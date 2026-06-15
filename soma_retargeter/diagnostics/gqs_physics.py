from __future__ import annotations

import csv
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_H4_MJCF = Path("/home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml")


@dataclass(frozen=True)
class RobotCSVMotion:
    name: str
    path: Path
    fps: float
    root_pos_m: np.ndarray
    root_quat_xyzw: np.ndarray
    joint_pos_rad: np.ndarray
    joint_names: list[str]


@dataclass(frozen=True)
class GQSPhysicsWeights:
    foot_sliding: float = 1.0
    velocity_violation: float = 500.0
    self_collision: float = 1000.0
    jerk: float = 0.1
    penetration: float = 10.0
    floating_frames_ratio: float = 200.0


@dataclass(frozen=True)
class GQSPhysicsConfig:
    fps: float = 120.0
    min_duration_sec: float = 0.5
    foot_contact_height_m: float = 0.05
    foot_slide_speed_threshold_m_s: float = 0.10
    floating_clearance_m: float = 0.05
    penetration_margin_m: float = 0.01
    joint_velocity_limit_deg_s: float = 600.0
    floor_size_m: float = 10.0
    enable_robot_self_contact: bool = True
    weights: GQSPhysicsWeights = GQSPhysicsWeights()


@dataclass(frozen=True)
class GQSPhysicsResult:
    motion: str
    path: str
    num_frames: int
    duration_sec: float
    score: float
    passed: bool
    metrics: dict[str, float]
    deductions: dict[str, float]
    error: str = ""

    def flat_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "motion": self.motion,
            "path": self.path,
            "num_frames": self.num_frames,
            "duration_sec": self.duration_sec,
            "score": self.score,
            "passed": self.passed,
            "error": self.error,
        }
        row.update({f"metric_{k}": v for k, v in self.metrics.items()})
        row.update({f"deduction_{k}": v for k, v in self.deductions.items()})
        return row


def load_robot_csv_motion(path: Path | str, fps: float = 120.0) -> RobotCSVMotion:
    path = Path(path).expanduser().resolve()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no frames: {path}")

    root_pos_cm = np.asarray(
        [
            [
                float(row["root_translateX"]),
                float(row["root_translateY"]),
                float(row["root_translateZ"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    root_euler_deg = np.asarray(
        [
            [
                float(row["root_rotateX"]),
                float(row["root_rotateY"]),
                float(row["root_rotateZ"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    joint_columns = [name for name in (reader.fieldnames or []) if name.endswith("_dof")]
    joint_names = [name[: -len("_dof")] for name in joint_columns]
    if not joint_names:
        raise ValueError(f"CSV has no joint *_dof columns: {path}")
    joint_pos_deg = np.asarray(
        [[float(row[name]) for name in joint_columns] for row in rows],
        dtype=np.float64,
    )

    return RobotCSVMotion(
        name=path.stem,
        path=path,
        fps=float(fps),
        root_pos_m=root_pos_cm * 0.01,
        root_quat_xyzw=R.from_euler("xyz", root_euler_deg, degrees=True).as_quat(),
        joint_pos_rad=np.deg2rad(joint_pos_deg),
        joint_names=joint_names,
    )


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values, dtype=np.float64)
    if values.shape[0] > 1:
        out[1:] = (values[1:] - values[:-1]) * float(fps)
    return out


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return quat / norm


def score_from_metrics(
    metrics: dict[str, float],
    weights: GQSPhysicsWeights = GQSPhysicsWeights(),
) -> tuple[float, dict[str, float]]:
    weight_map = asdict(weights)
    deductions = {
        key: float(metrics.get(key, 0.0)) * float(weight)
        for key, weight in weight_map.items()
    }
    score = max(0.0, 100.0 - float(sum(deductions.values())))
    return score, deductions


def build_model_qpos(motion: RobotCSVMotion, model: Any) -> np.ndarray:
    qpos = np.tile(np.asarray(model.qpos0, dtype=np.float64), (motion.root_pos_m.shape[0], 1))
    qpos[:, 0:3] = motion.root_pos_m
    quat = normalize_quat_xyzw(motion.root_quat_xyzw)
    qpos[:, 3:7] = quat[:, [3, 0, 1, 2]]

    missing: list[str] = []
    import mujoco

    for dof_idx, joint_name in enumerate(motion.joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            missing.append(joint_name)
            continue
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Expected hinge joint for {joint_name}, got type={int(model.jnt_type[joint_id])}")
        qpos_addr = int(model.jnt_qposadr[joint_id])
        qpos[:, qpos_addr] = motion.joint_pos_rad[:, dof_idx]

    if missing:
        raise ValueError(f"MJCF is missing motion joints: {', '.join(missing)}")
    return qpos


def build_qvel_from_qpos(qpos: np.ndarray, fps: float, model: Any) -> np.ndarray:
    qvel = np.zeros((qpos.shape[0], int(model.nv)), dtype=np.float64)
    if qpos.shape[0] <= 1:
        return qvel

    dt = 1.0 / float(fps)
    import mujoco

    for frame in range(1, qpos.shape[0]):
        mujoco.mj_differentiatePos(model, qvel[frame], dt, qpos[frame - 1], qpos[frame])
    qvel[0] = qvel[1]
    return qvel


def enable_robot_self_contact_mjcf(text: str) -> str:
    """Enable robot-robot contacts in the scoring-only MJCF.

    The H4 training MJCF intentionally has robot collision geoms inheriting
    conaffinity=0. For offline filtering we enable conaffinity on the shared
    collision default so MuJoCo can report self-contact pairs.
    """

    pattern = re.compile(r'(<default\s+class="collision"\s*>\s*<geom\b)([^>]*/>)', flags=re.MULTILINE)

    def repl(match: re.Match[str]) -> str:
        prefix, attrs = match.groups()
        if "conaffinity=" in attrs:
            attrs = re.sub(r'conaffinity="[^"]*"', 'conaffinity="1"', attrs)
        else:
            attrs = attrs.replace("/>", ' conaffinity="1" />')
        if "contype=" not in attrs:
            attrs = attrs.replace("/>", ' contype="1" />')
        return prefix + attrs

    text, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise ValueError("Could not find <default class=\"collision\"><geom .../> block to enable self-contact")
    return text


def make_floor_mjcf(
    base_xml: Path | str,
    floor_size_m: float = 10.0,
    enable_robot_self_contact: bool = True,
) -> Path:
    base_xml = Path(base_xml).expanduser().resolve()
    text = base_xml.read_text(encoding="utf-8")
    meshdir = (base_xml.parent / "../meshes/visual").resolve()
    text = text.replace('meshdir="../meshes/visual"', f'meshdir="{meshdir}"')
    if enable_robot_self_contact:
        text = enable_robot_self_contact_mjcf(text)
    if "<worldbody>" not in text:
        raise ValueError(f"{base_xml} has no <worldbody> tag")
    floor = (
        f'\n    <geom name="gqs_floor" type="plane" size="{float(floor_size_m)} {float(floor_size_m)} 0.01" '
        'pos="0 0 0" rgba="0.2 0.2 0.2 0.2" contype="1" conaffinity="15" group="0" />\n'
    )
    text = text.replace("<worldbody>", "<worldbody>" + floor, 1)
    tmp = tempfile.NamedTemporaryFile(prefix=f"{base_xml.stem}_gqs_floor_", suffix=".xml", delete=False)
    out = Path(tmp.name)
    tmp.close()
    out.write_text(text, encoding="utf-8")
    return out


def foot_collision_geom_ids(model: Any) -> list[int]:
    import mujoco

    ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        lower = name.lower()
        if ("left_foot" in lower or "right_foot" in lower) and "collision" in lower:
            ids.append(geom_id)
    if not ids:
        raise RuntimeError("No left/right foot collision geoms found")
    return ids


def geom_bottom_z(model: Any, data: Any, geom_id: int) -> float:
    import mujoco

    geom_type = int(model.geom_type[geom_id])
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)

    if geom_type in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        radius = float(size[0])
        half_length = float(size[1])
        axis = mat[:, 2]
        return float(pos[2] - abs(axis[2]) * half_length - radius)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(pos[2] - size[0])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        # Conservative support function for the z projection of an oriented box.
        return float(pos[2] - np.sum(np.abs(mat[2, :]) * size[:3]))
    return float(pos[2])


class GQSPhysicsScorer:
    def __init__(
        self,
        mjcf_path: Path | str = DEFAULT_H4_MJCF,
        config: GQSPhysicsConfig = GQSPhysicsConfig(),
    ):
        import mujoco

        self.mujoco = mujoco
        self.config = config
        self.floor_xml = make_floor_mjcf(
            mjcf_path,
            floor_size_m=config.floor_size_m,
            enable_robot_self_contact=config.enable_robot_self_contact,
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.floor_xml))
        self.data = mujoco.MjData(self.model)
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "gqs_floor")
        self.foot_geom_ids = foot_collision_geom_ids(self.model)
        self.foot_body_ids = sorted({int(self.model.geom_bodyid[gid]) for gid in self.foot_geom_ids})

    def close(self) -> None:
        try:
            self.floor_xml.unlink()
        except OSError:
            pass

    def __enter__(self) -> "GQSPhysicsScorer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def score_csv(self, csv_path: Path | str, threshold: float = 90.0) -> GQSPhysicsResult:
        return self.score_motion(load_robot_csv_motion(csv_path, fps=self.config.fps), threshold=threshold)

    def score_motion(self, motion: RobotCSVMotion, threshold: float = 90.0) -> GQSPhysicsResult:
        config = self.config
        duration_sec = motion.root_pos_m.shape[0] / float(config.fps)
        if motion.root_pos_m.shape[0] < 5 or duration_sec < config.min_duration_sec:
            metrics = {
                "foot_sliding": 100.0,
                "velocity_violation": 100.0,
                "self_collision": 100.0,
                "jerk": 100.0,
                "penetration": 100.0,
                "floating_frames_ratio": 1.0,
                "is_too_short": 1.0,
            }
            score, deductions = score_from_metrics(metrics, config.weights)
            return GQSPhysicsResult(
                motion=motion.name,
                path=str(motion.path),
                num_frames=int(motion.root_pos_m.shape[0]),
                duration_sec=float(duration_sec),
                score=score,
                passed=False,
                metrics=metrics,
                deductions=deductions,
            )

        qpos = build_model_qpos(motion, self.model)
        qvel = build_qvel_from_qpos(qpos, config.fps, self.model)
        joint_vel = finite_difference(motion.joint_pos_rad, config.fps)
        joint_acc = finite_difference(joint_vel, config.fps)
        joint_jerk = finite_difference(joint_acc, config.fps)

        velocity_limit = np.deg2rad(float(config.joint_velocity_limit_deg_s))
        velocity_violation = float(np.mean(np.maximum(0.0, np.abs(joint_vel) - velocity_limit)))
        jerk_metric = float(np.linalg.norm(joint_jerk, axis=1).mean() * 0.01)

        slide_sum = 0.0
        penetration_sum = 0.0
        self_collision_sum = 0.0
        floating_flags: list[float] = []
        min_foot_bottom_values: list[float] = []

        for frame in range(qpos.shape[0]):
            self.data.qpos[:] = qpos[frame]
            self.data.qvel[:] = qvel[frame]
            self.mujoco.mj_forward(self.model, self.data)

            foot_bottoms = [geom_bottom_z(self.model, self.data, geom_id) for geom_id in self.foot_geom_ids]
            min_foot_bottom = min(foot_bottoms)
            min_foot_bottom_values.append(float(min_foot_bottom))
            foot_contact = min_foot_bottom < config.foot_contact_height_m

            if foot_contact:
                body_speeds = [
                    float(np.linalg.norm(np.asarray(self.data.cvel[body_id], dtype=np.float64)[3:5]))
                    for body_id in self.foot_body_ids
                ]
                slide_sum += max(0.0, max(body_speeds, default=0.0) - config.foot_slide_speed_threshold_m_s) * 5.0

            active_contact = 0
            floor_dists: list[float] = []
            for contact_id in range(int(self.data.ncon)):
                contact = self.data.contact[contact_id]
                dist = float(contact.dist)
                g1 = int(contact.geom1)
                g2 = int(contact.geom2)
                has_floor = g1 == self.floor_geom_id or g2 == self.floor_geom_id
                if has_floor:
                    floor_dists.append(dist)
                elif dist < 0.0:
                    active_contact += 1
            self_collision_sum += min(float(active_contact), 10.0)

            if floor_dists:
                min_floor_dist = min(floor_dists)
                penetration_sum += max(0.0, -min_floor_dist - config.penetration_margin_m)
                floating_flags.append(1.0 if min_floor_dist > config.floating_clearance_m else 0.0)
            else:
                floating_flags.append(1.0 if min_foot_bottom > config.floating_clearance_m else 0.0)

        floating = np.asarray(floating_flags, dtype=np.float64)
        window_size = max(1, int(round(1.0 * config.fps)))
        if floating.shape[0] >= window_size:
            kernel = np.ones(window_size, dtype=np.float64)
            conv = np.convolve(floating, kernel, mode="same")
            floating_frames_ratio = float(np.mean(conv >= (window_size - 0.1)))
        else:
            floating_frames_ratio = float(np.mean(floating))

        metrics = {
            "foot_sliding": float(slide_sum / qpos.shape[0]),
            "velocity_violation": velocity_violation,
            "self_collision": float(self_collision_sum / qpos.shape[0]),
            "jerk": jerk_metric,
            "penetration": float(penetration_sum / qpos.shape[0]),
            "floating_frames_ratio": floating_frames_ratio,
            "min_foot_bottom_z_m": float(np.min(min_foot_bottom_values)),
            "max_joint_velocity_deg_s": float(np.rad2deg(np.max(np.abs(joint_vel)))),
            "max_joint_acceleration_deg_s2": float(np.rad2deg(np.max(np.abs(joint_acc)))),
            "max_joint_jerk_deg_s3": float(np.rad2deg(np.max(np.abs(joint_jerk)))),
        }
        score, deductions = score_from_metrics(metrics, config.weights)
        return GQSPhysicsResult(
            motion=motion.name,
            path=str(motion.path),
            num_frames=int(qpos.shape[0]),
            duration_sec=float(duration_sec),
            score=score,
            passed=score >= float(threshold),
            metrics=metrics,
            deductions=deductions,
        )


def score_motion_csv(
    csv_path: Path | str,
    *,
    mjcf_path: Path | str = DEFAULT_H4_MJCF,
    config: GQSPhysicsConfig = GQSPhysicsConfig(),
    threshold: float = 90.0,
) -> GQSPhysicsResult:
    with GQSPhysicsScorer(mjcf_path=mjcf_path, config=config) as scorer:
        return scorer.score_csv(csv_path, threshold=threshold)


def write_results(
    results: list[GQSPhysicsResult],
    output_dir: Path | str,
    config: GQSPhysicsConfig,
    threshold: float,
) -> None:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [result.flat_row() for result in results]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = ["motion", "passed", "score", "num_frames", "duration_sec", "path", "error"]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with (output_dir / "physics_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "threshold": float(threshold),
        "num_motions": len(results),
        "num_passed": sum(1 for result in results if result.passed),
        "num_failed": sum(1 for result in results if not result.passed),
        "config": asdict(config),
        "results": [asdict(result) for result in results],
    }
    (output_dir / "physics_scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
