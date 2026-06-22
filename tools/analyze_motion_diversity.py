#!/usr/bin/env python3
"""Analyze robot motion diversity without selecting or deleting motions.

The report is intentionally diagnostic-only.  It extracts a compact set of
motion features, normalizes them with robust min/max statistics, computes
weighted group-wise nearest-neighbor distances, and writes CSV/Markdown reports
that can be used to tune future diversity filtering.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - convenience fallback for bare system Python.
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable


DEFAULT_GROUP_WEIGHTS = {
    "root_locomotion": 0.20,
    "height_posture": 0.15,
    "foot_gait_proxy": 0.20,
    "upper_body_workspace_proxy": 0.20,
    "joint_range_group": 0.10,
    "dynamics_complexity": 0.10,
    "periodicity_symmetry": 0.05,
}

DEFAULT_MJCF = Path("/home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml")
FK_FRAME_STRIDE = 4  # 120 Hz source CSV -> 30 Hz FK features, without modifying source files.

FK_BODY_CANDIDATES = {
    "pelvis": ["pelvis_link"],
    "torso": ["torso_link"],
    "head": ["head_pitch_link", "head_yaw_link"],
    "left_wrist": ["left_wrist_pitch_link", "left_wrist_roll_link", "left_wrist_yaw_link"],
    "right_wrist": ["right_wrist_pitch_link", "right_wrist_roll_link", "right_wrist_yaw_link"],
    "left_foot": ["left_ankle_pitch_link", "left_ankle_roll_link"],
    "right_foot": ["right_ankle_pitch_link", "right_ankle_roll_link"],
}

_FK_MUJOCO: Any | None = None
_FK_MODEL: Any | None = None
_FK_DATA: Any | None = None
_FK_BODY_IDS: dict[str, int] = {}

JOINT_GROUPS = {
    "left_leg": [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_roll_joint",
        "left_ankle_pitch_joint",
    ],
    "right_leg": [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_roll_joint",
        "right_ankle_pitch_joint",
    ],
    "left_arm": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_roll_joint",
        "left_wrist_yaw_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
    ],
    "right_arm": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_roll_joint",
        "right_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
    ],
    "waist_head": ["waist_yaw_joint", "head_yaw_joint", "head_pitch_joint"],
}


@dataclass(frozen=True)
class MotionMeta:
    rel_path: str
    dataset: str
    category: str
    subcategory: str
    motion: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, required=True, help="Root containing CSV motions.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for diversity reports.")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="MJCF used for MuJoCo FK spatial features.")
    parser.add_argument("--no-mujoco-fk", action="store_true", help="Disable MuJoCo FK features and use CSV root/joint features only.")
    parser.add_argument("--score-root", type=Path, default=None, help="Optional root containing GQS physics_scores.csv files.")
    parser.add_argument("--glob", default="**/*.csv", help="CSV glob relative to --motion-root.")
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--nearest-k", type=int, default=5, help="Number of nearest neighbors to report per motion.")
    parser.add_argument("--robust-low", type=float, default=1.0, help="Low percentile for robust normalization.")
    parser.add_argument("--robust-high", type=float, default=99.0, help="High percentile for robust normalization.")
    parser.add_argument(
        "--group-weights",
        default=None,
        help="Optional comma list like root_locomotion=0.2,height_posture=0.15. Defaults to the initial 7-group weights.",
    )
    parser.add_argument("--redundant-quantile", type=float, default=0.05)
    parser.add_argument("--isolated-quantile", type=float, default=0.95)
    return parser.parse_args()


def init_fk_worker(mjcf_path: str | None) -> None:
    """Initialize one MuJoCo model per worker process."""
    global _FK_MUJOCO, _FK_MODEL, _FK_DATA, _FK_BODY_IDS
    if not mjcf_path:
        _FK_MUJOCO = None
        _FK_MODEL = None
        _FK_DATA = None
        _FK_BODY_IDS = {}
        return

    import mujoco

    _FK_MUJOCO = mujoco
    _FK_MODEL = mujoco.MjModel.from_xml_path(str(Path(mjcf_path).expanduser().resolve()))
    _FK_DATA = mujoco.MjData(_FK_MODEL)
    _FK_BODY_IDS = {}
    for key, candidates in FK_BODY_CANDIDATES.items():
        for name in candidates:
            idx = mujoco.mj_name2id(_FK_MODEL, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx >= 0:
                _FK_BODY_IDS[key] = int(idx)
                break


def parse_group_weights(text: str | None) -> dict[str, float]:
    weights = dict(DEFAULT_GROUP_WEIGHTS)
    if not text:
        return weights
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in weights:
            raise ValueError(f"Unknown group weight key: {key}")
        weights[key] = float(value)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Group weights sum to zero")
    return {key: value / total for key, value in weights.items()}


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.nanmean(values))


def safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.nanstd(values))


def safe_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.nanmax(values))


def safe_min(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.nanmin(values))


def safe_percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.nanpercentile(values, q))


def safe_range(values: np.ndarray) -> float:
    return safe_max(values) - safe_min(values)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or b.size < 3 or a.shape != b.shape:
        return 0.0
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def wrapped_diff_deg(next_angle: np.ndarray, prev_angle: np.ndarray) -> np.ndarray:
    return (next_angle - prev_angle + 180.0) % 360.0 - 180.0


def unwrap_deg(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(values.astype(np.float64))))


def spectral_peak_features(signal: np.ndarray, fps: float) -> tuple[float, float]:
    x = np.asarray(signal, dtype=np.float64)
    if x.size < 16:
        return 0.0, 0.0
    # Downsample long clips to keep feature extraction cheap and comparable.
    max_samples = 4096
    if x.size > max_samples:
        idx = np.linspace(0, x.size - 1, max_samples).astype(np.int64)
        x = x[idx]
        effective_fps = fps * (max_samples - 1) / max(1, signal.size - 1)
    else:
        effective_fps = fps
    x = x - np.mean(x)
    if float(np.std(x)) <= 1e-8:
        return 0.0, 0.0
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / effective_fps)
    mask = (freqs >= 0.2) & (freqs <= 8.0)
    if not np.any(mask):
        return 0.0, 0.0
    mag = spectrum[mask]
    fr = freqs[mask]
    total = float(np.sum(mag))
    if total <= 1e-12:
        return 0.0, 0.0
    idx = int(np.argmax(mag))
    return float(mag[idx] / total), float(fr[idx])


def metadata_from_path(path: Path, root: Path) -> MotionMeta:
    rel = path.relative_to(root)
    parts = rel.parts
    dataset = parts[0] if len(parts) > 0 else "unknown"
    if dataset == "seed" and len(parts) >= 2:
        category = parts[1]
        subcategory = category
    elif dataset == "grab" and len(parts) >= 2:
        category = "grab"
        subcategory = parts[1]
    else:
        category = dataset
        subcategory = parts[1] if len(parts) > 1 else dataset
    return MotionMeta(
        rel_path=rel.as_posix(),
        dataset=dataset,
        category=category,
        subcategory=subcategory,
        motion=path.stem,
    )


def read_csv_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if data.size == 0:
        data = np.zeros((0, len(header)), dtype=np.float64)
    return header, data


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return quat / norm


def build_fk_qpos(
    root_pos_m: np.ndarray,
    root_rot_deg: np.ndarray,
    joint_names: list[str],
    q_deg: np.ndarray,
) -> np.ndarray:
    if _FK_MUJOCO is None or _FK_MODEL is None:
        raise RuntimeError("MuJoCo FK model is not initialized")

    mujoco = _FK_MUJOCO
    model = _FK_MODEL
    qpos = np.tile(np.asarray(model.qpos0, dtype=np.float64), (root_pos_m.shape[0], 1))
    free_joint_ids = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    if free_joint_ids.size == 0:
        raise ValueError("MJCF has no free joint for root pose")
    free_qpos_addr = int(model.jnt_qposadr[int(free_joint_ids[0])])
    qpos[:, free_qpos_addr : free_qpos_addr + 3] = root_pos_m
    quat_xyzw = normalize_quat_xyzw(R.from_euler("xyz", root_rot_deg, degrees=True).as_quat())
    qpos[:, free_qpos_addr + 3 : free_qpos_addr + 7] = quat_xyzw[:, [3, 0, 1, 2]]

    missing: list[str] = []
    for dof_idx, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            missing.append(joint_name)
            continue
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Expected hinge joint for {joint_name}, got type={int(model.jnt_type[joint_id])}")
        qpos_addr = int(model.jnt_qposadr[joint_id])
        qpos[:, qpos_addr] = np.deg2rad(q_deg[:, dof_idx])
    if missing:
        raise ValueError(f"MJCF is missing motion joints: {', '.join(missing)}")
    return qpos


def compute_fk_spatial_series(
    root_pos_m: np.ndarray,
    root_rot_deg: np.ndarray,
    joint_names: list[str],
    q_deg: np.ndarray,
) -> dict[str, Any]:
    """Compute 30 Hz spatial feature series by taking every fourth source frame."""
    if _FK_MUJOCO is None or _FK_MODEL is None or _FK_DATA is None:
        return {}
    if root_pos_m.shape[0] == 0:
        return {}

    sample_idx = np.arange(0, root_pos_m.shape[0], FK_FRAME_STRIDE, dtype=np.int64)
    if sample_idx.size == 0:
        sample_idx = np.array([0], dtype=np.int64)

    root_pos_s = root_pos_m[sample_idx]
    root_rot_s = root_rot_deg[sample_idx]
    q_s = q_deg[sample_idx]
    qpos = build_fk_qpos(root_pos_s, root_rot_s, joint_names, q_s)

    mujoco = _FK_MUJOCO
    model = _FK_MODEL
    data = _FK_DATA
    body_pos: dict[str, np.ndarray] = {
        key: np.zeros((sample_idx.size, 3), dtype=np.float64)
        for key in _FK_BODY_IDS
    }
    for frame_idx in range(sample_idx.size):
        data.qpos[:] = qpos[frame_idx]
        mujoco.mj_forward(model, data)
        for key, body_id in _FK_BODY_IDS.items():
            body_pos[key][frame_idx] = np.asarray(data.xpos[body_id], dtype=np.float64)

    root_rot_inv = R.from_euler("xyz", root_rot_s, degrees=True).inv()
    rel_root: dict[str, np.ndarray] = {}
    for key, values in body_pos.items():
        rel_root[key] = root_rot_inv.apply(values - root_pos_s)

    return {
        "sample_idx": sample_idx,
        "fps": 120.0 / FK_FRAME_STRIDE,
        "body_pos": body_pos,
        "rel_root": rel_root,
    }


def group_indices(joint_names: list[str], group: str) -> list[int]:
    wanted = set(JOINT_GROUPS[group])
    return [idx for idx, name in enumerate(joint_names) if name in wanted]


def mean_abs_range(q: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return safe_mean(np.ptp(q[:, indices], axis=0))


def max_abs_range(q: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return safe_max(np.ptp(q[:, indices], axis=0))


def mean_abs_std(q: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return safe_mean(np.std(q[:, indices], axis=0))


def mean_abs_velocity(vel: np.ndarray, indices: list[int]) -> float:
    if not indices or vel.size == 0:
        return 0.0
    return safe_mean(np.abs(vel[:, indices]))


def extract_features_for_path(args: tuple[Path, Path, float]) -> dict[str, Any]:
    path, root, fps = args
    meta = metadata_from_path(path, root)
    header, data = read_csv_matrix(path)
    col = {name: idx for idx, name in enumerate(header)}
    if data.shape[0] < 2:
        raise ValueError(f"Motion has fewer than 2 frames: {path}")

    root_xyz = data[:, [col["root_translateX"], col["root_translateY"], col["root_translateZ"]]] * 0.01
    root_rot = data[:, [col["root_rotateX"], col["root_rotateY"], col["root_rotateZ"]]]
    joint_cols = [idx for idx, name in enumerate(header) if name.endswith("_dof")]
    joint_names = [header[idx].removesuffix("_dof") for idx in joint_cols]
    q = data[:, joint_cols]

    dq = wrapped_diff_deg(q[1:], q[:-1])
    joint_vel = dq * fps
    joint_acc = np.diff(joint_vel, axis=0) * fps if joint_vel.shape[0] >= 2 else np.zeros((0, q.shape[1]))
    joint_jerk = np.diff(joint_acc, axis=0) * fps if joint_acc.shape[0] >= 2 else np.zeros((0, q.shape[1]))

    dxy = np.diff(root_xyz[:, :2], axis=0)
    dz = np.diff(root_xyz[:, 2], axis=0)
    xy_step = np.linalg.norm(dxy, axis=1)
    xy_speed = xy_step * fps
    xy_acc = np.diff(xy_speed) * fps if xy_speed.size >= 2 else np.zeros((0,), dtype=np.float64)
    yaw = unwrap_deg(root_rot[:, 2])
    dyaw = np.diff(yaw)
    yaw_rate = dyaw * fps

    l_leg = group_indices(joint_names, "left_leg")
    r_leg = group_indices(joint_names, "right_leg")
    l_arm = group_indices(joint_names, "left_arm")
    r_arm = group_indices(joint_names, "right_arm")
    waist_head = group_indices(joint_names, "waist_head")
    lower = l_leg + r_leg
    upper = l_arm + r_arm

    def joint(name: str) -> np.ndarray:
        if name not in joint_names:
            return np.zeros((q.shape[0],), dtype=np.float64)
        return q[:, joint_names.index(name)]

    left_hip_pitch = joint("left_hip_pitch_joint")
    right_hip_pitch = joint("right_hip_pitch_joint")
    left_knee = joint("left_knee_joint")
    right_knee = joint("right_knee_joint")
    left_elbow = joint("left_elbow_roll_joint")
    right_elbow = joint("right_elbow_roll_joint")

    root_speed_peak_ratio, root_speed_peak_hz = spectral_peak_features(xy_speed, fps)
    hip_peak_ratio, hip_peak_hz = spectral_peak_features(left_hip_pitch - right_hip_pitch, fps)
    knee_peak_ratio, knee_peak_hz = spectral_peak_features(left_knee - right_knee, fps)

    features: dict[str, float] = {}
    groups: dict[str, list[str]] = {key: [] for key in DEFAULT_GROUP_WEIGHTS}

    def add(group: str, name: str, value: float) -> None:
        feature_name = f"{group}__{name}"
        if not math.isfinite(value):
            value = 0.0
        features[feature_name] = float(value)
        groups[group].append(feature_name)

    fk = compute_fk_spatial_series(root_xyz, root_rot, joint_names, q)
    fk_body: dict[str, np.ndarray] = fk.get("body_pos", {})
    fk_rel: dict[str, np.ndarray] = fk.get("rel_root", {})
    fk_fps = fps / FK_FRAME_STRIDE

    def body_series(name: str, *, rel: bool = False) -> np.ndarray:
        source = fk_rel if rel else fk_body
        return source.get(name, np.zeros((0, 3), dtype=np.float64))

    def axis_values(values: np.ndarray, axis: int) -> np.ndarray:
        return values[:, axis] if values.ndim == 2 and values.shape[0] > 0 else np.zeros((0,), dtype=np.float64)

    def body_speed(values: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[0] < 2:
            return np.zeros((0,), dtype=np.float64)
        return np.linalg.norm(np.diff(values, axis=0), axis=1) * fk_fps

    def body_xy_speed(values: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[0] < 2:
            return np.zeros((0,), dtype=np.float64)
        return np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1) * fk_fps

    def pair_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != b.shape or a.ndim != 2 or a.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        return np.linalg.norm(a - b, axis=1)

    def component_range(values: np.ndarray, axis: int) -> float:
        return safe_range(axis_values(values, axis))

    pelvis_w = body_series("pelvis")
    torso_w = body_series("torso")
    head_w = body_series("head")
    left_foot_w = body_series("left_foot")
    right_foot_w = body_series("right_foot")
    left_wrist_w = body_series("left_wrist")
    right_wrist_w = body_series("right_wrist")
    pelvis_r = body_series("pelvis", rel=True)
    torso_r = body_series("torso", rel=True)
    head_r = body_series("head", rel=True)
    left_foot_r = body_series("left_foot", rel=True)
    right_foot_r = body_series("right_foot", rel=True)
    left_wrist_r = body_series("left_wrist", rel=True)
    right_wrist_r = body_series("right_wrist", rel=True)

    def low_contact_proxy(foot_w: np.ndarray) -> np.ndarray:
        z = axis_values(foot_w, 2)
        if z.size == 0:
            return np.zeros((0,), dtype=bool)
        return z <= safe_percentile(z, 10) + 0.025

    left_low = low_contact_proxy(left_foot_w)
    right_low = low_contact_proxy(right_foot_w)
    left_foot_xy_speed = body_xy_speed(left_foot_w)
    right_foot_xy_speed = body_xy_speed(right_foot_w)

    duration = (data.shape[0] - 1) / fps
    path_xy = float(np.sum(xy_step))
    net_xy = float(np.linalg.norm(root_xyz[-1, :2] - root_xyz[0, :2]))

    add("root_locomotion", "duration_sec", duration)
    add("root_locomotion", "path_xy_m", path_xy)
    add("root_locomotion", "net_xy_m", net_xy)
    add("root_locomotion", "net_over_path_xy", net_xy / max(path_xy, 1e-6))
    add("root_locomotion", "mean_xy_speed_m_s", safe_mean(xy_speed))
    add("root_locomotion", "max_xy_speed_m_s", safe_max(xy_speed))
    add("root_locomotion", "std_xy_speed_m_s", safe_std(xy_speed))
    add("root_locomotion", "yaw_abs_change_deg", float(np.sum(np.abs(dyaw))))
    add("root_locomotion", "yaw_range_deg", safe_range(yaw))
    add("root_locomotion", "mean_abs_yaw_rate_deg_s", safe_mean(np.abs(yaw_rate)))
    add("root_locomotion", "max_abs_yaw_rate_deg_s", safe_max(np.abs(yaw_rate)))
    add("root_locomotion", "vertical_path_m", float(np.sum(np.abs(dz))))

    add("height_posture", "root_z_mean_m", safe_mean(root_xyz[:, 2]))
    add("height_posture", "root_z_min_m", safe_min(root_xyz[:, 2]))
    add("height_posture", "root_z_max_m", safe_max(root_xyz[:, 2]))
    add("height_posture", "root_z_range_m", safe_range(root_xyz[:, 2]))
    add("height_posture", "root_z_std_m", safe_std(root_xyz[:, 2]))
    add("height_posture", "root_pitch_range_deg", safe_range(unwrap_deg(root_rot[:, 1])))
    add("height_posture", "root_roll_range_deg", safe_range(unwrap_deg(root_rot[:, 0])))
    add("height_posture", "root_pitch_rms_deg", float(np.sqrt(safe_mean(root_rot[:, 1] ** 2))))
    add("height_posture", "root_roll_rms_deg", float(np.sqrt(safe_mean(root_rot[:, 0] ** 2))))
    add("height_posture", "knee_mean_abs_deg", safe_mean(np.abs(q[:, [joint_names.index(j) for j in ["left_knee_joint", "right_knee_joint"] if j in joint_names]])))
    add("height_posture", "knee_range_mean_deg", mean_abs_range(q, [joint_names.index(j) for j in ["left_knee_joint", "right_knee_joint"] if j in joint_names]))
    add("height_posture", "hip_pitch_mean_abs_deg", safe_mean(np.abs(np.column_stack([left_hip_pitch, right_hip_pitch]))))
    add("height_posture", "hip_pitch_range_mean_deg", safe_mean(np.ptp(np.column_stack([left_hip_pitch, right_hip_pitch]), axis=0)))
    add("height_posture", "ankle_pitch_mean_abs_deg", safe_mean(np.abs(q[:, [joint_names.index(j) for j in ["left_ankle_pitch_joint", "right_ankle_pitch_joint"] if j in joint_names]])))
    add("height_posture", "waist_head_range_mean_deg", mean_abs_range(q, waist_head))
    add("height_posture", "fk_pelvis_z_mean_m", safe_mean(axis_values(pelvis_w, 2)))
    add("height_posture", "fk_pelvis_z_range_m", component_range(pelvis_w, 2))
    add("height_posture", "fk_torso_z_mean_m", safe_mean(axis_values(torso_w, 2)))
    add("height_posture", "fk_torso_z_range_m", component_range(torso_w, 2))
    add("height_posture", "fk_head_z_mean_m", safe_mean(axis_values(head_w, 2)))
    add("height_posture", "fk_head_z_range_m", component_range(head_w, 2))
    add("height_posture", "fk_head_rel_z_mean_m", safe_mean(axis_values(head_r, 2)))
    add("height_posture", "fk_head_rel_z_range_m", component_range(head_r, 2))
    add("height_posture", "fk_torso_rel_x_range_m", component_range(torso_r, 0))
    add("height_posture", "fk_torso_rel_y_range_m", component_range(torso_r, 1))
    add("height_posture", "fk_torso_rel_z_range_m", component_range(torso_r, 2))

    add("foot_gait_proxy", "left_leg_range_mean_deg", mean_abs_range(q, l_leg))
    add("foot_gait_proxy", "right_leg_range_mean_deg", mean_abs_range(q, r_leg))
    add("foot_gait_proxy", "left_leg_range_max_deg", max_abs_range(q, l_leg))
    add("foot_gait_proxy", "right_leg_range_max_deg", max_abs_range(q, r_leg))
    add("foot_gait_proxy", "hip_pitch_lr_anti_corr", safe_corr(left_hip_pitch, -right_hip_pitch))
    add("foot_gait_proxy", "knee_lr_corr", safe_corr(left_knee, right_knee))
    add("foot_gait_proxy", "lower_joint_vel_mean_abs_deg_s", mean_abs_velocity(joint_vel, lower))
    add("foot_gait_proxy", "lower_joint_vel_p95_abs_deg_s", safe_percentile(np.abs(joint_vel[:, lower]) if lower and joint_vel.size else np.array([]), 95))
    left_leg_activity = mean_abs_range(q, l_leg) + mean_abs_velocity(joint_vel, l_leg) / 120.0
    right_leg_activity = mean_abs_range(q, r_leg) + mean_abs_velocity(joint_vel, r_leg) / 120.0
    add("foot_gait_proxy", "leg_activity_asymmetry", abs(left_leg_activity - right_leg_activity))
    add("foot_gait_proxy", "speed_to_leg_activity_ratio", safe_mean(xy_speed) / max(left_leg_activity + right_leg_activity, 1e-6))
    left_foot_z = axis_values(left_foot_w, 2)
    right_foot_z = axis_values(right_foot_w, 2)
    left_foot_rel_z = axis_values(left_foot_r, 2)
    right_foot_rel_z = axis_values(right_foot_r, 2)
    foot_sep = pair_distance(left_foot_r, right_foot_r)
    foot_step_width = np.abs(axis_values(left_foot_r - right_foot_r, 1)) if left_foot_r.shape == right_foot_r.shape and left_foot_r.size else np.zeros((0,), dtype=np.float64)
    foot_step_length = np.abs(axis_values(left_foot_r - right_foot_r, 0)) if left_foot_r.shape == right_foot_r.shape and left_foot_r.size else np.zeros((0,), dtype=np.float64)
    add("foot_gait_proxy", "fk_left_foot_z_min_m", safe_min(left_foot_z))
    add("foot_gait_proxy", "fk_right_foot_z_min_m", safe_min(right_foot_z))
    add("foot_gait_proxy", "fk_left_foot_z_range_m", safe_range(left_foot_z))
    add("foot_gait_proxy", "fk_right_foot_z_range_m", safe_range(right_foot_z))
    add("foot_gait_proxy", "fk_left_foot_rel_z_range_m", safe_range(left_foot_rel_z))
    add("foot_gait_proxy", "fk_right_foot_rel_z_range_m", safe_range(right_foot_rel_z))
    add("foot_gait_proxy", "fk_left_foot_xy_path_m", float(np.sum(np.linalg.norm(np.diff(left_foot_w[:, :2], axis=0), axis=1))) if left_foot_w.shape[0] >= 2 else 0.0)
    add("foot_gait_proxy", "fk_right_foot_xy_path_m", float(np.sum(np.linalg.norm(np.diff(right_foot_w[:, :2], axis=0), axis=1))) if right_foot_w.shape[0] >= 2 else 0.0)
    add("foot_gait_proxy", "fk_left_foot_speed_p95_m_s", safe_percentile(body_speed(left_foot_w), 95))
    add("foot_gait_proxy", "fk_right_foot_speed_p95_m_s", safe_percentile(body_speed(right_foot_w), 95))
    add("foot_gait_proxy", "fk_left_contact_proxy_ratio", float(np.mean(left_low)) if left_low.size else 0.0)
    add("foot_gait_proxy", "fk_right_contact_proxy_ratio", float(np.mean(right_low)) if right_low.size else 0.0)
    add("foot_gait_proxy", "fk_double_support_proxy_ratio", float(np.mean(left_low & right_low)) if left_low.size and right_low.size and left_low.shape == right_low.shape else 0.0)
    add("foot_gait_proxy", "fk_flight_proxy_ratio", float(np.mean((~left_low) & (~right_low))) if left_low.size and right_low.size and left_low.shape == right_low.shape else 0.0)
    add("foot_gait_proxy", "fk_left_low_xy_speed_mean_m_s", safe_mean(left_foot_xy_speed[left_low[1:]]) if left_low.size > 1 and left_foot_xy_speed.size == left_low.size - 1 else 0.0)
    add("foot_gait_proxy", "fk_right_low_xy_speed_mean_m_s", safe_mean(right_foot_xy_speed[right_low[1:]]) if right_low.size > 1 and right_foot_xy_speed.size == right_low.size - 1 else 0.0)
    add("foot_gait_proxy", "fk_foot_separation_mean_m", safe_mean(foot_sep))
    add("foot_gait_proxy", "fk_foot_separation_range_m", safe_range(foot_sep))
    add("foot_gait_proxy", "fk_step_width_mean_m", safe_mean(foot_step_width))
    add("foot_gait_proxy", "fk_step_width_range_m", safe_range(foot_step_width))
    add("foot_gait_proxy", "fk_step_length_mean_m", safe_mean(foot_step_length))
    add("foot_gait_proxy", "fk_step_length_range_m", safe_range(foot_step_length))
    add("foot_gait_proxy", "fk_foot_height_lr_corr", safe_corr(left_foot_z, right_foot_z))

    add("upper_body_workspace_proxy", "left_arm_range_mean_deg", mean_abs_range(q, l_arm))
    add("upper_body_workspace_proxy", "right_arm_range_mean_deg", mean_abs_range(q, r_arm))
    add("upper_body_workspace_proxy", "left_arm_range_max_deg", max_abs_range(q, l_arm))
    add("upper_body_workspace_proxy", "right_arm_range_max_deg", max_abs_range(q, r_arm))
    add("upper_body_workspace_proxy", "shoulder_range_mean_deg", mean_abs_range(q, [idx for idx, name in enumerate(joint_names) if "shoulder" in name]))
    add("upper_body_workspace_proxy", "elbow_range_mean_deg", mean_abs_range(q, [idx for idx, name in enumerate(joint_names) if "elbow" in name]))
    add("upper_body_workspace_proxy", "wrist_range_mean_deg", mean_abs_range(q, [idx for idx, name in enumerate(joint_names) if "wrist" in name]))
    add("upper_body_workspace_proxy", "wrist_range_max_deg", max_abs_range(q, [idx for idx, name in enumerate(joint_names) if "wrist" in name]))
    add("upper_body_workspace_proxy", "arm_joint_vel_mean_abs_deg_s", mean_abs_velocity(joint_vel, upper))
    left_arm_activity = mean_abs_range(q, l_arm) + mean_abs_velocity(joint_vel, l_arm) / 120.0
    right_arm_activity = mean_abs_range(q, r_arm) + mean_abs_velocity(joint_vel, r_arm) / 120.0
    add("upper_body_workspace_proxy", "arm_activity_asymmetry", abs(left_arm_activity - right_arm_activity))
    add("upper_body_workspace_proxy", "elbow_lr_corr", safe_corr(left_elbow, right_elbow))
    wrist_sep = pair_distance(left_wrist_r, right_wrist_r)
    for side, wrist_r, wrist_w in [
        ("left", left_wrist_r, left_wrist_w),
        ("right", right_wrist_r, right_wrist_w),
    ]:
        add("upper_body_workspace_proxy", f"fk_{side}_wrist_rel_x_mean_m", safe_mean(axis_values(wrist_r, 0)))
        add("upper_body_workspace_proxy", f"fk_{side}_wrist_rel_y_mean_m", safe_mean(axis_values(wrist_r, 1)))
        add("upper_body_workspace_proxy", f"fk_{side}_wrist_rel_z_mean_m", safe_mean(axis_values(wrist_r, 2)))
        add("upper_body_workspace_proxy", f"fk_{side}_wrist_rel_x_range_m", component_range(wrist_r, 0))
        add("upper_body_workspace_proxy", f"fk_{side}_wrist_rel_y_range_m", component_range(wrist_r, 1))
        add("upper_body_workspace_proxy", f"fk_{side}_wrist_rel_z_range_m", component_range(wrist_r, 2))
        add("upper_body_workspace_proxy", f"fk_{side}_wrist_path_m", float(np.sum(np.linalg.norm(np.diff(wrist_w, axis=0), axis=1))) if wrist_w.shape[0] >= 2 else 0.0)
        add("upper_body_workspace_proxy", f"fk_{side}_wrist_speed_p95_m_s", safe_percentile(body_speed(wrist_w), 95))
    add("upper_body_workspace_proxy", "fk_wrist_separation_mean_m", safe_mean(wrist_sep))
    add("upper_body_workspace_proxy", "fk_wrist_separation_range_m", safe_range(wrist_sep))
    add("upper_body_workspace_proxy", "fk_wrist_lr_height_corr", safe_corr(axis_values(left_wrist_w, 2), axis_values(right_wrist_w, 2)))

    joint_ranges = np.ptp(q, axis=0)
    joint_stds = np.std(q, axis=0)
    add("joint_range_group", "all_joint_range_mean_deg", safe_mean(joint_ranges))
    add("joint_range_group", "all_joint_range_max_deg", safe_max(joint_ranges))
    add("joint_range_group", "all_joint_range_std_deg", safe_std(joint_ranges))
    add("joint_range_group", "all_joint_std_mean_deg", safe_mean(joint_stds))
    add("joint_range_group", "active_joint_fraction_5deg", float(np.mean(joint_ranges >= 5.0)))
    add("joint_range_group", "active_joint_fraction_15deg", float(np.mean(joint_ranges >= 15.0)))
    add("joint_range_group", "lower_upper_range_ratio", mean_abs_range(q, lower) / max(mean_abs_range(q, upper), 1e-6))
    add("joint_range_group", "left_right_lower_range_asymmetry", abs(mean_abs_range(q, l_leg) - mean_abs_range(q, r_leg)))
    add("joint_range_group", "left_right_upper_range_asymmetry", abs(mean_abs_range(q, l_arm) - mean_abs_range(q, r_arm)))

    abs_vel = np.abs(joint_vel)
    abs_acc = np.abs(joint_acc)
    abs_jerk = np.abs(joint_jerk)
    add("dynamics_complexity", "joint_vel_mean_abs_deg_s", safe_mean(abs_vel))
    add("dynamics_complexity", "joint_vel_p95_abs_deg_s", safe_percentile(abs_vel, 95))
    add("dynamics_complexity", "joint_vel_max_abs_deg_s", safe_max(abs_vel))
    add("dynamics_complexity", "joint_acc_mean_abs_deg_s2", safe_mean(abs_acc))
    add("dynamics_complexity", "joint_acc_p95_abs_deg_s2", safe_percentile(abs_acc, 95))
    add("dynamics_complexity", "joint_acc_max_abs_deg_s2", safe_max(abs_acc))
    add("dynamics_complexity", "joint_jerk_mean_abs_deg_s3", safe_mean(abs_jerk))
    add("dynamics_complexity", "joint_jerk_p95_abs_deg_s3", safe_percentile(abs_jerk, 95))
    add("dynamics_complexity", "joint_jerk_max_abs_deg_s3", safe_max(abs_jerk))
    add("dynamics_complexity", "root_xy_acc_mean_abs_m_s2", safe_mean(np.abs(xy_acc)))
    add("dynamics_complexity", "root_xy_acc_max_abs_m_s2", safe_max(np.abs(xy_acc)))
    add("dynamics_complexity", "complexity_energy", safe_mean(abs_vel**2) + 0.0001 * safe_mean(abs_acc**2))
    for body_name, values in [
        ("left_foot", left_foot_w),
        ("right_foot", right_foot_w),
        ("left_wrist", left_wrist_w),
        ("right_wrist", right_wrist_w),
        ("head", head_w),
    ]:
        speed = body_speed(values)
        acc = np.diff(speed) * fk_fps if speed.size >= 2 else np.zeros((0,), dtype=np.float64)
        add("dynamics_complexity", f"fk_{body_name}_speed_mean_m_s", safe_mean(speed))
        add("dynamics_complexity", f"fk_{body_name}_speed_p95_m_s", safe_percentile(speed, 95))
        add("dynamics_complexity", f"fk_{body_name}_acc_p95_m_s2", safe_percentile(np.abs(acc), 95))

    add("periodicity_symmetry", "root_speed_fft_peak_ratio", root_speed_peak_ratio)
    add("periodicity_symmetry", "root_speed_fft_peak_hz", root_speed_peak_hz)
    add("periodicity_symmetry", "hip_pitch_diff_fft_peak_ratio", hip_peak_ratio)
    add("periodicity_symmetry", "hip_pitch_diff_fft_peak_hz", hip_peak_hz)
    add("periodicity_symmetry", "knee_diff_fft_peak_ratio", knee_peak_ratio)
    add("periodicity_symmetry", "knee_diff_fft_peak_hz", knee_peak_hz)
    add("periodicity_symmetry", "leg_lr_range_symmetry", 1.0 / (1.0 + abs(mean_abs_range(q, l_leg) - mean_abs_range(q, r_leg))))
    add("periodicity_symmetry", "arm_lr_range_symmetry", 1.0 / (1.0 + abs(mean_abs_range(q, l_arm) - mean_abs_range(q, r_arm))))
    left_foot_peak_ratio, left_foot_peak_hz = spectral_peak_features(left_foot_z, fk_fps)
    right_foot_peak_ratio, right_foot_peak_hz = spectral_peak_features(right_foot_z, fk_fps)
    foot_height_diff_peak_ratio, foot_height_diff_peak_hz = spectral_peak_features(left_foot_z - right_foot_z if left_foot_z.shape == right_foot_z.shape else np.zeros((0,), dtype=np.float64), fk_fps)
    add("periodicity_symmetry", "fk_left_foot_height_fft_peak_ratio", left_foot_peak_ratio)
    add("periodicity_symmetry", "fk_left_foot_height_fft_peak_hz", left_foot_peak_hz)
    add("periodicity_symmetry", "fk_right_foot_height_fft_peak_ratio", right_foot_peak_ratio)
    add("periodicity_symmetry", "fk_right_foot_height_fft_peak_hz", right_foot_peak_hz)
    add("periodicity_symmetry", "fk_foot_height_diff_fft_peak_ratio", foot_height_diff_peak_ratio)
    add("periodicity_symmetry", "fk_foot_height_diff_fft_peak_hz", foot_height_diff_peak_hz)
    add("periodicity_symmetry", "fk_foot_height_lr_corr", safe_corr(left_foot_z, right_foot_z))
    add("periodicity_symmetry", "fk_wrist_height_lr_corr", safe_corr(axis_values(left_wrist_w, 2), axis_values(right_wrist_w, 2)))

    out: dict[str, Any] = {
        "rel_path": meta.rel_path,
        "dataset": meta.dataset,
        "category": meta.category,
        "subcategory": meta.subcategory,
        "motion": meta.motion,
        "is_mirrored_name": bool(meta.motion.endswith("_M")),
        "num_frames": int(data.shape[0]),
        "duration_sec": duration,
    }
    out.update(features)
    return out


def discover_csvs(root: Path, pattern: str, limit: int | None) -> list[Path]:
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if limit is not None:
        paths = paths[: max(0, int(limit))]
    return paths


def load_score_map(score_root: Path | None, motion_root: Path) -> dict[str, dict[str, str]]:
    if score_root is None or not score_root.exists():
        return {}
    score_map: dict[str, dict[str, str]] = {}
    for score_file in sorted(score_root.rglob("physics_scores.csv")):
        with score_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path_text = row.get("path", "")
                rel: str | None = None
                if "/ok_only_csv/" in path_text:
                    rel = path_text.split("/ok_only_csv/", 1)[1]
                elif "/motions/" in path_text:
                    rel = path_text.split("/motions/", 1)[1]
                else:
                    try:
                        rel = Path(path_text).resolve().relative_to(motion_root).as_posix()
                    except Exception:
                        rel = None
                if rel:
                    score_map[rel] = row
    return score_map


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def robust_normalize(matrix: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.nanpercentile(matrix, low, axis=0)
    hi = np.nanpercentile(matrix, high, axis=0)
    denom = hi - lo
    valid = np.isfinite(denom) & (denom > 1e-12)
    norm = np.empty_like(matrix, dtype=np.float64)
    norm[:, valid] = np.clip((matrix[:, valid] - lo[valid]) / denom[valid], 0.0, 1.0)
    norm[:, ~valid] = 0.5
    norm[~np.isfinite(norm)] = 0.5
    return norm, lo, hi


def group_for_feature(feature: str) -> str:
    return feature.split("__", 1)[0]


def scaled_matrix(norm: np.ndarray, feature_names: list[str], weights: dict[str, float]) -> np.ndarray:
    scale = np.ones((len(feature_names),), dtype=np.float64)
    group_counts = {group: sum(1 for name in feature_names if group_for_feature(name) == group) for group in weights}
    for idx, name in enumerate(feature_names):
        group = group_for_feature(name)
        scale[idx] = weights.get(group, 0.0) / max(1, group_counts.get(group, 1))
    return norm * scale[None, :]


def compute_neighbors(
    norm: np.ndarray,
    feature_names: list[str],
    rows: list[dict[str, Any]],
    weights: dict[str, float],
    k: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if len(rows) <= 1:
        return [], np.zeros((len(rows),), dtype=np.float64)
    scaled = scaled_matrix(norm, feature_names, weights)
    tree = cKDTree(scaled)
    query_k = min(len(rows), k + 2)
    dists, inds = tree.query(scaled, k=query_k, p=1, workers=-1)
    if dists.ndim == 1:
        dists = dists[:, None]
        inds = inds[:, None]

    group_indices_map: dict[str, list[int]] = {
        group: [idx for idx, name in enumerate(feature_names) if group_for_feature(name) == group]
        for group in weights
    }

    neighbor_rows: list[dict[str, Any]] = []
    nearest_rank1 = np.full((len(rows),), np.nan, dtype=np.float64)
    for idx, row in enumerate(rows):
        rank = 0
        for dist, neighbor_idx in zip(dists[idx], inds[idx]):
            neighbor_idx = int(neighbor_idx)
            if neighbor_idx == idx:
                continue
            rank += 1
            if rank == 1:
                nearest_rank1[idx] = float(dist)
            pair: dict[str, Any] = {
                "rel_path": row["rel_path"],
                "dataset": row["dataset"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "motion": row["motion"],
                "neighbor_rank": rank,
                "neighbor_rel_path": rows[neighbor_idx]["rel_path"],
                "neighbor_dataset": rows[neighbor_idx]["dataset"],
                "neighbor_category": rows[neighbor_idx]["category"],
                "neighbor_subcategory": rows[neighbor_idx]["subcategory"],
                "neighbor_motion": rows[neighbor_idx]["motion"],
                "weighted_distance": float(dist),
                "same_dataset": row["dataset"] == rows[neighbor_idx]["dataset"],
                "same_category": row["category"] == rows[neighbor_idx]["category"],
                "same_subcategory": row["subcategory"] == rows[neighbor_idx]["subcategory"],
            }
            for group, indices in group_indices_map.items():
                if indices:
                    group_dist = float(np.mean(np.abs(norm[idx, indices] - norm[neighbor_idx, indices])))
                else:
                    group_dist = 0.0
                pair[f"group_distance_{group}"] = group_dist
                pair[f"group_weighted_{group}"] = group_dist * weights[group]
            neighbor_rows.append(pair)
            if rank >= k:
                break
    return neighbor_rows, nearest_rank1


def summarize_by_key(rows: list[dict[str, Any]], nearest: np.ndarray, key: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        buckets.setdefault(str(row[key]), []).append(idx)
    summary: list[dict[str, Any]] = []
    for bucket, indices in sorted(buckets.items()):
        vals = nearest[indices]
        vals = vals[np.isfinite(vals)]
        scores = np.array([float(rows[i].get("gqs_score") or np.nan) for i in indices], dtype=np.float64)
        summary.append(
            {
                key: bucket,
                "count": len(indices),
                "duration_mean_sec": safe_mean(np.array([float(rows[i]["duration_sec"]) for i in indices])),
                "duration_total_hr": safe_mean(np.array([float(rows[i]["duration_sec"]) for i in indices])) * len(indices) / 3600.0,
                "gqs_score_mean": safe_mean(scores[np.isfinite(scores)]),
                "nearest_p01": safe_percentile(vals, 1) if vals.size else 0.0,
                "nearest_p05": safe_percentile(vals, 5) if vals.size else 0.0,
                "nearest_p10": safe_percentile(vals, 10) if vals.size else 0.0,
                "nearest_p50": safe_percentile(vals, 50) if vals.size else 0.0,
                "nearest_p90": safe_percentile(vals, 90) if vals.size else 0.0,
                "nearest_p95": safe_percentile(vals, 95) if vals.size else 0.0,
            }
        )
    return summary


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 20) -> str:
    if not rows:
        return "_No rows._\n"
    rows = rows[:max_rows]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    feature_names: list[str],
    weights: dict[str, float],
    nearest_rank1: np.ndarray,
    neighbor_rows: list[dict[str, Any]],
    low: float,
    high: float,
    redundant_q: float,
    isolated_q: float,
) -> None:
    finite_nearest = nearest_rank1[np.isfinite(nearest_rank1)]
    global_p = {
        "p01": safe_percentile(finite_nearest, 1),
        "p05": safe_percentile(finite_nearest, 5),
        "p10": safe_percentile(finite_nearest, 10),
        "p50": safe_percentile(finite_nearest, 50),
        "p90": safe_percentile(finite_nearest, 90),
        "p95": safe_percentile(finite_nearest, 95),
        "p99": safe_percentile(finite_nearest, 99),
    }
    redundant_thr = safe_percentile(finite_nearest, redundant_q * 100.0)
    isolated_thr = safe_percentile(finite_nearest, isolated_q * 100.0)
    rank1_rows = [r for r in neighbor_rows if int(r["neighbor_rank"]) == 1]
    redundant = sorted(
        [r for r in rank1_rows if float(r["weighted_distance"]) <= redundant_thr],
        key=lambda r: float(r["weighted_distance"]),
    )[:20]
    redundant_all = [r for r in rank1_rows if float(r["weighted_distance"]) <= redundant_thr]
    redundant_same_subcategory = sum(1 for r in redundant_all if r["same_subcategory"] is True or str(r["same_subcategory"]) == "True")
    redundant_same_category = sum(1 for r in redundant_all if r["same_category"] is True or str(r["same_category"]) == "True")
    isolated_indices = np.argsort(-np.nan_to_num(nearest_rank1, nan=-1.0))[:20]
    isolated = [
        {
            "rel_path": rows[int(idx)]["rel_path"],
            "dataset": rows[int(idx)]["dataset"],
            "category": rows[int(idx)]["category"],
            "subcategory": rows[int(idx)]["subcategory"],
            "nearest_distance": float(nearest_rank1[int(idx)]),
            "gqs_score": rows[int(idx)].get("gqs_score", ""),
        }
        for idx in isolated_indices
        if np.isfinite(nearest_rank1[int(idx)])
    ]

    dataset_summary = summarize_by_key(rows, nearest_rank1, "dataset")
    category_summary = summarize_by_key(rows, nearest_rank1, "category")
    subcategory_summary = summarize_by_key(rows, nearest_rank1, "subcategory")
    category_by_redundancy = sorted(category_summary, key=lambda row: float(row["nearest_p50"]))
    dataset_by_redundancy = sorted(dataset_summary, key=lambda row: float(row["nearest_p50"]))
    most_redundant_category = category_by_redundancy[0] if category_by_redundancy else {}
    least_redundant_category = category_by_redundancy[-1] if category_by_redundancy else {}
    most_redundant_dataset = dataset_by_redundancy[0] if dataset_by_redundancy else {}
    least_redundant_dataset = dataset_by_redundancy[-1] if dataset_by_redundancy else {}

    lines = [
        "# Motion Diversity Diagnostic Report",
        "",
        "This report is diagnostic-only. No motion was selected, moved, deleted, or rejected by this analysis.",
        "",
        "## Scope",
        "",
        f"- Motions analyzed: {len(rows)}",
        f"- Feature count: {len(feature_names)}",
        f"- Robust normalization percentiles: p{low:g} to p{high:g}",
        "- Distance: group-wise weighted L1 over robust-normalized features.",
        f"- Spatial features: MuJoCo FK from H4 MJCF is computed on every {FK_FRAME_STRIDE}th source frame, i.e. 120 Hz CSVs contribute 30 Hz FK features without modifying source files.",
        "",
        "## Group Weights",
        "",
        markdown_table(
            [{"group": group, "weight": weight, "feature_count": sum(1 for f in feature_names if group_for_feature(f) == group)} for group, weight in weights.items()],
            ["group", "weight", "feature_count"],
            max_rows=20,
        ),
        "## Global Nearest-Neighbor Distance",
        "",
        markdown_table([{**global_p, "redundant_threshold": redundant_thr, "isolated_threshold": isolated_thr}], ["p01", "p05", "p10", "p50", "p90", "p95", "p99", "redundant_threshold", "isolated_threshold"], 5),
        "Interpretation: lower nearest distance means a motion is closer to another retained motion under the current feature/weight design.",
        "",
        "## Initial Observations",
        "",
        f"- The p{redundant_q * 100:g} redundancy band contains {len(redundant_all)} rank-1 pairs at distance <= {redundant_thr:.4f}.",
        f"- Inside that redundancy band, {redundant_same_category}/{len(redundant_all)} pairs are from the same category and {redundant_same_subcategory}/{len(redundant_all)} are from the same subcategory.",
        f"- Most redundant category by median nearest distance: `{most_redundant_category.get('category', '')}` (p50={float(most_redundant_category.get('nearest_p50', 0.0)):.4f}).",
        f"- Least redundant category by median nearest distance: `{least_redundant_category.get('category', '')}` (p50={float(least_redundant_category.get('nearest_p50', 0.0)):.4f}).",
        f"- Most redundant dataset by median nearest distance: `{most_redundant_dataset.get('dataset', '')}` (p50={float(most_redundant_dataset.get('nearest_p50', 0.0)):.4f}).",
        f"- Least redundant dataset by median nearest distance: `{least_redundant_dataset.get('dataset', '')}` (p50={float(least_redundant_dataset.get('nearest_p50', 0.0)):.4f}).",
        "- The closest examples are expected to include original/mirrored pairs and repeated idle/hold variants; these should be visually sampled before setting a final pruning threshold.",
        "",
        "## Dataset Summary",
        "",
        markdown_table(dataset_summary, ["dataset", "count", "duration_total_hr", "gqs_score_mean", "nearest_p05", "nearest_p50", "nearest_p95"], 20),
        "## Category Summary",
        "",
        markdown_table(category_summary, ["category", "count", "duration_total_hr", "gqs_score_mean", "nearest_p05", "nearest_p50", "nearest_p95"], 20),
        "## Subcategory Summary",
        "",
        markdown_table(subcategory_summary, ["subcategory", "count", "duration_total_hr", "gqs_score_mean", "nearest_p05", "nearest_p50", "nearest_p95"], 40),
        "## Most Redundant Rank-1 Pairs",
        "",
        markdown_table(redundant, ["rel_path", "neighbor_rel_path", "weighted_distance", "same_category", "same_subcategory"], 20),
        "## Most Isolated Motions",
        "",
        markdown_table(isolated, ["rel_path", "dataset", "category", "subcategory", "nearest_distance", "gqs_score"], 20),
        "## Next Use",
        "",
        "- Use `nearest_neighbors.csv` to inspect visually whether low-distance pairs are true duplicates or useful variations.",
        "- Tune the 7 group weights and rerun this script before deciding a final diversity threshold.",
        "- Operator labels are intentionally not used here; they should be intersected after the diversity policy is fixed.",
        "",
    ]
    (output_dir / "diversity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    motion_root = args.motion_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    score_root = args.score_root.expanduser().resolve() if args.score_root else None
    mjcf_path = None if args.no_mujoco_fk else args.mjcf.expanduser().resolve()
    weights = parse_group_weights(args.group_weights)
    if not motion_root.exists():
        raise FileNotFoundError(motion_root)
    if mjcf_path is not None and not mjcf_path.exists():
        raise FileNotFoundError(mjcf_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    csvs = discover_csvs(motion_root, args.glob, args.limit)
    if not csvs:
        raise SystemExit(f"No CSV files found under {motion_root}")

    print(f"[diversity] motions={len(csvs)} root={motion_root}")
    print(f"[diversity] output={output_dir}")
    print(f"[diversity] mjcf={mjcf_path if mjcf_path is not None else 'disabled'} fk_stride={FK_FRAME_STRIDE}")
    print(f"[diversity] weights={weights}")

    worker_args = [(path, motion_root, float(args.fps)) for path in csvs]
    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        init_fk_worker(str(mjcf_path) if mjcf_path is not None else None)
        for item in tqdm(worker_args, desc="extract features"):
            rows.append(extract_features_for_path(item))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_fk_worker,
            initargs=(str(mjcf_path) if mjcf_path is not None else None,),
        ) as executor:
            futures = [executor.submit(extract_features_for_path, item) for item in worker_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc="extract features"):
                rows.append(future.result())
    rows.sort(key=lambda row: row["rel_path"])

    score_map = load_score_map(score_root, motion_root)
    for row in rows:
        score = score_map.get(row["rel_path"])
        row["gqs_score"] = score.get("score", "") if score else ""
        row["gqs_passed"] = score.get("passed", "") if score else ""
        row["gqs_deduction_foot_sliding"] = score.get("deduction_foot_sliding", "") if score else ""
        row["gqs_deduction_self_collision"] = score.get("deduction_self_collision", "") if score else ""
        row["gqs_deduction_velocity_violation"] = score.get("deduction_velocity_violation", "") if score else ""
        row["gqs_deduction_jerk"] = score.get("deduction_jerk", "") if score else ""

    meta_fields = [
        "rel_path",
        "dataset",
        "category",
        "subcategory",
        "motion",
        "is_mirrored_name",
        "num_frames",
        "duration_sec",
        "gqs_score",
        "gqs_passed",
        "gqs_deduction_foot_sliding",
        "gqs_deduction_self_collision",
        "gqs_deduction_velocity_violation",
        "gqs_deduction_jerk",
    ]
    feature_names = sorted([name for name in rows[0] if "__" in name])
    matrix = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)
    norm, robust_lo, robust_hi = robust_normalize(matrix, float(args.robust_low), float(args.robust_high))

    feature_stats = []
    for idx, name in enumerate(feature_names):
        feature_stats.append(
            {
                "feature": name,
                "group": group_for_feature(name),
                "raw_p01": robust_lo[idx],
                "raw_p50": safe_percentile(matrix[:, idx], 50),
                "raw_p99": robust_hi[idx],
                "raw_mean": safe_mean(matrix[:, idx]),
                "raw_std": safe_std(matrix[:, idx]),
            }
        )

    write_csv(output_dir / "motion_diversity_features.csv", rows, meta_fields + feature_names)
    norm_rows = []
    for row_idx, row in enumerate(rows):
        norm_row = {field: row.get(field, "") for field in meta_fields}
        for feature_idx, name in enumerate(feature_names):
            norm_row[name] = float(norm[row_idx, feature_idx])
        norm_rows.append(norm_row)
    write_csv(output_dir / "motion_diversity_features_normalized.csv", norm_rows, meta_fields + feature_names)
    write_csv(output_dir / "feature_normalization_stats.csv", feature_stats, ["feature", "group", "raw_p01", "raw_p50", "raw_p99", "raw_mean", "raw_std"])

    neighbor_rows, nearest_rank1 = compute_neighbors(norm, feature_names, rows, weights, int(args.nearest_k))
    neighbor_fields = [
        "rel_path",
        "dataset",
        "category",
        "subcategory",
        "motion",
        "neighbor_rank",
        "neighbor_rel_path",
        "neighbor_dataset",
        "neighbor_category",
        "neighbor_subcategory",
        "neighbor_motion",
        "weighted_distance",
        "same_dataset",
        "same_category",
        "same_subcategory",
    ]
    for group in weights:
        neighbor_fields += [f"group_distance_{group}", f"group_weighted_{group}"]
    write_csv(output_dir / "nearest_neighbors.csv", neighbor_rows, neighbor_fields)

    for idx, row in enumerate(rows):
        row["nearest_rank1_distance"] = float(nearest_rank1[idx]) if np.isfinite(nearest_rank1[idx]) else ""
    write_csv(output_dir / "motion_diversity_features_with_nearest.csv", rows, meta_fields + ["nearest_rank1_distance"] + feature_names)

    category_summary = summarize_by_key(rows, nearest_rank1, "category")
    subcategory_summary = summarize_by_key(rows, nearest_rank1, "subcategory")
    dataset_summary = summarize_by_key(rows, nearest_rank1, "dataset")
    write_csv(output_dir / "dataset_summary.csv", dataset_summary, ["dataset", "count", "duration_mean_sec", "duration_total_hr", "gqs_score_mean", "nearest_p01", "nearest_p05", "nearest_p10", "nearest_p50", "nearest_p90", "nearest_p95"])
    write_csv(output_dir / "category_summary.csv", category_summary, ["category", "count", "duration_mean_sec", "duration_total_hr", "gqs_score_mean", "nearest_p01", "nearest_p05", "nearest_p10", "nearest_p50", "nearest_p90", "nearest_p95"])
    write_csv(output_dir / "subcategory_summary.csv", subcategory_summary, ["subcategory", "count", "duration_mean_sec", "duration_total_hr", "gqs_score_mean", "nearest_p01", "nearest_p05", "nearest_p10", "nearest_p50", "nearest_p90", "nearest_p95"])

    write_report(
        output_dir,
        rows,
        feature_names,
        weights,
        nearest_rank1,
        neighbor_rows,
        float(args.robust_low),
        float(args.robust_high),
        float(args.redundant_quantile),
        float(args.isolated_quantile),
    )
    metadata = {
        "motion_root": str(motion_root),
        "mjcf": str(mjcf_path) if mjcf_path is not None else None,
        "fk_frame_stride": FK_FRAME_STRIDE,
        "fk_effective_fps": float(args.fps) / FK_FRAME_STRIDE,
        "score_root": str(score_root) if score_root else None,
        "num_motions": len(rows),
        "fps": args.fps,
        "nearest_k": args.nearest_k,
        "robust_low": args.robust_low,
        "robust_high": args.robust_high,
        "group_weights": weights,
        "outputs": {
            "features": str(output_dir / "motion_diversity_features.csv"),
            "normalized_features": str(output_dir / "motion_diversity_features_normalized.csv"),
            "nearest_neighbors": str(output_dir / "nearest_neighbors.csv"),
            "report": str(output_dir / "diversity_report.md"),
        },
    }
    (output_dir / "diversity_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[diversity] wrote {output_dir / 'diversity_report.md'}")


if __name__ == "__main__":
    main()
