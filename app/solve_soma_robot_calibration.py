# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solve SOMA-to-robot scaler/offset configs from pose-pair samples.

Inputs are one-frame SOMA BVH files and matching robot pose JSON files. The
robot JSON can be exported by ``soma_robot_offset_calibrator.py`` using
``Save Pose Pair``; those files already include robot body FK transforms.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.utils.pose_utils as pose_utils
from soma_retargeter.utils.newton_utils import get_name_from_label
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROBOT_CONFIG_DEFAULTS = {
    "unitree_g1": (
        "unitree_g1/soma_to_g1_retargeter_config.json",
        "unitree_g1/soma_to_g1_scaler_config.json",
    ),
    "agile_one": (
        "agile_one/soma_to_agile_one_retargeter_config.json",
        "agile_one/soma_to_agile_one_scaler_config.json",
    ),
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def _resolve_config_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    repo_relative = (_REPO_ROOT / path).resolve()
    if repo_relative.exists():
        return repo_relative
    if len(path.parts) == 1:
        return io_utils.get_config_file(str(path)).resolve()
    return io_utils.get_config_file(*path.parts).resolve()


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def _average_quaternions_xyzw(quats: list[np.ndarray]) -> np.ndarray:
    if not quats:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    aligned = []
    ref = _quat_normalize(quats[0])
    for quat in quats:
        q = _quat_normalize(quat)
        if float(np.dot(ref, q)) < 0.0:
            q = -q
        aligned.append(q)
    m = np.zeros((4, 4), dtype=np.float64)
    for q in aligned:
        m += np.outer(q, q)
    eigvals, eigvecs = np.linalg.eigh(m)
    avg = eigvecs[:, int(np.argmax(eigvals))]
    if float(np.dot(ref, avg)) < 0.0:
        avg = -avg
    return _quat_normalize(avg)


def _rotation_error_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    return float(np.degrees((R.from_quat(q_a).inv() * R.from_quat(q_b)).magnitude()))


def _load_soma_global_transforms(path: Path, facing_direction: str) -> tuple[list[str], np.ndarray]:
    skeleton, animation = bvh_utils.load_bvh(str(path))
    if animation.num_frames != 1:
        raise ValueError(f"SOMA calibration pose must be a one-frame BVH: {path} has {animation.num_frames} frames")
    converter = SpaceConverter(get_facing_direction_type_from_str(facing_direction))
    root_tx = wp.transform(wp.vec3(0.0, 0.0, 0.0), converter.converter)
    global_tx = pose_utils.compute_global_pose(skeleton, animation.get_local_transforms(0), root_tx)
    return list(skeleton.joint_names), global_tx.astype(np.float64)


def _robot_body_transforms_from_pose_file(path: Path, robot_mjcf: Path | None = None) -> dict[str, list[float]]:
    data = _load_json(path)
    if isinstance(data.get("robot_body_transforms"), dict):
        return data["robot_body_transforms"]

    mjcf = robot_mjcf or Path(data.get("robot_mjcf", ""))
    if not mjcf:
        raise ValueError(f"Robot pose file has no robot_body_transforms and no robot_mjcf: {path}")
    mjcf = mjcf.expanduser().resolve()

    builder = newton.ModelBuilder()
    builder.add_mjcf(mjcf)
    model = builder.finalize()
    state = model.state()
    q = model.joint_q.numpy().astype(np.float32, copy=True)

    if "robot_joint_q" in data:
        raw_q = np.asarray(data["robot_joint_q"], dtype=np.float32)
        if len(raw_q) != len(q):
            raise ValueError(f"robot_joint_q length mismatch for {path}: {len(raw_q)} != {len(q)}")
        q[:] = raw_q
    else:
        if "robot_root_position" in data:
            q[:3] = np.asarray(data["robot_root_position"], dtype=np.float32)
        if "robot_root_rotation_xyzw" in data:
            q[3:7] = _quat_normalize(np.asarray(data["robot_root_rotation_xyzw"], dtype=np.float32))
        joints = data.get("robot_joints", {})
        if joints:
            q_start = model.joint_q_start.numpy()
            q_dim = np.diff(q_start)
            for joint_id, label in enumerate(model.joint_label):
                if q_dim[joint_id] != 1:
                    continue
                joint_name = get_name_from_label(label)
                if joint_name in joints:
                    q[int(q_start[joint_id])] = float(joints[joint_name])

    wp.copy(model.joint_q, wp.array(q, dtype=wp.float32), 0, 0, len(q))
    newton.eval_fk(model, model.joint_q, model.joint_qd, state, None)
    body_q = state.body_q.numpy()
    body_names = [get_name_from_label(label) for label in model.body_label]
    return {name: body_q[idx].astype(float).tolist() for idx, name in enumerate(body_names)}


def _build_pairs(args) -> list[tuple[Path, Path]]:
    if args.sample_files:
        pairs = []
        for sample in [Path(p).expanduser().resolve() for p in args.sample_files]:
            data = _load_json(sample)
            pairs.append((Path(data["soma_bvh"]).expanduser().resolve(), sample))
        return pairs

    if args.samples_dir:
        sample_files = sorted(Path(args.samples_dir).expanduser().resolve().glob("*.json"))
        pairs = []
        for sample in sample_files:
            data = _load_json(sample)
            if data.get("schema") != "soma_robot_pose_pair.v1" or "soma_bvh" not in data:
                continue
            pairs.append((Path(data["soma_bvh"]).expanduser().resolve(), sample))
        if not pairs:
            raise ValueError(f"No pose-pair JSON files found in {args.samples_dir}")
        return pairs

    soma_files = [Path(p).expanduser().resolve() for p in args.soma_files or []]
    robot_pose_files = [Path(p).expanduser().resolve() for p in args.robot_pose_files or []]
    if len(soma_files) != len(robot_pose_files):
        raise ValueError(f"--soma-files and --robot-pose-files must have equal length: {len(soma_files)} != {len(robot_pose_files)}")
    if not soma_files:
        raise ValueError("Provide --samples-dir, --sample-files, or equal-length --soma-files/--robot-pose-files")
    return list(zip(soma_files, robot_pose_files))


def _collect_observations(pairs, retargeter_config, args) -> dict:
    observations = {}
    for soma_path, robot_pose_path in pairs:
        joint_names, human_global = _load_soma_global_transforms(soma_path, args.soma_facing_direction)
        joint_index = {name: idx for idx, name in enumerate(joint_names)}
        robot_bodies = _robot_body_transforms_from_pose_file(robot_pose_path, Path(args.robot_mjcf).expanduser().resolve() if args.robot_mjcf else None)

        root_name = args.human_root_name
        if root_name not in joint_index:
            raise ValueError(f"Human root [{root_name}] not found in {soma_path}")
        human_root = human_global[joint_index[root_name]]
        root_mapping = retargeter_config.get("ik_map", {}).get(root_name, {})
        root_t_body = root_mapping.get("t_body")
        root_r_body = root_mapping.get("r_body")
        robot_root_position = (
            np.asarray(robot_bodies[root_t_body][:3], dtype=np.float64)
            if root_t_body in robot_bodies
            else None
        )
        robot_root_rotation = (
            _quat_normalize(np.asarray(robot_bodies[root_r_body][3:7], dtype=np.float64))
            if root_r_body in robot_bodies
            else None
        )

        for joint_name, mapping in retargeter_config.get("ik_map", {}).items():
            if joint_name not in joint_index:
                continue
            t_body = mapping["t_body"]
            r_body = mapping["r_body"]
            if t_body not in robot_bodies or r_body not in robot_bodies:
                continue
            observations.setdefault(joint_name, []).append(
                {
                    "soma_bvh": str(soma_path),
                    "robot_pose": str(robot_pose_path),
                    "human": human_global[joint_index[joint_name]],
                    "human_root": human_root,
                    "robot_root_position": robot_root_position,
                    "robot_root_rotation": robot_root_rotation,
                    "robot_position": np.asarray(robot_bodies[t_body][:3], dtype=np.float64),
                    "robot_rotation": _quat_normalize(np.asarray(robot_bodies[r_body][3:7], dtype=np.float64)),
                    "t_body": t_body,
                    "r_body": r_body,
                }
            )
    return observations


def _symmetric_partner(joint_name: str) -> str | None:
    if joint_name.startswith("Left"):
        return "Right" + joint_name[len("Left") :]
    if joint_name.startswith("Right"):
        return "Left" + joint_name[len("Right") :]
    return None


def _solve_offsets(observations: dict, scaler_config: dict, retargeter_config: dict, args) -> tuple[dict, dict]:
    human_height_assumption = float(scaler_config.get("human_height_assumption", 1.8))
    model_height = float(retargeter_config.get("model_height", human_height_assumption))
    height_ratio = model_height / human_height_assumption
    root_name = scaler_config.get("human_root_name", args.human_root_name)
    base_scales = scaler_config.get("joint_scales", {})
    root_scale_config = float(args.root_scale) if args.root_scale is not None else float(base_scales.get(root_name, 1.0))
    root_scale_eff = root_scale_config * height_ratio
    root_translation_offset_world = np.zeros(3, dtype=np.float64)
    root_align_positions = bool(getattr(args, "root_align_positions", True))

    solved_scaler = copy.deepcopy(scaler_config)
    solved_scaler["joint_scales"] = copy.deepcopy(scaler_config.get("joint_scales", {}))
    solved_scaler["joint_offsets"] = copy.deepcopy(scaler_config.get("joint_offsets", {}))

    report = {
        "num_pose_pairs": int(max((len(v) for v in observations.values()), default=0)),
        "root_scale_mode": "fixed" if args.root_scale is not None else "fit_root_z",
        "root_scale_config": root_scale_config,
        "root_position_alignment": root_align_positions,
        "root_position_offset_storage": "retargeter_config.ik_map.<root>.t_offset_world_xy",
        "height_ratio": height_ratio,
        "symmetric_scale_tying": True,
        "joints": [],
    }

    joint_scale_keys = list(solved_scaler.get("joint_scales", {}).keys())
    q_offset_by_joint = {}
    for joint_name in joint_scale_keys:
        obs = observations.get(joint_name, [])
        if not obs:
            continue
        q_offsets = []
        for item in obs:
            human_q = _quat_normalize(item["human"][3:7])
            robot_q = item["robot_rotation"]
            q_offsets.append((R.from_quat(human_q).inv() * R.from_quat(robot_q)).as_quat())
        q_offset_by_joint[joint_name] = _average_quaternions_xyzw(q_offsets)

    root_obs = observations.get(root_name, [])
    if root_obs and args.root_scale is None:
        human_z = np.asarray([item["human_root"][2] for item in root_obs], dtype=np.float64)
        robot_z = np.asarray([item["robot_position"][2] for item in root_obs], dtype=np.float64)
        valid = np.abs(human_z) > 1e-6
        if np.any(valid):
            unclamped_root_scale_eff = float(np.dot(human_z[valid], robot_z[valid]) / np.dot(human_z[valid], human_z[valid]))
            root_scale_eff = float(np.clip(unclamped_root_scale_eff, args.scale_min * height_ratio, args.scale_max * height_ratio))
            root_scale_config = float(root_scale_eff / height_ratio)
    if root_obs:
        xy_offsets = []
        for item in root_obs:
            h_root = item["human_root"][:3]
            xy_offsets.append(item["robot_position"][:2] - h_root[:2] * root_scale_eff)
        root_translation_offset_world[:2] = np.mean(np.asarray(xy_offsets), axis=0)
    report["root_scale_config"] = root_scale_config
    report["root_scale_effective"] = root_scale_eff
    report["root_translation_offset_world_xy"] = [float(x) for x in root_translation_offset_world]

    def refit_offset_for_scale(joint_name: str, scale_eff: float) -> np.ndarray:
        offsets = []
        q_offset = q_offset_by_joint[joint_name]
        for item in observations[joint_name]:
            human = item["human"]
            h_root = item["human_root"][:3]
            h_q = _quat_normalize(human[3:7])
            target_rot = R.from_quat(h_q) * R.from_quat(q_offset)
            if joint_name == root_name:
                return root_translation_offset_world.copy()
            else:
                if root_align_positions and item.get("robot_root_position") is not None:
                    root_base = item["robot_root_position"]
                else:
                    root_base = h_root * root_scale_eff
                b = item["robot_position"] - root_base - scale_eff * (human[:3] - h_root)
            offsets.append(target_rot.inv().apply(b))
        return np.mean(np.asarray(offsets), axis=0)

    def compute_errors(joint_name: str, scale_eff: float, offset_t: np.ndarray) -> tuple[list[float], list[float]]:
        pos_errors = []
        rot_errors = []
        q_offset = q_offset_by_joint[joint_name]
        for item in observations[joint_name]:
            human = item["human"]
            h_root = item["human_root"][:3]
            target_rot = R.from_quat(_quat_normalize(human[3:7])) * R.from_quat(q_offset)
            if joint_name == root_name:
                pred_pos = h_root * root_scale_eff + root_translation_offset_world
            else:
                if root_align_positions and item.get("robot_root_position") is not None:
                    root_base = item["robot_root_position"]
                else:
                    root_base = h_root * root_scale_eff
                pred_pos = (human[:3] - h_root) * scale_eff + root_base + target_rot.apply(offset_t)
            pred_q = target_rot.as_quat()
            pos_errors.append(float(np.linalg.norm(pred_pos - item["robot_position"])))
            rot_errors.append(_rotation_error_deg(pred_q, item["robot_rotation"]))
        return pos_errors, rot_errors

    def write_solution(joint_name: str, scale_eff: float, offset_t: np.ndarray, scale_group: str) -> None:
        q_offset = q_offset_by_joint[joint_name]
        scale_config = float(scale_eff / height_ratio)
        solved_scaler["joint_scales"][joint_name] = scale_config
        solved_scaler.setdefault("joint_offsets", {})[joint_name] = [
            [float(x) for x in offset_t],
            [float(x) for x in q_offset],
        ]
        pos_errors, rot_errors = compute_errors(joint_name, scale_eff, offset_t)
        report["joints"].append(
            {
                "joint": joint_name,
                "status": "solved",
                "samples": len(observations[joint_name]),
                "scale_group": scale_group,
                "scale_config": scale_config,
                "scale_effective": scale_eff,
                "translation_offset": [float(x) for x in offset_t],
                "rotation_offset_xyzw": [float(x) for x in q_offset],
                "mean_position_error_m": float(np.mean(pos_errors)),
                "max_position_error_m": float(np.max(pos_errors)),
                "mean_rotation_error_deg": float(np.mean(rot_errors)),
                "max_rotation_error_deg": float(np.max(rot_errors)),
            }
        )

    visited = set()
    for joint_name in joint_scale_keys:
        if joint_name in visited:
            continue
        obs = observations.get(joint_name, [])
        if not obs:
            report["joints"].append({"joint": joint_name, "status": "missing_observations"})
            visited.add(joint_name)
            continue

        partner = _symmetric_partner(joint_name)
        if (
            partner
            and partner in joint_scale_keys
            and partner not in visited
            and partner in observations
            and joint_name != root_name
        ):
            rows = []
            rhs = []
            for side_idx, side_joint in enumerate((joint_name, partner)):
                q_offset = q_offset_by_joint[side_joint]
                offset_col_start = 1 + side_idx * 3
                for item in observations[side_joint]:
                    human = item["human"]
                    h_root = item["human_root"][:3]
                    a = human[:3] - h_root
                    h_q = _quat_normalize(human[3:7])
                    target_rot = R.from_quat(h_q) * R.from_quat(q_offset)
                    if root_align_positions and item.get("robot_root_position") is not None:
                        root_base = item["robot_root_position"]
                    else:
                        root_base = h_root * root_scale_eff
                    b = item["robot_position"] - root_base
                    block = np.zeros((3, 7), dtype=np.float64)
                    block[:, 0] = a
                    block[:, offset_col_start : offset_col_start + 3] = target_rot.as_matrix()
                    rows.append(block)
                    rhs.append(b)
            mat = np.vstack(rows)
            y = np.concatenate(rhs)
            solution, residuals, rank, singular_values = np.linalg.lstsq(mat, y, rcond=None)
            scale_eff = float(solution[0])
            left_offset = solution[1:4]
            right_offset = solution[4:7]
            unclamped_scale_eff = scale_eff
            scale_eff = float(np.clip(scale_eff, args.scale_min * height_ratio, args.scale_max * height_ratio))
            if abs(scale_eff - unclamped_scale_eff) > 1e-9:
                left_offset = refit_offset_for_scale(joint_name, scale_eff)
                right_offset = refit_offset_for_scale(partner, scale_eff)

            group_name = f"{joint_name}<->{partner}"
            write_solution(joint_name, scale_eff, left_offset, group_name)
            write_solution(partner, scale_eff, right_offset, group_name)
            visited.add(joint_name)
            visited.add(partner)
            continue

        q_offset = q_offset_by_joint[joint_name]

        if joint_name == root_name:
            scale_eff = root_scale_eff
            offset_t = refit_offset_for_scale(joint_name, scale_eff)
        else:
            rows = []
            rhs = []
            for item in obs:
                human = item["human"]
                h_root = item["human_root"][:3]
                a = human[:3] - h_root
                h_q = _quat_normalize(human[3:7])
                target_rot = R.from_quat(h_q) * R.from_quat(q_offset)
                qmat = target_rot.as_matrix()
                if root_align_positions and item.get("robot_root_position") is not None:
                    root_base = item["robot_root_position"]
                else:
                    root_base = h_root * root_scale_eff
                b = item["robot_position"] - root_base
                block = np.zeros((3, 4), dtype=np.float64)
                block[:, 0] = a
                block[:, 1:4] = qmat
                rows.append(block)
                rhs.append(b)
            mat = np.vstack(rows)
            y = np.concatenate(rhs)
            solution, residuals, rank, singular_values = np.linalg.lstsq(mat, y, rcond=None)
            scale_eff = float(solution[0])
            offset_t = solution[1:4]
            unclamped_scale_eff = scale_eff
            scale_eff = float(np.clip(scale_eff, args.scale_min * height_ratio, args.scale_max * height_ratio))
            if abs(scale_eff - unclamped_scale_eff) > 1e-9:
                # Refit local offset after clamping scale.
                offset_t = refit_offset_for_scale(joint_name, scale_eff)

        write_solution(joint_name, scale_eff, offset_t, joint_name)
        visited.add(joint_name)

    solved_retargeter = copy.deepcopy(retargeter_config)
    root_offset_entry = solved_scaler.get("joint_offsets", {}).get(root_name)
    if root_offset_entry:
        root_translation_offset = [float(x) for x in root_translation_offset_world]
        solved_retargeter.setdefault("ik_map", {}).setdefault(root_name, {})["t_offset"] = root_translation_offset
        init_state = solved_retargeter.get("initial_robot_joint_positions")
        if isinstance(init_state, dict) and "root_position" in init_state:
            root_position = np.asarray(init_state["root_position"], dtype=np.float64)
            if root_position.shape == (3,):
                init_state["root_position"] = [float(x) for x in root_position - np.asarray(root_translation_offset)]
                report["initial_root_position_after_offset"] = init_state["root_position"]
        # Runtime applies this horizontal root offset globally from the
        # retargeter config. Keep the scaler's root translation zero to avoid
        # double-counting and to ensure root Z is controlled only by scale.
        solved_scaler["joint_offsets"][root_name][0] = [0.0, 0.0, 0.0]
        report["root_translation_offset"] = root_translation_offset
    return solved_scaler, solved_retargeter, report


def _write_report_md(path: Path, report: dict, scaler_path: Path, retargeter_path: Path) -> None:
    lines = [
        "# SOMA Robot Calibration Solve Report",
        "",
        f"- Pose pairs: {report['num_pose_pairs']}",
        f"- Root scale config: {report['root_scale_config']:.6f}",
        f"- Root scale effective: {report.get('root_scale_effective', 0.0):.6f}",
        f"- Root position alignment: `{report.get('root_position_alignment', False)}`",
        f"- Root position offset storage: `{report.get('root_position_offset_storage', 'scaler')}`",
        f"- Root translation offset: `{report.get('root_translation_offset', [0.0, 0.0, 0.0])}`",
        f"- Initial root position after offset: `{report.get('initial_root_position_after_offset', None)}`",
        f"- Height ratio: {report['height_ratio']:.6f}",
        f"- Scaler config: `{scaler_path}`",
        f"- Retargeter config: `{retargeter_path}`",
        "",
        "| Joint | Status | Samples | Scale | Mean Pos m | Max Pos m | Mean Rot deg | Max Rot deg |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["joints"]:
        lines.append(
            f"| {item['joint']} | {item['status']} | {item.get('samples', 0)} | "
            f"{item.get('scale_config', 0.0):.5f} | {item.get('mean_position_error_m', 0.0):.4f} | "
            f"{item.get('max_position_error_m', 0.0):.4f} | {item.get('mean_rotation_error_deg', 0.0):.2f} | "
            f"{item.get('max_rotation_error_deg', 0.0):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-type", choices=["unitree_g1", "agile_one"], required=True)
    parser.add_argument("--samples-dir", type=str, default=None)
    parser.add_argument("--sample-files", nargs="*", default=None)
    parser.add_argument("--soma-files", nargs="*", default=None)
    parser.add_argument("--robot-pose-files", nargs="*", default=None)
    parser.add_argument("--robot-mjcf", type=str, default=None)
    parser.add_argument("--base-retargeter-config", type=str, default=None)
    parser.add_argument("--base-scaler-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(_REPO_ROOT / "output/calibration_solve"))
    parser.add_argument("--soma-facing-direction", choices=["Mujoco", "Maya"], default="Mujoco")
    parser.add_argument("--human-root-name", type=str, default="Hips")
    parser.add_argument("--root-scale", type=float, default=None, help="Optional fixed config-space root scale. Defaults to base scaler Hips.")
    parser.add_argument(
        "--root-align-positions",
        dest="root_align_positions",
        action="store_true",
        default=True,
        help="Use the observed robot root position as the base before solving non-root position offsets.",
    )
    parser.add_argument(
        "--no-root-align-positions",
        dest="root_align_positions",
        action="store_false",
        help="Use scaled human root position as the non-root base, matching the old solve behavior.",
    )
    parser.add_argument("--scale-min", type=float, default=0.3)
    parser.add_argument("--scale-max", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_retargeter, default_scaler = _ROBOT_CONFIG_DEFAULTS[args.robot_type]
    retargeter_path = _resolve_config_path(args.base_retargeter_config or default_retargeter)
    retargeter_config = _load_json(retargeter_path)
    scaler_path = _resolve_config_path(args.base_scaler_config or retargeter_config.get("human_robot_scaler_config") or default_scaler)
    scaler_config = _load_json(scaler_path)

    pairs = _build_pairs(args)
    print(f"[calibration] pose pairs: {len(pairs)}")
    observations = _collect_observations(pairs, retargeter_config, args)
    solved_scaler, solved_retargeter, report = _solve_offsets(observations, scaler_config, retargeter_config, args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_scaler = output_dir / f"{args.robot_type}_recommended_scaler_config.json"
    out_retargeter = output_dir / f"{args.robot_type}_recommended_retargeter_config.json"
    out_report_json = output_dir / "calibration_solve_report.json"
    out_report_md = output_dir / "calibration_solve_report.md"

    solved_scaler["robot_type"] = args.robot_type
    solved_retargeter["human_robot_scaler_config"] = str(out_scaler)
    _write_json(out_scaler, solved_scaler)
    _write_json(out_retargeter, solved_retargeter)
    _write_json(out_report_json, report)
    _write_report_md(out_report_md, report, out_scaler, out_retargeter)

    solved = [item for item in report["joints"] if item["status"] == "solved"]
    print(f"[calibration] solved joints: {len(solved)}")
    print(f"[calibration] scaler: {out_scaler}")
    print(f"[calibration] retargeter: {out_retargeter}")
    print(f"[calibration] report: {out_report_md}")


if __name__ == "__main__":
    main()
