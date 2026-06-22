#!/usr/bin/env python3
"""Classify IK branch-switch risk from same-chain FK link-frame rotation.

This is intentionally separate from the older joint-diff classifier.  The
core assumption is stricter: a true IK branch switch must be supported by
evidence inside one kinematic chain.  Cross-chain synchronized spikes are
reported for review, but are not hard-rejected by themselves.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


ROOT_POSITION_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")
ROOT_ROTATION_COLUMNS = ("root_rotateX", "root_rotateY", "root_rotateZ")


CHAIN_BODIES: dict[str, dict[str, list[str]]] = {
    "left_arm": {
        "intermediate": [
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_roll_link",
        ],
        "distal": ["left_wrist_yaw_link", "left_wrist_roll_link", "left_wrist_pitch_link"],
    },
    "right_arm": {
        "intermediate": [
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_roll_link",
        ],
        "distal": ["right_wrist_yaw_link", "right_wrist_roll_link", "right_wrist_pitch_link"],
    },
    "left_leg": {
        "intermediate": [
            "left_hip_pitch_link",
            "left_hip_roll_link",
            "left_hip_yaw_link",
            "left_knee_link",
        ],
        "distal": ["left_ankle_roll_link", "left_ankle_pitch_link"],
    },
    "right_leg": {
        "intermediate": [
            "right_hip_pitch_link",
            "right_hip_roll_link",
            "right_hip_yaw_link",
            "right_knee_link",
        ],
        "distal": ["right_ankle_roll_link", "right_ankle_pitch_link"],
    },
}


_MUJOCO = None
_MODEL = None
_BODY_IDS: dict[str, int] = {}


@dataclass(frozen=True)
class ChainEvidence:
    chain: str
    risk: str
    reason: str
    frame0: int
    frame1: int
    persistent_score: int
    transient_score: int
    high_root_dynamic: bool
    root_z_range_2s_m: float
    root_vz_abs_max_2s_m_s: float
    max_intermediate_adj_deg_s: float
    max_distal_adj_deg_s: float
    max_intermediate_w24_deg: float
    max_distal_w24_deg: float
    max_intermediate_w48_deg: float
    max_distal_w48_deg: float
    distal_min_after_48_240_deg: float
    distal_event_outlier_ratio: float
    distal_position_pre_post_m: float
    distal_position_path_72_m: float
    distal_position_max_step_m: float
    distal_rot_per_path_deg_per_m: float
    top_intermediate_body: str
    top_distal_body: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, default=Path("/home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml"))
    parser.add_argument("--glob", default="**/*.csv")
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-pending",
        type=int,
        default=None,
        help="Maximum submitted-but-not-yet-collected jobs. Defaults to workers*4.",
    )
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument(
        "--include-list",
        type=Path,
        help="Optional newline-delimited relative CSV paths to process under motion-root.",
    )

    # Upper-chain stealth switch thresholds.  These catch smooth shoulder/arm
    # branch transitions that may stay below a simple per-frame joint threshold.
    parser.add_argument("--upper-intermediate-adj-deg-s", type=float, default=180.0)
    parser.add_argument("--upper-distal-adj-deg-s", type=float, default=500.0)
    parser.add_argument("--upper-spike-intermediate-adj-deg-s", type=float, default=400.0)
    parser.add_argument("--upper-spike-distal-adj-deg-s", type=float, default=650.0)
    parser.add_argument("--upper-intermediate-w24-deg", type=float, default=25.0)
    parser.add_argument("--upper-distal-w24-deg", type=float, default=65.0)
    parser.add_argument("--upper-distal-w48-deg", type=float, default=105.0)
    parser.add_argument("--persistent-max-distal-path-m", type=float, default=0.45)
    parser.add_argument("--persistent-max-distal-step-m", type=float, default=0.012)
    parser.add_argument("--persistent-min-distal-rot-per-path-deg-m", type=float, default=900.0)

    # Lower-chain hard thresholds are stricter because hopping/running can
    # produce large ankle motion without an IK branch switch.
    parser.add_argument("--lower-intermediate-adj-deg-s", type=float, default=500.0)
    parser.add_argument("--lower-distal-adj-deg-s", type=float, default=650.0)
    parser.add_argument("--lower-intermediate-w24-deg", type=float, default=55.0)
    parser.add_argument("--lower-distal-w24-deg", type=float, default=80.0)

    # Review-only thresholds: visible in reports but not hard-rejected without
    # same-chain intermediate evidence.
    parser.add_argument("--review-distal-adj-deg-s", type=float, default=360.0)
    parser.add_argument("--review-distal-w24-deg", type=float, default=45.0)
    return parser.parse_args()


def init_worker(mjcf_path: str) -> None:
    global _MUJOCO, _MODEL, _BODY_IDS
    import mujoco

    _MUJOCO = mujoco
    _MODEL = mujoco.MjModel.from_xml_path(str(Path(mjcf_path).expanduser().resolve()))
    _BODY_IDS = {}
    for groups in CHAIN_BODIES.values():
        for names in groups.values():
            for name in names:
                body_id = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    _BODY_IDS[name] = int(body_id)


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.where(norm < 1e-8, 1.0, norm)


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.where(norm < 1e-8, 1.0, norm)


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quat_angle_deg_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat_wxyz(quat)
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)
    w = np.clip(quat[..., 0], -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(w))


def canonicalize_quat_sequence_wxyz(quat: np.ndarray) -> np.ndarray:
    out = normalize_quat_wxyz(np.asarray(quat, dtype=np.float64)).copy()
    for i in range(1, out.shape[0]):
        dots = np.sum(out[i - 1] * out[i], axis=-1)
        out[i, dots < 0.0] *= -1.0
    return out


def infer_dataset_subset(motion_root: Path, csv_path: Path) -> tuple[str, str]:
    rel = csv_path.relative_to(motion_root)
    parts = rel.parts
    if len(parts) <= 1:
        return "nutan", "nutan"
    dataset = parts[0]
    if dataset in {"seed", "grab"} and len(parts) > 1:
        return dataset, parts[1]
    return dataset, ""


def load_csv_motion(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding=None)
    if data.shape == ():
        data = data.reshape(1)
    names = data.dtype.names
    if names is None:
        raise ValueError(f"{path} has no CSV header")
    missing = [name for name in ROOT_POSITION_COLUMNS + ROOT_ROTATION_COLUMNS if name not in names]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    root_pos = np.stack([data[name] for name in ROOT_POSITION_COLUMNS], axis=1) / 100.0
    root_euler = np.stack([data[name] for name in ROOT_ROTATION_COLUMNS], axis=1)
    root_quat_xyzw = R.from_euler("xyz", root_euler, degrees=True).as_quat()
    joint_cols = [name for name in names if name.endswith("_dof")]
    joint_names = [name[: -len("_dof")] for name in joint_cols]
    joint_pos = np.deg2rad(np.stack([data[name] for name in joint_cols], axis=1))
    return root_pos, root_quat_xyzw, joint_names, joint_pos


def make_model_qpos(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_names: list[str],
    joint_pos: np.ndarray,
) -> np.ndarray:
    if _MUJOCO is None or _MODEL is None:
        raise RuntimeError("MuJoCo worker is not initialized")
    model = _MODEL
    mujoco = _MUJOCO
    free_joint_ids = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    if len(free_joint_ids) != 1:
        raise ValueError(f"Expected one free joint, found {len(free_joint_ids)}")
    free_qpos_addr = int(model.jnt_qposadr[int(free_joint_ids[0])])
    qpos = np.tile(np.asarray(model.qpos0, dtype=np.float64), (root_pos.shape[0], 1))
    qpos[:, free_qpos_addr : free_qpos_addr + 3] = root_pos
    root_quat_xyzw = normalize_quat_xyzw(root_quat_xyzw)
    qpos[:, free_qpos_addr + 3 : free_qpos_addr + 7] = root_quat_xyzw[:, [3, 0, 1, 2]]

    missing: list[str] = []
    for joint_idx, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            missing.append(joint_name)
            continue
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        qpos[:, int(model.jnt_qposadr[joint_id])] = joint_pos[:, joint_idx]
    if missing:
        raise ValueError(f"MJCF missing joints: {missing}")
    return qpos


def compute_relative_body_state(qpos: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if _MUJOCO is None or _MODEL is None:
        raise RuntimeError("MuJoCo worker is not initialized")
    mujoco = _MUJOCO
    model = _MODEL
    data = mujoco.MjData(model)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    if pelvis_id < 0:
        raise ValueError("MJCF has no pelvis_link body")
    body_quats: dict[str, list[np.ndarray]] = {name: [] for name in _BODY_IDS}
    body_pos: dict[str, list[np.ndarray]] = {name: [] for name in _BODY_IDS}
    for frame in range(qpos.shape[0]):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        pelvis_inv = quat_conjugate_wxyz(data.xquat[pelvis_id])
        pelvis_pos = data.xpos[pelvis_id].copy()
        for name, body_id in _BODY_IDS.items():
            rel = quat_multiply_wxyz(pelvis_inv, data.xquat[body_id])
            body_quats[name].append(rel)
            body_pos[name].append(data.xpos[body_id].copy() - pelvis_pos)
    return (
        {name: canonicalize_quat_sequence_wxyz(np.asarray(values)) for name, values in body_quats.items()},
        {name: np.asarray(values, dtype=np.float64) for name, values in body_pos.items()},
    )


def adjacent_rot_vel_deg_s(quat: np.ndarray, fps: float) -> np.ndarray:
    if quat.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    delta = quat_multiply_wxyz(quat[1:], quat_conjugate_wxyz(quat[:-1]))
    return quat_angle_deg_wxyz(delta) * fps


def window_rot_deg(quat: np.ndarray, window: int) -> np.ndarray:
    if quat.shape[0] <= window:
        return np.zeros((0,), dtype=np.float64)
    delta = quat_multiply_wxyz(quat[window:], quat_conjugate_wxyz(quat[:-window]))
    return quat_angle_deg_wxyz(delta)


def max_series(series_by_body: dict[str, np.ndarray], body_names: list[str]) -> tuple[float, int, str]:
    best_val = 0.0
    best_frame = 0
    best_body = ""
    for body in body_names:
        values = series_by_body.get(body)
        if values is None or values.size == 0:
            continue
        idx = int(np.argmax(values))
        val = float(values[idx])
        if val > best_val:
            best_val = val
            best_frame = idx
            best_body = body
    return best_val, best_frame, best_body


def local_max_value(values: np.ndarray, frame: int, radius: int) -> float:
    if values.size == 0:
        return 0.0
    lo = max(0, frame - radius)
    hi = min(values.size, frame + radius + 1)
    return float(np.max(values[lo:hi])) if hi > lo else 0.0


def local_longest_run(values: np.ndarray, threshold: float, frame: int, radius: int) -> int:
    if values.size == 0:
        return 0
    lo = max(0, frame - radius)
    hi = min(values.size, frame + radius + 1)
    best = 0
    cur = 0
    for hit in values[lo:hi] >= threshold:
        if bool(hit):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def max_local_series(
    series_by_body: dict[str, np.ndarray],
    body_names: list[str],
    frame: int,
    radius: int,
) -> tuple[float, str]:
    best_val = 0.0
    best_body = ""
    for body in body_names:
        values = series_by_body.get(body)
        if values is None:
            continue
        val = local_max_value(values, frame, radius)
        if val > best_val:
            best_val = val
            best_body = body
    return best_val, best_body


def angle_between_rel_frames(quat: np.ndarray, a: int, b: int) -> float:
    if quat.shape[0] == 0:
        return 0.0
    a = int(np.clip(a, 0, quat.shape[0] - 1))
    b = int(np.clip(b, 0, quat.shape[0] - 1))
    delta = quat_multiply_wxyz(quat[b], quat_conjugate_wxyz(quat[a]))
    return float(quat_angle_deg_wxyz(delta))


def min_angle_to_pre_after(quat: np.ndarray, pre: int, lo: int, hi: int) -> float:
    if quat.shape[0] == 0:
        return 0.0
    pre = int(np.clip(pre, 0, quat.shape[0] - 1))
    lo = max(0, min(lo, quat.shape[0] - 1))
    hi = max(lo + 1, min(hi, quat.shape[0]))
    values = [angle_between_rel_frames(quat, pre, frame) for frame in range(lo, hi)]
    return float(min(values)) if values else 0.0


def position_window_features(
    pos: np.ndarray | None,
    center: int,
    pre_radius: int = 24,
    post_radius: int = 48,
) -> tuple[float, float, float]:
    if pos is None or pos.shape[0] == 0:
        return 0.0, 0.0, 0.0
    pre = int(np.clip(center - pre_radius, 0, pos.shape[0] - 1))
    post = int(np.clip(center + post_radius, 0, pos.shape[0] - 1))
    if post <= pre:
        return 0.0, 0.0, 0.0
    seg = pos[pre : post + 1]
    steps = np.linalg.norm(np.diff(seg, axis=0), axis=1) if seg.shape[0] >= 2 else np.zeros((0,), dtype=np.float64)
    pre_post = float(np.linalg.norm(pos[post] - pos[pre]))
    path_len = float(np.sum(steps)) if steps.size else 0.0
    max_step = float(np.max(steps)) if steps.size else 0.0
    return pre_post, path_len, max_step


def root_dynamic_around(root_pos: np.ndarray, frame: int, fps: float, radius: int = 120) -> tuple[bool, float, float]:
    lo = max(0, frame - radius)
    hi = min(root_pos.shape[0], frame + radius + 1)
    root_z = root_pos[lo:hi, 2]
    root_vz = np.diff(root_z) * fps if root_z.size >= 2 else np.zeros((0,), dtype=np.float64)
    z_range = float(np.ptp(root_z)) if root_z.size else 0.0
    vz_abs_max = float(np.max(np.abs(root_vz))) if root_vz.size else 0.0
    return z_range >= 0.12 or vz_abs_max >= 1.0, z_range, vz_abs_max


def classify_chain(
    chain: str,
    rel_quats: dict[str, np.ndarray],
    rel_pos: dict[str, np.ndarray],
    root_pos: np.ndarray,
    fps: float,
    ns: argparse.Namespace,
) -> ChainEvidence:
    bodies = CHAIN_BODIES[chain]
    intermediate = [b for b in bodies["intermediate"] if b in rel_quats]
    distal = [b for b in bodies["distal"] if b in rel_quats]
    adj = {body: adjacent_rot_vel_deg_s(rel_quats[body], fps) for body in intermediate + distal}
    w24 = {body: window_rot_deg(rel_quats[body], 24) for body in intermediate + distal}
    w48 = {body: window_rot_deg(rel_quats[body], 48) for body in intermediate + distal}

    int_adj, int_adj_frame, int_adj_body = max_series(adj, intermediate)
    dist_adj, dist_adj_frame, dist_adj_body = max_series(adj, distal)
    int_w24, int_w24_frame, int_w24_body = max_series(w24, intermediate)
    dist_w24, dist_w24_frame, dist_w24_body = max_series(w24, distal)
    int_w48, int_w48_frame, _ = max_series(w48, intermediate)
    dist_w48, dist_w48_frame, _ = max_series(w48, distal)

    is_arm = chain.endswith("_arm")
    persistent_score = 0
    transient_score = 0
    high_root_dynamic = False
    root_z_range = 0.0
    root_vz_abs_max = 0.0
    distal_min_after = 0.0
    distal_event_outlier_ratio = 0.0
    distal_position_pre_post = 0.0
    distal_position_path = 0.0
    distal_position_max_step = 0.0
    distal_rot_per_path = 0.0
    frame = int_adj_frame
    reason = "none"
    risk = "clean"
    if is_arm:
        num_frames = next(iter(rel_quats.values())).shape[0] if rel_quats else 0
        candidates = {
            int(np.clip(int_adj_frame + 1, 0, max(0, num_frames - 1))),
            int(np.clip(dist_adj_frame + 1, 0, max(0, num_frames - 1))),
            int(np.clip(int_w24_frame + 12, 0, max(0, num_frames - 1))),
            int(np.clip(dist_w24_frame + 12, 0, max(0, num_frames - 1))),
            int(np.clip(int_w48_frame + 24, 0, max(0, num_frames - 1))),
            int(np.clip(dist_w48_frame + 24, 0, max(0, num_frames - 1))),
        }
        best: dict | None = None
        for cand in sorted(candidates):
            dist_v_event, dist_v_body = max_local_series(adj, distal, cand, 3)
            int_v_event, int_v_body = max_local_series(adj, intermediate, cand, 3)
            dist_w48_event, dist_w48_body = max_local_series(w48, distal, cand, 72)
            if not dist_w48_body:
                dist_w48_body = dist_v_body
            int_w24_event, int_w24_body = max_local_series(w24, intermediate, cand, 48)
            dist_w24_event = local_max_value(w24.get(dist_w48_body, np.zeros((0,), dtype=np.float64)), cand, 48)

            dist_adj_for_event = adj.get(dist_v_body, np.zeros((0,), dtype=np.float64))
            dist_p99 = float(np.percentile(dist_adj_for_event, 99)) if dist_adj_for_event.size else 0.0
            dist_outlier_ratio = dist_v_event / max(dist_p99, 1e-6)
            dist_a = np.diff(dist_adj_for_event) * fps if dist_adj_for_event.size >= 2 else np.zeros((0,), dtype=np.float64)
            dist_j = np.diff(dist_a) * fps if dist_a.size >= 2 else np.zeros((0,), dtype=np.float64)

            dist_quat = rel_quats.get(dist_w48_body)
            int_quat = rel_quats.get(int_w24_body)
            pre = cand - 24
            min_after = min_angle_to_pre_after(dist_quat, pre, cand + 48, cand + 240) if dist_quat is not None else 0.0
            int_pre_post = angle_between_rel_frames(int_quat, pre, cand + 48) if int_quat is not None else 0.0
            high_root, local_root_z, local_root_vz = root_dynamic_around(root_pos, cand, fps)

            persistent_core = (
                dist_w48_event >= 95.0
                and dist_w24_event >= 65.0
                and int_w24_event >= 30.0
            )
            persistent_stay = min_after >= 80.0
            persistent_intermediate = int_pre_post >= 25.0
            dist_pos_pre_post, dist_pos_path, dist_pos_max_step = position_window_features(rel_pos.get(dist_w48_body), cand)
            dist_rot_per_path = dist_w48_event / max(dist_pos_path, 1e-6)
            persistent_spatial = (
                (
                    dist_pos_path <= ns.persistent_max_distal_path_m
                    and dist_pos_max_step <= ns.persistent_max_distal_step_m
                )
                or dist_rot_per_path >= ns.persistent_min_distal_rot_per_path_deg_m
            )
            p_score = int(persistent_core) + int(persistent_stay) + int(persistent_intermediate)

            t_score = 0
            t_score += int(dist_v_event >= 650.0)
            t_score += int(int_v_event >= 350.0)
            t_score += int(local_max_value(np.abs(dist_a), cand, 6) >= 30000.0)
            t_score += int(local_max_value(np.abs(dist_j), cand, 6) >= 3_000_000.0)
            t_score += int(local_longest_run(dist_adj_for_event, 500.0, cand, 12) >= 3)
            transient_outlier = dist_outlier_ratio >= 5.0

            hard_persistent = persistent_core and persistent_stay and persistent_intermediate and persistent_spatial and not high_root
            hard_transient = (t_score >= 4 or transient_outlier) and not high_root
            candidate_risk_rank = 3 if hard_persistent or hard_transient else 2 if (p_score >= 2 or t_score >= 3) else 1
            candidate_score = (
                candidate_risk_rank,
                p_score + t_score,
                dist_w48_event,
                dist_v_event,
            )
            row = {
                "frame": cand,
                "risk": "hard_ik_branch_switch" if candidate_risk_rank == 3 else "review_or_repairable_dynamic" if candidate_risk_rank == 2 else "clean",
                "reason": "upper_persistent_branch_drift" if hard_persistent else "upper_transient_snap" if hard_transient else "upper_borderline_same_chain" if candidate_risk_rank == 2 else "none",
                "p_score": p_score,
                "t_score": t_score,
                "high_root": high_root,
                "root_z": local_root_z,
                "root_vz": local_root_vz,
                "dist_min_after": min_after,
                "dist_outlier_ratio": dist_outlier_ratio,
                "dist_pos_pre_post": dist_pos_pre_post,
                "dist_pos_path": dist_pos_path,
                "dist_pos_max_step": dist_pos_max_step,
                "dist_rot_per_path": dist_rot_per_path,
                "int_body": int_v_body or int_w24_body,
                "dist_body": dist_v_body or dist_w48_body,
                "score": candidate_score,
            }
            if best is None or row["score"] > best["score"]:
                best = row

        if best is not None:
            frame = best["frame"]
            risk = best["risk"]
            reason = best["reason"]
            persistent_score = int(best["p_score"])
            transient_score = int(best["t_score"])
            high_root_dynamic = bool(best["high_root"])
            root_z_range = float(best["root_z"])
            root_vz_abs_max = float(best["root_vz"])
            distal_min_after = float(best["dist_min_after"])
            distal_event_outlier_ratio = float(best["dist_outlier_ratio"])
            distal_position_pre_post = float(best["dist_pos_pre_post"])
            distal_position_path = float(best["dist_pos_path"])
            distal_position_max_step = float(best["dist_pos_max_step"])
            distal_rot_per_path = float(best["dist_rot_per_path"])
            int_adj_body = best["int_body"] or int_adj_body
            dist_adj_body = best["dist_body"] or dist_adj_body
        if risk == "clean" and (dist_adj >= ns.review_distal_adj_deg_s or dist_w24 >= ns.review_distal_w24_deg):
            frame = dist_adj_frame if dist_adj >= ns.review_distal_adj_deg_s else dist_w24_frame
            reason = "upper_distal_review"
            risk = "review_or_repairable_dynamic"
    else:
        same_leg_switch = (
            (int_adj >= ns.lower_intermediate_adj_deg_s and dist_adj >= ns.lower_distal_adj_deg_s)
            or (int_w24 >= ns.lower_intermediate_w24_deg and dist_w24 >= ns.lower_distal_w24_deg)
        )
        frame = int(max(0, min(int_adj_frame, dist_adj_frame)))
        high_root_dynamic, root_z_range, root_vz_abs_max = root_dynamic_around(root_pos, frame, fps)
        int_series = adj.get(int_adj_body, np.zeros((0,), dtype=np.float64))
        dist_series = adj.get(dist_adj_body, np.zeros((0,), dtype=np.float64))
        int_outlier_ratio = int_adj / max(float(np.percentile(int_series, 99)) if int_series.size else 0.0, 1e-6)
        distal_event_outlier_ratio = dist_adj / max(float(np.percentile(dist_series, 99)) if dist_series.size else 0.0, 1e-6)
        lower_outlier = int_outlier_ratio >= 3.0 and distal_event_outlier_ratio >= 5.0
        if same_leg_switch and lower_outlier and not high_root_dynamic:
            frame = max(0, min(int_w24_frame, dist_w24_frame, int_adj_frame, dist_adj_frame))
            reason = "lower_same_chain_intermediate_distal"
            risk = "hard_ik_branch_switch"
        elif same_leg_switch:
            frame = max(0, min(int_w24_frame, dist_w24_frame, int_adj_frame, dist_adj_frame))
            reason = "lower_same_chain_dynamic_review"
            risk = "review_or_repairable_dynamic"
        elif dist_adj >= ns.review_distal_adj_deg_s or dist_w24 >= ns.review_distal_w24_deg:
            frame = dist_adj_frame if dist_adj >= ns.review_distal_adj_deg_s else dist_w24_frame
            reason = "lower_distal_review"
            risk = "review_or_repairable_dynamic"
        else:
            frame = int_adj_frame
            reason = "none"
            risk = "clean"

    return ChainEvidence(
        chain=chain,
        risk=risk,
        reason=reason,
        frame0=int(frame),
        frame1=int(frame + 1),
        persistent_score=persistent_score,
        transient_score=transient_score,
        high_root_dynamic=high_root_dynamic,
        root_z_range_2s_m=root_z_range,
        root_vz_abs_max_2s_m_s=root_vz_abs_max,
        max_intermediate_adj_deg_s=int_adj,
        max_distal_adj_deg_s=dist_adj,
        max_intermediate_w24_deg=int_w24,
        max_distal_w24_deg=dist_w24,
        max_intermediate_w48_deg=int_w48,
        max_distal_w48_deg=dist_w48,
        distal_min_after_48_240_deg=distal_min_after,
        distal_event_outlier_ratio=distal_event_outlier_ratio,
        distal_position_pre_post_m=distal_position_pre_post,
        distal_position_path_72_m=distal_position_path,
        distal_position_max_step_m=distal_position_max_step,
        distal_rot_per_path_deg_per_m=distal_rot_per_path,
        top_intermediate_body=int_adj_body or int_w24_body,
        top_distal_body=dist_adj_body or dist_w24_body,
    )


def classify_motion(job: tuple[str, str, str, float, dict]) -> tuple[dict, list[dict]]:
    motion_root_s, csv_path_s, mjcf_path_s, fps, args_dict = job
    # Reconstruct a namespace for threshold access in worker.
    ns = argparse.Namespace(**args_dict)
    motion_root = Path(motion_root_s)
    csv_path = Path(csv_path_s)
    if _MODEL is None:
        init_worker(mjcf_path_s)
    dataset, subset = infer_dataset_subset(motion_root, csv_path)
    root_pos, root_quat_xyzw, joint_names, joint_pos = load_csv_motion(csv_path)
    qpos = make_model_qpos(root_pos, root_quat_xyzw, joint_names, joint_pos)
    rel_quats, rel_pos = compute_relative_body_state(qpos)

    evidences = [classify_chain(chain, rel_quats, rel_pos, root_pos, fps, ns) for chain in CHAIN_BODIES]
    hard = [e for e in evidences if e.risk == "hard_ik_branch_switch"]
    review = [e for e in evidences if e.risk == "review_or_repairable_dynamic"]
    if hard:
        risk = "hard_ik_branch_switch"
    elif review:
        risk = "review_or_repairable_dynamic"
    else:
        risk = "clean"

    event_rows = [
        {
            "motion": csv_path.stem,
            "motion_path": str(csv_path),
            "relative_path": str(csv_path.relative_to(motion_root)),
            "dataset": dataset,
            "subset": subset,
            **e.__dict__,
        }
        for e in evidences
        if e.risk != "clean"
    ]
    top = max(evidences, key=lambda e: max(e.max_intermediate_adj_deg_s, e.max_distal_adj_deg_s, e.max_intermediate_w24_deg * fps / 24.0, e.max_distal_w24_deg * fps / 24.0))
    summary = {
        "motion": csv_path.stem,
        "motion_path": str(csv_path),
        "relative_path": str(csv_path.relative_to(motion_root)),
        "dataset": dataset,
        "subset": subset,
        "risk": risk,
        "num_frames": int(root_pos.shape[0]),
        "hard_chain_count": len(hard),
        "review_chain_count": len(review),
        "chains_hard": ";".join(e.chain for e in hard),
        "chains_review": ";".join(e.chain for e in review),
        "top_chain": top.chain,
        "top_reason": top.reason,
        "top_intermediate_body": top.top_intermediate_body,
        "top_distal_body": top.top_distal_body,
        "top_intermediate_adj_deg_s": top.max_intermediate_adj_deg_s,
        "top_distal_adj_deg_s": top.max_distal_adj_deg_s,
        "top_intermediate_w24_deg": top.max_intermediate_w24_deg,
        "top_distal_w24_deg": top.max_distal_w24_deg,
        "top_intermediate_w48_deg": top.max_intermediate_w48_deg,
        "top_distal_w48_deg": top.max_distal_w48_deg,
        "top_distal_position_pre_post_m": top.distal_position_pre_post_m,
        "top_distal_position_path_72_m": top.distal_position_path_72_m,
        "top_distal_position_max_step_m": top.distal_position_max_step_m,
        "top_distal_rot_per_path_deg_per_m": top.distal_rot_per_path_deg_per_m,
    }
    return summary, event_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ns = parse_args()
    motion_root = ns.motion_root.expanduser().resolve()
    output_dir = ns.output_dir.expanduser().resolve()
    mjcf_path = ns.mjcf.expanduser().resolve()
    csv_paths = sorted(motion_root.glob(ns.glob))
    if ns.include_list is not None:
        include_path = ns.include_list.expanduser().resolve()
        include = {
            line.strip()
            for line in include_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        csv_paths = [path for path in csv_paths if str(path.relative_to(motion_root)) in include]
    if ns.limit is not None:
        csv_paths = csv_paths[: ns.limit]
    if not csv_paths:
        raise FileNotFoundError(f"no CSVs matched {motion_root / ns.glob}")
    output_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(ns).copy()
    # Path objects are not needed inside workers except as strings in job.
    args_dict["motion_root"] = str(motion_root)
    args_dict["output_dir"] = str(output_dir)
    args_dict["mjcf"] = str(mjcf_path)
    jobs = [(str(motion_root), str(path), str(mjcf_path), float(ns.fps), args_dict) for path in csv_paths]

    summaries: list[dict] = []
    events: list[dict] = []
    started_at = time.perf_counter()
    max_pending = ns.max_pending or max(ns.workers, ns.workers * 4)
    print(
        f"[fk-ik] start jobs={len(jobs)} workers={ns.workers} max_pending={max_pending} "
        f"progress_interval={ns.progress_interval}",
        flush=True,
    )

    def print_progress(done_count: int) -> None:
        elapsed = max(time.perf_counter() - started_at, 1e-6)
        rate = done_count / elapsed
        eta = (len(jobs) - done_count) / max(rate, 1e-6)
        print(
            f"[fk-ik] {done_count}/{len(jobs)} elapsed={elapsed:.1f}s "
            f"rate={rate:.2f}/s eta={eta:.1f}s",
            flush=True,
        )

    if ns.workers == 1:
        init_worker(str(mjcf_path))
        for idx, job in enumerate(jobs, start=1):
            summary, rows = classify_motion(job)
            summaries.append(summary)
            events.extend(rows)
            if idx % ns.progress_interval == 0 or idx == len(jobs):
                print_progress(idx)
    else:
        with ProcessPoolExecutor(max_workers=ns.workers, initializer=init_worker, initargs=(str(mjcf_path),)) as pool:
            job_iter = iter(jobs)
            pending = set()

            def submit_next() -> bool:
                try:
                    job = next(job_iter)
                except StopIteration:
                    return False
                pending.add(pool.submit(classify_motion, job))
                return True

            for _ in range(min(max_pending, len(jobs))):
                submit_next()

            idx = 0
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    summary, rows = future.result()
                    idx += 1
                    summaries.append(summary)
                    events.extend(rows)
                    while len(pending) < max_pending and submit_next():
                        pass
                    if idx % ns.progress_interval == 0 or idx == len(jobs):
                        print_progress(idx)

    summaries.sort(key=lambda r: r["relative_path"])
    events.sort(key=lambda r: (r["relative_path"], r["chain"], r["frame0"]))
    summary_fields = [
        "motion",
        "motion_path",
        "relative_path",
        "dataset",
        "subset",
        "risk",
        "num_frames",
        "hard_chain_count",
        "review_chain_count",
        "chains_hard",
        "chains_review",
        "top_chain",
        "top_reason",
        "top_intermediate_body",
        "top_distal_body",
        "top_intermediate_adj_deg_s",
        "top_distal_adj_deg_s",
        "top_intermediate_w24_deg",
        "top_distal_w24_deg",
        "top_intermediate_w48_deg",
        "top_distal_w48_deg",
        "top_distal_position_pre_post_m",
        "top_distal_position_path_72_m",
        "top_distal_position_max_step_m",
        "top_distal_rot_per_path_deg_per_m",
    ]
    event_fields = [
        "motion",
        "motion_path",
        "relative_path",
        "dataset",
        "subset",
        "chain",
        "risk",
        "reason",
        "frame0",
        "frame1",
        "persistent_score",
        "transient_score",
        "high_root_dynamic",
        "root_z_range_2s_m",
        "root_vz_abs_max_2s_m_s",
        "max_intermediate_adj_deg_s",
        "max_distal_adj_deg_s",
        "max_intermediate_w24_deg",
        "max_distal_w24_deg",
        "max_intermediate_w48_deg",
        "max_distal_w48_deg",
        "distal_min_after_48_240_deg",
        "distal_event_outlier_ratio",
        "distal_position_pre_post_m",
        "distal_position_path_72_m",
        "distal_position_max_step_m",
        "distal_rot_per_path_deg_per_m",
        "top_intermediate_body",
        "top_distal_body",
    ]
    write_csv(output_dir / "motion_fk_same_chain_classification.csv", summaries, summary_fields)
    write_csv(output_dir / "fk_same_chain_events.csv", events, event_fields)

    counts = {risk: sum(1 for row in summaries if row["risk"] == risk) for risk in sorted({r["risk"] for r in summaries})}
    config = {
        "motion_root": str(motion_root),
        "mjcf": str(mjcf_path),
        "fps": ns.fps,
        "glob": ns.glob,
        "counts": counts,
        "thresholds": {
            key: value
            for key, value in vars(ns).items()
            if key.endswith("_deg") or key.endswith("_deg_s") or key.endswith("_m") or key.endswith("_deg_m")
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    md = [
        "# FK Same-Chain IK Classification",
        "",
        f"- Motion root: `{motion_root}`",
        f"- MJCF: `{mjcf_path}`",
        f"- Motions: `{len(summaries)}`",
        "",
        "## Counts",
        "",
        "| risk | count |",
        "| --- | ---: |",
    ]
    for risk, count in counts.items():
        md.append(f"| {risk} | {count} |")
    md.extend(
        [
            "",
            "## Files",
            "",
            "- `motion_fk_same_chain_classification.csv`",
            "- `fk_same_chain_events.csv`",
            "- `run_config.json`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[fk-ik] output={output_dir}", flush=True)
    print(counts, flush=True)


if __name__ == "__main__":
    main()
