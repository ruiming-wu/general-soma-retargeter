#!/usr/bin/env python3
"""Prototype dual-mode IK switch rules on manually labeled sanity cases.

The goal is not to materialize a dataset.  It tests whether two different
branch-switch modes are numerically separable:

1. persistent branch drift: slow-ish transition that stays on another branch;
2. transient snap/bounce: short velocity/jerk spike that may return.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.classify_ik_branch_switch_fk_same_chain import (  # noqa: E402
    CHAIN_BODIES,
    adjacent_rot_vel_deg_s,
    compute_relative_body_quats,
    init_worker,
    load_csv_motion,
    make_model_qpos,
    quat_angle_deg_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    window_rot_deg,
)


DEFAULT_MJCF = Path("/home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--fps", type=float, default=120.0)
    return parser.parse_args()


def load_body_rel_pos(model: mujoco.MjModel, qpos: np.ndarray, body_name: str) -> np.ndarray:
    data = mujoco.MjData(model)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    out = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        out.append(data.xpos[body_id] - data.xpos[pelvis_id])
    return np.asarray(out, dtype=np.float64)


def max_body_series(series_by_body: dict[str, np.ndarray], names: list[str]) -> tuple[str, np.ndarray]:
    best_name = ""
    best_series = np.zeros((0,), dtype=np.float64)
    best_val = -1.0
    for name in names:
        values = series_by_body.get(name, np.zeros((0,), dtype=np.float64))
        val = float(values.max(initial=0.0)) if values.size else 0.0
        if val > best_val:
            best_name = name
            best_series = values
            best_val = val
    return best_name, best_series


def local_max(values: np.ndarray, frame: int, radius: int) -> float:
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
    mask = values[lo:hi] >= threshold
    best = 0
    cur = 0
    for hit in mask:
        if hit:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def angle_between_rel_frames(quat: np.ndarray, a: int, b: int) -> float:
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
    values = [angle_between_rel_frames(quat, pre, t) for t in range(lo, hi)]
    return float(min(values)) if values else 0.0


def analyze_case(
    model: mujoco.MjModel,
    csv_path: Path,
    chain: str,
    event_frame: int,
    label: str,
    fps: float,
) -> dict:
    root_pos, root_quat, joint_names, joint_pos = load_csv_motion(csv_path)
    qpos = make_model_qpos(root_pos, root_quat, joint_names, joint_pos)
    rel_quats = compute_relative_body_quats(qpos)

    intermediate = [b for b in CHAIN_BODIES[chain]["intermediate"] if b in rel_quats]
    distal = [b for b in CHAIN_BODIES[chain]["distal"] if b in rel_quats]
    adj = {body: adjacent_rot_vel_deg_s(rel_quats[body], fps) for body in intermediate + distal}
    w24 = {body: window_rot_deg(rel_quats[body], 24) for body in intermediate + distal}
    w48 = {body: window_rot_deg(rel_quats[body], 48) for body in intermediate + distal}

    int_body, int_adj = max_body_series(adj, intermediate)
    dist_body, dist_adj = max_body_series(adj, distal)
    _, int_w24 = max_body_series(w24, intermediate)
    _, dist_w24 = max_body_series(w24, distal)
    _, int_w48 = max_body_series(w48, intermediate)
    _, dist_w48 = max_body_series(w48, distal)

    dist_quat = rel_quats[dist_body]
    int_quat = rel_quats[int_body]
    pre = event_frame - 24
    post = event_frame + 48
    post_long = event_frame + 96
    distal_pre_post_deg = angle_between_rel_frames(dist_quat, pre, post)
    distal_pre_long_deg = angle_between_rel_frames(dist_quat, pre, post_long)
    distal_min_after_48_240_deg = min_angle_to_pre_after(dist_quat, pre, event_frame + 48, event_frame + 240)
    intermediate_pre_post_deg = angle_between_rel_frames(int_quat, pre, post)

    # If the pose snaps and returns, pre->long can be small while velocity/jerk
    # around the event is huge.  If it drifts to another branch, pre->long remains
    # large.
    dist_v_event = local_max(dist_adj, event_frame, 3)
    int_v_event = local_max(int_adj, event_frame, 3)
    dist_a = np.diff(dist_adj) * fps if dist_adj.size >= 2 else np.zeros((0,), dtype=np.float64)
    int_a = np.diff(int_adj) * fps if int_adj.size >= 2 else np.zeros((0,), dtype=np.float64)
    dist_j = np.diff(dist_a) * fps if dist_a.size >= 2 else np.zeros((0,), dtype=np.float64)
    int_j = np.diff(int_a) * fps if int_a.size >= 2 else np.zeros((0,), dtype=np.float64)
    dist_p99 = float(np.percentile(dist_adj, 99)) if dist_adj.size else 0.0
    dist_event_outlier_ratio = dist_v_event / max(dist_p99, 1e-6)

    lo = max(0, event_frame - 120)
    hi = min(root_pos.shape[0], event_frame + 121)
    root_z = root_pos[lo:hi, 2]
    root_vz = np.diff(root_z) * fps if root_z.size >= 2 else np.zeros((0,), dtype=np.float64)
    high_root_dynamic = (float(np.ptp(root_z)) if root_z.size else 0.0) >= 0.12 or (
        float(np.max(np.abs(root_vz))) if root_vz.size else 0.0
    ) >= 1.0

    persistent_core = (
        local_max(dist_w48, event_frame, 72) >= 95.0
        and local_max(dist_w24, event_frame, 48) >= 65.0
        and local_max(int_w24, event_frame, 48) >= 30.0
    )
    persistent_stays_on_new_branch = distal_min_after_48_240_deg >= 80.0
    persistent_intermediate_support = intermediate_pre_post_deg >= 25.0
    persistent_score = 0
    persistent_score += int(persistent_core)
    persistent_score += int(persistent_stays_on_new_branch)
    persistent_score += int(persistent_intermediate_support)

    transient_score = 0
    transient_score += int(dist_v_event >= 650.0)
    transient_score += int(int_v_event >= 350.0)
    transient_score += int(local_max(np.abs(dist_a), event_frame, 6) >= 30000.0)
    transient_score += int(local_max(np.abs(dist_j), event_frame, 6) >= 3_000_000.0)
    transient_score += int(local_longest_run(dist_adj, 500.0, event_frame, 12) >= 3)
    transient_is_distribution_outlier = dist_event_outlier_ratio >= 5.0

    persistent = (
        persistent_core
        and persistent_stays_on_new_branch
        and persistent_intermediate_support
        and not high_root_dynamic
    )
    transient = transient_score >= 4 and transient_is_distribution_outlier and not high_root_dynamic
    if persistent:
        pred = "hard_persistent"
    elif transient:
        pred = "hard_transient"
    elif high_root_dynamic and (persistent_score >= 3 or transient_score >= 3):
        pred = "review_high_root_dynamic"
    elif persistent_score >= 3 or transient_score >= 3:
        pred = "review_borderline"
    else:
        pred = "clean_or_dynamic"

    return {
        "motion": csv_path.stem,
        "csv_path": str(csv_path),
        "chain": chain,
        "label": label,
        "event_frame": event_frame,
        "pred": pred,
        "persistent_score": persistent_score,
        "transient_score": transient_score,
        "high_root_dynamic": high_root_dynamic,
        "root_z_range_2s_m": float(np.ptp(root_z)) if root_z.size else 0.0,
        "root_vz_abs_max_2s_m_s": float(np.max(np.abs(root_vz))) if root_vz.size else 0.0,
        "intermediate_body": int_body,
        "distal_body": dist_body,
        "int_v_event_deg_s": int_v_event,
        "dist_v_event_deg_s": dist_v_event,
        "dist_acc_event_abs_max": local_max(np.abs(dist_a), event_frame, 6),
        "dist_jerk_event_abs_max": local_max(np.abs(dist_j), event_frame, 6),
        "dist_event_outlier_ratio": dist_event_outlier_ratio,
        "dist_w24_local_max_deg": local_max(dist_w24, event_frame, 48),
        "dist_w48_local_max_deg": local_max(dist_w48, event_frame, 72),
        "int_w24_local_max_deg": local_max(int_w24, event_frame, 48),
        "distal_pre_post_deg": distal_pre_post_deg,
        "distal_pre_long_deg": distal_pre_long_deg,
        "distal_min_after_48_240_deg": distal_min_after_48_240_deg,
        "intermediate_pre_post_deg": intermediate_pre_post_deg,
        "dist_adj_run_ge500": local_longest_run(dist_adj, 500.0, event_frame, 12),
        "int_adj_run_ge300": local_longest_run(int_adj, 300.0, event_frame, 12),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mjcf = args.mjcf.expanduser().resolve()
    init_worker(str(mjcf))
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    cases = [
        ("positive_transient", "slowmo2_nutan_2", "left_arm", 6188, "/home/ruiming.wu/data/nutan_retargeted/nutan_to_ao_h4_mjlab_triaxial_smooth5_first0_repeat3_heightfix/120hzcsv/slowmo2_nutan_2.csv"),
        ("positive_transient", "backwards_nutan", "left_arm", 2448, "/home/ruiming.wu/data/nutan_retargeted/nutan_to_ao_h4_mjlab_triaxial_smooth5_first0_repeat3_heightfix/120hzcsv/backwards_nutan.csv"),
        ("positive_transient", "walk100", "right_arm", 4218, "/home/ruiming.wu/data/nutan_retargeted/nutan_to_ao_h4_mjlab_triaxial_smooth5_first0_repeat3_heightfix/120hzcsv/walk_nutan_100bpm.csv"),
        ("positive_transient_small", "walk65", "left_arm", 7334, "/home/ruiming.wu/data/nutan_retargeted/nutan_to_ao_h4_mjlab_triaxial_smooth5_first0_repeat3_heightfix/120hzcsv/walk_nutan_65bpm.csv"),
        ("positive_persistent", "A522", "right_arm", 416, "/home/ruiming.wu/data/retarget_runs/soma_to_ao_h4_mjlab_triaxial_first0_repeat3_smooth5_batch128/03_limmt_gqs_score90_passed_csv/motions/seed/object_manipulation/body_motion/medium_big_heavy_one_hand_right_side_medium_to_front_medium_R_001__A522.csv"),
        ("positive_persistent", "A522_M", "left_arm", 416, "/home/ruiming.wu/data/retarget_runs/soma_to_ao_h4_mjlab_triaxial_first0_repeat3_smooth5_batch128/03_limmt_gqs_score90_passed_csv/motions/seed/object_manipulation/body_motion/medium_big_heavy_one_hand_right_side_medium_to_front_medium_R_001__A522_M.csv"),
        ("negative_high_root_dynamic", "hopping_R", "right_arm", 840, "/home/ruiming.wu/data/nutan_retargeted/nutan_to_ao_h4_mjlab_triaxial_smooth5_first0_repeat3_heightfix/120hzcsv/hopping_nutan.csv"),
        ("negative_high_root_dynamic", "hopping_L", "left_arm", 922, "/home/ruiming.wu/data/nutan_retargeted/nutan_to_ao_h4_mjlab_triaxial_smooth5_first0_repeat3_heightfix/120hzcsv/hopping_nutan.csv"),
        ("positive_transient", "jog_L", "left_arm", 7749, "/home/ruiming.wu/data/nutan_retargeted/nutan_to_ao_h4_mjlab_triaxial_smooth5_first0_repeat3_heightfix/120hzcsv/jog_nutan_140bpm.csv"),
        ("negative_dynamic", "jog_R", "right_arm", 14604, "/home/ruiming.wu/data/nutan_retargeted/nutan_to_ao_h4_mjlab_triaxial_smooth5_first0_repeat3_heightfix/120hzcsv/jog_nutan_140bpm.csv"),
    ]
    rows = [analyze_case(model, Path(path), chain, frame, label, args.fps) for label, _, chain, frame, path in cases]
    fields = list(rows[0].keys())
    with (args.output_dir / "dual_mode_case_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Dual-Mode IK Switch Prototype",
        "",
        "| motion | label | pred | persistent | transient | outlier | root_dynamic | dist_v | int_v | dist_w48 | min_after | root_z |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            "| {motion} | {label} | {pred} | {persistent_score} | {transient_score} | {dist_event_outlier_ratio:.2f} | {high_root_dynamic} | {dist_v_event_deg_s:.1f} | {int_v_event_deg_s:.1f} | {dist_w48_local_max_deg:.1f} | {distal_min_after_48_240_deg:.1f} | {root_z_range_2s_m:.3f} |".format(
                **r
            )
        )
    (args.output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(args.output_dir)
    print("\n".join(md))


if __name__ == "__main__":
    main()
