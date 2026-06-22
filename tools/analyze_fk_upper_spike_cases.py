#!/usr/bin/env python3
"""Analyze upper-chain FK spike morphology for selected retargeted CSV cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from classify_ik_branch_switch_fk_same_chain import (
    CHAIN_BODIES,
    adjacent_rot_vel_deg_s,
    compute_relative_body_quats,
    init_worker,
    load_csv_motion,
    make_model_qpos,
    window_rot_deg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, default=Path("/home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml"))
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("cases", nargs="+", help="case specs as csv_path:chain:event_frame")
    return parser.parse_args()


def summarize(values: np.ndarray, frame: int, fps: float, thresholds: tuple[float, ...]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {f"frames_ge_{thr:g}": 0 for thr in thresholds}
    out: dict[str, float] = {
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "event_value": float(values[min(max(frame, 0), values.size - 1)]),
    }
    for thr in thresholds:
        mask = values >= thr
        out[f"frames_ge_{thr:g}"] = int(np.sum(mask))
        # Longest contiguous run above threshold.
        best = 0
        cur = 0
        for hit in mask:
            if hit:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        out[f"longest_run_ge_{thr:g}"] = int(best)
        out[f"duration_sec_ge_{thr:g}"] = float(best / fps)
    return out


def read_root_series(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64, encoding=None)
    if data.shape == ():
        data = data.reshape(1)
    root_pos = np.stack([data[name] for name in ("root_translateX", "root_translateY", "root_translateZ")], axis=1) / 100.0
    euler = np.stack([data[name] for name in ("root_rotateX", "root_rotateY", "root_rotateZ")], axis=1)
    root_quat = R.from_euler("xyz", euler, degrees=True).as_quat()
    return root_pos, root_quat


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    init_worker(str(args.mjcf.expanduser().resolve()))
    rows: list[dict] = []
    for spec in args.cases:
        csv_s, chain, frame_s = spec.rsplit(":", 2)
        csv_path = Path(csv_s).expanduser().resolve()
        frame = int(frame_s)
        root_pos, root_quat_xyzw, joint_names, joint_pos = load_csv_motion(csv_path)
        qpos = make_model_qpos(root_pos, root_quat_xyzw, joint_names, joint_pos)
        rel = compute_relative_body_quats(qpos)
        bodies = CHAIN_BODIES[chain]
        intermediate = [b for b in bodies["intermediate"] if b in rel]
        distal = [b for b in bodies["distal"] if b in rel]
        adj = {b: adjacent_rot_vel_deg_s(rel[b], args.fps) for b in intermediate + distal}
        w12 = {b: window_rot_deg(rel[b], 12) for b in intermediate + distal}
        w24 = {b: window_rot_deg(rel[b], 24) for b in intermediate + distal}
        w48 = {b: window_rot_deg(rel[b], 48) for b in intermediate + distal}

        def group_max(series: dict[str, np.ndarray], names: list[str]) -> tuple[str, np.ndarray]:
            best_body = ""
            best_values = np.zeros((0,), dtype=np.float64)
            best = -1.0
            for body in names:
                values = series.get(body, np.zeros((0,), dtype=np.float64))
                val = float(values.max(initial=0.0)) if values.size else 0.0
                if val > best:
                    best = val
                    best_body = body
                    best_values = values
            return best_body, best_values

        int_body, int_adj = group_max(adj, intermediate)
        dist_body, dist_adj = group_max(adj, distal)
        _, int_w12 = group_max(w12, intermediate)
        _, dist_w12 = group_max(w12, distal)
        _, int_w24 = group_max(w24, intermediate)
        _, dist_w24 = group_max(w24, distal)
        _, int_w48 = group_max(w48, intermediate)
        _, dist_w48 = group_max(w48, distal)

        # Local root vertical velocity/height range around the event is a proxy
        # for landing or ballistic gait phases that can create inertial arm shake.
        lo = max(0, frame - 120)
        hi = min(root_pos.shape[0], frame + 121)
        root_z = root_pos[lo:hi, 2]
        root_vz = np.diff(root_z) * args.fps if root_z.size > 1 else np.zeros((0,), dtype=np.float64)

        row = {
            "motion": csv_path.stem,
            "csv_path": str(csv_path),
            "chain": chain,
            "event_frame": frame,
            "intermediate_body": int_body,
            "distal_body": dist_body,
            "root_z_range_2s_m": float(np.ptp(root_z)) if root_z.size else 0.0,
            "root_vz_abs_max_2s_m_s": float(np.max(np.abs(root_vz))) if root_vz.size else 0.0,
        }
        for prefix, values, thresholds in [
            ("int_adj", int_adj, (180.0, 300.0, 400.0, 800.0)),
            ("dist_adj", dist_adj, (360.0, 500.0, 650.0, 1000.0)),
            ("int_w12", int_w12, (15.0, 25.0, 35.0)),
            ("dist_w12", dist_w12, (25.0, 40.0, 55.0)),
            ("int_w24", int_w24, (25.0, 35.0, 45.0)),
            ("dist_w24", dist_w24, (45.0, 55.0, 65.0)),
            ("int_w48", int_w48, (35.0, 50.0, 65.0)),
            ("dist_w48", dist_w48, (60.0, 80.0, 100.0)),
        ]:
            for key, val in summarize(values, frame, args.fps, thresholds).items():
                row[f"{prefix}_{key}"] = val
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_dir / "upper_spike_case_features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md = ["# Upper FK Spike Case Features", "", "| motion | chain | frame | int_body | dist_body | int_adj max | dist_adj max | int_w24 max | dist_w24 max | dist_w48 max | root_z_range |", "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        md.append(
            "| {motion} | {chain} | {event_frame} | {intermediate_body} | {distal_body} | {int_adj_max:.1f} | {dist_adj_max:.1f} | {int_w24_max:.1f} | {dist_w24_max:.1f} | {dist_w48_max:.1f} | {root_z_range_2s_m:.3f} |".format(
                **r
            )
        )
    (args.output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
