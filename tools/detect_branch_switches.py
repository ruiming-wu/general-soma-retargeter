#!/usr/bin/env python3
"""Detect likely IK branch switches in retargeted robot CSV motions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from soma_retargeter.diagnostics.branch_switch import (
    BranchSwitchConfig,
    BranchSwitchEvent,
    detect_joint_branch_switch_events,
    upgrade_clustered_branch_switch_events,
    wrapped_angle_diff_rad,
)


ROOT_POSITION_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")
ROOT_ROTATION_COLUMNS = ("root_rotateX", "root_rotateY", "root_rotateZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, required=True, help="Directory containing retargeted CSV files.")
    parser.add_argument("--glob", default="**/*.csv", help="CSV glob relative to --motion-root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for diagnostic CSV reports.")
    parser.add_argument("--mjcf", type=Path, default=Path("/home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml"))
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--candidate-joint-diff-deg", type=float, default=3.0)
    parser.add_argument("--strong-joint-diff-deg", type=float, default=5.0)
    parser.add_argument("--severe-joint-diff-deg", type=float, default=10.0)
    parser.add_argument("--normalized-range-diff-threshold", type=float, default=0.08)
    parser.add_argument("--local-window", type=int, default=3)
    parser.add_argument("--dynamic-neighbor-min-deg", type=float, default=2.0)
    parser.add_argument("--dynamic-neighbor-fraction", type=float, default=0.45)
    parser.add_argument("--min-dynamic-support-frames", type=int, default=2)
    parser.add_argument("--isolation-neighbor-fraction", type=float, default=0.25)
    parser.add_argument("--cluster-strong-count", type=int, default=3)
    parser.add_argument("--cluster-severe-count", type=int, default=2)
    parser.add_argument("--cluster-min-max-dq-deg", type=float, default=8.0)
    parser.add_argument("--cluster-min-accel-deg-s2", type=float, default=50000.0)
    parser.add_argument("--clean-allow-possible", action="store_true", help="Allow possible_branch_switch in clean manifest.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_csv_motion(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding=None)
    if data.shape == ():
        data = data.reshape(1)
    names = data.dtype.names
    if names is None:
        raise ValueError(f"{path} has no CSV header")
    missing_root = [name for name in ROOT_POSITION_COLUMNS + ROOT_ROTATION_COLUMNS if name not in names]
    if missing_root:
        raise ValueError(f"{path} is missing root columns: {missing_root}")
    joint_cols = [name for name in names if name.endswith("_dof")]
    if not joint_cols:
        raise ValueError(f"{path} has no *_dof joint columns")
    joint_names = [name[: -len("_dof")] for name in joint_cols]
    joint_pos_rad = np.deg2rad(np.stack([data[name] for name in joint_cols], axis=1))
    return joint_names, joint_pos_rad


def parse_mjcf_joint_limits(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        return {}
    tree = ET.parse(path)
    root = tree.getroot()
    compiler = root.find("compiler")
    angle_unit = compiler.attrib.get("angle", "radian") if compiler is not None else "radian"
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        name = joint.attrib.get("name")
        range_text = joint.attrib.get("range")
        if not name or not range_text:
            continue
        parts = [float(x) for x in range_text.split()]
        if len(parts) != 2:
            continue
        lo, hi = parts
        if angle_unit == "degree":
            lo, hi = np.deg2rad([lo, hi])
        limits[name] = (float(lo), float(hi))
    return limits


def finite_motion_extrema(joint_pos_rad: np.ndarray, fps: float) -> tuple[float, float, float]:
    if joint_pos_rad.shape[0] < 2:
        return 0.0, 0.0, 0.0
    dq_deg = np.rad2deg(wrapped_angle_diff_rad(joint_pos_rad[1:], joint_pos_rad[:-1]))
    vel = dq_deg * fps
    acc = np.zeros_like(vel)
    if vel.shape[0] > 1:
        acc[1:] = (vel[1:] - vel[:-1]) * fps
    return float(np.max(np.abs(dq_deg))), float(np.max(np.abs(vel))), float(np.max(np.abs(acc)))


def event_rows(events: list[BranchSwitchEvent], motion_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        row = asdict(event)
        row["motion_path"] = str(motion_path)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = BranchSwitchConfig(
        candidate_joint_diff_deg=args.candidate_joint_diff_deg,
        strong_joint_diff_deg=args.strong_joint_diff_deg,
        severe_joint_diff_deg=args.severe_joint_diff_deg,
        normalized_range_diff_threshold=args.normalized_range_diff_threshold,
        local_window=args.local_window,
        dynamic_neighbor_min_deg=args.dynamic_neighbor_min_deg,
        dynamic_neighbor_fraction=args.dynamic_neighbor_fraction,
        min_dynamic_support_frames=args.min_dynamic_support_frames,
        isolation_neighbor_fraction=args.isolation_neighbor_fraction,
        cluster_strong_count=args.cluster_strong_count,
        cluster_severe_count=args.cluster_severe_count,
        cluster_min_max_dq_deg=args.cluster_min_max_dq_deg,
        cluster_min_accel_deg_s2=args.cluster_min_accel_deg_s2,
    )
    motion_root = args.motion_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    csv_paths = sorted(motion_root.glob(args.glob))
    if args.limit is not None:
        csv_paths = csv_paths[: args.limit]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matched {motion_root / args.glob}")
    joint_limits = parse_mjcf_joint_limits(args.mjcf.expanduser().resolve())

    all_events: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    clean_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []

    for csv_path in csv_paths:
        joint_names, joint_pos_rad = load_csv_motion(csv_path)
        events = detect_joint_branch_switch_events(
            joint_pos_rad,
            joint_names,
            args.fps,
            joint_limits_rad=joint_limits,
            motion_name=csv_path.stem,
            config=config,
        )
        events = upgrade_clustered_branch_switch_events(events, config=config)
        max_dq, max_vel, max_acc = finite_motion_extrema(joint_pos_rad, args.fps)
        counts = {name: 0 for name in [
            "probable_branch_switch",
            "possible_branch_switch",
            "high_dynamic_motion",
            "source_or_retarget_discontinuity",
        ]}
        for event in events:
            counts[event.classification] = counts.get(event.classification, 0) + 1
        probable_frames = len({e.frame0 for e in events if e.classification == "probable_branch_switch"})
        possible_frames = len({e.frame0 for e in events if e.classification == "possible_branch_switch"})
        severe_frames = len({e.frame0 for e in events if e.severity == "severe"})
        risk = (
            "probable_branch_switch"
            if counts["probable_branch_switch"]
            else "possible_branch_switch"
            if counts["possible_branch_switch"] and not args.clean_allow_possible
            else "clean_or_dynamic"
        )
        summary = {
            "motion": csv_path.stem,
            "motion_path": str(csv_path),
            "num_frames": joint_pos_rad.shape[0],
            "num_joints": joint_pos_rad.shape[1],
            "risk": risk,
            "probable_branch_switch_events": counts["probable_branch_switch"],
            "possible_branch_switch_events": counts["possible_branch_switch"],
            "high_dynamic_motion_events": counts["high_dynamic_motion"],
            "source_or_retarget_discontinuity_events": counts["source_or_retarget_discontinuity"],
            "probable_branch_switch_frames": probable_frames,
            "possible_branch_switch_frames": possible_frames,
            "severe_jump_frames": severe_frames,
            "max_abs_dq_deg": max_dq,
            "max_abs_velocity_deg_s": max_vel,
            "max_abs_accel_deg_s2": max_acc,
        }
        summaries.append(summary)
        all_events.extend(event_rows(events, csv_path))
        if risk == "clean_or_dynamic":
            clean_rows.append(summary)
        else:
            review_rows.append(summary)

    event_fields = list(asdict(BranchSwitchEvent("", 0, 1, "", "", "", 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, "")).keys()) + ["motion_path"]
    summary_fields = list(summaries[0].keys())
    write_csv(output_dir / "branch_switch_events.csv", all_events, event_fields)
    write_csv(output_dir / "motion_summary.csv", summaries, summary_fields)
    write_csv(output_dir / "clean_manifest.csv", clean_rows, summary_fields)
    write_csv(output_dir / "review_manifest.csv", review_rows, summary_fields)

    print(f"[branch-switch] motions={len(csv_paths)} events={len(all_events)} output={output_dir}")
    for summary in summaries:
        print(
            f"{summary['motion']}: risk={summary['risk']} "
            f"probable={summary['probable_branch_switch_events']} "
            f"possible={summary['possible_branch_switch_events']} "
            f"dynamic={summary['high_dynamic_motion_events']} "
            f"max_dq={summary['max_abs_dq_deg']:.2f}deg"
        )


if __name__ == "__main__":
    main()
