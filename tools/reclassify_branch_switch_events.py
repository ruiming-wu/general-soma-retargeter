#!/usr/bin/env python3
"""Reclassify branch-switch events into true IK switches vs repairable smooth conflicts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from soma_retargeter.diagnostics.branch_reclass import BranchReclassConfig, classify_motion_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--motion-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cluster-frame-gap", type=int, default=1)
    parser.add_argument("--strong-jump-deg", type=float, default=5.0)
    parser.add_argument("--severe-jump-deg", type=float, default=10.0)
    parser.add_argument("--true-min-strong-joints", type=int, default=2)
    parser.add_argument("--true-min-joint-groups", type=int, default=1)
    parser.add_argument("--true-min-severe-joints", type=int, default=2)
    return parser.parse_args()


def read_motion_summary(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.expanduser().open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[str(row["motion_path"])] = dict(row)
    return rows


def read_events_by_motion(path: Path) -> dict[str, list[dict[str, str]]]:
    events: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.expanduser().open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events[str(row["motion_path"])].append(dict(row))
    return events


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = BranchReclassConfig(
        cluster_frame_gap=args.cluster_frame_gap,
        strong_jump_deg=args.strong_jump_deg,
        severe_jump_deg=args.severe_jump_deg,
        true_min_strong_joints=args.true_min_strong_joints,
        true_min_joint_groups=args.true_min_joint_groups,
        true_min_severe_joints=args.true_min_severe_joints,
    )

    summaries = read_motion_summary(args.motion_summary_csv)
    events_by_motion = read_events_by_motion(args.events_csv)

    motion_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    for motion_path, summary in summaries.items():
        result = classify_motion_events(events_by_motion.get(motion_path, []), config)
        row: dict[str, object] = dict(summary)
        row.update(
            {
                "new_risk": result.risk,
                "true_ik_cluster_count": result.true_ik_cluster_count,
                "repairable_smooth_conflict_cluster_count": result.repairable_cluster_count,
                "low_amplitude_cluster_count": result.low_amplitude_cluster_count,
                "max_cluster_joint_count": result.max_cluster_joint_count,
                "max_cluster_joint_group_count": result.max_cluster_joint_group_count,
                "max_cluster_abs_dq_deg": result.max_abs_dq_deg,
            }
        )
        motion_rows.append(row)
        for idx, cluster in enumerate(result.clusters):
            cluster_row = asdict(cluster)
            cluster_row["motion"] = summary.get("motion", Path(motion_path).stem)
            cluster_row["motion_path"] = motion_path
            cluster_row["subset"] = summary.get("subset", "")
            cluster_row["cluster_index"] = idx
            cluster_row["joints"] = ";".join(cluster.joints)
            cluster_row["joint_groups"] = ";".join(cluster.joint_groups)
            cluster_rows.append(cluster_row)

    summary_fields = list(motion_rows[0].keys()) if motion_rows else []
    cluster_fields = [
        "motion",
        "motion_path",
        "subset",
        "cluster_index",
        "start_frame",
        "end_frame",
        "risk",
        "event_count",
        "strong_joint_count",
        "severe_joint_count",
        "joint_count",
        "joint_group_count",
        "max_abs_dq_deg",
        "joints",
        "joint_groups",
        "reason",
    ]
    write_csv(output_dir / "motion_reclass_summary.csv", motion_rows, summary_fields)
    write_csv(output_dir / "branch_switch_clusters.csv", cluster_rows, cluster_fields)
    write_csv(output_dir / "true_ik_reject_manifest.csv", [r for r in motion_rows if r["new_risk"] == "true_ik_branch_switch"], summary_fields)
    write_csv(
        output_dir / "repairable_smooth_conflict_manifest.csv",
        [r for r in motion_rows if r["new_risk"] == "repairable_smooth_conflict"],
        summary_fields,
    )
    write_csv(output_dir / "low_amplitude_possible_manifest.csv", [r for r in motion_rows if r["new_risk"] == "low_amplitude_possible"], summary_fields)
    write_csv(output_dir / "clean_or_dynamic_manifest.csv", [r for r in motion_rows if r["new_risk"] == "clean_or_dynamic"], summary_fields)
    write_csv(
        output_dir / "keep_without_true_ik_manifest.csv",
        [r for r in motion_rows if r["new_risk"] != "true_ik_branch_switch"],
        summary_fields,
    )

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in motion_rows:
        counts[(str(row.get("subset", "")), str(row["new_risk"]))] += 1
    overview_rows = [
        {"subset": subset, "new_risk": risk, "count": count}
        for (subset, risk), count in sorted(counts.items())
    ]
    write_csv(output_dir / "risk_by_subset.csv", overview_rows, ["subset", "new_risk", "count"])
    print(f"[reclass] motions={len(motion_rows)} clusters={len(cluster_rows)} output={output_dir}")
    for row in overview_rows:
        print(f"{row['subset'] or 'all'} {row['new_risk']}: {row['count']}")


if __name__ == "__main__":
    main()
