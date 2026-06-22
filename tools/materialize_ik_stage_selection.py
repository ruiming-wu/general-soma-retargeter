#!/usr/bin/env python3
"""Materialize IK-filtered CSVs from a classification report.

Clean motions are copied unchanged. Repairable single-joint jumps are copied
after cubic adaptive-window repair. Hard IK branch-switch motions are recorded
in manifests only and are not materialized into the accepted motion tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from repair_and_visualize_ik_jump_cases import read_csv, repair_values, wrapped_diff_deg, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-dir", type=Path, required=True)
    parser.add_argument("--output-motion-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--candidate-diff-deg", type=float, default=3.0)
    parser.add_argument("--repair-radius-frames", type=int, default=12)
    parser.add_argument("--max-repair-radius-frames", type=int, default=60)
    parser.add_argument("--radius-step-frames", type=int, default=6)
    parser.add_argument("--spike-anchor-threshold-deg", type=float, default=1.5)
    parser.add_argument("--endpoint-fit-frames", type=int, default=6)
    parser.add_argument("--repair-mode", choices=("linear", "cubic", "quintic"), default="cubic")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dump_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def max_event_frame(values: np.ndarray, start: int, end: int) -> int:
    start = max(0, int(start))
    end = min(values.shape[0] - 2, int(end))
    if end < start:
        return start
    local = values[start : end + 2]
    if local.shape[0] < 2:
        return start
    dq = np.abs(wrapped_diff_deg(local[1:], local[:-1]))
    return start + int(np.argmax(dq))


def copy_clean(row: dict[str, str], output_motion_dir: Path) -> dict[str, object]:
    src = Path(row["motion_path"])
    dst = output_motion_dir / row["relative_path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        **row,
        "accepted_status": "clean_copied",
        "accepted_path": str(dst),
        "repair_cluster_count": 0,
    }


def repair_motion(
    row: dict[str, str],
    clusters: list[dict[str, str]],
    output_motion_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    src = Path(row["motion_path"])
    dst = output_motion_dir / row["relative_path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    fieldnames, csv_rows = read_csv(src)
    repaired_rows = [dict(r) for r in csv_rows]
    cluster_records: list[dict[str, object]] = []

    # Work on one mutable numeric trace per touched joint so multiple clusters
    # for the same joint are repaired sequentially on the copied trajectory.
    joint_values: dict[str, np.ndarray] = {}
    sorted_clusters = sorted(clusters, key=lambda r: (int(r["start_frame"]), int(r["end_frame"]), r["joints"]))
    for cluster in sorted_clusters:
        joints = [j for j in cluster["joints"].split(";") if j]
        if len(joints) != 1:
            raise ValueError(f"Repairable cluster should contain one joint: {cluster}")
        joint = joints[0]
        col = f"{joint}_dof"
        if col not in fieldnames:
            raise ValueError(f"{src} missing {col}")
        if joint not in joint_values:
            joint_values[joint] = np.asarray([float(r[col]) for r in repaired_rows], dtype=np.float64)
        values = joint_values[joint]
        event_frame = max_event_frame(values, int(cluster["start_frame"]), int(cluster["end_frame"]))
        before_local = values.copy()
        after, metrics = repair_values(
            values,
            event_frame,
            int(args.repair_radius_frames),
            int(args.max_repair_radius_frames),
            int(args.radius_step_frames),
            float(args.spike_anchor_threshold_deg),
            True,
            str(args.repair_mode),
            int(args.endpoint_fit_frames),
        )
        joint_values[joint] = after
        cluster_records.append(
            {
                "motion": row["motion"],
                "relative_path": row["relative_path"],
                "source_path": row["motion_path"],
                "accepted_path": str(dst),
                "joint": joint,
                "cluster_start_frame": int(cluster["start_frame"]),
                "cluster_end_frame": int(cluster["end_frame"]),
                "selected_event_frame": event_frame,
                "cluster_max_abs_dq_deg": float(cluster["max_abs_dq_deg"]),
                "local_event_abs_dq_before_deg": float(
                    abs(wrapped_diff_deg(before_local[min(event_frame + 1, before_local.shape[0] - 1)], before_local[event_frame]))
                ),
                **metrics,
            }
        )

    for joint, values in joint_values.items():
        col = f"{joint}_dof"
        for csv_row, value in zip(repaired_rows, values, strict=True):
            csv_row[col] = f"{float(value):.10g}"

    write_csv(dst, fieldnames, repaired_rows)
    return (
        {
            **row,
            "accepted_status": "repairable_cubic_adaptive_repaired",
            "accepted_path": str(dst),
            "repair_cluster_count": len(sorted_clusters),
        },
        cluster_records,
    )


def main() -> None:
    args = parse_args()
    classification_dir = args.classification_dir.expanduser().resolve()
    output_motion_dir = args.output_motion_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    prepare_output(output_motion_dir, args.overwrite)
    prepare_output(report_dir, args.overwrite)

    summary_rows = load_csv(classification_dir / "motion_ik_classification.csv")
    cluster_rows = load_csv(classification_dir / "branch_switch_clusters.csv")
    clusters_by_path: dict[str, list[dict[str, str]]] = defaultdict(list)
    for cluster in cluster_rows:
        clusters_by_path[cluster["motion_path"]].append(cluster)

    clean_rows = [r for r in summary_rows if r["risk"] == "clean"]
    repairable_rows = [r for r in summary_rows if r["risk"] == "repairable_single_joint_jump"]
    hard_rows = [r for r in summary_rows if r["risk"] == "hard_ik_branch_switch"]

    accepted_rows: list[dict[str, object]] = []
    repair_records: list[dict[str, object]] = []
    for idx, row in enumerate(clean_rows, 1):
        accepted_rows.append(copy_clean(row, output_motion_dir))
        if idx % 2000 == 0 or idx == len(clean_rows):
            print(f"[ik-stage] copied clean {idx}/{len(clean_rows)}")

    for idx, row in enumerate(repairable_rows, 1):
        accepted, records = repair_motion(row, clusters_by_path[row["motion_path"]], output_motion_dir, args)
        accepted_rows.append(accepted)
        repair_records.extend(records)
        if idx % 250 == 0 or idx == len(repairable_rows):
            print(f"[ik-stage] repaired {idx}/{len(repairable_rows)}")

    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    accepted_fields = summary_fields + ["accepted_status", "accepted_path", "repair_cluster_count"]
    hard_fields = summary_fields + ["rejection_status"]
    hard_manifest = [{**r, "rejection_status": "hard_ik_manifest_only_not_materialized"} for r in hard_rows]

    dump_csv(report_dir / "accepted_clean_or_repaired_manifest.csv", accepted_rows, accepted_fields)
    dump_csv(report_dir / "clean_copied_manifest.csv", [r for r in accepted_rows if r["accepted_status"] == "clean_copied"], accepted_fields)
    dump_csv(
        report_dir / "repairable_cubic_adaptive_manifest.csv",
        [r for r in accepted_rows if r["accepted_status"] == "repairable_cubic_adaptive_repaired"],
        accepted_fields,
    )
    dump_csv(report_dir / "repairable_cubic_adaptive_cluster_repairs.csv", repair_records)
    dump_csv(report_dir / "hard_ik_manifest_only.csv", hard_manifest, hard_fields)

    meta = {
        "classification_dir": str(classification_dir),
        "output_motion_dir": str(output_motion_dir),
        "report_dir": str(report_dir),
        "candidate_diff_deg": args.candidate_diff_deg,
        "repair_mode": args.repair_mode,
        "adaptive_window": True,
        "repair_radius_frames": args.repair_radius_frames,
        "max_repair_radius_frames": args.max_repair_radius_frames,
        "radius_step_frames": args.radius_step_frames,
        "spike_anchor_threshold_deg": args.spike_anchor_threshold_deg,
        "endpoint_fit_frames": args.endpoint_fit_frames,
        "counts": {
            "clean_copied": len(clean_rows),
            "repairable_repaired": len(repairable_rows),
            "hard_manifest_only": len(hard_rows),
            "accepted_total": len(accepted_rows),
            "input_total": len(summary_rows),
        },
    }
    (report_dir / "ik_stage_materialization_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    md = [
        "# IK Stage Materialization",
        "",
        f"Classification report: `{classification_dir}`",
        f"Accepted motion output: `{output_motion_dir}`",
        "",
        "## Policy",
        "",
        "- Threshold baseline: 3 deg per-frame joint difference.",
        "- Clean motions are copied unchanged.",
        "- Repairable single-joint jumps are repaired with cubic Hermite interpolation and adaptive spike windows.",
        "- Hard IK branch-switch motions are kept in CSV manifests only and are not copied into the accepted motion tree.",
        "",
        "## Counts",
        "",
        f"- Input motions: {len(summary_rows)}",
        f"- Clean copied: {len(clean_rows)}",
        f"- Repairable repaired: {len(repairable_rows)}",
        f"- Accepted total: {len(accepted_rows)}",
        f"- Hard IK manifest-only rejected: {len(hard_rows)}",
        "",
        "## Reports",
        "",
        "- `accepted_clean_or_repaired_manifest.csv`",
        "- `clean_copied_manifest.csv`",
        "- `repairable_cubic_adaptive_manifest.csv`",
        "- `repairable_cubic_adaptive_cluster_repairs.csv`",
        "- `hard_ik_manifest_only.csv`",
        "- `ik_stage_materialization_meta.json`",
    ]
    (report_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ik-stage] accepted={len(accepted_rows)} hard_manifest_only={len(hard_rows)}")
    print(f"[ik-stage] motions={output_motion_dir}")
    print(f"[ik-stage] reports={report_dir}")


if __name__ == "__main__":
    main()
