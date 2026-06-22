#!/usr/bin/env python3
"""Materialize transient-IK clean motions after cubic repair and post-check.

First-pass clean motions are copied unchanged. Repairable transient single-chain
one/two-joint spikes are repaired into a temporary tree, reclassified with the
same transient-only detector, and only post-check clean results are copied into
the final accepted tree. Everything else is kept in manifests only.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from classify_transient_joint_spikes_dataset import classify_paths, emit_reports
from repair_and_visualize_ik_jump_cases import read_csv, repair_values, wrapped_diff_deg, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-dir", type=Path, required=True)
    parser.add_argument("--output-motion-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--repaired-temp-dir", type=Path, default=None)
    parser.add_argument("--candidate-diff-deg", type=float, default=3.0)
    parser.add_argument("--cluster-frame-gap", type=int, default=1)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=18)
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
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def max_event_frame(values: np.ndarray, start: int, end: int) -> tuple[int, float]:
    start = max(0, int(start))
    end = min(values.shape[0] - 2, int(end))
    if end < start or values.shape[0] < 2:
        return start, 0.0
    local = values[start : end + 2]
    dq = np.abs(wrapped_diff_deg(local[1:], local[:-1]))
    local_idx = int(np.argmax(dq))
    return start + local_idx, float(dq[local_idx])


def copy_motion(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def repair_motion(
    row: dict[str, str],
    clusters: list[dict[str, str]],
    repaired_temp_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, list[dict[str, object]]]:
    src = Path(row["motion_path"])
    rel = Path(row["relative_path"])
    dst = repaired_temp_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    fieldnames, csv_rows = read_csv(src)
    repaired_rows = [dict(r) for r in csv_rows]
    joint_values: dict[str, np.ndarray] = {}
    repair_records: list[dict[str, object]] = []

    for cluster in sorted(clusters, key=lambda r: (int(r["start_frame"]), int(r["end_frame"]), r["joints"])):
        joints = [joint for joint in cluster["joints"].split(";") if joint]
        if not (1 <= len(joints) <= 2):
            raise ValueError(f"Unexpected non-repairable cluster in repair set: {cluster}")
        for joint in joints:
            col = f"{joint}_dof"
            if col not in fieldnames:
                raise ValueError(f"{src} missing {col}")
            if joint not in joint_values:
                joint_values[joint] = np.asarray([float(r[col]) for r in repaired_rows], dtype=np.float64)
            before = joint_values[joint]
            event_frame, event_abs_dq = max_event_frame(before, int(cluster["start_frame"]), int(cluster["end_frame"]))
            if event_abs_dq <= float(args.candidate_diff_deg):
                repair_records.append(
                    {
                        "motion": row["motion"],
                        "relative_path": row["relative_path"],
                        "joint": joint,
                        "cluster_start_frame": int(cluster["start_frame"]),
                        "cluster_end_frame": int(cluster["end_frame"]),
                        "selected_event_frame": event_frame,
                        "local_event_abs_dq_before_deg": event_abs_dq,
                        "repair_status": "skipped_below_threshold",
                    }
                )
                continue
            after, metrics = repair_values(
                before,
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
            repair_records.append(
                {
                    "motion": row["motion"],
                    "relative_path": row["relative_path"],
                    "source_path": row["motion_path"],
                    "repaired_candidate_path": str(dst),
                    "joint": joint,
                    "cluster_start_frame": int(cluster["start_frame"]),
                    "cluster_end_frame": int(cluster["end_frame"]),
                    "selected_event_frame": event_frame,
                    "cluster_max_abs_dq_deg": float(cluster["max_abs_dq_deg"]),
                    "local_event_abs_dq_before_deg": event_abs_dq,
                    "repair_status": "repaired",
                    **metrics,
                }
            )

    for joint, values in joint_values.items():
        col = f"{joint}_dof"
        for csv_row, value in zip(repaired_rows, values, strict=True):
            csv_row[col] = f"{float(value):.10g}"
    write_csv(dst, fieldnames, repaired_rows)
    return dst, repair_records


def main() -> None:
    args = parse_args()
    classification_dir = args.classification_dir.expanduser().resolve()
    output_motion_dir = args.output_motion_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    repaired_temp_dir = (
        args.repaired_temp_dir.expanduser().resolve()
        if args.repaired_temp_dir is not None
        else report_dir / "repaired_candidates"
    )

    prepare_dir(output_motion_dir, args.overwrite)
    prepare_dir(report_dir, args.overwrite)
    prepare_dir(repaired_temp_dir, args.overwrite)

    summary_rows = load_csv(classification_dir / "motion_ik_classification.csv")
    cluster_rows = load_csv(classification_dir / "transient_clusters.csv")
    clusters_by_path: dict[str, list[dict[str, str]]] = defaultdict(list)
    for cluster in cluster_rows:
        clusters_by_path[cluster["motion_path"]].append(cluster)

    clean_rows = [row for row in summary_rows if row["risk"] == "clean"]
    repairable_rows = [row for row in summary_rows if row["risk"] == "repairable_transient_spike"]
    hard_rows = [row for row in summary_rows if row["risk"] == "hard_transient_multijoint_or_multichain"]

    firstpass_clean_accepted: list[dict[str, object]] = []
    for idx, row in enumerate(clean_rows, 1):
        dst = output_motion_dir / row["relative_path"]
        copy_motion(Path(row["motion_path"]), dst)
        firstpass_clean_accepted.append({**row, "accepted_status": "firstpass_clean_copied", "accepted_path": str(dst)})
        if idx % 2000 == 0 or idx == len(clean_rows):
            print(f"[transient-stage] copied first-pass clean {idx}/{len(clean_rows)}")

    repair_records: list[dict[str, object]] = []
    repaired_candidate_rows: list[dict[str, object]] = []
    for idx, row in enumerate(repairable_rows, 1):
        repaired_path, records = repair_motion(row, clusters_by_path[row["motion_path"]], repaired_temp_dir, args)
        repair_records.extend(records)
        repaired_candidate_rows.append({**row, "repaired_candidate_path": str(repaired_path)})
        if idx % 250 == 0 or idx == len(repairable_rows):
            print(f"[transient-stage] repaired candidates {idx}/{len(repairable_rows)}")

    postcheck_dir = report_dir / "postcheck_repaired_candidates"
    repaired_paths = sorted(repaired_temp_dir.glob("**/*.csv"))
    if repaired_paths:
        post_summaries, post_clusters = classify_paths(
            repaired_temp_dir,
            repaired_paths,
            float(args.candidate_diff_deg),
            int(args.cluster_frame_gap),
            float(args.fps),
            int(args.workers),
        )
        emit_reports(
            postcheck_dir,
            repaired_temp_dir,
            post_summaries,
            post_clusters,
            float(args.candidate_diff_deg),
            int(args.cluster_frame_gap),
        )
    else:
        post_summaries = []
        post_clusters = []
        postcheck_dir.mkdir(parents=True, exist_ok=True)

    post_by_rel = {row["relative_path"]: row for row in post_summaries}
    repaired_accepted: list[dict[str, object]] = []
    repaired_rejected: list[dict[str, object]] = []
    for row in repaired_candidate_rows:
        rel = row["relative_path"]
        post = post_by_rel.get(rel)
        if post is not None and post["risk"] == "clean":
            src = repaired_temp_dir / rel
            dst = output_motion_dir / rel
            copy_motion(src, dst)
            repaired_accepted.append(
                {
                    **row,
                    "accepted_status": "repaired_postcheck_clean_copied",
                    "accepted_path": str(dst),
                    "postcheck_risk": post["risk"],
                    "postcheck_candidate_event_count": post["candidate_event_count"],
                }
            )
        else:
            repaired_rejected.append(
                {
                    **row,
                    "rejection_status": "repair_postcheck_not_clean",
                    "postcheck_risk": post["risk"] if post else "missing_postcheck",
                    "postcheck_candidate_event_count": post["candidate_event_count"] if post else "",
                }
            )

    accepted_rows = firstpass_clean_accepted + repaired_accepted
    hard_manifest = [{**row, "rejection_status": "firstpass_hard_manifest_only"} for row in hard_rows]
    rejected_rows = hard_manifest + repaired_rejected

    base_fields = list(summary_rows[0].keys()) if summary_rows else []
    accepted_fields = base_fields + [
        "accepted_status",
        "accepted_path",
        "repaired_candidate_path",
        "postcheck_risk",
        "postcheck_candidate_event_count",
    ]
    rejected_fields = base_fields + [
        "rejection_status",
        "repaired_candidate_path",
        "postcheck_risk",
        "postcheck_candidate_event_count",
    ]
    dump_csv(report_dir / "accepted_clean_or_repaired_manifest.csv", accepted_rows, accepted_fields)
    dump_csv(report_dir / "firstpass_clean_copied_manifest.csv", firstpass_clean_accepted, accepted_fields)
    dump_csv(report_dir / "repaired_postcheck_clean_manifest.csv", repaired_accepted, accepted_fields)
    dump_csv(report_dir / "repaired_postcheck_rejected_manifest.csv", repaired_rejected, rejected_fields)
    dump_csv(report_dir / "firstpass_hard_manifest_only.csv", hard_manifest, rejected_fields)
    dump_csv(report_dir / "rejected_manifest.csv", rejected_rows, rejected_fields)
    dump_csv(report_dir / "repair_cluster_records.csv", repair_records)
    dump_csv(report_dir / "postcheck_repaired_clusters.csv", post_clusters)

    counts = {
        "input_total": len(summary_rows),
        "firstpass_clean_copied": len(firstpass_clean_accepted),
        "repairable_candidates": len(repairable_rows),
        "repaired_postcheck_clean": len(repaired_accepted),
        "repair_postcheck_rejected": len(repaired_rejected),
        "firstpass_hard_rejected": len(hard_manifest),
        "accepted_total": len(accepted_rows),
        "rejected_total": len(rejected_rows),
    }
    meta = {
        "classification_dir": str(classification_dir),
        "output_motion_dir": str(output_motion_dir),
        "report_dir": str(report_dir),
        "repaired_temp_dir": str(repaired_temp_dir),
        "candidate_diff_deg": args.candidate_diff_deg,
        "cluster_frame_gap": args.cluster_frame_gap,
        "repair_mode": args.repair_mode,
        "counts": counts,
    }
    (report_dir / "transient_ik_stage_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    md = [
        "# Transient IK Stage Materialization",
        "",
        f"First-pass classification: `{classification_dir}`",
        f"Accepted output: `{output_motion_dir}`",
        "",
        "## Policy",
        "",
        f"- Candidate threshold: adjacent-frame joint diff `>{args.candidate_diff_deg:g}` deg/frame.",
        "- First-pass clean motions are copied unchanged.",
        "- Repairable single-chain one/two-joint transient spikes are repaired with cubic interpolation.",
        "- Repaired motions are accepted only if transient-only postcheck returns `clean`.",
        "- First-pass hard and postcheck-non-clean motions are recorded in manifests only and discarded from the accepted tree.",
        "",
        "## Counts",
        "",
        "| item | count |",
        "| --- | ---: |",
    ]
    for key, value in counts.items():
        md.append(f"| `{key}` | {value} |")
    md.extend(
        [
            "",
            "## Reports",
            "",
            "- `accepted_clean_or_repaired_manifest.csv`",
            "- `firstpass_clean_copied_manifest.csv`",
            "- `repaired_postcheck_clean_manifest.csv`",
            "- `repaired_postcheck_rejected_manifest.csv`",
            "- `firstpass_hard_manifest_only.csv`",
            "- `rejected_manifest.csv`",
            "- `repair_cluster_records.csv`",
            "- `postcheck_repaired_candidates/`",
        ]
    )
    (report_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        "[transient-stage] "
        f"accepted={counts['accepted_total']} rejected={counts['rejected_total']} "
        f"output={output_motion_dir}"
    )


if __name__ == "__main__":
    main()
