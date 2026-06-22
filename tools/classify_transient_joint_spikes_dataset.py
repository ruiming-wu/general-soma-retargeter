#!/usr/bin/env python3
"""Transient-only joint spike classifier for retargeted robot CSVs.

This intentionally ignores persistent branch-drift heuristics. It only looks
for adjacent-frame joint jumps above a per-frame threshold and classifies each
local event cluster by chain/joint cardinality.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="**/*.csv")
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--candidate-diff-deg", type=float, default=3.0)
    parser.add_argument("--cluster-frame-gap", type=int, default=1)
    parser.add_argument("--workers", type=int, default=max(1, min(18, (os.cpu_count() or 2) - 1)))
    return parser.parse_args()


def wrapped_deg_diff(curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
    return (curr - prev + 180.0) % 360.0 - 180.0


def chain_for_joint(joint: str) -> str:
    if joint.startswith("left_") and any(token in joint for token in ("shoulder", "elbow", "wrist")):
        return "left_arm"
    if joint.startswith("right_") and any(token in joint for token in ("shoulder", "elbow", "wrist")):
        return "right_arm"
    if joint.startswith("left_") and any(token in joint for token in ("hip", "knee", "ankle")):
        return "left_leg"
    if joint.startswith("right_") and any(token in joint for token in ("hip", "knee", "ankle")):
        return "right_leg"
    if "waist" in joint or "head" in joint:
        return "torso_head"
    return "other"


def group_for_joint(joint: str) -> str:
    for token in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "waist", "head"):
        if token in joint:
            side = "left" if joint.startswith("left_") else "right" if joint.startswith("right_") else "center"
            return f"{side}_{token}"
    return "other"


def infer_dataset_subset(motion_root: Path, csv_path: Path) -> tuple[str, str]:
    rel = csv_path.relative_to(motion_root)
    parts = rel.parts
    if not parts:
        return "", ""
    dataset = parts[0]
    if dataset == "seed" and len(parts) > 1:
        return dataset, parts[1]
    if dataset == "grab" and len(parts) > 1:
        return dataset, parts[1]
    return dataset, dataset


def read_joint_matrix_deg(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    joint_indices = [idx for idx, name in enumerate(header) if name.endswith("_dof")]
    if not joint_indices:
        raise ValueError(f"{path} has no *_dof columns")
    joint_names = [header[idx][: -len("_dof")] for idx in joint_indices]
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=joint_indices, ndmin=2)
    return joint_names, data.astype(np.float64, copy=False)


def cluster_events(events: list[dict], frame_gap: int) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for event in sorted(events, key=lambda row: (row["frame0"], row["joint"])):
        if not clusters:
            clusters.append([event])
            continue
        last_frame = max(row["frame0"] for row in clusters[-1])
        if event["frame0"] - last_frame <= frame_gap:
            clusters[-1].append(event)
        else:
            clusters.append([event])
    return clusters


def classify_cluster(cluster: list[dict]) -> str:
    chains = {row["chain"] for row in cluster}
    joints = {row["joint"] for row in cluster}
    if len(chains) == 1 and 1 <= len(joints) <= 2:
        return "repairable_transient_spike"
    return "hard_transient_multijoint_or_multichain"


def classify_one(job: tuple[str, str, float, int, float]) -> tuple[dict, list[dict]]:
    motion_root_s, csv_path_s, candidate_deg, frame_gap, fps = job
    motion_root = Path(motion_root_s)
    csv_path = Path(csv_path_s)
    dataset, subset = infer_dataset_subset(motion_root, csv_path)
    joint_names, q_deg = read_joint_matrix_deg(csv_path)

    events: list[dict] = []
    if q_deg.shape[0] >= 2:
        dq = wrapped_deg_diff(q_deg[1:], q_deg[:-1])
        abs_dq = np.abs(dq)
        hit_frames, hit_joints = np.where(abs_dq > candidate_deg)
        for frame, joint_idx in zip(hit_frames.tolist(), hit_joints.tolist(), strict=True):
            joint = joint_names[joint_idx]
            val = float(abs_dq[frame, joint_idx])
            events.append(
                {
                    "motion": csv_path.stem,
                    "motion_path": str(csv_path),
                    "relative_path": str(csv_path.relative_to(motion_root)),
                    "dataset": dataset,
                    "subset": subset,
                    "frame0": int(frame),
                    "frame1": int(frame + 1),
                    "joint": joint,
                    "joint_group": group_for_joint(joint),
                    "chain": chain_for_joint(joint),
                    "abs_dq_deg": val,
                    "abs_velocity_deg_s": val * fps,
                }
            )
        max_abs_dq = float(abs_dq.max(initial=0.0))
    else:
        max_abs_dq = 0.0

    clusters = cluster_events(events, frame_gap)
    cluster_rows: list[dict] = []
    hard_count = 0
    repairable_count = 0
    for cluster_index, cluster in enumerate(clusters):
        joints = sorted({row["joint"] for row in cluster})
        groups = sorted({row["joint_group"] for row in cluster})
        chains = sorted({row["chain"] for row in cluster})
        risk = classify_cluster(cluster)
        hard_count += int(risk == "hard_transient_multijoint_or_multichain")
        repairable_count += int(risk == "repairable_transient_spike")
        max_abs = max(float(row["abs_dq_deg"]) for row in cluster)
        cluster_rows.append(
            {
                "motion": csv_path.stem,
                "motion_path": str(csv_path),
                "relative_path": str(csv_path.relative_to(motion_root)),
                "dataset": dataset,
                "subset": subset,
                "cluster_index": cluster_index,
                "start_frame": min(int(row["frame0"]) for row in cluster),
                "end_frame": max(int(row["frame0"]) for row in cluster),
                "risk": risk,
                "event_count": len(cluster),
                "joint_count": len(joints),
                "joint_group_count": len(groups),
                "chain_count": len(chains),
                "max_abs_dq_deg": max_abs,
                "max_abs_velocity_deg_s": max_abs * fps,
                "joints": ";".join(joints),
                "joint_groups": ";".join(groups),
                "chains": ";".join(chains),
            }
        )

    if hard_count:
        risk = "hard_transient_multijoint_or_multichain"
    elif repairable_count:
        risk = "repairable_transient_spike"
    else:
        risk = "clean"

    summary = {
        "motion": csv_path.stem,
        "motion_path": str(csv_path),
        "relative_path": str(csv_path.relative_to(motion_root)),
        "dataset": dataset,
        "subset": subset,
        "risk": risk,
        "num_frames": int(q_deg.shape[0]),
        "num_joints": int(q_deg.shape[1]),
        "candidate_event_count": len(events),
        "cluster_count": len(clusters),
        "repairable_cluster_count": repairable_count,
        "hard_cluster_count": hard_count,
        "max_abs_dq_deg": max_abs_dq,
        "max_abs_velocity_deg_s": max_abs_dq * fps,
    }
    return summary, cluster_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def classify_paths(
    motion_root: Path,
    paths: list[Path],
    candidate_diff_deg: float,
    cluster_frame_gap: int,
    fps: float,
    workers: int,
) -> tuple[list[dict], list[dict]]:
    jobs = [(str(motion_root), str(path), candidate_diff_deg, cluster_frame_gap, fps) for path in paths]
    summaries: list[dict] = []
    clusters: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(classify_one, job) for job in jobs]
        for idx, future in enumerate(as_completed(futures), 1):
            summary, cluster_rows = future.result()
            summaries.append(summary)
            clusters.extend(cluster_rows)
            if idx % 1000 == 0 or idx == len(futures):
                print(f"[transient-ik] classified {idx}/{len(futures)}")
    summaries.sort(key=lambda row: row["relative_path"])
    clusters.sort(key=lambda row: (row["relative_path"], row["cluster_index"]))
    return summaries, clusters


def emit_reports(
    output_dir: Path,
    motion_root: Path,
    summaries: list[dict],
    clusters: list[dict],
    candidate_diff_deg: float,
    cluster_frame_gap: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = list(summaries[0].keys()) if summaries else [
        "motion",
        "motion_path",
        "relative_path",
        "dataset",
        "subset",
        "risk",
    ]
    cluster_fields = [
        "motion",
        "motion_path",
        "relative_path",
        "dataset",
        "subset",
        "cluster_index",
        "start_frame",
        "end_frame",
        "risk",
        "event_count",
        "joint_count",
        "joint_group_count",
        "chain_count",
        "max_abs_dq_deg",
        "max_abs_velocity_deg_s",
        "joints",
        "joint_groups",
        "chains",
    ]
    write_csv(output_dir / "motion_ik_classification.csv", summaries, summary_fields)
    write_csv(output_dir / "transient_clusters.csv", clusters, cluster_fields)
    write_csv(output_dir / "branch_switch_clusters.csv", clusters, cluster_fields)
    for risk in ("clean", "repairable_transient_spike", "hard_transient_multijoint_or_multichain"):
        write_csv(output_dir / f"{risk}_manifest.csv", [row for row in summaries if row["risk"] == risk], summary_fields)

    counts: dict[tuple[str, str, str], int] = {}
    for row in summaries:
        key = (row["dataset"], row["subset"], row["risk"])
        counts[key] = counts.get(key, 0) + 1
    count_rows = [
        {"dataset": ds, "subset": subset, "risk": risk, "count": count}
        for (ds, subset, risk), count in sorted(counts.items())
    ]
    for risk in ("clean", "repairable_transient_spike", "hard_transient_multijoint_or_multichain"):
        count_rows.append({"dataset": "ALL", "subset": "ALL", "risk": risk, "count": sum(1 for row in summaries if row["risk"] == risk)})
    write_csv(output_dir / "risk_by_subset.csv", count_rows, ["dataset", "subset", "risk", "count"])

    md = [
        "# Transient IK Spike Classification",
        "",
        f"Input: `{motion_root}`",
        "",
        "## Policy",
        "",
        f"- Candidate threshold: adjacent-frame joint diff `>{candidate_diff_deg:g}` deg/frame.",
        f"- Cluster frame gap: `{cluster_frame_gap}`.",
        "- Persistent branch-drift checks are disabled.",
        "- Single-chain clusters with one or two joints are repairable.",
        "- Multi-chain clusters or single-chain clusters with three or more joints are manifest-only hard rejects.",
        "",
        "## Totals",
        "",
        "| risk | count |",
        "| --- | ---: |",
    ]
    for risk in ("clean", "repairable_transient_spike", "hard_transient_multijoint_or_multichain"):
        md.append(f"| `{risk}` | {sum(1 for row in summaries if row['risk'] == risk)} |")
    md.extend(["", "## Reports", "", "- `motion_ik_classification.csv`", "- `transient_clusters.csv`", "- `risk_by_subset.csv`"])
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    motion_root = args.motion_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    paths = sorted(motion_root.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No CSVs matched {motion_root / args.glob}")
    summaries, clusters = classify_paths(
        motion_root,
        paths,
        float(args.candidate_diff_deg),
        int(args.cluster_frame_gap),
        float(args.fps),
        int(args.workers),
    )
    emit_reports(output_dir, motion_root, summaries, clusters, float(args.candidate_diff_deg), int(args.cluster_frame_gap))
    print(f"[transient-ik] output={output_dir}")


if __name__ == "__main__":
    main()
