#!/usr/bin/env python3
"""Classify retargeted CSV motions into clean, repairable, and hard IK-switch sets."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="**/*.csv")
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--candidate-diff-deg", type=float, default=5.0)
    parser.add_argument("--severe-diff-deg", type=float, default=10.0)
    parser.add_argument(
        "--upper-candidate-diff-deg",
        type=float,
        default=1.0,
        help="Per-frame candidate threshold for arm joints. Arms need a lower threshold to catch smooth IK branch transitions.",
    )
    parser.add_argument(
        "--lower-candidate-diff-deg",
        type=float,
        default=5.0,
        help="Per-frame candidate threshold for leg joints. Lower-only events are treated as high dynamics, not hard IK.",
    )
    parser.add_argument(
        "--torso-candidate-diff-deg",
        type=float,
        default=3.0,
        help="Per-frame candidate threshold for waist/head joints.",
    )
    parser.add_argument(
        "--upper-chain-window-frames",
        type=int,
        default=24,
        help="Window length for arm-chain cumulative-change checks at CSV FPS.",
    )
    parser.add_argument(
        "--upper-chain-window-total-deg",
        type=float,
        default=45.0,
        help="Arm-chain L2 cumulative change threshold over the window. Catches smooth branch switches below per-frame threshold.",
    )
    parser.add_argument("--cluster-frame-gap", type=int, default=1)
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--materialize", action="store_true", help="Hardlink/copy CSVs into class folders.")
    return parser.parse_args()


def wrapped_deg_diff(curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
    return (curr - prev + 180.0) % 360.0 - 180.0


def coarse_group(joint: str) -> str:
    if joint.startswith("left_"):
        side = "left"
    elif joint.startswith("right_"):
        side = "right"
    else:
        side = "center"
    for token in ("ankle", "knee", "hip", "wrist", "elbow", "shoulder"):
        if token in joint:
            return f"{side}_{token}"
    if "waist" in joint or "head" in joint:
        return "torso_head"
    return f"{side}_other"


def chain_for_joint(joint: str) -> str:
    if any(token in joint for token in ("shoulder", "elbow", "wrist")):
        return "upper"
    if any(token in joint for token in ("hip", "knee", "ankle")):
        return "lower"
    if "waist" in joint or "head" in joint:
        return "torso_head"
    return "other"


def side_for_joint(joint: str) -> str:
    if joint.startswith("left_"):
        return "left"
    if joint.startswith("right_"):
        return "right"
    return "center"


def candidate_threshold_for_joint(joint: str, upper_deg: float, lower_deg: float, torso_deg: float, fallback_deg: float) -> float:
    chain = chain_for_joint(joint)
    if chain == "upper":
        return upper_deg
    if chain == "lower":
        return lower_deg
    if chain == "torso_head":
        return torso_deg
    return fallback_deg


def upper_chain_indices(joint_names: list[str], side: str) -> list[int]:
    return [
        idx
        for idx, joint in enumerate(joint_names)
        if joint.startswith(f"{side}_") and chain_for_joint(joint) == "upper"
    ]


def infer_dataset_subset(motion_root: Path, csv_path: Path) -> tuple[str, str]:
    rel = csv_path.relative_to(motion_root)
    parts = rel.parts
    dataset = parts[0] if parts else ""
    subset = ""
    if len(parts) == 1:
        return "nutan", "nutan"
    if dataset == "seed" and len(parts) > 1:
        subset = parts[1]
    elif dataset == "grab" and len(parts) > 1:
        subset = parts[1]
    elif dataset == "nutan":
        subset = "nutan"
    return dataset, subset


def read_joint_matrix_deg(path: Path) -> tuple[list[str], np.ndarray, int]:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    joint_indices = [idx for idx, name in enumerate(header) if name.endswith("_dof")]
    if not joint_indices:
        raise ValueError(f"{path} has no *_dof columns")
    joint_names = [header[idx][: -len("_dof")] for idx in joint_indices]
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=joint_indices, ndmin=2)
    return joint_names, data.astype(np.float64, copy=False), len(header)


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


def classify_one(args: tuple[str, str, float, float, float, float, float, int, float, float, int]) -> tuple[dict, list[dict]]:
    (
        motion_root_s,
        csv_path_s,
        candidate_deg,
        severe_deg,
        upper_candidate_deg,
        lower_candidate_deg,
        torso_candidate_deg,
        upper_window_frames,
        upper_window_total_deg,
        fps,
        frame_gap,
    ) = args
    motion_root = Path(motion_root_s)
    csv_path = Path(csv_path_s)
    dataset, subset = infer_dataset_subset(motion_root, csv_path)
    joint_names, q_deg, _ = read_joint_matrix_deg(csv_path)
    if q_deg.shape[0] < 2:
        max_abs_dq = 0.0
        max_abs_vel = 0.0
        events: list[dict] = []
    else:
        dq = wrapped_deg_diff(q_deg[1:], q_deg[:-1])
        abs_dq = np.abs(dq)
        max_abs_dq = float(abs_dq.max(initial=0.0))
        max_abs_vel = max_abs_dq * fps
        events = []
        for joint_idx, joint in enumerate(joint_names):
            threshold = candidate_threshold_for_joint(
                joint, upper_candidate_deg, lower_candidate_deg, torso_candidate_deg, candidate_deg
            )
            frames = np.where(abs_dq[:, joint_idx] >= threshold)[0]
            for frame in frames.tolist():
                val = float(abs_dq[frame, joint_idx])
                events.append(
                    {
                        "motion": csv_path.stem,
                        "motion_path": str(csv_path),
                        "dataset": dataset,
                        "subset": subset,
                        "frame0": int(frame),
                        "frame1": int(frame + 1),
                        "joint": joint,
                        "joint_group": coarse_group(joint),
                        "chain": chain_for_joint(joint),
                        "side": side_for_joint(joint),
                        "event_type": "joint_frame_diff",
                        "abs_dq_deg": val,
                        "severity": "severe" if val >= severe_deg else "candidate",
                    }
                )

        # Arm IK branch switches can happen as a smooth chain-level transition:
        # no single adjacent-frame joint diff exceeds the old global threshold,
        # but the shoulder/elbow/wrist chain moves to another solution branch
        # over a short window. Add synthetic chain-window events for that case.
        if upper_window_frames > 1 and q_deg.shape[0] > upper_window_frames:
            for side in ("left", "right"):
                arm_indices = upper_chain_indices(joint_names, side)
                if len(arm_indices) < 2:
                    continue
                delta = wrapped_deg_diff(q_deg[upper_window_frames:, arm_indices], q_deg[:-upper_window_frames, arm_indices])
                chain_norm = np.linalg.norm(delta, axis=1)
                frames = np.where(chain_norm >= upper_window_total_deg)[0]
                for frame in frames.tolist():
                    val = float(chain_norm[frame])
                    events.append(
                        {
                            "motion": csv_path.stem,
                            "motion_path": str(csv_path),
                            "dataset": dataset,
                            "subset": subset,
                            "frame0": int(frame),
                            "frame1": int(frame + upper_window_frames),
                            "joint": f"{side}_upper_chain_window",
                            "joint_group": f"{side}_upper_chain",
                            "chain": "upper",
                            "side": side,
                            "event_type": "upper_chain_window",
                            "abs_dq_deg": val,
                            "severity": "severe" if val >= max(severe_deg, upper_window_total_deg * 1.5) else "candidate",
                        }
                    )

    clusters = cluster_events(events, frame_gap)
    cluster_rows: list[dict] = []
    true_ik_clusters = 0
    repairable_clusters = 0
    lower_dynamic_clusters = 0
    max_cluster_joint_count = 0
    max_cluster_group_count = 0
    max_cluster_abs_dq = 0.0
    for cluster_idx, cluster in enumerate(clusters):
        joints = sorted({row["joint"] for row in cluster})
        groups = sorted({row["joint_group"] for row in cluster})
        chains = sorted({row.get("chain", chain_for_joint(row["joint"])) for row in cluster})
        event_types = sorted({row.get("event_type", "joint_frame_diff") for row in cluster})
        severe_joints = sorted({row["joint"] for row in cluster if row["severity"] == "severe"})
        cluster_max = max(float(row["abs_dq_deg"]) for row in cluster)
        upper_joints = sorted(
            {
                row["joint"]
                for row in cluster
                if row.get("chain", chain_for_joint(row["joint"])) == "upper" and row.get("event_type") == "joint_frame_diff"
            }
        )
        has_upper_chain_window = any(row.get("event_type") == "upper_chain_window" for row in cluster)
        is_lower_only = set(chains) == {"lower"}
        is_true_ik = False
        if is_lower_only:
            # A humanoid leg is only 6-DoF per side and these events are usually
            # high-dynamic locomotion rather than IK branch switches. Keep them
            # visible in reports, but do not reject or smooth them.
            risk = "lower_high_dynamic"
            lower_dynamic_clusters += 1
        else:
            is_true_ik = has_upper_chain_window or len(upper_joints) >= 2 or len(severe_joints) >= 2
            risk = "hard_ik_branch_switch" if is_true_ik else "repairable_single_joint_jump"
        true_ik_clusters += int(is_true_ik)
        repairable_clusters += int((not is_true_ik) and risk == "repairable_single_joint_jump")
        max_cluster_joint_count = max(max_cluster_joint_count, len(joints))
        max_cluster_group_count = max(max_cluster_group_count, len(groups))
        max_cluster_abs_dq = max(max_cluster_abs_dq, cluster_max)
        cluster_rows.append(
            {
                "motion": csv_path.stem,
                "motion_path": str(csv_path),
                "dataset": dataset,
                "subset": subset,
                "cluster_index": cluster_idx,
                "start_frame": min(row["frame0"] for row in cluster),
                "end_frame": max(row["frame0"] for row in cluster),
                "risk": risk,
                "event_types": ";".join(event_types),
                "event_count": len(cluster),
                "joint_count": len(joints),
                "joint_group_count": len(groups),
                "chain_count": len(chains),
                "severe_joint_count": len(severe_joints),
                "max_abs_dq_deg": cluster_max,
                "max_abs_velocity_deg_s": cluster_max * fps,
                "joints": ";".join(joints),
                "joint_groups": ";".join(groups),
                "chains": ";".join(chains),
            }
        )

    if true_ik_clusters:
        risk = "hard_ik_branch_switch"
    elif repairable_clusters:
        risk = "repairable_single_joint_jump"
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
        "hard_ik_cluster_count": true_ik_clusters,
        "repairable_cluster_count": repairable_clusters,
        "lower_high_dynamic_cluster_count": lower_dynamic_clusters,
        "max_cluster_joint_count": max_cluster_joint_count,
        "max_cluster_joint_group_count": max_cluster_group_count,
        "max_abs_dq_deg": max_abs_dq,
        "max_abs_velocity_deg_s": max_abs_vel,
        "max_cluster_abs_dq_deg": max_cluster_abs_dq,
    }
    return summary, cluster_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_rows(motion_root: Path, output_dir: Path, rows: list[dict]) -> None:
    for row in rows:
        src = Path(row["motion_path"])
        dst = output_dir / "motions" / row["risk"] / row["relative_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def main() -> None:
    ns = parse_args()
    motion_root = ns.motion_root.expanduser().resolve()
    output_dir = ns.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(motion_root.glob(ns.glob))
    if not paths:
        raise FileNotFoundError(f"No CSVs matched {motion_root / ns.glob}")

    jobs = [
        (
            str(motion_root),
            str(path),
            ns.candidate_diff_deg,
            ns.severe_diff_deg,
            ns.upper_candidate_diff_deg,
            ns.lower_candidate_diff_deg,
            ns.torso_candidate_diff_deg,
            ns.upper_chain_window_frames,
            ns.upper_chain_window_total_deg,
            ns.fps,
            ns.cluster_frame_gap,
        )
        for path in paths
    ]
    summaries: list[dict] = []
    clusters: list[dict] = []
    with ProcessPoolExecutor(max_workers=ns.workers) as pool:
        futures = [pool.submit(classify_one, job) for job in jobs]
        for idx, future in enumerate(as_completed(futures), 1):
            summary, cluster_rows = future.result()
            summaries.append(summary)
            clusters.extend(cluster_rows)
            if idx % 1000 == 0 or idx == len(futures):
                print(f"[ik-classify] {idx}/{len(futures)}")

    summaries.sort(key=lambda row: row["relative_path"])
    clusters.sort(key=lambda row: (row["motion_path"], row["cluster_index"]))

    summary_fields = list(summaries[0].keys())
    cluster_fields = [
        "motion",
        "motion_path",
        "dataset",
        "subset",
        "cluster_index",
        "start_frame",
        "end_frame",
        "risk",
        "event_types",
        "event_count",
        "joint_count",
        "joint_group_count",
        "chain_count",
        "severe_joint_count",
        "max_abs_dq_deg",
        "max_abs_velocity_deg_s",
        "joints",
        "joint_groups",
        "chains",
    ]
    write_csv(output_dir / "motion_ik_classification.csv", summaries, summary_fields)
    write_csv(output_dir / "branch_switch_clusters.csv", clusters, cluster_fields)
    for risk in ("clean", "repairable_single_joint_jump", "hard_ik_branch_switch"):
        write_csv(output_dir / f"{risk}_manifest.csv", [row for row in summaries if row["risk"] == risk], summary_fields)
    write_csv(
        output_dir / "lower_high_dynamic_clusters.csv",
        [row for row in clusters if row["risk"] == "lower_high_dynamic"],
        cluster_fields,
    )

    counts: dict[tuple[str, str, str], int] = {}
    for row in summaries:
        key = (row["dataset"], row["subset"], row["risk"])
        counts[key] = counts.get(key, 0) + 1
    count_rows = [
        {"dataset": ds, "subset": subset, "risk": risk, "count": count}
        for (ds, subset, risk), count in sorted(counts.items())
    ]
    for risk in ("clean", "repairable_single_joint_jump", "hard_ik_branch_switch"):
        count_rows.append({"dataset": "ALL", "subset": "ALL", "risk": risk, "count": sum(1 for row in summaries if row["risk"] == risk)})
    write_csv(output_dir / "risk_by_subset.csv", count_rows, ["dataset", "subset", "risk", "count"])

    if ns.materialize:
        materialize_rows(motion_root, output_dir, summaries)

    md = ["# IK Branch Switch Classification", "", f"Input: `{motion_root}`", "", "## Totals", "", "| risk | count |", "| --- | ---: |"]
    for risk in ("clean", "repairable_single_joint_jump", "hard_ik_branch_switch"):
        md.append(f"| `{risk}` | {sum(1 for row in summaries if row['risk'] == risk)} |")
    md.extend(
        [
            "",
            "## Detection Policy",
            "",
            f"- Upper joint candidate threshold: `{ns.upper_candidate_diff_deg}` deg/frame",
            f"- Lower joint candidate threshold: `{ns.lower_candidate_diff_deg}` deg/frame",
            f"- Torso/head candidate threshold: `{ns.torso_candidate_diff_deg}` deg/frame",
            f"- Upper chain window: `{ns.upper_chain_window_frames}` frames, cumulative L2 threshold `{ns.upper_chain_window_total_deg}` deg",
            "- Lower-only clusters are reported as `lower_high_dynamic` and do not hard-reject or smooth motions.",
            "",
            "## Reports",
            "",
            "- `motion_ik_classification.csv`",
            "- `branch_switch_clusters.csv`",
            "- `lower_high_dynamic_clusters.csv`",
            "- `risk_by_subset.csv`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ik-classify] output={output_dir}")


if __name__ == "__main__":
    main()
