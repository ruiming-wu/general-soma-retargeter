#!/usr/bin/env python3
"""Dry-run density-balanced motion selection.

This tool selects a smaller subset from an existing diversity feature table without
moving motion files. It is designed for the post-GQS/post-IK stage where dense
near-duplicates should be thinned while sparse motions and GRAB clips are kept.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TARGET_CATEGORIES = {
    "basic_locomotion_neutral",
    "object_interaction",
    "object_manipulation",
}

LOW_DYNAMIC_FAMILY_HINTS = {
    "idle",
    "look",
    "stand",
    "kneel",
    "sit",
    "crouch",
    "squat",
    "hold",
    "lift",
    "pocket_searching",
    "playing_recorder",
}

META_COLUMNS = {
    "rel_path",
    "dataset",
    "category",
    "subcategory",
    "motion",
    "is_mirrored_name",
    "num_frames",
    "duration_sec",
    "gqs_score",
    "gqs_passed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="04 motion_diversity_features.csv")
    parser.add_argument("--nearest", type=Path, required=True, help="04 nearest_neighbors.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=20_000)
    parser.add_argument("--side-min", type=float, default=0.48)
    parser.add_argument("--side-max", type=float, default=0.52)
    parser.add_argument("--dominance-threshold", type=float, default=0.12)
    parser.add_argument(
        "--target-category",
        action="append",
        default=None,
        help="Category allowed for pruning. May be repeated. Defaults to the three large seed categories.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows, fields


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: set[str] = set()
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str | None, default: float = 0.0) -> float:
    try:
        out = float(value if value is not None else "")
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def strip_mirror(stem: str) -> str:
    return re.sub(r"_M$", "", stem)


def actorless_name(stem: str) -> str:
    return re.sub(r"__A\d+$", "", strip_mirror(stem))


def family_name(path: str) -> str:
    name = actorless_name(Path(path).stem).lower()
    keys = [
        "idle",
        "look",
        "walk",
        "jog",
        "run",
        "turn",
        "jump",
        "hop",
        "kneel",
        "squat",
        "crouch",
        "sit",
        "stand",
        "lift",
        "pick",
        "put",
        "place",
        "throw",
        "catch",
        "push",
        "pull",
        "carry",
        "reach",
        "pocket_searching",
        "playing_recorder",
    ]
    hits = [key for key in keys if key in name]
    if hits:
        return "+".join(hits[:3])
    toks = [t for t in re.split(r"[_\W]+", name) if t and not t.isdigit() and t not in {"l", "r"}]
    return "_".join(toks[:3]) if toks else name


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks / max(1, values.shape[0] - 1)


def robust_range_retention(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    before_p01 = np.nanpercentile(before, 1, axis=0)
    before_p99 = np.nanpercentile(before, 99, axis=0)
    after_p01 = np.nanpercentile(after, 1, axis=0)
    after_p99 = np.nanpercentile(after, 99, axis=0)
    before_rng = np.maximum(before_p99 - before_p01, 1e-12)
    after_rng = np.maximum(after_p99 - after_p01, 1e-12)
    ret = after_rng / before_rng
    return {
        "geomean": float(np.exp(np.nanmean(np.log(np.maximum(ret, 1e-12))))),
        "arithmean": float(np.nanmean(ret)),
        "median": float(np.nanmedian(ret)),
        "p05": float(np.nanpercentile(ret, 5)),
        "min": float(np.nanmin(ret)),
    }


def group_for_feature(feature: str) -> str:
    return feature.split("__", 1)[0] if "__" in feature else "unknown"


def side_dominance(row: dict[str, str], threshold: float) -> tuple[str, float]:
    pairs = [
        (
            "upper_body_workspace_proxy__fk_left_wrist_path_m",
            "upper_body_workspace_proxy__fk_right_wrist_path_m",
        ),
        (
            "upper_body_workspace_proxy__fk_left_wrist_speed_p95_m_s",
            "upper_body_workspace_proxy__fk_right_wrist_speed_p95_m_s",
        ),
        (
            "upper_body_workspace_proxy__left_arm_range_mean_deg",
            "upper_body_workspace_proxy__right_arm_range_mean_deg",
        ),
        (
            "upper_body_workspace_proxy__left_arm_range_max_deg",
            "upper_body_workspace_proxy__right_arm_range_max_deg",
        ),
        (
            "upper_body_workspace_proxy__fk_left_wrist_rel_x_range_m",
            "upper_body_workspace_proxy__fk_right_wrist_rel_x_range_m",
        ),
        (
            "upper_body_workspace_proxy__fk_left_wrist_rel_y_range_m",
            "upper_body_workspace_proxy__fk_right_wrist_rel_y_range_m",
        ),
        (
            "upper_body_workspace_proxy__fk_left_wrist_rel_z_range_m",
            "upper_body_workspace_proxy__fk_right_wrist_rel_z_range_m",
        ),
    ]
    diffs = []
    for left_key, right_key in pairs:
        left = abs(to_float(row.get(left_key)))
        right = abs(to_float(row.get(right_key)))
        denom = left + right
        if denom > 1e-9:
            diffs.append((left - right) / denom)
    if not diffs:
        return "neutral", 0.0
    score = float(np.mean(diffs))
    if score > threshold:
        return "left", score
    if score < -threshold:
        return "right", score
    return "neutral", score


def side_counts(rows: list[dict[str, Any]], kept_mask: np.ndarray | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        if kept_mask is not None and not kept_mask[idx]:
            continue
        counts[str(row["dominant_side"])] += 1
    return counts


def side_fraction(counts: Counter[str]) -> float:
    left = counts.get("left", 0)
    right = counts.get("right", 0)
    denom = left + right
    return left / denom if denom else 0.5


def count_by(rows: list[dict[str, Any]], key: str, kept_mask: np.ndarray | None = None) -> Counter[str]:
    out: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        if kept_mask is not None and not kept_mask[idx]:
            continue
        out[str(row.get(key, ""))] += 1
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 50) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:max_rows]:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    features_path = args.features.expanduser().resolve()
    nearest_path = args.nearest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_categories = set(args.target_category or sorted(DEFAULT_TARGET_CATEGORIES))

    raw_rows, fields = read_csv(features_path)
    features = [name for name in fields if "__" in name and name not in META_COLUMNS]
    rows: list[dict[str, Any]] = []
    matrix = np.empty((len(raw_rows), len(features)), dtype=np.float64)
    for row_idx, row in enumerate(raw_rows):
        mutable: dict[str, Any] = dict(row)
        mutable["index"] = row_idx
        mutable["mirror_base"] = strip_mirror(Path(row["rel_path"]).stem)
        mutable["actorless_base"] = actorless_name(Path(row["rel_path"]).stem)
        mutable["family"] = family_name(row["rel_path"])
        mutable["dominant_side"], mutable["dominant_side_score"] = side_dominance(row, args.dominance_threshold)
        rows.append(mutable)
        for col_idx, feature in enumerate(features):
            matrix[row_idx, col_idx] = to_float(row.get(feature), np.nan)

    rel_to_idx = {str(row["rel_path"]): idx for idx, row in enumerate(rows)}
    rank1_dist = np.full(len(rows), np.nan, dtype=np.float64)
    rank1_neighbor = [""] * len(rows)
    rank1_same_mirror_base = np.zeros(len(rows), dtype=bool)
    rank1_mirror_pair = np.zeros(len(rows), dtype=bool)
    top_edges: list[tuple[float, int, int, int]] = []
    seen_edges: set[tuple[int, int]] = set()
    with nearest_path.open(newline="", encoding="utf-8") as f:
        for nearest_row in csv.DictReader(f):
            a = rel_to_idx.get(nearest_row["rel_path"])
            b = rel_to_idx.get(nearest_row["neighbor_rel_path"])
            if a is None or b is None or a == b:
                continue
            rank = int(nearest_row["neighbor_rank"])
            distance = float(nearest_row["weighted_distance"])
            if rank == 1:
                rank1_dist[a] = distance
                rank1_neighbor[a] = nearest_row["neighbor_rel_path"]
                same_base = rows[a]["mirror_base"] == rows[b]["mirror_base"]
                rank1_same_mirror_base[a] = same_base
                rank1_mirror_pair[a] = same_base and rows[a]["is_mirrored_name"] != rows[b]["is_mirrored_name"]
            if rank <= 5:
                key = (a, b) if a < b else (b, a)
                if key not in seen_edges:
                    seen_edges.add(key)
                    top_edges.append((distance, rank, a, b))
    finite_dist = rank1_dist[np.isfinite(rank1_dist)]
    p10 = float(np.nanpercentile(finite_dist, 10))
    p20 = float(np.nanpercentile(finite_dist, 20))
    p50 = float(np.nanpercentile(finite_dist, 50))

    complexity = np.array([to_float(row.get("dynamics_complexity__complexity_energy")) for row in rows])
    gqs = np.array([to_float(row.get("gqs_score")) for row in rows])
    complexity_rank = rank01(complexity)
    gqs_rank = np.nan_to_num(gqs, nan=0.0) / 100.0
    density_norm = np.clip((p50 - np.nan_to_num(rank1_dist, nan=p50)) / max(p50 - np.nanmin(finite_dist), 1e-9), 0.0, 1.0)

    removable = np.array([row["category"] in target_categories for row in rows], dtype=bool)
    kept = np.ones(len(rows), dtype=bool)
    target_remove = max(0, len(rows) - int(args.target_count))
    if target_remove > int(removable.sum()):
        raise ValueError(f"Cannot remove {target_remove}; only {int(removable.sum())} rows are in target categories")

    deletion_rows: list[dict[str, Any]] = []

    def removal_reason(idx: int) -> str:
        family = str(rows[idx]["family"])
        low_family = any(hint in family for hint in LOW_DYNAMIC_FAMILY_HINTS)
        if rank1_mirror_pair[idx] and rank1_dist[idx] <= p20:
            return "mirror_dense_pair"
        if low_family and rank1_dist[idx] <= p50:
            return "low_dynamic_dense_family"
        if rank1_dist[idx] <= p50:
            return "general_dense_neighbor"
        return "fallback_low_priority"

    def base_priority(idx: int) -> float:
        reason = removal_reason(idx)
        reason_bonus = {
            "mirror_dense_pair": 5.0,
            "low_dynamic_dense_family": 3.0,
            "general_dense_neighbor": 1.5,
            "fallback_low_priority": 0.0,
        }[reason]
        low_dynamic_bonus = 1.0 - float(complexity_rank[idx])
        low_gqs_bonus = 1.0 - float(gqs_rank[idx])
        return reason_bonus + 2.0 * float(density_norm[idx]) + 1.25 * low_dynamic_bonus + 0.5 * low_gqs_bonus

    def delete_idx(idx: int, reason: str, score: float) -> None:
        kept[idx] = False
        deletion_rows.append(
            {
                "rel_path": rows[idx]["rel_path"],
                "category": rows[idx]["category"],
                "subcategory": rows[idx]["subcategory"],
                "family": rows[idx]["family"],
                "dominant_side": rows[idx]["dominant_side"],
                "dominant_side_score": f"{float(rows[idx]['dominant_side_score']):.6f}",
                "rank1_distance": f"{float(rank1_dist[idx]):.8f}",
                "rank1_neighbor": rank1_neighbor[idx],
                "rank1_mirror_pair": bool(rank1_mirror_pair[idx]),
                "gqs_score": f"{gqs[idx]:.6f}",
                "complexity_energy": f"{complexity[idx]:.6f}",
                "removal_reason": reason,
                "removal_score": f"{score:.6f}",
            }
        )

    # First pass: consume top-5 dense edges, removing one endpoint from each live pair.
    top_edges.sort(key=lambda item: (item[0], item[1]))
    for distance, rank, a, b in top_edges:
        if len(deletion_rows) >= target_remove:
            break
        if not kept[a] or not kept[b] or not removable[a] and not removable[b]:
            continue
        candidates = [idx for idx in (a, b) if removable[idx]]
        if not candidates:
            continue
        side_now = side_fraction(side_counts(rows, kept))

        def adjusted_score(idx: int) -> float:
            score = base_priority(idx)
            side = rows[idx]["dominant_side"]
            if side == "left" and side_now > 0.50:
                score += 1.0
            elif side == "right" and side_now < 0.50:
                score += 1.0
            elif side in {"left", "right"}:
                score -= 0.6
            return score

        chosen = max(candidates, key=adjusted_score)
        side = rows[chosen]["dominant_side"]
        if side in {"left", "right"}:
            counts = side_counts(rows, kept)
            counts[side] -= 1
            projected = side_fraction(counts)
            if not (args.side_min <= projected <= args.side_max):
                other = [idx for idx in candidates if idx != chosen]
                if other:
                    alt = max(other, key=adjusted_score)
                    alt_side = rows[alt]["dominant_side"]
                    if alt_side in {"left", "right"}:
                        counts = side_counts(rows, kept)
                        counts[alt_side] -= 1
                        alt_projected = side_fraction(counts)
                    else:
                        alt_projected = side_now
                    if args.side_min <= alt_projected <= args.side_max:
                        chosen = alt
                    elif side_now < args.side_min and side == "left":
                        continue
                    elif side_now > args.side_max and side == "right":
                        continue
        delete_idx(chosen, removal_reason(chosen), adjusted_score(chosen))

    # Second pass: if needed, delete remaining highest-priority dense target rows.
    if len(deletion_rows) < target_remove:
        candidate_indices = [idx for idx in range(len(rows)) if kept[idx] and removable[idx]]
        candidate_indices.sort(key=base_priority, reverse=True)
        for idx in candidate_indices:
            if len(deletion_rows) >= target_remove:
                break
            side_now = side_fraction(side_counts(rows, kept))
            side = rows[idx]["dominant_side"]
            if side in {"left", "right"}:
                counts = side_counts(rows, kept)
                counts[side] -= 1
                projected = side_fraction(counts)
                if not (args.side_min <= projected <= args.side_max):
                    if side_now < args.side_min and side == "left":
                        continue
                    if side_now > args.side_max and side == "right":
                        continue
            delete_idx(idx, removal_reason(idx), base_priority(idx))

    kept_rows = [row for idx, row in enumerate(raw_rows) if kept[idx]]
    removed_set = {row["rel_path"] for row in deletion_rows}
    manifest_rows = []
    for idx, row in enumerate(rows):
        manifest_rows.append(
            {
                "rel_path": row["rel_path"],
                "decision": "remove" if row["rel_path"] in removed_set else "keep",
                "category": row["category"],
                "subcategory": row["subcategory"],
                "family": row["family"],
                "dominant_side": row["dominant_side"],
                "dominant_side_score": f"{float(row['dominant_side_score']):.6f}",
                "rank1_distance": f"{float(rank1_dist[idx]):.8f}",
                "rank1_neighbor": rank1_neighbor[idx],
                "rank1_mirror_pair": bool(rank1_mirror_pair[idx]),
                "removal_reason": next((r["removal_reason"] for r in deletion_rows if r["rel_path"] == row["rel_path"]), ""),
            }
        )

    write_csv(output_dir / "density_balance_manifest.csv", manifest_rows)
    write_csv(output_dir / "density_balance_removed.csv", deletion_rows)
    write_csv(output_dir / "selected_motion_diversity_features.csv", kept_rows, fieldnames=fields)

    before = matrix
    after = matrix[kept]
    overall_ret = robust_range_retention(before, after)
    group_rows = []
    for group in sorted({group_for_feature(feature) for feature in features}):
        cols = [idx for idx, feature in enumerate(features) if group_for_feature(feature) == group]
        ret = robust_range_retention(before[:, cols], after[:, cols])
        group_rows.append({"group": group, **ret})
    write_csv(output_dir / "density_balance_group_retention.csv", group_rows)

    cat_before = count_by(rows, "category")
    cat_after = count_by(rows, "category", kept)
    cat_removed = Counter(row["category"] for row in deletion_rows)
    cat_rows = []
    for category in sorted(cat_before):
        before_count = cat_before[category]
        after_count = cat_after[category]
        cat_rows.append(
            {
                "category": category,
                "before_count": before_count,
                "after_count": after_count,
                "removed_count": cat_removed[category],
                "retention": after_count / before_count if before_count else 0.0,
            }
        )
    write_csv(output_dir / "density_balance_category_counts.csv", cat_rows)

    reason_rows = []
    reason_counts = Counter(row["removal_reason"] for row in deletion_rows)
    for reason, count in reason_counts.most_common():
        reason_rows.append({"removal_reason": reason, "count": count, "pct_removed": count / max(1, len(deletion_rows))})
    write_csv(output_dir / "density_balance_removal_reasons.csv", reason_rows)

    side_before = side_counts(rows)
    side_after = side_counts(rows, kept)
    side_rows = []
    for side in ["left", "right", "neutral"]:
        side_rows.append({"side": side, "before_count": side_before[side], "after_count": side_after[side]})
    write_csv(output_dir / "density_balance_side_counts.csv", side_rows)

    report = []
    report.append("# Density Balance Dry Run\n")
    report.append(f"- Source motions: {len(rows)}")
    report.append(f"- Target count: {args.target_count}")
    report.append(f"- Selected motions: {int(kept.sum())}")
    report.append(f"- Removed motions: {len(deletion_rows)}")
    report.append(f"- Target categories: {', '.join(sorted(target_categories))}")
    report.append(f"- Protected categories/datasets are kept unless they are in target categories; `grab` remains untouched by category.\n")
    report.append("## Side Balance\n")
    report.append(f"- Before left fraction among non-neutral: {side_fraction(side_before):.4f}")
    report.append(f"- After left fraction among non-neutral: {side_fraction(side_after):.4f}")
    report.append(markdown_table(side_rows, ["side", "before_count", "after_count"]))
    report.append("## Category Counts\n")
    report.append(markdown_table(cat_rows, ["category", "before_count", "after_count", "removed_count", "retention"]))
    report.append("## Removal Reasons\n")
    report.append(markdown_table(reason_rows, ["removal_reason", "count", "pct_removed"]))
    report.append("## Diversity Retention\n")
    report.append(
        f"- Overall robust p01-p99 range geomean: {overall_ret['geomean']:.4f}\n"
        f"- Overall arithmetic mean: {overall_ret['arithmean']:.4f}\n"
        f"- Overall median: {overall_ret['median']:.4f}\n"
        f"- Feature p05: {overall_ret['p05']:.4f}\n"
        f"- Worst feature: {overall_ret['min']:.4f}\n"
    )
    report.append(markdown_table(group_rows, ["group", "geomean", "arithmean", "median", "p05", "min"]))
    report.append("## Files\n")
    report.append("- `density_balance_manifest.csv`")
    report.append("- `density_balance_removed.csv`")
    report.append("- `selected_motion_diversity_features.csv`")
    report.append("- `density_balance_category_counts.csv`")
    report.append("- `density_balance_side_counts.csv`")
    report.append("- `density_balance_group_retention.csv`")
    report.append("- `density_balance_removal_reasons.csv`")
    (output_dir / "density_balance_dry_run_report.md").write_text("\n".join(report), encoding="utf-8")

    print(output_dir / "density_balance_dry_run_report.md")
    print(f"selected={int(kept.sum())} removed={len(deletion_rows)} left_fraction={side_fraction(side_after):.4f}")


if __name__ == "__main__":
    main()

