#!/usr/bin/env python3
"""Compare diversity coverage between two motion diversity feature reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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
    "gqs_deduction_foot_sliding",
    "gqs_deduction_self_collision",
    "gqs_deduction_velocity_violation",
    "gqs_deduction_jerk",
    "nearest_rank1_distance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="Before-stage motion_diversity_features.csv")
    parser.add_argument("--after", type=Path, required=True, help="After-stage motion_diversity_features.csv")
    parser.add_argument("--before-name", default="02_ok")
    parser.add_argument("--after-name", default="04_transient_ik")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robust-low", type=float, default=1.0)
    parser.add_argument("--robust-high", type=float, default=99.0)
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows, fields


def to_float_matrix(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    matrix = np.empty((len(rows), len(features)), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for col_idx, feature in enumerate(features):
            try:
                value = float(row.get(feature, "nan"))
            except Exception:
                value = np.nan
            matrix[row_idx, col_idx] = value
    return matrix


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def stat(values: np.ndarray, q: float | None = None) -> float:
    values = finite(values)
    if values.size == 0:
        return 0.0
    if q is None:
        return float(np.nanmean(values))
    return float(np.nanpercentile(values, q))


def safe_range(low: float, high: float) -> float:
    value = high - low
    if not math.isfinite(value):
        return 0.0
    return max(0.0, float(value))


def group_for_feature(feature: str) -> str:
    return feature.split("__", 1)[0] if "__" in feature else "unknown"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for row in rows:
        out[row.get(key, "")] += 1
    return dict(out)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 30) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:max_rows]:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                if abs(value) >= 10000 or (abs(value) > 0 and abs(value) < 0.001):
                    vals.append(f"{value:.3e}")
                else:
                    vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def summarize_counts(before_rows: list[dict[str, str]], after_rows: list[dict[str, str]], key: str) -> list[dict[str, Any]]:
    before_counts = count_by(before_rows, key)
    after_counts = count_by(after_rows, key)
    out = []
    for item in sorted(set(before_counts) | set(after_counts)):
        b = before_counts.get(item, 0)
        a = after_counts.get(item, 0)
        out.append(
            {
                key: item,
                "before_count": b,
                "after_count": a,
                "kept_count_delta": a - b,
                "count_retention": a / b if b else 0.0,
                "dropped_count": max(0, b - a),
                "dropped_ratio": max(0, b - a) / b if b else 0.0,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    before_path = args.before.expanduser().resolve()
    after_path = args.after.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    before_rows, before_fields = load_rows(before_path)
    after_rows, after_fields = load_rows(after_path)
    features = sorted(
        [
            name
            for name in set(before_fields) & set(after_fields)
            if "__" in name and name not in META_COLUMNS
        ]
    )
    if not features:
        raise ValueError("No shared feature columns found")

    before_matrix = to_float_matrix(before_rows, features)
    after_matrix = to_float_matrix(after_rows, features)
    low = float(args.robust_low)
    high = float(args.robust_high)
    eps = 1e-9

    feature_rows: list[dict[str, Any]] = []
    raw_log10_ratio_sum = 0.0
    robust_log10_ratio_sum = 0.0
    raw_volume_dims = 0
    robust_volume_dims = 0
    for idx, feature in enumerate(features):
        b = before_matrix[:, idx]
        a = after_matrix[:, idx]
        b_min = stat(b, 0)
        b_p01 = stat(b, low)
        b_p05 = stat(b, 5)
        b_p50 = stat(b, 50)
        b_p95 = stat(b, 95)
        b_p99 = stat(b, high)
        b_max = stat(b, 100)
        a_min = stat(a, 0)
        a_p01 = stat(a, low)
        a_p05 = stat(a, 5)
        a_p50 = stat(a, 50)
        a_p95 = stat(a, 95)
        a_p99 = stat(a, high)
        a_max = stat(a, 100)
        raw_before = safe_range(b_min, b_max)
        raw_after = safe_range(a_min, a_max)
        robust_before = safe_range(b_p01, b_p99)
        robust_after = safe_range(a_p01, a_p99)
        p05p95_before = safe_range(b_p05, b_p95)
        p05p95_after = safe_range(a_p05, a_p95)
        raw_ret = raw_after / raw_before if raw_before > eps else 1.0
        robust_ret = robust_after / robust_before if robust_before > eps else 1.0
        p05p95_ret = p05p95_after / p05p95_before if p05p95_before > eps else 1.0
        if raw_before > eps and raw_after > eps:
            raw_log10_ratio_sum += math.log10(max(raw_ret, eps))
            raw_volume_dims += 1
        if robust_before > eps and robust_after > eps:
            robust_log10_ratio_sum += math.log10(max(robust_ret, eps))
            robust_volume_dims += 1
        low_tail_loss = max(0.0, a_p01 - b_p01) / robust_before if robust_before > eps else 0.0
        high_tail_loss = max(0.0, b_p99 - a_p99) / robust_before if robust_before > eps else 0.0
        row = {
            "feature": feature,
            "group": group_for_feature(feature),
            "before_min": b_min,
            "before_p01": b_p01,
            "before_p05": b_p05,
            "before_p50": b_p50,
            "before_p95": b_p95,
            "before_p99": b_p99,
            "before_max": b_max,
            "after_min": a_min,
            "after_p01": a_p01,
            "after_p05": a_p05,
            "after_p50": a_p50,
            "after_p95": a_p95,
            "after_p99": a_p99,
            "after_max": a_max,
            "raw_range_before": raw_before,
            "raw_range_after": raw_after,
            "raw_range_retention": raw_ret,
            "robust_range_before": robust_before,
            "robust_range_after": robust_after,
            "robust_range_retention": robust_ret,
            "p05p95_range_before": p05p95_before,
            "p05p95_range_after": p05p95_after,
            "p05p95_range_retention": p05p95_ret,
            "mean_before": stat(b, None),
            "mean_after": stat(a, None),
            "mean_delta": stat(a, None) - stat(b, None),
            "low_tail_loss_fraction_of_before_robust_range": low_tail_loss,
            "high_tail_loss_fraction_of_before_robust_range": high_tail_loss,
        }
        feature_rows.append(row)

    group_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        group_buckets[str(row["group"])].append(row)
    group_rows: list[dict[str, Any]] = []
    for group, rows in sorted(group_buckets.items()):
        robust_logs = [
            math.log10(max(float(row["robust_range_retention"]), eps))
            for row in rows
            if float(row["robust_range_before"]) > eps and float(row["robust_range_after"]) > eps
        ]
        raw_logs = [
            math.log10(max(float(row["raw_range_retention"]), eps))
            for row in rows
            if float(row["raw_range_before"]) > eps and float(row["raw_range_after"]) > eps
        ]
        group_rows.append(
            {
                "group": group,
                "feature_count": len(rows),
                "raw_log10_volume_ratio": float(np.sum(raw_logs)) if raw_logs else 0.0,
                "robust_log10_volume_ratio": float(np.sum(robust_logs)) if robust_logs else 0.0,
                "raw_geomean_range_retention": 10 ** (float(np.mean(raw_logs)) if raw_logs else 0.0),
                "robust_geomean_range_retention": 10 ** (float(np.mean(robust_logs)) if robust_logs else 0.0),
                "median_robust_range_retention": float(np.median([float(row["robust_range_retention"]) for row in rows])),
                "median_p05p95_range_retention": float(np.median([float(row["p05p95_range_retention"]) for row in rows])),
                "mean_high_tail_loss_fraction": float(np.mean([float(row["high_tail_loss_fraction_of_before_robust_range"]) for row in rows])),
                "mean_low_tail_loss_fraction": float(np.mean([float(row["low_tail_loss_fraction_of_before_robust_range"]) for row in rows])),
            }
        )

    category_rows = summarize_counts(before_rows, after_rows, "category")
    dataset_rows = summarize_counts(before_rows, after_rows, "dataset")
    subcategory_rows = summarize_counts(before_rows, after_rows, "subcategory")

    feature_fields = list(feature_rows[0].keys())
    write_csv(output_dir / "feature_range_retention.csv", feature_rows, feature_fields)
    write_csv(output_dir / "group_range_retention.csv", group_rows)
    write_csv(output_dir / "category_count_retention.csv", category_rows)
    write_csv(output_dir / "dataset_count_retention.csv", dataset_rows)
    write_csv(output_dir / "subcategory_count_retention.csv", subcategory_rows)

    sorted_lost = sorted(feature_rows, key=lambda row: float(row["robust_range_retention"]))
    sorted_high_tail = sorted(feature_rows, key=lambda row: float(row["high_tail_loss_fraction_of_before_robust_range"]), reverse=True)
    sorted_dynamics_lost = [
        row for row in sorted_lost if str(row["group"]) == "dynamics_complexity"
    ]
    write_csv(output_dir / "top_lost_dimensions_by_robust_range.csv", sorted_lost[:80], feature_fields)
    write_csv(output_dir / "top_high_tail_losses.csv", sorted_high_tail[:80], feature_fields)
    write_csv(output_dir / "top_lost_dynamics_dimensions.csv", sorted_dynamics_lost[:80], feature_fields)

    overall = {
        "before_name": args.before_name,
        "after_name": args.after_name,
        "before_count": len(before_rows),
        "after_count": len(after_rows),
        "count_retention": len(after_rows) / len(before_rows),
        "feature_count": len(features),
        "raw_volume_dims": raw_volume_dims,
        "robust_volume_dims": robust_volume_dims,
        "raw_log10_bbox_volume_ratio": raw_log10_ratio_sum,
        "robust_log10_bbox_volume_ratio": robust_log10_ratio_sum,
        "raw_geomean_range_retention": 10 ** (raw_log10_ratio_sum / raw_volume_dims) if raw_volume_dims else 1.0,
        "robust_geomean_range_retention": 10 ** (robust_log10_ratio_sum / robust_volume_dims) if robust_volume_dims else 1.0,
        "robust_low_percentile": low,
        "robust_high_percentile": high,
        "before_features": str(before_path),
        "after_features": str(after_path),
    }
    (output_dir / "coverage_comparison_meta.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    md = [
        "# Diversity Coverage Comparison",
        "",
        f"Before: `{args.before_name}`",
        f"After: `{args.after_name}`",
        "",
        "## Summary",
        "",
        markdown_table([overall], [
            "before_count",
            "after_count",
            "count_retention",
            "feature_count",
            "raw_log10_bbox_volume_ratio",
            "robust_log10_bbox_volume_ratio",
            "raw_geomean_range_retention",
            "robust_geomean_range_retention",
        ], 1),
        "The volume ratio is an axis-aligned approximation over feature ranges. The log10 value is additive across dimensions; the geometric mean range retention is usually easier to read per dimension.",
        "",
        "## Group Coverage",
        "",
        markdown_table(sorted(group_rows, key=lambda row: float(row["robust_geomean_range_retention"])), [
            "group",
            "feature_count",
            "robust_log10_volume_ratio",
            "robust_geomean_range_retention",
            "median_robust_range_retention",
            "mean_high_tail_loss_fraction",
            "mean_low_tail_loss_fraction",
        ], 20),
        "## Count Retention By Category",
        "",
        markdown_table(sorted(category_rows, key=lambda row: str(row["category"])), [
            "category",
            "before_count",
            "after_count",
            "count_retention",
            "dropped_count",
            "dropped_ratio",
        ], 30),
        "## Most Reduced Dimensions By Robust Range",
        "",
        markdown_table(sorted_lost, [
            "feature",
            "group",
            "robust_range_retention",
            "before_p01",
            "before_p99",
            "after_p01",
            "after_p99",
            "high_tail_loss_fraction_of_before_robust_range",
            "low_tail_loss_fraction_of_before_robust_range",
        ], 30),
        "## Largest High-Tail Losses",
        "",
        markdown_table(sorted_high_tail, [
            "feature",
            "group",
            "high_tail_loss_fraction_of_before_robust_range",
            "before_p99",
            "after_p99",
            "robust_range_retention",
        ], 30),
        "## Dynamics-Specific Losses",
        "",
        markdown_table(sorted_dynamics_lost, [
            "feature",
            "robust_range_retention",
            "before_p99",
            "after_p99",
            "high_tail_loss_fraction_of_before_robust_range",
            "mean_delta",
        ], 30),
        "## Output CSVs",
        "",
        "- `feature_range_retention.csv`",
        "- `group_range_retention.csv`",
        "- `category_count_retention.csv`",
        "- `dataset_count_retention.csv`",
        "- `subcategory_count_retention.csv`",
        "- `top_lost_dimensions_by_robust_range.csv`",
        "- `top_high_tail_losses.csv`",
        "- `top_lost_dynamics_dimensions.csv`",
    ]
    (output_dir / "diversity_coverage_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[diversity-compare] wrote {output_dir / 'diversity_coverage_comparison.md'}")


if __name__ == "__main__":
    main()
