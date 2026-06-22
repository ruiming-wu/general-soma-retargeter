#!/usr/bin/env python3
"""Compare seed5 SOMA retargeting metrics for single-axis and triaxial configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/home/ruiming.wu/data/seed-retargeted")
TRI_ROOT = ROOT / "seed5_triaxial_motionlibs_20260528"
OUT_DIR = TRI_ROOT / "comparison_reports"

CATEGORIES = [
    "basic_locomotion_neutral",
    "environments",
    "household",
    "object_interaction",
    "object_manipulation",
]

KEY_METRICS = [
    "root_position_error_m",
    "root_rotation_error_deg",
    "ankle_position_error_m",
    "ankle_rotation_error_deg",
    "wrist_position_error_m",
    "wrist_rotation_error_deg",
    "head_rotation_error_deg",
    "rotation_error_deg/all_judgement_targets",
]


def latest_run_dir(parent: Path) -> Path:
    if (parent / "summary.json").exists():
        return parent
    candidates = sorted(p for p in parent.iterdir() if p.is_dir() and (p / "summary.json").exists())
    if not candidates:
        raise FileNotFoundError(f"No summary.json found under {parent}")
    return candidates[-1]


def run_dir(robot: str, config: str, category: str) -> Path:
    if robot == "ao" and config == "single":
        return latest_run_dir(ROOT / "soma_agile_one" / category)
    if robot == "g1" and config == "single":
        return latest_run_dir(ROOT / "soma_g1" / category)
    if robot == "ao" and config == "triaxial":
        return latest_run_dir(TRI_ROOT / "ao_motionlib" / "raw_csv" / category)
    if robot == "g1" and config == "triaxial":
        return latest_run_dir(TRI_ROOT / "g1_motionlib" / "raw_csv" / category)
    raise ValueError((robot, config, category))


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    clip_rows: list[pd.DataFrame] = []
    metric_rows: list[pd.DataFrame] = []
    event_rows: list[pd.DataFrame] = []

    for robot in ["ao", "g1"]:
        for config in ["single", "triaxial"]:
            for category in CATEGORIES:
                d = run_dir(robot, config, category)
                summary = json.loads((d / "summary.json").read_text())
                counts = summary.get("counts", {})
                rates = summary.get("rates", {})
                summary_rows.append(
                    {
                        "robot": robot,
                        "config": config,
                        "category": category,
                        "run_dir": str(d),
                        "retargeter_config": summary.get("retargeter_config"),
                        "clips_total": counts.get("clips_total", np.nan),
                        "clips_success": counts.get("clips_succeeded", np.nan),
                        "clips_warning": counts.get("clips_warned", np.nan),
                        "clips_failure": counts.get("clips_failed", np.nan),
                        "success_rate": rates.get("success_rate", np.nan),
                        "warning_rate": rates.get("warning_rate", np.nan),
                        "failure_rate": rates.get("failure_rate", np.nan),
                        "elapsed_sec": summary.get("elapsed_sec", np.nan),
                        "batch_size": summary.get("batch_size", np.nan),
                    }
                )

                clips = read_csv_if_exists(d / "clips.csv")
                if not clips.empty:
                    clips["robot_key"] = robot
                    clips["config"] = config
                    clips["category"] = category
                    clips["source_name"] = clips["source_path"].map(lambda p: Path(str(p)).stem)
                    clips["run_dir"] = str(d)
                    clips["motion_csv_exists"] = clips.apply(
                        lambda r: Path(str(r["motion_csv"])).exists()
                        or (d / "motions" / f"{r['source_name']}.csv").exists(),
                        axis=1,
                    )
                    clip_rows.append(clips)

                metrics = read_csv_if_exists(d / "clip_metrics.csv")
                if not metrics.empty:
                    metrics["robot_key"] = robot
                    metrics["config"] = config
                    metrics["category"] = category
                    metrics["source_name"] = metrics["source_path"].map(lambda p: Path(str(p)).stem)
                    metric_rows.append(metrics)

                for file_name, event_type in [("warnings.csv", "warning"), ("failures.csv", "failure")]:
                    events = read_csv_if_exists(d / file_name)
                    if not events.empty:
                        events["robot_key"] = robot
                        events["config"] = config
                        events["category"] = category
                        events["event_type"] = event_type
                        events["source_name"] = events["source_path"].map(lambda p: Path(str(p)).stem)
                        event_rows.append(events)

    return (
        pd.DataFrame(summary_rows),
        pd.concat(clip_rows, ignore_index=True) if clip_rows else pd.DataFrame(),
        pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame(),
        pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame(),
    )


def aggregate_metrics(metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, g in metrics.groupby(group_cols + ["metric"], dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_dict = dict(zip(group_cols + ["metric"], key))
        count = pd.to_numeric(g["count"], errors="coerce")
        mean = pd.to_numeric(g["mean"], errors="coerce")
        max_v = pd.to_numeric(g["max"], errors="coerce")
        total_count = float(count.sum())
        weighted_mean = float((mean * count).sum() / total_count) if total_count else np.nan
        rows.append(
            {
                **key_dict,
                "clips": int(g["source_name"].nunique()) if "source_name" in g else len(g),
                "samples": int(total_count),
                "mean": weighted_mean,
                "clip_mean_mean": float(mean.mean()),
                "clip_mean_p50": float(mean.quantile(0.50)),
                "clip_mean_p90": float(mean.quantile(0.90)),
                "clip_mean_p95": float(mean.quantile(0.95)),
                "clip_max_p95": float(max_v.quantile(0.95)),
                "max": float(max_v.max()),
            }
        )
    return pd.DataFrame(rows)


def parse_frame_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    is_position = df["metric"].str.startswith("position_error_m/")
    is_rotation = df["metric"].str.startswith("rotation_error_deg/") & ~df["metric"].str.endswith(
        "all_judgement_targets"
    )
    df = df[is_position | is_rotation].copy()
    df["error_type"] = np.where(df["metric"].str.startswith("position_error_m/"), "position_m", "rotation_deg")
    target = df["metric"].str.split("->", n=1).str[-1]
    df["frame"] = target
    df["robot_body"] = df["metric"].str.split("/", n=1).str[-1].str.split("->", n=1).str[0]
    return df


def compare_single_triaxial(df: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    single = df[df["config"] == "single"].copy()
    tri = df[df["config"] == "triaxial"].copy()
    metric_cols = [c for c in ["mean", "clip_mean_p95", "clip_max_p95", "max"] if c in df.columns]
    merged = single[index_cols + metric_cols].merge(
        tri[index_cols + metric_cols], on=index_cols, suffixes=("_single", "_triaxial")
    )
    for c in metric_cols:
        merged[f"{c}_delta_tri_minus_single"] = merged[f"{c}_triaxial"] - merged[f"{c}_single"]
        merged[f"{c}_delta_pct"] = np.where(
            merged[f"{c}_single"].abs() > 1e-12,
            100.0 * (merged[f"{c}_triaxial"] / merged[f"{c}_single"] - 1.0),
            np.nan,
        )
    return merged


def fmt_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{100.0 * x:.2f}%"


def fmt_num(x: float, nd: int = 4) -> str:
    if pd.isna(x):
        return ""
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.{nd}f}"


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._\n"
    view = df.loc[:, columns].copy()
    for c in view.columns:
        if c.endswith("_pctpt"):
            view[c] = view[c].map(fmt_num)
        elif c.endswith("_rate") or "_rate_" in c:
            view[c] = view[c].map(fmt_pct)
        elif pd.api.types.is_float_dtype(view[c]) or pd.api.types.is_integer_dtype(view[c]):
            view[c] = view[c].map(fmt_num)
    view = view.astype(str).replace({"nan": "", "None": ""})
    widths = []
    for c in view.columns:
        values = [c] + view[c].tolist()
        widths.append(max(len(v) for v in values))
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(view.columns, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    rows = [
        "| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, clips, metrics, events = load_all()

    metric_cat = aggregate_metrics(metrics, ["robot_key", "config", "category"])
    metric_all = aggregate_metrics(metrics, ["robot_key", "config"])
    frame_metrics = parse_frame_metrics(metrics)
    frame_cat = aggregate_metrics(frame_metrics, ["robot_key", "config", "category", "frame", "error_type"])
    frame_all = aggregate_metrics(frame_metrics, ["robot_key", "config", "frame", "error_type"])

    key_cat = metric_cat[metric_cat["metric"].isin(KEY_METRICS)].copy()
    key_all = metric_all[metric_all["metric"].isin(KEY_METRICS)].copy()
    cmp_key_cat = compare_single_triaxial(key_cat, ["robot_key", "category", "metric"])
    cmp_key_all = compare_single_triaxial(key_all, ["robot_key", "metric"])
    cmp_frame_all = compare_single_triaxial(frame_all, ["robot_key", "frame", "error_type"])
    cmp_frame_cat = compare_single_triaxial(frame_cat, ["robot_key", "category", "frame", "error_type"])

    status_cmp = summary.pivot_table(
        index=["robot", "category"],
        columns="config",
        values=["clips_total", "clips_success", "clips_warning", "clips_failure", "success_rate", "warning_rate", "failure_rate"],
        aggfunc="first",
    )
    status_cmp.columns = [f"{a}_{b}" for a, b in status_cmp.columns]
    status_cmp = status_cmp.reset_index()
    if {"success_rate_triaxial", "success_rate_single"}.issubset(status_cmp.columns):
        status_cmp["success_rate_delta_pctpt"] = 100.0 * (
            status_cmp["success_rate_triaxial"] - status_cmp["success_rate_single"]
        )
        status_cmp["warning_rate_delta_pctpt"] = 100.0 * (
            status_cmp["warning_rate_triaxial"] - status_cmp["warning_rate_single"]
        )
        status_cmp["failure_rate_delta_pctpt"] = 100.0 * (
            status_cmp["failure_rate_triaxial"] - status_cmp["failure_rate_single"]
        )

    # Event tables.
    if not events.empty:
        event_reason = (
            events.groupby(["robot_key", "config", "category", "event_type", "reason", "joint", "robot_body"], dropna=False)
            .size()
            .reset_index(name="events")
            .sort_values(["robot_key", "config", "category", "events"], ascending=[True, True, True, False])
        )
        event_clip_reason = (
            events.groupby(["robot_key", "config", "category", "event_type", "reason"], dropna=False)["source_name"]
            .nunique()
            .reset_index(name="clips")
            .sort_values(["robot_key", "config", "category", "clips"], ascending=[True, True, True, False])
        )
    else:
        event_reason = pd.DataFrame()
        event_clip_reason = pd.DataFrame()

    # Saved-output audit: warning/failure clips should still have motion_csv.
    if not clips.empty:
        saved_audit = (
            clips.groupby(["robot_key", "config", "category", "status"], dropna=False)["motion_csv_exists"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "clips", "sum": "saved_csvs"})
        )
        saved_audit["saved_rate"] = saved_audit["saved_csvs"] / saved_audit["clips"]
    else:
        saved_audit = pd.DataFrame()

    # Persist detailed tables.
    summary.to_csv(OUT_DIR / "single_vs_triaxial_status_by_category.csv", index=False)
    metric_cat.to_csv(OUT_DIR / "single_vs_triaxial_metrics_by_category.csv", index=False)
    frame_cat.to_csv(OUT_DIR / "single_vs_triaxial_frame_metrics_by_category.csv", index=False)
    cmp_key_cat.to_csv(OUT_DIR / "single_vs_triaxial_key_metric_deltas_by_category.csv", index=False)
    cmp_frame_cat.to_csv(OUT_DIR / "single_vs_triaxial_frame_metric_deltas_by_category.csv", index=False)
    if not event_reason.empty:
        event_reason.to_csv(OUT_DIR / "single_vs_triaxial_event_reasons.csv", index=False)
    if not saved_audit.empty:
        saved_audit.to_csv(OUT_DIR / "single_vs_triaxial_saved_csv_audit.csv", index=False)

    report: list[str] = []
    report.append("# Seed5 SOMA Retargeting: Single-Axis vs Triaxial Comparison\n")
    report.append("Generated from existing `summary.json`, `clips.csv`, `clip_metrics.csv`, `warnings.csv`, and `failures.csv` files. No retargeting was re-run.\n")
    report.append("\n## Inputs\n")
    for _, r in summary.iterrows():
        report.append(
            f"- `{r.robot}/{r.config}/{r.category}`: `{r.run_dir}`"
        )
    report.append("\n## Interpretation Notes\n")
    report.append("- `single` means the earlier SOMA retarget configs under `soma_agile_one` and `soma_g1`.")
    report.append("- `triaxial` means the new solved SOMA triaxial configs under `seed5_triaxial_motionlibs_20260528`.")
    report.append("- Error means are weighted by evaluated frame/sample count from `clip_metrics.csv`; lower is better.")
    report.append("- `success` means no warning/failure under the thresholds: position warn/fail `0.25/0.5 m`, rotation warn/fail `45/90 deg`.")
    report.append("- Warning/failure clips are still audited for whether a robot CSV was saved.\n")

    report.append("\n## Recommendation\n")
    report.append("- **AO**: use the old single-axis config for training for now.")
    report.append(
        "  The triaxial AO result improves wrist/root averages, but it consistently worsens ankle position/rotation and slightly increases warning/failure pressure in object categories. Given the current AO training failure mode is ankle/leg posture, the safer training set is the single-axis AO motionlib. Use triaxial AO only if the next experiment explicitly prioritizes upper-body/manipulation visual quality over lower-body stability."
    )
    report.append("- **G1**: use the triaxial config for full-body training; keep single-axis as the locomotion-only fallback.")
    report.append(
        "  G1 triaxial slightly worsens lower-body/root errors, but the absolute errors remain small, success coverage is slightly better, and wrist position/rotation improves substantially. If the G1 policy is meant to cover upper-body operation as well as locomotion, triaxial is the better training candidate."
    )
    report.append("\n")

    report.append("## Coverage And Status By Category\n")
    report.append(md_table(status_cmp, [
        "robot",
        "category",
        "clips_total_single",
        "clips_total_triaxial",
        "success_rate_single",
        "success_rate_triaxial",
        "warning_rate_single",
        "warning_rate_triaxial",
        "failure_rate_single",
        "failure_rate_triaxial",
        "success_rate_delta_pctpt",
    ]))
    report.append("\n")

    report.append("## Overall Key Metrics\n")
    overall_view = key_all.sort_values(["robot_key", "metric", "config"])
    report.append(md_table(overall_view, [
        "robot_key",
        "config",
        "metric",
        "clips",
        "mean",
        "clip_mean_p95",
        "clip_max_p95",
        "max",
    ]))
    report.append("\n")

    report.append("## Overall Single-To-Triaxial Delta\n")
    report.append("Negative delta means triaxial is better; positive means triaxial is worse.\n")
    report.append(md_table(cmp_key_all.sort_values(["robot_key", "metric"]), [
        "robot_key",
        "metric",
        "mean_single",
        "mean_triaxial",
        "mean_delta_tri_minus_single",
        "mean_delta_pct",
        "clip_max_p95_single",
        "clip_max_p95_triaxial",
        "clip_max_p95_delta_pct",
    ]))
    report.append("\n")

    report.append("## Key Metrics By Category\n")
    cat_view = key_cat.sort_values(["robot_key", "category", "metric", "config"])
    report.append(md_table(cat_view, [
        "robot_key",
        "category",
        "config",
        "metric",
        "clips",
        "mean",
        "clip_mean_p95",
        "clip_max_p95",
        "max",
    ]))
    report.append("\n")

    report.append("## Category-Level Single-To-Triaxial Delta\n")
    report.append(md_table(cmp_key_cat.sort_values(["robot_key", "category", "metric"]), [
        "robot_key",
        "category",
        "metric",
        "mean_single",
        "mean_triaxial",
        "mean_delta_tri_minus_single",
        "mean_delta_pct",
        "clip_max_p95_delta_pct",
    ]))
    report.append("\n")

    report.append("## Overall Metrics By Judgement Frame\n")
    frame_overall_view = frame_all.sort_values(["robot_key", "frame", "error_type", "config"])
    report.append(md_table(frame_overall_view, [
        "robot_key",
        "frame",
        "error_type",
        "config",
        "clips",
        "mean",
        "clip_mean_p95",
        "clip_max_p95",
        "max",
    ]))
    report.append("\n")

    report.append("## Judgement Frame Single-To-Triaxial Delta\n")
    report.append(md_table(cmp_frame_all.sort_values(["robot_key", "frame", "error_type"]), [
        "robot_key",
        "frame",
        "error_type",
        "mean_single",
        "mean_triaxial",
        "mean_delta_tri_minus_single",
        "mean_delta_pct",
        "clip_max_p95_delta_pct",
    ]))
    report.append("\n")

    report.append("## Judgement Frame Metrics By Category\n")
    report.append("Full detailed table is also saved to `single_vs_triaxial_frame_metrics_by_category.csv`.\n")
    report.append(md_table(frame_cat.sort_values(["robot_key", "category", "frame", "error_type", "config"]), [
        "robot_key",
        "category",
        "frame",
        "error_type",
        "config",
        "mean",
        "clip_mean_p95",
        "clip_max_p95",
        "max",
    ]))
    report.append("\n")

    report.append("## Top Triaxial Regressions By Category/Frame\n")
    regress = cmp_frame_cat.sort_values("mean_delta_pct", ascending=False).head(30)
    report.append(md_table(regress, [
        "robot_key",
        "category",
        "frame",
        "error_type",
        "mean_single",
        "mean_triaxial",
        "mean_delta_pct",
        "clip_max_p95_delta_pct",
    ]))
    report.append("\n")

    report.append("## Top Triaxial Improvements By Category/Frame\n")
    improve = cmp_frame_cat.sort_values("mean_delta_pct", ascending=True).head(30)
    report.append(md_table(improve, [
        "robot_key",
        "category",
        "frame",
        "error_type",
        "mean_single",
        "mean_triaxial",
        "mean_delta_pct",
        "clip_max_p95_delta_pct",
    ]))
    report.append("\n")

    report.append("## Warning / Failure Sources\n")
    if event_clip_reason.empty:
        report.append("_No warnings/failures._\n")
    else:
        report.append("Clip counts by reason:\n")
        report.append(md_table(event_clip_reason, [
            "robot_key",
            "config",
            "category",
            "event_type",
            "reason",
            "clips",
        ]))
        report.append("\nEvent counts by reason/body:\n")
        report.append(md_table(event_reason, [
            "robot_key",
            "config",
            "category",
            "event_type",
            "reason",
            "joint",
            "robot_body",
            "events",
        ]))
    report.append("\n")

    report.append("## Saved CSV Audit For Non-Success Clips\n")
    if saved_audit.empty:
        report.append("_No clip audit available._\n")
    else:
        report.append(md_table(saved_audit[~saved_audit["status"].isin(["success", "succeeded"])].sort_values(["robot_key", "config", "category", "status"]), [
            "robot_key",
            "config",
            "category",
            "status",
            "clips",
            "saved_csvs",
            "saved_rate",
        ]))
    report.append("\n")

    report.append("## Output Files\n")
    for name in [
        "single_vs_triaxial_status_by_category.csv",
        "single_vs_triaxial_metrics_by_category.csv",
        "single_vs_triaxial_frame_metrics_by_category.csv",
        "single_vs_triaxial_key_metric_deltas_by_category.csv",
        "single_vs_triaxial_frame_metric_deltas_by_category.csv",
        "single_vs_triaxial_event_reasons.csv",
        "single_vs_triaxial_saved_csv_audit.csv",
    ]:
        path = OUT_DIR / name
        if path.exists():
            report.append(f"- `{path}`")

    report_path = OUT_DIR / "single_vs_triaxial_comparison.md"
    report_path.write_text("\n".join(report) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
