#!/usr/bin/env python3
"""Build a GQS-passed motion set with per-subset score thresholds.

The score CSVs were generated from the ok-only retargeted CSVs, but older
reports may store paths under an `ok_only_csv` alias. This script maps those
paths back to the canonical `02_retargeted_ok_only_csv/motions` tree and copies
the selected motions into a new stage folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--source-motion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--default-threshold", type=float, default=90.0)
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Subset threshold in the form `seed/basic_locomotion_neutral=80` or `grab=80`.",
    )
    parser.add_argument("--skip-subset", action="append", default=["nutan"])
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_thresholds(items: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --threshold {item!r}; expected subset=value")
        key, value = item.split("=", 1)
        thresholds[key.strip("/")] = float(value)
    return thresholds


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def subset_from_score_file(score_root: Path, score_csv: Path) -> str:
    return score_csv.parent.relative_to(score_root).as_posix()


def rel_from_report_path(report_path: str) -> Path:
    path = Path(report_path)
    parts = path.parts
    for marker in ("ok_only_csv", "motions"):
        if marker in parts:
            idx = parts.index(marker)
            return Path(*parts[idx + 1 :])
    raise ValueError(f"Cannot infer relative motion path from {report_path!r}")


def resolve_source_path(source_motion_root: Path, report_path: str) -> tuple[Path, Path]:
    rel = rel_from_report_path(report_path)
    src = source_motion_root / rel
    return src, rel


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    (path / "motions").mkdir(parents=True, exist_ok=True)
    (path / "reports").mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    score_root = args.score_root.expanduser().resolve()
    source_motion_root = args.source_motion_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    thresholds = parse_thresholds(args.threshold)
    skip_subsets = {item.strip("/") for item in args.skip_subset}

    prepare_output(output_root, args.overwrite)

    selected_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, object]] = {}
    seen_rel_paths: set[str] = set()

    score_files = sorted(score_root.glob("**/physics_scores.csv"))
    if not score_files:
        raise FileNotFoundError(f"No physics_scores.csv found under {score_root}")

    for score_csv in score_files:
        subset = subset_from_score_file(score_root, score_csv)
        if subset in skip_subsets:
            continue
        threshold = thresholds.get(subset, args.default_threshold)
        counts = {
            "scored": 0,
            "selected": 0,
            "rejected": 0,
            "missing": 0,
            "duplicate": 0,
            "threshold": threshold,
        }
        with score_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                counts["scored"] += 1
                try:
                    score = float(row["score"])
                except Exception:
                    score = float("-inf")
                src, rel = resolve_source_path(source_motion_root, row.get("path", ""))
                rel_s = rel.as_posix()
                base_record: dict[str, object] = {
                    "subset": subset,
                    "threshold": threshold,
                    "motion": row.get("motion", src.stem),
                    "score": score,
                    "source_path": str(src),
                    "relative_path": rel_s,
                    "score_csv": str(score_csv),
                }
                if score < threshold:
                    counts["rejected"] += 1
                    rejected_rows.append({**base_record, "reason": "score_below_threshold"})
                    continue
                if rel_s in seen_rel_paths:
                    counts["duplicate"] += 1
                    rejected_rows.append({**base_record, "reason": "duplicate_relative_path"})
                    continue
                if not src.exists():
                    counts["missing"] += 1
                    missing_rows.append({**base_record, "reason": "source_missing"})
                    continue

                dst = output_root / "motions" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                seen_rel_paths.add(rel_s)
                counts["selected"] += 1
                selected_rows.append({**base_record, "output_path": str(dst)})
        summary[subset] = counts
        print(
            f"[gqs-select] {subset}: selected={counts['selected']} "
            f"rejected={counts['rejected']} missing={counts['missing']} threshold={threshold:g}"
        )

    selected_fields = [
        "subset",
        "threshold",
        "motion",
        "score",
        "source_path",
        "relative_path",
        "score_csv",
        "output_path",
    ]
    rejected_fields = [
        "subset",
        "threshold",
        "motion",
        "score",
        "source_path",
        "relative_path",
        "score_csv",
        "reason",
    ]
    write_csv(output_root / "reports" / "passed_manifest.csv", selected_rows, selected_fields)
    write_csv(output_root / "reports" / "failed_manifest.csv", rejected_rows, rejected_fields)
    write_csv(output_root / "reports" / "missing_manifest.csv", missing_rows, rejected_fields)
    write_csv(output_root / "reports" / "file_manifest.csv", selected_rows, selected_fields)

    summary_rows = []
    for subset, counts in sorted(summary.items()):
        summary_rows.append({"subset": subset, **counts})
    total_selected = len(selected_rows)
    summary_rows.append(
        {
            "subset": "ALL",
            "threshold": "",
            "scored": sum(int(row["scored"]) for row in summary.values()),
            "selected": total_selected,
            "rejected": sum(int(row["rejected"]) for row in summary.values()),
            "missing": sum(int(row["missing"]) for row in summary.values()),
            "duplicate": sum(int(row["duplicate"]) for row in summary.values()),
        }
    )
    write_csv(
        output_root / "reports" / "gqs_summary.csv",
        summary_rows,
        ["subset", "threshold", "scored", "selected", "rejected", "missing", "duplicate"],
    )
    meta = {
        "score_root": str(score_root),
        "source_motion_root": str(source_motion_root),
        "output_root": str(output_root),
        "default_threshold": args.default_threshold,
        "thresholds": thresholds,
        "skip_subsets": sorted(skip_subsets),
        "selected_total": total_selected,
        "expected_count": args.expected_count,
    }
    (output_root / "reports" / "selection_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    md = [
        "# GQS Mixed-Threshold Selection",
        "",
        f"Source motions: `{source_motion_root}`",
        f"Score root: `{score_root}`",
        f"Output motions: `{output_root / 'motions'}`",
        "",
        "## Thresholds",
        "",
        f"- Default threshold: `{args.default_threshold:g}`",
    ]
    for key, value in sorted(thresholds.items()):
        md.append(f"- `{key}`: `{value:g}`")
    md.extend(["", "## Counts", "", "| subset | threshold | scored | selected | rejected | missing | duplicate |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in summary_rows:
        md.append(
            f"| `{row['subset']}` | {row['threshold']} | {row['scored']} | {row['selected']} | "
            f"{row['rejected']} | {row['missing']} | {row['duplicate']} |"
        )
    (output_root / "reports" / "selection_provenance.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    if args.expected_count is not None and total_selected != args.expected_count:
        raise RuntimeError(f"Selected {total_selected}, expected {args.expected_count}")
    print(f"[gqs-select] total_selected={total_selected} output={output_root}")


if __name__ == "__main__":
    main()
