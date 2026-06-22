#!/usr/bin/env python3
"""Repair specified local branch-switch jumps in retargeted motion CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from soma_retargeter.diagnostics.branch_repair import BranchRepairConfig, repair_joint_branch_jump_deg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", type=Path, required=True, help="JSON list of repair cases.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--pre-frames", type=int, default=4)
    parser.add_argument("--post-frames", type=int, default=8)
    parser.add_argument("--slope-clip-deg-per-frame", type=float, default=2.0)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.expanduser().open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fieldnames:
        raise ValueError(f"{path} has no CSV header")
    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def repair_case(case: dict, output_dir: Path, config: BranchRepairConfig) -> dict:
    csv_path = Path(case["csv"]).expanduser()
    joint = str(case["joint"])
    event_frame = int(case["event_frame"])
    name = str(case.get("name") or csv_path.stem)
    col = f"{joint}_dof"

    fieldnames, rows = read_csv(csv_path)
    if col not in fieldnames:
        raise ValueError(f"{csv_path} does not contain {col}")
    values = np.asarray([float(row[col]) for row in rows], dtype=np.float64)
    repaired, result = repair_joint_branch_jump_deg(values, event_frame0=event_frame, config=config)

    for row, value in zip(rows, repaired, strict=True):
        row[col] = f"{float(value):.10g}"

    out_csv = output_dir / "repaired_csv" / f"{name}.csv"
    write_csv(out_csv, fieldnames, rows)
    record = {
        "name": name,
        "source_csv": str(csv_path),
        "repaired_csv": str(out_csv),
        "joint": joint,
        **asdict(result),
    }
    return record


def main() -> None:
    args = parse_args()
    cases = json.loads(args.case_json.expanduser().read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("--case-json must contain a JSON list")
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = BranchRepairConfig(
        pre_frames=args.pre_frames,
        post_frames=args.post_frames,
        slope_clip_deg_per_frame=args.slope_clip_deg_per_frame,
        fps=args.fps,
    )

    records = [repair_case(case, output_dir, config) for case in cases]
    manifest = output_dir / "repair_manifest.csv"
    if records:
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    (output_dir / "repair_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
