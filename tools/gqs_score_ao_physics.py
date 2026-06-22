#!/usr/bin/env python3
"""Score retargeted Agile-One/H4 robot CSV motions with GQS-style physics metrics.

This is the AO/H4 adaptation of Humanoid-GPT's GQS physics filter. It only
uses robot CSV motions and the target MJCF; it intentionally does not merge
retarget warning/failure manifests or IK branch-switch diagnostics.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from soma_retargeter.diagnostics.gqs_physics import (
    DEFAULT_H4_MJCF,
    GQSPhysicsConfig,
    GQSPhysicsScorer,
    GQSPhysicsResult,
    GQSPhysicsWeights,
    write_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, required=True, help="Root containing retargeted robot CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for physics_scores.csv/json.")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_H4_MJCF, help="H4/AO MJCF used for scoring.")
    parser.add_argument("--glob", default="**/*.csv", help="CSV glob relative to --motion-root.")
    parser.add_argument("--fps", type=float, default=120.0, help="Source CSV FPS.")
    parser.add_argument("--threshold", type=float, default=90.0, help="Pass threshold for final GQS score.")
    parser.add_argument("--min-duration-sec", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker count. Use 1 for easiest debugging.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of CSVs for smoke tests.")
    parser.add_argument("--copy-passed-root", type=Path, default=None, help="Optional root to copy passed CSVs into.")
    parser.add_argument("--joint-velocity-limit-deg-s", type=float, default=600.0)
    parser.add_argument("--foot-contact-height-m", type=float, default=0.05)
    parser.add_argument("--foot-slide-speed-threshold-m-s", type=float, default=0.10)
    parser.add_argument("--floating-clearance-m", type=float, default=0.05)
    parser.add_argument("--penetration-margin-m", type=float, default=0.01)
    parser.add_argument("--floor-size-m", type=float, default=10.0)
    parser.add_argument("--weight-foot-sliding", type=float, default=1.0)
    parser.add_argument("--weight-velocity-violation", type=float, default=500.0)
    parser.add_argument("--weight-self-collision", type=float, default=1000.0)
    parser.add_argument("--weight-jerk", type=float, default=0.1)
    parser.add_argument("--weight-penetration", type=float, default=10.0)
    parser.add_argument("--weight-floating-frames-ratio", type=float, default=200.0)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GQSPhysicsConfig:
    weights = GQSPhysicsWeights(
        foot_sliding=args.weight_foot_sliding,
        velocity_violation=args.weight_velocity_violation,
        self_collision=args.weight_self_collision,
        jerk=args.weight_jerk,
        penetration=args.weight_penetration,
        floating_frames_ratio=args.weight_floating_frames_ratio,
    )
    return GQSPhysicsConfig(
        fps=args.fps,
        min_duration_sec=args.min_duration_sec,
        foot_contact_height_m=args.foot_contact_height_m,
        foot_slide_speed_threshold_m_s=args.foot_slide_speed_threshold_m_s,
        floating_clearance_m=args.floating_clearance_m,
        penetration_margin_m=args.penetration_margin_m,
        joint_velocity_limit_deg_s=args.joint_velocity_limit_deg_s,
        floor_size_m=args.floor_size_m,
        enable_robot_self_contact=True,
        weights=weights,
    )


def discover_csvs(root: Path, pattern: str, limit: int | None) -> list[Path]:
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if limit is not None:
        paths = paths[: max(0, int(limit))]
    return paths


def score_one_with_scorer(
    scorer: GQSPhysicsScorer,
    path: Path,
    threshold: float,
) -> GQSPhysicsResult:
    try:
        return scorer.score_csv(path, threshold=threshold)
    except Exception as exc:
        return GQSPhysicsResult(
            motion=path.stem,
            path=str(path),
            num_frames=0,
            duration_sec=0.0,
            score=0.0,
            passed=False,
            metrics={},
            deductions={},
            error=f"{type(exc).__name__}: {exc}",
        )


def score_chunk(paths: list[Path], mjcf: Path, config: GQSPhysicsConfig, threshold: float) -> list[GQSPhysicsResult]:
    with GQSPhysicsScorer(mjcf_path=mjcf, config=config) as scorer:
        return [score_one_with_scorer(scorer, path, threshold) for path in paths]


def split_chunks(paths: list[Path], num_chunks: int) -> list[list[Path]]:
    num_chunks = max(1, min(num_chunks, len(paths)))
    return [paths[idx::num_chunks] for idx in range(num_chunks)]


def copy_passed(results: list[GQSPhysicsResult], motion_root: Path, output_root: Path) -> int:
    count = 0
    output_root.mkdir(parents=True, exist_ok=True)
    for result in results:
        if not result.passed or result.error:
            continue
        src = Path(result.path)
        rel = src.relative_to(motion_root) if src.is_relative_to(motion_root) else Path(src.name)
        dst = output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    motion_root = args.motion_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    mjcf = args.mjcf.expanduser().resolve()
    if not motion_root.exists():
        raise FileNotFoundError(motion_root)
    if not mjcf.exists():
        raise FileNotFoundError(mjcf)

    config = build_config(args)
    csvs = discover_csvs(motion_root, args.glob, args.limit)
    print(f"[gqs] motions={len(csvs)} root={motion_root}")
    print(f"[gqs] mjcf={mjcf}")
    print(f"[gqs] output={output_dir}")
    if not csvs:
        raise SystemExit("[gqs] no CSV files found")

    if args.workers <= 1:
        with GQSPhysicsScorer(mjcf_path=mjcf, config=config) as scorer:
            results = [
                score_one_with_scorer(scorer, path, args.threshold)
                for path in tqdm(csvs, desc="gqs physics")
            ]
    else:
        # Keep config immutable when crossing processes; replace is a cheap dataclass sanity check.
        config = replace(config)
        results = []
        chunks = split_chunks(csvs, args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(score_chunk, chunk, mjcf, config, args.threshold): chunk
                for chunk in chunks
                if chunk
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="gqs physics"):
                results.extend(future.result())
        results.sort(key=lambda result: result.path)

    write_results(results, output_dir, config, args.threshold)
    num_errors = sum(1 for result in results if result.error)
    num_passed = sum(1 for result in results if result.passed)
    print(f"[gqs] wrote {output_dir / 'physics_scores.csv'}")
    print(f"[gqs] passed={num_passed}/{len(results)} errors={num_errors} threshold={args.threshold:g}")

    if args.copy_passed_root is not None:
        copied = copy_passed(results, motion_root, args.copy_passed_root.expanduser().resolve())
        print(f"[gqs] copied_passed={copied} to {args.copy_passed_root}")

    if num_errors:
        print("[gqs] errors are recorded in physics_scores.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
