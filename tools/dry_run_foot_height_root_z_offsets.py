#!/usr/bin/env python3
"""Dry-run per-motion root-z offsets from H4 foot collision bottoms."""

from __future__ import annotations

import argparse
import csv
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from soma_retargeter.diagnostics.gqs_physics import (
    build_model_qpos,
    foot_collision_geom_ids,
    geom_bottom_z,
    load_robot_csv_motion,
)


DEFAULT_XML = Path("/home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml")
_WORKER_MODEL: mujoco.MjModel | None = None
_WORKER_DATA: mujoco.MjData | None = None
_WORKER_FOOT_IDS: list[int] | None = None


@dataclass
class HeightDryRunResult:
    rel_path: str
    category: str
    num_frames: int
    num_foot_geoms: int
    offset_m: float
    offset_cm_csv_root_z: float
    before_global_min_m: float
    before_global_median_min_foot_m: float
    before_left_min_m: float
    before_right_min_m: float
    strict_min_offset_m: float
    chosen_offset_source: str
    chosen_offset_confidence: str
    high_conf_double_stance_frame_count: int
    high_conf_double_stance_frame_ratio: float
    stance_target_offset_m: float
    support_fallback_offset_m: float
    chosen_offset_m: float
    chosen_offset_cm_csv_root_z: float
    after_global_min_m: float
    max_penetration_after_m: float
    after_min_foot_p05_m: float
    after_min_foot_p50_m: float
    after_min_foot_p95_m: float
    support_sample_count: int
    support_median_raw_m: float
    support_after_p50_m: float
    support_after_p95_m: float
    support_frame_ratio: float
    support_min_foot_p50_m: float
    support_min_foot_p95_m: float
    double_support_frame_ratio: float
    double_support_left_p50_m: float
    double_support_right_p50_m: float
    double_support_abs_lr_diff_p50_m: float
    review_frame: int
    review_frame_left_bottom_after_m: float
    review_frame_right_bottom_after_m: float
    view_command: str
    status: str
    reasons: str


def make_loadable_mjcf(base_xml: Path) -> Path:
    text = base_xml.expanduser().resolve().read_text(encoding="utf-8")
    meshdir = (base_xml.parent / "../meshes/visual").resolve()
    text = text.replace('meshdir="../meshes/visual"', f'meshdir="{meshdir}"')
    tmp = tempfile.NamedTemporaryFile(prefix=f"{base_xml.stem}_height_dryrun_", suffix=".xml", delete=False)
    out = Path(tmp.name)
    tmp.close()
    out.write_text(text, encoding="utf-8")
    return out


def category_from_rel(rel_path: Path) -> str:
    parts = rel_path.parts
    if len(parts) >= 2 and parts[0] == "seed":
        return parts[1]
    if parts:
        return parts[0]
    return "unknown"


def foot_side(name: str) -> str | None:
    if name.startswith("left_foot"):
        return "left"
    if name.startswith("right_foot"):
        return "right"
    return None


def quantile_or_nan(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def finite_speed(positions: np.ndarray, fps: float = 120.0) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    speed = np.zeros(positions.shape[0], dtype=np.float64)
    if positions.shape[0] > 1:
        diff = np.linalg.norm(positions[1:] - positions[:-1], axis=1) * float(fps)
        speed[1:] = diff
        speed[0] = diff[0]
    return speed


def support_mask_for_side(bottoms: np.ndarray, centers: np.ndarray, fps: float, low_margin_m: float, max_speed_m_s: float) -> np.ndarray:
    z_ref = float(np.quantile(bottoms, 0.10))
    speed = finite_speed(centers, fps=fps)
    mask = (bottoms <= z_ref + low_margin_m) & (speed <= max_speed_m_s)
    if np.count_nonzero(mask) < max(5, int(0.005 * bottoms.shape[0])):
        # Some slow manipulation clips have tiny foot movement and narrow z spread.
        # Fall back to low-height frames so fully-floating clips can still be lowered.
        mask = bottoms <= float(np.quantile(bottoms, 0.20))
    return mask


def write_adjusted_csv(src: Path, dst: Path, offset_cm: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", newline="", encoding="utf-8") as f, dst.open("w", newline="", encoding="utf-8") as g:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {src}")
        if "root_translateZ" not in reader.fieldnames:
            raise ValueError(f"CSV has no root_translateZ column: {src}")
        writer = csv.DictWriter(g, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row["root_translateZ"] = f"{float(row['root_translateZ']) + offset_cm:.9f}"
            writer.writerow(row)


def init_worker(loadable_xml: str) -> None:
    global _WORKER_MODEL, _WORKER_DATA, _WORKER_FOOT_IDS
    _WORKER_MODEL = mujoco.MjModel.from_xml_path(loadable_xml)
    _WORKER_DATA = mujoco.MjData(_WORKER_MODEL)
    _WORKER_FOOT_IDS = foot_collision_geom_ids(_WORKER_MODEL)


def analyze_csv_job(job: dict[str, Any]) -> tuple[int, HeightDryRunResult]:
    if _WORKER_MODEL is None or _WORKER_DATA is None or _WORKER_FOOT_IDS is None:
        raise RuntimeError("Worker MuJoCo model was not initialized.")
    idx = int(job["idx"])
    result = analyze_csv(
        csv_path=Path(job["csv_path"]),
        csv_root=Path(job["csv_root"]),
        model=_WORKER_MODEL,
        data=_WORKER_DATA,
        foot_ids=_WORKER_FOOT_IDS,
        support_height_m=float(job["support_height_m"]),
        double_support_height_m=float(job["double_support_height_m"]),
        lr_balance_tolerance_m=float(job["lr_balance_tolerance_m"]),
        max_abs_offset_warn_m=float(job["max_abs_offset_warn_m"]),
        support_low_margin_m=float(job["support_low_margin_m"]),
        support_max_speed_m_s=float(job["support_max_speed_m_s"]),
        target_support_clearance_m=float(job["target_support_clearance_m"]),
        high_conf_double_height_m=float(job["high_conf_double_height_m"]),
        high_conf_lr_gap_m=float(job["high_conf_lr_gap_m"]),
        review_penetration_m=float(job["review_penetration_m"]),
        viewer_script=Path(job["viewer_script"]),
        write_root=Path(job["write_root"]) if job["write_root"] else None,
        write_filter=str(job["write_filter"]),
    )
    return idx, result


def analyze_csv(
    csv_path: Path,
    csv_root: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_ids: list[int],
    support_height_m: float,
    double_support_height_m: float,
    lr_balance_tolerance_m: float,
    max_abs_offset_warn_m: float,
    support_low_margin_m: float,
    support_max_speed_m_s: float,
    target_support_clearance_m: float,
    high_conf_double_height_m: float,
    high_conf_lr_gap_m: float,
    review_penetration_m: float,
    viewer_script: Path,
    write_root: Path | None = None,
    write_filter: str = "all",
) -> HeightDryRunResult:
    motion = load_robot_csv_motion(csv_path)
    qpos = build_model_qpos(motion, model)
    rel = csv_path.relative_to(csv_root)

    geom_names = {
        gid: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        for gid in foot_ids
    }
    left_ids = [gid for gid in foot_ids if foot_side(geom_names[gid]) == "left"]
    right_ids = [gid for gid in foot_ids if foot_side(geom_names[gid]) == "right"]
    if not left_ids or not right_ids:
        raise RuntimeError(f"Expected both left and right foot collision geoms, got {len(left_ids)} left/{len(right_ids)} right")

    left_min = np.empty(qpos.shape[0], dtype=np.float64)
    right_min = np.empty(qpos.shape[0], dtype=np.float64)
    all_min = np.empty(qpos.shape[0], dtype=np.float64)
    left_center = np.empty((qpos.shape[0], 3), dtype=np.float64)
    right_center = np.empty((qpos.shape[0], 3), dtype=np.float64)

    for frame in range(qpos.shape[0]):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        left_bottoms = [geom_bottom_z(model, data, gid) for gid in left_ids]
        right_bottoms = [geom_bottom_z(model, data, gid) for gid in right_ids]
        left_min[frame] = min(left_bottoms)
        right_min[frame] = min(right_bottoms)
        all_min[frame] = min(left_min[frame], right_min[frame])
        left_center[frame] = np.mean([np.asarray(data.geom_xpos[gid], dtype=np.float64) for gid in left_ids], axis=0)
        right_center[frame] = np.mean([np.asarray(data.geom_xpos[gid], dtype=np.float64) for gid in right_ids], axis=0)

    before_global_min = float(np.min(all_min))
    strict_min_offset = -before_global_min

    left_support = support_mask_for_side(
        left_min,
        left_center,
        fps=motion.fps,
        low_margin_m=support_low_margin_m,
        max_speed_m_s=support_max_speed_m_s,
    )
    right_support = support_mask_for_side(
        right_min,
        right_center,
        fps=motion.fps,
        low_margin_m=support_low_margin_m,
        max_speed_m_s=support_max_speed_m_s,
    )
    support_values = np.concatenate([left_min[left_support], right_min[right_support]])
    if support_values.size == 0:
        support_values = all_min[all_min <= float(np.quantile(all_min, 0.20))]

    support_median_raw = float(np.median(support_values))
    support_fallback_offset = float(target_support_clearance_m - support_median_raw)

    high_conf_double_raw_mask = (
        left_support
        & right_support
        & (np.maximum(left_min, right_min) <= high_conf_double_height_m)
        & (np.abs(left_min - right_min) <= high_conf_lr_gap_m)
    )
    high_conf_double_values = np.concatenate(
        [left_min[high_conf_double_raw_mask], right_min[high_conf_double_raw_mask]]
    )
    if high_conf_double_values.size > 0:
        stance_target_offset = float(target_support_clearance_m - float(np.median(high_conf_double_values)))
        offset = stance_target_offset
        chosen_source = "high_conf_double_stance"
        confidence = "high"
    elif support_values.size > 0:
        stance_target_offset = float("nan")
        offset = support_fallback_offset
        chosen_source = "single_foot_support_fallback"
        confidence = "low_no_double_stance"
    else:
        stance_target_offset = float("nan")
        offset = strict_min_offset
        chosen_source = "strict_min_fallback"
        confidence = "low_no_support"

    after_left = left_min + offset
    after_right = right_min + offset
    after_min = np.minimum(after_left, after_right)
    max_penetration = max(0.0, -float(np.min(after_min)))

    support_mask = after_min <= support_height_m
    double_support_mask = (
        (after_left <= double_support_height_m)
        & (after_right <= double_support_height_m)
        & (np.abs(after_left - after_right) <= lr_balance_tolerance_m)
    )

    reasons: list[str] = []
    support_after = support_values + offset
    notes: list[str] = []
    if abs(offset) > max_abs_offset_warn_m:
        notes.append(f"large_abs_offset>{max_abs_offset_warn_m:.3f}m")
    if confidence != "high":
        notes.append(confidence)
    if support_values.size < max(10, int(0.01 * qpos.shape[0])):
        notes.append("few_support_candidates")
    if np.mean(support_mask) < 0.01:
        notes.append("almost_no_support_frames_after_offset")
    if not double_support_mask.any():
        notes.append("no_balanced_double_support_frames_after_offset")
    if max_penetration > review_penetration_m:
        reasons.append(f"max_penetration>{review_penetration_m:.3f}m")

    status = "ok" if not reasons else "review"
    if reasons:
        review_frame = int(np.argmin(after_min))
    elif high_conf_double_raw_mask.any():
        candidates = np.where(high_conf_double_raw_mask)[0]
        target_value = float(np.median(np.minimum(after_left[candidates], after_right[candidates])))
        review_frame = int(candidates[np.argmin(np.abs(np.minimum(after_left[candidates], after_right[candidates]) - target_value))])
    else:
        review_frame = int(np.argmin(after_min))

    view_command = (
        f"cd /home/ruiming.wu/codes/general-soma-retargeter && "
        f"uv run python {viewer_script} "
        f"--xml /home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml "
        f"--csv {csv_path} --frame {review_frame} "
        f"--root-z-offset-cm {offset * 100.0:.6f} --show-visual"
    )
    should_write = True
    if write_filter == "ok":
        should_write = status == "ok"
    elif write_filter == "ok_high_conf":
        should_write = status == "ok" and confidence == "high"
    if write_root is not None and should_write:
        write_adjusted_csv(csv_path, write_root / rel, offset * 100.0)

    return HeightDryRunResult(
        rel_path=str(rel),
        category=category_from_rel(rel),
        num_frames=int(qpos.shape[0]),
        num_foot_geoms=len(foot_ids),
        offset_m=float(offset),
        offset_cm_csv_root_z=float(offset * 100.0),
        before_global_min_m=before_global_min,
        before_global_median_min_foot_m=float(np.median(all_min)),
        before_left_min_m=float(np.min(left_min)),
        before_right_min_m=float(np.min(right_min)),
        strict_min_offset_m=float(strict_min_offset),
        chosen_offset_source=chosen_source,
        chosen_offset_confidence=confidence,
        high_conf_double_stance_frame_count=int(np.count_nonzero(high_conf_double_raw_mask)),
        high_conf_double_stance_frame_ratio=float(np.mean(high_conf_double_raw_mask)),
        stance_target_offset_m=float(stance_target_offset),
        support_fallback_offset_m=float(support_fallback_offset),
        chosen_offset_m=float(offset),
        chosen_offset_cm_csv_root_z=float(offset * 100.0),
        after_global_min_m=float(np.min(after_min)),
        max_penetration_after_m=float(max_penetration),
        after_min_foot_p05_m=quantile_or_nan(after_min, 0.05),
        after_min_foot_p50_m=quantile_or_nan(after_min, 0.50),
        after_min_foot_p95_m=quantile_or_nan(after_min, 0.95),
        support_sample_count=int(support_values.size),
        support_median_raw_m=float(support_median_raw),
        support_after_p50_m=quantile_or_nan(support_after, 0.50),
        support_after_p95_m=quantile_or_nan(support_after, 0.95),
        support_frame_ratio=float(np.mean(support_mask)),
        support_min_foot_p50_m=quantile_or_nan(after_min[support_mask], 0.50),
        support_min_foot_p95_m=quantile_or_nan(after_min[support_mask], 0.95),
        double_support_frame_ratio=float(np.mean(double_support_mask)),
        double_support_left_p50_m=quantile_or_nan(after_left[double_support_mask], 0.50),
        double_support_right_p50_m=quantile_or_nan(after_right[double_support_mask], 0.50),
        double_support_abs_lr_diff_p50_m=quantile_or_nan(np.abs(after_left[double_support_mask] - after_right[double_support_mask]), 0.50),
        review_frame=review_frame,
        review_frame_left_bottom_after_m=float(after_left[review_frame]),
        review_frame_right_bottom_after_m=float(after_right[review_frame]),
        view_command=view_command,
        status=status,
        reasons=";".join(reasons + notes),
    )


def sample_csvs(csv_root: Path, sample_size: int, seed: int, use_all: bool) -> list[Path]:
    all_csvs = sorted(csv_root.rglob("*.csv"))
    if use_all:
        return all_csvs
    if sample_size >= len(all_csvs):
        return all_csvs
    rng = random.Random(seed)
    return sorted(rng.sample(all_csvs, sample_size))


def write_outputs(output_dir: Path, results: list[HeightDryRunResult], xml: Path, csv_root: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.__dict__ for r in results]
    fieldnames = list(rows[0].keys()) if rows else []
    with (output_dir / "foot_height_offset_dry_run.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Foot Height Root-Z Offset Dry Run",
        "",
        f"- XML: `{xml}`",
        f"- CSV root: `{csv_root}`",
        f"- Motions checked: {len(results)}",
        "- Offset convention: add `chosen_offset_cm_csv_root_z` to CSV `root_translateZ`.",
        "- `strict_min_offset_m` is `-min(all left/right foot collision capsule bottom z over the full motion)`.",
        "- `chosen_offset_m` prioritizes high-confidence double-stance frames; if absent, it falls back to low/slow single-foot support.",
        "- `status=review` only means post-offset maximum penetration exceeds the configured threshold.",
        "",
        "| motion | category | frames | source | confidence | chosen_offset_m | max_penetration_m | double_stance_raw | status | reasons | review_frame |",
        "|---|---|---:|---|---|---:|---:|---:|---|---|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r.rel_path}` | {r.category} | {r.num_frames} | {r.chosen_offset_source} | "
            f"{r.chosen_offset_confidence} | {r.chosen_offset_m:.6f} | {r.max_penetration_after_m:.6f} | "
            f"{r.high_conf_double_stance_frame_ratio:.3f} | {r.status} | {r.reasons or '-'} | {r.review_frame} |"
        )
    review_rows = [r for r in results if r.status != "ok"]
    if review_rows:
        lines.extend(["", "## Review Visualization Commands", ""])
        for r in review_rows:
            lines.extend(
                [
                    f"### {r.rel_path}",
                    "",
                    "```bash",
                    r.view_command,
                    "```",
                    "",
                ]
            )
    (output_dir / "FOOT_HEIGHT_OFFSET_DRY_RUN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--all", action="store_true", help="Process every CSV under --csv-root instead of sampling.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--support-height-m", type=float, default=0.03)
    parser.add_argument("--double-support-height-m", type=float, default=0.03)
    parser.add_argument("--lr-balance-tolerance-m", type=float, default=0.015)
    parser.add_argument("--max-abs-offset-warn-m", type=float, default=0.15)
    parser.add_argument("--support-low-margin-m", type=float, default=0.02)
    parser.add_argument("--support-max-speed-m-s", type=float, default=0.20)
    parser.add_argument("--target-support-clearance-m", type=float, default=0.001)
    parser.add_argument("--high-conf-double-height-m", type=float, default=0.04)
    parser.add_argument("--high-conf-lr-gap-m", type=float, default=0.02)
    parser.add_argument("--review-penetration-m", type=float, default=0.020)
    parser.add_argument("--write-root", type=Path, default=None, help="If set, write adjusted CSVs here preserving relative paths.")
    parser.add_argument(
        "--write-filter",
        choices=("all", "ok", "ok_high_conf"),
        default="all",
        help="Filter adjusted CSVs written to --write-root while still reporting all checked motions.",
    )
    parser.add_argument(
        "--viewer-script",
        type=Path,
        default=Path("tools/view_mujoco_retarget_frame.py"),
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_root = args.csv_root.expanduser().resolve()
    xml = args.xml.expanduser().resolve()
    loadable_xml = make_loadable_mjcf(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(loadable_xml))
        data = mujoco.MjData(model)
        foot_ids = foot_collision_geom_ids(model)
        foot_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) for gid in foot_ids]
        print(f"[foot-height] foot geoms={len(foot_ids)} names={foot_names}")

        csvs = sample_csvs(csv_root, args.sample_size, args.seed, args.all)
        write_root = args.write_root.expanduser().resolve() if args.write_root is not None else None
        if write_root is not None:
            write_root.mkdir(parents=True, exist_ok=True)
        results_by_idx: dict[int, HeightDryRunResult] = {}
        if args.workers <= 1:
            for idx, csv_path in enumerate(csvs, start=1):
                print(f"[foot-height] {idx}/{len(csvs)} {csv_path.relative_to(csv_root)}", flush=True)
                results_by_idx[idx] = analyze_csv(
                    csv_path=csv_path,
                    csv_root=csv_root,
                    model=model,
                    data=data,
                    foot_ids=foot_ids,
                    support_height_m=args.support_height_m,
                    double_support_height_m=args.double_support_height_m,
                    lr_balance_tolerance_m=args.lr_balance_tolerance_m,
                    max_abs_offset_warn_m=args.max_abs_offset_warn_m,
                    support_low_margin_m=args.support_low_margin_m,
                    support_max_speed_m_s=args.support_max_speed_m_s,
                    target_support_clearance_m=args.target_support_clearance_m,
                    high_conf_double_height_m=args.high_conf_double_height_m,
                    high_conf_lr_gap_m=args.high_conf_lr_gap_m,
                    review_penetration_m=args.review_penetration_m,
                    viewer_script=args.viewer_script,
                    write_root=write_root,
                    write_filter=args.write_filter,
                )
        else:
            jobs = [
                {
                    "idx": idx,
                    "csv_path": str(csv_path),
                    "csv_root": str(csv_root),
                    "support_height_m": args.support_height_m,
                    "double_support_height_m": args.double_support_height_m,
                    "lr_balance_tolerance_m": args.lr_balance_tolerance_m,
                    "max_abs_offset_warn_m": args.max_abs_offset_warn_m,
                    "support_low_margin_m": args.support_low_margin_m,
                    "support_max_speed_m_s": args.support_max_speed_m_s,
                    "target_support_clearance_m": args.target_support_clearance_m,
                    "high_conf_double_height_m": args.high_conf_double_height_m,
                    "high_conf_lr_gap_m": args.high_conf_lr_gap_m,
                    "review_penetration_m": args.review_penetration_m,
                    "viewer_script": str(args.viewer_script),
                    "write_root": str(write_root) if write_root is not None else "",
                    "write_filter": args.write_filter,
                }
                for idx, csv_path in enumerate(csvs, start=1)
            ]
            print(f"[foot-height] workers={args.workers} jobs={len(jobs)}", flush=True)
            ctx = get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=ctx,
                initializer=init_worker,
                initargs=(str(loadable_xml),),
            ) as executor:
                future_to_job = {executor.submit(analyze_csv_job, job): job for job in jobs}
                for completed, future in enumerate(as_completed(future_to_job), start=1):
                    job = future_to_job[future]
                    idx, result = future.result()
                    results_by_idx[idx] = result
                    print(
                        f"[foot-height] done {completed}/{len(jobs)} src_idx={idx} "
                        f"{result.status} {result.chosen_offset_confidence} {Path(job['csv_path']).relative_to(csv_root)}",
                        flush=True,
                    )
        results = [results_by_idx[idx] for idx in sorted(results_by_idx)]
        write_outputs(args.output_dir, results, xml, csv_root)
        print(f"[foot-height] wrote {args.output_dir}")
    finally:
        try:
            loadable_xml.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
