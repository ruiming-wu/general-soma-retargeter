#!/usr/bin/env python3
"""Dry-run stance-aware root-z offsets from H4 foot collision geometry.

This variant estimates the height offset from high-confidence double-support
frames instead of clamping the whole motion's global lowest foot point to zero.
It reports if the resulting stance-based offset would still cause non-stance
penetration elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from soma_retargeter.diagnostics.gqs_physics import (
    build_model_qpos,
    foot_collision_geom_ids,
    load_robot_csv_motion,
)


DEFAULT_XML = Path("/home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml")
DEFAULT_VIEWER_SCRIPT = Path("tools/view_mujoco_retarget_frame.py")


@dataclass
class StanceOffsetResult:
    rel_path: str
    category: str
    num_frames: int
    num_foot_geoms: int
    best_stance_frame: int
    best_stance_time_s: float
    best_stance_confidence: float
    review_frame: int
    review_frame_time_s: float
    strict_min_offset_m: float
    stance_weighted_mean_offset_m: float
    stance_weighted_p95_offset_m: float
    recommended_offset_m: float
    recommended_offset_cm_csv_root_z: float
    stance_candidate_count: int
    stance_candidate_ratio: float
    stance_after_pair_mean_p50_m: float
    stance_after_pair_mean_p95_abs_m: float
    best_stance_left_bottom_after_m: float
    best_stance_right_bottom_after_m: float
    best_stance_left_thickness_m: float
    best_stance_right_thickness_m: float
    best_stance_lr_gap_m: float
    best_stance_max_speed_m_s: float
    global_min_after_m: float
    global_min_after_frame: int
    global_min_left_after_m: float
    global_min_right_after_m: float
    status: str
    reasons: str
    best_stance_view_command: str
    review_view_command: str


def make_loadable_mjcf(base_xml: Path) -> Path:
    text = base_xml.expanduser().resolve().read_text(encoding="utf-8")
    meshdir = (base_xml.parent / "../meshes/visual").resolve()
    text = text.replace('meshdir="../meshes/visual"', f'meshdir="{meshdir}"')
    tmp = tempfile.NamedTemporaryFile(prefix=f"{base_xml.stem}_stance_height_", suffix=".xml", delete=False)
    out = Path(tmp.name)
    tmp.close()
    out.write_text(text, encoding="utf-8")
    return out


def category_from_rel(rel_path: Path) -> str:
    parts = rel_path.parts
    if len(parts) >= 2 and parts[0] == "seed":
        return parts[1]
    return parts[0] if parts else "unknown"


def foot_side(name: str) -> str | None:
    if name.startswith("left_foot"):
        return "left"
    if name.startswith("right_foot"):
        return "right"
    return None


def geom_bottom_top_z(model: Any, data: Any, geom_id: int) -> tuple[float, float]:
    geom_type = int(model.geom_type[geom_id])
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)

    if geom_type in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        radius = float(size[0])
        half_length = float(size[1])
        axis = mat[:, 2]
        half_z_extent = abs(float(axis[2])) * half_length + radius
        return float(pos[2] - half_z_extent), float(pos[2] + half_z_extent)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        radius = float(size[0])
        return float(pos[2] - radius), float(pos[2] + radius)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        half_z_extent = float(np.sum(np.abs(mat[2, :]) * size[:3]))
        return float(pos[2] - half_z_extent), float(pos[2] + half_z_extent)
    return float(pos[2]), float(pos[2])


def finite_speed(positions: np.ndarray, fps: float) -> np.ndarray:
    speed = np.zeros(positions.shape[0], dtype=np.float64)
    if positions.shape[0] > 1:
        diff = np.linalg.norm(positions[1:] - positions[:-1], axis=1) * float(fps)
        speed[1:] = diff
        speed[0] = diff[0]
    return speed


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    total = float(cumulative[-1])
    if total <= 1e-12:
        return float(np.quantile(values, q))
    return float(sorted_values[np.searchsorted(cumulative, q * total, side="left")])


def view_command(viewer_script: Path, csv_path: Path, frame: int, offset_m: float) -> str:
    return (
        "cd /home/ruiming.wu/codes/general-soma-retargeter && "
        f"uv run python {viewer_script} "
        "--xml /home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml "
        f"--csv {csv_path} --frame {frame} "
        f"--root-z-offset-cm {offset_m * 100.0:.6f} --show-visual"
    )


def analyze_csv(
    csv_path: Path,
    csv_root: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_ids: list[int],
    target_clearance_m: float,
    foot_flat_excess_scale_m: float,
    lr_gap_scale_m: float,
    speed_scale_m_s: float,
    min_confidence: float,
    min_candidate_ratio: float,
    penetration_review_m: float,
    stance_abs_p95_review_m: float,
    viewer_script: Path,
) -> StanceOffsetResult:
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
        raise RuntimeError("Expected both left and right foot collision geoms")

    radii = [float(model.geom_size[gid, 0]) for gid in foot_ids]
    nominal_diameter = 2.0 * float(np.median(radii))

    n = qpos.shape[0]
    left_bottom = np.empty(n, dtype=np.float64)
    right_bottom = np.empty(n, dtype=np.float64)
    left_thickness = np.empty(n, dtype=np.float64)
    right_thickness = np.empty(n, dtype=np.float64)
    left_center = np.empty((n, 3), dtype=np.float64)
    right_center = np.empty((n, 3), dtype=np.float64)

    for frame in range(n):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        left_bottoms, left_tops = zip(*(geom_bottom_top_z(model, data, gid) for gid in left_ids), strict=True)
        right_bottoms, right_tops = zip(*(geom_bottom_top_z(model, data, gid) for gid in right_ids), strict=True)
        left_bottom[frame] = min(left_bottoms)
        right_bottom[frame] = min(right_bottoms)
        left_thickness[frame] = max(left_tops) - min(left_bottoms)
        right_thickness[frame] = max(right_tops) - min(right_bottoms)
        left_center[frame] = np.mean([np.asarray(data.geom_xpos[gid], dtype=np.float64) for gid in left_ids], axis=0)
        right_center[frame] = np.mean([np.asarray(data.geom_xpos[gid], dtype=np.float64) for gid in right_ids], axis=0)

    left_speed = finite_speed(left_center, motion.fps)
    right_speed = finite_speed(right_center, motion.fps)
    max_speed = np.maximum(left_speed, right_speed)
    lr_gap = np.abs(left_bottom - right_bottom)
    pair_mean_bottom = 0.5 * (left_bottom + right_bottom)

    left_flat_excess = np.maximum(0.0, left_thickness - nominal_diameter)
    right_flat_excess = np.maximum(0.0, right_thickness - nominal_diameter)
    flat_score = np.exp(-(left_flat_excess + right_flat_excess) / foot_flat_excess_scale_m)
    gap_score = np.exp(-((lr_gap / lr_gap_scale_m) ** 2))
    speed_score = np.exp(-((max_speed / speed_scale_m_s) ** 2))
    confidence = flat_score * gap_score * speed_score

    candidate_mask = confidence >= min_confidence
    if not np.any(candidate_mask):
        # Keep the best few frames so the report still explains why confidence failed.
        best_count = max(1, min(10, n // 50))
        best_indices = np.argsort(confidence)[-best_count:]
        candidate_mask[best_indices] = True

    candidate_offsets = target_clearance_m - pair_mean_bottom[candidate_mask]
    candidate_weights = np.maximum(confidence[candidate_mask], 1e-6) ** 2
    weighted_mean_offset = float(np.average(candidate_offsets, weights=candidate_weights))
    weighted_p95_offset = weighted_quantile(candidate_offsets, candidate_weights, 0.95)
    recommended_offset = weighted_p95_offset

    after_left = left_bottom + recommended_offset
    after_right = right_bottom + recommended_offset
    after_min = np.minimum(after_left, after_right)
    after_pair_mean = pair_mean_bottom + recommended_offset
    best_stance_frame = int(np.argmax(confidence))
    global_min_frame = int(np.argmin(after_min))

    reasons: list[str] = []
    candidate_count = int(np.count_nonzero(candidate_mask))
    candidate_ratio = candidate_count / float(n)
    if candidate_ratio < min_candidate_ratio:
        reasons.append("few_high_confidence_double_support_candidates")
    if float(np.min(after_min)) < -penetration_review_m:
        reasons.append(f"nonstance_global_penetration>{penetration_review_m:.3f}m")
    stance_abs_p95 = weighted_quantile(np.abs(after_pair_mean[candidate_mask]), candidate_weights, 0.95)
    if stance_abs_p95 > stance_abs_p95_review_m:
        reasons.append(f"stance_pair_mean_abs_p95>{stance_abs_p95_review_m:.3f}m")
    if float(confidence[best_stance_frame]) < min_confidence:
        reasons.append("no_candidate_above_confidence_threshold")

    status = "ok" if not reasons else "review"
    review_frame = global_min_frame if any(reason.startswith("nonstance_global_penetration") for reason in reasons) else best_stance_frame

    return StanceOffsetResult(
        rel_path=str(rel),
        category=category_from_rel(rel),
        num_frames=int(n),
        num_foot_geoms=len(foot_ids),
        best_stance_frame=best_stance_frame,
        best_stance_time_s=float(best_stance_frame / motion.fps),
        best_stance_confidence=float(confidence[best_stance_frame]),
        review_frame=review_frame,
        review_frame_time_s=float(review_frame / motion.fps),
        strict_min_offset_m=float(-np.min(np.minimum(left_bottom, right_bottom))),
        stance_weighted_mean_offset_m=weighted_mean_offset,
        stance_weighted_p95_offset_m=float(weighted_p95_offset),
        recommended_offset_m=float(recommended_offset),
        recommended_offset_cm_csv_root_z=float(recommended_offset * 100.0),
        stance_candidate_count=candidate_count,
        stance_candidate_ratio=float(candidate_ratio),
        stance_after_pair_mean_p50_m=float(np.median(after_pair_mean[candidate_mask])),
        stance_after_pair_mean_p95_abs_m=float(stance_abs_p95),
        best_stance_left_bottom_after_m=float(after_left[best_stance_frame]),
        best_stance_right_bottom_after_m=float(after_right[best_stance_frame]),
        best_stance_left_thickness_m=float(left_thickness[best_stance_frame]),
        best_stance_right_thickness_m=float(right_thickness[best_stance_frame]),
        best_stance_lr_gap_m=float(lr_gap[best_stance_frame]),
        best_stance_max_speed_m_s=float(max_speed[best_stance_frame]),
        global_min_after_m=float(np.min(after_min)),
        global_min_after_frame=global_min_frame,
        global_min_left_after_m=float(after_left[global_min_frame]),
        global_min_right_after_m=float(after_right[global_min_frame]),
        status=status,
        reasons=";".join(reasons),
        best_stance_view_command=view_command(viewer_script, csv_path, best_stance_frame, recommended_offset),
        review_view_command=view_command(viewer_script, csv_path, review_frame, recommended_offset),
    )


def sample_csvs(csv_root: Path, sample_size: int, seed: int, explicit_csv: Path | None) -> list[Path]:
    if explicit_csv is not None:
        return [explicit_csv.expanduser().resolve()]
    all_csvs = sorted(csv_root.rglob("*.csv"))
    if sample_size >= len(all_csvs):
        return all_csvs
    rng = random.Random(seed)
    return sorted(rng.sample(all_csvs, sample_size))


def write_outputs(output_dir: Path, rows: list[StanceOffsetResult], xml: Path, csv_root: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [row.__dict__ for row in rows]
    with (output_dir / "stance_aware_foot_height_offset_dry_run.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(payloads[0].keys()))
        writer.writeheader()
        writer.writerows(payloads)

    lines = [
        "# Stance-Aware Foot Height Offset Dry Run",
        "",
        f"- XML: `{xml}`",
        f"- CSV root: `{csv_root}`",
        "- Primary recommendation: add `recommended_offset_cm_csv_root_z` to CSV `root_translateZ`.",
        "- Stance confidence uses both feet flatness, low foot speed, and left/right foot-bottom height agreement.",
        "- `strict_min_offset_m` is still reported for comparison, but not used as the primary offset.",
        "",
        "| motion | best_frame | best_t | conf | strict_m | rec_m | stance_abs_p95_m | global_min_after_m | status | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.rel_path}` | {row.best_stance_frame} | {row.best_stance_time_s:.3f} | "
            f"{row.best_stance_confidence:.3f} | {row.strict_min_offset_m:.6f} | "
            f"{row.recommended_offset_m:.6f} | {row.stance_after_pair_mean_p95_abs_m:.6f} | "
            f"{row.global_min_after_m:.6f} | {row.status} | {row.reasons or '-'} |"
        )
    lines.extend(["", "## Best Stance View Commands", ""])
    for row in rows:
        lines.extend(["### " + row.rel_path, "", "```bash", row.best_stance_view_command, "```", ""])
    review_rows = [row for row in rows if row.status != "ok"]
    if review_rows:
        lines.extend(["", "## Review View Commands", ""])
        for row in review_rows:
            lines.extend(["### " + row.rel_path, "", "```bash", row.review_view_command, "```", ""])
    (output_dir / "STANCE_AWARE_FOOT_HEIGHT_OFFSET_DRY_RUN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None, help="Analyze one explicit CSV instead of sampling.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-clearance-m", type=float, default=0.001)
    parser.add_argument("--foot-flat-excess-scale-m", type=float, default=0.012)
    parser.add_argument("--lr-gap-scale-m", type=float, default=0.012)
    parser.add_argument("--speed-scale-m-s", type=float, default=0.20)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--min-candidate-ratio", type=float, default=0.005)
    parser.add_argument("--penetration-review-m", type=float, default=0.010)
    parser.add_argument("--stance-abs-p95-review-m", type=float, default=0.003)
    parser.add_argument("--viewer-script", type=Path, default=DEFAULT_VIEWER_SCRIPT)
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
        print(
            "[stance-height] foot geoms=",
            [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) for gid in foot_ids],
        )
        csvs = sample_csvs(csv_root, args.sample_size, args.seed, args.csv)
        rows: list[StanceOffsetResult] = []
        for idx, path in enumerate(csvs, start=1):
            print(f"[stance-height] {idx}/{len(csvs)} {path.relative_to(csv_root) if path.is_relative_to(csv_root) else path}")
            rows.append(
                analyze_csv(
                    csv_path=path,
                    csv_root=csv_root,
                    model=model,
                    data=data,
                    foot_ids=foot_ids,
                    target_clearance_m=args.target_clearance_m,
                    foot_flat_excess_scale_m=args.foot_flat_excess_scale_m,
                    lr_gap_scale_m=args.lr_gap_scale_m,
                    speed_scale_m_s=args.speed_scale_m_s,
                    min_confidence=args.min_confidence,
                    min_candidate_ratio=args.min_candidate_ratio,
                    penetration_review_m=args.penetration_review_m,
                    stance_abs_p95_review_m=args.stance_abs_p95_review_m,
                    viewer_script=args.viewer_script,
                )
            )
        write_outputs(args.output_dir, rows, xml, csv_root)
        print(f"[stance-height] wrote {args.output_dir}")
    finally:
        try:
            loadable_xml.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
