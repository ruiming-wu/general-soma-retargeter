#!/usr/bin/env python3
"""Summarize key-frame retargeting offsets between final robot CSVs and SOMA BVH targets."""

from __future__ import annotations

import argparse
import csv
import io
import math
import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

import newton
import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.csv as csv_utils
import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
from soma_retargeter.utils import newton_utils
from soma_retargeter.utils.space_conversion_utils import (
    SpaceConverter,
    get_facing_direction_type_from_str,
)


KEY_TARGETS = {
    "pelvis_root": "Hips",
    "head": "Head",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
}

POSITION_BINS_M = np.linspace(0.0, 2.0, 4001, dtype=np.float64)
ROTATION_BINS_DEG = np.linspace(0.0, 180.0, 3601, dtype=np.float64)


@contextmanager
def quiet_output(enabled: bool = True):
    if not enabled:
        yield
        return
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield


@contextmanager
def quiet_newton_progress():
    original_trange = newton_pipeline.trange

    def disabled_trange(*args, **kwargs):
        kwargs["disable"] = True
        return original_trange(*args, **kwargs)

    newton_pipeline.trange = disabled_trange
    try:
        yield
    finally:
        newton_pipeline.trange = original_trange


def body_names(model) -> list[str]:
    return [newton_utils.get_name_from_label(label) for label in model.body_label]


def quat_error_deg_xyzw(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> float:
    a = Rotation.from_quat(a_xyzw)
    b = Rotation.from_quat(b_xyzw)
    return float(np.degrees((b.inv() * a).magnitude()))


def csv_to_bvh_path(csv_path: Path, csv_root: Path, soma_root: Path) -> Path:
    rel = csv_path.relative_to(csv_root)
    parts = list(rel.parts)
    if "body_motion" in parts:
        parts.remove("body_motion")
    return (soma_root / Path(*parts)).with_suffix(".bvh")


def category_from_rel(rel_path: Path) -> str:
    parts = rel_path.parts
    if len(parts) >= 2 and parts[0] == "seed":
        return parts[1]
    if parts:
        return parts[0]
    return "unknown"


@dataclass
class MetricAccum:
    count: int
    total: float
    min_value: float
    max_value: float
    hist: np.ndarray

    @classmethod
    def empty(cls, bins: np.ndarray) -> "MetricAccum":
        return cls(
            count=0,
            total=0.0,
            min_value=float("inf"),
            max_value=float("-inf"),
            hist=np.zeros(len(bins) - 1, dtype=np.int64),
        )

    def update_values(self, values: np.ndarray, bins: np.ndarray) -> None:
        if values.size == 0:
            return
        values = np.asarray(values, dtype=np.float64)
        self.count += int(values.size)
        self.total += float(values.sum())
        self.min_value = min(self.min_value, float(values.min()))
        self.max_value = max(self.max_value, float(values.max()))
        clipped = np.clip(values, bins[0], bins[-1])
        self.hist += np.histogram(clipped, bins=bins)[0]

    def merge(self, other: "MetricAccum") -> None:
        self.count += other.count
        self.total += other.total
        self.min_value = min(self.min_value, other.min_value)
        self.max_value = max(self.max_value, other.max_value)
        self.hist += other.hist

    def quantile(self, q: float, bins: np.ndarray) -> float:
        if self.count == 0:
            return float("nan")
        target = q * (self.count - 1)
        cumulative = np.cumsum(self.hist)
        idx = int(np.searchsorted(cumulative, target, side="right"))
        idx = min(max(idx, 0), len(bins) - 2)
        return float((bins[idx] + bins[idx + 1]) * 0.5)

    def summary(self, bins: np.ndarray) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "p01": float("nan"),
                "p99": float("nan"),
                "p99_range": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        p01 = self.quantile(0.01, bins)
        p99 = self.quantile(0.99, bins)
        return {
            "count": self.count,
            "mean": self.total / self.count,
            "median": self.quantile(0.50, bins),
            "p01": p01,
            "p99": p99,
            "p99_range": p99 - p01,
            "min": self.min_value,
            "max": self.max_value,
        }


def empty_scope_accum() -> dict[tuple[str, str], MetricAccum]:
    acc: dict[tuple[str, str], MetricAccum] = {}
    for key in KEY_TARGETS:
        acc[("position", key)] = MetricAccum.empty(POSITION_BINS_M)
        acc[("rotation", key)] = MetricAccum.empty(ROTATION_BINS_DEG)
    return acc


def merge_scope_accum(
    target: dict[tuple[str, str], MetricAccum],
    source: dict[tuple[str, str], MetricAccum],
) -> None:
    for metric_key, source_acc in source.items():
        if metric_key not in target:
            bins = POSITION_BINS_M if metric_key[0] == "position" else ROTATION_BINS_DEG
            target[metric_key] = MetricAccum.empty(bins)
        target[metric_key].merge(source_acc)


def process_motion(
    csv_path_str: str,
    csv_root_str: str,
    soma_root_str: str,
    robot: str,
    facing_direction: str,
    max_frames: int | None,
) -> dict[str, Any]:
    csv_path = Path(csv_path_str)
    csv_root = Path(csv_root_str)
    soma_root = Path(soma_root_str)
    rel_path = csv_path.relative_to(csv_root)
    bvh_path = csv_to_bvh_path(csv_path, csv_root, soma_root)
    if not bvh_path.exists():
        raise FileNotFoundError(f"Missing SOMA BVH for {rel_path}: {bvh_path}")

    with quiet_output(), quiet_newton_progress():
        skeleton, animation = bvh_utils.load_bvh(str(bvh_path))
        if max_frames is not None and animation.num_frames > max_frames:
            animation.local_transforms = np.copy(animation.local_transforms[:max_frames])
            animation.num_frames = max_frames

        converter = SpaceConverter(get_facing_direction_type_from_str(facing_direction))
        source_transform = converter.transform(wp.transform_identity())
        pipeline = newton_pipeline.NewtonPipeline(skeleton, "soma", robot)
        pipeline.add_input_motions([animation], [source_transform], True)

        buffer = csv_utils.load_csv(
            str(csv_path),
            fps=animation.sample_rate,
            csv_config=csv_utils.get_csv_config(robot),
            robot_name=robot,
        )

        targets = pipeline.input_targets[0]
        removed = pipeline.num_initialization_frames + pipeline.num_stabilization_frames
        usable_frames = min(buffer.num_frames, max(0, len(targets) - removed))
        if max_frames is not None:
            usable_frames = min(usable_frames, max_frames)
        if usable_frames <= 0:
            raise RuntimeError(f"No aligned frames for {rel_path}")

        eval_model = pipeline._build_model(1)
        pipeline._apply_initial_robot_joint_positions(eval_model, 1)
        eval_state = eval_model.state()
        names = body_names(eval_model)

        mapped_indices = {}
        for key, target_joint in KEY_TARGETS.items():
            if target_joint in pipeline.mapped_joints:
                mapped_indices[key] = pipeline.mapped_joints.index(target_joint)

        values: dict[tuple[str, str], list[float]] = defaultdict(list)
        for frame_idx in range(usable_frames):
            q = np.asarray(buffer.get_data(frame_idx), dtype=np.float32)
            wp.copy(eval_model.joint_q, wp.array(q, dtype=wp.float32), 0, 0, len(q))
            newton.eval_fk(eval_model, eval_model.joint_q, eval_model.joint_qd, eval_state, None)
            robot_body_q = eval_state.body_q.numpy()
            frame_targets = targets[removed + frame_idx]

            for key, map_idx in mapped_indices.items():
                target = np.asarray(frame_targets[map_idx], dtype=np.float64)
                pos_body_idx, pos_weight = pipeline.mapped_body_link_pos_data[map_idx]
                rot_body_idx, rot_weight = pipeline.mapped_body_link_rot_data[map_idx]

                if float(pos_weight) > 0.0:
                    pos_body = robot_body_q[pos_body_idx]
                    values[("position", key)].append(float(np.linalg.norm(pos_body[:3] - target[:3])))
                if float(rot_weight) > 0.0:
                    rot_body = robot_body_q[rot_body_idx]
                    values[("rotation", key)].append(quat_error_deg_xyzw(rot_body[3:7], target[3:7]))

        accum = empty_scope_accum()
        per_motion: dict[str, Any] = {
            "rel_path": str(rel_path),
            "category": category_from_rel(rel_path),
            "frames": usable_frames,
            "bvh_path": str(bvh_path),
            "csv_path": str(csv_path),
        }
        for metric_key, metric_values in values.items():
            metric_type, keyframe = metric_key
            bins = POSITION_BINS_M if metric_type == "position" else ROTATION_BINS_DEG
            arr = np.asarray(metric_values, dtype=np.float64)
            accum[metric_key].update_values(arr, bins)
            if arr.size:
                prefix = f"{metric_type}_{keyframe}"
                per_motion[f"{prefix}_mean"] = float(arr.mean())
                per_motion[f"{prefix}_median"] = float(np.median(arr))
                per_motion[f"{prefix}_p99"] = float(np.quantile(arr, 0.99))
                per_motion[f"{prefix}_min"] = float(arr.min())
                per_motion[f"{prefix}_max"] = float(arr.max())

        return {
            "rel_path": str(rel_path),
            "category": category_from_rel(rel_path),
            "accum": accum,
            "per_motion": per_motion,
            "body_names": names,
        }


def process_chunk(
    csv_paths: list[str],
    csv_root: str,
    soma_root: str,
    robot: str,
    facing_direction: str,
    max_frames: int | None,
) -> dict[str, Any]:
    global_accum = empty_scope_accum()
    category_accum: dict[str, dict[tuple[str, str], MetricAccum]] = {}
    per_motion_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    body_name_sample: list[str] | None = None

    for csv_path in csv_paths:
        try:
            result = process_motion(csv_path, csv_root, soma_root, robot, facing_direction, max_frames)
        except Exception as exc:  # noqa: BLE001 - keep full run progressing and report failures.
            errors.append(f"{csv_path}: {type(exc).__name__}: {exc}")
            continue

        merge_scope_accum(global_accum, result["accum"])
        category = result["category"]
        if category not in category_accum:
            category_accum[category] = empty_scope_accum()
        merge_scope_accum(category_accum[category], result["accum"])
        per_motion_rows.append(result["per_motion"])
        if body_name_sample is None:
            body_name_sample = result["body_names"]

    return {
        "global_accum": global_accum,
        "category_accum": category_accum,
        "per_motion_rows": per_motion_rows,
        "errors": errors,
        "body_name_sample": body_name_sample,
    }


def chunked(items: list[str], chunks: int) -> list[list[str]]:
    if chunks <= 1:
        return [items]
    chunk_size = max(1, math.ceil(len(items) / chunks))
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def write_summary_csv(
    path: Path,
    global_accum: dict[tuple[str, str], MetricAccum],
    category_accum: dict[str, dict[tuple[str, str], MetricAccum]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_scope(scope: str, accum: dict[tuple[str, str], MetricAccum]) -> None:
        for metric_type in ("position", "rotation"):
            bins = POSITION_BINS_M if metric_type == "position" else ROTATION_BINS_DEG
            unit = "m" if metric_type == "position" else "deg"
            for keyframe in KEY_TARGETS:
                stats = accum[(metric_type, keyframe)].summary(bins)
                rows.append(
                    {
                        "scope": scope,
                        "metric_type": metric_type,
                        "keyframe": keyframe,
                        "unit": unit,
                        **stats,
                    }
                )

    add_scope("all", global_accum)
    for category in sorted(category_accum):
        add_scope(category, category_accum[category])

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "metric_type",
        "keyframe",
        "unit",
        "count",
        "mean",
        "median",
        "p01",
        "p99",
        "p99_range",
        "min",
        "max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_per_motion_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["rel_path", "category", "frames", "csv_path", "bvh_path"]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], total_files: int, errors: list[str]) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return "nan"
            return f"{value:.6f}"
        return str(value)

    global_position = [
        row
        for row in rows
        if row["scope"] == "all" and row["metric_type"] == "position"
    ]
    global_rotation = [
        row
        for row in rows
        if row["scope"] == "all" and row["metric_type"] == "rotation"
    ]

    def table(section_rows: list[dict[str, Any]]) -> str:
        lines = [
            "| keyframe | count | mean | median | p01 | p99 | p99_range | min | max | unit |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in section_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["keyframe"]),
                        fmt(row["count"]),
                        fmt(row["mean"]),
                        fmt(row["median"]),
                        fmt(row["p01"]),
                        fmt(row["p99"]),
                        fmt(row["p99_range"]),
                        fmt(row["min"]),
                        fmt(row["max"]),
                        str(row["unit"]),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    content = [
        "# Keyframe Retarget Offset Summary",
        "",
        f"- Files requested: {total_files}",
        f"- Files failed: {len(errors)}",
        "- Quantiles are computed from fixed histograms: position bin = 0.5 mm, rotation bin = 0.05 deg.",
        "- `p99_range` is `p99 - p01`, intended as a robust min-max range.",
        "",
        "## Global Position Offset",
        "",
        table(global_position),
        "",
        "## Global Rotation Offset",
        "",
        table(global_rotation),
        "",
        "## Outputs",
        "",
        "- `keyframe_offset_summary.csv`: global and per-category summary.",
        "- `per_motion_keyframe_offsets.csv`: per-motion keyframe statistics.",
        "- `errors.txt`: files that could not be evaluated.",
    ]
    if errors:
        content.extend(["", "## First Errors", ""])
        content.extend(f"- {err}" for err in errors[:20])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--soma-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot", choices=("agile_one", "unitree_g1"), default="agile_one")
    parser.add_argument("--retarget-source-facing-direction", choices=("Mujoco", "Maya"), default="Mujoco")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1) // 2)))
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = sorted(str(path) for path in args.csv_root.rglob("*.csv"))
    if args.limit is not None:
        csv_paths = csv_paths[: args.limit]
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {args.csv_root}")

    chunks = [csv_paths[i : i + args.chunk_size] for i in range(0, len(csv_paths), args.chunk_size)]
    workers = max(1, min(args.workers, len(chunks)))
    print(f"[offsets] csvs={len(csv_paths)} chunks={len(chunks)} workers={workers}")

    global_accum = empty_scope_accum()
    category_accum: dict[str, dict[tuple[str, str], MetricAccum]] = {}
    per_motion_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    completed = 0

    if workers == 1:
        futures_iter = [
            process_chunk(
                chunk,
                str(args.csv_root),
                str(args.soma_root),
                args.robot,
                args.retarget_source_facing_direction,
                args.max_frames,
            )
            for chunk in chunks
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
            futures = [
                executor.submit(
                    process_chunk,
                    chunk,
                    str(args.csv_root),
                    str(args.soma_root),
                    args.robot,
                    args.retarget_source_facing_direction,
                    args.max_frames,
                )
                for chunk in chunks
            ]
            futures_iter = []
            for future in as_completed(futures):
                futures_iter.append(future.result())
                completed += 1
                processed = min(len(csv_paths), completed * args.chunk_size)
                print(f"[offsets] completed chunks {completed}/{len(chunks)} approx_csv={processed}/{len(csv_paths)}")

    if workers == 1:
        completed = 0

    for result in futures_iter:
        merge_scope_accum(global_accum, result["global_accum"])
        for category, accum in result["category_accum"].items():
            if category not in category_accum:
                category_accum[category] = empty_scope_accum()
            merge_scope_accum(category_accum[category], accum)
        per_motion_rows.extend(result["per_motion_rows"])
        errors.extend(result["errors"])
        if workers == 1:
            completed += 1
            processed = min(len(csv_paths), completed * args.chunk_size)
            print(f"[offsets] completed chunks {completed}/{len(chunks)} approx_csv={processed}/{len(csv_paths)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = write_summary_csv(args.output_dir / "keyframe_offset_summary.csv", global_accum, category_accum)
    write_per_motion_csv(args.output_dir / "per_motion_keyframe_offsets.csv", per_motion_rows)
    (args.output_dir / "errors.txt").write_text("\n".join(errors) + ("\n" if errors else ""), encoding="utf-8")
    write_markdown(args.output_dir / "KEYFRAME_OFFSET_SUMMARY.md", rows, len(csv_paths), errors)
    print(f"[offsets] wrote {args.output_dir}")
    print(f"[offsets] evaluated={len(per_motion_rows)} failed={len(errors)}")


if __name__ == "__main__":
    main()
