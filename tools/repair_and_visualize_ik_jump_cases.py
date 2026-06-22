#!/usr/bin/env python3
"""Repair selected single-joint IK jump cases and generate before/after plots/videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--repair-radius-frames", type=int, default=12)
    parser.add_argument("--max-repair-radius-frames", type=int, default=60)
    parser.add_argument("--radius-step-frames", type=int, default=6)
    parser.add_argument("--spike-anchor-threshold-deg", type=float, default=1.5)
    parser.add_argument("--disable-adaptive-spike-window", action="store_true")
    parser.add_argument("--repair-mode", choices=("linear", "cubic", "quintic"), default="quintic")
    parser.add_argument("--endpoint-fit-frames", type=int, default=6)
    parser.add_argument("--plot-window-sec", type=float, default=2.0)
    parser.add_argument("--render-window-sec", type=float, default=2.0)
    parser.add_argument("--robot", default="agile_one")
    parser.add_argument("--mjcf", type=Path, default=Path("/home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--python-command", default="uv run python")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fieldnames:
        raise ValueError(f"{path} has no header")
    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def wrapped_diff_deg(curr: np.ndarray | float, prev: np.ndarray | float) -> np.ndarray | float:
    return (curr - prev + 180.0) % 360.0 - 180.0


def unwrap_local(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for idx in range(1, values.shape[0]):
        out[idx] = out[idx - 1] + float(wrapped_diff_deg(values[idx], values[idx - 1]))
    return out


def unwrap_global(values: np.ndarray) -> np.ndarray:
    return unwrap_local(values)


def finite_dq(values: np.ndarray) -> np.ndarray:
    dq = np.zeros_like(values, dtype=np.float64)
    if values.shape[0] > 1:
        dq[1:] = wrapped_diff_deg(values[1:], values[:-1])
    return dq


def _local_polyfit_derivatives(unwrapped: np.ndarray, idx: int, fit_radius: int, side: str) -> tuple[float, float]:
    if side == "left":
        left = max(0, idx - fit_radius)
        right = idx
    elif side == "right":
        left = idx
        right = min(unwrapped.shape[0] - 1, idx + fit_radius)
    else:
        left = max(0, idx - fit_radius)
        right = min(unwrapped.shape[0] - 1, idx + fit_radius)
    xs = np.arange(left, right + 1, dtype=np.float64) - float(idx)
    ys = unwrapped[left : right + 1].astype(np.float64)
    if xs.shape[0] >= 5:
        coeff = np.polyfit(xs, ys, deg=3)
        d1 = float(coeff[-2])
        d2 = float(2.0 * coeff[-3])
        return d1, d2
    if xs.shape[0] >= 3:
        coeff = np.polyfit(xs, ys, deg=2)
        d1 = float(coeff[-2])
        d2 = float(2.0 * coeff[-3])
        return d1, d2
    if idx <= 0:
        d1 = float(unwrapped[min(1, unwrapped.shape[0] - 1)] - unwrapped[0])
    elif idx >= unwrapped.shape[0] - 1:
        d1 = float(unwrapped[-1] - unwrapped[-2])
    else:
        d1 = float((unwrapped[idx + 1] - unwrapped[idx - 1]) * 0.5)
    return d1, 0.0


def _linear_segment(y0: float, y1: float, n: int) -> np.ndarray:
    return np.linspace(y0, y1, n + 1)


def _cubic_hermite_segment(y0: float, y1: float, dy0_frame: float, dy1_frame: float, n: int) -> np.ndarray:
    if n <= 0:
        return np.asarray([y0], dtype=np.float64)
    t = np.linspace(0.0, 1.0, n + 1)
    dy0 = dy0_frame * n
    dy1 = dy1_frame * n
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return h00 * y0 + h10 * dy0 + h01 * y1 + h11 * dy1


def _quintic_hermite_segment(
    y0: float,
    y1: float,
    dy0_frame: float,
    dy1_frame: float,
    ddy0_frame: float,
    ddy1_frame: float,
    n: int,
) -> np.ndarray:
    if n <= 0:
        return np.asarray([y0], dtype=np.float64)
    t = np.linspace(0.0, 1.0, n + 1)
    v0 = dy0_frame * n
    v1 = dy1_frame * n
    a0 = ddy0_frame * n * n
    a1 = ddy1_frame * n * n
    c0 = y0
    c1 = v0
    c2 = 0.5 * a0
    rhs = np.asarray(
        [
            y1 - (c0 + c1 + c2),
            v1 - (c1 + 2.0 * c2),
            a1 - (2.0 * c2),
        ],
        dtype=np.float64,
    )
    mat = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0],
            [6.0, 12.0, 20.0],
        ],
        dtype=np.float64,
    )
    c3, c4, c5 = np.linalg.solve(mat, rhs)
    return c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5


def _smooth_segment(
    unwrapped: np.ndarray,
    left: int,
    right: int,
    mode: str,
    endpoint_fit_frames: int,
) -> np.ndarray:
    n = right - left
    y0 = float(unwrapped[left])
    y1 = float(unwrapped[right])
    if mode == "linear":
        return _linear_segment(y0, y1, n)
    dy0, ddy0 = _local_polyfit_derivatives(unwrapped, left, endpoint_fit_frames, "left")
    dy1, ddy1 = _local_polyfit_derivatives(unwrapped, right, endpoint_fit_frames, "right")
    if mode == "cubic":
        return _cubic_hermite_segment(y0, y1, dy0, dy1, n)
    return _quintic_hermite_segment(y0, y1, dy0, dy1, ddy0, ddy1, n)


def _repair_for_radius(
    values: np.ndarray,
    frame0: int,
    radius: int,
    mode: str,
    endpoint_fit_frames: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    frame0 = int(np.clip(frame0, 0, values.shape[0] - 2))
    frame1 = frame0 + 1
    left = max(0, frame0 - radius)
    right = min(values.shape[0] - 1, frame1 + radius)
    unwrapped = unwrap_global(values)
    anchor_delta = float(abs(wrapped_diff_deg(values[right], values[left])))
    repaired_local = _smooth_segment(unwrapped, left, right, mode, endpoint_fit_frames)
    repaired = values.copy()
    repaired[left : right + 1] = repaired_local
    before_dq = float(np.max(np.abs(finite_dq(values[left : right + 1])[1:]))) if right > left else 0.0
    after_dq = float(np.max(np.abs(finite_dq(repaired[left : right + 1])[1:]))) if right > left else 0.0
    kind = "spike_flatten" if anchor_delta <= 1.5 else "transition_average"
    return repaired, {
        "event_frame": frame0,
        "event_frame1": frame1,
        "repair_left": left,
        "repair_right": right,
        "repair_radius_frames": radius,
        "anchor_delta_deg": anchor_delta,
        "max_abs_dq_before_deg": before_dq,
        "max_abs_dq_after_deg": after_dq,
        "repair_kind": kind,
        "repair_mode": mode,
    }


def repair_values(
    values: np.ndarray,
    event_frame: int,
    radius: int,
    max_radius: int,
    radius_step: int,
    spike_anchor_threshold_deg: float,
    adaptive_spike_window: bool,
    mode: str,
    endpoint_fit_frames: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    frame0 = int(np.clip(event_frame, 0, values.shape[0] - 2))
    base_after, base_metrics = _repair_for_radius(values, frame0, radius, mode, endpoint_fit_frames)
    base_metrics["repair_kind"] = (
        "spike_flatten" if float(base_metrics["anchor_delta_deg"]) <= spike_anchor_threshold_deg else "transition_average"
    )
    if not adaptive_spike_window:
        return base_after, base_metrics

    best: tuple[tuple[float, float, int], np.ndarray, dict[str, float | int | str]] | None = None
    step = max(1, int(radius_step))
    for cand_radius in range(max(1, radius), max(radius, max_radius) + 1, step):
        repaired, metrics = _repair_for_radius(values, frame0, cand_radius, mode, endpoint_fit_frames)
        anchor_delta = float(metrics["anchor_delta_deg"])
        if anchor_delta > spike_anchor_threshold_deg:
            continue
        # Prefer a window whose two anchors are closest in value. This catches
        # two-sided spikes where a fixed small window only removes one edge.
        score = (anchor_delta, float(metrics["max_abs_dq_after_deg"]), int(metrics["repair_radius_frames"]))
        metrics["repair_kind"] = "spike_flatten_adaptive"
        if best is None or score < best[0]:
            best = (score, repaired, metrics)

    if best is None:
        return base_after, base_metrics
    return best[1], best[2]


def plot_trace(
    out_png: Path,
    values_before: np.ndarray,
    values_after: np.ndarray,
    joint: str,
    event_frame: int,
    repair_left: int,
    repair_right: int,
    fps: float,
    window_sec: float,
) -> None:
    half = max(1, int(round(window_sec * fps * 0.5)))
    start = max(0, event_frame - half)
    end = min(values_before.shape[0], event_frame + half)
    frames = np.arange(start, end)
    t = (frames - event_frame) / fps
    before = values_before[start:end]
    after = values_after[start:end]
    dq_before = finite_dq(values_before)[start:end] * fps
    dq_after = finite_dq(values_after)[start:end] * fps

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, before, label="before", color="#d62728", linewidth=1.5)
    axes[0].plot(t, after, label="after", color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel(f"{joint} position [deg]")
    axes[0].legend()
    axes[1].plot(t, dq_before, label="before", color="#d62728", linewidth=1.2)
    axes[1].plot(t, dq_after, label="after", color="#1f77b4", linewidth=1.2)
    axes[1].set_ylabel(f"{joint} velocity [deg/s]")
    axes[1].set_xlabel("time around event [s]")
    for ax in axes:
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.axvspan((repair_left - event_frame) / fps, (repair_right - event_frame) / fps, color="gray", alpha=0.15)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def render_window(
    csv_path: Path,
    output: Path,
    event_frame: int,
    fps: float,
    window_sec: float,
    robot: str,
    mjcf: Path,
    python_command: str,
) -> None:
    half = max(1, int(round(window_sec * fps * 0.5)))
    start = max(0, event_frame - half)
    end = event_frame + half
    cmd = [
        *shlex.split(python_command),
        str(REPO_ROOT / "tools" / "render_robot_motion.py"),
        str(csv_path),
        "--robot",
        robot,
        "--mjcf",
        str(mjcf),
        "--output",
        str(output),
        "--fps",
        str(fps),
        "--video-fps",
        "60",
        "--stride",
        "2",
        "--start-frame",
        str(start),
        "--end-frame",
        str(end),
        "--width",
        "960",
        "--height",
        "540",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def concat_side_by_side(left: Path, right: Path, output: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-an",
        str(output),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    args = parse_args()
    cases = json.loads(args.cases_json.read_text(encoding="utf-8"))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for case in cases:
        name = str(case["name"])
        csv_path = Path(case["csv"]).expanduser().resolve()
        joint = str(case["joint"])
        event_frame = int(case["event_frame"])
        col = f"{joint}_dof"
        case_dir = output_dir / name
        fieldnames, rows = read_csv(csv_path)
        if col not in fieldnames:
            raise ValueError(f"{csv_path} missing {col}")
        before = np.asarray([float(row[col]) for row in rows], dtype=np.float64)
        after, metrics = repair_values(
            before,
            event_frame,
            int(args.repair_radius_frames),
            int(args.max_repair_radius_frames),
            int(args.radius_step_frames),
            float(args.spike_anchor_threshold_deg),
            not bool(args.disable_adaptive_spike_window),
            str(args.repair_mode),
            int(args.endpoint_fit_frames),
        )
        repaired_rows = [dict(row) for row in rows]
        for row, value in zip(repaired_rows, after, strict=True):
            row[col] = f"{float(value):.10g}"

        repaired_csv = case_dir / "repaired_csv" / f"{csv_path.stem}__{joint}__f{event_frame}.csv"
        write_csv(repaired_csv, fieldnames, repaired_rows)
        plot_png = case_dir / "plots" / f"{csv_path.stem}__{joint}__f{event_frame}.png"
        plot_trace(
            plot_png,
            before,
            after,
            joint,
            event_frame,
            int(metrics["repair_left"]),
            int(metrics["repair_right"]),
            float(args.fps),
            float(args.plot_window_sec),
        )
        record: dict[str, object] = {
            "name": name,
            "source_csv": str(csv_path),
            "repaired_csv": str(repaired_csv),
            "joint": joint,
            **metrics,
            "plot_png": str(plot_png),
        }
        if args.render:
            orig_mp4 = case_dir / "videos" / f"{csv_path.stem}__before.mp4"
            repaired_mp4 = case_dir / "videos" / f"{csv_path.stem}__after.mp4"
            combined_mp4 = case_dir / "videos" / f"{csv_path.stem}__before_after.mp4"
            render_window(csv_path, orig_mp4, event_frame, args.fps, args.render_window_sec, args.robot, args.mjcf, args.python_command)
            render_window(repaired_csv, repaired_mp4, event_frame, args.fps, args.render_window_sec, args.robot, args.mjcf, args.python_command)
            concat_side_by_side(orig_mp4, repaired_mp4, combined_mp4)
            record.update({"before_video": str(orig_mp4), "after_video": str(repaired_mp4), "combined_video": str(combined_mp4)})
        manifest.append(record)

    with (output_dir / "repair_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(manifest[0].keys()) if manifest else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    (output_dir / "repair_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
