#!/usr/bin/env python3
"""Plot joint position/velocity/acceleration around a suspected branch switch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--joint", required=True, help="Joint name without _dof suffix.")
    parser.add_argument("--event-frame", type=int, required=True)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--compare-csv", type=Path, default=None, help="Optional repaired CSV to overlay.")
    parser.add_argument("--compare-label", default="after")
    parser.add_argument("--label", default="before")
    return parser.parse_args()


def wrapped_diff_deg(next_deg: np.ndarray, prev_deg: np.ndarray) -> np.ndarray:
    return (next_deg - prev_deg + 180.0) % 360.0 - 180.0


def load_joint(path: Path, joint: str) -> np.ndarray:
    col = f"{joint}_dof"
    with path.expanduser().open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if col not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain {col}")
        return np.asarray([float(row[col]) for row in reader], dtype=np.float64)


def finite_difference(values_deg: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    vel = np.zeros_like(values_deg)
    if values_deg.shape[0] > 1:
        vel[1:] = wrapped_diff_deg(values_deg[1:], values_deg[:-1]) * fps
    acc = np.zeros_like(values_deg)
    if values_deg.shape[0] > 1:
        acc[1:] = (vel[1:] - vel[:-1]) * fps
    return vel, acc


def plot_panel(
    ax,
    t: np.ndarray,
    y: np.ndarray,
    event_t: float,
    ylabel: str,
    title: str | None = None,
    *,
    label: str | None = None,
) -> None:
    ax.plot(t, y, linewidth=1.2, label=label)
    ax.axvline(event_t, color="crimson", linestyle="--", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    if title:
        ax.set_title(title)


def main() -> None:
    args = parse_args()
    q = load_joint(args.csv, args.joint)
    vel, acc = finite_difference(q, args.fps)
    q_cmp = vel_cmp = acc_cmp = None
    if args.compare_csv is not None:
        q_cmp = load_joint(args.compare_csv, args.joint)
        if q_cmp.shape[0] != q.shape[0]:
            raise ValueError(f"compare CSV has {q_cmp.shape[0]} frames, expected {q.shape[0]}")
        vel_cmp, acc_cmp = finite_difference(q_cmp, args.fps)
    frames = np.arange(q.shape[0])
    t = frames / args.fps
    event_frame = min(max(0, args.event_frame), q.shape[0] - 1)
    event_t = event_frame / args.fps

    half_window = max(1, int(round(args.window_sec * args.fps * 0.5)))
    start = max(0, event_frame - half_window)
    end = min(q.shape[0], event_frame + half_window + 1)
    window = slice(start, end)

    fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex="col")
    title = args.title or f"{args.csv.stem} | {args.joint} | frame {event_frame}"
    fig.suptitle(title, fontsize=14)

    plot_panel(axes[0, 0], t, q, event_t, "pos [deg]", "Full motion", label=args.label)
    plot_panel(axes[1, 0], t, vel, event_t, "vel [deg/s]", label=args.label)
    plot_panel(axes[2, 0], t, acc, event_t, "acc [deg/s^2]", label=args.label)
    axes[2, 0].set_xlabel("time [s]")

    plot_panel(axes[0, 1], t[window], q[window], event_t, "pos [deg]", f"Local window [{start}, {end})", label=args.label)
    plot_panel(axes[1, 1], t[window], vel[window], event_t, "vel [deg/s]", label=args.label)
    plot_panel(axes[2, 1], t[window], acc[window], event_t, "acc [deg/s^2]", label=args.label)
    axes[2, 1].set_xlabel("time [s]")
    if q_cmp is not None and vel_cmp is not None and acc_cmp is not None:
        for row, data in enumerate((q_cmp, vel_cmp, acc_cmp)):
            axes[row, 0].plot(t, data, linewidth=1.2, alpha=0.8, label=args.compare_label)
            axes[row, 1].plot(t[window], data[window], linewidth=1.2, alpha=0.8, label=args.compare_label)
        for ax in axes.flat:
            ax.legend(loc="best", fontsize=8)

    dq = 0.0 if event_frame == 0 else wrapped_diff_deg(q[event_frame], q[event_frame - 1])
    compare_text = ""
    if q_cmp is not None:
        dq_cmp = 0.0 if event_frame == 0 else wrapped_diff_deg(q_cmp[event_frame], q_cmp[event_frame - 1])
        compare_text = (
            f" | compare_dq_from_prev={dq_cmp:.4f}deg, compare_vel={vel_cmp[event_frame]:.2f}deg/s, "
            f"compare_acc={acc_cmp[event_frame]:.2f}deg/s^2"
        )
    fig.text(
        0.01,
        0.01,
        (
            f"csv={args.csv}\\n"
            f"joint={args.joint}, event_frame={event_frame}, event_t={event_t:.4f}s, "
            f"dq_from_prev={dq:.4f}deg, vel={vel[event_frame]:.2f}deg/s, acc={acc[event_frame]:.2f}deg/s^2"
            f"{compare_text}"
        ),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.expanduser(), dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
