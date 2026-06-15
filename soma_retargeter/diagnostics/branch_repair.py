"""Local repair utilities for suspected IK branch switches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soma_retargeter.diagnostics.branch_switch import wrapped_angle_diff_rad


@dataclass(frozen=True)
class BranchRepairConfig:
    pre_frames: int = 4
    post_frames: int = 8
    slope_clip_deg_per_frame: float = 2.0
    fps: float = 120.0


@dataclass(frozen=True)
class BranchRepairResult:
    event_frame0: int
    event_frame1: int
    anchor_left: int
    anchor_right: int
    max_abs_dq_before_deg: float
    max_abs_dq_after_deg: float
    max_abs_vel_before_deg_s: float
    max_abs_vel_after_deg_s: float
    max_abs_acc_before_deg_s2: float
    max_abs_acc_after_deg_s2: float


def wrapped_diff_deg(next_deg: np.ndarray | float, prev_deg: np.ndarray | float) -> np.ndarray | float:
    """Return shortest signed angular difference in degrees."""
    return np.rad2deg(wrapped_angle_diff_rad(np.deg2rad(next_deg), np.deg2rad(prev_deg)))


def finite_difference_deg(values_deg: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return adjacent dq, per-frame velocity and per-frame acceleration."""
    values = np.asarray(values_deg, dtype=np.float64)
    dq = np.zeros_like(values)
    if values.shape[0] > 1:
        dq[1:] = wrapped_diff_deg(values[1:], values[:-1])
    vel = dq * float(fps)
    acc = np.zeros_like(values)
    if values.shape[0] > 1:
        acc[1:] = (vel[1:] - vel[:-1]) * float(fps)
    return dq, vel, acc


def _window_metrics(values_deg: np.ndarray, start: int, stop: int, fps: float) -> tuple[float, float, float]:
    segment = np.asarray(values_deg[start:stop], dtype=np.float64)
    if segment.shape[0] < 2:
        return 0.0, 0.0, 0.0
    dq, vel, acc = finite_difference_deg(segment, fps)
    return (
        float(np.nanmax(np.abs(dq[1:]))),
        float(np.nanmax(np.abs(vel[1:]))),
        float(np.nanmax(np.abs(acc[1:]))),
    )


def _unwrap_local_deg(values_deg: np.ndarray) -> np.ndarray:
    values = np.asarray(values_deg, dtype=np.float64)
    out = np.empty_like(values)
    out[0] = values[0]
    for idx in range(1, values.shape[0]):
        out[idx] = out[idx - 1] + float(wrapped_diff_deg(values[idx], values[idx - 1]))
    return out


def _clipped_slope(values_deg: np.ndarray, index: int, config: BranchRepairConfig) -> float:
    if index <= 0 or index >= values_deg.shape[0]:
        return 0.0
    slope = float(wrapped_diff_deg(values_deg[index], values_deg[index - 1]))
    clip = abs(float(config.slope_clip_deg_per_frame))
    if clip <= 0.0:
        return 0.0
    return float(np.clip(slope, -clip, clip))


def _hermite(y0: float, y1: float, m0: float, m1: float, steps: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    return h00 * y0 + h10 * (steps * m0) + h01 * y1 + h11 * (steps * m1)


def repair_joint_branch_jump_deg(
    values_deg: np.ndarray,
    *,
    event_frame0: int,
    config: BranchRepairConfig | None = None,
) -> tuple[np.ndarray, BranchRepairResult]:
    """Smooth one suspected adjacent-frame branch switch in a joint trajectory.

    ``event_frame0`` denotes the jump from ``event_frame0`` to ``event_frame0 + 1``.
    The function preserves the left/right anchor frames and replaces only the
    interior of the local window with a clipped-slope cubic Hermite segment.
    """
    cfg = config or BranchRepairConfig()
    values = np.asarray(values_deg, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"values_deg must be 1-D, got shape {values.shape}")
    if values.shape[0] < 3:
        return values.copy(), BranchRepairResult(0, 0, 0, values.shape[0] - 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    frame0 = int(np.clip(event_frame0, 0, values.shape[0] - 2))
    frame1 = frame0 + 1
    anchor_left = max(0, frame0 - max(0, int(cfg.pre_frames)))
    anchor_right = min(values.shape[0] - 1, frame1 + max(0, int(cfg.post_frames)))
    if anchor_right <= anchor_left + 1:
        repaired = values.copy()
        before_dq, before_vel, before_acc = _window_metrics(values, anchor_left, anchor_right + 1, cfg.fps)
        return repaired, BranchRepairResult(
            frame0,
            frame1,
            anchor_left,
            anchor_right,
            before_dq,
            before_dq,
            before_vel,
            before_vel,
            before_acc,
            before_acc,
        )

    before_dq, before_vel, before_acc = _window_metrics(values, anchor_left, anchor_right + 1, cfg.fps)
    local = _unwrap_local_deg(values[anchor_left : anchor_right + 1])
    steps = anchor_right - anchor_left
    m0 = _clipped_slope(values, anchor_left, cfg)
    m1 = _clipped_slope(values, anchor_right, cfg)
    smoothed = _hermite(float(local[0]), float(local[-1]), m0, m1, steps)

    repaired = values.copy()
    repaired[anchor_left : anchor_right + 1] = smoothed
    repaired[anchor_left] = values[anchor_left]
    repaired[anchor_right] = values[anchor_right]

    after_dq, after_vel, after_acc = _window_metrics(repaired, anchor_left, anchor_right + 1, cfg.fps)
    return repaired, BranchRepairResult(
        frame0,
        frame1,
        anchor_left,
        anchor_right,
        before_dq,
        after_dq,
        before_vel,
        after_vel,
        before_acc,
        after_acc,
    )
