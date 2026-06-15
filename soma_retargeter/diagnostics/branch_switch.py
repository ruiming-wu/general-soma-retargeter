"""Detect likely IK branch switches in retargeted robot joint trajectories."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BranchSwitchConfig:
    candidate_joint_diff_deg: float = 3.0
    strong_joint_diff_deg: float = 5.0
    severe_joint_diff_deg: float = 10.0
    normalized_range_diff_threshold: float = 0.08
    local_window: int = 3
    dynamic_neighbor_min_deg: float = 2.0
    dynamic_neighbor_fraction: float = 0.45
    min_dynamic_support_frames: int = 2
    isolation_neighbor_fraction: float = 0.25
    body_small_motion_m: float = 0.03
    body_large_motion_m: float = 0.20
    body_small_rotation_deg: float = 8.0
    body_large_rotation_deg: float = 30.0
    cluster_strong_count: int = 3
    cluster_severe_count: int = 2
    cluster_min_max_dq_deg: float = 8.0
    cluster_min_accel_deg_s2: float = 50000.0


@dataclass(frozen=True)
class BranchSwitchEvent:
    motion: str
    frame0: int
    frame1: int
    joint: str
    classification: str
    severity: str
    abs_dq_deg: float
    signed_dq_deg: float
    velocity_deg_s: float
    accel_deg_s2: float
    jerk_deg_s3: float
    local_neighbor_max_dq_deg: float
    local_neighbor_median_dq_deg: float
    dynamic_support_frames: int
    normalized_range_diff: float | None
    representative_body_pos_delta_m: float | None
    representative_body_rot_delta_deg: float | None
    reason: str


def wrapped_angle_diff_rad(next_angle: np.ndarray | float, prev_angle: np.ndarray | float) -> np.ndarray | float:
    """Return the shortest signed angular difference ``next - prev`` in radians."""
    return (np.asarray(next_angle) - np.asarray(prev_angle) + np.pi) % (2.0 * np.pi) - np.pi


def _safe_abs_max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmax(np.abs(values)))


def _safe_abs_median(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmedian(np.abs(values)))


def _severity(abs_dq_deg: float, config: BranchSwitchConfig) -> str:
    if abs_dq_deg >= config.severe_joint_diff_deg:
        return "severe"
    if abs_dq_deg >= config.strong_joint_diff_deg:
        return "strong"
    return "candidate"


def _is_candidate(
    abs_dq_deg: float,
    normalized_range_diff: float | None,
    config: BranchSwitchConfig,
) -> bool:
    return (
        abs_dq_deg >= config.candidate_joint_diff_deg
        or (
            normalized_range_diff is not None
            and normalized_range_diff >= config.normalized_range_diff_threshold
            and abs_dq_deg >= config.dynamic_neighbor_min_deg
        )
    )


def _local_neighbors(values: np.ndarray, index: int, radius: int) -> np.ndarray:
    start = max(0, index - radius)
    stop = min(values.shape[0], index + radius + 1)
    if stop <= start:
        return np.empty((0,), dtype=values.dtype)
    return np.delete(values[start:stop], index - start)


def _classify_event(
    abs_dq_deg: float,
    neighbor_abs: np.ndarray,
    dynamic_support_frames: int,
    body_pos_delta_m: float | None,
    body_rot_delta_deg: float | None,
    config: BranchSwitchConfig,
) -> tuple[str, str]:
    neighbor_max = _safe_abs_max(neighbor_abs)
    locally_supported = dynamic_support_frames >= config.min_dynamic_support_frames
    isolated = (
        abs_dq_deg >= config.strong_joint_diff_deg
        and neighbor_max <= max(config.dynamic_neighbor_min_deg, abs_dq_deg * config.isolation_neighbor_fraction)
    )

    body_large = (
        (body_pos_delta_m is not None and body_pos_delta_m >= config.body_large_motion_m)
        or (body_rot_delta_deg is not None and body_rot_delta_deg >= config.body_large_rotation_deg)
    )
    body_small = (
        (body_pos_delta_m is None or body_pos_delta_m <= config.body_small_motion_m)
        and (body_rot_delta_deg is None or body_rot_delta_deg <= config.body_small_rotation_deg)
    )

    if locally_supported:
        return "high_dynamic_motion", "joint jump has nearby same-joint velocity support"
    if body_large:
        return "source_or_retarget_discontinuity", "joint jump coincides with large representative body motion"
    if isolated and body_small:
        return "probable_branch_switch", "isolated joint jump with little representative body motion"
    if isolated:
        return "probable_branch_switch", "isolated joint jump"
    return "possible_branch_switch", "joint jump lacks enough local dynamic support"


def detect_joint_branch_switch_events(
    joint_pos_rad: np.ndarray,
    joint_names: Iterable[str],
    fps: float,
    *,
    joint_limits_rad: dict[str, tuple[float, float]] | None = None,
    representative_body_pos_delta_m: np.ndarray | None = None,
    representative_body_rot_delta_deg: np.ndarray | None = None,
    motion_name: str = "",
    config: BranchSwitchConfig | None = None,
) -> list[BranchSwitchEvent]:
    """Detect suspicious adjacent-frame joint jumps.

    ``joint_pos_rad`` is shaped ``[num_frames, num_joints]``. Optional representative
    body deltas are shaped ``[num_frames - 1, num_joints]`` and should map each joint
    to its most relevant end-effector/body delta.
    """
    cfg = config or BranchSwitchConfig()
    q = np.asarray(joint_pos_rad, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"joint_pos_rad must be [frames, joints], got shape {q.shape}")
    if q.shape[0] < 2:
        return []
    names = list(joint_names)
    if len(names) != q.shape[1]:
        raise ValueError(f"joint_names has {len(names)} entries but joint_pos_rad has {q.shape[1]} joints")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    dq_rad = wrapped_angle_diff_rad(q[1:], q[:-1])
    dq_deg = np.rad2deg(dq_rad)
    abs_dq_deg = np.abs(dq_deg)
    vel_deg_s = dq_deg * float(fps)
    accel_deg_s2 = np.zeros_like(vel_deg_s)
    if vel_deg_s.shape[0] > 1:
        accel_deg_s2[1:] = (vel_deg_s[1:] - vel_deg_s[:-1]) * float(fps)
    jerk_deg_s3 = np.zeros_like(vel_deg_s)
    if accel_deg_s2.shape[0] > 1:
        jerk_deg_s3[1:] = (accel_deg_s2[1:] - accel_deg_s2[:-1]) * float(fps)

    joint_ranges_deg = np.full((q.shape[1],), np.nan, dtype=np.float64)
    if joint_limits_rad:
        for joint_idx, joint_name in enumerate(names):
            if joint_name not in joint_limits_rad:
                continue
            lo, hi = joint_limits_rad[joint_name]
            width = float(hi - lo)
            if width > 1e-8:
                joint_ranges_deg[joint_idx] = np.rad2deg(width)

    normalized_diffs = np.full_like(abs_dq_deg, np.nan)
    valid_range = np.isfinite(joint_ranges_deg) & (joint_ranges_deg > 1e-8)
    if np.any(valid_range):
        normalized_diffs[:, valid_range] = abs_dq_deg[:, valid_range] / joint_ranges_deg[valid_range]

    candidate_mask = abs_dq_deg >= cfg.candidate_joint_diff_deg
    candidate_mask |= (
        np.isfinite(normalized_diffs)
        & (normalized_diffs >= cfg.normalized_range_diff_threshold)
        & (abs_dq_deg >= cfg.dynamic_neighbor_min_deg)
    )
    candidate_indices = np.argwhere(candidate_mask)

    events: list[BranchSwitchEvent] = []
    for frame_idx, joint_idx in candidate_indices:
        frame_idx = int(frame_idx)
        joint_idx = int(joint_idx)
        joint_name = names[joint_idx]
        abs_jump = float(abs_dq_deg[frame_idx, joint_idx])
        normalized = float(normalized_diffs[frame_idx, joint_idx]) if np.isfinite(normalized_diffs[frame_idx, joint_idx]) else None
        neighbor = _local_neighbors(abs_dq_deg[:, joint_idx], frame_idx, cfg.local_window)
        support_threshold = max(cfg.dynamic_neighbor_min_deg, abs_jump * cfg.dynamic_neighbor_fraction)
        signed_neighbors = _local_neighbors(dq_deg[:, joint_idx], frame_idx, cfg.local_window)
        current_sign = np.sign(dq_deg[frame_idx, joint_idx])
        if current_sign == 0:
            dynamic_support = 0
        else:
            dynamic_support = int(
                np.sum((np.abs(signed_neighbors) >= support_threshold) & (np.sign(signed_neighbors) == current_sign))
            )

        body_pos = None
        if representative_body_pos_delta_m is not None:
            body_pos = float(representative_body_pos_delta_m[frame_idx, joint_idx])
        body_rot = None
        if representative_body_rot_delta_deg is not None:
            body_rot = float(representative_body_rot_delta_deg[frame_idx, joint_idx])

        classification, reason = _classify_event(abs_jump, neighbor, dynamic_support, body_pos, body_rot, cfg)
        events.append(
            BranchSwitchEvent(
                motion=motion_name,
                frame0=frame_idx,
                frame1=frame_idx + 1,
                joint=joint_name,
                classification=classification,
                severity=_severity(abs_jump, cfg),
                abs_dq_deg=abs_jump,
                signed_dq_deg=float(dq_deg[frame_idx, joint_idx]),
                velocity_deg_s=float(vel_deg_s[frame_idx, joint_idx]),
                accel_deg_s2=float(accel_deg_s2[frame_idx, joint_idx]),
                jerk_deg_s3=float(jerk_deg_s3[frame_idx, joint_idx]),
                local_neighbor_max_dq_deg=_safe_abs_max(neighbor),
                local_neighbor_median_dq_deg=_safe_abs_median(neighbor),
                dynamic_support_frames=dynamic_support,
                normalized_range_diff=normalized,
                representative_body_pos_delta_m=body_pos,
                representative_body_rot_delta_deg=body_rot,
                reason=reason,
            )
        )
    events.sort(key=lambda e: (e.frame0, e.joint))
    return events


def upgrade_clustered_branch_switch_events(
    events: Iterable[BranchSwitchEvent],
    *,
    config: BranchSwitchConfig | None = None,
) -> list[BranchSwitchEvent]:
    """Upgrade same-frame multi-joint jump clusters to probable branch switches.

    IK branch switches often appear as several joints jumping on the same frame.
    Single-joint local velocity support can make those jumps look like "dynamic"
    motion, so this pass adds a frame-level signal without changing source/body
    discontinuity classifications.
    """
    cfg = config or BranchSwitchConfig()
    event_list = list(events)
    by_frame: dict[tuple[str, int], list[int]] = {}
    for idx, event in enumerate(event_list):
        by_frame.setdefault((event.motion, event.frame0), []).append(idx)

    upgraded = list(event_list)
    for indices in by_frame.values():
        cluster = [event_list[i] for i in indices if event_list[i].classification != "source_or_retarget_discontinuity"]
        if not cluster:
            continue
        strong_count = sum(event.abs_dq_deg >= cfg.strong_joint_diff_deg for event in cluster)
        severe_count = sum(event.abs_dq_deg >= cfg.severe_joint_diff_deg for event in cluster)
        max_abs_dq = max(event.abs_dq_deg for event in cluster)
        max_abs_accel = max(abs(event.accel_deg_s2) for event in cluster)
        has_cluster_signature = (
            max_abs_accel >= cfg.cluster_min_accel_deg_s2
            and max_abs_dq >= cfg.cluster_min_max_dq_deg
            and (severe_count >= cfg.cluster_severe_count or strong_count >= cfg.cluster_strong_count)
        )
        if not has_cluster_signature:
            continue
        reason = (
            f"same-frame multi-joint jump cluster "
            f"(strong={strong_count}, severe={severe_count}, max_dq={max_abs_dq:.2f}deg)"
        )
        for idx in indices:
            event = event_list[idx]
            if event.classification == "source_or_retarget_discontinuity":
                continue
            upgraded[idx] = replace(event, classification="probable_branch_switch", reason=reason)

    upgraded.sort(key=lambda e: (e.motion, e.frame0, e.joint))
    return upgraded
