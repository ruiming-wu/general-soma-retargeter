"""Reclassify branch-switch detections using multijoint cluster criteria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BranchReclassConfig:
    cluster_frame_gap: int = 1
    strong_jump_deg: float = 5.0
    severe_jump_deg: float = 10.0
    true_min_strong_joints: int = 2
    true_min_joint_groups: int = 1
    true_min_severe_joints: int = 2


@dataclass(frozen=True)
class BranchCluster:
    start_frame: int
    end_frame: int
    risk: str
    event_count: int
    strong_joint_count: int
    severe_joint_count: int
    joint_count: int
    joint_group_count: int
    max_abs_dq_deg: float
    joints: tuple[str, ...]
    joint_groups: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class MotionReclassResult:
    risk: str
    true_ik_cluster_count: int
    repairable_cluster_count: int
    low_amplitude_cluster_count: int
    max_cluster_joint_count: int
    max_cluster_joint_group_count: int
    max_abs_dq_deg: float
    clusters: tuple[BranchCluster, ...]


def joint_group(joint: str) -> str:
    """Return a coarse kinematic group for branch-switch cluster analysis."""
    if joint.startswith("left_"):
        side = "left"
    elif joint.startswith("right_"):
        side = "right"
    else:
        side = "center"

    if "ankle" in joint:
        return f"{side}_ankle"
    if "knee" in joint:
        return f"{side}_knee"
    if "hip" in joint:
        return f"{side}_hip"
    if "wrist" in joint:
        return f"{side}_wrist"
    if "elbow" in joint:
        return f"{side}_elbow"
    if "shoulder" in joint:
        return f"{side}_shoulder"
    if "waist" in joint or "head" in joint:
        return "torso_head"
    return f"{side}_other"


def _float_event(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def _int_event(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return int(float(value))


def _relevant_events(events: Iterable[dict], config: BranchReclassConfig) -> list[dict]:
    relevant: list[dict] = []
    for event in events:
        abs_dq = _float_event(event, "abs_dq_deg")
        if abs_dq >= config.strong_jump_deg:
            relevant.append(event)
    relevant.sort(key=lambda row: (_int_event(row, "frame0"), str(row.get("joint", ""))))
    return relevant


def _cluster_events(events: list[dict], config: BranchReclassConfig) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for event in events:
        frame = _int_event(event, "frame0")
        if not clusters:
            clusters.append([event])
            continue
        last_frame = max(_int_event(row, "frame0") for row in clusters[-1])
        if frame - last_frame <= config.cluster_frame_gap:
            clusters[-1].append(event)
        else:
            clusters.append([event])
    return clusters


def classify_cluster(events: list[dict], config: BranchReclassConfig | None = None) -> BranchCluster:
    cfg = config or BranchReclassConfig()
    frames = [_int_event(event, "frame0") for event in events]
    joints = tuple(sorted({str(event.get("joint", "")) for event in events}))
    groups = tuple(sorted({joint_group(joint) for joint in joints}))
    abs_dqs = [_float_event(event, "abs_dq_deg") for event in events]
    strong_joints = {
        str(event.get("joint", "")) for event in events if _float_event(event, "abs_dq_deg") >= cfg.strong_jump_deg
    }
    severe_joints = {
        str(event.get("joint", "")) for event in events if _float_event(event, "abs_dq_deg") >= cfg.severe_jump_deg
    }

    strong_joint_count = len(strong_joints)
    severe_joint_count = len(severe_joints)
    group_count = len({joint_group(joint) for joint in strong_joints}) if strong_joints else len(groups)
    max_abs_dq = max(abs_dqs) if abs_dqs else 0.0

    is_true_ik = severe_joint_count >= cfg.true_min_severe_joints or strong_joint_count >= cfg.true_min_strong_joints
    if is_true_ik:
        risk = "true_ik_branch_switch"
        reason = "multiple candidate joints in the same local window"
    elif strong_joint_count > 0:
        risk = "repairable_smooth_conflict"
        reason = "local one/few-joint jump; treat as smoothing conflict"
    else:
        risk = "low_amplitude_possible"
        reason = "no strong joint jump in cluster"

    return BranchCluster(
        start_frame=min(frames) if frames else 0,
        end_frame=max(frames) if frames else 0,
        risk=risk,
        event_count=len(events),
        strong_joint_count=strong_joint_count,
        severe_joint_count=severe_joint_count,
        joint_count=len(joints),
        joint_group_count=group_count,
        max_abs_dq_deg=max_abs_dq,
        joints=joints,
        joint_groups=groups,
        reason=reason,
    )


def classify_motion_events(events: Iterable[dict], config: BranchReclassConfig | None = None) -> MotionReclassResult:
    cfg = config or BranchReclassConfig()
    relevant = _relevant_events(events, cfg)
    clusters = tuple(classify_cluster(cluster, cfg) for cluster in _cluster_events(relevant, cfg))
    true_count = sum(cluster.risk == "true_ik_branch_switch" for cluster in clusters)
    repairable_count = sum(cluster.risk == "repairable_smooth_conflict" for cluster in clusters)
    low_count = sum(cluster.risk == "low_amplitude_possible" for cluster in clusters)
    if true_count:
        risk = "true_ik_branch_switch"
    elif repairable_count:
        risk = "repairable_smooth_conflict"
    elif low_count:
        risk = "low_amplitude_possible"
    else:
        risk = "clean_or_dynamic"
    return MotionReclassResult(
        risk=risk,
        true_ik_cluster_count=int(true_count),
        repairable_cluster_count=int(repairable_count),
        low_amplitude_cluster_count=int(low_count),
        max_cluster_joint_count=max((cluster.joint_count for cluster in clusters), default=0),
        max_cluster_joint_group_count=max((cluster.joint_group_count for cluster in clusters), default=0),
        max_abs_dq_deg=max((cluster.max_abs_dq_deg for cluster in clusters), default=0.0),
        clusters=clusters,
    )
