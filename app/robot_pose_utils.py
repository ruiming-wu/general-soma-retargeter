#!/usr/bin/env python3
"""Shared helpers for robot pose and robot-to-robot calibration scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROBOT_POSE_SCHEMA = "robot_pose.v1"
ROBOT_ROBOT_POSE_PAIR_SCHEMA = "robot_robot_pose_pair.v1"


def load_json(path: Path | str) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, data: dict[str, Any]) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def normalized_quat_xyzw(quat: list[float] | np.ndarray) -> np.ndarray:
    quat_np = np.asarray(quat, dtype=np.float64)
    return quat_np / max(float(np.linalg.norm(quat_np)), 1e-12)


def pose_transform_from_bodies(
    body_transforms: dict[str, list[float]],
    t_body: str,
    r_body: str,
) -> list[float]:
    if t_body not in body_transforms:
        raise KeyError(f"Missing translation body [{t_body}] in robot pose")
    if r_body not in body_transforms:
        raise KeyError(f"Missing rotation body [{r_body}] in robot pose")
    t_row = body_transforms[t_body]
    r_row = body_transforms[r_body]
    return [
        float(t_row[0]),
        float(t_row[1]),
        float(t_row[2]),
        *[float(x) for x in normalized_quat_xyzw(r_row[3:7])],
    ]


def build_semantic_targets(
    body_transforms: dict[str, list[float]],
    semantic_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for semantic_name, mapping in semantic_map.items():
        t_body = mapping["t_body"]
        r_body = mapping["r_body"]
        transform = pose_transform_from_bodies(body_transforms, t_body, r_body)
        targets[semantic_name] = {
            "t_body": t_body,
            "r_body": r_body,
            "target_position": transform[:3],
            "target_rotation_xyzw": transform[3:7],
        }
    return targets


def robot_pose_body_transforms(pose: dict[str, Any]) -> dict[str, list[float]]:
    body_transforms = pose.get("robot_body_transforms")
    if not isinstance(body_transforms, dict):
        raise ValueError("Robot pose JSON must include robot_body_transforms")
    return body_transforms


def common_semantic_keys(source_map: dict[str, Any], target_map: dict[str, Any]) -> list[str]:
    source_keys = set(source_map)
    target_keys = set(target_map)
    missing_source = sorted(target_keys - source_keys)
    missing_target = sorted(source_keys - target_keys)
    if missing_source or missing_target:
        raise ValueError(
            "source_ik_map and ik_map must have identical semantic keys; "
            f"missing_source={missing_source}; missing_target={missing_target}"
        )
    return sorted(source_keys)
