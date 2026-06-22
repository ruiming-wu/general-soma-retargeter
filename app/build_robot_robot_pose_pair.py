#!/usr/bin/env python3
"""Build a robot-to-robot pose-pair JSON from two one-frame robot pose JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.robot_pose_utils import (
    ROBOT_ROBOT_POSE_PAIR_SCHEMA,
    build_semantic_targets,
    common_semantic_keys,
    load_json,
    robot_pose_body_transforms,
    write_json,
)


def build_robot_robot_pose_pair(
    source_pose_path: Path,
    target_pose_path: Path,
    base_retargeter_config_path: Path,
    pose_name: str | None = None,
) -> dict:
    source_pose_path = Path(source_pose_path).expanduser().resolve()
    target_pose_path = Path(target_pose_path).expanduser().resolve()
    base_retargeter_config_path = Path(base_retargeter_config_path).expanduser().resolve()

    source_pose = load_json(source_pose_path)
    target_pose = load_json(target_pose_path)
    retargeter_config = load_json(base_retargeter_config_path)
    source_map = retargeter_config.get("source_ik_map", {})
    target_map = retargeter_config.get("ik_map", {})
    common_semantic_keys(source_map, target_map)

    source_body_transforms = robot_pose_body_transforms(source_pose)
    target_body_transforms = robot_pose_body_transforms(target_pose)
    source_targets = build_semantic_targets(source_body_transforms, source_map)
    target_targets = build_semantic_targets(target_body_transforms, target_map)

    return {
        "schema": ROBOT_ROBOT_POSE_PAIR_SCHEMA,
        "pose_name": pose_name or source_pose_path.stem,
        "source_robot_type": source_pose.get("robot_type", retargeter_config.get("source_type", "")),
        "target_robot_type": target_pose.get("robot_type", retargeter_config.get("target_type", "")),
        "source_robot_mjcf": source_pose.get("robot_mjcf", retargeter_config.get("source_robot_mjcf_path", "")),
        "target_robot_mjcf": target_pose.get("robot_mjcf", retargeter_config.get("robot_mjcf_path", "")),
        "source_pose": str(source_pose_path),
        "target_pose": str(target_pose_path),
        "base_retargeter_config": str(base_retargeter_config_path),
        "source_robot_joint_q": source_pose.get("robot_joint_q", []),
        "target_robot_joint_q": target_pose.get("robot_joint_q", []),
        "source_robot_joints": source_pose.get("robot_joints", {}),
        "target_robot_joints": target_pose.get("robot_joints", {}),
        "source_robot_body_transforms": source_body_transforms,
        "target_robot_body_transforms": target_body_transforms,
        "source_ik_targets": source_targets,
        "target_ik_targets": target_targets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pose", type=Path, required=True)
    parser.add_argument("--target-pose", type=Path, required=True)
    parser.add_argument("--base-retargeter-config", type=Path, required=True)
    parser.add_argument("--pose-name", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair = build_robot_robot_pose_pair(
        source_pose_path=args.source_pose,
        target_pose_path=args.target_pose,
        base_retargeter_config_path=args.base_retargeter_config,
        pose_name=args.pose_name,
    )
    write_json(args.output, pair)
    print(f"[robot-robot-pose-pair] wrote: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
