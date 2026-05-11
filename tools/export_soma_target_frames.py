#!/usr/bin/env python3
"""Export SOMA retargeting target frames for video overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str


TARGET_JOINTS = {"Hips", "Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot"}


def export_targets(bvh_path: Path, robot: str, output: Path, facing_direction: str, max_frames: int | None) -> None:
    skeleton, animation = bvh_utils.load_bvh(str(bvh_path))
    if max_frames is not None and animation.num_frames > max_frames:
        animation.local_transforms = np.copy(animation.local_transforms[:max_frames])
        animation.num_frames = max_frames

    converter = SpaceConverter(get_facing_direction_type_from_str(facing_direction))
    source_transform = converter.transform(wp.transform_identity())
    pipeline = newton_pipeline.NewtonPipeline(skeleton, "soma", robot)
    pipeline.add_input_motions([animation], [source_transform], True)

    removed = pipeline.num_initialization_frames + pipeline.num_stabilization_frames
    mapped_indices = [idx for idx, name in enumerate(pipeline.mapped_joints) if name in TARGET_JOINTS]
    names = [pipeline.mapped_joints[idx] for idx in mapped_indices]
    raw_targets = pipeline.input_targets[0][removed:]
    if max_frames is not None:
        raw_targets = raw_targets[:max_frames]

    positions = []
    quats = []
    for frame_targets in raw_targets:
        frame_positions = []
        frame_quats = []
        for idx in mapped_indices:
            target = np.asarray(frame_targets[idx], dtype=np.float64)
            frame_positions.append(target[:3])
            frame_quats.append(target[3:7])
        positions.append(frame_positions)
        quats.append(frame_quats)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        names=np.asarray(names, dtype=object),
        positions=np.asarray(positions, dtype=np.float64),
        quats_xyzw=np.asarray(quats, dtype=np.float64),
        source=str(bvh_path),
        method="soma",
        robot=robot,
    )
    print(f"[soma-targets] wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--robot", choices=("unitree_g1", "agile_one"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retarget-source-facing-direction", choices=("Mujoco", "Maya"), default="Mujoco")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_targets(
        bvh_path=args.bvh,
        robot=args.robot,
        output=args.output,
        facing_direction=args.retarget_source_facing_direction,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
