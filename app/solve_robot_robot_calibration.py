#!/usr/bin/env python3
"""Solve robot-to-robot triaxial scaler/offset configs from canonical pose pairs."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.robot_pose_utils import (
    ROBOT_ROBOT_POSE_PAIR_SCHEMA,
    common_semantic_keys,
    load_json,
    normalized_quat_xyzw,
    pose_transform_from_bodies,
    robot_pose_body_transforms,
    write_json,
)
from app.solve_soma_robot_calibration import (
    _format_scale_for_md,
    _resolve_config_path,
    _robot_body_transforms_from_pose_file,
    _solve_offsets,
    _write_report_md,
)

_DEFAULT_RETARGETER = "agile_one/g1_to_ao_triaxial_retargeter_config.json"
_DEFAULT_SCALER = "agile_one/g1_to_ao_triaxial_scaler_config.json"


def _load_robot_pose(path: Path, fallback_mjcf: str | None = None) -> dict:
    data = load_json(path)
    if "robot_body_transforms" not in data:
        mjcf = Path(fallback_mjcf).expanduser().resolve() if fallback_mjcf else None
        data = copy.deepcopy(data)
        data["robot_body_transforms"] = _robot_body_transforms_from_pose_file(path, mjcf)
    return data


def _pair_paths_from_sample(sample_path: Path) -> tuple[Path, Path]:
    data = load_json(sample_path)
    if data.get("schema") != ROBOT_ROBOT_POSE_PAIR_SCHEMA:
        raise ValueError(f"Expected {ROBOT_ROBOT_POSE_PAIR_SCHEMA}: {sample_path}")
    source_pose = Path(data["source_pose"]).expanduser().resolve()
    target_pose = Path(data["target_pose"]).expanduser().resolve()
    return source_pose, target_pose


def build_pairs(args: argparse.Namespace) -> list[tuple[Path, Path, Path | None]]:
    if args.sample_files:
        return [
            (Path(p).expanduser().resolve(), *_pair_paths_from_sample(Path(p).expanduser().resolve()))
            for p in args.sample_files
        ]

    if args.samples_dir:
        sample_files = sorted(Path(args.samples_dir).expanduser().resolve().glob("*.json"))
        pairs = []
        for sample in sample_files:
            try:
                data = load_json(sample)
            except Exception:
                continue
            if data.get("schema") != ROBOT_ROBOT_POSE_PAIR_SCHEMA:
                continue
            pairs.append((sample, *_pair_paths_from_sample(sample)))
        if not pairs:
            raise ValueError(f"No robot-to-robot pose-pair JSON files found in {args.samples_dir}")
        return pairs

    source_pose_files = [Path(p).expanduser().resolve() for p in args.source_pose_files or []]
    target_pose_files = [Path(p).expanduser().resolve() for p in args.target_pose_files or []]
    if len(source_pose_files) != len(target_pose_files):
        raise ValueError(
            "--source-pose-files and --target-pose-files must have equal length: "
            f"{len(source_pose_files)} != {len(target_pose_files)}"
        )
    if not source_pose_files:
        raise ValueError("Provide --samples-dir, --sample-files, or equal-length --source-pose-files/--target-pose-files")
    return [(None, source, target) for source, target in zip(source_pose_files, target_pose_files)]


def collect_robot_robot_observations(
    sample_files: list[Path],
    retargeter_config: dict,
    root_name: str = "Hips",
) -> dict[str, list[dict]]:
    pairs = [(Path(sample).expanduser().resolve(), *_pair_paths_from_sample(Path(sample).expanduser().resolve())) for sample in sample_files]
    return collect_observations_from_pairs(pairs, retargeter_config, root_name=root_name)


def collect_observations_from_pairs(
    pairs: list[tuple[Path | None, Path, Path]],
    retargeter_config: dict,
    root_name: str = "Hips",
) -> dict[str, list[dict]]:
    source_map = retargeter_config.get("source_ik_map", {})
    target_map = retargeter_config.get("ik_map", {})
    semantic_keys = common_semantic_keys(source_map, target_map)

    source_root_mapping = source_map.get(root_name)
    target_root_mapping = target_map.get(root_name)
    if source_root_mapping is None or target_root_mapping is None:
        raise ValueError(f"Root semantic key [{root_name}] must exist in both source_ik_map and ik_map")

    observations: dict[str, list[dict]] = {}
    for sample_path, source_pose_path, target_pose_path in pairs:
        source_pose = _load_robot_pose(source_pose_path, retargeter_config.get("source_robot_mjcf_path"))
        target_pose = _load_robot_pose(target_pose_path, retargeter_config.get("robot_mjcf_path"))
        source_bodies = robot_pose_body_transforms(source_pose)
        target_bodies = robot_pose_body_transforms(target_pose)

        source_root = np.asarray(
            pose_transform_from_bodies(
                source_bodies,
                source_root_mapping["t_body"],
                source_root_mapping["r_body"],
            ),
            dtype=np.float64,
        )
        target_root = np.asarray(
            pose_transform_from_bodies(
                target_bodies,
                target_root_mapping["t_body"],
                target_root_mapping["r_body"],
            ),
            dtype=np.float64,
        )

        for semantic_name in semantic_keys:
            source_mapping = source_map[semantic_name]
            target_mapping = target_map[semantic_name]
            source_tx = np.asarray(
                pose_transform_from_bodies(source_bodies, source_mapping["t_body"], source_mapping["r_body"]),
                dtype=np.float64,
            )
            target_tx = np.asarray(
                pose_transform_from_bodies(target_bodies, target_mapping["t_body"], target_mapping["r_body"]),
                dtype=np.float64,
            )
            observations.setdefault(semantic_name, []).append(
                {
                    "sample": str(sample_path.resolve()) if sample_path else "",
                    "source_pose": str(source_pose_path.resolve()),
                    "target_pose": str(target_pose_path.resolve()),
                    "human": source_tx,
                    "human_root": source_root,
                    "robot_root_position": target_root[:3],
                    "robot_root_rotation": normalized_quat_xyzw(target_root[3:7]),
                    "robot_position": target_tx[:3],
                    "robot_rotation": normalized_quat_xyzw(target_tx[3:7]),
                    "t_body": target_mapping["t_body"],
                    "r_body": target_mapping["r_body"],
                    "source_t_body": source_mapping["t_body"],
                    "source_r_body": source_mapping["r_body"],
                }
            )
    return observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=str, default=None)
    parser.add_argument("--sample-files", nargs="*", default=None)
    parser.add_argument("--source-pose-files", nargs="*", default=None)
    parser.add_argument("--target-pose-files", nargs="*", default=None)
    parser.add_argument("--base-retargeter-config", type=str, default=_DEFAULT_RETARGETER)
    parser.add_argument("--base-scaler-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(_REPO_ROOT / "output/g1_to_ao_calibration_solve"))
    parser.add_argument("--output-prefix", type=str, default="g1_to_ao_triaxial")
    parser.add_argument("--root-name", type=str, default="Hips")
    parser.add_argument("--root-scale", type=float, default=None)
    parser.add_argument("--scale-min", type=float, default=0.3)
    parser.add_argument("--scale-max", type=float, default=1.5)
    parser.add_argument("--anisotropic-scale", action="store_true", default=True)
    parser.add_argument("--scalar-scale", dest="anisotropic_scale", action="store_false")
    parser.add_argument("--axis-motion-min", type=float, default=0.02)
    parser.add_argument("--root-align-positions", dest="root_align_positions", action="store_true", default=True)
    parser.add_argument("--no-root-align-positions", dest="root_align_positions", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retargeter_path = _resolve_config_path(args.base_retargeter_config)
    retargeter_config = load_json(retargeter_path)
    scaler_path = _resolve_config_path(
        args.base_scaler_config
        or retargeter_config.get("source_robot_to_robot_scaler_config")
        or retargeter_config.get("human_robot_scaler_config")
        or _DEFAULT_SCALER
    )
    scaler_config = load_json(scaler_path)

    pairs = build_pairs(args)
    print(f"[robot-robot-calibration] pose pairs: {len(pairs)}")
    observations = collect_observations_from_pairs(pairs, retargeter_config, root_name=args.root_name)

    solve_args = SimpleNamespace(
        human_root_name=args.root_name,
        root_scale=args.root_scale,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        anisotropic_scale=args.anisotropic_scale,
        axis_motion_min=args.axis_motion_min,
        root_align_positions=args.root_align_positions,
    )
    solved_scaler, solved_retargeter, report = _solve_offsets(observations, scaler_config, retargeter_config, solve_args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_scaler = output_dir / f"{args.output_prefix}_scaler_config.json"
    out_retargeter = output_dir / f"{args.output_prefix}_retargeter_config.json"
    out_report_json = output_dir / "calibration_solve_report.json"
    out_report_md = output_dir / "calibration_solve_report.md"

    solved_scaler["source_type"] = retargeter_config.get("source_type", "unitree_g1")
    solved_scaler["target_type"] = retargeter_config.get("target_type", "agile_one")
    solved_scaler["robot_type"] = retargeter_config.get("target_type", "agile_one")
    solved_retargeter["source_robot_to_robot_scaler_config"] = str(out_scaler)
    solved_retargeter["human_robot_scaler_config"] = str(out_scaler)

    write_json(out_scaler, solved_scaler)
    write_json(out_retargeter, solved_retargeter)
    write_json(out_report_json, report)
    _write_report_md(out_report_md, report, out_scaler, out_retargeter)

    solved = [item for item in report["joints"] if item["status"] == "solved"]
    print(f"[robot-robot-calibration] solved joints: {len(solved)}")
    print(f"[robot-robot-calibration] scaler: {out_scaler}")
    print(f"[robot-robot-calibration] retargeter: {out_retargeter}")
    print(f"[robot-robot-calibration] report: {out_report_md}")


if __name__ == "__main__":
    main()
