#!/usr/bin/env python3
"""Export seed-retargeted G1 robot_motionlib data to TWIST2 MotionLib format.

The seed files are SONIC/joblib PKLs with ``root_trans_offset``, ``root_rot``,
``pose_aa`` and ``dof``.  TWIST2's original IsaacGym loader expects plain
pickle files with FK-derived ``local_body_pos`` plus a YAML motion list.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
import yaml


DEFAULT_SONIC_REPO = Path("/home/ruiming.wu/codes/GR00T-WholeBodyControl")
if DEFAULT_SONIC_REPO.exists() and str(DEFAULT_SONIC_REPO) not in sys.path:
    sys.path.insert(0, str(DEFAULT_SONIC_REPO))

from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch  # noqa: E402


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "motion"


def _quat_rotate_inverse_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_xyz = -quat[:, None, :3]
    q_w = quat[:, None, 3:4]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + q_w * t + np.cross(q_xyz, t)


def _load_humanoid(mjcf_file: Path, device: str) -> Humanoid_Batch:
    cfg = SimpleNamespace(
        asset=SimpleNamespace(
            assetRoot=str(mjcf_file.parent),
            assetFileName=mjcf_file.name,
        ),
        extend_config=[],
    )
    return Humanoid_Batch(cfg, torch.device(device))


def _iter_source_files(input_root: Path, limit: int | None) -> list[Path]:
    files = sorted(input_root.rglob("*.pkl"))
    if limit is not None and limit >= 0:
        files = files[:limit]
    return files


def _convert_motion(
    humanoid: Humanoid_Batch,
    motion_name: str,
    motion_data: dict,
    out_file: Path,
    overwrite: bool,
    device: str,
) -> bool:
    if out_file.exists() and not overwrite:
        return False

    required = ("root_trans_offset", "root_rot", "pose_aa")
    missing = [key for key in required if key not in motion_data]
    if missing:
        raise KeyError(f"{motion_name}: missing required fields {missing}")

    root_pos = np.asarray(motion_data["root_trans_offset"], dtype=np.float32)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float32)
    pose_aa = np.asarray(motion_data["pose_aa"], dtype=np.float32)
    dof_pos = np.asarray(motion_data.get("dof", motion_data.get("dof_pos")), dtype=np.float32)
    fps = float(motion_data.get("fps", 30.0))

    if dof_pos.shape[-1] < humanoid.num_dof:
        raise ValueError(f"{motion_name}: expected >= {humanoid.num_dof} DoFs, got {dof_pos.shape[-1]}")
    dof_pos = dof_pos[:, : humanoid.num_dof]

    with torch.no_grad():
        fk = humanoid.fk_batch(
            torch.from_numpy(pose_aa).to(device).unsqueeze(0),
            torch.from_numpy(root_pos).to(device).unsqueeze(0),
            return_full=True,
            fps=fps,
            target_fps=fps,
            interpolate_data=False,
            use_parallel_fk=True,
        )
    global_body_pos = fk.global_translation.squeeze(0).detach().cpu().numpy().astype(np.float32)
    local_body_pos = _quat_rotate_inverse_xyzw(root_rot, global_body_pos - root_pos[:, None, :])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("wb") as f:
        pickle.dump(
            {
                "fps": fps,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": local_body_pos.astype(np.float32),
                "link_body_list": list(humanoid.body_names),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/ruiming.wu/data/seed-retargeted/g1_motionlib/robot_motionlib"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/ruiming.wu/data/seed-retargeted/g1_motionlib/twist2_motionlib/robot_motionlib"),
    )
    parser.add_argument(
        "--yaml-out",
        type=Path,
        default=Path("/home/ruiming.wu/data/seed-retargeted/g1_motionlib/twist2_motionlib/seed_g1_twist2.yaml"),
    )
    parser.add_argument(
        "--twist2-config-out",
        type=Path,
        default=Path("/home/ruiming.wu/codes/TWIST2/legged_gym/motion_data_configs/seed_g1_twist2.yaml"),
        help="Optional copy of the motion YAML inside TWIST2. Use an empty string to skip.",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("/home/ruiming.wu/codes/TWIST2/assets/g1/g1_mocap_29dof.xml"),
        help="TWIST2 G1 mocap MJCF used to generate the 38-body FK fields expected by TWIST2.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    humanoid = _load_humanoid(args.mjcf, args.device)
    source_files = _iter_source_files(args.input_root, args.limit)
    motions = []
    written = 0

    for src in source_files:
        loaded = joblib.load(src)
        if not isinstance(loaded, dict):
            raise TypeError(f"{src} did not contain a motion dictionary")
        rel_dir = src.parent.relative_to(args.input_root)
        for key, motion_data in loaded.items():
            motion_name = _safe_name(str(key))
            out_rel = rel_dir / f"{motion_name}.pkl"
            out_file = args.output_root / out_rel
            if _convert_motion(humanoid, motion_name, motion_data, out_file, args.overwrite, args.device):
                written += 1
            motions.append(
                {
                    "file": os.fspath(out_rel),
                    "weight": 1.0,
                    "description": "seed-retargeted G1 robot_motionlib exported for TWIST2",
                }
            )

    payload = {"root_path": os.fspath(args.output_root), "motions": motions}
    args.yaml_out.parent.mkdir(parents=True, exist_ok=True)
    with args.yaml_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    if args.twist2_config_out and str(args.twist2_config_out):
        args.twist2_config_out.parent.mkdir(parents=True, exist_ok=True)
        with args.twist2_config_out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

    print(f"Converted {written} motions from {len(source_files)} files")
    print(f"Motion YAML: {args.yaml_out}")
    if args.twist2_config_out and str(args.twist2_config_out):
        print(f"TWIST2 config YAML: {args.twist2_config_out}")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
