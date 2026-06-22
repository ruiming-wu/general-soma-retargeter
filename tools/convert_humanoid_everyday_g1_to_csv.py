#!/usr/bin/env python3
"""Convert Humanoid Everyday G1 lite parquet files to retargeter G1 CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyarrow is required. Run with: uv run --with pyarrow python ...") from exc

try:
    from scipy.spatial.transform import Rotation as R
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required. Run with: uv run --with scipy --with pyarrow python ...") from exc


G1_CSV_HEADER = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]


LEG_TO_RETARGETER_G1 = [
    2, 1, 0, 3, 4, 5,
    8, 7, 6, 9, 10, 11,
    12, 13, 14,
]

ARM_TO_RETARGETER_G1 = list(range(14))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _episode_path(root: Path, episode_index: int) -> Path:
    return root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def _output_path(output_root: Path, episode_index: int) -> Path:
    return output_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.csv"


def _list_episode_indices(root: Path, limit: int | None, episodes: list[int] | None) -> list[int]:
    if episodes:
        indices = sorted(set(episodes))
    else:
        indices = sorted(int(path.stem.removeprefix("episode_")) for path in (root / "data").glob("chunk-*/*.parquet"))
    if limit is not None:
        indices = indices[:limit]
    return indices


def convert_episode(input_path: Path, output_path: Path, use_action_arm: bool, root_source: str) -> tuple[int, int]:
    columns = ["observation.leg_joints", "observation.arm_joints", "action"]
    if root_source == "odometry":
        columns.extend(["observation.odometry.position", "observation.odometry.quat"])

    table = pq.read_table(
        input_path,
        columns=columns,
    )
    leg_col = table["observation.leg_joints"]
    arm_col = table["observation.arm_joints"]
    action_col = table["action"]
    odom_pos_col = table["observation.odometry.position"] if root_source == "odometry" else None
    odom_quat_col = table["observation.odometry.quat"] if root_source == "odometry" else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(G1_CSV_HEADER)
        for frame_idx in range(table.num_rows):
            leg = np.asarray(leg_col[frame_idx].as_py(), dtype=np.float64)
            if use_action_arm:
                action = np.asarray(action_col[frame_idx].as_py(), dtype=np.float64)
                arm = action[:14]
            else:
                arm = np.asarray(arm_col[frame_idx].as_py(), dtype=np.float64)

            if leg.shape[0] != 15:
                raise ValueError(f"{input_path}: expected 15 leg joints, got {leg.shape[0]}")
            if arm.shape[0] != 14:
                raise ValueError(f"{input_path}: expected 14 arm joints, got {arm.shape[0]}")

            joints_rad = np.concatenate([leg[LEG_TO_RETARGETER_G1], arm[ARM_TO_RETARGETER_G1]])
            joints_deg = np.rad2deg(joints_rad)
            if root_source == "odometry":
                root_xyz_cm = np.asarray(odom_pos_col[frame_idx].as_py(), dtype=np.float64) * 100.0
                quat_wxyz = np.asarray(odom_quat_col[frame_idx].as_py(), dtype=np.float64)
                quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
                root_euler_deg = R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
            else:
                # Humanoid-Everyday-G1 lite does not include base odometry, so keep root fixed.
                root_xyz_cm = np.zeros(3, dtype=np.float64)
                root_euler_deg = np.zeros(3, dtype=np.float64)
            writer.writerow([frame_idx, *root_xyz_cm.tolist(), *root_euler_deg.tolist(), *joints_deg.tolist()])

    return table.num_rows, output_path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/home/ruiming.wu/data/Humanoid-Everyday-G1"))
    parser.add_argument("--output-root", type=Path, default=Path("/home/ruiming.wu/data/Humanoid-Everyday-G1/g1_csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument(
        "--use-action-arm",
        action="store_true",
        help="Use action[:14] arm IK solution instead of observation.arm_joints.",
    )
    parser.add_argument(
        "--root-source",
        choices=("fixed", "odometry"),
        default="fixed",
        help="Use fixed zero root or odometry.position/quat from rooted parquet.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    indices = _list_episode_indices(root, args.limit, args.episodes)
    if not indices:
        raise SystemExit(f"No parquet episodes found under {root / 'data'}")

    episodes_meta = {row.get("episode_index"): row for row in _load_jsonl(root / "meta" / "episodes.jsonl")}
    tasks_meta = {row.get("task_index"): row for row in _load_jsonl(root / "meta" / "tasks.jsonl")}
    manifest_path = output_root / "manifest.csv"
    output_root.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    converted = 0
    skipped = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=[
                "episode_index",
                "frames",
                "csv_path",
                "task_index",
                "category",
                "task",
                "instruction",
                "root_policy",
                "arm_source",
            ],
        )
        writer.writeheader()
        for n, idx in enumerate(indices, 1):
            src = _episode_path(root, idx)
            dst = _output_path(output_root, idx)
            if not src.exists():
                raise FileNotFoundError(src)
            if dst.exists() and not args.overwrite:
                skipped += 1
                frames = sum(1 for _ in dst.open("r", encoding="utf-8")) - 1
            else:
                frames, _ = convert_episode(src, dst, args.use_action_arm, args.root_source)
                converted += 1
            total_frames += frames

            ep = episodes_meta.get(idx, {})
            task_index = (ep.get("tasks") or [None])[0]
            task = tasks_meta.get(task_index, {})
            writer.writerow(
                {
                    "episode_index": idx,
                    "frames": frames,
                    "csv_path": str(dst),
                    "task_index": task_index,
                    "category": task.get("category", ""),
                    "task": task.get("task", ""),
                    "instruction": ep.get("instruction", ""),
                    "root_policy": args.root_source,
                    "arm_source": "action[:14]" if args.use_action_arm else "observation.arm_joints",
                }
            )
            if n % 250 == 0 or n == len(indices):
                print(
                    f"{n}/{len(indices)} episodes processed; converted={converted}; "
                    f"skipped={skipped}; frames={total_frames}",
                    flush=True,
                )

    print(f"output: {output_root}")
    print(f"manifest: {manifest_path}")
    print(f"episodes: {len(indices)} frames: {total_frames} hours@30Hz: {total_frames / 30.0 / 3600.0:.6f}")


if __name__ == "__main__":
    main()
