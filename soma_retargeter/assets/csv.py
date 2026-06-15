# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from dataclasses import dataclass
from typing import Protocol, ClassVar, List

import numpy as np
import warp as wp

from scipy.spatial.transform import Rotation as R
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer


AGILE_ONE_FIXED_TEKKEN2_HANDS_ALIGNED_JOINT_NAMES: List[str] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_roll_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint", "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_roll_joint",
    "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_thumb_joint_0", "left_thumb_joint_1", "left_thumb_joint_2",
    "left_thumb_joint_3", "left_thumb_joint_4_follower",
    "left_index_pitch_joint_reorder_1", "left_index_swing_joint_reorder_0",
    "left_index_pip_joint", "left_index_dip_joint_follower",
    "left_middle_pitch_joint_reorder_1", "left_middle_swing_joint_reorder_0",
    "left_middle_pip_joint", "left_middle_dip_joint_follower",
    "left_ring_pitch_joint_reorder_1", "left_ring_swing_joint_reorder_0",
    "left_ring_pip_joint", "left_ring_dip_joint_follower",
    "left_little_pitch_joint_reorder_1", "left_little_swing_joint_reorder_0",
    "left_little_pip_joint", "left_little_dip_joint_follower",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_roll_joint",
    "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_thumb_joint_0", "right_thumb_joint_1", "right_thumb_joint_2",
    "right_thumb_joint_3", "right_thumb_joint_4_follower",
    "right_index_pitch_joint_reorder_1", "right_index_swing_joint_reorder_0",
    "right_index_pip_joint", "right_index_dip_joint_follower",
    "right_middle_pitch_joint_reorder_1", "right_middle_swing_joint_reorder_0",
    "right_middle_pip_joint", "right_middle_dip_joint_follower",
    "right_ring_pitch_joint_reorder_1", "right_ring_swing_joint_reorder_0",
    "right_ring_pip_joint", "right_ring_dip_joint_follower",
    "right_little_pitch_joint_reorder_1", "right_little_swing_joint_reorder_0",
    "right_little_pip_joint", "right_little_dip_joint_follower",
]


def _to_csv_row_with_joints(frame_idx: int, anim_row: np.ndarray, joint_values_rad: np.ndarray) -> List[float]:
    # translation (m -> cm)
    t = wp.vec3(*anim_row[0:3]) * 100.0
    # root rotation (quat -> euler deg)
    q = wp.quat(*anim_row[3:7])
    euler = R.from_quat([q[0], q[1], q[2], q[3]]).as_euler("xyz", degrees=True)

    row = [frame_idx, t[0], t[1], t[2], euler[0], euler[1], euler[2]]
    row.extend(np.rad2deg(joint_values_rad).tolist())

    return row


class RobotCSVConfig(Protocol):
    name: str
    csv_header: List[str]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        ...
    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> List[float]:
        ...


@dataclass
class UnitreeG129DOF_CSVConfig:
    name: str = "unitree_g1_29dof"
    csv_header: ClassVar[List[str]] = [
        "Frame",
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
        "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
        "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
        "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
        "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
        "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
        "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
        "left_shoulder_yaw_joint_dof", "left_elbow_joint_dof",
        "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
        "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
        "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof",
        "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof",
        "right_wrist_yaw_joint_dof"]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        """
        Convert one CSV row (including frame index) into one anim buffer frame.
        """
        # csv_row layout: [frame index, tx, ty, tz, rx, ry, rz, dof0, ...]
        num_joint_dofs = csv_row.shape[0] - 1 # Remove frame index
        anim_row = np.zeros(
            num_joint_dofs + 1, # euler rotate xyz values converted to quat
            dtype=np.float32)

        # translation (cm -> m)
        anim_row[0:3] = csv_row[1:4] * 0.01

        # rotation (euler deg -> quat)
        euler = np.deg2rad(csv_row[4:7])
        quat = wp.quat_rpy(euler[0], euler[1], euler[2])
        anim_row[3:7] = quat

        # remaining joints (deg -> rad)
        anim_row[7:] = np.deg2rad(csv_row[7:])

        return anim_row

    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> List[float]:
        """
        Convert one anim buffer row into a CSV row with this config's layout.
        """
        return _to_csv_row_with_joints(frame_idx, anim_row, np.asarray(anim_row[7:]))


@dataclass
class AgileOneNoHandsCSVConfig:
    name: str = "agile_one"
    csv_header: ClassVar[List[str]] = [
        "Frame",
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
        "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
        "left_knee_joint_dof", "left_ankle_roll_joint_dof", "left_ankle_pitch_joint_dof",
        "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
        "right_knee_joint_dof", "right_ankle_roll_joint_dof", "right_ankle_pitch_joint_dof",
        "waist_yaw_joint_dof", "head_yaw_joint_dof", "head_pitch_joint_dof",
        "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
        "left_shoulder_yaw_joint_dof", "left_elbow_roll_joint_dof",
        "left_wrist_yaw_joint_dof", "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof",
        "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
        "right_shoulder_yaw_joint_dof", "right_elbow_roll_joint_dof",
        "right_wrist_yaw_joint_dof", "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof"]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        return UnitreeG129DOF_CSVConfig().to_anim_frame(csv_row)

    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> List[float]:
        joint_values = np.asarray(anim_row[7:])
        target_joint_names = [name.removesuffix("_dof") for name in self.csv_header[7:]]

        if len(joint_values) == len(target_joint_names):
            selected_joint_values = joint_values
        elif len(joint_values) == len(AGILE_ONE_FIXED_TEKKEN2_HANDS_ALIGNED_JOINT_NAMES):
            # Fixed-hands Agile One retargeting includes hand joints in the Newton model.
            # Downstream SONIC motionlib remains the legacy 29-DOF no-hands layout.
            fixed_index = {
                name: idx for idx, name in enumerate(AGILE_ONE_FIXED_TEKKEN2_HANDS_ALIGNED_JOINT_NAMES)
            }
            selected_joint_values = np.asarray([joint_values[fixed_index[name]] for name in target_joint_names])
        else:
            raise ValueError(
                f"[ERROR]: Agile One CSV export expected {len(target_joint_names)} no-hands joints "
                f"or {len(AGILE_ONE_FIXED_TEKKEN2_HANDS_ALIGNED_JOINT_NAMES)} fixed-hands joints, "
                f"got {len(joint_values)}."
            )

        return _to_csv_row_with_joints(frame_idx, anim_row, selected_joint_values)


def get_csv_config(robot_type: str) -> RobotCSVConfig:
    if robot_type == "unitree_g1":
        return UnitreeG129DOF_CSVConfig()
    if robot_type == "agile_one":
        return AgileOneNoHandsCSVConfig()

    raise ValueError(f"[ERROR]: Unknown robot CSV config type: {robot_type}")


def load_csv(
    file_path: str,
    fps: float = 120.0,
    csv_config: RobotCSVConfig = UnitreeG129DOF_CSVConfig(),
    robot_name: str | None = None,
) -> CSVAnimationBuffer:
    """
    Load a robot motion CSV file into a ``CSVAnimationBuffer``.
    Args:
        file_path (str): Path to the CSV file to load.
        fps (float, optional): Frames per second for the animation. Defaults to 120.0.
        csv_config (RobotCSVConfig, optional): Configuration object that defines how to parse
            CSV rows into animation frames. Defaults to ``UnitreeG129DOF_CSVConfig``.
    Returns:
        CSVAnimationBuffer: An animation buffer containing the loaded and converted animation data.
    Raises:
        FileNotFoundError: If the CSV file at file_path does not exist.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"[INFO]: Loading CSV [{file_path}] for robot [{robot_name or csv_config.name}]")
        csv_data = np.loadtxt(f, delimiter=",", skiprows=1)
        num_frames = csv_data.shape[0]

        # Each anim row is derived by config, so infer size from first row
        first_row_anim = csv_config.to_anim_frame(csv_data[0])
        anim_data = np.zeros((num_frames, first_row_anim.shape[0]), dtype=np.float32)
        anim_data[0, :] = first_row_anim

        for i in range(1, num_frames):
            anim_data[i, :] = csv_config.to_anim_frame(csv_data[i])

        return CSVAnimationBuffer.create_from_raw_data(anim_data, fps)


def save_csv(file_path: str, buffer: CSVAnimationBuffer, csv_config: RobotCSVConfig = UnitreeG129DOF_CSVConfig()) -> None:
    """
    Save a ``CSVAnimationBuffer`` to a robot motion CSV file.

    Args:
        file_path (str): The path where the CSV file will be saved.
        buffer (CSVAnimationBuffer): The animation buffer containing frame data to be saved.
        csv_config (RobotCSVConfig, optional): Configuration object that defines CSV format and headers.
            Defaults to ``UnitreeG129DOF_CSVConfig``.

    Raises:
        RuntimeError: If the buffer is empty or invalid.
        OSError: If the file cannot be opened or written.
    """
    if buffer is None or buffer.num_frames == 0:
        raise RuntimeError("[ERROR]: Empty or invalid buffer.")

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_config.csv_header)

        for i in range(buffer.num_frames):
            data = buffer.get_data(i)
            row = csv_config.to_csv_row(i, data)
            writer.writerow(row)
