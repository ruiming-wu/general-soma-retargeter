#!/usr/bin/env python3
"""Export seed-retargeted G1 robot_motionlib data for OmniH2O IsaacGym.

The seed files are SONIC/joblib PKLs with ``root_trans_offset``, ``root_rot``,
``pose_aa`` and ``dof``.  This script writes one OmniH2O-compatible PHC motion
PKL per source motion while preserving a directory-style motion library.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np


G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "motion"


def _iter_source_files(input_root: Path, limit: int | None) -> list[Path]:
    files = sorted(input_root.rglob("*.pkl"))
    if limit is not None and limit >= 0:
        files = files[:limit]
    return files


def _as_float32(data, field: str, motion_name: str) -> np.ndarray:
    if field not in data:
        raise KeyError(f"{motion_name}: missing required field {field}")
    return np.asarray(data[field], dtype=np.float32)


def _convert_motion(motion_name: str, motion_data: dict) -> dict:
    root_trans = _as_float32(motion_data, "root_trans_offset", motion_name)
    pose_aa = _as_float32(motion_data, "pose_aa", motion_name)
    dof = _as_float32(motion_data, "dof", motion_name)
    root_rot = _as_float32(motion_data, "root_rot", motion_name)

    if pose_aa.ndim != 3 or pose_aa.shape[1:] != (30, 3):
        raise ValueError(f"{motion_name}: OmniH2O G1 expects pose_aa shape (T, 30, 3), got {pose_aa.shape}")
    if dof.ndim != 2 or dof.shape[1] < 29:
        raise ValueError(f"{motion_name}: OmniH2O G1 expects at least 29 DoFs, got {dof.shape}")

    smpl_joints = np.asarray(
        motion_data.get("smpl_joints", np.zeros((pose_aa.shape[0], 24, 3), dtype=np.float32)),
        dtype=np.float32,
    )
    return {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa,
        "dof": dof[:, :29],
        "root_rot": root_rot,
        "smpl_joints": smpl_joints,
        "fps": int(round(float(motion_data.get("fps", 30)))),
    }


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding="utf-8")


def _write_raw(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _install_omnih2o_g1_configs(omnih2o_repo: Path, twist2_g1_asset_dir: Path, motion_file: Path) -> None:
    g1_asset_dst = omnih2o_repo / "resources/robots/g1"
    shutil.copytree(twist2_g1_asset_dir, g1_asset_dst, dirs_exist_ok=True)

    cfg_root = omnih2o_repo / "legged_gym/legged_gym/cfg"

    _write(
        cfg_root / "config_g1_teleop.yaml",
        """
        defaults:
          - _self_
          - asset: asset_g1_teleop
          - commands: commands_teleop
          - control: control_g1_teleop
          - domain_rand: domain_rand_g1_teleop
          - env: env_g1_teleop
          - init_state: init_state_g1_teleop
          - motion: motion_g1_teleop
          - noise: noise_teleop
          - normalization: normalization_teleop
          - train: ppo_teleop
          - rewards: rewards_g1_teleop_omnih2o_teacher
          - sim: sim_teleop
          - terrain: terrain_teleop
          - viewer: viewer_base

        project_name: "G1"
        notes: "Seed G1 motions through OmniH2O teleop task"
        exp_name: &exp_name g1_seed_teleop
        headless: True
        seed: 1
        no_log: False
        test: False
        sim_device: "cuda:0"
        rl_device: "cuda:0"
        sim_device_id: 0
        metadata: false
        play: ${test}
        train: True
        im_dump: False
        task: "h1:teleop"
        load_run: ""
        num_envs: 4096
        checkpoint: 0

        joystick: False
        tmp_freeze_upper: False
        max_iterations: 1000000
        horovod: False
        resume: False
        experiment_name: null
        run_name: null
        compute_device_id: 0
        graphics_device_id: 0
        flex: False

        use_gpu: True
        use_gpu_pipeline: True
        subscenes: 0
        slices: 0
        num_threads: 0

        server_mode: False
        no_virtual_display: False
        render_o3d: False
        debug: False
        follow: False
        add_proj: False
        real_traj: False

        hydra:
          job:
            name: ${exp_name}
            env_set:
              OMP_NUM_THREADS: 1
          run:
            dir: output/g1/${exp_name}

        use_wandb: True
        train_velocity_estimation: False
        use_velocity_estimation: False
        """,
    )

    _write(
        cfg_root / "asset/asset_g1_teleop.yaml",
        """
        defaults:
          - asset_base

        file : 'resources/robots/g1/g1_custom_collision_29dof.urdf'
        name : "g1"
        foot_name : "ankle_roll_link"
        penalize_contacts_on : []
        terminate_after_contacts_on : ["pelvis", "shoulder", "hip", "knee"]
        base_orientation_body_name : "torso_link"
        self_collisions : 1
        replace_cylinder_with_capsule : True
        flip_visual_attachments : False

        density : 0.001
        angular_damping : 0.
        linear_damping : 0.
        set_dof_properties : True
        default_dof_prop_damping : []
        default_dof_prop_stiffness : []
        default_dof_prop_friction : []
        max_angular_velocity : 1000.
        max_linear_velocity : 1000.
        armature : 0.
        thickness : 0.01

        terminate_by_knee_distance : False
        terminate_by_lin_vel : False
        terminate_by_ang_vel : False
        terminate_by_gravity : True
        terminate_by_low_height : False
        terminate_by_ref_motion_distance : True
        terminate_by_1time_motion : True

        local_upper_reward : False
        zero_out_far: False
        zero_out_far_change_obs: False
        close_distance : 1.0
        far_distance : 1.0

        termination_scales:
            base_height : 0.3
            base_vel : 10.0
            base_ang_vel : 5.0
            gravity_x : 0.7
            gravity_y : 0.7
            min_knee_distance : 0.
            max_ref_motion_distance : 5.0

        clip_motion_goal: True
        clip_motion_goal_distance: 1.0
        """,
    )

    _write(
        cfg_root / "env/env_g1_teleop.yaml",
        """
        defaults:
          - env_base

        num_envs : 4096
        num_observations : 168
        num_privileged_obs : 275
        num_actions : 29
        im_eval : False
        add_short_history: False
        short_history_length: 5
        """,
    )

    _write(
        cfg_root / "motion/motion_g1_teleop.yaml",
        f"""
        teleop : True
        visualize : False
        recycle_motion : True
        terrain_level_down_distance : 0.5
        num_markers : 29

        motion_file : '{motion_file}'
        skeleton_file : 'resources/robots/g1/g1_29dof_rev_1_0.xml'
        marker_file : 'resources/objects/Marker/traj_marker.urdf'
        num_dof_pos_reference : 29
        num_dof_vel_reference : 29

        extend_hand: True
        extend_hand_parent_names: ['left_wrist_yaw_link', 'right_wrist_yaw_link']
        extend_hand_offsets: [[0.15, 0, 0], [0.15, 0, 0]]
        extend_head: False

        future_tracks: False
        num_traj_samples: 1
        traj_sample_timestep_inv: 50

        curriculum : False
        obs_noise_by_curriculum: False
        push_robot_by_curriculum: False
        kpkd_by_curriculum: False
        rfi_by_curriculum: False
        teleop_level_up_episode_length : 100
        teleop_level_down_episode_length : 30

        teleop_obs_version : 'v-teleop-extend-max'
        teleop_selected_keypoints_names : ['left_ankle_roll_link', 'right_ankle_roll_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_elbow_link', 'right_elbow_link']

        resample_motions_for_envs : True
        resample_motions_for_envs_interval_s : 1000

        visualize_config:
            customize_color : False
            marker_joint_colors : []

        realtime_vr_keypoints : False
        """,
    )

    _write(
        cfg_root / "control/control_g1_teleop.yaml",
        """
        control_type : 'P'
        stiffness :
          hip_yaw: 100
          hip_roll: 100
          hip_pitch: 100
          knee: 150
          ankle: 40
          waist: 150
          shoulder: 40
          elbow: 40
          wrist: 40
        damping :
          hip_yaw: 2
          hip_roll: 2
          hip_pitch: 2
          knee: 4
          ankle: 2
          waist: 4
          shoulder: 5
          elbow: 5
          wrist: 5
        action_scale : 0.5
        decimation : 4
        action_filt : False
        action_cutfreq : 4.0
        """,
    )

    default_angles = {
        "left_hip_pitch_joint": -0.2,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.4,
        "left_ankle_pitch_joint": -0.2,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.2,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.4,
        "right_ankle_pitch_joint": -0.2,
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.4,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 1.2,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.4,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 1.2,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }
    angles_yaml = "\n".join(f"    {name} : {value}" for name, value in default_angles.items())
    _write_raw(
        cfg_root / "init_state/init_state_g1_teleop.yaml",
        (
            "pos : [0.0, 0.0, 1.0]\n"
            "rot : [0.0, 0.0, 0.0, 1.0]\n"
            "lin_vel : [0.0, 0.0, 0.0]\n"
            "ang_vel : [0.0, 0.0, 0.0]\n"
            "max_linvel : 0.5\n"
            "max_angvel : 0.5\n"
            "default_joint_angles :\n"
            f"{angles_yaml}\n"
        ),
    )

    _write(
        cfg_root / "domain_rand/domain_rand_g1_teleop.yaml",
        """
        defaults:
          - domain_rand_base

        push_robots : True
        push_interval_s : 5
        max_push_vel_xy : 1.0
        randomize_friction : True
        friction_range : [-0.6, 1.2]
        randomize_base_mass : False
        added_mass_range : [-5., 10.]

        randomize_base_com : True
        base_com_range:
            x : [-0.1, 0.1]
            y : [-0.1, 0.1]
            z : [-0.1, 0.1]

        randomize_link_mass : True
        link_mass_range : [0.7, 1.3]
        randomize_link_body_names : [
            'pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
            'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'torso_link',
        ]

        randomize_pd_gain : True
        kp_range : [0.75, 1.25]
        kd_range : [0.75, 1.25]
        randomize_torque_rfi : True
        rfi_lim : 0.1
        randomize_rfi_lim : True
        rfi_lim_range : [0.5, 1.5]
        randomize_ctrl_delay : True
        ctrl_delay_step_range : [0, 3]

        randomize_motion_ref_xyz: True
        motion_ref_xyz_range : [[-0.02, 0.02],[-0.02, 0.02],[-0.1, 0.1]]
        motion_package_loss: False
        package_loss_range: [1, 10]
        package_loss_interval_s : 2

        born_offset : False
        born_offset_curriculum: False
        born_offset_level_down_threshold: 50
        born_offset_level_up_threshold: 120
        level_degree: 0.00005
        born_distance : 0.25
        born_offset_range: [0.0, 1]
        born_offset_possibility : 1.0
        born_heading_curriculum: False
        born_heading_randomization : False
        born_heading_level_down_threshold: 50
        born_heading_level_up_threshold: 120
        born_heading_degree: 10
        born_heading_range: [0, 180]
        born_heading_level_degree: 0.00005
        """,
    )

    joint_weights = "\n".join(f"  {name}: 1.0" for name in G1_JOINT_NAMES)
    rewards_prefix = dedent(
        """
        scales:
          torques : -0.0001
          torque_limits : -2.0
          dof_acc : -0.000011
          dof_vel : -0.004
          lower_action_rate : -3.0
          upper_action_rate : -0.625
          dof_pos_limits : -125.0
          dof_vel_limits : -50.0
          termination : -250.0
          feet_contact_forces : -0.75
          stumble : -1250.0
          feet_air_time_teleop : 1000.0
          slippage : -37.5
          feet_ori : -62.5
          in_the_air: -200.0
          stable_lower_when_vrclose: 0.0
          stable_lower_when_vrclose_positive: 0.0
          orientation : -200.0
          feet_height : 0.0
          feet_max_height_for_this_air : -2500.0
          closing: 0.0
          teleop_selected_joint_position : 32.0
          teleop_selected_joint_vel : 16.0
          teleop_body_position : 0.0
          teleop_body_position_extend : 30.0
          teleop_body_position_extend_small_sigma : 0.0
          teleop_body_position_extend_upper: 0.0
          teleop_body_position_vr_3keypoints : 50.0
          teleop_body_rotation : 20.0
          teleop_body_vel : 8.0
          teleop_body_ang_vel : 8.0

        desired_feet_max_height_for_this_air : 0.25
        feet_height_target: 0.2
        vrclose_threshold: 0.10
        ref_stable_velocity_threshold: 0.05
        only_positive_rewards : False
        tracking_sigma : 0.25
        soft_dof_pos_limit : 0.85
        soft_dof_vel_limit : 0.85
        soft_torque_limit : 0.85
        max_contact_force : 500.
        base_height_target : 1.
        body_pos_sigma : 0.5
        body_rot_sigma : 1.
        body_vel_sigma : 1.
        body_ang_vel_sigma : 1.
        joint_pos_sigma : 1.
        joint_vel_sigma : 1.
        max_penalty_compared_to_positive : False
        max_penalty_compared_to_positive_coef : 0.5
        scaling_down_body_pos_sigma : True
        teleop_body_pos_sigma_scaling_down_coef : 0.999
        teleop_joint_pos_sigma : 0.5
        teleop_joint_vel_sigma : 10
        teleop_body_pos_lowerbody_sigma : 0.5
        teleop_body_pos_0dot5sigma : 0.5
        teleop_body_pos_upperbody_sigma : 0.03
        teleop_body_pos_vr_3keypoints_sigma : 0.03
        teleop_body_pos_lowerbody_weight : 0.5
        teleop_body_pos_upperbody_weight : 1.0
        teleop_body_rot_sigma : 0.1
        teleop_body_vel_sigma : 10
        teleop_body_ang_vel_sigma : 10
        teleop_body_rot_selection : ['pelvis']
        teleop_body_vel_selection : ['pelvis']
        teleop_body_pos_selection : ['pelvis']
        teleop_body_ang_vel_selection : ['pelvis']
        teleop_joint_pos_selection :
        """
    ).lstrip()
    rewards_suffix = dedent(
        """
        sigma_curriculum: False
        num_compute_average_epl : 10000
        teleop_body_pos_upperbody_sigma_range: [0.02, 1.0]
        reward_position_sigma_level_up_threshold: 50
        reward_position_sigma_level_down_threshold: 120
        penalty_curriculum: False
        penalty_scale : 1.0
        penalty_scale_range: [0.25, 1.0]
        penalty_level_down_threshold: 50
        penalty_level_up_threshold: 120
        level_degree: 0.00001
        penalty_reward_names : [
          "torques", "torque_limits", "dof_acc", "dof_vel", "lower_action_rate",
          "upper_action_rate", "dof_pos_limits", "termination", "feet_contact_forces",
          "stumble", "feet_air_time_teleop", "slippage", "feet_ori", "orientation",
          "in_the_air", "stable_lower_when_vrclose"
        ]
        """
    ).lstrip()
    _write_raw(
        cfg_root / "rewards/rewards_g1_teleop_omnih2o_teacher.yaml",
        f"{rewards_prefix}{joint_weights}\n{rewards_suffix}",
    )


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
        default=Path("/home/ruiming.wu/data/seed-retargeted/g1_motionlib/omnih2o_motionlib/robot_motionlib"),
    )
    parser.add_argument(
        "--config-motion-file",
        type=Path,
        default=None,
        help="Motion directory path written into generated OmniH2O configs. Defaults to --output-root.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-install-omnih2o-g1", action="store_true")
    parser.add_argument("--omnih2o-repo", type=Path, default=Path("/home/ruiming.wu/codes/human2humanoid"))
    parser.add_argument("--twist2-g1-asset-dir", type=Path, default=Path("/home/ruiming.wu/codes/TWIST2/assets/g1"))
    args = parser.parse_args()

    source_files = _iter_source_files(args.input_root, args.limit)
    written = 0
    total_frames = 0
    motion_files = []
    for src in source_files:
        loaded = joblib.load(src)
        if not isinstance(loaded, dict):
            raise TypeError(f"{src} did not contain a motion dictionary")
        rel_dir = src.parent.relative_to(args.input_root)
        for key, motion_data in loaded.items():
            motion_key = _safe_name(str(key))
            motion = _convert_motion(motion_key, motion_data)
            out_file = args.output_root / rel_dir / f"{motion_key}.pkl"
            if out_file.exists() and not args.overwrite:
                motion_files.append(str(out_file.relative_to(args.output_root)))
                total_frames += motion["root_trans_offset"].shape[0]
                continue

            out_file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(motion, out_file, compress=3)
            written += 1
            motion_files.append(str(out_file.relative_to(args.output_root)))
            total_frames += motion["root_trans_offset"].shape[0]

    metadata = {
        "source_root": str(args.input_root),
        "output_root": str(args.output_root),
        "num_source_files": len(source_files),
        "num_motions": len(motion_files),
        "num_written": written,
        "total_frames": total_frames,
        "format": "OmniH2O PHC MotionLibH1 G1 directory of one-motion joblib PKLs",
        "g1_joint_names": G1_JOINT_NAMES,
        "motion_files": motion_files,
    }
    metadata_path = args.output_root.parent / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.no_install_omnih2o_g1:
        _install_omnih2o_g1_configs(args.omnih2o_repo, args.twist2_g1_asset_dir, args.config_motion_file or args.output_root)

    print(f"Exported {len(motion_files)} motions from {len(source_files)} files")
    print(f"Written this run: {written}")
    print(f"Total frames: {total_frames}")
    print(f"Output root: {args.output_root}")
    print(f"Metadata: {metadata_path}")
    if not args.no_install_omnih2o_g1:
        print(f"Installed G1 OmniH2O configs under: {args.omnih2o_repo / 'legged_gym/legged_gym/cfg'}")


if __name__ == "__main__":
    main()
