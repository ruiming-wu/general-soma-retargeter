# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive source-to-robot offset calibration tool.

This app is intentionally separate from ``bvh_to_csv_converter.py``. It uses
the existing Newton/OpenGL/ImGui stack so SOMA's USD skinned mesh can be reused
directly when available while the robot is loaded from an MJCF file. For
non-SOMA BVH skeletons, pass ``--source_mesh none`` to run in skeleton-only mode.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from datetime import datetime
from types import MethodType

import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.usd as usd_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.utils.pose_utils as pose_utils
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.renderers.mesh_renderer import SkeletalMeshRenderer
from soma_retargeter.renderers.skeleton_renderer import SkeletonRenderer
from soma_retargeter.utils.newton_utils import get_name_from_label
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SOMA_USD = _REPO_ROOT / "soma_retargeter/configs/soma/soma_base_skel_minimal.usd"
_DEFAULT_SOMA_BVH = _REPO_ROOT / "soma_retargeter/configs/soma/soma_zero_frame0.bvh"
_DEFAULT_G1_MJCF = Path("/home/ruiming.wu/codes/GMR-PH/assets/unitree_g1/g1_mocap_29dof.xml")
_DEFAULT_EXPORT_DIR = _REPO_ROOT / "output/offset_calibrator_exports"
_ROBOT_CONFIG_DEFAULTS = {
    "unitree_g1": (
        "unitree_g1/soma_to_g1_retargeter_config.json",
        "unitree_g1/soma_to_g1_scaler_config.json",
    ),
    "agile_one": (
        "agile_one/soma_to_agile_one_retargeter_config.json",
        "agile_one/soma_to_agile_one_scaler_config.json",
    ),
}

_PANEL_MARGIN = 10
_SIDE_PANEL_WIDTH = 390
_BOTTOM_PANEL_HEIGHT = 125
_PANEL_ALPHA = 0.92
_SOMA_COLOR = (235.0 / 255.0, 245.0 / 255.0, 112.0 / 255.0)
_PREVIEW_COLOR = (116.0 / 255.0, 235.0 / 255.0, 210.0 / 255.0)
_AXIS_SCALE = 0.08
_AXIS_THICKNESS = 0.006
_POS_WARN_M = 0.05
_ROT_WARN_DEG = 15.0
_AXIS_COLORS = (
    (1.0, 0.05, 0.05),
    (0.05, 1.0, 0.05),
    (0.1, 0.35, 1.0),
)


class SomaRobotOffsetCalibrator:
    """Display a source skeleton, robot, and calibrated offset preview for manual calibration."""

    def __init__(self, viewer, args: argparse.Namespace):
        if isinstance(viewer, newton.viewer.ViewerNull):
            raise ValueError("soma_robot_offset_calibrator requires an interactive viewer, e.g. --viewer gl")

        self.viewer = viewer
        self.viewer.vsync = True
        self.source_name = args.source_name
        self.source_mesh_mode = args.source_mesh
        self.viewer.renderer.set_title(f"{self.source_name} Robot Offset Calibrator")
        self._suppress_default_viewer_ui()
        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")

        self.args = args
        self.initial_offset_x = float(args.initial_offset_x)
        self.overlap_x = 0.0
        self.soma_root_offset = np.zeros(3, dtype=np.float32)
        self.show_soma_mesh = self.source_mesh_mode != "none"
        self.show_soma_skeleton = self.source_mesh_mode == "none"
        self.show_preview_mesh = self.source_mesh_mode != "none"
        self.show_preview_skeleton = True
        self.show_preview_mapped_axes = True
        self.show_preview_mapped_names = False
        self.show_robot_mesh = True
        self.show_gizmos = True
        self.time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.last_export_path = ""
        self.last_export_status = ""
        self.last_pose_pair_path = ""
        self.last_pose_pair_status = ""

        self.converter = SpaceConverter(FacingDirectionType.MUJOCO)
        self._load_soma(Path(args.soma_bvh), Path(args.soma_usd))
        self._load_calibration_configs(args)
        self._load_robot(self._resolve_robot_mjcf(args))
        self._apply_initial_robot_joint_positions()
        self._validate_ik_map_bodies()
        self.preview_report = self._compute_preview_report()

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets([0, 0, 0])
        self._update_scene()

    def _suppress_default_viewer_ui(self) -> None:
        """Keep ImGui enabled while removing Newton's built-in side/stats panels."""
        self.viewer._render_left_panel = MethodType(lambda _viewer: None, self.viewer)
        self.viewer._render_stats_overlay = MethodType(lambda _viewer: None, self.viewer)

    def _load_soma(self, soma_bvh: Path, soma_usd: Path) -> None:
        self.soma_bvh = soma_bvh.expanduser().resolve()
        self.soma_usd = soma_usd.expanduser().resolve()
        self.skeleton, animation = bvh_utils.load_bvh(str(self.soma_bvh))
        self.soma_reference_local = np.copy(animation.get_local_transforms(0))
        self.soma_current_local = np.copy(self.soma_reference_local)
        self.soma_euler_deg = np.zeros((self.skeleton.num_joints, 3), dtype=np.float32)
        self.soma_show_names = np.zeros(self.skeleton.num_joints, dtype=bool)
        self.soma_show_axes = np.zeros(self.skeleton.num_joints, dtype=bool)
        self.soma_instance = SkeletonInstance(
            self.skeleton,
            _SOMA_COLOR,
            self._soma_world_transform(),
        )
        self.soma_instance.set_local_transforms(self.soma_current_local)
        self.soma_mesh = None
        self.soma_mesh_renderer = None
        if self.source_mesh_mode == "soma":
            self.soma_mesh = usd_utils.load_skeletal_mesh_from_usd(
                str(self.soma_usd),
                self.skeleton,
                "/OUTPUT/c_geometry_grp",
                "/OUTPUT/c_skeleton_grp/Root",
            )
            self.soma_mesh_renderer = SkeletalMeshRenderer(self.soma_mesh)
        self.soma_skeleton_renderer = SkeletonRenderer(self.skeleton, [0])
        self.preview_instance = SkeletonInstance(
            self.skeleton,
            _PREVIEW_COLOR,
            self._soma_world_transform(),
        )
        self.preview_current_local = np.copy(self.soma_current_local)
        self.preview_instance.set_local_transforms(self.preview_current_local)
        self.preview_mesh_renderer = SkeletalMeshRenderer(self.soma_mesh) if self.soma_mesh is not None else None
        self.preview_skeleton_renderer = SkeletonRenderer(self.skeleton, [0])

    def _load_soma_pose(self, soma_bvh: Path) -> None:
        """Load another single-frame source BVH without rebuilding the robot."""
        if getattr(self, "soma_mesh_renderer", None) is not None:
            self.soma_mesh_renderer.clear(self.viewer)
        if hasattr(self, "soma_skeleton_renderer"):
            self.soma_skeleton_renderer.clear(self.viewer)
        if getattr(self, "preview_mesh_renderer", None) is not None:
            self.preview_mesh_renderer.clear(self.viewer)
        if hasattr(self, "preview_skeleton_renderer"):
            self.preview_skeleton_renderer.clear(self.viewer)
        self._load_soma(soma_bvh, self.soma_usd)
        self._load_calibration_configs(self.args)
        self.preview_report = self._compute_preview_report()
        self._update_scene()

    def _load_robot(self, robot_mjcf: Path) -> None:
        self.robot_mjcf = robot_mjcf.expanduser().resolve()
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_mjcf(self.robot_mjcf)
        self.model = builder.finalize()
        self.state = self.model.state()
        self.robot_q_default = self.model.joint_q.numpy().astype(np.float32, copy=True)
        self.robot_q = self.robot_q_default.copy()
        self.robot_body_names = [get_name_from_label(label) for label in self.model.body_label]
        self.robot_body_name_to_idx = {name: i for i, name in enumerate(self.robot_body_names)}
        self.robot_joint_controls = self._build_robot_joint_controls()
        self.robot_pelvis_control = self._build_robot_pelvis_control()
        self.robot_root_euler_deg = R.from_quat(self.robot_q[3:7]).as_euler("xyz", degrees=True).astype(np.float32)

    def _load_calibration_configs(self, args: argparse.Namespace) -> None:
        self.robot_type = args.robot_type or (self._infer_robot_type(args.robot_mjcf) if args.robot_mjcf else "unitree_g1")
        self.export_dir = Path(args.export_dir).expanduser().resolve()

        default_retargeter, default_scaler = _ROBOT_CONFIG_DEFAULTS[self.robot_type]
        self.base_retargeter_path = self._resolve_config_path(args.base_retargeter_config or default_retargeter)
        self.base_retargeter_config = self._load_json(self.base_retargeter_path)

        scaler_config = args.base_scaler_config or self.base_retargeter_config.get("human_robot_scaler_config") or default_scaler
        self.base_scaler_path = self._resolve_config_path(scaler_config)
        self.base_scaler_config = self._load_json(self.base_scaler_path)

        self.preview_joint_scales = {
            name: float(value) for name, value in self.base_scaler_config.get("joint_scales", {}).items()
        }
        self.preview_joint_offsets_t = {}
        self.preview_joint_offsets_rpy = {}
        self.preview_show_names = {}
        self.preview_show_axes = {}
        for name, entry in self.base_scaler_config.get("joint_offsets", {}).items():
            t_offset, q_offset = entry
            self.preview_joint_offsets_t[name] = np.asarray(t_offset, dtype=np.float32)
            self.preview_joint_offsets_rpy[name] = R.from_quat(q_offset).as_euler("xyz", degrees=True).astype(np.float32)
            self.preview_show_names[name] = False
            self.preview_show_axes[name] = False

        self.mapped_joint_names = [
            name for name in self.base_retargeter_config.get("ik_map", {}).keys()
            if self.skeleton.joint_index(name) != -1
        ]
        self.preview_joint_names = [
            name for name in self.skeleton.joint_names
            if name in self.preview_joint_scales and name in self.preview_joint_offsets_t
        ]
        for name in self.mapped_joint_names:
            if name in self.preview_show_axes:
                self.preview_show_axes[name] = bool(self.show_preview_mapped_axes)
            if name in self.preview_show_names:
                self.preview_show_names[name] = bool(self.show_preview_mapped_names)

    def _resolve_robot_mjcf(self, args: argparse.Namespace) -> Path:
        if args.robot_mjcf:
            return Path(args.robot_mjcf)
        if self.robot_type == "agile_one" and self.base_retargeter_config.get("robot_mjcf_path"):
            return Path(self.base_retargeter_config["robot_mjcf_path"])
        if self.robot_type == "unitree_g1":
            return _DEFAULT_G1_MJCF
        raise ValueError(f"No default MJCF is configured for robot_type [{self.robot_type}]. Pass --robot_mjcf explicitly.")

    def _reset_preview_config(self) -> None:
        self.preview_joint_scales = {
            name: float(value) for name, value in self.base_scaler_config.get("joint_scales", {}).items()
        }
        self.preview_joint_offsets_t = {}
        self.preview_joint_offsets_rpy = {}
        for name, entry in self.base_scaler_config.get("joint_offsets", {}).items():
            t_offset, q_offset = entry
            self.preview_joint_offsets_t[name] = np.asarray(t_offset, dtype=np.float32)
            self.preview_joint_offsets_rpy[name] = R.from_quat(q_offset).as_euler("xyz", degrees=True).astype(np.float32)
        self.preview_report = self._compute_preview_report()

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            f.write("\n")

    @staticmethod
    def _infer_robot_type(robot_mjcf: str) -> str:
        lowered = robot_mjcf.lower()
        if "g1" in lowered or "unitree" in lowered:
            return "unitree_g1"
        if "agile" in lowered or "h4" in lowered:
            return "agile_one"
        return "agile_one"

    @staticmethod
    def _resolve_config_path(path_like: str | Path) -> Path:
        path = Path(path_like).expanduser()
        if path.is_absolute():
            return path.resolve()
        repo_relative = (_REPO_ROOT / path).resolve()
        if repo_relative.exists():
            return repo_relative
        if len(path.parts) == 1:
            return io_utils.get_config_file(str(path)).resolve()
        return io_utils.get_config_file(*path.parts).resolve()

    def _validate_ik_map_bodies(self) -> None:
        missing = []
        for joint_name, mapping in self.base_retargeter_config.get("ik_map", {}).items():
            for key in ("t_body", "r_body"):
                if mapping[key] not in self.robot_body_name_to_idx:
                    missing.append(f"{joint_name}.{key}={mapping[key]}")
        if missing:
            raise ValueError("IK map references robot bodies that are not in the loaded MJCF: " + ", ".join(missing))

    def _build_robot_joint_controls(self) -> list[dict]:
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        q_dim = np.diff(q_start)
        joint_type = self.model.joint_type.numpy()
        joint_child = self.model.joint_child.numpy()
        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        controls = []

        for joint_id, label in enumerate(self.model.joint_label):
            if q_dim[joint_id] != 1:
                continue

            q_index = int(q_start[joint_id])
            dof_index = int(qd_start[joint_id])
            name = get_name_from_label(label)
            is_revolute = int(joint_type[joint_id]) == 1
            if is_revolute:
                lo = float(lower[dof_index])
                hi = float(upper[dof_index])
                if not np.isfinite(lo) or not np.isfinite(hi) or abs(lo) > 1e9 or abs(hi) > 1e9:
                    min_value, max_value = -180.0, 180.0
                else:
                    min_value, max_value = np.degrees([lo, hi]).astype(float)
                value = float(np.degrees(self.robot_q[q_index]))
                unit = "deg"
            else:
                lo = float(lower[dof_index])
                hi = float(upper[dof_index])
                if not np.isfinite(lo) or not np.isfinite(hi) or abs(lo) > 1e9 or abs(hi) > 1e9:
                    min_value, max_value = -1.0, 1.0
                else:
                    min_value, max_value = lo, hi
                value = float(self.robot_q[q_index])
                unit = "q"

            controls.append(
                {
                    "name": name,
                    "joint_id": joint_id,
                    "child_body_id": int(joint_child[joint_id]),
                    "q_index": q_index,
                    "is_revolute": is_revolute,
                    "min": float(min_value),
                    "max": float(max_value),
                    "value": value,
                    "unit": unit,
                    "show_name": False,
                    "show_axis": False,
                }
            )
        return controls

    def _build_robot_pelvis_control(self) -> dict | None:
        for body_id, name in enumerate(self.robot_body_names):
            if name.lower() in {"pelvis", "pelvis_link"}:
                return {"name": name, "body_id": body_id, "show_name": True, "show_axis": True}
        for body_id, name in enumerate(self.robot_body_names):
            if "pelvis" in name.lower():
                return {"name": name, "body_id": body_id, "show_name": True, "show_axis": True}
        return None

    def _apply_initial_robot_joint_positions(self) -> None:
        initial = self.base_retargeter_config.get("initial_robot_joint_positions")
        if not isinstance(initial, dict):
            return
        if "root_position" in initial:
            self.robot_q[:3] = np.asarray(initial["root_position"], dtype=np.float32)
        if "root_rotation_xyzw" in initial:
            q = np.asarray(initial["root_rotation_xyzw"], dtype=np.float32)
            q = q / max(float(np.linalg.norm(q)), 1e-12)
            self.robot_q[3:7] = q
            self.robot_root_euler_deg = R.from_quat(q).as_euler("xyz", degrees=True).astype(np.float32)
        joints = initial.get("joints", {})
        if isinstance(joints, dict):
            control_by_name = {control["name"]: control for control in self.robot_joint_controls}
            for joint_name, value in joints.items():
                control = control_by_name.get(joint_name)
                if control is None:
                    continue
                q_value = float(value)
                self.robot_q[control["q_index"]] = q_value
                control["value"] = float(np.degrees(q_value)) if control["is_revolute"] else q_value

    def _soma_world_transform(self):
        # overlap_x == 0 keeps the requested default separation.
        # overlap_x == initial_offset_x moves SOMA onto the robot.
        x = self.initial_offset_x - self.overlap_x + float(self.soma_root_offset[0])
        return wp.transform(
            wp.vec3(x, float(self.soma_root_offset[1]), float(self.soma_root_offset[2])),
            self.converter.converter,
        )

    def _set_soma_joint_euler(self, joint_index: int, euler_deg: np.ndarray) -> None:
        reference = self.soma_reference_local[joint_index]
        reference_rot = R.from_quat(reference[3:7])
        offset_rot = R.from_euler("xyz", euler_deg, degrees=True)
        quat_xyzw = (reference_rot * offset_rot).as_quat().astype(np.float32)
        self.soma_current_local[joint_index, :3] = reference[:3]
        self.soma_current_local[joint_index, 3:7] = quat_xyzw

    def _update_scene(self) -> None:
        self.soma_instance.xform = self._soma_world_transform()
        self.soma_instance.set_local_transforms(self.soma_current_local)
        wp.copy(self.model.joint_q, wp.array(self.robot_q, dtype=wp.float32), 0, 0, len(self.robot_q))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)
        if hasattr(self, "preview_joint_names"):
            self._sync_preview_avatar()
            self.preview_report = self._compute_preview_report()

    def _set_robot_root_euler(self, euler_deg: np.ndarray) -> None:
        q = R.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float32)
        self.robot_q[3:7] = q / max(float(np.linalg.norm(q)), 1e-12)

    def _soma_global_transforms_np(self) -> np.ndarray:
        global_tx = self.soma_instance.compute_global_transforms()
        return wp.array(global_tx, dtype=wp.transform).numpy()

    def _robot_body_transforms_np(self) -> np.ndarray:
        return self.state.body_q.numpy()

    def _height_ratio(self) -> float:
        human_height_assumption = float(self.base_scaler_config.get("human_height_assumption", 1.8))
        model_height = float(self.base_retargeter_config.get("model_height", human_height_assumption))
        return model_height / human_height_assumption

    def _effective_scale(self, joint_name: str) -> float:
        return float(self.preview_joint_scales[joint_name]) * self._height_ratio()

    def _preview_global_transforms_np(self) -> np.ndarray:
        return pose_utils.compute_global_pose(self.skeleton, self.preview_current_local, self.preview_instance.xform)

    def _preview_scaling_reference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        original_global = self._soma_global_transforms_np()
        root_name = self.base_scaler_config.get("human_root_name", "Hips")
        root_idx = self.skeleton.joint_index(root_name)
        if root_idx == -1:
            root_idx = self.skeleton.joint_index("Hips")
        root_t = original_global[root_idx, :3] if root_idx != -1 else original_global[0, :3]
        root_scale = self._effective_scale(root_name) if root_name in self.preview_joint_scales else 1.0
        return original_global, root_t, root_t * root_scale

    def _preview_effector_transforms(self) -> dict[str, np.ndarray]:
        """Config-space preview targets: scale + translation offset + rotation offset."""
        original_global, root_t, scaled_root_t = self._preview_scaling_reference()
        effectors: dict[str, np.ndarray] = {}
        for joint_name in self.preview_joint_names:
            joint_idx = self.skeleton.joint_index(joint_name)
            if joint_idx == -1:
                continue
            pose = original_global[joint_idx]
            pose_rot = R.from_quat(pose[3:7])
            offset_rot = R.from_euler("xyz", self.preview_joint_offsets_rpy[joint_name], degrees=True)
            target_rot = pose_rot * offset_rot
            scale = self._effective_scale(joint_name)
            target_t = (pose[:3] - root_t) * scale + scaled_root_t
            target_t = target_t + target_rot.apply(self.preview_joint_offsets_t[joint_name])
            effectors[joint_name] = self._compose_transform(
                target_t.astype(np.float32),
                target_rot.as_quat().astype(np.float32),
            )
        return effectors

    def _descendant_indices(self, joint_idx: int) -> list[int]:
        descendants = []
        for candidate in range(joint_idx, self.skeleton.num_joints):
            current = candidate
            while current != -1:
                if current == joint_idx:
                    descendants.append(candidate)
                    break
                current = int(self.skeleton.parent_indices[current])
        return descendants

    @staticmethod
    def _compose_transform(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        return np.asarray([position[0], position[1], position[2], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]], dtype=np.float32)

    @staticmethod
    def _transform_inverse(row: np.ndarray) -> np.ndarray:
        rot = R.from_quat(row[3:7])
        inv_rot = rot.inv()
        inv_t = inv_rot.apply(-row[:3])
        return SomaRobotOffsetCalibrator._compose_transform(inv_t, inv_rot.as_quat())

    @staticmethod
    def _transform_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        lhs_rot = R.from_quat(lhs[3:7])
        rhs_rot = R.from_quat(rhs[3:7])
        rot = lhs_rot * rhs_rot
        pos = lhs[:3] + lhs_rot.apply(rhs[:3])
        return SomaRobotOffsetCalibrator._compose_transform(pos, rot.as_quat())

    def _sync_preview_avatar(self) -> None:
        self.preview_instance.xform = self.soma_instance.xform
        original_global, root_t, scaled_root_t = self._preview_scaling_reference()
        preview_global = np.copy(original_global)

        for joint_name in self.preview_joint_names:
            joint_idx = self.skeleton.joint_index(joint_name)
            if joint_idx == -1:
                continue
            pose = original_global[joint_idx]
            pose_rot = R.from_quat(pose[3:7])
            scale = self._effective_scale(joint_name)
            target_t = (pose[:3] - root_t) * scale + scaled_root_t
            target_t = target_t + pose_rot.apply(self.preview_joint_offsets_t[joint_name])
            target_tx = self._compose_transform(target_t.astype(np.float32), pose[3:7].astype(np.float32))
            delta = self._transform_multiply(target_tx, self._transform_inverse(preview_global[joint_idx]))
            for descendant_idx in self._descendant_indices(joint_idx):
                preview_global[descendant_idx] = self._transform_multiply(delta, preview_global[descendant_idx])

        self.preview_current_local = pose_utils.compute_local_pose(
            self.skeleton,
            preview_global,
            self.preview_instance.xform,
        )
        self.preview_instance.set_local_transforms(self.preview_current_local)

    @staticmethod
    def _rotation_error_deg(q_a, q_b) -> float:
        return float(np.degrees((R.from_quat(q_a).inv() * R.from_quat(q_b)).magnitude()))

    def _compute_preview_report(self) -> dict:
        if not hasattr(self, "preview_current_local"):
            return {"summary": {}, "joints": []}

        preview_effectors = self._preview_effector_transforms()
        robot_body_q = self._robot_body_transforms_np()
        joints = []

        for joint_name in self.mapped_joint_names:
            mapping = self.base_retargeter_config["ik_map"][joint_name]
            if joint_name not in preview_effectors:
                continue
            t_body_idx = self.robot_body_name_to_idx[mapping["t_body"]]
            r_body_idx = self.robot_body_name_to_idx[mapping["r_body"]]
            effector_tx = preview_effectors[joint_name]
            pos_error = float(np.linalg.norm(effector_tx[:3] - robot_body_q[t_body_idx, :3]))
            rot_error = self._rotation_error_deg(effector_tx[3:7], robot_body_q[r_body_idx, 3:7])
            joints.append(
                {
                    "joint": joint_name,
                    "t_body": mapping["t_body"],
                    "r_body": mapping["r_body"],
                    "position_error_m": pos_error,
                    "rotation_error_deg": rot_error,
                    "scale": float(self.preview_joint_scales.get(joint_name, 1.0)),
                    "effective_scale": float(self._effective_scale(joint_name)) if joint_name in self.preview_joint_scales else 1.0,
                    "translation_offset": self.preview_joint_offsets_t.get(joint_name, np.zeros(3)).astype(float).tolist(),
                    "rotation_offset_rpy_deg": self.preview_joint_offsets_rpy.get(joint_name, np.zeros(3)).astype(float).tolist(),
                    "warning": pos_error > _POS_WARN_M or rot_error > _ROT_WARN_DEG,
                }
            )

        pos_errors = [item["position_error_m"] for item in joints]
        rot_errors = [item["rotation_error_deg"] for item in joints]
        worst = max(joints, key=lambda item: item["position_error_m"] + item["rotation_error_deg"] / 100.0, default=None)
        return {
            "summary": {
                "mean_position_error_m": float(np.mean(pos_errors)) if pos_errors else 0.0,
                "max_position_error_m": float(np.max(pos_errors)) if pos_errors else 0.0,
                "mean_rotation_error_deg": float(np.mean(rot_errors)) if rot_errors else 0.0,
                "max_rotation_error_deg": float(np.max(rot_errors)) if rot_errors else 0.0,
                "worst_joint": worst["joint"] if worst else "",
                "warnings": int(sum(1 for item in joints if item["warning"])),
            },
            "joints": joints,
        }

    def _candidate_scaler_config(self) -> dict:
        config = copy.deepcopy(self.base_scaler_config)
        config["robot_type"] = self.robot_type
        config["joint_scales"] = {
            name: float(self.preview_joint_scales[name])
            for name in config.get("joint_scales", {}).keys()
            if name in self.preview_joint_scales
        }
        config["joint_offsets"] = {}
        for name in self.base_scaler_config.get("joint_offsets", {}).keys():
            if name not in self.preview_joint_offsets_t:
                continue
            q_offset = R.from_euler("xyz", self.preview_joint_offsets_rpy[name], degrees=True).as_quat()
            config["joint_offsets"][name] = [
                [float(x) for x in self.preview_joint_offsets_t[name]],
                [float(x) for x in q_offset],
            ]
        return config

    def _candidate_retargeter_config(self, scaler_path: Path) -> dict:
        config = copy.deepcopy(self.base_retargeter_config)
        config["human_robot_scaler_config"] = str(scaler_path.resolve())
        if self.robot_type == "agile_one":
            config["robot_mjcf_path"] = str(self.robot_mjcf)
        return config

    def _state_snapshot(self) -> dict:
        return {
            "robot_type": self.robot_type,
            "robot_mjcf": str(self.robot_mjcf),
            "soma_bvh": str(self.soma_bvh),
            "soma_usd": str(self.soma_usd),
            "base_retargeter_config": str(self.base_retargeter_path),
            "base_scaler_config": str(self.base_scaler_path),
            "overlap_x": float(self.overlap_x),
            "initial_offset_x": float(self.initial_offset_x),
            "soma_root_offset": self.soma_root_offset.astype(float).tolist(),
            "soma_local_transforms": self.soma_current_local.astype(float).tolist(),
            "soma_global_transforms": self._soma_global_transforms_np().astype(float).tolist(),
            "preview_local_transforms": self.preview_current_local.astype(float).tolist(),
            "preview_global_transforms": self._preview_global_transforms_np().astype(float).tolist(),
            "preview_effector_transforms": {
                name: transform.astype(float).tolist()
                for name, transform in self._preview_effector_transforms().items()
            },
            "robot_joint_q": self.robot_q.astype(float).tolist(),
            "robot_body_transforms": self._robot_body_transforms_np().astype(float).tolist(),
        }

    def _robot_joint_values_by_name(self) -> dict:
        values = {}
        for control in self.robot_joint_controls:
            q_value = float(self.robot_q[control["q_index"]])
            values[control["name"]] = q_value
        return values

    def _pose_pair_sample(self) -> dict:
        self._update_scene()
        robot_body_q = self._robot_body_transforms_np()
        robot_body_transforms = {
            name: robot_body_q[idx].astype(float).tolist()
            for idx, name in enumerate(self.robot_body_names)
        }
        ik_targets = {}
        for joint_name, mapping in self.base_retargeter_config.get("ik_map", {}).items():
            t_body = mapping["t_body"]
            r_body = mapping["r_body"]
            if t_body not in robot_body_transforms or r_body not in robot_body_transforms:
                continue
            ik_targets[joint_name] = {
                "t_body": t_body,
                "r_body": r_body,
                "target_position": robot_body_transforms[t_body][:3],
                "target_rotation_xyzw": robot_body_transforms[r_body][3:7],
            }
        return {
            "schema": "soma_robot_pose_pair.v1",
            "robot_type": self.robot_type,
            "robot_mjcf": str(self.robot_mjcf),
            "soma_bvh": str(self.soma_bvh),
            "soma_usd": str(self.soma_usd),
            "base_retargeter_config": str(self.base_retargeter_path),
            "base_scaler_config": str(self.base_scaler_path),
            "soma_root_view_offset_m": self.soma_root_offset.astype(float).tolist(),
            "robot_root_position": self.robot_q[:3].astype(float).tolist(),
            "robot_root_rotation_xyzw": self.robot_q[3:7].astype(float).tolist(),
            "robot_joint_q": self.robot_q.astype(float).tolist(),
            "robot_joints": self._robot_joint_values_by_name(),
            "robot_body_transforms": robot_body_transforms,
            "ik_targets": ik_targets,
        }

    def save_pose_pair(self) -> Path:
        pose_pair_dir = self.export_dir / "pose_pairs"
        pose_pair_dir.mkdir(parents=True, exist_ok=True)
        path = pose_pair_dir / f"{self.soma_bvh.stem}__{self.robot_type}.json"
        suffix = 1
        while path.exists():
            path = pose_pair_dir / f"{self.soma_bvh.stem}__{self.robot_type}_{suffix:02d}.json"
            suffix += 1
        self._write_json(path, self._pose_pair_sample())
        self.last_pose_pair_path = str(path)
        self.last_pose_pair_status = "saved"
        return path

    def _write_report_md(self, path: Path, report: dict) -> None:
        summary = report["summary"]
        lines = [
            "# SOMA to Robot Calibration Report",
            "",
            f"- Robot type: `{self.robot_type}`",
            f"- Robot MJCF: `{self.robot_mjcf}`",
            f"- Base retargeter config: `{self.base_retargeter_path}`",
            f"- Base scaler config: `{self.base_scaler_path}`",
            f"- Mean position error: {summary.get('mean_position_error_m', 0.0):.4f} m",
            f"- Max position error: {summary.get('max_position_error_m', 0.0):.4f} m",
            f"- Mean rotation error: {summary.get('mean_rotation_error_deg', 0.0):.2f} deg",
            f"- Max rotation error: {summary.get('max_rotation_error_deg', 0.0):.2f} deg",
            f"- Worst joint: `{summary.get('worst_joint', '')}`",
            f"- Warning joints: {summary.get('warnings', 0)}",
            "",
            "| Joint | t_body | r_body | pos m | rot deg | scale | warning |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for item in report["joints"]:
            lines.append(
                f"| {item['joint']} | {item['t_body']} | {item['r_body']} | "
                f"{item['position_error_m']:.4f} | {item['rotation_error_deg']:.2f} | "
                f"{item['scale']:.4f} | {'YES' if item['warning'] else ''} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def export_calibration(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.export_dir / timestamp
        export_path.mkdir(parents=True, exist_ok=False)

        scaler_path = export_path / "candidate_scaler_config.json"
        retargeter_path = export_path / "candidate_retargeter_config.json"
        state_path = export_path / "state_snapshot.json"
        report_json_path = export_path / "calibration_report.json"
        report_md_path = export_path / "calibration_report.md"

        report = self._compute_preview_report()
        self._write_json(scaler_path, self._candidate_scaler_config())
        self._write_json(retargeter_path, self._candidate_retargeter_config(scaler_path))
        self._write_json(state_path, self._state_snapshot())
        self._write_json(report_json_path, report)
        self._write_report_md(report_md_path, report)

        self.last_export_path = str(export_path)
        self.last_export_status = "saved"
        return export_path

    @staticmethod
    def _append_thick_axes(
        starts: list[np.ndarray],
        ends: list[np.ndarray],
        colors: list[tuple[float, float, float]],
        transform_row: np.ndarray,
        scale: float = _AXIS_SCALE,
        thickness: float = _AXIS_THICKNESS,
    ) -> None:
        position = np.asarray(transform_row[:3], dtype=np.float32)
        basis = R.from_quat(transform_row[3:7]).as_matrix().astype(np.float32)
        offsets = (
            np.zeros(3, dtype=np.float32),
            basis[:, 0] * thickness,
            -basis[:, 0] * thickness,
            basis[:, 1] * thickness,
            -basis[:, 1] * thickness,
            basis[:, 2] * thickness,
            -basis[:, 2] * thickness,
        )

        for axis_index in range(3):
            direction = basis[:, axis_index] * scale
            for offset in offsets:
                starts.append(position + offset)
                ends.append(position + direction + offset)
                colors.append(_AXIS_COLORS[axis_index])

    def _log_axis_batch(self, name: str, transforms: list[np.ndarray]) -> None:
        if not transforms:
            self.viewer.log_lines(name, None, None, None)
            return

        starts: list[np.ndarray] = []
        ends: list[np.ndarray] = []
        colors: list[tuple[float, float, float]] = []
        for transform_row in transforms:
            self._append_thick_axes(starts, ends, colors, transform_row)

        self.viewer.log_lines(
            name,
            wp.array(np.asarray(starts, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(ends, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(colors, dtype=np.float32), dtype=wp.vec3),
            width=0.04,
        )

    def _render_selected_axes(self) -> None:
        soma_transforms = self._soma_global_transforms_np()
        selected_soma = [soma_transforms[i] for i, enabled in enumerate(self.soma_show_axes) if enabled]
        self._log_axis_batch("/soma_selected_joint_axes", selected_soma)

        preview_transforms = self._preview_effector_transforms()
        selected_preview = []
        for joint_name in self.preview_joint_names:
            if joint_name not in preview_transforms:
                continue
            if self.preview_show_axes.get(joint_name, False):
                selected_preview.append(preview_transforms[joint_name])
        self._log_axis_batch("/preview_selected_joint_axes", selected_preview)

        robot_transforms = self._robot_body_transforms_np()
        selected_robot = []
        for control in self.robot_joint_controls:
            body_id = control["child_body_id"]
            if control["show_axis"] and 0 <= body_id < len(robot_transforms):
                selected_robot.append(robot_transforms[body_id])
        if self.robot_pelvis_control is not None and self.robot_pelvis_control["show_axis"]:
            body_id = self.robot_pelvis_control["body_id"]
            if 0 <= body_id < len(robot_transforms):
                selected_robot.append(robot_transforms[body_id])
        self._log_axis_batch("/robot_selected_joint_axes", selected_robot)

    def _set_robot_mesh_hidden(self, hidden: bool) -> None:
        for batch in getattr(self.viewer, "_shape_instances", {}).values():
            for name in (batch.name, f"{batch.name}/capsule_cylinder", f"{batch.name}/capsule_caps"):
                obj = self.viewer.objects.get(name)
                if obj is not None:
                    obj.hidden = hidden

    def _project_world_to_screen(self, position: np.ndarray, ui):
        camera = getattr(self.viewer, "camera", None)
        if camera is None:
            return None

        view = np.asarray(camera.get_view_matrix(), dtype=np.float32).reshape(4, 4)
        projection = np.asarray(camera.get_projection_matrix(), dtype=np.float32).reshape(4, 4)
        point = np.array([position[0], position[1], position[2], 1.0], dtype=np.float32)

        ndc = None
        for view_matrix, projection_matrix in ((view, projection), (view.T, projection.T)):
            clip = projection_matrix @ view_matrix @ point
            if abs(float(clip[3])) < 1e-6 or float(clip[3]) <= 0.0:
                continue
            candidate = clip[:3] / clip[3]
            if np.any(np.abs(candidate[:2]) > 1.2):
                continue
            ndc = candidate
            break
        if ndc is None:
            return None

        io = ui.get_io()
        display_size = io.display_size
        display_width = display_size.x if hasattr(display_size, "x") else display_size[0]
        display_height = display_size.y if hasattr(display_size, "y") else display_size[1]
        x = (ndc[0] * 0.5 + 0.5) * display_width
        y = (1.0 - (ndc[1] * 0.5 + 0.5)) * display_height
        return float(x), float(y)

    def _draw_label(self, ui, draw_list, text: str, position: np.ndarray, color) -> None:
        screen = self._project_world_to_screen(position, ui)
        if screen is None:
            return
        draw_list.add_text(ui.ImVec2(screen[0] + 5.0, screen[1] - 5.0), ui.get_color_u32(ui.ImVec4(*color)), text)

    def _draw_selected_labels(self, ui) -> None:
        draw_list = ui.get_foreground_draw_list()
        soma_transforms = self._soma_global_transforms_np()
        for joint_index, enabled in enumerate(self.soma_show_names):
            if enabled:
                self._draw_label(
                    ui,
                    draw_list,
                    self.skeleton.joint_names[joint_index],
                    soma_transforms[joint_index, :3],
                    (0.98, 0.95, 0.35, 1.0),
                )

        robot_transforms = self._robot_body_transforms_np()
        if self.robot_pelvis_control is not None and self.robot_pelvis_control["show_name"]:
            body_id = self.robot_pelvis_control["body_id"]
            if 0 <= body_id < len(robot_transforms):
                self._draw_label(
                    ui,
                    draw_list,
                    self.robot_pelvis_control["name"],
                    robot_transforms[body_id, :3],
                    (1.0, 0.55, 0.25, 1.0),
                )
        for control in self.robot_joint_controls:
            body_id = control["child_body_id"]
            if control["show_name"] and 0 <= body_id < len(robot_transforms):
                self._draw_label(
                    ui,
                    draw_list,
                    control["name"],
                    robot_transforms[body_id, :3],
                    (0.55, 0.8, 1.0, 1.0),
                )

        preview_transforms = self._preview_effector_transforms()
        for joint_name in self.preview_joint_names:
            if joint_name not in preview_transforms:
                continue
            if self.preview_show_names.get(joint_name, False):
                self._draw_label(
                    ui,
                    draw_list,
                    f"preview:{joint_name}",
                    preview_transforms[joint_name][:3],
                    (0.45, 1.0, 0.9, 1.0),
                )

    def step(self) -> None:
        self._update_scene()
        self.time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.time)
        if self.show_soma_mesh and self.soma_mesh_renderer is not None:
            self.soma_mesh_renderer.draw(self.viewer, self.soma_instance, self.soma_instance.color, 0)
        if self.show_soma_skeleton and self.soma_skeleton_renderer is not None:
            self.soma_skeleton_renderer.draw(self.viewer, self.soma_instance, 0)
        if self.show_preview_mesh and self.preview_mesh_renderer is not None:
            self.preview_mesh_renderer.draw(self.viewer, self.preview_instance, self.preview_instance.color, 1)
        if self.show_preview_skeleton and self.preview_skeleton_renderer is not None:
            self.preview_skeleton_renderer.draw(self.viewer, self.preview_instance, 1)
        self._render_selected_axes()
        if self.show_gizmos:
            self.viewer.log_gizmo("soma_offset", self.soma_instance.xform)
        self.viewer.log_state(self.state)
        self._set_robot_mesh_hidden(not self.show_robot_mesh)
        self.viewer.end_frame()

    def run(self) -> None:
        while self.viewer.is_running():
            with wp.ScopedTimer("step", active=False):
                self.step()
            with wp.ScopedTimer("render", active=False):
                self.render()
        self.viewer.close()

    def gui(self, ui) -> None:
        self._ui_soma_panel(ui)
        self._ui_preview_panel(ui)
        self._ui_robot_panel(ui)
        self._ui_bottom_panel(ui)
        self._draw_selected_labels(ui)

    @staticmethod
    def _slider_input_float(ui, label: str, value: float, min_value: float, max_value: float, fmt: str, width: float):
        ui.set_next_item_width(max(90, width - 125))
        slider_changed, slider_value = ui.slider_float(label, float(value), float(min_value), float(max_value), fmt)
        ui.same_line()
        ui.set_next_item_width(72)
        input_changed, input_value = ui.input_float(f"##{label}_input", float(value), 0.0, 0.0, fmt)
        if slider_changed or input_changed:
            return True, float(np.clip(input_value if input_changed else slider_value, min_value, max_value))
        return False, float(value)

    def _ui_soma_panel(self, ui) -> None:
        import tkinter as tk
        from tkinter import filedialog as tk_filedialog

        viewport = ui.get_main_viewport()
        height = viewport.size.y - _BOTTOM_PANEL_HEIGHT - 3 * _PANEL_MARGIN
        ui.set_next_window_pos(ui.ImVec2(_PANEL_MARGIN, _PANEL_MARGIN))
        ui.set_next_window_size(ui.ImVec2(_SIDE_PANEL_WIDTH, height))
        ui.set_next_window_bg_alpha(_PANEL_ALPHA)
        ui.begin("SOMA Joints", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        ui.text(f"{self.source_name} joints: {self.skeleton.num_joints}")
        ui.text(f"BVH: {self.soma_bvh.name}")
        ui.text(f"Source mesh: {self.source_mesh_mode}")
        if self.source_mesh_mode != "none":
            ui.text(f"USD: {self.soma_usd.name}")
        ui.separator()
        if ui.button("Load single-frame BVH"):
            root = tk.Tk()
            root.withdraw()
            bvh_path = tk_filedialog.askopenfilename(
                title=f"Load {self.source_name} single-frame BVH",
                initialdir=str(self.soma_bvh.parent),
                defaultextension=".bvh",
                filetypes=[("BVH files", "*.bvh")],
            )
            root.destroy()
            if bvh_path:
                try:
                    self._load_soma_pose(Path(bvh_path))
                    self.last_pose_pair_status = f"loaded: {Path(bvh_path).name}"
                except Exception as exc:
                    self.last_pose_pair_status = f"load failed: {exc}"
        if self.soma_mesh_renderer is not None:
            changed, self.show_soma_mesh = ui.checkbox("Show mesh", self.show_soma_mesh)
        else:
            self.show_soma_mesh = False
            changed = False
            ui.text("Show mesh: unavailable in source_mesh=none mode")
        if changed and not self.show_soma_mesh and self.soma_mesh_renderer is not None:
            self.soma_mesh_renderer.clear(self.viewer)
        changed, self.show_soma_skeleton = ui.checkbox("Show skeleton", self.show_soma_skeleton)
        if changed and not self.show_soma_skeleton:
            self.soma_skeleton_renderer.clear(self.viewer)
        if ui.button("Reset SOMA"):
            self.soma_euler_deg[:] = 0.0
            self.soma_current_local[:] = self.soma_reference_local
            self.soma_root_offset[:] = 0.0
            self.soma_show_names[:] = False
            self.soma_show_axes[:] = False
        ui.separator()

        for joint_index, joint_name in enumerate(self.skeleton.joint_names):
            if not ui.collapsing_header(joint_name):
                continue
            ui.push_id(joint_index)
            _, self.soma_show_names[joint_index] = ui.checkbox("Name", bool(self.soma_show_names[joint_index]))
            ui.same_line()
            _, self.soma_show_axes[joint_index] = ui.checkbox("Axis", bool(self.soma_show_axes[joint_index]))
            if joint_index == 0:
                ui.text("Root Translation")
                for axis_idx, axis_name in enumerate(("tx", "ty", "tz")):
                    changed, value = self._slider_input_float(
                        ui,
                        axis_name,
                        float(self.soma_root_offset[axis_idx]),
                        -2.0,
                        2.0,
                        "%.3f",
                        _SIDE_PANEL_WIDTH,
                    )
                    if changed:
                        self.soma_root_offset[axis_idx] = value
            for axis_index, axis_name in enumerate(("rx", "ry", "rz")):
                ui.push_id(joint_index * 10 + axis_index)
                current = float(self.soma_euler_deg[joint_index, axis_index])
                ui.set_next_item_width(_SIDE_PANEL_WIDTH - 170)
                slider_changed, value = ui.slider_float(
                    axis_name,
                    current,
                    -180.0,
                    180.0,
                    "%.1f deg",
                )
                ui.same_line()
                ui.set_next_item_width(72)
                input_changed, input_value = ui.input_float("##deg", current, 0.0, 0.0, "%.1f")
                ui.pop_id()
                if slider_changed or input_changed:
                    self.soma_euler_deg[joint_index, axis_index] = float(np.clip(input_value if input_changed else value, -180.0, 180.0))
                    self._set_soma_joint_euler(joint_index, self.soma_euler_deg[joint_index])
            ui.pop_id()
        ui.end()

    def _ui_robot_panel(self, ui) -> None:
        viewport = ui.get_main_viewport()
        height = viewport.size.y - _BOTTOM_PANEL_HEIGHT - 3 * _PANEL_MARGIN
        x = viewport.size.x - _SIDE_PANEL_WIDTH - _PANEL_MARGIN
        ui.set_next_window_pos(ui.ImVec2(x, _PANEL_MARGIN))
        ui.set_next_window_size(ui.ImVec2(_SIDE_PANEL_WIDTH, height))
        ui.set_next_window_bg_alpha(_PANEL_ALPHA)
        ui.begin("Robot Joints", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        ui.text(f"Robot joints: {len(self.robot_joint_controls)} scalar controls")
        ui.text(f"MJCF: {self.robot_mjcf.name}")
        ui.separator()
        _, self.show_robot_mesh = ui.checkbox("Show robot mesh", self.show_robot_mesh)
        if ui.button("Reset Robot"):
            self.robot_q[:] = self.robot_q_default
            self.robot_root_euler_deg = R.from_quat(self.robot_q[3:7]).as_euler("xyz", degrees=True).astype(np.float32)
            for control in self.robot_joint_controls:
                q_value = float(self.robot_q[control["q_index"]])
                control["value"] = float(np.degrees(q_value)) if control["is_revolute"] else q_value
                control["show_name"] = False
                control["show_axis"] = False
            if self.robot_pelvis_control is not None:
                self.robot_pelvis_control["show_name"] = True
                self.robot_pelvis_control["show_axis"] = True
        ui.separator()

        if self.robot_pelvis_control is not None:
            if ui.collapsing_header(f"Pelvis Link: {self.robot_pelvis_control['name']}"):
                _, self.robot_pelvis_control["show_name"] = ui.checkbox(
                    "Pelvis name",
                    bool(self.robot_pelvis_control["show_name"]),
                )
                ui.same_line()
                _, self.robot_pelvis_control["show_axis"] = ui.checkbox(
                    "Pelvis axis",
                    bool(self.robot_pelvis_control["show_axis"]),
                )
            ui.separator()

        if ui.collapsing_header("Floating Root", flags=ui.TreeNodeFlags_.default_open):
            ui.text("Root position")
            for axis_idx, axis_name in enumerate(("x", "y", "z")):
                changed, value = self._slider_input_float(
                    ui,
                    axis_name,
                    float(self.robot_q[axis_idx]),
                    -2.0 if axis_idx < 2 else -0.5,
                    2.0,
                    "%.3f",
                    _SIDE_PANEL_WIDTH,
                )
                if changed:
                    self.robot_q[axis_idx] = value
            ui.text("Root rotation")
            for axis_idx, axis_name in enumerate(("rx", "ry", "rz")):
                changed, value = self._slider_input_float(
                    ui,
                    axis_name,
                    float(self.robot_root_euler_deg[axis_idx]),
                    -180.0,
                    180.0,
                    "%.2f",
                    _SIDE_PANEL_WIDTH,
                )
                if changed:
                    self.robot_root_euler_deg[axis_idx] = value
                    self._set_robot_root_euler(self.robot_root_euler_deg)
            ui.separator()

        for i, control in enumerate(self.robot_joint_controls):
            if not ui.collapsing_header(control["name"]):
                continue
            ui.push_id(10_000 + i)
            _, control["show_name"] = ui.checkbox("Name", bool(control["show_name"]))
            ui.same_line()
            _, control["show_axis"] = ui.checkbox("Axis", bool(control["show_axis"]))
            ui.set_next_item_width(_SIDE_PANEL_WIDTH - 170)
            slider_changed, value = ui.slider_float(
                control["unit"],
                float(control["value"]),
                float(control["min"]),
                float(control["max"]),
                "%.2f deg" if control["is_revolute"] else "%.3f",
            )
            ui.same_line()
            ui.set_next_item_width(72)
            input_changed, input_value = ui.input_float("##value", float(control["value"]), 0.0, 0.0, "%.2f")
            if slider_changed or input_changed:
                control["value"] = float(np.clip(input_value if input_changed else value, control["min"], control["max"]))
                self.robot_q[control["q_index"]] = np.radians(control["value"]) if control["is_revolute"] else control["value"]
            ui.pop_id()
        ui.end()

    def _ui_preview_panel(self, ui) -> None:
        viewport = ui.get_main_viewport()
        height = viewport.size.y - _BOTTOM_PANEL_HEIGHT - 3 * _PANEL_MARGIN
        x = _SIDE_PANEL_WIDTH + 2 * _PANEL_MARGIN
        ui.set_next_window_pos(ui.ImVec2(x, _PANEL_MARGIN))
        ui.set_next_window_size(ui.ImVec2(_SIDE_PANEL_WIDTH, height))
        ui.set_next_window_bg_alpha(_PANEL_ALPHA)
        ui.begin("Config Preview SOMA", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        ui.text(f"Robot type: {self.robot_type}")
        ui.text(f"Mapped joints: {len(self.preview_joint_names)}")
        summary = self.preview_report.get("summary", {})
        ui.text(f"Mean pos: {summary.get('mean_position_error_m', 0.0):.3f}m")
        ui.same_line()
        ui.text(f"Max pos: {summary.get('max_position_error_m', 0.0):.3f}m")
        ui.text(f"Mean rot: {summary.get('mean_rotation_error_deg', 0.0):.1f}deg")
        ui.same_line()
        ui.text(f"Max rot: {summary.get('max_rotation_error_deg', 0.0):.1f}deg")
        ui.text(f"Worst: {summary.get('worst_joint', '')}")
        ui.separator()
        if self.preview_mesh_renderer is not None:
            changed, self.show_preview_mesh = ui.checkbox("Show preview mesh", self.show_preview_mesh)
        else:
            self.show_preview_mesh = False
            changed = False
            ui.text("Show preview mesh: unavailable in source_mesh=none mode")
        if changed and not self.show_preview_mesh and self.preview_mesh_renderer is not None:
            self.preview_mesh_renderer.clear(self.viewer)
        changed, self.show_preview_skeleton = ui.checkbox("Show preview skeleton", self.show_preview_skeleton)
        if changed and not self.show_preview_skeleton:
            self.preview_skeleton_renderer.clear(self.viewer)
        mapped_preview_names = [name for name in self.mapped_joint_names if name in self.preview_show_axes]
        all_mapped_axes = bool(mapped_preview_names) and all(self.preview_show_axes.get(name, False) for name in mapped_preview_names)
        all_mapped_names = bool(mapped_preview_names) and all(self.preview_show_names.get(name, False) for name in mapped_preview_names)
        changed, value = ui.checkbox("Show mapped axes", all_mapped_axes)
        if changed:
            self.show_preview_mapped_axes = bool(value)
            for name in mapped_preview_names:
                self.preview_show_axes[name] = bool(value)
        changed, value = ui.checkbox("Show mapped names", all_mapped_names)
        if changed:
            self.show_preview_mapped_names = bool(value)
            for name in mapped_preview_names:
                self.preview_show_names[name] = bool(value)

        if ui.button("Reset preview config"):
            self._reset_preview_config()
        ui.same_line()
        if ui.button("Save Calibration"):
            try:
                self.export_calibration()
            except Exception as exc:
                self.last_export_status = f"export failed: {exc}"
        ui.same_line()
        if ui.button("Save Pose Pair"):
            try:
                self.save_pose_pair()
            except Exception as exc:
                self.last_pose_pair_status = f"pose-pair save failed: {exc}"
        if self.last_export_status:
            ui.text(f"Status: {self.last_export_status}")
        if self.last_export_path:
            ui.text(f"Path: {self.last_export_path}")
        if self.last_pose_pair_status:
            ui.text(f"Pose pair: {self.last_pose_pair_status}")
        if self.last_pose_pair_path:
            ui.text(f"Pose pair path: {self.last_pose_pair_path}")
        ui.separator()

        for joint_name in self.preview_joint_names:
            if not ui.collapsing_header(joint_name):
                continue
            ui.push_id(f"preview_{joint_name}")
            _, self.preview_show_names[joint_name] = ui.checkbox("Name", bool(self.preview_show_names[joint_name]))
            ui.same_line()
            _, self.preview_show_axes[joint_name] = ui.checkbox("Axis", bool(self.preview_show_axes[joint_name]))

            changed, value = self._slider_input_float(
                ui,
                "scale",
                self.preview_joint_scales[joint_name],
                0.3,
                1.5,
                "%.4f",
                _SIDE_PANEL_WIDTH,
            )
            if changed:
                self.preview_joint_scales[joint_name] = value

            for axis_idx, axis_name in enumerate(("tx", "ty", "tz")):
                changed, value = self._slider_input_float(
                    ui,
                    axis_name,
                    float(self.preview_joint_offsets_t[joint_name][axis_idx]),
                    -0.3,
                    0.3,
                    "%.4f",
                    _SIDE_PANEL_WIDTH,
                )
                if changed:
                    self.preview_joint_offsets_t[joint_name][axis_idx] = value

            for axis_idx, axis_name in enumerate(("rx", "ry", "rz")):
                changed, value = self._slider_input_float(
                    ui,
                    axis_name,
                    float(self.preview_joint_offsets_rpy[joint_name][axis_idx]),
                    -180.0,
                    180.0,
                    "%.2f",
                    _SIDE_PANEL_WIDTH,
                )
                if changed:
                    self.preview_joint_offsets_rpy[joint_name][axis_idx] = value
            ui.pop_id()
        ui.end()

    def _ui_bottom_panel(self, ui) -> None:
        viewport = ui.get_main_viewport()
        x = _PANEL_MARGIN
        width = max(260, viewport.size.x - 2 * _PANEL_MARGIN)
        y = viewport.size.y - _BOTTOM_PANEL_HEIGHT - _PANEL_MARGIN
        ui.set_next_window_pos(ui.ImVec2(x, y))
        ui.set_next_window_size(ui.ImVec2(width, _BOTTOM_PANEL_HEIGHT))
        ui.set_next_window_bg_alpha(_PANEL_ALPHA)
        ui.begin("Alignment", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        ui.text("Overlap moves SOMA from +X offset to robot alignment.")
        ui.set_next_item_width(max(120, width - 170))
        changed, value = ui.slider_float(
            "Overlap (m)",
            float(self.overlap_x),
            0.0,
            self.initial_offset_x,
            "%.3f",
        )
        if changed:
            self.overlap_x = float(np.clip(value, 0.0, self.initial_offset_x))
        ui.same_line()
        ui.text(f"SOMA x offset: {self.initial_offset_x - self.overlap_x:.3f}m")
        _, self.show_gizmos = ui.checkbox("Show gizmos", self.show_gizmos)
        ui.same_line()
        if ui.button("Reset All"):
            self.overlap_x = 0.0
            self.soma_euler_deg[:] = 0.0
            self.soma_current_local[:] = self.soma_reference_local
            self.soma_root_offset[:] = 0.0
            self.soma_show_names[:] = False
            self.soma_show_axes[:] = False
            self.robot_q[:] = self.robot_q_default
            self.robot_root_euler_deg = R.from_quat(self.robot_q[3:7]).as_euler("xyz", degrees=True).astype(np.float32)
            for control in self.robot_joint_controls:
                q_value = float(self.robot_q[control["q_index"]])
                control["value"] = float(np.degrees(q_value)) if control["is_revolute"] else q_value
                control["show_name"] = False
                control["show_axis"] = False
            if self.robot_pelvis_control is not None:
                self.robot_pelvis_control["show_name"] = True
                self.robot_pelvis_control["show_axis"] = True
        ui.end()


def parse_args():
    import newton.examples

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="gl")
    parser.add_argument("--soma_bvh", type=str, default=str(_DEFAULT_SOMA_BVH))
    parser.add_argument("--robot_mjcf", type=str, default=None)
    parser.add_argument("--soma_usd", type=str, default=str(_DEFAULT_SOMA_USD))
    parser.add_argument(
        "--source_name",
        type=str,
        default="SOMA",
        help="Display name for the source skeleton. Use e.g. Nutan for non-SOMA BVH files.",
    )
    parser.add_argument(
        "--source_mesh",
        choices=["soma", "none"],
        default="soma",
        help="Use 'none' for source BVH skeletons that do not match the SOMA USD mesh.",
    )
    parser.add_argument("--initial_offset_x", type=float, default=1.0)
    parser.add_argument("--robot_type", choices=["unitree_g1", "agile_one"], default=None)
    parser.add_argument("--base_retargeter_config", type=str, default=None)
    parser.add_argument("--base_scaler_config", type=str, default=None)
    parser.add_argument("--export_dir", type=str, default=str(_DEFAULT_EXPORT_DIR))
    return parser


def main() -> None:
    import newton.examples

    parser = parse_args()
    viewer, args = newton.examples.init(parser)
    with wp.ScopedDevice(args.device):
        app = SomaRobotOffsetCalibrator(viewer, args)
        app.run()


if __name__ == "__main__":
    main()
