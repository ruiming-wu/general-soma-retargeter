#!/usr/bin/env python3
"""Interactive one-frame robot pose editor for calibration pose libraries."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from types import MethodType

import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.robot_pose_utils import ROBOT_POSE_SCHEMA, build_semantic_targets, load_json, write_json
from soma_retargeter.utils.newton_utils import get_name_from_label


_DEFAULT_EXPORT_DIR = _REPO_ROOT / "output/robot_pose_editor"
_DEFAULT_G1_MJCF = Path("/home/ruiming.wu/codes/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
_DEFAULT_AO_MJCF = Path("/home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml")
_PANEL_WIDTH = 420
_PANEL_MARGIN = 10
_PANEL_ALPHA = 0.92
_AXIS_SCALE = 0.08
_AXIS_THICKNESS = 0.006
_AXIS_COLORS = (
    (1.0, 0.05, 0.05),
    (0.05, 1.0, 0.05),
    (0.1, 0.35, 1.0),
)


def build_robot_pose_sample(
    robot_type: str,
    robot_mjcf: Path,
    pose_name: str,
    robot_q: np.ndarray,
    robot_joints: dict[str, float],
    robot_body_names: list[str],
    robot_body_q: np.ndarray,
    semantic_map: dict | None = None,
    base_retargeter_config: Path | None = None,
) -> dict:
    body_transforms = {
        name: [float(x) for x in robot_body_q[idx]]
        for idx, name in enumerate(robot_body_names)
    }
    semantic_targets = build_semantic_targets(body_transforms, semantic_map or {}) if semantic_map else {}
    return {
        "schema": ROBOT_POSE_SCHEMA,
        "pose_name": pose_name,
        "robot_type": robot_type,
        "robot_mjcf": str(Path(robot_mjcf).expanduser().resolve()),
        "base_retargeter_config": str(Path(base_retargeter_config).expanduser().resolve()) if base_retargeter_config else "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "robot_root_position": [float(x) for x in robot_q[:3]],
        "robot_root_rotation_xyzw": [float(x) for x in robot_q[3:7]],
        "robot_joint_q": [float(x) for x in robot_q],
        "robot_joints": {name: float(value) for name, value in robot_joints.items()},
        "robot_body_transforms": body_transforms,
        "semantic_targets": semantic_targets,
    }


def write_robot_pose_sample_to_path(path: Path, pose: dict) -> Path:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    pose = dict(pose)
    pose["pose_name"] = path.stem
    write_json(path, pose)
    return path


class RobotPoseEditor:
    def __init__(self, viewer, args: argparse.Namespace):
        if isinstance(viewer, newton.viewer.ViewerNull):
            raise ValueError("robot_pose_editor requires an interactive viewer, e.g. --viewer gl")
        self.viewer = viewer
        self.viewer.vsync = True
        self.viewer.renderer.set_title(f"{args.robot_type} Robot Pose Editor")
        self.viewer._render_left_panel = MethodType(lambda _viewer: None, self.viewer)
        self.viewer._render_stats_overlay = MethodType(lambda _viewer: None, self.viewer)
        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")

        self.args = args
        self.robot_type = args.robot_type
        self.pose_name = args.pose_name
        self.export_dir = Path(args.export_dir).expanduser().resolve()
        self.robot_mjcf = self._resolve_robot_mjcf(args)
        self.base_retargeter_path = Path(args.base_retargeter_config).expanduser().resolve() if args.base_retargeter_config else None
        self.base_retargeter_config = load_json(self.base_retargeter_path) if self.base_retargeter_path else {}
        self.semantic_map = self._select_semantic_map(args.map_side)

        self.show_robot_mesh = True
        self.show_all_semantic_axes = True
        self.show_gizmo = True
        self.last_save_status = ""
        self.last_save_path = ""
        self.time = 0.0
        self.frame_dt = 1.0 / 60.0

        self._load_robot()
        if args.load_pose:
            self._load_pose_json(Path(args.load_pose))
        else:
            self._apply_initial_robot_joint_positions()
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets([0, 0, 0])

    def _resolve_robot_mjcf(self, args: argparse.Namespace) -> Path:
        if args.robot_mjcf:
            return Path(args.robot_mjcf).expanduser().resolve()
        if args.robot_type == "unitree_g1":
            return _DEFAULT_G1_MJCF.resolve()
        if args.robot_type == "agile_one":
            return _DEFAULT_AO_MJCF.resolve()
        raise ValueError(f"No default MJCF for robot_type={args.robot_type}")

    def _select_semantic_map(self, map_side: str) -> dict:
        if not self.base_retargeter_config:
            return {}
        if map_side == "source":
            return self.base_retargeter_config.get("source_ik_map", {})
        if map_side == "target":
            return self.base_retargeter_config.get("ik_map", {})
        source_type = self.base_retargeter_config.get("source_type")
        target_type = self.base_retargeter_config.get("target_type")
        if self.robot_type == source_type:
            return self.base_retargeter_config.get("source_ik_map", {})
        if self.robot_type == target_type:
            return self.base_retargeter_config.get("ik_map", {})
        return self.base_retargeter_config.get("ik_map", {})

    def _load_robot(self) -> None:
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_mjcf(self.robot_mjcf)
        self.model = builder.finalize()
        self.state = self.model.state()
        self.robot_q_default = self.model.joint_q.numpy().astype(np.float32, copy=True)
        self.robot_q = self.robot_q_default.copy()
        self.robot_body_names = [get_name_from_label(label) for label in self.model.body_label]
        self.robot_body_name_to_idx = {name: idx for idx, name in enumerate(self.robot_body_names)}
        self.robot_joint_controls = self._build_robot_joint_controls()
        self.robot_pelvis_control = self._build_robot_pelvis_control()
        self.robot_root_euler_deg = R.from_quat(self.robot_q[3:7]).as_euler("xyz", degrees=True).astype(np.float32)
        self._validate_semantic_bodies()

    def _validate_semantic_bodies(self) -> None:
        missing = []
        for semantic_name, mapping in self.semantic_map.items():
            for key in ("t_body", "r_body"):
                if mapping[key] not in self.robot_body_name_to_idx:
                    missing.append(f"{semantic_name}.{key}={mapping[key]}")
        if missing:
            raise ValueError("Semantic map references bodies missing from MJCF: " + ", ".join(missing))

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
                min_value, max_value = (-180.0, 180.0) if not np.isfinite(lo + hi) or abs(lo) > 1e9 or abs(hi) > 1e9 else np.degrees([lo, hi]).astype(float)
                value = float(np.degrees(self.robot_q[q_index]))
                unit = "deg"
            else:
                lo = float(lower[dof_index])
                hi = float(upper[dof_index])
                min_value, max_value = (-1.0, 1.0) if not np.isfinite(lo + hi) or abs(lo) > 1e9 or abs(hi) > 1e9 else (lo, hi)
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
                return {"name": name, "body_id": body_id, "show_axis": True}
        return None

    def _apply_initial_robot_joint_positions(self) -> None:
        initial = self.base_retargeter_config.get("initial_robot_joint_positions")
        if not isinstance(initial, dict):
            return
        if "root_position" in initial:
            self.robot_q[:3] = np.asarray(initial["root_position"], dtype=np.float32)
        if "root_rotation_xyzw" in initial:
            quat = np.asarray(initial["root_rotation_xyzw"], dtype=np.float32)
            self.robot_q[3:7] = quat / max(float(np.linalg.norm(quat)), 1e-12)
            self.robot_root_euler_deg = R.from_quat(self.robot_q[3:7]).as_euler("xyz", degrees=True).astype(np.float32)
        controls = {control["name"]: control for control in self.robot_joint_controls}
        for joint_name, value in initial.get("joints", {}).items():
            control = controls.get(joint_name)
            if control is None:
                continue
            q_value = float(value)
            self.robot_q[control["q_index"]] = q_value
            control["value"] = float(np.degrees(q_value)) if control["is_revolute"] else q_value

    def _load_pose_json(self, path: Path) -> None:
        data = load_json(path)
        q = np.asarray(data.get("robot_joint_q", []), dtype=np.float32)
        if q.shape != self.robot_q.shape:
            raise ValueError(f"robot_joint_q shape mismatch: {q.shape} != {self.robot_q.shape}")
        self.robot_q[:] = q
        self.robot_root_euler_deg = R.from_quat(self.robot_q[3:7]).as_euler("xyz", degrees=True).astype(np.float32)
        controls = {control["name"]: control for control in self.robot_joint_controls}
        for joint_name, value in data.get("robot_joints", {}).items():
            control = controls.get(joint_name)
            if control is None:
                continue
            control["value"] = float(np.degrees(value)) if control["is_revolute"] else float(value)

    def _set_robot_root_euler(self, euler_deg: np.ndarray) -> None:
        quat = R.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float32)
        self.robot_q[3:7] = quat / max(float(np.linalg.norm(quat)), 1e-12)

    def _update_scene(self) -> None:
        wp.copy(self.model.joint_q, wp.array(self.robot_q, dtype=wp.float32), 0, 0, len(self.robot_q))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)

    def _robot_body_transforms_np(self) -> np.ndarray:
        return self.state.body_q.numpy()

    def _robot_joint_values_by_name(self) -> dict[str, float]:
        return {control["name"]: float(self.robot_q[control["q_index"]]) for control in self.robot_joint_controls}

    def build_pose_sample(self) -> dict:
        self._update_scene()
        return build_robot_pose_sample(
            robot_type=self.robot_type,
            robot_mjcf=self.robot_mjcf,
            pose_name=self.pose_name,
            robot_q=self.robot_q,
            robot_joints=self._robot_joint_values_by_name(),
            robot_body_names=self.robot_body_names,
            robot_body_q=self._robot_body_transforms_np(),
            semantic_map=self.semantic_map,
            base_retargeter_config=self.base_retargeter_path,
        )

    def save_pose(self, path: Path | None = None) -> Path:
        output_dir = self.export_dir / self.robot_type
        if path is None:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{self.pose_name}.json"
            suffix = 1
            while path.exists():
                path = output_dir / f"{self.pose_name}_{suffix:02d}.json"
                suffix += 1
        path = write_robot_pose_sample_to_path(path, self.build_pose_sample())
        self.pose_name = path.stem
        self.last_save_path = str(path)
        self.last_save_status = "saved"
        return path

    def save_pose_as_dialog(self) -> None:
        import tkinter as tk
        from tkinter import filedialog as tk_filedialog

        output_dir = self.export_dir / self.robot_type
        output_dir.mkdir(parents=True, exist_ok=True)
        root = tk.Tk()
        root.withdraw()
        save_path = tk_filedialog.asksaveasfilename(
            title="Save robot pose JSON",
            initialdir=str(output_dir),
            initialfile=f"{self.pose_name}.json",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        root.destroy()
        if save_path:
            self.save_pose(Path(save_path))

    def reset_robot(self) -> None:
        self.robot_q[:] = self.robot_q_default
        self.robot_root_euler_deg = R.from_quat(self.robot_q[3:7]).as_euler("xyz", degrees=True).astype(np.float32)
        for control in self.robot_joint_controls:
            q_value = float(self.robot_q[control["q_index"]])
            control["value"] = float(np.degrees(q_value)) if control["is_revolute"] else q_value
            control["show_axis"] = False

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

    @staticmethod
    def _append_thick_axes(starts, ends, colors, transform_row: np.ndarray, scale: float = _AXIS_SCALE, thickness: float = _AXIS_THICKNESS) -> None:
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
        starts, ends, colors = [], [], []
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
        body_q = self._robot_body_transforms_np()
        selected = []
        if self.robot_pelvis_control is not None and self.robot_pelvis_control["show_axis"]:
            selected.append(body_q[self.robot_pelvis_control["body_id"]])
        for control in self.robot_joint_controls:
            if control["show_axis"]:
                selected.append(body_q[control["child_body_id"]])
        self._log_axis_batch("/robot_selected_axes", selected)

        if self.show_all_semantic_axes and self.semantic_map:
            semantic_transforms = []
            for mapping in self.semantic_map.values():
                body_name = mapping["t_body"]
                if body_name in self.robot_body_name_to_idx:
                    semantic_transforms.append(body_q[self.robot_body_name_to_idx[body_name]])
            self._log_axis_batch("/robot_semantic_axes", semantic_transforms)

    def _set_robot_mesh_hidden(self, hidden: bool) -> None:
        for batch in getattr(self.viewer, "_shape_instances", {}).values():
            for name in (batch.name, f"{batch.name}/capsule_cylinder", f"{batch.name}/capsule_caps"):
                obj = self.viewer.objects.get(name)
                if obj is not None:
                    obj.hidden = hidden

    def step(self) -> None:
        self._update_scene()
        self.time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.time)
        self._render_selected_axes()
        if self.show_gizmo:
            self.viewer.log_gizmo("robot_root", wp.transform(wp.vec3(*[float(x) for x in self.robot_q[:3]]), wp.quat(*[float(x) for x in self.robot_q[3:7]])))
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
        viewport = ui.get_main_viewport()
        ui.set_next_window_pos(ui.ImVec2(_PANEL_MARGIN, _PANEL_MARGIN))
        ui.set_next_window_size(ui.ImVec2(_PANEL_WIDTH, max(300, viewport.size.y - 2 * _PANEL_MARGIN)))
        ui.set_next_window_bg_alpha(_PANEL_ALPHA)
        ui.begin("Robot Pose", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        ui.text(f"Robot type: {self.robot_type}")
        ui.text(f"MJCF: {self.robot_mjcf.name}")
        ui.text(f"Pose name: {self.pose_name}")
        ui.text(f"Semantic frames: {len(self.semantic_map)}")
        _, self.show_robot_mesh = ui.checkbox("Show robot mesh", self.show_robot_mesh)
        _, self.show_all_semantic_axes = ui.checkbox("Show semantic axes", self.show_all_semantic_axes)
        _, self.show_gizmo = ui.checkbox("Show root gizmo", self.show_gizmo)
        if ui.button("Save Pose JSON As..."):
            try:
                self.save_pose_as_dialog()
            except Exception as exc:
                self.last_save_status = f"save failed: {exc}"
        ui.same_line()
        if ui.button("Quick Save"):
            try:
                self.save_pose()
            except Exception as exc:
                self.last_save_status = f"save failed: {exc}"
        ui.same_line()
        if ui.button("Reset Robot"):
            self.reset_robot()
        if self.last_save_status:
            ui.text(f"Status: {self.last_save_status}")
        if self.last_save_path:
            ui.text(f"Path: {self.last_save_path}")
        ui.separator()

        if self.robot_pelvis_control is not None and ui.collapsing_header(f"Pelvis: {self.robot_pelvis_control['name']}"):
            _, self.robot_pelvis_control["show_axis"] = ui.checkbox("Axis", bool(self.robot_pelvis_control["show_axis"]))

        if ui.collapsing_header("Floating Root", flags=ui.TreeNodeFlags_.default_open):
            for axis_idx, axis_name in enumerate(("x", "y", "z")):
                changed, value = self._slider_input_float(ui, axis_name, float(self.robot_q[axis_idx]), -2.0 if axis_idx < 2 else -0.5, 2.0, "%.3f", _PANEL_WIDTH)
                if changed:
                    self.robot_q[axis_idx] = value
            for axis_idx, axis_name in enumerate(("rx", "ry", "rz")):
                changed, value = self._slider_input_float(ui, axis_name, float(self.robot_root_euler_deg[axis_idx]), -180.0, 180.0, "%.2f", _PANEL_WIDTH)
                if changed:
                    self.robot_root_euler_deg[axis_idx] = value
                    self._set_robot_root_euler(self.robot_root_euler_deg)

        for i, control in enumerate(self.robot_joint_controls):
            if not ui.collapsing_header(control["name"]):
                continue
            ui.push_id(10_000 + i)
            _, control["show_axis"] = ui.checkbox("Axis", bool(control["show_axis"]))
            changed, value = self._slider_input_float(
                ui,
                control["unit"],
                float(control["value"]),
                float(control["min"]),
                float(control["max"]),
                "%.2f" if control["is_revolute"] else "%.3f",
                _PANEL_WIDTH,
            )
            if changed:
                control["value"] = value
                self.robot_q[control["q_index"]] = np.radians(value) if control["is_revolute"] else value
            ui.pop_id()
        ui.end()


def parse_args() -> argparse.Namespace:
    import newton.examples

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="gl")
    parser.add_argument("--robot-type", choices=["unitree_g1", "agile_one"], required=True)
    parser.add_argument("--robot-mjcf", type=str, default=None)
    parser.add_argument("--base-retargeter-config", type=str, default=None)
    parser.add_argument("--map-side", choices=["auto", "source", "target"], default="auto")
    parser.add_argument("--pose-name", type=str, default="robot_custom_pose")
    parser.add_argument("--load-pose", type=str, default=None)
    parser.add_argument("--export-dir", type=str, default=str(_DEFAULT_EXPORT_DIR))
    return parser


def main() -> None:
    import newton.examples

    parser = parse_args()
    viewer, args = newton.examples.init(parser)
    with wp.ScopedDevice(args.device):
        app = RobotPoseEditor(viewer, args)
        app.run()


if __name__ == "__main__":
    main()
