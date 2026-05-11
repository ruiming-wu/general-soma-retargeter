# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive SOMA single-pose editor.

This tool starts from a reference SOMA BVH pose, lets the user rotate joints in
Newton's GL viewer, and saves the current pose as a one-frame BVH that can be
used as a retargeter initialization/calibration pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.pipelines.utils as pipeline_utils
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.renderers.coordinate_renderer import CoordinateRenderer
from soma_retargeter.renderers.mesh_renderer import SkeletalMeshRenderer
from soma_retargeter.renderers.skeleton_renderer import SkeletonRenderer
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SOMA_BVH = _REPO_ROOT / "soma_retargeter/configs/soma/soma_zero_frame0.bvh"
_DEFAULT_EXPORT_DIR = _REPO_ROOT / "soma_retargeter/configs/soma/calibration_poses"

_PANEL_WIDTH = 390
_PANEL_MARGIN = 10
_PANEL_ALPHA = 0.9
_AXIS_SCALE = 0.08
_COLOR = (235.0 / 255.0, 245.0 / 255.0, 112.0 / 255.0)


@dataclass
class BVHChannelLayout:
    hierarchy_lines: list[str]
    joint_channels: list[tuple[str, list[str]]]
    frame_time: float


def _parse_bvh_channel_layout(path: Path) -> BVHChannelLayout:
    lines = path.read_text(encoding="utf-8").splitlines()
    motion_idx = None
    frame_time = 1.0 / 120.0
    joint_stack: list[str] = []
    joint_channels: list[tuple[str, list[str]]] = []
    ignore_next_brackets = False

    for idx, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "ROOT":
            joint_stack.append(tokens[1].split(":")[-1])
        elif tokens[0] == "JOINT":
            joint_stack.append(tokens[1].split(":")[-1])
        elif line.strip() == "End Site":
            ignore_next_brackets = True
        elif tokens[0] == "CHANNELS":
            if not joint_stack:
                raise ValueError(f"CHANNELS appears before a joint in {path}")
            joint_channels.append((joint_stack[-1], tokens[2:]))
        elif tokens[0] == "}":
            if ignore_next_brackets:
                ignore_next_brackets = False
            elif joint_stack:
                joint_stack.pop()
        elif tokens[0] == "MOTION":
            motion_idx = idx
            break

    if motion_idx is None:
        raise ValueError(f"BVH file has no MOTION section: {path}")

    for line in lines[motion_idx + 1 :]:
        tokens = line.split()
        if len(tokens) >= 3 and tokens[0] == "Frame" and tokens[1] == "Time:":
            frame_time = float(tokens[2])
            break

    return BVHChannelLayout(lines[:motion_idx], joint_channels, frame_time)


def _rotation_order_from_channels(channels: list[str]) -> str:
    return "".join(channel[0].upper() for channel in channels if "rotation" in channel)


def _pose_to_motion_row(skeleton, local_transforms: np.ndarray, layout: BVHChannelLayout) -> list[float]:
    values: list[float] = []
    for joint_name, channels in layout.joint_channels:
        joint_index = skeleton.joint_index(joint_name)
        if joint_index == -1:
            raise ValueError(f"Template BVH joint is missing from loaded skeleton: {joint_name}")

        tx = local_transforms[joint_index]
        pos_cm = np.asarray(tx[:3], dtype=np.float64) * 100.0
        quat_xyzw = np.asarray(tx[3:7], dtype=np.float64)
        rot = R.from_quat(quat_xyzw / max(np.linalg.norm(quat_xyzw), 1e-12))
        rot_order = _rotation_order_from_channels(channels)
        euler_by_axis = {}
        if rot_order:
            # Upper-case scipy sequences match the loader's q = q_axis0 * q_axis1 * q_axis2 convention.
            euler = rot.as_euler(rot_order, degrees=True)
            euler_by_axis = {axis: float(euler[idx]) for idx, axis in enumerate(rot_order)}

        for channel in channels:
            axis = channel[0].upper()
            if "position" in channel:
                values.append(float(pos_cm["XYZ".index(axis)]))
            elif "rotation" in channel:
                values.append(float(euler_by_axis[axis]))
            else:
                raise ValueError(f"Unsupported BVH channel: {channel}")
    return values


def save_one_frame_bvh(template_bvh: Path, output_bvh: Path, skeleton, local_transforms: np.ndarray) -> None:
    layout = _parse_bvh_channel_layout(template_bvh)
    row = _pose_to_motion_row(skeleton, local_transforms, layout)
    output_bvh.parent.mkdir(parents=True, exist_ok=True)
    with output_bvh.open("w", encoding="utf-8") as f:
        f.write("\n".join(layout.hierarchy_lines))
        f.write("\nMOTION\n")
        f.write("Frames: 1\n")
        f.write(f"Frame Time: {layout.frame_time:.6f}\n")
        f.write(" ".join(f"{value:.10g}" for value in row))
        f.write("\n")


class SomaPoseEditor:
    def __init__(self, viewer, args) -> None:
        self.viewer = viewer
        self.viewer.vsync = True
        self.source_name = args.source_name
        self.source_mesh_mode = args.source_mesh
        self.viewer.renderer.set_title(f"{self.source_name} Pose Editor")
        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")

        self.soma_bvh = Path(args.soma_bvh).expanduser().resolve()
        self.export_dir = Path(args.export_dir).expanduser().resolve()
        self.export_name = args.export_name
        self.converter = SpaceConverter(get_facing_direction_type_from_str(args.soma_facing_direction))

        self.time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.show_mesh = self.source_mesh_mode != "none"
        self.show_skeleton = self.source_mesh_mode == "none"
        self.show_axes = True
        self.show_gizmo = True
        self.last_save_status = ""

        self.skeleton, animation = bvh_utils.load_bvh(str(self.soma_bvh))
        self.reference_local = np.copy(animation.get_local_transforms(0))
        self.current_local = np.copy(self.reference_local)
        self.euler_deg = np.zeros((self.skeleton.num_joints, 3), dtype=np.float32)
        self.root_offset = np.zeros(3, dtype=np.float32)

        self.instance = SkeletonInstance(self.skeleton, _COLOR, self._world_transform())
        self.instance.set_local_transforms(self.current_local)
        self.skeleton_renderer = SkeletonRenderer(self.skeleton, [0])
        self.coordinate_renderer = CoordinateRenderer()
        self.skeletal_mesh = None
        self.mesh_renderer = None
        if self.source_mesh_mode == "soma":
            self.skeletal_mesh = pipeline_utils.get_source_model_mesh(pipeline_utils.SourceType.SOMA, self.skeleton)
            self.mesh_renderer = SkeletalMeshRenderer(self.skeletal_mesh)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets([0, 0, 0])
        self.state = self.model.state()

    def _world_transform(self):
        return wp.transform(
            wp.vec3(float(self.root_offset[0]), float(self.root_offset[1]), float(self.root_offset[2])),
            self.converter.converter,
        )

    @staticmethod
    def _slider_input_float(ui, label: str, value: float, min_value: float, max_value: float, fmt: str, width: float):
        ui.set_next_item_width(max(90, width - 130))
        slider_changed, slider_value = ui.slider_float(label, float(value), float(min_value), float(max_value), fmt)
        ui.same_line()
        ui.set_next_item_width(78)
        input_changed, input_value = ui.input_float(f"##{label}_input", float(value), 0.0, 0.0, fmt)
        if slider_changed or input_changed:
            return True, float(np.clip(input_value if input_changed else slider_value, min_value, max_value))
        return False, float(value)

    def _set_joint_euler(self, joint_index: int, euler_deg: np.ndarray) -> None:
        reference = self.reference_local[joint_index]
        reference_rot = R.from_quat(reference[3:7])
        offset_rot = R.from_euler("xyz", euler_deg, degrees=True)
        self.current_local[joint_index, :3] = reference[:3]
        self.current_local[joint_index, 3:7] = (reference_rot * offset_rot).as_quat().astype(np.float32)
        self.instance.set_local_transforms(self.current_local)

    @staticmethod
    def _normalized_quat(quat_xyzw: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat_xyzw, dtype=np.float64)
        return quat / max(float(np.linalg.norm(quat)), 1e-12)

    def _sync_joint_euler_from_local(self, joint_index: int) -> None:
        reference_rot = R.from_quat(self._normalized_quat(self.reference_local[joint_index, 3:7]))
        current_rot = R.from_quat(self._normalized_quat(self.current_local[joint_index, 3:7]))
        offset_rot = reference_rot.inv() * current_rot
        self.euler_deg[joint_index] = offset_rot.as_euler("xyz", degrees=True).astype(np.float32)

    def _align_joint_rotation_to_root(self, joint_index: int) -> None:
        if joint_index == 0:
            self.euler_deg[0] = np.zeros(3, dtype=np.float32)
            self.current_local[0, 3:7] = self.reference_local[0, 3:7]
            self.instance.set_local_transforms(self.current_local)
            return

        global_transforms = self.instance.compute_global_transforms()
        root_global_rot = R.from_quat(self._normalized_quat(global_transforms[0, 3:7]))
        parent_index = int(self.skeleton.joint_parent(joint_index))
        parent_global_rot = R.from_quat(self._normalized_quat(global_transforms[parent_index, 3:7]))
        target_local_rot = parent_global_rot.inv() * root_global_rot
        self.current_local[joint_index, 3:7] = target_local_rot.as_quat().astype(np.float32)
        self._sync_joint_euler_from_local(joint_index)
        self.instance.set_local_transforms(self.current_local)

    def _snap_root_rotation_to_world_90_grid(self) -> None:
        global_transforms = self.instance.compute_global_transforms()
        current_global_rot = R.from_quat(self._normalized_quat(global_transforms[0, 3:7]))
        current_euler_deg = current_global_rot.as_euler("xyz", degrees=True)
        snapped_euler_deg = np.round(current_euler_deg / 90.0) * 90.0
        target_global_rot = R.from_euler("xyz", snapped_euler_deg, degrees=True)

        world_rot = R.from_quat(self._normalized_quat(np.asarray(self._world_transform()[3:7], dtype=np.float64)))
        target_local_rot = world_rot.inv() * target_global_rot
        self.current_local[0, 3:7] = target_local_rot.as_quat().astype(np.float32)
        self._sync_joint_euler_from_local(0)
        self.instance.set_local_transforms(self.current_local)

    def _reset_pose(self) -> None:
        self.euler_deg[:] = 0.0
        self.root_offset[:] = 0.0
        self.current_local[:] = self.reference_local
        self.instance.set_local_transforms(self.current_local)

    def _save_to_path(self, path: Path) -> None:
        save_one_frame_bvh(self.soma_bvh, path, self.skeleton, self.current_local)
        metadata_path = path.with_suffix(path.suffix + ".json")
        metadata = {
            "template_bvh": str(self.soma_bvh),
            "output_bvh": str(path),
            "joint_rotation_offsets_xyz_deg": {
                name: self.euler_deg[idx].astype(float).tolist()
                for idx, name in enumerate(self.skeleton.joint_names)
                if np.linalg.norm(self.euler_deg[idx]) > 1e-6
            },
            "root_view_offset_m": self.root_offset.astype(float).tolist(),
            "note": "BVH stores the edited single-frame local joint transforms. root_view_offset_m is only a viewer offset.",
        }
        import json

        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.last_save_status = f"saved: {path}"

    def _save_as_dialog(self) -> None:
        import tkinter as tk
        from tkinter import filedialog as tk_filedialog

        root = tk.Tk()
        root.withdraw()
        initial_file = self.export_name if self.export_name.endswith(".bvh") else f"{self.export_name}.bvh"
        save_path = tk_filedialog.asksaveasfilename(
            title="Save SOMA one-frame BVH pose",
            initialdir=str(self.export_dir),
            initialfile=initial_file,
            defaultextension=".bvh",
            filetypes=[("BVH files", "*.bvh")],
        )
        root.destroy()
        if save_path:
            self._save_to_path(Path(save_path).expanduser().resolve())

    def _quick_save(self) -> None:
        filename = self.export_name if self.export_name.endswith(".bvh") else f"{self.export_name}.bvh"
        self._save_to_path((self.export_dir / filename).resolve())

    def step(self) -> None:
        self.time += self.frame_dt
        self.instance.xform = self._world_transform()

    def render(self) -> None:
        self.viewer.begin_frame(self.time)
        if self.show_skeleton:
            self.skeleton_renderer.draw(self.viewer, self.instance, 0)
        if self.show_axes:
            self.coordinate_renderer.draw(self.viewer, self.instance.compute_global_transforms(), _AXIS_SCALE, 0)
        if self.show_mesh and self.mesh_renderer is not None:
            self.mesh_renderer.draw(self.viewer, self.instance, self.instance.color, 0)
        if self.show_gizmo:
            self.viewer.log_gizmo("soma_pose", self.instance.xform)
        self.viewer.log_state(self.state)
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
        height = viewport.size.y - 2 * _PANEL_MARGIN
        ui.set_next_window_pos(ui.ImVec2(_PANEL_MARGIN, _PANEL_MARGIN))
        ui.set_next_window_size(ui.ImVec2(_PANEL_WIDTH, height))
        ui.set_next_window_bg_alpha(_PANEL_ALPHA)
        ui.begin(f"{self.source_name} Pose Editor", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))

        ui.text(f"Template: {self.soma_bvh.name}")
        ui.text(f"Joints: {self.skeleton.num_joints}")
        ui.text(f"Source mesh: {self.source_mesh_mode}")
        ui.separator()

        if self.mesh_renderer is not None:
            _, self.show_mesh = ui.checkbox("Show mesh", self.show_mesh)
        else:
            self.show_mesh = False
            ui.text("Show mesh: unavailable in source_mesh=none mode")
        if not self.show_mesh and self.mesh_renderer is not None:
            self.mesh_renderer.clear(self.viewer)
        _, self.show_skeleton = ui.checkbox("Show skeleton", self.show_skeleton)
        if not self.show_skeleton:
            self.skeleton_renderer.clear(self.viewer)
        _, self.show_axes = ui.checkbox("Show axes", self.show_axes)
        if not self.show_axes:
            self.coordinate_renderer.clear(self.viewer)
        _, self.show_gizmo = ui.checkbox("Show root gizmo", self.show_gizmo)

        if ui.button("Reset Pose"):
            self._reset_pose()
        ui.same_line()
        if ui.button("Quick Save"):
            try:
                self._quick_save()
            except Exception as exc:
                self.last_save_status = f"save failed: {exc}"
        ui.same_line()
        if ui.button("Save As"):
            try:
                self._save_as_dialog()
            except Exception as exc:
                self.last_save_status = f"save failed: {exc}"

        ui.text(f"Quick save dir: {self.export_dir}")
        ui.text(f"Quick save name: {self.export_name}")
        if self.last_save_status:
            ui.text(self.last_save_status)

        ui.separator()
        if ui.collapsing_header("Viewer Root Offset"):
            for axis_idx, axis_name in enumerate(("tx", "ty", "tz")):
                changed, value = self._slider_input_float(
                    ui, axis_name, float(self.root_offset[axis_idx]), -2.0, 2.0, "%.3f", _PANEL_WIDTH
                )
                if changed:
                    self.root_offset[axis_idx] = value

        ui.separator()
        for joint_index, joint_name in enumerate(self.skeleton.joint_names):
            if not ui.collapsing_header(joint_name):
                continue
            ui.push_id(joint_index)
            if joint_index == 0:
                ui.text("Root frame is the alignment reference.")
                if ui.button("Snap root rotation to world 90 deg grid"):
                    self._snap_root_rotation_to_world_90_grid()
            elif ui.button("Align rotation to root frame"):
                self._align_joint_rotation_to_root(joint_index)
            for axis_idx, axis_name in enumerate(("rx", "ry", "rz")):
                changed, value = self._slider_input_float(
                    ui,
                    axis_name,
                    float(self.euler_deg[joint_index, axis_idx]),
                    -180.0,
                    180.0,
                    "%.1f",
                    _PANEL_WIDTH,
                )
                if changed:
                    self.euler_deg[joint_index, axis_idx] = value
                    self._set_joint_euler(joint_index, self.euler_deg[joint_index])
            ui.pop_id()

        ui.end()


def parse_args():
    import newton.examples

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="gl")
    parser.add_argument("--soma_bvh", type=str, default=str(_DEFAULT_SOMA_BVH))
    parser.add_argument("--export_dir", type=str, default=str(_DEFAULT_EXPORT_DIR))
    parser.add_argument("--export_name", type=str, default="soma_custom_pose.bvh")
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
        help="Use 'none' for BVH skeletons that do not match the SOMA USD mesh.",
    )
    parser.add_argument(
        "--soma_facing_direction",
        choices=["Mujoco", "Maya"],
        default="Mujoco",
        help="Coordinate convention used only for viewing the SOMA mesh.",
    )
    return parser


def main() -> None:
    import newton.examples

    parser = parse_args()
    viewer, args = newton.examples.init(parser)
    with wp.ScopedDevice(args.device):
        app = SomaPoseEditor(viewer, args)
        app.run()


if __name__ == "__main__":
    main()
