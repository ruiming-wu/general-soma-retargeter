#!/usr/bin/env python3
"""View one retargeted robot CSV frame in MuJoCo with a debug ground plane.

The source H4 MJCF intentionally has no floor. This tool writes a temporary
MJCF with a visual floor at z=0 and a red marker at the lowest foot collision
point, then loads a selected CSV frame into qpos for inspection.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_XML = Path("/home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml")
DEFAULT_CSV = Path(
    "/home/ruiming.wu/codes/general-soma-retargeter/output/"
    "nutan_to_ao_h4_mjlab_triaxial_smooth5_first0_repeat3_heightfix_conservative_split/"
    "normal12/motions/test_nutan.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Base robot MJCF without debug floor.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Retargeted robot CSV to inspect.")
    parser.add_argument("--frame", type=int, default=0, help="CSV frame index to load.")
    parser.add_argument("--ground-size", type=float, default=5.0, help="Half-size of debug ground plane in meters.")
    parser.add_argument("--marker-radius", type=float, default=0.025, help="Lowest-point marker radius in meters.")
    parser.add_argument("--show-visual", action="store_true", help="Also show visual mesh geoms, not only collisions.")
    parser.add_argument("--check-only", action="store_true", help="Only compute foot bottom height; do not open viewer.")
    parser.add_argument(
        "--root-z-offset-cm",
        type=float,
        default=0.0,
        help="Temporary root_translateZ offset in CSV centimeters for inspection only.",
    )
    parser.add_argument(
        "--keep-temp-xml",
        action="store_true",
        help="Keep the generated MJCF path after exiting for manual inspection.",
    )
    return parser.parse_args()


def read_csv_row(path: Path, frame: int) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    if frame < 0:
        frame = len(rows) + frame
    if frame < 0 or frame >= len(rows):
        raise IndexError(f"frame {frame} out of range [0, {len(rows) - 1}] for {path}")
    return rows[frame]


def build_debug_xml(base_xml: Path, ground_size: float, marker_radius: float, keep: bool) -> Path:
    text = base_xml.read_text()
    meshdir = (base_xml.parent / "../meshes/visual").resolve()
    text = text.replace('meshdir="../meshes/visual"', f'meshdir="{meshdir}"')
    debug_geoms = f"""
    <geom name="debug_floor" type="plane" size="{ground_size} {ground_size} 0.01"
          pos="0 0 0" rgba="0.15 0.35 0.15 0.35" material="groundplane"
          contype="1" conaffinity="15" group="0" />
    <site name="debug_lowest_foot_point" type="sphere" size="{marker_radius}"
          pos="0 0 0" rgba="1 0.05 0.05 1" group="0" />
"""
    if "<worldbody>" not in text:
        raise ValueError(f"{base_xml} has no <worldbody> tag")
    text = text.replace("<worldbody>", "<worldbody>" + debug_geoms, 1)

    if keep:
        out = Path(tempfile.gettempdir()) / f"{base_xml.stem}_debug_floor.xml"
        out.write_text(text)
        return out
    tmp = tempfile.NamedTemporaryFile(prefix=f"{base_xml.stem}_debug_floor_", suffix=".xml", delete=False)
    out = Path(tmp.name)
    tmp.close()
    out.write_text(text)
    return out


def set_qpos_from_csv(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    row: dict[str, str],
    root_z_offset_cm: float = 0.0,
) -> None:
    qpos = np.asarray(model.qpos0, dtype=float).copy()
    root_pos = np.array(
        [float(row["root_translateX"]), float(row["root_translateY"]), float(row["root_translateZ"])],
        dtype=float,
    ) / 100.0
    root_pos[2] += float(root_z_offset_cm) / 100.0
    euler_deg = np.array(
        [float(row["root_rotateX"]), float(row["root_rotateY"]), float(row["root_rotateZ"])],
        dtype=float,
    )
    quat_xyzw = R.from_euler("xyz", euler_deg, degrees=True).as_quat()
    qpos[:3] = root_pos
    qpos[3:7] = quat_xyzw[[3, 0, 1, 2]]

    for col, value in row.items():
        if not col.endswith("_dof"):
            continue
        joint_name = col[: -len("_dof")]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"CSV joint {joint_name!r} does not exist in {model.names.decode(errors='ignore')[:0]}")
        qpos[int(model.jnt_qposadr[joint_id])] = np.deg2rad(float(value))

    data.qpos[:] = qpos
    data.qvel[:] = 0.0


def foot_collision_geom_ids(model: mujoco.MjModel) -> list[int]:
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if ("left_foot" in name or "right_foot" in name) and name.endswith("_collision"):
            ids.append(geom_id)
    if not ids:
        raise RuntimeError("No left/right foot collision geoms found")
    return ids


def capsule_lowest_point(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> tuple[float, np.ndarray]:
    pos = np.asarray(data.geom_xpos[geom_id], dtype=float)
    mat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    radius = float(model.geom_size[geom_id, 0])
    half_length = float(model.geom_size[geom_id, 1])
    axis = mat[:, 2]
    centerline_end = pos - np.sign(axis[2]) * half_length * axis
    point = centerline_end.copy()
    point[2] -= radius
    bottom_z = float(pos[2] - abs(axis[2]) * half_length - radius)
    point[2] = bottom_z
    return bottom_z, point


def update_lowest_marker(model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int]) -> tuple[str, float, np.ndarray]:
    candidates = []
    for geom_id in geom_ids:
        bottom_z, point = capsule_lowest_point(model, data, geom_id)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        candidates.append((bottom_z, point, name))
    bottom_z, point, name = min(candidates, key=lambda item: item[0])
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "debug_lowest_foot_point")
    if site_id >= 0:
        model.site_pos[site_id] = point
    return name, bottom_z, point


def configure_viewer(viewer, show_visual: bool) -> None:
    viewer.opt.geomgroup[:] = 0
    viewer.opt.geomgroup[0] = 1  # debug floor and marker
    viewer.opt.geomgroup[3] = 1  # collision geoms
    if show_visual:
        viewer.opt.geomgroup[2] = 1
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True


def main() -> None:
    args = parse_args()
    debug_xml = build_debug_xml(args.xml.resolve(), args.ground_size, args.marker_radius, args.keep_temp_xml)
    model = mujoco.MjModel.from_xml_path(str(debug_xml))
    data = mujoco.MjData(model)
    row = read_csv_row(args.csv, args.frame)
    set_qpos_from_csv(model, data, row, root_z_offset_cm=args.root_z_offset_cm)
    mujoco.mj_forward(model, data)
    foot_ids = foot_collision_geom_ids(model)
    lowest_name, lowest_z, lowest_point = update_lowest_marker(model, data, foot_ids)
    mujoco.mj_forward(model, data)

    print(f"debug_xml: {debug_xml}")
    print(f"csv: {args.csv}")
    print(f"frame: {args.frame}")
    print(f"lowest_foot_geom: {lowest_name}")
    print(f"lowest_foot_bottom_z_m: {lowest_z:.9f}")
    print(f"lowest_foot_point_xyz_m: {lowest_point.tolist()}")
    print(f"root_z_offset_cm_applied: {args.root_z_offset_cm:.9f}")
    print("foot_collision_geoms:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) for gid in foot_ids])
    if args.check_only:
        return

    from mujoco import viewer as mujoco_viewer

    with mujoco_viewer.launch_passive(model, data) as viewer:
        configure_viewer(viewer, args.show_visual)
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = data.qpos[:3]
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)

    if not args.keep_temp_xml:
        try:
            debug_xml.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
