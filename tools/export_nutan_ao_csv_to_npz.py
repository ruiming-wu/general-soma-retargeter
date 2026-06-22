#!/usr/bin/env python3
"""Export retargeted Nutan-to-AO CSV motions to one-motion NPZ files.

The input CSV format matches the retargeter/rendering tools in this repository:
root translations are stored in centimeters, root rotations are xyz Euler angles
in degrees, and joint columns end with ``_dof`` and are stored in degrees.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


DEFAULT_INPUT_ROOT = Path(
    "/home/ruiming.wu/codes/general-soma-retargeter/"
    "output/nutan_to_ao_tekken2_aligned/motions"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/ruiming.wu/codes/general-soma-retargeter/"
    "output/nutan_to_ao_tekken2_aligned/motion_npz"
)
DEFAULT_MJCF = Path(
    "/home/ruiming.wu/codes/H4/mjcf/agile_one_fixed_tekken2_hands_aligned.xml"
)

ROOT_POSITION_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")
ROOT_ROTATION_COLUMNS = ("root_rotateX", "root_rotateY", "root_rotateZ")
DEFAULT_SOURCE_FPS = 120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing retargeted motion CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where one NPZ per motion will be written.",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=DEFAULT_MJCF,
        help="AO MJCF used for forward kinematics.",
    )
    parser.add_argument("--glob", default="*.csv", help="Input glob relative to --input-root.")
    parser.add_argument("--fps", type=float, default=50.0, help="Target FPS stored in NPZ.")
    parser.add_argument(
        "--source-fps",
        type=float,
        default=DEFAULT_SOURCE_FPS,
        help="FPS of the input CSV samples before resampling.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Convert at most this many CSV files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NPZ files.")
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed. Default is uncompressed NPZ for faster loading.",
    )
    return parser.parse_args()


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return quat / norm


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return quat / norm


def canonicalize_quat_sequence_wxyz(quat: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so adjacent frames take the short path."""
    out = normalize_quat_wxyz(np.asarray(quat, dtype=np.float64)).copy()
    for i in range(1, out.shape[0]):
        if np.sum(out[i - 1] * out[i]) < 0.0:
            out[i] *= -1.0
    return out


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., 1:] *= -1.0
    return result


def quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quat_delta_to_rotvec_wxyz(delta: np.ndarray) -> np.ndarray:
    delta = normalize_quat_wxyz(delta)
    delta = np.where(delta[..., :1] < 0.0, -delta, delta)
    w = np.clip(delta[..., 0], -1.0, 1.0)
    xyz = delta[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    scale = np.divide(angle, sin_half, out=np.full_like(angle, 2.0), where=sin_half > 1e-8)
    return xyz * scale[..., None]


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    edge_order = 2 if values.shape[0] > 2 else 1
    return np.gradient(values, 1.0 / fps, axis=0, edge_order=edge_order).astype(np.float32)


def angular_velocity_from_quat_wxyz(quat: np.ndarray, fps: float) -> np.ndarray:
    quat = canonicalize_quat_sequence_wxyz(quat)
    num_frames = quat.shape[0]
    if num_frames <= 1:
        return np.zeros(quat.shape[:-1] + (3,), dtype=np.float32)

    ang_vel = np.zeros(quat.shape[:-1] + (3,), dtype=np.float64)
    dt = 1.0 / fps

    delta = quat_multiply_wxyz(quat[1], quat_conjugate_wxyz(quat[0]))
    ang_vel[0] = quat_delta_to_rotvec_wxyz(delta) / dt

    delta = quat_multiply_wxyz(quat[-1], quat_conjugate_wxyz(quat[-2]))
    ang_vel[-1] = quat_delta_to_rotvec_wxyz(delta) / dt

    if num_frames > 2:
        delta = quat_multiply_wxyz(quat[2:], quat_conjugate_wxyz(quat[:-2]))
        ang_vel[1:-1] = quat_delta_to_rotvec_wxyz(delta) / (2.0 * dt)

    return ang_vel.astype(np.float32)


def load_csv_motion(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding=None)
    if data.shape == ():
        data = data.reshape(1)

    names = data.dtype.names
    if names is None:
        raise ValueError(f"{path} has no CSV header")

    missing_root = [name for name in ROOT_POSITION_COLUMNS + ROOT_ROTATION_COLUMNS if name not in names]
    if missing_root:
        raise ValueError(f"{path} is missing root columns: {missing_root}")

    root_pos = np.stack([data[name] for name in ROOT_POSITION_COLUMNS], axis=1) / 100.0
    euler_deg = np.stack([data[name] for name in ROOT_ROTATION_COLUMNS], axis=1)
    root_quat_xyzw = R.from_euler("xyz", euler_deg, degrees=True).as_quat()

    joint_cols = [name for name in names if name.endswith("_dof")]
    if not joint_cols:
        raise ValueError(f"{path} has no *_dof joint columns")
    joint_names = [name[: -len("_dof")] for name in joint_cols]
    joint_pos = np.deg2rad(np.stack([data[name] for name in joint_cols], axis=1))

    return root_pos, root_quat_xyzw, joint_names, joint_pos


def resample_motion(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_pos: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample CSV motion arrays from source FPS to target FPS."""
    if root_pos.shape[0] != joint_pos.shape[0] or root_pos.shape[0] != root_quat_xyzw.shape[0]:
        raise ValueError("root and joint arrays have inconsistent frame counts")
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError(f"source_fps and target_fps must be positive, got {source_fps}, {target_fps}")

    num_source_frames = root_pos.shape[0]
    if num_source_frames <= 1 or abs(source_fps - target_fps) < 1e-6:
        return root_pos, normalize_quat_xyzw(root_quat_xyzw), joint_pos

    num_target_frames = max(1, int(round(num_source_frames * target_fps / source_fps)))
    source_times = np.arange(num_source_frames, dtype=np.float64) / float(source_fps)
    target_times = np.arange(num_target_frames, dtype=np.float64) / float(target_fps)
    target_times = np.minimum(target_times, source_times[-1])

    root_pos_rs = np.empty((num_target_frames, 3), dtype=np.float64)
    for axis in range(3):
        root_pos_rs[:, axis] = np.interp(target_times, source_times, root_pos[:, axis])

    joint_pos_rs = np.empty((num_target_frames, joint_pos.shape[1]), dtype=np.float64)
    for joint_idx in range(joint_pos.shape[1]):
        joint_pos_rs[:, joint_idx] = np.interp(target_times, source_times, joint_pos[:, joint_idx])

    root_quat_xyzw = normalize_quat_xyzw(root_quat_xyzw)
    slerp = Slerp(source_times, R.from_quat(root_quat_xyzw))
    root_quat_rs = slerp(target_times).as_quat()
    return root_pos_rs, normalize_quat_xyzw(root_quat_rs), joint_pos_rs


def model_body_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        names.append(name if name else f"body_{body_id}")
    return names


def make_model_qpos(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_names: list[str],
    joint_pos: np.ndarray,
) -> np.ndarray:
    if len(joint_names) != joint_pos.shape[1]:
        raise ValueError(
            f"joint_names has {len(joint_names)} entries, but joint_pos has {joint_pos.shape[1]} columns"
        )

    free_joint_ids = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    if len(free_joint_ids) != 1:
        raise ValueError(f"Expected exactly one free joint in MJCF, found {len(free_joint_ids)}")
    free_qpos_addr = int(model.jnt_qposadr[int(free_joint_ids[0])])

    qpos = np.tile(np.asarray(model.qpos0, dtype=np.float64), (root_pos.shape[0], 1))
    qpos[:, free_qpos_addr : free_qpos_addr + 3] = root_pos
    root_quat_xyzw = normalize_quat_xyzw(root_quat_xyzw)
    qpos[:, free_qpos_addr + 3 : free_qpos_addr + 7] = root_quat_xyzw[:, [3, 0, 1, 2]]

    missing: list[str] = []
    non_hinge: list[str] = []
    for joint_idx, joint_name in enumerate(joint_names):
        model_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if model_joint_id < 0:
            missing.append(joint_name)
            continue
        if int(model.jnt_type[model_joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            non_hinge.append(joint_name)
            continue
        qpos_addr = int(model.jnt_qposadr[model_joint_id])
        qpos[:, qpos_addr] = joint_pos[:, joint_idx]

    if missing:
        raise ValueError(f"MJCF is missing motion joints: {', '.join(missing)}")
    if non_hinge:
        raise ValueError(f"MJCF joints are not hinge joints: {', '.join(non_hinge)}")
    return qpos


def forward_kinematics(
    model: mujoco.MjModel, qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    num_frames = qpos.shape[0]
    num_bodies = model.nbody - 1
    body_pos_w = np.empty((num_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.empty((num_frames, num_bodies, 4), dtype=np.float32)

    for frame in range(num_frames):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        body_pos_w[frame] = data.xpos[1:].astype(np.float32)
        body_quat_w[frame] = data.xquat[1:].astype(np.float32)

    body_quat_w = canonicalize_quat_sequence_wxyz(body_quat_w).astype(np.float32)
    return body_pos_w, body_quat_w


def export_motion(
    csv_path: Path,
    output_path: Path,
    model: mujoco.MjModel,
    body_names: list[str],
    source_fps: float,
    fps: float,
    compressed: bool,
) -> dict[str, tuple[int, ...]]:
    root_pos, root_quat_xyzw, joint_names, joint_pos = load_csv_motion(csv_path)
    root_pos, root_quat_xyzw, joint_pos = resample_motion(
        root_pos, root_quat_xyzw, joint_pos, source_fps=source_fps, target_fps=fps
    )
    qpos = make_model_qpos(model, root_pos, root_quat_xyzw, joint_names, joint_pos)
    body_pos_w, body_quat_w = forward_kinematics(model, qpos)

    base_orientations = normalize_quat_xyzw(root_quat_xyzw)[:, [3, 0, 1, 2]].astype(np.float32)
    base_orientations = canonicalize_quat_sequence_wxyz(base_orientations).astype(np.float32)
    payload = {
        "joint_names": np.asarray(joint_names, dtype=str),
        "body_names": np.asarray(body_names, dtype=str),
        "fps": np.asarray(fps, dtype=np.float32),
        "base_positions": root_pos.astype(np.float32),
        "base_orientations": base_orientations,
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": finite_difference(joint_pos, fps),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": finite_difference(body_pos_w, fps),
        "body_ang_vel_w": angular_velocity_from_quat_wxyz(body_quat_w, fps),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(output_path, **payload)

    return {key: value.shape for key, value in payload.items()}


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    mjcf_path = args.mjcf.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF does not exist: {mjcf_path}")
    if args.fps <= 0.0:
        raise ValueError(f"fps must be positive, got {args.fps}")
    if args.source_fps <= 0.0:
        raise ValueError(f"source-fps must be positive, got {args.source_fps}")

    csv_paths = sorted(input_root.glob(args.glob))
    if args.limit is not None:
        csv_paths = csv_paths[: args.limit]
    if not csv_paths:
        raise FileNotFoundError(f"no CSV files matched {input_root / args.glob}")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    body_names = model_body_names(model)

    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"MJCF: {mjcf_path}")
    print(f"Source FPS: {args.source_fps:g}")
    print(f"Target FPS metadata/differentiation rate: {args.fps:g}")
    print(f"Body count (excluding world): {len(body_names)}")
    print(f"Converting {len(csv_paths)} motion CSV files")

    for csv_path in csv_paths:
        rel = csv_path.relative_to(input_root)
        output_path = output_root / rel.with_suffix(".npz")
        if output_path.exists() and not args.overwrite:
            print(f"SKIP existing {output_path}")
            continue
        shapes = export_motion(
            csv_path, output_path, model, body_names, args.source_fps, args.fps, args.compressed
        )
        print(
            f"WROTE {output_path} "
            f"frames={shapes['base_positions'][0]} joints={shapes['joint_pos'][1]} bodies={shapes['body_pos_w'][1]}"
        )


if __name__ == "__main__":
    main()
