#!/usr/bin/env python3
"""Verify Nutan-to-AO NPZ exports against their source CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from export_nutan_ao_csv_to_npz import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_MJCF,
    DEFAULT_OUTPUT_ROOT,
    angular_velocity_from_quat_wxyz,
    finite_difference,
    forward_kinematics,
    load_csv_motion,
    make_model_qpos,
    model_body_names,
    normalize_quat_xyzw,
    resample_motion,
)


REQUIRED_KEYS = (
    "joint_names",
    "body_names",
    "fps",
    "base_positions",
    "base_orientations",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--npz-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--glob", default="*.csv")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--source-fps", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--quat-atol-deg", type=float, default=1e-3)
    return parser.parse_args()


def quat_angle_error_deg_wxyz(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    a = a / np.where(a_norm < 1e-8, 1.0, a_norm)
    b = b / np.where(b_norm < 1e-8, 1.0, b_norm)
    dots = np.abs(np.sum(a * b, axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    return float(np.max(2.0 * np.arccos(dots)) * 180.0 / np.pi)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def assert_close(name: str, value: float, atol: float) -> None:
    if value > atol:
        raise AssertionError(f"{name} max error {value:.6g} exceeds tolerance {atol:.6g}")


def verify_motion(
    csv_path: Path,
    npz_path: Path,
    model: mujoco.MjModel,
    expected_body_names: list[str],
    source_fps: float,
    fps: float,
    atol: float,
    quat_atol_deg: float,
) -> dict[str, float]:
    if not npz_path.exists():
        raise FileNotFoundError(f"missing NPZ for {csv_path.name}: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as npz:
        missing = [key for key in REQUIRED_KEYS if key not in npz]
        if missing:
            raise AssertionError(f"{npz_path} is missing keys: {missing}")

        root_pos, root_quat_xyzw, joint_names, joint_pos = load_csv_motion(csv_path)
        root_pos, root_quat_xyzw, joint_pos = resample_motion(
            root_pos, root_quat_xyzw, joint_pos, source_fps=source_fps, target_fps=fps
        )
        qpos = make_model_qpos(model, root_pos, root_quat_xyzw, joint_names, joint_pos)
        body_pos_w, body_quat_w = forward_kinematics(model, qpos)

        expected_base_orientations = normalize_quat_xyzw(root_quat_xyzw)[:, [3, 0, 1, 2]]
        expected_joint_vel = finite_difference(joint_pos, fps)
        expected_body_lin_vel = finite_difference(body_pos_w, fps)
        expected_body_ang_vel = angular_velocity_from_quat_wxyz(body_quat_w, fps)

        npz_joint_names = [str(name) for name in npz["joint_names"]]
        npz_body_names = [str(name) for name in npz["body_names"]]
        if npz_joint_names != joint_names:
            raise AssertionError(f"{npz_path} joint_names differ from CSV columns")
        if npz_body_names != expected_body_names:
            raise AssertionError(f"{npz_path} body_names differ from MJCF body order")

        expected_shapes = {
            "base_positions": root_pos.shape,
            "base_orientations": expected_base_orientations.shape,
            "joint_pos": joint_pos.shape,
            "joint_vel": joint_pos.shape,
            "body_pos_w": body_pos_w.shape,
            "body_quat_w": body_quat_w.shape,
            "body_lin_vel_w": body_pos_w.shape,
            "body_ang_vel_w": body_pos_w.shape,
        }
        for key, shape in expected_shapes.items():
            if npz[key].shape != shape:
                raise AssertionError(f"{npz_path} {key} shape {npz[key].shape} != expected {shape}")
            if not np.all(np.isfinite(npz[key])):
                raise AssertionError(f"{npz_path} {key} contains non-finite values")

        errors = {
            "fps": abs(float(npz["fps"]) - fps),
            "base_pos_m": max_abs(npz["base_positions"], root_pos),
            "base_quat_deg": quat_angle_error_deg_wxyz(npz["base_orientations"], expected_base_orientations),
            "joint_pos_rad": max_abs(npz["joint_pos"], joint_pos),
            "joint_vel_rad_s": max_abs(npz["joint_vel"], expected_joint_vel),
            "body_pos_m": max_abs(npz["body_pos_w"], body_pos_w),
            "body_quat_deg": quat_angle_error_deg_wxyz(npz["body_quat_w"], body_quat_w),
            "body_lin_vel_m_s": max_abs(npz["body_lin_vel_w"], expected_body_lin_vel),
            "body_ang_vel_rad_s": max_abs(npz["body_ang_vel_w"], expected_body_ang_vel),
        }

    assert_close("fps", errors["fps"], atol)
    assert_close("base_pos_m", errors["base_pos_m"], atol)
    assert_close("joint_pos_rad", errors["joint_pos_rad"], atol)
    assert_close("joint_vel_rad_s", errors["joint_vel_rad_s"], atol)
    assert_close("body_pos_m", errors["body_pos_m"], atol)
    assert_close("body_lin_vel_m_s", errors["body_lin_vel_m_s"], atol)
    assert_close("body_ang_vel_rad_s", errors["body_ang_vel_rad_s"], atol)
    if errors["base_quat_deg"] > quat_atol_deg:
        raise AssertionError(
            f"base_quat_deg max error {errors['base_quat_deg']:.6g} exceeds tolerance {quat_atol_deg:.6g}"
        )
    if errors["body_quat_deg"] > quat_atol_deg:
        raise AssertionError(
            f"body_quat_deg max error {errors['body_quat_deg']:.6g} exceeds tolerance {quat_atol_deg:.6g}"
        )
    return errors


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    npz_root = args.npz_root.expanduser().resolve()
    mjcf_path = args.mjcf.expanduser().resolve()

    csv_paths = sorted(input_root.glob(args.glob))
    if args.limit is not None:
        csv_paths = csv_paths[: args.limit]
    if not csv_paths:
        raise FileNotFoundError(f"no CSV files matched {input_root / args.glob}")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    expected_body_names = model_body_names(model)
    worst: dict[str, float] = {}

    for csv_path in csv_paths:
        npz_path = npz_root / csv_path.relative_to(input_root).with_suffix(".npz")
        errors = verify_motion(
            csv_path,
            npz_path,
            model,
            expected_body_names,
            args.source_fps,
            args.fps,
            args.atol,
            args.quat_atol_deg,
        )
        for key, value in errors.items():
            worst[key] = max(worst.get(key, 0.0), value)
        print(
            f"OK {csv_path.name}: "
            f"base_pos={errors['base_pos_m']:.3g}m joint={errors['joint_pos_rad']:.3g}rad "
            f"body_pos={errors['body_pos_m']:.3g}m body_quat={errors['body_quat_deg']:.3g}deg"
        )

    print("Worst errors:")
    for key in sorted(worst):
        print(f"  {key}: {worst[key]:.9g}")


if __name__ == "__main__":
    main()
