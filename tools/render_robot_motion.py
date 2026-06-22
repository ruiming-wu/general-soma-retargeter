#!/usr/bin/env python3
"""Render retargeted robot motions from flat CSV or SONIC motionlib PKL.

The script is intentionally independent from the interactive Newton viewer so
it can be used in batch/debug sessions.  It loads a retargeted motion, writes
root pose + joint DOFs directly into the robot MJCF qpos, and records an MP4.

Examples:
    micromamba run -n gmr_env python tools/render_robot_motion.py \
        /home/ruiming.wu/data/seed-retargeted/ao_motionlib/robot_motionlib_slow2x/object_interaction/foo.pkl

    micromamba run -n gmr_env python tools/render_robot_motion.py \
        /home/ruiming.wu/data/seed-retargeted/gmr_g1/basic_locomotion_neutral/motions/foo.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

# Prefer EGL for headless/offscreen rendering.  This must be set before importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency message
    raise SystemExit("cv2 is required. Run this with gmr_env or install opencv-python.") from exc

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - dependency message
    raise SystemExit("mujoco is required. Run this with gmr_env or install mujoco.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MJCF = {
    "unitree_g1": Path(
        "/home/ruiming.wu/codes/GR00T-WholeBodyControl/"
        "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
    ),
    "agile_one": Path(
        "/home/ruiming.wu/codes/H4/mjcf/"
        "agile_one_fixed_tekken2_hands_aligned_body_only.xml"
    ),
}

FALLBACK_MJCF = {
    "agile_one": Path(
        "/home/ruiming.wu/codes/H4/mjcf/agile_one_fixed_tekken2_hands_aligned.xml"
    ),
}

CSV_ROBOT_HINT_COLUMNS = {
    "unitree_g1": {"waist_roll_joint_dof", "waist_pitch_joint_dof", "left_elbow_joint_dof"},
    "agile_one": {"head_yaw_joint_dof", "head_pitch_joint_dof", "left_elbow_roll_joint_dof"},
}

PATH_ROBOT_HINTS = {
    "unitree_g1": ("g1", "unitree"),
    "agile_one": ("ao", "agile_one", "agile-one", "agile"),
}

FOOT_BODY_CANDIDATES = {
    "unitree_g1": {
        "left": ["left_ankle_roll_link", "left_ankle_pitch_link"],
        "right": ["right_ankle_roll_link", "right_ankle_pitch_link"],
    },
    "agile_one": {
        "left": ["left_ankle_pitch_link", "left_ankle_roll_link"],
        "right": ["right_ankle_pitch_link", "right_ankle_roll_link"],
    },
}


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def infer_robot(path: Path, header: list[str] | None = None) -> str | None:
    """Infer robot type from CSV header or path names."""
    if header is not None:
        header_set = set(header)
        matches = [
            robot for robot, hints in CSV_ROBOT_HINT_COLUMNS.items() if hints.intersection(header_set)
        ]
        if len(matches) == 1:
            return matches[0]

    parts = [p.lower() for p in path.parts]
    # Prefer more specific AO hints before "g1", because some AO folders may include generic names.
    for robot in ("agile_one", "unitree_g1"):
        if any(any(hint == part or hint in part for hint in PATH_ROBOT_HINTS[robot]) for part in parts):
            return robot
    return None


def _load_joblib(path: Path) -> Any:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - dependency message
        raise SystemExit("joblib is required to read compressed motionlib PKLs.") from exc
    return joblib.load(path)


def _motion_entry_from_pkl(path: Path) -> tuple[str, dict[str, Any]]:
    payload = _load_joblib(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported PKL payload type: {type(payload)!r}")

    required = {"root_trans_offset", "root_rot", "dof"}
    if required.issubset(payload.keys()):
        return path.stem, payload

    entries = [(k, v) for k, v in payload.items() if isinstance(v, dict) and required.issubset(v.keys())]
    if len(entries) != 1:
        raise ValueError(
            f"Expected one motionlib entry with keys {sorted(required)}, found {len(entries)}"
        )
    return str(entries[0][0]), entries[0][1]


def load_motion(
    path: Path, robot_arg: str, fps_arg: float | None
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, list[str] | None, float]:
    """Load motion and return name, robot, root_pos, root_quat_xyzw, dof_rad, joint_names, fps."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        header = _read_csv_header(path)
        robot = robot_arg if robot_arg != "auto" else infer_robot(path, header)
        if robot is None:
            raise ValueError("Could not infer robot from CSV header/path; pass --robot unitree_g1|agile_one")

        data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding=None)
        if data.shape == ():
            data = data.reshape(1)

        root_pos = np.stack(
            [data["root_translateX"], data["root_translateY"], data["root_translateZ"]], axis=1
        ).astype(np.float32) / 100.0
        euler_deg = np.stack(
            [data["root_rotateX"], data["root_rotateY"], data["root_rotateZ"]], axis=1
        )
        root_quat_xyzw = R.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float32)

        joint_cols = [name for name in data.dtype.names or [] if name.endswith("_dof")]
        joint_names = [name[: -len("_dof")] for name in joint_cols]
        dof_rad = np.deg2rad(np.stack([data[name] for name in joint_cols], axis=1)).astype(np.float32)
        fps = float(fps_arg or 120.0)
        return path.stem, robot, root_pos, root_quat_xyzw, dof_rad, joint_names, fps

    if suffix == ".pkl":
        motion_name, entry = _motion_entry_from_pkl(path)
        robot = robot_arg if robot_arg != "auto" else infer_robot(path)
        if robot is None:
            # PKL entries do not currently store joint names or robot type, so path inference is
            # the only safe automatic route for same-DoF robots.
            raise ValueError("Could not infer robot from PKL path; pass --robot unitree_g1|agile_one")

        root_pos = np.asarray(entry["root_trans_offset"], dtype=np.float32)
        root_quat_xyzw = np.asarray(entry["root_rot"], dtype=np.float32)
        dof_rad = np.asarray(entry["dof"], dtype=np.float32)
        fps = float(fps_arg or entry.get("fps", 30.0))
        return motion_name, robot, root_pos, root_quat_xyzw, dof_rad, None, fps

    raise ValueError(f"Unsupported input suffix: {suffix}. Expected .csv or .pkl")


def resolve_mjcf(robot: str, override: str | None) -> Path:
    if override:
        path = Path(override).expanduser()
    else:
        path = DEFAULT_MJCF[robot]
        if not path.exists() and robot in FALLBACK_MJCF:
            path = FALLBACK_MJCF[robot]
    if not path.exists():
        raise FileNotFoundError(f"MJCF not found for {robot}: {path}")
    return path


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return quat / norm


def make_qpos(root_pos: np.ndarray, root_quat_xyzw: np.ndarray, dof_rad: np.ndarray) -> np.ndarray:
    qpos = np.zeros((root_pos.shape[0], 7 + dof_rad.shape[1]), dtype=np.float64)
    qpos[:, 0:3] = root_pos
    quat = normalize_quat_xyzw(root_quat_xyzw.astype(np.float64))
    qpos[:, 3:7] = quat[:, [3, 0, 1, 2]]  # MuJoCo freejoint quaternion is wxyz.
    qpos[:, 7:] = dof_rad
    return qpos


def make_model_qpos(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    dof_rad: np.ndarray,
    joint_names: list[str] | None,
) -> np.ndarray:
    """Build model-sized qpos and place motion joints by MJCF joint name when available."""
    if joint_names is None:
        return make_qpos(root_pos, root_quat_xyzw, dof_rad)

    if len(joint_names) != dof_rad.shape[1]:
        raise ValueError(f"joint_names has {len(joint_names)} entries, but dof has {dof_rad.shape[1]} columns")

    qpos = np.tile(np.asarray(model.qpos0, dtype=np.float64), (root_pos.shape[0], 1))
    qpos[:, 0:3] = root_pos
    quat = normalize_quat_xyzw(root_quat_xyzw.astype(np.float64))
    qpos[:, 3:7] = quat[:, [3, 0, 1, 2]]  # MuJoCo freejoint quaternion is wxyz.

    missing: list[str] = []
    for dof_idx, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            missing.append(joint_name)
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        if int(model.jnt_type[joint_id]) != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Expected hinge joint for {joint_name}, got type={int(model.jnt_type[joint_id])}")
        qpos[:, qpos_addr] = dof_rad[:, dof_idx]

    if missing:
        raise ValueError(f"MJCF is missing motion joints: {', '.join(missing)}")
    return qpos


def ensure_qpos_compatible(model: mujoco.MjModel, qpos: np.ndarray) -> None:
    if model.nq != qpos.shape[1]:
        raise ValueError(f"MJCF nq={model.nq}, but motion qpos has {qpos.shape[1]} columns")


def body_id(model: mujoco.MjModel, name: str) -> int | None:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return idx if idx >= 0 else None


def choose_foot_bodies(model: mujoco.MjModel, robot: str) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for side, candidates in FOOT_BODY_CANDIDATES[robot].items():
        result[side] = None
        for name in candidates:
            idx = body_id(model, name)
            if idx is not None:
                result[side] = idx
                break
    return result


def set_camera(
    cam: mujoco.MjvCamera,
    qpos: np.ndarray,
    frame: int,
    follow: bool,
    distance: float,
    azimuth: float,
    elevation: float,
) -> None:
    root = qpos[frame, 0:3]
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    if follow:
        cam.lookat[:] = [root[0], root[1], max(0.75, root[2] * 0.65)]


def render_motion(
    qpos: np.ndarray,
    robot: str,
    mjcf_path: Path,
    output_path: Path,
    video_fps: float,
    width: int,
    height: int,
    stride: int,
    max_frames: int | None,
    start_frame: int,
    end_frame: int | None,
    follow: bool,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    ensure_qpos_compatible(model, qpos)

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(video_fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    foot_ids = choose_foot_bodies(model, robot)
    foot_z: dict[str, list[float]] = {"left": [], "right": []}
    root_z: list[float] = []

    start = max(0, int(start_frame))
    end = qpos.shape[0] if end_frame is None else min(qpos.shape[0], max(start + 1, int(end_frame)))
    frame_indices = list(range(start, end, max(1, stride)))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]

    try:
        for out_i, frame in enumerate(frame_indices):
            data.qpos[:] = qpos[frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            set_camera(
                cam,
                qpos,
                frame,
                follow=follow,
                distance=camera_distance,
                azimuth=camera_azimuth,
                elevation=camera_elevation,
            )
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            root_z.append(float(data.qpos[2]))
            for side, idx in foot_ids.items():
                if idx is not None:
                    foot_z[side].append(float(data.xpos[idx, 2]))
    finally:
        writer.release()
        renderer.close()

    return {
        "robot": robot,
        "mjcf_path": str(mjcf_path),
        "output_video": str(output_path),
        "num_source_frames": int(qpos.shape[0]),
        "num_rendered_frames": int(len(frame_indices)),
        "start_frame": int(start),
        "end_frame_exclusive": int(end),
        "video_fps": float(video_fps),
        "stride": int(stride),
        "root_z": summarize_series(root_z),
        "foot_body_z": {side: summarize_series(values) for side, values in foot_z.items()},
    }


def summarize_series(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())}


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="Input flat robot CSV or SONIC motionlib PKL.")
    parser.add_argument("--robot", choices=["auto", "unitree_g1", "agile_one"], default="auto")
    parser.add_argument("--mjcf", type=str, default=None, help="Override robot MJCF path.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--output", type=Path, default=None, help="Override output MP4 path.")
    parser.add_argument("--fps", type=float, default=None, help="Override source motion FPS.")
    parser.add_argument("--video-fps", type=float, default=None, help="Output video FPS. Defaults to source FPS.")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth source frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap rendered frames for quick checks.")
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame to render.")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive source frame where rendering stops.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-follow", action="store_true", help="Use fixed camera lookat after initialization.")
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_path = args.motion.expanduser().resolve()
    if not motion_path.exists():
        raise FileNotFoundError(motion_path)

    motion_name, robot, root_pos, root_quat_xyzw, dof_rad, joint_names, source_fps = load_motion(
        motion_path, args.robot, args.fps
    )
    if dof_rad.shape[1] != 29:
        raise ValueError(f"Expected 29 DOFs, got {dof_rad.shape[1]}")

    mjcf_path = resolve_mjcf(robot, args.mjcf)
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    qpos = make_model_qpos(model, root_pos, root_quat_xyzw, dof_rad, joint_names)

    video_fps = float(args.video_fps or source_fps)
    output_path = args.output
    if output_path is None:
        output_path = args.output_root / robot / f"{safe_name(motion_name)}.mp4"
    output_path = output_path.expanduser().resolve()

    print(
        f"[render] motion={motion_path}\n"
        f"[render] robot={robot} frames={qpos.shape[0]} source_fps={source_fps:g} video_fps={video_fps:g}\n"
        f"[render] mjcf={mjcf_path}\n"
        f"[render] output={output_path}"
    )

    summary = render_motion(
        qpos=qpos,
        robot=robot,
        mjcf_path=mjcf_path,
        output_path=output_path,
        video_fps=video_fps,
        width=args.width,
        height=args.height,
        stride=args.stride,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        follow=not args.no_follow,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
    )
    summary["input_motion"] = str(motion_path)
    summary["motion_name"] = motion_name
    summary["source_fps"] = float(source_fps)

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[render] wrote {output_path}")
    print(f"[render] wrote {json_path}")
    print("[render] height summary:")
    print(json.dumps({k: summary[k] for k in ("root_z", "foot_body_z")}, indent=2))


if __name__ == "__main__":
    main()
