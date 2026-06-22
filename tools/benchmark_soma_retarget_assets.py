#!/usr/bin/env python3
"""Benchmark SOMA retargeting on the bundled BVH assets.

The script evaluates SOMA -> robot retargeting with the retargeter's own IK
targets. It reports both strict-success aggregate metrics and all-samples
aggregate metrics when clip metrics are available.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import io
import json
import math
import multiprocessing as mp
import os
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm, trange

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.csv as csv_utils
import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.utils import newton_utils
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BVH_DIR = REPO_ROOT / "assets/motions/bvh"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output/retarget_benchmark/soma"
ROBOT_CHOICES = ("unitree_g1", "agile_one")

ROOT_JOINTS = {"Hips"}
HEAD_JOINTS = {"Head"}
WRIST_JOINTS = {"LeftHand", "RightHand"}
ANKLE_JOINTS = {"LeftFoot", "RightFoot"}
JUDGEMENT_POSITION_JOINTS = ROOT_JOINTS | WRIST_JOINTS | ANKLE_JOINTS
JUDGEMENT_ROTATION_JOINTS = ROOT_JOINTS | HEAD_JOINTS | WRIST_JOINTS | ANKLE_JOINTS
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


@dataclass
class OnlineStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min: float = math.inf
    max: float = -math.inf

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def merge(self, other: "OnlineStats") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.min = other.min
            self.max = other.max
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / total
        self.mean = (self.mean * self.count + other.mean * other.count) / total
        self.count = total
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)

    def as_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "min": None,
                "max": None,
                "c95_low": None,
                "c95_high": None,
                "c99_low": None,
                "c99_high": None,
            }
        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        sem = math.sqrt(variance / self.count) if self.count > 0 else 0.0
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "c95_low": self.mean - 1.96 * sem,
            "c95_high": self.mean + 1.96 * sem,
            "c99_low": self.mean - 2.576 * sem,
            "c99_high": self.mean + 2.576 * sem,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OnlineStats":
        obj = cls()
        obj.count = int(payload.get("count", 0))
        obj.mean = float(payload.get("mean", 0.0) or 0.0)
        obj.m2 = float(payload.get("m2", 0.0) or 0.0)
        obj.min = float(payload.get("min", math.inf))
        obj.max = float(payload.get("max", -math.inf))
        return obj

    def to_payload(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "min": self.min,
            "max": self.max,
        }


class StatsBank:
    def __init__(self) -> None:
        self._stats: dict[str, OnlineStats] = {}

    def update(self, key: str, value: float) -> None:
        self._stats.setdefault(key, OnlineStats()).update(value)

    def merge_payload(self, payload: dict[str, dict[str, Any]]) -> None:
        for key, value in payload.items():
            self._stats.setdefault(key, OnlineStats()).merge(OnlineStats.from_dict(value))

    def as_payload(self) -> dict[str, dict[str, float | int]]:
        return {key: value.to_payload() for key, value in sorted(self._stats.items())}

    def as_summary(self) -> dict[str, dict[str, float | int | None]]:
        return {key: value.as_dict() for key, value in sorted(self._stats.items())}


def quat_error_deg_xyzw(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> float:
    a = R.from_quat(a_xyzw)
    b = R.from_quat(b_xyzw)
    return float(np.degrees((b.inv() * a).magnitude()))


def joint_position_group(joint_name: str) -> str | None:
    if joint_name in ROOT_JOINTS:
        return "root"
    if joint_name in WRIST_JOINTS:
        return "wrist"
    if joint_name in ANKLE_JOINTS:
        return "ankle"
    return None


def joint_rotation_group(joint_name: str) -> str | None:
    if joint_name in ROOT_JOINTS:
        return "root"
    if joint_name in HEAD_JOINTS:
        return "head"
    if joint_name in WRIST_JOINTS:
        return "wrist"
    if joint_name in ANKLE_JOINTS:
        return "ankle"
    return None


def body_names(model) -> list[str]:
    return [newton_utils.get_name_from_label(label) for label in model.body_label]


def update_status_counts(counts: dict[str, int], result: dict[str, Any]) -> None:
    counts["done"] += 1
    if result.get("failure") is not None:
        counts["fail"] += 1
    elif result.get("warning") is not None:
        counts["warn"] += 1
    else:
        counts["ok"] += 1


def status_desc(prefix: str, counts: dict[str, int], total: int) -> str:
    return (
        f"{prefix} status\t"
        f"{GREEN}OK {counts['ok']:>7}{RESET}\t"
        f"{YELLOW}WARN {counts['warn']:>7}{RESET}\t"
        f"{RED}FAIL {counts['fail']:>7}{RESET}\t"
        f"DONE {counts['done']:>7}/{total:<7}"
    )


def cache_key(args: argparse.Namespace, robot: str) -> dict[str, Any]:
    return {
        "robot": robot,
        "max_frames": args.max_frames,
        "continue_after_failure": args.continue_after_failure,
        "rotation_fail_deg": args.rotation_fail_deg,
        "position_fail_m": args.position_fail_m,
        "rotation_warn_deg": args.rotation_warn_deg,
        "position_warn_m": args.position_warn_m,
        "retarget_source_facing_direction": args.retarget_source_facing_direction,
        "retargeter_config": str(args.retargeter_config.resolve()) if args.retargeter_config else "",
        "preserve_motion_subdirs": args.preserve_motion_subdirs,
        "metrics_policy": "judgement_targets_v3_all_status_clip_metrics",
    }


def result_matches_cache_key(result: dict[str, Any], key: dict[str, Any]) -> bool:
    return result.get("_cache_key") == key


def load_result_cache(path: Path, key: dict[str, Any], source_paths: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    cached: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_path = result.get("source_path")
            if source_path not in source_paths:
                continue
            if not result_matches_cache_key(result, key):
                continue
            cached[source_path] = result
    return cached


def append_result_cache(path: Path, results: list[dict[str, Any]], key: dict[str, Any]) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for result in results:
            payload = dict(result)
            payload["_cache_key"] = key
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def quiet_newton_pipeline_progress():
    original_trange = newton_pipeline.trange

    def disabled_trange(*args, **kwargs):
        kwargs["disable"] = True
        return trange(*args, **kwargs)

    newton_pipeline.trange = disabled_trange
    try:
        yield
    finally:
        newton_pipeline.trange = original_trange


@contextmanager
def quiet_internal_output(enabled: bool):
    if not enabled:
        yield
        return
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield


def make_exception_result(bvh_path: Path, robot: str, exc: BaseException) -> dict[str, Any]:
    failure = {
        "reason": "exception",
        "source_path": str(bvh_path),
        "robot": robot,
        "frame": "",
        "joint": "",
        "robot_body": "",
        "value": "",
        "threshold": "",
        "message": str(exc),
    }
    return {
        "status": "failed",
        "source_path": str(bvh_path),
        "robot": robot,
        "frames_evaluated": 0,
        "source_frames": 0,
        "sample_rate": "",
        "warning": None,
        "failure": failure,
        "motion_csv": "",
        "max_rotation_error_deg": "",
        "max_position_error_m": "",
        "max_key_position_error_m": "",
        "stats": {},
        "clip_metrics": {},
    }


def motion_csv_path_for(
    bvh_path: Path,
    motion_csv_dir: Path,
    bvh_root: Path | None,
    preserve_motion_subdirs: bool,
) -> Path:
    if preserve_motion_subdirs and bvh_root is not None:
        try:
            rel_path = bvh_path.relative_to(bvh_root)
        except ValueError:
            try:
                rel_path = bvh_path.absolute().relative_to(bvh_root.absolute())
            except ValueError:
                rel_path = Path(bvh_path.name)
        return (motion_csv_dir / rel_path).with_suffix(".csv")
    return motion_csv_dir / f"{bvh_path.stem}.csv"


def trim_animation(animation: AnimationBuffer, max_frames: int | None) -> AnimationBuffer:
    if max_frames is None or animation.num_frames <= max_frames:
        return animation
    return AnimationBuffer(
        animation.skeleton,
        max_frames,
        animation.sample_rate,
        np.copy(animation.local_transforms[:max_frames]),
    )


def evaluate_buffer(
    bvh_path: Path,
    robot: str,
    animation: AnimationBuffer,
    pipeline: newton_pipeline.NewtonPipeline,
    buffer,
    target_index: int,
    motion_csv_dir: Path | None,
    bvh_root: Path | None,
    preserve_motion_subdirs: bool,
    save_motion_csv: bool,
    continue_after_failure: bool,
    rotation_fail_deg: float,
    position_fail_m: float,
    rotation_warn_deg: float,
    position_warn_m: float,
) -> dict[str, Any]:
    motion_csv_path = None
    if motion_csv_dir is not None:
        motion_csv_path = motion_csv_path_for(bvh_path, motion_csv_dir, bvh_root, preserve_motion_subdirs)
        if save_motion_csv:
            motion_csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_utils.save_csv(str(motion_csv_path), buffer, csv_utils.get_csv_config(robot))

    targets = pipeline.input_targets[target_index]
    removed = pipeline.num_initialization_frames + pipeline.num_stabilization_frames
    usable_frames = min(buffer.num_frames, max(0, len(targets) - removed))
    if usable_frames == 0:
        raise RuntimeError("No aligned frames are available for evaluation")

    eval_model = pipeline._build_model(1)
    pipeline._apply_initial_robot_joint_positions(eval_model, 1)
    eval_state = eval_model.state()
    names = body_names(eval_model)

    stats = StatsBank()
    failure: dict[str, Any] | None = None
    warning: dict[str, Any] | None = None
    max_rotation = 0.0
    max_position = 0.0
    max_key_position = 0.0

    for frame_idx in range(usable_frames):
        q = np.asarray(buffer.get_data(frame_idx), dtype=np.float32)
        wp.copy(eval_model.joint_q, wp.array(q, dtype=wp.float32), 0, 0, len(q))
        newton.eval_fk(eval_model, eval_model.joint_q, eval_model.joint_qd, eval_state, None)
        robot_body_q = eval_state.body_q.numpy()
        frame_targets = targets[removed + frame_idx]

        for map_idx, joint_name in enumerate(pipeline.mapped_joints):
            target = np.asarray(frame_targets[map_idx], dtype=np.float64)
            pos_body_idx, pos_weight = pipeline.mapped_body_link_pos_data[map_idx]
            rot_body_idx, rot_weight = pipeline.mapped_body_link_rot_data[map_idx]

            if float(pos_weight) > 0.0:
                group = joint_position_group(joint_name)
                if group is not None:
                    pos_body = robot_body_q[pos_body_idx]
                    pos_error = float(np.linalg.norm(pos_body[:3] - target[:3]))
                    pos_body_name = names[pos_body_idx]
                    metric = f"position_error_m/{pos_body_name}->{joint_name}"
                    stats.update(metric, pos_error)
                    stats.update(f"{group}_position_error_m", pos_error)
                    max_position = max(max_position, pos_error)
                    max_key_position = max(max_key_position, pos_error)
                    if pos_error > position_fail_m and failure is None:
                        failure = {
                            "reason": f"{group}_position_error_threshold",
                            "source_path": str(bvh_path),
                            "robot": robot,
                            "frame": frame_idx,
                            "joint": joint_name,
                            "robot_body": pos_body_name,
                            "value": pos_error,
                            "threshold": position_fail_m,
                        }
                    elif pos_error > position_warn_m and warning is None:
                        warning = {
                            "reason": f"{group}_position_error_warning",
                            "source_path": str(bvh_path),
                            "robot": robot,
                            "frame": frame_idx,
                            "joint": joint_name,
                            "robot_body": pos_body_name,
                            "value": pos_error,
                            "threshold": position_warn_m,
                        }

            if float(rot_weight) > 0.0:
                group = joint_rotation_group(joint_name)
                if group is None:
                    continue
                rot_body = robot_body_q[rot_body_idx]
                rot_error = quat_error_deg_xyzw(rot_body[3:7], target[3:7])
                rot_body_name = names[rot_body_idx]
                metric = f"rotation_error_deg/{rot_body_name}->{joint_name}"
                stats.update(metric, rot_error)
                stats.update("rotation_error_deg/all_judgement_targets", rot_error)
                stats.update(f"{group}_rotation_error_deg", rot_error)
                max_rotation = max(max_rotation, rot_error)
                if rot_error > rotation_fail_deg and failure is None:
                    failure = {
                        "reason": f"{group}_rotation_error_threshold",
                        "source_path": str(bvh_path),
                        "robot": robot,
                        "frame": frame_idx,
                        "joint": joint_name,
                        "robot_body": rot_body_name,
                        "value": rot_error,
                        "threshold": rotation_fail_deg,
                    }
                elif rot_error > rotation_warn_deg and warning is None:
                    warning = {
                        "reason": f"{group}_rotation_error_warning",
                        "source_path": str(bvh_path),
                        "robot": robot,
                        "frame": frame_idx,
                        "joint": joint_name,
                        "robot_body": rot_body_name,
                        "value": rot_error,
                        "threshold": rotation_warn_deg,
                    }

        if failure is not None and not continue_after_failure:
            break

    status = "failed" if failure else "warning" if warning else "succeeded"
    return {
        "status": status,
        "source_path": str(bvh_path),
        "robot": robot,
        "frames_evaluated": usable_frames if failure is None or continue_after_failure else failure["frame"] + 1,
        "source_frames": animation.num_frames,
        "sample_rate": animation.sample_rate,
        "warning": None if failure is not None else warning,
        "failure": failure,
        "motion_csv": str(motion_csv_path) if motion_csv_path is not None else "",
        "max_rotation_error_deg": max_rotation,
        "max_position_error_m": max_position,
        "max_key_position_error_m": max_key_position,
        "stats": stats.as_payload(),
        "clip_metrics": stats.as_summary(),
    }


def evaluate_batch(
    bvh_paths: list[Path],
    robot: str,
    max_frames: int | None,
    retarget_source_facing_direction: str,
    motion_csv_dir: Path | None,
    bvh_root: Path | None,
    preserve_motion_subdirs: bool,
    resume: bool,
    continue_after_failure: bool,
    rotation_fail_deg: float,
    position_fail_m: float,
    rotation_warn_deg: float,
    position_warn_m: float,
    retargeter_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not bvh_paths:
        return []

    existing_paths: list[Path] = []
    missing_paths: list[Path] = []
    if resume and motion_csv_dir is not None:
        for bvh_path in bvh_paths:
            motion_csv_path = motion_csv_path_for(bvh_path, motion_csv_dir, bvh_root, preserve_motion_subdirs)
            if motion_csv_path.exists() and motion_csv_path.stat().st_size > 0:
                existing_paths.append(bvh_path)
            else:
                missing_paths.append(bvh_path)
    else:
        missing_paths = bvh_paths

    results: list[dict[str, Any]] = []
    if existing_paths:
        results.extend(
                evaluate_existing_csv_batch(
                bvh_paths=existing_paths,
                robot=robot,
                max_frames=max_frames,
                retarget_source_facing_direction=retarget_source_facing_direction,
                motion_csv_dir=motion_csv_dir,
                bvh_root=bvh_root,
                preserve_motion_subdirs=preserve_motion_subdirs,
                continue_after_failure=continue_after_failure,
                rotation_fail_deg=rotation_fail_deg,
                position_fail_m=position_fail_m,
                rotation_warn_deg=rotation_warn_deg,
                position_warn_m=position_warn_m,
                retargeter_config=retargeter_config,
            )
        )

    if not missing_paths:
        return results

    try:
        skeleton, first_animation = bvh_utils.load_bvh(str(missing_paths[0]))
        animations = [trim_animation(first_animation, max_frames)]
        for bvh_path in missing_paths[1:]:
            _, animation = bvh_utils.load_bvh(str(bvh_path), skeleton)
            animations.append(trim_animation(animation, max_frames))

        source_converter = SpaceConverter(get_facing_direction_type_from_str(retarget_source_facing_direction))
        source_transform = source_converter.transform(wp.transform_identity())
        pipeline = newton_pipeline.NewtonPipeline(skeleton, "soma", robot, retarget_config=retargeter_config)
        with quiet_newton_pipeline_progress():
            pipeline.add_input_motions(animations, [source_transform] * len(animations), True)
            buffers = pipeline.execute()
        if not buffers or len(buffers) != len(animations):
            raise RuntimeError(f"NewtonPipeline returned {len(buffers) if buffers else 0} buffers for {len(animations)} inputs")
    except Exception as exc:
        return results + [make_exception_result(path, robot, exc) for path in missing_paths]

    for idx, (bvh_path, animation, buffer) in enumerate(zip(missing_paths, animations, buffers, strict=True)):
        try:
            results.append(
                evaluate_buffer(
                    bvh_path=bvh_path,
                    robot=robot,
                    animation=animation,
                    pipeline=pipeline,
                    buffer=buffer,
                    target_index=idx,
                    motion_csv_dir=motion_csv_dir,
                    bvh_root=bvh_root,
                    preserve_motion_subdirs=preserve_motion_subdirs,
                    save_motion_csv=True,
                    continue_after_failure=continue_after_failure,
                    rotation_fail_deg=rotation_fail_deg,
                    position_fail_m=position_fail_m,
                    rotation_warn_deg=rotation_warn_deg,
                    position_warn_m=position_warn_m,
                )
            )
        except Exception as exc:
            results.append(make_exception_result(bvh_path, robot, exc))

    return results


def evaluate_existing_csv_batch(
    bvh_paths: list[Path],
    robot: str,
    max_frames: int | None,
    retarget_source_facing_direction: str,
    motion_csv_dir: Path,
    bvh_root: Path | None,
    preserve_motion_subdirs: bool,
    continue_after_failure: bool,
    rotation_fail_deg: float,
    position_fail_m: float,
    rotation_warn_deg: float,
    position_warn_m: float,
    retargeter_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not bvh_paths:
        return []

    try:
        skeleton, first_animation = bvh_utils.load_bvh(str(bvh_paths[0]))
        animations = [trim_animation(first_animation, max_frames)]
        for bvh_path in bvh_paths[1:]:
            _, animation = bvh_utils.load_bvh(str(bvh_path), skeleton)
            animations.append(trim_animation(animation, max_frames))

        source_converter = SpaceConverter(get_facing_direction_type_from_str(retarget_source_facing_direction))
        source_transform = source_converter.transform(wp.transform_identity())
        pipeline = newton_pipeline.NewtonPipeline(skeleton, "soma", robot, retarget_config=retargeter_config)
        with quiet_newton_pipeline_progress():
            pipeline.add_input_motions(animations, [source_transform] * len(animations), True)
    except Exception as exc:
        return [make_exception_result(path, robot, exc) for path in bvh_paths]

    results = []
    csv_config = csv_utils.get_csv_config(robot)
    for idx, (bvh_path, animation) in enumerate(zip(bvh_paths, animations, strict=True)):
        try:
            motion_csv_path = motion_csv_path_for(bvh_path, motion_csv_dir, bvh_root, preserve_motion_subdirs)
            buffer = csv_utils.load_csv(str(motion_csv_path), fps=animation.sample_rate, csv_config=csv_config)
            results.append(
                evaluate_buffer(
                    bvh_path=bvh_path,
                    robot=robot,
                    animation=animation,
                    pipeline=pipeline,
                    buffer=buffer,
                    target_index=idx,
                    motion_csv_dir=motion_csv_dir,
                    bvh_root=bvh_root,
                    preserve_motion_subdirs=preserve_motion_subdirs,
                    save_motion_csv=False,
                    continue_after_failure=continue_after_failure,
                    rotation_fail_deg=rotation_fail_deg,
                    position_fail_m=position_fail_m,
                    rotation_warn_deg=rotation_warn_deg,
                    position_warn_m=position_warn_m,
                )
            )
        except Exception as exc:
            results.append(make_exception_result(bvh_path, robot, exc))

    return results


def process_batch_job(args: tuple[list[Path], str, argparse.Namespace]) -> list[dict[str, Any]]:
    bvh_paths, robot, ns = args
    retargeter_config = None
    if getattr(ns, "retargeter_config", None):
        with Path(ns.retargeter_config).open("r", encoding="utf-8") as f:
            retargeter_config = json.load(f)
    with quiet_internal_output(not ns.verbose_internal_output):
        return evaluate_batch(
            bvh_paths=bvh_paths,
            robot=robot,
            max_frames=ns.max_frames,
            retarget_source_facing_direction=ns.retarget_source_facing_direction,
            motion_csv_dir=Path(ns.motion_csv_dir) if ns.motion_csv_dir else None,
            bvh_root=Path(ns.bvh_dir),
            preserve_motion_subdirs=ns.preserve_motion_subdirs,
            resume=ns.resume,
            continue_after_failure=ns.continue_after_failure,
            rotation_fail_deg=ns.rotation_fail_deg,
            position_fail_m=ns.position_fail_m,
            rotation_warn_deg=ns.rotation_warn_deg,
            position_warn_m=ns.position_warn_m,
            retargeter_config=retargeter_config,
        )


def process_batch_job_isolated(job: tuple[list[Path], str, argparse.Namespace]) -> list[dict[str, Any]]:
    """Run one batch in a short-lived child so Warp/Newton memory is released."""

    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        mp_context=ctx,
        max_tasks_per_child=1,
    ) as executor:
        future = executor.submit(process_batch_job, job)
        return future.result()


def process_batch_job_with_fallback(
    job: tuple[list[Path], str, argparse.Namespace],
    use_process_isolation: bool,
) -> list[dict[str, Any]]:
    bvh_paths, robot, ns = job
    try:
        if use_process_isolation:
            return process_batch_job_isolated(job)
        return process_batch_job(job)
    except Exception as exc:
        if len(bvh_paths) <= 1:
            return [make_exception_result(bvh_paths[0], robot, exc)]
        mid = len(bvh_paths) // 2
        print(
            f"[soma-benchmark] batch size {len(bvh_paths)} crashed; "
            f"retrying as {mid}+{len(bvh_paths) - mid}. error={exc}",
            flush=True,
        )
        left = process_batch_job_with_fallback((bvh_paths[:mid], robot, ns), use_process_isolation)
        right = process_batch_job_with_fallback((bvh_paths[mid:], robot, ns), use_process_isolation)
        return left + right


METRIC_FIELDS = ["source_path", "robot", "metric", "count", "mean", "min", "max", "c95_low", "c95_high", "c99_low", "c99_high"]
CLIP_FIELDS = [
    "status",
    "source_path",
    "robot",
    "frames_evaluated",
    "source_frames",
    "sample_rate",
    "motion_csv",
    "warning_reason",
    "failure_reason",
    "max_rotation_error_deg",
    "max_position_error_m",
    "max_key_position_error_m",
]
EVENT_FIELDS = ["source_path", "robot", "frame", "reason", "joint", "robot_body", "value", "threshold", "message"]


def write_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = METRIC_FIELDS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_clip_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = CLIP_FIELDS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_event_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = EVENT_FIELDS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


class StreamingReportWriter:
    """Write per-clip reports immediately and keep only aggregate state in RAM."""

    def __init__(self, summary_dir: Path) -> None:
        self.clip_csv = summary_dir / "clips.csv"
        self.metric_csv = summary_dir / "clip_metrics.csv"
        self.failure_csv = summary_dir / "failures.csv"
        self.warning_csv = summary_dir / "warnings.csv"
        self.aggregate = StatsBank()
        self.aggregate_all = StatsBank()
        self.total = 0
        self.succeeded = 0
        self.failed = 0
        self.warned = 0
        self.failures_preview: list[dict[str, Any]] = []
        self.warnings_preview: list[dict[str, Any]] = []
        self._files = [
            self.clip_csv.open("w", newline="", encoding="utf-8"),
            self.metric_csv.open("w", newline="", encoding="utf-8"),
            self.failure_csv.open("w", newline="", encoding="utf-8"),
            self.warning_csv.open("w", newline="", encoding="utf-8"),
        ]
        self._clip_writer = csv.DictWriter(self._files[0], fieldnames=CLIP_FIELDS)
        self._metric_writer = csv.DictWriter(self._files[1], fieldnames=METRIC_FIELDS)
        self._failure_writer = csv.DictWriter(self._files[2], fieldnames=EVENT_FIELDS)
        self._warning_writer = csv.DictWriter(self._files[3], fieldnames=EVENT_FIELDS)
        for writer in (self._clip_writer, self._metric_writer, self._failure_writer, self._warning_writer):
            writer.writeheader()

    def close(self) -> None:
        for f in self._files:
            f.close()

    def flush(self) -> None:
        for f in self._files:
            f.flush()

    def record_batch(self, results: list[dict[str, Any]]) -> None:
        for result in results:
            self.record_result(result)
        self.flush()

    def record_result(self, result: dict[str, Any]) -> None:
        failure = result.get("failure")
        warning = result.get("warning")
        self.total += 1
        if failure is not None:
            self.failed += 1
            self._failure_writer.writerow({key: failure.get(key, "") for key in EVENT_FIELDS})
            if len(self.failures_preview) < 30:
                self.failures_preview.append(failure)
        elif warning is not None:
            self.warned += 1
            self._warning_writer.writerow({key: warning.get(key, "") for key in EVENT_FIELDS})
            if len(self.warnings_preview) < 30:
                self.warnings_preview.append(warning)
        else:
            self.succeeded += 1
            self.aggregate.merge_payload(result["stats"])
        if result.get("stats"):
            self.aggregate_all.merge_payload(result["stats"])

        self._clip_writer.writerow(
            {
                "status": result["status"],
                "source_path": result["source_path"],
                "robot": result["robot"],
                "frames_evaluated": result["frames_evaluated"],
                "source_frames": result["source_frames"],
                "sample_rate": result["sample_rate"],
                "motion_csv": result.get("motion_csv", ""),
                "warning_reason": warning.get("reason") if warning else "",
                "failure_reason": failure.get("reason") if failure else "",
                "max_rotation_error_deg": result["max_rotation_error_deg"],
                "max_position_error_m": result["max_position_error_m"],
                "max_key_position_error_m": result["max_key_position_error_m"],
            }
        )
        for metric, values in result.get("clip_metrics", {}).items():
            self._metric_writer.writerow(
                {
                    "source_path": result["source_path"],
                    "robot": result["robot"],
                    "metric": metric,
                    **values,
                }
            )


def append_event_section(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.extend(["", f"## {title}", ""])
    if not rows:
        lines.append("无。")
        return

    lines.extend([
        "| 文件 | 机器人 | 帧 | 原因 | 目标 | 机器人 body | 数值 | 阈值 |",
        "|---|---:|---:|---|---|---|---:|---:|",
    ])
    for row in rows[:30]:
        file_name = Path(row.get("source_path", "")).name
        target = row.get("joint", "")
        value = row.get("value", "")
        threshold = row.get("threshold", "")
        value_text = "" if value == "" else f"{float(value):.6g}"
        threshold_text = "" if threshold == "" else f"{float(threshold):.6g}"
        lines.append(
            f"| `{file_name}` | `{row.get('robot', '')}` | `{row.get('frame', '')}` | "
            f"`{row.get('reason', '')}` | `{target}` | `{row.get('robot_body', '')}` | "
            f"`{value_text}` | `{threshold_text}` |"
        )
    if len(rows) > 30:
        lines.append(f"\n仅显示前 30 条。完整列表见 CSV。")


def fmt_metric_value(value: Any) -> str:
    return "" if value is None else f"{float(value):.6g}"


def fmt_metric_interval(values: dict[str, Any], low_key: str, high_key: str) -> str:
    low = values.get(low_key)
    high = values.get(high_key)
    if low is None or high is None:
        return ""
    return f"[{float(low):.6g}, {float(high):.6g}]"


def append_metrics_section(
    lines: list[str],
    metrics: dict[str, dict[str, Any]],
    title: str = "聚合指标",
    scope_text: str = "统计范围：仅严格成功 clips；仅统计判定目标：头部旋转、双腕位置/旋转、root 位置/旋转、双踝位置/旋转。",
) -> None:
    lines.extend(["", f"## {title}", ""])
    if not metrics:
        lines.append("没有 clips 可用于聚合统计。")
        return

    lines.extend([
        scope_text,
        "",
        "| 指标 | 样本数 | 均值 | C95 区间 | C99 区间 | 最小-最大区间 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, values in metrics.items():
        lines.append(
            f"| `{name}` | `{values.get('count', 0)}` | `{fmt_metric_value(values.get('mean'))}` | "
            f"`{fmt_metric_interval(values, 'c95_low', 'c95_high')}` | "
            f"`{fmt_metric_interval(values, 'c99_low', 'c99_high')}` | "
            f"`{fmt_metric_interval(values, 'min', 'max')}` |"
        )


def write_summary_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# SOMA 重定向评估",
        "",
        f"- 创建时间：`{report['created_at']}`",
        f"- BVH 目录：`{report['bvh_dir']}`",
        f"- 机器人：`{', '.join(report['robots'])}`",
        f"- Batch size：`{report['batch_size']}`",
        f"- Resume：`{report['resume']}`",
        f"- 复用已有 motion CSV：`{report['resume_existing_motion_csvs']}`",
        f"- 复用缓存结果：`{report['resume_cached_results']}`",
        f"- Clips 总数：`{report['counts']['clips_total']}`",
        f"- 成功：`{report['counts']['clips_succeeded']}` (`{report['rates']['success_rate']:.4f}`)",
        f"- 警告：`{report['counts']['clips_warned']}` (`{report['rates']['warning_rate']:.4f}`)",
        f"- 失败：`{report['counts']['clips_failed']}` (`{report['rates']['failure_rate']:.4f}`)",
        "",
        "## 判定目标",
        "",
        "- 头部：旋转",
        "- 双腕：位置和旋转",
        "- Root/Hips：位置和旋转",
        "- 双踝：位置和旋转",
        "- 其他辅助追踪关节：不参与成功/警告/失败判定，也不进入聚合统计",
        "",
        "## 阈值",
        "",
        f"- 失败旋转阈值：`{report['thresholds']['rotation_fail_deg']} deg`",
        f"- 失败位置阈值：`{report['thresholds']['position_fail_m']} m`",
        f"- 警告旋转阈值：`{report['thresholds']['rotation_warn_deg']} deg`",
        f"- 警告位置阈值：`{report['thresholds']['position_warn_m']} m`",
        "",
        "## 文件",
        "",
        f"- Clip CSV：`{report['clip_csv']}`",
        f"- Metric CSV：`{report['metric_csv']}`",
        f"- Failures CSV：`{report['failure_csv']}`",
        f"- Warnings CSV：`{report['warning_csv']}`",
        f"- Result cache JSONL：`{report['result_cache_jsonl']}`",
        f"- JSON：`{report['summary_json']}`",
    ]
    append_metrics_section(lines, report["aggregate_metrics"])
    append_metrics_section(
        lines,
        report.get("aggregate_metrics_all_samples", {}),
        title="全样本聚合指标",
        scope_text=(
            "统计范围：所有已成功完成指标评估的 clips，包括 succeeded、warning 和 continue-after-failure "
            "下保留完整指标的 failed clips；仅统计判定目标。"
        ),
    )
    append_event_section(lines, "失败样例", report["failures"])
    append_event_section(lines, "警告样例", report["warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-dir", type=Path, default=DEFAULT_BVH_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--robot", choices=(*ROBOT_CHOICES, "all"), default="all")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of BVH clips to retarget in one NewtonPipeline call. This enables SOMA's single-GPU multi-env path.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-motion-csv", action="store_true", help="Save one retargeted robot CSV per BVH clip.")
    parser.add_argument(
        "--preserve-motion-subdirs",
        action="store_true",
        help="Save motion CSVs under motions/<relative BVH path>. This avoids stem collisions across GRAB subjects.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run. Existing motion CSVs are reused and re-evaluated so final metrics cover all clips.",
    )
    parser.add_argument(
        "--resume-summary-dir",
        type=Path,
        default=None,
        help="Existing timestamp summary directory to resume in-place, e.g. output/.../soma_g1/20260429_181454.",
    )
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="Continue evaluating a clip after the first failure so full motion CSVs can be written.",
    )
    parser.add_argument("--rotation-fail-deg", type=float, default=90.0)
    parser.add_argument("--position-fail-m", type=float, default=0.5)
    parser.add_argument("--rotation-warn-deg", type=float, default=45.0)
    parser.add_argument("--position-warn-m", type=float, default=0.25)
    parser.add_argument(
        "--retarget-source-facing-direction",
        choices=("Mujoco", "Maya"),
        default="Mujoco",
        help="Source BVH coordinate convention. The official converter config uses Mujoco for SEED BVH files.",
    )
    parser.add_argument(
        "--retargeter-config",
        type=Path,
        default=None,
        help="Optional explicit retargeter JSON config. Useful for A/B tests without replacing the default config file.",
    )
    parser.add_argument(
        "--verbose-internal-output",
        action="store_true",
        help="Show verbose output from BVH loading/Newton internals. Disabled by default so tqdm bars stay at the bottom.",
    )
    parser.add_argument(
        "--no-batch-process-isolation",
        action="store_true",
        help=(
            "Run all SOMA batches in the main process. Faster startup, but long runs can keep growing "
            "RSS because Warp/Newton/NumPy allocators do not reliably return memory to the OS."
        ),
    )
    return parser.parse_args()


def chunked(items: list[Path], size: int) -> list[list[Path]]:
    if size < 1:
        raise ValueError("--batch-size must be >= 1")
    return [items[i:i + size] for i in range(0, len(items), size)]


def latest_summary_dir(output_root: Path) -> Path:
    candidates = [p for p in output_root.iterdir() if p.is_dir()] if output_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"--resume could not find an existing timestamp directory under {output_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_robot(args: argparse.Namespace, bvh_files: list[Path], robot: str, timestamp: str) -> dict[str, Any]:
    if args.resume_summary_dir is not None:
        summary_dir = args.resume_summary_dir / robot if args.robot == "all" else args.resume_summary_dir
    elif args.resume:
        summary_dir = latest_summary_dir(args.output_root)
        if args.robot == "all":
            summary_dir = summary_dir / robot
    else:
        summary_dir = args.output_root / timestamp / robot if args.robot == "all" else args.output_root / timestamp
    summary_dir.mkdir(parents=True, exist_ok=True)
    args.motion_csv_dir = str(summary_dir / "motions") if args.save_motion_csv else ""
    result_cache_jsonl = summary_dir / "clip_results.jsonl"
    current_cache_key = cache_key(args, robot)
    source_paths = {str(path) for path in bvh_files}
    cached_results_by_source = (
        load_result_cache(result_cache_jsonl, current_cache_key, source_paths)
        if args.resume
        else {}
    )
    uncached_bvh_files = [path for path in bvh_files if str(path) not in cached_results_by_source]

    batches = chunked(uncached_bvh_files, args.batch_size)
    jobs = [(batch, robot, args) for batch in batches]
    started = time.time()
    existing_motion_csvs = 0
    if args.resume and args.motion_csv_dir:
        motion_csv_dir = Path(args.motion_csv_dir)
        existing_motion_csvs = sum(
            1
            for path in bvh_files
            if motion_csv_path_for(path, motion_csv_dir, args.bvh_dir, args.preserve_motion_subdirs).exists()
        )
    print(
        f"[soma-benchmark] clips={len(bvh_files)} robot={robot} "
        f"batches={len(jobs)} batch_size={args.batch_size} resume={args.resume} "
        f"existing_csv={existing_motion_csvs} cached_results={len(cached_results_by_source)} "
        f"batch_process_isolation={not args.no_batch_process_isolation}"
    )
    print(f"[soma-benchmark] output={summary_dir}")

    clip_csv = summary_dir / "clips.csv"
    metric_csv = summary_dir / "clip_metrics.csv"
    failure_csv = summary_dir / "failures.csv"
    warning_csv = summary_dir / "warnings.csv"
    summary_json = summary_dir / "summary.json"
    summary_md = summary_dir / "summary.md"

    report_writer = StreamingReportWriter(summary_dir)
    status_counts = {"ok": 0, "warn": 0, "fail": 0, "done": 0}
    try:
        cached_results = list(cached_results_by_source.values())
        if cached_results:
            report_writer.record_batch(cached_results)
            for result in cached_results:
                update_status_counts(status_counts, result)
        initial_done = report_writer.total
        with tqdm(
            total=len(bvh_files),
            initial=initial_done,
            desc=f"soma {robot}",
            dynamic_ncols=True,
            position=0,
            leave=True,
        ) as progress, tqdm(
            total=0,
            bar_format="{desc}",
            dynamic_ncols=True,
            position=1,
            leave=True,
        ) as status:
            status.set_description_str(status_desc("soma", status_counts, len(bvh_files)), refresh=True)
            for job in jobs:
                if args.no_batch_process_isolation:
                    batch_results = process_batch_job_with_fallback(job, use_process_isolation=False)
                else:
                    batch_results = process_batch_job_with_fallback(job, use_process_isolation=True)
                report_writer.record_batch(batch_results)
                append_result_cache(result_cache_jsonl, batch_results, current_cache_key)
                for result in batch_results:
                    update_status_counts(status_counts, result)
                progress.update(len(batch_results))
                status.set_description_str(status_desc("soma", status_counts, len(bvh_files)), refresh=True)
                del batch_results
                gc.collect()
    finally:
        report_writer.close()

    total = report_writer.total
    failed = report_writer.failed
    warned = report_writer.warned
    succeeded = report_writer.succeeded
    report = {
        "script": str(Path(__file__).resolve()),
        "created_at": timestamp,
        "bvh_dir": str(args.bvh_dir.resolve()),
        "robots": [robot],
        "summary_dir": str(summary_dir),
        "clip_csv": str(clip_csv),
        "metric_csv": str(metric_csv),
        "failure_csv": str(failure_csv),
        "warning_csv": str(warning_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "result_cache_jsonl": str(result_cache_jsonl),
        "elapsed_sec": time.time() - started,
        "batch_size": args.batch_size,
        "batch_process_isolation": not args.no_batch_process_isolation,
        "resume": args.resume,
        "resume_existing_motion_csvs": existing_motion_csvs,
        "resume_cached_results": len(cached_results_by_source),
        "thresholds": {
            "rotation_fail_deg": args.rotation_fail_deg,
            "position_fail_m": args.position_fail_m,
            "rotation_warn_deg": args.rotation_warn_deg,
            "position_warn_m": args.position_warn_m,
        },
        "retarget_source_facing_direction": args.retarget_source_facing_direction,
        "retargeter_config": str(args.retargeter_config.resolve()) if args.retargeter_config else "",
        "preserve_motion_subdirs": args.preserve_motion_subdirs,
        "judgement_targets": {
            "rotation": sorted(JUDGEMENT_ROTATION_JOINTS),
            "position": sorted(JUDGEMENT_POSITION_JOINTS),
        },
        "counts": {
            "clips_total": total,
            "clips_succeeded": succeeded,
            "clips_failed": failed,
            "clips_warned": warned,
        },
        "rates": {
            "success_rate": succeeded / total if total else 0.0,
            "failure_rate": failed / total if total else 0.0,
            "warning_rate": warned / total if total else 0.0,
        },
        "aggregate_metrics_scope": "strict_succeeded_clips_only_judgement_targets_only",
        "aggregate_metrics": report_writer.aggregate.as_summary(),
        "aggregate_metrics_all_samples_scope": "all_statuses_with_available_clip_metrics_judgement_targets_only",
        "aggregate_metrics_all_samples": report_writer.aggregate_all.as_summary(),
        "failures": report_writer.failures_preview,
        "warnings": report_writer.warnings_preview,
        "failures_csv_is_complete": True,
        "warnings_csv_is_complete": True,
        "summary_event_lists_are_previews": True,
    }
    summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary_md(summary_md, report)
    print(f"[soma-benchmark] success={succeeded} fail={failed} warn={warned}")
    print(f"[soma-benchmark] summary={summary_dir}")
    return report


def main() -> None:
    args = parse_args()
    if args.resume:
        args.save_motion_csv = True
    if args.resume_summary_dir is not None:
        args.resume = True
        args.save_motion_csv = True
    bvh_files = sorted(args.bvh_dir.rglob("*.bvh"))
    if not bvh_files:
        raise FileNotFoundError(f"No BVH files found under {args.bvh_dir}")

    robots = list(ROBOT_CHOICES) if args.robot == "all" else [args.robot]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports = [run_robot(args, bvh_files, robot, timestamp) for robot in robots]
    print("[soma-benchmark] completed summaries:")
    for report in reports:
        print(f"[soma-benchmark] {report['robots'][0]} -> {report['summary_dir']}")


if __name__ == "__main__":
    main()
