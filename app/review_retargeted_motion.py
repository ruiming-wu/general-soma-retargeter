#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive visual review tool for human source motions and robot CSV retargets.

The tool scans one or more roots, matches human motions and retargeted robot CSVs
by file stem, and opens an interactive Newton viewer for manual screening.

Current rendering support is intentionally conservative:
  * human BVH/SOMA sources are renderable;
  * SMPL/SMPLX-like files are indexed for matching/manifesting, but require a
    converted SOMA BVH with the same stem to be displayed.

Example:
    .venv/bin/python app/review_retargeted_motion.py \
        --soma-root /home/ruiming.wu/data/dataset/human_motionlib/seed_soma_bvh/household \
        --smpl-root /home/ruiming.wu/data/dataset/human_motionlib/seed_smpl_filtered/household \
        --robot-root /home/ruiming.wu/data/dataset/ao_motionlib/seed/household/body_motion \
        --output output/motion_review
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.csv as csv_utils
import soma_retargeter.pipelines.utils as pipeline_utils
import soma_retargeter.utils.newton_utils as newton_utils
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.renderers.coordinate_renderer import CoordinateRenderer
from soma_retargeter.renderers.mesh_renderer import SkeletalMeshRenderer
from soma_retargeter.renderers.skeleton_renderer import SkeletonRenderer
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str


_PANEL_MARGIN = 10
_PANEL_ALPHA = 0.92
_TOP_PANEL_HEIGHT = 210
_PLAYBACK_PANEL_HEIGHT = 130
_DEFAULT_UI_SCALE = 1.25
_DEFAULT_HUMAN_COLOR = (235.0 / 255.0, 245.0 / 255.0, 112.0 / 255.0)
_DEFAULT_CAMERA_TARGET = np.array([0.0, 0.0, 0.9], dtype=np.float32)
_DEFAULT_CAMERA_DISTANCE = 5.0
_DEFAULT_CAMERA_HEIGHT = 1.7
_DEFAULT_CAMERA_LEFT_FRONT_DEG = 45.0
_REVIEW_ROBOT = "agile_one"
_REVIEW_FIELDNAMES = [
    "time_unix",
    "index",
    "key",
    "label",
    "reason",
    "playback_time",
    "human_path",
    "robot_path",
    "robot_type",
]
_MAYBE_REASONS = [
    "Questionable Motion",
    "Self-Collision Risk",
    "Retargeting Artifact",
    "Other",
]
_REJECT_REASONS = [
    "Inappropriate Motion",
    "Self-Collision",
    "Retargeting Error",
    "Other",
]


def enable_cpu_only_viewer_fallback() -> None:
    """Let Newton's GL viewer run on machines without a CUDA driver."""

    try:
        has_cuda = any(getattr(device, "is_cuda", False) for device in wp.get_devices())
    except Exception:
        has_cuda = False
    if has_cuda:
        return

    original_empty = wp.empty

    def empty_without_cpu_pinned(*args, **kwargs):
        if kwargs.get("pinned") is True:
            device = kwargs.get("device")
            if device is None or str(device) == "cpu" or getattr(device, "is_cpu", False):
                kwargs = dict(kwargs)
                kwargs["pinned"] = False
        return original_empty(*args, **kwargs)

    wp.empty = empty_without_cpu_pinned
    print("[review] CUDA device not available; using non-pinned CPU viewer buffers.")


_LABEL_COLORS = {
    "keep": (0.30, 0.90, 0.42, 1.0),
    "maybe": (1.00, 0.78, 0.20, 1.0),
    "reject": (1.00, 0.30, 0.25, 1.0),
    "unmarked": (0.80, 0.80, 0.80, 1.0),
}
_LABEL_BUTTON_COLORS = {
    "keep": (
        (0.12, 0.50, 0.22, 1.0),
        (0.18, 0.66, 0.30, 1.0),
        (0.08, 0.40, 0.17, 1.0),
        (1.00, 1.00, 1.00, 1.0),
    ),
    "maybe": (
        (0.82, 0.58, 0.10, 1.0),
        (0.95, 0.70, 0.18, 1.0),
        (0.68, 0.46, 0.06, 1.0),
        (0.08, 0.07, 0.03, 1.0),
    ),
    "reject": (
        (0.62, 0.12, 0.10, 1.0),
        (0.82, 0.18, 0.15, 1.0),
        (0.48, 0.08, 0.07, 1.0),
        (1.00, 1.00, 1.00, 1.0),
    ),
}
_HUMAN_SUFFIXES = {".bvh", ".smpl", ".smplx", ".npz", ".pkl"}
_ROBOT_CSV_SKIP_NAMES = {
    "clips.csv",
    "clip_metrics.csv",
    "summary.csv",
    "metrics.csv",
    "review_manifest.csv",
    "review_labels.csv",
}
_HUMAN_PRIORITY = {
    ".bvh": 0,
    ".smpl": 1,
    ".smplx": 1,
    ".npz": 2,
    ".pkl": 2,
}
_CSV_ROBOT_HINT_COLUMNS = {
    "unitree_g1": {"waist_roll_joint_dof", "waist_pitch_joint_dof", "left_elbow_joint_dof"},
    "agile_one": {"head_yaw_joint_dof", "head_pitch_joint_dof", "left_elbow_roll_joint_dof"},
}
_PATH_ROBOT_HINTS = {
    "unitree_g1": ("g1", "unitree"),
    "agile_one": ("ao", "agile_one", "agile-one", "agile"),
}
_DEFAULT_AO_FULL_HANDS_MJCF = (
    Path(__file__).resolve().parents[1] / "assets/robots/agile_one_fixed_tekken2_hands_aligned.xml",
    Path(__file__).resolve().parents[1].parent / "H4/mjcf/agile_one_fixed_tekken2_hands_aligned.xml",
    Path.home() / "motion_review/H4/mjcf/agile_one_fixed_tekken2_hands_aligned.xml",
    Path("/home/ruiming.wu/codes/H4/mjcf/agile_one_fixed_tekken2_hands_aligned.xml"),
    Path(
        "/home/ruiming.wu/codes/GR00T-WholeBodyControl/"
        "gear_sonic/data/assets/robot_description/mjcf/agile_one_fixed_tekken2_hands_aligned.xml"
    ),
)
_DEFAULT_G1_MJCF = (
    Path(
        "/home/ruiming.wu/codes/GR00T-WholeBodyControl/"
        "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
    ),
)


@dataclass(frozen=True)
class MotionPair:
    key: str
    human_path: Path
    robot_path: Path
    robot_type: str

    @property
    def renderable(self) -> bool:
        return self.human_path.suffix.lower() == ".bvh"

    @property
    def name(self) -> str:
        return self.human_path.stem


def _path_parts_with_stem(path: Path) -> list[str]:
    parts = list(path.parts)
    parts[-1] = path.stem
    return parts


def _strip_robot_motion_channel(parts: list[str]) -> list[str]:
    return [part for part in parts if part not in {"body_motion", "left_hand_motion", "right_hand_motion"}]


def motion_match_key(path: Path, root: Path | None = None) -> str:
    """Return a dataset-aware matching key for human and robot motion files.

    The old stem-only key silently collapsed GRAB motions that share the same
    file name across subjects. These keys preserve dataset/category/subject
    identity while keeping single-folder review backwards compatible.
    """

    if root is not None:
        try:
            rel_parts = _path_parts_with_stem(path.relative_to(root))
            return "/".join(_strip_robot_motion_channel(rel_parts))
        except ValueError:
            pass

    parts = _path_parts_with_stem(path)
    marker_replacements = {
        "seed_soma_bvh": "seed",
        "seed_smpl_filtered": "seed",
        "grab_soma_bvh": "grab",
        "grab_smplx": "grab",
    }
    for marker, replacement in marker_replacements.items():
        if marker in parts:
            idx = parts.index(marker)
            return "/".join([replacement, *_strip_robot_motion_channel(parts[idx + 1 :])])

    for marker in ("ao_motionlib", "g1_motionlib"):
        if marker in parts:
            idx = parts.index(marker)
            return "/".join(_strip_robot_motion_channel(parts[idx + 1 :]))
    return path.stem


def _read_csv_header(path: Path) -> list[str] | None:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return next(csv.reader(f))
    except Exception:
        return None


def _is_robot_motion_csv(path: Path) -> bool:
    if path.name in _ROBOT_CSV_SKIP_NAMES:
        return False
    header = _read_csv_header(path)
    if not header:
        return False
    header_set = set(header)
    return {
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
    }.issubset(header_set) and any(name.endswith("_dof") for name in header)


def infer_robot_type(path: Path, header: list[str] | None = None) -> str | None:
    if header is not None:
        header_set = set(header)
        matches = [
            robot for robot, hints in _CSV_ROBOT_HINT_COLUMNS.items() if hints.intersection(header_set)
        ]
        if len(matches) == 1:
            return matches[0]

    parts = [part.lower() for part in path.parts]
    for robot in ("agile_one", "unitree_g1"):
        if any(any(hint == part or hint in part for hint in _PATH_ROBOT_HINTS[robot]) for part in parts):
            return robot
    return None


def _append_candidate(mapping: dict[str, list[Path]], path: Path, root: Path | None = None) -> None:
    mapping.setdefault(motion_match_key(path, root), []).append(path)


def collect_human_stems(roots: list[Path]) -> tuple[set[str], dict]:
    keys: set[str] = set()
    files = 0
    suffix_counts: dict[str, int] = {}
    missing_roots: list[str] = []
    for root in roots:
        if not root.exists():
            missing_roots.append(str(root))
            continue
        for path in sorted(root.rglob("*")):
            suffix = path.suffix.lower()
            if path.is_file() and suffix in _HUMAN_SUFFIXES:
                keys.add(motion_match_key(path, root))
                files += 1
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return keys, {
        "roots": [str(root) for root in roots],
        "missing_roots": missing_roots,
        "files": files,
        "unique_keys": len(keys),
        "suffix_counts": suffix_counts,
    }


def infer_required_human_roots(human_roots: list[Path]) -> list[Path]:
    required: list[Path] = []
    for root in human_roots:
        parts = list(root.parts)
        if "seed_soma_bvh" not in parts:
            continue
        idx = parts.index("seed_soma_bvh")
        candidate = Path(*parts[:idx], "seed_smpl_filtered", *parts[idx + 1 :])
        if candidate.exists():
            required.append(candidate)
    return required


def _choose_human(candidates: list[Path]) -> Path:
    return sorted(
        candidates,
        key=lambda p: (_HUMAN_PRIORITY.get(p.suffix.lower(), 99), len(p.parts), str(p)),
    )[0]


def _choose_robot(candidates: list[Path], robot: str) -> tuple[Path, str] | None:
    typed: list[tuple[Path, str]] = []
    for path in sorted(candidates):
        inferred = infer_robot_type(path, _read_csv_header(path))
        if inferred is None:
            continue
        if robot != "auto" and inferred != robot:
            continue
        typed.append((path, inferred))
    if not typed:
        return None
    if robot == "auto":
        robot_types = {robot_type for _, robot_type in typed}
        if len(robot_types) > 1:
            raise ValueError(
                "Matched both agile_one and unitree_g1 CSVs. Pass --robot agile_one or --robot unitree_g1."
            )
    return typed[0]


def find_motion_pairs(human_roots: list[Path], robot_roots: list[Path], robot: str) -> tuple[list[MotionPair], dict]:
    human: dict[str, list[Path]] = {}
    robot_csvs: dict[str, list[Path]] = {}

    for root in human_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in _HUMAN_SUFFIXES:
                _append_candidate(human, path, root)

    for root in robot_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            if path.is_file() and _is_robot_motion_csv(path):
                _append_candidate(robot_csvs, path, root)

    pairs: list[MotionPair] = []
    common_keys = sorted(set(human).intersection(robot_csvs))
    for key in common_keys:
        human_path = _choose_human(human[key])
        robot_choice = _choose_robot(robot_csvs[key], robot)
        if robot_choice is None:
            continue
        robot_path, robot_type = robot_choice
        pairs.append(MotionPair(key=key, human_path=human_path, robot_path=robot_path, robot_type=robot_type))

    stats = {
        "human_files": sum(len(v) for v in human.values()),
        "robot_csvs": sum(len(v) for v in robot_csvs.values()),
        "human_unique_keys": len(human),
        "robot_unique_keys": len(robot_csvs),
        "matched_pairs": len(pairs),
        "renderable_pairs": sum(1 for pair in pairs if pair.renderable),
        "non_renderable_pairs": sum(1 for pair in pairs if not pair.renderable),
    }
    return pairs, stats


def write_manifest(path: Path, pairs: list[MotionPair], stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "renderable",
                "robot_type",
                "human_path",
                "robot_path",
            ],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "key": pair.key,
                    "renderable": pair.renderable,
                    "robot_type": pair.robot_type,
                    "human_path": pair.human_path,
                    "robot_path": pair.robot_path,
                }
            )

    stats_path = path.with_suffix(path.suffix + ".json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def parse_review_dir_timestamp(path: Path) -> float:
    prefix = "review_"
    if not path.name.startswith(prefix):
        return 0.0
    try:
        return datetime.strptime(path.name[len(prefix) :], "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return 0.0


def latest_label_time(labels_path: Path) -> tuple[float, int]:
    latest_time = 0.0
    valid_rows = 0
    if not labels_path.exists():
        return latest_time, valid_rows

    with labels_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("key") or not row.get("label"):
                continue
            valid_rows += 1
            try:
                latest_time = max(latest_time, float(row.get("time_unix") or 0.0))
            except ValueError:
                continue
    return latest_time, valid_rows


def review_output_sort_key(path: Path) -> tuple[float, float, float, float, int, str]:
    labels_path = path / "review_labels.csv"
    label_time, valid_rows = latest_label_time(labels_path)
    try:
        labels_mtime = labels_path.stat().st_mtime
    except FileNotFoundError:
        labels_mtime = 0.0
    try:
        dir_mtime = path.stat().st_mtime
    except FileNotFoundError:
        dir_mtime = 0.0
    return (
        label_time,
        labels_mtime,
        parse_review_dir_timestamp(path),
        dir_mtime,
        valid_rows,
        path.name,
    )


def find_latest_review_output_dir(base_output: Path, exclude: Path | None = None) -> Path | None:
    base_output = base_output.expanduser().resolve()
    exclude = exclude.expanduser().resolve() if exclude is not None else None
    if not base_output.exists():
        return None
    candidates = [
        path
        for path in base_output.iterdir()
        if path.is_dir()
        and path.name.startswith("review_")
        and path.resolve() != exclude
        and (path / "review_labels.csv").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=review_output_sort_key)


def resolve_resume_output_dir(cli_args: argparse.Namespace) -> Path | None:
    if cli_args.resume_root is not None:
        resume_root = cli_args.resume_root.expanduser().resolve()
        labels_path = resume_root / "review_labels.csv"
        if not resume_root.is_dir():
            raise SystemExit(f"--resume-root is not a directory: {resume_root}")
        if not labels_path.exists():
            raise SystemExit(f"--resume-root does not contain review_labels.csv: {resume_root}")
        return resume_root

    if cli_args.resume:
        return find_latest_review_output_dir(cli_args.output)
    return None


def copy_resume_labels(previous_output_dir: Path | None, new_labels_path: Path) -> int:
    if previous_output_dir is None:
        return 0
    previous_labels_path = previous_output_dir / "review_labels.csv"
    if not previous_labels_path.exists():
        return 0

    new_labels_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with previous_labels_path.open("r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        with new_labels_path.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=_REVIEW_FIELDNAMES)
            writer.writeheader()
            for row in reader:
                if not row.get("key") or not row.get("label"):
                    continue
                writer.writerow({field: row.get(field, "") for field in _REVIEW_FIELDNAMES})
                rows += 1
    return rows


def ensure_review_labels_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_REVIEW_FIELDNAMES)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())


def load_resume_config(previous_output_dir: Path | None) -> dict:
    if previous_output_dir is None:
        return {}

    run_config_path = previous_output_dir / "run_config.json"
    if run_config_path.exists():
        return json.loads(run_config_path.read_text(encoding="utf-8"))

    manifest_path = previous_output_dir / "review_manifest.csv"
    manifest_stats_path = previous_output_dir / "review_manifest.csv.json"
    if not manifest_path.exists():
        return {}

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        first_row = next(csv.DictReader(f), None)
    if first_row is None:
        return {}

    config = {
        "soma_root": str(Path(first_row["human_path"]).parent),
        "robot_root": str(Path(first_row["robot_path"]).parent),
        "robot": first_row["robot_type"],
    }
    if manifest_stats_path.exists():
        stats = json.loads(manifest_stats_path.read_text(encoding="utf-8"))
        roots = stats.get("smpl_filter", {}).get("roots", [])
        if roots:
            config["smpl_root"] = roots[0]
    return config


def save_run_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def first_unreviewed_index(pairs: list[MotionPair], review_output: Path) -> int:
    labels, _ = RetargetReviewViewer._load_existing_reviews(review_output)
    for idx, pair in enumerate(pairs):
        if pair.key not in labels:
            return idx
    return 0


def resolve_robot_mjcf(robot_type: str, robot_mjcf: Path | None) -> Path:
    if robot_mjcf is not None:
        path = robot_mjcf.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = _DEFAULT_AO_FULL_HANDS_MJCF if robot_type == "agile_one" else _DEFAULT_G1_MJCF
    for candidate in candidates:
        try:
            exists = candidate.exists()
        except PermissionError:
            continue
        if exists:
            return candidate

    if robot_type == "unitree_g1":
        return newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml"

    raise FileNotFoundError(f"No default MJCF found for {robot_type}. Update resolve_robot_mjcf().")


def build_robot_model(robot_type: str, robot_mjcf: Path | None):
    robot_mjcf_path = resolve_robot_mjcf(robot_type, robot_mjcf)
    robot_builder = newton.ModelBuilder()
    robot_builder.add_mjcf(robot_mjcf_path)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_builder(robot_builder, wp.transform_identity())
    return builder.finalize(), robot_mjcf_path


def _flag_value(flag):
    return getattr(flag, "value", flag)


def _push_scaled_font(ui, scale: float) -> bool:
    if scale <= 0.0 or abs(scale - 1.0) < 1e-3:
        return False
    try:
        ui.push_font(ui.get_font(), ui.get_font_size() * scale)
        return True
    except Exception:
        return False


def _pop_scaled_font(ui, pushed: bool) -> None:
    if pushed:
        ui.pop_font()


def _install_review_viewer_ui_overrides(viewer, ui_scale: float) -> None:
    """Hide Newton's default side panel and place its stats overlay at right-center."""
    if hasattr(viewer, "_render_left_panel"):
        viewer._render_left_panel = lambda: None

    if not hasattr(viewer, "_render_stats_overlay"):
        return

    def _render_stats_overlay():
        ui_obj = getattr(viewer, "ui", None)
        if not ui_obj or not ui_obj.is_available:
            return

        imgui = ui_obj.imgui
        io = ui_obj.io
        flags = (
            _flag_value(imgui.WindowFlags_.no_decoration)
            | _flag_value(imgui.WindowFlags_.always_auto_resize)
            | _flag_value(imgui.WindowFlags_.no_resize)
            | _flag_value(imgui.WindowFlags_.no_saved_settings)
            | _flag_value(imgui.WindowFlags_.no_focus_on_appearing)
            | _flag_value(imgui.WindowFlags_.no_nav)
            | _flag_value(imgui.WindowFlags_.no_move)
        )

        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size[0] - _PANEL_MARGIN, io.display_size[1] * 0.5),
            pivot=imgui.ImVec2(1.0, 0.5),
        )
        try:
            imgui.set_next_window_bg_alpha(0.7)
        except AttributeError:
            pass

        if imgui.begin("Performance Stats", flags=flags):
            pushed_font = _push_scaled_font(imgui, ui_scale)
            imgui.text(f"FPS: {getattr(viewer, '_current_fps', 0.0):.1f}")
            model = getattr(viewer, "model", None)
            if model is not None:
                imgui.separator()
                imgui.text(f"Bodies: {model.body_count}")
                imgui.text(f"Shapes: {model.shape_count}")
                imgui.text(f"Joints: {model.joint_count}")
            _pop_scaled_font(imgui, pushed_font)

        for callback in getattr(viewer, "_ui_callbacks", {}).get("stats", []):
            callback(imgui)
        imgui.end()

    viewer._render_stats_overlay = _render_stats_overlay


def _model_joint_q_lookup(model) -> dict[str, int]:
    names = [newton_utils.get_name_from_label(label) for label in model.joint_label]
    q_starts = model.joint_q_start.numpy()
    return {name: int(q_starts[idx]) for idx, name in enumerate(names)}


def load_robot_csv_for_model(path: Path, robot_type: str, model, fps: float) -> csv_utils.CSVAnimationBuffer:
    """Load a robot CSV into model-sized Newton joint_q rows.

    CSV files may contain only the 29 no-hands body joints while the viewing
    model may include hand joints. Missing joints are intentionally left at the
    model default pose so fixed-hands MJCFs can be used for visual review.
    """
    csv_config = csv_utils.get_csv_config(robot_type)
    compact = csv_utils.load_csv(str(path), fps=fps, csv_config=csv_config, robot_name=robot_type)

    if compact.data[0].shape[0] == model.joint_coord_count:
        return compact

    with path.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    q_default = np.asarray(model.joint_q.numpy(), dtype=np.float32)[: model.joint_coord_count]
    q_data = np.tile(q_default[None, :], (raw.shape[0], 1))
    q_data[:, :7] = np.asarray([row[:7] for row in compact.data], dtype=np.float32)

    q_lookup = _model_joint_q_lookup(model)
    missing: list[str] = []
    for col_idx, col_name in enumerate(header):
        if not col_name.endswith("_dof"):
            continue
        joint_name = col_name[: -len("_dof")]
        q_start = q_lookup.get(joint_name)
        if q_start is None:
            missing.append(joint_name)
            continue
        q_data[:, q_start] = np.deg2rad(raw[:, col_idx]).astype(np.float32)

    if missing:
        raise ValueError(
            f"Robot model is missing CSV joints from {path}: {', '.join(missing[:12])}"
            + (" ..." if len(missing) > 12 else "")
        )

    return csv_utils.CSVAnimationBuffer.create_from_raw_data(q_data, fps)


class RetargetReviewViewer:
    def __init__(self, viewer, args: argparse.Namespace, pairs: list[MotionPair]) -> None:
        if not pairs:
            raise ValueError("No renderable motion pairs found.")

        self.viewer = viewer
        self.viewer.vsync = True
        self.args = args
        self.pairs = pairs
        self.index = max(0, min(args.start_index, len(pairs) - 1))
        self.converter = SpaceConverter(get_facing_direction_type_from_str(args.human_facing_direction))
        _install_review_viewer_ui_overrides(self.viewer, float(args.ui_scale))
        self.viewer.renderer.set_title("Retargeted Motion Review")
        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")

        self.robot_type = pairs[0].robot_type
        if any(pair.robot_type != self.robot_type for pair in pairs):
            raise ValueError("Renderable pairs contain multiple robot types. Use an Agile One CSV root for this review tool.")
        self.model, self.robot_mjcf_path = build_robot_model(self.robot_type, args.robot_mjcf)
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets([0, 0, 0])
        self.state = self.model.state()
        self.robot_num_joint_q = self.model.joint_coord_count // self.model.articulation_count
        self.robot_default_joint_q_values = self.model.joint_q.numpy()
        self.robot_offset = wp.transform(wp.vec3(float(args.robot_offset_x), float(args.robot_offset_y), 0.0), wp.quat_identity())
        self.human_offset = wp.transform(wp.vec3(float(args.human_offset_x), float(args.human_offset_y), 0.0), wp.quat_identity())

        self.coordinate_renderer = CoordinateRenderer()
        self.skeleton_renderer: SkeletonRenderer | None = None
        self.skeletal_mesh_renderer: SkeletalMeshRenderer | None = None
        self.skeleton_instance: SkeletonInstance | None = None
        self.human_buffer = None
        self.robot_buffer = None

        self.show_human_mesh = True
        self.show_human_skeleton = False
        self.show_joint_axes = False
        self.show_gizmos = False
        self.is_playing = True
        self.playback_time = 0.0
        self.playback_total_time = 0.0
        self.playback_speed = 1.5
        self.playback_loop = True
        self.fps = float(args.viewer_fps)
        self.frame_dt = 1.0 / self.fps
        self.time = 0.0
        self.last_status = ""
        self.pending_label: str | None = None
        self.pending_reason_text = ""
        self.pending_other_selected = False
        self.labels, self.reasons = self._load_existing_reviews(args.review_output)

        self.load_pair(self.index)
        self._set_initial_camera()

    @staticmethod
    def _load_existing_reviews(path: Path) -> tuple[dict[str, str], dict[str, str]]:
        if not path.exists():
            return {}, {}
        labels = {}
        reasons = {}
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get("key")
                label = row.get("label")
                if key and label:
                    labels[key] = label
                    reasons[key] = row.get("reason", "")
        return labels, reasons

    def _review_output_needs_new_header(self) -> bool:
        if not self.args.review_output.exists():
            return True
        with self.args.review_output.open("r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        return header != _REVIEW_FIELDNAMES

    def _mark(self, label: str, reason: str = "") -> None:
        pair = self.pairs[self.index]
        self.args.review_output.parent.mkdir(parents=True, exist_ok=True)
        write_header = self._review_output_needs_new_header()
        if self.args.review_output.exists() and write_header:
            backup = self.args.review_output.with_suffix(
                self.args.review_output.suffix + f".pre_reason_{int(time.time())}.bak"
            )
            self.args.review_output.rename(backup)
            self.last_status = f"backed up old review schema to {backup}"
        with self.args.review_output.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_REVIEW_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "time_unix": f"{time.time():.3f}",
                    "index": self.index,
                    "key": pair.key,
                    "label": label,
                    "reason": reason,
                    "playback_time": f"{self.playback_time:.4f}",
                    "human_path": pair.human_path,
                    "robot_path": pair.robot_path,
                    "robot_type": pair.robot_type,
                }
            )
            f.flush()
            os.fsync(f.fileno())
        self.labels[pair.key] = label
        self.reasons[pair.key] = reason
        self.pending_label = None
        self.pending_reason_text = ""
        self.pending_other_selected = False
        suffix = f" ({reason})" if reason else ""
        saved_status = f"marked {label}{suffix}: {pair.key}"
        self.last_status = saved_status
        print(f"[review] saved label={label}{suffix} key={pair.key} -> {self.args.review_output}", flush=True)
        if len(self.pairs) > 1:
            self.next_pair(1)
            self.last_status = f"{saved_status}; advanced to {self.index + 1}/{len(self.pairs)}"

    def _clear_renderers(self) -> None:
        if self.skeleton_renderer is not None:
            self.skeleton_renderer.clear(self.viewer)
        if self.skeletal_mesh_renderer is not None:
            self.skeletal_mesh_renderer.clear(self.viewer)
        if self.coordinate_renderer is not None:
            self.coordinate_renderer.clear(self.viewer)

    def load_pair(self, index: int) -> None:
        self.index = max(0, min(index, len(self.pairs) - 1))
        self.pending_label = None
        self.pending_reason_text = ""
        self.pending_other_selected = False
        pair = self.pairs[self.index]
        if not pair.renderable:
            raise ValueError(f"Pair is not renderable without a BVH source: {pair.human_path}")

        self._clear_renderers()
        skeleton, animation = bvh_utils.load_bvh(str(pair.human_path))
        self.human_buffer = animation
        self.skeleton_renderer = SkeletonRenderer(skeleton, [0])
        self.skeleton_instance = SkeletonInstance(
            skeleton,
            _DEFAULT_HUMAN_COLOR,
            self.converter.transform(wp.transform_identity()),
        )
        self.skeleton_instance.set_local_transforms(animation.get_local_transforms(0))

        self.skeletal_mesh_renderer = None
        if self.args.human_mesh:
            try:
                skeletal_mesh = pipeline_utils.get_source_model_mesh(pipeline_utils.SourceType.SOMA, skeleton)
                self.skeletal_mesh_renderer = SkeletalMeshRenderer(skeletal_mesh)
            except Exception as exc:
                self.last_status = f"human mesh unavailable: {exc}"
                self.show_human_mesh = False

        csv_fps = float(self.args.robot_fps or animation.sample_rate)
        self.robot_buffer = load_robot_csv_for_model(pair.robot_path, pair.robot_type, self.model, csv_fps)

        self.playback_time = 0.0
        self.compute_playback_total_time()
        self.last_status = f"loaded {self.index + 1}/{len(self.pairs)}: {pair.key}"

    def compute_playback_total_time(self) -> None:
        human_time = 0.0 if self.human_buffer is None else self.human_buffer.num_frames / self.human_buffer.sample_rate
        robot_time = 0.0 if self.robot_buffer is None else self.robot_buffer.num_frames / self.robot_buffer.sample_rate
        self.playback_total_time = max(human_time, robot_time)
        self.playback_time = max(0.0, min(self.playback_time, self.playback_total_time))

    def next_pair(self, delta: int) -> None:
        self.load_pair((self.index + delta) % len(self.pairs))

    def next_unmarked(self, direction: int) -> None:
        for step in range(1, len(self.pairs) + 1):
            idx = (self.index + direction * step) % len(self.pairs)
            if self.pairs[idx].key not in self.labels:
                self.load_pair(idx)
                return
        self.last_status = "all renderable pairs already have labels"

    def _set_initial_camera(self) -> None:
        if not hasattr(self.viewer, "camera"):
            return
        angle = np.deg2rad(_DEFAULT_CAMERA_LEFT_FRONT_DEG)
        pos = _DEFAULT_CAMERA_TARGET + np.array(
            [
                _DEFAULT_CAMERA_DISTANCE * np.cos(angle),
                -_DEFAULT_CAMERA_DISTANCE * np.sin(angle),
                _DEFAULT_CAMERA_HEIGHT - _DEFAULT_CAMERA_TARGET[2],
            ],
            dtype=np.float32,
        )
        front = _DEFAULT_CAMERA_TARGET - pos
        norm = float(np.linalg.norm(front))
        if norm < 1.0e-6:
            return
        front /= norm
        pitch = float(np.rad2deg(np.arcsin(front[2])))
        yaw = float(np.rad2deg(np.arctan2(front[1], front[0])))
        camera_pos_type = type(self.viewer.camera.pos)
        self.viewer.set_camera(camera_pos_type(float(pos[0]), float(pos[1]), float(pos[2])), pitch, yaw)

    def step_frame(self, delta: int) -> None:
        sample_rate = self.human_buffer.sample_rate if self.human_buffer is not None else self.fps
        self.playback_time = max(
            0.0,
            min(self.playback_total_time, self.playback_time + delta / float(sample_rate)),
        )

    def update_robot_state(self) -> None:
        if self.robot_buffer is None:
            wp.copy(
                self.model.joint_q,
                wp.array(self.robot_default_joint_q_values[: self.robot_num_joint_q], dtype=wp.float32),
                0,
                0,
                self.robot_num_joint_q,
            )
        else:
            prev_xform = wp.transform(self.robot_buffer.xform)
            self.robot_buffer.xform = self.robot_offset
            data = self.robot_buffer.sample(self.playback_time)
            wp.copy(self.model.joint_q, wp.array(data, dtype=wp.float32), 0, 0, self.robot_num_joint_q)
            self.robot_buffer.xform = prev_xform
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)

    def step(self) -> None:
        self.time += self.frame_dt
        if self.is_playing:
            self.playback_time += self.frame_dt * self.playback_speed
            if self.playback_loop and self.playback_total_time > 0.0:
                self.playback_time %= self.playback_total_time
            else:
                self.playback_time = max(0.0, min(self.playback_time, self.playback_total_time))

        if self.human_buffer is not None and self.skeleton_instance is not None:
            self.skeleton_instance.set_local_transforms(self.human_buffer.sample(self.playback_time))
        self.update_robot_state()

    def render(self) -> None:
        self.viewer.begin_frame(self.time)
        if self.skeleton_instance is not None:
            prev_xform = wp.transform(self.skeleton_instance.xform)
            self.skeleton_instance.xform = wp.mul(self.human_offset, self.skeleton_instance.xform)
            if self.show_human_skeleton and self.skeleton_renderer is not None:
                self.skeleton_renderer.draw(self.viewer, self.skeleton_instance, 0)
            if self.show_joint_axes:
                self.coordinate_renderer.draw(self.viewer, self.skeleton_instance.compute_global_transforms(), 0.1, 0)
            if self.show_human_mesh and self.skeletal_mesh_renderer is not None:
                self.skeletal_mesh_renderer.draw(
                    self.viewer,
                    self.skeleton_instance,
                    self.skeleton_instance.color,
                    0,
                )
            self.skeleton_instance.xform = prev_xform

        if self.show_gizmos:
            self.viewer.log_gizmo("human_offset", self.human_offset)
            self.viewer.log_gizmo("robot_offset", self.robot_offset)
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
        self.ui_review_panel(ui)
        self.ui_playback_controls(ui)

    @staticmethod
    def _text_wrapped(ui, text: object) -> None:
        if hasattr(ui, "text_wrapped"):
            ui.text_wrapped(str(text))
        else:
            ui.text(str(text))

    @staticmethod
    def _text_colored(ui, text: object, rgba: tuple[float, float, float, float]) -> None:
        if hasattr(ui, "text_colored") and hasattr(ui, "ImVec4"):
            ui.text_colored(ui.ImVec4(*rgba), str(text))
        else:
            ui.text(str(text))

    @staticmethod
    def _pop_style_colors(ui, count: int) -> None:
        try:
            ui.pop_style_color(count)
        except TypeError:
            for _ in range(count):
                ui.pop_style_color()

    @staticmethod
    def _colored_button(ui, text: str, label: str) -> bool:
        button_style = _LABEL_BUTTON_COLORS.get(label)
        if (
            button_style
            and hasattr(ui, "push_style_color")
            and hasattr(ui, "pop_style_color")
            and hasattr(ui, "Col_")
            and hasattr(ui, "ImVec4")
        ):
            button, hovered, active, text_color = button_style
            pushed = 0
            try:
                ui.push_style_color(ui.Col_.button, ui.ImVec4(*button))
                pushed += 1
                ui.push_style_color(ui.Col_.button_hovered, ui.ImVec4(*hovered))
                pushed += 1
                ui.push_style_color(ui.Col_.button_active, ui.ImVec4(*active))
                pushed += 1
                ui.push_style_color(ui.Col_.text, ui.ImVec4(*text_color))
                pushed += 1
                return ui.button(text)
            finally:
                if pushed:
                    RetargetReviewViewer._pop_style_colors(ui, pushed)
        RetargetReviewViewer._text_colored(ui, text.split("##", 1)[0], _LABEL_COLORS.get(label, _LABEL_COLORS["unmarked"]))
        ui.same_line()
        return ui.button(text)

    def _reason_buttons(self, ui, label: str, reasons: list[str], buttons_per_row: int = 4) -> None:
        self._text_colored(ui, f"Select {label} reason:", _LABEL_COLORS.get(label, _LABEL_COLORS["unmarked"]))
        for idx, reason in enumerate(reasons):
            if idx % buttons_per_row != 0:
                ui.same_line()
            if self._colored_button(ui, f"{reason}##{label}_reason_{idx}", label):
                if reason == "Other":
                    self.pending_other_selected = True
                    self.pending_reason_text = ""
                    self.last_status = f"Enter custom {label} reason before submitting."
                else:
                    self._mark(label, reason)
        if self.pending_other_selected:
            ui.set_next_item_width(520)
            _, self.pending_reason_text = ui.input_text(
                f"Other reason##{label}_custom_reason",
                self.pending_reason_text,
            )
            ui.same_line()
            if (
                self._colored_button(ui, f"Submit other {label} reason##submit_reason", label)
                and self.pending_reason_text.strip()
            ):
                self._mark(label, self.pending_reason_text.strip())
            ui.same_line()
        if ui.button(f"Cancel {label}##cancel_reason"):
            self.pending_label = None
            self.pending_reason_text = ""
            self.pending_other_selected = False

    def _ui_scale(self) -> float:
        return max(1.0, float(self.args.ui_scale))

    def ui_review_panel(self, ui) -> None:
        viewport = ui.get_main_viewport()
        scale = self._ui_scale()
        height_scale = 1.0 + 0.55 * (scale - 1.0)
        panel_width = max(760, viewport.size.x - 2 * _PANEL_MARGIN)
        extra_height = 70 if self.pending_label else 0
        if self.pending_other_selected:
            extra_height += 45
        panel_height = int((_TOP_PANEL_HEIGHT + extra_height) * height_scale)
        ui.set_next_window_pos(ui.ImVec2(_PANEL_MARGIN, _PANEL_MARGIN))
        ui.set_next_window_size(ui.ImVec2(panel_width, panel_height))
        ui.set_next_window_bg_alpha(_PANEL_ALPHA)
        ui.begin(
            "Motion Review",
            flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize | ui.WindowFlags_.no_move),
        )
        pushed_font = _push_scaled_font(ui, float(self.args.ui_scale))

        pair = self.pairs[self.index]
        label = self.labels.get(pair.key, "unmarked")
        reason = self.reasons.get(pair.key, "")
        ui.text(f"{self.index + 1}/{len(self.pairs)}  label:")
        ui.same_line()
        self._text_colored(ui, label, _LABEL_COLORS.get(label, _LABEL_COLORS["unmarked"]))
        if reason:
            ui.same_line()
            ui.text(f"reason: {reason}")
        ui.text(f"name: {pair.name}")
        self._text_wrapped(ui, f"human: {pair.human_path}")
        self._text_wrapped(ui, f"robot: {pair.robot_path}")
        ui.separator()

        if ui.button("< Prev"):
            self.next_pair(-1)
        ui.same_line()
        if ui.button("Next >"):
            self.next_pair(1)
        ui.same_line()
        if ui.button("< Prev unreviewed"):
            self.next_unmarked(-1)
        ui.same_line()
        if ui.button("Next unreviewed >"):
            self.next_unmarked(1)
        ui.same_line()
        ui.text("unreviewed = no keep/maybe/reject label yet")

        if self._colored_button(ui, "Keep", "keep"):
            self._mark("keep")
        ui.same_line()
        if self._colored_button(ui, "Maybe", "maybe"):
            self.pending_label = "maybe"
            self.pending_reason_text = ""
            self.pending_other_selected = False
        ui.same_line()
        if self._colored_button(ui, "Reject", "reject"):
            self.pending_label = "reject"
            self.pending_reason_text = ""
            self.pending_other_selected = False

        if self.pending_label == "maybe":
            self._reason_buttons(ui, "maybe", _MAYBE_REASONS, buttons_per_row=5)
        elif self.pending_label == "reject":
            self._reason_buttons(ui, "reject", _REJECT_REASONS, buttons_per_row=4)

        if self.last_status:
            ui.text(self.last_status)
        _pop_scaled_font(ui, pushed_font)
        ui.end()

    def ui_playback_controls(self, ui) -> None:
        viewport = ui.get_main_viewport()
        scale = self._ui_scale()
        height_scale = 1.0 + 0.55 * (scale - 1.0)
        panel_height = int(_PLAYBACK_PANEL_HEIGHT * height_scale)
        panel_width = max(760, viewport.size.x - 2 * _PANEL_MARGIN)
        ui.set_next_window_pos(ui.ImVec2(_PANEL_MARGIN, viewport.size.y - _PANEL_MARGIN - panel_height))
        ui.set_next_window_size(ui.ImVec2(panel_width, panel_height))
        ui.set_next_window_bg_alpha(_PANEL_ALPHA)
        ui.begin(
            "Playback Controls",
            flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize | ui.WindowFlags_.no_move),
        )
        pushed_font = _push_scaled_font(ui, float(self.args.ui_scale))

        ui.align_text_to_frame_padding()
        ui.text("Time (s):")
        ui.same_line()
        ui.set_next_item_width(max(240, panel_width - int(300 * scale)))
        changed, new_time = ui.slider_float(
            "##TimeSlider",
            self.playback_time,
            0.0,
            max(1e-6, self.playback_total_time),
            "%.2f",
        )
        if changed:
            self.playback_time = max(0.0, min(new_time, self.playback_total_time))
        ui.same_line()
        ui.text(f"{self.playback_time:.2f}/{self.playback_total_time:.2f}s")

        self.is_playing = not ui.button("Pause") if self.is_playing else ui.button("Play ")
        ui.same_line()
        ui.text("Speed")
        ui.same_line()
        ui.set_next_item_width(100)
        changed, new_speed = ui.slider_float("##SpeedSlider", self.playback_speed, -2.0, 2.0, "%.2f")
        if changed:
            self.playback_speed = new_speed
        ui.same_line()
        _, self.playback_loop = ui.checkbox("Loop", self.playback_loop)
        _pop_scaled_font(ui, pushed_font)
        ui.end()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Review matched SOMA BVH and retargeted robot CSV motions.",
    )
    parser.add_argument("--soma-root", type=Path, help="Root containing renderable SOMA BVH motions.")
    parser.add_argument("--smpl-root", type=Path, help="Root containing SMPL/SMPLX filtered motions.")
    parser.add_argument("--robot-root", type=Path, help="Root containing Agile One retargeted robot CSV motions.")
    parser.add_argument("--robot-mjcf", type=Path, help="Explicit Agile One MJCF path.")
    parser.add_argument("--output", type=Path, default=Path("output/motion_review"), help="Base output directory.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Inherit labels from the latest review_* directory under --output, "
            "then continue writing into a fresh timestamped directory."
        ),
    )
    parser.add_argument(
        "--resume-root",
        "-resume-root",
        type=Path,
        help=(
            "Resume from a specific prior review_* directory instead of auto-selecting "
            "one under --output. This implies --resume."
        ),
    )
    parser.add_argument("--ui-scale", type=float, default=_DEFAULT_UI_SCALE)
    return parser.parse_args()


def make_run_output_dir(base_output: Path) -> Path:
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output.expanduser().resolve() / f"review_{time_tag}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def main() -> None:
    import newton.viewer

    cli_args = parse_args()
    is_resume = bool(cli_args.resume or cli_args.resume_root is not None)
    if not is_resume and (
        cli_args.soma_root is None
        or cli_args.smpl_root is None
        or cli_args.robot_root is None
    ):
        raise SystemExit(
            "--soma-root, --smpl-root, and --robot-root are required unless --resume or --resume-root is set."
        )

    previous_output_dir = resolve_resume_output_dir(cli_args)
    resume_config = load_resume_config(previous_output_dir)
    if is_resume and previous_output_dir is None:
        raise SystemExit(f"--resume requested, but no prior review_* directory was found under {cli_args.output}")

    soma_root_arg = cli_args.soma_root or (
        Path(resume_config["soma_root"]) if "soma_root" in resume_config else None
    )
    smpl_root_arg = cli_args.smpl_root or (
        Path(resume_config["smpl_root"]) if "smpl_root" in resume_config else None
    )
    robot_root_arg = cli_args.robot_root or (
        Path(resume_config["robot_root"]) if "robot_root" in resume_config else None
    )
    robot_mjcf_arg = cli_args.robot_mjcf or (
        Path(resume_config["robot_mjcf"]) if "robot_mjcf" in resume_config else None
    )
    robot_arg = _REVIEW_ROBOT
    if soma_root_arg is None or smpl_root_arg is None or robot_root_arg is None:
        raise SystemExit(
            "--resume could not recover all required inputs from the latest run. "
            "Pass --soma-root, --smpl-root, and --robot-root once."
        )

    output_dir = make_run_output_dir(cli_args.output)
    if previous_output_dir is not None and previous_output_dir.resolve() == output_dir.resolve():
        previous_output_dir = find_latest_review_output_dir(cli_args.output, exclude=output_dir)
    review_output = output_dir / "review_labels.csv"
    resumed_rows = copy_resume_labels(previous_output_dir, review_output)
    ensure_review_labels_file(review_output)
    args = SimpleNamespace(
        start_index=0,
        human_facing_direction="Mujoco",
        ui_scale=cli_args.ui_scale,
        robot_mjcf=robot_mjcf_arg,
        robot_offset_x=0.0,
        robot_offset_y=-0.85,
        human_offset_x=0.0,
        human_offset_y=0.85,
        viewer_fps=60.0,
        review_output=review_output,
        manifest_output=output_dir / "review_manifest.csv",
        human_mesh=True,
        robot_fps=None,
    )

    soma_root = soma_root_arg.expanduser().resolve()
    smpl_root = smpl_root_arg.expanduser().resolve()
    robot_root = robot_root_arg.expanduser().resolve()
    pairs, stats = find_motion_pairs([soma_root], [robot_root], robot_arg)

    before_required_filter = len(pairs)
    required_stems, required_stats = collect_human_stems([smpl_root])
    pairs = [pair for pair in pairs if pair.key in required_stems]
    stats["matched_pairs_before_smpl_filter"] = before_required_filter
    stats["smpl_filter"] = required_stats
    stats["smpl_filter_removed"] = before_required_filter - len(pairs)
    stats["matched_pairs"] = len(pairs)
    stats["renderable_pairs"] = sum(1 for pair in pairs if pair.renderable)
    stats["non_renderable_pairs"] = sum(1 for pair in pairs if not pair.renderable)

    renderable_pairs = [pair for pair in pairs if pair.renderable]
    args.start_index = first_unreviewed_index(renderable_pairs, args.review_output)
    write_manifest(args.manifest_output, pairs, stats)
    save_run_config(
        output_dir / "run_config.json",
        {
            "soma_root": str(soma_root),
            "smpl_root": str(smpl_root),
            "robot_root": str(robot_root),
            "robot_mjcf": str(robot_mjcf_arg.expanduser().resolve()) if robot_mjcf_arg is not None else None,
            "robot": robot_arg,
            "ui_scale": cli_args.ui_scale,
            "created_from_resume": is_resume,
            "resumed_from": str(previous_output_dir) if previous_output_dir is not None else None,
        },
    )

    print("[review] scan stats:")
    print(json.dumps(stats, indent=2))
    print(f"[review] output_dir={output_dir}")
    print(f"[review] manifest={args.manifest_output}")
    print(f"[review] labels={args.review_output}")
    if is_resume:
        print(f"[review] resumed_from={previous_output_dir}")
        print(f"[review] resumed_labels={resumed_rows}")
        print(f"[review] start_index={args.start_index}")

    if not renderable_pairs:
        raise SystemExit(
            "No renderable pairs found. The viewer currently needs a BVH/SOMA source with the same stem as the robot CSV."
        )

    enable_cpu_only_viewer_fallback()
    viewer = newton.viewer.ViewerGL()
    app = RetargetReviewViewer(viewer, args, renderable_pairs)
    app.run()


if __name__ == "__main__":
    main()
