#!/usr/bin/env python3
"""Render short videos around detected branch-switch events."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="motion_summary/probable_manifest CSV.")
    parser.add_argument("--events", type=Path, required=True, help="branch_switch_events.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for MP4 slices and manifest.")
    parser.add_argument("--subset-prefix", default="grab/", help="Only render rows whose subset starts with this prefix.")
    parser.add_argument("--limit", type=int, default=80, help="Number of motions to render. Use 0 for all.")
    parser.add_argument("--fps", type=float, default=120.0, help="Source CSV FPS.")
    parser.add_argument("--window-sec", type=float, default=2.0, help="Total rendered window centered on the event frame.")
    parser.add_argument("--robot", default="agile_one")
    parser.add_argument("--mjcf", type=Path, default=Path("/home/ruiming.wu/codes/H4/mjcf/h4_mjlab.xml"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=2, help="Render every Nth source frame.")
    parser.add_argument("--python-command", default="uv run python", help="Command prefix used to invoke render_robot_motion.py.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def sort_key(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        float(row.get("probable_branch_switch_events") or 0.0),
        float(row.get("max_abs_dq_deg") or 0.0),
        float(row.get("max_abs_accel_deg_s2") or 0.0),
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    probable_rows = [
        row
        for row in read_csv(args.summary)
        if row.get("risk") == "probable_branch_switch" and row.get("subset", "").startswith(args.subset_prefix)
    ]
    probable_rows.sort(key=sort_key, reverse=True)
    if args.limit > 0:
        probable_rows = probable_rows[: args.limit]

    events_by_motion_path: dict[str, list[dict[str, str]]] = {}
    for event in read_csv(args.events):
        if event.get("classification") != "probable_branch_switch":
            continue
        if not event.get("subset", "").startswith(args.subset_prefix):
            continue
        events_by_motion_path.setdefault(event["motion_path"], []).append(event)

    manifest_rows: list[dict[str, str | int | float]] = []
    half_window_frames = max(1, int(round(args.window_sec * args.fps * 0.5)))
    render_script = REPO_ROOT / "tools" / "render_robot_motion.py"

    for idx, row in enumerate(probable_rows, start=1):
        motion_path = row["motion_path"]
        events = events_by_motion_path.get(motion_path, [])
        if not events:
            continue
        event = max(events, key=lambda e: (float(e.get("abs_dq_deg") or 0.0), abs(float(e.get("accel_deg_s2") or 0.0))))
        event_frame = int(float(event["frame0"]))
        num_frames = int(float(row["num_frames"]))
        start_frame = max(0, event_frame - half_window_frames)
        end_frame = min(num_frames, event_frame + half_window_frames)
        subset = row["subset"]
        rel_dir = output_dir / subset
        filename = (
            f"{idx:04d}__{safe_name(row['motion'])}"
            f"__f{event_frame:06d}__{safe_name(event['joint'])}"
            f"__dq{float(event['abs_dq_deg']):05.2f}.mp4"
        )
        output_path = rel_dir / filename
        cmd = [
            *shlex.split(args.python_command),
            str(render_script),
            motion_path,
            "--robot",
            args.robot,
            "--mjcf",
            str(args.mjcf.expanduser()),
            "--output",
            str(output_path),
            "--fps",
            str(args.fps),
            "--video-fps",
            str(args.video_fps),
            "--stride",
            str(args.stride),
            "--start-frame",
            str(start_frame),
            "--end-frame",
            str(end_frame),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
        ]
        manifest_rows.append(
            {
                "rank": idx,
                "motion": row["motion"],
                "subset": subset,
                "motion_path": motion_path,
                "output_video": str(output_path),
                "event_frame": event_frame,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "event_time_sec": event_frame / args.fps,
                "joint": event["joint"],
                "abs_dq_deg": event["abs_dq_deg"],
                "probable_branch_switch_events": row["probable_branch_switch_events"],
                "possible_branch_switch_events": row["possible_branch_switch_events"],
                "max_abs_dq_deg": row["max_abs_dq_deg"],
            }
        )
        print(f"[{idx}/{len(probable_rows)}] {output_path}")
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else [
        "rank",
        "motion",
        "subset",
        "motion_path",
        "output_video",
        "event_frame",
        "start_frame",
        "end_frame",
        "event_time_sec",
        "joint",
        "abs_dq_deg",
        "probable_branch_switch_events",
        "possible_branch_switch_events",
        "max_abs_dq_deg",
    ]
    with (output_dir / "slice_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[slice] wrote manifest: {output_dir / 'slice_manifest.csv'}")
    print(f"[slice] selected={len(manifest_rows)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
