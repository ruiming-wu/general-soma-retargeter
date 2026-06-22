#!/usr/bin/env python3
"""Extract G1 low-dimensional rooted trajectories from Humanoid Everyday full parquet."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyarrow is required. Run with: uv run --with pyarrow --with huggingface_hub python ...") from exc

try:
    from huggingface_hub import HfFileSystem
except ImportError as exc:  # pragma: no cover
    raise SystemExit("huggingface_hub is required. Run with: uv run --with huggingface_hub --with pyarrow python ...") from exc


REPO_ID = "USC-PSI-Lab/humanoid-everyday"
REMOTE_PREFIX = f"datasets/{REPO_ID}"
META_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/meta"

ROOTED_COLUMNS = [
    "observation.imu.quaternion",
    "observation.imu.accelerometer",
    "observation.imu.gyroscope",
    "observation.imu.rpy",
    "observation.odometry.position",
    "observation.odometry.velocity",
    "observation.odometry.rpy",
    "observation.odometry.quat",
    "observation.arm_joints",
    "observation.leg_joints",
    "observation.hand_joints",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "next.done",
]


def _download_meta(output_root: Path) -> tuple[dict, list[dict], list[dict]]:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for name in ["info.json", "episodes.jsonl", "tasks.jsonl"]:
        dst = meta_dir / name
        if not dst.exists() or dst.stat().st_size == 0:
            url = f"{META_URL}/{name}"
            for attempt in range(12):
                try:
                    with urllib.request.urlopen(url, timeout=60) as r:
                        dst.write_bytes(r.read())
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        time.sleep(20 + attempt * 10)
                        continue
                    raise
            else:
                raise RuntimeError(f"failed to download {url}")
        loaded[name] = dst.read_text(encoding="utf-8")

    info = json.loads(loaded["info.json"])
    episodes = [json.loads(line) for line in loaded["episodes.jsonl"].splitlines() if line.strip()]
    tasks = [json.loads(line) for line in loaded["tasks.jsonl"].splitlines() if line.strip()]
    return info, episodes, tasks


def _remote_episode_path(episode_index: int) -> str:
    return f"{REMOTE_PREFIX}/data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"


def _local_episode_path(output_root: Path, episode_index: int) -> Path:
    return output_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def _extract_one(output_root: Path, episode: dict, overwrite: bool) -> tuple[bool, int, int, str | None]:
    episode_index = int(episode["episode_index"])
    dst = _local_episode_path(output_root, episode_index)
    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return True, episode_index, int(episode.get("length", 0)), None

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    remote_path = _remote_episode_path(episode_index)
    last_error = None
    for attempt in range(8):
        try:
            fs = HfFileSystem()
            with fs.open(remote_path, "rb") as f:
                table = pq.ParquetFile(f).read(columns=ROOTED_COLUMNS)
            pq.write_table(table, tmp, compression="zstd")
            tmp.replace(dst)
            return True, episode_index, table.num_rows, None
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(5 + attempt * 5)
    return False, episode_index, 0, last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/home/ruiming.wu/data/Humanoid-Everyday-G1-rooted"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    info, episodes, tasks = _download_meta(output_root)
    g1_episodes = [row for row in episodes if row.get("robot_type") == "g1"]
    if args.limit is not None:
        g1_episodes = g1_episodes[: args.limit]

    task_by_idx = {row["task_index"]: row for row in tasks}
    manifest_path = output_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"full_total_episodes={info.get('total_episodes')} "
        f"g1_episodes={len(g1_episodes)} workers={args.workers}"
    )

    failed = []
    total_frames = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(_extract_one, output_root, ep, args.overwrite) for ep in g1_episodes]
        for n, fut in enumerate(cf.as_completed(futures), 1):
            ok, episode_index, frames, err = fut.result()
            if ok:
                total_frames += frames
            else:
                failed.append((episode_index, err))
            if n % 100 == 0 or n == len(futures):
                print(f"{n}/{len(futures)} done; failed={len(failed)}; frames={total_frames}", flush=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("episode_index,frames,task_index,category,task,instruction,path\n")
        for ep in g1_episodes:
            task_index = (ep.get("tasks") or [""])[0]
            task = task_by_idx.get(task_index, {})
            path = _local_episode_path(output_root, int(ep["episode_index"]))
            f.write(
                f"{ep['episode_index']},{ep.get('length','')},{task_index},"
                f"{task.get('category','')},{task.get('task','')},"
                f"{json.dumps(ep.get('instruction',''))},{path}\n"
            )

    if failed:
        print("FAILED first 20:", failed[:20])
        raise SystemExit(2)

    print(f"output: {output_root}")
    print(f"manifest: {manifest_path}")
    print(f"episodes: {len(g1_episodes)} frames: {total_frames} hours@30Hz: {total_frames / 30.0 / 3600.0:.6f}")


if __name__ == "__main__":
    main()
