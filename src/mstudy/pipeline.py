"""Video -> tracks.parquet.

The expensive stage. A 30-minute video takes roughly the same again in
processing on a CPU-only machine, so this stage is written to be resumable:
results are flushed to disk in shards and an interrupted run picks up from the
last completed shard instead of starting over.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .config import Config
from .detect import PersonAnalyser, empty_pose
from .ingest import VideoInfo, iter_frames, probe_video
from .track import build_tracker, update as tracker_update

TRACK_COLUMNS = [
    "frame_idx", "t", "track_id",
    "x1", "y1", "x2", "y2", "conf",
    "has_pose", "kp_x", "kp_y", "kp_conf",
]

# Flush roughly every 5 minutes of analysed footage.
SHARD_SECONDS = 300.0


def _shard_dir(out_dir: Path) -> Path:
    return out_dir / "_shards"


def _completed_shards(out_dir: Path) -> list[Path]:
    d = _shard_dir(out_dir)
    return sorted(d.glob("shard_*.parquet")) if d.exists() else []


def _resume_point(out_dir: Path, cfg: Config) -> tuple[float, list[Path]]:
    """How far a previous run got, if its config still matches."""
    marker = out_dir / "_shards" / "state.json"
    shards = _completed_shards(out_dir)
    if not shards or not marker.exists():
        return 0.0, []
    try:
        state = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return 0.0, []
    # A changed config invalidates everything already computed.
    if state.get("config_fingerprint") != cfg.fingerprint():
        return 0.0, []
    return float(state.get("next_start_s", 0.0)), shards


def run_tracking(
    video_path: str | Path,
    cfg: Config,
    out_dir: Path,
    quick: bool = False,
    resume: bool = True,
    console: Console | None = None,
) -> tuple[pd.DataFrame, VideoInfo]:
    console = console or Console()
    video_path = Path(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(video_path)
    console.print(f"[bold]{video_path.name}[/bold]  {info.describe()}")

    max_seconds = 120.0 if quick else None
    analysis_fps = 1.0 if quick else cfg.video.analysis_fps
    pose_stride = max(1, int(round(analysis_fps / max(cfg.video.pose_fps, 0.01))))

    start_s, kept_shards = _resume_point(out_dir, cfg) if resume else (0.0, [])
    if start_s > 0:
        console.print(f"[yellow]Resuming from {start_s / 60:.1f} min[/yellow]")
    else:
        if _shard_dir(out_dir).exists():
            shutil.rmtree(_shard_dir(out_dir))
        kept_shards = []
    _shard_dir(out_dir).mkdir(parents=True, exist_ok=True)

    analyser = PersonAnalyser(cfg)
    tracker = build_tracker(cfg)

    total_s = min(info.duration_s, (start_s + max_seconds) if max_seconds else info.duration_s)
    rows: list[dict] = []
    shard_no = len(kept_shards)
    last_flush_t = start_s
    n_analysed = 0
    t_begin = time.time()

    def flush(next_start: float) -> None:
        nonlocal rows, shard_no
        if not rows:
            return
        path = _shard_dir(out_dir) / f"shard_{shard_no:04d}.parquet"
        pd.DataFrame(rows, columns=TRACK_COLUMNS).to_parquet(path, index=False)
        (_shard_dir(out_dir) / "state.json").write_text(
            json.dumps(
                {"config_fingerprint": cfg.fingerprint(), "next_start_s": next_start},
                indent=2,
            )
        )
        shard_no += 1
        rows = []

    columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[rate]}"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
    ]

    with Progress(*columns, console=console) as progress:
        task = progress.add_task(
            "detect+track", total=max(1.0, total_s - start_s), rate="")

        for i, (frame_idx, t, frame) in enumerate(
            iter_frames(info, analysis_fps, cfg.video.rotate, start_s, max_seconds)
        ):
            boxes, scores = analyser.detect(frame)
            boxes, scores, ids = tracker_update(tracker, boxes, scores)

            want_pose = (i % pose_stride == 0) and len(boxes) > 0
            if want_pose:
                kpts, kp_scores = analyser.pose(frame, boxes)
            else:
                kpts, kp_scores = empty_pose(len(boxes))

            for j, tid in enumerate(ids):
                rows.append(
                    {
                        "frame_idx": int(frame_idx),
                        "t": float(t),
                        "track_id": int(tid),
                        "x1": float(boxes[j, 0]), "y1": float(boxes[j, 1]),
                        "x2": float(boxes[j, 2]), "y2": float(boxes[j, 3]),
                        "conf": float(scores[j]),
                        "has_pose": bool(want_pose),
                        "kp_x": kpts[j, :, 0].astype(np.float32).tolist(),
                        "kp_y": kpts[j, :, 1].astype(np.float32).tolist(),
                        "kp_conf": kp_scores[j].astype(np.float32).tolist(),
                    }
                )

            n_analysed += 1
            if t - last_flush_t >= SHARD_SECONDS:
                flush(t)
                last_flush_t = t

            elapsed = max(time.time() - t_begin, 1e-6)
            progress.update(
                task,
                completed=max(0.0, t - start_s),
                rate=f"{n_analysed / elapsed:.1f} fr/s ",
            )

        flush(total_s)

    shards = _completed_shards(out_dir)
    if not shards:
        tracks = pd.DataFrame(columns=TRACK_COLUMNS)
    else:
        tracks = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)

    tracks.to_parquet(out_dir / "tracks.parquet", index=False)
    shutil.rmtree(_shard_dir(out_dir), ignore_errors=True)

    wall = time.time() - t_begin
    console.print(
        f"[green]Tracked[/green] {len(tracks):,} detections, "
        f"{tracks['track_id'].nunique() if len(tracks) else 0} track id(s) "
        f"in {wall / 60:.1f} min"
    )
    return tracks, info
