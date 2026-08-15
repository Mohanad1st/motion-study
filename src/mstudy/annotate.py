"""Render the annotated verification video.

This is the trust-builder. Numbers in a report are arguable; a supervisor
watching two minutes of their own line with the system's labels burned on top
can see for themselves whether it is right.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from .config import Config
from .identify import apply_names, load_names
from .ingest import VideoInfo, probe_video
from .metrics import format_hms

# BGR. Mirrors the report's Okabe-Ito palette so a worker is the same colour
# in the video as on the Gantt chart.
WORKER_COLOURS = [
    (178, 114, 0), (0, 159, 230), (115, 158, 0), (167, 121, 204),
    (233, 180, 86), (0, 94, 213), (49, 109, 140), (155, 58, 93),
]
STOP_COLOUR = (0, 176, 255)
MAX_OUT_W = 1280


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with:  winget install Gyan.FFmpeg"
        )
    return exe


def _label(img, text, org, colour, scale=0.55, thick=1):
    """Text with a dark outline, so it stays readable over any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _step_at(segments: pd.DataFrame, worker: str, t: float) -> tuple[str, str]:
    rows = segments[
        (segments["worker"] == worker)
        & (segments["start_s"] <= t)
        & (segments["end_s"] > t)
    ]
    if rows.empty:
        return "", ""
    r = rows.iloc[0]
    return str(r["step"]), str(r["kind"])


def render_annotated(
    video: Path,
    cfg: Config,
    run_dir: Path,
    max_seconds: float | None = None,
    console: Console | None = None,
) -> Path:
    console = console or Console()

    tracks_path = run_dir / "tracks.parquet"
    if not tracks_path.exists():
        raise FileNotFoundError(f"No tracks at {tracks_path}. Run `mstudy track` first.")

    tracks = apply_names(pd.read_parquet(tracks_path), load_names(run_dir))
    segments = (
        pd.read_csv(run_dir / "segments.csv")
        if (run_dir / "segments.csv").exists() else pd.DataFrame()
    )
    stops = (
        pd.read_csv(run_dir / "stops.csv")
        if (run_dir / "stops.csv").exists() else pd.DataFrame()
    )

    info: VideoInfo = probe_video(video)
    workers = sorted(tracks["worker"].unique()) if not tracks.empty else []
    colours = {w: WORKER_COLOURS[i % len(WORKER_COLOURS)] for i, w in enumerate(workers)}

    scale = min(1.0, MAX_OUT_W / info.width)
    out_w = int(info.width * scale) // 2 * 2   # H.264 needs even dimensions
    out_h = int(info.height * scale) // 2 * 2

    # Snap each native frame to the most recent analysed sample.
    sample_times = np.sort(tracks["t"].unique()) if not tracks.empty else np.array([0.0])
    by_time = {t: g for t, g in tracks.groupby("t")} if not tracks.empty else {}

    out_path = run_dir / "annotated.mp4"
    cmd = [
        _ffmpeg(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}", "-r", f"{info.fps:.4f}",
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(str(video))
    limit = min(info.duration_s, max_seconds or info.duration_s)
    total = int(limit * info.fps)

    zone_polys = [
        (z.name, (np.array(z.polygon, np.float32) * scale).astype(np.int32), z.kind)
        for z in cfg.zones
    ]

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"), BarColumn(),
            TaskProgressColumn(), TimeRemainingColumn(), console=console
        ) as progress:
            task = progress.add_task("annotate", total=max(total, 1))

            idx = 0
            while idx < total:
                ok, frame = cap.read()
                if not ok:
                    break
                t = idx / info.fps

                if scale < 1.0:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                else:
                    frame = frame[:out_h, :out_w]

                # Station outlines, so a viewer can see the zones being applied.
                overlay = frame.copy()
                for name, poly, kind in zone_polys:
                    col = (200, 150, 60) if kind == "work" else (140, 140, 140)
                    cv2.fillPoly(overlay, [poly], col)
                    cv2.polylines(frame, [poly], True, col, 1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

                pos = np.searchsorted(sample_times, t, side="right") - 1
                if pos >= 0:
                    rows = by_time.get(sample_times[pos])
                    if rows is not None:
                        for _, r in rows.iterrows():
                            worker = r["worker"]
                            colour = colours.get(worker, (200, 200, 200))
                            x1, y1 = int(r["x1"] * scale), int(r["y1"] * scale)
                            x2, y2 = int(r["x2"] * scale), int(r["y2"] * scale)

                            step, kind = _step_at(segments, worker, t) if len(segments) else ("", "")
                            stopped = (
                                len(stops)
                                and (
                                    (stops["worker"] == worker)
                                    & (stops["start_s"] <= t)
                                    & (stops["end_s"] > t)
                                ).any()
                            )
                            box_colour = STOP_COLOUR if stopped else colour
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour,
                                          3 if stopped else 2)
                            _label(frame, worker, (x1, max(16, y1 - 24)), box_colour, 0.62, 2)
                            if step:
                                tag = f"STOPPED - {step}" if stopped else step
                                _label(frame, tag, (x1, max(30, y1 - 6)),
                                       STOP_COLOUR if stopped else (235, 235, 235), 0.5, 1)

                cv2.rectangle(frame, (0, 0), (out_w, 30), (24, 24, 24), -1)
                _label(frame, f"{format_hms(t)}   {video.name}", (10, 21),
                       (245, 245, 245), 0.58, 1)

                proc.stdin.write(frame.tobytes())
                idx += 1
                if idx % 10 == 0:
                    progress.update(task, completed=idx)
            progress.update(task, completed=total)
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode not in (0, None):
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")

    console.print(f"[green]Wrote[/green] {out_path} ({out_w}x{out_h}, {info.fps:.1f} fps)")
    return out_path
