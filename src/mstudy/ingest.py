"""Video reading and frame sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}


class VideoError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def describe(self) -> str:
        m, s = divmod(int(self.duration_s), 60)
        return f"{self.width}x{self.height} @ {self.fps:.2f} fps, {m}m{s:02d}s ({self.frame_count} frames)"


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise VideoError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(
            f"Could not open {path.name}. The file may be corrupt or use a codec "
            "OpenCV cannot read - try re-encoding with: "
            f'ffmpeg -i "{path.name}" -c:v libx264 -an fixed.mp4'
        )
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if fps <= 0 or fps > 1000:
        # Some phone/DVR files report nonsense. 25 fps is the safest guess and
        # only affects the timestamp mapping, which we recompute per frame.
        fps = 25.0
    if frame_count <= 0:
        raise VideoError(f"{path.name} reports zero frames - the file looks truncated.")

    return VideoInfo(path=path, fps=fps, frame_count=frame_count, width=width, height=height)


def _rotate(frame: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def iter_frames(
    info: VideoInfo,
    analysis_fps: float,
    rotate: int = 0,
    start_s: float = 0.0,
    max_seconds: float | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_seconds, frame) at the analysis rate.

    Frames we do not need are consumed with `grab()`, which advances the decoder
    without doing the expensive colour conversion. Sequential reading beats
    seeking here: seeking on long H.264 files lands on the nearest keyframe and
    silently shifts every timestamp, which would corrupt the whole time study.
    """
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise VideoError(f"Could not open {info.path}")

    stride = max(1, int(round(info.fps / max(analysis_fps, 0.01))))
    start_frame = int(start_s * info.fps)
    end_frame = info.frame_count
    if max_seconds is not None:
        end_frame = min(end_frame, int((start_s + max_seconds) * info.fps))

    try:
        idx = 0
        while idx < end_frame:
            if idx < start_frame or (idx - start_frame) % stride != 0:
                if not cap.grab():
                    break
                idx += 1
                continue

            ok, frame = cap.read()
            if not ok:
                break
            yield idx, idx / info.fps, _rotate(frame, rotate)
            idx += 1
    finally:
        cap.release()


def read_frame_at(info: VideoInfo, t_seconds: float, rotate: int = 0) -> np.ndarray:
    """Grab a single representative frame - used by the zone editor and reports."""
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise VideoError(f"Could not open {info.path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t_seconds * info.fps)))
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            raise VideoError(f"Could not read any frame from {info.path.name}")
        return _rotate(frame, rotate)
    finally:
        cap.release()


def find_videos(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
