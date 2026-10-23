"""Multi-worker tracking.

Uses ByteTrack from Roboflow's `trackers` package (Apache-2.0) rather than the
copy inside `supervision`, which is deprecated and scheduled for removal.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from .config import Config


def build_tracker(cfg: Config) -> ByteTrackTracker:
    """Create a tracker tuned for a LOW analysis frame rate.

    The defaults assume ~30 fps video. We feed it 2 fps, so between two frames a
    walking worker moves most of their own body width and the overlap between
    successive boxes is far smaller than the tracker expects. Two adjustments
    follow from that:

    * `minimum_iou_threshold` is loosened, or a walking worker would be dropped
      and re-acquired as a new person every second.
    * `frame_rate` is told the truth, so the internal motion model and the
      lost-track timeout are expressed in real seconds rather than being 15x
      too short.

    Identity mistakes that survive this are absorbed downstream: the worker
    naming step maps several track IDs onto one person, so a re-acquired worker
    still lands on a single row of the Gantt chart.
    """
    fps = cfg.video.analysis_fps
    return ByteTrackTracker(
        frame_rate=fps,
        # Keep a lost worker's identity alive for ~10 seconds of real time.
        lost_track_buffer=max(5, int(round(10.0 * fps))),
        track_activation_threshold=max(0.30, cfg.detection.min_confidence),
        high_conf_det_threshold=max(0.45, cfg.detection.min_confidence + 0.1),
        minimum_iou_threshold=0.20,
        # At 2 fps, demanding several consecutive frames would delay every
        # worker's first appearance by seconds and lose the start of the cycle.
        minimum_consecutive_frames=1,
    )


def to_detections(boxes: np.ndarray, scores: np.ndarray) -> sv.Detections:
    if len(boxes) == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.asarray(boxes, np.float32).reshape(-1, 4),
        confidence=np.asarray(scores, np.float32).reshape(-1),
        class_id=np.zeros(len(boxes), dtype=int),
    )


def update(tracker: ByteTrackTracker, boxes: np.ndarray, scores: np.ndarray):
    """Advance the tracker one frame.

    Returns (boxes, scores, track_ids) keeping only CONFIRMED tracks. The
    tracker marks provisional detections with id -1; letting those through would
    create phantom one-frame workers on the Gantt chart.
    """
    dets = tracker.update(to_detections(boxes, scores))
    if dets is None or len(dets) == 0 or dets.tracker_id is None:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), int)

    ids = np.asarray(dets.tracker_id)
    keep = ids > 0
    conf = dets.confidence if dets.confidence is not None else np.ones(len(dets), np.float32)
    return (
        np.asarray(dets.xyxy, np.float32)[keep],
        np.asarray(conf, np.float32)[keep],
        ids[keep].astype(int),
    )
