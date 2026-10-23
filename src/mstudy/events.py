"""The industrial-engineering layer: turn raw tracks into work events.

This is the only module that encodes *time study rules* rather than computer
vision, and it is where the accuracy of every reported number is decided. It is
written against plain DataFrames so it can be tested on synthetic tracks with no
video, no model and no GPU.

Vocabulary
----------
segment   a continuous period one worker spent doing one thing
step      a segment inside a `work` zone - the thing being timed
walking   a segment inside a `walkway` zone - transport, a classic waste
off_station a segment visible in frame but in no defined zone
absent    a segment where the worker was not detected at all ("left")
stop      a period inside a step where the worker was present but not moving
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import LABEL_ABSENT, LABEL_OUTSIDE, WORK_KEYPOINTS
from .config import Config
from .geometry import assign_zones, foot_points

# Keypoints below this confidence are ignored when measuring motion; a
# hallucinated wrist jitters wildly and would mask a genuine stop.
MIN_KP_CONFIDENCE = 0.30


# ---------------------------------------------------------------------------
# 1. Motion
# ---------------------------------------------------------------------------


def compute_motion(tracks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add a `motion` column measured in BODY HEIGHTS PER SECOND.

    Normalising by the worker's on-screen height is what makes a single
    threshold valid across the frame: a worker at the back of the bay is half
    the pixels of one at the front, and a pixels-per-second threshold would
    call them idle while they work.

    Motion is taken from hands and forearms (see WORK_KEYPOINTS). A worker
    hand-tightening a cuplock stands perfectly still from the waist down, and a
    body-centroid metric would wrongly report them as stopped.
    """
    tracks = tracks.sort_values(["worker", "t"]).copy()
    if "has_pose" not in tracks.columns:
        tracks["has_pose"] = True
    out = []

    for worker, whole in tracks.groupby("worker", sort=False):
        whole = whole.reset_index(drop=True)
        whole["motion"] = np.nan

        # Motion is measured between CONSECUTIVE POSE SAMPLES only. Pose runs at
        # a lower rate than detection, and comparing a pose frame against a
        # frame that has no pose would silently fall back to box movement -
        # which is precisely the signal that fails for a worker whose hands are
        # busy but whose feet are planted.
        pose_idx = np.flatnonzero(whole["has_pose"].to_numpy())
        if len(pose_idx) < 2:
            whole["motion"] = 0.0
            out.append(whole)
            continue

        grp = whole.iloc[pose_idx].reset_index(drop=True)
        n = len(grp)
        motion = np.zeros(n, dtype=float)

        kp_x = np.stack(grp["kp_x"].to_numpy())  # (n, 17)
        kp_y = np.stack(grp["kp_y"].to_numpy())
        kp_c = np.stack(grp["kp_conf"].to_numpy())

        sel = np.array(WORK_KEYPOINTS)
        kx, ky, kc = kp_x[:, sel], kp_y[:, sel], kp_c[:, sel]

        dx = np.diff(kx, axis=0)
        dy = np.diff(ky, axis=0)
        dist = np.sqrt(dx**2 + dy**2)  # (n-1, len(sel))

        # A keypoint counts only if it was confident in BOTH frames of the pair.
        valid = (kc[:-1] >= MIN_KP_CONFIDENCE) & (kc[1:] >= MIN_KP_CONFIDENCE)
        dist_masked = np.where(valid, dist, np.nan)

        with np.errstate(invalid="ignore"):
            per_frame = np.nanmedian(dist_masked, axis=1)

        # Fallback for frames where pose was unusable: box centroid movement.
        cx = (grp["x1"].to_numpy() + grp["x2"].to_numpy()) / 2
        cy = (grp["y1"].to_numpy() + grp["y2"].to_numpy()) / 2
        centroid_move = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
        per_frame = np.where(np.isnan(per_frame), centroid_move, per_frame)

        body_h = np.clip(grp["y2"].to_numpy() - grp["y1"].to_numpy(), 1e-6, None)
        pair_h = (body_h[:-1] + body_h[1:]) / 2
        dt = np.clip(np.diff(grp["t"].to_numpy()), 1e-6, None)

        motion[1:] = per_frame / pair_h / dt
        motion[0] = motion[1]

        # Scatter back onto the full row set; rows between pose samples stay NaN
        # and are filled forward when the timeline is built.
        whole.loc[pose_idx, "motion"] = motion
        whole["motion"] = whole["motion"].ffill().bfill()
        out.append(whole)

    result = pd.concat(out, ignore_index=True) if out else tracks.assign(motion=0.0)

    # Smooth with a rolling median: robust to the single-frame keypoint jumps
    # that a mean would smear across the whole window.
    win = max(3, int(round(cfg.events.motion_smoothing_seconds * cfg.video.analysis_fps)))
    if win % 2 == 0:
        win += 1
    result["motion"] = (
        result.groupby("worker", sort=False)["motion"]
        .transform(lambda s: s.rolling(win, center=True, min_periods=1).median())
    )
    return result


# ---------------------------------------------------------------------------
# 2. Per-frame step labels
# ---------------------------------------------------------------------------


def label_zones(tracks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add a `zone` column naming the station each worker occupies per frame."""
    tracks = tracks.copy()
    if tracks.empty:
        tracks["zone"] = pd.Series(dtype=object)
        return tracks
    boxes = tracks[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    tracks["zone"] = assign_zones(foot_points(boxes), cfg.zones)
    return tracks


def build_timeline(
    tracks: pd.DataFrame, cfg: Config, duration_s: float
) -> pd.DataFrame:
    """Resample every worker onto one uniform clock spanning the whole video.

    Two things happen here that matter for the report:

    * Every worker gets a row for every time step, so a worker who is missing
      is explicitly ABSENT rather than silently skipped. That is what makes
      "left the station" measurable.
    * Short detection dropouts (a worker passes behind a stack of tubes) are
      bridged, so they do not masquerade as departures.
    """
    dt = 1.0 / cfg.video.analysis_fps
    grid = np.arange(0.0, max(duration_s, dt), dt)
    grace_frames = max(0, int(round(cfg.events.missing_grace_seconds / dt)))

    workers = sorted(tracks["worker"].unique()) if not tracks.empty else []
    frames = []

    for worker in workers:
        grp = tracks[tracks["worker"] == worker].sort_values("t")

        # Where a worker somehow has two detections at the same instant (tracker
        # split), keep the more confident one.
        grp = grp.sort_values("conf", ascending=False).drop_duplicates("t").sort_values("t")

        idx = np.searchsorted(grid, grp["t"].to_numpy())
        idx = np.clip(idx, 0, len(grid) - 1)

        zone = np.full(len(grid), LABEL_ABSENT, dtype=object)
        motion = np.full(len(grid), np.nan)
        present = np.zeros(len(grid), dtype=bool)

        zone[idx] = grp["zone"].to_numpy()
        motion[idx] = grp["motion"].to_numpy()
        present[idx] = True

        # Bridge dropouts: carry the last known zone forward, but only for as
        # long as the grace period allows.
        if grace_frames:
            last_zone, gap = LABEL_ABSENT, 10**9
            for i in range(len(grid)):
                if present[i]:
                    last_zone, gap = zone[i], 0
                else:
                    gap += 1
                    if gap <= grace_frames and last_zone != LABEL_ABSENT:
                        zone[i] = last_zone

        motion = pd.Series(motion).ffill().bfill().to_numpy()

        frames.append(
            pd.DataFrame(
                {
                    "worker": worker,
                    "t": grid,
                    "zone": zone,
                    "motion": motion,
                    "detected": present,
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=["worker", "t", "zone", "motion", "detected"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Hysteresis / despeckling
# ---------------------------------------------------------------------------


def _runs(labels: np.ndarray) -> list[tuple[int, int, object]]:
    """Run-length encode: list of (start_index, end_index_exclusive, label)."""
    if len(labels) == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate([[0], change, [len(labels)]])
    return [(int(bounds[i]), int(bounds[i + 1]), labels[bounds[i]]) for i in range(len(bounds) - 1)]


def despeckle(labels: np.ndarray, dt: float, min_seconds: float) -> np.ndarray:
    """Absorb runs shorter than `min_seconds` into their neighbours.

    Without this, a worker leaning across a boundary produces dozens of
    one-frame "steps" and the Gantt chart becomes unreadable confetti. We
    repeatedly dissolve the *shortest* offending run into whichever neighbour is
    longer, which converges and never flip-flops.
    """
    labels = np.asarray(labels, dtype=object).copy()
    if len(labels) == 0 or min_seconds <= 0:
        return labels

    min_len = max(1, int(round(min_seconds / dt)))

    while True:
        runs = _runs(labels)
        if len(runs) <= 1:
            break
        short = [r for r in runs if (r[1] - r[0]) < min_len]
        if not short:
            break

        # Dissolve the shortest run first.
        start, end, _ = min(short, key=lambda r: r[1] - r[0])
        pos = runs.index((start, end, labels[start]))
        prev_run = runs[pos - 1] if pos > 0 else None
        next_run = runs[pos + 1] if pos < len(runs) - 1 else None

        if prev_run is None:
            winner = next_run[2]
        elif next_run is None:
            winner = prev_run[2]
        else:
            prev_len = prev_run[1] - prev_run[0]
            next_len = next_run[1] - next_run[0]
            winner = prev_run[2] if prev_len >= next_len else next_run[2]

        labels[start:end] = winner

    return labels


# ---------------------------------------------------------------------------
# 4. Segments
# ---------------------------------------------------------------------------


def _classify(zone: str, cfg: Config) -> str:
    if zone == LABEL_ABSENT:
        return "absent"
    if zone == LABEL_OUTSIDE:
        return "off_station"
    kind = cfg.zone_kind(zone)
    return "walking" if kind == "walkway" else "work"


def build_segments(timeline: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Collapse the per-frame timeline into the time-study table.

    One row per (worker, continuous activity). This table is the direct source
    of the Gantt chart and of every duration statistic in the report.
    """
    dt = 1.0 / cfg.video.analysis_fps
    rows = []

    for worker, grp in timeline.groupby("worker", sort=True):
        grp = grp.sort_values("t").reset_index(drop=True)
        labels = despeckle(
            grp["zone"].to_numpy(dtype=object), dt, cfg.events.min_segment_seconds
        )
        times = grp["t"].to_numpy()
        motion = grp["motion"].to_numpy()

        for start, end, zone in _runs(labels):
            t0 = float(times[start])
            t1 = float(times[end - 1] + dt)
            kind = _classify(zone, cfg)

            # A momentary step outside a zone is not a departure.
            if kind in ("absent", "off_station") and (t1 - t0) < cfg.events.left_min_seconds:
                kind = "brief_excursion"

            seg_motion = motion[start:end]
            rows.append(
                {
                    "worker": worker,
                    "zone": zone,
                    "step": cfg.step_label(zone) if kind == "work" else zone,
                    "kind": kind,
                    "start_s": t0,
                    "end_s": t1,
                    "duration_s": t1 - t0,
                    "mean_motion": float(np.nanmean(seg_motion)) if len(seg_motion) else 0.0,
                    "start_frame_idx": start,
                }
            )

    cols = [
        "worker", "zone", "step", "kind", "start_s", "end_s",
        "duration_s", "mean_motion", "start_frame_idx",
    ]
    segments = pd.DataFrame(rows, columns=cols)
    if segments.empty:
        return segments
    return segments.sort_values(["worker", "start_s"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Stops
# ---------------------------------------------------------------------------


def detect_stops(timeline: pd.DataFrame, segments: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Find periods where a worker was at their station but not working.

    Only searched inside `work` segments: a worker standing still in the aisle
    is already counted as walking/idle, and double-reporting it would inflate
    the loss figures.
    """
    if timeline.empty or segments.empty:
        return pd.DataFrame(columns=["worker", "step", "start_s", "end_s", "duration_s"])

    dt = 1.0 / cfg.video.analysis_fps
    min_len = max(1, int(round(cfg.events.stop_min_seconds / dt)))
    rows = []

    work_segments = segments[segments["kind"] == "work"]

    for worker, grp in timeline.groupby("worker", sort=True):
        grp = grp.sort_values("t").reset_index(drop=True)
        times = grp["t"].to_numpy()
        still = (grp["motion"].to_numpy() < cfg.events.stop_motion_threshold) & grp[
            "detected"
        ].to_numpy()

        for _, seg in work_segments[work_segments["worker"] == worker].iterrows():
            in_seg = (times >= seg["start_s"]) & (times < seg["end_s"])
            local = still & in_seg
            for start, end, is_still in _runs(local.astype(object)):
                if not is_still or (end - start) < min_len:
                    continue
                rows.append(
                    {
                        "worker": worker,
                        "step": seg["step"],
                        "start_s": float(times[start]),
                        "end_s": float(times[end - 1] + dt),
                        "duration_s": float(times[end - 1] + dt - times[start]),
                    }
                )

    return pd.DataFrame(
        rows, columns=["worker", "step", "start_s", "end_s", "duration_s"]
    ).sort_values(["worker", "start_s"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Public entry point
# ---------------------------------------------------------------------------


def analyse(tracks: pd.DataFrame, cfg: Config, duration_s: float) -> dict[str, pd.DataFrame]:
    """Full raw-tracks -> work-events pipeline.

    Returns {'timeline', 'segments', 'stops'}.
    """
    tracks = label_zones(tracks, cfg)
    tracks = compute_motion(tracks, cfg)
    timeline = build_timeline(tracks, cfg, duration_s)
    segments = build_segments(timeline, cfg)
    stops = detect_stops(timeline, segments, cfg)
    return {"timeline": timeline, "segments": segments, "stops": stops}
