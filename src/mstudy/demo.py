"""Synthetic cuplock-ledger line, for seeing the report before any filming.

Everything downstream of the camera - steps, stops, departures, cycles, the
Gantt chart - is exercised by this, so the output format can be reviewed and
argued about before a single real video exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    Allowances, Config, CycleSettings, DetectionSettings, EventSettings,
    TimeStudySettings, VideoSettings, Zone,
)

FRAME_W, FRAME_H = 1280, 720
FPS = 2.0
POSE_FPS = 1.0
DT = 1.0 / FPS

STATIONS = [
    ("Material rack", 60, 240), ("Cutting bench", 300, 480),
    ("Press", 540, 720), ("Welding jig", 780, 960),
    ("Finished stack", 1010, 1220),
]
STATION_Y = (380, 620)
AISLE_Y = (170, 360)

# One repetition of the operation: station and its nominal seconds.
CYCLE = [
    ("Material rack", 11.0), ("Cutting bench", 34.0), ("Press", 22.0),
    ("Welding jig", 48.0), ("Finished stack", 13.0),
]
WALK_S = 5.0

WORKERS = [
    # name, pace multiplier (>1 is slower than nominal)
    ("Ahmed", 1.00),
    ("Mahmoud", 1.16),
    ("Youssef", 0.93),
]


def demo_config() -> Config:
    zones = [
        Zone(name=name, kind="work",
             polygon=[[x0, STATION_Y[0]], [x1, STATION_Y[0]],
                      [x1, STATION_Y[1]], [x0, STATION_Y[1]]])
        for name, x0, x1 in STATIONS
    ]
    zones.append(
        Zone(name="Aisle", kind="walkway",
             polygon=[[40, AISLE_Y[0]], [1240, AISLE_Y[0]],
                      [1240, AISLE_Y[1]], [40, AISLE_Y[1]]])
    )
    return Config(
        camera_id="cuplock_line_cam1",
        description="DEMO - synthetic data, not real footage",
        zones=zones,
        steps={
            "Material rack": "Load tube",
            "Cutting bench": "Cut to length",
            "Press": "Press cup",
            "Welding jig": "Weld ledger blade",
            "Finished stack": "Stack finished",
        },
        video=VideoSettings(analysis_fps=FPS, pose_fps=POSE_FPS),
        detection=DetectionSettings(),
        events=EventSettings(
            min_segment_seconds=1.5, stop_motion_threshold=0.08,
            stop_min_seconds=5.0, left_min_seconds=2.0, missing_grace_seconds=1.0,
        ),
        cycles=CycleSettings(boundary_zone="Material rack", min_cycle_seconds=30.0),
        time_study=TimeStudySettings(
            rating_factor=1.00,
            allowances=Allowances(personal=0.05, fatigue=0.07, delay=0.03),
            takt_seconds=150.0,
        ),
    )


def _station_centre(name: str) -> tuple[float, float]:
    for n, x0, x1 in STATIONS:
        if n == name:
            return (x0 + x1) / 2.0, 520.0
    return FRAME_W / 2.0, 265.0  # aisle


def _plan(
    rng: np.random.Generator,
    duration_s: float,
    pace: float,
    stop_spec: tuple[str, int, float] | None = None,
    leave_spec: tuple[str, int, float] | None = None,
):
    """Build a (start, end, station, working) schedule for one worker.

    Stops and departures are specified relative to a NAMED VISIT
    (station, which visit, how long) rather than an absolute clock time, so the
    injected event is guaranteed to land where it is meant to. Specifying them
    by wall-clock instead lets random cycle variation drift the worker into the
    aisle, and the "ground truth" then no longer describes what the data shows.
    """
    events, t, visits = [], 0.0, {}
    while t < duration_s:
        for station, nominal in CYCLE:
            if t >= duration_s:
                break
            visits[station] = visits.get(station, 0) + 1
            dur = max(4.0, nominal * pace * float(rng.normal(1.0, 0.11)))
            events.append([t, min(t + dur, duration_s), station, True, visits[station]])
            t += dur
            if t < duration_s:
                events.append([t, min(t + WALK_S, duration_s), "Aisle", True, 0])
                t += WALK_S

    truth: dict[str, tuple[float, float] | None] = {"stop": None, "leave": None}

    def split_visit(spec) -> None:
        """Carve an idle window out of the Nth visit to a station."""
        station, occurrence, length = spec
        for i, (a, b, st, _w, visit) in enumerate(events):
            if st != station or visit != occurrence:
                continue
            s0 = a + max(0.0, (b - a - length) / 2.0)
            s1 = min(b, s0 + length)
            if s1 - s0 < 1.0:
                return
            events[i:i + 1] = [
                seg for seg in ([a, s0, st, True, visit],
                                [s0, s1, st, False, visit],
                                [s1, b, st, True, visit])
                if seg[1] - seg[0] > 0.4
            ]
            truth["stop"] = (s0, s1)
            return

    def remove_window(spec):
        """Delete a stretch of time entirely - the worker walks off camera.

        Unlike a stop, a departure normally outlasts the visit it starts in, so
        this clips across however many segments the absence covers.
        """
        station, occurrence, length = spec
        start = next(
            (a for a, _b, st, _w, visit in events
             if st == station and visit == occurrence), None
        )
        if start is None:
            return
        end = start + length
        kept = []
        for a, b, st, w, visit in events:
            if b <= start or a >= end:
                kept.append([a, b, st, w, visit])
                continue
            if a < start:
                kept.append([a, start, st, w, visit])
            if b > end:
                kept.append([end, b, st, w, visit])
        events[:] = kept
        truth["leave"] = (start, end)

    if stop_spec:
        split_visit(stop_spec)      # present at station but idle
    if leave_spec:
        remove_window(leave_spec)   # off camera entirely

    plan = [(a, b, st, w) for a, b, st, w, _v in events if b - a > 0.4]
    return plan, truth


def generate(
    duration_s: float = 720.0, seed: int = 7
) -> tuple[pd.DataFrame, Config, float, dict]:
    """Returns (tracks, config, duration, ground_truth).

    `ground_truth` records exactly when each stop and departure was injected, so
    tests can measure how closely the detector recovers events whose real
    answers are known. That is the only accuracy check available before real
    footage exists.
    """
    rng = np.random.default_rng(seed)
    cfg = demo_config()
    pose_stride = int(round(FPS / POSE_FPS))

    specs = {
        "Ahmed": dict(pace=1.00),
        "Mahmoud": dict(pace=1.16, stop_spec=("Welding jig", 2, 18.0)),
        "Youssef": dict(pace=0.93, stop_spec=("Cutting bench", 3, 12.0),
                        leave_spec=("Material rack", 5, 45.0)),
    }
    plans, truth = {}, {"stops": [], "departures": []}
    for worker, spec in specs.items():
        plan, t = _plan(rng, duration_s, **spec)
        plans[worker] = plan
        if t["stop"]:
            truth["stops"].append((worker, *t["stop"]))
        if t["leave"]:
            truth["departures"].append((worker, *t["leave"]))

    rows = []
    for track_id, (worker, _) in enumerate(WORKERS, start=1):
        offset = (track_id - 1) * 26.0  # keep workers physically apart
        hand_phase = 0.0

        for a, b, station, working in plans[worker]:
            cx, cy = _station_centre(station)
            cx = float(np.clip(cx + offset, 70, FRAME_W - 70))

            n = max(1, int(round((b - a) / DT)))
            for i in range(n):
                t = a + i * DT
                if t >= duration_s:
                    break
                frame_idx = int(round(t * 25.0))
                jitter_x = float(rng.normal(0, 3.0))
                jitter_y = float(rng.normal(0, 2.0))
                # Walking drifts across the aisle rather than standing in it.
                if station == "Aisle":
                    cx_t = cx + (i / max(n - 1, 1)) * 90.0 - 45.0
                else:
                    cx_t = cx

                x_c, y_b = cx_t + jitter_x, cy + jitter_y
                body_h = 190.0
                has_pose = (i % pose_stride == 0)

                # Hands sweep while working; barely move while stopped.
                amp = 26.0 if working else 0.7
                hand_phase += 1.0
                hx = x_c + amp * np.sin(hand_phase * 0.9)
                hy = y_b - 95.0 + amp * 0.45 * np.cos(hand_phase * 0.7)

                kp_x = [x_c] * 17
                kp_y = [y_b - 150.0] * 17
                for j in (7, 8, 9, 10):
                    kp_x[j] = float(hx + (j - 8) * 6.0)
                    kp_y[j] = float(hy)

                rows.append({
                    "frame_idx": frame_idx, "t": float(t), "track_id": track_id,
                    "worker": worker,
                    "x1": x_c - 42.0, "y1": y_b - body_h,
                    "x2": x_c + 42.0, "y2": y_b,
                    "conf": float(np.clip(rng.normal(0.88, 0.05), 0.4, 0.99)),
                    "has_pose": has_pose,
                    "kp_x": kp_x, "kp_y": kp_y,
                    "kp_conf": [0.85 if has_pose else 0.0] * 17,
                })

    tracks = pd.DataFrame(rows).sort_values(["t", "track_id"]).reset_index(drop=True)
    return tracks, cfg, duration_s, truth


def background_image(cfg: Config) -> bytes:
    """A schematic 'floor' so the spaghetti diagram has something to sit on."""
    import cv2

    img = np.full((FRAME_H, FRAME_W, 3), 232, np.uint8)
    for zone in cfg.zones:
        poly = np.array(zone.polygon, np.int32)
        shade = (208, 226, 240) if zone.kind == "work" else (226, 226, 226)
        cv2.fillPoly(img, [poly], shade)
        cv2.polylines(img, [poly], True, (150, 158, 166), 2)
        x, y = poly[:, 0].min() + 10, poly[:, 1].min() + 26
        cv2.putText(img, zone.name, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 78, 86), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""
