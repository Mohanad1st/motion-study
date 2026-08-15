"""Tests for the industrial-engineering layer.

These run on synthetic tracks - no video, no models. The numbers a supervisor
will argue about are produced here, so this is the layer that has to be right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mstudy import LABEL_ABSENT, LABEL_OUTSIDE
from mstudy.config import Config, EventSettings, VideoSettings, Zone
from mstudy.events import analyse, build_segments, build_timeline, compute_motion, despeckle
from mstudy.geometry import assign_zones, foot_points, points_in_polygon

FPS = 2.0
DT = 1.0 / FPS
BODY_H = 100.0

# Three stations side by side, all at floor level y in [300, 500].
ZONES = [
    Zone(name="Cutting bench", polygon=[[100, 300], [200, 300], [200, 500], [100, 500]], kind="work"),
    Zone(name="Aisle", polygon=[[200, 300], [400, 300], [400, 500], [200, 500]], kind="walkway"),
    Zone(name="Welding jig", polygon=[[400, 300], [500, 300], [500, 500], [400, 500]], kind="work"),
]


def make_config(**event_overrides) -> Config:
    return Config(
        camera_id="test",
        zones=ZONES,
        steps={"Cutting bench": "Cut to length", "Welding jig": "Weld ledger"},
        video=VideoSettings(analysis_fps=FPS, pose_fps=FPS),
        events=EventSettings(
            min_segment_seconds=1.5,
            stop_motion_threshold=0.08,
            stop_min_seconds=5.0,
            motion_smoothing_seconds=1.0,
            left_min_seconds=2.0,
            missing_grace_seconds=1.0,
            **event_overrides,
        ),
    )


def make_tracks(spans: list[tuple[float, float, float, float]], worker: str = "Ali") -> pd.DataFrame:
    """Build synthetic tracks.

    spans: list of (start_s, end_s, centre_x, hand_movement_px_per_frame)
    """
    rows = []
    for start, end, cx, hand_move in spans:
        n = int(round((end - start) / DT))
        for i in range(n):
            t = start + i * DT
            # Hands drift steadily; magnitude sets whether this reads as working.
            hx = 500.0 + i * hand_move
            kp_x = [cx] * 17
            kp_y = [350.0] * 17
            for j in (7, 8, 9, 10):  # elbows and wrists
                kp_x[j] = hx
            rows.append(
                {
                    "worker": worker,
                    "t": t,
                    "x1": cx - 20, "y1": 400 - BODY_H, "x2": cx + 20, "y2": 400.0,
                    "conf": 0.9,
                    "has_pose": True,
                    "kp_x": kp_x, "kp_y": kp_y, "kp_conf": [0.9] * 17,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_foot_point_is_bottom_centre():
    pts = foot_points(np.array([[100, 200, 140, 400]]))
    assert pts.tolist() == [[120.0, 400.0]]


def test_point_in_polygon_basic():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    pts = np.array([[5, 5], [15, 5], [-1, -1], [9.9, 9.9]])
    assert points_in_polygon(pts, square).tolist() == [True, False, False, True]


def test_horizontal_edges_do_not_break_ray_casting():
    """A zone drawn with perfectly horizontal edges must not divide by zero."""
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert points_in_polygon(np.array([[5, 0.0]]), poly).size == 1


def test_overlapping_zones_resolve_in_config_order():
    zones = [
        Zone(name="Specific", polygon=[[0, 0], [10, 0], [10, 10], [0, 10]]),
        Zone(name="Broad", polygon=[[0, 0], [100, 0], [100, 100], [0, 100]]),
    ]
    labels = assign_zones(np.array([[5, 5], [50, 50]]), zones)
    assert labels.tolist() == ["Specific", "Broad"]


def test_point_outside_all_zones():
    assert assign_zones(np.array([[999, 999]]), ZONES).tolist() == [LABEL_OUTSIDE]


# --------------------------------------------------------------------------
# despeckle / hysteresis
# --------------------------------------------------------------------------


def test_despeckle_absorbs_short_runs():
    # 10 x A, 1 x B (a single-frame flicker), 10 x A  ->  all A
    labels = np.array(["A"] * 10 + ["B"] + ["A"] * 10, dtype=object)
    out = despeckle(labels, dt=DT, min_seconds=1.5)
    assert set(out.tolist()) == {"A"}


def test_despeckle_keeps_genuine_changes():
    labels = np.array(["A"] * 10 + ["B"] * 10, dtype=object)
    out = despeckle(labels, dt=DT, min_seconds=1.5)
    assert out[0] == "A" and out[-1] == "B"
    assert len(set(out.tolist())) == 2


def test_despeckle_converges_on_alternating_noise():
    """Pathological input must terminate, not spin forever."""
    labels = np.array(["A", "B"] * 20, dtype=object)
    out = despeckle(labels, dt=DT, min_seconds=1.5)
    assert len(set(out.tolist())) == 1


# --------------------------------------------------------------------------
# motion
# --------------------------------------------------------------------------


def test_motion_is_scale_invariant():
    """The same physical movement must read the same near and far from camera.

    This is why motion is normalised by body height: a pixel threshold would
    call distant workers idle while they work.
    """
    cfg = make_config()

    def motion_for(scale: float) -> float:
        rows = []
        for i in range(20):
            kp_x = [100.0] * 17
            kp_y = [100.0] * 17
            for j in (7, 8, 9, 10):
                kp_x[j] = 100.0 + i * 10.0 * scale
            rows.append(
                {
                    "worker": "W", "t": i * DT,
                    "x1": 0.0, "y1": 0.0, "x2": 40.0 * scale, "y2": BODY_H * scale,
                    "conf": 0.9, "has_pose": True,
                    "kp_x": kp_x, "kp_y": kp_y, "kp_conf": [0.9] * 17,
                }
            )
        return compute_motion(pd.DataFrame(rows), cfg)["motion"].median()

    near, far = motion_for(1.0), motion_for(0.5)
    assert near == pytest.approx(far, rel=1e-6)


def test_frames_without_pose_do_not_fabricate_motion():
    """Rows between pose samples inherit motion, never invent it from the box."""
    cfg = make_config()
    rows = []
    for i in range(10):
        has_pose = i % 2 == 0
        kp_x = [100.0] * 17
        for j in (7, 8, 9, 10):
            kp_x[j] = 100.0  # hands perfectly still
        rows.append(
            {
                "worker": "W", "t": i * DT,
                # Box jitters a lot; hands do not. Must still read as stopped.
                "x1": i * 30.0, "y1": 0.0, "x2": i * 30.0 + 40, "y2": BODY_H,
                "conf": 0.9, "has_pose": has_pose,
                "kp_x": kp_x, "kp_y": [100.0] * 17,
                "kp_conf": [0.9 if has_pose else 0.0] * 17,
            }
        )
    motion = compute_motion(pd.DataFrame(rows), cfg)["motion"]
    assert motion.max() < cfg.events.stop_motion_threshold


# --------------------------------------------------------------------------
# segments end-to-end
# --------------------------------------------------------------------------


def test_full_scenario_steps_stop_and_departure():
    """A realistic shift fragment, checked end to end.

    0-20 s   cutting bench, hands busy
    20-24 s  walking the aisle
    24-50 s  welding jig, with a 10 s stop from t=30
    50-60 s  gone from frame entirely
    """
    cfg = make_config()
    tracks = pd.concat(
        [
            make_tracks([(0, 20, 150, 12.0)]),          # working at bench
            make_tracks([(20, 24, 300, 12.0)]),         # walking
            make_tracks([(24, 30, 450, 12.0)]),         # welding, active
            make_tracks([(30, 40, 450, 0.05)]),         # STOPPED at the jig
            make_tracks([(40, 50, 450, 12.0)]),         # welding, active again
            # 50-60 s: no rows at all -> absent
        ],
        ignore_index=True,
    )

    result = analyse(tracks, cfg, duration_s=60.0)
    segs, stops = result["segments"], result["stops"]

    steps = segs[segs["kind"] == "work"]["step"].tolist()
    assert "Cut to length" in steps
    assert "Weld ledger" in steps

    bench = segs[segs["step"] == "Cut to length"].iloc[0]
    assert bench["duration_s"] == pytest.approx(20.0, abs=1.0)

    assert (segs["kind"] == "walking").any(), "aisle transit should be reported as walking"

    absent = segs[segs["kind"] == "absent"]
    assert len(absent) == 1
    assert absent.iloc[0]["duration_s"] == pytest.approx(10.0, abs=1.5)

    assert len(stops) == 1, f"expected exactly one stop, got {stops.to_dict('records')}"
    assert stops.iloc[0]["duration_s"] == pytest.approx(10.0, abs=2.0)
    assert stops.iloc[0]["step"] == "Weld ledger"


def test_brief_excursion_is_not_a_departure():
    """Stepping out of the zone for a moment must not be reported as leaving."""
    cfg = make_config()
    tracks = pd.concat(
        [
            make_tracks([(0, 20, 150, 12.0)]),
            # 1 second outside every zone
            make_tracks([(20, 21, 900, 12.0)]),
            make_tracks([(21, 40, 150, 12.0)]),
        ],
        ignore_index=True,
    )
    segs = analyse(tracks, cfg, duration_s=40.0)["segments"]
    assert not (segs["kind"] == "absent").any()
    # The flicker is absorbed by hysteresis rather than becoming its own step.
    assert len(segs[segs["kind"] == "work"]) == 1


def test_detection_dropout_is_bridged_not_reported_as_leaving():
    cfg = make_config()
    tracks = pd.concat(
        [make_tracks([(0, 20, 150, 12.0)]), make_tracks([(20.5, 40, 150, 12.0)])],
        ignore_index=True,
    )
    segs = analyse(tracks, cfg, duration_s=40.0)["segments"]
    assert not (segs["kind"] == "absent").any()


def test_two_workers_get_separate_rows():
    cfg = make_config()
    tracks = pd.concat(
        [
            make_tracks([(0, 30, 150, 12.0)], worker="Ali"),
            make_tracks([(0, 30, 450, 12.0)], worker="Omar"),
        ],
        ignore_index=True,
    )
    segs = analyse(tracks, cfg, duration_s=30.0)["segments"]
    assert set(segs["worker"]) == {"Ali", "Omar"}
    assert segs[segs["worker"] == "Ali"]["step"].iloc[0] == "Cut to length"
    assert segs[segs["worker"] == "Omar"]["step"].iloc[0] == "Weld ledger"


def test_empty_input_does_not_crash():
    cfg = make_config()
    empty = pd.DataFrame(
        columns=["worker", "t", "x1", "y1", "x2", "y2", "conf", "has_pose",
                 "kp_x", "kp_y", "kp_conf"]
    )
    result = analyse(empty, cfg, duration_s=10.0)
    assert result["segments"].empty
    assert result["stops"].empty
