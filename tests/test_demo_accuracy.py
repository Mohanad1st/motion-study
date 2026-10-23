"""Accuracy of the detector against known ground truth.

The synthetic line injects stops and departures at exactly known times, so we
can measure how well the event engine recovers them. This is the closest thing
to the stopwatch validation until real footage exists - and it is the test that
would catch a regression that silently shifts every reported duration.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mstudy.cycles import detect_cycles
from mstudy.demo import CYCLE, WALK_S, generate
from mstudy.events import analyse
from mstudy.metrics import step_statistics, worker_statistics

# At 2 fps, one frame is 0.5 s. Motion smoothing costs a couple of frames at the
# start of a stop, so a ~3 s tolerance is the honest bar for stop edges.
STOP_TOLERANCE_S = 4.0
DEPARTURE_TOLERANCE_S = 2.0


@pytest.fixture(scope="module")
def study():
    tracks, cfg, duration, truth = generate(duration_s=720.0)
    result = analyse(tracks, cfg, duration)
    cycles = detect_cycles(result["segments"], cfg)
    return dict(tracks=tracks, cfg=cfg, duration=duration, truth=truth,
                cycles=cycles, **result)


def test_every_injected_departure_is_found(study):
    absent = study["segments"]
    absent = absent[absent["kind"] == "absent"]

    for worker, start, end in study["truth"]["departures"]:
        match = absent[
            (absent["worker"] == worker)
            & (absent["start_s"] < end)
            & (absent["end_s"] > start)
        ]
        assert len(match) == 1, f"no departure detected for {worker} at {start:.0f}s"
        row = match.iloc[0]
        assert row["start_s"] == pytest.approx(start, abs=DEPARTURE_TOLERANCE_S)
        assert row["duration_s"] == pytest.approx(end - start, abs=2 * DEPARTURE_TOLERANCE_S)


def test_every_injected_stop_is_found(study):
    stops = study["stops"]
    for worker, start, end in study["truth"]["stops"]:
        match = stops[
            (stops["worker"] == worker)
            & (stops["start_s"] < end)
            & (stops["end_s"] > start)
        ]
        assert len(match) == 1, (
            f"no stop detected for {worker} at {start:.0f}-{end:.0f}s; "
            f"detected: {stops.to_dict('records')}"
        )
        assert match.iloc[0]["duration_s"] == pytest.approx(
            end - start, abs=STOP_TOLERANCE_S
        )


def test_no_phantom_stops(study):
    """Every reported stop must correspond to one we injected.

    False stops are worse than missed ones: they accuse a worker of idling when
    they were working.
    """
    truth = study["truth"]["stops"]
    for _, row in study["stops"].iterrows():
        assert any(
            row["worker"] == w and row["start_s"] < end and row["end_s"] > start
            for w, start, end in truth
        ), f"phantom stop reported: {row.to_dict()}"


def test_no_phantom_departures(study):
    truth = study["truth"]["departures"]
    absent = study["segments"]
    for _, row in absent[absent["kind"] == "absent"].iterrows():
        assert any(
            row["worker"] == w and row["start_s"] < end and row["end_s"] > start
            for w, start, end in truth
        ), f"phantom departure reported: {row.to_dict()}"


def test_step_times_track_the_nominal_durations(study):
    """Measured element times must land near the durations we simulated.

    Cycle-to-cycle variation is +/-11% by construction and each element is
    observed ~5 times, so the mean should sit well inside 25%.
    """
    stats = step_statistics(study["segments"], study["cfg"]).set_index("step")
    label = {
        "Material rack": "Load tube", "Cutting bench": "Cut to length",
        "Press": "Press cup", "Welding jig": "Weld ledger blade",
        "Finished stack": "Stack finished",
    }
    # Average pace across the three workers.
    mean_pace = (1.00 + 1.16 + 0.93) / 3

    for station, nominal in CYCLE:
        step = label[station]
        assert step in stats.index, f"{step} was never measured"
        measured = stats.loc[step, "mean_s"]
        expected = nominal * mean_pace
        assert measured == pytest.approx(expected, rel=0.25), (
            f"{step}: measured {measured:.1f}s vs expected ~{expected:.1f}s"
        )


def test_cycle_time_matches_the_simulated_cycle(study):
    nominal = sum(d for _, d in CYCLE) + WALK_S * len(CYCLE)
    mean_pace = (1.00 + 1.16 + 0.93) / 3
    # Walking is not scaled by pace, so scale only the work portion.
    expected = sum(d for _, d in CYCLE) * mean_pace + WALK_S * len(CYCLE)

    cycles = study["cycles"]
    assert not cycles.empty, "no cycles detected"
    assert cycles["duration_s"].mean() == pytest.approx(expected, rel=0.20), (
        f"mean cycle {cycles['duration_s'].mean():.1f}s vs expected ~{expected:.1f}s "
        f"(unpaced nominal {nominal:.0f}s)"
    )


def test_every_worker_and_station_appears(study):
    segs = study["segments"]
    assert set(segs["worker"]) == {"Ahmed", "Mahmoud", "Youssef"}
    work_steps = set(segs[segs["kind"] == "work"]["step"])
    assert work_steps == {
        "Load tube", "Cut to length", "Press cup",
        "Weld ledger blade", "Stack finished",
    }


def test_utilisation_is_plausible(study):
    """Workers who only walk between stations should still read as mostly busy."""
    workers = worker_statistics(
        study["segments"], study["stops"], study["cfg"], study["duration"]
    )
    assert len(workers) == 3
    for _, row in workers.iterrows():
        assert 60.0 <= row["utilisation_percent"] <= 95.0, row.to_dict()
        assert row["walking_percent"] > 0.0


def test_report_renders_end_to_end(tmp_path, study):
    """The whole chain must produce a real, self-contained HTML file."""
    from mstudy.demo import FRAME_H, FRAME_W, background_image
    from mstudy.study import run_study

    res = run_study(
        tracks=study["tracks"], cfg=study["cfg"], duration_s=study["duration"],
        out_dir=tmp_path, video_name="test",
        background_png=background_image(study["cfg"]), frame_size=(FRAME_W, FRAME_H),
    )
    html = res["report"].read_text(encoding="utf-8")
    assert len(html) > 500_000, "plotly.js should be embedded for offline use"
    assert "Motion &amp; Time Study" in html or "Motion & Time Study" in html
    for worker in ("Ahmed", "Mahmoud", "Youssef"):
        assert worker in html
    for f in ("segments.csv", "stops.csv", "step_statistics.csv",
              "worker_statistics.csv", "events.jsonl", "run.json"):
        assert (tmp_path / f).exists(), f"{f} was not written"
