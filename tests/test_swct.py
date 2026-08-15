"""Standard Work Combination Table and the text step breakdown."""

from __future__ import annotations

import pandas as pd
import pytest

from mstudy.breakdown import build_text, consistency_label
from mstudy.cycles import assign_cycle_numbers, detect_cycles
from mstudy.demo import generate
from mstudy.events import analyse
from mstudy.metrics import step_statistics, worker_statistics
from mstudy.swct import build_swct, swct_figure, swct_summary, swct_text


@pytest.fixture(scope="module")
def study():
    tracks, cfg, duration, truth = generate(duration_s=720.0)
    res = analyse(tracks, cfg, duration)
    cycles = detect_cycles(res["segments"], cfg)
    segments = assign_cycle_numbers(res["segments"], cycles)
    swct = build_swct(segments, res["stops"], cycles, cfg)
    return dict(cfg=cfg, duration=duration, truth=truth, cycles=cycles,
                segments=segments, stops=res["stops"], swct=swct)


# --------------------------------------------------------------------------
# SWCT structure
# --------------------------------------------------------------------------


def test_swct_has_a_row_per_element_per_operator(study):
    swct = study["swct"]
    assert set(swct["worker"]) == {"Ahmed", "Mahmoud", "Youssef"}
    for worker, g in swct.groupby("worker"):
        assert set(g["element"]) == {
            "Load tube", "Cut to length", "Press cup",
            "Weld ledger blade", "Stack finished",
        }
        # Sequence numbers must be 1..n with no gaps, or the chart rows misalign.
        assert g["seq"].tolist() == list(range(1, len(g) + 1))


def test_elements_follow_the_real_cycle_order(study):
    """Row order must match the order the work is actually done in."""
    for _, g in study["swct"].groupby("worker"):
        order = g.sort_values("seq")["element"].tolist()
        assert order == [
            "Load tube", "Cut to length", "Press cup",
            "Weld ledger blade", "Stack finished",
        ]


def test_manual_plus_auto_plus_walk_equals_element_total(study):
    swct = study["swct"]
    computed = swct["manual_s"] + swct["auto_s"] + swct["walk_s"]
    pd.testing.assert_series_equal(
        computed, swct["element_total_s"], check_names=False, rtol=1e-9
    )


def test_cumulative_time_is_a_running_total(study):
    """Each element must start exactly where the previous one ended."""
    for _, g in study["swct"].groupby("worker"):
        g = g.sort_values("seq")
        assert g.iloc[0]["start_s"] == pytest.approx(0.0)
        starts = g["start_s"].to_numpy()[1:]
        prev_ends = g["cumulative_s"].to_numpy()[:-1]
        assert starts == pytest.approx(prev_ends)


def test_idle_time_lands_in_the_auto_column_not_manual(study):
    """A worker standing still at a station is auto/wait, never manual work."""
    swct, truth = study["swct"], study["truth"]
    stopped_workers = {w for w, _s, _e in truth["stops"]}

    for worker in stopped_workers:
        g = swct[swct["worker"] == worker]
        assert g["auto_s"].sum() > 0, f"{worker} had a stop but no auto/wait time"

    # And a worker with no stop must have none.
    clean = set(swct["worker"]) - stopped_workers
    for worker in clean:
        g = swct[swct["worker"] == worker]
        assert g["auto_s"].sum() == pytest.approx(0.0, abs=0.01)


def test_walking_is_charged_to_the_element_it_follows(study):
    """Every element except possibly the last is followed by a walk."""
    for _, g in study["swct"].groupby("worker"):
        assert g["walk_s"].sum() > 0
        assert (g.sort_values("seq")["walk_s"].to_numpy()[:-1] > 0).all()


def test_cycle_total_matches_the_measured_cycle_time(study):
    """SWCT cycle length must agree with independently detected cycle times."""
    swct, cycles = study["swct"], study["cycles"]
    for worker, g in swct.groupby("worker"):
        swct_cycle = g["element_total_s"].sum()
        measured = cycles[cycles["worker"] == worker]["duration_s"].mean()
        assert swct_cycle == pytest.approx(measured, rel=0.12), (
            f"{worker}: SWCT {swct_cycle:.1f}s vs measured cycle {measured:.1f}s"
        )


def test_slowest_operator_is_flagged_over_takt(study):
    """Mahmoud is the slow-paced operator; takt is 150s. He must be flagged."""
    totals = swct_summary(study["swct"], study["cfg"].time_study.takt_seconds)
    row = totals[totals["worker"] == "Mahmoud"].iloc[0]
    assert "OVER TAKT" in row["vs_takt"]

    fast = totals[totals["worker"] == "Youssef"].iloc[0]
    assert "within takt" in fast["vs_takt"]


def test_summary_totals_match_the_element_rows(study):
    totals = swct_summary(study["swct"], 150.0)
    for _, row in totals.iterrows():
        g = study["swct"][study["swct"]["worker"] == row["worker"]]
        assert row["manual_s"] == pytest.approx(g["manual_s"].sum(), abs=0.1)
        assert row["cycle_s"] == pytest.approx(g["element_total_s"].sum(), abs=0.1)


def test_swct_without_cycles_still_produces_rows(study):
    """No boundary zone configured must degrade gracefully, not crash."""
    empty_cycles = pd.DataFrame(
        columns=["worker", "cycle_no", "start_s", "end_s", "duration_s"]
    )
    swct = build_swct(study["segments"], study["stops"], empty_cycles, study["cfg"])
    assert not swct.empty
    assert set(swct["worker"]) == {"Ahmed", "Mahmoud", "Youssef"}


def test_swct_handles_empty_input():
    from mstudy.demo import demo_config

    empty = pd.DataFrame(columns=["worker", "kind", "step", "start_s", "end_s",
                                  "duration_s", "zone"])
    swct = build_swct(empty, empty, empty, demo_config())
    assert swct.empty
    assert swct_summary(swct, 150.0).empty
    assert "No standard work data" in swct_text(swct, 150.0)


def test_swct_figure_renders(study):
    fig = swct_figure(study["swct"], "Ahmed", 150.0)
    assert len(fig.data) > 0
    # Manual, auto and walk should each contribute at least one trace overall.
    names = {t.name for t in fig.data}
    assert any("Manual" in n for n in names)
    assert any("Walk" in n for n in names)


# --------------------------------------------------------------------------
# Text breakdown
# --------------------------------------------------------------------------


def test_consistency_bands():
    assert consistency_label(5) == "consistent"
    assert consistency_label(18) == "variable"
    assert consistency_label(40) == "UNSTABLE"


@pytest.fixture(scope="module")
def text(study):
    return build_text(
        segments=study["segments"], stops=study["stops"], cycles=study["cycles"],
        step_stats=step_statistics(study["segments"], study["cfg"]),
        worker_stats=worker_statistics(
            study["segments"], study["stops"], study["cfg"], study["duration"]),
        cfg=study["cfg"], video_name="demo", duration_s=study["duration"],
    )


def test_breakdown_lists_every_worker_and_element(text):
    for worker in ("AHMED", "MAHMOUD", "YOUSSEF"):
        assert worker in text
    for element in ("Load tube", "Cut to length", "Press cup",
                    "Weld ledger blade", "Stack finished"):
        assert element in text


def test_breakdown_shows_cycles_times_and_events(text):
    assert "CYCLE 1" in text
    assert "walking / transport" in text
    assert "STOPPED" in text
    assert "LEFT" in text
    assert "ELEMENT SUMMARY" in text


def test_breakdown_states_the_rating_factor_caveat(text):
    """The report must never let a standard time pass without the caveat."""
    assert "HUMAN judgement" in text
    assert "qualified observer" in text


def test_breakdown_has_no_trailing_whitespace(text):
    offenders = [ln for ln in text.splitlines() if ln != ln.rstrip()]
    assert not offenders, f"{len(offenders)} line(s) with trailing whitespace"


def test_swct_text_reports_totals_and_takt(study):
    out = swct_text(study["swct"], 150.0)
    assert "TOTAL" in out
    assert "Takt" in out
    assert "OVER TAKT" in out  # Mahmoud
