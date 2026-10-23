"""Tracks survive a save/load cycle unchanged.

The real pipeline always writes tracks.parquet and reads it back in a separate
process, and parquet returns the keypoint list columns as NumPy arrays rather
than Python lists. If the motion maths did not cope with that, every test above
would still pass while every real run produced wrong stop detection.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mstudy.demo import generate
from mstudy.events import analyse
from mstudy.identify import apply_names, track_summary
from mstudy.pipeline import TRACK_COLUMNS


@pytest.fixture(scope="module")
def demo_data():
    return generate(duration_s=300.0)


def test_analysis_is_identical_after_a_parquet_round_trip(tmp_path, demo_data):
    tracks, cfg, duration, _ = demo_data

    before = analyse(tracks, cfg, duration)["segments"]

    path = tmp_path / "tracks.parquet"
    tracks.to_parquet(path, index=False)
    reloaded = pd.read_parquet(path)

    # Parquet gives back arrays, not lists - the thing this test exists for.
    assert not isinstance(reloaded["kp_x"].iloc[0], list)

    after = analyse(reloaded, cfg, duration)["segments"]

    pd.testing.assert_frame_equal(
        before.reset_index(drop=True), after.reset_index(drop=True)
    )


def test_stops_survive_the_round_trip(tmp_path, demo_data):
    tracks, cfg, duration, truth = demo_data
    path = tmp_path / "t.parquet"
    tracks.to_parquet(path, index=False)

    stops = analyse(pd.read_parquet(path), cfg, duration)["stops"]
    expected = [s for s in truth["stops"] if s[1] < duration]
    assert len(stops) == len(expected)


def test_pipeline_columns_match_what_the_analysis_needs(demo_data):
    """The writer and the reader must agree on the schema."""
    tracks, _, _, _ = demo_data
    required = {"frame_idx", "t", "track_id", "x1", "y1", "x2", "y2",
                "conf", "has_pose", "kp_x", "kp_y", "kp_conf"}
    assert required <= set(TRACK_COLUMNS)
    assert required <= set(tracks.columns)


def test_naming_then_analysis_works_on_reloaded_tracks(tmp_path, demo_data):
    """The real path: track ids from parquet, names applied, then analysed."""
    tracks, cfg, duration, _ = demo_data

    raw = tracks.drop(columns=["worker"])
    path = tmp_path / "t.parquet"
    raw.to_parquet(path, index=False)
    reloaded = pd.read_parquet(path)

    summary = track_summary(reloaded)
    assert len(summary) == 3

    # Two track ids deliberately given the same name, as a returning worker would be.
    names = {1: "Ali", 2: "Ali", 3: "Omar"}
    named = apply_names(reloaded, names)
    assert set(named["worker"]) == {"Ali", "Omar"}

    segments = analyse(named, cfg, duration)["segments"]
    assert set(segments["worker"]) == {"Ali", "Omar"}
