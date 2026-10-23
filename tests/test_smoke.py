"""Import and wiring checks.

Cheap tests that catch the errors which would otherwise only appear an hour
into a real run: a broken import in a rarely-used module, a CLI command that
does not resolve, a config file that no longer parses.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

MODULES = [
    "mstudy", "mstudy.annotate", "mstudy.cli", "mstudy.config", "mstudy.cycles",
    "mstudy.demo", "mstudy.detect", "mstudy.events", "mstudy.geometry",
    "mstudy.identify", "mstudy.ingest", "mstudy.metrics", "mstudy.pipeline",
    "mstudy.report", "mstudy.study", "mstudy.track", "mstudy.zones",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_cli_commands_are_registered():
    from mstudy.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "track", "name", "report", "zones", "annotate", "demo", "watch"):
        assert command in result.stdout, f"`{command}` missing from CLI help"


def test_example_config_parses():
    from mstudy.config import load_config

    cfg = load_config(REPO_ROOT / "configs" / "example_line.yaml")
    assert cfg.camera_id
    assert cfg.video.pose_fps <= cfg.video.analysis_fps
    assert cfg.time_study.allowances.total == pytest.approx(0.15)


def test_report_template_ships_with_the_package():
    from mstudy.report import TEMPLATE_DIR

    assert (TEMPLATE_DIR / "report.html.jinja").exists()


def test_pose_faster_than_analysis_is_rejected():
    """A config that asks for more pose than frames must fail loudly, not silently."""
    from mstudy.config import ConfigError, VideoSettings

    with pytest.raises(ConfigError):
        VideoSettings(analysis_fps=1.0, pose_fps=4.0)


def test_duplicate_zone_names_are_rejected(tmp_path):
    from mstudy.config import ConfigError, load_config

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "camera_id: x\n"
        "zones:\n"
        "  - {name: Bench, polygon: [[0,0],[1,0],[1,1]]}\n"
        "  - {name: Bench, polygon: [[2,2],[3,2],[3,3]]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Duplicate zone"):
        load_config(bad)


def test_unknown_boundary_zone_is_rejected(tmp_path):
    from mstudy.config import ConfigError, load_config

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "camera_id: x\n"
        "zones:\n"
        "  - {name: Bench, polygon: [[0,0],[1,0],[1,1]]}\n"
        "cycles:\n"
        "  boundary_zone: Nonexistent\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="boundary_zone"):
        load_config(bad)


def test_names_merge_two_tracks_into_one_worker():
    """A worker who left and returned must land on ONE Gantt row, not two."""
    import pandas as pd

    from mstudy.identify import apply_names

    tracks = pd.DataFrame({"track_id": [1, 2, 3], "t": [0.0, 1.0, 2.0]})
    named = apply_names(tracks, {1: "Ali", 2: "Ali", 3: "__ignore__"})
    assert named["worker"].tolist() == ["Ali", "Ali"]


def test_unnamed_tracks_are_kept_not_dropped():
    import pandas as pd

    from mstudy.identify import apply_names

    tracks = pd.DataFrame({"track_id": [7], "t": [0.0]})
    assert apply_names(tracks, {})["worker"].tolist() == ["Worker #7"]
