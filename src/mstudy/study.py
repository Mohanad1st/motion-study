"""Orchestration: tracks -> every output artefact.

Kept separate from the video pipeline so the whole analysis can be re-run in
seconds from a saved tracks.parquet when only a threshold or a chart changed.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import __version__
from .breakdown import build_text
from .config import Config
from .cycles import assign_cycle_numbers, detect_cycles
from .events import analyse
from .metrics import (
    cycle_statistics, headline, step_statistics, worker_statistics, yamazumi_table,
)
from .report import render_report
from .swct import build_swct, swct_summary, swct_text


def _warnings(cfg: Config, segments: pd.DataFrame, tracks: pd.DataFrame,
              cycles: pd.DataFrame, duration_s: float) -> list[str]:
    """Honest caveats printed at the top of the report.

    A report that hides its own weak spots gets trusted once and then discarded.
    """
    out = []
    if segments.empty:
        out.append("No workers were detected. Check the camera view and the zone polygons.")
        return out

    if not cfg.zones:
        out.append("No zones are defined, so no work elements could be measured. "
                   "Run `mstudy zones` to draw the stations.")

    absent = segments[segments["kind"] == "absent"]["duration_s"].sum()
    if duration_s > 0 and absent / (duration_s * segments["worker"].nunique()) > 0.30:
        out.append(
            f"Workers were out of view for {absent / 60:.0f} minutes in total. "
            "Either the camera does not cover the whole operation, or detection is "
            "failing - check the annotated video before trusting the utilisation figures."
        )

    off = segments[segments["kind"] == "off_station"]["duration_s"].sum()
    on_screen = segments[segments["kind"] != "absent"]["duration_s"].sum()
    if on_screen > 0 and off / on_screen > 0.25:
        out.append(
            "More than a quarter of on-camera time fell outside every defined zone. "
            "Some stations are probably missing from the config."
        )

    if cfg.cycles.boundary_zone and cycles.empty:
        out.append(
            f"No complete cycles were found at boundary zone "
            f"'{cfg.cycles.boundary_zone}'. Cycle and Yamazumi figures are unavailable."
        )

    if not tracks.empty and "has_pose" in tracks.columns and not tracks["has_pose"].any():
        out.append("Pose estimation never ran, so stop detection is unavailable.")

    if cfg.time_study.rating_factor == 1.0:
        out.append(
            "Rating factor is 1.00 (the default). Standard times below assume the "
            "observed workers were performing at a normal pace - confirm that with a "
            "qualified observer before using these figures to set a standard."
        )
    return out


def run_study(
    tracks: pd.DataFrame,
    cfg: Config,
    duration_s: float,
    out_dir: Path,
    video_name: str,
    background_png: bytes | None = None,
    frame_size: tuple[int, int] = (1280, 720),
) -> dict:
    """Run the full analysis and write every artefact into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    result = analyse(tracks, cfg, duration_s)
    segments, stops, timeline = result["segments"], result["stops"], result["timeline"]

    cycles = detect_cycles(segments, cfg)
    segments = assign_cycle_numbers(segments, cycles)

    steps = step_statistics(segments, cfg)
    workers = worker_statistics(segments, stops, cfg, duration_s)
    cyc_stats = cycle_statistics(cycles)
    yama = yamazumi_table(segments, cycles)
    head = headline(workers, steps, cyc_stats)

    takt = cfg.time_study.takt_seconds
    swct = build_swct(segments, stops, cycles, cfg)
    swct_totals = swct_summary(swct, takt)

    # Raw tables, so the IE team can pivot the numbers themselves.
    segments.to_csv(out_dir / "segments.csv", index=False)
    stops.to_csv(out_dir / "stops.csv", index=False)
    steps.to_csv(out_dir / "step_statistics.csv", index=False)
    workers.to_csv(out_dir / "worker_statistics.csv", index=False)
    cycles.to_csv(out_dir / "cycles.csv", index=False)
    swct.to_csv(out_dir / "standard_work_combination.csv", index=False)

    # Plain-text breakdown: every step, in order, with its time.
    breakdown = build_text(
        segments=segments, stops=stops, cycles=cycles, step_stats=steps,
        worker_stats=workers, cfg=cfg, video_name=video_name, duration_s=duration_s,
    )
    breakdown += (
        "\n" + "=" * 96 + "\n STANDARD WORK COMBINATION TABLE\n" + "=" * 96 + "\n"
        "\n   Manual    = operator's hands are on the work\n"
        "   Auto/wait = operator present at the station but not moving. Video cannot\n"
        "               separate machine cycle time from avoidable waiting - that split\n"
        "               is a judgement for whoever knows the equipment cycle times.\n"
        "   Walk      = transport between stations, attributed to the element it follows\n"
        + swct_text(swct, takt)
    )
    (out_dir / "steps.txt").write_text(breakdown, encoding="utf-8")

    with (out_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        for _, r in segments.iterrows():
            fh.write(json.dumps({
                "type": "segment", "worker": r["worker"], "kind": r["kind"],
                "zone": r["zone"], "step": r["step"],
                "start_s": round(float(r["start_s"]), 2),
                "end_s": round(float(r["end_s"]), 2),
                "duration_s": round(float(r["duration_s"]), 2),
            }) + "\n")
        for _, r in stops.iterrows():
            fh.write(json.dumps({
                "type": "stop", "worker": r["worker"], "step": r["step"],
                "start_s": round(float(r["start_s"]), 2),
                "end_s": round(float(r["end_s"]), 2),
                "duration_s": round(float(r["duration_s"]), 2),
            }) + "\n")

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_path = render_report(
        out_path=out_dir / "report.html",
        cfg=cfg, video_name=video_name, duration_s=duration_s,
        segments=segments, stops=stops, tracks=tracks,
        step_stats=steps, worker_stats=workers, cycle_stats=cyc_stats,
        yamazumi=yama, headline=head,
        swct=swct, swct_totals=swct_totals, breakdown_text=breakdown,
        background_png=background_png, frame_size=frame_size,
        generated_at=generated_at,
        warnings=_warnings(cfg, segments, tracks, cycles, duration_s),
    )

    (out_dir / "run.json").write_text(json.dumps({
        "mstudy_version": __version__,
        "video": video_name,
        "generated_at": generated_at,
        "duration_s": round(duration_s, 2),
        "config_fingerprint": cfg.fingerprint(),
        "config": cfg.to_dict(),
        "headline": head,
    }, indent=2, default=str), encoding="utf-8")

    return {
        "report": report_path, "segments": segments, "stops": stops,
        "steps": steps, "workers": workers, "cycles": cycles,
        "timeline": timeline, "headline": head,
        "swct": swct, "swct_totals": swct_totals,
        "steps_txt": out_dir / "steps.txt", "breakdown": breakdown,
    }
