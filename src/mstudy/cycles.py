"""Cycle detection.

A cycle is one repetition of the operation. Everything a time study says about
consistency depends on splitting the recording into cycles correctly, so the
rule is deliberately simple and declared by the user rather than inferred: a new
cycle begins each time the worker re-enters a nominated boundary zone (typically
where they pick up the next piece of material).
"""

from __future__ import annotations

import pandas as pd

from .config import Config


def detect_cycles(segments: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return one row per completed cycle: worker, cycle_no, start_s, end_s, duration_s.

    Only *completed* cycles are returned. The trailing partial cycle at the end
    of the recording is dropped, because including a half-finished repetition
    would drag the average cycle time down and understate the real workload.
    """
    empty = pd.DataFrame(columns=["worker", "cycle_no", "start_s", "end_s", "duration_s"])
    boundary = cfg.cycles.boundary_zone
    if not boundary or segments.empty:
        return empty

    rows = []
    for worker, grp in segments.groupby("worker", sort=True):
        grp = grp.sort_values("start_s")
        starts = grp[grp["zone"] == boundary]["start_s"].tolist()
        if len(starts) < 2:
            continue

        cycle_no = 0
        for a, b in zip(starts[:-1], starts[1:]):
            if (b - a) < cfg.cycles.min_cycle_seconds:
                # Two entries in quick succession are one visit split by a
                # momentary step out, not two cycles.
                continue
            cycle_no += 1
            rows.append(
                {
                    "worker": worker,
                    "cycle_no": cycle_no,
                    "start_s": float(a),
                    "end_s": float(b),
                    "duration_s": float(b - a),
                }
            )

    return pd.DataFrame(rows, columns=empty.columns) if rows else empty


def assign_cycle_numbers(segments: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    """Tag each segment with the cycle it belongs to (NaN outside any cycle)."""
    segments = segments.copy()
    segments["cycle_no"] = pd.NA
    if cycles.empty or segments.empty:
        return segments

    for _, cyc in cycles.iterrows():
        mask = (
            (segments["worker"] == cyc["worker"])
            & (segments["start_s"] >= cyc["start_s"])
            & (segments["start_s"] < cyc["end_s"])
        )
        segments.loc[mask, "cycle_no"] = cyc["cycle_no"]
    return segments
