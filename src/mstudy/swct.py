"""Standard Work Combination Table (SWCT).

The lean standard-work chart: for each element of one cycle, how much time is
the operator's hands (manual), how much is the operator standing while something
else happens (auto/wait), and how much is walking - all laid on one time axis
against takt.

A note on what "auto" means here, because it matters when someone challenges the
chart. Classic SWCT distinguishes MACHINE time (the press is cycling, the
operator is free) from WAITING. Video cannot tell those apart: both look like a
person standing still at a station. So this chart reports the honest thing it
can measure - "operator not moving at the station" - and labels it AUTO / WAIT.
Splitting that column into real machine time versus avoidable waiting is a
judgement for whoever knows the equipment cycle times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import Config

MANUAL_COLOUR = "#0072B2"
AUTO_COLOUR = "#E69F00"
WALK_COLOUR = "#009E73"
TAKT_COLOUR = "#D62728"

SWCT_COLUMNS = [
    "worker", "seq", "element", "manual_s", "auto_s", "walk_s",
    "start_s", "element_total_s", "cumulative_s", "observations",
]


def _stopped_within(stops: pd.DataFrame, worker: str, start: float, end: float) -> float:
    """Seconds the worker was idle inside one work segment."""
    if stops.empty:
        return 0.0
    rows = stops[
        (stops["worker"] == worker) & (stops["start_s"] < end) & (stops["end_s"] > start)
    ]
    if rows.empty:
        return 0.0
    lo = rows["start_s"].clip(lower=start).to_numpy()
    hi = rows["end_s"].clip(upper=end).to_numpy()
    return float(np.clip(hi - lo, 0, None).sum())


def _sequence_for(
    segs: pd.DataFrame, stops: pd.DataFrame, worker: str
) -> list[dict]:
    """Walk one cycle in time order, splitting each element into manual/auto/walk.

    Walking is attributed to the element it FOLLOWS, which is the SWCT
    convention: the operator finishes at the press, then walks to the jig, and
    that walk belongs to the press row.
    """
    seq: list[dict] = []
    current: dict | None = None

    for _, seg in segs.sort_values("start_s").iterrows():
        kind = seg["kind"]
        if kind == "work":
            if current is not None:
                seq.append(current)
            idle = _stopped_within(stops, worker, seg["start_s"], seg["end_s"])
            current = {
                "element": seg["step"],
                "manual_s": max(0.0, float(seg["duration_s"]) - idle),
                "auto_s": idle,
                "walk_s": 0.0,
            }
        elif kind in ("walking", "off_station", "brief_excursion"):
            if current is not None:
                current["walk_s"] += float(seg["duration_s"])
        # 'absent' is excluded: a departure is not part of standard work.

    if current is not None:
        seq.append(current)
    return seq


def build_swct(
    segments: pd.DataFrame,
    stops: pd.DataFrame,
    cycles: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """One standard-work row set per worker, averaged across observed cycles.

    Standard work describes a REPEATABLE cycle, so each element is averaged over
    every cycle it was observed in rather than taken from a single lucky one.
    """
    if segments.empty:
        return pd.DataFrame(columns=SWCT_COLUMNS)

    rows: list[dict] = []

    for worker in sorted(segments["worker"].unique()):
        wsegs = segments[segments["worker"] == worker]
        wcycles = cycles[cycles["worker"] == worker] if not cycles.empty else cycles

        sequences: list[list[dict]] = []
        if not wcycles.empty:
            for _, cyc in wcycles.iterrows():
                in_cycle = wsegs[
                    (wsegs["start_s"] >= cyc["start_s"]) & (wsegs["start_s"] < cyc["end_s"])
                ]
                seq = _sequence_for(in_cycle, stops, worker)
                if seq:
                    sequences.append(seq)
        if not sequences:
            # No cycle boundary configured: treat the whole recording as one
            # pass and let the per-element averaging below do the work.
            seq = _sequence_for(wsegs, stops, worker)
            if not seq:
                continue
            sequences = [seq]

        # Average each element across cycles, and order by where it typically
        # falls in the cycle.
        agg: dict[str, dict] = {}
        for seq in sequences:
            for position, item in enumerate(seq):
                a = agg.setdefault(
                    item["element"],
                    {"manual": [], "auto": [], "walk": [], "positions": []},
                )
                a["manual"].append(item["manual_s"])
                a["auto"].append(item["auto_s"])
                a["walk"].append(item["walk_s"])
                a["positions"].append(position)

        ordered = sorted(agg.items(), key=lambda kv: float(np.mean(kv[1]["positions"])))

        cumulative = 0.0
        for seq_no, (element, a) in enumerate(ordered, start=1):
            manual = float(np.mean(a["manual"]))
            auto = float(np.mean(a["auto"]))
            walk = float(np.mean(a["walk"]))
            total = manual + auto + walk
            rows.append(
                {
                    "worker": worker, "seq": seq_no, "element": element,
                    "manual_s": manual, "auto_s": auto, "walk_s": walk,
                    "start_s": cumulative,
                    "element_total_s": total,
                    "cumulative_s": cumulative + total,
                    "observations": len(a["manual"]),
                }
            )
            cumulative += total

    return pd.DataFrame(rows, columns=SWCT_COLUMNS)


def _zigzag(x0: float, x1: float, y: float, amplitude: float = 0.17):
    """Triangle wave - the SWCT convention for walking time."""
    span = x1 - x0
    if span <= 0:
        return [x0], [y]
    teeth = max(2, min(24, int(span / 2.0)))
    xs = np.linspace(x0, x1, teeth * 2 + 1)
    ys = np.full(len(xs), y, dtype=float)
    ys[1::2] += amplitude
    ys[2::2] -= amplitude * 0.0
    return xs.tolist(), ys.tolist()


def swct_figure(swct: pd.DataFrame, worker: str, takt_seconds: float | None) -> go.Figure:
    """The chart for one operator."""
    fig = go.Figure()
    rows = swct[swct["worker"] == worker].sort_values("seq")
    if rows.empty:
        fig.add_annotation(text="No standard work data", showarrow=False)
        return fig

    labels = [f"{int(r.seq)}. {r.element}" for r in rows.itertuples()]
    y_of = {int(r.seq): i for i, r in enumerate(rows.itertuples())}
    first = True

    for r in rows.itertuples():
        y = y_of[int(r.seq)]
        x = float(r.start_s)

        if r.manual_s > 0:
            fig.add_trace(go.Scatter(
                x=[x, x + r.manual_s], y=[y, y], mode="lines",
                line=dict(color=MANUAL_COLOUR, width=7),
                name="Manual (operator working)", legendgroup="manual",
                showlegend=first,
                hovertemplate=f"<b>{r.element}</b><br>Manual: {r.manual_s:.1f}s<extra></extra>",
            ))
            x += r.manual_s

        if r.auto_s > 0:
            fig.add_trace(go.Scatter(
                x=[x, x + r.auto_s], y=[y, y], mode="lines",
                line=dict(color=AUTO_COLOUR, width=4, dash="dash"),
                name="Auto / wait (operator idle at station)", legendgroup="auto",
                showlegend=first,
                hovertemplate=f"<b>{r.element}</b><br>Auto/wait: {r.auto_s:.1f}s<extra></extra>",
            ))
            x += r.auto_s

        if r.walk_s > 0:
            zx, zy = _zigzag(x, x + r.walk_s, y)
            fig.add_trace(go.Scatter(
                x=zx, y=zy, mode="lines",
                line=dict(color=WALK_COLOUR, width=2),
                name="Walk", legendgroup="walk", showlegend=first,
                hovertemplate=f"Walk after {r.element}: {r.walk_s:.1f}s<extra></extra>",
            ))

        first = False

    total = float(rows["cumulative_s"].max())

    fig.add_vline(x=total, line=dict(color="#555", width=1.5, dash="dot"),
                  annotation_text=f"cycle {total:.0f}s", annotation_position="top left")

    if takt_seconds:
        fig.add_vline(x=takt_seconds, line=dict(color=TAKT_COLOUR, width=3),
                      annotation_text=f"TAKT {takt_seconds:.0f}s",
                      annotation_position="top right")
        if total > takt_seconds:
            # Over takt: this operator cannot keep up with demand.
            fig.add_vrect(
                x0=takt_seconds, x1=total, fillcolor=TAKT_COLOUR,
                opacity=0.10, line_width=0,
            )

    fig.update_layout(
        height=max(300, 46 * len(rows) + 170),
        xaxis=dict(title="Seconds from start of cycle", showgrid=True,
                   gridcolor="rgba(0,0,0,0.08)", zeroline=False,
                   range=[0, max(total, takt_seconds or 0) * 1.08]),
        yaxis=dict(
            tickmode="array", tickvals=list(range(len(labels))), ticktext=labels,
            autorange="reversed", showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0),
        margin=dict(l=10, r=20, t=64, b=46),
        plot_bgcolor="white", hovermode="closest",
    )
    return fig


def swct_summary(swct: pd.DataFrame, takt_seconds: float | None) -> pd.DataFrame:
    """Per-operator totals: manual, auto/wait, walk, cycle, and takt headroom."""
    cols = ["worker", "elements", "manual_s", "auto_wait_s", "walk_s",
            "cycle_s", "vs_takt"]
    if swct.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for worker, g in swct.groupby("worker"):
        cycle = float(g["element_total_s"].sum())
        if takt_seconds:
            gap = takt_seconds - cycle
            verdict = (f"{gap:+.0f}s ({'OVER TAKT' if gap < 0 else 'within takt'})")
        else:
            verdict = "no takt set"
        rows.append({
            "worker": worker,
            "elements": int(len(g)),
            "manual_s": round(float(g["manual_s"].sum()), 1),
            "auto_wait_s": round(float(g["auto_s"].sum()), 1),
            "walk_s": round(float(g["walk_s"].sum()), 1),
            "cycle_s": round(cycle, 1),
            "vs_takt": verdict,
        })
    return pd.DataFrame(rows, columns=cols)


def swct_text(swct: pd.DataFrame, takt_seconds: float | None) -> str:
    """Text rendering of the table, for the steps.txt breakdown."""
    if swct.empty:
        return "   No standard work data - set cycles.boundary_zone in the config.\n"

    lines: list[str] = []
    for worker, g in swct.groupby("worker"):
        g = g.sort_values("seq")
        cycle = float(g["element_total_s"].sum())
        lines.append("")
        lines.append(f"   {worker.upper()}")
        lines.append(f"   {'#':>3}  {'Element':<28}{'Manual':>9}{'Auto/wait':>11}"
                     f"{'Walk':>8}{'Total':>9}{'Cum.':>9}")
        for r in g.itertuples():
            lines.append(
                f"   {int(r.seq):>3}  {str(r.element)[:28]:<28}"
                f"{r.manual_s:>8.1f}s{r.auto_s:>10.1f}s{r.walk_s:>7.1f}s"
                f"{r.element_total_s:>8.1f}s{r.cumulative_s:>8.1f}s"
            )
        lines.append(
            f"   {'':>3}  {'TOTAL':<28}{g['manual_s'].sum():>8.1f}s"
            f"{g['auto_s'].sum():>10.1f}s{g['walk_s'].sum():>7.1f}s"
            f"{cycle:>8.1f}s"
        )
        if takt_seconds:
            gap = takt_seconds - cycle
            state = "OVER TAKT - cannot meet demand" if gap < 0 else "within takt"
            lines.append(f"        Takt {takt_seconds:.0f}s -> {gap:+.1f}s  ({state})")
    lines.append("")
    return "\n".join(lines)
