"""Build the interactive HTML report.

One self-contained file: no server, no internet, no external assets. It has to
open on a supervisor's laptop in a factory office with the WiFi down.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import Config
from .metrics import format_hms

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Okabe-Ito: distinguishable for the ~8% of men with colour vision deficiency,
# which in a factory review meeting is not a hypothetical.
WORKER_COLOURS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#8C6D31", "#5D3A9B",
]
STEP_COLOURS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#79706E",
]

KIND_STYLE = {
    "work": dict(opacity=1.0, pattern=""),
    "walking": dict(opacity=0.55, pattern="/"),
    "off_station": dict(opacity=0.30, pattern="."),
    "brief_excursion": dict(opacity=0.30, pattern="."),
    "absent": dict(opacity=0.12, pattern="x"),
}

KIND_LABEL = {
    "work": "At station (working)",
    "walking": "Walking / transport",
    "off_station": "Off station (in view)",
    "brief_excursion": "Brief step away",
    "absent": "Left / out of view",
}


def _colour_map(names: list[str], palette: list[str]) -> dict[str, str]:
    return {n: palette[i % len(palette)] for i, n in enumerate(sorted(names))}


def _axis_ticks(duration_s: float) -> tuple[list[float], list[str]]:
    """Tick every whole minute (or 30 s for short clips), labelled mm:ss."""
    step = 30.0 if duration_s <= 600 else 60.0 if duration_s <= 3600 else 300.0
    vals = list(np.arange(0, duration_s + step, step))
    return vals, [format_hms(v) for v in vals]


def build_gantt(
    segments: pd.DataFrame,
    stops: pd.DataFrame,
    duration_s: float,
    colour_by: str = "worker",
) -> go.Figure:
    """The multi-worker Gantt: one row per employee, across the operation clock.

    Colour carries identity (or work element, on the second tab). Texture
    carries activity type, so walking and absence stay readable even in a
    printed black-and-white copy of the report.
    """
    fig = go.Figure()
    if segments.empty:
        fig.add_annotation(text="No activity detected", showarrow=False)
        return fig

    workers = sorted(segments["worker"].unique())
    steps = sorted(segments[segments["kind"] == "work"]["step"].unique())
    worker_colours = _colour_map(workers, WORKER_COLOURS)
    step_colours = _colour_map(steps, STEP_COLOURS)

    seen_legend: set[str] = set()

    for kind in ("work", "walking", "off_station", "brief_excursion", "absent"):
        subset = segments[segments["kind"] == kind]
        if subset.empty:
            continue
        style = KIND_STYLE[kind]

        for worker in workers:
            rows = subset[subset["worker"] == worker]
            if rows.empty:
                continue

            if colour_by == "worker":
                colours = [worker_colours[worker]] * len(rows)
                legend_key = worker
            else:
                colours = [
                    step_colours.get(s, "#B0B0B0") if kind == "work" else "#B0B0B0"
                    for s in rows["step"]
                ]
                legend_key = f"{kind}"

            show_legend = legend_key not in seen_legend and kind == "work"
            if show_legend:
                seen_legend.add(legend_key)

            # Only label bars wide enough to hold text, or the chart turns to soup.
            labels = [
                (s if d > duration_s * 0.035 else "")
                for s, d in zip(rows["step"], rows["duration_s"])
            ]

            fig.add_trace(
                go.Bar(
                    x=rows["duration_s"],
                    y=[worker] * len(rows),
                    base=rows["start_s"],
                    orientation="h",
                    marker=dict(
                        color=colours,
                        opacity=style["opacity"],
                        line=dict(color="rgba(0,0,0,0.35)", width=1),
                        pattern=dict(shape=style["pattern"], fgcolor="rgba(255,255,255,0.55)",
                                     size=6, solidity=0.25),
                    ),
                    text=labels,
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(size=11, color="white"),
                    cliponaxis=False,
                    name=legend_key if colour_by == "worker" else KIND_LABEL[kind],
                    legendgroup=legend_key,
                    showlegend=show_legend,
                    customdata=np.stack(
                        [rows["step"], rows["duration_s"],
                         [KIND_LABEL[kind]] * len(rows),
                         [format_hms(v) for v in rows["start_s"]]],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>%{customdata[2]}<br>"
                        "Element: %{customdata[0]}<br>"
                        "Starts: %{customdata[3]}<br>"
                        "Duration: %{customdata[1]:.1f} s<extra></extra>"
                    ),
                )
            )

    # --- event markers -------------------------------------------------
    # STOPPED: worker present at the station but not moving.
    if not stops.empty:
        fig.add_trace(
            go.Scatter(
                x=stops["start_s"] + stops["duration_s"] / 2,
                y=stops["worker"],
                mode="markers",
                marker=dict(
                    symbol="square", size=13, color="#FFB000",
                    line=dict(color="#5A3E00", width=1.6),
                ),
                name="■ Stopped (idle at station)",
                customdata=np.stack(
                    [stops["duration_s"], [format_hms(v) for v in stops["start_s"]],
                     stops["step"]], axis=-1
                ),
                hovertemplate=(
                    "<b>%{y}</b> — STOPPED<br>At: %{customdata[1]}<br>"
                    "For: %{customdata[0]:.1f} s<br>"
                    "During: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    # LEFT: worker gone from view.
    departures = segments[segments["kind"] == "absent"]
    if not departures.empty:
        fig.add_trace(
            go.Scatter(
                x=departures["start_s"] + departures["duration_s"] / 2,
                y=departures["worker"],
                mode="markers",
                marker=dict(
                    symbol="x", size=15, color="#D62728",
                    line=dict(color="#5A0000", width=1.2),
                ),
                name="✖ Left (out of view)",
                customdata=np.stack(
                    [departures["duration_s"],
                     [format_hms(v) for v in departures["start_s"]]], axis=-1
                ),
                hovertemplate=(
                    "<b>%{y}</b> — LEFT<br>At: %{customdata[1]}<br>"
                    "For: %{customdata[0]:.1f} s<extra></extra>"
                ),
            )
        )

    tickvals, ticktext = _axis_ticks(duration_s)
    fig.update_layout(
        barmode="overlay",
        height=max(320, 90 * len(workers) + 190),
        xaxis=dict(
            title="Elapsed time (mm:ss)", tickvals=tickvals, ticktext=ticktext,
            range=[0, duration_s], showgrid=True, gridcolor="rgba(0,0,0,0.08)",
        ),
        yaxis=dict(title="", categoryorder="array", categoryarray=workers[::-1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=20, t=60, b=50),
        plot_bgcolor="white",
        bargap=0.45,
        hovermode="closest",
    )
    return fig


def build_spaghetti(
    tracks: pd.DataFrame, background_png: bytes | None, width: int, height: int
) -> go.Figure:
    """Movement paths drawn over the workshop floor.

    The classic motion-study diagram. Excess walking is very hard to argue with
    once it is a picture.
    """
    fig = go.Figure()
    if tracks.empty:
        fig.add_annotation(text="No movement data", showarrow=False)
        return fig

    workers = sorted(tracks["worker"].unique())
    colours = _colour_map(workers, WORKER_COLOURS)

    if background_png:
        uri = "data:image/png;base64," + base64.b64encode(background_png).decode()
        fig.add_layout_image(
            dict(source=uri, xref="x", yref="y", x=0, y=0,
                 sizex=width, sizey=height, sizing="stretch", layer="below", opacity=0.55)
        )

    for worker in workers:
        rows = tracks[tracks["worker"] == worker].sort_values("t")
        fx = (rows["x1"] + rows["x2"]) / 2
        fy = rows["y2"]
        fig.add_trace(
            go.Scatter(
                x=fx, y=fy, mode="lines",
                line=dict(color=colours[worker], width=1.8),
                opacity=0.75, name=worker,
                hovertemplate=f"<b>{worker}</b><br>%{{x:.0f}}, %{{y:.0f}}<extra></extra>",
            )
        )

    fig.update_layout(
        height=max(380, int(380 * height / max(width, 1)) + 80),
        xaxis=dict(range=[0, width], visible=False, constrain="domain"),
        # Image coordinates run downward, so the y axis is reversed to match.
        yaxis=dict(range=[height, 0], visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#f4f4f4",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    return fig


def build_yamazumi(yamazumi: pd.DataFrame, takt_seconds: float | None) -> go.Figure:
    """Stacked work content per operator - the line-balancing view."""
    fig = go.Figure()
    if yamazumi.empty:
        fig.add_annotation(text="No cycle data - set cycles.boundary_zone to enable",
                           showarrow=False)
        return fig

    steps = sorted(yamazumi["step"].unique())
    colours = _colour_map(steps, STEP_COLOURS)
    workers = sorted(yamazumi["worker"].unique())

    for step in steps:
        rows = yamazumi[yamazumi["step"] == step].set_index("worker")
        fig.add_trace(
            go.Bar(
                x=workers,
                y=[rows["seconds_per_cycle"].get(w, 0.0) for w in workers],
                name=step,
                marker_color=colours[step],
                hovertemplate=f"<b>%{{x}}</b><br>{step}: %{{y:.1f}} s/cycle<extra></extra>",
            )
        )

    if takt_seconds:
        fig.add_hline(
            y=takt_seconds, line=dict(color="#D62728", width=2, dash="dash"),
            annotation_text=f"Takt {takt_seconds:.0f}s", annotation_position="top right",
        )

    fig.update_layout(
        barmode="stack", height=420,
        yaxis_title="Seconds per cycle", xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=40), plot_bgcolor="white",
    )
    return fig


def _fig_html(fig: go.Figure, include_js: bool) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=True if include_js else False,
        config={"displaylogo": False, "responsive": True},
    )


def _table(df: pd.DataFrame, rounding: dict[str, int] | None = None) -> str:
    if df.empty:
        return "<p class='empty'>No data.</p>"
    shown = df.copy()
    for col, digits in (rounding or {}).items():
        if col in shown.columns:
            shown[col] = shown[col].astype(float).round(digits)
    return shown.to_html(index=False, classes="data", border=0, justify="left")


def render_report(
    out_path: Path,
    cfg: Config,
    video_name: str,
    duration_s: float,
    segments: pd.DataFrame,
    stops: pd.DataFrame,
    tracks: pd.DataFrame,
    step_stats: pd.DataFrame,
    worker_stats: pd.DataFrame,
    cycle_stats: pd.DataFrame,
    yamazumi: pd.DataFrame,
    headline: dict,
    swct: pd.DataFrame,
    swct_totals: pd.DataFrame,
    breakdown_text: str,
    background_png: bytes | None,
    frame_size: tuple[int, int],
    generated_at: str,
    warnings: list[str],
) -> Path:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.jinja")

    gantt_worker = build_gantt(segments, stops, duration_s, colour_by="worker")
    gantt_step = build_gantt(segments, stops, duration_s, colour_by="step")
    spaghetti = build_spaghetti(tracks, background_png, *frame_size)
    yama = build_yamazumi(yamazumi, cfg.time_study.takt_seconds)

    # One Standard Work Combination chart per operator - standard work describes
    # one person's repeatable cycle, so they are never merged onto one axis.
    from .swct import swct_figure

    swct_blocks = []
    for worker in (sorted(swct["worker"].unique()) if not swct.empty else []):
        fig = swct_figure(swct, worker, cfg.time_study.takt_seconds)
        rows = swct[swct["worker"] == worker]
        swct_blocks.append({
            "worker": worker,
            "html": _fig_html(fig, include_js=False),
            "cycle_s": float(rows["element_total_s"].sum()),
            "observations": int(rows["observations"].max()) if len(rows) else 0,
        })

    events = pd.concat(
        [
            stops.assign(event="STOPPED")[["worker", "event", "start_s", "duration_s", "step"]]
            if not stops.empty else pd.DataFrame(),
            segments[segments["kind"] == "absent"].assign(event="LEFT")[
                ["worker", "event", "start_s", "duration_s"]
            ]
            if not segments.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    if not events.empty:
        events = events.sort_values("start_s")
        events.insert(2, "at", [format_hms(v) for v in events["start_s"]])
        events = events.drop(columns=["start_s"])
        events["duration_s"] = events["duration_s"].round(1)

    html = template.render(
        video_name=video_name,
        camera_id=cfg.camera_id,
        description=cfg.description,
        generated_at=generated_at,
        duration_hms=format_hms(duration_s),
        headline=headline,
        rating_factor=cfg.time_study.rating_factor,
        allowance_percent=100.0 * cfg.time_study.allowances.total,
        analysis_fps=cfg.video.analysis_fps,
        pose_fps=cfg.video.pose_fps,
        stop_threshold=cfg.events.stop_motion_threshold,
        stop_min_seconds=cfg.events.stop_min_seconds,
        warnings=warnings,
        gantt_worker=_fig_html(gantt_worker, include_js=True),
        gantt_step=_fig_html(gantt_step, include_js=False),
        spaghetti=_fig_html(spaghetti, include_js=False),
        yamazumi=_fig_html(yama, include_js=False),
        swct_blocks=swct_blocks,
        takt_seconds=cfg.time_study.takt_seconds,
        breakdown_text=breakdown_text,
        table_swct=_table(swct, {
            "manual_s": 1, "auto_s": 1, "walk_s": 1, "start_s": 1,
            "element_total_s": 1, "cumulative_s": 1}),
        table_swct_totals=_table(swct_totals),
        table_steps=_table(step_stats, {
            "mean_s": 1, "median_s": 1, "min_s": 1, "max_s": 1, "std_s": 1,
            "cv_percent": 0, "total_s": 1, "normal_s": 1, "standard_s": 1}),
        table_workers=_table(worker_stats, {
            "on_camera_s": 0, "work_s": 0, "walking_s": 0, "off_station_s": 0,
            "absent_s": 0, "utilisation_percent": 1, "walking_percent": 1,
            "stop_s": 0, "stop_percent": 1}),
        table_cycles=_table(cycle_stats, {
            "mean_cycle_s": 1, "median_cycle_s": 1, "min_cycle_s": 1,
            "max_cycle_s": 1, "std_cycle_s": 1}),
        table_events=_table(events),
    )

    out_path.write_text(html, encoding="utf-8")
    return out_path
