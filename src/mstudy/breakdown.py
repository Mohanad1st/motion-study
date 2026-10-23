"""Plain-text breakdown of the labour steps.

A readable, printable listing: every step each worker performed, in order, with
its start time and duration. The charts are for spotting patterns; this is for
checking the system against what you saw on the shop floor, line by line.
"""

from __future__ import annotations

import pandas as pd

from .config import Config
from .metrics import format_hms

WIDTH = 96

# How consistent an element is, from its coefficient of variation (spread as a
# percentage of the mean). These bands are the ones IE practice tends to use.
def consistency_label(cv_percent: float) -> str:
    if cv_percent < 10:
        return "consistent"
    if cv_percent < 25:
        return "variable"
    return "UNSTABLE"


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _stopped_within(stops: pd.DataFrame, worker: str, start: float, end: float) -> float:
    if stops.empty:
        return 0.0
    rows = stops[
        (stops["worker"] == worker) & (stops["start_s"] < end) & (stops["end_s"] > start)
    ]
    if rows.empty:
        return 0.0
    overlap = rows[["end_s"]].clip(upper=end).to_numpy().ravel() - rows[["start_s"]].clip(
        lower=start
    ).to_numpy().ravel()
    return float(overlap.clip(min=0).sum())


def _arrow(kind: str) -> str:
    return {
        "work": " ",
        "walking": ">",
        "off_station": "~",
        "brief_excursion": "~",
        "absent": "X",
    }.get(kind, " ")


def _describe(kind: str, step: str) -> str:
    if kind == "work":
        return step
    if kind == "walking":
        return "walking / transport"
    if kind in ("off_station", "brief_excursion"):
        return "off station (in view)"
    if kind == "absent":
        return "LEFT - out of view"
    return step


def build_text(
    segments: pd.DataFrame,
    stops: pd.DataFrame,
    cycles: pd.DataFrame,
    step_stats: pd.DataFrame,
    worker_stats: pd.DataFrame,
    cfg: Config,
    video_name: str,
    duration_s: float,
) -> str:
    out: list[str] = []
    add = out.append

    add(_rule("="))
    add(f" LABOUR STEP BREAKDOWN - {video_name}")
    add(f" Camera {cfg.camera_id} | recording {format_hms(duration_s)} "
        f"| analysed at {cfg.video.analysis_fps} fps")
    add(_rule("="))

    if segments.empty:
        add("")
        add(" No workers were detected.")
        return "\n".join(out)

    # ---------------- per worker, step by step ----------------
    for worker in sorted(segments["worker"].unique()):
        rows = segments[segments["worker"] == worker].sort_values("start_s")
        wstat = worker_stats[worker_stats["worker"] == worker]
        wcycles = cycles[cycles["worker"] == worker] if not cycles.empty else cycles

        add("")
        add(_rule())
        header = f" {worker.upper()}"
        if not wstat.empty:
            r = wstat.iloc[0]
            header += (f"  |  {r['utilisation_percent']:.0f}% utilisation"
                       f"  |  {int(r['stop_count'])} stop(s)")
        if not wcycles.empty:
            header += f"  |  {len(wcycles)} cycles, mean {wcycles['duration_s'].mean():.1f}s"
        add(header)
        add(_rule())
        add("")
        add(f"   {'#':>3}  {'Start':>7}  {'Step':<34}{'Duration':>10}   Notes")
        add(f"   {'-' * 3}  {'-' * 7}  {'-' * 34}{'-' * 10}   {'-' * 22}")

        current_cycle = None
        for n, (_, seg) in enumerate(rows.iterrows(), start=1):
            cyc = seg.get("cycle_no")
            if pd.notna(cyc) and cyc != current_cycle:
                current_cycle = cyc
                match = wcycles[wcycles["cycle_no"] == cyc]
                length = f"{match.iloc[0]['duration_s']:.1f}s" if not match.empty else "?"
                add("")
                add(f"   --- CYCLE {int(cyc)} "
                    f"(starts {format_hms(seg['start_s'])}, lasts {length}) ---")

            note = ""
            if seg["kind"] == "work":
                idle = _stopped_within(stops, worker, seg["start_s"], seg["end_s"])
                if idle > 0:
                    note = f"includes {idle:.0f}s STOPPED"
            elif seg["kind"] == "absent":
                note = "departure"

            add(
                f"   {n:>3}{_arrow(seg['kind'])} {format_hms(seg['start_s']):>7}  "
                f"{_describe(seg['kind'], seg['step'])[:34]:<34}"
                f"{seg['duration_s']:>8.1f} s   {note}"
            )

        # ---------------- this worker's element summary ----------------
        work = rows[rows["kind"] == "work"]
        if not work.empty:
            g = work.groupby("step")["duration_s"]
            add("")
            add(f"   ELEMENT SUMMARY - {worker}")
            add(f"   {'Step':<30}{'Obs':>5}{'Mean':>9}{'Min':>9}{'Max':>9}"
                f"{'SD':>8}   Consistency")
            for step in g.mean().sort_values(ascending=False).index:
                mean = g.mean()[step]
                sd = g.std(ddof=0).fillna(0.0)[step]
                cv = 100 * sd / mean if mean else 0.0
                add(
                    f"   {step[:30]:<30}{int(g.count()[step]):>5}"
                    f"{mean:>8.1f}s{g.min()[step]:>8.1f}s{g.max()[step]:>8.1f}s"
                    f"{sd:>7.1f}s   {consistency_label(cv)} ({cv:.0f}%)"
                )

    # ---------------- across all workers ----------------
    add("")
    add(_rule("="))
    add(" ALL WORKERS - WORK ELEMENT TIMES")
    add(_rule("="))
    add("")
    if step_stats.empty:
        add("   No work elements measured - check the zone polygons.")
    else:
        add(f"   {'Step':<30}{'Obs':>5}{'Mean':>9}{'Median':>9}{'SD':>8}"
            f"{'Normal':>9}{'Std':>9}   Consistency")
        for _, r in step_stats.iterrows():
            add(
                f"   {str(r['step'])[:30]:<30}{int(r['observations']):>5}"
                f"{r['mean_s']:>8.1f}s{r['median_s']:>8.1f}s{r['std_s']:>7.1f}s"
                f"{r['normal_s']:>8.1f}s{r['standard_s']:>8.1f}s   "
                f"{consistency_label(r['cv_percent'])} ({r['cv_percent']:.0f}%)"
            )
        add("")
        add(f"   Normal   = mean x rating factor {cfg.time_study.rating_factor}")
        add(f"   Standard = normal x (1 + {100 * cfg.time_study.allowances.total:.0f}% allowances)")
        add("   The rating factor is a HUMAN judgement. This system does not estimate")
        add("   worker pace - no standard time above is valid without a qualified observer.")

    # ---------------- stops ----------------
    add("")
    add(_rule("="))
    add(" STOPS AND DEPARTURES")
    add(_rule("="))
    add("")
    events = []
    for _, r in stops.iterrows():
        events.append((r["start_s"], r["worker"], "STOPPED",
                       f"{r['duration_s']:.1f}s during {r['step']}"))
    for _, r in segments[segments["kind"] == "absent"].iterrows():
        events.append((r["start_s"], r["worker"], "LEFT", f"{r['duration_s']:.1f}s out of view"))

    if not events:
        add("   None detected.")
    else:
        add(f"   {'At':>7}  {'Worker':<16}{'Event':<10}Detail")
        for t, worker, kind, detail in sorted(events):
            add(f"   {format_hms(t):>7}  {worker[:16]:<16}{kind:<10}{detail}")

    add("")
    # Trailing spaces on the many blank Notes cells would otherwise show up as
    # ragged whitespace in the HTML <pre> block and in any diff of this file.
    return "\n".join(line.rstrip() for line in out)
