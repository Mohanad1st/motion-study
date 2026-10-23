"""Time-study statistics.

Converts segments into the numbers an industrial engineer actually reports:
element times with their spread, labour utilisation, and standard time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

# Segment kinds that represent a worker present and adding value.
PRODUCTIVE = ("work",)


def step_statistics(segments: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per work element: how long it takes and how consistent it is.

    The standard deviation column is the one worth reading first. A step with a
    high mean is simply slow and everyone already knows it; a step with a high
    *spread* is unstable, and instability is usually where the recoverable time
    is hiding.
    """
    cols = [
        "step", "observations", "mean_s", "median_s", "min_s", "max_s",
        "std_s", "cv_percent", "total_s", "normal_s", "standard_s",
    ]
    work = segments[segments["kind"].isin(PRODUCTIVE)]
    if work.empty:
        return pd.DataFrame(columns=cols)

    g = work.groupby("step")["duration_s"]
    stats = pd.DataFrame(
        {
            "step": g.mean().index,
            "observations": g.count().to_numpy(),
            "mean_s": g.mean().to_numpy(),
            "median_s": g.median().to_numpy(),
            "min_s": g.min().to_numpy(),
            "max_s": g.max().to_numpy(),
            # ddof=0 keeps a single-observation step at 0 rather than NaN.
            "std_s": g.std(ddof=0).fillna(0.0).to_numpy(),
            "total_s": g.sum().to_numpy(),
        }
    )

    # Coefficient of variation: spread as a percentage of the mean, so steps of
    # very different lengths can be compared for consistency on one scale.
    stats["cv_percent"] = np.where(
        stats["mean_s"] > 0, 100.0 * stats["std_s"] / stats["mean_s"], 0.0
    )

    rating = cfg.time_study.rating_factor
    allowance = cfg.time_study.allowances.total
    stats["normal_s"] = stats["mean_s"] * rating
    stats["standard_s"] = stats["normal_s"] * (1.0 + allowance)

    return stats.sort_values("total_s", ascending=False).reset_index(drop=True)[cols]


def worker_statistics(
    segments: pd.DataFrame, stops: pd.DataFrame, cfg: Config, duration_s: float
) -> pd.DataFrame:
    """Per worker: where the time went.

    `utilisation_percent` is time at a work station as a share of the time the
    worker was on camera at all - not of the whole recording. Dividing by the
    recording length would punish a worker for a shift that started late and
    would not be defensible in a review.
    """
    cols = [
        "worker", "on_camera_s", "work_s", "walking_s", "off_station_s",
        "absent_s", "utilisation_percent", "walking_percent",
        "stop_count", "stop_s", "stop_percent",
    ]
    if segments.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for worker, grp in segments.groupby("worker", sort=True):
        by_kind = grp.groupby("kind")["duration_s"].sum()

        def kind(name: str) -> float:
            return float(by_kind.get(name, 0.0))

        work = kind("work")
        walking = kind("walking")
        off = kind("off_station") + kind("brief_excursion")
        absent = kind("absent")
        on_camera = work + walking + off

        w_stops = stops[stops["worker"] == worker] if not stops.empty else stops
        stop_s = float(w_stops["duration_s"].sum()) if len(w_stops) else 0.0

        rows.append(
            {
                "worker": worker,
                "on_camera_s": on_camera,
                "work_s": work,
                "walking_s": walking,
                "off_station_s": off,
                "absent_s": absent,
                "utilisation_percent": 100.0 * work / on_camera if on_camera > 0 else 0.0,
                "walking_percent": 100.0 * walking / on_camera if on_camera > 0 else 0.0,
                "stop_count": int(len(w_stops)),
                "stop_s": stop_s,
                "stop_percent": 100.0 * stop_s / work if work > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def cycle_statistics(cycles: pd.DataFrame) -> pd.DataFrame:
    cols = ["worker", "cycles", "mean_cycle_s", "median_cycle_s", "min_cycle_s",
            "max_cycle_s", "std_cycle_s"]
    if cycles.empty:
        return pd.DataFrame(columns=cols)

    g = cycles.groupby("worker")["duration_s"]
    return pd.DataFrame(
        {
            "worker": g.mean().index,
            "cycles": g.count().to_numpy(),
            "mean_cycle_s": g.mean().to_numpy(),
            "median_cycle_s": g.median().to_numpy(),
            "min_cycle_s": g.min().to_numpy(),
            "max_cycle_s": g.max().to_numpy(),
            "std_cycle_s": g.std(ddof=0).fillna(0.0).to_numpy(),
        }
    )[cols].reset_index(drop=True)


def yamazumi_table(segments: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    """Work content per operator per cycle - the line-balancing view.

    Dividing by the number of cycles turns a raw recording into a comparable
    per-piece figure, which is what tells you whether labour is spread evenly
    across the operators or piled onto one of them.
    """
    cols = ["worker", "step", "seconds_per_cycle"]
    work = segments[segments["kind"].isin(PRODUCTIVE)]
    if work.empty:
        return pd.DataFrame(columns=cols)

    counts = (
        cycles.groupby("worker")["cycle_no"].count().to_dict() if not cycles.empty else {}
    )
    rows = []
    for (worker, step), total in work.groupby(["worker", "step"])["duration_s"].sum().items():
        n = max(1, counts.get(worker, 1))
        rows.append({"worker": worker, "step": step, "seconds_per_cycle": total / n})

    return pd.DataFrame(rows, columns=cols)


def headline(
    workers: pd.DataFrame, steps: pd.DataFrame, cycle_stats: pd.DataFrame
) -> dict[str, float | int | str]:
    """A handful of numbers for the top of the report."""
    return {
        "workers": int(len(workers)),
        "distinct_steps": int(len(steps)),
        "mean_utilisation": float(workers["utilisation_percent"].mean()) if len(workers) else 0.0,
        "total_stops": int(workers["stop_count"].sum()) if len(workers) else 0,
        "total_stop_s": float(workers["stop_s"].sum()) if len(workers) else 0.0,
        "total_walking_s": float(workers["walking_s"].sum()) if len(workers) else 0.0,
        "total_absent_s": float(workers["absent_s"].sum()) if len(workers) else 0.0,
        "mean_cycle_s": float(cycle_stats["mean_cycle_s"].mean()) if len(cycle_stats) else 0.0,
    }


def format_hms(seconds: float) -> str:
    seconds = int(round(max(0.0, seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
