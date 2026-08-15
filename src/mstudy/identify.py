"""Put a worker's name to each tracked person.

You asked to name workers once per video. Naming them in the first frame breaks
the moment someone walks out of shot and comes back - the tracker gives them a
new number and the Gantt chart grows a phantom fourth worker. So naming happens
*after* tracking, against a contact sheet of every track that was found: a
worker who left and returned simply appears twice and you give both thumbnails
the same name, which merges them onto one row.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from .config import Config
from .ingest import VideoInfo, probe_video
from .metrics import format_hms

THUMB_W, THUMB_H = 150, 260
COLS = 6
PAD = 12
HEADER = 34

# Tracks this short are almost always a detector flicker, not a person.
MIN_TRACK_SECONDS = 3.0


def track_summary(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per track: when it appeared, how long it lasted, how solid it is."""
    if tracks.empty:
        return pd.DataFrame(columns=["track_id", "first_s", "last_s", "duration_s",
                                     "detections", "mean_conf"])
    g = tracks.groupby("track_id")
    out = pd.DataFrame(
        {
            "track_id": g["t"].min().index,
            "first_s": g["t"].min().to_numpy(),
            "last_s": g["t"].max().to_numpy(),
            "detections": g.size().to_numpy(),
            "mean_conf": g["conf"].mean().to_numpy(),
        }
    )
    out["duration_s"] = out["last_s"] - out["first_s"]
    return out.sort_values("first_s").reset_index(drop=True)


def _crop(frame: np.ndarray, row: pd.Series) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, int(row["x1"]) - 10)
    y1 = max(0, int(row["y1"]) - 10)
    x2 = min(w, int(row["x2"]) + 10)
    y2 = min(h, int(row["y2"]) + 10)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((THUMB_H, THUMB_W, 3), np.uint8)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)


def build_contact_sheet(
    tracks: pd.DataFrame, info: VideoInfo, summary: pd.DataFrame, rotate: int = 0
) -> np.ndarray:
    """One labelled thumbnail per track, taken at that track's clearest moment."""
    from .ingest import read_frame_at

    ids = summary["track_id"].tolist()
    if not ids:
        return np.zeros((200, 600, 3), np.uint8)

    rows = int(np.ceil(len(ids) / COLS))
    sheet = np.full(
        (rows * (THUMB_H + HEADER + PAD) + PAD, COLS * (THUMB_W + PAD) + PAD, 3),
        245, np.uint8,
    )

    for i, tid in enumerate(ids):
        sub = tracks[tracks["track_id"] == tid]
        # Pick the frame where the detector was most confident - the clearest
        # view of that person, which is what makes them recognisable.
        best = sub.loc[sub["conf"].idxmax()]
        try:
            frame = read_frame_at(info, float(best["t"]), rotate)
            thumb = _crop(frame, best)
        except Exception:
            thumb = np.zeros((THUMB_H, THUMB_W, 3), np.uint8)

        r, c = divmod(i, COLS)
        x = PAD + c * (THUMB_W + PAD)
        y = PAD + r * (THUMB_H + HEADER + PAD)

        cv2.rectangle(sheet, (x - 2, y - 2), (x + THUMB_W + 2, y + THUMB_H + HEADER),
                      (190, 190, 190), 1)
        sheet[y:y + THUMB_H, x:x + THUMB_W] = thumb

        info_row = summary[summary["track_id"] == tid].iloc[0]
        cv2.putText(sheet, f"#{tid}", (x + 4, y + THUMB_H + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(
            sheet,
            f"{format_hms(info_row['first_s'])}  {info_row['duration_s']:.0f}s",
            (x + 42, y + THUMB_H + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA,
        )
    return sheet


def names_cache_path(cfg: Config, config_dir: Path) -> Path:
    return config_dir / f"{cfg.camera_id}.names.json"


def load_names(run_dir: Path) -> dict[int, str]:
    path = run_dir / "names.json"
    if not path.exists():
        return {}
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def save_names(run_dir: Path, names: dict[int, str]) -> None:
    (run_dir / "names.json").write_text(
        json.dumps({str(k): v for k, v in names.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def apply_names(tracks: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """Attach a `worker` column, merging every track id that shares a name.

    Tracks left unnamed keep a neutral 'Worker #n' label rather than being
    dropped: an unnamed person is still occupying a station, and silently
    deleting them would understate the labour on the line.
    """
    tracks = tracks.copy()
    if tracks.empty:
        tracks["worker"] = pd.Series(dtype=object)
        return tracks
    tracks["worker"] = [
        names.get(int(t), f"Worker #{int(t)}") for t in tracks["track_id"]
    ]
    return tracks[tracks["worker"] != "__ignore__"].reset_index(drop=True)


def run_naming(
    video: Path, cfg: Config, run_dir: Path, console: Console | None = None
) -> dict[int, str]:
    console = console or Console()

    tracks_path = run_dir / "tracks.parquet"
    if not tracks_path.exists():
        console.print(f"[red]No tracks at {tracks_path}. Run `mstudy track` first.[/red]")
        return {}

    tracks = pd.read_parquet(tracks_path)
    info = probe_video(video)
    summary = track_summary(tracks)

    solid = summary[summary["duration_s"] >= MIN_TRACK_SECONDS]
    flimsy = summary[summary["duration_s"] < MIN_TRACK_SECONDS]

    console.print(f"\n[bold]{len(summary)} track(s) found[/bold] "
                  f"({len(flimsy)} shorter than {MIN_TRACK_SECONDS:.0f}s, auto-ignored)")

    sheet = build_contact_sheet(tracks, info, solid, cfg.video.rotate)
    sheet_path = run_dir / "contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    console.print(f"Contact sheet: [bold]{sheet_path}[/bold]")

    table = Table("Track", "First seen", "Duration", "Detections", "Confidence")
    for _, r in solid.iterrows():
        table.add_row(f"#{int(r['track_id'])}", format_hms(r["first_s"]),
                      f"{r['duration_s']:.0f}s", str(int(r["detections"])),
                      f"{r['mean_conf']:.2f}")
    console.print(table)

    cached_path = (cfg.source_path.parent / f"{cfg.camera_id}.names.json"
                   if cfg.source_path else run_dir / "cache.names.json")
    known: list[str] = []
    if cached_path.exists():
        try:
            known = sorted(set(json.loads(cached_path.read_text(encoding="utf-8"))))
        except Exception:
            known = []

    console.print(
        "\nOpen the contact sheet, then name each track below.\n"
        "  - Give the SAME name to two tracks if they are the same person "
        "(someone who left and came back).\n"
        "  - Press Enter to skip a track (it becomes 'Worker #n').\n"
        "  - Type [bold]x[/bold] to exclude a track entirely (a visitor, a supervisor "
        "walking through)."
    )
    if known:
        console.print(f"  Previously named on this camera: {', '.join(known)}")

    names: dict[int, str] = {}
    for _, r in solid.iterrows():
        tid = int(r["track_id"])
        entry = input(
            f"  Track #{tid} (first seen {format_hms(r['first_s'])}, "
            f"{r['duration_s']:.0f}s) name: "
        ).strip()
        if entry.lower() == "x":
            names[tid] = "__ignore__"
        elif entry:
            names[tid] = entry

    for _, r in flimsy.iterrows():
        names[int(r["track_id"])] = "__ignore__"

    save_names(run_dir, names)

    real = sorted({v for v in names.values() if v != "__ignore__"})
    if real:
        try:
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            cached_path.write_text(
                json.dumps(sorted(set(known) | set(real)), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    console.print(f"\n[green]Named {len(real)} worker(s):[/green] {', '.join(real) or '(none)'}")
    return names
