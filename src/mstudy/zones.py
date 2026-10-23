"""Click-to-draw station zones on a reference frame.

Run once per camera position. The polygons are pixel coordinates, so they stay
valid only while the camera does not move - which is why the runbook insists on
a fixed tripod.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from rich.console import Console

from .config import Zone, load_config, save_zones
from .geometry import polygon_area
from .ingest import probe_video, read_frame_at

WINDOW = "mstudy - draw zones"
MAX_DISPLAY_W, MAX_DISPLAY_H = 1500, 820

WORK_COLOUR = (210, 160, 60)      # BGR
WALKWAY_COLOUR = (140, 140, 140)
ACTIVE_COLOUR = (60, 200, 255)

HELP = """
  Left click   add a point
  Z            undo last point
  ENTER        close this zone and name it
  D            delete the last saved zone
  S            save to the config file and quit
  Q / ESC      quit without saving
"""


def _fit(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    return min(1.0, MAX_DISPLAY_W / w, MAX_DISPLAY_H / h)


def _draw(base: np.ndarray, zones: list[Zone], pending: list[tuple[int, int]],
          scale: float) -> np.ndarray:
    canvas = base.copy()
    overlay = canvas.copy()

    for zone in zones:
        poly = (np.array(zone.polygon, np.float32) * scale).astype(np.int32)
        colour = WORK_COLOUR if zone.kind == "work" else WALKWAY_COLOUR
        cv2.fillPoly(overlay, [poly], colour)
        cv2.polylines(canvas, [poly], True, colour, 2, cv2.LINE_AA)
        x, y = poly[:, 0].min() + 8, poly[:, 1].min() + 24
        label = f"{zone.name} [{zone.kind}]"
        cv2.putText(canvas, label, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.32, canvas, 0.68, 0, canvas)

    if pending:
        pts = np.array(pending, np.int32)
        if len(pending) > 1:
            cv2.polylines(canvas, [pts], False, ACTIVE_COLOUR, 2, cv2.LINE_AA)
        for p in pending:
            cv2.circle(canvas, p, 4, ACTIVE_COLOUR, -1, cv2.LINE_AA)

    banner = "ENTER close zone | Z undo | D delete last | S save+quit | Q cancel"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (24, 24, 24), -1)
    cv2.putText(canvas, banner, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def run_zone_editor(
    video: Path, config_path: Path, at_seconds: float = 30.0,
    console: Console | None = None,
) -> list[Zone]:
    console = console or Console()

    info = probe_video(video)
    frame = read_frame_at(info, min(at_seconds, max(info.duration_s - 1, 0)), rotate=0)

    # Honour an existing rotate setting so zones match what the pipeline sees.
    if config_path.exists():
        try:
            rotate = load_config(config_path).video.rotate
            if rotate:
                frame = read_frame_at(info, min(at_seconds, info.duration_s - 1), rotate)
        except Exception:
            pass

    scale = _fit(frame)
    display = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()

    zones: list[Zone] = []
    if config_path.exists():
        try:
            zones = list(load_config(config_path).zones)
            if zones:
                console.print(f"Loaded {len(zones)} existing zone(s) - add to or delete them.")
        except Exception as exc:
            console.print(f"[yellow]Could not read existing zones: {exc}[/yellow]")

    pending: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((x, y))

    console.print(f"[bold]Reference frame:[/bold] {video.name} at {at_seconds:.0f}s "
                  f"({info.width}x{info.height})")
    console.print(HELP)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    saved = False

    try:
        while True:
            cv2.imshow(WINDOW, _draw(display, zones, pending, scale))
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):  # ENTER
                if len(pending) < 3:
                    console.print("[yellow]A zone needs at least 3 points.[/yellow]")
                    continue
                # Back to original-frame coordinates before storing.
                poly = [[int(round(x / scale)), int(round(y / scale))] for x, y in pending]
                if polygon_area(np.array(poly)) < 400:
                    console.print("[yellow]That zone is tiny - draw it larger.[/yellow]")
                    continue

                console.print(f"[bold]Zone {len(zones) + 1}[/bold] ({len(poly)} points)")
                name = input("  Station name: ").strip()
                if not name:
                    console.print("  [yellow]No name given - discarded.[/yellow]")
                    pending.clear()
                    continue
                kind_in = input("  Kind - [w]ork station or [a]isle/walkway? [w]: ").strip().lower()
                kind = "walkway" if kind_in.startswith("a") else "work"
                zones.append(Zone(name=name, polygon=poly, kind=kind))
                console.print(f"  [green]Added '{name}' as {kind}.[/green]")
                pending.clear()

            elif key in (ord("z"), ord("Z")):
                if pending:
                    pending.pop()

            elif key in (ord("d"), ord("D")):
                if zones:
                    removed = zones.pop()
                    console.print(f"[yellow]Deleted '{removed.name}'.[/yellow]")

            elif key in (ord("s"), ord("S")):
                saved = True
                break

            elif key in (ord("q"), ord("Q"), 27):
                break

            # Window closed with the X button.
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if not saved:
        console.print("[yellow]Quit without saving.[/yellow]")
        return zones

    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(
                {
                    "camera_id": config_path.stem,
                    "description": f"Zones drawn from {video.name}",
                    "video": {"analysis_fps": 2.0, "pose_fps": 1.0, "rotate": 0},
                    "time_study": {"rating_factor": 1.00},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    save_zones(config_path, zones)
    console.print(f"[green]Saved {len(zones)} zone(s) to {config_path}[/green]")
    for z in zones:
        console.print(f"  - {z.name} ({z.kind})")
    console.print(
        "\n[bold]Next:[/bold] add friendly work-element names under `steps:` in the "
        "config, and set `cycles.boundary_zone` to whichever station starts a new cycle."
    )
    return zones
