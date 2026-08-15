"""Command line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import ConfigError, load_config

app = typer.Typer(add_completion=False, help="Motion and time study from video.")
console = Console()

PROJECT_ROOT = Path.cwd()


def _run_dir(video: Path, root: Path | None = None) -> Path:
    return (root or PROJECT_ROOT) / "runs" / video.stem


def _print_summary(res: dict) -> None:
    head = res["headline"]
    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_row("Workers", str(head["workers"]))
    t.add_row("Mean utilisation", f"{head['mean_utilisation']:.0f}%")
    t.add_row("Work elements", str(head["distinct_steps"]))
    t.add_row("Stops detected", f"{head['total_stops']} ({head['total_stop_s']:.0f}s idle)")
    t.add_row("Walking time", f"{head['total_walking_s']:.0f}s")
    if head["mean_cycle_s"]:
        t.add_row("Mean cycle time", f"{head['mean_cycle_s']:.1f}s")
    console.print(t)


@app.command()
def demo(
    minutes: float = typer.Option(12.0, help="Length of the synthetic recording."),
    out: Path = typer.Option(Path("runs/demo"), help="Where to write the report."),
) -> None:
    """Generate a report from synthetic data - no video or models needed.

    Use this to review the report format and argue about what it should show
    before anyone spends a day filming.
    """
    from .demo import FRAME_H, FRAME_W, background_image, generate
    from .study import run_study

    console.print("[bold]Generating synthetic cuplock-ledger line...[/bold]")
    tracks, cfg, duration, _truth = generate(duration_s=minutes * 60.0)
    console.print(f"  {len(tracks):,} synthetic detections, "
                  f"{tracks['worker'].nunique()} workers, {duration / 60:.0f} min")

    res = run_study(
        tracks=tracks, cfg=cfg, duration_s=duration, out_dir=out,
        video_name="DEMO - synthetic data",
        background_png=background_image(cfg), frame_size=(FRAME_W, FRAME_H),
    )
    _print_summary(res)
    console.print(f"\n[green]Report:[/green] {res['report'].resolve()}")


@app.command()
def zones(
    video: Path = typer.Argument(..., help="Video to draw zones on."),
    config: Path = typer.Option(..., "--config", "-c", help="Config file to write into."),
    at: float = typer.Option(30.0, help="Seconds into the video for the reference frame."),
) -> None:
    """Draw the station polygons on a frame. Do this once per camera position."""
    from .zones import run_zone_editor

    try:
        run_zone_editor(video, config, at_seconds=at, console=console)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command()
def track(
    video: Path = typer.Argument(..., help="Video to process."),
    config: Path = typer.Option(..., "--config", "-c"),
    quick: bool = typer.Option(False, help="Only the first 2 minutes, at 1 fps."),
    no_resume: bool = typer.Option(False, help="Ignore any partial previous run."),
) -> None:
    """Detect and track workers, writing tracks.parquet. The slow stage."""
    from .pipeline import run_tracking

    cfg = load_config(config)
    out_dir = _run_dir(video)
    run_tracking(video, cfg, out_dir, quick=quick, resume=not no_resume, console=console)
    console.print(f"\nNext: [bold]mstudy name {video} -c {config}[/bold]")


@app.command()
def name(
    video: Path = typer.Argument(..., help="Video whose tracks should be named."),
    config: Path = typer.Option(..., "--config", "-c"),
) -> None:
    """Put a worker's name to each tracked person, via a thumbnail contact sheet."""
    from .identify import run_naming

    cfg = load_config(config)
    run_naming(video, cfg, _run_dir(video), console=console)


@app.command()
def report(
    video: Path = typer.Argument(..., help="Video whose run should be re-analysed."),
    config: Path = typer.Option(..., "--config", "-c"),
) -> None:
    """Rebuild the report from an existing tracks.parquet.

    Fast - use this after changing a threshold or a zone, instead of
    re-processing the video.
    """
    import pandas as pd

    from .identify import apply_names, load_names
    from .ingest import probe_video, read_frame_at
    from .study import run_study

    cfg = load_config(config)
    out_dir = _run_dir(video)
    tracks_path = out_dir / "tracks.parquet"
    if not tracks_path.exists():
        console.print(f"[red]No tracks found at {tracks_path}. Run `mstudy track` first.[/red]")
        raise typer.Exit(1)

    tracks = pd.read_parquet(tracks_path)
    tracks = apply_names(tracks, load_names(out_dir))

    info = probe_video(video)
    frame = read_frame_at(info, min(30.0, info.duration_s / 2), cfg.video.rotate)
    import cv2
    ok, buf = cv2.imencode(".png", frame)

    res = run_study(
        tracks=tracks, cfg=cfg, duration_s=info.duration_s, out_dir=out_dir,
        video_name=video.name,
        background_png=buf.tobytes() if ok else None,
        frame_size=(frame.shape[1], frame.shape[0]),
    )
    _print_summary(res)
    console.print(f"\n[green]Report:[/green] {res['report'].resolve()}")


@app.command()
def annotate(
    video: Path = typer.Argument(...),
    config: Path = typer.Option(..., "--config", "-c"),
    max_seconds: float = typer.Option(
        0.0, help="Only annotate the first N seconds (0 = the whole video)."
    ),
) -> None:
    """Write annotated.mp4 - the original video with names, steps and a clock."""
    from .annotate import render_annotated

    cfg = load_config(config)
    render_annotated(
        video, cfg, _run_dir(video),
        max_seconds=max_seconds or None, console=console,
    )


@app.command()
def run(
    video: Path = typer.Argument(..., help="Video to analyse."),
    config: Path = typer.Option(..., "--config", "-c"),
    quick: bool = typer.Option(False, help="Only the first 2 minutes, at 1 fps."),
    skip_naming: bool = typer.Option(False, help="Use track numbers instead of names."),
    with_video: bool = typer.Option(False, help="Also render the annotated video."),
) -> None:
    """Track, name, analyse and report - the whole thing in one command."""
    track(video=video, config=config, quick=quick, no_resume=False)
    if not skip_naming:
        name(video=video, config=config)
    report(video=video, config=config)
    if with_video:
        annotate(video=video, config=config, max_seconds=0.0)


@app.command()
def watch(
    config: Path = typer.Option(..., "--config", "-c"),
    folder: Path = typer.Option(Path("input"), help="Folder to monitor."),
    interval: float = typer.Option(60.0, help="Seconds between checks."),
) -> None:
    """Watch a folder and analyse any new video that appears."""
    import time

    from .ingest import find_videos

    console.print(f"Watching [bold]{folder}[/bold] every {interval:.0f}s. Ctrl-C to stop.")
    seen = {p.name for p in find_videos(folder) if _run_dir(p).joinpath("report.html").exists()}
    try:
        while True:
            for video in find_videos(folder):
                if video.name in seen:
                    continue
                console.print(f"\n[bold]New video:[/bold] {video.name}")
                try:
                    track(video=video, config=config, quick=False, no_resume=False)
                    report(video=video, config=config)
                except Exception as exc:  # keep watching despite one bad file
                    console.print(f"[red]Failed on {video.name}: {exc}[/red]")
                seen.add(video.name)
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\nStopped.")


if __name__ == "__main__":
    app()
