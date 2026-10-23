"""Configuration loading and validation.

One config file describes one *camera position* on one production line: where the
stations are, what counts as a stop, and the time-study constants. Because zones
are pixel coordinates, a config is only valid while the camera does not move.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import yaml

ZoneKind = Literal["work", "walkway"]


class ConfigError(ValueError):
    """Raised with a message aimed at the person editing the YAML, not a developer."""


@dataclass
class Zone:
    name: str
    polygon: list[list[int]]
    kind: ZoneKind = "work"

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ConfigError(
                f"Zone '{self.name}' has only {len(self.polygon)} points. "
                "A zone needs at least 3 to form an area."
            )
        if self.kind not in ("work", "walkway"):
            raise ConfigError(
                f"Zone '{self.name}' has kind '{self.kind}'. "
                "Use 'work' for a station where value is added, or 'walkway' for "
                "aisles and transport routes."
            )


@dataclass
class VideoSettings:
    # We analyse a few frames per second, not all 25-30. Work elements last
    # seconds, so 2 fps (0.5 s resolution) is finer than any stopwatch operator
    # and an order of magnitude less compute than full frame rate.
    analysis_fps: float = 2.0

    # Pose estimation runs at its own, lower rate. Measured on the target
    # machine, pose costs ~120 ms PER WORKER per frame while detection costs
    # ~330 ms per frame regardless of headcount - so with three workers, pose
    # was the majority of the run time. Pose exists only to tell "working" from
    # "stopped", and the shortest stop we report is 5 s, so sampling it once a
    # second loses nothing and roughly halves total processing time.
    pose_fps: float = 1.0

    rotate: int = 0  # 0, 90, 180 or 270 - for cameras mounted sideways

    def __post_init__(self) -> None:
        if self.pose_fps > self.analysis_fps:
            raise ConfigError(
                f"video.pose_fps ({self.pose_fps}) cannot exceed video.analysis_fps "
                f"({self.analysis_fps}) - pose is sampled from analysed frames."
            )


@dataclass
class DetectionSettings:
    # rtmlib 'mode' presets trade accuracy for speed: performance | balanced | lightweight
    mode: str = "balanced"
    min_confidence: float = 0.40
    device: str = "cpu"
    backend: str = "onnxruntime"


@dataclass
class EventSettings:
    # Hysteresis. A worker leaning across a zone boundary must not create a
    # flurry of one-frame "steps" - this is the single setting that separates a
    # readable Gantt chart from confetti.
    min_segment_seconds: float = 1.5

    # Motion is measured in BODY HEIGHTS PER SECOND, not pixels per second, so
    # the threshold does not change when a worker stands further from the camera.
    stop_motion_threshold: float = 0.08
    stop_min_seconds: float = 5.0
    motion_smoothing_seconds: float = 1.0

    # A worker who steps out of a zone for half a second has not "left".
    left_min_seconds: float = 2.0

    # Detection drops out briefly (someone walks behind a stack of tubes).
    # Bridge gaps up to this long rather than reporting a departure.
    missing_grace_seconds: float = 1.0


@dataclass
class CycleSettings:
    # A new cycle begins each time the worker re-enters this zone. Leave null to
    # skip cycle analysis (the report still gives steps, stops and utilisation).
    boundary_zone: str | None = None
    min_cycle_seconds: float = 10.0


@dataclass
class Allowances:
    personal: float = 0.05
    fatigue: float = 0.07
    delay: float = 0.03

    @property
    def total(self) -> float:
        return self.personal + self.fatigue + self.delay


@dataclass
class TimeStudySettings:
    # THE RATING FACTOR IS A HUMAN JUDGEMENT. The system will not invent one.
    # 1.00 = the observed worker was performing at a normal pace. A qualified
    # observer sets this; the report always prints which value was used.
    rating_factor: float = 1.00
    allowances: Allowances = field(default_factory=Allowances)
    takt_seconds: float | None = None  # customer demand rate, for the Yamazumi chart


@dataclass
class Config:
    camera_id: str
    zones: list[Zone] = field(default_factory=list)
    steps: dict[str, str] = field(default_factory=dict)
    video: VideoSettings = field(default_factory=VideoSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    events: EventSettings = field(default_factory=EventSettings)
    cycles: CycleSettings = field(default_factory=CycleSettings)
    time_study: TimeStudySettings = field(default_factory=TimeStudySettings)
    description: str = ""
    source_path: Path | None = None

    # ---- lookups -------------------------------------------------------

    def zone_kind(self, zone_name: str) -> str | None:
        for z in self.zones:
            if z.name == zone_name:
                return z.kind
        return None

    def step_label(self, zone_name: str) -> str:
        """Friendly work-element name for a zone, falling back to the zone name."""
        return self.steps.get(zone_name, zone_name)

    @property
    def work_zone_names(self) -> list[str]:
        return [z.name for z in self.zones if z.kind == "work"]

    @property
    def walkway_zone_names(self) -> list[str]:
        return [z.name for z in self.zones if z.kind == "walkway"]

    # ---- reproducibility ----------------------------------------------

    def fingerprint(self) -> str:
        """Stable hash of the analysis-affecting settings.

        Written into run.json so a report from six months ago can be traced to
        the exact configuration that produced it.
        """
        payload = asdict(self)
        payload.pop("source_path", None)
        payload.pop("description", None)
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("source_path", None)
        return d


def _require(mapping: dict, key: str, ctx: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key '{key}' in {ctx}.")
    return mapping[key]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    zones = [
        Zone(
            name=_require(z, "name", "a zone entry"),
            polygon=[[int(p[0]), int(p[1])] for p in _require(z, "polygon", "a zone entry")],
            kind=z.get("kind", "work"),
        )
        for z in raw.get("zones", [])
    ]

    names = [z.name for z in zones]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ConfigError(
            f"Duplicate zone name(s): {', '.join(sorted(duplicates))}. "
            "Each station needs a unique name so its times can be reported separately."
        )

    ts_raw = raw.get("time_study", {}) or {}
    time_study = TimeStudySettings(
        rating_factor=float(ts_raw.get("rating_factor", 1.00)),
        allowances=Allowances(**(ts_raw.get("allowances", {}) or {})),
        takt_seconds=ts_raw.get("takt_seconds"),
    )

    cfg = Config(
        camera_id=_require(raw, "camera_id", "the config file"),
        description=raw.get("description", ""),
        zones=zones,
        steps=raw.get("steps", {}) or {},
        video=VideoSettings(**(raw.get("video", {}) or {})),
        detection=DetectionSettings(**(raw.get("detection", {}) or {})),
        events=EventSettings(**(raw.get("events", {}) or {})),
        cycles=CycleSettings(**(raw.get("cycles", {}) or {})),
        time_study=time_study,
        source_path=path,
    )

    if cfg.cycles.boundary_zone and cfg.cycles.boundary_zone not in names:
        raise ConfigError(
            f"cycles.boundary_zone is '{cfg.cycles.boundary_zone}' but no zone has "
            f"that name. Available zones: {', '.join(names) or '(none yet)'}"
        )

    unknown_steps = set(cfg.steps) - set(names)
    if unknown_steps:
        raise ConfigError(
            f"steps refers to zone(s) that do not exist: {', '.join(sorted(unknown_steps))}"
        )

    return cfg


def save_zones(cfg_path: str | Path, zones: list[Zone]) -> None:
    """Write zones back into an existing config, preserving all other settings.

    Used by the zone editor so drawing polygons never clobbers hand-tuned
    thresholds.
    """
    cfg_path = Path(cfg_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    raw = raw or {}
    raw["zones"] = [
        {"name": z.name, "kind": z.kind, "polygon": [[int(x), int(y)] for x, y in z.polygon]}
        for z in zones
    ]
    cfg_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
