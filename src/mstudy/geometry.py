"""Zone geometry: where is a worker standing?

Deliberately pure NumPy (no OpenCV) so the step-assignment logic can be unit
tested without a video, a display, or model weights.
"""

from __future__ import annotations

import numpy as np

from . import LABEL_OUTSIDE
from .config import Zone


def foot_points(boxes: np.ndarray) -> np.ndarray:
    """Bottom-centre of each bounding box, i.e. roughly where the feet are.

    Zones are drawn on the floor, so the worker's *ground contact point* decides
    which station they are at. Using the box centre instead would place a tall
    worker in the wrong zone whenever they lean.

    boxes: (N, 4) array of x1, y1, x2, y2  ->  returns (N, 2)
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]], axis=1)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting. Returns a boolean mask over points."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)

    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        # Does a horizontal ray from the point cross this edge?
        straddles = (yi > y) != (yj > y)
        denom = yj - yi
        # Where the edge is horizontal, `straddles` is already False, so the
        # value computed here is irrelevant - guard only to avoid a divide error.
        safe_denom = np.where(denom == 0.0, 1.0, denom)
        x_cross = (xj - xi) * (y - yi) / safe_denom + xi
        inside ^= straddles & (x < x_cross)
        j = i

    return inside


def assign_zones(points: np.ndarray, zones: list[Zone]) -> np.ndarray:
    """Label each point with the zone containing it.

    Zones are tested in config order, so **earlier zones win** where polygons
    overlap. That gives you a deliberate priority: list the specific station
    before the broad aisle it sits inside.

    Returns an array of zone-name strings, LABEL_OUTSIDE where no zone matches.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    labels = np.full(len(points), LABEL_OUTSIDE, dtype=object)
    if len(points) == 0:
        return labels

    unassigned = np.ones(len(points), dtype=bool)
    for zone in zones:
        if not unassigned.any():
            break
        hit = points_in_polygon(points, np.array(zone.polygon)) & unassigned
        labels[hit] = zone.name
        unassigned &= ~hit

    return labels


def polygon_area(polygon: np.ndarray) -> float:
    """Shoelace area in square pixels. Used to sanity-check tiny mis-drawn zones."""
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
