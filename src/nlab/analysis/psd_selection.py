"""Exact event-coordinate selection, independent of PSD histogram binning."""

from __future__ import annotations

import numpy as np


def points_in_polygon(x: np.ndarray, y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Even-odd containment, including edges/vertices and excluding nonfinite data."""
    x, y = np.asarray(x), np.asarray(y)
    polygon = np.asarray(vertices, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("Selection coordinates must be equal-length one-dimensional arrays")
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.isfinite(polygon).all()):
        raise ValueError("Selection needs at least three finite vertices")
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    scale = high - low
    if np.any(scale <= 0):
        raise ValueError("Selection must have nonzero width and height")
    polygon = (polygon - low) / scale
    result = np.zeros(x.shape, dtype=bool)
    candidate = np.flatnonzero(
        np.isfinite(x) & np.isfinite(y)
        & (x >= low[0]) & (x <= high[0]) & (y >= low[1]) & (y <= high[1])
    )
    px, py = (x[candidate] - low[0]) / scale[0], (y[candidate] - low[1]) / scale[1]
    inside = np.zeros(len(candidate), dtype=bool)
    boundary = inside.copy()
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        dx, dy = b - a
        if dx == 0 and dy == 0:
            continue
        cross = (px - a[0]) * dy - (py - a[1]) * dx
        boundary |= (
            (np.abs(cross) <= 1e-12)
            & (px >= min(a[0], b[0]) - 1e-12) & (px <= max(a[0], b[0]) + 1e-12)
            & (py >= min(a[1], b[1]) - 1e-12) & (py <= max(a[1], b[1]) + 1e-12)
        )
        if dy != 0:
            inside ^= ((a[1] > py) != (b[1] > py)) & (px < a[0] + (py - a[1]) * dx / dy)
    result[candidate] = inside | boundary
    return result
