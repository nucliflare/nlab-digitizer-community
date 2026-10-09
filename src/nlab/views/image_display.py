"""Shared presentation helpers for count-density images."""

from __future__ import annotations

import numpy as np


def transparent_sparse_bins(
    displayed: np.ndarray,
    counts: np.ndarray,
    *,
    maximum_hidden_count: float = 1.0,
) -> np.ndarray:
    """Return a float image with low-occupancy bins marked transparent.

    Pyqtgraph renders NaN image pixels with zero alpha. Signed corrected
    coincidence values use their magnitude for the occupancy threshold.
    """
    image = np.asarray(displayed, dtype=np.float64).copy()
    bin_counts = np.asarray(counts)
    if image.shape != bin_counts.shape:
        raise ValueError("display image and count matrix must have matching shapes")
    image[np.abs(bin_counts) <= maximum_hidden_count] = np.nan
    return image


__all__ = ["transparent_sparse_bins"]
