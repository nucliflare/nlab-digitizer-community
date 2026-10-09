"""Small, fit-free estimators for MCA gain-drift diagnostics.

The live diagnostic intentionally records the literal maximum requested by the
operator as well as a lightly smoothed, parabolic sub-bin estimate.  This is
not a photopeak fit: retaining both values and the complete differential ROI
makes jumps caused by low statistics or a competing peak visible afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class McaPeakEstimate:
    """Peak-location and quality measurements from one differential spectrum."""

    maximum_channel: int | None
    smoothed_channel: float
    local_centroid_channel: float
    peak_counts: float
    background_counts_per_bin: float
    prominence_counts: float
    signal_to_noise: float
    roi_counts: int
    valid: bool


def differential_counts(
    current: NDArray[np.integer],
    previous: NDArray[np.integer],
) -> tuple[NDArray[np.uint64], int]:
    """Return a non-negative interval spectrum and its negative-bin count.

    Histogram attributes are read in sequential chunks and are not an atomic
    snapshot while acquisition is running.  A reset, transport inconsistency,
    or concurrent clear can therefore make individual signed differences
    negative.  The count is retained as a quality flag while negative values
    are clipped instead of wrapping unsigned integers.
    """

    current_array = np.asarray(current)
    previous_array = np.asarray(previous)
    if current_array.ndim != 1 or previous_array.ndim != 1:
        raise ValueError("MCA histograms must be one-dimensional")
    if current_array.shape != previous_array.shape:
        raise ValueError("current and previous MCA histograms must have equal shape")

    delta = current_array.astype(np.int64) - previous_array.astype(np.int64)
    negative_bins = int(np.count_nonzero(delta < 0))
    return np.maximum(delta, 0).astype(np.uint64), negative_bins


def estimate_peak_maximum(
    differential: NDArray[np.integer],
    roi_min: int,
    roi_max: int,
    *,
    smoothing_sigma_bins: float = 2.0,
    centroid_half_width_bins: int = 40,
    minimum_roi_counts: int = 20,
) -> McaPeakEstimate:
    """Estimate the strongest peak in the inclusive channel ROI.

    ``maximum_channel`` is the literal maximum-count bin.  ``smoothed_channel``
    is the maximum after Gaussian smoothing followed by a three-point
    parabolic interpolation.  The local centroid is background-subtracted and
    limited around that maximum so distant structures do not pull it away.
    """

    counts = np.asarray(differential)
    if counts.ndim != 1:
        raise ValueError("differential spectrum must be one-dimensional")
    if roi_min < 0 or roi_max < roi_min or roi_max >= counts.size:
        raise ValueError(
            f"ROI [{roi_min}, {roi_max}] is outside histogram with {counts.size} bins"
        )
    if smoothing_sigma_bins < 0:
        raise ValueError("smoothing_sigma_bins must be non-negative")
    if centroid_half_width_bins < 1:
        raise ValueError("centroid_half_width_bins must be positive")

    roi = counts[roi_min : roi_max + 1].astype(np.float64)
    roi_counts = int(np.sum(roi, dtype=np.float64))
    if roi.size == 0 or roi_counts == 0:
        return McaPeakEstimate(
            maximum_channel=None,
            smoothed_channel=float("nan"),
            local_centroid_channel=float("nan"),
            peak_counts=0.0,
            background_counts_per_bin=0.0,
            prominence_counts=0.0,
            signal_to_noise=0.0,
            roi_counts=roi_counts,
            valid=False,
        )

    raw_index = int(np.argmax(roi))
    smoothed = (
        gaussian_filter1d(roi, smoothing_sigma_bins, mode="nearest")
        if smoothing_sigma_bins > 0
        else roi.copy()
    )
    smooth_index = int(np.argmax(smoothed))
    sub_bin = 0.0
    if 0 < smooth_index < smoothed.size - 1:
        left = float(smoothed[smooth_index - 1])
        centre = float(smoothed[smooth_index])
        right = float(smoothed[smooth_index + 1])
        denominator = left - 2.0 * centre + right
        if denominator < 0:
            sub_bin = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))

    edge_width = max(1, min(20, roi.size // 10))
    edge_samples = np.concatenate((roi[:edge_width], roi[-edge_width:]))
    background = float(np.median(edge_samples))
    peak_counts = float(roi[raw_index])
    smoothed_peak = float(smoothed[smooth_index])
    prominence = max(0.0, smoothed_peak - background)
    noise = np.sqrt(max(smoothed_peak + background, 1.0))
    signal_to_noise = prominence / noise

    local_start = max(0, smooth_index - centroid_half_width_bins)
    local_stop = min(roi.size, smooth_index + centroid_half_width_bins + 1)
    local_counts = np.maximum(roi[local_start:local_stop] - background, 0.0)
    local_total = float(np.sum(local_counts))
    if local_total > 0:
        local_channels = np.arange(local_start, local_stop, dtype=np.float64) + roi_min
        centroid = float(np.dot(local_channels, local_counts) / local_total)
    else:
        centroid = float("nan")

    valid = bool(roi_counts >= minimum_roi_counts and signal_to_noise >= 3.0)
    return McaPeakEstimate(
        maximum_channel=roi_min + raw_index,
        smoothed_channel=roi_min + smooth_index + sub_bin,
        local_centroid_channel=centroid,
        peak_counts=peak_counts,
        background_counts_per_bin=background,
        prominence_counts=prominence,
        signal_to_noise=signal_to_noise,
        roi_counts=roi_counts,
        valid=valid,
    )


__all__ = [
    "McaPeakEstimate",
    "differential_counts",
    "estimate_peak_maximum",
]
