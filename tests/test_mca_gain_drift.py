from __future__ import annotations

import numpy as np
import pytest

from nlab.analysis.mca_gain_drift import differential_counts, estimate_peak_maximum


def test_differential_counts_clips_resets_without_unsigned_wrap() -> None:
    current = np.array([12, 3, 9], dtype=np.uint32)
    previous = np.array([10, 8, 4], dtype=np.uint32)

    delta, negative_bins = differential_counts(current, previous)

    assert delta.dtype == np.uint64
    assert delta.tolist() == [2, 0, 5]
    assert negative_bins == 1


def test_peak_estimator_reports_literal_and_sub_bin_maximum() -> None:
    channels = np.arange(2048, dtype=np.float64)
    spectrum = 3.0 + 400.0 * np.exp(-0.5 * ((channels - 1337.4) / 12.0) ** 2)
    counts = np.rint(spectrum).astype(np.uint64)

    estimate = estimate_peak_maximum(counts, 1000, 1700)

    assert estimate.maximum_channel in (1337, 1338)
    assert estimate.smoothed_channel == pytest.approx(1337.4, abs=0.15)
    assert estimate.local_centroid_channel == pytest.approx(1337.4, abs=0.15)
    assert estimate.signal_to_noise > 3
    assert estimate.valid


def test_peak_estimator_marks_empty_interval_invalid() -> None:
    estimate = estimate_peak_maximum(np.zeros(2048, dtype=np.uint64), 1000, 1700)

    assert estimate.maximum_channel is None
    assert np.isnan(estimate.smoothed_channel)
    assert estimate.roi_counts == 0
    assert not estimate.valid


@pytest.mark.parametrize("roi", [(-1, 4), (4, 3), (0, 10)])
def test_peak_estimator_rejects_invalid_roi(roi: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="ROI"):
        estimate_peak_maximum(np.zeros(10, dtype=np.uint64), *roi)
