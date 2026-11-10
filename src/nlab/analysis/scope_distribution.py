"""Bounded histograms of received Scope samples, peaks and pulse areas."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class HistogramMode(StrEnum):
    SAMPLES = "samples"
    AMPLITUDE = "amplitude"
    INTEGRAL = "integral"


def pulse_integral(
    samples: NDArray[np.generic], *, sample_period_ns: float,
    pretrigger_ns: float, negative: bool,
) -> float:
    """Return a positive, polarity-selected post-trigger area in ADC codes*ns.

    The baseline is the arithmetic mean of complete pretrigger sample bins.
    A bin crossing the trigger is omitted from both baseline and integration.
    Sample-hold integration sums only deviations on the selected side of the
    baseline. Thus opposite-polarity excursions do not cancel pulse area, and
    rectified noise contributes a positive area. No noise correction is implied.
    """
    if not np.isfinite([sample_period_ns, pretrigger_ns]).all() or sample_period_ns <= 0:
        raise ValueError("Integral needs finite sample timing and a positive sample period")
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Integral needs a finite waveform")
    ratio = pretrigger_ns / sample_period_ns
    if not np.isfinite(ratio):
        raise ValueError("Integral needs a finite pretrigger/sample-period ratio")
    baseline_count = int(np.floor(ratio))
    signal_start = int(np.ceil(ratio))
    if baseline_count < 1 or signal_start >= values.size:
        raise ValueError("Integral needs complete pretrigger baseline and post-trigger samples")
    baseline = float(values[:baseline_count].mean())
    deviations = values[signal_start:] - baseline
    if negative:
        deviations = -deviations
    with np.errstate(over="ignore", invalid="ignore"):
        area = float(np.maximum(deviations, 0.0).sum() * sample_period_ns)
    if not np.isfinite(area):
        raise ValueError("Integral is outside the finite numeric range")
    return area


class AccumulatedAmplitudeHistogram:
    """Histogram received values with fixed memory and explicit mode semantics.

    Sample/peak modes store signed 16-bit codes at one-code resolution, and
    display groups of 100 codes. Fractional values round to the nearest code
    (ties upward). Peak mode adds one raw maximum/minimum per received frame.

    Integral mode adds one positive area per valid frame. Its bins start at
    100 ADC codes times the sample period. Adjacent bins merge in pairs if a
    larger area needs more range, preserving counts within the fixed array.
    Readout/polarity/window changes reset derived distributions to keep their
    measurement definitions separate. No waveforms or event lists are retained.
    """

    ADC_BIN_WIDTH = 100

    def __init__(self) -> None:
        self.counts: NDArray[np.uint64] = np.zeros(65536, dtype=np.uint64)
        self.mode = HistogramMode.SAMPLES
        # Historical name: histogram entries (individual samples, peaks or integrals).
        self.sample_count = 0
        self.frame_count = 0
        self.rejected_count = 0
        self.rejected_frames = 0
        self.last_rejection = ""
        self.source: str | None = None
        self.integral_bin_width = float(self.ADC_BIN_WIDTH)
        self._context: tuple[object, ...] | None = None

    def clear(self) -> None:
        self.counts.fill(0)
        self.sample_count = 0
        self.frame_count = 0
        self.rejected_count = 0
        self.rejected_frames = 0
        self.last_rejection = ""
        self.source = None
        self.integral_bin_width = float(self.ADC_BIN_WIDTH)
        self._context = None

    def set_mode(self, mode: HistogramMode) -> None:
        mode = HistogramMode(mode)
        if self.mode != mode:
            self.mode = mode
            self.clear()

    def add(
        self, samples: NDArray[np.generic], *, source: str,
        negative: bool = False, sample_period_ns: float = 1.0,
        pretrigger_ns: float = 0.0,
    ) -> None:
        values = np.asarray(samples).reshape(-1)
        if not values.size:
            return
        context: tuple[object, ...] = (source,)
        if self.mode != HistogramMode.SAMPLES:
            context += (negative, values.size)
        if self.mode == HistogramMode.INTEGRAL:
            context += (sample_period_ns, pretrigger_ns)
        if self._context != context:
            self.clear()
            self.source = source
            self._context = context
            if (np.isfinite(sample_period_ns) and sample_period_ns > 0
                    and np.isfinite(self.ADC_BIN_WIDTH * sample_period_ns)):
                self.integral_bin_width = self.ADC_BIN_WIDTH * sample_period_ns

        valid = np.isfinite(values) & (values >= -32768) & (values <= 32767)
        if self.mode != HistogramMode.SAMPLES and not valid.all():
            self._reject_frame("Peak/integral needs a complete finite ADC waveform")
            return
        if self.mode == HistogramMode.INTEGRAL:
            try:
                area = pulse_integral(
                    values, sample_period_ns=sample_period_ns,
                    pretrigger_ns=pretrigger_ns, negative=negative,
                )
            except ValueError as exc:
                self._reject_frame(str(exc))
                return
            self._check_capacity(1)
            while area >= self.counts.size * self.integral_bin_width:
                half = self.counts.size // 2
                merged = self.counts.reshape(-1, 2).sum(axis=1, dtype=np.uint64)
                self.counts[:half] = merged
                self.counts[half:] = 0
                self.integral_bin_width *= 2
            self.counts[int(area // self.integral_bin_width)] += np.uint64(1)
            self.sample_count += 1
        else:
            accepted = values[valid]
            if self.mode == HistogramMode.AMPLITUDE:
                accepted = np.array([accepted.min() if negative else accepted.max()])
            self._check_capacity(int(accepted.size))
            indices = np.floor(accepted.astype(np.float64) + 32768.5).astype(np.int64)
            self.counts += np.bincount(indices, minlength=65536).astype(np.uint64)
            self.sample_count += int(accepted.size)
            self.rejected_count += int(values.size - valid.sum())
        self.frame_count += 1
        self.last_rejection = ""

    def _reject_frame(self, reason: str) -> None:
        self.frame_count += 1
        self.rejected_frames += 1
        self.last_rejection = reason

    def _check_capacity(self, count: int) -> None:
        if self.sample_count + count > np.iinfo(np.uint64).max:
            raise OverflowError("Scope distribution counter is full; clear it to continue")

    def projected_bins(
        self, lower: float, upper: float
    ) -> tuple[NDArray[np.float64], NDArray[np.uint64]]:
        """Return visible 100-code bins for samples/peaks without losing counts."""
        if self.mode == HistogramMode.INTEGRAL:
            occupied = np.flatnonzero(self.counts)
            stop = int(occupied[-1]) + 1 if occupied.size else 1
            edges = np.arange(stop + 1, dtype=np.float64) * self.integral_bin_width
            return edges, self.counts[:stop].copy()
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            return np.empty(0), np.empty(0, dtype=np.uint64)
        width = self.ADC_BIN_WIDTH
        total_bins = (self.counts.size + width - 1) // width
        first = int(np.clip(np.floor((lower + 32768.5) / width), 0, total_bins))
        last = int(np.clip(np.ceil((upper + 32768.5) / width), 0, total_bins))
        if last <= first:
            return np.empty(0), np.empty(0, dtype=np.uint64)
        first_code, last_code = first * width, last * width
        codes = self.counts[first_code:min(last_code, self.counts.size)]
        if last_code > self.counts.size:
            codes = np.pad(codes, (0, last_code - self.counts.size))
        edges = np.arange(first, last + 1, dtype=np.float64) * width - 32768.5
        return edges, codes.reshape(-1, width).sum(axis=1, dtype=np.uint64)
