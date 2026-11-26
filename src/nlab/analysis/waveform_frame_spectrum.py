"""Noise and mean-pulse spectra of recorded waveform frames (offline, no Qt).

Frames come from NLab NDMA recordings (raw 2 ns samples) or CAEN files, whose
sample period the operator enters. Each frame contributes one segment to a
Welch average of the one-sided PSD in codes^2/Hz, with the normalisation of
``waveform_spectrum.frame_psd`` (windowed, mean removed, linear power).

Two regions answer different questions:

- ``PRETRIGGER``: the baseline region before the trigger. In a triggered
  recording this is the only part without the pulse, so it estimates the
  baseline noise. It is short, so the resolution is coarse: delta_f = fs/n
  for n baseline samples (64 samples at 2 ns give 7.8 MHz bins).
- ``FRAME``: the whole frame. It contains the pulses, so it is the spectrum of
  pulse plus noise, not of noise.

The optional mean-pulse spectrum is a different quantity: ``|X(f)|^2`` of the
rectangular-windowed mean of baseline-subtracted, polarity-normalised frames
(the pulse shape, its energy spectral density), reported in dB relative to
its own maximum. Comparing where the pulse has its energy with where the
noise lies supports shaping and filter choices; the two curves are not in
the same units and must not be read as a signal-to-noise ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from nlab.analysis.waveform_spectrum import (
    MIN_SAMPLES,
    Detrend,
    FramePsd,
    SpectrumWindow,
    format_frequency,
    one_sided_factor,
    window_values,
)

_BATCH_VALUES = 1 << 22  # bounds the float copy of one batch


class SpectrumRegion(StrEnum):
    PRETRIGGER = "pretrigger"
    FRAME = "frame"


@dataclass(frozen=True)
class FrameSpectrumRequest:
    """What to read from every frame and how to transform it."""

    sample_period_ns: float
    frame_samples: int
    region: SpectrumRegion
    baseline_start: int
    baseline_end: int
    polarity: Literal[-1, 1]
    baseline_method: str = "median"
    window: SpectrumWindow = SpectrumWindow.HANN
    mean_pulse: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_period_ns) or self.sample_period_ns <= 0:
            raise ValueError("Spectrum needs a positive, finite sample period")
        if not 0 <= self.baseline_start < self.baseline_end <= self.frame_samples:
            raise ValueError("The baseline region must lie inside the frame")
        if self.polarity not in (-1, 1) or self.baseline_method not in ("median", "mean"):
            raise ValueError("Invalid polarity or baseline method")
        start, stop = self.segment
        if stop - start < MIN_SAMPLES:
            raise ValueError(
                f"The {self.region.value} segment has {stop - start} samples; "
                f"a spectrum needs at least {MIN_SAMPLES}"
            )

    @property
    def segment(self) -> tuple[int, int]:
        """First and one-past-last sample of the transformed segment."""
        if self.region == SpectrumRegion.PRETRIGGER:
            return self.baseline_start, self.baseline_end
        return 0, self.frame_samples

    @property
    def batch_rows(self) -> int:
        return max(1, _BATCH_VALUES // self.frame_samples)


@dataclass(frozen=True)
class FrameSpectrumResult:
    request: FrameSpectrumRequest
    noise: FramePsd | None
    pulse_frequencies: NDArray[np.float64] | None
    pulse_db: NDArray[np.float64] | None
    frames: int
    skipped: int
    population: str = ""

    @property
    def message(self) -> str:
        """Caveats a reader must see beside the plot."""
        if not self.frames:
            return (
                f"No frame had {self.request.frame_samples:,} finite samples; nothing to "
                "transform."
            )
        start, stop = self.request.segment
        resolution = format_frequency(1e9 / self.request.sample_period_ns / (stop - start))
        if self.request.region == SpectrumRegion.PRETRIGGER:
            return (
                f"Pretrigger segments of {stop - start} samples: Δf = {resolution}, so "
                "lines closer than that merge and nothing below it is resolved."
            )
        return (
            "Whole frames contain the pulses: this is pulse plus noise. Use the "
            f"pretrigger region for baseline noise. Δf = {resolution}."
        )


@dataclass
class FrameSpectrumAccumulator:
    """Linear-power Welch average over frames, plus the mean pulse."""

    request: FrameSpectrumRequest
    frames: int = 0
    skipped: int = 0
    _power: NDArray[np.float64] = field(init=False)
    _mean_square: float = field(init=False, default=0.0)
    _pulse_sum: NDArray[np.float64] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        start, stop = self.request.segment
        self._power = np.zeros((stop - start) // 2 + 1)
        if self.request.mean_pulse:
            self._pulse_sum = np.zeros(self.request.frame_samples)

    def add(self, rows: NDArray[np.generic], complete: NDArray[np.bool_] | None = None) -> None:
        """Add a (frames x samples) block; rows shorter than the frame are skipped."""
        request = self.request
        block = np.asarray(rows)
        if block.ndim != 2:
            raise ValueError("Frames must be a two-dimensional block")
        usable = np.ones(block.shape[0], dtype=np.bool_) if complete is None else complete.copy()
        if block.shape[1] < request.frame_samples:
            usable[:] = False
        for first in range(0, block.shape[0], request.batch_rows):
            selected = usable[first:first + request.batch_rows]
            part = block[first:first + request.batch_rows][selected]
            data = part[:, :request.frame_samples].astype(np.float64)
            finite = np.isfinite(data).all(axis=1)
            data = data[finite]
            self.skipped += int(selected.size - data.shape[0])
            if not data.shape[0]:
                continue
            start, stop = request.segment
            segment = data[:, start:stop]
            centred = segment - segment.mean(axis=1, keepdims=True)
            self._mean_square += float(np.einsum("ij,ij->", centred, centred)) / (stop - start)
            spectra = np.fft.rfft(centred * window_values(request.window, stop - start), axis=1)
            self._power += np.sum(spectra.real**2 + spectra.imag**2, axis=0)
            if self._pulse_sum is not None:
                region = data[:, request.baseline_start:request.baseline_end]
                baseline = (
                    np.median(region, axis=1) if request.baseline_method == "median"
                    else region.mean(axis=1)
                )
                self._pulse_sum += request.polarity * (data - baseline[:, None]).sum(axis=0)
            self.frames += data.shape[0]

    def result(self, population: str = "") -> FrameSpectrumResult:
        request = self.request
        noise: FramePsd | None = None
        pulse_frequencies = pulse_db = None
        if self.frames:
            start, stop = request.segment
            n = stop - start
            weights = window_values(request.window, n)
            s1, s2 = float(weights.sum()), float(np.dot(weights, weights))
            fs = 1e9 / request.sample_period_ns
            noise = FramePsd(
                frequencies=np.asarray(np.fft.rfftfreq(n, d=1.0 / fs), dtype=np.float64),
                psd=self._power / self.frames * one_sided_factor(n) / (fs * s2),
                n=n,
                sample_period_ns=request.sample_period_ns,
                window=request.window,
                detrend=Detrend.CONSTANT,
                s1=s1,
                s2=s2,
                input_variance=self._mean_square / self.frames,
                frames=self.frames,
                averages=float(self.frames),
            )
            if self._pulse_sum is not None:
                pulse = self._pulse_sum / self.frames
                spectrum = np.fft.rfft(pulse)
                energy = spectrum.real**2 + spectrum.imag**2
                peak = float(energy.max())
                pulse_frequencies = np.asarray(
                    np.fft.rfftfreq(pulse.size, d=request.sample_period_ns * 1e-9),
                    dtype=np.float64,
                )
                pulse_db = 10 * np.log10(
                    np.maximum(energy / peak if peak > 0 else energy, 1e-30)
                )
        return FrameSpectrumResult(
            request, noise, pulse_frequencies, pulse_db, self.frames, self.skipped, population,
        )
