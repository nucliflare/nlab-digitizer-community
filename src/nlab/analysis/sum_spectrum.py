"""Power spectrum of recorded periodic-sum window means (offline, no Qt).

A periodic-sum capture stores the mean of W consecutive 2 ns ADC samples once
per interval ``T = (W/4 + gap) x 8 ns``, the interval
``ScopeSumAnalysis.snapshot()`` reports; ``fs' = 1/T``. The noise inside one
window is lost, but the series of window means has a valid spectrum from 0 to
fs'/2. That is where mains, supply ripple and slow drift become visible, which
a Scope frame (bins of at least 61 kHz) cannot resolve.

Each value is a boxcar average of width 2W ns sampled once per T, a duty cycle
``D = 2W ns / T``. Its response is ``|H(f)| = |sin(pi f 2W ns) / (W sin(pi f 2
ns))|``: 0.637 (-3.9 dB) at fs'/2 when D = 1, first null at fs'/D. Content
above fs'/2 aliases with little attenuation.

Missing windows break the series, and a spectrum must not run across them:
IP124 skips one window while sealing each 508-window packet at gap 0, and the
host can lose more. The series is split into gapless runs. Welch (1967)
segments of L windows, Hann-windowed with 50 % overlap, are taken inside runs
only and averaged over all runs in linear power, with the normalisation of
``waveform_spectrum.frame_psd``. ``delta_f = fs'/L``: resolving 50 Hz needs
delta_f of about 10 Hz or less, so runs of at least fs'/10 windows. Runs
shorter than L, and the tail of each run after its last segment, are counted
as discarded windows. Overlapping segments are correlated; the effective
number of averages follows Welch's overlap correlation.

Optional bridging replaces each isolated single missing window by the mean of
its two neighbours. For a tone of frequency f the bridged value equals the
true one times cos(2 pi f / fs'): exact well below fs'/2, sign-flipped at
fs'/2. For white noise a bridged value carries half the power of a recorded
one, so one bridged window in 508 lowers the noise level by about 0.1 %.
Bridged windows recur once per packet, so this error modulates a tone and
adds side lines spaced fs'/508 around it, strongest for tones far from DC.
Bridging invents data: it is off by default, and bridged windows are counted
and their timestamps kept.

References
----------
Welch, P. D. (1967). The use of fast Fourier transform for the estimation
    of power spectra: a method based on time averaging over short, modified
    periodograms. IEEE Trans. Audio Electroacoust. AU-15, 70-73.
    doi:10.1109/TAU.1967.1161901
Harris, F. J. (1978). On the use of windows for harmonic analysis with the
    discrete Fourier transform. Proc. IEEE 66, 51-83.
    doi:10.1109/PROC.1978.10837
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nlab.analysis.current_capture import CurrentCaptureInfo, iter_window_means
from nlab.analysis.waveform_spectrum import (
    Detrend,
    FramePsd,
    SpectrumWindow,
    format_frequency,
    mean_decimation_response,
    one_sided_factor,
    window_values,
)
from nlab.hardware.digitizer.scope import (
    SCOPE_ADC_SAMPLE_PERIOD_NS,
    SCOPE_DATAPATH_CLOCK_PERIOD_NS,
)
from nlab.hardware.digitizer.scope_sums import MASK64

MIN_SEGMENT_LENGTH = 16
MAX_SEGMENT_LENGTH = 1 << 24
_BATCH_VALUES = 1 << 22  # bounds the memory of one vectorised FFT batch


def sum_interval_ns(window_samples: int, gap_cycles: int) -> float:
    """Window spacing ``(W/4 + gap) x 8 ns``."""
    return float((window_samples // 4 + gap_cycles) * SCOPE_DATAPATH_CLOCK_PERIOD_NS)


def window_mean_response(
    frequency: float | NDArray[np.float64], window_samples: int,
) -> NDArray[np.float64]:
    """|H(f)| of a W-sample window mean of 2 ns ADC samples."""
    return mean_decimation_response(frequency, window_samples, SCOPE_ADC_SAMPLE_PERIOD_NS)


def windows_for_resolution(delta_f_hz: float, interval_ns: float) -> int:
    """Contiguous windows that a bin spacing ``delta_f_hz`` needs: ceil(fs'/delta_f)."""
    if not delta_f_hz > 0 or not interval_ns > 0:
        raise ValueError("Resolution and interval must be positive")
    return math.ceil(1e9 / interval_ns / delta_f_hz - 1e-9)


@dataclass(frozen=True)
class SumSpectrum:
    """Welch estimate over gapless runs, with the bookkeeping to judge it."""

    spectrum: FramePsd | None
    window_samples: int
    gap_cycles: int
    segment_length: int
    bridge_single_gaps: bool
    segments: int
    runs: int
    runs_used: int
    longest_run: int
    windows: int
    used_windows: int
    discarded_windows: int
    missing_windows: int
    gaps: int
    single_gaps: int
    bridged_windows: int
    bridged_timestamps: NDArray[np.uint64]

    @property
    def interval_ns(self) -> float:
        return sum_interval_ns(self.window_samples, self.gap_cycles)

    @property
    def fs(self) -> float:
        return 1e9 / self.interval_ns

    @property
    def delta_f(self) -> float:
        return self.fs / self.segment_length

    @property
    def duty_cycle(self) -> float:
        """Fraction of each interval the window mean covers, 2W ns / T."""
        return self.window_samples * SCOPE_ADC_SAMPLE_PERIOD_NS / self.interval_ns

    @property
    def nyquist_response(self) -> float:
        return float(window_mean_response(self.fs / 2, self.window_samples)[()])

    @property
    def largest_reachable_length(self) -> int | None:
        """The longest power-of-two segment some run can hold, or None."""
        if self.longest_run < MIN_SEGMENT_LENGTH:
            return None
        return min(MAX_SEGMENT_LENGTH, 1 << int(math.log2(self.longest_run)))

    @property
    def message(self) -> str:
        """Why the requested resolution is unreachable or poorly supported, or ''."""
        if not self.windows:
            return "The capture has no windows."
        length = self.segment_length
        if not self.segments:
            text = (
                f"L = {length:,} windows (Δf = {format_frequency(self.delta_f)}) needs a "
                f"run of {length:,} contiguous windows; the longest gapless run has "
                f"{self.longest_run:,}."
            )
            largest = self.largest_reachable_length
            if largest is not None:
                text += (
                    f" The finest reachable resolution is L = {largest:,} "
                    f"(Δf = {format_frequency(self.fs / largest)})."
                )
            if self.single_gaps and not self.bridge_single_gaps:
                text += (
                    f" Bridging the {self.single_gaps:,} isolated single missing "
                    "window(s) would join runs."
                )
            return text
        if self.discarded_windows * 2 > self.windows:
            return (
                f"{self.discarded_windows:,} of {self.windows:,} windows lie in runs "
                f"shorter than L = {length:,} or in run tails and are not used."
            )
        return ""


class SumWelch:
    """Streaming Welch average of window means over gapless runs.

    Feed timestamp/mean blocks in recording order with ``append`` and call
    ``finish`` once. Memory is bounded by about one segment plus one block.
    """

    def __init__(
        self,
        *,
        window_samples: int,
        gap_cycles: int,
        segment_length: int,
        bridge_single_gaps: bool = False,
    ) -> None:
        if (
            type(segment_length) is not int
            or not MIN_SEGMENT_LENGTH <= segment_length <= MAX_SEGMENT_LENGTH
            or segment_length & (segment_length - 1)
        ):
            raise ValueError(
                f"Segment length must be a power of two from {MIN_SEGMENT_LENGTH} "
                f"to {MAX_SEGMENT_LENGTH:,}"
            )
        self.window_samples = int(window_samples)
        self.gap_cycles = int(gap_cycles)
        self.interval_ticks = self.window_samples // 4 + self.gap_cycles
        if self.window_samples < 4 or self.interval_ticks <= 0:
            raise ValueError("Invalid sum window geometry")
        self.segment_length = segment_length
        self.bridge_single_gaps = bool(bridge_single_gaps)
        self._step = segment_length // 2
        self._weights = window_values(SpectrumWindow.HANN, segment_length)
        self._power = np.zeros(segment_length // 2 + 1)
        self._mean_square_sum = 0.0
        self._chunks: list[NDArray[np.float64]] = []
        self._pending = 0
        self._run_length = 0
        self._run_segments = 0
        self._last_timestamp: int | None = None
        self._last_value = 0.0
        self._bridged: list[int] = []
        self.segments = self.runs = self.runs_used = self.longest_run = 0
        self.windows = self.used_windows = self.discarded_windows = 0
        self.missing_windows = self.gaps = self.single_gaps = self.bridged_windows = 0

    def append(self, timestamps: NDArray[np.generic], means: NDArray[np.generic]) -> None:
        stamps = np.asarray(timestamps, dtype=np.uint64).reshape(-1)
        values = np.asarray(means, dtype=np.float64).reshape(-1)
        if stamps.size != values.size:
            raise ValueError("Timestamps and window means differ in length")
        if not values.size:
            return
        if not np.isfinite(values).all():
            raise ValueError("Window means must be finite")
        interval = np.uint64(self.interval_ticks)
        steps = np.empty(stamps.size, dtype=np.uint64)
        # uint64 subtraction wraps like the native 64-bit counter.
        steps[0] = (
            stamps[0] - np.uint64(self._last_timestamp)
            if self._last_timestamp is not None else interval
        )
        steps[1:] = stamps[1:] - stamps[:-1]
        self.windows += values.size
        start = 0
        for index in np.flatnonzero(steps != interval).tolist():
            self._extend(values[start:index])
            step = int(steps[index])
            if step == 0 or step >= 1 << 63 or step % self.interval_ticks:
                raise ValueError("Window timestamps are nonmonotonic or off the interval grid")
            missing = step // self.interval_ticks - 1
            self.missing_windows += missing
            self.gaps += 1
            self.single_gaps += int(missing == 1)
            if missing == 1 and self.bridge_single_gaps:
                left = self._last_value if index == 0 else float(values[index - 1])
                self._extend(np.array([0.5 * (left + float(values[index]))]))
                self.bridged_windows += 1
                self._bridged.append((int(stamps[index]) - self.interval_ticks) & MASK64)
            else:
                self._end_run()
            start = index
        self._extend(values[start:])
        self._last_timestamp = int(stamps[-1])
        self._last_value = float(values[-1])

    def _extend(self, values: NDArray[np.float64]) -> None:
        if not values.size:
            return
        self._chunks.append(values)
        self._pending += values.size
        self._run_length += values.size
        if self._pending < self.segment_length:
            return
        data = np.concatenate(self._chunks)
        length, step = self.segment_length, self._step
        count = 1 + (data.size - length) // step
        segments = np.lib.stride_tricks.sliding_window_view(data, length)[::step]
        rows = max(1, _BATCH_VALUES // length)
        for first in range(0, count, rows):
            block = segments[first:min(count, first + rows)]
            centred = block - block.mean(axis=1, keepdims=True)
            self._mean_square_sum += float(np.einsum("ij,ij->", centred, centred)) / length
            spectra = np.fft.rfft(centred * self._weights, axis=1)
            self._power += np.sum(spectra.real**2 + spectra.imag**2, axis=0)
        self.segments += count
        self._run_segments += count
        remainder = data[count * step:]
        self._chunks = [remainder]
        self._pending = remainder.size

    def _end_run(self) -> None:
        if self._run_length:
            covered = (
                (self._run_segments - 1) * self._step + self.segment_length
                if self._run_segments else 0
            )
            self.runs += 1
            self.runs_used += int(self._run_segments > 0)
            self.used_windows += covered
            self.discarded_windows += self._run_length - covered
            self.longest_run = max(self.longest_run, self._run_length)
        self._chunks = []
        self._pending = self._run_length = self._run_segments = 0

    def finish(self) -> SumSpectrum:
        self._end_run()
        spectrum: FramePsd | None = None
        if self.segments:
            weights = self._weights
            s1, s2 = float(weights.sum()), float(np.dot(weights, weights))
            interval_ns = sum_interval_ns(self.window_samples, self.gap_cycles)
            fs = 1e9 / interval_ns
            psd = self._power / self.segments * one_sided_factor(self.segment_length) / (fs * s2)
            # Welch (1967): correlation of periodograms of segments overlapping
            # by one step; only segments within one run overlap.
            rho = (float(np.dot(weights[self._step:], weights[:-self._step])) / s2) ** 2
            total = self.segments
            averages = total**2 / (total + 2 * rho * (total - self.runs_used))
            spectrum = FramePsd(
                frequencies=np.asarray(
                    np.fft.rfftfreq(self.segment_length, d=interval_ns * 1e-9), dtype=np.float64
                ),
                psd=psd,
                n=self.segment_length,
                sample_period_ns=interval_ns,
                window=SpectrumWindow.HANN,
                detrend=Detrend.CONSTANT,
                s1=s1,
                s2=s2,
                input_variance=self._mean_square_sum / total,
                frames=total,
                averages=averages,
            )
        return SumSpectrum(
            spectrum=spectrum,
            window_samples=self.window_samples,
            gap_cycles=self.gap_cycles,
            segment_length=self.segment_length,
            bridge_single_gaps=self.bridge_single_gaps,
            segments=self.segments,
            runs=self.runs,
            runs_used=self.runs_used,
            longest_run=self.longest_run,
            windows=self.windows,
            used_windows=self.used_windows,
            discarded_windows=self.discarded_windows,
            missing_windows=self.missing_windows,
            gaps=self.gaps,
            single_gaps=self.single_gaps,
            bridged_windows=self.bridged_windows,
            bridged_timestamps=np.asarray(self._bridged, dtype=np.uint64),
        )


def capture_sum_spectrum(
    info: CurrentCaptureInfo,
    *,
    segment_length: int,
    bridge_single_gaps: bool = False,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[float], None] | None = None,
) -> SumSpectrum:
    """Welch spectrum of every window mean in a recorded sum capture.

    Reads the whole capture in recording order with the same packet checks as
    the browser. Raises CurrentReadCancelledError when ``cancelled``.
    """
    geometry = info.geometry
    if geometry is None:
        raise ValueError("The capture has no sum windows")
    welch = SumWelch(
        window_samples=geometry.frame_samples,
        gap_cycles=geometry.gap_cycles,
        segment_length=segment_length,
        bridge_single_gaps=bridge_single_gaps,
    )
    total = max(1, info.windows)
    for first, stamps, means in iter_window_means(info, cancelled=cancelled):
        welch.append(stamps, means)
        if progress is not None:
            progress((first + stamps.size) / total)
    return welch.finish()
