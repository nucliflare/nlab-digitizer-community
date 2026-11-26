"""Power spectral density of digitizer waveform frames, in ADC codes.

This is a diagnostic of the analogue path and baseline: interference lines,
supply ripple, the noise floor and filter responses. It does not measure
energy. Pulses arriving at random times add a continuum, not lines.

Each frame is one segment of a modified periodogram (Welch 1967): it is
detrended, multiplied by a periodic (DFT-even) window ``w`` of length N and
transformed. With ``S1 = sum(w)`` and ``S2 = sum(w**2)`` the one-sided density

    P(f_k) = c_k |X_k|^2 / (fs S2),   c_k = 2 except c = 1 at DC and Nyquist,

in codes^2/Hz at ``f_k = k fs / N``, is what
``scipy.signal.periodogram(scaling="density")`` returns. Its integral over
0..fs/2 equals the window-weighted mean square of the detrended frame
(Parseval), so for stationary noise ``parseval_check()`` tends to 1. The
equivalent noise bandwidth is ``ENBW = fs S2 / S1^2`` and the coherent gain
``S1 / N`` (Heinzel et al. 2002). Frames are averaged in linear power and
converted to dB only for display: the mean of log power is biased about
2.5 dB low for noise.

A sinusoid of peak amplitude A gives ``|X_k| = A S1 / 2`` at its bin, so
``tone_amplitude()`` reads A back through the coherent gain. Between bins
the reading falls by the window's scalloping loss (Harris 1978). Measured
for these periodic windows with N = 4096, tone half a bin off centre:

=================  ===========  =============  ======================
Window             ENBW [bins]  Coherent gain  Worst scalloping loss
=================  ===========  =============  ======================
``boxcar``         1.000        1.000          3.92 dB
``hann``           1.500        0.500          1.42 dB
``blackmanharris`` 2.004        0.359          0.83 dB
``flattop``        3.770        0.216          0.01 dB
=================  ===========  =============  ======================

Use ``flattop`` for tone amplitudes and ``hann`` for noise floors.

A Scope frame is short: ``delta_f = fs / N``, at least 61 kHz for the
longest 8188-point viewer frame at 8 ns. Mains and supply ripple belong to
the periodic-sum window-mean series instead. The live viewer and IP122+
periodic DMA deliver mean4 points (the mean of four 2 ns ADC samples), whose
response ``mean_decimation_response()`` falls to 0.653 (-3.7 dB) at the
62.5 MHz Nyquist frequency; content between 62.5 and 125 MHz aliases with
weak attenuation. Results are in ADC codes. No code full scale is assumed,
so no dBFS scale is offered: the int16 transport carries 12-bit ADS5407
codes, and no attribute or document in this repository defines their full
scale. A dBFS axis would need that scale from the PetaLinux scope driver
(user-api.md) as an advertised attribute.

References
----------
Welch, P. D. (1967). The use of fast Fourier transform for the estimation
    of power spectra: a method based on time averaging over short, modified
    periodograms. IEEE Trans. Audio Electroacoust. AU-15, 70-73.
    doi:10.1109/TAU.1967.1161901
Harris, F. J. (1978). On the use of windows for harmonic analysis with the
    discrete Fourier transform. Proc. IEEE 66, 51-83.
    doi:10.1109/PROC.1978.10837
Heinzel, G., Ruediger, A., Schilling, R. (2002). Spectrum and spectral
    density estimation by the discrete Fourier transform (DFT), including a
    comprehensive list of window functions and some new flat-top windows.
    Max-Planck-Institut fuer Gravitationsphysik (Albert-Einstein-Institut),
    Hannover. hdl:11858/00-001M-0000-0013-557A-5
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

# SciPy is imported inside the functions that need it: the Scope view imports
# this module, and application start-up must not load SciPy (see
# tests/test_startup_imports.py).

MIN_SAMPLES = 16
"""Shortest accepted frame; shorter segments have no useful resolution."""

MINIMUM_COMPENSATED_RESPONSE = 0.1
"""``compensated_psd()`` never divides where |H| is below this."""


class SpectrumWindow(StrEnum):
    HANN = "hann"
    BLACKMAN_HARRIS = "blackmanharris"
    FLATTOP = "flattop"
    BOXCAR = "boxcar"


SCALLOPING_LOSS_DB: dict[SpectrumWindow, float] = {
    SpectrumWindow.BOXCAR: 3.92,
    SpectrumWindow.HANN: 1.42,
    SpectrumWindow.BLACKMAN_HARRIS: 0.83,
    SpectrumWindow.FLATTOP: 0.01,
}
"""Worst-case amplitude loss for a tone half a bin from a bin centre."""


class Detrend(StrEnum):
    CONSTANT = "constant"
    LINEAR = "linear"
    NONE = "none"


class SpectrumAveraging(StrEnum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    MAX_HOLD = "max"
    MIN_HOLD = "min"


class SpectrumUnit(StrEnum):
    DB = "db"
    PSD = "psd"
    ASD = "asd"


UNIT_LABELS: dict[SpectrumUnit, str] = {
    SpectrumUnit.DB: "dB re 1 code²/Hz",
    SpectrumUnit.PSD: "codes²/Hz",
    SpectrumUnit.ASD: "codes/√Hz",
}


@dataclass(frozen=True)
class FramePsd:
    """One-sided PSD in codes^2/Hz with the metadata needed to interpret it.

    ``frames`` counts the segments averaged into ``psd``; ``averages`` is the
    equivalent number of independent averages, so ``relative_std`` is the
    expected relative standard deviation of each noise bin.
    """

    frequencies: NDArray[np.float64]
    psd: NDArray[np.float64]
    n: int
    sample_period_ns: float
    window: SpectrumWindow
    detrend: Detrend
    s1: float
    s2: float
    input_variance: float
    frames: int = 1
    averages: float = 1.0

    @property
    def fs(self) -> float:
        """Sample rate in Hz."""
        return 1e9 / self.sample_period_ns

    @property
    def delta_f(self) -> float:
        """Bin spacing in Hz."""
        return self.fs / self.n

    @property
    def enbw(self) -> float:
        """Equivalent noise bandwidth in Hz."""
        return self.fs * self.s2 / self.s1**2

    @property
    def enbw_bins(self) -> float:
        return self.n * self.s2 / self.s1**2

    @property
    def relative_std(self) -> float:
        return 1.0 / math.sqrt(self.averages) if self.averages > 0 else math.inf


@dataclass(frozen=True)
class Spur:
    """A narrow line above the local median floor."""

    frequency: float
    level: float
    excess_db: float
    width_bins: int
    bin: int


@lru_cache(maxsize=32)
def window_values(window: SpectrumWindow, n: int) -> NDArray[np.float64]:
    """Read-only periodic (DFT-even) window of length ``n``."""
    from scipy.signal import get_window

    values = np.asarray(get_window(str(window), n, fftbins=True), dtype=np.float64)
    values.setflags(write=False)
    return values


def _finite_frame(samples: NDArray[np.generic], sample_period_ns: float) -> NDArray[np.float64]:
    if not math.isfinite(sample_period_ns) or sample_period_ns <= 0:
        raise ValueError("Spectrum needs a positive, finite sample period")
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size < MIN_SAMPLES:
        raise ValueError(f"Spectrum needs at least {MIN_SAMPLES} samples; got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("Spectrum needs a finite waveform")
    return values


def _detrended(values: NDArray[np.float64], detrend: Detrend) -> NDArray[np.float64]:
    if detrend == Detrend.CONSTANT:
        return np.asarray(values - values.mean(), dtype=np.float64)
    if detrend == Detrend.LINEAR:
        from scipy.signal import detrend as scipy_detrend

        return np.asarray(scipy_detrend(values, type="linear"), dtype=np.float64)
    return values


def one_sided_factor(n: int) -> NDArray[np.float64]:
    """Fold ``|X_k|^2`` of an ``n``-point rfft into a one-sided density: 2, or 1 at DC/Nyquist."""
    factor = np.full(n // 2 + 1, 2.0)
    factor[0] = 1.0
    if n % 2 == 0:
        factor[-1] = 1.0
    return factor


def frame_psd(
    samples: NDArray[np.generic],
    sample_period_ns: float,
    *,
    window: SpectrumWindow | str = SpectrumWindow.HANN,
    detrend: Detrend | str = Detrend.CONSTANT,
) -> FramePsd:
    """Return the one-sided PSD of one frame in codes^2/Hz, 0..fs/2.

    Equal to ``scipy.signal.periodogram(..., window, scaling="density")``
    with a periodic window. Raises ValueError for fewer than 16 or
    non-finite samples, and for a non-positive or non-finite sample period.
    """
    window = SpectrumWindow(window)
    detrend = Detrend(detrend)
    values = _finite_frame(samples, float(sample_period_ns))
    n = values.size
    weights = window_values(window, n)
    s1 = float(weights.sum())
    s2 = float(np.dot(weights, weights))
    detrended = _detrended(values, detrend)
    fs = 1e9 / float(sample_period_ns)
    spectrum = np.fft.rfft(detrended * weights)
    power = (spectrum.real**2 + spectrum.imag**2) * one_sided_factor(n) / (fs * s2)
    return FramePsd(
        frequencies=np.asarray(np.fft.rfftfreq(n, d=1.0 / fs), dtype=np.float64),
        psd=power,
        n=n,
        sample_period_ns=float(sample_period_ns),
        window=window,
        detrend=detrend,
        s1=s1,
        s2=s2,
        input_variance=float(np.dot(detrended, detrended) / n),
    )


def _is_edge_bin(index: int, n: int) -> bool:
    return index == 0 or (n % 2 == 0 and index == n // 2)


def tone_amplitude(spectrum: FramePsd, frequency: float, *, search_bins: int = 1) -> float:
    """Peak amplitude in codes of a sinusoid near ``frequency``.

    Uses the largest bin within ``search_bins`` of the nearest bin and the
    window's coherent gain, ``A = 2 |X_k| / S1``. An off-bin tone reads low
    by at most the window's scalloping loss (``SCALLOPING_LOSS_DB``). For an
    averaged spectrum the result is the RMS-equivalent amplitude.
    """
    if not math.isfinite(frequency) or frequency < 0 or frequency > spectrum.fs / 2:
        raise ValueError("Tone frequency must lie within 0..fs/2")
    centre = int(round(frequency / spectrum.delta_f))
    low = max(0, centre - search_bins)
    high = min(spectrum.psd.size, centre + search_bins + 1)
    index = low + int(np.argmax(spectrum.psd[low:high]))
    edge = _is_edge_bin(index, spectrum.n)
    factor = 1.0 if edge else 2.0
    magnitude_squared = spectrum.psd[index] * spectrum.fs * spectrum.s2 / factor
    return float((1.0 if edge else 2.0) * math.sqrt(magnitude_squared) / spectrum.s1)


def parseval_check(spectrum: FramePsd) -> float:
    """Ratio of the integrated PSD to the detrended input's mean square.

    1 for a boxcar window, and 1 on average for stationary noise with any
    window. NaN when the detrended input has no variance.
    """
    if spectrum.input_variance <= 0:
        return math.nan
    return float(spectrum.psd.sum() * spectrum.delta_f / spectrum.input_variance)


def mean_decimation_response(
    frequency: float | NDArray[np.float64],
    factor: int = 4,
    base_period_ns: float = 2.0,
) -> NDArray[np.float64]:
    """|H(f)| of an M-sample mean on a ``base_period_ns`` grid.

    ``H = sin(M pi f / f0) / (M sin(pi f / f0))`` with ``f0 = 1 / base``.
    For mean4 of 2 ns samples |H| is 0.906 at 31.25 MHz and 0.653
    (-3.7 dB) at the 62.5 MHz Nyquist frequency of 8 ns points.
    """
    if factor < 1 or not math.isfinite(base_period_ns) or base_period_ns <= 0:
        raise ValueError("Decimation needs a positive factor and base period")
    f0 = 1e9 / base_period_ns
    x = np.pi * np.asarray(frequency, dtype=np.float64) / f0
    numerator = np.sin(factor * x)
    denominator = factor * np.sin(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        response = np.where(np.abs(denominator) > 1e-15, numerator / denominator, 1.0)
    return np.asarray(np.abs(response), dtype=np.float64)


def compensated_psd(
    spectrum: FramePsd,
    factor: int,
    base_period_ns: float = 2.0,
) -> NDArray[np.float64]:
    """Divide out an M-sample mean's power response where |H| >= 0.1.

    Bins with a weaker response are returned unchanged, never amplified.
    Aliased content from above the Nyquist frequency is not recovered.
    """
    response = mean_decimation_response(spectrum.frequencies, factor, base_period_ns)
    result = spectrum.psd.copy()
    usable = response >= MINIMUM_COMPENSATED_RESPONSE
    result[usable] /= response[usable] ** 2
    return result


def decimation_factor(sample_period_ns: float, base_period_ns: float = 2.0) -> int:
    """M of an M-sample mean grid, or 1 when the period is not a multiple."""
    ratio = sample_period_ns / base_period_ns
    rounded = int(round(ratio))
    return rounded if rounded >= 1 and abs(ratio - rounded) < 1e-9 else 1


def find_spurs(
    spectrum: FramePsd,
    *,
    threshold_db_over_median: float = 10.0,
    max_count: int = 5,
    floor_bins: int | None = None,
) -> list[Spur]:
    """Narrow lines more than ``threshold_db_over_median`` above the local floor.

    The floor is a running median of ``floor_bins`` bins (odd), wide enough
    that a line does not lift it. Each contiguous run of bins above the
    threshold yields one spur at its maximum, refined by three-point
    parabolic interpolation of the dB values. The DC bin is excluded.
    Strongest excess first.
    """
    from scipy.ndimage import median_filter

    power = spectrum.psd
    if power.size < 5 or max_count < 1:
        return []
    if floor_bins is None:
        floor_bins = min(63, max(5, (power.size // 16) * 2 + 1))
    floor_bins = min(floor_bins | 1, power.size - (1 - power.size % 2))
    floor = median_filter(power, size=floor_bins, mode="nearest")
    tiny = np.finfo(np.float64).tiny
    level_db = 10.0 * np.log10(np.maximum(power, tiny))
    excess = level_db - 10.0 * np.log10(np.maximum(floor, tiny))
    above = excess >= threshold_db_over_median
    above[0] = False
    spurs: list[Spur] = []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], above.astype(np.int8), [0]))))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        index = int(start + np.argmax(excess[start:stop]))
        offset = 0.0
        if 0 < index < power.size - 1:
            a, b, c = level_db[index - 1 : index + 2]
            curvature = a - 2.0 * b + c
            if curvature < 0:
                offset = float(np.clip(0.5 * (a - c) / curvature, -0.5, 0.5))
        spurs.append(Spur(
            frequency=(index + offset) * spectrum.delta_f,
            level=float(power[index]),
            excess_db=float(excess[index]),
            width_bins=int(stop - start),
            bin=index,
        ))
    spurs.sort(key=lambda spur: spur.excess_db, reverse=True)
    return spurs[:max_count]


def line_threshold_db(averages: float) -> float:
    """Excess over the median floor that averaged noise alone almost never reaches.

    A noise bin averaged over K independent periodograms is chi-square with
    2K degrees of freedom, scaled by 1/2K. Its 1 - 1e-6 quantile relative to
    its median (Wilson-Hilferty approximation) plus 1 dB margin, clamped to
    4..10 dB: 10 dB for one or two frames, about 6 dB at K = 10 and 4 dB from
    about K = 40. A fixed 10 dB would hide real lines in long averages.

    Wilson, E. B., Hilferty, M. M. (1931). The distribution of chi-square.
    Proc. Natl. Acad. Sci. USA 17, 684-688. doi:10.1073/pnas.17.12.684
    """
    k = max(1.0, float(averages)) if math.isfinite(averages) else 1.0
    z = NormalDist().inv_cdf(1 - 1e-6)
    c = 1.0 / (9.0 * k)
    ratio = (1 - c + z * math.sqrt(c)) ** 3 / (1 - c) ** 3
    return min(10.0, max(4.0, 10 * math.log10(ratio) + 1.0))


def plotted_bins(n: int) -> slice:
    """The rfft bins of an N-point spectrum worth plotting.

    Drops DC, which detrending removes and a log axis cannot place, and the
    Nyquist bin of an even N: the one-sided density does not double it (as in
    SciPy), so drawn it looks like a spurious 3 dB drop at the right edge.
    """
    return slice(1, n // 2) if n % 2 == 0 else slice(1, n // 2 + 1)


def format_frequency(hz: float) -> str:
    """Compact SI frequency, for example ``61.04 kHz`` or ``125 MHz``."""
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if abs(hz) >= scale:
            return f"{hz / scale:.4g} {unit}"
    return f"{hz:.4g} Hz"


def to_display_units(psd: NDArray[np.float64], unit: SpectrumUnit | str) -> NDArray[np.float64]:
    """Convert linear codes^2/Hz for display only; never average the result."""
    unit = SpectrumUnit(unit)
    values = np.asarray(psd, dtype=np.float64)
    if unit == SpectrumUnit.PSD:
        return values.copy()
    if unit == SpectrumUnit.ASD:
        return np.sqrt(np.maximum(values, 0.0))
    return 10.0 * np.log10(np.maximum(values, np.finfo(np.float64).tiny))


class AccumulatedSpectrum:
    """Fixed-memory average of frame spectra with explicit reset semantics.

    Mean, maximum and minimum are all kept on linear power. ``LINEAR`` is the
    Welch mean of independent frames and holds once ``average_count`` frames
    are in (0 = no limit); ``EXPONENTIAL`` weights new frames by
    ``max(1/K, 1/N)``, a linear warm-up followed by alpha = 1/N. ``MAX_HOLD``
    and ``MIN_HOLD`` select which trace is shown; their mean stays linear.

    Spectra of different length, sample period, source or trigger mode are
    never combined: such a frame restarts the average and records why. So do
    window, detrend, averaging and count changes. Invalid frames are counted
    and skipped, not raised.
    """

    def __init__(
        self,
        *,
        window: SpectrumWindow | str = SpectrumWindow.HANN,
        averaging: SpectrumAveraging | str = SpectrumAveraging.LINEAR,
        average_count: int = 100,
        detrend: Detrend | str = Detrend.CONSTANT,
    ) -> None:
        self.window = SpectrumWindow(window)
        self.averaging = SpectrumAveraging(averaging)
        self.detrend = Detrend(detrend)
        if average_count < 0:
            raise ValueError("Average count must be zero (no limit) or positive")
        self.average_count = int(average_count)
        self.mean: NDArray[np.float64] | None = None
        self.maximum: NDArray[np.float64] | None = None
        self.minimum: NDArray[np.float64] | None = None
        self.frames = 0
        self.rejected_frames = 0
        self.last_rejection = ""
        self.reset_reason = ""
        self.source: str | None = None
        self.trigger_mode: str | None = None
        self.n = 0
        self.sample_period_ns = 0.0
        self.mean_input_variance = 0.0
        self._template: FramePsd | None = None

    # -- configuration -------------------------------------------------
    def clear(self, reason: str = "cleared") -> None:
        self.mean = self.maximum = self.minimum = None
        self.frames = 0
        self.rejected_frames = 0
        self.last_rejection = ""
        self.reset_reason = reason
        self.source = None
        self.trigger_mode = None
        self.n = 0
        self.sample_period_ns = 0.0
        self.mean_input_variance = 0.0
        self._template = None

    def set_window(self, window: SpectrumWindow | str) -> None:
        window = SpectrumWindow(window)
        if window != self.window:
            self.window = window
            self.clear(f"window changed to {window.value}")

    def set_detrend(self, detrend: Detrend | str) -> None:
        detrend = Detrend(detrend)
        if detrend != self.detrend:
            self.detrend = detrend
            self.clear(f"detrend changed to {detrend.value}")

    def set_averaging(self, averaging: SpectrumAveraging | str) -> None:
        averaging = SpectrumAveraging(averaging)
        if averaging == self.averaging:
            return
        exponential = SpectrumAveraging.EXPONENTIAL
        restart = (averaging == exponential) != (self.averaging == exponential)
        self.averaging = averaging
        if restart:
            self.clear(f"averaging changed to {averaging.value}")

    def set_average_count(self, count: int) -> None:
        if count < 0:
            raise ValueError("Average count must be zero (no limit) or positive")
        if count != self.average_count:
            self.average_count = int(count)
            self.clear(f"average count changed to {count or 'unlimited'}")

    # -- accumulation --------------------------------------------------
    @property
    def complete(self) -> bool:
        """The linear average has its requested number of frames and holds."""
        return (
            self.averaging == SpectrumAveraging.LINEAR
            and self.average_count > 0
            and self.frames >= self.average_count
        )

    @property
    def effective_averages(self) -> float:
        if self.averaging == SpectrumAveraging.EXPONENTIAL and self.average_count > 0:
            return float(min(self.frames, 2 * self.average_count - 1))
        return float(self.frames)

    @property
    def relative_std(self) -> float:
        """Expected relative standard deviation of a mean noise bin, ~1/sqrt(K)."""
        averages = self.effective_averages
        return 1.0 / math.sqrt(averages) if averages > 0 else math.inf

    def _context_change(
        self, n: int, sample_period_ns: float, source: str, trigger_mode: str | None,
    ) -> str:
        if self._template is None:
            return ""
        if n != self.n:
            return f"frame length {self.n} → {n} points"
        if sample_period_ns != self.sample_period_ns:
            return f"sample period {self.sample_period_ns:g} → {sample_period_ns:g} ns"
        if source != self.source:
            return f"source {self.source} → {source}"
        if trigger_mode != self.trigger_mode:
            return f"trigger mode {self.trigger_mode} → {trigger_mode}"
        return ""

    def add(
        self,
        samples: NDArray[np.generic],
        *,
        source: str,
        sample_period_ns: float,
        trigger_mode: str | None = None,
    ) -> bool:
        """Add one frame; return whether the spectrum changed."""
        values = np.asarray(samples).reshape(-1)
        try:
            spectrum = frame_psd(
                values, sample_period_ns, window=self.window, detrend=self.detrend,
            )
        except ValueError as exc:
            self.rejected_frames += 1
            self.last_rejection = str(exc)
            return False
        change = self._context_change(spectrum.n, spectrum.sample_period_ns, source, trigger_mode)
        if change:
            self.clear(change)
        if self.complete:
            return False
        if self._template is None:
            self._template = replace(spectrum, psd=np.empty(0))
            self.source = source
            self.trigger_mode = trigger_mode
            self.n = spectrum.n
            self.sample_period_ns = spectrum.sample_period_ns
        power = spectrum.psd
        self.frames += 1
        if self.mean is None or self.maximum is None or self.minimum is None:
            self.mean = power.copy()
            self.maximum = power.copy()
            self.minimum = power.copy()
            self.mean_input_variance = spectrum.input_variance
        else:
            weight = 1.0 / self.frames
            if self.averaging == SpectrumAveraging.EXPONENTIAL and self.average_count > 0:
                weight = max(weight, 1.0 / self.average_count)
            self.mean += weight * (power - self.mean)
            self.mean_input_variance += weight * (
                spectrum.input_variance - self.mean_input_variance
            )
            np.maximum(self.maximum, power, out=self.maximum)
            np.minimum(self.minimum, power, out=self.minimum)
        self.last_rejection = ""
        return True

    def result(self, trace: str = "display") -> FramePsd | None:
        """The ``mean``, ``max`` or ``min`` trace, or the selected ``display`` one."""
        if self._template is None or self.mean is None:
            return None
        if trace == "display":
            trace = {SpectrumAveraging.MAX_HOLD: "max", SpectrumAveraging.MIN_HOLD: "min"}.get(
                self.averaging, "mean"
            )
        arrays = {"mean": self.mean, "max": self.maximum, "min": self.minimum}
        if trace not in arrays:
            raise ValueError(f"Unknown spectrum trace {trace!r}")
        power = arrays[trace]
        assert power is not None
        return replace(
            self._template,
            psd=power.copy(),
            input_variance=self.mean_input_variance,
            frames=self.frames,
            averages=self.effective_averages if trace == "mean" else 1.0,
        )


def write_spectrum_csv(
    path: Path, spectrum: AccumulatedSpectrum, metadata: Mapping[str, object],
) -> None:
    """Write frequency, mean, max and min in codes^2/Hz after ``#`` metadata lines."""
    mean = spectrum.result("mean")
    maximum = spectrum.result("max")
    minimum = spectrum.result("min")
    if mean is None or maximum is None or minimum is None:
        raise ValueError("No spectrum has been accumulated")
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write("# Scope noise spectrum (one-sided PSD, linear power average)\n")
        header: dict[str, object] = {
            "source": spectrum.source,
            "sample_rate_hz": f"{mean.fs:.9g}",
            "points_n": mean.n,
            "bin_spacing_hz": f"{mean.delta_f:.9g}",
            "window": mean.window.value,
            "enbw_hz": f"{mean.enbw:.9g}",
            "detrend": mean.detrend.value,
            "averaging": spectrum.averaging.value,
            "average_count": spectrum.average_count or "unlimited",
            "frames_k": spectrum.frames,
            "relative_std": f"{spectrum.relative_std:.4g}",
            "parseval_ratio": f"{parseval_check(mean):.6g}",
            "trigger_mode": spectrum.trigger_mode,
            "units": "codes^2/Hz (ADC codes; no full scale assumed)",
            **metadata,
        }
        for key, value in header.items():
            stream.write(f"# {key}: {value}\n")
        writer = csv.writer(stream)
        writer.writerow(["frequency_hz", "mean_psd", "max_psd", "min_psd"])
        for row in zip(mean.frequencies, mean.psd, maximum.psd, minimum.psd, strict=True):
            writer.writerow([f"{value:.9g}" for value in row])
