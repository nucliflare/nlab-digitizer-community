"""Offline PSD reconstruction from stored pulse waveforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from fast_histogram import histogram2d

BaselineMethod = Literal["median", "mean"]


@dataclass(frozen=True)
class WaveformPsdSettings:
    """Sample-domain preprocessing, integration gates, and histogram geometry."""

    baseline_start: int
    baseline_end: int
    gate_start: int
    short_end: int
    long_end: int
    polarity: Literal[-1, 1]
    baseline_method: BaselineMethod = "median"
    energy_bins: int = 512
    energy_range: tuple[float, float] = (0.0, 1_000_000.0)
    ratio_bins: int = 256
    ratio_range: tuple[float, float] = (-0.1, 1.0)

    def __post_init__(self) -> None:
        if not 0 <= self.baseline_start < self.baseline_end <= self.gate_start:
            raise ValueError("baseline must end no later than the integration start")
        if not self.gate_start < self.short_end <= self.long_end:
            raise ValueError("integration gates must satisfy start < short <= long")
        if self.polarity not in {-1, 1}:
            raise ValueError("waveform polarity must be -1 or 1")
        if self.baseline_method not in {"median", "mean"}:
            raise ValueError("baseline method must be median or mean")
        if self.energy_bins <= 0 or self.ratio_bins <= 0:
            raise ValueError("histogram bin counts must be positive")
        _validate_range(self.energy_range, "energy")
        _validate_range(self.ratio_range, "ratio")


@dataclass(frozen=True)
class WaveformCharge:
    baseline: float
    baseline_rms: float
    short_charge: float
    long_charge: float
    ratio: float


@dataclass(frozen=True)
class WaveformPsdStatistics:
    received: int
    accepted: int
    too_short: int
    nonpositive_long: int
    outside_range: int


@dataclass(frozen=True)
class WaveformPsdResult:
    matrix: np.ndarray
    statistics: WaveformPsdStatistics
    energy_range: tuple[float, float]
    ratio_range: tuple[float, float]
    calculated_long: np.ndarray
    stored_long: np.ndarray
    calculated_short: np.ndarray
    stored_short: np.ndarray

    @property
    def energy_edges(self) -> np.ndarray:
        return np.linspace(*self.energy_range, self.matrix.shape[0] + 1)

    @property
    def ratio_edges(self) -> np.ndarray:
        return np.linspace(*self.ratio_range, self.matrix.shape[1] + 1)


def infer_waveform_polarity(
    samples: np.ndarray,
    *,
    baseline_start: int,
    baseline_end: int,
    search_start: int,
) -> Literal[-1, 1]:
    """Choose one file-wide polarity from a representative pulse."""
    values = np.asarray(samples, dtype=np.float64)
    if not 0 <= baseline_start < baseline_end <= search_start < len(values):
        raise ValueError("polarity inference regions are outside the waveform")
    baseline = float(np.median(values[baseline_start:baseline_end]))
    pulse = values[search_start:] - baseline
    positive = float(np.max(pulse, initial=0.0))
    negative = abs(float(np.min(pulse, initial=0.0)))
    return -1 if negative > positive else 1


def integrate_waveform(
    samples: np.ndarray,
    settings: WaveformPsdSettings,
) -> WaveformCharge | None:
    """Baseline, orient, and integrate one waveform; return ``None`` if unusable."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or len(values) < settings.long_end:
        return None
    baseline_samples = values[settings.baseline_start : settings.baseline_end]
    baseline = float(
        np.median(baseline_samples)
        if settings.baseline_method == "median"
        else np.mean(baseline_samples)
    )
    baseline_rms = float(np.sqrt(np.mean((baseline_samples - baseline) ** 2)))
    pulse = settings.polarity * (values[settings.gate_start : settings.long_end] - baseline)
    short_width = settings.short_end - settings.gate_start
    short_charge = float(np.sum(pulse[:short_width], dtype=np.float64))
    long_charge = float(np.sum(pulse, dtype=np.float64))
    if not np.isfinite(long_charge) or long_charge <= 0:
        return WaveformCharge(
            baseline=baseline,
            baseline_rms=baseline_rms,
            short_charge=short_charge,
            long_charge=long_charge,
            ratio=float("nan"),
        )
    ratio = (long_charge - short_charge) / long_charge
    return WaveformCharge(
        baseline=baseline,
        baseline_rms=baseline_rms,
        short_charge=short_charge,
        long_charge=long_charge,
        ratio=float(ratio),
    )


class WaveformPsdAccumulator:
    """Incrementally histogram waveform-derived charges with bounded scratch memory."""

    _FLUSH_EVENTS = 8192
    _COMPARISON_EVENTS = 20_000

    def __init__(self, settings: WaveformPsdSettings) -> None:
        self.settings = settings
        self.matrix = np.zeros(
            (settings.energy_bins, settings.ratio_bins),
            dtype=np.uint64,
        )
        self._received = 0
        self._accepted = 0
        self._too_short = 0
        self._nonpositive_long = 0
        self._outside_range = 0
        self._energies: list[float] = []
        self._ratios: list[float] = []
        self._calculated_long: list[float] = []
        self._stored_long: list[float] = []
        self._calculated_short: list[float] = []
        self._stored_short: list[float] = []

    def add_waveform(
        self,
        samples: np.ndarray,
        *,
        stored_long: int | None = None,
        stored_short: int | None = None,
    ) -> WaveformCharge | None:
        self._received += 1
        if len(samples) < self.settings.long_end:
            self._too_short += 1
            return None
        charge = integrate_waveform(samples, self.settings)
        if charge is None:
            self._too_short += 1
            return None
        if not np.isfinite(charge.ratio) or charge.long_charge <= 0:
            self._nonpositive_long += 1
            return charge
        energy_low, energy_high = self.settings.energy_range
        ratio_low, ratio_high = self.settings.ratio_range
        if not (
            energy_low <= charge.long_charge < energy_high
            and ratio_low <= charge.ratio <= ratio_high
        ):
            self._outside_range += 1
            return charge
        self._accepted += 1
        self._energies.append(charge.long_charge)
        self._ratios.append(min(charge.ratio, np.nextafter(ratio_high, ratio_low)))
        if (
            stored_long is not None
            and stored_short is not None
            and len(self._stored_long) < self._COMPARISON_EVENTS
        ):
            self._calculated_long.append(charge.long_charge)
            self._stored_long.append(float(stored_long))
            self._calculated_short.append(charge.short_charge)
            self._stored_short.append(float(stored_short))
        if len(self._energies) >= self._FLUSH_EVENTS:
            self._flush()
        return charge

    def add_waveforms(
        self,
        samples: np.ndarray,
        *,
        complete: np.ndarray | None = None,
        stored_long: np.ndarray | None = None,
        stored_short: np.ndarray | None = None,
    ) -> None:
        """Integrate and histogram a rectangular waveform batch.

        The charge equations are algebraically identical to
        :func:`integrate_waveform`, but reductions run along the event axis in
        NumPy rather than dispatching several array operations per event.
        """
        values = np.asarray(samples)
        if values.ndim != 2:
            raise ValueError("waveform batch must be a two-dimensional array")
        received = len(values)
        self._received += received
        if complete is None:
            usable = np.full(received, values.shape[1] >= self.settings.long_end)
        else:
            usable = np.asarray(complete, dtype=np.bool_)
            if usable.shape != (received,):
                raise ValueError("waveform completeness mask has the wrong shape")
            if values.shape[1] < self.settings.long_end and np.any(usable):
                raise ValueError("complete waveforms do not reach the long gate")
        usable_count = int(np.count_nonzero(usable))
        self._too_short += received - usable_count
        if not usable_count:
            return

        all_usable = usable_count == received
        selected = values if all_usable else values[usable]
        baseline_samples = selected[
            :, self.settings.baseline_start : self.settings.baseline_end
        ]
        baseline = (
            np.median(baseline_samples, axis=1)
            if self.settings.baseline_method == "median"
            else np.mean(baseline_samples, axis=1)
        )
        short_width = self.settings.short_end - self.settings.gate_start
        long_width = self.settings.long_end - self.settings.gate_start
        short_sum = np.sum(
            selected[:, self.settings.gate_start : self.settings.short_end],
            axis=1,
            dtype=np.float64,
        )
        long_sum = np.sum(
            selected[:, self.settings.gate_start : self.settings.long_end],
            axis=1,
            dtype=np.float64,
        )
        short_charge = self.settings.polarity * (
            short_sum - baseline * short_width
        )
        long_charge = self.settings.polarity * (long_sum - baseline * long_width)
        ratio = np.full_like(long_charge, np.nan)
        positive = np.isfinite(long_charge) & (long_charge > 0.0)
        np.divide(
            long_charge - short_charge,
            long_charge,
            out=ratio,
            where=positive,
        )
        nonpositive = ~positive | ~np.isfinite(ratio)
        self._nonpositive_long += int(np.count_nonzero(nonpositive))

        energy_low, energy_high = self.settings.energy_range
        ratio_low, ratio_high = self.settings.ratio_range
        accepted = (
            ~nonpositive
            & (long_charge >= energy_low)
            & (long_charge < energy_high)
            & (ratio >= ratio_low)
            & (ratio <= ratio_high)
        )
        accepted_count = int(np.count_nonzero(accepted))
        self._accepted += accepted_count
        self._outside_range += usable_count - int(np.count_nonzero(nonpositive)) - accepted_count
        if not accepted_count:
            return

        self._flush()
        accepted_ratio = np.minimum(
            ratio[accepted],
            np.nextafter(ratio_high, ratio_low),
        )
        counts = histogram2d(
            long_charge[accepted],
            accepted_ratio,
            bins=(self.settings.energy_bins, self.settings.ratio_bins),
            range=(self.settings.energy_range, self.settings.ratio_range),
        )
        self.matrix += counts.astype(np.uint64, copy=False)

        if stored_long is None or stored_short is None:
            return
        long_values = np.asarray(stored_long)
        short_values = np.asarray(stored_short)
        if long_values.shape != (received,) or short_values.shape != (received,):
            raise ValueError("stored gate arrays have the wrong shape")
        remaining = self._COMPARISON_EVENTS - len(self._stored_long)
        if remaining <= 0:
            return
        selected_long = long_values if all_usable else long_values[usable]
        selected_short = short_values if all_usable else short_values[usable]
        accepted_long = selected_long[accepted][:remaining]
        accepted_short = selected_short[accepted][:remaining]
        calculated_long = long_charge[accepted][:remaining]
        calculated_short = short_charge[accepted][:remaining]
        self._calculated_long.extend(calculated_long.tolist())
        self._stored_long.extend(accepted_long.astype(np.float64).tolist())
        self._calculated_short.extend(calculated_short.tolist())
        self._stored_short.extend(accepted_short.astype(np.float64).tolist())

    def result(self) -> WaveformPsdResult:
        self._flush()
        return WaveformPsdResult(
            matrix=_readonly(self.matrix),
            statistics=WaveformPsdStatistics(
                received=self._received,
                accepted=self._accepted,
                too_short=self._too_short,
                nonpositive_long=self._nonpositive_long,
                outside_range=self._outside_range,
            ),
            energy_range=self.settings.energy_range,
            ratio_range=self.settings.ratio_range,
            calculated_long=_readonly(np.asarray(self._calculated_long)),
            stored_long=_readonly(np.asarray(self._stored_long)),
            calculated_short=_readonly(np.asarray(self._calculated_short)),
            stored_short=_readonly(np.asarray(self._stored_short)),
        )

    def _flush(self) -> None:
        if not self._energies:
            return
        counts = histogram2d(
            np.asarray(self._energies),
            np.asarray(self._ratios),
            bins=(self.settings.energy_bins, self.settings.ratio_bins),
            range=(self.settings.energy_range, self.settings.ratio_range),
        )
        self.matrix += counts.astype(np.uint64, copy=False)
        self._energies.clear()
        self._ratios.clear()


def _validate_range(value: tuple[float, float], name: str) -> None:
    low, high = value
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"{name} range must contain increasing finite values")


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result
