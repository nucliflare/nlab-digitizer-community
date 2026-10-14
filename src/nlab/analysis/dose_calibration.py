"""Data contracts and CSV persistence for dose-rate calibration logging."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np


@dataclass(frozen=True)
class McaDoseCalibrationSnapshot:
    """Latest MCA statistics and accumulated spectrum for one channel."""

    histogram: np.ndarray
    count_rate_raw: int
    pulse_deadtime_ms: float
    acquisition_elapsed_s: float
    roi: tuple[int, int] | None


@dataclass(frozen=True)
class PsuDoseCalibrationSnapshot:
    """Latest detector-bias and temperature values for one channel."""

    hv_setpoint_v: float
    hv_readback_v: float | None
    sipm_setpoint_v: float | None
    sipm_readback_v: float | None
    temperature_analog_c: float | None
    temperature_digital_c: float | None
    adc_temperature_c: float | None


@dataclass(frozen=True)
class DigitizerTemperatureSnapshot:
    """Latest board-level temperature diagnostics for one digitizer."""

    hat_temperature_c: float | None
    fpga_temperature_c: float | None
    ads5407_temperature_raw: float | None


@dataclass(frozen=True)
class PsdDoseCalibrationSnapshot:
    """Current cumulative PSD population integrals and selection geometry."""

    cut: float
    roi: tuple[float, float]
    below_cut_total_counts: int
    above_cut_total_counts: int
    below_cut_roi_counts: int
    above_cut_roi_counts: int


@dataclass(frozen=True)
class DifferentialSpectrumIntegrals:
    """Counts added since the preceding spectrum snapshot."""

    total_counts: int
    roi_counts: int | None


def differential_spectrum_integrals(
    current: np.ndarray,
    previous: np.ndarray,
    roi: tuple[int, int] | None,
) -> DifferentialSpectrumIntegrals:
    """Integrate the non-negative per-bin difference of two MCA spectra."""
    current_counts = np.asarray(current, dtype=np.uint64)
    previous_counts = np.asarray(previous, dtype=np.uint64)
    if current_counts.ndim != 1 or previous_counts.ndim != 1:
        raise ValueError("MCA spectra must be one-dimensional")
    if current_counts.shape != previous_counts.shape:
        raise ValueError("consecutive MCA spectra must have the same shape")

    difference = np.zeros(current_counts.shape, dtype=np.uint64)
    increasing = current_counts >= previous_counts
    np.subtract(current_counts, previous_counts, out=difference, where=increasing)
    total = int(np.sum(difference, dtype=np.uint64))
    if roi is None or len(difference) == 0:
        return DifferentialSpectrumIntegrals(total_counts=total, roi_counts=None)

    low, high = sorted((int(roi[0]), int(roi[1])))
    low = int(np.clip(low, 0, len(difference) - 1))
    high = int(np.clip(high, 0, len(difference) - 1))
    roi_counts = int(np.sum(difference[low : high + 1], dtype=np.uint64))
    return DifferentialSpectrumIntegrals(total_counts=total, roi_counts=roi_counts)


class DoseCalibrationCsvWriter:
    """Create one new CSV and flush every appended calibration row."""

    def __init__(self, path: Path, fieldnames: tuple[str, ...]) -> None:
        if not fieldnames:
            raise ValueError("dose calibration CSV requires at least one column")
        self.path = path
        self.fieldnames = fieldnames
        self._stream: TextIO = path.open("x", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._stream,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        self._writer.writeheader()
        self._stream.flush()

    def append(self, row: dict[str, object]) -> None:
        """Append exactly the configured schema and make it durable immediately."""
        unexpected = set(row).difference(self.fieldnames)
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"dose calibration row has unexpected columns: {names}")
        self._writer.writerow({name: row.get(name, "") for name in self.fieldnames})
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

