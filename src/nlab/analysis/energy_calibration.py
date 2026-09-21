"""Energy-scale fitting and serialisation for MCA histogram channels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

import numpy as np

CalibrationModel = Literal["linear", "quadratic"]
FingerprintValue = bool | int | float | str


@dataclass(frozen=True)
class SpectrumSnapshot:
    snapshot_id: str
    channel: int
    counts: np.ndarray
    label: str
    created_utc: str
    elapsed_s: float
    live: bool
    fingerprint: dict[str, FingerprintValue]

    @classmethod
    def create(
        cls,
        *,
        channel: int,
        counts: np.ndarray,
        label: str,
        elapsed_s: float,
        live: bool,
        fingerprint: Mapping[str, FingerprintValue],
    ) -> SpectrumSnapshot:
        copied = np.asarray(counts, dtype=np.uint32).copy()
        copied.setflags(write=False)
        return cls(
            snapshot_id=uuid4().hex,
            channel=channel,
            counts=copied,
            label=label,
            created_utc=datetime.now(UTC).isoformat(),
            elapsed_s=float(elapsed_s),
            live=live,
            fingerprint=dict(fingerprint),
        )


@dataclass(frozen=True)
class CalibrationPoint:
    channel: float
    energy_kev: float
    label: str = ""
    source: str = ""
    enabled: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "energy_kev": self.energy_kev,
            "label": self.label,
            "source": self.source,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, value: object) -> CalibrationPoint:
        if not isinstance(value, Mapping):
            raise ValueError("calibration point must be a mapping")
        try:
            channel = float(value["channel"])
            energy = float(value["energy_kev"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("calibration point needs numeric channel and energy_kev") from exc
        if not np.isfinite(channel) or not np.isfinite(energy):
            raise ValueError("calibration point values must be finite")
        return cls(
            channel=channel,
            energy_kev=energy,
            label=str(value.get("label", "")),
            source=str(value.get("source", "")),
            enabled=bool(value.get("enabled", True)),
        )


@dataclass(frozen=True)
class EnergyCalibration:
    """Polynomial energy mapping with coefficients in ascending power order."""

    model: CalibrationModel
    coefficients_kev: tuple[float, ...]
    points: tuple[CalibrationPoint, ...]
    rms_residual_kev: float
    max_residual_kev: float
    fingerprint: dict[str, FingerprintValue]
    updated_utc: str

    def energy(self, channel: float | np.ndarray) -> float | np.ndarray:
        value = np.polynomial.polynomial.polyval(channel, self.coefficients_kev)
        if np.ndim(value) == 0:
            return float(value)
        return np.asarray(value)

    def channel_scale_for_binning(self, binning_index: int) -> float:
        """Map channels at the current binning to the calibration's channel basis."""
        reference = self.fingerprint.get("binning")
        if isinstance(reference, bool) or not isinstance(reference, int):
            return 1.0
        return 2.0 ** (binning_index - reference)

    def energy_at_binning(
        self,
        channel: float | np.ndarray,
        binning_index: int,
    ) -> float | np.ndarray:
        """Evaluate energy after accounting for the current power-of-two bin width."""
        return self.energy(channel * self.channel_scale_for_binning(binning_index))

    def settings_compatible(
        self,
        current: Mapping[str, FingerprintValue],
        *,
        allow_binning_rescale: bool,
    ) -> bool:
        """Return whether current processing settings can use this calibration."""
        if not self.fingerprint:
            return True
        reference = dict(self.fingerprint)
        current_values = dict(current)
        if allow_binning_rescale:
            reference.pop("binning", None)
            current_values.pop("binning", None)
        return reference == current_values

    def residuals(self) -> np.ndarray:
        enabled = [point for point in self.points if point.enabled]
        channels = np.asarray([point.channel for point in enabled], dtype=np.float64)
        energies = np.asarray([point.energy_kev for point in enabled], dtype=np.float64)
        return energies - np.asarray(self.energy(channels), dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "coefficients_kev": list(self.coefficients_kev),
            "points": [point.to_dict() for point in self.points],
            "rms_residual_kev": self.rms_residual_kev,
            "max_residual_kev": self.max_residual_kev,
            "fingerprint": dict(self.fingerprint),
            "updated_utc": self.updated_utc,
        }

    @classmethod
    def from_dict(cls, value: object) -> EnergyCalibration:
        if not isinstance(value, Mapping):
            raise ValueError("energy calibration must be a mapping")
        model = str(value.get("model", "linear"))
        if model not in {"linear", "quadratic"}:
            raise ValueError(f"unsupported energy calibration model: {model}")
        raw_points = value.get("points")
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
            raise ValueError("energy calibration points must be a list")
        points = tuple(CalibrationPoint.from_dict(point) for point in raw_points)
        raw_fingerprint = value.get("fingerprint", {})
        if not isinstance(raw_fingerprint, Mapping):
            raise ValueError("energy calibration fingerprint must be a mapping")
        fingerprint: dict[str, FingerprintValue] = {}
        for key, item in raw_fingerprint.items():
            if not isinstance(item, (bool, int, float, str)):
                raise ValueError(f"invalid fingerprint value for {key}")
            fingerprint[str(key)] = item
        fitted = fit_energy_calibration(
            points,
            model=model,  # type: ignore[arg-type]
            fingerprint=fingerprint,
        )
        updated = str(value.get("updated_utc", fitted.updated_utc))
        return replace(fitted, updated_utc=updated)


def fit_energy_calibration(
    points: Sequence[CalibrationPoint],
    *,
    model: CalibrationModel = "linear",
    fingerprint: Mapping[str, FingerprintValue] | None = None,
    max_channel: float = 16_383.0,
) -> EnergyCalibration:
    """Fit a monotonic channel-to-keV mapping from enabled reference points."""

    all_points = tuple(points)
    enabled = tuple(point for point in all_points if point.enabled)
    degree = 1 if model == "linear" else 2
    required = degree + 1
    if len(enabled) < required:
        raise ValueError(f"{model} calibration requires at least {required} enabled points")

    channels = np.asarray([point.channel for point in enabled], dtype=np.float64)
    energies = np.asarray([point.energy_kev for point in enabled], dtype=np.float64)
    if not np.all(np.isfinite(channels)) or not np.all(np.isfinite(energies)):
        raise ValueError("calibration points must be finite")
    if np.any(channels < 0) or np.any(channels > max_channel):
        raise ValueError(f"calibration channels must be within 0-{max_channel:g}")
    if np.any(energies < 0):
        raise ValueError("calibration energies cannot be negative")
    if len(np.unique(channels)) != len(channels):
        raise ValueError("calibration points must use distinct channel positions")

    coefficients = np.polynomial.polynomial.polyfit(channels, energies, degree)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("energy calibration fit is not finite")
    derivative_at_ends = coefficients[1] + 2.0 * coefficients[2] * np.asarray(
        [0.0, max_channel]
    ) if degree == 2 else np.asarray([coefficients[1], coefficients[1]])
    if np.any(derivative_at_ends <= 0):
        raise ValueError("energy calibration must increase across the MCA channel range")

    fitted = np.polynomial.polynomial.polyval(channels, coefficients)
    residuals = energies - fitted
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    maximum = float(np.max(np.abs(residuals)))
    return EnergyCalibration(
        model=model,
        coefficients_kev=tuple(float(value) for value in coefficients),
        points=all_points,
        rms_residual_kev=rms,
        max_residual_kev=maximum,
        fingerprint=dict(fingerprint or {}),
        updated_utc=datetime.now(UTC).isoformat(),
    )
