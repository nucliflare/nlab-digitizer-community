"""Immutable one-dimensional spectra and count-aware arithmetic operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

import numpy as np

SpectrumAxisUnit = Literal["channel", "keV"]
SpectrumSource = Literal["mca", "csv", "spe", "wdm", "caen_text", "root", "derived"]
NormalizationMode = Literal["area", "maximum", "elapsed"]


@dataclass(frozen=True)
class Spectrum:
    """Validated spectrum with variance and provenance kept beside its counts.

    ``x`` contains bin centres. Raw spectra use Poisson variances by default;
    derived spectra retain explicitly propagated variances and can contain
    negative counts after subtraction.
    """

    spectrum_id: str
    label: str
    x: np.ndarray
    counts: np.ndarray
    variances: np.ndarray
    axis_unit: SpectrumAxisUnit
    source: SpectrumSource
    metadata: dict[str, object]
    history: tuple[str, ...] = ()
    energy_kev: np.ndarray | None = None
    poisson_counts: bool = True

    def __post_init__(self) -> None:
        x = _readonly_float_array(self.x, "x")
        counts = _readonly_float_array(self.counts, "counts")
        variances = _readonly_float_array(self.variances, "variances")
        if not len(counts):
            raise ValueError("spectrum must contain at least one bin")
        if len(x) != len(counts) or len(variances) != len(counts):
            raise ValueError("spectrum x, counts, and variances must have equal lengths")
        if len(x) > 1 and np.any(np.diff(x) <= 0):
            raise ValueError("spectrum coordinates must be strictly increasing")
        if np.any(variances < 0):
            raise ValueError("spectrum variances cannot be negative")
        energy = None
        if self.energy_kev is not None:
            energy = _readonly_float_array(self.energy_kev, "energy_kev")
            if len(energy) != len(counts):
                raise ValueError("energy_kev must have one value per spectrum bin")
            if len(energy) > 1 and np.any(np.diff(energy) <= 0):
                raise ValueError("calibrated energies must be strictly increasing")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "variances", variances)
        object.__setattr__(self, "energy_kev", energy)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "history", tuple(self.history))

    @classmethod
    def create(
        cls,
        *,
        label: str,
        counts: np.ndarray,
        x: np.ndarray | None = None,
        variances: np.ndarray | None = None,
        axis_unit: SpectrumAxisUnit = "channel",
        source: SpectrumSource,
        metadata: dict[str, object] | None = None,
        history: tuple[str, ...] = (),
        energy_kev: np.ndarray | None = None,
        poisson_counts: bool = True,
        spectrum_id: str | None = None,
    ) -> Spectrum:
        values = np.asarray(counts, dtype=np.float64)
        coordinates = (
            np.arange(len(values), dtype=np.float64)
            if x is None
            else np.asarray(x, dtype=np.float64)
        )
        variance_values = (
            np.clip(values, 0.0, None)
            if variances is None
            else np.asarray(variances, dtype=np.float64)
        )
        return cls(
            spectrum_id=spectrum_id or uuid4().hex,
            label=label,
            x=coordinates,
            counts=values,
            variances=variance_values,
            axis_unit=axis_unit,
            source=source,
            metadata=dict(metadata or {}),
            history=history,
            energy_kev=energy_kev,
            poisson_counts=poisson_counts,
        )

    @property
    def elapsed_s(self) -> float | None:
        value = self.metadata.get("elapsed_s")
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        elapsed = float(value)
        return elapsed if np.isfinite(elapsed) and elapsed > 0 else None

    @property
    def normalization_time_s(self) -> float | None:
        """Prefer detector live time, falling back to elapsed acquisition time."""
        value = self.metadata.get("live_time_s")
        if not isinstance(value, bool) and isinstance(value, int | float):
            live_time = float(value)
            if np.isfinite(live_time) and live_time > 0:
                return live_time
        return self.elapsed_s

    def energy(self, coordinate: float) -> float | None:
        """Return a calibrated energy for a fitted coordinate when available."""
        if self.axis_unit == "keV":
            return float(coordinate)
        if self.energy_kev is None:
            return None
        return float(np.interp(coordinate, self.x, self.energy_kev))


def spectra_compatible(left: Spectrum, right: Spectrum) -> bool:
    """Return whether two spectra share an arithmetic-compatible coordinate grid."""
    return bool(
        left.axis_unit == right.axis_unit
        and left.x.shape == right.x.shape
        and np.allclose(left.x, right.x, rtol=1e-10, atol=1e-12)
    )


def scale_spectrum(spectrum: Spectrum, factor: float, *, label: str | None = None) -> Spectrum:
    if not np.isfinite(factor):
        raise ValueError("scale factor must be finite")
    return _derived(
        spectrum,
        label=label or f"{spectrum.label} x {factor:g}",
        counts=spectrum.counts * factor,
        variances=spectrum.variances * factor**2,
        operation=f"scale({factor:.17g})",
        poisson_counts=spectrum.poisson_counts and factor == 1.0,
    )


def normalize_spectrum(
    spectrum: Spectrum,
    mode: NormalizationMode,
    *,
    label: str | None = None,
) -> Spectrum:
    if mode == "area":
        denominator = float(np.sum(spectrum.counts))
    elif mode == "maximum":
        denominator = float(np.max(spectrum.counts))
    elif mode == "elapsed":
        elapsed = spectrum.normalization_time_s
        if elapsed is None:
            raise ValueError(
                "acquisition-time normalization requires positive live_time_s or elapsed_s metadata"
            )
        denominator = elapsed
    else:
        raise ValueError(f"unsupported normalization mode {mode!r}")
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"cannot normalize by non-positive {mode}")
    return _derived(
        spectrum,
        label=label or f"{spectrum.label} normalized by {mode}",
        counts=spectrum.counts / denominator,
        variances=spectrum.variances / denominator**2,
        operation=f"normalize({mode}, denominator={denominator:.17g})",
        poisson_counts=False,
    )


def combine_spectra(
    left: Spectrum,
    right: Spectrum,
    *,
    operation: Literal["add", "subtract"],
    right_scale: float = 1.0,
    label: str | None = None,
) -> Spectrum:
    if not spectra_compatible(left, right):
        raise ValueError("spectrum arithmetic requires identical coordinate grids and units")
    if not np.isfinite(right_scale):
        raise ValueError("spectrum arithmetic scale must be finite")
    sign = 1.0 if operation == "add" else -1.0
    counts = left.counts + sign * right_scale * right.counts
    variances = left.variances + right_scale**2 * right.variances
    energy = _compatible_energy(left, right)
    symbol = "+" if operation == "add" else "-"
    return Spectrum.create(
        label=label or f"{left.label} {symbol} {right_scale:g} x {right.label}",
        x=left.x,
        counts=counts,
        variances=variances,
        axis_unit=left.axis_unit,
        source="derived",
        metadata={
            **left.metadata,
            "operation": operation,
            "left_spectrum_id": left.spectrum_id,
            "right_spectrum_id": right.spectrum_id,
            "right_scale": right_scale,
        },
        history=left.history
        + right.history
        + (f"{operation}({right.spectrum_id}, scale={right_scale:.17g})",),
        energy_kev=energy,
        poisson_counts=(
            operation == "add"
            and right_scale == 1.0
            and left.poisson_counts
            and right.poisson_counts
        ),
    )


def subtract_background(
    sample: Spectrum,
    background: Spectrum,
    *,
    scale: float | None = None,
    label: str | None = None,
) -> Spectrum:
    if scale is None:
        sample_elapsed = sample.normalization_time_s
        background_elapsed = background.normalization_time_s
        if sample_elapsed is None or background_elapsed is None:
            raise ValueError(
                "automatic background scaling requires positive elapsed_s metadata on both spectra"
            )
        scale = sample_elapsed / background_elapsed
    return combine_spectra(
        sample,
        background,
        operation="subtract",
        right_scale=scale,
        label=label or f"{sample.label} - background {background.label}",
    )


def rebin_spectrum(spectrum: Spectrum, factor: int, *, label: str | None = None) -> Spectrum:
    if factor < 1:
        raise ValueError("rebin factor must be at least 1")
    if factor == 1:
        return _derived(
            spectrum,
            label=label or f"{spectrum.label} rebinned x1",
            counts=spectrum.counts,
            variances=spectrum.variances,
            operation="rebin(1)",
            poisson_counts=spectrum.poisson_counts,
        )
    starts = np.arange(0, len(spectrum.counts), factor)
    counts = np.add.reduceat(spectrum.counts, starts)
    variances = np.add.reduceat(spectrum.variances, starts)
    x = np.asarray(
        [np.mean(spectrum.x[start : start + factor]) for start in starts],
        dtype=np.float64,
    )
    energy = None
    if spectrum.energy_kev is not None:
        energy = np.asarray(
            [np.mean(spectrum.energy_kev[start : start + factor]) for start in starts],
            dtype=np.float64,
        )
    return Spectrum.create(
        label=label or f"{spectrum.label} rebinned x{factor}",
        x=x,
        counts=counts,
        variances=variances,
        axis_unit=spectrum.axis_unit,
        source="derived",
        metadata={**spectrum.metadata, "rebin_factor": factor},
        history=spectrum.history + (f"rebin({factor})",),
        energy_kev=energy,
        poisson_counts=spectrum.poisson_counts,
    )


def crop_spectrum(
    spectrum: Spectrum,
    low: float,
    high: float,
    *,
    label: str | None = None,
) -> Spectrum:
    low, high = sorted((float(low), float(high)))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        raise ValueError("crop bounds must be distinct finite values")
    mask = (spectrum.x >= low) & (spectrum.x <= high)
    if np.count_nonzero(mask) < 2:
        raise ValueError("crop range must contain at least two spectrum bins")
    energy = spectrum.energy_kev[mask] if spectrum.energy_kev is not None else None
    return Spectrum.create(
        label=label or f"{spectrum.label} cropped {low:g}-{high:g}",
        x=spectrum.x[mask],
        counts=spectrum.counts[mask],
        variances=spectrum.variances[mask],
        axis_unit=spectrum.axis_unit,
        source="derived",
        metadata={**spectrum.metadata, "crop": [low, high]},
        history=spectrum.history + (f"crop({low:.17g}, {high:.17g})",),
        energy_kev=energy,
        poisson_counts=spectrum.poisson_counts,
    )


def _derived(
    source: Spectrum,
    *,
    label: str,
    counts: np.ndarray,
    variances: np.ndarray,
    operation: str,
    poisson_counts: bool,
) -> Spectrum:
    return Spectrum.create(
        label=label,
        x=source.x,
        counts=counts,
        variances=variances,
        axis_unit=source.axis_unit,
        source="derived",
        metadata={**source.metadata, "parent_spectrum_id": source.spectrum_id},
        history=source.history + (operation,),
        energy_kev=source.energy_kev,
        poisson_counts=poisson_counts,
    )


def _compatible_energy(left: Spectrum, right: Spectrum) -> np.ndarray | None:
    if left.energy_kev is None or right.energy_kev is None:
        return None
    if not np.allclose(left.energy_kev, right.energy_kev, rtol=1e-10, atol=1e-12):
        return None
    return left.energy_kev


def _readonly_float_array(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"spectrum {name} must be a finite one-dimensional array")
    result.setflags(write=False)
    return result
