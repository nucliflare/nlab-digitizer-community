"""Constrained lmfit models for one-dimensional MCA photopeak analysis."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from lmfit import Minimizer, Model, Parameters
from lmfit.models import ConstantModel, GaussianModel, LinearModel
from scipy.signal import find_peaks
from scipy.special import erfc, expit

from nlab.analysis.spectrum import Spectrum

BackgroundModel = Literal["none", "constant", "linear", "exponential", "compton", "fermi"]
FitStatistic = Literal["auto", "poisson", "weighted", "unweighted"]
ResolvedFitStatistic = Literal["poisson", "weighted", "unweighted"]


class PeakFitCancelledError(RuntimeError):
    """Raised when the caller asks an active peak fit to stop."""


@dataclass(frozen=True)
class PeakFitSpec:
    peak_count: int = 1
    background: BackgroundModel = "linear"
    statistic: FitStatistic = "auto"
    fit_min: float | None = None
    fit_max: float | None = None
    peak_centers: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.peak_count not in {1, 2, 3}:
            raise ValueError("peak count must be 1, 2, or 3")
        if self.background not in {
            "none",
            "constant",
            "linear",
            "exponential",
            "compton",
            "fermi",
        }:
            raise ValueError(f"unsupported background model {self.background!r}")
        if self.statistic not in {"auto", "poisson", "weighted", "unweighted"}:
            raise ValueError(f"unsupported fit statistic {self.statistic!r}")
        if self.peak_centers and len(self.peak_centers) != self.peak_count:
            raise ValueError("manual peak centres must match the selected peak count")


@dataclass(frozen=True)
class FitParameter:
    name: str
    value: float
    stderr: float | None
    minimum: float
    maximum: float
    varying: bool
    expression: str | None


@dataclass(frozen=True)
class PeakComponent:
    index: int
    center: float
    center_stderr: float | None
    sigma: float
    sigma_stderr: float | None
    fwhm: float
    fwhm_stderr: float | None
    area: float
    area_stderr: float | None
    height: float
    resolution_percent: float | None
    energy_kev: float | None
    energy_fwhm_kev: float | None


@dataclass(frozen=True)
class PeakFitResult:
    success: bool
    message: str
    statistic: ResolvedFitStatistic
    x: np.ndarray
    observed: np.ndarray
    best_fit: np.ndarray
    residual: np.ndarray
    components: dict[str, np.ndarray]
    peaks: tuple[PeakComponent, ...]
    parameters: tuple[FitParameter, ...]
    chi_square: float
    reduced_chi_square: float
    aic: float
    bic: float
    evaluations: int
    warnings: tuple[str, ...]


def fit_spectrum_peaks(
    spectrum: Spectrum,
    spec: PeakFitSpec,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> PeakFitResult:
    """Fit one to three Gaussian peaks and one selectable background."""
    mask = np.ones(len(spectrum.counts), dtype=bool)
    if spec.fit_min is not None:
        mask &= spectrum.x >= spec.fit_min
    if spec.fit_max is not None:
        mask &= spectrum.x <= spec.fit_max
    x = np.asarray(spectrum.x[mask], dtype=np.float64)
    y = np.asarray(spectrum.counts[mask], dtype=np.float64)
    variances = np.asarray(spectrum.variances[mask], dtype=np.float64)
    if len(x) < max(12, spec.peak_count * 5):
        raise ValueError("fit range contains too few bins for the selected model")
    if np.ptp(x) <= 0:
        raise ValueError("fit coordinates must span a non-zero range")
    if not np.any(y != y[0]):
        raise ValueError("fit range contains no spectral structure")

    statistic = _resolve_statistic(spectrum, spec.statistic, y)
    model, parameters = _build_model(x, y, spec)
    cancel_check = cancelled or (lambda: False)

    def objective(params: Parameters) -> np.ndarray:
        calculated = np.asarray(model.eval(params=params, x=x), dtype=np.float64)
        if statistic == "poisson":
            return _poisson_deviance_residual(y, calculated)
        if statistic == "weighted":
            uncertainties = np.sqrt(np.clip(variances, 1.0, None))
            return (y - calculated) / uncertainties
        return y - calculated

    def iteration_callback(
        _params: Parameters,
        _iteration: int,
        _residual: np.ndarray,
        *_args: object,
        **_kwargs: object,
    ) -> bool:
        return cancel_check()

    minimizer = Minimizer(objective, parameters, iter_cb=iteration_callback)
    result = minimizer.minimize(method="least_squares", max_nfev=10_000)
    if cancel_check() or result.aborted:
        raise PeakFitCancelledError("peak fit cancelled")

    best_fit = np.asarray(model.eval(params=result.params, x=x), dtype=np.float64)
    evaluated = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in model.eval_components(params=result.params, x=x).items()
    }
    peaks = tuple(
        _peak_component(spectrum, result.params, index)
        for index in range(1, spec.peak_count + 1)
    )
    warnings = _fit_warnings(result.params, peaks, bool(result.errorbars), statistic, y)
    parameters_out = tuple(
        FitParameter(
            name=name,
            value=float(parameter.value),
            stderr=_finite_optional(parameter.stderr),
            minimum=float(parameter.min),
            maximum=float(parameter.max),
            varying=bool(parameter.vary),
            expression=parameter.expr,
        )
        for name, parameter in result.params.items()
    )
    return PeakFitResult(
        success=bool(result.success),
        message=str(result.message),
        statistic=statistic,
        x=_readonly(x),
        observed=_readonly(y),
        best_fit=_readonly(best_fit),
        residual=_readonly(y - best_fit),
        components={name: _readonly(values) for name, values in evaluated.items()},
        peaks=peaks,
        parameters=parameters_out,
        chi_square=float(result.chisqr),
        reduced_chi_square=float(result.redchi),
        aic=float(result.aic),
        bic=float(result.bic),
        evaluations=int(result.nfev),
        warnings=warnings,
    )


def export_peak_fit_result(
    path: Path,
    *,
    spectrum: Spectrum,
    spec: PeakFitSpec,
    result: PeakFitResult,
) -> None:
    """Export a compact fit report as JSON or CSV without overwriting."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        document = {
            "kind": "nlab_mca_peak_fit",
            "spectrum": {
                "id": spectrum.spectrum_id,
                "label": spectrum.label,
                "source": spectrum.source,
                "metadata": spectrum.metadata,
                "history": spectrum.history,
            },
            "model": asdict(spec),
            "statistics": {
                "success": result.success,
                "message": result.message,
                "statistic": result.statistic,
                "chi_square": result.chi_square,
                "reduced_chi_square": result.reduced_chi_square,
                "aic": result.aic,
                "bic": result.bic,
                "evaluations": result.evaluations,
                "warnings": result.warnings,
            },
            "peaks": [asdict(peak) for peak in result.peaks],
            "parameters": [asdict(parameter) for parameter in result.parameters],
        }
        with path.open("x", encoding="utf-8") as output:
            json.dump(document, output, indent=2, default=str)
        return
    if path.suffix.lower() == ".csv":
        with path.open("x", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(
                (
                    "peak",
                    "center",
                    "center_stderr",
                    "energy_kev",
                    "fwhm",
                    "energy_fwhm_kev",
                    "area",
                    "area_stderr",
                    "resolution_percent",
                )
            )
            for peak in result.peaks:
                writer.writerow(
                    (
                        peak.index,
                        peak.center,
                        peak.center_stderr,
                        peak.energy_kev,
                        peak.fwhm,
                        peak.energy_fwhm_kev,
                        peak.area,
                        peak.area_stderr,
                        peak.resolution_percent,
                    )
                )
        return
    raise ValueError("peak-fit export must use .json or .csv")


def suggest_peak_centers(x: np.ndarray, counts: np.ndarray, peak_count: int) -> tuple[float, ...]:
    """Return stable, separated centre guesses ordered by coordinate."""
    if peak_count not in {1, 2, 3}:
        raise ValueError("peak count must be 1, 2, or 3")
    x = np.asarray(x, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if len(x) != len(counts) or not len(x):
        raise ValueError("peak suggestion needs equally sized non-empty arrays")
    baseline = float(np.percentile(counts, 15))
    signal = counts - baseline
    distance = max(1, len(signal) // (peak_count * 4))
    prominence = max(float(np.ptp(signal)) * 0.02, 1.0e-12)
    candidates, properties = find_peaks(signal, distance=distance, prominence=prominence)
    if len(candidates):
        prominences = np.asarray(properties["prominences"])
        chosen = candidates[np.argsort(prominences)[-peak_count:]].tolist()
    else:
        chosen = []
    if len(chosen) < peak_count:
        order = np.argsort(signal)[::-1]
        minimum_separation = max(1, len(signal) // (peak_count * 5))
        for index in order:
            candidate = int(index)
            if all(abs(candidate - existing) >= minimum_separation for existing in chosen):
                chosen.append(candidate)
            if len(chosen) == peak_count:
                break
    if len(chosen) < peak_count:
        chosen.extend(
            int(index)
            for index in np.linspace(0, len(x) - 1, peak_count + 2)[1:-1]
            if int(index) not in chosen
        )
    return tuple(float(x[index]) for index in sorted(chosen[:peak_count]))


def _build_model(x: np.ndarray, y: np.ndarray, spec: PeakFitSpec) -> tuple[Model, Parameters]:
    centres = (
        tuple(sorted(spec.peak_centers))
        if spec.peak_centers
        else suggest_peak_centers(x, y, spec.peak_count)
    )
    if len(set(centres)) != len(centres):
        raise ValueError("peak centres must be distinct")
    if any(centre < x[0] or centre > x[-1] for centre in centres):
        raise ValueError("peak centres must fall inside the fit range")
    peaks = [GaussianModel(prefix=f"p{index}_") for index in range(1, spec.peak_count + 1)]
    model: Model = peaks[0]
    for peak in peaks[1:]:
        model = model + peak
    background = _background_model(spec.background)
    if background is not None:
        model = model + background
    params = model.make_params()

    span = float(x[-1] - x[0])
    bin_width = float(np.median(np.diff(x)))
    sigma_guess = max(abs(bin_width), span / 50.0)
    background_level = max(0.0, float(np.percentile(y, 15)))
    for index, centre in enumerate(centres, start=1):
        left = float(x[0]) if index == 1 else (centres[index - 2] + centre) * 0.5
        right = float(x[-1]) if index == len(centres) else (centre + centres[index]) * 0.5
        sample_index = int(np.argmin(np.abs(x - centre)))
        height = max(float(y[sample_index]) - background_level, float(np.ptp(y)) * 0.05, 1.0)
        sigma = min(sigma_guess, max(abs(bin_width), (right - left) / 4.0))
        params[f"p{index}_center"].set(value=centre, min=left, max=right)
        params[f"p{index}_sigma"].set(
            value=sigma,
            min=max(abs(bin_width) * 0.2, np.finfo(float).eps),
            max=max(abs(bin_width), span * 0.5),
        )
        params[f"p{index}_amplitude"].set(
            value=height * sigma * np.sqrt(2.0 * np.pi),
            min=0.0,
        )

    if spec.background == "constant":
        params["bg_c"].set(value=background_level, min=0.0)
    elif spec.background == "linear":
        slope = float((y[-1] - y[0]) / span)
        params["bg_intercept"].set(value=max(0.0, float(y[0] - slope * x[0])))
        params["bg_slope"].set(value=slope)
    elif spec.background == "exponential":
        params["bg_amplitude"].set(value=max(background_level, 1.0e-6), min=0.0)
        params["bg_rate"].set(value=0.0, min=-20.0 / span, max=20.0 / span)
        params["bg_x_ref"].set(value=float(np.mean(x)), vary=False)
    elif spec.background == "compton":
        params["bg_amplitude"].set(value=max(float(np.ptp(y)) * 0.25, 1.0e-6), min=0.0)
        params["bg_center"].set(expr="p1_center")
        params["bg_sigma"].set(expr="p1_sigma")
    elif spec.background == "fermi":
        params["bg_amplitude"].set(value=max(float(np.ptp(y)) * 0.25, 1.0e-6), min=0.0)
        params["bg_center"].set(value=centres[0], min=float(x[0]), max=float(x[-1]))
        params["bg_width"].set(
            value=sigma_guess,
            min=max(abs(bin_width) * 0.2, np.finfo(float).eps),
            max=span,
        )
    return model, params


def _background_model(background: BackgroundModel) -> Model | None:
    if background == "none":
        return None
    if background == "constant":
        return ConstantModel(prefix="bg_")
    if background == "linear":
        return LinearModel(prefix="bg_")
    if background == "exponential":
        return Model(_centred_exponential, prefix="bg_")
    if background == "compton":
        return Model(_compton_step, prefix="bg_")
    if background == "fermi":
        return Model(_fermi_step, prefix="bg_")
    raise ValueError(f"unsupported background model {background!r}")


def _centred_exponential(
    x: np.ndarray,
    amplitude: float,
    rate: float,
    x_ref: float,
) -> np.ndarray:
    return amplitude * np.exp(np.clip(rate * (x - x_ref), -100.0, 100.0))


def _compton_step(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
) -> np.ndarray:
    return np.asarray(
        0.5 * amplitude * erfc((x - center) / (np.sqrt(2.0) * sigma)),
        dtype=np.float64,
    )


def _fermi_step(
    x: np.ndarray,
    amplitude: float,
    center: float,
    width: float,
) -> np.ndarray:
    return np.asarray(amplitude * expit(-(x - center) / width), dtype=np.float64)


def _resolve_statistic(
    spectrum: Spectrum,
    requested: FitStatistic,
    counts: np.ndarray,
) -> ResolvedFitStatistic:
    if requested == "auto":
        return "poisson" if spectrum.poisson_counts and np.all(counts >= 0) else "weighted"
    if requested == "poisson":
        if not spectrum.poisson_counts or np.any(counts < 0):
            raise ValueError("Poisson fitting requires an unscaled, non-negative count spectrum")
        return "poisson"
    return requested


def _poisson_deviance_residual(observed: np.ndarray, calculated: np.ndarray) -> np.ndarray:
    floor = np.finfo(np.float64).tiny
    expected = np.clip(calculated, floor, None)
    log_term = np.zeros_like(observed)
    positive = observed > 0
    log_term[positive] = observed[positive] * (
        np.log(observed[positive]) - np.log(expected[positive])
    )
    deviance = np.maximum(2.0 * (expected - observed + log_term), 0.0)
    residual = np.sign(observed - expected) * np.sqrt(deviance)
    negative_model = calculated < floor
    if np.any(negative_model):
        scale = max(float(np.max(np.abs(observed))), 1.0)
        residual[negative_model] -= (floor - calculated[negative_model]) / scale
    return np.asarray(residual, dtype=np.float64)


def _peak_component(spectrum: Spectrum, params: Parameters, index: int) -> PeakComponent:
    prefix = f"p{index}_"
    center = float(params[f"{prefix}center"].value)
    sigma = float(params[f"{prefix}sigma"].value)
    fwhm = float(params[f"{prefix}fwhm"].value)
    area = float(params[f"{prefix}amplitude"].value)
    height = float(params[f"{prefix}height"].value)
    energy = spectrum.energy(center)
    energy_fwhm = None
    if energy is not None:
        low_energy = spectrum.energy(center - fwhm * 0.5)
        high_energy = spectrum.energy(center + fwhm * 0.5)
        if low_energy is not None and high_energy is not None:
            energy_fwhm = high_energy - low_energy
    reference_center = energy if energy is not None else center
    reference_fwhm = energy_fwhm if energy_fwhm is not None else fwhm
    resolution = (
        100.0 * reference_fwhm / reference_center if reference_center > 0 else None
    )
    return PeakComponent(
        index=index,
        center=center,
        center_stderr=_finite_optional(params[f"{prefix}center"].stderr),
        sigma=sigma,
        sigma_stderr=_finite_optional(params[f"{prefix}sigma"].stderr),
        fwhm=fwhm,
        fwhm_stderr=_finite_optional(params[f"{prefix}fwhm"].stderr),
        area=area,
        area_stderr=_finite_optional(params[f"{prefix}amplitude"].stderr),
        height=height,
        resolution_percent=resolution,
        energy_kev=energy,
        energy_fwhm_kev=energy_fwhm,
    )


def _fit_warnings(
    params: Parameters,
    peaks: tuple[PeakComponent, ...],
    errorbars: bool,
    statistic: ResolvedFitStatistic,
    counts: np.ndarray,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if not errorbars:
        warnings.append("Parameter covariance could not be estimated.")
    if statistic == "weighted" and np.any(counts < 0):
        warnings.append("Negative derived counts were fitted with propagated-variance weights.")
    for peak in peaks:
        for name in (f"p{peak.index}_center", f"p{peak.index}_sigma"):
            parameter = params[name]
            tolerance = max(abs(parameter.value) * 1.0e-6, 1.0e-9)
            if np.isfinite(parameter.min) and abs(parameter.value - parameter.min) <= tolerance:
                warnings.append(f"{name} reached its lower bound.")
            if np.isfinite(parameter.max) and abs(parameter.value - parameter.max) <= tolerance:
                warnings.append(f"{name} reached its upper bound.")
    return tuple(warnings)


def _finite_optional(value: float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None


def _readonly(values: np.ndarray) -> np.ndarray:
    copied = np.asarray(values, dtype=np.float64).copy()
    copied.setflags(write=False)
    return copied
