"""Joint and independent Gaussian figure-of-merit fits for PSD projections."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from lmfit import Minimizer, Parameters
from lmfit.minimizer import MinimizerResult

_FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))
_MIN_FIT_BINS = 12


@dataclass(frozen=True)
class PsdFomFit:
    """Result of joint or independent Poisson-deviance Gaussian fits."""

    success: bool
    message: str
    x: np.ndarray
    observed: np.ndarray
    best_fit: np.ndarray
    first_component: np.ndarray
    second_component: np.ndarray
    background_component: np.ndarray
    centroids: tuple[float, float]
    centroid_uncertainties: tuple[float | None, float | None]
    sigmas: tuple[float, float]
    fwhms: tuple[float, float]
    fwhm_uncertainties: tuple[float | None, float | None]
    fom: float
    reduced_deviance: float
    evaluations: int
    warnings: tuple[str, ...]
    fit_mode: Literal["double", "separate"] = "double"
    fit_regions: tuple[tuple[float, float], ...] = ()
    use_background: bool = True

    @property
    def poor_convergence(self) -> bool:
        """Whether diagnostics make the numerical result unsafe to trust."""
        return bool(self.warnings) or not self.success

    @property
    def varied_parameters(self) -> int:
        """Six peak parameters plus one joint or two independent offsets."""
        offsets = 2 if self.fit_mode == "separate" else 1
        return 6 + offsets * int(self.use_background)

    @property
    def reduced_chi_square(self) -> float:
        """Compatibility alias for callers written before Poisson weighting."""
        return self.reduced_deviance


def fit_psd_fom_lm(
    ratio: np.ndarray,
    counts: np.ndarray,
    *,
    roi: tuple[float, float],
    centroid_seeds: tuple[float, float],
    max_nfev: int = 10_000,
    use_background: bool = True,
) -> PsdFomFit:
    """Fit two Gaussian populations with a constant background using LM.

    The objective is the signed Poisson-deviance residual, so empty and sparse
    bins are handled without observed-count inverse-variance bias. Bounds are
    applied through lmfit's internal parameter transforms while MINPACK's
    Levenberg-Marquardt solver performs the fit. With use_background=False,
    the constant count offset is fixed to zero rather than fitted.
    """
    ratio_values, count_values = _validated_projection(ratio, counts, max_nfev)

    roi_low, roi_high = sorted(map(float, roi))
    if not math.isfinite(roi_low) or not math.isfinite(roi_high) or roi_low >= roi_high:
        raise ValueError("fit ROI must contain two increasing finite bounds")
    selected = (ratio_values >= roi_low) & (ratio_values <= roi_high)
    x = ratio_values[selected]
    y = count_values[selected]
    if len(x) < _MIN_FIT_BINS:
        raise ValueError(f"fit ROI must contain at least {_MIN_FIT_BINS} bins")
    if np.ptp(x) <= 0.0:
        raise ValueError("fit ROI must span a non-zero ratio range")
    if float(np.sum(y)) <= 0.0 or not np.any(y != y[0]):
        raise ValueError("fit ROI contains no resolvable count structure")

    first_seed, second_seed = sorted(map(float, centroid_seeds))
    if not (roi_low < first_seed < second_seed < roi_high):
        raise ValueError("centroid estimates must be ordered inside the fit ROI")

    bin_width = float(np.median(np.diff(x)))
    if not math.isfinite(bin_width) or bin_width <= 0.0:
        raise ValueError("PSD ratio bins must be strictly increasing")
    span = roi_high - roi_low
    split = 0.5 * (first_seed + second_seed)
    background_guess = max(0.1, float(np.percentile(y, 10.0))) if use_background else 0.0
    sigma_min = max(bin_width / 4.0, np.finfo(np.float64).eps)
    sigma_max = span / 2.0
    first_height, first_sigma = _initial_peak_shape(
        x,
        y,
        search_range=(roi_low, split),
        background=background_guess,
        bin_width=bin_width,
        sigma_bounds=(sigma_min, sigma_max),
    )
    second_height, second_sigma = _initial_peak_shape(
        x,
        y,
        search_range=(split, roi_high),
        background=background_guess,
        bin_width=bin_width,
        sigma_bounds=(sigma_min, sigma_max),
    )

    # Express the second centroid as a positive fraction of the distance from
    # the first centroid to the ROI high edge. This prevents label swapping
    # without trapping either centroid on one side of the seed midpoint.
    separation_fraction = (second_seed - first_seed) / (roi_high - first_seed)
    fraction_margin = min(1.0e-6, bin_width / span / 10.0)

    parameters = Parameters()
    parameters.add("background", value=background_guess, min=0.0, vary=use_background)
    parameters.add("first_height", value=first_height, min=0.0)
    parameters.add(
        "first_center",
        value=first_seed,
        min=roi_low,
        max=roi_high - bin_width,
    )
    parameters.add("first_sigma", value=first_sigma, min=sigma_min, max=sigma_max)
    parameters.add("second_height", value=second_height, min=0.0)
    parameters.add(
        "separation_fraction",
        value=separation_fraction,
        min=fraction_margin,
        max=1.0 - fraction_margin,
    )
    parameters.add("roi_high_bound", value=roi_high, vary=False)
    parameters.add(
        "second_center",
        expr=("first_center + separation_fraction * (roi_high_bound - first_center)"),
    )
    parameters.add("second_sigma", value=second_sigma, min=sigma_min, max=sigma_max)

    def objective(params: Parameters) -> np.ndarray:
        return _poisson_deviance_residual(y, _model(params, x))

    result = Minimizer(objective, parameters).minimize(method="leastsq", max_nfev=max_nfev)
    first = _gaussian(
        x,
        float(result.params["first_height"].value),
        float(result.params["first_center"].value),
        float(result.params["first_sigma"].value),
    )
    second = _gaussian(
        x,
        float(result.params["second_height"].value),
        float(result.params["second_center"].value),
        float(result.params["second_sigma"].value),
    )
    background = np.full_like(x, float(result.params["background"].value))
    best_fit = background + first + second
    centroids = (
        float(result.params["first_center"].value),
        float(result.params["second_center"].value),
    )
    sigmas = (
        float(result.params["first_sigma"].value),
        float(result.params["second_sigma"].value),
    )
    fwhms = (_FWHM_FACTOR * sigmas[0], _FWHM_FACTOR * sigmas[1])
    denominator = fwhms[0] + fwhms[1]
    fom = abs(centroids[1] - centroids[0]) / denominator
    degrees_of_freedom = max(len(y) - len(result.var_names), 1)
    reduced_deviance = (
        float(np.sum(np.square(_poisson_deviance_residual(y, best_fit)))) / degrees_of_freedom
    )
    warnings = _fit_warnings(
        result=result,
        y=y,
        roi=(roi_low, roi_high),
        bin_width=bin_width,
        centroids=centroids,
        sigmas=sigmas,
        heights=(
            float(result.params["first_height"].value),
            float(result.params["second_height"].value),
        ),
        background=float(result.params["background"].value),
        reduced_deviance=reduced_deviance,
    )
    centroid_uncertainties = (
        _finite_optional(result.params["first_center"].stderr),
        _finite_optional(result.params["second_center"].stderr),
    )
    sigma_uncertainties = (
        _finite_optional(result.params["first_sigma"].stderr),
        _finite_optional(result.params["second_sigma"].stderr),
    )
    fwhm_uncertainties = (
        None if sigma_uncertainties[0] is None else _FWHM_FACTOR * sigma_uncertainties[0],
        None if sigma_uncertainties[1] is None else _FWHM_FACTOR * sigma_uncertainties[1],
    )
    return PsdFomFit(
        success=bool(result.success),
        message=str(result.message),
        x=_readonly(x),
        observed=_readonly(y),
        best_fit=_readonly(best_fit),
        first_component=_readonly(first),
        second_component=_readonly(second),
        background_component=_readonly(background),
        centroids=centroids,
        centroid_uncertainties=centroid_uncertainties,
        sigmas=sigmas,
        fwhms=fwhms,
        fwhm_uncertainties=fwhm_uncertainties,
        fom=float(fom),
        reduced_deviance=reduced_deviance,
        evaluations=int(result.nfev),
        warnings=warnings,
        fit_regions=((roi_low, roi_high),),
        use_background=use_background,
    )


def _validated_projection(
    ratio: np.ndarray, counts: np.ndarray, max_nfev: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(ratio, dtype=np.float64)
    y = np.asarray(counts, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("PSD ratio and count inputs must be one-dimensional")
    if x.shape != y.shape:
        raise ValueError("PSD ratio and count inputs must have the same length")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("PSD projection contains non-finite values")
    if np.any(y < 0):
        raise ValueError("PSD projection counts cannot be negative")
    if np.any(np.diff(x) <= 0):
        raise ValueError("PSD ratio bins must be strictly increasing")
    if max_nfev <= 0:
        raise ValueError("maximum fit evaluations must be positive")
    return x, y


def fit_psd_fom_separate_lm(
    ratio: np.ndarray,
    counts: np.ndarray,
    *,
    rois: tuple[tuple[float, float], tuple[float, float]],
    centroid_seeds: tuple[float, float],
    max_nfev: int = 10_000,
    use_background: bool = True,
) -> PsdFomFit:
    """Fit an independent Gaussian plus constant background in each window.

    Use only for well-separated populations whose other Gaussian tail is
    negligible inside each window. The gap contributes neither observations
    nor residuals. Each window has four varied parameters with its offset
    enabled, or three with the offset fixed to zero. Combined reduced deviance
    uses the total fitted bins minus the total varied parameters.
    Counts and widths have the same meaning as in the joint fit.
    """
    ratio_values, count_values = _validated_projection(ratio, counts, max_nfev)
    regions = tuple(tuple(sorted(map(float, roi))) for roi in rois)
    if len(regions) != 2 or any(
        len(region) != 2 or not all(map(math.isfinite, region)) or region[0] >= region[1]
        for region in regions
    ):
        raise ValueError("each peak ROI must contain two increasing finite bounds")
    if regions[0][1] >= regions[1][0]:
        raise ValueError("peak ROIs must be ordered and non-overlapping, with a gap between them")
    seeds = tuple(map(float, centroid_seeds))
    if len(seeds) != 2:
        raise ValueError("two centroid estimates are required")
    results: list[MinimizerResult] = []
    coordinates: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    components: list[np.ndarray] = []
    backgrounds: list[np.ndarray] = []
    warnings: list[str] = []
    deviance = 0.0
    for index, ((low, high), seed) in enumerate(zip(regions, seeds, strict=True), start=1):
        selected = (ratio_values >= low) & (ratio_values <= high)
        x, y = ratio_values[selected], count_values[selected]
        if len(x) < _MIN_FIT_BINS:
            raise ValueError(
                f"population {index} fit ROI must contain at least {_MIN_FIT_BINS} bins"
            )
        if not low < seed < high:
            raise ValueError(f"population {index} centroid estimate must be inside its fit ROI")
        if np.sum(y) <= 0 or not np.any(y != y[0]):
            raise ValueError(f"population {index} fit ROI contains no resolvable count structure")
        width = float(np.median(np.diff(x)))
        sigma_min, sigma_max = width / 4.0, (high - low) / 2.0
        background = max(0.1, float(np.percentile(y, 10))) if use_background else 0.0
        height, sigma = _initial_peak_shape(
            x, y, search_range=(low, high), background=background,
            bin_width=width, sigma_bounds=(sigma_min, sigma_max),
        )
        parameters = Parameters()
        parameters.add("background", value=background, min=0.0, vary=use_background)
        parameters.add("height", value=height, min=0.0)
        parameters.add("center", value=seed, min=low, max=high)
        parameters.add("sigma", value=sigma, min=sigma_min, max=sigma_max)

        def objective(params: Parameters) -> np.ndarray:
            model = float(params["background"].value) + _gaussian(
                x, float(params["height"].value), float(params["center"].value),
                float(params["sigma"].value),
            )
            return _poisson_deviance_residual(y, model)

        result = Minimizer(objective, parameters).minimize(method="leastsq", max_nfev=max_nfev)
        center = float(result.params["center"].value)
        sigma = float(result.params["sigma"].value)
        height = float(result.params["height"].value)
        background = float(result.params["background"].value)
        peak = _gaussian(x, height, center, sigma)
        local_deviance = float(np.sum(np.square(_poisson_deviance_residual(y, peak + background))))
        local_warnings = _fit_warnings(
            result=result, y=y, roi=(low, high), bin_width=width,
            centroids=(center, center), sigmas=(sigma, sigma), heights=(height, height),
            background=background,
            reduced_deviance=local_deviance / (len(x) - len(result.var_names)),
            check_separation=False,
        )
        warnings.extend(f"Population {index}: {warning}" for warning in local_warnings)
        results.append(result)
        coordinates.append(x)
        observations.append(y)
        components.append(peak)
        backgrounds.append(np.full_like(x, background))
        deviance += local_deviance

    first, second = results
    centers = (float(first.params["center"].value), float(second.params["center"].value))
    sigmas = (float(first.params["sigma"].value), float(second.params["sigma"].value))
    first_error = _finite_optional(first.params["sigma"].stderr)
    second_error = _finite_optional(second.params["sigma"].stderr)
    component1 = np.concatenate((components[0], np.zeros_like(components[1])))
    component2 = np.concatenate((np.zeros_like(components[0]), components[1]))
    background_values = np.concatenate(backgrounds)
    x = np.concatenate(coordinates)
    return PsdFomFit(
        success=bool(first.success and second.success),
        message=f"Population 1: {first.message}; Population 2: {second.message}",
        x=_readonly(x), observed=_readonly(np.concatenate(observations)),
        best_fit=_readonly(component1 + component2 + background_values),
        first_component=_readonly(component1), second_component=_readonly(component2),
        background_component=_readonly(background_values), centroids=centers,
        centroid_uncertainties=(
            _finite_optional(first.params["center"].stderr),
            _finite_optional(second.params["center"].stderr),
        ),
        sigmas=sigmas, fwhms=(_FWHM_FACTOR * sigmas[0], _FWHM_FACTOR * sigmas[1]),
        fwhm_uncertainties=(
            None if first_error is None else _FWHM_FACTOR * first_error,
            None if second_error is None else _FWHM_FACTOR * second_error,
        ),
        fom=(centers[1] - centers[0]) / (_FWHM_FACTOR * sum(sigmas)),
        reduced_deviance=deviance / (len(x) - sum(len(result.var_names) for result in results)),
        evaluations=int(first.nfev + second.nfev), warnings=tuple(warnings),
        fit_mode="separate", fit_regions=((regions[0][0], regions[0][1]),
                                         (regions[1][0], regions[1][1])),
        use_background=use_background,
    )


def _model(parameters: Parameters, x: np.ndarray) -> np.ndarray:
    background = float(parameters["background"].value)
    first = _gaussian(
        x,
        float(parameters["first_height"].value),
        float(parameters["first_center"].value),
        float(parameters["first_sigma"].value),
    )
    second = _gaussian(
        x,
        float(parameters["second_height"].value),
        float(parameters["second_center"].value),
        float(parameters["second_sigma"].value),
    )
    return np.asarray(background + first + second, dtype=np.float64)


def _gaussian(x: np.ndarray, height: float, center: float, sigma: float) -> np.ndarray:
    return np.asarray(
        height * np.exp(-0.5 * np.square((x - center) / sigma)),
        dtype=np.float64,
    )


def _initial_peak_shape(
    x: np.ndarray,
    y: np.ndarray,
    *,
    search_range: tuple[float, float],
    background: float,
    bin_width: float,
    sigma_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Estimate an independent height and sigma from local half-height crossings."""
    low, high = search_range
    selected = np.flatnonzero((x >= low) & (x <= high))
    fallback_sigma = max(2.0 * bin_width, (high - low) / 8.0)
    if len(selected) < 3:
        return 1.0, _interior_clip(fallback_sigma, *sigma_bounds)

    smoothed = np.convolve(y, np.asarray((1.0, 2.0, 3.0, 2.0, 1.0)) / 9.0, mode="same")
    peak_index = int(selected[np.argmax(smoothed[selected])])
    peak_level = float(smoothed[peak_index])
    height = max(peak_level - background, 1.0)
    half_level = background + 0.5 * height
    selected_low = int(selected[0])
    selected_high = int(selected[-1])

    left = peak_index
    while left > selected_low and smoothed[left] > half_level:
        left -= 1
    right = peak_index
    while right < selected_high and smoothed[right] > half_level:
        right += 1
    if left == selected_low or right == selected_high:
        sigma = fallback_sigma
    else:
        left_crossing = _linear_crossing(
            x[left],
            smoothed[left],
            x[left + 1],
            smoothed[left + 1],
            half_level,
        )
        right_crossing = _linear_crossing(
            x[right - 1],
            smoothed[right - 1],
            x[right],
            smoothed[right],
            half_level,
        )
        sigma = (right_crossing - left_crossing) / _FWHM_FACTOR
    return height, _interior_clip(sigma, *sigma_bounds)


def _linear_crossing(
    first_x: float,
    first_y: float,
    second_x: float,
    second_y: float,
    level: float,
) -> float:
    if second_y == first_y:
        return 0.5 * (first_x + second_x)
    fraction = (level - first_y) / (second_y - first_y)
    return float(first_x + fraction * (second_x - first_x))


def _interior_clip(value: float, minimum: float, maximum: float) -> float:
    margin = max((maximum - minimum) * 1.0e-6, np.finfo(np.float64).eps)
    return float(np.clip(value, minimum + margin, maximum - margin))


def _poisson_deviance_residual(
    observed: np.ndarray,
    expected: np.ndarray,
) -> np.ndarray:
    """Return signed square-root Poisson-deviance residuals."""
    safe_expected = np.maximum(np.asarray(expected, dtype=np.float64), 1.0e-12)
    observed_values = np.asarray(observed, dtype=np.float64)
    terms = safe_expected - observed_values
    positive = observed_values > 0.0
    terms[positive] += observed_values[positive] * np.log(
        observed_values[positive] / safe_expected[positive]
    )
    return np.asarray(
        np.sign(observed_values - safe_expected) * np.sqrt(2.0 * np.maximum(terms, 0.0)),
        dtype=np.float64,
    )


def _fit_warnings(
    *,
    result: object,
    y: np.ndarray,
    roi: tuple[float, float],
    bin_width: float,
    centroids: tuple[float, float],
    sigmas: tuple[float, float],
    heights: tuple[float, float],
    background: float,
    reduced_deviance: float,
    check_separation: bool = True,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if not bool(getattr(result, "success", False)):
        warnings.append("LM solver did not converge")
    if not bool(getattr(result, "errorbars", False)):
        warnings.append("parameter covariance could not be estimated")
    if not math.isfinite(reduced_deviance) or reduced_deviance > 3.0:
        warnings.append("reduced Poisson deviance indicates a poor Gaussian fit")
    if float(np.sum(y)) < 100.0:
        warnings.append("fit ROI contains fewer than 100 counts")

    roi_low, roi_high = roi
    boundary_tolerance = max(bin_width, (roi_high - roi_low) * 0.005)
    if (
        centroids[0] - roi_low <= boundary_tolerance
        or roi_high - centroids[1] <= boundary_tolerance
    ):
        warnings.append("a fitted centroid is pinned near a fit boundary")
    if any(sigma <= bin_width / 2.0 for sigma in sigmas):
        warnings.append("a fitted width is below the histogram resolution")
    if any(sigma >= (roi_high - roi_low) * 0.49 for sigma in sigmas):
        warnings.append("a fitted width is pinned near its upper bound")
    if any(height / math.sqrt(max(height + background, 1.0)) < 3.0 for height in heights):
        warnings.append("a fitted population has less than 3-sigma peak significance")
    if check_separation and abs(centroids[1] - centroids[0]) <= 0.5 * (sigmas[0] + sigmas[1]):
        warnings.append("the fitted populations are not independently resolved")
    return tuple(dict.fromkeys(warnings))


def _finite_optional(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.flags.writeable = False
    return result
