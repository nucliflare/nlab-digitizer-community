"""Private helpers shared by the MCA spectrum fitters.

The photopeak fitter (:mod:`nlab.analysis.peak_fitting`) and the Compton-edge
fitter (:mod:`nlab.analysis.compton_edge`) use the same count-aware statistic
selection, Poisson deviance residual and background shapes. They live here so
that both fitters stay numerically identical. Nothing in this module imports
lmfit or scipy.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from nlab.analysis.spectrum import Spectrum

FitStatistic = Literal["auto", "poisson", "weighted", "unweighted"]
ResolvedFitStatistic = Literal["poisson", "weighted", "unweighted"]
FIT_STATISTICS: frozenset[str] = frozenset({"auto", "poisson", "weighted", "unweighted"})


def resolve_statistic(
    spectrum: Spectrum,
    requested: FitStatistic,
    counts: np.ndarray,
) -> ResolvedFitStatistic:
    """Resolve ``auto`` and validate an explicit Poisson request for one spectrum."""
    if requested == "auto":
        return "poisson" if spectrum.poisson_counts and np.all(counts >= 0) else "weighted"
    if requested == "poisson":
        if not spectrum.poisson_counts or np.any(counts < 0):
            raise ValueError("Poisson fitting requires an unscaled, non-negative count spectrum")
        return "poisson"
    return requested


def poisson_deviance_residual(observed: np.ndarray, calculated: np.ndarray) -> np.ndarray:
    """Signed square-root Poisson deviance with a penalty for a non-positive model.

    The squared residuals sum to the Poisson likelihood chi-square of Baker and
    Cousins (1984), doi:10.1016/0167-5087(84)90016-4.
    """
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


def poisson_deviance_residual_derivative(
    observed: np.ndarray,
    calculated: np.ndarray,
) -> np.ndarray:
    """Derivative of :func:`poisson_deviance_residual` with respect to the model.

    For ``r = sign(y - m) sqrt(D)`` with ``D = 2 (m - y + y ln(y/m))`` the
    derivative is ``-|m - y| / (m sqrt(D))``, which tends to ``-1/sqrt(m)`` as
    ``m -> y``. Where the model is below the positive floor the residual is the
    linear penalty term, whose derivative is ``1/scale``.
    """
    floor = np.finfo(np.float64).tiny
    expected = np.clip(calculated, floor, None)
    log_term = np.zeros_like(observed)
    positive = observed > 0
    log_term[positive] = observed[positive] * (
        np.log(observed[positive]) - np.log(expected[positive])
    )
    deviance = np.maximum(2.0 * (expected - observed + log_term), 0.0)
    root = np.sqrt(deviance)
    limit = -1.0 / np.sqrt(expected)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = -np.abs(expected - observed) / (expected * root)
    relative = np.abs(expected - observed) / np.maximum(expected, 1.0)
    derivative = np.where((root > 0) & (relative > 1.0e-6), exact, limit)
    negative_model = calculated < floor
    if np.any(negative_model):
        scale = max(float(np.max(np.abs(observed))), 1.0)
        derivative[negative_model] = 1.0 / scale
    return np.asarray(derivative, dtype=np.float64)


def statistic_uncertainties(variances: np.ndarray) -> np.ndarray:
    """Per-bin standard deviations used by the weighted least-squares statistic."""
    return np.asarray(np.sqrt(np.clip(variances, 1.0, None)), dtype=np.float64)


def statistic_residual(
    statistic: ResolvedFitStatistic,
    observed: np.ndarray,
    calculated: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Residual vector whose sum of squares is the minimised fit statistic."""
    if statistic == "poisson":
        return poisson_deviance_residual(observed, calculated)
    if statistic == "weighted":
        return np.asarray(
            (observed - calculated) / statistic_uncertainties(variances), dtype=np.float64
        )
    return np.asarray(observed - calculated, dtype=np.float64)


def centred_exponential(
    x: np.ndarray,
    amplitude: float,
    rate: float,
    x_ref: float,
) -> np.ndarray:
    """Exponential background ``amplitude * exp(rate * (x - x_ref))`` with a clipped exponent."""
    return np.asarray(amplitude * np.exp(np.clip(rate * (x - x_ref), -100.0, 100.0)))


def finite_optional(value: float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None


def readonly(values: np.ndarray) -> np.ndarray:
    copied = np.asarray(values, dtype=np.float64).copy()
    copied.setflags(write=False)
    return copied
