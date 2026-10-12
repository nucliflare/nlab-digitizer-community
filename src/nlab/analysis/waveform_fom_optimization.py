"""Discrete short/long-gate optimization for waveform-derived PSD."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nlab.analysis.psd_fom import PsdFomFit, fit_psd_fom_lm
from nlab.analysis.waveform_psd import WaveformPsdSettings


class WaveformFomOptimizationCancelledError(RuntimeError):
    """Raised internally when a gate search is cancelled."""


@dataclass(frozen=True)
class WaveformFomOptimizationRequest:
    """Fixed analysis selections and inclusive gate bounds for one search."""

    settings: WaveformPsdSettings
    energy_roi: tuple[float, float]
    fit_roi: tuple[float, float]
    centroid_seeds: tuple[float, float]
    short_bounds: tuple[int, int]
    long_bounds: tuple[int, int]
    initial_step: int

    def __post_init__(self) -> None:
        gate_start = self.settings.gate_start
        short_low, short_high = self.short_bounds
        long_low, long_high = self.long_bounds
        if not gate_start < short_low <= short_high:
            raise ValueError("short-gate search bounds must follow the integration start")
        if not gate_start < long_low <= long_high:
            raise ValueError("long-gate search bounds must follow the integration start")
        if short_low >= long_high:
            raise ValueError("gate search bounds contain no short-before-long pair")
        if not (
            short_low <= self.settings.short_end <= short_high
            and long_low <= self.settings.long_end <= long_high
        ):
            raise ValueError("current gate endpoints must lie inside the search bounds")
        if self.initial_step <= 0:
            raise ValueError("initial gate-search step must be positive")
        _validate_range(self.energy_roi, "energy ROI")
        _validate_range(self.fit_roi, "fit ROI")
        first_seed, second_seed = sorted(map(float, self.centroid_seeds))
        fit_low, fit_high = sorted(map(float, self.fit_roi))
        if not fit_low < first_seed < second_seed < fit_high:
            raise ValueError("centroid seeds must be ordered inside the fit ROI")


@dataclass(frozen=True)
class WaveformGateCandidate:
    """One gate pair with a trustworthy fitted FOM."""

    short_end: int
    long_end: int
    fom: float
    projected_events: int
    fit: PsdFomFit


@dataclass(frozen=True)
class WaveformFomOptimizationResult:
    """Best local gate pair and search diagnostics."""

    best: WaveformGateCandidate
    evaluated_candidates: int
    valid_candidates: int
    sampled_waveforms: int


OptimizationProgress = Callable[[int, int, WaveformGateCandidate | None], None]


def optimize_waveform_gate_ends(
    cumulative_charge: np.ndarray,
    request: WaveformFomOptimizationRequest,
    *,
    progress: OptimizationProgress | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> WaveformFomOptimizationResult:
    """Maximize fitted FOM with a coarse-to-fine integer pattern search.

    ``cumulative_charge[:, i]`` is the baseline-corrected integral from the
    fixed integration start through sample ``gate_start + i``. Candidate fits
    with any convergence warning are excluded from the objective.
    """
    integrals = np.asarray(cumulative_charge, dtype=np.float64)
    if integrals.ndim != 2 or not integrals.shape[0] or not integrals.shape[1]:
        raise ValueError("gate optimization requires a non-empty charge matrix")
    required_width = request.long_bounds[1] - request.settings.gate_start
    if integrals.shape[1] < required_width:
        raise ValueError("charge matrix does not reach the long-gate search bound")
    if not np.all(np.isfinite(integrals)):
        raise ValueError("charge matrix contains non-finite values")

    cache: dict[tuple[int, int], WaveformGateCandidate | None] = {}
    best: WaveformGateCandidate | None = None
    valid_candidates = 0

    def evaluate(short_end: int, long_end: int) -> WaveformGateCandidate | None:
        nonlocal best, valid_candidates
        if cancelled is not None and cancelled():
            raise WaveformFomOptimizationCancelledError
        key = (short_end, long_end)
        if key in cache:
            return cache[key]
        candidate = _evaluate_candidate(integrals, request, short_end, long_end)
        cache[key] = candidate
        if candidate is not None:
            valid_candidates += 1
            if best is None or _candidate_rank(candidate, request) > _candidate_rank(
                best, request
            ):
                best = candidate
        if progress is not None:
            progress(len(cache), valid_candidates, best)
        return candidate

    current_pair = (request.settings.short_end, request.settings.long_end)
    current = evaluate(*current_pair)
    step = max(1, int(request.initial_step))
    while step >= 1:
        moved = False
        neighbours: list[tuple[int, int]] = []
        for short_delta in (-step, 0, step):
            for long_delta in (-step, 0, step):
                if short_delta == 0 and long_delta == 0:
                    continue
                pair = (current_pair[0] + short_delta, current_pair[1] + long_delta)
                if _valid_pair(pair, request):
                    neighbours.append(pair)
        candidates = [candidate for pair in neighbours if (candidate := evaluate(*pair))]
        if candidates:
            neighbour = max(candidates, key=lambda value: _candidate_rank(value, request))
            if current is None or _candidate_rank(neighbour, request) > _candidate_rank(
                current, request
            ):
                current = neighbour
                current_pair = (neighbour.short_end, neighbour.long_end)
                moved = True
        if not moved:
            step //= 2

    if best is None:
        raise ValueError(
            "No gate pair produced a trustworthy double-Gaussian FOM fit; "
            "check the energy/fit ROIs, centroid seeds, and event count"
        )
    return WaveformFomOptimizationResult(
        best=best,
        evaluated_candidates=len(cache),
        valid_candidates=valid_candidates,
        sampled_waveforms=len(integrals),
    )


def _evaluate_candidate(
    cumulative_charge: np.ndarray,
    request: WaveformFomOptimizationRequest,
    short_end: int,
    long_end: int,
) -> WaveformGateCandidate | None:
    if not _valid_pair((short_end, long_end), request):
        return None
    gate_start = request.settings.gate_start
    short_charge = cumulative_charge[:, short_end - gate_start - 1]
    long_charge = cumulative_charge[:, long_end - gate_start - 1]
    usable = np.isfinite(long_charge) & (long_charge > 0.0)
    ratio = np.full_like(long_charge, np.nan)
    np.divide(long_charge - short_charge, long_charge, out=ratio, where=usable)

    energy_low, energy_high = _effective_energy_roi(request)
    ratio_low, ratio_high = request.settings.ratio_range
    selected = (
        usable
        & np.isfinite(ratio)
        & (long_charge >= energy_low)
        & (long_charge < energy_high)
        & (ratio >= ratio_low)
        & (ratio <= ratio_high)
    )
    projected_events = int(np.count_nonzero(selected))
    if projected_events == 0:
        return None
    counts, edges = np.histogram(
        ratio[selected],
        bins=request.settings.ratio_bins,
        range=request.settings.ratio_range,
    )
    centers = (edges[:-1] + edges[1:]) * 0.5
    try:
        fit = fit_psd_fom_lm(
            centers,
            counts,
            roi=request.fit_roi,
            centroid_seeds=request.centroid_seeds,
        )
    except (RuntimeError, TypeError, ValueError):
        return None
    if fit.poor_convergence or not math.isfinite(fit.fom) or fit.fom <= 0.0:
        return None
    return WaveformGateCandidate(
        short_end=short_end,
        long_end=long_end,
        fom=fit.fom,
        projected_events=projected_events,
        fit=fit,
    )


def _effective_energy_roi(
    request: WaveformFomOptimizationRequest,
) -> tuple[float, float]:
    edges = np.linspace(
        *request.settings.energy_range,
        request.settings.energy_bins + 1,
    )
    roi_low, roi_high = sorted(map(float, request.energy_roi))
    first = int(np.clip(np.searchsorted(edges, roi_low, side="right") - 1, 0, len(edges) - 1))
    last = int(np.clip(np.searchsorted(edges, roi_high, side="left"), 0, len(edges) - 1))
    if last <= first:
        raise ValueError("energy ROI does not include a histogram bin")
    return float(edges[first]), float(edges[last])


def _valid_pair(
    pair: tuple[int, int],
    request: WaveformFomOptimizationRequest,
) -> bool:
    short_end, long_end = pair
    return (
        request.short_bounds[0] <= short_end <= request.short_bounds[1]
        and request.long_bounds[0] <= long_end <= request.long_bounds[1]
        and request.settings.gate_start < short_end < long_end
    )


def _candidate_rank(
    candidate: WaveformGateCandidate,
    request: WaveformFomOptimizationRequest,
) -> tuple[float, int, int]:
    distance = abs(candidate.short_end - request.settings.short_end) + abs(
        candidate.long_end - request.settings.long_end
    )
    return candidate.fom, -distance, candidate.projected_events


def _validate_range(value: tuple[float, float], name: str) -> None:
    low, high = sorted(map(float, value))
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError(f"{name} must contain two increasing finite values")
