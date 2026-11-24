"""Klein–Nishina Compton-edge model, fit, diagnostics and calibration checks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import nlab.analysis.compton_edge as compton_module
from nlab.analysis import dietze_klein_table
from nlab.analysis.compton_edge import (
    COMPTON_EDGE_ENERGY_UNIT,
    ComptonBackground,
    ComptonEdgeDataset,
    ComptonEdgeFitCancelledError,
    ComptonEdgeFitProgress,
    ComptonEdgeFitSpec,
    ComptonLine,
    KleinNishinaResponse,
    ResolutionTermName,
    SimulatedResponse,
    compton_edge_calibration_points,
    compton_edge_kev,
    compton_edge_summary_rows,
    compton_shape_diagnostics,
    export_compton_edge_result,
    extrapolation_warning,
    fit_compton_edges,
    folded_line_response,
    klein_nishina_electron_spectrum,
    load_simulated_response,
    nuclide_compton_lines,
    parse_custom_energies,
)
from nlab.analysis.energy_calibration import (
    EnergyCalibration,
    EnergyUnit,
    fit_energy_calibration,
)
from nlab.analysis.peak_fitting import PeakFitCancelledError
from nlab.analysis.spectrum import Spectrum, scale_spectrum

TRUE_GAIN = 5.0
TRUE_OFFSET = -60.0
TRUE_RESOLUTION = (0.04, 1.2, 0.0)
CS = (ComptonLine(661.657, nuclide="Cs-137", emission_probability_percent=85.1),)
MIX = CS + (
    ComptonLine(1173.228, nuclide="Co-60", emission_probability_percent=99.85),
    ComptonLine(1332.492, nuclide="Co-60", emission_probability_percent=99.98),
    ComptonLine(511.0, nuclide="Na-22", emission_probability_percent=180.76),
    ComptonLine(1274.537, nuclide="Na-22", emission_probability_percent=99.94),
)


def _expected(
    edges: np.ndarray,
    lines_and_counts: list[tuple[ComptonLine, float]],
    *,
    gain: float = TRUE_GAIN,
    offset: float = TRUE_OFFSET,
    resolution: tuple[float, float, float] = TRUE_RESOLUTION,
    background: float = 40.0,
) -> np.ndarray:
    centres = 0.5 * (edges[1:] + edges[:-1])
    expected = background * np.exp(-(centres - centres[0]) / (0.4 * np.ptp(centres)))
    for line, counts in lines_and_counts:
        expected = expected + counts * folded_line_response(
            edges, line, gain=gain, offset=offset, resolution=resolution
        )
    return np.asarray(expected)


def _spectrum(
    lines_and_counts: list[tuple[ComptonLine, float]],
    *,
    bins: int,
    width: float,
    seed: int,
    offset: float = TRUE_OFFSET,
    resolution: tuple[float, float, float] = TRUE_RESOLUTION,
    background: float = 40.0,
) -> Spectrum:
    edges = width * np.arange(bins + 1, dtype=np.float64)
    expected = _expected(
        edges, lines_and_counts, offset=offset, resolution=resolution, background=background
    )
    counts = np.random.default_rng(seed).poisson(np.clip(expected, 0.0, None))
    return Spectrum.create(
        label=f"synthetic {seed}",
        x=0.5 * (edges[1:] + edges[:-1]),
        counts=counts,
        source="derived",
        poisson_counts=True,
    )


def _joint_spec(**overrides: Any) -> ComptonEdgeFitSpec:
    cs = _spectrum([(CS[0], 3.0e6)], bins=700, width=5.0, seed=11)
    mix = _spectrum([(line, 1.5e6) for line in MIX], bins=900, width=8.0, seed=12)
    spec = ComptonEdgeFitSpec(
        datasets=(
            ComptonEdgeDataset(cs, CS, 1400.0, 3300.0),
            ComptonEdgeDataset(mix, MIX, 1000.0, 6800.0),
        ),
        statistic="poisson",
    )
    return replace(spec, **overrides)


@pytest.fixture(scope="module")
def joint_result() -> compton_module.ComptonEdgeFitResult:
    return fit_compton_edges(_joint_spec())


# ------------------------------------------------------------------ physics


@pytest.mark.parametrize(
    ("gamma", "edge"),
    [
        (511.0, 340.67),
        (661.657, 477.33),
        (1173.228, 963.42),
        (1274.537, 1061.70),
        (1332.492, 1118.10),
    ],
)
def test_compton_edge_energies(gamma: float, edge: float) -> None:
    assert compton_edge_kev(gamma) == pytest.approx(edge, abs=0.01)


def test_klein_nishina_shape_is_non_negative_zero_above_edge_and_finite_at_edge() -> None:
    edge = compton_edge_kev(661.657)
    t = np.linspace(-10.0, 700.0, 7101)
    shape = klein_nishina_electron_spectrum(t, 661.657)
    assert np.all(np.isfinite(shape))
    assert np.all(shape >= 0)
    assert np.all(shape[t > edge] == 0)
    assert np.all(shape[t < 0] == 0)
    at_edge = klein_nishina_electron_spectrum(np.asarray([edge, edge - 1e-9]), 661.657)
    assert np.all(np.isfinite(at_edge)) and at_edge[0] > 2.0
    assert at_edge[0] == pytest.approx(at_edge[1], rel=1e-6)
    assert klein_nishina_electron_spectrum(0.0, 661.657)[0] == pytest.approx(2.0)


def test_analytic_cell_weights_match_numeric_integration() -> None:
    energy = 1173.228
    edge = compton_edge_kev(energy)
    fine = np.linspace(0.0, edge, 400_001)
    density = klein_nishina_electron_spectrum(fine, energy)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(fine)))
    )
    cells = np.asarray([0.0, 100.0, 400.0, 900.0, edge])
    numeric = np.diff(np.interp(cells, fine, cumulative)) / cumulative[-1]
    analytic = KleinNishinaResponse(energy).weights(cells)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-10)


def test_cached_grid_weights_are_normalised_and_reused() -> None:
    first = compton_module._kn_grid_weights(661.657, 1.0, 480)
    second = compton_module._kn_grid_weights(661.657, 1.0, 480)
    assert first is second
    assert not first.flags.writeable
    assert float(np.sum(first)) == pytest.approx(1.0, abs=1e-12)
    assert np.all(first[478:] == 0)


def test_folding_with_vanishing_resolution_reproduces_binned_histogram() -> None:
    line = CS[0]
    edges = np.linspace(-80.5, 2600.5, 1342)
    folded = folded_line_response(
        edges, line, gain=TRUE_GAIN, offset=TRUE_OFFSET, resolution=(0.0, 0.0, 0.0)
    )
    cells = compton_module._grid_cells((line,), 1.0)
    centres, weights = compton_module._line_grid(line, 1.0, cells)
    histogram, _ = np.histogram(TRUE_GAIN * centres + TRUE_OFFSET, edges, weights=weights)
    np.testing.assert_allclose(folded, histogram, atol=1e-12)


def test_folded_area_is_conserved_inside_the_integration_range() -> None:
    edges = np.linspace(-2_000.0, 9_000.0, 2_201)
    for line in MIX:
        folded = folded_line_response(
            edges, line, gain=TRUE_GAIN, offset=TRUE_OFFSET, resolution=TRUE_RESOLUTION
        )
        assert float(np.sum(folded)) == pytest.approx(1.0, abs=1e-9)
        assert np.all(folded >= -1e-12)


def test_folded_response_agrees_with_independent_monte_carlo() -> None:
    rng = np.random.default_rng(7)
    energy = 661.657
    edge = compton_edge_kev(energy)
    fine = np.linspace(0.0, edge, 200_001)
    density = klein_nishina_electron_spectrum(fine, energy)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(fine)))
    )
    samples = 2_000_000
    t = np.interp(rng.random(samples) * cumulative[-1], cumulative, fine)
    a, b, c = TRUE_RESOLUTION
    sigma = TRUE_GAIN * t * np.sqrt(a**2 + b**2 / t + c**2 / t**2)
    observed = TRUE_GAIN * t + TRUE_OFFSET + sigma * rng.standard_normal(samples)
    edges = np.linspace(600.0, 2800.0, 111)
    counts, _ = np.histogram(observed, edges)
    expected = samples * folded_line_response(
        edges, CS[0], gain=TRUE_GAIN, offset=TRUE_OFFSET, resolution=TRUE_RESOLUTION
    )
    chi2 = float(np.sum((counts - expected) ** 2 / np.maximum(expected, 1.0)))
    assert chi2 / len(counts) < 1.6


# ------------------------------------------------------------- model engine


def test_tabulated_and_direct_paths_agree() -> None:
    spec = _joint_spec()
    model = compton_module._ComptonModel(spec)
    values = model.initial_values(lambda: False)
    values[:5] = (TRUE_GAIN, TRUE_OFFSET, *TRUE_RESOLUTION)
    model.path = "direct"
    model._cache_key = None
    direct = model.evaluate(values)
    for step in (2.0, 1.0):
        model.path = "tabulated"
        model.tabulation_step = step
        model._cache_key = None
        tabulated = model.evaluate(values)
        for exact, approximate in zip(direct.models, tabulated.models, strict=True):
            scale = float(np.max(exact))
            np.testing.assert_allclose(approximate, exact, atol=2e-5 * scale)


@pytest.mark.parametrize("path", ["direct", "tabulated"])
def test_analytic_jacobian_matches_finite_differences(path: str) -> None:
    spec = _joint_spec(resolution_terms=("a", "b", "c"))
    model = compton_module._ComptonModel(spec)
    values = model.initial_values(lambda: False)
    values[:5] = (TRUE_GAIN * 1.01, TRUE_OFFSET + 5.0, 0.05, 1.1, 2.0)
    model.path = cast(compton_module.EvaluationPath, path)
    model.tabulation_step = 1.0
    model._cache_key = None
    analytic = model.jacobian(values)
    numeric = np.zeros_like(analytic)
    for index in range(len(values)):
        if not model.layout.vary[index]:
            continue
        step = 1e-4 * max(abs(values[index]), 1e-2)
        upper, lower = values.copy(), values.copy()
        upper[index] += step
        lower[index] -= step
        numeric[:, index] = (model.residuals(upper) - model.residuals(lower)) / (2.0 * step)
    free = model.layout.vary
    scale = np.maximum(np.max(np.abs(numeric[:, free]), axis=0), 1e-12)
    error = np.max(np.abs(analytic[:, free] - numeric[:, free]), axis=0) / scale
    tolerance = 2e-4 if path == "direct" else 2e-2
    assert np.all(error < tolerance), dict(zip(np.asarray(model.layout.names)[free], error))


# --------------------------------------------------------- parameter recovery


def test_joint_fit_recovers_truth_within_reported_uncertainties(
    joint_result: compton_module.ComptonEdgeFitResult,
) -> None:
    result = joint_result
    assert result.success
    assert result.statistic == "poisson"
    assert result.energy_unit == "keVee"
    assert result.gain_stderr is not None and result.offset_stderr is not None
    assert abs(result.gain - TRUE_GAIN) < 3.0 * result.gain_stderr
    assert abs(result.offset - TRUE_OFFSET) < 3.0 * result.offset_stderr
    for term, truth in zip(result.resolution[:2], TRUE_RESOLUTION[:2], strict=True):
        assert term.varying and term.sigma_stderr is not None
        assert abs(term.sigma_value - truth) < 3.0 * term.sigma_stderr
        assert term.fwhm_value == pytest.approx(2.3548200450309493 * term.sigma_value)
    assert not result.resolution[2].varying and result.resolution[2].sigma_value == 0.0
    assert 0.8 < result.reduced_deviance < 1.25
    assert result.edge_span_kev == pytest.approx((340.6669, 1118.101), abs=1e-3)
    assert len(result.lines) == 6
    for line in result.lines:
        truth = TRUE_GAIN * line.edge_kev + TRUE_OFFSET
        assert line.edge_x_stderr is not None
        assert abs(line.edge_x - truth) < 3.0 * line.edge_x_stderr
        assert line.in_fit_range
        expected = np.sqrt(0.04**2 + 1.2**2 / line.edge_kev)
        assert line.sigma_over_energy == pytest.approx(expected, rel=0.1)
        assert line.fwhm_over_energy == pytest.approx(2.3548200450309493 * line.sigma_over_energy)
    covariance = result.gain_offset_covariance
    assert covariance.shape == (2, 2) and covariance[0, 1] < 0
    position, error = result.edge_position(477.334)
    assert position == pytest.approx(result.gain * 477.334 + result.offset)
    assert error == pytest.approx(result.lines[0].edge_x_stderr)


def test_joint_fit_reports_per_dataset_components(
    joint_result: compton_module.ComptonEdgeFitResult,
) -> None:
    first, second = joint_result.datasets
    assert set(first.components) == {"Cs-137 661.657 keV", "background"}
    assert len(second.components) == 6
    total = sum(second.components.values())
    np.testing.assert_allclose(total, second.best_fit, rtol=1e-10)
    np.testing.assert_allclose(second.residual, second.observed - second.best_fit)
    assert len(second.bin_edges) == len(second.x) + 1
    assert first.diagnostics is not None and second.diagnostics is None
    assert first.x[0] >= 1400.0 and first.x[-1] <= 3300.0


def test_single_dataset_with_fixed_zero_offset_recovers_gain() -> None:
    cs = _spectrum([(CS[0], 2.0e6)], bins=700, width=5.0, seed=21, offset=0.0)
    result = fit_compton_edges(
        ComptonEdgeFitSpec(
            datasets=(ComptonEdgeDataset(cs, CS, 1500.0, 3300.0),),
            offset_free=False,
        )
    )
    assert result.success
    assert result.offset == 0.0 and result.offset_stderr is None
    assert result.gain_stderr is not None
    assert abs(result.gain - TRUE_GAIN) < 3.0 * result.gain_stderr
    assert not any("degenerate" in warning for warning in result.warnings)


def test_explicit_starting_values_and_linear_background() -> None:
    cs = _spectrum([(CS[0], 2.0e6)], bins=700, width=5.0, seed=31)
    mix = _spectrum([(line, 1.0e6) for line in MIX], bins=900, width=8.0, seed=32)
    result = fit_compton_edges(
        ComptonEdgeFitSpec(
            datasets=(
                ComptonEdgeDataset(cs, CS, 1400.0, 3300.0, background="linear"),
                ComptonEdgeDataset(mix, MIX, 1000.0, 6800.0, background="constant"),
            ),
            initial_gain=4.6,
            initial_offset=0.0,
        )
    )
    assert result.success
    assert result.gain == pytest.approx(TRUE_GAIN, rel=0.01)
    names = {parameter.name for parameter in result.parameters}
    assert {"d0_bg_intercept", "d0_bg_slope", "d0_bg_x_ref", "d1_bg_c"} <= names


def test_linked_amplitudes_follow_emission_yields() -> None:
    result = fit_compton_edges(_joint_spec(amplitude_mode="linked"))
    names = [parameter.name for parameter in result.parameters]
    assert "d1_Co_60_scale" in names and "d1_Na_22_scale" in names
    assert not any("Co_60_1173" in name for name in names)
    amplitudes = {line.label: line.amplitude for line in result.lines if line.dataset_index == 1}
    ratio = amplitudes["Co-60 1332.49 keV"] / amplitudes["Co-60 1173.23 keV"]
    assert ratio == pytest.approx(99.98 / 99.85)
    ratio = amplitudes["Na-22 511 keV"] / amplitudes["Na-22 1274.54 keV"]
    assert ratio == pytest.approx(180.76 / 99.94)


def test_weighted_statistic_is_used_for_derived_non_count_spectra() -> None:
    cs = scale_spectrum(_spectrum([(CS[0], 2.0e6)], bins=700, width=5.0, seed=41), 0.5)
    result = fit_compton_edges(
        ComptonEdgeFitSpec(
            datasets=(ComptonEdgeDataset(cs, CS, 1400.0, 3300.0),),
            offset_free=False,
            initial_offset=TRUE_OFFSET,
        )
    )
    assert result.statistic == "weighted"
    assert result.gain == pytest.approx(TRUE_GAIN, rel=0.01)
    with pytest.raises(ValueError, match="Poisson"):
        fit_compton_edges(
            ComptonEdgeFitSpec(
                datasets=(ComptonEdgeDataset(cs, CS, 1400.0, 3300.0),), statistic="poisson"
            )
        )


# ---------------------------------------------------------------- edge cases


def test_dataset_and_spec_validation() -> None:
    spectrum = _spectrum([(CS[0], 1.0e5)], bins=200, width=10.0, seed=1)
    with pytest.raises(ValueError, match="at least one gamma line"):
        ComptonEdgeDataset(spectrum, ())
    with pytest.raises(ValueError, match="background"):
        ComptonEdgeDataset(spectrum, CS, background=cast(ComptonBackground, "compton"))
    with pytest.raises(ValueError, match="fit_min < fit_max"):
        ComptonEdgeDataset(spectrum, CS, 500.0, 400.0)
    with pytest.raises(ValueError, match="at least one dataset"):
        ComptonEdgeFitSpec(datasets=())
    dataset = ComptonEdgeDataset(spectrum, CS)
    with pytest.raises(ValueError, match="resolution terms"):
        ComptonEdgeFitSpec(
            datasets=(dataset,),
            resolution_terms=cast(tuple[ResolutionTermName, ...], ("d",)),
        )
    with pytest.raises(ValueError, match="grid step"):
        ComptonEdgeFitSpec(datasets=(dataset,), grid_step_kev=2.0)
    with pytest.raises(ValueError, match="1-4"):
        ComptonEdgeFitSpec(datasets=(dataset,), dietze_klein_detector=5)
    with pytest.raises(ValueError, match="positive"):
        ComptonLine(-1.0)


def test_empty_and_too_narrow_fit_ranges_are_rejected() -> None:
    spectrum = _spectrum([(CS[0], 1.0e5)], bins=200, width=10.0, seed=2)
    with pytest.raises(ValueError, match="contains no bins"):
        fit_compton_edges(
            ComptonEdgeFitSpec(datasets=(ComptonEdgeDataset(spectrum, CS, 5000.0, 6000.0),))
        )
    with pytest.raises(ValueError, match="too few bins"):
        fit_compton_edges(
            ComptonEdgeFitSpec(datasets=(ComptonEdgeDataset(spectrum, CS, 1000.0, 1050.0),))
        )
    flat = Spectrum.create(label="flat", counts=np.full(200, 5.0), source="derived")
    with pytest.raises(ValueError, match="no spectral structure"):
        fit_compton_edges(ComptonEdgeFitSpec(datasets=(ComptonEdgeDataset(flat, CS),)))


def test_cancellation_raises_the_peak_fit_cancellation_type() -> None:
    with pytest.raises(ComptonEdgeFitCancelledError) as caught:
        fit_compton_edges(_joint_spec(), cancelled=lambda: True)
    assert isinstance(caught.value, PeakFitCancelledError)
    calls = 0

    def later() -> bool:
        nonlocal calls
        calls += 1
        return calls > 3

    with pytest.raises(ComptonEdgeFitCancelledError):
        fit_compton_edges(_joint_spec(initial_gain=4.8), cancelled=later)


def test_progress_reports_every_stage_in_order() -> None:
    updates: list[ComptonEdgeFitProgress] = []
    result = fit_compton_edges(_joint_spec(), progress=updates.append)

    stages = [update.stage for update in updates]
    assert stages[0] == "prepare" and stages[-1] == "finalize"
    # Stages never interleave: each one appears as a single contiguous run.
    runs = [stage for index, stage in enumerate(stages) if index == 0 or stage != stages[index - 1]]
    assert runs == ["prepare", "gain_scan", "optimize", "finalize"]
    scan = [update for update in updates if update.stage == "gain_scan"]
    assert [update.done for update in scan] == list(range(len(scan)))
    assert all(update.total == len(scan) - 1 for update in scan)
    optimize = [update for update in updates if update.stage == "optimize"]
    assert all(update.total is None for update in optimize)
    # The live count matches the evaluations shown after the fit.
    assert optimize[-1].done == result.evaluations


def test_progress_skips_gain_scan_when_starting_gain_is_given() -> None:
    updates: list[ComptonEdgeFitProgress] = []
    fit_compton_edges(_joint_spec(initial_gain=4.8), progress=updates.append)
    assert "gain_scan" not in {update.stage for update in updates}


def test_offset_free_fit_with_one_edge_energy_warns_about_degeneracy() -> None:
    cs = _spectrum([(CS[0], 2.0e6)], bins=700, width=5.0, seed=51)
    result = fit_compton_edges(
        ComptonEdgeFitSpec(datasets=(ComptonEdgeDataset(cs, CS, 1400.0, 3300.0),))
    )
    assert any("degenerate" in warning for warning in result.warnings)


def test_edge_outside_fit_range_is_reported() -> None:
    mix = _spectrum([(line, 1.0e6) for line in MIX], bins=900, width=8.0, seed=61)
    result = fit_compton_edges(
        ComptonEdgeFitSpec(
            datasets=(ComptonEdgeDataset(mix, MIX, 1000.0, 5000.0),),
            initial_gain=TRUE_GAIN,
            initial_offset=TRUE_OFFSET,
        )
    )
    outside = [line for line in result.lines if not line.in_fit_range]
    assert {line.label for line in outside} == {"Co-60 1332.49 keV", "Na-22 1274.54 keV"}
    assert sum("outside the fit range" in warning for warning in result.warnings) == 2


# -------------------------------------------------------------- calibration


def test_calibration_points_feed_energy_calibration_in_kevee(
    joint_result: compton_module.ComptonEdgeFitResult,
) -> None:
    points = compton_edge_calibration_points(joint_result)
    assert [point.energy_kev for point in points] == pytest.approx(
        [340.6669, 477.334, 963.4195, 1061.7032, 1118.101], abs=1e-3
    )
    assert all(point.source == "compton-edge fit" for point in points)
    assert points[1].label == "Compton edge 661.657 keV"
    calibration = fit_energy_calibration(points, energy_unit=COMPTON_EDGE_ENERGY_UNIT)
    assert calibration.energy_unit == "keVee"
    assert calibration.energy_quantity == "Light output"
    assert calibration.coefficients_kev[1] == pytest.approx(1.0 / joint_result.gain)
    assert calibration.energy(joint_result.offset) == pytest.approx(0.0, abs=1e-6)
    assert calibration.rms_residual_kev < 1e-6
    restored = EnergyCalibration.from_dict(json.loads(json.dumps(calibration.to_dict())))
    assert restored.energy_unit == "keVee"
    assert restored.coefficients_kev == pytest.approx(calibration.coefficients_kev)


def test_fixed_offset_adds_an_explicit_zero_light_point() -> None:
    cs = _spectrum([(CS[0], 2.0e6)], bins=700, width=5.0, seed=71, offset=0.0)
    result = fit_compton_edges(
        ComptonEdgeFitSpec(datasets=(ComptonEdgeDataset(cs, CS, 1500.0, 3300.0),),
                           offset_free=False)
    )
    points = compton_edge_calibration_points(result)
    assert len(points) == 2
    assert points[0].channel == 0.0 and points[0].energy_kev == 0.0
    assert "fixed" in points[0].source
    calibration = fit_energy_calibration(points, energy_unit="keVee")
    assert calibration.coefficients_kev[1] == pytest.approx(1.0 / result.gain)


def test_old_calibration_documents_without_unit_still_load_as_kev() -> None:
    legacy = {
        "model": "linear",
        "coefficients_kev": [0.0, 0.5],
        "points": [
            {"channel": 100.0, "energy_kev": 50.0},
            {"channel": 1000.0, "energy_kev": 500.0},
        ],
        "fingerprint": {"binning": 0},
        "updated_utc": "2026-01-01T00:00:00+00:00",
    }
    calibration = EnergyCalibration.from_dict(legacy)
    assert calibration.energy_unit == "keV"
    assert calibration.energy_quantity == "Energy"
    assert calibration.to_dict()["energy_unit"] == "keV"
    with pytest.raises(ValueError, match="unit"):
        EnergyCalibration.from_dict({**legacy, "energy_unit": "MeVee"})
    with pytest.raises(ValueError, match="unit"):
        fit_energy_calibration(
            EnergyCalibration.from_dict(legacy).points,
            energy_unit=cast(EnergyUnit, "eV"),
        )


def test_extrapolation_warning_names_the_constrained_span() -> None:
    assert extrapolation_warning((340.7, 1118.1), 400.0, 1000.0) is None
    message = extrapolation_warning((340.7, 1118.1), 50.0, 1500.0)
    assert message is not None and "340.7" in message and "extrapolated" in message


# ------------------------------------------------------------- diagnostics


def _shape_truth(resolution: tuple[float, float, float]) -> tuple[float, float, float]:
    edges = np.linspace(1000.0, 3500.0, 25_001)
    model = folded_line_response(
        edges, CS[0], gain=TRUE_GAIN, offset=0.0, resolution=resolution
    ) / np.diff(edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    peak = int(np.argmax(model))
    half = model[peak] / 2.0
    crossing = peak + int(np.flatnonzero(model[peak:] <= half)[0])
    falling = slice(crossing, crossing - 2, -1)
    l_half = float(np.interp(half, model[falling], centres[falling]))
    edge = TRUE_GAIN * CS[0].edge_kev
    height = float(np.interp(edge, centres, model)) / model[peak]
    return float(centres[peak]), l_half, height


def test_diagnostics_recover_known_single_scatter_shape_points() -> None:
    resolution = (0.05, 1.0, 0.0)
    l_max, l_half, height = _shape_truth(resolution)
    spectrum = _spectrum(
        [(CS[0], 4.0e7)], bins=2000, width=2.0, seed=81, offset=0.0, resolution=resolution,
        background=0.0,
    )
    edge = TRUE_GAIN * CS[0].edge_kev
    sigma = edge * np.sqrt(0.05**2 + 1.0 / CS[0].edge_kev)
    diagnostics = compton_shape_diagnostics(
        spectrum.x, spectrum.counts, edge_x=edge, edge_sigma_x=sigma,
        line_energy_kev=661.657,
    )
    assert diagnostics.l_max / diagnostics.l_half == pytest.approx(l_max / l_half, abs=0.004)
    assert diagnostics.l_max == pytest.approx(l_max, rel=0.004)
    assert diagnostics.l_half == pytest.approx(l_half, rel=0.002)
    assert diagnostics.edge_height_fraction == pytest.approx(height, abs=0.02)
    assert diagnostics.dk_resolution_fwhm == pytest.approx(
        1.5 * (diagnostics.l_half - diagnostics.l_max) / diagnostics.l_half
    )
    assert diagnostics.dietze_klein is None


def test_dietze_klein_estimate_interpolates_table_three() -> None:
    l_max, l_half = 7555.0, 8482.0
    x = np.linspace(4500.0, 12500.0, 801)
    rising = 1000.0 * (1.0 - (l_max - x) / 6000.0)
    falling = 1000.0 * (1.0 - 0.5 * (x - l_max) / (l_half - l_max))
    counts = np.clip(np.where(x < l_max, rising, falling), 5.0, None)
    diagnostics = compton_shape_diagnostics(
        x, counts, edge_x=8578.0, edge_sigma_x=705.0, line_energy_kev=661.657,
        dietze_klein_detector=2,
    )
    resolution = 1.5 * (diagnostics.l_half - diagnostics.l_max) / diagnostics.l_half
    block = dietze_klein_table.block_for(661.657, 2)
    assert block is not None and block.detector == 2
    estimate = diagnostics.dietze_klein
    assert estimate is not None
    expected_lmax = np.interp(100 * resolution, block.resolution_fwhm_percent,
                              block.lmax_over_edge)
    assert estimate.lmax_over_edge == pytest.approx(expected_lmax)
    assert estimate.edge_x == pytest.approx(
        0.5 * (diagnostics.l_max / estimate.lmax_over_edge
               + diagnostics.l_half / estimate.lhalf_over_edge)
    )
    assert "Table 3" in estimate.citation
    # Values from the published EJ-309 showcase: 16.4 % FWHM and an edge near 8200.
    published = 1.5 * (l_half - l_max) / l_half
    assert published == pytest.approx(0.164, abs=0.001)
    lmax_ratio = np.interp(100 * published, block.resolution_fwhm_percent, block.lmax_over_edge)
    lhalf_ratio = np.interp(100 * published, block.resolution_fwhm_percent, block.lhalf_over_edge)
    assert 0.5 * (l_max / lmax_ratio + l_half / lhalf_ratio) == pytest.approx(8200.0, abs=10.0)
    assert dietze_klein_table.block_for(1274.537, 2) is None


def _multiple_scatter_response() -> SimulatedResponse:
    energy = 661.657
    edge = compton_edge_kev(energy)
    centres = np.arange(0.5, energy, 1.0)
    single = klein_nishina_electron_spectrum(centres, energy)
    single /= single.sum()
    tail = ((centres > edge - 40.0) & (centres < energy)).astype(float)
    tail /= tail.sum()
    return SimulatedResponse(centres, 0.7 * single + 0.3 * tail, source="synthetic.csv")


def test_multiple_scatter_tail_biases_kn_and_simulated_response_removes_bias() -> None:
    response = _multiple_scatter_response()
    simulated_line = CS[0].with_response(response)
    spectrum = _spectrum(
        [(simulated_line, 4.0e6)], bins=700, width=5.0, seed=91, offset=0.0, background=20.0
    )
    single = fit_compton_edges(
        ComptonEdgeFitSpec(
            datasets=(ComptonEdgeDataset(spectrum, CS, 1500.0, 3300.0),),
            offset_free=False,
        )
    )
    assert single.gain > TRUE_GAIN * 1.01
    assert any("Compton maximum" in warning for warning in single.warnings)
    diagnostics = single.datasets[0].diagnostics
    assert diagnostics is not None and diagnostics.edge_height_fraction is not None
    assert not 0.65 <= diagnostics.edge_height_fraction <= 0.90
    simulated = fit_compton_edges(
        ComptonEdgeFitSpec(
            datasets=(ComptonEdgeDataset(spectrum, (simulated_line,), 1500.0, 3300.0),),
            offset_free=False,
        )
    )
    assert simulated.gain_stderr is not None
    assert abs(simulated.gain - TRUE_GAIN) < 3.0 * simulated.gain_stderr
    assert abs(simulated.gain - TRUE_GAIN) < abs(single.gain - TRUE_GAIN) / 5.0
    assert simulated.lines[0].response.startswith("simulated response (synthetic.csv)")


def test_reduced_deviance_warning_mentions_multiple_scattering() -> None:
    response = _multiple_scatter_response()
    spectrum = _spectrum(
        [(CS[0].with_response(response), 4.0e7)], bins=700, width=5.0, seed=92, offset=0.0,
    )
    result = fit_compton_edges(
        ComptonEdgeFitSpec(
            datasets=(ComptonEdgeDataset(spectrum, CS, 1500.0, 3300.0),), offset_free=False
        )
    )
    assert result.reduced_deviance > 3.0
    assert any("multiple Compton scattering" in warning for warning in result.warnings)


# --------------------------------------------------------- lines and inputs


def test_nuclide_presets_use_catalogue_lines_with_energy_and_yield_filters() -> None:
    sodium = nuclide_compton_lines("na22")
    assert [line.energy_kev for line in sodium] == pytest.approx([511.0, 1274.537])
    assert sodium[0].emission_probability_percent == pytest.approx(180.76, abs=0.01)
    assert all(line.nuclide == "Na-22" for line in sodium)
    barium = nuclide_compton_lines("Ba-133")
    assert all(line.energy_kev >= 150.0 for line in barium)
    assert 356.013 in [round(line.energy_kev, 3) for line in barium]


def test_custom_energy_parsing() -> None:
    assert parse_custom_energies("661.657, 1173.2;1332.5  511") == (
        661.657, 1173.2, 1332.5, 511.0,
    )
    assert parse_custom_energies("  ") == ()
    with pytest.raises(ValueError):
        parse_custom_energies("662, abc")
    with pytest.raises(ValueError):
        parse_custom_energies("-5")


def test_simulated_response_loader_reads_csv_with_header(tmp_path: Path) -> None:
    path = tmp_path / "cs137_response.csv"
    path.write_text(
        "# Geant4 deposited energy, 2 keV bins\nenergy_kev,weight\n"
        + "\n".join(f"{energy:.1f},{weight:.6g}" for energy, weight in
                    zip(np.arange(1.0, 700.0, 2.0), np.linspace(1.0, 2.0, 350), strict=True)),
        encoding="utf-8",
    )
    response = load_simulated_response(path)
    assert response.max_energy_kev == pytest.approx(700.0)
    assert response.label == "simulated response (cs137_response.csv)"
    weights = response.weights(np.arange(0.0, 702.0, 1.0))
    assert float(np.sum(weights)) == pytest.approx(1.0)
    assert np.all(weights >= 0)
    assert isinstance(response, compton_module.ResponseShape)
    bad = tmp_path / "bad.csv"
    bad.write_text("energy_kev,weight\n10,1\n5,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="increasing"):
        load_simulated_response(bad)
    negative = tmp_path / "negative.csv"
    negative.write_text("1,1\n2,-1\n3,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-negative"):
        load_simulated_response(negative)


# ---------------------------------------------------------------- reporting


def test_summary_rows_and_exports(
    tmp_path: Path,
    joint_result: compton_module.ComptonEdgeFitResult,
) -> None:
    rows = compton_edge_summary_rows(joint_result)
    quantities = [row.quantity for row in rows]
    assert quantities[:2] == ["gain g", "offset o"]
    assert "a (σ)" in quantities and "α (FWHM)" in quantities and "β (FWHM)" in quantities
    assert "[1] 661.657 keV L_max" in quantities
    assert "reduced deviance" in quantities
    spec = _joint_spec()
    json_path = tmp_path / "edges.json"
    export_compton_edge_result(json_path, spec=spec, result=joint_result)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["kind"] == "nlab_compton_edge_fit"
    assert document["energy_unit"] == "keVee"
    assert document["calibration"]["gain_x_per_kevee"] == pytest.approx(joint_result.gain)
    assert len(document["lines"]) == 6
    assert document["resolution"][1]["fwhm_value"] == pytest.approx(
        2.3548200450309493 * document["resolution"][1]["sigma_value"]
    )
    assert any("Klein" in reference for reference in document["references"])
    csv_path = tmp_path / "edges.csv"
    export_compton_edge_result(csv_path, spec=spec, result=joint_result)
    text = csv_path.read_text(encoding="utf-8")
    assert text.startswith("section,quantity,value,stderr,unit,note")
    with pytest.raises(FileExistsError):
        export_compton_edge_result(csv_path, spec=spec, result=joint_result)
    with pytest.raises(ValueError, match=".json or .csv"):
        export_compton_edge_result(tmp_path / "edges.txt", spec=spec, result=joint_result)


def test_csv_export_labels_kevee_light_output_and_round_trips(tmp_path: Path) -> None:
    from nlab.analysis.energy_calibration import CalibrationPoint
    from nlab.analysis.spectrum_io import export_spectrum_csv, load_spectra

    calibration = fit_energy_calibration(
        (CalibrationPoint(100.0, 340.0), CalibrationPoint(300.0, 1118.0)),
        energy_unit="keVee",
    )
    channels = np.arange(400, dtype=np.float64)
    spectrum = Spectrum.create(
        label="EJ-309",
        counts=channels,
        source="mca",
        metadata={"energy_calibration": calibration.to_dict()},
        energy_kev=np.asarray(calibration.energy(channels)),
    )
    assert spectrum.energy_unit == "keVee"
    path = tmp_path / "ej309.csv"
    export_spectrum_csv(path, spectrum)
    assert "# energy_unit=keVee" in path.read_text(encoding="utf-8")
    assert load_spectra(path)[0].energy_unit == "keVee"
    plain = Spectrum.create(label="NaI", counts=channels, source="mca")
    plain_path = tmp_path / "nai.csv"
    export_spectrum_csv(plain_path, plain)
    assert "energy_unit" not in plain_path.read_text(encoding="utf-8")
    assert plain.energy_unit == "keV"


def test_gain_and_offset_bounds_from_the_start_reach_the_optimiser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scipy.optimize

    captured: dict[str, Any] = {}
    real_least_squares = scipy.optimize.least_squares

    def spy(fun: Any, x0: Any, **kwargs: Any) -> Any:
        captured["bounds"] = kwargs["bounds"]
        return real_least_squares(fun, x0, **kwargs)

    monkeypatch.setattr(scipy.optimize, "least_squares", spy)
    result = fit_compton_edges(_joint_spec(initial_gain=4.8, initial_offset=0.0))
    lower, upper = captured["bounds"]
    assert (lower[0], upper[0]) == pytest.approx((1.6, 14.4))
    assert lower[1] < 0.0 < upper[1] and np.isfinite(lower[1]) and np.isfinite(upper[1])
    gain = next(parameter for parameter in result.parameters if parameter.name == "gain")
    assert (gain.minimum, gain.maximum) == pytest.approx((1.6, 14.4))
    assert result.gain == pytest.approx(TRUE_GAIN, rel=0.01)


def test_linked_mode_starts_near_the_yield_scaled_amplitudes() -> None:
    scale = 1.5e6
    mix = _spectrum(
        [(line, scale * (line.emission_probability_percent or 0.0) / 100.0) for line in MIX],
        bins=900,
        width=8.0,
        seed=102,
    )
    spec = ComptonEdgeFitSpec(
        datasets=(ComptonEdgeDataset(mix, MIX, 1000.0, 6800.0),),
        amplitude_mode="linked",
        initial_gain=TRUE_GAIN,
        initial_offset=TRUE_OFFSET,
        initial_resolution=(0.04, 1.2, 0.0),
    )
    model = compton_module._ComptonModel(spec)
    values = model.initial_values(lambda: False)
    starts = dict(zip(model.layout.names, values, strict=True))
    assert starts["d0_Co_60_scale"] == pytest.approx(scale, rel=0.1)
    assert starts["d0_Na_22_scale"] == pytest.approx(scale, rel=0.1)


def test_a_line_without_influence_loses_only_its_own_uncertainty() -> None:
    cs = _spectrum([(CS[0], 2.0e6)], bins=700, width=5.0, seed=103, offset=0.0)
    # A 100 keV line has its whole continuum (edge at x ≈ 140) far below the window.
    lines = CS + (ComptonLine(100.0, label="far line"),)
    result = fit_compton_edges(
        ComptonEdgeFitSpec(
            datasets=(ComptonEdgeDataset(cs, lines, 1500.0, 3300.0),),
            offset_free=False,
            initial_gain=TRUE_GAIN,
        )
    )
    assert result.gain_stderr is not None
    far = next(line for line in result.lines if line.label == "far line")
    assert far.amplitude_stderr is None and not far.in_fit_range
    assert any("could not be estimated for: d0_far_line_amplitude" in warning
               for warning in result.warnings)
