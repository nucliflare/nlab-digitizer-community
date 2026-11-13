"""PSD double-Gaussian figure-of-merit fitting tests."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from nlab.analysis.psd_fom import fit_psd_fom_lm, fit_psd_fom_separate_lm
from nlab.views.offline_psd_plot import OfflinePsdPlot


def _double_gaussian_projection() -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(42)
    ratio = np.linspace(-0.2, 0.8, 256)
    expectation = (
        8.0
        + 260.0 * np.exp(-0.5 * np.square((ratio - 0.12) / 0.035))
        + 180.0 * np.exp(-0.5 * np.square((ratio - 0.48) / 0.055))
    )
    return ratio, random.poisson(expectation)


def test_lm_double_gaussian_recovers_conventional_fom() -> None:
    ratio, counts = _double_gaussian_projection()

    fit = fit_psd_fom_lm(
        ratio,
        counts,
        roi=(-0.05, 0.65),
        centroid_seeds=(0.1, 0.5),
    )

    expected_fom = (0.48 - 0.12) / (2.354820045 * (0.035 + 0.055))
    assert fit.success
    assert fit.warnings == ()
    assert fit.use_background and fit.varied_parameters == 7
    assert fit.centroids == pytest.approx((0.12, 0.48), abs=0.005)
    assert fit.fwhms == pytest.approx((0.035 * 2.35482, 0.055 * 2.35482), rel=0.08)
    assert fit.fom == pytest.approx(expected_fom, rel=0.08)
    assert fit.reduced_deviance == pytest.approx(1.0, abs=0.35)
    assert not fit.best_fit.flags.writeable


def test_zero_background_well_separated_peaks_converge_to_independent_widths() -> None:
    ratio = np.linspace(-1.0, 1.0, 256)
    expectation = 210.0 * np.exp(-0.5 * np.square((ratio - 0.225) / 0.025)) + 375.0 * np.exp(
        -0.5 * np.square((ratio - 0.808) / 0.019)
    )
    counts = np.random.default_rng(108).poisson(expectation)

    fit = fit_psd_fom_lm(
        ratio,
        counts,
        roi=(-0.1, 0.98),
        centroid_seeds=(0.22, 0.81),
    )

    assert fit.success
    assert fit.centroids == pytest.approx((0.225, 0.808), abs=0.004)
    assert fit.fwhms == pytest.approx((0.05887, 0.04474), rel=0.12)
    assert fit.fwhms[0] != pytest.approx(fit.fwhms[1])
    assert fit.fom > 5.0
    assert fit.reduced_deviance < 2.0


def test_lm_fit_rejects_centroid_estimates_outside_roi() -> None:
    ratio, counts = _double_gaussian_projection()

    with pytest.raises(ValueError, match="inside the fit ROI"):
        fit_psd_fom_lm(
            ratio,
            counts,
            roi=(0.0, 0.7),
            centroid_seeds=(-0.1, 0.5),
        )


def test_sparse_fit_is_returned_with_poor_convergence_warning() -> None:
    ratio = np.linspace(-0.2, 0.8, 128)
    expectation = (
        0.1
        + 3.0 * np.exp(-0.5 * np.square((ratio - 0.12) / 0.035))
        + 2.0 * np.exp(-0.5 * np.square((ratio - 0.48) / 0.055))
    )
    counts = np.random.default_rng(8).poisson(expectation)

    fit = fit_psd_fom_lm(
        ratio,
        counts,
        roi=(-0.05, 0.65),
        centroid_seeds=(0.1, 0.5),
    )

    assert fit.poor_convergence
    assert "fit ROI contains fewer than 100 counts" in fit.warnings


def test_offline_plot_fits_on_button_and_toggles_guides(qtbot: QtBot) -> None:
    ratio_edges = np.linspace(-0.2, 0.8, 257)
    ratio = (ratio_edges[:-1] + ratio_edges[1:]) * 0.5
    expectation = (
        8.0
        + 260.0 * np.exp(-0.5 * np.square((ratio - 0.12) / 0.035))
        + 180.0 * np.exp(-0.5 * np.square((ratio - 0.48) / 0.055))
    )
    matrix = np.random.default_rng(12).poisson(expectation, size=(32, 256)).astype(np.uint64)
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    plot.resize(1100, 650)
    plot.show()
    plot.set_data(
        matrix,
        energy_range=(0.0, 32.0),
        ratio_range=(-0.2, 0.8),
    )
    plot.energy_region.setRegion((0.0, 32.0))
    plot._render_projections()
    plot.fom_widget.fit_region.setRegion((-0.05, 0.65))
    plot.fom_widget.centroid_lines[0].setPos(0.1)
    plot.fom_widget.centroid_lines[1].setPos(0.5)
    assert not plot.fom_widget.copy_button.isEnabled()

    QTest.mouseClick(plot.fom_widget.fit_button, Qt.MouseButton.LeftButton)

    assert plot.fom_widget.last_fit is not None
    assert plot.fom_widget.last_fit.fom == pytest.approx(1.7, rel=0.12)
    assert plot.fom_widget.fom_label.text() != "—"
    assert "converged" in plot.fom_widget.status_label.text()
    assert plot.fom_widget.copy_button.isEnabled()

    QTest.mouseClick(plot.fom_widget.copy_button, Qt.MouseButton.LeftButton)
    rows = [line.split("\t") for line in QApplication.clipboard().text().splitlines()]
    assert rows[0] == [
        "Population",
        "Centroid",
        "Centroid uncertainty",
        "FWHM",
        "FWHM uncertainty",
        "FOM",
        "Reduced Poisson deviance",
        "Fit status",
        "Evaluations",
        "Warnings",
    ]
    assert [rows[1][0], rows[2][0]] == ["1", "2"]
    assert float(rows[1][1]) == pytest.approx(plot.fom_widget.last_fit.centroids[0])
    assert float(rows[2][3]) == pytest.approx(plot.fom_widget.last_fit.fwhms[1])
    assert float(rows[1][5]) == pytest.approx(plot.fom_widget.last_fit.fom)

    plot.fom_widget.guides_checkbox.setChecked(False)
    assert not plot.fom_widget.fit_region.isVisible()
    assert all(not line.isVisible() for line in plot.fom_widget.centroid_lines)

    first_line = plot.fom_widget.centroid_lines[0]
    first_line.sigPositionChangeFinished.emit(first_line)
    assert plot.fom_widget.last_fit is None
    assert not plot.fom_widget.copy_button.isEnabled()


def test_offline_plot_report_state_is_a_read_only_snapshot(qtbot: QtBot) -> None:
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    matrix = np.arange(32 * 64, dtype=np.uint64).reshape(32, 64)
    plot.set_data(
        matrix,
        energy_range=(0.0, 32.0),
        ratio_range=(-0.2, 0.8),
        energy_label="Integrated charge",
    )
    plot.cut_spin.setValue(0.25)
    plot.energy_region.setRegion((4.0, 20.0))
    plot.set_energy_log_y(True)

    state = plot.report_state()

    assert not state.matrix.flags.writeable
    np.testing.assert_array_equal(state.matrix, matrix)
    assert state.energy_label == "Integrated charge"
    assert state.cut == pytest.approx(0.25)
    assert state.energy_roi == pytest.approx((4.0, 20.0))
    assert state.matrix_scale == "log"
    assert state.energy_log_y is True


def test_independent_fits_ignore_gap_and_recover_distinct_local_backgrounds() -> None:
    # Generated Poisson counts, with different baselines in the two fit windows.
    ratio = np.linspace(0, 1, 1024)
    baseline = np.where(ratio < 0.5, 3.0, 17.0)
    expected = baseline + 300 * np.exp(-0.5 * ((ratio - 0.2) / 0.016) ** 2)
    expected += 120 * np.exp(-0.5 * ((ratio - 0.8) / 0.027) ** 2)
    counts = np.random.default_rng(123).poisson(expected)
    rois = ((0.11, 0.29), (0.65, 0.95))
    fit = fit_psd_fom_separate_lm(ratio, counts, rois=rois, centroid_seeds=(0.19, 0.81))
    assert fit.success
    assert fit.fit_mode == "separate"
    assert fit.use_background and fit.varied_parameters == 8
    assert fit.fit_regions == rois
    assert fit.centroids == pytest.approx((0.2, 0.8), abs=0.002)
    assert fit.fwhms == pytest.approx(np.asarray((0.016, 0.027)) * 2.35482, rel=0.06)
    assert fit.fom == pytest.approx(0.6 / (2.35482 * 0.043), rel=0.06)
    assert fit.reduced_deviance == pytest.approx(1, abs=0.3)
    assert np.mean(fit.background_component[fit.x < 0.5]) == pytest.approx(3, abs=0.5)
    assert np.mean(fit.background_component[fit.x > 0.5]) == pytest.approx(17, abs=1.5)
    assert not np.any((fit.x > 0.29) & (fit.x < 0.65))
    assert not fit.best_fit.flags.writeable
    contaminated = counts.copy()
    contaminated[(ratio > 0.3) & (ratio < 0.64)] = 1_000_000
    refit = fit_psd_fom_separate_lm(
        ratio, contaminated, rois=rois, centroid_seeds=(0.19, 0.81),
    )
    np.testing.assert_array_equal(refit.best_fit, fit.best_fit)
    np.testing.assert_array_equal(refit.observed, fit.observed)


@pytest.mark.parametrize(
    "rois,seeds,message",
    [
        (((0.0, 0.4), (0.3, 0.7)), (0.2, 0.5), "non-overlapping"),
        (((0.0, 0.3), (0.4, 0.7)), (0.35, 0.5), "inside its fit ROI"),
        (((0.0, 0.01), (0.4, 0.7)), (0.005, 0.5), "at least 12 bins"),
        (((0.0, np.nan), (0.4, 0.7)), (0.2, 0.5), "finite"),
    ],
)
def test_separate_fit_rejects_invalid_peak_windows(
    rois: tuple[tuple[float, float], tuple[float, float]],
    seeds: tuple[float, float], message: str,
) -> None:
    ratio, counts = _double_gaussian_projection()
    with pytest.raises(ValueError, match=message):
        fit_psd_fom_separate_lm(ratio, counts, rois=rois, centroid_seeds=seeds)


def test_separate_fit_reports_sparse_population_warning() -> None:
    ratio = np.linspace(0, 1, 512)
    counts = np.random.default_rng(7).poisson(
        0.01 + 2 * np.exp(-0.5 * ((ratio - 0.2) / 0.018) ** 2)
        + 200 * np.exp(-0.5 * ((ratio - 0.8) / 0.025) ** 2)
    )
    fit = fit_psd_fom_separate_lm(
        ratio, counts, rois=((0.1, 0.3), (0.65, 0.95)), centroid_seeds=(0.2, 0.8),
    )
    assert fit.poor_convergence
    assert any("Population 1" in warning and "100 counts" in warning for warning in fit.warnings)


def test_separate_guides_fit_and_preserve_both_ranges(qtbot: QtBot) -> None:
    ratio, counts = _double_gaussian_projection()
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    widget = plot.fom_widget
    widget.set_ratio_range((-0.2, 0.8))
    widget.set_projection(ratio, counts)
    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("separate"))
    widget.peak_regions[0].setRegion((-0.02, 0.26))
    widget.peak_regions[1].setRegion((0.32, 0.66))
    widget.centroid_lines[0].setPos(0.1)
    widget.centroid_lines[1].setPos(0.5)
    assert not widget.fit_region.isVisible()
    assert all(region.isVisible() for region in widget.peak_regions)
    fit = widget.fit_current_projection()
    assert fit is not None and fit.success and fit.fit_mode == "separate"
    assert fit.centroids == pytest.approx((0.12, 0.48), abs=0.006)
    _, plotted_ratio = widget._fit_curve.getData()
    assert np.any(np.isnan(plotted_ratio))  # Never draw a fit through the excluded gap.
    widget.set_ratio_range((0, 0.6))
    assert widget.peak_regions[0].getRegion() == pytest.approx((0, 0.26))
    assert widget.peak_regions[1].getRegion() == pytest.approx((0.32, 0.6))
    widget.guides_checkbox.setChecked(False)
    assert all(not region.isVisible() for region in widget.peak_regions)
    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("double"))
    widget.guides_checkbox.setChecked(True)
    assert widget.fit_region.isVisible()
    assert all(not region.isVisible() for region in widget.peak_regions)


def test_continuous_fit_coalesces_updates_and_stops_when_unchecked(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ratio, counts = _double_gaussian_projection()
    fit = fit_psd_fom_lm(ratio, counts, roi=(-0.05, 0.65), centroid_seeds=(0.1, 0.5))
    fitter = Mock(return_value=fit)
    monkeypatch.setattr("nlab.views.psd_fom_widget.fit_psd_fom_lm", fitter)
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    widget = plot.fom_widget
    widget.set_ratio_range((-0.2, 0.8))
    widget._continuous_timer.setInterval(20)
    assert not widget._continuous_timer.isActive()
    widget.continuous_checkbox.setChecked(True)
    for update in range(30):
        widget.set_projection(ratio, counts + update)
    qtbot.waitUntil(lambda: fitter.call_count == 1, timeout=1500)
    np.testing.assert_array_equal(fitter.call_args.args[1], counts + 29)
    assert fitter.call_args.kwargs["max_nfev"] == 400
    qtbot.wait(80)
    assert fitter.call_count == 1
    widget.centroid_lines[0].sigPositionChangeFinished.emit(widget.centroid_lines[0])
    qtbot.waitUntil(lambda: fitter.call_count == 2, timeout=1500)
    widget.continuous_checkbox.setChecked(False)
    assert not widget._continuous_timer.isActive()
    widget.set_projection(ratio, counts + 100)
    qtbot.wait(80)
    assert fitter.call_count == 2


def test_continuous_fit_waits_for_enabled_fit_control(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ratio, counts = _double_gaussian_projection()
    fit = fit_psd_fom_lm(ratio, counts, roi=(-0.05, 0.65), centroid_seeds=(0.1, 0.5))
    fitter = Mock(return_value=fit)
    monkeypatch.setattr("nlab.views.psd_fom_widget.fit_psd_fom_lm", fitter)
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    widget = plot.fom_widget
    widget.set_projection(ratio, counts)
    widget._continuous_timer.setInterval(20)
    widget.fit_button.setEnabled(False)
    widget.continuous_checkbox.setChecked(True)
    qtbot.wait(80)
    fitter.assert_not_called()
    widget.fit_button.setEnabled(True)
    qtbot.waitUntil(lambda: fitter.call_count == 1, timeout=1500)
    widget.continuous_checkbox.setChecked(False)


def test_bounded_fit_returns_a_warning_when_evaluation_budget_is_exhausted() -> None:
    ratio, counts = _double_gaussian_projection()
    fit = fit_psd_fom_separate_lm(
        ratio, counts, rois=((-0.02, 0.26), (0.32, 0.66)),
        centroid_seeds=(0.1, 0.5), max_nfev=2,
    )
    assert not fit.success and fit.poor_convergence
    assert any("did not converge" in warning for warning in fit.warnings)


def test_continuous_fit_does_not_retry_invalid_data_until_something_changes(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitter = Mock(side_effect=ValueError("No resolvable peak"))
    monkeypatch.setattr("nlab.views.psd_fom_widget.fit_psd_fom_lm", fitter)
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    widget = plot.fom_widget
    widget._continuous_timer.setInterval(20)
    widget.continuous_checkbox.setChecked(True)
    qtbot.waitUntil(lambda: fitter.call_count == 1, timeout=1500)
    qtbot.wait(80)
    assert fitter.call_count == 1
    assert "Fit unavailable" in widget.status_label.text()
    widget.continuous_checkbox.setChecked(False)



def test_continuous_updates_keep_readout_but_invalidate_stale_exports(qtbot: QtBot) -> None:
    ratio, counts = _double_gaussian_projection()
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    widget = plot.fom_widget
    widget.set_ratio_range((-0.2, 0.8))
    widget.set_projection(ratio, counts)
    previous_fit = widget.fit_current_projection()
    assert previous_fit is not None
    previous_readout = widget.fom_label.text()
    widget._continuous_timer.setInterval(20)
    widget.continuous_checkbox.setChecked(True)
    widget.set_projection(ratio, counts + 6)
    assert widget.last_fit is None
    assert not widget.copy_button.isEnabled()
    assert widget.fom_label.text() == previous_readout
    assert widget.status_label.text() == "Updating fit..."
    assert widget._fit_curve.opacity() == pytest.approx(0.4)
    qtbot.waitUntil(lambda: widget.last_fit is not None, timeout=1500)
    assert widget.copy_button.isEnabled()
    assert widget._fit_curve.opacity() == pytest.approx(1)
    assert widget.last_fit is not None
    np.testing.assert_array_equal(widget.last_fit.observed, previous_fit.observed + 6)
    widget.continuous_checkbox.setChecked(False)



@pytest.mark.parametrize("mode", ("double", "separate"))
def test_zero_offset_models_recover_peaks_and_use_six_varied_parameters(mode: str) -> None:
    ratio = np.linspace(0, 1, 1024)
    expected = 300 * np.exp(-0.5 * ((ratio - 0.25) / 0.025) ** 2)
    expected += 180 * np.exp(-0.5 * ((ratio - 0.75) / 0.035) ** 2)
    counts = np.random.default_rng(314).poisson(expected)
    if mode == "double":
        fit = fit_psd_fom_lm(
            ratio, counts, roi=(0.08, 0.94), centroid_seeds=(0.24, 0.76),
            use_background=False,
        )
    else:
        fit = fit_psd_fom_separate_lm(
            ratio, counts, rois=((0.08, 0.42), (0.54, 0.94)),
            centroid_seeds=(0.24, 0.76), use_background=False,
        )
    assert fit.success and not fit.use_background
    assert fit.varied_parameters == 6
    assert np.all(fit.background_component == 0)
    assert fit.centroids == pytest.approx((0.25, 0.75), abs=0.002)
    assert fit.fwhms == pytest.approx(np.asarray((0.025, 0.035)) * 2.35482, rel=0.04)
    model = np.maximum(fit.best_fit, 1e-12)
    y = fit.observed
    positive = y > 0
    deviance = np.sum(model - y)
    deviance += np.sum(y[positive] * np.log(y[positive] / model[positive]))
    assert fit.reduced_deviance == pytest.approx(2 * deviance / (len(y) - 6))


@pytest.mark.parametrize("mode", ("double", "separate"))
def test_offset_checkbox_invalidates_fit_and_controls_continuous_model(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    ratio, counts = _double_gaussian_projection()
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    widget = plot.fom_widget
    widget.set_ratio_range((-0.2, 0.8))
    widget.set_projection(ratio, counts)
    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData(mode))
    assert widget.background_checkbox.isChecked()
    assert widget.fit_current_projection() is not None
    widget.background_checkbox.setChecked(False)
    assert widget.last_fit is None and not widget.copy_button.isEnabled()
    function_name = "fit_psd_fom_lm" if mode == "double" else "fit_psd_fom_separate_lm"
    fitter = Mock(side_effect=ValueError("Test routing only"))
    monkeypatch.setattr(f"nlab.views.psd_fom_widget.{function_name}", fitter)
    widget.continuous_checkbox.setChecked(True)
    assert widget._continuous_timer.isActive()
    # Exercise model routing without depending on GUI timer scheduling.
    widget._continuous_timer.timeout.emit()
    fitter.assert_called_once()
    assert fitter.call_args.kwargs["use_background"] is False
    assert fitter.call_args.kwargs["max_nfev"] == 400
    widget.continuous_checkbox.setChecked(False)
    assert not widget._continuous_timer.isActive()
