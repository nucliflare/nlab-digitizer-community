"""PSD double-Gaussian figure-of-merit fitting tests."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from nlab.analysis.psd_fom import fit_psd_fom_lm
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

    qtbot.mouseClick(plot.fom_widget.fit_button, Qt.MouseButton.LeftButton)

    assert plot.fom_widget.last_fit is not None
    assert plot.fom_widget.last_fit.fom == pytest.approx(1.7, rel=0.12)
    assert plot.fom_widget.fom_label.text() != "—"
    assert "converged" in plot.fom_widget.status_label.text()
    assert plot.fom_widget.copy_button.isEnabled()

    qtbot.mouseClick(plot.fom_widget.copy_button, Qt.MouseButton.LeftButton)
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
