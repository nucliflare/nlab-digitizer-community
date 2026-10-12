from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from PySide6.QtWidgets import QMainWindow
from pytestqt.qtbot import QtBot

from nlab.analysis.energy_calibration import (
    CalibrationPoint,
    EnergyCalibration,
    SpectrumSnapshot,
    fit_calibration_peak,
    fit_energy_calibration,
)
from nlab.ui.ui_main_window import Ui_MainWindow
from nlab.views.energy_calibration_dialog import EnergyCalibrationDialog
from nlab.views.plot_viewbox import ModifierZoomViewBox


def test_linear_calibration_uses_all_points_and_round_trips() -> None:
    points = (
        CalibrationPoint(100.0, 510.5, source="source-a"),
        CalibrationPoint(200.0, 1_021.0, source="source-b"),
        CalibrationPoint(300.0, 1_531.5, source="source-c"),
    )

    calibration = fit_energy_calibration(
        points,
        model="linear",
        fingerprint={"binning": 0},
    )

    assert calibration.coefficients_kev[0] == pytest.approx(0.0, abs=1e-10)
    assert calibration.coefficients_kev[1] == pytest.approx(5.105)
    assert calibration.energy(250.0) == pytest.approx(1_276.25)
    assert calibration.energy_at_binning(125.0, 1) == pytest.approx(1_276.25)
    assert calibration.channel_scale_for_binning(1) == 2.0
    assert calibration.settings_compatible(
        {"binning": 4},
        allow_binning_rescale=True,
    )
    assert not calibration.settings_compatible(
        {"binning": 4},
        allow_binning_rescale=False,
    )
    assert calibration.rms_residual_kev == pytest.approx(0.0, abs=1e-10)
    restored = EnergyCalibration.from_dict(calibration.to_dict())
    assert restored.model == "linear"
    assert restored.fingerprint == {"binning": 0}
    assert restored.energy(250.0) == pytest.approx(1_276.25)
    assert [point.source for point in restored.points] == [
        "source-a",
        "source-b",
        "source-c",
    ]


def test_quadratic_calibration_and_monotonic_validation() -> None:
    points = tuple(
        CalibrationPoint(channel, 2.0 + 0.5 * channel + 0.001 * channel**2)
        for channel in (100.0, 500.0, 1_000.0, 2_000.0)
    )

    calibration = fit_energy_calibration(points, model="quadratic")

    assert calibration.coefficients_kev == pytest.approx((2.0, 0.5, 0.001))
    assert calibration.energy(750.0) == pytest.approx(939.5)

    with pytest.raises(ValueError, match="must increase"):
        fit_energy_calibration(
            (CalibrationPoint(100, 1_000), CalibrationPoint(200, 500)),
            model="linear",
        )


def test_gaussian_centroid_fit_refines_a_dragged_seed() -> None:
    x = np.arange(256, dtype=np.float64)
    center = 113.35
    counts = 14.0 + 0.025 * x
    counts += 45_000.0 * np.exp(-0.5 * np.square((x - center) / 4.8)) / (
        4.8 * np.sqrt(2.0 * np.pi)
    )
    snapshot = SpectrumSnapshot.create(
        channel=0,
        counts=np.rint(counts).astype(np.uint32),
        label="Calibration source",
        elapsed_s=30.0,
        live=False,
        fingerprint={"binning": 0},
    )

    result = fit_calibration_peak(snapshot, 109.0, half_width=24.0)

    assert result.seed_channel == 109.0
    assert result.centroid_channel == pytest.approx(center, abs=0.08)
    assert result.fwhm_channel == pytest.approx(2.35482 * 4.8, abs=0.15)
    assert result.centroid_stderr is not None
    assert len(result.x) == len(result.observed) == len(result.best_fit)


def test_dialog_uses_gaussian_centroids_for_energy_fit(qtbot: QtBot) -> None:
    x = np.arange(256, dtype=np.float64)
    centers = (72.4, 181.65)
    counts = np.full_like(x, 10.0)
    for center, area, sigma in ((centers[0], 35_000.0, 3.8), (centers[1], 48_000.0, 5.2)):
        counts += area * np.exp(-0.5 * np.square((x - center) / sigma)) / (
            sigma * np.sqrt(2.0 * np.pi)
        )
    snapshot = SpectrumSnapshot.create(
        channel=0,
        counts=np.rint(counts).astype(np.uint32),
        label="Two-line source",
        elapsed_s=30.0,
        live=False,
        fingerprint={"binning": 0},
    )
    controller = _FakeMcaController(0, snapshot)
    dialog = EnergyCalibrationDialog([controller])  # type: ignore[list-item]
    qtbot.addWidget(dialog)

    dialog._add_point(69.0)
    dialog._add_point(177.0)
    workspace = dialog._workspace()
    assert workspace is not None
    workspace.points[0].energy_kev = 511.0
    workspace.points[1].energy_kev = 1274.537
    dialog._rebuild_table()
    dialog._refit()

    assert [point.seed_channel for point in workspace.points] == [69.0, 177.0]
    assert [point.channel for point in workspace.points] == pytest.approx(centers, abs=0.1)
    assert workspace.fit is not None
    assert [point.channel for point in workspace.fit.points] == pytest.approx(centers, abs=0.1)
    assert dialog.report_button.isEnabled()
    assert all(point.centroid_line is not None for point in workspace.points)

    original_seeds = [point.seed_channel for point in workspace.points]
    dialog._apply()

    assert controller.energy_calibration is workspace.fit
    assert workspace.show_peak_fits
    assert original_seeds == [69.0, 177.0]
    assert [point.seed_channel for point in workspace.points] == pytest.approx(centers, abs=0.1)
    assert [point.line.value() for point in workspace.points if point.line is not None] == (
        pytest.approx(centers, abs=0.1)
    )
    assert all(point.centroid_line is None for point in workspace.points)
    assert all(point.fit_curve is not None for point in workspace.points)
    assert all(point.residual_curve is not None for point in workspace.points)
    for point in workspace.points:
        assert point.fit_curve is not None
        assert point.residual_curve is not None
        assert point.peak_fit is not None
        _, displayed_fit = point.fit_curve.getData()
        _, displayed_residual = point.residual_curve.getData()
        np.testing.assert_allclose(displayed_fit, point.peak_fit.best_fit)
        np.testing.assert_allclose(
            displayed_residual,
            point.peak_fit.observed - point.peak_fit.best_fit,
        )

    applied_fits = [point.peak_fit for point in workspace.points]
    dialog._show_channel()
    assert all(
        point.peak_fit is applied_fit
        for point, applied_fit in zip(workspace.points, applied_fits, strict=True)
    )
    assert all(point.fit_curve is not None for point in workspace.points)
    assert all(point.residual_curve is not None for point in workspace.points)
    dialog.close_without_prompt()


@dataclass
class _FakeMcaController:
    channel: int
    _snapshot: SpectrumSnapshot
    energy_calibration: EnergyCalibration | None = None

    def spectrum_snapshot(self) -> SpectrumSnapshot:
        return self._snapshot

    def energy_calibration_fingerprint(self) -> dict[str, int]:
        return {"binning": 0}

    def apply_energy_calibration(self, calibration: EnergyCalibration) -> None:
        self.energy_calibration = calibration

    def clear_energy_calibration(self) -> None:
        self.energy_calibration = None


def _snapshot(label: str, peak: int) -> SpectrumSnapshot:
    x = np.arange(32, dtype=np.float64)
    counts = 4.0 + 2_000.0 * np.exp(-0.5 * np.square((x - peak) / 1.5)) / (
        1.5 * np.sqrt(2.0 * np.pi)
    )
    return SpectrumSnapshot.create(
        channel=0,
        counts=np.rint(counts).astype(np.uint32),
        label=label,
        elapsed_s=10.0,
        live=False,
        fingerprint={"binning": 0},
    )


def test_dialog_keeps_points_when_a_new_source_spectrum_is_overlaid(qtbot: QtBot) -> None:
    controller = _FakeMcaController(0, _snapshot("Cs-137", 5))
    dialog = EnergyCalibrationDialog([controller])  # type: ignore[list-item]
    qtbot.addWidget(dialog)
    workspace = dialog._workspace()
    assert workspace is not None
    assert [snapshot.label for snapshot in workspace.snapshots] == ["Cs-137"]

    dialog._add_point(5.25)
    workspace.points[0].energy_kev = 661.657

    controller._snapshot = _snapshot("Co-60", 15)
    dialog._load_current(replace=False)

    assert [snapshot.label for snapshot in workspace.snapshots] == ["Cs-137", "Co-60"]
    assert len(workspace.points) == 1
    dialog._add_point(15.5)
    workspace.points[1].energy_kev = 1_332.492
    dialog._rebuild_table()
    dialog._refit()
    assert dialog.apply_button.isEnabled()
    assert len(workspace.points) == 2
    dialog._apply()
    assert controller.energy_calibration is not None
    assert len(controller.energy_calibration.points) == 2
    dialog.close_without_prompt()


def test_energy_calibration_plots_use_modifier_zoom(qtbot: QtBot) -> None:
    controller = _FakeMcaController(0, _snapshot("Cs-137", 5))
    dialog = EnergyCalibrationDialog([controller])  # type: ignore[list-item]
    qtbot.addWidget(dialog)

    assert isinstance(dialog.plot.getViewBox(), ModifierZoomViewBox)
    assert isinstance(dialog.residual_plot.getViewBox(), ModifierZoomViewBox)
    assert dialog.centroid_half_width.value() == 300.0
    dialog.close_without_prompt()


def test_dragging_reference_line_updates_channel_table(qtbot: QtBot) -> None:
    controller = _FakeMcaController(0, _snapshot("Cs-137", 5))
    dialog = EnergyCalibrationDialog([controller])  # type: ignore[list-item]
    qtbot.addWidget(dialog)
    dialog._add_point(5.0)
    workspace = dialog._workspace()
    assert workspace is not None
    line = workspace.points[0].line
    assert line is not None

    line.setValue(7.25)

    assert workspace.points[0].channel == pytest.approx(7.25)
    channel_item = dialog.table.item(0, 3)
    assert channel_item is not None
    assert float(channel_item.text()) == pytest.approx(7.25)
    dialog.close_without_prompt()


def test_all_calibration_table_columns_fit_at_minimum_window_size(qtbot: QtBot) -> None:
    controller = _FakeMcaController(0, _snapshot("Cs-137", 5))
    dialog = EnergyCalibrationDialog([controller])  # type: ignore[list-item]
    qtbot.addWidget(dialog)
    dialog.resize(dialog.minimumSize())
    dialog.show()
    qtbot.waitExposed(dialog)

    last_column = dialog.table.columnCount() - 1
    last_column_right = (
        dialog.table.columnViewportPosition(last_column)
        + dialog.table.columnWidth(last_column)
    )
    assert last_column_right <= dialog.table.viewport().width()
    assert dialog.table.horizontalScrollBar().maximum() == 0
    dialog.close_without_prompt()


def test_energy_calibration_action_is_in_tools_menu(qtbot: QtBot) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.actionEnergyCalibration in ui.menuTools.actions()
    assert ui.actionEnergyCalibration not in ui.menuDeveloper.actions()
