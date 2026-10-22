"""MCA peak-analysis workbench integration checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QMessageBox
from pytestqt.qtbot import QtBot

import nlab.views.mca_peak_analysis_dialog as peak_analysis_dialog_module
from nlab.analysis.energy_calibration import (
    CalibrationPoint,
    EnergyCalibration,
    SpectrumSnapshot,
    fit_energy_calibration,
)
from nlab.analysis.spectrum import Spectrum
from nlab.ui.ui_main_window import Ui_MainWindow
from nlab.views.mca_peak_analysis_dialog import McaPeakAnalysisDialog
from nlab.views.plot_viewbox import ModifierZoomViewBox


@dataclass
class _FakeMcaController:
    channel: int
    counts: np.ndarray
    elapsed_s: float = 10.0
    live_time_s: float | None = 9.5
    energy_calibration: EnergyCalibration | None = None
    coincidence_energy_bin: int = 0
    calibration_stale: bool = False
    snapshots: int = 0

    def spectrum_snapshot(self) -> SpectrumSnapshot:
        self.snapshots += 1
        return SpectrumSnapshot.create(
            channel=self.channel,
            counts=self.counts,
            label=f"MCA {self.channel} snapshot {self.snapshots}",
            elapsed_s=self.elapsed_s,
            live=True,
            fingerprint={"binning": self.coincidence_energy_bin},
            live_time_s=self.live_time_s,
            n42_instrument={"identifier": f"D1:CH{self.channel}@192.0.2.10:30431"},
            n42_detector={"name": f"D1:CH{self.channel}", "category": "Other"},
        )

    def energy_calibration_is_stale(self) -> bool:
        return self.calibration_stale


def _photopeak(center: float = 180.0) -> np.ndarray:
    x = np.arange(400, dtype=np.float64)
    counts = 12.0 + 0.01 * x
    counts += 80_000 * np.exp(-0.5 * ((x - center) / 7.0) ** 2) / (
        7.0 * np.sqrt(2.0 * np.pi)
    )
    return np.rint(counts).astype(np.uint32)


def test_workbench_automatically_snapshots_all_mcas_and_refreshes_selected(
    qtbot: QtBot,
) -> None:
    controllers = [_FakeMcaController(0, _photopeak()), _FakeMcaController(1, _photopeak(220))]
    dialog = McaPeakAnalysisDialog(controllers)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    assert dialog.source_list.count() == 2
    assert [controller.snapshots for controller in controllers] == [1, 1]
    spectrum_id = dialog._current_id()
    assert spectrum_id is not None
    original = dialog._spectra[spectrum_id]
    assert original.metadata["live_time_s"] == 9.5
    assert original.metadata["dead_time_s"] == 0.5
    assert original.metadata["n42_detector"] == {"name": "D1:CH0", "category": "Other"}

    controllers[0].counts = np.full(400, 7, dtype=np.uint32)
    dialog._refresh_selected()

    refreshed = dialog._spectra[spectrum_id]
    assert refreshed.spectrum_id == original.spectrum_id
    np.testing.assert_array_equal(refreshed.counts, 7)
    assert controllers[0].snapshots == 2
    dialog.close_without_prompt()


def test_workbench_operation_creates_immutable_derived_spectrum(qtbot: QtBot) -> None:
    controller = _FakeMcaController(0, _photopeak())
    dialog = McaPeakAnalysisDialog([controller])  # type: ignore[list-item]
    qtbot.addWidget(dialog)
    source_id = dialog._current_id()
    assert source_id is not None
    source = dialog._spectra[source_id]
    dialog.operation_combo.setCurrentIndex(dialog.operation_combo.findData("scale"))
    dialog.operation_factor.setValue(0.5)

    dialog._apply_operation()

    derived = dialog._current_spectrum()
    assert derived is not None
    assert derived.source == "derived"
    assert derived.spectrum_id != source.spectrum_id
    np.testing.assert_allclose(derived.counts, source.counts * 0.5)
    np.testing.assert_allclose(dialog._spectra[source_id].counts, source.counts)
    assert not derived.poisson_counts
    dialog.close_without_prompt()


def test_workbench_removes_only_managed_replay_spectra(qtbot: QtBot) -> None:
    dialog = McaPeakAnalysisDialog(  # type: ignore[list-item]
        [_FakeMcaController(0, _photopeak())]
    )
    qtbot.addWidget(dialog)
    live_id = dialog._current_id()
    assert live_id is not None
    replay = Spectrum.create(
        label="Recorded MCA Ch 0",
        counts=np.arange(16, dtype=np.uint32),
        source="mca",
        spectrum_id="session-replay-ch0",
    )

    dialog.add_spectra([replay])
    dialog.remove_spectra({replay.spectrum_id})

    assert replay.spectrum_id not in dialog._spectra
    assert live_id in dialog._spectra
    assert dialog.source_list.count() == 1
    dialog.close_without_prompt()


def test_workbench_runs_peak_fit_off_the_gui_thread(qtbot: QtBot) -> None:
    dialog = McaPeakAnalysisDialog(  # type: ignore[list-item]
        [_FakeMcaController(0, _photopeak())]
    )
    qtbot.addWidget(dialog)
    dialog.fit_region.setRegion((130, 230))
    dialog._suggest_peaks()
    assert all(
        line.zValue() > dialog.fit_region.zValue() for line in dialog._peak_lines
    )

    dialog.fit_action.trigger()
    assert dialog._fit_thread is not None
    qtbot.waitUntil(lambda: dialog._fit_thread is None, timeout=10_000)

    result = dialog._fit_result
    assert result is not None
    assert result.success
    assert result.peaks[0].center == pytest.approx(180, abs=0.2)
    assert result.statistic == "poisson"
    assert dialog.peak_table.rowCount() == 1
    assert dialog.export_fit_action.isEnabled()
    assert dialog.report_fit_action.isEnabled()
    dialog.close_without_prompt()


def test_peak_fit_failure_is_shown_in_status_and_message_box(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = McaPeakAnalysisDialog(  # type: ignore[list-item]
        [_FakeMcaController(0, _photopeak())]
    )
    qtbot.addWidget(dialog)
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: messages.append((title, message)),
    )

    dialog._fit_failed("peak centres must fall inside the fit range")

    assert dialog.fit_status.text() == (
        "Fit failed: peak centres must fall inside the fit range"
    )
    assert dialog.workbench_status.text() == (
        "Peak fit failed: peak centres must fall inside the fit range"
    )
    assert messages == [
        ("Peak Fit Failed", "peak centres must fall inside the fit range")
    ]
    dialog.close_without_prompt()


def test_peak_results_header_fits_compact_results_panel(qtbot: QtBot) -> None:
    dialog = McaPeakAnalysisDialog(  # type: ignore[list-item]
        [_FakeMcaController(0, _photopeak())]
    )
    qtbot.addWidget(dialog)

    header = dialog.peak_table.horizontalHeader()
    assert header.font().pointSizeF() < dialog.peak_table.font().pointSizeF()
    assert dialog.peak_table.horizontalHeaderItem(2).text() == "E\n(keV)"
    assert dialog.peak_table.horizontalHeaderItem(5).text() == "Res.\n(%)"
    assert dialog.peak_table.horizontalHeaderItem(6).text() == "Centre\nSE"
    assert dialog.peak_table.horizontalHeaderItem(5).toolTip()
    dialog.close_without_prompt()


def test_peak_spectrum_and_residual_use_modifier_zoom(qtbot: QtBot) -> None:
    dialog = McaPeakAnalysisDialog(  # type: ignore[list-item]
        [_FakeMcaController(0, _photopeak())]
    )
    qtbot.addWidget(dialog)

    assert isinstance(dialog.spectrum_plot.getViewBox(), ModifierZoomViewBox)
    assert isinstance(dialog.residual_plot.getViewBox(), ModifierZoomViewBox)
    dialog.close_without_prompt()


def test_reference_lines_overlay_na22_on_calibrated_channel_axis(qtbot: QtBot) -> None:
    calibration = fit_energy_calibration(
        (CalibrationPoint(0.0, 0.0), CalibrationPoint(399.0, 2000.0))
    )
    controller = _FakeMcaController(0, _photopeak(), energy_calibration=calibration)
    dialog = McaPeakAnalysisDialog([controller])  # type: ignore[list-item]
    qtbot.addWidget(dialog)

    na22 = next(
        dialog.reference_nuclide_list.item(row)
        for row in range(dialog.reference_nuclide_list.count())
        if dialog.reference_nuclide_list.item(row).data(Qt.ItemDataRole.UserRole) == "Na-22"
    )
    na22.setCheckState(Qt.CheckState.Checked)

    assert len(dialog._reference_lines) == 2
    assert [line.value() for line in dialog._reference_lines] == pytest.approx(
        [511.0 * 399.0 / 2000.0, 1274.537 * 399.0 / 2000.0]
    )
    assert dialog.reference_table.rowCount() == 2
    assert dialog.reference_table.item(0, 3).text() == "Annihilation"
    assert "Marker height is not a predicted peak height" in dialog.reference_status.text()
    dialog.close_without_prompt()


def test_reference_lines_require_current_energy_calibration(qtbot: QtBot) -> None:
    dialog = McaPeakAnalysisDialog(  # type: ignore[list-item]
        [_FakeMcaController(0, _photopeak())]
    )
    qtbot.addWidget(dialog)
    first = dialog.reference_nuclide_list.item(0)

    first.setCheckState(Qt.CheckState.Checked)

    assert dialog._reference_lines == []
    assert "Apply an energy calibration" in dialog.reference_status.text()
    assert not dialog.reference_roi_filter.isEnabled()
    assert "unavailable" in dialog.reference_filter_status.text()
    dialog.close_without_prompt()


def test_reference_favourites_are_pinned_editable_and_persistent(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_settings: dict[str, object] = {}

    class MemorySettings:
        def contains(self, key: str) -> bool:
            return key in stored_settings

        def value(self, key: str, default: object = None) -> object:
            return stored_settings.get(key, default)

        def setValue(self, key: str, value: object) -> None:  # noqa: N802 - Qt API
            stored_settings[key] = value

    monkeypatch.setattr(peak_analysis_dialog_module, "QSettings", MemorySettings)
    dialog = McaPeakAnalysisDialog(  # type: ignore[list-item]
        [_FakeMcaController(0, _photopeak())]
    )
    qtbot.addWidget(dialog)

    expected_defaults = [
        "Cs-137",
        "Na-22",
        "Bi-207",
        "Mn-54",
        "Cr-51",
        "Co-57",
        "Cd-109",
        "Am-241",
        "Pm-145",
        "Ba-133",
    ]
    listed = [
        str(dialog.reference_nuclide_list.item(row).data(Qt.ItemDataRole.UserRole))
        for row in range(len(expected_defaults))
    ]
    assert listed == expected_defaults
    assert all(
        dialog.reference_nuclide_list.item(row).text().startswith("★ ")
        for row in range(len(expected_defaults))
    )

    co60 = next(
        dialog.reference_nuclide_list.item(row)
        for row in range(dialog.reference_nuclide_list.count())
        if dialog.reference_nuclide_list.item(row).data(Qt.ItemDataRole.UserRole) == "Co-60"
    )
    dialog.reference_nuclide_list.setCurrentItem(co60)
    assert dialog.reference_favourite_action.text() == "Add to favourites"
    dialog.reference_favourite_action.trigger()

    assert dialog._reference_favourites == expected_defaults + ["Co-60"]
    assert stored_settings["analysis/radionuclide_reference_favourites"] == (
        expected_defaults + ["Co-60"]
    )
    assert dialog.reference_nuclide_list.item(len(expected_defaults)).data(
        Qt.ItemDataRole.UserRole
    ) == "Co-60"
    dialog.close_without_prompt()

    reopened = McaPeakAnalysisDialog(  # type: ignore[list-item]
        [_FakeMcaController(0, _photopeak())]
    )
    qtbot.addWidget(reopened)
    cs137 = next(
        reopened.reference_nuclide_list.item(row)
        for row in range(reopened.reference_nuclide_list.count())
        if reopened.reference_nuclide_list.item(row).data(Qt.ItemDataRole.UserRole) == "Cs-137"
    )
    reopened.reference_nuclide_list.setCurrentItem(cs137)
    assert reopened.reference_favourite_action.text() == "Remove from favourites"
    reopened.reference_favourite_action.trigger()

    assert "Cs-137" not in reopened._reference_favourites
    assert reopened.reference_nuclide_list.item(0).data(Qt.ItemDataRole.UserRole) == "Na-22"
    unpinned_cs137 = next(
        reopened.reference_nuclide_list.item(row)
        for row in range(reopened.reference_nuclide_list.count())
        if reopened.reference_nuclide_list.item(row).data(Qt.ItemDataRole.UserRole) == "Cs-137"
    )
    assert not unpinned_cs137.text().startswith("★ ")
    reopened.close_without_prompt()


def test_reference_candidate_filter_uses_calibrated_fit_roi(qtbot: QtBot) -> None:
    calibration = fit_energy_calibration(
        (CalibrationPoint(0.0, 0.0), CalibrationPoint(399.0, 2000.0))
    )
    controller = _FakeMcaController(0, _photopeak(), energy_calibration=calibration)
    dialog = McaPeakAnalysisDialog([controller])  # type: ignore[list-item]
    qtbot.addWidget(dialog)
    dialog.fit_region.setRegion((650.0 * 399.0 / 2000.0, 670.0 * 399.0 / 2000.0))

    assert dialog.reference_roi_filter.isEnabled()
    dialog.reference_roi_filter.setChecked(True)

    items = {
        str(dialog.reference_nuclide_list.item(row).data(Qt.ItemDataRole.UserRole)):
        dialog.reference_nuclide_list.item(row)
        for row in range(dialog.reference_nuclide_list.count())
    }
    assert "Cs-137" in items
    assert "1 ROI line" in items["Cs-137"].text()
    assert "strongest 85.1%" in items["Cs-137"].text()
    assert "Na-22" not in items
    assert "A match is not an identification" in dialog.reference_filter_status.text()

    dialog.reference_probability.setValue(90.0)
    remaining = {
        str(dialog.reference_nuclide_list.item(row).data(Qt.ItemDataRole.UserRole))
        for row in range(dialog.reference_nuclide_list.count())
    }
    assert "Cs-137" not in remaining
    dialog.close_without_prompt()


def test_reference_candidate_filter_clips_negative_calibrated_roi(qtbot: QtBot) -> None:
    calibration = fit_energy_calibration(
        (CalibrationPoint(100.0, 10.0), CalibrationPoint(200.0, 30.0))
    )
    dialog = McaPeakAnalysisDialog(  # type: ignore[list-item]
        [_FakeMcaController(0, _photopeak(), energy_calibration=calibration)]
    )
    qtbot.addWidget(dialog)
    dialog.fit_region.setRegion((0.0, 100.0))

    dialog.reference_roi_filter.setChecked(True)

    assert dialog.reference_roi_filter.isChecked()
    assert dialog._reference_roi_energy_range == pytest.approx((0.0, 10.0))
    assert "0–10 keV" in dialog.reference_filter_status.text()
    dialog.close_without_prompt()


def test_reference_candidate_filter_keeps_selected_nonmatch_manageable(qtbot: QtBot) -> None:
    calibration = fit_energy_calibration(
        (CalibrationPoint(0.0, 0.0), CalibrationPoint(399.0, 2000.0))
    )
    dialog = McaPeakAnalysisDialog(  # type: ignore[list-item]
        [_FakeMcaController(0, _photopeak(), energy_calibration=calibration)]
    )
    qtbot.addWidget(dialog)
    na22 = next(
        dialog.reference_nuclide_list.item(row)
        for row in range(dialog.reference_nuclide_list.count())
        if dialog.reference_nuclide_list.item(row).data(Qt.ItemDataRole.UserRole) == "Na-22"
    )
    na22.setCheckState(Qt.CheckState.Checked)
    dialog.fit_region.setRegion((650.0 * 399.0 / 2000.0, 670.0 * 399.0 / 2000.0))
    dialog.reference_roi_filter.setChecked(True)

    retained = next(
        dialog.reference_nuclide_list.item(row)
        for row in range(dialog.reference_nuclide_list.count())
        if dialog.reference_nuclide_list.item(row).data(Qt.ItemDataRole.UserRole) == "Na-22"
    )
    assert retained.checkState() == Qt.CheckState.Checked
    assert "selected; no ROI match" in retained.text()

    retained.setCheckState(Qt.CheckState.Unchecked)
    assert retained.isHidden()
    dialog.close_without_prompt()


def test_peak_analysis_action_is_in_tools_menu(qtbot: QtBot) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.actionMcaPeakAnalysis in ui.menuTools.actions()
    assert ui.actionMcaPeakAnalysis not in ui.menuDeveloper.actions()
