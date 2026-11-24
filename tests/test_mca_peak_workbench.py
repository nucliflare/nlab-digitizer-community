"""MCA peak-analysis workbench integration checks."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
from pytestqt.qtbot import QtBot

import nlab.views.mca_peak_analysis_dialog as peak_analysis_dialog_module
from nlab.analysis.energy_calibration import (
    CalibrationPoint,
    CalibrationPointTransfer,
    EnergyCalibration,
    SpectrumSnapshot,
    fit_energy_calibration,
)
from nlab.analysis.spectrum import Spectrum
from nlab.ui.ui_main_window import Ui_MainWindow
from nlab.views.energy_calibration_dialog import EnergyCalibrationDialog
from nlab.views.mca_peak_analysis_dialog import McaPeakAnalysisDialog
from nlab.views.plot_viewbox import ModifierZoomViewBox

if TYPE_CHECKING:
    from nlab.analysis.compton_edge import ComptonEdgeFitSpec
    from nlab.controllers.mca_controller import MCAController
    from nlab.workers.compton_edge_fit_worker import ComptonEdgeFitWorker


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


# ------------------------------------------------- Compton-edge (keVee) workflow


def _compton_mix_counts() -> np.ndarray:
    from nlab.analysis.compton_edge import folded_line_response, nuclide_compton_lines

    edges = np.arange(4001, dtype=np.float64) - 0.5
    centres = np.arange(4000, dtype=np.float64)
    expected = 30.0 * np.exp(-centres / 1500.0)
    for nuclide in ("Cs-137", "Co-60", "Na-22"):
        for line in nuclide_compton_lines(nuclide):
            expected = expected + 4.0e5 * folded_line_response(
                edges, line, gain=3.0, offset=-30.0, resolution=(0.04, 1.2, 0.0)
            )
    return np.random.default_rng(5).poisson(np.clip(expected, 0.0, None)).astype(np.uint32)


@dataclass
class _CalibratingFakeMcaController(_FakeMcaController):
    channel_label: str = "CH0"
    applied: EnergyCalibration | None = None

    def energy_calibration_fingerprint(self) -> dict[str, int]:
        return {"binning": self.coincidence_energy_bin}

    def apply_energy_calibration(self, calibration: EnergyCalibration) -> None:
        self.applied = calibration


def _controllers(*controllers: _FakeMcaController) -> list[MCAController]:
    return cast("list[MCAController]", list(controllers))


def _check_compton_sources(dialog: McaPeakAnalysisDialog, nuclides: set[str]) -> None:
    for row in range(dialog.compton_nuclide_list.count()):
        item = dialog.compton_nuclide_list.item(row)
        if item.data(Qt.ItemDataRole.UserRole) in nuclides:
            item.setCheckState(Qt.CheckState.Checked)


def test_compton_edge_tab_fits_in_background_and_overlays_edges(qtbot: QtBot) -> None:
    from nlab.analysis.compton_edge import ComptonEdgeFitResult

    dialog = McaPeakAnalysisDialog(_controllers(_FakeMcaController(0, _compton_mix_counts())))
    qtbot.addWidget(dialog)
    assert dialog.side_tabs.tabText(1) == "Compton edge"
    assert dialog.compton_fit_action in dialog.analysis_menu.actions()
    assert dialog.compton_calibration_action in dialog.analysis_menu.actions()
    assert dialog.compton_export_action in dialog.file_menu.actions()
    assert not dialog.compton_export_action.isEnabled()
    dialog.side_tabs.setCurrentWidget(dialog.compton_tab)
    _check_compton_sources(dialog, {"Cs-137", "Co-60", "Na-22"})
    dialog.fit_region.setRegion((600.0, 3600.0))
    spectrum_id = dialog._current_id()
    assert spectrum_id is not None
    config = dialog._compton_configs[spectrum_id]
    assert config.include and config.nuclides == ["Cs-137", "Na-22", "Co-60"]
    assert config.fit_range == pytest.approx((600.0, 3600.0))
    assert dialog.compton_dataset_list.item(0).checkState() == Qt.CheckState.Checked

    assert dialog.workbench_progress.isHidden()
    dialog.compton_fit_action.trigger()
    assert dialog._compton_thread is not None
    assert not dialog.workbench_progress.isHidden()
    assert dialog.cursor().shape() == Qt.CursorShape.BusyCursor
    qtbot.waitUntil(lambda: dialog._compton_thread is None, timeout=60_000)
    assert dialog.workbench_progress.isHidden()
    assert dialog.cursor().shape() == Qt.CursorShape.ArrowCursor
    assert dialog.workbench_status.text() == "Compton-edge fit complete."

    result = dialog._compton_result
    assert isinstance(result, ComptonEdgeFitResult)
    assert result.success
    assert result.gain == pytest.approx(3.0, rel=0.01)
    assert dialog.compton_result_table.rowCount() > 10
    assert dialog.compton_result_table.item(0, 0).text() == "gain g"
    assert "keVee" in dialog.compton_summary.text()
    names = [item.name() for item in dialog.spectrum_plot.listDataItems()]
    assert "Compton-edge model" in names and "Cs-137 661.657 keV" in names
    markers = [
        item for item in dialog.spectrum_plot.items
        if isinstance(item, pg.InfiniteLine) and "edge" in item.label.format
    ]
    assert len(markers) == 5
    assert dialog.compton_export_action.isEnabled()
    assert dialog.compton_copy_action.isEnabled()
    dialog.compton_copy_action.trigger()
    assert QApplication.clipboard().text().startswith("Quantity\tValue")
    dialog.close_without_prompt()


def test_compton_progress_drives_the_bottom_bar_until_cancel(qtbot: QtBot) -> None:
    from nlab.analysis.compton_edge import ComptonEdgeFitProgress

    dialog = McaPeakAnalysisDialog([])
    qtbot.addWidget(dialog)
    worker = SimpleNamespace(stop_requested=False)
    dialog._compton_worker = cast("ComptonEdgeFitWorker", worker)
    bar = dialog.workbench_progress

    dialog._compton_fit_progress(ComptonEdgeFitProgress("gain_scan", 40, 160))
    assert (bar.minimum(), bar.maximum(), bar.value()) == (0, 160, 40)
    assert "starting gain (40/160)" in dialog.workbench_status.text()

    dialog._compton_fit_progress(ComptonEdgeFitProgress("optimize", 12))
    assert (bar.minimum(), bar.maximum()) == (0, 0)
    assert "12 evaluation(s)" in dialog.workbench_status.text()

    # Updates still queued behind a cancel request must not replace its status.
    worker.stop_requested = True
    dialog._compton_fit_progress(ComptonEdgeFitProgress("finalize"))
    assert "12 evaluation(s)" in dialog.workbench_status.text()
    dialog._compton_worker = None
    dialog.close_without_prompt()


def test_compton_fit_worker_throttles_progress_within_a_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nlab.analysis.compton_edge import ComptonEdgeFitProgress
    from nlab.workers import compton_edge_fit_worker as worker_module

    times = iter([0.0, 0.01, 0.02, 0.5, 0.51, 0.52, 0.53])
    monkeypatch.setattr(worker_module, "time", SimpleNamespace(monotonic=lambda: next(times)))
    worker = worker_module.ComptonEdgeFitWorker(cast("ComptonEdgeFitSpec", None))
    emitted: list[object] = []
    worker.progress.connect(emitted.append)

    updates = [
        ComptonEdgeFitProgress("prepare"),
        ComptonEdgeFitProgress("gain_scan", 0, 160),
        ComptonEdgeFitProgress("gain_scan", 1, 160),  # within 100 ms: dropped
        ComptonEdgeFitProgress("gain_scan", 2, 160),
        ComptonEdgeFitProgress("gain_scan", 160, 160),  # stage complete: always sent
        ComptonEdgeFitProgress("optimize", 1),
        ComptonEdgeFitProgress("optimize", 2),  # within 100 ms: dropped
    ]
    for update in updates:
        worker._report_progress(update)

    assert emitted == [updates[index] for index in (0, 1, 3, 4, 5)]
    worker.deleteLater()


def test_compton_edges_feed_energy_calibration_in_kevee(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _CalibratingFakeMcaController(0, _compton_mix_counts())
    dialog = McaPeakAnalysisDialog(_controllers(controller))
    qtbot.addWidget(dialog)
    dialog.side_tabs.setCurrentWidget(dialog.compton_tab)
    _check_compton_sources(dialog, {"Cs-137", "Co-60", "Na-22"})
    dialog.fit_region.setRegion((600.0, 3600.0))
    dialog.compton_fit_action.trigger()
    qtbot.waitUntil(lambda: dialog._compton_thread is None, timeout=60_000)
    result = dialog._compton_result
    assert result is not None
    # Nothing listening: refuse instead of reporting that the edges were opened.
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda _parent, _title, text: warnings.append(text)
    )
    status_before = dialog.compton_status.text()
    dialog.compton_calibration_action.trigger()
    assert warnings and "not connected" in warnings[0]
    assert dialog.compton_status.text() == status_before

    transfers: list[object] = []
    dialog.compton_calibration_requested.connect(transfers.append)

    dialog.compton_calibration_action.trigger()

    assert len(transfers) == 1
    transfer = transfers[0]
    assert isinstance(transfer, CalibrationPointTransfer)
    assert transfer.energy_unit == "keVee"
    assert transfer.channel == 0
    assert transfer.fingerprint == {"binning": 0}
    assert len(transfer.points) == 5
    assert "extrapolated" in dialog.compton_status.text()

    calibration_dialog = EnergyCalibrationDialog(_controllers(controller))
    qtbot.addWidget(calibration_dialog)
    workspace = calibration_dialog._workspace()
    assert workspace is not None
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes
    )
    calibration_dialog.add_calibration_points(transfer)

    assert workspace.energy_unit == "keVee"
    assert all(point.locked and point.snapshot_id is None for point in workspace.points)
    assert all(point.line is not None and not point.line.movable for point in workspace.points)
    header = calibration_dialog.table.horizontalHeaderItem(5)
    assert header is not None and header.text() == "Energy [keVee]"
    assert workspace.fit is not None and workspace.fit.energy_unit == "keVee"
    assert "[keVee]" in calibration_dialog.fit_status.text()
    assert "span" in calibration_dialog.fit_status.text()
    assert workspace.fit.coefficients_kev[1] == pytest.approx(1.0 / result.gain)
    assert controller.applied is None
    assert calibration_dialog.apply_action.isEnabled()
    assert "Apply calibration" in calibration_dialog.snapshot_status.text()
    calibration_dialog.apply_action.trigger()
    assert controller.applied is not None and controller.applied.energy_unit == "keVee"

    calibration_dialog.add_calibration_points(transfer)
    assert len(workspace.points) == 5
    calibration_dialog.close_without_prompt()
    dialog.close_without_prompt()


def test_energy_calibration_opens_snapshots_for_compton_fit(qtbot: QtBot) -> None:
    controller = _CalibratingFakeMcaController(0, _compton_mix_counts())
    calibration_dialog = EnergyCalibrationDialog(_controllers(controller))
    qtbot.addWidget(calibration_dialog)
    requested: list[object] = []
    calibration_dialog.compton_fit_requested.connect(requested.append)

    calibration_dialog.compton_action.trigger()

    assert len(requested) == 1
    spectra = requested[0]
    assert isinstance(spectra, list) and len(spectra) == 1
    spectrum = spectra[0]
    assert isinstance(spectrum, Spectrum)
    assert spectrum.metadata["settings_fingerprint"] == {"binning": 0}
    workbench = McaPeakAnalysisDialog([])
    qtbot.addWidget(workbench)
    workbench.add_compton_datasets(spectra)
    assert workbench.side_tabs.currentWidget() is workbench.compton_tab
    assert workbench._compton_configs[spectrum.spectrum_id].include
    assert workbench.compton_dataset_list.count() == 1
    workbench.close_without_prompt()
    calibration_dialog.close_without_prompt()


def test_kevee_calibration_labels_spectrum_axes(qtbot: QtBot) -> None:
    points = (
        CalibrationPoint(channel=100.0, energy_kev=340.0, source="compton-edge fit"),
        CalibrationPoint(channel=300.0, energy_kev=1118.0, source="compton-edge fit"),
    )
    calibration = fit_energy_calibration(points, energy_unit="keVee")
    controller = _FakeMcaController(0, _photopeak(), energy_calibration=calibration)
    dialog = McaPeakAnalysisDialog(_controllers(controller))
    qtbot.addWidget(dialog)
    spectrum = dialog._current_spectrum()
    assert spectrum is not None and spectrum.energy_unit == "keVee"
    assert dialog.energy_axis.labelText == "Light output"
    assert dialog.energy_axis.labelUnits == "keVee"
    dialog.close_without_prompt()


def test_compton_tab_controls_resolve_to_compton_help(qtbot: QtBot) -> None:
    from nlab.help.content import load_topics
    from nlab.help.context import resolve_context

    dialog = McaPeakAnalysisDialog([])
    qtbot.addWidget(dialog)
    controls = (dialog.compton_offset_free, dialog.compton_dk_combo, dialog.compton_result_table)
    for widget in controls:
        assert resolve_context(widget).topic == "compton-edge"
    assert resolve_context(dialog.peak_count).topic == "peak-analysis"
    assert "compton-edge" in load_topics()
    dialog.close_without_prompt()
