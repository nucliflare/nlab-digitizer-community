"""MCA peak-analysis workbench integration checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from PySide6.QtWidgets import QMainWindow, QMessageBox
from pytestqt.qtbot import QtBot

from nlab.analysis.energy_calibration import EnergyCalibration, SpectrumSnapshot
from nlab.ui.ui_main_window import Ui_MainWindow
from nlab.views.mca_peak_analysis_dialog import McaPeakAnalysisDialog
from nlab.views.plot_viewbox import ModifierZoomViewBox


@dataclass
class _FakeMcaController:
    channel: int
    counts: np.ndarray
    elapsed_s: float = 10.0
    energy_calibration: EnergyCalibration | None = None
    coincidence_energy_bin: int = 0
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
        )

    def energy_calibration_is_stale(self) -> bool:
        return False


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

    dialog._start_fit()
    assert dialog._fit_thread is not None
    qtbot.waitUntil(lambda: dialog._fit_thread is None, timeout=10_000)

    result = dialog._fit_result
    assert result is not None
    assert result.success
    assert result.peaks[0].center == pytest.approx(180, abs=0.2)
    assert result.statistic == "poisson"
    assert dialog.peak_table.rowCount() == 1
    assert dialog.export_fit_button.isEnabled()
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


def test_peak_analysis_action_is_in_tools_menu(qtbot: QtBot) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.actionMcaPeakAnalysis in ui.menuTools.actions()
    assert ui.actionMcaPeakAnalysis not in ui.menuDeveloper.actions()
