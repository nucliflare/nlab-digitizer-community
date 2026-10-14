from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QMainWindow
from pytestqt.qtbot import QtBot

from nlab.analysis.dose_calibration import (
    DigitizerTemperatureSnapshot,
    DoseCalibrationCsvWriter,
    McaDoseCalibrationSnapshot,
    PsdDoseCalibrationSnapshot,
    PsuDoseCalibrationSnapshot,
    differential_spectrum_integrals,
)
from nlab.analysis.psd import PsdAccumulator
from nlab.controllers.global_controller import GlobalController
from nlab.controllers.mca_controller import MCAController
from nlab.controllers.psd_controller import PSDController
from nlab.controllers.psu_controller import PSUController
from nlab.hardware.digitizer.diagnostics import GlobalDiagnosticReading
from nlab.ui.ui_main_window import Ui_MainWindow
from nlab.views.dose_calibration_dialog import (
    DoseCalibrationChannelSource,
    DoseCalibrationDialog,
)
from nlab.workers.mca_worker import MCAReadback
from nlab.workers.psu_worker import PSUReadback
from nlab.workers.temperature_correction_worker import TemperatureCorrectionReadback


def test_differential_spectrum_integrals_clip_resets_and_integrate_roi() -> None:
    result = differential_spectrum_integrals(
        np.array([5, 1, 10, 4], dtype=np.uint32),
        np.array([2, 3, 5, 4], dtype=np.uint32),
        (1, 2),
    )

    assert result.total_counts == 8
    assert result.roi_counts == 5


def test_csv_writer_flushes_rows_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "dose.csv"
    writer = DoseCalibrationCsvWriter(path, ("dose", "counts"))
    writer.append({"dose": 1.5, "counts": 42})

    assert path.read_text(encoding="utf-8").splitlines() == [
        "dose,counts",
        "1.5,42",
    ]
    writer.close()
    with pytest.raises(FileExistsError):
        DoseCalibrationCsvWriter(path, ("dose",))


def test_controller_snapshots_expose_cached_calibration_inputs() -> None:
    mca = SimpleNamespace(
        _latest_readback=MCAReadback(
            histogram=np.array([10, 20, 30], dtype=np.uint32),
            debug1=np.empty(0),
            debug2=np.empty(0),
            count_rate=123,
            pulse_deadtime_ms=0.25,
            elapsed_time=85,
        ),
        _roi=SimpleNamespace(isVisible=lambda: True),
        _roi_bounds=lambda _length: (1, 2),
    )
    mca_snapshot = MCAController.dose_calibration_snapshot(mca)
    assert mca_snapshot is not None
    assert mca_snapshot.count_rate_raw == 123
    assert mca_snapshot.acquisition_elapsed_s == pytest.approx(8.5)
    assert mca_snapshot.roi == (1, 2)

    psu = SimpleNamespace(
        _latest_readback=PSUReadback(
            timestamp=1.0,
            hv_voltage=498.0,
            hv_compens_output=500.0,
            temp_analog=24.0,
            temp_digital=25.0,
            temp_digital_status=1,
            ads_temp=68.0,
            sipm_voltage=None,
        ),
        _sipm_available=False,
        ui=SimpleNamespace(spinHvVoltage=SimpleNamespace(value=lambda: 500.0)),
    )
    psu_snapshot = PSUController.dose_calibration_snapshot(psu)
    assert psu_snapshot.hv_setpoint_v == pytest.approx(500.0)
    assert psu_snapshot.hv_readback_v == pytest.approx(498.0)
    assert psu_snapshot.adc_temperature_c == pytest.approx(68.0)

    accumulator = PsdAccumulator(energy_bins=4, ratio_bins=4, ratio_range=(0.0, 1.0))
    accumulator.matrix[:] = np.array(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
        ],
        dtype=np.uint64,
    )
    psd = SimpleNamespace(
        ui=SimpleNamespace(spinCut=SimpleNamespace(value=lambda: 0.5)),
        _accumulator=accumulator,
        _energy_roi=SimpleNamespace(getRegion=lambda: (0.0, 32768.0)),
    )
    psd_snapshot = PSDController.dose_calibration_snapshot(psd)
    assert psd_snapshot.below_cut_total_counts == 24
    assert psd_snapshot.above_cut_total_counts == 40
    assert psd_snapshot.below_cut_roi_counts == 8
    assert psd_snapshot.above_cut_roi_counts == 16

    global_controller = SimpleNamespace(
        _latest_diagnostics_readback=(
            GlobalDiagnosticReading("hat_temperature", "HAT", 43.2),
            GlobalDiagnosticReading("fpga_temperature", "FPGA", 68.9),
        ),
        _latest_temperature_readback=TemperatureCorrectionReadback(71.0, 0.0, 0),
    )
    temperatures = GlobalController.dose_calibration_temperature_snapshot(global_controller)
    assert temperatures.hat_temperature_c == pytest.approx(43.2)
    assert temperatures.fpga_temperature_c == pytest.approx(68.9)
    assert temperatures.ads5407_temperature_raw == pytest.approx(71.0)


@dataclass
class _McaSource:
    snapshot: McaDoseCalibrationSnapshot

    def dose_calibration_snapshot(self) -> McaDoseCalibrationSnapshot:
        return self.snapshot


@dataclass(frozen=True)
class _PsuSource:
    def dose_calibration_snapshot(self) -> PsuDoseCalibrationSnapshot:
        return PsuDoseCalibrationSnapshot(
            hv_setpoint_v=500.0,
            hv_readback_v=498.5,
            sipm_setpoint_v=None,
            sipm_readback_v=None,
            temperature_analog_c=24.5,
            temperature_digital_c=25.0,
            adc_temperature_c=68.0,
        )


@dataclass(frozen=True)
class _PsdSource:
    def dose_calibration_snapshot(self) -> PsdDoseCalibrationSnapshot:
        return PsdDoseCalibrationSnapshot(
            cut=0.25,
            roi=(100.0, 500.0),
            below_cut_total_counts=900,
            above_cut_total_counts=100,
            below_cut_roi_counts=450,
            above_cut_roi_counts=50,
        )


@dataclass(frozen=True)
class _TemperatureSource:
    def dose_calibration_temperature_snapshot(self) -> DigitizerTemperatureSnapshot:
        return DigitizerTemperatureSnapshot(
            hat_temperature_c=43.2,
            fpga_temperature_c=68.9,
            ads5407_temperature_raw=71.0,
        )


def _mca_snapshot(counts: list[int]) -> McaDoseCalibrationSnapshot:
    return McaDoseCalibrationSnapshot(
        histogram=np.asarray(counts, dtype=np.uint32),
        count_rate_raw=123,
        pulse_deadtime_ms=0.125,
        acquisition_elapsed_s=12.0,
        roi=(1, 2),
    )


def test_dialog_records_committed_dose_and_selected_live_data(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    mca = _McaSource(_mca_snapshot([10, 20, 30]))
    dialog = DoseCalibrationDialog(
        [
            DoseCalibrationChannelSource(
                label="D1:CH0",
                mca=mca,
                psu=_PsuSource(),
                psd=_PsdSource(),
                temperatures=_TemperatureSource(),
            )
        ]
    )
    qtbot.addWidget(dialog)
    path = tmp_path / "dose.csv"
    dialog.output_path.setText(str(path))
    dialog.interval_ms.setValue(60_000)
    dialog.source_label.setText("Cs-137 at 10 cm")
    dialog.dose_rate.setValue(1.25)
    dialog._commit_reference_fields()
    dialog._start_recording()

    mca.snapshot = _mca_snapshot([12, 21, 35])
    dialog._record_row()
    dialog.dose_rate.setValue(2.5)
    mca.snapshot = _mca_snapshot([13, 23, 40])
    dialog._record_row()
    dialog._commit_reference_fields()
    mca.snapshot = _mca_snapshot([14, 25, 45])
    dialog._record_row()
    dialog._stop_recording()

    assert dialog.plot_data_combo.currentData() == "expected_dose_rate"
    plot_times, plotted_doses = dialog.plot_curve.getData()
    assert plot_times is not None
    assert plotted_doses is not None
    assert len(plot_times) == 3
    assert plotted_doses == pytest.approx([1.25, 1.25, 2.5])

    differential_index = dialog.plot_data_combo.findData("mca_differential_total_counts")
    assert differential_index >= 0
    dialog.plot_data_combo.setCurrentIndex(differential_index)
    _, plotted_counts = dialog.plot_curve.getData()
    assert plotted_counts == pytest.approx([8.0, 8.0, 8.0])

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert [float(row["expected_dose_rate"]) for row in rows] == [1.25, 1.25, 2.5]
    assert rows[0]["source_label"] == "Cs-137 at 10 cm"
    assert rows[0]["channel"] == "D1:CH0"
    assert int(rows[0]["mca_differential_total_counts"]) == 8
    assert int(rows[0]["mca_differential_roi_counts"]) == 6
    assert float(rows[0]["hv_readback_v"]) == pytest.approx(498.5)
    assert float(rows[0]["digitizer_fpga_temperature_c"]) == pytest.approx(68.9)
    assert int(rows[0]["psd_below_cut_total_counts"]) == 900
    assert int(rows[0]["psd_above_cut_roi_counts"]) == 50


def test_plot_selector_only_lists_available_selected_data(qtbot: QtBot) -> None:
    dialog = DoseCalibrationDialog(
        [DoseCalibrationChannelSource(label="D1:CH0", mca=_McaSource(_mca_snapshot([1])))]
    )
    qtbot.addWidget(dialog)

    assert dialog.plot_data_combo.findData("expected_dose_rate") >= 0
    assert dialog.plot_data_combo.findData("mca_count_rate_raw") >= 0
    assert dialog.plot_data_combo.findData("hv_readback_v") == -1
    assert dialog.plot_data_combo.findData("psd_above_cut_roi_counts") == -1

    dialog.save_mca_statistics.setChecked(False)
    assert dialog.plot_data_combo.findData("mca_count_rate_raw") == -1
    assert dialog.plot_data_combo.findData("expected_dose_rate") >= 0


def test_dose_calibration_action_is_in_tools_menu(qtbot: QtBot) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.actionDoseCalibration in ui.menuTools.actions()
