"""Modeless CSV logger for reference dose-rate calibration measurements."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.dose_calibration import (
    DigitizerTemperatureSnapshot,
    DoseCalibrationCsvWriter,
    McaDoseCalibrationSnapshot,
    PsdDoseCalibrationSnapshot,
    PsuDoseCalibrationSnapshot,
    differential_spectrum_integrals,
)
from nlab.views.plot_viewbox import NLabPlotWidget, style_plot_widget

log = logging.getLogger(__name__)

_OUTPUT_FOLDER_KEY = "analysis/dose_calibration_folder"
_MAX_PLOT_POINTS = 10_000
_DOSE_UNIT_TO_SV_PER_H = {
    "nSv/h": 1.0e-9,
    "\N{MICRO SIGN}Sv/h": 1.0e-6,
    "mSv/h": 1.0e-3,
    "Sv/h": 1.0,
}
_BASE_FIELDS = (
    "timestamp_utc",
    "elapsed_s",
    "sample_interval_s",
    "channel",
    "source_label",
    "expected_dose_rate",
    "dose_rate_unit",
)
_MCA_STAT_FIELDS = (
    "mca_count_rate_raw",
    "mca_pulse_deadtime_ms",
    "mca_acquisition_elapsed_s",
)
_MCA_DIFFERENTIAL_FIELDS = (
    "mca_differential_total_counts",
    "mca_roi_enabled",
    "mca_roi_low_channel",
    "mca_roi_high_channel",
    "mca_differential_roi_counts",
)
_PSU_FIELDS = (
    "hv_setpoint_v",
    "hv_readback_v",
    "sipm_setpoint_v",
    "sipm_readback_v",
)
_TEMPERATURE_FIELDS = (
    "detector_temperature_analog_c",
    "detector_temperature_digital_c",
    "adc_temperature_c",
    "digitizer_hat_temperature_c",
    "digitizer_fpga_temperature_c",
    "ads5407_temperature_raw",
)
_PSD_FIELDS = (
    "psd_cut",
    "psd_roi_low_energy_raw",
    "psd_roi_high_energy_raw",
    "psd_below_cut_total_counts",
    "psd_above_cut_total_counts",
    "psd_below_cut_roi_counts",
    "psd_above_cut_roi_counts",
)


@dataclass(frozen=True)
class _PlotSeries:
    field: str
    label: str
    axis_label: str
    unit: str | None = None


_PLOT_SERIES = (
    _PlotSeries("expected_dose_rate", "Expected dose rate", "Expected dose rate"),
    _PlotSeries("mca_count_rate_raw", "MCA count rate (raw)", "MCA count rate"),
    _PlotSeries(
        "mca_pulse_deadtime_ms",
        "MCA pulse dead time",
        "MCA pulse dead time",
        "ms",
    ),
    _PlotSeries(
        "mca_acquisition_elapsed_s",
        "MCA acquisition elapsed time",
        "MCA acquisition elapsed time",
        "s",
    ),
    _PlotSeries(
        "mca_differential_total_counts",
        "MCA differential total counts",
        "Differential counts",
    ),
    _PlotSeries("mca_roi_low_channel", "MCA ROI low channel", "MCA channel"),
    _PlotSeries("mca_roi_high_channel", "MCA ROI high channel", "MCA channel"),
    _PlotSeries(
        "mca_differential_roi_counts",
        "MCA differential ROI counts",
        "Differential ROI counts",
    ),
    _PlotSeries("hv_setpoint_v", "HV setpoint", "High voltage", "V"),
    _PlotSeries("hv_readback_v", "HV readback", "High voltage", "V"),
    _PlotSeries("sipm_setpoint_v", "SiPM setpoint", "SiPM voltage", "V"),
    _PlotSeries("sipm_readback_v", "SiPM readback", "SiPM voltage", "V"),
    _PlotSeries(
        "detector_temperature_analog_c",
        "Detector temperature (analogue)",
        "Temperature",
        "\N{DEGREE SIGN}C",
    ),
    _PlotSeries(
        "detector_temperature_digital_c",
        "Detector temperature (digital)",
        "Temperature",
        "\N{DEGREE SIGN}C",
    ),
    _PlotSeries("adc_temperature_c", "ADC temperature", "Temperature", "\N{DEGREE SIGN}C"),
    _PlotSeries(
        "digitizer_hat_temperature_c",
        "Digitizer HAT temperature",
        "Temperature",
        "\N{DEGREE SIGN}C",
    ),
    _PlotSeries(
        "digitizer_fpga_temperature_c",
        "Digitizer FPGA temperature",
        "Temperature",
        "\N{DEGREE SIGN}C",
    ),
    _PlotSeries(
        "ads5407_temperature_raw",
        "ADS5407 temperature (raw)",
        "ADS5407 temperature (raw)",
    ),
    _PlotSeries("psd_cut", "PSD cut", "PSD ratio"),
    _PlotSeries("psd_roi_low_energy_raw", "PSD ROI low energy", "Raw energy"),
    _PlotSeries("psd_roi_high_energy_raw", "PSD ROI high energy", "Raw energy"),
    _PlotSeries(
        "psd_below_cut_total_counts",
        "PSD below-cut total counts",
        "Cumulative counts",
    ),
    _PlotSeries(
        "psd_above_cut_total_counts",
        "PSD above-cut total counts",
        "Cumulative counts",
    ),
    _PlotSeries(
        "psd_below_cut_roi_counts",
        "PSD below-cut ROI counts",
        "Cumulative ROI counts",
    ),
    _PlotSeries(
        "psd_above_cut_roi_counts",
        "PSD above-cut ROI counts",
        "Cumulative ROI counts",
    ),
)
_PLOT_SERIES_BY_FIELD = {series.field: series for series in _PLOT_SERIES}


class _McaSource(Protocol):
    def dose_calibration_snapshot(self) -> McaDoseCalibrationSnapshot | None: ...


class _PsuSource(Protocol):
    def dose_calibration_snapshot(self) -> PsuDoseCalibrationSnapshot: ...


class _PsdSource(Protocol):
    def dose_calibration_snapshot(self) -> PsdDoseCalibrationSnapshot: ...


class _TemperatureSource(Protocol):
    def dose_calibration_temperature_snapshot(
        self,
    ) -> DigitizerTemperatureSnapshot: ...


@dataclass(frozen=True)
class DoseCalibrationChannelSource:
    """Live read-only data providers for one displayed detector channel."""

    label: str
    mca: _McaSource
    psu: _PsuSource | None = None
    psd: _PsdSource | None = None
    temperatures: _TemperatureSource | None = None


class DoseCalibrationDialog(QDialog):
    """Periodically append selected detector observables beside a reference dose."""

    def __init__(
        self,
        sources: list[DoseCalibrationChannelSource],
        parent: QWidget | None = None,
    ) -> None:
        if not sources:
            raise ValueError("dose calibration requires at least one MCA channel")
        super().__init__(parent)
        self._sources = tuple(sources)
        self._writer: DoseCalibrationCsvWriter | None = None
        self._previous_histogram: np.ndarray | None = None
        self._started_monotonic = 0.0
        self._last_sample_monotonic = 0.0
        self._row_count = 0
        self._committed_dose_rate = 0.0
        self._committed_source_label = ""
        self._committed_dose_unit = "\N{MICRO SIGN}Sv/h"
        self._plot_elapsed_s: deque[float] = deque(maxlen=_MAX_PLOT_POINTS)
        self._plot_values: dict[str, deque[float]] = {
            series.field: deque(maxlen=_MAX_PLOT_POINTS) for series in _PLOT_SERIES
        }

        self.setWindowTitle("Dose Calibration Logger")
        self.setModal(False)
        self.setMinimumSize(1_040, 620)
        self.resize(1_240, 700)
        self._build_ui()
        self._connect_signals()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._record_row)
        self._channel_changed()
        self._commit_reference_fields()

    def _build_ui(self) -> None:
        root_layout = QHBoxLayout(self)
        self.control_panel = QWidget(self)
        layout = QVBoxLayout(self.control_panel)
        intro = QLabel(
            "Record a known reference dose rate together with selected live detector "
            "observables. The dose value used in CSV rows changes only after editing "
            "the field is finished.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        reference_group = QGroupBox("Reference measurement", self)
        reference_form = QFormLayout(reference_group)
        self.channel_combo = QComboBox(reference_group)
        for source in self._sources:
            self.channel_combo.addItem(source.label)
        reference_form.addRow("Detector channel:", self.channel_combo)

        self.source_label = QLineEdit(reference_group)
        self.source_label.setPlaceholderText("e.g. Cs-137, 10 cm geometry")
        reference_form.addRow("Source / setup:", self.source_label)

        dose_row = QHBoxLayout()
        self.dose_rate = QDoubleSpinBox(reference_group)
        self.dose_rate.setRange(0.0, 1.0e12)
        self.dose_rate.setDecimals(6)
        self.dose_rate.setKeyboardTracking(False)
        self.dose_rate.setSingleStep(0.1)
        self.dose_unit = QComboBox(reference_group)
        self.dose_unit.addItems(["nSv/h", "\N{MICRO SIGN}Sv/h", "mSv/h", "Sv/h"])
        self.dose_unit.setCurrentText("\N{MICRO SIGN}Sv/h")
        dose_row.addWidget(self.dose_rate, 1)
        dose_row.addWidget(self.dose_unit)
        reference_form.addRow("Expected dose rate:", dose_row)
        self.committed_value = QLabel(reference_group)
        self.committed_value.setStyleSheet("color: #2e7d32;")
        reference_form.addRow("Value used for rows:", self.committed_value)
        layout.addWidget(reference_group)

        data_group = QGroupBox("CSV data columns", self)
        data_layout = QGridLayout(data_group)
        self.save_mca_statistics = QCheckBox("MCA count rate and dead time", data_group)
        self.save_mca_differential = QCheckBox(
            "MCA differential total and visible ROI integrals",
            data_group,
        )
        self.save_psu_voltage = QCheckBox("PSU setpoint and voltage readback", data_group)
        self.save_temperatures = QCheckBox(
            "Detector, ADC, HAT, and FPGA temperatures",
            data_group,
        )
        self.save_psd_integrals = QCheckBox(
            "PSD below/above-cut totals and energy-ROI integrals",
            data_group,
        )
        for checkbox in (
            self.save_mca_statistics,
            self.save_mca_differential,
            self.save_psu_voltage,
            self.save_temperatures,
            self.save_psd_integrals,
        ):
            checkbox.setChecked(True)
        data_layout.addWidget(self.save_mca_statistics, 0, 0)
        data_layout.addWidget(self.save_mca_differential, 1, 0)
        data_layout.addWidget(self.save_psu_voltage, 0, 1)
        data_layout.addWidget(self.save_temperatures, 1, 1)
        data_layout.addWidget(self.save_psd_integrals, 2, 0, 1, 2)
        layout.addWidget(data_group)

        output_group = QGroupBox("CSV recording", self)
        output_layout = QFormLayout(output_group)
        path_row = QHBoxLayout()
        self.output_path = QLineEdit(output_group)
        self.output_path.setPlaceholderText("Choose a new CSV file")
        self.browse_button = QPushButton("Browse...", output_group)
        path_row.addWidget(self.output_path, 1)
        path_row.addWidget(self.browse_button)
        output_layout.addRow("Output file:", path_row)
        self.interval_ms = QSpinBox(output_group)
        self.interval_ms.setRange(100, 3_600_000)
        self.interval_ms.setValue(1_000)
        self.interval_ms.setSingleStep(100)
        self.interval_ms.setSuffix(" ms")
        output_layout.addRow("Sampling interval:", self.interval_ms)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start CSV recording", output_group)
        self.stop_button = QPushButton("Stop", output_group)
        self.start_button.setStyleSheet(
            "QPushButton:enabled { background-color: #45b649; color: white; }"
        )
        self.stop_button.setStyleSheet(
            "QPushButton:enabled { background-color: #ef5350; color: white; }"
        )
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        output_layout.addRow(button_row)
        self.status = QLabel("Choose a new CSV file to begin.", output_group)
        self.status.setWordWrap(True)
        output_layout.addRow(self.status)
        layout.addWidget(output_group)
        layout.addStretch(1)

        plot_group = QGroupBox("Recorded data plot", self)
        plot_layout = QVBoxLayout(plot_group)
        plot_select_row = QHBoxLayout()
        plot_select_row.addWidget(QLabel("Data:", plot_group))
        self.plot_data_combo = QComboBox(plot_group)
        plot_select_row.addWidget(self.plot_data_combo, 1)
        plot_layout.addLayout(plot_select_row)

        self.plot_widget = NLabPlotWidget(plot_group)
        style_plot_widget(self.plot_widget)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.showAxis("top")
        self.plot_widget.showAxis("right")
        self.plot_widget.setLabel("bottom", "Recording elapsed time", units="s")
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen("#1f77b4", width=2.0))
        plot_layout.addWidget(self.plot_widget, 1)
        self.plot_status = QLabel("The plot updates after each recorded CSV row.", plot_group)
        self.plot_status.setWordWrap(True)
        plot_layout.addWidget(self.plot_status)

        root_layout.addWidget(self.control_panel, 1)
        root_layout.addWidget(plot_group, 1)

    def _connect_signals(self) -> None:
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)
        self.dose_rate.editingFinished.connect(self._commit_reference_fields)
        self.source_label.editingFinished.connect(self._commit_reference_fields)
        self.dose_unit.currentTextChanged.connect(self._commit_reference_fields)
        self.plot_data_combo.currentIndexChanged.connect(self._refresh_plot)
        for checkbox in (
            self.save_mca_statistics,
            self.save_mca_differential,
            self.save_psu_voltage,
            self.save_temperatures,
            self.save_psd_integrals,
        ):
            checkbox.toggled.connect(self._refresh_plot_options)
        self.browse_button.clicked.connect(self._browse_output)
        self.start_button.clicked.connect(self._start_recording)
        self.stop_button.clicked.connect(self._stop_recording)

    def _source(self) -> DoseCalibrationChannelSource:
        return self._sources[self.channel_combo.currentIndex()]

    def _channel_changed(self, *_args: object) -> None:
        source = self._source()
        self.save_psu_voltage.setEnabled(source.psu is not None)
        self.save_temperatures.setEnabled(source.psu is not None or source.temperatures is not None)
        self.save_psd_integrals.setEnabled(source.psd is not None)
        self._refresh_plot_options()

    def _commit_reference_fields(self, *_args: object) -> None:
        self._committed_dose_rate = float(self.dose_rate.value())
        self._committed_dose_unit = self.dose_unit.currentText()
        self._committed_source_label = self.source_label.text().strip()
        self.committed_value.setText(f"{self._committed_dose_rate:.6g} {self._committed_dose_unit}")
        self._refresh_plot()

    def _refresh_plot_options(self, *_args: object) -> None:
        previous_field = self.plot_data_combo.currentData()
        available_fields = set(self._fieldnames())
        self.plot_data_combo.blockSignals(True)
        self.plot_data_combo.clear()
        for series in _PLOT_SERIES:
            if series.field in available_fields:
                self.plot_data_combo.addItem(series.label, series.field)
        previous_index = self.plot_data_combo.findData(previous_field)
        self.plot_data_combo.setCurrentIndex(max(previous_index, 0))
        self.plot_data_combo.blockSignals(False)
        self._refresh_plot()

    def _refresh_plot(self, *_args: object) -> None:
        field = self.plot_data_combo.currentData()
        series = _PLOT_SERIES_BY_FIELD.get(field)
        if series is None:
            self.plot_curve.setData([], [])
            return

        values = np.fromiter(
            self._plot_values[field],
            dtype=np.float64,
            count=len(self._plot_values[field]),
        )
        unit = series.unit
        if field == "expected_dose_rate":
            unit = self._committed_dose_unit
            values = values / _DOSE_UNIT_TO_SV_PER_H[unit]
        elapsed = np.fromiter(
            self._plot_elapsed_s,
            dtype=np.float64,
            count=len(self._plot_elapsed_s),
        )
        self.plot_widget.setLabel("left", series.axis_label, units=unit)
        self.plot_curve.setData(elapsed, values)

        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            self.plot_status.setText(
                f"Latest {series.label.lower()}: {finite_values[-1]:.6g}"
                + (f" {unit}" if unit else "")
            )
        elif self._plot_elapsed_s:
            self.plot_status.setText(f"No {series.label.lower()} value is available yet.")
        else:
            self.plot_status.setText("The plot updates after each recorded CSV row.")

    def _clear_plot_history(self) -> None:
        self._plot_elapsed_s.clear()
        for values in self._plot_values.values():
            values.clear()
        self._refresh_plot()

    def _append_plot_sample(self, row: dict[str, object]) -> None:
        self._plot_elapsed_s.append(float(str(row["elapsed_s"])))
        for field, values in self._plot_values.items():
            value = row.get(field, "")
            if field == "expected_dose_rate":
                dose_unit = str(row["dose_rate_unit"])
                value = float(str(value)) * _DOSE_UNIT_TO_SV_PER_H[dose_unit]
            try:
                numeric_value = float(str(value))
            except (TypeError, ValueError):
                numeric_value = np.nan
            values.append(numeric_value)
        self._refresh_plot()

    def _browse_output(self, *_args: object) -> None:
        folder = str(QSettings().value(_OUTPUT_FOLDER_KEY, "measurements"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Create Dose Calibration CSV",
            str(Path(folder) / f"dose_calibration_{timestamp}.csv"),
            "CSV files (*.csv);;All files (*)",
        )
        if selected:
            path = Path(selected)
            if path.suffix.lower() != ".csv":
                path = path.with_suffix(".csv")
            self.output_path.setText(str(path))
            QSettings().setValue(_OUTPUT_FOLDER_KEY, str(path.parent))

    def _fieldnames(self) -> tuple[str, ...]:
        fields = list(_BASE_FIELDS)
        if self.save_mca_statistics.isChecked():
            fields.extend(_MCA_STAT_FIELDS)
        if self.save_mca_differential.isChecked():
            fields.extend(_MCA_DIFFERENTIAL_FIELDS)
        if self.save_psu_voltage.isChecked() and self.save_psu_voltage.isEnabled():
            fields.extend(_PSU_FIELDS)
        if self.save_temperatures.isChecked() and self.save_temperatures.isEnabled():
            fields.extend(_TEMPERATURE_FIELDS)
        if self.save_psd_integrals.isChecked() and self.save_psd_integrals.isEnabled():
            fields.extend(_PSD_FIELDS)
        return tuple(fields)

    def _has_selected_measurements(self) -> bool:
        return len(self._fieldnames()) > len(_BASE_FIELDS)

    def _start_recording(self, *_args: object) -> None:
        if self._writer is not None:
            return
        self._commit_reference_fields()
        path_text = self.output_path.text().strip()
        if not path_text:
            self._browse_output()
            path_text = self.output_path.text().strip()
        if not path_text:
            return
        if not self._has_selected_measurements():
            QMessageBox.information(
                self,
                "Dose Calibration",
                "Select at least one detector measurement for the CSV.",
            )
            return
        path = Path(path_text)
        if path.exists():
            QMessageBox.warning(
                self,
                "Dose Calibration",
                "The selected CSV already exists. Choose a new file so calibration "
                "data are not overwritten.",
            )
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = DoseCalibrationCsvWriter(path, self._fieldnames())
        except Exception as exc:
            log.exception("Could not create dose calibration CSV")
            QMessageBox.critical(self, "Dose Calibration", str(exc))
            self._writer = None
            return

        snapshot = self._source().mca.dose_calibration_snapshot()
        self._previous_histogram = None if snapshot is None else snapshot.histogram.copy()
        self._started_monotonic = time.monotonic()
        self._last_sample_monotonic = self._started_monotonic
        self._row_count = 0
        self._clear_plot_history()
        self._set_recording_controls(True)
        self._timer.start(self.interval_ms.value())
        self.status.setText(f"Recording to {path}. Waiting for the first interval.")

    def _set_recording_controls(self, recording: bool) -> None:
        self.channel_combo.setEnabled(not recording)
        self.output_path.setEnabled(not recording)
        self.browse_button.setEnabled(not recording)
        self.interval_ms.setEnabled(not recording)
        for checkbox in (
            self.save_mca_statistics,
            self.save_mca_differential,
            self.save_psu_voltage,
            self.save_temperatures,
            self.save_psd_integrals,
        ):
            checkbox.setEnabled(not recording)
        if not recording:
            self._channel_changed()
        self.start_button.setEnabled(not recording)
        self.stop_button.setEnabled(recording)

    @staticmethod
    def _value(value: object | None) -> object:
        return "" if value is None else value

    def _record_row(self) -> None:
        writer = self._writer
        if writer is None:
            return
        now_monotonic = time.monotonic()
        source = self._source()
        row: dict[str, object] = {
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "elapsed_s": now_monotonic - self._started_monotonic,
            "sample_interval_s": now_monotonic - self._last_sample_monotonic,
            "channel": source.label,
            "source_label": self._committed_source_label,
            "expected_dose_rate": self._committed_dose_rate,
            "dose_rate_unit": self._committed_dose_unit,
        }
        self._last_sample_monotonic = now_monotonic

        try:
            mca = source.mca.dose_calibration_snapshot()
            if self.save_mca_statistics.isChecked():
                row.update(
                    {
                        "mca_count_rate_raw": "" if mca is None else mca.count_rate_raw,
                        "mca_pulse_deadtime_ms": ("" if mca is None else mca.pulse_deadtime_ms),
                        "mca_acquisition_elapsed_s": (
                            "" if mca is None else mca.acquisition_elapsed_s
                        ),
                    }
                )
            if self.save_mca_differential.isChecked():
                self._add_mca_differential(row, mca)

            psu = source.psu.dose_calibration_snapshot() if source.psu else None
            if self.save_psu_voltage.isChecked() and source.psu is not None:
                row.update(
                    {
                        "hv_setpoint_v": "" if psu is None else psu.hv_setpoint_v,
                        "hv_readback_v": self._value(None if psu is None else psu.hv_readback_v),
                        "sipm_setpoint_v": self._value(
                            None if psu is None else psu.sipm_setpoint_v
                        ),
                        "sipm_readback_v": self._value(
                            None if psu is None else psu.sipm_readback_v
                        ),
                    }
                )
            if self.save_temperatures.isChecked() and (
                source.psu is not None or source.temperatures is not None
            ):
                self._add_temperatures(row, psu, source)
            if self.save_psd_integrals.isChecked() and source.psd is not None:
                self._add_psd_integrals(row, source)
            writer.append(row)
            self._append_plot_sample(row)
        except Exception as exc:
            log.exception("Dose calibration CSV recording failed")
            self._finish_recording(f"Recording stopped after an error: {exc}")
            QMessageBox.critical(self, "Dose Calibration Recording Failed", str(exc))
            return

        self._row_count += 1
        self.status.setText(
            f"Recording {source.label}: {self._row_count:,} row(s) saved to {writer.path}."
        )

    def _add_mca_differential(
        self,
        row: dict[str, object],
        snapshot: McaDoseCalibrationSnapshot | None,
    ) -> None:
        roi = None if snapshot is None else snapshot.roi
        row["mca_roi_enabled"] = roi is not None
        row["mca_roi_low_channel"] = "" if roi is None else roi[0]
        row["mca_roi_high_channel"] = "" if roi is None else roi[1]
        previous = self._previous_histogram
        if snapshot is None:
            row["mca_differential_total_counts"] = ""
            row["mca_differential_roi_counts"] = ""
            return
        self._previous_histogram = snapshot.histogram.copy()
        if previous is None or previous.shape != snapshot.histogram.shape:
            row["mca_differential_total_counts"] = ""
            row["mca_differential_roi_counts"] = ""
            return
        integrals = differential_spectrum_integrals(
            snapshot.histogram,
            previous,
            roi,
        )
        row["mca_differential_total_counts"] = integrals.total_counts
        row["mca_differential_roi_counts"] = self._value(integrals.roi_counts)

    def _add_temperatures(
        self,
        row: dict[str, object],
        psu: PsuDoseCalibrationSnapshot | None,
        source: DoseCalibrationChannelSource,
    ) -> None:
        board = (
            source.temperatures.dose_calibration_temperature_snapshot()
            if source.temperatures is not None
            else None
        )
        row.update(
            {
                "detector_temperature_analog_c": self._value(
                    None if psu is None else psu.temperature_analog_c
                ),
                "detector_temperature_digital_c": self._value(
                    None if psu is None else psu.temperature_digital_c
                ),
                "adc_temperature_c": self._value(None if psu is None else psu.adc_temperature_c),
                "digitizer_hat_temperature_c": self._value(
                    None if board is None else board.hat_temperature_c
                ),
                "digitizer_fpga_temperature_c": self._value(
                    None if board is None else board.fpga_temperature_c
                ),
                "ads5407_temperature_raw": self._value(
                    None if board is None else board.ads5407_temperature_raw
                ),
            }
        )

    def _add_psd_integrals(
        self,
        row: dict[str, object],
        source: DoseCalibrationChannelSource,
    ) -> None:
        snapshot = source.psd.dose_calibration_snapshot() if source.psd else None
        row.update(
            {
                "psd_cut": "" if snapshot is None else snapshot.cut,
                "psd_roi_low_energy_raw": "" if snapshot is None else snapshot.roi[0],
                "psd_roi_high_energy_raw": "" if snapshot is None else snapshot.roi[1],
                "psd_below_cut_total_counts": (
                    "" if snapshot is None else snapshot.below_cut_total_counts
                ),
                "psd_above_cut_total_counts": (
                    "" if snapshot is None else snapshot.above_cut_total_counts
                ),
                "psd_below_cut_roi_counts": (
                    "" if snapshot is None else snapshot.below_cut_roi_counts
                ),
                "psd_above_cut_roi_counts": (
                    "" if snapshot is None else snapshot.above_cut_roi_counts
                ),
            }
        )

    def _stop_recording(self, *_args: object) -> None:
        self._finish_recording(f"Recording stopped; {self._row_count:,} row(s) saved.")

    def _finish_recording(self, message: str) -> None:
        self._timer.stop()
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.close()
        self._previous_histogram = None
        self._set_recording_controls(False)
        self.status.setText(message)

    def show_workspace(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def close_without_prompt(self) -> None:
        if self._writer is not None:
            self._finish_recording(
                f"Recording stopped while the workspace changed; {self._row_count:,} row(s) saved."
            )
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._writer is not None:
            self._stop_recording()
        event.accept()
