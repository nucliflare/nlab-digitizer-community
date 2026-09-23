from __future__ import annotations

import logging
from collections import deque

import pyqtgraph as pg
from nlab_modbus.core.base_modbus_device import BaseModbusDevice
from nlab_modbus.core.register_specs import RegisterSpec, RegisterType
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDoubleSpinBox,
    QHeaderView,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from nlab.ui.ui_external_device_view import Ui_ExternalDeviceView
from nlab.views.responsive_layout import configure_external_layout
from nlab.workers.external_device_worker import ExternalDeviceWorker

log = logging.getLogger(__name__)


def _is_boolean_spec(spec: RegisterSpec) -> bool:
    return spec.scale == 1.0 and spec.min == 0 and spec.max == 1


def _is_service_register(name: str) -> bool:
    """RS-485 diagnostics/config, the static-pass register, and reserved
    (dummy_*) registers are service-mode only — not useful in normal operation.
    """
    return name.startswith("rs485") or name.startswith("dummy") or name == "pass_static"


# Human-readable labels for register names shown in the Settings/Telemetry
# tables. Covers the registers across SiPM, Geiger, and PMT HV PSU maps that
# survive the filters above; anything missing falls back to a generic
# snake_case -> Title Case conversion in _label_for(). Edit freely — these are
# best-effort guesses from the register name/unit, not firmware documentation.
_REGISTER_LABELS: dict[str, str] = {
    # shared across device types
    "cpu_temp": "CPU Temperature",
    "board_temp": "Board Temperature",
    "eeprom_error": "EEPROM Error",
    "supply_voltage": "Supply Voltage",
    "hv_voltage": "HV Voltage",
    "pwm_enable": "HV Enable",
    "pwm_set_voltage": "HV Voltage Setpoint",
    "pwm_duty": "HV PWM Duty Cycle",
    "pwm_cmpss_action": "PWM Comparator Action",
    "vout_ripple": "Output Voltage Ripple",
    "pid_saturation": "PID Saturation",
    # SiPM
    "vout_pwr_en": "Output Enable",
    "vout_set": "Output Voltage Setpoint",
    "sipm_comp_en": "Temp. Compensation Enable",
    "sipm_comp_tref": "Temp. Compensation Reference",
    "sipm_comp_ct": "Temp. Compensation Coefficient",
    "led_drv_enable": "LED Driver Enable",
    "sipm_voltage_10mv": "SiPM Output Voltage",
    "sipm_current_ua": "SiPM Output Current",
    "board_temp_adc": "Board Temperature (ADC)",
    "adc_sipm_vout": "SiPM Output Voltage (ADC)",
    "sipm_board_temp": "SiPM Board Temperature",
    "sipm_correct_voltage_mv": "SiPM Corrected Voltage",
    "vout_supply_fault": "Output Supply Fault",
    "ext_amp_supply_fault": "External Amplifier Supply Fault",
    "led_drv_temp": "LED Driver Temperature",
    "led_drv_status": "LED Driver Status",
    # Geiger
    "pulses_per_sec": "Pulse Rate",
    "dose_level_msvh": "Dose Rate",
    "dose_msv": "Accumulated Dose",
    "pulse_integer": "Pulse Integral",
    "gm_supply_fault": "GM Tube Supply Fault",
    # PMT HV PSU
    "hv_current": "HV Current",
    "pmt_temp": "PMT Temperature",
}


def _label_for(name: str) -> str:
    return _REGISTER_LABELS.get(name, name.replace("_", " ").title())


def _format_register_value(value: int | float) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _value_placeholder(spec: RegisterSpec) -> str:
    """Widest value this register can show, used to pre-size the Value column.

    Telemetry text is filled in after the first poll, but the column width is
    frozen (see _shrink_table_to_contents) right after the table is built —
    sizing it from an empty string would later clip real, wider readings.
    """
    lo = spec.min * spec.scale if spec.scale != 1.0 else spec.min
    hi = spec.max * spec.scale if spec.scale != 1.0 else spec.max
    return max(_format_register_value(lo), _format_register_value(hi), key=len)


class ExternalDeviceController(QWidget):
    """View + controller for one external Modbus device (SiPM / Geiger / PMT HV PSU).

    Loaded from ui_external_device_view.ui. The Settings/Telemetry tables and
    the Monitoring controls are static, defined in the form; only their row
    contents are built generically here from the device's REGISTER_MAP, since
    that set varies per device type and can't be known at design time.
    """

    def __init__(self, device: BaseModbusDevice, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.device = device
        self.ui = Ui_ExternalDeviceView()
        self.ui.setupUi(self)
        configure_external_layout(self, self.ui)

        self._worker: ExternalDeviceWorker | None = None
        self._worker_thread: QThread | None = None
        self._worker_stop_requested = False
        self._row_by_name: dict[str, int] = {}

        self._plot_colors: dict[str, object] = {}
        self._plot_curves: dict[str, pg.PlotDataItem] = {}
        self._plot_ts: dict[str, deque[float]] = {}
        self._plot_values: dict[str, deque[float]] = {}

        self._build_ui()
        self._load_settings()
        self._on_start_monitor()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.ui.lblHeader.setText(
            f"{self.device.device_type.name}  —  {self.device.connection_info()}"
        )

        self._configure_table(self.ui.settingsTable)
        self._configure_table(self.ui.telemetryTable)

        self.ui.btnStartMonitor.clicked.connect(self._on_start_monitor)
        self.ui.btnStopMonitor.clicked.connect(self._on_stop_monitor)
        self.ui.spinRefreshRate.valueChanged.connect(self._on_refresh_rate_changed)

        self.ui.plotWidget.setBackground("#f8f9fa")
        self.ui.plotWidget.showGrid(x=True, y=True, alpha=0.2)
        self.ui.plotWidget.setLabel("bottom", "Time", units="s")
        self.ui.plotWidget.addLegend()

        self._populate_settings_rows()
        self._populate_telemetry_rows()
        self._shrink_table_to_contents(self.ui.settingsTable)
        self._shrink_table_to_contents(self.ui.telemetryTable)
        for row in range(self.ui.telemetryTable.rowCount()):
            item = self.ui.telemetryTable.item(row, 1)
            if item is not None:
                item.setText("")

    @staticmethod
    def _configure_table(table: QTableWidget) -> None:
        table.verticalHeader().setVisible(False)
        header_view = table.horizontalHeader()
        for col in range(table.columnCount()):
            header_view.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

    @staticmethod
    def _shrink_table_to_contents(table: QTableWidget) -> None:
        """Size the table exactly to its rows/columns so it doesn't stretch.

        Without this, QTableWidget's default Expanding size policy makes it
        fill all leftover vertical space in the layout — wasted, since these
        tables have a small, fixed row count set once at construction.

        Columns are measured once with ResizeToContents, then frozen to Fixed
        at that width. Left in ResizeToContents, a cell widget's checked-state
        indicator (e.g. QCheckBox) can paint slightly wider than unchecked,
        and the header reflows every column to fit the still-fixed table
        width — clipping the Register column's text.
        """
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        header_view = table.horizontalHeader()
        widths = [table.columnWidth(c) for c in range(table.columnCount())]
        for col, w in enumerate(widths):
            header_view.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            table.setColumnWidth(col, w)
        width = sum(widths) + table.frameWidth() * 2
        height = table.horizontalHeader().height()
        height += sum(table.rowHeight(r) for r in range(table.rowCount()))
        height += table.frameWidth() * 2
        table.setFixedSize(width, height)

    def _holding_specs(self) -> list[tuple[str, RegisterSpec]]:
        """Holding (writable) registers, excluding password-protected and service ones."""
        items = self.device.REGISTER_MAP.items()
        return [
            (n, s)
            for n, s in items
            if s.reg_type == RegisterType.HOLDING
            and not s.password_protected
            and not _is_service_register(n)
        ]

    def _input_specs(self) -> list[tuple[str, RegisterSpec]]:
        """Input (telemetry) registers covered by the device's read_snapshot().

        hardware_version/firmware_version sit outside the READOUT_START–STOP
        block every device class reads — read_snapshot() never returns them,
        so a polled table would show them as permanently blank. Restricting
        to the snapshot range keeps the table consistent with the live data.
        """
        start = getattr(self.device, "READOUT_START", None)
        stop = getattr(self.device, "READOUT_STOP", None)
        items = self.device.REGISTER_MAP.items()
        specs = [
            (n, s)
            for n, s in items
            if s.reg_type == RegisterType.INPUT and not _is_service_register(n)
        ]
        if start is not None and stop is not None:
            specs = [(n, s) for n, s in specs if start <= s.address < stop]
        return specs

    def _populate_settings_rows(self) -> None:
        specs = self._holding_specs()
        self.ui.settingsTable.setRowCount(len(specs))
        for row, (name, spec) in enumerate(specs):
            self.ui.settingsTable.setItem(row, 0, QTableWidgetItem(_label_for(name)))
            self.ui.settingsTable.setItem(row, 2, QTableWidgetItem(spec.unit or ""))
            self.ui.settingsTable.setCellWidget(row, 1, self._make_editor(name, spec))
            self._row_by_name[f"settings:{name}"] = row

    def _populate_telemetry_rows(self) -> None:
        specs = self._input_specs()
        self.ui.telemetryTable.setRowCount(len(specs))
        for row, (name, spec) in enumerate(specs):
            self.ui.telemetryTable.setItem(row, 0, QTableWidgetItem(_label_for(name)))
            self.ui.telemetryTable.setItem(row, 1, QTableWidgetItem(_value_placeholder(spec)))
            self.ui.telemetryTable.setItem(row, 2, QTableWidgetItem(spec.unit or ""))
            self._row_by_name[f"telemetry:{name}"] = row

            self._plot_colors[name] = pg.intColor(row, hues=max(len(specs), 1))
            checkbox = QCheckBox()
            checkbox.toggled.connect(lambda checked, n=name: self._on_plot_toggled(n, checked))
            self.ui.telemetryTable.setCellWidget(row, 3, checkbox)

    def _on_plot_toggled(self, name: str, checked: bool) -> None:
        if checked:
            pen = pg.mkPen(self._plot_colors[name], width=2)
            self._plot_curves[name] = self.ui.plotWidget.plot(pen=pen, name=_label_for(name))
            self._plot_ts[name] = deque()
            self._plot_values[name] = deque()
        else:
            curve = self._plot_curves.pop(name, None)
            if curve is not None:
                self.ui.plotWidget.removeItem(curve)
            self._plot_ts.pop(name, None)
            self._plot_values.pop(name, None)

    def _make_editor(self, name: str, spec: RegisterSpec) -> QAbstractSpinBox | QCheckBox:
        if _is_boolean_spec(spec):
            box = QCheckBox()
            box.toggled.connect(lambda v, n=name: self._on_setting_changed(n, int(v)))
            return box
        editor: QAbstractSpinBox
        if spec.scale != 1.0:
            double_editor = QDoubleSpinBox()
            double_editor.setDecimals(4)
            double_editor.setRange(spec.min * spec.scale, spec.max * spec.scale)
            double_editor.setSingleStep(spec.scale)
            double_editor.editingFinished.connect(
                lambda n=name, e=double_editor: self._on_setting_changed(n, e.value())
            )
            editor = double_editor
        else:
            int_editor = QSpinBox()
            int_editor.setRange(spec.min, spec.max)
            int_editor.editingFinished.connect(
                lambda n=name, e=int_editor: self._on_setting_changed(n, e.value())
            )
            editor = int_editor
        return editor

    # ------------------------------------------------------------------
    # Initial hardware readout
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        try:
            values = self.device.get_all_holding_registers()
        except Exception:
            log.exception("Failed to read initial settings for %s", self.device.connection_info())
            return
        for name, value in values.items():
            row = self._row_by_name.get(f"settings:{name}")
            if row is None:
                continue
            widget = self.ui.settingsTable.cellWidget(row, 1)
            if widget is None:
                continue
            widget.blockSignals(True)
            try:
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QDoubleSpinBox | QSpinBox):
                    widget.setValue(value)
            finally:
                widget.blockSignals(False)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def _on_setting_changed(self, name: str, value: int | float) -> None:
        log.debug("%s: %s=%s", self.device.connection_info(), name, value)
        try:
            self.device.write(name, value)
        except Exception:
            log.exception("Write failed for %s.%s", self.device.connection_info(), name)

    @property
    def configuration_id(self) -> str:
        """Stable human-readable identifier within one discovered bus."""
        return f"{self.device.device_type.name}:{self.device.device_id}"

    def hardware_configuration_settings(self) -> dict[str, int | float]:
        values: dict[str, int | float] = {}
        for name, _spec in self._holding_specs():
            row = self._row_by_name.get(f"settings:{name}")
            widget = self.ui.settingsTable.cellWidget(row, 1) if row is not None else None
            if isinstance(widget, QCheckBox):
                values[name] = int(widget.isChecked())
            elif isinstance(widget, QDoubleSpinBox | QSpinBox):
                values[name] = widget.value()
        return values

    def apply_hardware_configuration_settings(self, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        specs = dict(self._holding_specs())
        for name, value in settings.items():
            if name not in specs or not isinstance(value, int | float):
                continue
            row = self._row_by_name.get(f"settings:{name}")
            widget = self.ui.settingsTable.cellWidget(row, 1) if row is not None else None
            if widget is not None:
                widget.blockSignals(True)
                try:
                    if isinstance(widget, QCheckBox):
                        widget.setChecked(bool(value))
                    elif isinstance(widget, QDoubleSpinBox):
                        widget.setValue(float(value))
                    elif isinstance(widget, QSpinBox):
                        widget.setValue(int(value))
                finally:
                    widget.blockSignals(False)
            self.device.write(name, value)

    def configuration_settings(self) -> dict[str, object]:
        plotted: list[str] = []
        for name, _spec in self._input_specs():
            row = self._row_by_name.get(f"telemetry:{name}")
            widget = self.ui.telemetryTable.cellWidget(row, 3) if row is not None else None
            if isinstance(widget, QCheckBox) and widget.isChecked():
                plotted.append(name)
        return {
            "refresh_interval_ms": self.ui.spinRefreshRate.value(),
            "plot_time_range_s": self.ui.spinTimeRange.value(),
            "plotted_registers": plotted,
        }

    def apply_configuration_settings(self, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        if "refresh_interval_ms" in settings:
            self.ui.spinRefreshRate.setValue(int(settings["refresh_interval_ms"]))
        if "plot_time_range_s" in settings:
            self.ui.spinTimeRange.setValue(int(settings["plot_time_range_s"]))
        plotted = settings.get("plotted_registers")
        if isinstance(plotted, list):
            selected = {str(name) for name in plotted}
            for name, _spec in self._input_specs():
                row = self._row_by_name.get(f"telemetry:{name}")
                widget = self.ui.telemetryTable.cellWidget(row, 3) if row is not None else None
                if isinstance(widget, QCheckBox):
                    widget.setChecked(name in selected)

    # ------------------------------------------------------------------
    # Polling start / stop
    # ------------------------------------------------------------------

    def start_polling(self, interval_ms: int = 1000) -> None:
        self._worker = ExternalDeviceWorker(self.device, interval_ms=interval_ms)
        self._worker_thread = QThread(self)
        self._worker_stop_requested = False
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.readback.connect(self._on_readback)
        self._worker.finished.connect(
            self._worker_thread.quit,
            Qt.ConnectionType.DirectConnection,
        )
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(self._on_polling_finished)

        self._worker_thread.start()
        log.info("External device polling started: %s", self.device.connection_info())

    def _on_start_monitor(self) -> None:
        self.start_polling(interval_ms=self.ui.spinRefreshRate.value())
        self.ui.btnStartMonitor.setEnabled(False)
        self.ui.btnStopMonitor.setEnabled(True)

    def _on_stop_monitor(self) -> None:
        self.request_polling_stop()
        self.ui.btnStartMonitor.setEnabled(True)
        self.ui.btnStopMonitor.setEnabled(False)

    def _on_refresh_rate_changed(self, value: int) -> None:
        if self._worker is not None:
            self._worker.change_interval.emit(value)

    def _on_polling_finished(self) -> None:
        self._worker = None
        self._worker_thread = None
        self._worker_stop_requested = False

    def request_polling_stop(self) -> None:
        """Ask the Modbus worker to stop without blocking the GUI thread."""
        if self._worker_stop_requested:
            return
        self._worker_stop_requested = True
        worker = self._worker
        if worker is None:
            return
        try:
            worker.request_stop.emit()
        except RuntimeError:
            log.debug("External device worker was already deleted during shutdown")

    def stop_polling_sync(self) -> None:
        """Blocking stop for use during application shutdown only."""
        thread = self._worker_thread
        self.request_polling_stop()
        if thread is not None:
            if not thread.wait(3000):
                log.warning("External device worker thread did not stop in time, terminating")
                thread.terminate()
                thread.wait()
        self._worker_thread = None
        self._worker = None
        self._worker_stop_requested = False

    def _on_readback(self, timestamp: float, snapshot: dict[str, int | float]) -> None:
        for name, value in snapshot.items():
            row = self._row_by_name.get(f"telemetry:{name}")
            if row is None:
                continue
            item = self.ui.telemetryTable.item(row, 1)
            if item is None:
                continue
            item.setText(_format_register_value(value))
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            curve = self._plot_curves.get(name)
            if curve is None:
                continue
            ts = self._plot_ts[name]
            values = self._plot_values[name]
            ts.append(timestamp)
            values.append(value)
            cutoff = timestamp - self.ui.spinTimeRange.value()
            while ts and ts[0] < cutoff:
                ts.popleft()
                values.popleft()
            curve.setData(list(ts), list(values))
