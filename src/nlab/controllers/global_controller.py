from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

from PySide6.QtCore import QSettings, Qt, QThread, QTime
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHeaderView, QTableWidgetItem, QWidget

from nlab.hardware.digitizer.diagnostics import GlobalDiagnosticReading
from nlab.hardware.digitizer.digitizer import Digitizer
from nlab.hardware.digitizer.mca import MCA_PARAMETER_SPECS, MCAParam
from nlab.ui.ui_global_view import Ui_GlobalView
from nlab.views.responsive_layout import configure_global_layout
from nlab.workers.global_diagnostics_worker import GlobalDiagnosticsWorker
from nlab.workers.temperature_correction_worker import (
    T_MAX,
    TemperatureCorrectionReadback,
    TemperatureCorrectionWorker,
)

log = logging.getLogger(__name__)


class GlobalController(QWidget):
    """Global digitizer controls shared by every MCA channel."""

    _INTERVAL_SETTINGS_KEY = "global/diagnostics_interval_ms"
    _TEMPERATURE_INTERVAL_SETTINGS_KEY = "global/temperature_interval_ms"
    _TEMPERATURE_COEFFICIENT_SETTINGS_KEY = "global/temperature_coefficient"
    _TEMPERATURE_OFFSET_SETTINGS_KEY = "global/temperature_offset"
    _TEMPERATURE_ENABLED_SETTINGS_KEY = "global/temperature_enabled"

    def __init__(
        self,
        devices: Sequence[Digitizer],
        channel_labels: Sequence[int] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        if not devices:
            raise ValueError("GlobalController requires at least one digitizer channel")
        super().__init__(parent)

        # The first channel owns the shared sync core. Global diagnostics and
        # temperature correction may use any channel's IDS context because
        # IIO firmware can expose IDS hardware for only a subset of channels.
        self.device = next(
            (device for device in devices if device.hv is not None), devices[0]
        )
        self._devices = tuple(devices)
        self._channel_labels = tuple(
            channel_labels if channel_labels is not None else range(len(devices))
        )
        if len(self._channel_labels) != len(self._devices):
            raise ValueError("channel_labels must match the number of devices")
        self._sync = devices[0].mca.sync

        self.ui = Ui_GlobalView()
        self.ui.setupUi(self)  # type: ignore[no-untyped-call]
        configure_global_layout(self, self.ui)
        self.ui.tableDiagnostics.verticalHeader().setVisible(False)
        header = self.ui.tableDiagnostics.horizontalHeader()
        for column in range(self.ui.tableDiagnostics.columnCount()):
            header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )

        self._worker: GlobalDiagnosticsWorker | None = None
        self._worker_thread: QThread | None = None
        self._worker_stop_requested = False
        self._temperature_worker: TemperatureCorrectionWorker | None = None
        self._temperature_worker_thread: QThread | None = None
        self._temperature_worker_stop_requested = False
        self._reset_sync_for_initialization()
        self._sync_available = self._load_sync_state()
        self._load_settings()
        self._temperature_available = self.device.hv is not None and all(
            device.mca_available() for device in self._devices
        )
        self.ui.groupTemperatureCorrection.setEnabled(self._temperature_available)
        if not self._temperature_available:
            self._set_temperature_status(
                "Temperature correction unavailable: ADS5407 or MCA hardware is missing.",
                error=True,
            )
        self._connect_signals()
        self._start_diagnostics()
        if self._temperature_available and self.ui.cbTemperatureCorrectionEnabled.isChecked():
            self._start_temperature_correction()

    def _load_settings(self) -> None:
        settings = QSettings()
        value = cast(
            int,
            settings.value(self._INTERVAL_SETTINGS_KEY, 1000, type=int),
        )
        self.ui.spinDiagnosticsInterval.setValue(value)
        specs = MCA_PARAMETER_SPECS
        coefficient = cast(
            float,
            settings.value(
                self._TEMPERATURE_COEFFICIENT_SETTINGS_KEY,
                float(specs[MCAParam.TEMP_COEFF].default),
                type=float,
            ),
        )
        offset = cast(
            int,
            settings.value(
                self._TEMPERATURE_OFFSET_SETTINGS_KEY,
                int(specs[MCAParam.TEMP_OFFSET].default),
                type=int,
            ),
        )
        interval = cast(
            int,
            settings.value(
                self._TEMPERATURE_INTERVAL_SETTINGS_KEY,
                1000,
                type=int,
            ),
        )
        enabled = cast(
            bool,
            settings.value(
                self._TEMPERATURE_ENABLED_SETTINGS_KEY,
                True,
                type=bool,
            ),
        )
        self.ui.spinTempCoeff.setValue(coefficient)
        self.ui.spinTempOffset.setValue(offset)
        self.ui.spinTemperatureInterval.setValue(interval)
        self.ui.cbTemperatureCorrectionEnabled.setChecked(enabled)
        self.ui.lblTemperatureMax.setText(f"{T_MAX:g} raw")

    def _load_sync_state(self) -> bool:
        try:
            source = self._sync.get_trig_src()
            enabled = self._sync.get_enable()
            software_state = self._sync.get_sw_trig()
        except Exception as exc:
            self.ui.groupSoftwareStart.setEnabled(False)
            self._set_sync_status(f"Trigger sync unavailable: {exc}", error=True)
            return False

        self.ui.comboSyncSource.blockSignals(True)
        self.ui.comboSyncSource.setCurrentIndex(source)
        self.ui.comboSyncSource.blockSignals(False)
        self.ui.cbSyncEnable.blockSignals(True)
        self.ui.cbSyncEnable.setChecked(enabled)
        self.ui.cbSyncEnable.blockSignals(False)
        self.ui.comboSyncSource.setEnabled(not enabled)
        self.ui.lblSoftwareState.setText("HIGH" if software_state else "LOW")
        self._set_sync_status("Shared trigger state loaded.")
        return True

    def _reset_sync_for_initialization(self) -> None:
        """Leave the shared start output gated off and LOW on every launch."""
        try:
            self._sync.set_enable(False)
            self._sync.set_sw_trig(0)
        except Exception:
            # _load_sync_state() below supplies the user-facing unavailable
            # state. Keep this reset best-effort for older firmware.
            log.warning("Could not reset shared sync state during startup", exc_info=True)

    def _connect_signals(self) -> None:
        if self._sync_available:
            self.ui.comboSyncSource.currentIndexChanged.connect(
                self._on_sync_source_changed,
            )
            self.ui.cbSyncEnable.toggled.connect(self._on_sync_enable_changed)
            self.ui.btnPrepareSoftware.clicked.connect(self._prepare_software_sync)
            self.ui.btnSoftwareStart.clicked.connect(self._start_armed_channels)
            self.ui.btnSoftwareReset.clicked.connect(self._reset_software_level)
        if self._temperature_available:
            self.ui.spinTempCoeff.editingFinished.connect(
                self._on_temperature_coefficient_changed,
            )
            self.ui.spinTempOffset.editingFinished.connect(
                self._on_temperature_offset_changed,
            )
            self.ui.spinTemperatureInterval.valueChanged.connect(
                self._on_temperature_interval_changed,
            )
            self.ui.cbTemperatureCorrectionEnabled.toggled.connect(
                self._on_temperature_correction_toggled,
            )
        self.ui.spinDiagnosticsInterval.valueChanged.connect(
            self._on_diagnostics_interval_changed,
        )

    def _set_temperature_status(self, message: str, *, error: bool = False) -> None:
        self.ui.lblTemperatureStatus.setText(message)
        color = "#b00020" if error else "#2e7d32"
        self.ui.lblTemperatureStatus.setStyleSheet(f"color: {color};")

    def _on_temperature_coefficient_changed(self) -> None:
        QSettings().setValue(
            self._TEMPERATURE_COEFFICIENT_SETTINGS_KEY,
            self.ui.spinTempCoeff.value(),
        )
        self._send_temperature_parameters()

    def _on_temperature_offset_changed(self) -> None:
        QSettings().setValue(
            self._TEMPERATURE_OFFSET_SETTINGS_KEY,
            self.ui.spinTempOffset.value(),
        )
        self._send_temperature_parameters()

    def _send_temperature_parameters(self) -> None:
        worker = self._temperature_worker
        if worker is not None:
            worker.change_parameters.emit(
                self.ui.spinTempCoeff.value(),
                self.ui.spinTempOffset.value(),
            )

    def _on_temperature_interval_changed(self, interval_ms: int) -> None:
        QSettings().setValue(
            self._TEMPERATURE_INTERVAL_SETTINGS_KEY,
            interval_ms,
        )
        if self._temperature_worker is not None:
            self._temperature_worker.change_interval.emit(interval_ms)

    def _on_temperature_correction_toggled(self, enabled: bool) -> None:
        QSettings().setValue(self._TEMPERATURE_ENABLED_SETTINGS_KEY, enabled)
        if enabled:
            self._start_temperature_correction()
        else:
            self._stop_temperature_correction_sync()
            self._set_temperature_status(
                "Temperature-correction loop disabled; the last correction remains applied.",
            )

    def _start_temperature_correction(self) -> None:
        if self._temperature_worker is not None:
            return
        hv = self.device.hv
        if hv is None:
            self._set_temperature_status(
                "Temperature correction unavailable: ADS5407 backend is missing.",
                error=True,
            )
            return
        worker = TemperatureCorrectionWorker(
            hv,
            [device.mca for device in self._devices],
            self.ui.spinTempCoeff.value(),
            self.ui.spinTempOffset.value(),
            self.ui.spinTemperatureInterval.value(),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.readback.connect(self._on_temperature_readback)
        worker.error.connect(self._on_temperature_error)
        # shutdown() synchronously waits in the GUI thread. QThread itself
        # also lives there, so an AutoConnection would queue quit() behind
        # that blocking wait and consume the entire timeout. quit() is
        # thread-safe; invoke it directly when the worker finishes.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            self._on_temperature_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._temperature_worker = worker
        self._temperature_worker_thread = thread
        self._temperature_worker_stop_requested = False
        thread.start()
        self._set_temperature_status("Temperature-correction loop started.")

    def _on_temperature_readback(
        self,
        readback: TemperatureCorrectionReadback,
    ) -> None:
        self.ui.lblAdsTemperature.setText(f"{readback.temperature:g} raw")
        self.ui.lblAppliedTempCoeff.setText(
            f"{readback.applied_coefficient:.9f}",
        )
        self.ui.lblAppliedTempOffset.setText(str(readback.applied_offset))
        self._set_temperature_status(
            f"Correction applied to {len(self._devices)} MCA channel(s).",
        )

    def _on_temperature_error(self, message: str) -> None:
        self._set_temperature_status(
            f"Temperature-correction cycle failed: {message}",
            error=True,
        )

    def _on_temperature_finished(self) -> None:
        self._temperature_worker = None
        self._temperature_worker_thread = None
        self._temperature_worker_stop_requested = False

    def _stop_temperature_correction_sync(self) -> None:
        thread = self._temperature_worker_thread
        self._request_temperature_correction_stop()
        if thread is not None and not thread.wait(5000):
            log.warning("Waiting for in-flight temperature-correction cycle")
            thread.quit()
            thread.wait()
        self._temperature_worker = None
        self._temperature_worker_thread = None
        self._temperature_worker_stop_requested = False

    def _request_temperature_correction_stop(self) -> None:
        if self._temperature_worker_stop_requested:
            return
        self._temperature_worker_stop_requested = True
        # Keep the Python wrapper alive until _stop_temperature_correction_sync()
        # joins its QThread. Dropping the last reference here can destroy the
        # worker-owned QTimer from this GUI thread and trigger Qt's
        # "Timers cannot be stopped from another thread" warning.
        worker = self._temperature_worker
        if worker is None:
            return
        try:
            worker.request_shutdown()
        except RuntimeError:
            # Idempotent shutdown: a queued thread-finished callback may not
            # yet have cleared a wrapper whose C++ QObject is already gone.
            log.debug("Temperature worker was already deleted during shutdown")

    def _request_diagnostics_stop(self) -> None:
        if self._worker_stop_requested:
            return
        self._worker_stop_requested = True
        worker = self._worker
        if worker is None:
            return
        try:
            worker.request_shutdown()
        except RuntimeError:
            log.debug("Diagnostics worker was already deleted during shutdown")

    def request_polling_stop(self) -> None:
        """Signal both global workers without waiting for either one."""
        self._request_temperature_correction_stop()
        self._request_diagnostics_stop()

    def refresh_temperature_state(self) -> None:
        """Reapply the global base values after channel settings are loaded."""
        self._send_temperature_parameters()

    def hardware_configuration_settings(self) -> dict[str, int | bool]:
        """Return the user-configurable state of the shared trigger core."""
        if not self._sync_available:
            return {}
        return {
            "trigger_source": self._sync.get_trig_src(),
            "enabled": self._sync.get_enable(),
        }

    def apply_hardware_configuration_settings(self, settings: object) -> None:
        if not self._sync_available or not isinstance(settings, dict):
            return
        # Gate first, configure the source, and only then restore the requested
        # gate. The software trigger level deliberately remains LOW.
        self._sync.set_enable(False)
        self._sync.set_sw_trig(0)
        if "trigger_source" in settings:
            self._sync.set_trig_src(int(settings["trigger_source"]))
        if "enabled" in settings:
            self._sync.set_enable(bool(settings["enabled"]))
        self._load_sync_state()

    def configuration_settings(self) -> dict[str, int | float | bool]:
        """Return global settings implemented by GUI workers."""
        return {
            "diagnostics_interval_ms": self.ui.spinDiagnosticsInterval.value(),
            "temperature_correction_enabled": (self.ui.cbTemperatureCorrectionEnabled.isChecked()),
            "temperature_coefficient": self.ui.spinTempCoeff.value(),
            "temperature_offset": self.ui.spinTempOffset.value(),
            "temperature_interval_ms": self.ui.spinTemperatureInterval.value(),
        }

    def apply_configuration_settings(self, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        if "diagnostics_interval_ms" in settings:
            self.ui.spinDiagnosticsInterval.setValue(int(settings["diagnostics_interval_ms"]))
        if "temperature_coefficient" in settings:
            self.ui.spinTempCoeff.setValue(float(settings["temperature_coefficient"]))
            self._on_temperature_coefficient_changed()
        if "temperature_offset" in settings:
            self.ui.spinTempOffset.setValue(int(settings["temperature_offset"]))
            self._on_temperature_offset_changed()
        if "temperature_interval_ms" in settings:
            self.ui.spinTemperatureInterval.setValue(int(settings["temperature_interval_ms"]))
        if "temperature_correction_enabled" in settings:
            self.ui.cbTemperatureCorrectionEnabled.setChecked(
                bool(settings["temperature_correction_enabled"])
            )

    def _set_sync_status(self, message: str, *, error: bool = False) -> None:
        self.ui.lblSyncStatus.setText(message)
        color = "#b00020" if error else "#2e7d32"
        self.ui.lblSyncStatus.setStyleSheet(f"color: {color};")

    def _on_sync_source_changed(self, source: int) -> None:
        try:
            self._sync.set_trig_src(source)
        except Exception as exc:
            self._load_sync_state()
            self._set_sync_status(f"Cannot change start source: {exc}", error=True)
            return
        self._set_sync_status(
            "Start source set to software."
            if source == 0
            else "Start source set to synchronized hardware input.",
        )

    def _on_sync_enable_changed(self, enabled: bool) -> None:
        try:
            self._sync.set_enable(enabled)
        except Exception as exc:
            self._load_sync_state()
            self._set_sync_status(f"Cannot change common gate: {exc}", error=True)
            return
        self.ui.comboSyncSource.setEnabled(not enabled)
        self._set_sync_status(
            "Common output gate enabled." if enabled else "Common output gate disabled.",
        )

    def _prepare_software_sync(self) -> None:
        """Establish disabled/source-software/LOW/enabled in the safe order."""
        try:
            self._sync.set_enable(False)
            self._sync.set_sw_trig(0)
            self._sync.set_trig_src(0)
            self._sync.set_enable(True)
        except Exception as exc:
            self._load_sync_state()
            self._set_sync_status(f"Software-sync preparation failed: {exc}", error=True)
            return
        self._load_sync_state()
        self._set_sync_status(
            "Software sync prepared at LOW. Arm every MCA channel, then press Start.",
        )

    def set_coincidence_locked(self, locked: bool) -> None:
        """Reserve manual shared-start controls for a two-channel session."""
        for control in (
            self.ui.comboSyncSource,
            self.ui.cbSyncEnable,
            self.ui.btnPrepareSoftware,
            self.ui.btnSoftwareStart,
            self.ui.btnSoftwareReset,
        ):
            control.setEnabled(not locked)
        if not locked:
            self._load_sync_state()

    @property
    def sync_available(self) -> bool:
        return self._sync_available

    def _arming_errors(self) -> list[str]:
        errors: list[str] = []
        for label, device in zip(self._channel_labels, self._devices, strict=True):
            if not device.mca.get_global_enable():
                errors.append(f"Ch {label} is not armed")
            if not device.mca.get_ext_trig_enable():
                errors.append(f"Ch {label} has External Trigger disabled")
        return errors

    def _start_armed_channels(self) -> None:
        try:
            if self._sync.get_trig_src() != 0:
                raise RuntimeError("shared start source is not Software")
            if not self._sync.get_enable():
                raise RuntimeError("common output gate is disabled")
            if self._sync.get_sw_trig() != 0:
                raise RuntimeError("software level is already HIGH; reset it before rearming")
            arming_errors = self._arming_errors()
            if arming_errors:
                raise RuntimeError("; ".join(arming_errors))
            self._sync.set_sw_trig(1)
        except Exception as exc:
            self._set_sync_status(f"Synchronized start blocked: {exc}", error=True)
            return

        self.ui.lblSoftwareState.setText("HIGH")
        self._set_sync_status("Software start asserted for all armed MCA channels.")
        log.info("Global software start asserted for %d MCA channel(s)", len(self._devices))

    def _reset_software_level(self) -> None:
        try:
            self._sync.set_sw_trig(0)
        except Exception as exc:
            self._set_sync_status(f"Cannot reset software level: {exc}", error=True)
            return
        self.ui.lblSoftwareState.setText("LOW")
        self._set_sync_status("Software start level reset to LOW.")

    def disarm_sync(self) -> None:
        """Best-effort shutdown reset: gate OFF first, then software LOW."""
        if not self._sync_available:
            return
        try:
            self._sync.set_enable(False)
            self._sync.set_sw_trig(0)
        except Exception:
            log.warning("Failed to disarm shared MCA sync during shutdown", exc_info=True)
            return
        self.ui.cbSyncEnable.blockSignals(True)
        self.ui.cbSyncEnable.setChecked(False)
        self.ui.cbSyncEnable.blockSignals(False)
        self.ui.comboSyncSource.setEnabled(True)
        self.ui.lblSoftwareState.setText("LOW")

    def _start_diagnostics(self) -> None:
        interval_ms = self.ui.spinDiagnosticsInterval.value()
        self._worker = GlobalDiagnosticsWorker(self.device, interval_ms)
        self._worker_thread = QThread(self)
        self._worker_stop_requested = False
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.readback.connect(self._on_diagnostics_readback)
        self._worker.error.connect(self._on_diagnostics_error)
        self._worker.finished.connect(
            self._worker_thread.quit,
            Qt.ConnectionType.DirectConnection,
        )
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(
            self._on_diagnostics_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker_thread.start()

    def _on_diagnostics_interval_changed(self, interval_ms: int) -> None:
        QSettings().setValue(self._INTERVAL_SETTINGS_KEY, interval_ms)
        if self._worker is not None:
            self._worker.change_interval.emit(interval_ms)

    def _on_diagnostics_readback(
        self,
        readings: list[GlobalDiagnosticReading],
    ) -> None:
        table = self.ui.tableDiagnostics
        table.setRowCount(len(readings))
        for row, reading in enumerate(readings):
            value = self._format_diagnostic_value(reading)
            items = (
                QTableWidgetItem(reading.label),
                QTableWidgetItem(value),
                QTableWidgetItem(reading.unit),
            )
            if reading.healthy is not None:
                color = QColor("#dff0d8" if reading.healthy else "#f8d7da")
                for item in items:
                    item.setBackground(color)
            for column, item in enumerate(items):
                table.setItem(row, column, item)
        table.resizeRowsToContents()
        table.resizeColumnsToContents()
        height = table.horizontalHeader().height()
        height += sum(table.rowHeight(row) for row in range(table.rowCount()))
        height += table.frameWidth() * 2 + 2
        table.setFixedHeight(height)
        width = sum(table.columnWidth(column) for column in range(table.columnCount()))
        width += table.frameWidth() * 2 + 2
        table.setFixedWidth(width)
        self.ui.lblLastUpdate.setText(QTime.currentTime().toString("HH:mm:ss"))

    @staticmethod
    def _format_diagnostic_value(reading: GlobalDiagnosticReading) -> str:
        if isinstance(reading.value, bool):
            return "Yes" if reading.value else "No"
        if isinstance(reading.value, float):
            return f"{reading.value:.{reading.precision}f}"
        return f"{reading.value:,}"

    def _on_diagnostics_error(self, message: str) -> None:
        self.ui.lblLastUpdate.setText(f"Read error: {message}")
        self.ui.lblLastUpdate.setStyleSheet("color: #b00020;")

    def _on_diagnostics_finished(self) -> None:
        self._worker = None
        self._worker_thread = None
        self._worker_stop_requested = False

    def stop_polling_sync(self) -> None:
        """Stop both global workers before closing their device contexts."""
        # Start both shutdowns together. If each is inside a synchronous
        # network read, their bounded completion times now overlap instead of
        # being paid serially.
        self.request_polling_stop()
        self._stop_temperature_correction_sync()
        thread = self._worker_thread
        self._request_diagnostics_stop()
        if thread is not None and not thread.wait(5000):
            # A remote libiio attribute read is synchronous.  Do not let the
            # application close its contexts underneath that call; request a
            # clean event-loop exit and wait for the current read to return.
            log.warning("Waiting for in-flight global diagnostics read to finish")
            thread.quit()
            thread.wait()
        self._worker = None
        self._worker_thread = None
        self._worker_stop_requested = False
