from __future__ import annotations

import logging
from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QWidget

from nlab.hardware.digitizer.hv import HV_PARAMETER_SPECS, HVParam, HVSupply
from nlab.hardware.digitizer.scope import RangeSpec
from nlab.ui.ui_psu_view import Ui_PSUView
from nlab.views.responsive_layout import configure_psu_layout
from nlab.workers.psu_worker import PSUReadback, PSUWorker

log = logging.getLogger(__name__)


class PSUController(QWidget):
    """View + controller for a single channel's power supply (IDS subsystem).

    Loaded from ui_psu_view.ui.  Accepts an HVSupply instance which wraps
    the IDS gRPC backend for SiPM/HV bias and temperature monitoring.
    """

    def __init__(self, hv: HVSupply, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.hv = hv
        self.ui = Ui_PSUView()
        self.ui.setupUi(self)
        configure_psu_layout(self, self.ui)

        self._sipm_available = hv.sipm_available()
        if not self._sipm_available:
            self.ui.groupSipm.setVisible(False)
            self.ui.groupSipmCompens.setVisible(False)

        self._ts: deque[float] = deque()
        self._voltages: deque[float] = deque()

        self._worker: PSUWorker | None = None
        self._worker_thread: QThread | None = None
        self._worker_stop_requested = False

        self._setup_plot()
        self._apply_parameter_specs()
        self._send_defaults()
        self._load_hardware_state()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Plot setup
    # ------------------------------------------------------------------

    def _setup_plot(self) -> None:
        self.ui.plotHvVoltage.setBackground("#f8f9fa")
        self.ui.plotHvVoltage.showGrid(x=True, y=True, alpha=0.2)
        self.ui.plotHvVoltage.setLabel("left", "HV Voltage", units="V")
        self.ui.plotHvVoltage.setLabel("bottom", "Time", units="s")
        self._plot_curve = self.ui.plotHvVoltage.plot(pen=pg.mkPen("#e04040", width=2))

    # ------------------------------------------------------------------
    # Spec application — runs before signals are connected
    # ------------------------------------------------------------------

    def _apply_parameter_specs(self) -> None:
        specs = HV_PARAMETER_SPECS

        self._apply_range_to_double_spinbox(self.ui.spinHvVoltage, specs[HVParam.HV_VOLTAGE])
        self._apply_range_to_double_spinbox(self.ui.spinHvCompensCt, specs[HVParam.HV_COMPENS_CT])
        self._apply_range_to_double_spinbox(
            self.ui.spinHvCompensTref, specs[HVParam.HV_COMPENS_TREF]
        )

        if self._sipm_available:
            self._apply_range_to_double_spinbox(
                self.ui.spinSipmVoltage, specs[HVParam.SIPM_VOLTAGE]
            )
            self._apply_range_to_double_spinbox(
                self.ui.spinSipmCompensCt, specs[HVParam.SIPM_COMPENS_CT]
            )
            self._apply_range_to_double_spinbox(
                self.ui.spinSipmCompensTref, specs[HVParam.SIPM_COMPENS_TREF]
            )

    @staticmethod
    def _apply_range_to_double_spinbox(spinbox, spec: RangeSpec) -> None:
        spinbox.setMinimum(spec.min_val)
        spinbox.setMaximum(spec.max_val)
        spinbox.setSingleStep(spec.step)
        spinbox.setValue(spec.default)

    # ------------------------------------------------------------------
    # Write defaults to hardware, then read back
    # ------------------------------------------------------------------

    def _send_defaults(self) -> None:
        specs = HV_PARAMETER_SPECS
        try:
            self.hv.set_hv_voltage(specs[HVParam.HV_VOLTAGE].default)
            self.hv.set_hv_compens_ct(specs[HVParam.HV_COMPENS_CT].default)
            self.hv.set_hv_compens_tref(specs[HVParam.HV_COMPENS_TREF].default)
            self.hv.set_hv_compens_mode(specs[HVParam.HV_COMPENS_MODE].default)
            self.hv.set_temp_digital_enable(specs[HVParam.TEMP_DIGITAL_ENABLE].default)
            if self._sipm_available:
                self.hv.set_sipm_enable(specs[HVParam.SIPM_ENABLE].default)
                self.hv.set_sipm_voltage(specs[HVParam.SIPM_VOLTAGE].default)
                self.hv.set_sipm_compens_ct(specs[HVParam.SIPM_COMPENS_CT].default)
                self.hv.set_sipm_compens_tref(specs[HVParam.SIPM_COMPENS_TREF].default)
                self.hv.set_sipm_compens_mode(specs[HVParam.SIPM_COMPENS_MODE].default)
        except Exception:
            log.exception("Failed to send HV defaults")

    def _load_hardware_state(self) -> None:
        try:
            self.ui.spinHvVoltage.setValue(self.hv.get_hv_adc_voltage())
            self.ui.lblHvVoltage.setText(f"{self.hv.get_hv_adc_voltage():.1f}")
            self.ui.lblHvCompensOutput.setText(f"{self.hv.get_hv_compens_output():.1f}")

            self.ui.lblTempAnalog.setText(f"{self.hv.get_temp_analog():.1f}")
            self.ui.lblTempDigital.setText(f"{self.hv.get_temp_digital():.1f}")
            self.ui.lblTempDigitalStatus.setText(str(self.hv.get_temp_digital_status()))
            self.ui.lblAdsTemp.setText(f"{self.hv.get_ads_temp():.1f}")

            if self._sipm_available:
                self.ui.spinSipmVoltage.setValue(self.hv.get_sipm_adc_voltage())
                self.ui.lblSipmVoltage.setText(f"{self.hv.get_sipm_adc_voltage():.2f}")
                self.ui.lblSipmCurrent.setText(f"{self.hv.get_sipm_adc_current():.4f}")
                self.ui.lblSipmOverload.setText(str(self.hv.get_sipm_overload()))
                self.ui.lblSipmCompensOutput.setText(f"{self.hv.get_sipm_compens_output():.2f}")
        except Exception:
            log.exception("Failed to read initial HV state")

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        if self._sipm_available:
            self.ui.cbSipmEnable.toggled.connect(
                lambda v: (log.debug("PSU: sipm_enable=%s", v), self.hv.set_sipm_enable(int(v)))
            )
            self.ui.spinSipmVoltage.editingFinished.connect(self._on_sipm_voltage_changed)
            self.ui.comboSipmCompensMode.currentIndexChanged.connect(
                lambda i: (
                    log.debug("PSU: sipm_compens_mode=%d", i),
                    self.hv.set_sipm_compens_mode(i),
                )
            )
            self.ui.spinSipmCompensCt.editingFinished.connect(
                lambda: (
                    log.debug("PSU: sipm_compens_ct=%.4f", self.ui.spinSipmCompensCt.value()),
                    self.hv.set_sipm_compens_ct(self.ui.spinSipmCompensCt.value()),
                )
            )
            self.ui.spinSipmCompensTref.editingFinished.connect(
                lambda: (
                    log.debug("PSU: sipm_compens_tref=%.2f", self.ui.spinSipmCompensTref.value()),
                    self.hv.set_sipm_compens_tref(self.ui.spinSipmCompensTref.value()),
                )
            )

        self.ui.spinHvVoltage.editingFinished.connect(self._on_hv_voltage_changed)

        self.ui.comboHvCompensMode.currentIndexChanged.connect(
            lambda i: (log.debug("PSU: hv_compens_mode=%d", i), self.hv.set_hv_compens_mode(i))
        )
        self.ui.spinHvCompensCt.editingFinished.connect(
            lambda: (
                log.debug("PSU: hv_compens_ct=%.4f", self.ui.spinHvCompensCt.value()),
                self.hv.set_hv_compens_ct(self.ui.spinHvCompensCt.value()),
            )
        )
        self.ui.spinHvCompensTref.editingFinished.connect(
            lambda: (
                log.debug("PSU: hv_compens_tref=%.2f", self.ui.spinHvCompensTref.value()),
                self.hv.set_hv_compens_tref(self.ui.spinHvCompensTref.value()),
            )
        )

        self.ui.comboTempDigitalEnable.currentIndexChanged.connect(
            lambda i: (
                log.debug("PSU: temp_digital_enable=%d", i),
                self.hv.set_temp_digital_enable(i),
            )
        )

        self.ui.btnStartMonitor.clicked.connect(self._on_start_monitor)
        self.ui.btnStopMonitor.clicked.connect(self._on_stop_monitor)
        self.ui.spinRefreshRate.valueChanged.connect(self._on_refresh_rate_changed)

    # ------------------------------------------------------------------
    # Monitoring start / stop
    # ------------------------------------------------------------------

    def _on_start_monitor(self) -> None:
        self._ts.clear()
        self._voltages.clear()

        interval_ms = self.ui.spinRefreshRate.value()
        self._worker = PSUWorker(
            self.hv,
            read_sipm=self._sipm_available,
            interval_ms=interval_ms,
        )
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
        self._worker_thread.finished.connect(
            self._on_monitor_finished,
            Qt.ConnectionType.QueuedConnection,
        )

        self.ui.btnStartMonitor.setEnabled(False)
        self.ui.btnStopMonitor.setEnabled(True)
        self._worker_thread.start()
        log.info(
            "PSU: monitoring started (interval %d ms, sipm=%s)", interval_ms, self._sipm_available
        )

    def _on_stop_monitor(self) -> None:
        self.request_monitor_stop()
        self.ui.btnStartMonitor.setEnabled(True)
        self.ui.btnStopMonitor.setEnabled(False)
        log.info("PSU: monitoring stop requested")

    def _on_monitor_finished(self) -> None:
        self._worker = None
        self._worker_thread = None
        self._worker_stop_requested = False
        log.info("PSU: monitoring stopped")

    def request_monitor_stop(self) -> None:
        """Ask the PSU monitor to stop without waiting for its QThread."""
        if self._worker_stop_requested:
            return
        self._worker_stop_requested = True
        worker = self._worker
        if worker is None:
            return
        try:
            worker.request_stop.emit()
        except RuntimeError:
            log.debug("PSU worker was already deleted during shutdown")

    def stop_monitor_sync(self) -> None:
        """Blocking stop for use during application shutdown only."""
        thread = self._worker_thread
        self.request_monitor_stop()
        if thread is not None:
            if not thread.wait(3000):
                log.warning("PSU worker thread did not stop in time, terminating")
                thread.terminate()
                thread.wait()
        self._worker_thread = None
        self._worker = None
        self._worker_stop_requested = False

    def _on_refresh_rate_changed(self, value: int) -> None:
        if self._worker is not None:
            self._worker.change_interval.emit(value)

    def _on_hv_voltage_changed(self) -> None:
        val = self.ui.spinHvVoltage.value()
        log.debug("PSU: hv_voltage=%.2f", val)
        self.hv.set_hv_voltage(val)
        if self._worker is None:
            readback = self.hv.get_hv_adc_voltage()
            self.ui.lblHvVoltage.setText(f"{readback:.1f}")
            log.debug("PSU: hv readback=%.1f", readback)

    def _on_sipm_voltage_changed(self) -> None:
        val = self.ui.spinSipmVoltage.value()
        log.debug("PSU: sipm_voltage=%.3f", val)
        self.hv.set_sipm_voltage(val)
        if self._worker is None:
            readback = self.hv.get_sipm_adc_voltage()
            self.ui.lblSipmVoltage.setText(f"{readback:.2f}")
            log.debug("PSU: sipm readback=%.2f", readback)

    def hardware_configuration_settings(self) -> dict[str, int | float]:
        """Return all IDS setpoint widgets (the backend exposes no getters)."""
        settings: dict[str, int | float] = {
            "hv_voltage": self.ui.spinHvVoltage.value(),
            "hv_compens_ct": self.ui.spinHvCompensCt.value(),
            "hv_compens_tref": self.ui.spinHvCompensTref.value(),
            "hv_compens_mode": self.ui.comboHvCompensMode.currentIndex(),
            "temp_digital_enable": self.ui.comboTempDigitalEnable.currentIndex(),
        }
        if self._sipm_available:
            settings.update(
                {
                    "sipm_enable": int(self.ui.cbSipmEnable.isChecked()),
                    "sipm_voltage": self.ui.spinSipmVoltage.value(),
                    "sipm_compens_ct": self.ui.spinSipmCompensCt.value(),
                    "sipm_compens_tref": self.ui.spinSipmCompensTref.value(),
                    "sipm_compens_mode": self.ui.comboSipmCompensMode.currentIndex(),
                }
            )
        return settings

    def populate_hardware_configuration_settings(self, settings: object) -> None:
        """Populate write-only IDS controls after their values are applied."""
        if not isinstance(settings, dict):
            return
        widgets = {
            "hv_voltage": self.ui.spinHvVoltage,
            "hv_compens_ct": self.ui.spinHvCompensCt,
            "hv_compens_tref": self.ui.spinHvCompensTref,
            "hv_compens_mode": self.ui.comboHvCompensMode,
            "temp_digital_enable": self.ui.comboTempDigitalEnable,
            "sipm_enable": self.ui.cbSipmEnable,
            "sipm_voltage": self.ui.spinSipmVoltage,
            "sipm_compens_ct": self.ui.spinSipmCompensCt,
            "sipm_compens_tref": self.ui.spinSipmCompensTref,
            "sipm_compens_mode": self.ui.comboSipmCompensMode,
        }
        for name, value in settings.items():
            widget = widgets.get(name)
            if widget is None:
                continue
            widget.blockSignals(True)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            widget.blockSignals(False)

    def configuration_settings(self) -> dict[str, int]:
        return {
            "refresh_interval_ms": self.ui.spinRefreshRate.value(),
            "plot_time_range_s": self.ui.spinTimeRange.value(),
        }

    def apply_configuration_settings(self, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        if "refresh_interval_ms" in settings:
            self.ui.spinRefreshRate.setValue(int(settings["refresh_interval_ms"]))
        if "plot_time_range_s" in settings:
            self.ui.spinTimeRange.setValue(int(settings["plot_time_range_s"]))

    # ------------------------------------------------------------------
    # Readback handling (main thread, via signal)
    # ------------------------------------------------------------------

    def _on_readback(self, rb: PSUReadback) -> None:
        time_range = self.ui.spinTimeRange.value()

        self._ts.append(rb.timestamp)
        self._voltages.append(rb.hv_voltage)

        cutoff = rb.timestamp - time_range
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()
            self._voltages.popleft()

        self._update_plot()
        self._update_readback_labels(rb)

    def _update_plot(self) -> None:
        if not self._ts:
            return
        t = np.array(self._ts)
        v = np.array(self._voltages)
        self._plot_curve.setData(t, v)
        vb = self.ui.plotHvVoltage.getViewBox()
        vb.setXRange(t[-1] - self.ui.spinTimeRange.value(), t[-1], padding=0.02)

    def _update_readback_labels(self, rb: PSUReadback) -> None:
        if self._sipm_available:
            self.ui.lblSipmVoltage.setText(f"{rb.sipm_voltage:.2f}")
            self.ui.lblSipmCurrent.setText(f"{rb.sipm_current:.4f}")
            self.ui.lblSipmOverload.setText(str(rb.sipm_overload))
            self.ui.lblSipmCompensOutput.setText(f"{rb.sipm_compens_output:.2f}")

        self.ui.lblHvVoltage.setText(f"{rb.hv_voltage:.1f}")
        self.ui.lblHvCompensOutput.setText(f"{rb.hv_compens_output:.1f}")

        self.ui.lblTempAnalog.setText(f"{rb.temp_analog:.1f}")
        self.ui.lblTempDigital.setText(f"{rb.temp_digital:.1f}")
        self.ui.lblTempDigitalStatus.setText(str(rb.temp_digital_status))
        self.ui.lblAdsTemp.setText(f"{rb.ads_temp:.1f}")
