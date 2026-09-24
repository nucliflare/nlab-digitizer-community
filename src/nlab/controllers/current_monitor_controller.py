from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtWidgets import QWidget

from nlab.hardware.digitizer.dma import ScopeDmaFrame, ScopeFrameBuffer
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.hardware.digitizer.scope import SCOPE_DATAPATH_CLOCK_PERIOD_NS
from nlab.ui.ui_current_monitor_view import Ui_CurrentMonitorView
from nlab.workers.current_monitor_worker import (
    CurrentMonitorWorker,
    CurrentSample,
    CurrentSampleBuffer,
)

if TYPE_CHECKING:
    from nlab.controllers.scope_controller import ScopeController

log = logging.getLogger(__name__)

_DISPLAY_HZ = 60
_DISPLAY_PERIOD_NS = round(1_000_000_000 / _DISPLAY_HZ)
_TARGET_SAMPLE_HZ = 1000
_HISTORY_NS = 10_000_000_000
_RATE_WINDOW_NS = 1_000_000_000
_ZERO_WINDOW_NS = 1_000_000_000
_MIN_STALE_NS = 50_000_000
_MODE_IIR = 0
_MODE_SCOPE_DMA = 1


class CurrentMonitorController(QWidget):
    """Live current estimates from IIR polling or periodic Scope DMA."""

    def __init__(
        self,
        mca: MultiChannelAnalyzer,
        channel: int,
        parent: QWidget | None = None,
        *,
        auto_start: bool = False,
        scope_controller: ScopeController | None = None,
        scope_frame_buffer: ScopeFrameBuffer | None = None,
    ) -> None:
        super().__init__(parent)
        self._mca = mca
        self._channel = channel
        self._scope_controller = scope_controller
        self._scope_frame_buffer = scope_frame_buffer
        self._scope_frame_consumer = ScopeFrameBuffer(max_frames=1024)
        self._dma_active = False
        self._dma_sequence = 0
        self._sample_buffer = CurrentSampleBuffer()
        self._worker: CurrentMonitorWorker | None = None
        self._worker_thread: QThread | None = None
        self._stop_requested = False
        self._latest: CurrentSample | None = None
        self._history: deque[CurrentSample] = deque()
        self._last_batch: tuple[CurrentSample, ...] = ()
        self._dropped_samples = 0
        self._monitor_error: str | None = None

        self.ui = Ui_CurrentMonitorView()
        self.ui.setupUi(self)  # type: ignore[no-untyped-call]
        self._setup_plot()
        self._connect_signals()
        if self._scope_controller is not None:
            self._scope_controller.current_dma_state_changed.connect(
                self._on_scope_dma_state_changed
            )
        self._on_mode_changed(self.ui.comboMode.currentIndex())

        self._display_timer = QTimer(self)
        self._display_timer.setSingleShot(True)
        self._display_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._display_timer.timeout.connect(self._render_pending)
        self._next_display_ns = time.perf_counter_ns() + _DISPLAY_PERIOD_NS

        if auto_start:
            self.start_monitor()

    @property
    def channel(self) -> int:
        return self._channel

    def _setup_plot(self) -> None:
        self.ui.plotCurrent.setBackground("#f8f9fa")
        plot = self.ui.plotCurrent.getPlotItem()
        plot.showAxis("right")
        plot.showAxis("top")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setLabel("bottom", "Time before newest sample", units="s")
        plot.setLabel("left", "IIR output", units="raw")
        self._plot = plot
        self._curve = plot.plot(pen=pg.mkPen("#277da1", width=1.5))

    def _connect_signals(self) -> None:
        self.ui.btnStart.clicked.connect(self.start_monitor)
        self.ui.btnStop.clicked.connect(self.request_monitor_stop)
        self.ui.btnZero.clicked.connect(self._set_zero_from_recent)
        self.ui.spinZero.valueChanged.connect(self._calibration_changed)
        self.ui.spinScale.valueChanged.connect(self._calibration_changed)
        self.ui.comboUnit.currentTextChanged.connect(self._calibration_changed)
        self.ui.comboMode.currentIndexChanged.connect(self._on_mode_changed)

    def _is_running(self) -> bool:
        return self._worker_thread is not None or self._dma_active

    def _dma_supported(self) -> bool:
        return (
            self._scope_controller is not None
            and self._scope_frame_buffer is not None
            and self._scope_controller.current_dma_supported()
        )

    def _set_running_controls(self, running: bool) -> None:
        self.ui.btnStart.setEnabled(not running)
        self.ui.btnStop.setEnabled(running)
        self.ui.comboMode.setEnabled(not running)

    @Slot(int)
    def _on_mode_changed(self, index: int) -> None:
        dma_mode = index == _MODE_SCOPE_DMA
        self.ui.labelRaw.setText("Raw frame mean:" if dma_mode else "Raw IIR code:")
        base_label = "Scope frame mean" if dma_mode else "IIR output"
        label = "Current" if self._unit() != "raw" else base_label
        self._plot.setLabel("left", label, units=self._unit())
        if self._is_running():
            return
        if dma_mode and not self._dma_supported():
            self.ui.lblStatus.setText(
                "Scope DMA mode is unavailable on this backend/firmware."
            )
        else:
            self.ui.lblStatus.setText("Monitor stopped.")

    def _schedule_display(self) -> None:
        if not self._display_timer.isActive():
            delay_ns = max(0, self._next_display_ns - time.perf_counter_ns())
            self._display_timer.start(max(0, round(delay_ns / 1_000_000)))

    @Slot()
    def start_monitor(self) -> None:
        if self._is_running():
            return
        self._reset_measurement_state()
        if self.ui.comboMode.currentIndex() == _MODE_SCOPE_DMA:
            self._start_dma_monitor()
            return
        self._sample_buffer = CurrentSampleBuffer()
        worker = CurrentMonitorWorker(
            self._mca.create_current_monitor_client,
            self._sample_buffer,
            target_hz=_TARGET_SAMPLE_HZ,
        )
        thread = QThread(self)
        self._worker = worker
        self._worker_thread = thread
        self._stop_requested = False
        self._monitor_error = None
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            self._on_worker_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._set_running_controls(True)
        self.ui.lblStatus.setText("Opening an isolated current-monitor connection…")
        self._next_display_ns = time.perf_counter_ns() + _DISPLAY_PERIOD_NS
        self._schedule_display()
        thread.start()

    def _reset_measurement_state(self) -> None:
        self._latest = None
        self._history.clear()
        self._last_batch = ()
        self._dropped_samples = 0
        self._monitor_error = None
        self._dma_sequence = 0
        self.ui.btnZero.setEnabled(False)
        self._curve.setData([], [])

    def _start_dma_monitor(self) -> None:
        scope_controller = self._scope_controller
        publisher = self._scope_frame_buffer
        if not self._dma_supported() or scope_controller is None or publisher is None:
            self._monitor_error = (
                "Scope DMA mode requires the direct IIO Scope backend with "
                "periodic-frame support."
            )
            self.ui.lblStatus.setText(self._monitor_error)
            return

        self._scope_frame_consumer.clear()
        publisher.subscribe(self._scope_frame_consumer)
        self._stop_requested = False
        self._monitor_error = None
        try:
            scope_controller.start_current_dma_monitor()
        except Exception as exc:
            publisher.unsubscribe(self._scope_frame_consumer)
            self._monitor_error = f"Could not start Scope DMA current monitor: {exc}"
            self.ui.lblStatus.setText(self._monitor_error)
            log.exception("Starting Scope DMA current monitor failed")
            return

        self._dma_active = True
        self._set_running_controls(True)
        self.ui.lblStatus.setText("Starting optimized periodic Scope DMA...")
        self._next_display_ns = time.perf_counter_ns() + _DISPLAY_PERIOD_NS
        self._schedule_display()

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        self._monitor_error = message

    def request_monitor_stop(self) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        if self._dma_active:
            if self._scope_controller is not None:
                self._scope_controller.stop_current_dma_monitor()
            self.ui.btnStop.setEnabled(False)
            return
        worker = self._worker
        if worker is not None:
            worker.request_shutdown()
        self.ui.btnStop.setEnabled(False)

    def stop_monitor_sync(self) -> None:
        """Stop the selected transport and wait for resource release."""
        if self._dma_active:
            self._stop_requested = True
            if self._scope_controller is not None:
                self._scope_controller.stop_current_dma_monitor(wait=True)
            if self._dma_active:
                self._finish_dma_monitor("Monitor stopped.")
            return

        self.request_monitor_stop()
        thread = self._worker_thread
        if thread is not None and not thread.wait(4000):
            # A remote attribute read is synchronous. Never close its context
            # from another thread; wait for the bounded transport call to
            # return instead of using QThread.terminate().
            log.warning("Waiting for in-flight current-monitor IIO read to finish")
            thread.wait()
        self._worker = None
        self._worker_thread = None
        self._stop_requested = False
        self._display_timer.stop()
        self._render_pending()
        self._set_running_controls(False)
        if self._monitor_error is None:
            self.ui.lblStatus.setText("Monitor stopped.")

    @Slot()
    def _on_worker_finished(self) -> None:
        self._worker = None
        self._worker_thread = None
        self._stop_requested = False
        self._display_timer.stop()
        self._render_pending()
        self._set_running_controls(False)
        if self._monitor_error is None:
            self.ui.lblStatus.setText("Monitor stopped.")

    @Slot(bool, str)
    def _on_scope_dma_state_changed(self, running: bool, message: str) -> None:
        if not self._dma_active:
            return
        if running:
            self.ui.lblStatus.setText(message)
            return
        self._finish_dma_monitor(message)

    def _finish_dma_monitor(self, message: str) -> None:
        publisher = self._scope_frame_buffer
        if publisher is not None:
            publisher.unsubscribe(self._scope_frame_consumer)
        self._display_timer.stop()
        self._dma_active = False
        self._render_pending()
        self._stop_requested = False
        self._set_running_controls(False)
        self.ui.lblStatus.setText(message)

    def _convert(self, raw_code: float) -> float:
        return (raw_code - self.ui.spinZero.value()) * self.ui.spinScale.value()

    def _unit(self) -> str:
        return self.ui.comboUnit.currentText()

    @staticmethod
    def _format_value(value: float) -> str:
        magnitude = abs(value)
        if magnitude == 0:
            return "0"
        if magnitude >= 100_000 or magnitude < 0.001:
            return f"{value:.5e}"
        return f"{value:,.6g}"

    def _calibration_changed(self, *_args: object) -> None:
        raw_label = (
            "Scope frame mean"
            if self.ui.comboMode.currentIndex() == _MODE_SCOPE_DMA
            else "IIR output"
        )
        label = "Current" if self._unit() != "raw" else raw_label
        self._plot.setLabel("left", label, units=self._unit())
        self._render_values(time.perf_counter_ns())

    @Slot()
    def _set_zero_from_recent(self) -> None:
        latest = self._latest
        if latest is None:
            return
        cutoff = latest.timestamp_ns - _ZERO_WINDOW_NS
        values = [sample.raw_code for sample in self._history if sample.timestamp_ns >= cutoff]
        if values:
            self.ui.spinZero.setValue(statistics.fmean(values))

    @Slot()
    def _render_pending(self) -> None:
        now_ns = time.perf_counter_ns()
        if self.ui.comboMode.currentIndex() == _MODE_SCOPE_DMA:
            samples, dropped = self._drain_dma_samples()
            self._dropped_samples += dropped
        else:
            batch = self._sample_buffer.drain()
            self._monitor_error = batch.error
            self._dropped_samples += batch.dropped_samples
            samples = batch.samples
        if samples:
            self._last_batch = samples
            self._latest = samples[-1]
            self._history.extend(samples)
            cutoff = self._latest.timestamp_ns - _HISTORY_NS
            while self._history and self._history[0].timestamp_ns < cutoff:
                self._history.popleft()
            self.ui.btnZero.setEnabled(True)
        self._render_values(now_ns)

        if self._is_running():
            self._next_display_ns += _DISPLAY_PERIOD_NS
            if self._next_display_ns <= now_ns:
                self._next_display_ns = now_ns + _DISPLAY_PERIOD_NS
            self._schedule_display()

    def _drain_dma_samples(self) -> tuple[tuple[CurrentSample, ...], int]:
        frames, dropped = self._scope_frame_consumer.drain()
        samples: list[CurrentSample] = []
        for frame in frames:
            if not frame.samples.size:
                continue
            self._dma_sequence += 1
            samples.append(self._current_sample_from_dma_frame(frame))
        return tuple(samples), dropped

    def _current_sample_from_dma_frame(self, frame: ScopeDmaFrame) -> CurrentSample:
        return CurrentSample(
            sequence=self._dma_sequence,
            timestamp_ns=frame.received_ns,
            raw_code=float(np.mean(frame.samples, dtype=np.float64)),
            read_latency_ns=0,
            coverage_ns=frame.geometry.capture_duration_ns,
            hardware_timestamp=frame.timestamp,
            sample_count=frame.geometry.waveform_samples,
        )

    def _render_values(self, now_ns: int) -> None:
        latest = self._latest
        if latest is None:
            if self._monitor_error is not None:
                self.ui.lblStatus.setText(self._monitor_error)
            return

        unit = self._unit()
        current = self._convert(latest.raw_code)
        self.ui.lblCurrent.setText(f"{self._format_value(current)} {unit}")
        self.ui.lblRaw.setText(self._format_value(latest.raw_code))

        batch_values = [self._convert(sample.raw_code) for sample in self._last_batch]
        if batch_values:
            self.ui.lblInterval.setText(
                f"mean {self._format_value(statistics.fmean(batch_values))}; "
                f"min {self._format_value(min(batch_values))}; "
                f"max {self._format_value(max(batch_values))} {unit}; "
                f"n={len(batch_values)}"
            )

        rate_samples = [
            sample
            for sample in self._history
            if sample.timestamp_ns >= latest.timestamp_ns - _RATE_WINDOW_NS
        ]
        dma_mode = self.ui.comboMode.currentIndex() == _MODE_SCOPE_DMA
        intervals_ns = [
            self._sample_interval_ns(left, right, dma_mode=dma_mode)
            for left, right in zip(rate_samples, rate_samples[1:])
            if self._sample_interval_ns(left, right, dma_mode=dma_mode) > 0
        ]
        rate_hz = 0.0
        median_interval_ns = 0.0
        if intervals_ns:
            median_interval_ns = statistics.median(intervals_ns)
            rate_hz = 1_000_000_000 / statistics.fmean(intervals_ns)
        age_ns = max(0, now_ns - latest.timestamp_ns)
        if dma_mode:
            coverage_ns = sum(sample.coverage_ns for sample in rate_samples)
            observation_ns = (
                self._sample_interval_ns(
                    rate_samples[0],
                    rate_samples[-1],
                    dma_mode=True,
                )
                + round(statistics.fmean(intervals_ns))
                if len(rate_samples) > 1 and intervals_ns
                else latest.coverage_ns
            )
            coverage_percent = 100 * coverage_ns / max(1, observation_ns)
            self.ui.lblAcquisition.setText(
                f"{rate_hz:,.1f} frames/s; median spacing "
                f"{median_interval_ns / 1_000_000:.3f} ms; frame average "
                f"{latest.coverage_ns / 1000:.3f} us; "
                f"coverage {coverage_percent:.2f}%; age {age_ns / 1_000_000:.1f} ms"
            )
        else:
            latencies_ms = [
                sample.read_latency_ns / 1_000_000 for sample in rate_samples
            ]
            median_latency_ms = (
                statistics.median(latencies_ms) if latencies_ms else 0.0
            )
            self.ui.lblAcquisition.setText(
                f"{rate_hz:,.1f} Hz; median read {median_latency_ms:.3f} ms; "
                f"age {age_ns / 1_000_000:.1f} ms"
            )

        stale_ns = max(_MIN_STALE_NS, round(3 * median_interval_ns))
        if self._monitor_error is not None:
            status = self._monitor_error
        elif age_ns > stale_ns:
            status = f"Stale: newest sample is {age_ns / 1_000_000:.1f} ms old."
        elif dma_mode:
            sample_period_ns = (
                latest.coverage_ns / latest.sample_count if latest.sample_count else 0
            )
            status = (
                "Live periodic Scope DMA; each point averages "
                f"{latest.sample_count:,} samples at {sample_period_ns:g} ns spacing; "
                "frame spacing uses the hardware trigger timestamp."
            )
        else:
            status = "Live FPGA IIR status; host timestamps are transaction midpoints."
        if self._dropped_samples:
            noun = "frame(s)" if dma_mode else "sample(s)"
            status += f" Display buffer dropped {self._dropped_samples:,} {noun}."
        self.ui.lblStatus.setText(status)

        if self._history:
            newest = self._history[-1]
            x = [
                -self._sample_interval_ns(sample, newest, dma_mode=dma_mode)
                / 1_000_000_000
                for sample in self._history
            ]
            y = [self._convert(sample.raw_code) for sample in self._history]
            self._curve.setData(x, y)

    @staticmethod
    def _sample_interval_ns(
        left: CurrentSample,
        right: CurrentSample,
        *,
        dma_mode: bool,
    ) -> int:
        if (
            dma_mode
            and left.hardware_timestamp is not None
            and right.hardware_timestamp is not None
        ):
            return (
                right.hardware_timestamp - left.hardware_timestamp
            ) * SCOPE_DATAPATH_CLOCK_PERIOD_NS
        return right.timestamp_ns - left.timestamp_ns

    def reset_zoom(self) -> None:
        self._plot.enableAutoRange()

    def configuration_settings(self) -> dict[str, object]:
        return {
            "mode": "scope_dma" if self.ui.comboMode.currentIndex() else "iir",
            "zero_code": self.ui.spinZero.value(),
            "scale_per_code": self.ui.spinScale.value(),
            "unit": self._unit(),
        }

    def apply_configuration_settings(self, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        requested_mode = settings.get("mode")
        if requested_mode in ("iir", "scope_dma"):
            target = _MODE_SCOPE_DMA if requested_mode == "scope_dma" else _MODE_IIR
            if target != self.ui.comboMode.currentIndex():
                restart = self._is_running()
                if restart:
                    self.stop_monitor_sync()
                self.ui.comboMode.setCurrentIndex(target)
                if restart:
                    self.start_monitor()
        if "zero_code" in settings:
            self.ui.spinZero.setValue(float(settings["zero_code"]))
        if "scale_per_code" in settings:
            self.ui.spinScale.setValue(float(settings["scale_per_code"]))
        if "unit" in settings:
            index = self.ui.comboUnit.findText(str(settings["unit"]))
            if index >= 0:
                self.ui.comboUnit.setCurrentIndex(index)
