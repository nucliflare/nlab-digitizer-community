from __future__ import annotations

import logging
import time
from enum import IntEnum
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSettings, QThread, QThreadPool, QTimer
from PySide6.QtWidgets import QFileDialog, QSlider, QSpinBox, QWidget

from nlab.hardware.digitizer.dma import (
    FILE_HEADER_STRUCT,
    IIOScopeDmaStreamer,
    ScopeDmaStreamer,
)
from nlab.hardware.digitizer.scope import (
    ListSpec,
    RangeSpec,
    Scope,
    ScopeParam,
    TriggerMode,
)
from nlab.ui.ui_scope_view import Ui_ScopeView
from nlab.views.plot_viewbox import ModifierZoomViewBox
from nlab.workers.dma_workers import IIOScopeDmaWorker, ScopeDmaWorker
from nlab.workers.scope_worker import ScopeWorker

log = logging.getLogger(__name__)


class DisplayMode(IntEnum):
    PERSISTENCE = 0
    RAW = 1


class ScopeController(QWidget):
    """View + controller for a single Scope channel.

    Loaded from ui_scope_view.ui.  Drop into any QTabWidget via addTab().
    Widget ranges and defaults are driven entirely by Scope.specs at runtime.
    """

    _Y_SCALE_FACTOR = 20
    _Y_MIN = -32_000
    _Y_MAX = 32_000
    _SAMPLE_PERIOD_NS = 2

    def __init__(
        self,
        scope: Scope,
        scope_dma: ScopeDmaStreamer | IIOScopeDmaStreamer | None = None,
        channel: int = 1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._scope = scope
        self._scope_dma = scope_dma
        self._channel = channel
        self.ui = Ui_ScopeView()
        self.ui.setupUi(self)

        self._acquiring = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._request_frame)

        self._dma_worker: ScopeDmaWorker | IIOScopeDmaWorker | None = None
        self._dma_thread: QThread | None = None
        self._dma_filepath: Path | None = None
        self._dma_counter = 0

        self._measurement_timer = QTimer(self)
        self._measurement_timer.setSingleShot(True)
        self._measurement_timer.timeout.connect(self._on_measurement_timeout)

        self._apply_parameter_specs()
        self._send_defaults()
        self._load_hardware_state()
        self._restore_display_settings()
        self._setup_graph()
        self._connect_signals()
        self._update_frame_gap_enabled()
        self._update_viewer_transport_hint()
        self.ui.btnStop.setEnabled(False)

    # ------------------------------------------------------------------
    # Spec application
    # ------------------------------------------------------------------

    def _apply_parameter_specs(self) -> None:
        specs = self._scope.specs

        spec = specs[ScopeParam.TRIGGER_LEVEL]
        assert isinstance(spec, RangeSpec)
        self._apply_range_to_spinbox(self.ui.spinTriggerLevel, spec)
        self._apply_range_to_slider(self.ui.sliderTriggerLevel, spec)

        spec = specs[ScopeParam.DAC_VALUE]
        assert isinstance(spec, RangeSpec)
        self._apply_range_to_spinbox(self.ui.spinDacValue, spec)
        self._apply_range_to_slider(self.ui.sliderDacValue, spec)

        spec = specs[ScopeParam.PRETRIGGER_SAMPLES]
        assert isinstance(spec, RangeSpec)
        self._apply_range_to_spinbox(self.ui.spinPretrigger, spec)

        spec = specs[ScopeParam.FRAME_SAMPLES]
        assert isinstance(spec, RangeSpec)
        self._apply_range_to_spinbox(self.ui.spinFrameSamples, spec)

        spec = specs[ScopeParam.FRAME_PERIOD_CYCLES]
        assert isinstance(spec, RangeSpec)
        self._apply_range_to_spinbox(self.ui.spinFrameGap, spec)

        spec = specs[ScopeParam.EDGE_MODE]
        assert isinstance(spec, ListSpec)
        if spec.default in spec.items:
            self.ui.comboTriggerMode.setCurrentIndex(list(spec.items).index(spec.default))

        spec = specs[ScopeParam.DMA_ENABLED]
        assert isinstance(spec, ListSpec)
        self.ui.cbDmaEnable.setChecked(bool(spec.default))

    @staticmethod
    def _apply_range_to_spinbox(spinbox: QSpinBox, spec: RangeSpec) -> None:
        spinbox.setMinimum(int(spec.min_val))
        spinbox.setMaximum(int(spec.max_val))
        spinbox.setSingleStep(int(spec.step) or 1)
        spinbox.setValue(int(spec.default))

    @staticmethod
    def _apply_range_to_slider(slider: QSlider, spec: RangeSpec) -> None:
        slider.setMinimum(int(spec.min_val))
        slider.setMaximum(int(spec.max_val))
        slider.setSingleStep(int(spec.step) or 1)
        slider.setValue(int(spec.default))

    # ------------------------------------------------------------------
    # Write defaults to hardware, then read back
    # ------------------------------------------------------------------

    def _send_defaults(self) -> None:
        specs = self._scope.specs
        self._scope.set_trigger_level(int(specs[ScopeParam.TRIGGER_LEVEL].default))
        self._scope.set_dac_value(int(specs[ScopeParam.DAC_VALUE].default))
        self._scope.set_pretrigger_samples(int(specs[ScopeParam.PRETRIGGER_SAMPLES].default))
        self._scope.set_frame_samples(int(specs[ScopeParam.FRAME_SAMPLES].default))
        self._scope.set_frame_period_cycles(
            int(specs[ScopeParam.FRAME_PERIOD_CYCLES].default)
        )
        self._scope.set_trigger_mode(TriggerMode(specs[ScopeParam.EDGE_MODE].default))
        self._scope.set_dma_enable(bool(specs[ScopeParam.DMA_ENABLED].default))

    def _load_hardware_state(self) -> None:
        self.ui.spinTriggerLevel.setValue(self._scope.get_trigger_level())
        self.ui.sliderTriggerLevel.setValue(self._scope.get_trigger_level())
        self.ui.spinDacValue.setValue(self._scope.get_dac_value())
        self.ui.sliderDacValue.setValue(self._scope.get_dac_value())
        self.ui.spinPretrigger.setValue(self._scope.get_pretrigger_samples())
        self.ui.spinFrameSamples.setValue(self._scope.get_frame_samples())
        self.ui.spinFrameGap.setValue(self._scope.get_frame_period_cycles())
        self.ui.comboTriggerMode.setCurrentIndex(self._scope.get_trigger_mode().value)
        self.ui.cbDmaEnable.setChecked(self._scope.get_dma_enable())

    # ------------------------------------------------------------------
    # Display settings persistence
    # ------------------------------------------------------------------

    def _settings_key(self, name: str) -> str:
        return f"scope/ch{self._channel}/{name}"

    def _restore_display_settings(self) -> None:
        s = QSettings()
        if (v := s.value(self._settings_key("display_mode"))) is not None:
            self.ui.comboDisplayMode.setCurrentIndex(int(v))
        if (v := s.value(self._settings_key("persistence"))) is not None:
            self.ui.dialPersistence.setValue(int(v))
        if (v := s.value(self._settings_key("refresh_rate"))) is not None:
            self.ui.spinRefreshRate.setValue(int(v))

    def save_display_settings(self) -> None:
        s = QSettings()
        s.setValue(self._settings_key("display_mode"), self.ui.comboDisplayMode.currentIndex())
        s.setValue(self._settings_key("persistence"), self.ui.dialPersistence.value())
        s.setValue(self._settings_key("refresh_rate"), self.ui.spinRefreshRate.value())

    def configuration_settings(self) -> dict[str, int]:
        """Return settings owned by this view rather than by scope hardware."""
        return {
            "display_mode": self.ui.comboDisplayMode.currentIndex(),
            "persistence": self.ui.dialPersistence.value(),
            "refresh_rate_hz": self.ui.spinRefreshRate.value(),
            "measurement_time_s": self.ui.spinTime.value(),
        }

    def apply_configuration_settings(self, settings: object) -> None:
        """Populate app-only controls from a YAML channel section."""
        if not isinstance(settings, dict):
            return
        if "display_mode" in settings:
            self.ui.comboDisplayMode.setCurrentIndex(int(settings["display_mode"]))
        if "persistence" in settings:
            self.ui.dialPersistence.setValue(int(settings["persistence"]))
        if "refresh_rate_hz" in settings:
            self.ui.spinRefreshRate.setValue(int(settings["refresh_rate_hz"]))
        if "measurement_time_s" in settings:
            self.ui.spinTime.setValue(int(settings["measurement_time_s"]))
        self.save_display_settings()

    def reset_zoom(self) -> None:
        """Reset the waveform plot to its default fixed range."""
        self._update_axis_ranges()

    # ------------------------------------------------------------------
    # Enable/disable parameter controls during DMA
    # ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.ui.groupTrigger.setEnabled(enabled)
        self.ui.groupTiming.setEnabled(enabled)
        self.ui.cbDmaEnable.setEnabled(enabled)

    def _viewer_frame_limit(self) -> int | None:
        return self._scope.get_viewer_frame_samples_limit()

    def _frame_exceeds_viewer_limit(self) -> bool:
        limit = self._viewer_frame_limit()
        return limit is not None and self.ui.spinFrameSamples.value() > limit

    def _viewer_limit_message(self) -> str:
        limit = self._viewer_frame_limit()
        assert limit is not None
        return (
            f"The IIO live viewer is limited to {limit} samples by its "
            "page-sized text transport. Enable Record DMA frames for this "
            "frame length; live preview will be paused."
        )

    def _update_viewer_transport_hint(self) -> None:
        limit = self._viewer_frame_limit()
        if limit is None:
            return
        tip = (
            f"Hardware and DMA support up to 8188 samples. The IIO live "
            f"viewer is guaranteed complete through {limit} samples because "
            "viewer_data is limited to one text page."
        )
        self.ui.spinFrameSamples.setToolTip(tip)
        self.ui.plotWaveform.setToolTip(tip)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._wire_slider_spinbox(
            self.ui.sliderTriggerLevel,
            self.ui.spinTriggerLevel,
            lambda v: self._scope.set_trigger_level(v),
        )
        self._wire_slider_spinbox(
            self.ui.sliderDacValue,
            self.ui.spinDacValue,
            lambda v: self._scope.set_dac_value(v),
        )

        self.ui.spinPretrigger.editingFinished.connect(
            lambda: self._scope.set_pretrigger_samples(self.ui.spinPretrigger.value())
        )
        self.ui.spinFrameSamples.editingFinished.connect(self._on_frame_samples_changed)
        self.ui.spinFrameGap.editingFinished.connect(
            lambda: self._scope.set_frame_period_cycles(self.ui.spinFrameGap.value())
        )
        self.ui.comboTriggerMode.currentIndexChanged.connect(
            self._on_trigger_mode_changed
        )
        self.ui.cbDmaEnable.toggled.connect(self._on_dma_toggled)
        self.ui.btnStart.clicked.connect(self._on_start)
        self.ui.btnStop.clicked.connect(self._on_stop)
        self.ui.btnAcquireFrame.clicked.connect(self._on_acquire_frame)
        self.ui.btnDmaFile.clicked.connect(self._on_dma_file)

        self.ui.comboDisplayMode.currentIndexChanged.connect(self._on_display_mode_changed)
        self.ui.dialPersistence.valueChanged.connect(self._on_persistence_changed)
        self.ui.spinRefreshRate.valueChanged.connect(self._on_refresh_rate_changed)

    def _on_trigger_mode_changed(self, index: int) -> None:
        self._scope.set_trigger_mode(TriggerMode(index))
        self._update_frame_gap_enabled()

    def _update_frame_gap_enabled(self) -> None:
        periodic = self.ui.comboTriggerMode.currentIndex() == TriggerMode.TIMED
        enabled = periodic and self._scope.frame_period_cycles_supported()
        self.ui.labelFrameGap.setEnabled(enabled)
        self.ui.spinFrameGap.setEnabled(enabled)

    def _wire_slider_spinbox(
        self,
        slider: QSlider,
        spinbox: QSpinBox,
        set_fn: object,
    ) -> None:
        slider.valueChanged.connect(spinbox.setValue)

        def on_spin_changed(v: int) -> None:
            slider.blockSignals(True)
            slider.setValue(v)
            slider.blockSignals(False)

        def on_spin_committed() -> None:
            set_fn(spinbox.value())  # type: ignore[operator]

        def on_slider_released() -> None:
            set_fn(spinbox.value())  # type: ignore[operator]

        spinbox.valueChanged.connect(on_spin_changed)
        spinbox.editingFinished.connect(on_spin_committed)
        slider.sliderReleased.connect(on_slider_released)

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def _on_dma_toggled(self, enabled: bool) -> None:
        self._scope.set_dma_enable(enabled)
        self._update_viewer_transport_hint()

    def _on_start(self) -> None:
        if self._dma_worker is not None or self._dma_thread is not None:
            log.warning(
                "Scope ch%d: refusing to start while the previous DMA worker is still closing",
                self._channel,
            )
            return
        use_dma = self.ui.cbDmaEnable.isChecked() and self._scope_dma is not None
        if self._frame_exceeds_viewer_limit() and not use_dma:
            message = self._viewer_limit_message()
            self.ui.btnStart.setChecked(False)
            self.ui.lblRecordingStatus.setText(message)
            log.warning("Scope ch%d: %s", self._channel, message)
            return
        self.ui.btnStart.setChecked(True)
        self.ui.btnStart.setEnabled(False)
        self.ui.btnStop.setChecked(False)
        self.ui.btnAcquireFrame.setEnabled(False)
        self.ui.cbDmaEnable.setEnabled(False)
        self.ui.btnDmaFile.setEnabled(False)

        if self.ui.cbDmaEnable.isChecked() and self._scope_dma is not None:
            self._start_with_dma()
        else:
            self._start_polling_only()

    def _start_polling_only(self) -> None:
        self._scope.start()
        self.ui.btnStop.setEnabled(True)
        interval_ms = 1000 // self.ui.spinRefreshRate.value()
        self._refresh_timer.start(interval_ms)
        self._start_measurement_timer()
        log.info("Scope ch%d: acquisition started (refresh %d ms)", self._channel, interval_ms)

    def _start_with_dma(self) -> None:
        filepath = self._dma_filepath or self._generate_filepath()
        self._dma_filepath = None

        if isinstance(self._scope_dma, IIOScopeDmaStreamer):
            log.debug("Scope ch%d IIO DMA [1/4]: creating worker, file=%s", self._channel, filepath)
            self._dma_worker = IIOScopeDmaWorker(
                streamer=self._scope_dma,
                filepath=filepath,
            )
        else:
            frame_samples = self.ui.spinFrameSamples.value()
            log.debug(
                "Scope ch%d DMA [1/6]: creating worker, file=%s, frame_samples=%d",
                self._channel,
                filepath,
                frame_samples,
            )
            self._dma_worker = ScopeDmaWorker(
                streamer=self._scope_dma,
                filepath=filepath,
                frame_samples=frame_samples,
            )
        self._dma_thread = QThread(self)
        self._dma_worker.moveToThread(self._dma_thread)

        self._dma_thread.started.connect(self._dma_worker.run)
        self._dma_worker.ready.connect(self._on_dma_ready)
        self._dma_worker.progress.connect(self._on_dma_progress)
        self._dma_worker.error.connect(self._on_dma_error)
        self._dma_worker.finished.connect(self._dma_thread.quit)
        self._dma_worker.finished.connect(self._dma_worker.deleteLater)
        self._dma_thread.finished.connect(self._dma_thread.deleteLater)
        self._dma_thread.finished.connect(self._on_dma_finished)

        self._set_controls_enabled(False)
        self.ui.lblRecordingStatus.setText("Connecting...")
        log.debug(
            "Scope ch%d DMA [2/6]: starting worker thread (ZMQ connect + subscribe)", self._channel
        )
        self._dma_thread.start()
        log.info("Scope DMA: worker started, waiting for socket ready, file=%s", filepath)

    def _on_dma_ready(self) -> None:
        if isinstance(self._scope_dma, IIOScopeDmaStreamer):
            # The backend's first read starts the blocking refill and then
            # writes enable=1 in that order. The queued ready signal is only
            # a UI transition; a second start write here would race it.
            log.debug(
                "Scope ch%d IIO DMA [3/4]: backend reader owns ordered start",
                self._channel,
            )
        else:
            log.debug(
                "Scope ch%d DMA [3/6]: ZMQ socket ready, DMA already enabled via checkbox",
                self._channel,
            )
            log.debug(
                "Scope ch%d DMA [4/6]: calling scope.start() -> set_enable(True) "
                "(HW fires start_irq -> server sends StreamSTART)",
                self._channel,
            )
            self._scope.start()
        interval_ms = 1000 // self.ui.spinRefreshRate.value()
        if self._frame_exceeds_viewer_limit():
            self._refresh_timer.stop()
        else:
            self._refresh_timer.start(interval_ms)
        self.ui.btnStop.setEnabled(True)
        self.ui.lblRecordingStatus.setText(
            "Recording... (live viewer paused for this frame length)"
            if self._frame_exceeds_viewer_limit()
            else "Recording..."
        )
        self._start_measurement_timer()
        log.info("Scope ch%d: DMA + acquisition started (socket was ready)", self._channel)

    def _on_stop(self) -> None:
        self._refresh_timer.stop()
        self._measurement_timer.stop()
        self.ui.btnStop.setChecked(True)

        if self._dma_worker is not None:
            log.debug("Scope ch%d: stopping with DMA", self._channel)
            if isinstance(self._scope_dma, IIOScopeDmaStreamer):
                # Current vdpp_scope explicitly accepts enable=0 while a
                # buffer is armed. Stop new triggers first, then have the
                # worker cancel any blocked refill and close the buffer.
                self._scope.stop()
                self._dma_worker.stop()
            else:
                self._scope.stop()
                self._dma_worker.stop()
            self._acquiring = False
            self.ui.btnStart.setChecked(False)
            self.ui.btnStart.setEnabled(False)
            self.ui.btnStop.setEnabled(False)
            self.ui.btnAcquireFrame.setEnabled(False)
            self.ui.cbDmaEnable.setEnabled(False)
            self.ui.btnDmaFile.setEnabled(False)
            self.ui.lblRecordingStatus.setText("Stopping...")
            log.info("Scope ch%d: DMA stop requested; waiting for teardown", self._channel)
            return
        else:
            self._scope.stop()

        self._acquiring = False
        self.ui.btnStart.setChecked(False)
        self.ui.btnStart.setEnabled(True)
        self.ui.btnStop.setEnabled(False)
        self.ui.btnAcquireFrame.setEnabled(True)
        self.ui.cbDmaEnable.setEnabled(True)
        self.ui.btnDmaFile.setEnabled(True)
        log.info("Scope ch%d: acquisition stopped", self._channel)

    def _start_measurement_timer(self) -> None:
        time_s = self.ui.spinTime.value()
        if time_s > 0:
            self._measurement_timer.start(time_s * 1000)
            log.info("Scope ch%d: measurement timer set to %d s", self._channel, time_s)

    def _on_measurement_timeout(self) -> None:
        log.info("Scope ch%d: measurement time limit reached", self._channel)
        self._on_stop()

    def _on_refresh_rate_changed(self, value: int) -> None:
        if self._refresh_timer.isActive():
            self._refresh_timer.setInterval(1000 // value)

    def _request_frame(self) -> None:
        if self._frame_exceeds_viewer_limit():
            self._refresh_timer.stop()
            self.ui.lblRecordingStatus.setText(self._viewer_limit_message())
            return
        if self._acquiring:
            return
        self._acquiring = True
        worker = ScopeWorker(self._scope)
        worker.signals.ready.connect(self._on_frame_received)
        QThreadPool.globalInstance().start(worker)

    def _on_frame_received(self, data: list[np.ndarray]) -> None:
        self._acquiring = False
        if data is None:
            return
        x_time, y_voltage = data
        if self._display_mode == DisplayMode.RAW:
            self._raw_curve.setData(x_time, y_voltage)
        else:
            self._rasterize_frame(y_voltage)
            self._persistence_img.setImage(
                self._persistence_buffer, autoLevels=False, levels=(0, 1)
            )

    def _on_frame_samples_changed(self) -> None:
        value = self.ui.spinFrameSamples.value()
        if (
            self._refresh_timer.isActive()
            and self._dma_worker is None
            and self._frame_exceeds_viewer_limit()
        ):
            # Do not alter the hardware geometry under a viewer read that
            # may already be in flight. Restore the last accepted value and
            # explain how to capture the requested long frame.
            previous = self._scope.get_frame_samples()
            self.ui.spinFrameSamples.blockSignals(True)
            self.ui.spinFrameSamples.setValue(previous)
            self.ui.spinFrameSamples.blockSignals(False)
            message = self._viewer_limit_message()
            self.ui.lblRecordingStatus.setText(message)
            log.warning("Scope ch%d: %s", self._channel, message)
            return
        self._scope.set_frame_samples(value)
        self._display_nx = value // 4
        self._display_ny = self._display_nx * self._Y_SCALE_FACTOR
        self._persistence_buffer = np.zeros((self._display_nx, self._display_ny), dtype=np.float32)
        self._update_axis_ranges()
        self._update_viewer_transport_hint()

    def _on_acquire_frame(self) -> None:
        if self._frame_exceeds_viewer_limit():
            message = self._viewer_limit_message()
            self.ui.lblRecordingStatus.setText(message)
            log.warning("Scope ch%d: %s", self._channel, message)
            return
        try:
            self._scope.start()
            raw_frame = self._scope.acquire_frame()
        except Exception as exc:
            log.exception("Scope ch%d: single-frame acquisition failed", self._channel)
            self.ui.lblRecordingStatus.setText(f"Viewer error: {exc}")
            return
        finally:
            try:
                self._scope.stop()
            except Exception:
                log.exception("Scope ch%d: failed to disarm after viewer read", self._channel)
        value = self.ui.spinFrameSamples.value()
        frame = raw_frame[: value // 4]
        time_arr = np.arange(0, 8 * len(frame), 8)
        if self._display_mode == DisplayMode.RAW:
            self._raw_curve.setData(time_arr, frame)
        else:
            self._rasterize_frame(frame)
            self._persistence_img.setImage(
                self._persistence_buffer, autoLevels=False, levels=(0, 1)
            )

    def _rasterize_frame(self, frame: np.ndarray) -> None:
        self._persistence_buffer *= self.persistence
        n_samples = len(frame)
        x_float = np.linspace(0, self._display_nx - 1, n_samples)
        y_float = (
            (frame.astype(np.float32) - self._Y_MIN)
            / (self._Y_MAX - self._Y_MIN)
            * (self._display_ny - 1)
        )
        y_float = np.clip(y_float, 0, self._display_ny - 1)

        for i in range(n_samples - 1):
            x0, y0 = x_float[i], y_float[i]
            x1, y1 = x_float[i + 1], y_float[i + 1]
            n_pts = max(int(max(abs(x1 - x0), abs(y1 - y0))), 1) + 1
            xs = np.linspace(x0, x1, n_pts).astype(int)
            ys = np.linspace(y0, y1, n_pts).astype(int)
            np.clip(xs, 0, self._display_nx - 1, out=xs)
            np.clip(ys, 0, self._display_ny - 1, out=ys)
            self._persistence_buffer[xs, ys] = 1.0

    # ------------------------------------------------------------------
    # Graph setup
    # ------------------------------------------------------------------

    def _setup_graph(self) -> None:
        self._setup_plot_layout()
        self._setup_persistence_layer()
        self._setup_raw_layer()
        self._update_axis_ranges()
        self._set_display_mode(DisplayMode(self.ui.comboDisplayMode.currentIndex()))

    def _setup_plot_layout(self) -> None:
        layout = pg.GraphicsLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        self.ui.plotWaveform.setCentralItem(layout)
        self.ui.plotWaveform.setBackground("#f8f9fa")

        layout.addLabel("Amplitude [raw]", angle=-90)
        self._plot_item = layout.addPlot(viewBox=ModifierZoomViewBox())
        self._plot_item.showAxis("right")
        self._plot_item.showAxis("top")
        self._plot_item.showGrid(x=True, y=True, alpha=0.2)
        layout.nextRow()
        layout.addLabel("Time [ns]", col=1)

    def _setup_persistence_layer(self) -> None:
        self._display_nx = self.ui.spinFrameSamples.value() // 4
        self._display_ny = self._display_nx * self._Y_SCALE_FACTOR
        self._persistence_buffer = np.zeros((self._display_nx, self._display_ny), dtype=np.float32)
        self._persistence_img = pg.ImageItem()
        self._persistence_img.setImage(self._persistence_buffer, autoLevels=False, levels=(0, 1))
        self._persistence_img.setColorMap("viridis")
        self._plot_item.addItem(self._persistence_img)

    def _setup_raw_layer(self) -> None:
        self._raw_curve = self._plot_item.plot(pen=pg.mkPen("#00bfff", width=1))
        self._raw_curve.setVisible(False)

    def _update_axis_ranges(self) -> None:
        frame = self.ui.spinFrameSamples.value()
        duration_ns = frame * self._SAMPLE_PERIOD_NS
        vb = self._plot_item.getViewBox()
        vb.setXRange(0, duration_ns, padding=0)
        vb.setYRange(self._Y_MIN, self._Y_MAX, padding=0)
        self._persistence_img.setRect(
            QRectF(0, self._Y_MIN, duration_ns, self._Y_MAX - self._Y_MIN)
        )

    # ------------------------------------------------------------------
    # Display mode
    # ------------------------------------------------------------------

    def _set_display_mode(self, mode: DisplayMode) -> None:
        self._display_mode = mode
        is_persistence = mode == DisplayMode.PERSISTENCE
        self._persistence_img.setVisible(is_persistence)
        self._raw_curve.setVisible(not is_persistence)
        self.ui.dialPersistence.setEnabled(is_persistence)
        self.ui.labelPersistenceValue.setEnabled(is_persistence)

    def _on_display_mode_changed(self, index: int) -> None:
        self._set_display_mode(DisplayMode(index))

    def _on_persistence_changed(self, value: int) -> None:
        self.ui.labelPersistenceValue.setText(f"{value / 1000:.3f}")

    @property
    def persistence(self) -> float:
        return self.ui.dialPersistence.value() / 1000.0

    # ------------------------------------------------------------------
    # DMA helpers
    # ------------------------------------------------------------------

    def _generate_filepath(self) -> Path:
        folder = Path(QSettings().value("dma/save_folder", "measurements"))
        folder.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._dma_counter += 1
        name = f"scope_ch{self._channel}_{ts}_{self._dma_counter:03d}.bin"
        filepath = folder / name
        log.info("Scope DMA: auto-generated filepath: %s", filepath)
        return filepath

    def _on_dma_file(self) -> None:
        default_dir = str(QSettings().value("dma/save_folder", "measurements"))
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Scope DMA File",
            default_dir,
            "Binary files (*.bin);;All files (*)",
        )
        if path:
            self._dma_filepath = Path(path)
            log.info("Scope DMA: user selected filepath: %s", self._dma_filepath)

    def _on_dma_progress(self, bytes_written: int) -> None:
        # Both scope formats prepend the same 24-byte NDMA header. The
        # streamer progress value counts payload only, so add the header to
        # make the displayed number equal the actual file size.
        file_bytes = bytes_written + FILE_HEADER_STRUCT.size
        suffix = "; viewer paused" if self._frame_exceeds_viewer_limit() else ""
        if file_bytes < 1024 * 1024:
            self.ui.lblRecordingStatus.setText(
                f"Recording: {file_bytes / 1024:.1f} KiB{suffix}"
            )
        else:
            self.ui.lblRecordingStatus.setText(
                f"Recording: {file_bytes / (1024 * 1024):.1f} MiB{suffix}"
            )

    def _on_dma_error(self, message: str) -> None:
        log.error("Scope DMA error: %s", message)
        self.ui.lblRecordingStatus.setText(f"Error: {message}")

    def _on_dma_finished(self) -> None:
        self._dma_worker = None
        self._dma_thread = None
        log.info("Scope DMA: worker finished")

        self._set_controls_enabled(True)
        self.ui.btnStart.setChecked(False)
        self.ui.btnStart.setEnabled(True)
        self.ui.btnStop.setEnabled(False)
        self.ui.btnAcquireFrame.setEnabled(True)
        self.ui.cbDmaEnable.setEnabled(True)
        self.ui.btnDmaFile.setEnabled(True)

        if self._scope.dma_fault_is_latched():
            # A genuine EIO during the just-finished session latched a
            # fault (see IIODigitizerBackend._refill_dma_buffer()'s
            # docstring) -- automatic DMA rearm is now blocked until this
            # is cleared. acknowledge_dma_recovery() is purely a passive
            # readback check (never writes dma_enable), so attempting it
            # right away is safe: either the fault was transient and this
            # clears it, or the channel is genuinely wedged and it raises
            # again, in which case a board restart is needed regardless.
            try:
                self._scope.acknowledge_dma_recovery()
            except RuntimeError:
                log.error(
                    "Scope ch%d: DMA fault did not clear -- board restart likely required",
                    self._channel,
                    exc_info=True,
                )
                self.ui.lblRecordingStatus.setText("DMA fault -- restart the board, then reconnect")
                return
            log.info("Scope ch%d: DMA fault cleared, DMA capture is usable again", self._channel)
            self.ui.lblRecordingStatus.setText("Stopped (recovered from DMA fault)")
            return

        self.ui.lblRecordingStatus.setText("Stopped")

    def stop_dma_sync(self) -> None:
        """Blocking stop for use during application shutdown/reconnect only.

        Handles both halves of what the UI's own Stop button does: an
        in-progress DMA worker (if any), and -- regardless of whether DMA
        was involved -- the hardware ENABLE bit itself. Closing the app
        (or reconnecting) without clicking Stop first used to leave a
        plain viewer-only measurement armed indefinitely, since nothing
        in the shutdown path ever called scope.stop() for that case.

        Confirmed live against the IIO backend as the root cause of a
        "frozen viewer" that looked like a stuck device or driver bug:
        ENABLE is a real, persistent hardware register there (not a
        software flag like the previous driver), so it stays set across
        the whole app being closed and reopened. Re-arming with the exact
        same trigger settings on relaunch writes the same value (1) over
        an already-1 register -- no 0-to-1 transition -- so the core
        never gets a fresh re-arm pulse, and the on-chip pulse-viewer
        memory just keeps showing whatever it last captured before the
        app closed, looking permanently frozen.
        """
        worker = self._dma_worker
        thread = self._dma_thread
        self._ensure_disarmed()
        if worker is not None:
            worker.stop()
        if thread is not None:
            if not thread.wait(5000):
                # QThread.terminate() can bypass stream_to_file()'s finally
                # block and strand the native IIO buffer. Leave the worker
                # alive to complete its cancellation cleanup instead.
                log.error(
                    "Scope DMA thread did not stop within 5 seconds; "
                    "leaving it alive rather than bypassing buffer cleanup"
                )
            else:
                self._dma_thread = None
                self._dma_worker = None

        self._ensure_disarmed()

    def _ensure_disarmed(self) -> None:
        """Best-effort scope.stop() for shutdown/reconnect -- see
        stop_dma_sync()'s docstring for why this exists. Retries briefly
        on the known transient -EBUSY window right after a DMA buffer
        closes; logs and gives up rather than blocking or crashing
        shutdown if it never clears.
        """
        for attempt in range(10):
            try:
                if self._scope.get_enable():
                    self._scope.stop()
                return
            except OSError as e:
                if getattr(e, "errno", None) != 16 or attempt == 9:
                    log.warning(
                        "Scope ch%d: failed to disarm during shutdown", self._channel, exc_info=True
                    )
                    return
                time.sleep(0.02)
            except Exception:
                log.warning(
                    "Scope ch%d: failed to disarm during shutdown", self._channel, exc_info=True
                )
                return
