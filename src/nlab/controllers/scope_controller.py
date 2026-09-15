from __future__ import annotations

import logging
import time
from enum import IntEnum
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSettings, Qt, QThread, QThreadPool, QTimer
from PySide6.QtWidgets import QFileDialog, QSlider, QSpinBox, QWidget

from nlab.analysis.waveform_file import (
    MappedWaveformFile,
    WaveformFileIndex,
    WaveformFrame,
)
from nlab.hardware.digitizer.dma import (
    FILE_HEADER_STRUCT,
    IIOScopeDmaStreamer,
    ScopeDmaStreamer,
)
from nlab.hardware.digitizer.scope import (
    SCOPE_ADC_SAMPLE_PERIOD_NS,
    SCOPE_DATAPATH_CLOCK_PERIOD_NS,
    ListSpec,
    RangeSpec,
    Scope,
    ScopeParam,
    TriggerMode,
)
from nlab.ui.ui_scope_view import Ui_ScopeView
from nlab.views.plot_viewbox import ModifierZoomViewBox
from nlab.views.responsive_layout import configure_scope_layout
from nlab.views.time_axis import format_duration_ns, time_axis_scale
from nlab.workers.dma_workers import IIOScopeDmaWorker, ScopeDmaWorker
from nlab.workers.scope_auto_setup_worker import (
    ScopeAutoSetupResult,
    ScopeAutoSetupWorker,
)
from nlab.workers.scope_worker import ScopeWorker
from nlab.workers.waveform_file_worker import WaveformFileIndexWorker

log = logging.getLogger(__name__)

_SCOPE_CONTROL_TOOLTIPS = {
    "comboTriggerMode": "Selects the condition that starts each scope frame.",
    "spinTriggerLevel": "Sets the raw ADC threshold used by the selected trigger mode.",
    "spinDacValue": "Sets the analog front-end baseline (DC offset) DAC code.",
    "spinPretrigger": (
        "Sets the pretrigger duration in nanoseconds (2 ns per full-rate ADC sample)."
    ),
    "spinFrameSamples": "Sets the frame duration in nanoseconds.",
    "spinFrameGap": (
        "In Periodic mode, adds this delay in nanoseconds after each frame."
    ),
    "spinTime": (
        "Stops the acquisition automatically after this many seconds; 0 runs until stopped."
    ),
    "comboDisplayMode": "Chooses a raw waveform trace or an accumulated persistence display.",
    "spinRefreshRate": "Sets how often the live waveform is requested and redrawn, in hertz.",
    "btnAutoSetup": (
        "Finds a unipolar pulse, moves its baseline near the opposite ADC rail "
        "for maximum dynamic range, and selects a noise-aware edge trigger. "
        "Click again to cancel."
    ),
    "spinFileSamplePeriod": (
        "CAEN binary files do not store their ADC sample period. Set it here "
        "to obtain the correct time axis."
    ),
    "comboFileChannel": (
        "Selects the CAEN board and channel whose waveform events are displayed."
    ),
    "spinFileFrame": "Selects the waveform event or NDMA scope frame to display.",
}


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
    # The scope captures raw ADC samples at 500 MSPS (2 ns/sample).  Its
    # lightweight viewer memory contains one four-sample boxcar average per
    # 125 MHz datapath beat, hence one displayed point every 8 ns. See the
    # PetaLinux project's user API, "Timebase" and "Viewer".
    _SAMPLE_PERIOD_NS = SCOPE_ADC_SAMPLE_PERIOD_NS
    _VIEWER_POINT_PERIOD_NS = SCOPE_DATAPATH_CLOCK_PERIOD_NS

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
        configure_scope_layout(self, self.ui)
        self._apply_control_tooltips()

        self._acquiring = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._request_frame)

        self._dma_worker: ScopeDmaWorker | IIOScopeDmaWorker | None = None
        self._dma_thread: QThread | None = None
        self._dma_filepath: Path | None = None
        self._dma_counter = 0

        self._auto_setup_worker: ScopeAutoSetupWorker | None = None
        self._auto_setup_thread: QThread | None = None
        self._auto_setup_result: ScopeAutoSetupResult | None = None
        self._auto_setup_error: str | None = None

        self._waveform_index_worker: WaveformFileIndexWorker | None = None
        self._waveform_index_thread: QThread | None = None
        self._waveform_index_result: WaveformFileIndex | None = None
        self._waveform_index_error: str | None = None
        self._waveform_index_abandon = False
        self._waveform_file: MappedWaveformFile | None = None
        self._file_previous_display_mode = DisplayMode.RAW
        self._file_frame_timer = QTimer(self)
        self._file_frame_timer.setSingleShot(True)
        self._file_frame_timer.setInterval(30)
        self._file_frame_timer.timeout.connect(self._render_file_frame)

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
        self.ui.groupFileBrowser.hide()

    @property
    def channel(self) -> int:
        return self._channel

    # ------------------------------------------------------------------
    # Spec application
    # ------------------------------------------------------------------

    def _apply_control_tooltips(self) -> None:
        for object_name, tooltip in _SCOPE_CONTROL_TOOLTIPS.items():
            getattr(self.ui, object_name).setToolTip(tooltip)

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
        self._apply_scaled_range_to_spinbox(
            self.ui.spinPretrigger, spec, self._SAMPLE_PERIOD_NS
        )
        self.ui.labelPretrigger.setText("Pretrigger:")

        spec = specs[ScopeParam.FRAME_SAMPLES]
        assert isinstance(spec, RangeSpec)
        self._apply_scaled_range_to_spinbox(
            self.ui.spinFrameSamples, spec, self._SAMPLE_PERIOD_NS
        )
        self.ui.labelFrameSamples.setText("Frame:")

        spec = specs[ScopeParam.FRAME_PERIOD_CYCLES]
        assert isinstance(spec, RangeSpec)
        self._apply_scaled_range_to_spinbox(
            self.ui.spinFrameGap, spec, self._VIEWER_POINT_PERIOD_NS
        )
        self.ui.labelFrameGap.setText("Periodic gap:")

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
    def _apply_scaled_range_to_spinbox(
        spinbox: QSpinBox, spec: RangeSpec, scale: int
    ) -> None:
        """Apply a hardware range after converting its units to nanoseconds."""
        spinbox.setMinimum(int(spec.min_val) * scale)
        spinbox.setMaximum(int(spec.max_val) * scale)
        spinbox.setSingleStep((int(spec.step) or 1) * scale)
        spinbox.setValue(int(spec.default) * scale)
        spinbox.setSuffix(" ns")

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
        self.ui.spinPretrigger.setValue(
            self._scope.get_pretrigger_samples() * self._SAMPLE_PERIOD_NS
        )
        self.ui.spinFrameSamples.setValue(
            self._scope.get_frame_samples() * self._SAMPLE_PERIOD_NS
        )
        self.ui.spinFrameGap.setValue(
            self._scope.get_frame_period_cycles() * self._VIEWER_POINT_PERIOD_NS
        )
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
        if self._waveform_file is not None:
            self._render_file_frame()
            return
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

    def _frame_samples_from_ui(self) -> int:
        return self.ui.spinFrameSamples.value() // self._SAMPLE_PERIOD_NS

    def _frame_exceeds_viewer_limit(self) -> bool:
        limit = self._viewer_frame_limit()
        return limit is not None and self._frame_samples_from_ui() > limit

    def _viewer_limit_message(self) -> str:
        limit = self._viewer_frame_limit()
        assert limit is not None
        return (
            f"Live preview is truncated above {limit * self._SAMPLE_PERIOD_NS} ns "
            f"({limit} ADC samples). "
            "Enable Record DMA frames to save complete frames."
        )

    def _viewer_truncation_message(self, displayed: int, expected: int) -> str:
        suffix = (
            "DMA is recording the full frame."
            if self._dma_worker is not None
            else "Enable Record DMA frames to save the full frame."
        )
        return f"Live preview truncated: showing {displayed} of {expected} points. {suffix}"

    def _update_viewer_transport_hint(self) -> None:
        frame_ns = self.ui.spinFrameSamples.value()
        frame_samples = self._frame_samples_from_ui()
        tip = (
            f"Sets the frame length: {format_duration_ns(frame_ns)} "
            f"({frame_samples} ADC samples at {self._SAMPLE_PERIOD_NS} ns/sample). "
            f"The live viewer plots {self._VIEWER_POINT_PERIOD_NS} ns averages."
        )
        limit = self._viewer_frame_limit()
        if limit is not None:
            tip += (
                f" Legacy live preview may truncate above {limit} samples; "
                "DMA records the complete frame."
            )
        self.ui.spinFrameSamples.setToolTip(tip)

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
            lambda: self._scope.set_pretrigger_samples(
                self.ui.spinPretrigger.value() // self._SAMPLE_PERIOD_NS
            )
        )
        self.ui.spinFrameSamples.editingFinished.connect(self._on_frame_samples_changed)
        self.ui.spinFrameGap.editingFinished.connect(
            lambda: self._scope.set_frame_period_cycles(
                self.ui.spinFrameGap.value() // self._VIEWER_POINT_PERIOD_NS
            )
        )
        self.ui.comboTriggerMode.currentIndexChanged.connect(
            self._on_trigger_mode_changed
        )
        self.ui.cbDmaEnable.toggled.connect(self._on_dma_toggled)
        self.ui.btnStart.clicked.connect(self._on_start)
        self.ui.btnStop.clicked.connect(self._on_stop)
        self.ui.btnAutoSetup.clicked.connect(self._on_auto_setup)
        self.ui.btnAcquireFrame.clicked.connect(self._on_acquire_frame)
        self.ui.btnDmaFile.clicked.connect(self._on_dma_file)
        self.ui.btnCloseWaveformFile.clicked.connect(self._on_close_waveform_file)
        self.ui.comboFileChannel.currentIndexChanged.connect(
            self._on_file_channel_changed
        )
        self.ui.sliderFileFrame.valueChanged.connect(self.ui.spinFileFrame.setValue)
        self.ui.spinFileFrame.valueChanged.connect(self.ui.sliderFileFrame.setValue)
        self.ui.spinFileFrame.valueChanged.connect(self._queue_file_frame)
        self.ui.btnPreviousFileFrame.clicked.connect(
            lambda: self.ui.spinFileFrame.setValue(self.ui.spinFileFrame.value() - 1)
        )
        self.ui.btnNextFileFrame.clicked.connect(
            lambda: self.ui.spinFileFrame.setValue(self.ui.spinFileFrame.value() + 1)
        )
        self.ui.spinFileSamplePeriod.editingFinished.connect(self._render_file_frame)

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
        if self._waveform_file is not None or self._waveform_index_thread is not None:
            self.ui.lblRecordingStatus.setText(
                "Close the waveform file before starting live acquisition"
            )
            return
        if self._auto_setup_thread is not None:
            return
        if self._dma_worker is not None or self._dma_thread is not None:
            log.warning(
                "Scope ch%d: refusing to start while the previous DMA worker is still closing",
                self._channel,
            )
            return
        self.ui.btnStart.setChecked(True)
        self.ui.btnStart.setEnabled(False)
        self.ui.btnStop.setChecked(False)
        self.ui.btnAcquireFrame.setEnabled(False)
        self.ui.cbDmaEnable.setEnabled(False)
        self.ui.btnDmaFile.setEnabled(False)
        self.ui.btnAutoSetup.setEnabled(False)

        if self.ui.cbDmaEnable.isChecked() and self._scope_dma is not None:
            self._start_with_dma()
        else:
            self._start_polling_only()

    def _start_polling_only(self) -> None:
        self._scope.start()
        self.ui.btnStop.setEnabled(True)
        interval_ms = 1000 // self.ui.spinRefreshRate.value()
        self._refresh_timer.start(interval_ms)
        if self._frame_exceeds_viewer_limit():
            self.ui.lblRecordingStatus.setText(self._viewer_limit_message())
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
            frame_samples = self._frame_samples_from_ui()
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
        self._dma_worker.finished.connect(
            self._dma_thread.quit,
            Qt.ConnectionType.DirectConnection,
        )
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
        self._refresh_timer.start(interval_ms)
        self.ui.btnStop.setEnabled(True)
        self.ui.lblRecordingStatus.setText(
            "Recording full frames; live preview is truncated."
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
        self.ui.btnAutoSetup.setEnabled(True)
        log.info("Scope ch%d: acquisition stopped", self._channel)

    # ------------------------------------------------------------------
    # Auto Setup
    # ------------------------------------------------------------------

    def _on_auto_setup(self) -> None:
        if self._waveform_file is not None or self._waveform_index_thread is not None:
            self.ui.lblRecordingStatus.setText(
                "Close the waveform file before running Auto Setup"
            )
            return
        if self._auto_setup_worker is not None:
            self._auto_setup_worker.stop()
            self.ui.btnAutoSetup.setEnabled(False)
            self.ui.lblRecordingStatus.setText("Cancelling Auto Setup...")
            return
        if self._dma_worker is not None or self._dma_thread is not None:
            self.ui.lblRecordingStatus.setText("Stop DMA recording before Auto Setup")
            return
        if self._refresh_timer.isActive() or self._scope.get_enable():
            self.ui.lblRecordingStatus.setText("Stop acquisition before Auto Setup")
            return
        if self.ui.cbDmaEnable.isChecked():
            self.ui.lblRecordingStatus.setText("Disable Record DMA frames before Auto Setup")
            return

        self._auto_setup_result = None
        self._auto_setup_error = None
        worker = ScopeAutoSetupWorker(self._scope.create_isolated_client)
        thread = QThread(self)
        self._auto_setup_worker = worker
        self._auto_setup_thread = thread
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self.ui.lblRecordingStatus.setText)
        worker.succeeded.connect(self._on_auto_setup_succeeded)
        worker.error.connect(self._on_auto_setup_error)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_auto_setup_finished)

        self._set_auto_setup_busy(True)
        self.ui.lblRecordingStatus.setText("Auto Setup: connecting...")
        thread.start()
        log.info("Scope ch%d: Auto Setup started", self._channel)

    def _set_auto_setup_busy(self, busy: bool) -> None:
        self._set_controls_enabled(not busy)
        self.ui.btnStart.setEnabled(not busy)
        self.ui.btnStop.setEnabled(False)
        self.ui.btnAcquireFrame.setEnabled(not busy)
        self.ui.btnDmaFile.setEnabled(not busy)
        self.ui.btnAutoSetup.setEnabled(True)
        self.ui.btnAutoSetup.setText("Cancel Auto" if busy else "Auto Setup")

    def _on_auto_setup_succeeded(self, result: ScopeAutoSetupResult) -> None:
        self._auto_setup_result = result

    def _on_auto_setup_error(self, message: str) -> None:
        self._auto_setup_error = message

    def _on_auto_setup_finished(self) -> None:
        result = self._auto_setup_result
        error = self._auto_setup_error
        self._auto_setup_worker = None
        self._auto_setup_thread = None
        self._set_auto_setup_busy(False)

        try:
            self._load_hardware_state()
            self._update_frame_gap_enabled()
            self._update_axis_ranges()
        except Exception as exc:
            log.exception("Scope ch%d: failed to refresh after Auto Setup", self._channel)
            self.ui.lblRecordingStatus.setText(f"Auto Setup refresh failed: {exc}")
            return

        if result is None:
            self.ui.lblRecordingStatus.setText(f"Auto Setup failed: {error or 'unknown error'}")
            return

        frame = result.frame[: self._frame_samples_from_ui() // 4]
        raw_time = np.arange(len(frame)) * self._VIEWER_POINT_PERIOD_NS
        self._on_frame_received([raw_time, frame])
        verification = "verified" if result.verified else "set; pulse not re-observed"
        self.ui.lblRecordingStatus.setText(
            f"Auto Setup {verification}: DAC {result.dac_value}, "
            f"{result.trigger_mode.name.lower().replace('_', ' ')} at "
            f"{result.trigger_level}"
        )
        log.info(
            "Scope ch%d: Auto Setup complete: DAC=%d, mode=%s, level=%d, "
            "baseline=%.1f, noise=%.1f, amplitude=%.1f, verified=%s",
            self._channel,
            result.dac_value,
            result.trigger_mode.name,
            result.trigger_level,
            result.baseline,
            result.noise_sigma,
            result.pulse_amplitude,
            result.verified,
        )

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
        expected = self._frame_samples_from_ui() // 4
        if len(y_voltage) < expected:
            self.ui.lblRecordingStatus.setText(
                self._viewer_truncation_message(len(y_voltage), expected)
            )
        if self._display_mode == DisplayMode.RAW:
            self._raw_curve.setData(x_time / self._time_scale.ns_per_unit, y_voltage)
        else:
            self._rasterize_frame(y_voltage)
            self._persistence_img.setImage(
                self._persistence_buffer, autoLevels=False, levels=(0, 1)
            )

    def _on_frame_samples_changed(self) -> None:
        frame_samples = self._frame_samples_from_ui()
        self._scope.set_frame_samples(frame_samples)
        self._display_nx = frame_samples // 4
        self._display_ny = self._display_nx * self._Y_SCALE_FACTOR
        self._persistence_buffer = np.zeros((self._display_nx, self._display_ny), dtype=np.float32)
        self._update_axis_ranges()
        if self._frame_exceeds_viewer_limit():
            self.ui.lblRecordingStatus.setText(self._viewer_limit_message())

    def _on_acquire_frame(self) -> None:
        if self._waveform_file is not None or self._waveform_index_thread is not None:
            self.ui.lblRecordingStatus.setText(
                "Close the waveform file before acquiring a live frame"
            )
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
        frame_samples = self._frame_samples_from_ui()
        frame = raw_frame[: frame_samples // 4]
        expected = frame_samples // 4
        if len(frame) < expected:
            self.ui.lblRecordingStatus.setText(
                self._viewer_truncation_message(len(frame), expected)
            )
        time_arr = (
            np.arange(len(frame))
            * self._VIEWER_POINT_PERIOD_NS
            / self._time_scale.ns_per_unit
        )
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
        # A legacy text viewer can return only the prefix that fits in one
        # sysfs page. Keep that prefix at its real time coordinates instead
        # of stretching it across the full configured frame.
        x_float = np.arange(n_samples, dtype=np.float32)
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
        self._time_axis_label = layout.addLabel("Time [ns]", col=1)

    def _setup_persistence_layer(self) -> None:
        self._display_nx = self._frame_samples_from_ui() // 4
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
        frame_samples = self._frame_samples_from_ui()
        duration_ns = self.ui.spinFrameSamples.value()
        self._time_scale = time_axis_scale(duration_ns)
        duration = self._time_scale.from_nanoseconds(duration_ns)
        self._time_axis_label.setText(
            f"Time [{self._time_scale.unit}]  "
            f"({self._SAMPLE_PERIOD_NS} ns/ADC sample; "
            f"{self._VIEWER_POINT_PERIOD_NS} ns/viewer point)"
        )
        _, plotted_frame = self._raw_curve.getData()
        if plotted_frame is not None and len(plotted_frame) > 0:
            plotted_frame = plotted_frame[: frame_samples // 4]
            plotted_time = (
                np.arange(len(plotted_frame))
                * self._VIEWER_POINT_PERIOD_NS
                / self._time_scale.ns_per_unit
            )
            self._raw_curve.setData(plotted_time, plotted_frame)
        vb = self._plot_item.getViewBox()
        vb.setXRange(0, duration, padding=0)
        vb.setYRange(self._Y_MIN, self._Y_MAX, padding=0)
        self._persistence_img.setRect(
            QRectF(0, self._Y_MIN, duration, self._Y_MAX - self._Y_MIN)
        )
        self._update_viewer_transport_hint()

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

    # ------------------------------------------------------------------
    # File waveform browser
    # ------------------------------------------------------------------

    def open_waveform_file(self, path: Path) -> None:
        """Index a CAEN or NLab scope binary without blocking the GUI."""
        if self._refresh_timer.isActive() or self._scope.get_enable():
            raise RuntimeError("Stop live scope acquisition before opening a waveform file")
        if self._dma_worker is not None or self._dma_thread is not None:
            raise RuntimeError("Stop scope DMA before opening a waveform file")
        if self._auto_setup_thread is not None:
            raise RuntimeError("Wait for Auto Setup before opening a waveform file")
        if self._waveform_index_thread is not None:
            raise RuntimeError("A waveform file is already being indexed")

        self._close_mapped_waveform()
        self._waveform_index_result = None
        self._waveform_index_error = None
        self._waveform_index_abandon = False
        worker = WaveformFileIndexWorker(path)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_waveform_index_progress)
        worker.loaded.connect(self._on_waveform_index_loaded)
        worker.error.connect(self._on_waveform_index_error)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_waveform_index_finished)
        self._waveform_index_worker = worker
        self._waveform_index_thread = thread
        self.ui.labelFileChannel.hide()
        self.ui.comboFileChannel.hide()
        self.ui.groupFileBrowser.show()
        self._set_file_browser_busy(True)
        self.ui.lblFileFrameInfo.setText(f"Indexing {path.name}...")
        thread.start()

    def _set_file_browser_busy(self, busy: bool) -> None:
        self._set_controls_enabled(not busy)
        self.ui.groupDisplay.setEnabled(not busy)
        self.ui.btnStart.setEnabled(not busy)
        self.ui.btnStop.setEnabled(False)
        self.ui.btnAcquireFrame.setEnabled(not busy)
        self.ui.btnDmaFile.setEnabled(not busy)
        self.ui.btnAutoSetup.setEnabled(not busy)
        self.ui.btnCloseWaveformFile.setEnabled(busy)

    def _on_waveform_index_progress(self, processed: int, total: int) -> None:
        percent = 100.0 * processed / total if total else 0.0
        self.ui.lblFileFrameInfo.setText(f"Indexing waveform records: {percent:.1f}%")

    def _on_waveform_index_loaded(self, index: object) -> None:
        if self._waveform_index_abandon:
            return
        if isinstance(index, WaveformFileIndex):
            self._waveform_index_result = index
        else:
            self._waveform_index_error = "Waveform index worker returned invalid data"

    def _on_waveform_index_error(self, message: str) -> None:
        self._waveform_index_error = message

    def _on_waveform_index_finished(self) -> None:
        abandoned = self._waveform_index_abandon
        index = None if abandoned else self._waveform_index_result
        error = None if abandoned else self._waveform_index_error
        self._waveform_index_worker = None
        self._waveform_index_thread = None
        self._waveform_index_result = None
        self._waveform_index_error = None
        self._waveform_index_abandon = False
        if index is None:
            self._set_file_browser_busy(False)
            if abandoned:
                self.ui.btnCloseWaveformFile.setEnabled(False)
                self.ui.lblFileFrameInfo.setText("No waveform file loaded.")
                self.ui.groupFileBrowser.hide()
                return
            self.ui.btnCloseWaveformFile.setEnabled(True)
            self.ui.lblFileFrameInfo.setText(
                f"Waveform file error: {error}" if error else "Waveform indexing cancelled."
            )
            return

        try:
            self._waveform_file = MappedWaveformFile(index)
        except Exception as exc:
            self._set_file_browser_busy(False)
            self.ui.btnCloseWaveformFile.setEnabled(True)
            self.ui.lblFileFrameInfo.setText(f"Waveform file error: {exc}")
            return

        self.ui.comboFileChannel.blockSignals(True)
        try:
            self.ui.comboFileChannel.clear()
            for source in index.channels:
                label = f"Channel {source.channel}"
                if source.board is not None:
                    label = f"Board {source.board} / {label}"
                self.ui.comboFileChannel.addItem(label)
            initial_source_index = next(
                (
                    source_index
                    for source_index, source in enumerate(index.channels)
                    if source.channel == self._channel
                ),
                0,
            )
            self.ui.comboFileChannel.setCurrentIndex(initial_source_index)
        finally:
            self.ui.comboFileChannel.blockSignals(False)
        has_multiple_sources = len(index.channels) > 1
        self.ui.labelFileChannel.setVisible(has_multiple_sources)
        self.ui.comboFileChannel.setVisible(has_multiple_sources)
        self.ui.comboFileChannel.setEnabled(has_multiple_sources)
        self.ui.spinFileSamplePeriod.setValue(index.sample_period_ns or 2.0)
        self.ui.spinFileSamplePeriod.setEnabled(index.sample_period_ns is None)
        self._file_previous_display_mode = self._display_mode
        self.ui.comboDisplayMode.setCurrentIndex(DisplayMode.RAW)
        self._set_file_browser_busy(True)
        self._on_file_channel_changed(initial_source_index)

    def _on_file_channel_changed(self, source_index: int) -> None:
        source_file = self._waveform_file
        if source_file is None or not 0 <= source_index < len(source_file.index.channels):
            return
        self._file_frame_timer.stop()
        source = source_file.index.channels[source_index]
        maximum = source.frame_count - 1
        self.ui.sliderFileFrame.blockSignals(True)
        self.ui.spinFileFrame.blockSignals(True)
        try:
            self.ui.sliderFileFrame.setRange(0, maximum)
            self.ui.spinFileFrame.setRange(0, maximum)
            self.ui.sliderFileFrame.setValue(0)
            self.ui.spinFileFrame.setValue(0)
            self.ui.spinFileFrame.setSuffix(f" / {maximum}")
        finally:
            self.ui.sliderFileFrame.blockSignals(False)
            self.ui.spinFileFrame.blockSignals(False)
        self.ui.sliderFileFrame.setEnabled(True)
        self.ui.spinFileFrame.setEnabled(True)
        self.ui.btnPreviousFileFrame.setEnabled(maximum > 0)
        self.ui.btnNextFileFrame.setEnabled(maximum > 0)
        self._render_file_frame()

    def _queue_file_frame(self) -> None:
        if self._waveform_file is not None:
            self._file_frame_timer.start()

    def _render_file_frame(self) -> None:
        source = self._waveform_file
        if source is None:
            return
        try:
            frame_index = self.ui.spinFileFrame.value()
            source_index = self.ui.comboFileChannel.currentIndex()
            frame = source.frame(frame_index, source_index)
            sample_period_ns = self.ui.spinFileSamplePeriod.value()
            duration_ns = max(sample_period_ns, len(frame.samples) * sample_period_ns)
            scale = time_axis_scale(duration_ns)
            x_time = (
                np.arange(len(frame.samples), dtype=np.float64)
                * sample_period_ns
                / scale.ns_per_unit
            )
            self._raw_curve.setData(x_time, frame.samples)
            # Match pyqtgraph's ``A`` action for every selected file frame:
            # the waveform may occupy only a small part of the ADC's full
            # signed range, so fixed hardware limits make it look flat.
            self._plot_item.getViewBox().autoRange(
                items=[self._raw_curve],
                padding=0.02,
            )
            self._time_axis_label.setText(
                f"Time [{scale.unit}] ({sample_period_ns:g} ns/file sample)"
            )
            self.ui.lblFileFrameInfo.setText(self._file_frame_description(frame_index, frame))
        except Exception as exc:
            log.exception("Failed to display waveform file frame")
            self.ui.lblFileFrameInfo.setText(f"Waveform frame error: {exc}")

    def _file_frame_description(self, frame_index: int, frame: WaveformFrame) -> str:
        source = self._waveform_file
        assert source is not None
        if source.index.caen_info is None:
            return (
                f"{source.index.path.name} | frame {frame_index:,} | "
                f"timestamp {frame.timestamp} (8 ns ticks) | {len(frame.samples):,} samples"
            )
        ratio = (
            (frame.long_gate - frame.short_gate) / frame.long_gate
            if frame.long_gate and frame.short_gate is not None
            else float("nan")
        )
        return (
            f"{source.index.path.name} | event {frame_index:,} | board {frame.board}, "
            f"channel {frame.channel} | timestamp {frame.timestamp} ps | "
            f"long {frame.long_gate}, short {frame.short_gate}, PSD {ratio:.4f} | "
            f"flags 0x{(frame.flags or 0):08X} | {len(frame.samples):,} samples"
        )

    def _on_close_waveform_file(self) -> None:
        if self._waveform_index_worker is not None:
            self._waveform_index_abandon = True
            self._waveform_index_worker.stop()
            self.ui.lblFileFrameInfo.setText("Cancelling waveform indexing...")
            return
        self._close_mapped_waveform()

    def _close_mapped_waveform(self) -> None:
        self._file_frame_timer.stop()
        source = self._waveform_file
        had_source = source is not None
        if source is not None:
            self._raw_curve.setData([], [])
            source.close()
            self._waveform_file = None
        if not hasattr(self, "ui"):
            return
        self.ui.sliderFileFrame.setEnabled(False)
        self.ui.spinFileFrame.setEnabled(False)
        self.ui.btnPreviousFileFrame.setEnabled(False)
        self.ui.btnNextFileFrame.setEnabled(False)
        self.ui.spinFileSamplePeriod.setEnabled(False)
        self.ui.comboFileChannel.blockSignals(True)
        try:
            self.ui.comboFileChannel.clear()
        finally:
            self.ui.comboFileChannel.blockSignals(False)
        self.ui.comboFileChannel.setEnabled(False)
        self.ui.comboFileChannel.hide()
        self.ui.labelFileChannel.hide()
        self.ui.btnCloseWaveformFile.setEnabled(False)
        self.ui.lblFileFrameInfo.setText("No waveform file loaded.")
        self._set_file_browser_busy(False)
        self.ui.groupFileBrowser.hide()
        if had_source:
            self.ui.comboDisplayMode.setCurrentIndex(self._file_previous_display_mode)
            self._update_axis_ranges()

    def _stop_waveform_file_sync(self) -> None:
        worker = self._waveform_index_worker
        thread = self._waveform_index_thread
        if worker is not None:
            self._waveform_index_abandon = True
            worker.stop()
        if thread is not None and not thread.wait(5000):
            log.error("Waveform index thread did not stop within 5 seconds")
        else:
            self._waveform_index_worker = None
            self._waveform_index_thread = None
        self._waveform_index_result = None
        self._waveform_index_error = None
        self._close_mapped_waveform()

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
        self.ui.btnAutoSetup.setEnabled(True)

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
        self._stop_waveform_file_sync()

        auto_worker = self._auto_setup_worker
        auto_thread = self._auto_setup_thread
        if auto_worker is not None:
            auto_worker.stop()
        if auto_thread is not None and not auto_thread.wait(5000):
            log.error(
                "Scope Auto Setup thread did not stop within 5 seconds; "
                "leaving it alive to finish rollback"
            )

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
