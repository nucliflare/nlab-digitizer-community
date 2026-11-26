from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    QPointF,
    QRectF,
    QSettings,
    Qt,
    QThread,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QActionGroup, QCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMenu,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QWidget,
)

from nlab import __version__, settings_keys
from nlab.analysis.scope_distribution import AccumulatedAmplitudeHistogram, HistogramMode
from nlab.analysis.waveform_spectrum import (
    UNIT_LABELS,
    AccumulatedSpectrum,
    FramePsd,
    SpectrumAveraging,
    SpectrumUnit,
    SpectrumWindow,
    Spur,
    compensated_psd,
    decimation_factor,
    find_spurs,
    format_frequency,
    line_threshold_db,
    mean_decimation_response,
    parseval_check,
    plotted_bins,
    to_display_units,
    write_spectrum_csv,
)
from nlab.controllers.measurement_series import (
    MeasurementSeries,
    owner_series,
    time_limit_required_message,
)
from nlab.hardware.digitizer.current_monitor import (
    ScopeCurrentAccumulator,
    ScopeCurrentPreview,
    ScopeCurrentRuntime,
)
from nlab.hardware.digitizer.dma import (
    FILE_HEADER_STRUCT,
    IIOScopeDmaStreamer,
    ScopeFrameBuffer,
)
from nlab.hardware.digitizer.scope import (
    SCOPE_ADC_SAMPLE_PERIOD_NS,
    SCOPE_DATAPATH_CLOCK_PERIOD_NS,
    ListSpec,
    RangeSpec,
    Scope,
    ScopeParam,
    TriggerMode,
    range_spec,
)
from nlab.help.context import GRAPHICS_HELP_TOPIC_KEY
from nlab.ui.ui_scope_view import Ui_ScopeView
from nlab.utils.recording_paths import (
    available_path,
    chosen_recording_path,
    measurement_folder,
    unique_recording_path,
)
from nlab.views import theme
from nlab.views.frequency_axis import set_frequency_axis
from nlab.views.numeric_controls import normalize_numeric_control
from nlab.views.panel_status import PanelStatus
from nlab.views.plot_viewbox import (
    DraggableScopeCurve,
    HistogramViewBox,
    ModifierZoomViewBox,
    show_frame_axes,
)
from nlab.views.responsive_layout import configure_scope_layout
from nlab.views.spec_binding import bind_slider, bind_spinbox
from nlab.views.time_axis import format_duration_ns, time_axis_scale
from nlab.views.tooltips import wrap_tooltip
from nlab.workers.dma_workers import IIOScopeDmaWorker
from nlab.workers.scope_auto_setup_worker import (
    ScopeAutoSetupResult,
    ScopeAutoSetupWorker,
)
from nlab.workers.scope_sum_worker import ScopeSumWorker
from nlab.workers.scope_worker import ScopeWorker

log = logging.getLogger(__name__)

_CURRENT_DMA_OPTIMIZED_IP_VERSION = 122
_CURRENT_DMA_FRAME_SAMPLES = 8188
_CURRENT_DMA_GAP_CYCLES = 12_500

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
}

_SPECTRUM_COUNTS = (10, 30, 100, 300, 1000, 0)
_SPECTRUM_SETTINGS = (
    "window", "averaging", "average_count", "unit", "log_frequency", "mean4_compensation",
)
# Level and edge triggers synchronise frames to pulses (pulse content, not baseline).
_PULSE_SYNCHRONISED_TRIGGERS = (
    TriggerMode.ANY_ABOVE, TriggerMode.ANY_BELOW,
    TriggerMode.RISING_EDGE, TriggerMode.FALLING_EDGE,
)
_SPECTRUM_SPUR_INTERVAL_S = 0.5
_SPECTRUM_NOT_VALIDATED = "Software analysis; not validated on live hardware."


class DisplayMode(IntEnum):
    PERSISTENCE = 0
    RAW = 1


@dataclass(frozen=True)
class _WaveformDragState:
    x: np.ndarray
    y: np.ndarray
    dac_value: int
    pretrigger_ns: int
    adc_per_dac: float
    status_text: str


@dataclass(frozen=True)
class _PretriggerLineDragState:
    x: np.ndarray | None
    y: np.ndarray | None
    pretrigger_ns: int
    status_text: str


class ScopeController(QWidget):
    """View + controller for a single Scope channel.

    Loaded from ui_scope_view.ui.  Drop into any QTabWidget via addTab().
    Widget ranges and defaults are driven entirely by Scope.specs at runtime.
    """

    # Fixed vertical raster resolution (~62 ADC codes per row over the 64k
    # display range). It no longer scales with frame length: at 20 rows per
    # column a 4096-sample frame needed an 84 MB float image (~100 ms per
    # frame to rasterize and paint); this keeps it at a few ms.
    _PERSISTENCE_ROWS = 1024
    _Y_MIN = -32_000
    _Y_MAX = 32_000
    # The scope captures raw ADC samples at 500 MSPS (2 ns/sample).  Its
    # lightweight viewer memory contains one four-sample boxcar average per
    # 125 MHz datapath beat, hence one displayed point every 8 ns. See the
    # PetaLinux project's user API, "Timebase" and "Viewer".
    _SAMPLE_PERIOD_NS = SCOPE_ADC_SAMPLE_PERIOD_NS
    _VIEWER_POINT_PERIOD_NS = SCOPE_DATAPATH_CLOCK_PERIOD_NS
    _THRESHOLD_MARKER_COLOR = "#a66f6f"
    _PRETRIGGER_MARKER_COLOR = "#648b71"
    current_dma_state_changed = Signal(bool, str)
    session_viewer_frame = Signal(int, object, int)
    measurement_active_changed = Signal(bool)
    connection_lost = Signal(str)
    """The board connection was lost; carries the transport error message."""

    def __init__(
        self,
        scope: Scope,
        scope_dma: IIOScopeDmaStreamer | None = None,
        channel: int = 0,
        parent: QWidget | None = None,
        *,
        dma_frame_buffer: ScopeFrameBuffer | None = None,
        current_accumulator: ScopeCurrentAccumulator | None = None,
        measurement_configuration: Callable[[], dict[str, object]] | None = None,
    ) -> None:
        super().__init__(parent)
        self._scope = scope
        self._measurement_configuration = measurement_configuration
        self._scope_dma = scope_dma
        self._sum_worker: ScopeSumWorker | None = None
        self._sum_previous_settings: tuple[TriggerMode, int, int, int] | None = None
        self._dma_frame_buffer = dma_frame_buffer
        self._current_accumulator = current_accumulator
        self._channel = channel
        self.ui = Ui_ScopeView()
        self.ui.setupUi(self)
        self.status = PanelStatus(self)
        configure_scope_layout(self, self.ui)
        self._apply_control_tooltips()

        self._acquiring = False
        # _acquiring describes one in-flight viewer request, not the run.
        self._measurement_active = False
        self._discard_inflight_frame = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._request_frame)
        self._viewer_scope: Scope | None = None
        self._viewer_scope_lock = threading.Lock()

        self._dma_worker: IIOScopeDmaWorker | None = None
        self._dma_thread: QThread | None = None
        self._dma_filepath: Path | None = None
        self._dma_stopping = False
        self._dma_online_current = False
        self._dma_last_error: str | None = None

        self._auto_setup_worker: ScopeAutoSetupWorker | None = None
        self._auto_setup_thread: QThread | None = None
        self._auto_setup_result: ScopeAutoSetupResult | None = None
        self._auto_setup_error: str | None = None
        self._dac_adc_slope: float | None = None
        self._waveform_drag: _WaveformDragState | None = None
        self._pretrigger_line_drag: _PretriggerLineDragState | None = None
        self._waveform_pending_status: str | None = None

        self._measurement_timer = QTimer(self)
        self._measurement_timer.setSingleShot(True)
        self._measurement_timer.timeout.connect(self._on_measurement_timeout)
        self._elapsed_started_at: float | None = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(250)
        self._elapsed_timer.timeout.connect(self._update_elapsed_time)
        self._distribution_enabled = False
        self._amplitude_histogram = AccumulatedAmplitudeHistogram()
        self._spectrum_enabled = False
        self._spectrum = AccumulatedSpectrum()
        self._spectrum_unit = SpectrumUnit.DB
        self._spectrum_log_frequency = False
        self._spectrum_compensate = False
        self._spectrum_display: tuple[np.ndarray, np.ndarray] | None = None
        self._spectrum_spurs: list[Spur] = []
        self._spectrum_spurs_at = -np.inf
        self._spectrum_spurs_frames = 0
        self._spectrum_info = ""
        self._spectrum_cursor = ""
        self._spectrum_log_mode = (False, False)
        self._session_replay_active = False
        self._session_replay_display_mode: int | None = None
        self._replay_hardware_interlocked = False

        self._series = MeasurementSeries(
            self,
            spinbox=self.ui.spinSeriesRuns,
            label=self.ui.labelSeriesRuns,
            name=f"Scope ch{channel}",
            is_idle=self._series_idle,
            start_run=self._start_series_run,
        )

        self._apply_parameter_specs()
        self._send_defaults()
        self._load_hardware_state()
        self._restore_display_settings()
        self._setup_graph()
        self._connect_signals()
        self._update_frame_gap_enabled()
        self._update_viewer_transport_hint()
        self.ui.btnStop.setEnabled(False)
        self._configure_sum_controls()

    @property
    def channel(self) -> int:
        return self._channel

    @property
    def measurement_active(self) -> bool:
        """Last lifecycle-confirmed continuous run; excludes single-frame reads."""
        return bool(getattr(self, "_measurement_active", False))

    @property
    def last_dma_error(self) -> str | None:
        """Error of the latest DMA session, or None; cleared when the next one starts."""
        return self._dma_last_error

    def _set_measurement_active(self, active: bool) -> None:
        if getattr(self, "_measurement_active", False) != active:
            self._measurement_active = active
            # Share the run lifecycle used by the measurement badge, not the
            # per-frame viewer refresh. No hardware counter or polling required.
            elapsed_timer = getattr(self, "_elapsed_timer", None)
            if elapsed_timer is not None:
                if active:
                    self._elapsed_started_at = monotonic()
                    self.ui.labelElapsedTime.setText("Elapsed: 0.0 s")
                    elapsed_timer.start()
                else:
                    self._update_elapsed_time()
                    elapsed_timer.stop()
                    self._elapsed_started_at = None
            signal = getattr(self, "measurement_active_changed", None)
            if signal is not None:
                signal.emit(active)

    def _update_elapsed_time(self) -> None:
        """Display host run duration; neither hardware live time nor sample time."""
        if self._elapsed_started_at is not None:
            seconds = max(0.0, monotonic() - self._elapsed_started_at)
            self.ui.labelElapsedTime.setText(f"Elapsed: {seconds:.1f} s")

    # ------------------------------------------------------------------
    # Spec application
    # ------------------------------------------------------------------

    def _apply_control_tooltips(self) -> None:
        for object_name, tooltip in _SCOPE_CONTROL_TOOLTIPS.items():
            getattr(self.ui, object_name).setToolTip(tooltip)

    def _apply_parameter_specs(self) -> None:
        specs = self._scope.specs

        for parameter, spinbox, slider in (
            (ScopeParam.TRIGGER_LEVEL, self.ui.spinTriggerLevel, self.ui.sliderTriggerLevel),
            (ScopeParam.DAC_VALUE, self.ui.spinDacValue, self.ui.sliderDacValue),
        ):
            spec = range_spec(specs, parameter)
            bind_spinbox(spinbox, spec)
            bind_slider(slider, spec)

        # Timing fields are edited in nanoseconds.
        for parameter, spinbox, scale in (
            (ScopeParam.PRETRIGGER_SAMPLES, self.ui.spinPretrigger, self._SAMPLE_PERIOD_NS),
            (ScopeParam.FRAME_SAMPLES, self.ui.spinFrameSamples, self._SAMPLE_PERIOD_NS),
            (ScopeParam.FRAME_PERIOD_CYCLES, self.ui.spinFrameGap, self._VIEWER_POINT_PERIOD_NS),
        ):
            bind_spinbox(spinbox, range_spec(specs, parameter).scaled(scale), suffix=" ns")
        self.ui.labelPretrigger.setText("Pretrigger:")
        self.ui.labelFrameSamples.setText("Frame:")
        self.ui.labelFrameGap.setText("Periodic gap:")

        self.ui.comboTriggerMode.clear()
        self.ui.comboTriggerMode.setPlaceholderText("Select a waveform trigger")
        for mode in self._scope.supported_trigger_modes():
            if mode == TriggerMode.PERIODIC_SUM:
                continue  # Structured sums are not waveforms; Current owns them.
            label = {0: "Any above", 1: "Any below", 2: "Falling edge",
                     3: "Rising edge", 4: "Periodic"}[int(mode)]
            self.ui.comboTriggerMode.addItem(label, int(mode))
        edge_spec = specs[ScopeParam.EDGE_MODE]
        assert isinstance(edge_spec, ListSpec)
        if edge_spec.default in edge_spec.items:
            self.ui.comboTriggerMode.setCurrentIndex(
                list(edge_spec.items).index(edge_spec.default)
            )

        self.ui.cbDmaEnable.setChecked(False)

    # ------------------------------------------------------------------
    # Write defaults to hardware, then read back
    # ------------------------------------------------------------------

    def _send_defaults(self) -> None:
        specs = self._scope.specs
        if self._scope.dma_active():
            raise RuntimeError("stop the active Scope DMA session before applying defaults")
        if self._scope.get_enable():
            self._scope.stop()
        # vdpp-scope.c pretrigger_samples_store() rejects nonzero pretrigger
        # in mode 5. Leave a previous sum session before restoring waveform
        # geometry; trigger_mode_store() also requires sum acquisition stopped.
        self._scope.set_trigger_mode(TriggerMode(int(specs[ScopeParam.EDGE_MODE].default)))
        self._scope.set_trigger_level(int(specs[ScopeParam.TRIGGER_LEVEL].default))
        if self._scope.dac_available():
            self._scope.set_dac_value(int(specs[ScopeParam.DAC_VALUE].default))
        self._scope.set_pretrigger_samples(int(specs[ScopeParam.PRETRIGGER_SAMPLES].default))
        self._scope.set_frame_samples(int(specs[ScopeParam.FRAME_SAMPLES].default))
        self._scope.set_frame_period_cycles(
            int(specs[ScopeParam.FRAME_PERIOD_CYCLES].default)
        )

    def _load_hardware_state(self) -> None:
        self.ui.spinTriggerLevel.setValue(self._scope.get_trigger_level())
        self.ui.sliderTriggerLevel.setValue(self._scope.get_trigger_level())
        dac_available = self._scope.dac_available()
        self.ui.spinDacValue.setEnabled(dac_available)
        self.ui.sliderDacValue.setEnabled(dac_available)
        self.ui.btnAutoSetup.setEnabled(dac_available)
        if dac_available:
            self.ui.spinDacValue.setValue(self._scope.get_dac_value())
            self.ui.sliderDacValue.setValue(self._scope.get_dac_value())
        else:
            self.ui.spinDacValue.setToolTip("No AFE DAC output is available for this channel")
        self.ui.spinPretrigger.setValue(
            self._scope.get_pretrigger_samples() * self._SAMPLE_PERIOD_NS
        )
        self.ui.spinFrameSamples.setValue(
            self._scope.get_frame_samples() * self._SAMPLE_PERIOD_NS
        )
        self.ui.spinFrameGap.setValue(
            self._scope.get_frame_period_cycles() * self._VIEWER_POINT_PERIOD_NS
        )
        for slider, spinbox in (
            (self.ui.sliderPretrigger, self.ui.spinPretrigger),
            (self.ui.sliderFrameSamples, self.ui.spinFrameSamples),
            (self.ui.sliderFrameGap, self.ui.spinFrameGap),
        ):
            slider.setValue(spinbox.value())
        self.ui.comboTriggerMode.setCurrentIndex(self.ui.comboTriggerMode.findData(self._scope.get_trigger_mode().value))

    def reload_hardware_state(self) -> None:
        """Show the settings the hardware now holds, e.g. after Load Settings.

        Use this, not _load_hardware_state(), once the plot exists: it also
        fits the plot to the frame length that was loaded.
        """
        self._load_hardware_state()
        self._update_frame_gap_enabled()
        self._show_frame_geometry()

    def _show_frame_geometry(self) -> None:
        """Fit the plot and persistence image to the frame the widgets show.

        A frame length set in code never commits; only editingFinished or a
        slider release reaches _on_frame_samples_changed(). Code that changes
        the frame widgets must call this, or the plot keeps the old time range.
        """
        frame_samples = self._frame_samples_from_ui()
        if self._display_nx != frame_samples // 4:
            self._reset_persistence_buffer(frame_samples)
            self._persistence_img.setImage(
                self._persistence_buffer, autoLevels=False, levels=(0, 1)
            )
        self._update_axis_ranges()

    # ------------------------------------------------------------------
    # Display settings persistence
    # ------------------------------------------------------------------

    def _settings_key(self, name: str) -> str:
        return settings_keys.scope_channel(self._channel, name)

    def _restore_display_settings(self) -> None:
        s = QSettings()
        if (v := s.value(self._settings_key("display_mode"))) is not None:
            self.ui.comboDisplayMode.setCurrentIndex(int(v))
        if (v := s.value(self._settings_key("persistence"))) is not None:
            self.ui.dialPersistence.setValue(int(v))
        if (v := s.value(self._settings_key("refresh_rate"))) is not None:
            self.ui.spinRefreshRate.setValue(int(v))
        stored = {
            name: s.value(self._settings_key(f"spectrum_{name}")) for name in _SPECTRUM_SETTINGS
        }
        self._apply_spectrum_settings(
            {name: value for name, value in stored.items() if value is not None}
        )

    def save_display_settings(self) -> None:
        s = QSettings()
        s.setValue(self._settings_key("display_mode"), self.ui.comboDisplayMode.currentIndex())
        s.setValue(self._settings_key("persistence"), self.ui.dialPersistence.value())
        s.setValue(self._settings_key("refresh_rate"), self.ui.spinRefreshRate.value())
        for name, value in self._spectrum_settings().items():
            s.setValue(self._settings_key(f"spectrum_{name}"), value)

    def configuration_settings(self) -> dict[str, int | bool | str]:
        """Return settings owned by this view rather than by scope hardware."""
        return {
            "display_mode": self.ui.comboDisplayMode.currentIndex(),
            "persistence": self.ui.dialPersistence.value(),
            "refresh_rate_hz": self.ui.spinRefreshRate.value(),
            "measurement_time_s": self.ui.spinTime.value(),
            "series_runs": self._series.runs(),
            "dma_enabled": self.ui.cbDmaEnable.isChecked(),
            "histogram_mode": self._amplitude_histogram.mode.value,
            **{f"spectrum_{name}": value for name, value in self._spectrum_settings().items()},
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
        if "series_runs" in settings:
            self._series.set_runs(int(settings["series_runs"]))
        if "dma_enabled" in settings:
            self.ui.cbDmaEnable.setChecked(bool(settings["dma_enabled"]))
        if "histogram_mode" in settings:
            try:
                self.set_scope_distribution_mode(HistogramMode(settings["histogram_mode"]))
            except ValueError:
                log.warning("Ignoring unknown Scope histogram mode %r", settings["histogram_mode"])
        self._apply_spectrum_settings({
            name: settings[f"spectrum_{name}"]
            for name in _SPECTRUM_SETTINGS if f"spectrum_{name}" in settings
        })
        self.save_display_settings()

    def reset_zoom(self) -> None:
        """Reset the waveform plot to its default fixed range."""
        self._update_axis_ranges()
        if self._amplitude_histogram.mode == HistogramMode.INTEGRAL:
            self._reset_distribution_zoom()

    @property
    def session_replay_active(self) -> bool:
        return self._session_replay_active

    def begin_session_replay(self, *, frame_count: int, duration_ns: int) -> None:
        """Stop live acquisition and isolate this view for recorded playback."""
        if getattr(self, "_session_replay_active", False):
            self.end_session_replay()
        self.clear_scope_distribution()
        self.clear_scope_spectrum("session replay")
        self._session_replay_active = True
        self._stop_live_for_replay()
        self._session_replay_display_mode = self.ui.comboDisplayMode.currentIndex()
        blocked = self.ui.comboDisplayMode.blockSignals(True)
        try:
            self.ui.comboDisplayMode.setCurrentIndex(int(DisplayMode.RAW))
        finally:
            self.ui.comboDisplayMode.blockSignals(blocked)
        self._set_display_mode(DisplayMode.RAW)
        self._raw_curve.setData([], [])
        self._threshold_line.setVisible(False)
        self._pretrigger_line.setVisible(False)
        self._apply_session_replay_controls()
        duration_s = duration_ns / 1_000_000_000
        self.status.setText(
            f"Scope replay ready: {frame_count} recorded frame(s), {duration_s:.3f} s"
        )

    def _stop_live_for_replay(self) -> None:
        self._refresh_timer.stop()
        self._measurement_timer.stop()
        self._discard_inflight_frame = True
        self._acquiring = False
        online_current = self._dma_online_current
        try:
            self.stop_dma_sync()
            dma_running = self._dma_thread is not None and self._dma_thread.isRunning()
            auto_running = (
                self._auto_setup_thread is not None
                and self._auto_setup_thread.isRunning()
            )
            if dma_running or auto_running:
                raise RuntimeError(
                    "Scope background activity did not stop; replay was not started"
                )
        except Exception:
            self._session_replay_active = False
            self._restore_controls_after_session_replay()
            raise

        if online_current:
            self._dma_online_current = False
            self._set_dma_checkbox_without_write(False)
            self.current_dma_state_changed.emit(
                False,
                "Live Scope DMA stopped for session replay.",
            )
        self._dma_stopping = False

    def set_replay_hardware_interlock(
        self, enabled: bool, *, resume_background: bool = True,
    ) -> None:
        """Prevent every Scope hardware action while replay data are loaded."""
        enabled = bool(enabled)
        if enabled == self._replay_hardware_interlocked:
            return
        self._replay_hardware_interlocked = enabled
        if enabled:
            self._stop_live_for_replay()
            self._apply_hardware_interlock_controls()
            return
        if not self._session_replay_active:
            self._restore_controls_after_session_replay()

    def _apply_hardware_interlock_controls(self) -> None:
        self._set_controls_enabled(False)
        self.ui.btnStart.setChecked(False)
        self.ui.btnStart.setEnabled(False)
        self.ui.btnStop.setChecked(False)
        self.ui.btnStop.setEnabled(False)
        self.ui.btnAcquireFrame.setEnabled(False)
        self.ui.btnDmaFile.setEnabled(False)
        self.ui.btnAutoSetup.setEnabled(False)

    def _apply_session_replay_controls(self) -> None:
        self._apply_hardware_interlock_controls()
        self.ui.comboDisplayMode.setEnabled(False)
        self.ui.dialPersistence.setEnabled(False)
        self.ui.labelPersistenceValue.setEnabled(False)

    def display_session_replay_frame(
        self,
        samples: np.ndarray,
        *,
        sample_period_ns: int,
        frame_index: int,
        frame_count: int,
        elapsed_ns: int,
        duration_ns: int,
        source: str,
    ) -> None:
        """Render one recorded frame without publishing or touching hardware."""
        if not self._session_replay_active:
            return
        waveform = np.asarray(samples, dtype="<i2").reshape(-1)
        if len(waveform) == 0 or sample_period_ns <= 0:
            return
        frame_duration_ns = max(sample_period_ns, len(waveform) * sample_period_ns)
        self._time_scale = time_axis_scale(frame_duration_ns)
        x_time = (
            np.arange(len(waveform), dtype=np.float64)
            * sample_period_ns
            / self._time_scale.ns_per_unit
        )
        self._raw_curve.setData(x_time, waveform)
        self._time_axis_label.setText(
            f"Recorded time [{self._time_scale.unit}]  "
            f"({sample_period_ns} ns/sample; {source} stream)"
        )
        view = self._plot_item.getViewBox()
        view.setXRange(
            0,
            frame_duration_ns / self._time_scale.ns_per_unit,
            padding=0,
        )
        view.setYRange(self._Y_MIN, self._Y_MAX, padding=0)
        self.status.setText(
            "Replaying recorded Scope: "
            f"{elapsed_ns / 1_000_000_000:.3f} / "
            f"{duration_ns / 1_000_000_000:.3f} s "
            f"(frame {frame_index}/{frame_count})"
        )

    def finish_session_replay(self, *, frame_count: int, duration_ns: int) -> None:
        """Hold the last replay frame and keep live acquisition isolated at EOF."""
        if not self._session_replay_active:
            return
        self.status.setText(
            "Scope replay complete: "
            f"{frame_count} frame(s), {duration_ns / 1_000_000_000:.3f} s; "
            "final recorded frame held"
        )

    def end_session_replay(self) -> None:
        """Leave replay isolation without automatically restarting acquisition."""
        if not self._session_replay_active:
            return
        self._session_replay_active = False
        self.clear_scope_distribution()
        self.clear_scope_spectrum("session replay ended")
        self._restore_controls_after_session_replay()
        self.status.setText(
            "Session replay unloaded; Scope acquisition remains stopped"
        )

    def _restore_controls_after_session_replay(self) -> None:
        display_index = self._session_replay_display_mode
        self._session_replay_display_mode = None
        if display_index is not None:
            blocked = self.ui.comboDisplayMode.blockSignals(True)
            try:
                self.ui.comboDisplayMode.setCurrentIndex(display_index)
            finally:
                self.ui.comboDisplayMode.blockSignals(blocked)
        self.ui.comboDisplayMode.setEnabled(True)
        self._set_display_mode(DisplayMode(self.ui.comboDisplayMode.currentIndex()))
        self._threshold_line.setVisible(True)
        self._pretrigger_line.setVisible(True)
        if self._replay_hardware_interlocked:
            self._apply_hardware_interlock_controls()
            self._update_axis_ranges()
            return
        self._set_controls_enabled(True)
        self.ui.btnStart.setChecked(False)
        self.ui.btnStart.setEnabled(True)
        self.ui.btnStop.setChecked(False)
        self.ui.btnStop.setEnabled(False)
        self.ui.btnAcquireFrame.setEnabled(True)
        self.ui.btnDmaFile.setEnabled(True)
        self.ui.btnAutoSetup.setEnabled(self._scope.dac_available())
        self._update_axis_ranges()

    # ------------------------------------------------------------------
    # Enable/disable parameter controls during DMA
    # ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        if not enabled:
            self._cancel_pretrigger_line_drag()
        self.ui.groupTrigger.setEnabled(enabled)
        self.ui.groupTiming.setEnabled(enabled)
        self.ui.cbDmaEnable.setEnabled(enabled)
        self._threshold_line.setMovable(enabled)
        self._update_waveform_drag_enabled()

    def _set_dma_checkbox_without_write(self, checked: bool) -> None:
        """Mirror buffer-owned DMA state without writing its hardware gate."""
        blocked = self.ui.cbDmaEnable.blockSignals(True)
        try:
            self.ui.cbDmaEnable.setChecked(checked)
        finally:
            self.ui.cbDmaEnable.blockSignals(blocked)
        self._update_viewer_transport_hint()

    def reset_defaults(self) -> None:
        """Restore Scope hardware/UI defaults while the channel is idle."""
        if self._replay_hardware_interlocked:
            raise RuntimeError("exit session replay before resetting Scope defaults")
        if (
            self._sum_worker is not None
            or self._auto_setup_thread is not None
            or self._dma_worker is not None
            or self._dma_thread is not None
            or self._refresh_timer.isActive()
            or self._scope.get_enable()
        ):
            raise RuntimeError("stop Scope acquisition before resetting defaults")

        self._raw_curve.cancelDrag()
        self._cancel_pretrigger_line_drag()
        self._send_defaults()
        self._load_hardware_state()
        self._set_dma_checkbox_without_write(False)
        self._reset_persistence_buffer(self._frame_samples_from_ui())
        self._persistence_img.setImage(
            self._persistence_buffer,
            autoLevels=False,
            levels=(0, 1),
        )
        self._raw_curve.setData([], [])
        self._sync_pretrigger_line_from_widget()
        self._update_axis_ranges()
        self._update_frame_gap_enabled()
        self._update_viewer_transport_hint()
        self.status.setText("Scope defaults restored")

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
        self.ui.sliderFrameSamples.setToolTip(tip)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._wire_slider_spinbox(
            self.ui.sliderTriggerLevel,
            self.ui.spinTriggerLevel,
            lambda v: self._scope.set_trigger_level(v),
        )
        self.ui.spinTriggerLevel.valueChanged.connect(self._threshold_line.setValue)
        self._threshold_line.sigPositionChanged.connect(self._on_threshold_line_changed)
        self._threshold_line.sigPositionChangeFinished.connect(
            self._on_threshold_line_drag_finished
        )
        self._wire_slider_spinbox(
            self.ui.sliderDacValue,
            self.ui.spinDacValue,
            lambda v: self._scope.set_dac_value(v),
        )
        self._raw_curve.drag_started.connect(self._on_waveform_drag_started)
        self._raw_curve.drag_moved.connect(self._on_waveform_drag_moved)
        self._raw_curve.drag_finished.connect(self._on_waveform_drag_finished)

        self._wire_slider_spinbox(
            self.ui.sliderPretrigger,
            self.ui.spinPretrigger,
            lambda _value: self._on_pretrigger_committed(),
        )
        self.ui.spinPretrigger.valueChanged.connect(
            self._sync_pretrigger_line_from_widget
        )
        self._pretrigger_line.sigPositionChanged.connect(
            self._on_pretrigger_line_changed
        )
        self._pretrigger_line.sigPositionChangeFinished.connect(
            self._on_pretrigger_line_drag_finished
        )
        self._wire_slider_spinbox(
            self.ui.sliderFrameSamples,
            self.ui.spinFrameSamples,
            lambda _value: self._on_frame_samples_changed(),
        )
        self._wire_slider_spinbox(
            self.ui.sliderFrameGap,
            self.ui.spinFrameGap,
            lambda _value: self._on_frame_gap_committed(),
        )
        self.ui.comboTriggerMode.currentIndexChanged.connect(
            self._on_trigger_mode_changed
        )
        self.ui.cbDmaEnable.toggled.connect(self._on_dma_toggled)
        self.ui.btnStart.clicked.connect(self._on_start_clicked)
        self.ui.btnStop.clicked.connect(self._on_stop)
        self.ui.btnAutoSetup.clicked.connect(self._on_auto_setup)
        self.ui.btnAcquireFrame.clicked.connect(self._on_acquire_frame)
        self.ui.btnDmaFile.clicked.connect(self._on_dma_file)
        self.ui.comboDisplayMode.currentIndexChanged.connect(self._on_display_mode_changed)
        self.ui.dialPersistence.valueChanged.connect(self._on_persistence_changed)
        self.ui.spinRefreshRate.valueChanged.connect(self._on_refresh_rate_changed)

    def _on_threshold_line_changed(self) -> None:
        """Keep the plot marker and integer-valued threshold widgets in sync."""
        value = round(self._threshold_line.value())
        value = max(self.ui.spinTriggerLevel.minimum(), value)
        value = min(self.ui.spinTriggerLevel.maximum(), value)
        if self._threshold_line.value() != value:
            self._threshold_line.blockSignals(True)
            try:
                self._threshold_line.setValue(value)
            finally:
                self._threshold_line.blockSignals(False)
        self.ui.spinTriggerLevel.setValue(value)

    def _on_threshold_line_drag_finished(self) -> None:
        if self._replay_hardware_interlocked:
            return
        if self._threshold_line.movable:
            self._scope.set_trigger_level(self.ui.spinTriggerLevel.value())

    def _sync_pretrigger_line_from_widget(self) -> None:
        scale = self._time_scale.ns_per_unit
        line = self._pretrigger_line
        line.blockSignals(True)
        try:
            line.setBounds(
                (
                    self.ui.spinPretrigger.minimum() / scale,
                    self.ui.spinPretrigger.maximum() / scale,
                )
            )
            line.setValue(self.ui.spinPretrigger.value() / scale)
        finally:
            line.blockSignals(False)
        line.label.setFormat(f"Pretrigger {self.ui.spinPretrigger.value()} ns")

    def _on_pretrigger_line_changed(self) -> None:
        line = self._pretrigger_line
        spin = self.ui.spinPretrigger
        if line.moving and self._pretrigger_line_drag is None:
            x, y = self._raw_curve.getData()
            self._pretrigger_line_drag = _PretriggerLineDragState(
                x=np.array(x, copy=True) if x is not None else None,
                y=np.array(y, copy=True) if y is not None else None,
                pretrigger_ns=spin.value(),
                status_text=self.status.text(),
            )
            self._update_waveform_drag_enabled()
        target_ns = line.value() * self._time_scale.ns_per_unit
        normalize_numeric_control(spin, target_ns)
        self._sync_pretrigger_line_from_widget()
        state = self._pretrigger_line_drag
        if state is not None:
            if state.x is not None and state.y is not None:
                shift = (spin.value() - state.pretrigger_ns) / self._time_scale.ns_per_unit
                self._raw_curve.setData(state.x + shift, state.y)
            self.status.setText(
                f"Pretrigger preview: {spin.value()} ns"
            )

    def _cancel_pretrigger_line_drag(self) -> None:
        state = self._pretrigger_line_drag
        if state is None:
            return
        self._pretrigger_line_drag = None
        self._restore_pretrigger_line_drag(state)

    def _on_pretrigger_line_drag_finished(self) -> None:
        if self._replay_hardware_interlocked:
            return
        state = self._pretrigger_line_drag
        if state is None:
            return
        self._pretrigger_line_drag = None
        value_ns = self.ui.spinPretrigger.value()
        if not self._pretrigger_line.movable or value_ns == state.pretrigger_ns:
            self._restore_pretrigger_line_drag(state)
            return
        try:
            # vdpp-scope.c accepts live pretrigger writes. Commit once on
            # release, matching the raw-waveform drag path.
            self._scope.set_pretrigger_samples(value_ns // self._SAMPLE_PERIOD_NS)
        except Exception as exc:
            log.exception("Scope ch%d: pretrigger marker update failed", self._channel)
            try:
                self.ui.spinPretrigger.setValue(
                    self._scope.get_pretrigger_samples() * self._SAMPLE_PERIOD_NS
                )
                detail = ""
            except Exception:
                log.exception("Scope ch%d: pretrigger marker readback failed", self._channel)
                detail = "; hardware state unknown—reconnect"
            self._sync_pretrigger_line_from_widget()
            if state.x is not None and state.y is not None:
                self._raw_curve.setData(state.x, state.y)
            self.status.setText(f"Pretrigger update failed: {exc}{detail}")
            self._update_waveform_drag_enabled()
            return

        self._waveform_pending_status = state.status_text
        self._update_waveform_drag_enabled()
        if self._refresh_timer.isActive():
            self.status.setText(
                "Pretrigger applied; showing preview until the next live frame"
            )
            self._request_frame()
        else:
            if state.x is not None and state.y is not None:
                self._raw_curve.setData(state.x, state.y)
            self.status.setText(
                "Pretrigger applied; acquire a frame to see the actual waveform"
            )

    def _restore_pretrigger_line_drag(self, state: _PretriggerLineDragState) -> None:
        self.ui.spinPretrigger.setValue(state.pretrigger_ns)
        self._sync_pretrigger_line_from_widget()
        if state.x is not None and state.y is not None:
            self._raw_curve.setData(state.x, state.y)
        self.status.setText(state.status_text)
        self._update_waveform_drag_enabled()

    def _update_waveform_drag_enabled(self) -> None:
        marker_enabled = (
            self.ui.groupTiming.isEnabled()
            and self._dma_worker is None
            and self._auto_setup_thread is None
            and self._waveform_pending_status is None
            and self._waveform_drag is None
        )
        self._pretrigger_line.setMovable(marker_enabled)
        enabled = (
            self._display_mode == DisplayMode.RAW
            and self._scope.dac_available()
            and self.ui.groupTrigger.isEnabled()
            and self.ui.groupTiming.isEnabled()
            and self._dma_worker is None
            and self._auto_setup_thread is None
            and self._waveform_pending_status is None
            and self._pretrigger_line_drag is None
        )
        self._raw_curve.setDragEnabled(enabled)
        self._raw_curve.setToolTip(
            "Drag the live waveform: horizontal = pretrigger, vertical = DAC baseline"
            if enabled else ""
        )

    def _on_waveform_drag_started(self) -> None:
        x, y = self._raw_curve.getData()
        if x is None or y is None or len(x) == 0:
            self._raw_curve.cancelDrag()
            return
        dac_spec = self._scope.specs[ScopeParam.DAC_VALUE]
        adc_spec = self._scope.specs[ScopeParam.TRIGGER_LEVEL]
        assert isinstance(dac_spec, RangeSpec)
        assert isinstance(adc_spec, RangeSpec)
        # The AFE DAC driver does not define an analogue ADC-counts/code scale.
        # Operator observation on the current board (2026-09-16): increasing
        # the DAC moves the signal down. Auto Setup measures the real slope;
        # until then only the negative sign is observed, and the full ADC span
        # per DAC span is an approximate UI preview sensitivity.
        fallback_slope = -(adc_spec.max_val - adc_spec.min_val) / (
            dac_spec.max_val - dac_spec.min_val
        )
        self._waveform_drag = _WaveformDragState(
            x=np.array(x, copy=True),
            y=np.array(y, copy=True),
            dac_value=self.ui.spinDacValue.value(),
            pretrigger_ns=self.ui.spinPretrigger.value(),
            adc_per_dac=self._dac_adc_slope or fallback_slope,
            status_text=self.status.text(),
        )
        self._pretrigger_line.setMovable(False)

    def _on_waveform_drag_moved(self, dx: float, dy: float) -> None:
        state = self._waveform_drag
        if state is None:
            return
        pre_spin = self.ui.spinPretrigger
        target_ns = state.pretrigger_ns + dx * self._time_scale.ns_per_unit
        normalize_numeric_control(pre_spin, target_ns)
        pre_ns = pre_spin.value()
        dac_spin = self.ui.spinDacValue
        dac_value = state.dac_value + round(dy / state.adc_per_dac)
        dac_value = max(dac_spin.minimum(), min(dac_spin.maximum(), dac_value))
        dac_spin.setValue(dac_value)
        x_shift = (pre_ns - state.pretrigger_ns) / self._time_scale.ns_per_unit
        y_shift = (dac_value - state.dac_value) * state.adc_per_dac
        self._raw_curve.setData(state.x + x_shift, state.y + y_shift)
        calibration = "" if self._dac_adc_slope is not None else " (DAC preview approximate)"
        self.status.setText(
            f"Drag preview: DAC {dac_value}, pretrigger {pre_ns} ns{calibration}"
        )

    def _on_waveform_drag_finished(self, cancelled: bool) -> None:
        if self._replay_hardware_interlocked:
            self._raw_curve.cancelDrag()
            return
        state = self._waveform_drag
        if state is None:
            return
        self._waveform_drag = None
        dac_value = self.ui.spinDacValue.value()
        pretrigger_ns = self.ui.spinPretrigger.value()
        if cancelled or (
            dac_value == state.dac_value and pretrigger_ns == state.pretrigger_ns
        ):
            self.ui.spinDacValue.setValue(state.dac_value)
            self.ui.spinPretrigger.setValue(state.pretrigger_ns)
            self._raw_curve.setData(state.x, state.y)
            self.status.setText(state.status_text)
            self._update_waveform_drag_enabled()
            return

        try:
            # vdpp-scope.c permits live pretrigger writes, and the AFE DAC is
            # a separate IIO device. Commit once on release, never per pixel.
            if pretrigger_ns != state.pretrigger_ns:
                self._scope.set_pretrigger_samples(
                    pretrigger_ns // self._SAMPLE_PERIOD_NS
                )
            if dac_value != state.dac_value:
                self._scope.set_dac_value(dac_value)
        except Exception as exc:
            log.exception("Scope ch%d: waveform drag update failed", self._channel)
            readback_failed = False
            try:
                self.ui.spinPretrigger.setValue(
                    self._scope.get_pretrigger_samples() * self._SAMPLE_PERIOD_NS
                )
                self.ui.spinDacValue.setValue(self._scope.get_dac_value())
            except Exception:
                readback_failed = True
                log.exception("Scope ch%d: waveform drag readback failed", self._channel)
            self._raw_curve.setData(state.x, state.y)
            detail = "; hardware state unknown—reconnect" if readback_failed else ""
            self.status.setText(f"Waveform drag failed: {exc}{detail}")
            self._update_waveform_drag_enabled()
            return

        self._waveform_pending_status = state.status_text
        self._update_waveform_drag_enabled()
        if self._refresh_timer.isActive():
            self.status.setText(
                "DAC/pretrigger applied; showing preview until the next live frame"
            )
            self._request_frame()
        else:
            self._raw_curve.setData(state.x, state.y)
            self.status.setText(
                "DAC/pretrigger applied; acquire a frame to see the actual waveform"
            )

    def _on_trigger_mode_changed(self, index: int) -> None:
        if self._replay_hardware_interlocked:
            return
        value = self.ui.comboTriggerMode.itemData(index)
        if value is None:
            return
        mode = TriggerMode(value)
        self._scope.set_trigger_mode(mode)
        self._configure_sum_controls()
        self._update_frame_gap_enabled()

    def _configure_sum_controls(self) -> None:
        is_sum = self._scope.get_trigger_mode() == TriggerMode.PERIODIC_SUM
        for control in (self.ui.spinPretrigger, self.ui.sliderPretrigger,
                        self.ui.btnAcquireFrame):
            control.setEnabled(not is_sum and not self._replay_hardware_interlocked)
        self.ui.btnAutoSetup.setEnabled(
            not is_sum and not self._replay_hardware_interlocked and self._scope.dac_available()
        )
        self.ui.cbDmaEnable.setEnabled(not is_sum and not self._replay_hardware_interlocked)
        if is_sum:
            self._raw_curve.clear()
            self.status.setText(
                "Periodic sums belong to Current. Select a waveform trigger "
                "to use Scope."
            )

    def current_sum_supported(self) -> bool:
        """Whether Current can acquire structured periodic sums on this channel."""
        return (
            self._scope_dma is not None
            and TriggerMode.PERIODIC_SUM in self._scope.supported_trigger_modes()
        )

    def start_current_sum_monitor(
        self, *, gap_cycles: int, frame_samples: int | None = None, output: Path | None = None,
        overwrite: bool = False,
    ) -> ScopeSumWorker:
        """Reserve/configure Scope DMA; Current renders its bounded analysis."""
        if self._replay_hardware_interlocked or self._session_replay_active:
            raise RuntimeError("exit session replay before starting periodic sums")
        if not self.current_sum_supported():
            raise RuntimeError("periodic sums require supported IP123/IP124 Scope DMA")
        if (
            self._sum_worker is not None
            or self._auto_setup_thread is not None
            or self._dma_worker is not None
            or self._dma_thread is not None
            or self._acquiring
            or self.measurement_active
            or self._refresh_timer.isActive()
            or self._scope.get_enable()
            or self._scope.dma_active()
        ):
            raise RuntimeError("stop the active Scope acquisition before starting periodic sums")
        if (series := owner_series(self)) is not None:
            series.cancel()  # Current now owns the Scope transport.
        mode = self._scope.get_trigger_mode()
        if mode == TriggerMode.PERIODIC_SUM:
            raise RuntimeError("select a waveform trigger in Scope before starting periodic sums")
        previous_frame = self._scope.get_frame_samples()
        selected_frame = previous_frame if frame_samples is None else frame_samples
        self._scope.specs[ScopeParam.FRAME_SAMPLES].validate(selected_frame, "frame_samples")
        self._scope.specs[ScopeParam.FRAME_PERIOD_CYCLES].validate(
            gap_cycles, "frame_period_cycles"
        )
        self._sum_previous_settings = (
            mode, self._scope.get_pretrigger_samples(), self._scope.get_frame_period_cycles(),
            previous_frame,
        )
        assert self._scope_dma is not None
        worker = None
        try:
            self._scope.set_pretrigger_samples(0)
            self._scope.set_frame_samples(selected_frame)
            self._scope.set_trigger_mode(TriggerMode.PERIODIC_SUM)
            self._scope.set_frame_period_cycles(gap_cycles)
            from nlab.utils.settings_io import configuration_yaml

            # Snapshot the effective sum settings after configuration, before the
            # DMA worker/enable. Never collect full settings in the reader loop.
            snapshot = (
                configuration_yaml(self._measurement_configuration())
                if output is not None and self._measurement_configuration is not None else ""
            )
            worker = ScopeSumWorker(
                capture=self._scope_dma.create_sum_capture(configuration_yaml=snapshot)
            )
            self._sum_worker = worker
            self._set_sum_controls_busy(True)
            worker.start(output, overwrite=overwrite)
            self._set_measurement_active(True)
            return worker
        except Exception:
            if worker is not None:
                worker.request_stop()
                worker.join()
            self._sum_worker = None
            self._restore_current_sum_settings()
            raise

    def _set_sum_controls_busy(self, busy: bool) -> None:
        enabled = not busy and not self._replay_hardware_interlocked
        self._set_controls_enabled(enabled)
        for control in (
            self.ui.btnStart, self.ui.btnAcquireFrame, self.ui.btnAutoSetup, self.ui.btnDmaFile
        ):
            control.setEnabled(enabled)
        self.ui.btnStop.setEnabled(False)
        self._configure_sum_controls()

    def finish_current_sum_monitor(self) -> None:
        """GUI-thread completion, after the reader has drained and released DMA."""
        worker = self._sum_worker
        if worker is None or not worker.done:
            return
        worker.join()
        self._sum_worker = None
        try:
            self._restore_current_sum_settings()
        except Exception as exc:
            worker.error = (worker.error + "; " if worker.error else "") + f"Restore: {exc}"
        finally:
            self._set_measurement_active(False)
        self.status.setText(worker.message)

    def _restore_current_sum_settings(self) -> None:
        previous = self._sum_previous_settings
        if previous is None:
            return
        self._sum_previous_settings = None
        # Release the reader and restore before Digitizer.close().
        mode, pretrigger, gap, frame = previous
        self._scope.set_trigger_mode(mode)
        self._scope.set_frame_samples(frame)
        self._scope.set_pretrigger_samples(pretrigger)
        self._scope.set_frame_period_cycles(gap)
        self._set_sum_controls_busy(False)
        self.status.setText("Current periodic sums stopped; Scope waveform settings restored.")

    def stop_current_sum_monitor(self, *, wait: bool = False) -> None:
        """Stop/drain Current sums; shutdown joins before restoring settings."""
        worker = getattr(self, "_sum_worker", None)
        if worker is not None:
            worker.request_stop()
            if wait:
                worker.join()
                self.finish_current_sum_monitor()

    def _on_pretrigger_committed(self) -> None:
        if self._replay_hardware_interlocked:
            return
        self._scope.set_pretrigger_samples(
            self.ui.spinPretrigger.value() // self._SAMPLE_PERIOD_NS
        )

    def _on_frame_gap_committed(self) -> None:
        if self._replay_hardware_interlocked:
            return
        self._scope.set_frame_period_cycles(
            self.ui.spinFrameGap.value() // self._VIEWER_POINT_PERIOD_NS
        )

    def _update_frame_gap_enabled(self) -> None:
        periodic = self.ui.comboTriggerMode.currentData() == int(TriggerMode.TIMED)
        enabled = (
            periodic
            and self._scope.frame_period_cycles_supported()
            and not self._replay_hardware_interlocked
        )
        for control in (self.ui.labelFrameGap, self.ui.spinFrameGap, self.ui.sliderFrameGap):
            control.show()
            control.setEnabled(enabled)

    def _wire_slider_spinbox(
        self,
        slider: QSlider,
        spinbox: QSpinBox,
        set_fn: Callable[[int], None],
    ) -> None:
        slider.setRange(spinbox.minimum(), spinbox.maximum())
        slider.setSingleStep(spinbox.singleStep())
        slider.setPageStep(spinbox.singleStep() * 10)
        slider.setValue(spinbox.value())
        slider.setToolTip(spinbox.toolTip())
        slider.valueChanged.connect(spinbox.setValue)

        def on_spin_changed(v: int) -> None:
            blocked = slider.blockSignals(True)
            try:
                slider.setValue(v)
            finally:
                slider.blockSignals(blocked)

        def commit() -> None:
            normalize_numeric_control(spinbox)
            if not self._replay_hardware_interlocked:
                set_fn(spinbox.value())

        spinbox.valueChanged.connect(on_spin_changed)
        spinbox.editingFinished.connect(commit)
        slider.sliderReleased.connect(commit)

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def current_dma_supported(self) -> bool:
        """Whether this Scope can feed the Current view without a file."""
        return (
            self._scope_dma is not None
            and self._current_accumulator is not None
            and self._scope.frame_period_cycles_supported()
        )

    def start_current_dma_monitor(self, *, gap_cycles: int | None = None) -> None:
        """Configure and start online periodic DMA for current monitoring.

        The Scope controller remains the sole DMA owner. The Current view is
        only a subscriber to decoded complete frames, exactly as PSD is a
        subscriber to MCA list-mode batches. IP122 uses the live-qualified
        maximum-coverage frame/gap preset; the selected values remain visible
        in the Scope controls. The displayed trigger level and DAC baseline are
        applied unchanged so Current observes the same analogue operating
        point as Scope. Other firmware retains the operator-selected Scope
        geometry because that preset has not been qualified there.
        """
        if self._replay_hardware_interlocked:
            raise RuntimeError("exit session replay before starting Scope DMA")
        if not self.current_dma_supported():
            raise RuntimeError(
                "Scope DMA current monitoring requires the direct IIO backend "
                "with periodic-frame support"
            )
        if self._sum_worker is not None:
            raise RuntimeError("Current periodic sums already own the Scope transport")
        if self._auto_setup_thread is not None:
            raise RuntimeError("Scope Auto Setup is active")
        if self._dma_worker is not None or self._dma_thread is not None:
            raise RuntimeError("Scope DMA is already active on this channel")
        if (series := owner_series(self)) is not None:
            series.cancel()  # Current now owns the Scope transport.

        if self.ui.btnStart.isChecked() or self._refresh_timer.isActive():
            self._on_stop()
        elif self._scope.get_enable():
            self._scope.stop()

        ip_version = self._scope.get_ip_version()
        if ip_version == _CURRENT_DMA_OPTIMIZED_IP_VERSION:
            # Live-qualified on .128 with IP122: the maximum frame amortizes
            # per-frame transport overhead, while a 100 us gap produced the
            # best timestamp-derived coverage. Missing trigger opportunities
            # remain explicit in the FPGA timestamps.
            frame_samples = _CURRENT_DMA_FRAME_SAMPLES
            selected_gap_cycles = (
                _CURRENT_DMA_GAP_CYCLES if gap_cycles is None else gap_cycles
            )
        else:
            frame_samples = self._frame_samples_from_ui()
            selected_gap_cycles = (
                self.ui.spinFrameGap.value() // self._VIEWER_POINT_PERIOD_NS
                if gap_cycles is None
                else gap_cycles
            )
        frame_spec = self._scope.specs[ScopeParam.FRAME_SAMPLES]
        assert isinstance(frame_spec, RangeSpec)
        if not int(frame_spec.min_val) <= frame_samples <= int(frame_spec.max_val):
            raise ValueError("Scope frame setting is outside the supported range")
        gap_spec = self._scope.specs[ScopeParam.FRAME_PERIOD_CYCLES]
        assert isinstance(gap_spec, RangeSpec)
        if not int(gap_spec.min_val) <= selected_gap_cycles <= int(gap_spec.max_val):
            raise ValueError("Scope DMA gap is outside the supported range")
        gap_cycles = selected_gap_cycles
        trigger_level = self.ui.spinTriggerLevel.value()
        dac_value = self.ui.spinDacValue.value()

        # Write the hardware explicitly. Updating the widgets with blocked
        # signals below is presentation only and must not duplicate writes.
        # Periodic triggering does not evaluate trigger_level, but preserving
        # it here keeps the Scope operating point exact when the operator later
        # returns to a threshold/edge mode. The DAC baseline remains active for
        # every mode and therefore directly affects the Current waveform mean.
        self._scope.set_trigger_level(trigger_level)
        if self._scope.dac_available():
            self._scope.set_dac_value(dac_value)
        self._scope.set_pretrigger_samples(0)
        self._scope.set_frame_samples(frame_samples)
        self._scope.set_trigger_mode(TriggerMode.TIMED)
        self._scope.set_frame_period_cycles(gap_cycles)

        def set_without_signal(
            widget: QWidget,
            setter: Callable[[int], None],
            value: int,
        ) -> None:
            blocked = widget.blockSignals(True)
            try:
                setter(value)
            finally:
                widget.blockSignals(blocked)

        for spinbox, slider, value in (
            (self.ui.spinPretrigger, self.ui.sliderPretrigger, 0),
            (
                self.ui.spinFrameSamples,
                self.ui.sliderFrameSamples,
                frame_samples * self._SAMPLE_PERIOD_NS,
            ),
            (
                self.ui.spinFrameGap,
                self.ui.sliderFrameGap,
                gap_cycles * self._VIEWER_POINT_PERIOD_NS,
            ),
        ):
            set_without_signal(spinbox, spinbox.setValue, value)
            set_without_signal(slider, slider.setValue, value)
        set_without_signal(
            self.ui.comboTriggerMode,
            self.ui.comboTriggerMode.setCurrentIndex,
            self.ui.comboTriggerMode.findData(int(TriggerMode.TIMED)),
        )
        self._set_dma_checkbox_without_write(True)
        self._update_frame_gap_enabled()
        self._show_frame_geometry()  # also refreshes the viewer transport hint

        assert self._scope_dma is not None
        assert self._current_accumulator is not None
        geometry = self._scope_dma.capture_geometry()
        if geometry.waveform_samples <= 0:
            raise RuntimeError("Scope DMA geometry contains no waveform samples")
        self._current_accumulator.start_session(
            geometry,
            ScopeCurrentRuntime(
                channel=self._channel,
                ip_version=ip_version,
                gap_cycles=gap_cycles,
                expected_interval_ticks=gap_cycles + frame_samples // 4,
                viewer_state="dma-preview" if self.isVisible() else "disabled",
            ),
        )
        self._dma_online_current = True
        self._dma_last_error = None
        try:
            self._on_start()
        except Exception:
            self._dma_online_current = False
            self._current_accumulator.finish_session()
            raise
        if self._dma_worker is None:
            self._dma_online_current = False
            self._current_accumulator.finish_session()
            raise RuntimeError("Scope DMA did not start")

    def stop_current_dma_monitor(self, *, wait: bool = False) -> None:
        """Stop a DMA session that was started on behalf of Current."""
        if not self._dma_online_current:
            return
        self._on_stop()
        if wait:
            self.stop_dma_sync()
            if self._dma_online_current:
                self._dma_online_current = False
                self._set_dma_checkbox_without_write(False)
                self.current_dma_state_changed.emit(False, "Scope DMA stopped.")

    def _on_dma_toggled(self, _enabled: bool) -> None:
        self._update_viewer_transport_hint()

    # ------------------------------------------------------------------
    # Measurement series
    # ------------------------------------------------------------------

    def set_measurement_series_enabled(self, enabled: bool) -> None:
        """Apply Settings > Enable Series to this view."""
        self._series.set_enabled(enabled)

    def _on_start_clicked(self) -> None:
        """Operator Start: begins a series when Settings > Enable Series is on."""
        series = self._series
        if not series.begin(self.ui.spinTime.value()):
            self.ui.btnStart.setChecked(False)
            self.status.setText(time_limit_required_message(series.runs()))
            return
        try:
            self._on_start()
        finally:
            if series.active and self._series_idle():
                series.cancel()  # Start was declined and already reported why.

    def _series_idle(self) -> bool:
        """No run, reader thread, sum session or Auto Setup holds this Scope."""
        return (
            not self.measurement_active
            and self._dma_worker is None
            and self._dma_thread is None
            and self._sum_worker is None
            and self._auto_setup_thread is None
            and not self._refresh_timer.isActive()
        )

    def _start_series_run(self) -> None:
        if self.ui.btnStart.isEnabled():
            self._on_start()

    def _next_dma_filepath(self) -> Path:
        """The chosen or generated recording, numbered while a series runs."""

        def chosen_or_generated() -> Path:
            chosen = self._dma_filepath
            if chosen is None:
                return self._generate_filepath()
            return chosen_recording_path(chosen)

        series = owner_series(self)
        numbered = None if series is None else series.recording_path(chosen_or_generated)
        if numbered is None:
            return chosen_or_generated()
        filepath = available_path(numbered)
        log.info("Scope DMA: series filepath: %s", filepath)
        return filepath

    def _on_start(self) -> None:
        if getattr(self, "_replay_hardware_interlocked", False) or getattr(
            self,
            "_session_replay_active",
            False,
        ):
            self.status.setText(
                "Exit session replay before starting live Scope acquisition"
            )
            return
        if (
            self._sum_worker is not None
            or self._scope.get_trigger_mode() == TriggerMode.PERIODIC_SUM
        ):
            self.status.setText(
                "Stop Current periodic sums and select a waveform trigger to use Scope."
            )
            return
        self._raw_curve.cancelDrag()
        self._cancel_pretrigger_line_drag()
        if self._waveform_pending_status is not None:
            self._raw_curve.setData([], [])
            self._waveform_pending_status = None
            self._update_waveform_drag_enabled()
        if self._auto_setup_thread is not None:
            return
        if self._dma_worker is not None or self._dma_thread is not None:
            log.warning(
                "Scope ch%d: refusing to start while the previous DMA worker is still closing",
                self._channel,
            )
            return
        self._discard_inflight_frame = False
        self._dma_stopping = False
        self._dma_last_error = None
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
        ScopeController._set_measurement_active(self, True)
        self.ui.btnStop.setEnabled(True)
        interval_ms = 1000 // self.ui.spinRefreshRate.value()
        self._refresh_timer.start(interval_ms)
        if self._frame_exceeds_viewer_limit():
            self.status.setText(self._viewer_limit_message())
        self._start_measurement_timer()
        log.info("Scope ch%d: acquisition started (refresh %d ms)", self._channel, interval_ms)

    def _start_with_dma(self) -> None:
        filepath = (
            None
            if self._dma_online_current
            else ScopeController._next_dma_filepath(self)
        )
        self._dma_filepath = None

        assert self._scope_dma is not None
        log.debug(
            "Scope ch%d IIO DMA [1/4]: creating worker, destination=%s",
            self._channel,
            filepath or "live Current view",
        )
        self._dma_worker = IIOScopeDmaWorker(
            streamer=self._scope_dma,
            filepath=filepath,
            # The publisher and current accumulator are independent consumers.
            # Keep both attached so passive recording also sees Current-only DMA.
            frame_buffer=self._dma_frame_buffer,
            current_accumulator=(
                self._current_accumulator if self._dma_online_current else None
            ),
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
        self._dma_thread.finished.connect(self._dma_worker.deleteLater)
        self._dma_thread.finished.connect(
            self._on_dma_finished,
            Qt.ConnectionType.QueuedConnection,
        )

        self._set_controls_enabled(False)
        self.status.setText("Arming IIO DMA...")
        log.debug(
            "Scope ch%d IIO DMA [2/4]: starting worker thread", self._channel
        )
        self._dma_thread.start()
        log.info(
            "Scope DMA: worker started, waiting for ready, destination=%s",
            filepath or "live Current view",
        )

    @Slot()
    def _on_dma_ready(self) -> None:
        if self._dma_stopping:
            return
        ScopeController._set_measurement_active(self, True)
        # The backend's first read starts the blocking refill and then writes
        # enable=1 in that order. The queued ready signal is only a UI
        # transition; a second start write here would race it.
        log.debug(
            "Scope ch%d IIO DMA [3/4]: backend reader owns ordered start",
            self._channel,
        )
        interval_ms = 1000 // self.ui.spinRefreshRate.value()
        self._refresh_timer.start(interval_ms)
        self.ui.btnStop.setEnabled(True)
        if self._dma_online_current:
            self.status.setText(
                "Streaming periodic DMA frames to Current; preview uses the latest DMA frame."
            )
            self.current_dma_state_changed.emit(True, "Scope DMA is streaming.")
        else:
            self.status.setText(
                "Recording full frames; live preview is truncated."
                if self._frame_exceeds_viewer_limit()
                else "Recording..."
            )
        self._start_measurement_timer()
        log.info("Scope ch%d: IIO DMA + acquisition started", self._channel)

    def _on_stop(self) -> None:
        if getattr(self, "_sum_worker", None) is not None:
            self.stop_current_sum_monitor()
            self.status.setText("Stopping Current periodic sums; draining accepted windows...")
            return
        series = owner_series(self)
        if series is not None:
            series.stop_requested()
        self._raw_curve.cancelDrag()
        self._cancel_pretrigger_line_drag()
        if self._waveform_pending_status is not None:
            self._raw_curve.setData([], [])
            self._waveform_pending_status = None
            self._update_waveform_drag_enabled()
        self._refresh_timer.stop()
        self._measurement_timer.stop()
        self._discard_inflight_frame = True
        self.ui.btnStop.setChecked(True)

        if self._dma_worker is not None:
            self._dma_stopping = True
            log.debug("Scope ch%d: stopping with DMA", self._channel)
            # Current vdpp_scope explicitly accepts enable=0 while a buffer
            # is armed. Stop new triggers first, then have the worker cancel
            # any blocked refill and close the buffer.
            self._scope.stop()
            ScopeController._set_measurement_active(self, False)
            self._dma_worker.stop()
            self._acquiring = False
            self.ui.btnStart.setChecked(False)
            self.ui.btnStart.setEnabled(False)
            self.ui.btnStop.setEnabled(False)
            self.ui.btnAcquireFrame.setEnabled(False)
            self.ui.cbDmaEnable.setEnabled(False)
            self.ui.btnDmaFile.setEnabled(False)
            self.status.setText("Stopping...")
            log.info("Scope ch%d: DMA stop requested; waiting for teardown", self._channel)
            return
        else:
            self._scope.stop()

        ScopeController._set_measurement_active(self, False)
        self._acquiring = False
        self.ui.btnStart.setChecked(False)
        self.ui.btnStart.setEnabled(True)
        self.ui.btnStop.setEnabled(False)
        self.ui.btnAcquireFrame.setEnabled(True)
        self.ui.cbDmaEnable.setEnabled(True)
        self.ui.btnDmaFile.setEnabled(True)
        self.ui.btnAutoSetup.setEnabled(self._scope.dac_available())
        log.info("Scope ch%d: acquisition stopped", self._channel)
        if series is not None:
            series.run_ended()

    # ------------------------------------------------------------------
    # Auto Setup
    # ------------------------------------------------------------------

    def _on_auto_setup(self) -> None:
        if not self._scope.dac_available():
            self.status.setText("Auto Setup requires an available AFE DAC output")
            return
        if getattr(self, "_replay_hardware_interlocked", False):
            self.status.setText(
                "Exit session replay before running Scope Auto Setup"
            )
            return
        if self._auto_setup_worker is not None:
            self._auto_setup_worker.stop()
            self.ui.btnAutoSetup.setEnabled(False)
            self.status.setText("Cancelling Auto Setup...")
            return
        if self._dma_worker is not None or self._dma_thread is not None:
            self.status.setText("Stop DMA recording before Auto Setup")
            return
        if (series := owner_series(self)) is not None and series.active:
            self.status.setText("Stop the measurement series before Auto Setup")
            return
        if self._refresh_timer.isActive() or self._scope.get_enable():
            self.status.setText("Stop acquisition before Auto Setup")
            return
        if self.ui.cbDmaEnable.isChecked():
            self.status.setText("Disable Record DMA frames before Auto Setup")
            return

        self._auto_setup_result = None
        self._auto_setup_error = None
        worker = ScopeAutoSetupWorker(self._scope.create_isolated_client)
        thread = QThread(self)
        self._auto_setup_worker = worker
        self._auto_setup_thread = thread
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self.status.setText)
        worker.succeeded.connect(self._on_auto_setup_succeeded)
        worker.error.connect(self._on_auto_setup_error)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            self._on_auto_setup_finished,
            Qt.ConnectionType.QueuedConnection,
        )

        self._set_auto_setup_busy(True)
        self.status.setText("Auto Setup: connecting...")
        thread.start()
        log.info("Scope ch%d: Auto Setup started", self._channel)

    def _set_auto_setup_busy(self, busy: bool) -> None:
        self._set_controls_enabled(not busy)
        self.ui.btnStart.setEnabled(not busy)
        self.ui.btnStop.setEnabled(False)
        self.ui.btnAcquireFrame.setEnabled(not busy)
        self.ui.btnDmaFile.setEnabled(not busy)
        self.ui.btnAutoSetup.setEnabled(self._scope.dac_available())
        self.ui.btnAutoSetup.setText("Cancel Auto" if busy else "Auto Setup")

    def _on_auto_setup_succeeded(self, result: ScopeAutoSetupResult) -> None:
        self._auto_setup_result = result

    def _on_auto_setup_error(self, message: str) -> None:
        self._auto_setup_error = message

    @Slot()
    def _on_auto_setup_finished(self) -> None:
        thread = self._auto_setup_thread
        if thread is None:
            return  # synchronous shutdown already reaped this worker
        if not thread.wait(2000):
            log.error(
                "Scope ch%d: Auto Setup thread emitted finished but did not exit",
                self._channel,
            )
            QTimer.singleShot(100, self._on_auto_setup_finished)
            return

        # QThread.finished can be queued while the native thread is still
        # completing teardown. Keep both Python wrappers alive until GUI and
        # hardware-state restoration is done; releasing them earlier can make
        # PySide destroy a still-running QThread and terminate the process.
        worker = self._auto_setup_worker
        log.info("Scope ch%d: Auto Setup worker exited; restoring controls", self._channel)
        result = self._auto_setup_result
        error = self._auto_setup_error
        self._auto_setup_worker = None
        self._auto_setup_thread = None
        try:
            if getattr(self, "_session_replay_active", False):
                self._apply_session_replay_controls()
                return
            self._set_auto_setup_busy(False)
            self.reload_hardware_state()
        except Exception as exc:
            log.exception("Scope ch%d: failed to refresh after Auto Setup", self._channel)
            self.status.setText(f"Auto Setup refresh failed: {exc}")
        else:
            if result is None:
                self.status.setText(
                    f"Auto Setup failed: {error or 'unknown error'}"
                )
            else:
                self._dac_adc_slope = result.dac_slope

                frame = result.frame[: self._frame_samples_from_ui() // 4]
                raw_time = np.arange(len(frame)) * self._VIEWER_POINT_PERIOD_NS
                self._on_frame_received([raw_time, frame])
                verification = "verified" if result.verified else "set; pulse not re-observed"
                self.status.setText(
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
        finally:
            thread.deleteLater()
            del worker

    def _start_measurement_timer(self) -> None:
        time_s = self.ui.spinTime.value()
        if time_s > 0:
            self._measurement_timer.start(time_s * 1000)
            log.info("Scope ch%d: measurement timer set to %d s", self._channel, time_s)

    def _on_measurement_timeout(self) -> None:
        log.info("Scope ch%d: measurement time limit reached", self._channel)
        if (series := owner_series(self)) is not None:
            series.mark_completed()
        self._on_stop()

    def _on_refresh_rate_changed(self, value: int) -> None:
        if self._refresh_timer.isActive():
            self._refresh_timer.setInterval(1000 // value)

    def _request_frame(self) -> None:
        # Runs on every refresh tick (up to 60 Hz): use the mirrored mode
        # rather than a blocking network read on the GUI thread.
        if self._scope.cached_trigger_mode() == TriggerMode.PERIODIC_SUM:
            return  # Viewer memory is unsupported for structured sums.
        if getattr(self, "_session_replay_active", False):
            return
        if self._dma_online_current and self._current_accumulator is not None:
            # Hidden Current-only operation performs no viewer transaction or
            # waveform conversion. A visible Scope consumes only the latest
            # immutable DMA record; intermediate previews are replaceable.
            if not self.isVisible():
                return
            preview = self._current_accumulator.take_latest_preview()
            if preview is not None:
                self._on_dma_preview_received(preview)
            return
        if self._acquiring:
            return
        self._acquiring = True
        worker = ScopeWorker(self._viewer_scope_for_worker)
        worker.signals.failed.connect(self._on_viewer_frame_failed)
        worker.signals.ready.connect(self._on_viewer_frame_received)
        QThreadPool.globalInstance().start(worker)

    def _on_dma_preview_received(self, preview: ScopeCurrentPreview) -> None:
        if getattr(self, "_session_replay_active", False):
            return
        if self._waveform_drag is not None or self._pretrigger_line_drag is not None:
            return
        geometry = preview.geometry
        frame = np.frombuffer(
            preview.record,
            dtype="<i2",
            count=geometry.waveform_samples,
            offset=8,
        )
        source = f"DMA preview ({geometry.sample_period_ns} ns/sample)"
        self._accumulate_scope_distribution(
            frame, source=source, sample_period_ns=geometry.sample_period_ns,
        )
        self._accumulate_scope_spectrum(
            frame, source=source, sample_period_ns=geometry.sample_period_ns,
        )
        x_time = np.arange(frame.size) * geometry.sample_period_ns
        if self._display_mode == DisplayMode.RAW:
            self._raw_curve.setData(x_time / self._time_scale.ns_per_unit, frame)
        else:
            self._rasterize_frame(frame)
            self._persistence_img.setImage(
                self._persistence_buffer, autoLevels=False, levels=(0, 1)
            )

    def _viewer_scope_for_worker(self) -> Scope:
        """Lazily open a viewer-only IIO connection on the worker thread."""
        with self._viewer_scope_lock:
            if self._viewer_scope is None:
                self._viewer_scope = self._scope.create_isolated_client()
            return self._viewer_scope

    def close_viewer_client(self) -> None:
        """Call only after the pool's in-flight viewer work has finished."""
        self.stop_current_sum_monitor(wait=True)
        with self._viewer_scope_lock:
            viewer_scope = self._viewer_scope
            self._viewer_scope = None
        if viewer_scope is not None:
            viewer_scope.close()

    @Slot(str, bool)
    def _on_viewer_frame_failed(self, message: str, connection_lost: bool) -> None:
        """Stop live viewing once the board connection is gone, without hardware writes.

        A reset network context never recovers, so requesting further frames
        would only repeat the failure at the refresh rate. The normal Stop path
        writes ``enable=0`` and would fail the same way; a DMA session keeps
        its own reader-process teardown.
        """
        if (
            not connection_lost
            or getattr(self, "_session_replay_active", False)
            or not self._refresh_timer.isActive()
        ):
            return
        self._refresh_timer.stop()
        self._discard_inflight_frame = True
        self._acquiring = False
        if (series := owner_series(self)) is not None:
            series.cancel()
        if self._dma_worker is None:
            self._raw_curve.cancelDrag()
            self._cancel_pretrigger_line_drag()
            self._measurement_timer.stop()
            ScopeController._set_measurement_active(self, False)
            self.ui.btnStop.setChecked(True)
            self.ui.btnStart.setChecked(False)
            self.ui.btnStart.setEnabled(True)
            self.ui.btnStop.setEnabled(False)
            self.ui.btnAcquireFrame.setEnabled(True)
            self.ui.cbDmaEnable.setEnabled(True)
            self.ui.btnDmaFile.setEnabled(True)
        self.status.setText(
            f"Connection lost: {message}; use Connection → Reconnect Device."
        )
        log.warning("Scope ch%d: live view stopped after the board connection was lost",
                    self._channel)
        self.connection_lost.emit(message)

    @Slot(object)
    def _on_viewer_frame_received(self, data: list[np.ndarray] | None) -> None:
        self._acquiring = False
        if (
            data is None
            or self._discard_inflight_frame
            or getattr(self, "_session_replay_active", False)
        ):
            return
        self._on_frame_received(data)

    def _on_frame_received(self, data: list[np.ndarray]) -> None:
        self._acquiring = False
        if getattr(self, "_session_replay_active", False):
            return
        _x_time, captured_samples = data
        signal = getattr(self, "session_viewer_frame", None)
        if signal is not None:
            signal.emit(
                self._channel,
                captured_samples,
                self._VIEWER_POINT_PERIOD_NS,
            )
        if self._waveform_drag is not None or self._pretrigger_line_drag is not None:
            return
        pending_status = self._waveform_pending_status
        if pending_status is not None:
            self.status.setText(pending_status)
            self._waveform_pending_status = None
        x_time, y_voltage = data
        self._accumulate_scope_distribution(
            y_voltage, source="Scope viewer (8 ns mean4)",
            sample_period_ns=self._VIEWER_POINT_PERIOD_NS,
        )
        self._accumulate_scope_spectrum(
            y_voltage, source="Scope viewer (8 ns mean4)",
            sample_period_ns=self._VIEWER_POINT_PERIOD_NS,
        )
        expected = self._frame_samples_from_ui() // 4
        if len(y_voltage) < expected:
            self.status.setText(
                self._viewer_truncation_message(len(y_voltage), expected)
            )
        if self._display_mode == DisplayMode.RAW:
            self._raw_curve.setData(x_time / self._time_scale.ns_per_unit, y_voltage)
        else:
            self._rasterize_frame(y_voltage)
            self._persistence_img.setImage(
                self._persistence_buffer, autoLevels=False, levels=(0, 1)
            )
        if pending_status is not None:
            self._update_waveform_drag_enabled()

    def _on_frame_samples_changed(self) -> None:
        if self._replay_hardware_interlocked:
            return
        self._raw_curve.cancelDrag()
        self._cancel_pretrigger_line_drag()
        if self._waveform_pending_status is not None:
            self._raw_curve.setData([], [])
            self._waveform_pending_status = None
            self._update_waveform_drag_enabled()
        frame_samples = self._frame_samples_from_ui()
        self._scope.set_frame_samples(frame_samples)
        self._reset_persistence_buffer(frame_samples)
        self._update_axis_ranges()
        self.clear_scope_spectrum("frame duration changed")
        if self._frame_exceeds_viewer_limit():
            self.status.setText(self._viewer_limit_message())

    def _on_acquire_frame(self) -> None:
        if getattr(self, "_replay_hardware_interlocked", False):
            self.status.setText(
                "Exit session replay before acquiring a live Scope frame"
            )
            return
        if self._scope.get_trigger_mode() == TriggerMode.PERIODIC_SUM:
            self.status.setText(
                "Periodic sums belong to Current. Select a waveform trigger to use Scope."
            )
            return
        self._raw_curve.cancelDrag()
        self._cancel_pretrigger_line_drag()
        try:
            self._scope.start()
            raw_frame = self._scope.acquire_frame()
        except Exception as exc:
            log.exception("Scope ch%d: single-frame acquisition failed", self._channel)
            self.status.setText(f"Viewer error: {exc}")
            return
        finally:
            try:
                self._scope.stop()
            except Exception:
                log.exception("Scope ch%d: failed to disarm after viewer read", self._channel)
        frame_samples = self._frame_samples_from_ui()
        frame = raw_frame[: frame_samples // 4]
        self._accumulate_scope_distribution(
            frame, source="Scope viewer (8 ns mean4)",
            sample_period_ns=self._VIEWER_POINT_PERIOD_NS,
        )
        self._accumulate_scope_spectrum(
            frame, source="Scope viewer (8 ns mean4)",
            sample_period_ns=self._VIEWER_POINT_PERIOD_NS,
        )
        signal = getattr(self, "session_viewer_frame", None)
        if signal is not None:
            signal.emit(
                self._channel,
                frame,
                self._VIEWER_POINT_PERIOD_NS,
            )
        expected = frame_samples // 4
        if len(frame) < expected:
            self.status.setText(
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
        pending_status = self._waveform_pending_status
        if pending_status is not None:
            self.status.setText(pending_status)
            self._waveform_pending_status = None
            self._update_waveform_drag_enabled()

    def _reset_persistence_buffer(self, frame_samples: int) -> None:
        """One column per viewer point, fixed rows; see _PERSISTENCE_ROWS."""
        self._display_nx = frame_samples // 4
        self._display_ny = self._PERSISTENCE_ROWS
        self._persistence_buffer = np.zeros(
            (self._display_nx, self._display_ny), dtype=np.float32
        )

    def _rasterize_frame(self, frame: np.ndarray) -> None:
        buffer = self._persistence_buffer
        buffer *= self.persistence
        n_samples = min(len(frame), self._display_nx)
        if not n_samples:
            return
        # A legacy text viewer can return only the prefix that fits in one
        # sysfs page. Keep that prefix at its real time coordinates instead
        # of stretching it across the full configured frame.
        y_float = (
            (frame[:n_samples].astype(np.float32) - self._Y_MIN)
            / (self._Y_MAX - self._Y_MIN)
            * (self._display_ny - 1)
        )
        y_pixels = np.clip(y_float, 0, self._display_ny - 1).astype(np.intp)
        # Match the raw trace's sample hold: column i spans the transition
        # from sample i-1 to sample i (column 0 is the first sample alone),
        # rather than interpolating amplitude between sample times.
        low = y_pixels.copy()
        high = y_pixels.copy()
        low[1:] = np.minimum(y_pixels[:-1], y_pixels[1:])
        high[1:] = np.maximum(y_pixels[:-1], y_pixels[1:])
        rows = np.arange(self._display_ny)
        buffer[:n_samples][(rows >= low[:, None]) & (rows <= high[:, None])] = 1.0

    # ------------------------------------------------------------------
    # Graph setup
    # ------------------------------------------------------------------

    def _setup_graph(self) -> None:
        self._setup_plot_layout()
        self._setup_distribution_plot()
        self._setup_spectrum_plot()
        self._setup_persistence_layer()
        self._setup_raw_layer()
        self._setup_threshold_line()
        self._setup_pretrigger_line()
        self._update_axis_ranges()
        self._set_display_mode(DisplayMode(self.ui.comboDisplayMode.currentIndex()))

    def _setup_plot_layout(self) -> None:
        layout = pg.GraphicsLayout()
        self._scope_plot_layout = layout
        layout.setContentsMargins(10, 10, 10, 10)
        self.ui.plotWaveform.setCentralItem(layout)
        self.ui.plotWaveform.setBackground(theme.colors().plot_background)

        layout.addLabel("Amplitude [raw]", angle=-90)
        self._plot_item = layout.addPlot(viewBox=ModifierZoomViewBox())
        show_frame_axes(self._plot_item)
        self._plot_item.showGrid(x=True, y=True, alpha=0.2)
        layout.nextRow()
        self._time_axis_label = layout.addLabel("Time [ns]", col=1)

    def _setup_distribution_plot(self) -> None:
        distribution_view = HistogramViewBox()
        self._distribution_plot = pg.PlotItem(viewBox=distribution_view)
        self._distribution_plot.setParentItem(self._scope_plot_layout)
        self._distribution_plot.hideAxis("left")
        # Same zero-height top frame as the waveform plot: their Y-linked view
        # boxes keep the same top and bottom edges.
        show_frame_axes(self._distribution_plot, "top")
        self._distribution_plot.showGrid(x=True, y=True, alpha=0.2)
        self._distribution_plot.setYLink(self._plot_item)
        self._distribution_curve = pg.PlotDataItem(
            fillLevel="enclosed",
            fillBrush=pg.mkBrush(39, 125, 161, 60),
            pen=pg.mkPen("#277da1"),
        )
        self._distribution_plot.addItem(self._distribution_curve)
        self._distribution_plot.ctrl.logXCheck.toggled.connect(
            self._on_distribution_scale_changed
        )
        self._distribution_plot.ctrl.logYCheck.setEnabled(False)
        self._distribution_plot.ctrl.logYCheck.setToolTip(
            "Amplitude stays linear and shares the Scope Y axis."
        )
        self._setup_distribution_menu(distribution_view)
        self._distribution_label = pg.LabelItem("Counts")
        self._distribution_label.setParentItem(self._scope_plot_layout)
        self._distribution_plot.hide()
        self._distribution_label.hide()
        self._plot_item.getViewBox().sigYRangeChanged.connect(self._refresh_scope_distribution)
        self.ui.comboTriggerMode.currentIndexChanged.connect(self._sync_distribution_direction)

    def _setup_distribution_menu(self, view: HistogramViewBox) -> None:
        self._distribution_keep_auto_y = False
        self._distribution_menu = QMenu(self)
        self._distribution_mode_group = QActionGroup(self)
        self._distribution_mode_actions: dict[HistogramMode, QAction] = {}
        for mode, text in (
            (HistogramMode.SAMPLES, "Sample distribution (all samples)"),
            (HistogramMode.AMPLITUDE, "Peak amplitude (one per waveform)"),
            (HistogramMode.INTEGRAL, "Pulse integral (one per waveform)"),
        ):
            action = self._distribution_menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(mode == self._amplitude_histogram.mode)
            action.setToolTip("Changing mode clears this channel's accumulated histogram.")
            self._distribution_mode_group.addAction(action)
            action.triggered.connect(
                lambda _checked=False, selected=mode: self.set_scope_distribution_mode(selected)
            )
            self._distribution_mode_actions[mode] = action
        self._distribution_menu.addSeparator()
        self._distribution_log_action = self._distribution_menu.addAction("Logarithmic counts (L)")
        self._distribution_log_action.setCheckable(True)
        self._distribution_log_action.toggled.connect(self._set_distribution_log_counts)
        self._distribution_menu.addAction("Reset histogram zoom", self._reset_distribution_zoom)
        self._distribution_menu.addAction("Clear histogram", self.clear_scope_distribution)
        self._distribution_menu.aboutToShow.connect(self._sync_distribution_menu)
        view.set_histogram_menu(self._distribution_menu)

    def _sync_distribution_menu(self) -> None:
        blocked = self._distribution_log_action.blockSignals(True)
        self._distribution_log_action.setChecked(self._distribution_plot.getAxis("bottom").logMode)
        self._distribution_log_action.blockSignals(blocked)

    def set_scope_distribution_mode(self, mode: HistogramMode) -> None:
        """Choose this channel's histogram observable and start new counts."""
        mode = HistogramMode(mode)
        if self._amplitude_histogram.mode == mode:
            return
        self._amplitude_histogram.set_mode(mode)
        self._distribution_mode_actions[mode].setChecked(True)
        if mode == HistogramMode.INTEGRAL:
            self._distribution_plot.setYLink(None)
            self._distribution_plot.showAxis("left")
            self._distribution_plot.setLabel("left", "Pulse integral [ADC codes·ns]")
            self._distribution_plot.getAxis("left").enableAutoSIPrefix(False)
        else:
            self._distribution_plot.hideAxis("left")
            self._distribution_plot.setYLink(self._plot_item)
        self._reset_distribution_zoom()

    def _reset_distribution_zoom(self) -> None:
        if self._amplitude_histogram.mode == HistogramMode.INTEGRAL:
            self._distribution_plot.enableAutoRange(axis="y", enable=True)
        else:
            self._plot_item.setYRange(self._Y_MIN, self._Y_MAX, padding=0)
        self._refresh_scope_distribution()

    def set_scope_distribution_enabled(self, enabled: bool) -> None:
        """Show and accumulate received live waveforms; hiding pauses counting."""
        if self._distribution_enabled == enabled:
            return
        self._distribution_enabled = enabled
        layout = self._scope_plot_layout
        if enabled:
            layout.addItem(self._distribution_plot, row=0, col=2)
            layout.addItem(self._distribution_label, row=1, col=2)
            layout.layout.setColumnStretchFactor(1, 4)
            layout.layout.setColumnStretchFactor(2, 1)
            self._distribution_plot.show()
            self._distribution_label.show()
            self._refresh_scope_distribution()
        else:
            for item in (self._distribution_plot, self._distribution_label):
                layout.removeItem(item)
                item.setParentItem(layout)
                item.hide()
            layout.layout.setColumnStretchFactor(2, 0)

    def toggle_scope_distribution_log_under_mouse(self) -> bool:
        """Toggle count scale only while the pointer is over this distribution."""
        if not self._distribution_enabled or not self.isVisible():
            return False
        widget = self.ui.plotWaveform
        if not widget.viewport().underMouse():
            return False
        position = widget.mapToScene(widget.viewport().mapFromGlobal(QCursor.pos()))
        if not self._distribution_plot.sceneBoundingRect().contains(position):
            return False
        logarithmic = self._distribution_plot.getAxis("bottom").logMode
        self._set_distribution_log_counts(not logarithmic)
        return True

    def _set_distribution_log_counts(self, enabled: bool) -> None:
        self._distribution_keep_auto_y = bool(
            self._distribution_plot.getViewBox().autoRangeEnabled()[1]
        )
        self._distribution_plot.setLogMode(x=enabled, y=False)

    def _on_distribution_scale_changed(self, _enabled: bool) -> None:
        # PlotItem's log toggle enables both auto ranges. Preserve both
        # amplitude and integral zoom while changing only the count scale.
        main_range = self._plot_item.getViewBox().viewRange()[1][:]
        side_range = self._distribution_plot.getViewBox().viewRange()[1][:]
        integral = self._amplitude_histogram.mode == HistogramMode.INTEGRAL
        self._distribution_plot.disableAutoRange()
        self._plot_item.setYRange(*main_range, padding=0)
        if integral:
            self._distribution_plot.setYRange(*side_range, padding=0)
            if self._distribution_keep_auto_y:
                self._distribution_plot.enableAutoRange(axis="y", enable=True)
        self._refresh_scope_distribution()

    def clear_scope_distribution(self) -> None:
        """Start a new live amplitude distribution on this channel."""
        self._amplitude_histogram.clear()
        if self._amplitude_histogram.mode == HistogramMode.INTEGRAL:
            self._distribution_plot.enableAutoRange(axis="y", enable=True)
        self._refresh_scope_distribution()

    def _distribution_negative_polarity(self) -> bool:
        trigger = self.ui.comboTriggerMode.currentData()
        return trigger in (int(TriggerMode.ANY_BELOW), int(TriggerMode.FALLING_EDGE))

    def _sync_distribution_direction(self, *_args: object) -> None:
        inverted = self._plot_item.getViewBox().yInverted()
        if (
            self._amplitude_histogram.mode == HistogramMode.INTEGRAL
            and self._distribution_negative_polarity()
        ):
            inverted = not inverted
        self._distribution_plot.getViewBox().invertY(inverted)

    def _accumulate_scope_distribution(
        self, samples: np.ndarray, *, source: str, sample_period_ns: float
    ) -> None:
        if not self._distribution_enabled or self._session_replay_active:
            return
        negative = self._distribution_negative_polarity()
        self._amplitude_histogram.add(
            samples, source=source, negative=negative,
            sample_period_ns=sample_period_ns,
            pretrigger_ns=self.ui.spinPretrigger.value(),
        )
        self._refresh_scope_distribution()

    def _refresh_scope_distribution(self, *_args: object) -> None:
        self._sync_distribution_direction()
        if not self._distribution_enabled:
            return
        histogram = self._amplitude_histogram
        lower, upper = self._plot_item.getViewBox().viewRange()[1]
        edges, counts = histogram.projected_bins(lower, upper)
        logarithmic = self._distribution_plot.getAxis("bottom").logMode
        if histogram.sample_count and counts.size:
            count_path = np.concatenate(([0], np.repeat(counts, 2), [0, 0]))
            if logarithmic:
                # Zero has no logarithm; clip only for rendering. Positive
                # integer counts keep their exact values, including ones.
                count_path = np.maximum(count_path, 0.5)
            amplitude_path = np.concatenate(
                ([edges[0]], np.repeat(edges, 2)[1:-1], [edges[-1], edges[0]])
            )
            self._distribution_curve.setData(count_path, amplitude_path)
        else:
            self._distribution_curve.clear()
        maximum = float(counts.max()) if counts.size else 0.0
        if logarithmic:
            self._distribution_plot.setXRange(
                np.log10(0.5), np.log10(max(1.0, maximum)) + 0.05, padding=0
            )
        else:
            self._distribution_plot.setXRange(0, max(1.0, maximum * 1.05), padding=0)
        mode_name = {
            HistogramMode.SAMPLES: "Sample distribution",
            HistogramMode.AMPLITUDE: "Peak amplitude",
            HistogramMode.INTEGRAL: "Pulse integral",
        }[histogram.mode]
        entry_unit = "samples" if histogram.mode == HistogramMode.SAMPLES else "entries"
        self._distribution_label.setText(
            f"{mode_name} — Counts{' (log)' if logarithmic else ''}<br>"
            f"<span style='font-size:8pt'>{histogram.sample_count:,} {entry_unit} / "
            f"{histogram.frame_count:,} frames; skipped {histogram.rejected_frames:,}</span>"
        )
        # A plot item's tooltip bypasses the application's widget-tooltip
        # wrapping, so keep it short and wrap it here.
        tooltip = [
            f"{histogram.source or 'Live Scope distribution; waiting for waveforms'}.",
            "Received waveforms only; polling and preview omit frames.",
            {
                HistogramMode.SAMPLES: "Each display bin spans 100 ADC codes.",
                HistogramMode.AMPLITUDE: (
                    "One raw minimum per frame for falling/any-below triggers, "
                    "otherwise one maximum; bins span 100 ADC codes."
                ),
                HistogramMode.INTEGRAL: (
                    "Pulse-side area beyond the pretrigger mean, ADC codes·ns, bin "
                    f"{histogram.integral_bin_width:g}. Noise adds area; no baseline skips."
                ),
            }[histogram.mode],
            "Right-click: mode, zoom, clear. Hover and press L: linear/log counts. "
            "Paused while hidden; session replay adds nothing.",
        ]
        if histogram.rejected_count or histogram.rejected_frames:
            tooltip.append(
                f"Rejected samples {histogram.rejected_count:,}, skipped frames "
                f"{histogram.rejected_frames:,}. {histogram.last_rejection}".rstrip()
            )
        self._distribution_plot.setToolTip(wrap_tooltip("\n".join(tooltip)))

    # ------------------------------------------------------------------
    # Noise spectrum (display-only consumer of the same received frames)
    # ------------------------------------------------------------------

    def _setup_spectrum_plot(self) -> None:
        layout = self._scope_plot_layout
        view = HistogramViewBox()
        self._spectrum_plot = pg.PlotItem(viewBox=view)
        self._spectrum_plot.setParentItem(layout)
        show_frame_axes(self._spectrum_plot)
        self._spectrum_plot.showGrid(x=True, y=True, alpha=0.2)
        set_frequency_axis(self._spectrum_plot, log=False)
        self._spectrum_curve = pg.PlotDataItem(pen=pg.mkPen("#277da1"))
        self._spectrum_markers = pg.PlotDataItem(
            pen=None, symbol="t", symbolSize=9,
            symbolBrush=pg.mkBrush("#c0563b"), symbolPen=None,
        )
        self._spectrum_plot.addItem(self._spectrum_curve)
        self._spectrum_plot.addItem(self._spectrum_markers)
        self._spectrum_axis_label = pg.LabelItem("", angle=-90)
        self._spectrum_axis_label.setParentItem(layout)
        for item in (self._spectrum_plot, self._spectrum_axis_label):
            item.setData(GRAPHICS_HELP_TOPIC_KEY, "scope-spectrum")
            item.hide()
        # A plain wrapping label: a pg.LabelItem pins its layout column to the
        # text width, which pushed the amplitude distribution off screen.
        self._spectrum_label = QLabel(self)
        self._spectrum_label.setObjectName("labelSpectrumInfo")
        self._spectrum_label.setWordWrap(True)
        self._spectrum_label.setTextFormat(Qt.TextFormat.RichText)
        self._spectrum_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        self._spectrum_label.hide()
        self.ui.waveformLayout.addWidget(self._spectrum_label)
        self._setup_spectrum_menu(view)
        scene = self.ui.plotWaveform.scene()
        if scene is not None:
            scene.sigMouseMoved.connect(self._on_spectrum_mouse_moved)
        self.ui.comboTriggerMode.currentIndexChanged.connect(
            lambda _index: self.clear_scope_spectrum("trigger mode changed")
        )
        self.ui.spinDacValue.valueChanged.connect(
            lambda _value: self.clear_scope_spectrum("DAC baseline changed")
        )
        # Tick labels are measured when painted, so re-align shortly after a
        # range change rather than before the new labels exist.
        self._spectrum_axis_timer = QTimer(self)
        self._spectrum_axis_timer.setSingleShot(True)
        self._spectrum_axis_timer.setInterval(100)
        self._spectrum_axis_timer.timeout.connect(self._align_spectrum_axes)
        for plot in (self._plot_item, self._spectrum_plot):
            plot.getViewBox().sigYRangeChanged.connect(self._schedule_spectrum_axis_alignment)

    def _schedule_spectrum_axis_alignment(self, *_args: object) -> None:
        if self._spectrum_enabled:
            self._spectrum_axis_timer.start()

    def _align_spectrum_axes(self) -> None:
        """Give both left axes one width so the two data areas line up.

        Each axis otherwise sizes itself to its own tick labels (``-30000``
        versus ``-15``), offsetting the spectrum from the waveform above it.
        Hidden, the waveform axis returns to automatic sizing.
        """
        main = self._plot_item.getAxis("left")
        if not self._spectrum_enabled:
            if main.fixedWidth is not None:
                main.setWidth(None)
            return
        side = self._spectrum_plot.getAxis("left")

        def needed(axis: Any) -> float:
            # AxisItem._updateWidth without a label: tick text, offset, ticks.
            style = axis.style
            return float(axis.textWidth + style["tickTextOffset"][0]
                         + max(0, style["tickLength"]))

        width = int(np.ceil(max(needed(main), needed(side))))
        for axis in (main, side):
            if axis.fixedWidth != width:
                axis.setWidth(width)

    def _setup_spectrum_menu(self, view: HistogramViewBox) -> None:
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        self._spectrum_menu = menu
        self._spectrum_actions: dict[tuple[str, object], QAction] = {}

        def choices(
            title: str, kind: str, handler: Callable[[Any], None],
            options: tuple[tuple[object, str, str], ...],
        ) -> None:
            submenu = menu.addMenu(title)
            submenu.setToolTipsVisible(True)
            group = QActionGroup(self)
            for value, text, tip in options:
                action = submenu.addAction(text)
                action.setCheckable(True)
                action.setToolTip(tip)
                group.addAction(action)
                action.triggered.connect(
                    lambda _checked=False, selected=value: handler(selected)
                )
                self._spectrum_actions[(kind, value)] = action

        choices("Window", "window", self.set_scope_spectrum_window, (
            (SpectrumWindow.HANN, "Hann (noise floor)",
             "1.5-bin noise bandwidth; a tone between bins reads up to 1.42 dB low."),
            (SpectrumWindow.BLACKMAN_HARRIS, "Blackman-Harris (low leakage)",
             "2.0-bin noise bandwidth; up to 0.83 dB low between bins."),
            (SpectrumWindow.FLATTOP, "Flat-top (tone amplitude)",
             "Tone amplitude within 0.01 dB; 3.8-bin noise bandwidth."),
            (SpectrumWindow.BOXCAR, "Rectangular",
             "Narrowest bins but strong leakage; up to 3.92 dB low between bins."),
        ))
        choices("Averaging", "averaging", self.set_scope_spectrum_averaging, (
            (SpectrumAveraging.LINEAR, "Linear (N frames, then hold)",
             "Mean of independent frames in linear power; holds after N frames."),
            (SpectrumAveraging.EXPONENTIAL, "Exponential (α = 1/N)",
             "Running average in linear power; follows changes over about N frames."),
            (SpectrumAveraging.MAX_HOLD, "Max hold",
             "Largest power seen in each bin since the last reset."),
            (SpectrumAveraging.MIN_HOLD, "Min hold",
             "Smallest power seen in each bin since the last reset."),
        ))
        choices("Frames N", "count", self.set_scope_spectrum_average_count, tuple(
            (count, f"{count:,}" if count else "Unlimited",
             "Frames per linear average, or the exponential time constant.")
            for count in _SPECTRUM_COUNTS
        ))
        choices("Units", "unit", self.set_scope_spectrum_unit, tuple(
            (unit, UNIT_LABELS[unit], "Display only; averaging always uses linear power.")
            for unit in SpectrumUnit
        ))
        self._spectrum_log_action = menu.addAction("Logarithmic frequency axis")
        self._spectrum_log_action.setCheckable(True)
        self._spectrum_log_action.toggled.connect(self.set_scope_spectrum_log_frequency)
        self._spectrum_compensation_action = menu.addAction(
            "Compensate mean4 response (display only)"
        )
        self._spectrum_compensation_action.setCheckable(True)
        self._spectrum_compensation_action.setToolTip(
            "Divides out the four-sample mean's |H|² where |H| ≥ 0.1. Aliased content "
            "from 62.5-125 MHz is not removed. Exported values stay uncompensated."
        )
        self._spectrum_compensation_action.toggled.connect(self.set_scope_spectrum_compensation)
        menu.addSeparator()
        menu.addAction("Reset spectrum", lambda: self.clear_scope_spectrum("reset"))
        menu.addAction("Export spectrum CSV…", self.export_scope_spectrum)
        menu.aboutToShow.connect(self._sync_spectrum_menu)
        view.set_histogram_menu(menu)
        self._sync_spectrum_menu()

    def _sync_spectrum_menu(self) -> None:
        selected = {
            ("window", self._spectrum.window),
            ("averaging", self._spectrum.averaging),
            ("count", self._spectrum.average_count),
            ("unit", self._spectrum_unit),
        }
        for key, action in self._spectrum_actions.items():
            action.setChecked(key in selected)
        for action, checked in (
            (self._spectrum_log_action, self._spectrum_log_frequency),
            (self._spectrum_compensation_action, self._spectrum_compensate),
        ):
            blocked = action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(blocked)

    def _spectrum_settings(self) -> dict[str, int | bool | str]:
        return {
            "window": self._spectrum.window.value,
            "averaging": self._spectrum.averaging.value,
            "average_count": self._spectrum.average_count,
            "unit": self._spectrum_unit.value,
            "log_frequency": self._spectrum_log_frequency,
            "mean4_compensation": self._spectrum_compensate,
        }

    def _apply_spectrum_settings(self, values: object) -> None:
        """Apply saved spectrum options; skip (and log) any invalid entry."""
        if not isinstance(values, Mapping):
            return

        def flag(value: object) -> bool:
            return str(value).strip().lower() in ("1", "true", "yes", "on")

        appliers: dict[str, Callable[[object], None]] = {
            "window": lambda value: self._spectrum.set_window(str(value)),
            "averaging": lambda value: self._spectrum.set_averaging(str(value)),
            "average_count": lambda value: self._spectrum.set_average_count(int(str(value))),
            "unit": lambda value: setattr(self, "_spectrum_unit", SpectrumUnit(str(value))),
            "log_frequency": lambda value: setattr(self, "_spectrum_log_frequency", flag(value)),
            "mean4_compensation": lambda value: setattr(
                self, "_spectrum_compensate", flag(value)
            ),
        }
        for name, apply in appliers.items():
            if name in values:
                try:
                    apply(values[name])
                except (TypeError, ValueError):
                    log.warning("Ignoring invalid Scope spectrum %s %r", name, values[name])
        if hasattr(self, "_spectrum_menu"):
            self._sync_spectrum_menu()
            self._spectrum_plot.enableAutoRange()
            self._refresh_scope_spectrum(force_spurs=True)

    def set_scope_spectrum_window(self, window: SpectrumWindow | str) -> None:
        self._spectrum.set_window(window)
        self._sync_spectrum_menu()
        self._refresh_scope_spectrum(force_spurs=True)

    def set_scope_spectrum_averaging(self, averaging: SpectrumAveraging | str) -> None:
        self._spectrum.set_averaging(averaging)
        self._sync_spectrum_menu()
        self._refresh_scope_spectrum(force_spurs=True)

    def set_scope_spectrum_average_count(self, count: int) -> None:
        self._spectrum.set_average_count(int(count))
        self._sync_spectrum_menu()
        self._refresh_scope_spectrum(force_spurs=True)

    def set_scope_spectrum_unit(self, unit: SpectrumUnit | str) -> None:
        self._spectrum_unit = SpectrumUnit(unit)
        self._sync_spectrum_menu()
        self._spectrum_plot.enableAutoRange()
        self._refresh_scope_spectrum(force_spurs=True)

    def set_scope_spectrum_log_frequency(self, enabled: bool) -> None:
        self._spectrum_log_frequency = bool(enabled)
        self._sync_spectrum_menu()
        self._spectrum_plot.enableAutoRange()
        self._refresh_scope_spectrum()

    def set_scope_spectrum_compensation(self, enabled: bool) -> None:
        self._spectrum_compensate = bool(enabled)
        self._sync_spectrum_menu()
        self._refresh_scope_spectrum(force_spurs=True)

    def set_scope_spectrum_enabled(self, enabled: bool) -> None:
        """Show and average received live waveforms' spectra; hiding pauses."""
        if self._spectrum_enabled == enabled:
            return
        self._spectrum_enabled = enabled
        layout = self._scope_plot_layout
        items = ((self._spectrum_axis_label, 0), (self._spectrum_plot, 1))
        grid = layout.layout
        if enabled:
            for item, column in items:
                layout.addItem(item, row=2, col=column)
                item.show()
            grid.setRowSpacing(1, grid.verticalSpacing())
            grid.setRowStretchFactor(0, 3)
            grid.setRowStretchFactor(2, 2)
            self._spectrum_label.show()
            self._refresh_scope_spectrum(force_spurs=True)
            self._schedule_spectrum_axis_alignment()
        else:
            for item, _column in items:
                layout.removeItem(item)
                item.setParentItem(layout)
                item.hide()
            self._spectrum_label.hide()
            # An explicit 0 would override the plot's Expanding policy and
            # collapse it; 1 restores the original share of the free height.
            # The empty row 2 would otherwise still add its row spacing.
            grid.setRowStretchFactor(0, 1)
            grid.setRowStretchFactor(2, 0)
            grid.setRowSpacing(1, 0)
            self._spectrum_axis_timer.stop()
            self._align_spectrum_axes()

    def clear_scope_spectrum(self, reason: str = "cleared") -> None:
        """Start a new spectrum average on this channel and record why."""
        self._spectrum.clear(reason)
        self._spectrum_spurs = []
        self._spectrum_spurs_at = -np.inf
        self._spectrum_spurs_frames = 0
        if hasattr(self, "_spectrum_plot"):
            self._spectrum_plot.enableAutoRange()
            self._refresh_scope_spectrum()

    def _spectrum_trigger_mode(self) -> str | None:
        text = self.ui.comboTriggerMode.currentText()
        return text or None

    def _accumulate_scope_spectrum(
        self, samples: np.ndarray, *, source: str, sample_period_ns: float
    ) -> None:
        if not self._spectrum_enabled or self._session_replay_active:
            return
        if self._waveform_drag is not None or self._pretrigger_line_drag is not None:
            return
        rejected = self._spectrum.rejected_frames
        added = self._spectrum.add(
            samples, source=source, sample_period_ns=sample_period_ns,
            trigger_mode=self._spectrum_trigger_mode(),
        )
        # Repaint only when the spectrum, or its rejection report, changed.
        if added or self._spectrum.rejected_frames != rejected:
            self._refresh_scope_spectrum()

    def _refresh_scope_spectrum(self, *, force_spurs: bool = False) -> None:
        if not self._spectrum_enabled:
            return
        unit = self._spectrum_unit
        log_mode = (self._spectrum_log_frequency, unit != SpectrumUnit.DB)
        if log_mode != self._spectrum_log_mode:
            # setLogMode re-transforms every item; do it only when it changes.
            self._spectrum_plot.setLogMode(x=log_mode[0], y=log_mode[1])
            set_frequency_axis(self._spectrum_plot, log=log_mode[0])
            self._spectrum_log_mode = log_mode
        quantity = "ASD" if unit == SpectrumUnit.ASD else "PSD"
        self._spectrum_axis_label.setText(f"{quantity} [{UNIT_LABELS[unit]}]")
        result = self._spectrum.result()
        if result is None:
            self._spectrum_curve.clear()
            self._spectrum_markers.clear()
            self._spectrum_display = None
            self._spectrum_spurs = []
            self._set_spectrum_info(None)
            self._align_spectrum_axes()
            return
        factor = decimation_factor(result.sample_period_ns, SCOPE_ADC_SAMPLE_PERIOD_NS)
        compensated = self._spectrum_compensate and factor > 1
        linear = (
            compensated_psd(result, factor, SCOPE_ADC_SAMPLE_PERIOD_NS)
            if compensated else result.psd
        )
        values = to_display_units(linear, unit)
        bins = plotted_bins(result.n)  # Without DC and the un-doubled Nyquist bin.
        frequencies, shown = result.frequencies[bins], values[bins]
        self._spectrum_display = (frequencies, shown)
        if unit != SpectrumUnit.DB:
            shown = np.maximum(shown, np.finfo(np.float64).tiny)
        self._spectrum_curve.setData(frequencies, shown)
        self._update_spectrum_spurs(force=force_spurs, factor=factor, compensated=compensated)
        spurs = [
            spur for spur in self._spectrum_spurs if bins.start <= spur.bin < bins.stop
        ]
        if spurs:
            marker_values = values[[spur.bin for spur in spurs]]
            if unit != SpectrumUnit.DB:
                marker_values = np.maximum(marker_values, np.finfo(np.float64).tiny)
            self._spectrum_markers.setData([spur.frequency for spur in spurs], marker_values)
        else:
            self._spectrum_markers.clear()
        self._set_spectrum_info(result, factor=factor, compensated=compensated, spurs=spurs)
        self._align_spectrum_axes()

    def _update_spectrum_spurs(self, *, force: bool, factor: int, compensated: bool) -> None:
        """Search the mean (never max-hold) for lines, at most every 0.5 s.

        A single frame's noise bins are exponentially distributed: one in
        about a thousand exceeds the median by 10 dB, so lines are only
        searched from K = 2. The threshold then falls with averaging
        (``line_threshold_db``: 10 dB to 4 dB), keeping false lines near one
        in a million bins. The search also reruns whenever K doubles, so a
        burst of frames cannot keep a K = 2 result.
        """
        frames = self._spectrum.frames
        if frames < 2:
            self._spectrum_spurs = []
            self._spectrum_spurs_frames = 0
            return
        now = monotonic()
        if not (
            force
            or now - self._spectrum_spurs_at >= _SPECTRUM_SPUR_INTERVAL_S
            or frames >= 2 * self._spectrum_spurs_frames
        ):
            return
        mean = self._spectrum.result("mean")
        if mean is None:
            return
        linear = (
            compensated_psd(mean, factor, SCOPE_ADC_SAMPLE_PERIOD_NS) if compensated else mean.psd
        )
        self._spectrum_spurs = find_spurs(
            replace(mean, psd=linear),
            threshold_db_over_median=line_threshold_db(mean.averages),
        )
        self._spectrum_spurs_at = now
        self._spectrum_spurs_frames = frames

    def _set_spectrum_info(
        self, result: FramePsd | None, *, factor: int = 1, compensated: bool = False,
        spurs: list[Spur] | None = None,
    ) -> None:
        spectrum = self._spectrum
        lines: list[str] = []
        if result is None:
            lines.append("Noise spectrum — waiting for waveforms")
        else:
            limit = (
                f"/{spectrum.average_count:,}"
                if spectrum.averaging == SpectrumAveraging.LINEAR and spectrum.average_count
                else ""
            )
            lines.append(
                f"{spectrum.source} · fs {format_frequency(result.fs)} · N {result.n:,} · "
                f"Δf {format_frequency(result.delta_f)} · ENBW {format_frequency(result.enbw)} "
                f"({result.window.value})"
            )
            lines.append(
                f"K {spectrum.frames:,}{limit}{' (complete)' if spectrum.complete else ''} · "
                f"±{100 * spectrum.relative_std:.1f}% per bin · "
                f"Parseval {parseval_check(result):.3f} · {spectrum.averaging.value} average"
            )
            if factor > 1:
                nyquist = result.fs / 2
                response = float(mean_decimation_response(nyquist, factor,
                                                          SCOPE_ADC_SAMPLE_PERIOD_NS)[()])
                lines.append(
                    f"mean{factor} points: |H| = {response:.3f} "
                    f"({20 * np.log10(response):.1f} dB) at {format_frequency(nyquist)}; "
                    f"{format_frequency(nyquist)}–{format_frequency(result.fs)} aliases"
                    f"{' · compensated for display' if compensated else ''}"
                )
        trigger = self.ui.comboTriggerMode.currentData()
        if trigger is not None and int(trigger) in {int(mode) for mode in
                                                    _PULSE_SYNCHRONISED_TRIGGERS}:
            lines.append(
                "<span style='color:#c0563b'>Triggered frames are synchronised to pulses: "
                "the spectrum includes pulse content. Use Periodic for baseline noise.</span>"
            )
        if spectrum.reset_reason and spectrum.reset_reason != "cleared":
            lines.append(f"Restarted: {spectrum.reset_reason}")
        if spectrum.last_rejection:
            lines.append(
                f"Skipped {spectrum.rejected_frames:,} frame(s): {spectrum.last_rejection}"
            )
        self._spectrum_info = "<br>".join(lines)
        self._update_spectrum_label()
        # A plot item's tooltip bypasses the application's widget-tooltip
        # wrapping, so wrap it here; the details live in F1 help.
        unit = UNIT_LABELS[self._spectrum_unit]
        tooltip = [
            "Noise spectrum of received Scope frames, in ADC codes.",
            "Right-click: window, averaging, units, reset, CSV export. F1: help.",
            _SPECTRUM_NOT_VALIDATED,
        ]
        if spurs:
            tooltip.append("Strongest lines:")
            tooltip.extend(
                f"{format_frequency(spur.frequency)}: "
                f"{to_display_units(np.array([spur.level]), self._spectrum_unit)[0]:.3g} "
                f"{unit} (+{spur.excess_db:.0f} dB)"
                for spur in spurs
            )
        self._spectrum_plot.setToolTip(wrap_tooltip("\n".join(tooltip)))

    def _update_spectrum_label(self) -> None:
        cursor = f"<br>{self._spectrum_cursor}" if self._spectrum_cursor else ""
        self._spectrum_label.setText(
            f"<span style='font-size:8pt'>{self._spectrum_info}{cursor}</span>"
        )

    def _on_spectrum_mouse_moved(self, position: QPointF) -> None:
        if not self._spectrum_enabled or self._spectrum_display is None:
            return
        if not self._spectrum_plot.sceneBoundingRect().contains(position):
            if self._spectrum_cursor:
                self._spectrum_cursor = ""
                self._update_spectrum_label()
            return
        frequencies, values = self._spectrum_display
        if not frequencies.size:
            return
        x = self._spectrum_plot.getViewBox().mapSceneToView(position).x()
        frequency = 10.0**x if self._spectrum_log_frequency else x
        index = int(np.argmin(np.abs(frequencies - frequency)))
        self._spectrum_cursor = (
            f"Cursor: {format_frequency(float(frequencies[index]))}, "
            f"{values[index]:.4g} {UNIT_LABELS[self._spectrum_unit]}"
        )
        self._update_spectrum_label()

    def _spectrum_export_metadata(self, stamp: datetime) -> dict[str, object]:
        factor = decimation_factor(self._spectrum.sample_period_ns, SCOPE_ADC_SAMPLE_PERIOD_NS)
        return {
            "channel": self._channel,
            "utc_timestamp": stamp.isoformat(timespec="seconds"),
            "app_version": __version__,
            "mean_decimation_factor": factor,
            "mean4_compensation": "not applied to exported values",
            "reset_reason": self._spectrum.reset_reason or "none",
            "validation": _SPECTRUM_NOT_VALIDATED,
        }

    def export_scope_spectrum(self) -> Path | None:
        """Save frequency, mean, max and min PSD with a metadata header."""
        if self._spectrum.result("mean") is None:
            self.status.setText("No Scope spectrum to export yet")
            return None
        stamp = datetime.now(UTC)
        default = (
            measurement_folder()
            / f"scope_ch{self._channel}_spectrum_{stamp:%Y%m%d_%H%M%S}.csv"
        )
        path_text, _ = QFileDialog.getSaveFileName(
            self, "Export Scope Spectrum", str(default), "CSV files (*.csv);;All files (*)",
        )
        if not path_text:
            return None
        path = Path(path_text)
        if not path.suffix:
            path = path.with_suffix(".csv")
        try:
            write_spectrum_csv(path, self._spectrum, self._spectrum_export_metadata(stamp))
        except (OSError, ValueError) as exc:
            log.exception("Scope ch%d: spectrum export to %s failed", self._channel, path)
            self.status.setText(f"Spectrum export failed: {exc}")
            return None
        self.status.setText(f"Scope spectrum saved to {path}")
        return path

    def _setup_persistence_layer(self) -> None:
        self._reset_persistence_buffer(self._frame_samples_from_ui())
        # Downsample to screen pixels before colour-mapping long frames.
        self._persistence_img = pg.ImageItem(autoDownsample=True)
        self._persistence_img.setImage(self._persistence_buffer, autoLevels=False, levels=(0, 1))
        self._persistence_img.setColorMap(theme.persistence_colormap())
        self._plot_item.addItem(self._persistence_img)

    def _setup_raw_layer(self) -> None:
        self._raw_curve = DraggableScopeCurve(pen=pg.mkPen(theme.colors().scope_trace, width=1))
        self._plot_item.addItem(self._raw_curve)
        self._raw_curve.setVisible(False)

    def _setup_threshold_line(self) -> None:
        spec = self._scope.specs[ScopeParam.TRIGGER_LEVEL]
        assert isinstance(spec, RangeSpec)
        self._threshold_line = pg.InfiniteLine(
            pos=self.ui.spinTriggerLevel.value(),
            angle=0,
            movable=True,
            pen=pg.mkPen(self._THRESHOLD_MARKER_COLOR, width=1.5, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen("#bd8b8b", width=2, style=Qt.PenStyle.DashLine),
            label="Threshold {value:.0f}",
            labelOpts={"color": self._THRESHOLD_MARKER_COLOR, "position": 0.98},
        )
        self._threshold_line.setBounds((int(spec.min_val), int(spec.max_val)))
        self._threshold_line.setZValue(10)
        self._threshold_line.setToolTip("Drag to set the Scope trigger threshold")
        self._plot_item.addItem(self._threshold_line)
        self.ui.labelTriggerLevel.setStyleSheet(
            f"color: {self._THRESHOLD_MARKER_COLOR};"
        )

    def _setup_pretrigger_line(self) -> None:
        self._pretrigger_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=True,
            pen=pg.mkPen(self._PRETRIGGER_MARKER_COLOR, width=1.5, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen("#7ea28a", width=2, style=Qt.PenStyle.DashLine),
            label="Pretrigger",
            labelOpts={"color": self._PRETRIGGER_MARKER_COLOR, "position": 0.98},
        )
        self._pretrigger_line.setZValue(10)
        self._pretrigger_line.setToolTip("Drag to shift the Scope pretrigger position")
        self._plot_item.addItem(self._pretrigger_line)
        self.ui.labelPretrigger.setStyleSheet(
            f"color: {self._PRETRIGGER_MARKER_COLOR};"
        )

    def _update_axis_ranges(self) -> None:
        frame_samples = self._frame_samples_from_ui()
        duration_ns = self.ui.spinFrameSamples.value()
        self._time_scale = time_axis_scale(duration_ns)
        self._sync_pretrigger_line_from_widget()
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
        self._update_waveform_drag_enabled()

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
        filepath = unique_recording_path(f"scope_ch{self._channel}", ".bin")
        log.info("Scope DMA: auto-generated filepath: %s", filepath)
        return filepath

    def _on_dma_file(self) -> None:
        default_dir = str(measurement_folder())
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Scope DMA File",
            default_dir,
            "Binary files (*.bin);;All files (*)",
        )
        if path:
            self._dma_filepath = Path(path)
            log.info("Scope DMA: user selected filepath: %s", self._dma_filepath)

    @Slot(object)
    def _on_dma_progress(self, bytes_written: int) -> None:
        if self._dma_stopping or self._dma_thread is None:
            return
        if self._dma_online_current:
            self.status.setText(
                f"Streaming to Current: {bytes_written / 1024**2:.1f} MiB received"
            )
            return
        # Both scope formats prepend the same 24-byte NDMA header. The
        # streamer progress value counts payload only, so add the header to
        # make the displayed number equal the actual file size.
        file_bytes = bytes_written + FILE_HEADER_STRUCT.size
        suffix = "; viewer paused" if self._frame_exceeds_viewer_limit() else ""
        if file_bytes < 1024**2:
            self.status.setText(
                f"Recording: {file_bytes / 1024:.1f} KiB{suffix}"
            )
        elif file_bytes < 1024**3:
            self.status.setText(
                f"Recording: {file_bytes / 1024**2:.1f} MiB{suffix}"
            )
        else:
            self.status.setText(
                f"Recording: {file_bytes / 1024**3:.2f} GiB{suffix}"
            )

    @Slot(str)
    def _on_dma_error(self, message: str) -> None:
        log.error("Scope DMA error: %s", message)
        self._dma_last_error = message
        self.status.setText(f"Error: {message}")
        if (series := owner_series(self)) is not None:
            series.cancel()

    @Slot()
    def _on_dma_finished(self) -> None:
        thread = self._dma_thread
        if thread is None:
            return  # synchronous shutdown already reaped this worker
        if not thread.wait(2000):
            log.error("Scope ch%d: DMA thread emitted finished but did not exit", self._channel)
            QTimer.singleShot(100, self._on_dma_finished)
            return

        # Keep both wrappers alive through GUI restoration. The thread's
        # finished signal can be queued alongside earlier worker callbacks.
        worker = self._dma_worker
        online_current = self._dma_online_current
        if online_current and self._current_accumulator is not None:
            snapshot = self._current_accumulator.snapshot(now_ns=time.perf_counter_ns())
            runtime = snapshot.runtime
            log.info(
                "Scope ch%d current DMA summary: received=%d (%d bytes), "
                "analyzed=%d, rejected=%d, discarded=%d, protocol_errors=%d, "
                "rate=%.3f MB/s/%.1f fps, skipped=%d, off_grid=%d, "
                "preview_replaced=%d, display_updates=%d, display_replaced=%d, "
                "analysis_lag=%.3f ms (max %.3f ms), display_p95=%.3f ms, "
                "transport=%s, buffers=%d, READBUF=%d, queue=%d/high=%d",
                self._channel,
                snapshot.received_frames,
                snapshot.received_bytes,
                snapshot.analyzed_frames,
                snapshot.rejected_frames,
                snapshot.discarded_analysis_frames,
                snapshot.protocol_errors,
                snapshot.payload_mb_s,
                snapshot.received_fps,
                snapshot.skipped_opportunities,
                snapshot.off_grid_intervals,
                snapshot.replaced_preview_frames,
                snapshot.display_updates,
                snapshot.replaced_display_generations,
                snapshot.analysis_lag_ns / 1_000_000,
                snapshot.maximum_analysis_lag_ns / 1_000_000,
                snapshot.display_p95_duration_ns / 1_000_000,
                runtime.transport if runtime is not None else "unknown",
                runtime.kernel_buffers if runtime is not None else 0,
                runtime.readbuf_batch_frames if runtime is not None else 0,
                runtime.queued_blocks if runtime is not None else 0,
                runtime.queue_high_watermark if runtime is not None else 0,
            )
        log.info("Scope DMA: worker finished; restoring controls")
        ScopeController._set_measurement_active(self, False)
        series_error = self._dma_last_error
        try:
            if getattr(self, "_session_replay_active", False):
                self._apply_session_replay_controls()
                return
            self._set_controls_enabled(True)
            self.ui.btnStart.setChecked(False)
            self.ui.btnStart.setEnabled(True)
            self.ui.btnStop.setEnabled(False)
            self.ui.btnAcquireFrame.setEnabled(True)
            self.ui.cbDmaEnable.setEnabled(True)
            self.ui.btnDmaFile.setEnabled(True)
            self.ui.btnAutoSetup.setEnabled(self._scope.dac_available())

            if self._scope.dma_fault_is_latched():
                series_error = series_error or "DMA fault"
                # A genuine EIO blocks rearm until the passive recovery check.
                try:
                    self._scope.acknowledge_dma_recovery()
                except RuntimeError:
                    log.error(
                        "Scope ch%d: DMA fault did not clear -- board restart likely required",
                        self._channel,
                        exc_info=True,
                    )
                    self.status.setText(
                        "DMA fault -- restart the board, then reconnect"
                    )
                    return
                log.info(
                    "Scope ch%d: DMA fault cleared, DMA capture is usable again",
                    self._channel,
                )
                self.status.setText("Stopped (recovered from DMA fault)")
                return

            self.status.setText(
                f"Error: {self._dma_last_error}"
                if self._dma_last_error is not None
                else "Stopped"
            )
        except Exception:
            log.exception("Scope ch%d: failed to restore GUI after DMA stop", self._channel)
            self.status.setText("DMA stopped; GUI refresh failed")
        finally:
            self._dma_worker = None
            self._dma_thread = None
            self._dma_stopping = False
            self._dma_online_current = False
            if online_current:
                # IIO buffer teardown owns and has already released the DMA
                # gate. Clear the online-only selection so Auto Setup is not
                # blocked by stale GUI state after Current reports stopped.
                self._set_dma_checkbox_without_write(False)
            thread.deleteLater()
            del worker
            if online_current:
                self.current_dma_state_changed.emit(
                    False,
                    self._dma_last_error or "Scope DMA stopped.",
                )
            elif (series := owner_series(self)) is not None:
                # The buffer and reader thread are released: the next series
                # run, if any, re-arms from a closed buffer.
                series.run_ended(series_error)

    def stop_dma_sync(self) -> None:
        """Blocking stop for shutdown, reconnect, or isolated session replay.

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
        if (series := owner_series(self)) is not None:
            series.cancel()
        self.stop_current_sum_monitor(wait=True)
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
                thread.deleteLater()

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
                ScopeController._set_measurement_active(self, False)
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
