from __future__ import annotations

import logging
import math
import statistics
import time
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QLocale, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from nlab.controllers.measurement_series import (
    MeasurementSeries,
    owner_series,
    time_limit_required_message,
)
from nlab.hardware.digitizer.current_monitor import (
    ScopeCurrentAccumulator,
    ScopeCurrentSnapshot,
)
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.hardware.digitizer.scope import (
    PARAMETER_SPECS,
    SCOPE_ADC_SAMPLE_PERIOD_NS,
    SCOPE_DATAPATH_CLOCK_PERIOD_NS,
    ScopeParam,
    range_spec,
)
from nlab.ui.ui_current_monitor_view import Ui_CurrentMonitorView
from nlab.utils.recording_paths import (
    available_path,
    chosen_recording_path,
    measurement_folder,
    unique_recording_path,
)
from nlab.utils.sample_history import SampleHistory
from nlab.views import theme
from nlab.views.numeric_controls import (
    bind_numeric_control,
    nearest_step_value,
    normalize_numeric_control,
)
from nlab.views.panel_status import PanelStatus
from nlab.views.plot_viewbox import show_frame_axes, waveform_step_mode
from nlab.views.spec_binding import bind_spinbox
from nlab.views.time_axis import time_axis_scale
from nlab.workers.current_monitor_worker import (
    CurrentMonitorWorker,
    CurrentSample,
    CurrentSampleBuffer,
)
from nlab.workers.scope_sum_worker import ScopeSumWorker
from nlab.workers.worker_thread import create_worker_thread, join_worker_thread

if TYPE_CHECKING:
    from nlab.controllers.scope_controller import ScopeController

log = logging.getLogger(__name__)


class CurrentAcquisition(Enum):
    """The one live acquisition this Current view owns (at most one at a time)."""

    IDLE = auto()
    IIR = auto()  # input-filter polling on this view's worker thread
    SCOPE_DMA = auto()  # periodic Scope DMA owned by the Scope controller
    SUMS = auto()  # periodic sums in a Scope sum reader process

_DISPLAY_HZ = 30
_DISPLAY_PERIOD_NS = round(1_000_000_000 / _DISPLAY_HZ)
_TARGET_SAMPLE_HZ = 1000
_HISTORY_NS = 10_000_000_000
_RATE_WINDOW_NS = 1_000_000_000
_ZERO_WINDOW_NS = 1_000_000_000
_MIN_STALE_NS = 50_000_000
_MODE_IIR = 0
_MODE_SCOPE_DMA = 1
_MODE_PERIODIC_SUM = 2
_MODE_NAMES = ("iir", "scope_dma", "periodic_sum")
_DMA_GAP_CYCLE_NS = SCOPE_DATAPATH_CLOCK_PERIOD_NS
_DMA_GAP_MAX_CYCLES = 65_535
_DEFAULT_DMA_GAP_CYCLES = 12_500
_DMA_PLOT_HZ = 2
_DMA_PLOT_PERIOD_NS = round(1_000_000_000 / _DMA_PLOT_HZ)
# comboSumFormat order; the extension selects the recorder (scope_sums.open_sum_writer).
_SUM_FORMATS = ("binary", "hdf5")
_SUM_FORMAT_SUFFIX = {"binary": ".bin", "hdf5": ".h5"}
_SUFFIX_SUM_FORMAT = {".bin": "binary", ".h5": "hdf5", ".hdf5": "hdf5"}
_SUM_FILE_FILTERS = {
    "binary": "Binary sum capture (*.bin)",
    "hdf5": "HDF5 sum capture (*.h5 *.hdf5)",
}


class CurrentMonitorController(QWidget):
    """Current monitoring from IIR, Scope waveforms, or structured periodic sums."""

    session_samples = Signal(int, object)
    # True while a live acquisition wants MCA/HV/temperature polling paused.
    readback_pause_changed = Signal(bool)

    def __init__(
        self,
        mca: MultiChannelAnalyzer,
        channel: int,
        parent: QWidget | None = None,
        *,
        auto_start: bool = False,
        scope_controller: ScopeController | None = None,
        scope_current_accumulator: ScopeCurrentAccumulator | None = None,
    ) -> None:
        super().__init__(parent)
        self._mca = mca
        self._channel = channel
        self._scope_controller = scope_controller
        self._scope_current_accumulator = scope_current_accumulator
        self._acquisition = CurrentAcquisition.IDLE
        self._sum_worker: ScopeSumWorker | None = None
        self._last_mode_index: int | None = None
        self._sample_buffer = CurrentSampleBuffer()
        self._worker: CurrentMonitorWorker | None = None
        self._worker_thread: QThread | None = None
        self._stop_requested = False
        self._latest: CurrentSample | None = None
        self._history = SampleHistory()
        self._last_batch: tuple[CurrentSample, ...] = ()
        self._dropped_samples = 0
        self._monitor_error: str | None = None
        self._display_hz = _DISPLAY_HZ
        self._display_period_ns = _DISPLAY_PERIOD_NS
        self._last_dma_render_generation = -1
        self._last_sum_render_key: tuple[int, int, bool] | None = None
        self._last_dma_plot_points = 0
        self._last_dma_plot_update_ns = 0
        self._calibration_dirty = True
        self._replay_hardware_interlocked = False
        self._session_replay_active = False

        self.ui = Ui_CurrentMonitorView()
        self.ui.setupUi(self)
        if self._sum_supported():
            self.ui.comboMode.addItem("Periodic sums")
        self.status = PanelStatus(self, "Monitor stopped.")
        # Host-clock duration of one run, for every acquisition mode.
        self._measurement_timer = QTimer(self)
        self._measurement_timer.setSingleShot(True)
        self._measurement_timer.timeout.connect(self._on_measurement_timeout)
        self._series = MeasurementSeries(
            self,
            spinbox=self.ui.spinSeriesRuns,
            label=self.ui.labelSeriesRuns,
            name=f"Current ch{channel}",
            is_idle=self._series_idle,
            start_run=self._start_series_run,
        )
        bind_numeric_control(self.ui.spinDmaGap, snap_to_step=True)
        self.ui.spinDmaGap.setLocale(QLocale.c())
        bind_spinbox(self.ui.spinSumSamples, range_spec(PARAMETER_SPECS, ScopeParam.FRAME_SAMPLES))
        self.ui.sumTimingControls.setProperty("nlabHelpTopic", "scope-sums")
        self.ui.sumControls.setProperty("nlabHelpTopic", "scope-sums")
        self._setup_plot()
        self._connect_signals()
        self._update_sum_timing()
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

    def _set_status(self, text: str, details: str = "") -> None:
        """Short status-bar message; diagnostic detail goes in its tooltip."""
        self.status.setText(text)
        self.status.setToolTip(details)

    def _sum_format(self) -> str:
        return _SUM_FORMATS[max(0, self.ui.comboSumFormat.currentIndex())]

    def _select_sum_format(self, suffix: str) -> None:
        file_format = _SUFFIX_SUM_FORMAT.get(suffix.lower())
        if file_format is not None:
            self.ui.comboSumFormat.setCurrentIndex(_SUM_FORMATS.index(file_format))

    @Slot()
    def _on_sum_file_edited(self) -> None:
        self._select_sum_format(Path(self.ui.editSumFile.text().strip()).suffix)

    @Slot(int)
    def _on_sum_format_changed(self, _index: int) -> None:
        # Keep a typed name's extension in step with the selected format.
        text = self.ui.editSumFile.text().strip()
        typed = _SUFFIX_SUM_FORMAT.get(Path(text).suffix.lower())
        if text and typed is not None and typed != self._sum_format():
            wanted = _SUM_FORMAT_SUFFIX[self._sum_format()]
            self.ui.editSumFile.setText(str(Path(text).with_suffix(wanted)))

    def _setup_plot(self) -> None:
        self.ui.plotCurrent.setBackground(theme.colors().plot_background)
        plot = self.ui.plotCurrent.getPlotItem()
        show_frame_axes(plot)
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setLabel("bottom", "Time before newest sample", units="s")
        plot.setLabel("left", "IIR output", units="raw")
        self._plot = plot
        for side in ("bottom", "top"):
            plot.getAxis(side).enableAutoSIPrefix(False)
        plot.getViewBox().sigXRangeChanged.connect(self._update_time_axis)
        self._update_time_axis()
        self._curve = plot.plot(
            pen=pg.mkPen("#277da1", width=1.5),
            autoDownsample=True,
            downsampleMethod="peak",
            clipToView=True,
        )

    def _update_time_axis(self, *_args: object) -> None:
        """Keep seconds as plot coordinates; select display units by zoom span."""
        low, high = self._plot.viewRange()[0]
        scale = time_axis_scale((high - low) * 1_000_000_000)
        bottom = self._plot.getAxis("bottom")
        if bottom.labelUnits != scale.unit:
            bottom.setLabel("Time before newest sample", units=scale.unit)
        for side in ("bottom", "top"):
            self._plot.getAxis(side).setScale(1_000_000_000 / scale.ns_per_unit)

    def _connect_signals(self) -> None:
        self.ui.btnStart.clicked.connect(self._on_start_clicked)
        self.ui.btnStop.clicked.connect(self.request_monitor_stop)
        self.ui.btnResetScopeDefaults.clicked.connect(self._reset_scope_defaults)
        self.ui.btnZero.clicked.connect(self._set_zero_from_recent)
        self.ui.spinDmaGap.editingFinished.connect(self._normalize_dma_gap)
        self.ui.spinDmaGap.valueChanged.connect(self._update_sum_timing)
        self.ui.spinSumSamples.valueChanged.connect(self._update_sum_timing)
        self.ui.spinZero.valueChanged.connect(self._calibration_changed)
        self.ui.spinScale.valueChanged.connect(self._calibration_changed)
        self.ui.comboUnit.currentTextChanged.connect(self._calibration_changed)
        self.ui.comboMode.currentIndexChanged.connect(self._on_mode_changed)
        self.ui.btnSumFile.clicked.connect(self._choose_sum_output)
        self.ui.editSumFile.editingFinished.connect(self._on_sum_file_edited)
        self.ui.comboSumFormat.currentIndexChanged.connect(self._on_sum_format_changed)
        self.ui.cbPauseReadbacks.toggled.connect(
            lambda _checked: self._sync_readback_pause(
                getattr(self, "_controls_running", False)
            )
        )

    def _is_running(self) -> bool:
        return (
            self._acquisition is not CurrentAcquisition.IDLE
        )

    def _sum_supported(self) -> bool:
        return (
            self._scope_controller is not None
            and self._scope_controller.current_sum_supported()
        )

    def _dma_supported(self) -> bool:
        return (
            self._scope_controller is not None
            and self._scope_current_accumulator is not None
            and self._scope_controller.current_dma_supported()
        )

    def _sync_readback_pause(self, running: bool) -> None:
        """Request a board readback pause while a live acquisition runs (opt-in).

        MCA, HV and temperature polling share the board's iiod with Current.
        Live, MCA polling alone cut periodic-sum coverage (256 samples, gap 0)
        from ~99% to ~92%.
        """
        self._controls_running = running
        wanted = (
            running
            and self.ui.cbPauseReadbacks.isChecked()
            and not self._replay_hardware_interlocked
        )
        if wanted != getattr(self, "_readback_pause_requested", False):
            self._readback_pause_requested = wanted
            self.readback_pause_changed.emit(wanted)

    def _set_running_controls(self, running: bool) -> None:
        self._sync_readback_pause(running)
        sum_mode = self.ui.comboMode.currentIndex() == _MODE_PERIODIC_SUM
        record_enabled = (
            sum_mode and not running and self._sum_supported()
            and not self._replay_hardware_interlocked
        )
        for record_widget in (
            self.ui.cbRecordSums, self.ui.comboSumFormat, self.ui.editSumFile, self.ui.btnSumFile,
        ):
            record_widget.setEnabled(record_enabled)
        self.ui.spinSumSamples.setEnabled(record_enabled)
        self.ui.btnStop.setText(
            "Stop and drain" if self._acquisition is CurrentAcquisition.SUMS else "Stop"
        )
        if self._replay_hardware_interlocked:
            for widget in (
                self.ui.btnStart,
                self.ui.btnStop,
                self.ui.comboMode,
                self.ui.spinDmaGap,
                self.ui.btnResetScopeDefaults,
                self.ui.spinTime,
            ):
                widget.setEnabled(False)
            self.ui.labelDmaGap.setEnabled(False)
            self.ui.btnStop.setEnabled(False)
            return
        self.ui.btnStart.setEnabled(not running and (not sum_mode or self._sum_supported()))
        self.ui.btnStop.setEnabled(running)
        self.ui.comboMode.setEnabled(not running)
        self.ui.spinTime.setEnabled(not running)
        dma_gap_enabled = (
            not running
            and (
                (self.ui.comboMode.currentIndex() == _MODE_SCOPE_DMA and self._dma_supported())
                or (
                    self.ui.comboMode.currentIndex() == _MODE_PERIODIC_SUM
                    and self._sum_supported()
                )
            )
        )
        self.ui.labelDmaGap.setEnabled(dma_gap_enabled)
        self.ui.spinDmaGap.setEnabled(dma_gap_enabled)
        self.ui.btnResetScopeDefaults.setEnabled(
            dma_gap_enabled and self.ui.comboMode.currentIndex() == _MODE_SCOPE_DMA
        )

    @Slot(int)
    def _on_mode_changed(self, index: int) -> None:
        if index != self._last_mode_index and not self._is_running():
            self._sum_worker = None
            self._reset_measurement_state()
        self._last_mode_index = index
        dma_mode = index == _MODE_SCOPE_DMA
        sum_mode = index == _MODE_PERIODIC_SUM
        self.ui.btnStart.setText("Start")
        self.ui.sumControls.setVisible(sum_mode)
        self.ui.sumTimingControls.setVisible(sum_mode or dma_mode)
        for widget in (self.ui.labelSumSamples, self.ui.spinSumSamples, self.ui.labelSumTiming):
            widget.setVisible(sum_mode)
        self.ui.labelDmaGap.setText("Window gap:" if sum_mode else "DMA gap:")
        self.ui.spinDmaGap.setToolTip(
            "Gap between sum windows; 0 to 524.280 us in 0.008 us (8 ns) steps. "
            "Editable while stopped; applied on Start."
            if sum_mode else
            "Post-frame gap for Scope DMA; 0.008 us (8 ns) hardware resolution."
        )
        self.ui.groupReading.setTitle("Window means" if sum_mode else "Live current")
        # Sums are ADC codes; the IIR/waveform calibration never applies to them.
        self.ui.groupCalibration.setVisible(not sum_mode)
        self.ui.btnZero.setEnabled(not sum_mode and self._latest is not None)
        self.ui.labelRaw.setText(
            "Window mean:" if sum_mode else "Raw DMA sample:" if dma_mode else "Raw IIR code:"
        )
        self.ui.labelInterval.setText(
            "Session mean:" if sum_mode else (
                "Latest DMA fragment:" if dma_mode else "Last screen interval:"
            )
        )
        base_label = "DMA waveform" if dma_mode else "IIR output"
        label = "Current" if self._unit() != "raw" else base_label
        self._plot.setLabel(
            "left", "Window mean" if sum_mode else label,
            units="ADC codes" if sum_mode else self._unit(),
        )
        if self._is_running():
            return
        self._set_running_controls(False)
        if sum_mode:
            self._set_status(
                "Periodic sums stopped.",
                "Start plots measured window means in ADC codes, without the "
                "IIR/waveform calibration. Window size and gap apply on Start; "
                "recording is optional.",
            )
        elif dma_mode and not self._dma_supported():
            self.status.setText(
                "Scope DMA mode is unavailable on this backend/firmware."
            )
        else:
            self.status.setText("Monitor stopped.")

    @Slot()
    def _reset_scope_defaults(self) -> None:
        if self._replay_hardware_interlocked:
            return
        scope_controller = self._scope_controller
        if self._is_running() or scope_controller is None:
            return
        try:
            scope_controller.reset_defaults()
        except Exception as exc:
            self.status.setText(f"Could not reset Scope defaults: {exc}")
            log.exception("Resetting Scope defaults failed")
            return
        self._set_dma_gap_cycles(_DEFAULT_DMA_GAP_CYCLES)
        self._set_status(
            "Scope defaults restored.", "Current DMA optimization is inactive."
        )

    def _sum_frame_samples(self) -> int:
        spinbox = self.ui.spinSumSamples
        return int(nearest_step_value(
            spinbox.value(), spinbox.minimum(), spinbox.maximum(), spinbox.singleStep()
        ))

    @Slot()
    def _update_sum_timing(self) -> None:
        duration_ns = self._sum_frame_samples() * SCOPE_ADC_SAMPLE_PERIOD_NS
        interval_ns = duration_ns + self._dma_gap_cycles() * _DMA_GAP_CYCLE_NS
        self.ui.labelSumTiming.setText(
            f"= {duration_ns / 1000:.3f} µs every {interval_ns / 1000:.3f} µs"
        )

    def _dma_gap_cycles(self) -> int:
        gap_us = nearest_step_value(
            self.ui.spinDmaGap.value(),
            0,
            _DMA_GAP_MAX_CYCLES * _DMA_GAP_CYCLE_NS / 1000,
            _DMA_GAP_CYCLE_NS / 1000,
        )
        return round(gap_us * 1000 / _DMA_GAP_CYCLE_NS)

    def _set_dma_gap_cycles(self, cycles: int) -> None:
        if not 0 <= cycles <= _DMA_GAP_MAX_CYCLES:
            raise ValueError("DMA gap cycles must be 0..65535")
        blocked = self.ui.spinDmaGap.blockSignals(True)
        try:
            self.ui.spinDmaGap.setValue(cycles * _DMA_GAP_CYCLE_NS / 1000)
        finally:
            self.ui.spinDmaGap.blockSignals(blocked)
        self._update_sum_timing()

    @Slot()
    def _normalize_dma_gap(self) -> None:
        self._set_dma_gap_cycles(self._dma_gap_cycles())

    # ------------------------------------------------------------------
    # Time limit and measurement series
    # ------------------------------------------------------------------

    def set_measurement_series_enabled(self, enabled: bool) -> None:
        """Apply Settings > Enable Series to this view."""
        self._series.set_enabled(enabled)

    @Slot()
    def _on_start_clicked(self) -> None:
        """Operator Start: begins a series when Settings > Enable Series is on."""
        series = self._series
        if not series.begin(self.ui.spinTime.value()):
            self.status.setText(time_limit_required_message(series.runs()))
            return
        try:
            self.start_monitor()
        finally:
            if series.active and self._series_idle():
                series.cancel()  # Start was declined and already reported why.

    def _series_idle(self) -> bool:
        """No acquisition, and the IIR worker thread (if any) has exited.

        Scope DMA and periodic sums report their end only after the Scope
        controller has released the buffer and reader.
        """
        return not self._is_running() and self._worker_thread is None

    def _start_series_run(self) -> None:
        if self.ui.btnStart.isEnabled():
            self.start_monitor()

    def _start_measurement_timer(self) -> None:
        time_s = self.ui.spinTime.value()
        if time_s > 0:
            self._measurement_timer.start(time_s * 1000)
            log.info("Current ch%d: measurement timer set to %d s", self._channel, time_s)

    @Slot()
    def _on_measurement_timeout(self) -> None:
        log.info("Current ch%d: measurement time limit reached", self._channel)
        if (series := owner_series(self)) is not None:
            series.mark_completed()
        self.request_monitor_stop()

    def _end_series_run(self, error: str | None) -> None:
        """The acquisition released its resources; continue or end the series."""
        self._measurement_timer.stop()
        if (series := owner_series(self)) is not None:
            series.run_ended(error)

    def _schedule_display(self) -> None:
        if not self._display_timer.isActive():
            delay_ns = max(0, self._next_display_ns - time.perf_counter_ns())
            self._display_timer.start(max(0, round(delay_ns / 1_000_000)))

    @Slot()
    def start_monitor(self) -> None:
        if self._replay_hardware_interlocked:
            return
        if self._is_running():
            return
        if self.ui.comboMode.currentIndex() == _MODE_PERIODIC_SUM:
            self._start_sum_monitor()
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
        thread = create_worker_thread(self, worker, on_finished=self._on_worker_finished)
        self._worker = worker
        self._worker_thread = thread
        self._acquisition = CurrentAcquisition.IIR
        self._stop_requested = False
        self._monitor_error = None
        self._last_dma_render_generation = -1
        self._calibration_dirty = True
        worker.error.connect(self._on_worker_error)
        self._set_running_controls(True)
        self.status.setText("Opening an isolated current-monitor connection…")
        self._next_display_ns = time.perf_counter_ns() + self._display_period_ns
        self._schedule_display()
        thread.start()
        self._start_measurement_timer()

    def set_replay_hardware_interlock(
        self, enabled: bool, *, resume_background: bool = True,
    ) -> None:
        """Stop live Current acquisition and block controls that touch Scope/IIO."""
        enabled = bool(enabled)
        if enabled == self._replay_hardware_interlocked:
            return
        self._replay_hardware_interlocked = enabled
        if enabled and (series := owner_series(self)) is not None:
            series.cancel()
        if enabled and self._is_running():
            self.stop_monitor_sync()
        self._set_running_controls(self._is_running())

    def begin_session_replay(
        self,
        *,
        settings: object = None,
        dropped_samples: int = 0,
    ) -> None:
        """Prepare the current view for transport-free recorded IIR samples."""
        self._display_timer.stop()
        self._session_replay_active = True
        self._reset_measurement_state()
        self._dropped_samples = max(0, int(dropped_samples))
        self.ui.comboMode.blockSignals(True)
        self.ui.comboMode.setCurrentIndex(_MODE_IIR)
        self.ui.comboMode.blockSignals(False)
        self._on_mode_changed(_MODE_IIR)
        if isinstance(settings, dict):
            if "zero_code" in settings:
                self.ui.spinZero.setValue(float(settings["zero_code"]))
            if "scale_per_code" in settings:
                self.ui.spinScale.setValue(float(settings["scale_per_code"]))
            if "unit" in settings:
                index = self.ui.comboUnit.findText(str(settings["unit"]))
                if index >= 0:
                    self.ui.comboUnit.setCurrentIndex(index)
        self.status.setText("Session current replay ready.")

    def display_session_replay_samples(
        self,
        samples: tuple[CurrentSample, ...],
    ) -> None:
        """Render recorded samples without feeding the live recorder signal."""
        if not samples:
            return
        self._last_batch = samples
        self._latest = samples[-1]
        self._history.extend(samples)
        self._history.trim_before(self._latest.timestamp_ns - _HISTORY_NS)
        self.ui.btnZero.setEnabled(True)
        self._render_values(self._latest.timestamp_ns)
        status = "Recorded FPGA IIR samples; replay does not access hardware."
        if self._dropped_samples:
            status += f" Capture dropped {self._dropped_samples:,} sample(s)."
        self.status.setText(status)

    def finish_session_replay(self) -> None:
        if self._session_replay_active:
            self.status.setText("Session current replay complete; final view held.")

    def end_session_replay(self) -> None:
        self._session_replay_active = False

    def _reset_measurement_state(self) -> None:
        self._latest = None
        self._history.clear()
        self._last_batch = ()
        self._dropped_samples = 0
        self._monitor_error = None
        self._last_dma_plot_points = 0
        self._last_dma_plot_update_ns = 0
        self.ui.btnZero.setEnabled(False)
        self._curve.setData([], [])
        for label in (
            self.ui.lblCurrent, self.ui.lblRaw, self.ui.lblInterval, self.ui.lblAcquisition
        ):
            label.setText("-")

    def _start_sum_monitor(self) -> None:
        scope_controller = self._scope_controller
        if scope_controller is None or not self._sum_supported():
            self.status.setText("Periodic sums are unavailable on this backend/firmware.")
            return
        series = owner_series(self)
        numbered = (
            series.recording_path(self._sum_output_path)
            if series is not None and self.ui.cbRecordSums.isChecked()
            else None
        )
        if numbered is not None:
            # Series recordings are numbered and never replace a file.
            output: Path | None = available_path(numbered)
        else:
            output = self._sum_output_path() if self.ui.cbRecordSums.isChecked() else None
        overwrite = False
        if numbered is None and output is not None and output.is_file():
            answer = QMessageBox.question(
                self,
                "Overwrite sum recording?",
                f'"{output}" already exists.\n\nOverwrite it with a new recording?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        # The confirmation runs a nested event loop; replay or another acquisition
        # can begin while it is open. Scope also rechecks its ownership below.
        if self._replay_hardware_interlocked or self._is_running():
            return
        self._reset_measurement_state()
        self._normalize_dma_gap()
        normalize_numeric_control(self.ui.spinSumSamples)
        self._last_sum_render_key = None
        try:
            self._sum_worker = scope_controller.start_current_sum_monitor(
                frame_samples=self._sum_frame_samples(),
                gap_cycles=self._dma_gap_cycles(), output=output, overwrite=overwrite
            )
        except Exception as exc:
            self._monitor_error = f"Could not start periodic sums: {exc}"
            self.status.setText(self._monitor_error)
            log.exception("Starting Current periodic sums failed")
            return
        self._acquisition = CurrentAcquisition.SUMS
        self._stop_requested = False
        self._set_running_controls(True)
        self._set_status(
            "Acquiring periodic sums…",
            "Stop and drain keeps the windows the board already accepted.",
        )
        self._next_display_ns = time.perf_counter_ns() + self._display_period_ns
        self._schedule_display()
        self._start_measurement_timer()

    def _sum_output_path(self) -> Path:
        """Resolve explicit names or generate a fresh name in the measurement folder.

        A .bin, .h5 or .hdf5 extension selects that format; any other name
        gains the extension of the format selected in the GUI. An explicit name
        gets the start timestamp when General Settings asks for it.
        """
        suffix = _SUM_FORMAT_SUFFIX[self._sum_format()]
        filename = self.ui.editSumFile.text().strip()
        if filename:
            output = Path(filename).expanduser()
            if output.suffix.lower() not in _SUFFIX_SUM_FORMAT:
                output = output.with_name(output.name + suffix)
            if not output.is_absolute():
                output = measurement_folder() / output
            return chosen_recording_path(output)
        return unique_recording_path(f"ch{self._channel}_current", suffix, create_folder=False)

    def _choose_sum_output(self) -> None:
        folder = str(measurement_folder())
        selected = self._sum_format()
        filters = [_SUM_FILE_FILTERS[selected]] + [
            text for key, text in _SUM_FILE_FILTERS.items() if key != selected
        ]
        filename, chosen_filter = QFileDialog.getSaveFileName(
            self, "Sum recording", folder, ";;".join(filters),
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if filename:
            output = Path(filename)
            if output.suffix.lower() not in _SUFFIX_SUM_FORMAT:
                chosen = next(
                    (key for key, text in _SUM_FILE_FILTERS.items() if text == chosen_filter),
                    selected,
                )
                output = output.with_name(output.name + _SUM_FORMAT_SUFFIX[chosen])
            self.ui.editSumFile.setText(str(output))
            self._select_sum_format(output.suffix)
            self.ui.cbRecordSums.setChecked(True)

    def _finish_sum_monitor(self) -> None:
        worker = self._sum_worker
        if worker is None or not worker.done:
            return
        if self._acquisition is CurrentAcquisition.SUMS and self._scope_controller is not None:
            self._scope_controller.finish_current_sum_monitor()
        worker.join()
        self._display_timer.stop()
        self._acquisition = CurrentAcquisition.IDLE
        self._stop_requested = False
        self._set_running_controls(False)
        self.status.setText(worker.message)
        self._end_series_run(worker.error or None)

    def _render_sum_snapshot(self) -> None:
        worker = self._sum_worker
        if worker is None:
            return
        self.ui.groupReading.setTitle(
            "Recorded window means" if worker.capture is None else "Window means"
        )
        snapshot = worker.analysis.snapshot()
        # The reader process publishes at most ten snapshots a second; the
        # display timer runs faster. Repainting an unchanged snapshot costs
        # GUI CPU that, on a laptop, competes with that same reader process.
        render_key = (id(worker), snapshot["packets"], worker.done)
        if render_key == self._last_sum_render_key:
            return
        self._last_sum_render_key = render_key
        data, mean = snapshot["points"], snapshot["mean"]
        if data.size and mean is not None:
            connect = np.zeros(len(data), dtype=bool)
            # Adjacent accepted windows differ by one scheduled interval. Allow
            # float display rounding, but never bridge a missing native window.
            connect[:-1] = np.diff(data[:, 0]) < snapshot["interval_ns"] * 1.5e-9
            self._curve.setData(
                data[:, 0] - data[-1, 0], data[:, 1], stepMode=None, connect=connect
            )
            latest = float(data[-1, 1])
            self.ui.lblCurrent.setText(f"{self._format_value(latest)} ADC codes")
            self.ui.lblRaw.setText(self._format_value(latest))
            self.ui.lblInterval.setText(f"{self._format_value(mean)} ADC codes")
            self.ui.lblInterval.setToolTip("Sample-weighted mean of every window this session.")
            windows, missed = snapshot["windows"], snapshot["missed"]
            offered = windows + missed
            coverage = 100 * windows / offered if offered else 100.0
            self.ui.lblAcquisition.setText(
                f"{windows:,} windows · {coverage:.2f}% coverage"
            )
            self.ui.lblAcquisition.setToolTip(
                f"{snapshot['samples']:,} ADC samples; exposure "
                f"{snapshot['coverage_ns'] * 1e-9:.6g} s. Coverage counts the windows "
                "the board missed; at gap 0 it skips one while sealing each "
                "508-window packet (about 0.2% at 256 samples)."
            )
            if not worker.done:
                state = "Draining tail" if self._stop_requested else "Live sums"
                self._set_status(
                    f"{state} · missed {missed:,} · gaps {snapshot['skipped']:,}",
                    f"IP{snapshot.get('ip_version')}; "
                    f"N={snapshot.get('samples_per_window')} samples; "
                    f"gap={snapshot.get('gap_cycles')} cycles; "
                    f"interval={snapshot.get('interval_ns')} ns. Missed windows come "
                    "from the hardware counter, gaps from the timestamps. Values are "
                    "ADC codes; physical current needs a calibration.",
                )
        if worker.done:
            self._finish_sum_monitor()

    def _start_dma_monitor(self) -> None:
        scope_controller = self._scope_controller
        accumulator = self._scope_current_accumulator
        if not self._dma_supported() or scope_controller is None or accumulator is None:
            self._monitor_error = (
                "Scope DMA mode requires the direct IIO Scope backend with "
                "periodic-frame support."
            )
            self.status.setText(self._monitor_error)
            return

        self._stop_requested = False
        self._monitor_error = None
        gap_cycles = self._dma_gap_cycles()
        self._set_dma_gap_cycles(gap_cycles)
        try:
            scope_controller.start_current_dma_monitor(gap_cycles=gap_cycles)
        except Exception as exc:
            self._monitor_error = f"Could not start Scope DMA current monitor: {exc}"
            self.status.setText(self._monitor_error)
            log.exception("Starting Scope DMA current monitor failed")
            return

        self._acquisition = CurrentAcquisition.SCOPE_DMA
        self._set_running_controls(True)
        self.status.setText(
            "Starting periodic Scope DMA current monitoring "
            f"(gap {gap_cycles * _DMA_GAP_CYCLE_NS / 1000:.3f} us)..."
        )
        self._next_display_ns = time.perf_counter_ns() + self._display_period_ns
        self._schedule_display()

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        self._monitor_error = message

    def request_monitor_stop(self) -> None:
        if self._stop_requested:
            return
        self._measurement_timer.stop()
        if (series := owner_series(self)) is not None:
            series.stop_requested()
        self._stop_requested = True
        if self._acquisition is CurrentAcquisition.SUMS:
            if self._sum_worker is not None:
                self._sum_worker.request_stop()
            self.ui.btnStop.setEnabled(False)
            self.status.setText("Stopping; draining accepted windows and closing the transport...")
            return
        if self._acquisition is CurrentAcquisition.SCOPE_DMA:
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
        if (series := owner_series(self)) is not None:
            series.cancel()
        self._measurement_timer.stop()
        if self._acquisition is CurrentAcquisition.SUMS:
            worker = self._sum_worker
            if worker is not None:
                worker.request_stop()
                worker.join()
                self._render_sum_snapshot()
            return
        if self._acquisition is CurrentAcquisition.SCOPE_DMA:
            self._stop_requested = True
            if self._scope_controller is not None:
                self._scope_controller.stop_current_dma_monitor(wait=True)
            if self._acquisition is CurrentAcquisition.SCOPE_DMA:
                self._finish_dma_monitor("Monitor stopped.")
            return

        self.request_monitor_stop()
        # A remote attribute read is synchronous: never close its context from
        # another thread; wait for the bounded transport call to return.
        join_worker_thread(self._worker_thread, label="Current monitor", warn_after_ms=4000)
        self._worker = None
        self._worker_thread = None
        if self._acquisition is CurrentAcquisition.IIR:
            self._acquisition = CurrentAcquisition.IDLE
        self._stop_requested = False
        self._display_timer.stop()
        self._render_pending()
        self._set_running_controls(False)
        if self._monitor_error is None:
            self.status.setText("Monitor stopped.")

    @Slot()
    def _on_worker_finished(self, thread: QThread) -> None:
        if thread is not self._worker_thread:
            return  # an earlier monitor; a restarted one is running
        self._worker = None
        self._worker_thread = None
        if self._acquisition is CurrentAcquisition.IIR:
            self._acquisition = CurrentAcquisition.IDLE
        self._stop_requested = False
        self._display_timer.stop()
        self._render_pending()
        self._set_running_controls(False)
        if self._monitor_error is None:
            self.status.setText("Monitor stopped.")
        self._end_series_run(self._monitor_error)

    @Slot(bool, str)
    def _on_scope_dma_state_changed(self, running: bool, message: str) -> None:
        if self._acquisition is not CurrentAcquisition.SCOPE_DMA:
            return
        if running:
            self.status.setText(message)
            # Time the run from DMA streaming, not from arming.
            self._start_measurement_timer()
            return
        self._finish_dma_monitor(message)

    def _finish_dma_monitor(self, message: str) -> None:
        self._display_timer.stop()
        self._acquisition = CurrentAcquisition.IDLE
        self._render_pending()
        self._stop_requested = False
        self._set_running_controls(False)
        self.status.setText(message)
        scope_controller = self._scope_controller
        self._end_series_run(
            self._monitor_error
            or (scope_controller.last_dma_error if scope_controller is not None else None)
        )

    @staticmethod
    def _convert(raw_code: float, zero: float, scale: float) -> float:
        return (raw_code - zero) * scale

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
        if self.ui.comboMode.currentIndex() == _MODE_PERIODIC_SUM:
            return  # The sum view reports measured ADC codes, without calibration.
        self._calibration_dirty = True
        self._last_dma_plot_update_ns = 0
        raw_label = (
            "DMA waveform"
            if self.ui.comboMode.currentIndex() == _MODE_SCOPE_DMA
            else "IIR output"
        )
        label = "Current" if self._unit() != "raw" else raw_label
        self._plot.setLabel("left", label, units=self._unit())
        now_ns = time.perf_counter_ns()
        if self.ui.comboMode.currentIndex() == _MODE_SCOPE_DMA:
            accumulator = self._scope_current_accumulator
            if accumulator is not None:
                snapshot = accumulator.snapshot(now_ns=now_ns)
                self._render_dma_snapshot(snapshot)
                self._last_dma_render_generation = snapshot.generation
                self._calibration_dirty = False
        else:
            self._render_values(now_ns)

    @Slot()
    def _set_zero_from_recent(self) -> None:
        if self.ui.comboMode.currentIndex() == _MODE_SCOPE_DMA:
            accumulator = self._scope_current_accumulator
            if accumulator is None:
                return
            snapshot = accumulator.snapshot(now_ns=time.perf_counter_ns())
            if not snapshot.bins:
                return
            newest = snapshot.bins[-1]
            cutoff_index = newest.index - max(
                1, round(_ZERO_WINDOW_NS / snapshot.bin_width_ns)
            )
            recent = [item for item in snapshot.bins if item.index >= cutoff_index]
            sample_count = sum(item.sample_count for item in recent)
            if sample_count:
                self.ui.spinZero.setValue(
                    sum(item.raw_sum for item in recent) / sample_count
                )
            return
        latest = self._latest
        if latest is None:
            return
        history = self._history
        values = history.raw_codes[history.since(latest.timestamp_ns - _ZERO_WINDOW_NS)]
        if values.size:
            self.ui.spinZero.setValue(float(values.mean()))

    @Slot()
    def _render_pending(self) -> None:
        now_ns = time.perf_counter_ns()
        if self.ui.comboMode.currentIndex() == _MODE_PERIODIC_SUM:
            self._render_sum_snapshot()
        elif self.ui.comboMode.currentIndex() == _MODE_SCOPE_DMA:
            accumulator = self._scope_current_accumulator
            if accumulator is not None:
                snapshot = accumulator.snapshot(now_ns=now_ns)
                if (
                    snapshot.generation != self._last_dma_render_generation
                    or self._calibration_dirty
                    or self._monitor_error is not None
                ):
                    render_started_ns = time.perf_counter_ns()
                    self._render_dma_snapshot(snapshot)
                    accumulator.note_display_update(
                        snapshot.generation,
                        time.perf_counter_ns() - render_started_ns,
                    )
                    self._last_dma_render_generation = snapshot.generation
                    self._calibration_dirty = False
        else:
            batch = self._sample_buffer.drain()
            signal = getattr(self, "session_samples", None)
            if signal is not None:
                signal.emit(self._channel, batch)
            self._monitor_error = batch.error
            self._dropped_samples += batch.dropped_samples
            samples = batch.samples
            if samples:
                self._last_batch = samples
                self._latest = samples[-1]
                self._history.extend(samples)
                self._history.trim_before(self._latest.timestamp_ns - _HISTORY_NS)
                self.ui.btnZero.setEnabled(True)
            self._render_values(now_ns)

        if self._is_running():
            self._next_display_ns += self._display_period_ns
            if self._next_display_ns <= now_ns:
                self._next_display_ns = now_ns + self._display_period_ns
            self._schedule_display()

    def _render_dma_snapshot(self, snapshot: ScopeCurrentSnapshot) -> None:
        latest_raw_sample = snapshot.latest_raw_sample
        if latest_raw_sample is None or not snapshot.bins:
            if self._monitor_error is not None:
                self.status.setText(self._monitor_error)
            return

        zero = self.ui.spinZero.value()
        scale = self.ui.spinScale.value()
        unit = self._unit()
        current = self._convert(latest_raw_sample, zero, scale)
        self.ui.lblCurrent.setText(f"{self._format_value(current)} {unit}")
        self.ui.lblRaw.setText(self._format_value(latest_raw_sample))
        self.ui.btnZero.setEnabled(True)

        frame_minimum = snapshot.latest_frame_minimum
        frame_maximum = snapshot.latest_frame_maximum
        geometry = snapshot.geometry
        converted_minimum = (
            self._convert(frame_minimum, zero, scale)
            if frame_minimum is not None
            else math.nan
        )
        converted_maximum = (
            self._convert(frame_maximum, zero, scale)
            if frame_maximum is not None
            else math.nan
        )
        self.ui.lblInterval.setText(
            f"min {self._format_value(converted_minimum)}; "
            f"max {self._format_value(converted_maximum)} {unit}"
        )
        self.ui.lblInterval.setToolTip(
            f"Latest frame: last {self._format_value(current)} {unit}; "
            f"{geometry.waveform_samples if geometry is not None else 0:,} waveform samples."
        )

        runtime = snapshot.runtime
        window_us = geometry.capture_duration_ns / 1000 if geometry is not None else 0.0
        timing_text = (
            f"median spacing {snapshot.median_interval_ns / 1_000_000:.3f} ms"
        )
        if runtime is not None:
            gap_ms = (
                runtime.gap_cycles * SCOPE_DATAPATH_CLOCK_PERIOD_NS / 1_000_000
            )
            nominal_ms = (
                runtime.expected_interval_ticks
                * SCOPE_DATAPATH_CLOCK_PERIOD_NS
                / 1_000_000
            )
            timing_text = (
                f"gap {gap_ms:.3f} ms; nominal/median spacing "
                f"{nominal_ms:.3f}/{snapshot.median_interval_ns / 1_000_000:.3f} ms"
            )
        self.ui.lblAcquisition.setText(
            f"{snapshot.received_fps:,.1f} frames/s · "
            f"{snapshot.observed_coverage_percent:.2f}% coverage"
        )
        self.ui.lblAcquisition.setToolTip(
            f"{snapshot.payload_mb_s:.3f} MB/s; {timing_text}; "
            f"window {window_us:.3f} us; "
            f"age {snapshot.data_age_ns / 1_000_000:.1f} ms"
        )

        runtime_text = "transport pending"
        if runtime is not None:
            runtime_text = (
                f"{runtime.transport}, buffers={runtime.kernel_buffers}, "
                f"READBUF x{runtime.readbuf_batch_frames or 1}, "
                f"queue={runtime.queued_blocks}/high {runtime.queue_high_watermark}, "
                f"viewer={runtime.viewer_state}"
            )
        render_time_ns = time.perf_counter_ns()
        update_plot = (
            not self._last_dma_plot_update_ns
            or render_time_ns - self._last_dma_plot_update_ns >= _DMA_PLOT_PERIOD_NS
            or not snapshot.active
        )
        if update_plot:
            self._render_dma_plot(snapshot, zero, scale)
            self._last_dma_plot_update_ns = render_time_ns
        status = (
            f"Live DMA · {snapshot.analyzed_frames:,}/{snapshot.received_frames:,} frames · "
            f"skipped {snapshot.skipped_opportunities:,}"
        )
        details = (
            f"Analyzed {snapshot.analyzed_frames:,}/{snapshot.received_frames:,} "
            f"received frames; {runtime_text}; skipped opportunities "
            f"{snapshot.skipped_opportunities:,}; off-grid "
            f"{snapshot.off_grid_intervals:,}; inline analysis lag "
            f"{snapshot.analysis_lag_ns / 1_000_000:.3f} ms "
            f"(queue 0); update preparation p95 "
            f"{snapshot.display_p95_duration_ns / 1_000_000:.1f} ms; "
            f"plot {len(snapshot.waveform_records):,} raw fragments/"
            f"{self._last_dma_plot_points:,} samples; "
            f"preview/display replacements {snapshot.replaced_preview_frames:,}/"
            f"{snapshot.replaced_display_generations:,}; protocol errors "
            f"{snapshot.protocol_errors:,}."
        )
        if snapshot.rejected_frames or snapshot.discarded_analysis_frames:
            status += f" · rejected {snapshot.rejected_frames:,}"
            details += (
                f" Rejected {snapshot.rejected_frames:,}; analysis discarded "
                f"{snapshot.discarded_analysis_frames:,}."
            )
        if self._monitor_error is not None:
            status = self._monitor_error
        elif snapshot.data_age_ns > max(
            _MIN_STALE_NS, round(3 * snapshot.median_interval_ns)
        ):
            status = (
                f"Stale: newest DMA frame is "
                f"{snapshot.data_age_ns / 1_000_000:.1f} ms old. " + status
            )
        self._set_status(status, details)

    def _render_dma_plot(
        self,
        snapshot: ScopeCurrentSnapshot,
        zero: float,
        scale: float,
    ) -> None:
        """Plot bounded raw DMA fragments independently from the 30 Hz readout."""
        geometry = snapshot.geometry
        records = snapshot.waveform_records
        if geometry is None or not records:
            self._last_dma_plot_points = 0
            self._curve.setData([], [])
            return

        samples_per_frame = geometry.waveform_samples
        sample_points = len(records) * samples_per_frame
        array_points = sample_points + len(records) - 1
        x_values = np.empty(array_points, dtype=np.float64)
        y_values = np.empty(array_points, dtype=np.float64)
        sample_offsets_ns = (
            np.arange(samples_per_frame, dtype=np.int64) * geometry.sample_period_ns
        )
        newest_timestamp = int.from_bytes(records[-1][:8], "little")
        newest_sample_ns = (
            newest_timestamp * SCOPE_DATAPATH_CLOCK_PERIOD_NS
            + sample_offsets_ns[-1]
        )
        cursor = 0
        for index, record in enumerate(records):
            timestamp = int.from_bytes(record[:8], "little")
            stop = cursor + samples_per_frame
            x_values[cursor:stop] = (
                timestamp * SCOPE_DATAPATH_CLOCK_PERIOD_NS
                + sample_offsets_ns
                - newest_sample_ns
            )
            values = np.frombuffer(
                record,
                dtype="<i2",
                count=samples_per_frame,
                offset=8,
            )
            y_values[cursor:stop] = (values - zero) * scale
            cursor = stop
            if index + 1 < len(records):
                x_values[cursor] = x_values[cursor - 1]
                y_values[cursor] = np.nan
                cursor += 1
        x_values /= 1_000_000_000
        self._last_dma_plot_points = sample_points
        self._curve.setData(x_values, y_values, stepMode=waveform_step_mode(), connect="finite")

    def _render_values(self, now_ns: int) -> None:
        latest = self._latest
        if latest is None:
            if self._monitor_error is not None:
                self.status.setText(self._monitor_error)
            return

        zero = self.ui.spinZero.value()
        scale = self.ui.spinScale.value()
        unit = self._unit()
        current = self._convert(latest.raw_code, zero, scale)
        self.ui.lblCurrent.setText(f"{self._format_value(current)} {unit}")
        self.ui.lblRaw.setText(self._format_value(latest.raw_code))

        batch_values = [
            self._convert(sample.raw_code, zero, scale) for sample in self._last_batch
        ]
        if batch_values:
            self.ui.lblInterval.setText(
                f"mean {self._format_value(statistics.fmean(batch_values))}; "
                f"min {self._format_value(min(batch_values))}; "
                f"max {self._format_value(max(batch_values))} {unit}; "
                f"n={len(batch_values)}"
            )

        history = self._history
        recent = history.since(latest.timestamp_ns - _RATE_WINDOW_NS)
        intervals_ns = np.diff(history.timestamps[recent])
        intervals_ns = intervals_ns[intervals_ns > 0]
        rate_hz = 0.0
        median_interval_ns = 0.0
        if intervals_ns.size:
            median_interval_ns = float(np.median(intervals_ns))
            rate_hz = 1_000_000_000 / float(intervals_ns.mean())
        age_ns = max(0, now_ns - latest.timestamp_ns)
        latencies_ns = history.read_latencies_ns[recent]
        median_latency_ms = (
            float(np.median(latencies_ns)) / 1_000_000 if latencies_ns.size else 0.0
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
        else:
            status = "Live IIR polling."
        if self._dropped_samples:
            status += f" Display buffer dropped {self._dropped_samples:,} sample(s)."
        self._set_status(status, "FPGA IIR output; host timestamps are transaction midpoints.")

        if len(history):
            timestamps = history.timestamps
            x = (timestamps - timestamps[-1]) / 1_000_000_000
            y = (history.raw_codes - zero) * scale
            # The same item is reused after switching back from raw DMA pulses
            # to the host-timed scalar IIR trend; restore ordinary line drawing.
            self._curve.setData(x, y, stepMode=None, connect="auto")

    def reset_zoom(self) -> None:
        self._plot.enableAutoRange()

    def configuration_settings(self) -> dict[str, object]:
        return {
            "mode": _MODE_NAMES[self.ui.comboMode.currentIndex()],
            "zero_code": self.ui.spinZero.value(),
            "scale_per_code": self.ui.spinScale.value(),
            "unit": self._unit(),
            "display_fps": self._display_hz,
            "dma_gap_cycles": self._dma_gap_cycles(),
            "sum_frame_samples": self._sum_frame_samples(),
            "sum_file_format": self._sum_format(),
            "pause_readbacks": self.ui.cbPauseReadbacks.isChecked(),
            "measurement_time_s": self.ui.spinTime.value(),
            "series_runs": self._series.runs(),
            "analysis_bin_ms": (
                self._scope_current_accumulator.bin_width_ms
                if self._scope_current_accumulator is not None
                else 100
            ),
        }

    def apply_configuration_settings(self, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        if "sum_frame_samples" in settings:
            if self._is_running():
                raise RuntimeError("cannot change sum window size while monitor is running")
            normalize_numeric_control(self.ui.spinSumSamples, int(settings["sum_frame_samples"]))
        if "dma_gap_cycles" in settings:
            if self._is_running():
                raise RuntimeError("cannot change DMA gap while monitor is running")
            self._set_dma_gap_cycles(int(settings["dma_gap_cycles"]))
        requested_mode = settings.get("mode")
        if requested_mode in _MODE_NAMES:
            target = _MODE_NAMES.index(requested_mode)
            if target == _MODE_PERIODIC_SUM and not self._sum_supported():
                raise ValueError("periodic_sum requires supported IP123/IP124 Scope DMA")
            if target != self.ui.comboMode.currentIndex():
                restart = self._is_running()
                if restart:
                    self.stop_monitor_sync()
                self.ui.comboMode.setCurrentIndex(target)
                if restart:
                    self.start_monitor()
        if "pause_readbacks" in settings:
            self.ui.cbPauseReadbacks.setChecked(bool(settings["pause_readbacks"]))
        if "measurement_time_s" in settings:
            self.ui.spinTime.setValue(int(settings["measurement_time_s"]))
        if "series_runs" in settings:
            self._series.set_runs(int(settings["series_runs"]))
        if settings.get("sum_file_format") in _SUM_FORMATS:
            if self._is_running():
                raise RuntimeError("cannot change the sum file format while monitor is running")
            self.ui.comboSumFormat.setCurrentIndex(
                _SUM_FORMATS.index(str(settings["sum_file_format"]))
            )
        if "zero_code" in settings:
            self.ui.spinZero.setValue(float(settings["zero_code"]))
        if "scale_per_code" in settings:
            self.ui.spinScale.setValue(float(settings["scale_per_code"]))
        if "unit" in settings:
            index = self.ui.comboUnit.findText(str(settings["unit"]))
            if index >= 0:
                self.ui.comboUnit.setCurrentIndex(index)
        if "display_fps" in settings:
            requested_fps = int(settings["display_fps"])
            if not 1 <= requested_fps <= _DISPLAY_HZ:
                raise ValueError(f"display_fps must be 1..{_DISPLAY_HZ}")
            self._display_hz = requested_fps
            self._display_period_ns = round(1_000_000_000 / requested_fps)
        if "analysis_bin_ms" in settings:
            if self._is_running():
                raise RuntimeError("cannot change analysis bins while monitor is running")
            accumulator = self._scope_current_accumulator
            if accumulator is not None:
                accumulator.configure_bin_width_ms(int(settings["analysis_bin_ms"]))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.stop_monitor_sync()
        super().closeEvent(event)
