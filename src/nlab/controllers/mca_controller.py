from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt, QThread, QTimer
from PySide6.QtWidgets import QDoubleSpinBox, QFileDialog, QSlider, QSpinBox, QWidget

from nlab.hardware.digitizer.dma import IIOMcaDmaStreamer, McaDmaStreamer, McaEventBuffer
from nlab.hardware.digitizer.mca import MCA_PARAMETER_SPECS, MCAParam, MultiChannelAnalyzer
from nlab.hardware.digitizer.scope import RangeSpec
from nlab.ui.ui_mca_view import Ui_MCAView
from nlab.views.plot_viewbox import ModifierZoomViewBox
from nlab.views.responsive_layout import configure_mca_layout
from nlab.views.time_axis import format_duration_ns, time_axis_scale
from nlab.workers.dma_workers import IIOMcaDmaWorker, McaDmaWorker
from nlab.workers.mca_worker import MCAReadback, MCAWorker

log = logging.getLogger(__name__)

_DEBUG_SIGNAL_NAMES = [
    "Input signal",
    "Trigger signal",
    "Trapezoid signal",
    "Trapezoid energy",
    "CFD signal",
    "CFD window",
    "Charge comparison window",
    "PSD ZC window",
    "Logic trigger",
]

_BINNING_LABELS = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"]
_WINDOW_LABELS = ["8 ns", "16 ns", "32 ns", "64 ns", "128 ns", "256 ns", "512 ns"]
_TRIGGER_SOURCE_LABELS = ["Threshold", "CR-RC2", "CR2-RC2"]
_LP_PRESET_LABELS = ["200 MHz", "70 MHz", "Moving average"]

# Hardware/network polling can run faster than Qt can repaint the two debug
# curves plus a 16384-bin histogram. Keep acquisition at the requested rate,
# but cap presentation so queued readback signals cannot starve mouse/keyboard
# events in the GUI thread. 15 Hz is still visually continuous and leaves a
# comfortable event-loop budget on the machine used for the live review.
_MAX_GUI_RENDER_HZ = 15

# The diagnostic memories publish one entry per 125 MHz datapath beat.  The
# board timebase is therefore 8 ns per displayed debug sample; see
# hw_description/user-api.md, "Timebase" and "Diagnostic memories".
_DEBUG_SAMPLE_PERIOD_NS = 8

_MCA_CONTROL_TOOLTIPS = {
    "comboPulsePolarity": "Selects whether pulses are expected to be negative or positive.",
    "comboBaseline": "Selects the time window used to estimate the signal baseline.",
    "comboDebug1": "Selects the internal signal captured in diagnostic waveform bank 1.",
    "comboDebug2": "Selects the internal signal captured in diagnostic waveform bank 2.",
    "spinPileupWindow": "Sets the interval in which a second pulse is classified as pile-up.",
    "comboBinning": "Sets the histogram energy-bin width; larger factors combine more codes.",
    "spinTimeLimit": "Sets the acquisition duration in seconds; 0 runs until stopped.",
    "spinRefreshRate": "Sets the requested rate for waveform, spectrum, and statistics updates.",
    "spinTriggerLevel": "Sets the raw signal threshold used by the selected trigger source.",
    "spinFrameSamples": "Sets the diagnostic waveform capture-window duration in nanoseconds.",
    "spinPretrigger": "Sets the diagnostic waveform offset before the trigger in nanoseconds.",
    "comboTriggerSource": "Selects threshold, CR-RC2, or CR2-RC2 pulse triggering.",
    "spinEdgeDetCoeff": "Sets the legacy edge-detector coefficient when supported by hardware.",
    "comboLpPreset": "Selects the input FIR low-pass response or moving-average filter.",
    "spinCrrc2Cdelay": "Sets the C-stage delay of the CR-RC2 shaping filter in nanoseconds.",
    "spinCrrc2Fdelay": "Sets the F-stage delay of the CR-RC2 shaping filter in nanoseconds.",
    "spinCrrc2Pzc": "Sets the raw pole-zero correction coefficient for CR-RC2 shaping.",
    "spinCfdFactor": "Sets the constant-fraction multiplier used to form the CFD signal.",
    "spinCfdDelay": "Sets the CFD signal delay in nanoseconds.",
    "spinCfdTwLow": "Sets the lower CFD time-walk boundary in nanoseconds.",
    "spinCfdTwHigh": "Sets the upper CFD time-walk boundary in nanoseconds.",
    "spinTrapR": "Sets the trapezoidal-filter rise time in nanoseconds.",
    "spinTrapM": "Sets the trapezoidal-filter flat-top time in nanoseconds.",
    "spinTrapT": "Sets the pole-zero time constant in nanoseconds; IIO converts it to beta.",
    "spinTrapE": "Sets the trapezoidal-filter energy sampling time in nanoseconds.",
    "comboTrapFt": "Selects the trapezoidal filter's flat-top window duration.",
    "spinCcTime": "Sets the charge-comparison integration time in nanoseconds.",
    "comboPsdZcMode": "Selects hardware mode 0 or 1 for PSD zero-crossing analysis.",
    "spinPsdZcLow": "Sets the lower PSD zero-crossing boundary in nanoseconds.",
    "spinPsdZcHigh": "Sets the upper PSD zero-crossing boundary in nanoseconds.",
}


class _PsdCaptureSink(Protocol):
    def begin_capture(self, enabled: bool, note: str = "") -> None: ...

    def finish_capture(self) -> None: ...

    def set_capture_error(self, message: str) -> None: ...


class MCAController(QWidget):
    """View + controller for a single MultiChannelAnalyzer channel."""

    def __init__(
        self,
        mca: MultiChannelAnalyzer,
        mca_dma: McaDmaStreamer | IIOMcaDmaStreamer | None = None,
        channel: int = 1,
        parent: QWidget | None = None,
        event_buffer: McaEventBuffer | None = None,
        psd_capture: _PsdCaptureSink | None = None,
    ) -> None:
        super().__init__(parent)
        self._mca = mca
        self._mca_dma = mca_dma
        self._channel = channel
        self._psd_capture = psd_capture
        self.ui = Ui_MCAView()
        self.ui.setupUi(self)
        self._responsive_layout = configure_mca_layout(self, self.ui)
        self._apply_control_tooltips()

        self._worker: MCAWorker | None = None
        self._worker_thread: QThread | None = None

        self._dma_worker: McaDmaWorker | IIOMcaDmaWorker | None = None
        self._dma_thread: QThread | None = None
        self._dma_filepath: Path | None = None
        self._dma_counter = 0
        self._event_buffer = event_buffer or McaEventBuffer()
        self._psd_capture_enabled = False

        self._last_histogram: np.ndarray | None = None
        self._last_elapsed_s: float = 0.0
        self._pending_readback: MCAReadback | None = None
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_pending_readback)

        self._populate_combos()
        self._apply_parameter_specs()
        self._disarm_before_initialization()
        self._send_defaults()
        self._load_hardware_state()
        self._setup_debug_plot()
        self._setup_histogram_plot()
        self._connect_signals()
        self.ui.btnStop.setEnabled(False)

    # ------------------------------------------------------------------
    # Combo population
    # ------------------------------------------------------------------

    def _apply_control_tooltips(self) -> None:
        for object_name, tooltip in _MCA_CONTROL_TOOLTIPS.items():
            getattr(self.ui, object_name).setToolTip(tooltip)

    def _populate_combos(self) -> None:
        self.ui.comboPulsePolarity.clear()
        self.ui.comboPulsePolarity.addItems(["Negative", "Positive"])
        self.ui.comboBaseline.clear()
        self.ui.comboBaseline.addItems(_WINDOW_LABELS)

        self.ui.comboDebug1.clear()
        self.ui.comboDebug2.clear()
        for selector in self._mca.get_debug_signal_selectors():
            if not 0 <= selector < len(_DEBUG_SIGNAL_NAMES):
                log.warning("Ignoring unknown MCA debug selector %d", selector)
                continue
            name = _DEBUG_SIGNAL_NAMES[selector]
            self.ui.comboDebug1.addItem(name, selector)
            self.ui.comboDebug2.addItem(name, selector)
        self.ui.comboDebug2.setCurrentIndex(self.ui.comboDebug2.findData(1))

        for label in _BINNING_LABELS:
            self.ui.comboBinning.addItem(label)

        self.ui.comboTriggerSource.clear()
        self.ui.comboTriggerSource.addItems(_TRIGGER_SOURCE_LABELS)
        self.ui.comboLpPreset.clear()
        self.ui.comboLpPreset.addItems(_LP_PRESET_LABELS)
        self.ui.comboTrapFt.addItems(_WINDOW_LABELS)

    # ------------------------------------------------------------------
    # Hardware range application
    # ------------------------------------------------------------------

    def _apply_parameter_specs(self) -> None:
        """Drive every numerical hardware control from MCA specs.

        The specs mirror ``user-api.md`` and the deployed v101 IIO
        attributes. Keeping the form and validation layer tied to the same
        table prevents a widget from silently clamping a valid readback or
        offering a value the driver will reject.
        """
        pairs = (
            (MCAParam.TRIGGER_LEVEL, self.ui.spinTriggerLevel, self.ui.sliderTriggerLevel),
            (MCAParam.PRETRIGGER_SAMPLES, self.ui.spinPretrigger, self.ui.sliderPretrigger),
            (MCAParam.FRAME_SAMPLES, self.ui.spinFrameSamples, self.ui.sliderFrameSamples),
            (MCAParam.CRRC2_CDELAY, self.ui.spinCrrc2Cdelay, self.ui.sliderCrrc2Cdelay),
            (MCAParam.CRRC2_FDELAY, self.ui.spinCrrc2Fdelay, self.ui.sliderCrrc2Fdelay),
            (MCAParam.CRRC2_PZC, self.ui.spinCrrc2Pzc, self.ui.sliderCrrc2Pzc),
            (MCAParam.CFD_DELAY, self.ui.spinCfdDelay, self.ui.sliderCfdDelay),
            (MCAParam.TRAPEZ_R, self.ui.spinTrapR, self.ui.sliderTrapR),
            (MCAParam.TRAPEZ_M, self.ui.spinTrapM, self.ui.sliderTrapM),
            (MCAParam.TRAPEZ_E, self.ui.spinTrapE, self.ui.sliderTrapE),
        )
        for parameter, spinbox, slider in pairs:
            spec = MCA_PARAMETER_SPECS[parameter]
            assert isinstance(spec, RangeSpec)
            self._apply_range_to_spinbox(spinbox, spec)
            self._apply_range_to_slider(slider, spec)

        singles = (
            (MCAParam.PILEUP_WINDOW, self.ui.spinPileupWindow),
            (MCAParam.TIME_LIMIT, self.ui.spinTimeLimit),
            (MCAParam.CFD_TW_LOW, self.ui.spinCfdTwLow),
            (MCAParam.CFD_TW_HIGH, self.ui.spinCfdTwHigh),
            (MCAParam.CC_TIME, self.ui.spinCcTime),
            (MCAParam.PSD_ZC_LOW, self.ui.spinPsdZcLow),
            (MCAParam.PSD_ZC_HIGH, self.ui.spinPsdZcHigh),
        )
        for parameter, spinbox in singles:
            spec = MCA_PARAMETER_SPECS[parameter]
            assert isinstance(spec, RangeSpec)
            self._apply_range_to_spinbox(spinbox, spec)

        for parameter, double_spinbox in (
            (MCAParam.CFD_FACTOR, self.ui.spinCfdFactor),
            (MCAParam.TRAPEZ_T, self.ui.spinTrapT),
            (MCAParam.EDGE_DET_COEFF, self.ui.spinEdgeDetCoeff),
        ):
            spec = MCA_PARAMETER_SPECS[parameter]
            assert isinstance(spec, RangeSpec)
            self._apply_range_to_double_spinbox(double_spinbox, spec)

        for control in (
            self.ui.spinPretrigger,
            self.ui.spinFrameSamples,
            self.ui.spinCrrc2Cdelay,
            self.ui.spinCrrc2Fdelay,
            self.ui.spinCfdDelay,
            self.ui.spinCfdTwLow,
            self.ui.spinCfdTwHigh,
            self.ui.spinTrapR,
            self.ui.spinTrapM,
            self.ui.spinTrapT,
            self.ui.spinTrapE,
            self.ui.spinCcTime,
            self.ui.spinPsdZcLow,
            self.ui.spinPsdZcHigh,
        ):
            control.setSuffix(" ns")
        # Keep runtime-generated UI modules made before the form update correct.
        self.ui.labelTrapT.setText("Pole-zero time:")

        # There is no edge-detector-coefficient attribute in the current
        # IIO pulse processor. Hide the legacy-only compatibility control so
        # its in-memory shadow cannot be mistaken for hardware readback.
        edge_backed = self._mca.edge_det_coeff_is_hardware_backed()
        self.ui.labelEdgeDetCoeff.setVisible(edge_backed)
        self.ui.spinEdgeDetCoeff.setVisible(edge_backed)

    @staticmethod
    def _apply_range_to_spinbox(spinbox: QSpinBox, spec: RangeSpec) -> None:
        spinbox.setRange(int(spec.min_val), int(spec.max_val))
        spinbox.setSingleStep(int(spec.step) or 1)
        spinbox.setValue(int(spec.default))

    @staticmethod
    def _apply_range_to_slider(slider: QSlider, spec: RangeSpec) -> None:
        slider.setRange(int(spec.min_val), int(spec.max_val))
        slider.setSingleStep(int(spec.step) or 1)
        slider.setPageStep(max(int(spec.step), 1))
        slider.setValue(int(spec.default))

    @staticmethod
    def _apply_range_to_double_spinbox(spinbox: QDoubleSpinBox, spec: RangeSpec) -> None:
        spinbox.setRange(float(spec.min_val), float(spec.max_val))
        spinbox.setSingleStep(float(spec.step) or 1.0)
        spinbox.setValue(float(spec.default))

    # ------------------------------------------------------------------
    # Write defaults to hardware, then read back
    # ------------------------------------------------------------------

    def _disarm_before_initialization(self) -> None:
        """Clear a stale enable gate before writing configuration defaults.

        ``measurement_in_progress`` is not the ownership gate: an MCA armed
        for an external trigger can report false there while ``enable`` is
        still one. Per vdpp-pulse-processor.c's pp_enable_store(), writing
        enable=0 is always accepted, including after a previous GUI crashed.
        """
        self._mca.stop()
        log.debug("MCA ch%d: startup enable gate cleared", self._channel)

    def _send_defaults(self) -> None:
        specs = MCA_PARAMETER_SPECS

        # Signal
        self._mca.set_trigger_level(int(specs[MCAParam.TRIGGER_LEVEL].default))
        self._mca.set_pulse_polarity(int(specs[MCAParam.PULSE_POLARITY].default))
        self._mca.set_baseline_window(int(specs[MCAParam.BASELINE_WINDOW].default))
        self._mca.set_pretrigger_samples(int(specs[MCAParam.PRETRIGGER_SAMPLES].default))
        self._mca.set_frame_samples(int(specs[MCAParam.FRAME_SAMPLES].default))
        self._mca.set_trg_source(int(specs[MCAParam.TRG_SOURCE].default))
        self._mca.set_ext_trig_enable(bool(specs[MCAParam.EXT_TRIG_ENABLE].default))
        self._mca.set_edge_det_coeff(int(specs[MCAParam.EDGE_DET_COEFF].default))

        # Acquisition
        self._mca.set_energy_bin(int(specs[MCAParam.ENERGY_BIN].default))
        self._mca.set_pileup_window(int(specs[MCAParam.PILEUP_WINDOW].default))
        self._mca.set_time_limit(int(specs[MCAParam.TIME_LIMIT].default))

        # Debug signal routing
        self._mca.set_mem1_sig_select(int(specs[MCAParam.MEM1_SIG_SELECT].default))
        self._mca.set_mem2_sig_select(int(specs[MCAParam.MEM2_SIG_SELECT].default))

        # CR-RC2 / LP filter
        self._mca.filters.lp.set_preset(int(specs[MCAParam.LP_PRESET].default))
        self._mca.filters.crrc2.set_Cdelay(int(specs[MCAParam.CRRC2_CDELAY].default))
        self._mca.filters.crrc2.set_Fdelay(int(specs[MCAParam.CRRC2_FDELAY].default))
        self._mca.filters.crrc2.set_pzc_coeff(int(specs[MCAParam.CRRC2_PZC].default))

        # CFD
        self._mca.filters.cfd.set_enable(bool(specs[MCAParam.CFD_ENABLE].default))
        self._mca.filters.cfd.set_factor(float(specs[MCAParam.CFD_FACTOR].default))
        self._mca.filters.cfd.set_delay(int(specs[MCAParam.CFD_DELAY].default))
        self._mca.filters.cfd.set_time_window_low(int(specs[MCAParam.CFD_TW_LOW].default))
        self._mca.filters.cfd.set_time_window_high(int(specs[MCAParam.CFD_TW_HIGH].default))

        # Trapezoid
        self._mca.filters.trapezoid.set_enable(bool(specs[MCAParam.TRAPEZ_ENABLE].default))
        self._mca.filters.trapezoid.set_R(int(specs[MCAParam.TRAPEZ_R].default))
        self._mca.filters.trapezoid.set_M(int(specs[MCAParam.TRAPEZ_M].default))
        self._mca.filters.trapezoid.set_T(int(specs[MCAParam.TRAPEZ_T].default))
        self._mca.filters.trapezoid.set_E(int(specs[MCAParam.TRAPEZ_E].default))
        self._mca.filters.trapezoid.set_FT(int(specs[MCAParam.TRAPEZ_FT].default))

        # Charge comparison (PSD)
        self._mca.filters.charge_comparison.set_enable(bool(specs[MCAParam.CC_ENABLE].default))
        self._mca.filters.charge_comparison.set_time(int(specs[MCAParam.CC_TIME].default))

        # PSD zero-crossing
        self._mca.filters.psd_zc.set_enable(bool(specs[MCAParam.PSD_ZC_ENABLE].default))
        self._mca.filters.psd_zc.set_mode(int(specs[MCAParam.PSD_ZC_MODE].default))
        self._mca.filters.psd_zc.set_time_window_low(int(specs[MCAParam.PSD_ZC_LOW].default))
        self._mca.filters.psd_zc.set_time_window_high(int(specs[MCAParam.PSD_ZC_HIGH].default))

        log.info("MCA ch%d: channel defaults sent to hardware", self._channel)

    def _load_hardware_state(self) -> None:
        self.ui.spinTriggerLevel.setValue(self._mca.get_trigger_level())
        self.ui.sliderTriggerLevel.setValue(self._mca.get_trigger_level())
        self.ui.comboPulsePolarity.setCurrentIndex(self._mca.get_pulse_polarity())
        self.ui.comboBaseline.setCurrentIndex(self._mca.get_baseline_window())
        self.ui.spinPretrigger.setValue(self._mca.get_pretrigger_samples())
        self.ui.sliderPretrigger.setValue(self._mca.get_pretrigger_samples())
        self.ui.spinFrameSamples.setValue(self._mca.get_frame_samples())
        self.ui.sliderFrameSamples.setValue(self._mca.get_frame_samples())
        self.ui.comboBinning.setCurrentIndex(self._mca.get_energy_bin())
        self.ui.spinPileupWindow.setValue(self._mca.get_pileup_window())
        self.ui.spinTimeLimit.setValue(self._mca.get_time_limit())
        self.ui.comboTriggerSource.setCurrentIndex(self._mca.get_trg_source())
        self.ui.cbExtTrigger.setChecked(self._mca.get_ext_trig_enable())
        self.ui.comboDebug1.setCurrentIndex(
            self.ui.comboDebug1.findData(self._mca.get_mem1_sig_select())
        )
        self.ui.comboDebug2.setCurrentIndex(
            self.ui.comboDebug2.findData(self._mca.get_mem2_sig_select())
        )
        self.ui.spinEdgeDetCoeff.setValue(self._mca.get_edge_det_coeff())

        # CR-RC2 (LP preset is write-only on the device, no readback)
        self.ui.spinCrrc2Cdelay.setValue(self._mca.filters.crrc2.get_Cdelay())
        self.ui.sliderCrrc2Cdelay.setValue(self._mca.filters.crrc2.get_Cdelay())
        self.ui.spinCrrc2Fdelay.setValue(self._mca.filters.crrc2.get_Fdelay())
        self.ui.sliderCrrc2Fdelay.setValue(self._mca.filters.crrc2.get_Fdelay())
        self.ui.spinCrrc2Pzc.setValue(self._mca.filters.crrc2.get_pzc_coeff())
        self.ui.sliderCrrc2Pzc.setValue(self._mca.filters.crrc2.get_pzc_coeff())

        # CFD
        self.ui.cbCfdEnable.setChecked(self._mca.filters.cfd.get_enable())
        self.ui.spinCfdFactor.setValue(self._mca.filters.cfd.get_factor())
        self.ui.spinCfdDelay.setValue(self._mca.filters.cfd.get_delay())
        self.ui.sliderCfdDelay.setValue(self._mca.filters.cfd.get_delay())
        self.ui.spinCfdTwLow.setValue(self._mca.filters.cfd.get_time_window_low())
        self.ui.spinCfdTwHigh.setValue(self._mca.filters.cfd.get_time_window_high())

        # Trapezoid
        self.ui.cbTrapezEnable.setChecked(self._mca.filters.trapezoid.get_enable())
        self.ui.spinTrapR.setValue(self._mca.filters.trapezoid.get_R())
        self.ui.sliderTrapR.setValue(self._mca.filters.trapezoid.get_R())
        self.ui.spinTrapM.setValue(self._mca.filters.trapezoid.get_M())
        self.ui.sliderTrapM.setValue(self._mca.filters.trapezoid.get_M())
        self.ui.spinTrapT.setValue(self._mca.filters.trapezoid.get_T())
        self.ui.spinTrapE.setValue(self._mca.filters.trapezoid.get_E())
        self.ui.sliderTrapE.setValue(self._mca.filters.trapezoid.get_E())
        self.ui.comboTrapFt.setCurrentIndex(self._mca.filters.trapezoid.get_FT())

        # Charge comparison (PSD)
        self.ui.cbCcEnable.setChecked(self._mca.filters.charge_comparison.get_enable())
        self.ui.spinCcTime.setValue(self._mca.filters.charge_comparison.get_time())

        # PSD zero-crossing
        self.ui.cbPsdZcEnable.setChecked(self._mca.filters.psd_zc.get_enable())
        self.ui.comboPsdZcMode.setCurrentIndex(self._mca.filters.psd_zc.get_mode())
        self.ui.spinPsdZcLow.setValue(self._mca.filters.psd_zc.get_time_window_low())
        self.ui.spinPsdZcHigh.setValue(self._mca.filters.psd_zc.get_time_window_high())

        log.info("MCA ch%d: hardware state loaded into UI", self._channel)

    # ------------------------------------------------------------------
    # Enable/disable parameter controls during DMA
    # ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.ui.groupMca.setEnabled(enabled)
        self.ui.groupSignal.setEnabled(enabled)
        self.ui.tabFilters.setEnabled(enabled)
        self.ui.spinTimeLimit.setEnabled(enabled)
        self.ui.spinRefreshRate.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Debug plot (scope-like, two curves)
    # ------------------------------------------------------------------

    def _setup_debug_plot(self) -> None:
        layout = pg.GraphicsLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        self.ui.plotDebug.setCentralItem(layout)
        self.ui.plotDebug.setBackground("#f8f9fa")

        layout.addLabel("Amplitude", angle=-90)
        self._debug_plot = layout.addPlot(viewBox=ModifierZoomViewBox())
        self._debug_plot.showAxis("right")
        self._debug_plot.showAxis("top")
        self._debug_plot.showGrid(x=True, y=True, alpha=0.2)
        self._debug_plot.addLegend(offset=(10, 10))
        layout.nextRow()
        self._debug_time_axis_label = layout.addLabel("Time [ns]", col=1)

        self._debug1_curve = self._debug_plot.plot(
            pen=pg.mkPen("#00bfff", width=1),
            name="Debug 1",
        )
        self._debug2_curve = self._debug_plot.plot(
            pen=pg.mkPen("#ff8c00", width=1),
            name="Debug 2",
        )
        self._update_debug_time_axis()

    def _update_debug_time_axis(self, frame_samples: int | None = None) -> None:
        if frame_samples is None:
            frame_samples = self.ui.spinFrameSamples.value()
        displayed_samples = frame_samples // _DEBUG_SAMPLE_PERIOD_NS
        duration_ns = displayed_samples * _DEBUG_SAMPLE_PERIOD_NS
        self._debug_time_scale = time_axis_scale(duration_ns)
        self._debug_time_axis_label.setText(
            f"Time [{self._debug_time_scale.unit}]  "
            f"({_DEBUG_SAMPLE_PERIOD_NS} ns/debug sample)"
        )
        for curve in (self._debug1_curve, self._debug2_curve):
            _, plotted_samples = curve.getData()
            if plotted_samples is None or len(plotted_samples) == 0:
                continue
            plotted_samples = plotted_samples[:displayed_samples]
            plotted_time = (
                np.arange(len(plotted_samples))
                * _DEBUG_SAMPLE_PERIOD_NS
                / self._debug_time_scale.ns_per_unit
            )
            curve.setData(plotted_time, plotted_samples)
        tip = (
            "Sets the diagnostic waveform capture-window length. "
            f"Debug sample period: {_DEBUG_SAMPLE_PERIOD_NS} ns "
            f"(125 MHz datapath). The current frame displays up to "
            f"{displayed_samples} debug samples ({format_duration_ns(duration_ns)})."
        )
        self.ui.spinFrameSamples.setToolTip(tip)

    # ------------------------------------------------------------------
    # Histogram plot (channels vs counts, with ROI)
    # ------------------------------------------------------------------

    def _setup_histogram_plot(self) -> None:
        layout = pg.GraphicsLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        self.ui.plotHistogram.setCentralItem(layout)
        self.ui.plotHistogram.setBackground("#f8f9fa")

        layout.addLabel("Counts", angle=-90)
        self._hist_plot = layout.addPlot(viewBox=ModifierZoomViewBox())
        self._hist_plot.showAxis("right")
        self._hist_plot.showAxis("top")
        self._hist_plot.showGrid(x=True, y=True, alpha=0.2)
        layout.nextRow()
        layout.addLabel("Channel", col=1)

        self._hist_curve = self._hist_plot.plot(
            pen=pg.mkPen("#1f77b4", width=1),
            stepMode="center",
        )

        self._roi = pg.LinearRegionItem(values=[100, 200], movable=True)
        self._roi_dragging = False
        self._roi.setZValue(10)
        self._hist_plot.addItem(self._roi)
        self._roi.setVisible(False)
        # This pyqtgraph version exposes no sigRegionChangeStarted. The first
        # sigRegionChanged event marks an active drag; unlike the old direct
        # connection, this handler only flips a flag and performs no ROI
        # calculations. sigRegionChangeFinished performs the one update.
        self._roi.sigRegionChanged.connect(self._on_roi_region_changed)
        self._roi.sigRegionChangeFinished.connect(self._on_roi_change_finished)

    def _on_roi_region_changed(self) -> None:
        # Histogram readbacks can continue at up to 60 Hz while the mouse is
        # moving. _update_histogram() consults this flag so none of those
        # readbacks trigger ROI calculations during the drag either.
        self._roi_dragging = True

    def _on_roi_change_finished(self) -> None:
        self._roi_dragging = False
        self._update_roi_stats()

    def set_roi_visible(self, visible: bool) -> None:
        """Show/hide the ROI selection tool and its stats panel."""
        self._roi.setVisible(visible)
        self.ui.roiStatsPanel.setVisible(visible)
        if not visible:
            self._roi_dragging = False
        if visible:
            self._update_roi_stats()

    def set_log_y(self, enabled: bool) -> None:
        """Toggle logarithmic Y-axis on the histogram plot."""
        self._hist_plot.setLogMode(y=enabled)

    def reset_zoom(self) -> None:
        """Auto-range the debug and histogram plots."""
        self._debug_plot.autoRange()
        self._hist_plot.autoRange()

    def hardware_configuration_settings(self) -> dict[str, int]:
        """Return write-only hardware choices that cannot be read back."""
        return {"low_pass_preset": self.ui.comboLpPreset.currentIndex()}

    def populate_hardware_configuration_settings(self, settings: object) -> None:
        if not isinstance(settings, dict) or "low_pass_preset" not in settings:
            return
        self.ui.comboLpPreset.blockSignals(True)
        self.ui.comboLpPreset.setCurrentIndex(int(settings["low_pass_preset"]))
        self.ui.comboLpPreset.blockSignals(False)

    def configuration_settings(self) -> dict[str, object]:
        """Return controls that affect only GUI polling and presentation."""
        return {
            "refresh_rate_hz": self.ui.spinRefreshRate.value(),
            "roi": [float(value) for value in self._roi.getRegion()],
        }

    def apply_configuration_settings(self, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        if "refresh_rate_hz" in settings:
            self.ui.spinRefreshRate.setValue(int(settings["refresh_rate_hz"]))
        roi = settings.get("roi")
        if isinstance(roi, list) and len(roi) == 2:
            self._roi.setRegion((float(roi[0]), float(roi[1])))

    # ------------------------------------------------------------------
    # ROI statistics (gross counts, peak centroid/FWHM estimate — no curve fit)
    # ------------------------------------------------------------------

    def _update_roi_stats(self) -> None:
        if not self._roi.isVisible():
            return
        if self._last_histogram is None or len(self._last_histogram) == 0:
            self.ui.lblRoiStats.setText("No histogram data yet.")
            return

        histogram = self._last_histogram
        n_bins = len(histogram)

        left, right = self._roi.getRegion()
        low_bin = int(np.clip(round(left), 0, n_bins - 1))
        high_bin = int(np.clip(round(right), 0, n_bins - 1))
        if high_bin < low_bin:
            low_bin, high_bin = high_bin, low_bin
        width = high_bin - low_bin + 1

        window = histogram[low_bin : high_bin + 1].astype(np.float64)
        bins = np.arange(low_bin, high_bin + 1, dtype=np.float64)

        gross_counts = float(window.sum())
        gross_unc = np.sqrt(gross_counts) if gross_counts > 0 else 0.0
        rel_unc_pct = (gross_unc / gross_counts * 100.0) if gross_counts > 0 else float("nan")

        if self._last_elapsed_s > 0:
            cps_line = f"  CPS:                {gross_counts / self._last_elapsed_s:.2f}"
        else:
            cps_line = "  CPS:                N/A (no live time)"

        if gross_counts > 0:
            max_idx_local = int(np.argmax(window))
            max_bin_pos = low_bin + max_idx_local
            max_bin_counts = window[max_idx_local]

            centroid = float((bins * window).sum() / gross_counts)
            variance = float((window * (bins - centroid) ** 2).sum() / gross_counts)
            sigma = np.sqrt(variance) if variance > 0 else 0.0
            fwhm = 2.3548 * sigma
            resolution_pct = (fwhm / centroid * 100.0) if centroid > 0 else float("nan")

            peak_lines = (
                f"  Max bin position:   {max_bin_pos}\n"
                f"  Max bin counts:     {max_bin_counts:.0f}\n"
                f"  Centroid:           {centroid:.2f}\n"
                f"  Weighted sigma:     {sigma:.2f}\n"
                f"  Approx. FWHM:       {fwhm:.2f}\n"
                f"  Approx. resolution: {resolution_pct:.2f} %"
            )
        else:
            peak_lines = "  No counts in ROI"

        text = (
            f"ROI\n"
            f"  Left marker:        {low_bin}\n"
            f"  Right marker:       {high_bin}\n"
            f"  Width [bins]:       {width}\n"
            f"  Energy/ch range:    {low_bin}-{high_bin}\n"
            f"\n"
            f"Counts\n"
            f"  Gross counts:       {gross_counts:.0f}\n"
            f"  Gross uncertainty:  {gross_unc:.2f}\n"
            f"  Relative unc.:      {rel_unc_pct:.2f} %\n"
            f"{cps_line}\n"
            f"\n"
            f"Peak\n"
            f"{peak_lines}"
        )
        self.ui.lblRoiStats.setText(text)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _apply_hardware_setting(self, write: Callable[[], None]) -> None:
        """Apply a setting immediately, preserving a polling measurement.

        IIO MCA fields are immutable while enabled. Unlike list-mode DMA,
        an ordinary histogram/viewer acquisition can be briefly stopped and
        restarted safely. MultiChannelAnalyzer owns the synchronization that
        prevents MCAWorker from treating the short stop as time-limit
        completion. DMA controls remain disabled because an armed list-mode
        buffer requires its full stop/drain/close lifecycle.
        """
        if self._dma_worker is not None:
            log.warning(
                "MCA ch%d: ignored configuration write while DMA is active",
                self._channel,
            )
            return
        if self._worker is None:
            write()
            return

        log.debug("MCA ch%d: pausing acquisition for live reconfiguration", self._channel)
        restarted = self._mca.reconfigure_while_running(write)
        if restarted:
            # A fresh enable starts a new accumulation. Do not present the
            # previous run's spectrum/elapsed time as belonging to it while
            # the polling worker waits for its next readback.
            self._last_histogram = None
            self._last_elapsed_s = 0.0
            self._hist_curve.setData([], [])

    def _connect_signals(self) -> None:
        self.ui.comboPulsePolarity.currentIndexChanged.connect(
            lambda i: (
                log.debug("MCA ch%d: polarity=%d", self._channel, i),
                self._apply_hardware_setting(lambda: self._mca.set_pulse_polarity(i)),
            )
        )
        self.ui.comboBaseline.currentIndexChanged.connect(
            lambda i: self._apply_hardware_setting(lambda: self._mca.set_baseline_window(i))
        )
        self.ui.comboDebug1.currentIndexChanged.connect(
            lambda i: self._mca.set_mem1_sig_select(int(self.ui.comboDebug1.itemData(i)))
        )
        self.ui.comboDebug2.currentIndexChanged.connect(
            lambda i: self._mca.set_mem2_sig_select(int(self.ui.comboDebug2.itemData(i)))
        )
        self.ui.spinPileupWindow.editingFinished.connect(
            lambda: self._apply_hardware_setting(
                lambda: self._mca.set_pileup_window(self.ui.spinPileupWindow.value())
            )
        )
        self.ui.comboBinning.currentIndexChanged.connect(
            lambda i: self._apply_hardware_setting(lambda: self._mca.set_energy_bin(i))
        )
        self.ui.cbExtTrigger.toggled.connect(
            lambda value: self._apply_hardware_setting(lambda: self._mca.set_ext_trig_enable(value))
        )

        self.ui.spinTimeLimit.editingFinished.connect(
            lambda: self._apply_hardware_setting(
                lambda: self._mca.set_time_limit(self.ui.spinTimeLimit.value())
            )
        )
        self.ui.btnStart.clicked.connect(self._on_start)
        self.ui.btnStop.clicked.connect(self._on_stop)
        self.ui.btnClearSpectrum.clicked.connect(self._on_clear_spectrum)
        self.ui.btnExportCsv.clicked.connect(self._on_export_csv)
        self.ui.cbDmaEnable.toggled.connect(lambda v: self._mca.set_dma_enable(v))
        self.ui.btnDmaFile.clicked.connect(self._on_dma_file)
        self.ui.spinRefreshRate.valueChanged.connect(self._on_refresh_rate_changed)

        self._wire_slider_spinbox(
            self.ui.sliderTriggerLevel,
            self.ui.spinTriggerLevel,
            lambda v: self._mca.set_trigger_level(v),
        )
        self._wire_slider_spinbox(
            self.ui.sliderFrameSamples,
            self.ui.spinFrameSamples,
            lambda v: self._mca.set_frame_samples(v),
        )
        self.ui.spinFrameSamples.valueChanged.connect(self._update_debug_time_axis)
        self._wire_slider_spinbox(
            self.ui.sliderPretrigger,
            self.ui.spinPretrigger,
            lambda v: self._mca.set_pretrigger_samples(v),
        )
        self.ui.comboTriggerSource.currentIndexChanged.connect(
            lambda i: self._apply_hardware_setting(lambda: self._mca.set_trg_source(i))
        )
        self.ui.spinEdgeDetCoeff.editingFinished.connect(
            lambda: self._apply_hardware_setting(
                lambda: self._mca.set_edge_det_coeff(int(self.ui.spinEdgeDetCoeff.value()))
            )
        )

        self.ui.comboLpPreset.currentIndexChanged.connect(
            lambda i: self._apply_hardware_setting(lambda: self._mca.filters.lp.set_preset(i))
        )

        self._wire_slider_spinbox(
            self.ui.sliderCrrc2Cdelay,
            self.ui.spinCrrc2Cdelay,
            lambda v: self._mca.filters.crrc2.set_Cdelay(v),
        )
        self._wire_slider_spinbox(
            self.ui.sliderCrrc2Fdelay,
            self.ui.spinCrrc2Fdelay,
            lambda v: self._mca.filters.crrc2.set_Fdelay(v),
        )
        self._wire_slider_spinbox(
            self.ui.sliderCrrc2Pzc,
            self.ui.spinCrrc2Pzc,
            lambda v: self._mca.filters.crrc2.set_pzc_coeff(v),
        )

        self.ui.cbCfdEnable.toggled.connect(
            lambda value: self._apply_hardware_setting(
                lambda: self._mca.filters.cfd.set_enable(value)
            )
        )
        self.ui.spinCfdFactor.editingFinished.connect(
            lambda: self._apply_hardware_setting(
                lambda: self._mca.filters.cfd.set_factor(self.ui.spinCfdFactor.value())
            )
        )
        self._wire_slider_spinbox(
            self.ui.sliderCfdDelay,
            self.ui.spinCfdDelay,
            lambda v: self._mca.filters.cfd.set_delay(v),
        )
        self.ui.spinCfdTwLow.editingFinished.connect(
            lambda: self._apply_hardware_setting(
                lambda: self._mca.filters.cfd.set_time_window_low(self.ui.spinCfdTwLow.value())
            )
        )
        self.ui.spinCfdTwHigh.editingFinished.connect(
            lambda: self._apply_hardware_setting(
                lambda: self._mca.filters.cfd.set_time_window_high(self.ui.spinCfdTwHigh.value())
            )
        )

        self.ui.cbTrapezEnable.toggled.connect(
            lambda value: self._apply_hardware_setting(
                lambda: self._mca.filters.trapezoid.set_enable(value)
            )
        )
        self._wire_slider_spinbox(
            self.ui.sliderTrapR,
            self.ui.spinTrapR,
            lambda v: self._mca.filters.trapezoid.set_R(v),
        )
        self._wire_slider_spinbox(
            self.ui.sliderTrapM,
            self.ui.spinTrapM,
            lambda v: self._mca.filters.trapezoid.set_M(v),
        )
        self.ui.spinTrapT.editingFinished.connect(
            lambda: self._apply_hardware_setting(
                lambda: self._mca.filters.trapezoid.set_T(int(self.ui.spinTrapT.value()))
            )
        )
        self._wire_slider_spinbox(
            self.ui.sliderTrapE,
            self.ui.spinTrapE,
            lambda v: self._mca.filters.trapezoid.set_E(v),
        )
        self.ui.comboTrapFt.currentIndexChanged.connect(
            lambda i: self._apply_hardware_setting(lambda: self._mca.filters.trapezoid.set_FT(i))
        )

        self.ui.cbCcEnable.toggled.connect(
            lambda value: self._apply_hardware_setting(
                lambda: self._mca.filters.charge_comparison.set_enable(value)
            )
        )
        self.ui.spinCcTime.editingFinished.connect(
            lambda: self._apply_hardware_setting(
                lambda: self._mca.filters.charge_comparison.set_time(self.ui.spinCcTime.value())
            )
        )
        self.ui.cbPsdZcEnable.toggled.connect(
            lambda value: self._apply_hardware_setting(
                lambda: self._mca.filters.psd_zc.set_enable(value)
            )
        )
        self.ui.comboPsdZcMode.currentIndexChanged.connect(
            lambda i: self._apply_hardware_setting(lambda: self._mca.filters.psd_zc.set_mode(i))
        )
        self.ui.spinPsdZcLow.editingFinished.connect(
            lambda: self._apply_hardware_setting(
                lambda: self._mca.filters.psd_zc.set_time_window_low(self.ui.spinPsdZcLow.value())
            )
        )
        self.ui.spinPsdZcHigh.editingFinished.connect(
            lambda: self._apply_hardware_setting(
                lambda: self._mca.filters.psd_zc.set_time_window_high(self.ui.spinPsdZcHigh.value())
            )
        )

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

        spinbox.valueChanged.connect(on_spin_changed)
        spinbox.editingFinished.connect(
            lambda: self._apply_hardware_setting(
                lambda: set_fn(spinbox.value())  # type: ignore[operator]
            )
        )
        slider.sliderReleased.connect(
            lambda: self._apply_hardware_setting(
                lambda: set_fn(spinbox.value())  # type: ignore[operator]
            )
        )

    # ------------------------------------------------------------------
    # Measurement Start / Stop / Clear
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        self.ui.btnStart.setChecked(True)
        self.ui.btnStart.setEnabled(False)
        self.ui.btnStop.setChecked(False)
        self.ui.cbDmaEnable.setEnabled(False)
        self.ui.btnDmaFile.setEnabled(False)
        try:
            if self.ui.cbDmaEnable.isChecked() and self._mca_dma is not None:
                self._start_with_dma()
            else:
                self._start_polling_only()
        except Exception:
            # Qt prints an uncaught slot exception and leaves Start disabled.
            # Restore a retryable, definitely-disarmed state instead.  The
            # driver accepts enable=0 even when a configuration write has
            # just failed with EBUSY.
            log.exception("MCA ch%d: measurement start failed", self._channel)
            self._stop_worker()
            try:
                self._mca.stop()
            except Exception:
                log.warning(
                    "MCA ch%d: failed to disarm after start error",
                    self._channel,
                    exc_info=True,
                )
            self._set_controls_enabled(True)
            self.ui.btnStart.setChecked(False)
            self.ui.btnStart.setEnabled(True)
            self.ui.btnStop.setChecked(False)
            self.ui.btnStop.setEnabled(False)
            self.ui.cbDmaEnable.setEnabled(True)
            self.ui.btnDmaFile.setEnabled(True)
            self._finish_psd_capture()

    def _start_polling_only(self) -> None:
        # A completed timed measurement clears measurement_in_progress but
        # leaves the pulse processor's enable ownership gate asserted.  Clear
        # it unconditionally before writing measurement_time_raw, otherwise
        # pp_field_store() returns EBUSY on the next run.
        self._mca.stop()
        self._mca.set_time_limit(self.ui.spinTimeLimit.value())
        self._mca.start()
        self.ui.btnStop.setEnabled(True)
        self._start_worker()
        log.info(
            "MCA ch%d: measurement started (time_limit=%d s)",
            self._channel,
            self.ui.spinTimeLimit.value(),
        )

    def _on_stop(self) -> None:
        self.ui.btnStop.setChecked(True)

        if self._dma_worker is not None:
            log.debug("MCA ch%d: stopping with DMA", self._channel)
            self._stop_worker()
            if isinstance(self._mca_dma, IIOMcaDmaStreamer):
                # Set the worker's stop flag first so it cannot begin a new
                # steady-state refill after the final block. Then stop the
                # pulse processor; either the in-flight refill receives the
                # padded final frame or streamer's close path performs the
                # same stop and one-second inactivity drain.
                self._dma_worker.stop()
                self._mca.stop()
            else:
                self._mca.stop()
                self._mca.set_dma_enable(False)
                self._dma_worker.stop()
            self._set_controls_enabled(True)
        else:
            self._stop_worker()
            self._mca.stop()

        self.ui.btnStart.setChecked(False)
        self.ui.btnStart.setEnabled(True)
        self.ui.btnStop.setEnabled(False)
        self.ui.cbDmaEnable.setEnabled(True)
        self.ui.btnDmaFile.setEnabled(True)
        log.info("MCA ch%d: measurement stopped", self._channel)

    def _on_measurement_done(self) -> None:
        """Called when the hardware stops the measurement (time limit reached).

        The worker has already stopped its own timer before emitting this signal.
        The driver's measurement_in_progress bit is only status; reaching the
        time limit does not release the separate enable ownership gate.  Stop
        explicitly before making Start available again.
        """
        log.info("MCA ch%d: measurement completed by hardware (time limit)", self._channel)
        if self._dma_worker is not None:
            if isinstance(self._mca_dma, IIOMcaDmaStreamer):
                # Match the manual-stop ordering: prevent another refill,
                # then release enable so the close path can drain the tail.
                self._dma_worker.stop()
                self._mca.stop()
            else:
                self._mca.stop()
                self._mca.set_dma_enable(False)
                self._dma_worker.stop()
            self._set_controls_enabled(True)
        else:
            self._mca.stop()
        self.ui.btnStart.setChecked(False)
        self.ui.btnStart.setEnabled(True)
        self.ui.btnStop.setChecked(True)
        self.ui.btnStop.setEnabled(False)
        self.ui.cbDmaEnable.setEnabled(True)
        self.ui.btnDmaFile.setEnabled(True)

    def _on_clear_spectrum(self) -> None:
        self._mca.clear_spectrum()
        self._hist_curve.setData([], [])
        self._last_histogram = None
        if self._roi.isVisible() and not self._roi_dragging:
            self._update_roi_stats()

    def _on_export_csv(self) -> None:
        if self._last_histogram is None or len(self._last_histogram) == 0:
            log.warning("MCA ch%d: no spectrum to export", self._channel)
            return

        default_dir = str(QSettings().value("dma/save_folder", "measurements"))
        ts = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"{default_dir}/ch{self._channel}_spectrum_{ts}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Spectrum CSV",
            default_name,
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        with open(path, "w", newline="") as f:
            f.write(f"# MCA channel {self._channel} spectrum export\n")
            f.write(f"# elapsed_s={self._last_elapsed_s:.1f}\n")
            if self._roi.isVisible():
                for line in self.ui.lblRoiStats.text().splitlines():
                    f.write(f"# {line}\n")
            f.write("channel,counts\n")
            for ch, counts in enumerate(self._last_histogram):
                f.write(f"{ch},{int(counts)}\n")

        log.info("MCA ch%d: spectrum exported to %s", self._channel, path)

    # ------------------------------------------------------------------
    # Polling worker lifecycle (gRPC histogram/stats)
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        interval_ms = 1000 // self.ui.spinRefreshRate.value()
        self._worker = MCAWorker(self._mca, interval_ms=interval_ms)
        self._worker_thread = QThread(self)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.readback.connect(self._on_readback)
        self._worker.measurement_done.connect(self._on_measurement_done)
        self._worker.finished.connect(
            self._worker_thread.quit,
            Qt.ConnectionType.DirectConnection,
        )
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(self._on_worker_finished)

        self._worker_thread.start()

    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.request_stop.emit()

    def _on_worker_finished(self) -> None:
        self._worker = None
        self._worker_thread = None

    def _on_refresh_rate_changed(self, value: int) -> None:
        if self._worker is not None:
            self._worker.change_interval.emit(1000 // value)

    def stop_worker_sync(self) -> None:
        """Blocking stop for use during application shutdown only."""
        worker = self._worker
        thread = self._worker_thread
        if worker is not None:
            worker.request_stop.emit()
        if thread is not None:
            if not thread.wait(3000):
                log.warning("MCA worker thread did not stop in time, terminating")
                thread.terminate()
                thread.wait()
        self._worker_thread = None
        self._worker = None

    # ------------------------------------------------------------------
    # DMA listmode recording
    # ------------------------------------------------------------------

    def _generate_filepath(self) -> Path:
        folder = Path(QSettings().value("dma/save_folder", "measurements"))
        folder.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._dma_counter += 1
        name = f"ch{self._channel}_{ts}_{self._dma_counter:03d}.bin"
        filepath = folder / name
        log.info("MCA DMA: auto-generated filepath: %s", filepath)
        return filepath

    def _on_dma_file(self) -> None:
        default_dir = str(QSettings().value("dma/save_folder", "measurements"))
        path, _ = QFileDialog.getSaveFileName(
            self,
            "MCA DMA File",
            default_dir,
            "Binary files (*.bin);;All files (*)",
        )
        if path:
            self._dma_filepath = Path(path)
            log.info("MCA DMA: user selected filepath: %s", self._dma_filepath)

    def _start_with_dma(self) -> None:
        filepath = self._dma_filepath or self._generate_filepath()
        self._dma_filepath = None
        self._event_buffer.clear()
        log.debug("MCA ch%d DMA [1/6]: creating worker, file=%s", self._channel, filepath)

        # Both the pulse-processor configuration path and lm_buffer_preenable
        # require enable=0.  This also recovers a stale timed acquisition
        # before set_time_limit() below touches measurement_time_raw.
        self._mca.stop()

        display_buffer = self._prepare_psd_capture()

        if isinstance(self._mca_dma, IIOMcaDmaStreamer):
            # Pulse-processor fields are immutable while list_buffer_active
            # is set, so apply the duration before the worker's first read
            # creates/arms the lm_frame buffer.
            self._mca.set_time_limit(self.ui.spinTimeLimit.value())
            self._dma_worker = IIOMcaDmaWorker(
                streamer=self._mca_dma,
                filepath=filepath,
                event_buffer=display_buffer,
            )
        else:
            assert isinstance(self._mca_dma, McaDmaStreamer)
            self._dma_worker = McaDmaWorker(
                streamer=self._mca_dma,
                filepath=filepath,
                event_buffer=display_buffer,
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
        self.ui.lblDmaStatus.setText("Connecting...")
        if isinstance(self._mca_dma, IIOMcaDmaStreamer):
            log.debug(
                "MCA ch%d DMA [2/6]: starting worker thread (IIO buffer arm)",
                self._channel,
            )
        else:
            log.debug(
                "MCA ch%d DMA [2/6]: starting worker thread (ZMQ connect + subscribe)",
                self._channel,
            )
        self._dma_thread.start()
        readiness = "IIO DMA arm" if isinstance(self._mca_dma, IIOMcaDmaStreamer) else "ZMQ socket"
        log.info(
            "MCA DMA: worker started, waiting for %s, file=%s",
            readiness,
            filepath,
        )

    def _prepare_psd_capture(self) -> McaEventBuffer | None:
        """Arm display-only event interception when Charge Comparison is on."""
        self._psd_capture_enabled = bool(
            self._psd_capture is not None and self.ui.cbCcEnable.isChecked()
        )
        if self._psd_capture is not None:
            self._psd_capture.begin_capture(
                self._psd_capture_enabled,
                (
                    "Recording live PSD."
                    if self._psd_capture_enabled
                    else "PSD inactive: enable Charge Comparison before starting DMA."
                ),
            )
        return self._event_buffer if self._psd_capture_enabled else None

    def _on_dma_ready(self) -> None:
        if isinstance(self._mca_dma, IIOMcaDmaStreamer):
            # The backend emits ready only after it has armed all eight
            # kernel blocks, entered the first blocking refill and written
            # enable=1. A GUI-thread mca.start() here would duplicate backend
            # ownership of that lifecycle.
            log.debug(
                "MCA ch%d IIO DMA: buffer armed, reader active, measurement started",
                self._channel,
            )
        else:
            log.debug(
                "MCA ch%d DMA [3/6]: ZMQ socket ready, DMA already enabled via checkbox",
                self._channel,
            )
            log.debug(
                "MCA ch%d DMA [4/6]: calling mca.start() -> set_global_enable(True) "
                "(HW fires list_start_irq -> server sends StreamSTART)",
                self._channel,
            )
            self._mca.set_time_limit(self.ui.spinTimeLimit.value())
            self._mca.start()
        log.debug(
            "MCA ch%d DMA: starting polling worker (time_limit=%d s)",
            self._channel,
            self.ui.spinTimeLimit.value(),
        )
        self._start_worker()
        self.ui.btnStop.setEnabled(True)
        self.ui.lblDmaStatus.setText("Recording...")
        log.info("MCA ch%d: DMA + measurement started", self._channel)

    def _on_dma_progress(self, event_count: int) -> None:
        unit = "records" if isinstance(self._mca_dma, IIOMcaDmaStreamer) else "events"
        self.ui.lblDmaStatus.setText(f"Recording: {event_count} {unit}")

    def _on_dma_error(self, message: str) -> None:
        log.error("MCA DMA error: %s", message)
        self.ui.lblDmaStatus.setText(f"Error: {message}")
        if self._psd_capture is not None:
            self._psd_capture.set_capture_error(message)

    def _on_dma_finished(self) -> None:
        self._dma_worker = None
        self._dma_thread = None
        self.ui.lblDmaStatus.setText("Stopped")
        self._finish_psd_capture()
        log.info("MCA DMA: worker finished")

    def _finish_psd_capture(self) -> None:
        if self._psd_capture is not None:
            self._psd_capture.finish_capture()
        self._psd_capture_enabled = False

    def stop_dma_sync(self) -> None:
        """Blocking stop for use during application shutdown/reconnect only.

        Handles both halves of what the UI's own Stop button does (see
        _on_stop()): an in-progress DMA worker (if any), and -- regardless
        of whether DMA was involved -- the hardware measurement-enable bit
        itself, via mca.stop()/set_dma_enable(False). Mirrors the same
        fix in ScopeController.stop_dma_sync(): closing the app (or
        reconnecting) without clicking Stop first used to leave a
        measurement armed indefinitely, since nothing in the shutdown
        path called mca.stop() for that case.
        """
        worker = self._dma_worker
        thread = self._dma_thread
        if worker is not None:
            worker.stop()
            if isinstance(self._mca_dma, IIOMcaDmaStreamer):
                # Unblock a possibly empty partial frame so the worker can
                # receive the final padded block and run its drain/close.
                try:
                    self._mca.stop()
                except Exception:
                    log.warning(
                        "MCA ch%d: failed to stop before IIO DMA shutdown",
                        self._channel,
                        exc_info=True,
                    )
        if thread is not None:
            if not thread.wait(3000):
                log.warning("MCA DMA thread did not stop in time, terminating")
                thread.terminate()
                thread.wait()
        had_dma_worker = worker is not None
        self._dma_thread = None
        self._dma_worker = None

        self._ensure_disarmed(had_dma_worker)
        self._finish_psd_capture()

    def _ensure_disarmed(self, had_dma_worker: bool) -> None:
        """Best-effort mca.stop()/set_dma_enable(False) for shutdown or
        reconnect -- see stop_dma_sync()'s docstring for why this exists.
        """
        try:
            # Stop unconditionally. An externally armed channel may have
            # measurement_in_progress=0 while global enable remains one;
            # testing the former was what left hardware owned across restart.
            self._mca.stop()
            if had_dma_worker:
                self._mca.set_dma_enable(False)
        except Exception:
            log.warning("MCA ch%d: failed to disarm during shutdown", self._channel, exc_info=True)

    # ------------------------------------------------------------------
    # Readback handling (gRPC polling)
    # ------------------------------------------------------------------

    def _on_readback(self, rb: MCAReadback) -> None:
        """Coalesce worker readbacks instead of rendering every signal.

        This slot runs in the GUI thread. At 30 Hz, directly calling three
        PlotDataItem.setData() methods here kept Qt continuously busy with
        readback/paint events and starved ordinary clicks even though the
        plots themselves appeared current. Replacing the pending value is
        O(1); a single-shot timer renders only the newest snapshot at the
        capped presentation rate, so stale frames never form a queue.
        """
        self._pending_readback = rb
        if not self._render_timer.isActive():
            requested_hz = self.ui.spinRefreshRate.value()
            render_hz = min(requested_hz, _MAX_GUI_RENDER_HZ)
            self._render_timer.start(max(1, 1000 // render_hz))

    def _render_pending_readback(self) -> None:
        rb = self._pending_readback
        self._pending_readback = None
        if rb is None:
            return

        self._update_debug_plot(rb.debug1, rb.debug2)
        # Update elapsed time before ROI statistics, which are invoked from
        # _update_histogram(); the old ordering displayed CPS using the
        # previous readback's elapsed value.
        self._update_statistics(rb)
        self._update_histogram(rb.histogram)

    def _update_debug_plot(self, raw_debug1: np.ndarray, raw_debug2: np.ndarray) -> None:
        samples = self.ui.sliderFrameSamples.value()
        self._update_debug_time_axis(samples)
        if raw_debug1 is not None and len(raw_debug1) > 0:
            debug1 = raw_debug1[: samples // _DEBUG_SAMPLE_PERIOD_NS]
            t1 = (
                np.arange(len(debug1))
                * _DEBUG_SAMPLE_PERIOD_NS
                / self._debug_time_scale.ns_per_unit
            )
            self._debug1_curve.setData(t1, debug1)
        if raw_debug2 is not None and len(raw_debug2) > 0:
            debug2 = raw_debug2[: samples // _DEBUG_SAMPLE_PERIOD_NS]
            t2 = (
                np.arange(len(debug2))
                * _DEBUG_SAMPLE_PERIOD_NS
                / self._debug_time_scale.ns_per_unit
            )
            self._debug2_curve.setData(t2, debug2)

    def _update_histogram(self, histogram: np.ndarray) -> None:
        if histogram is None or len(histogram) == 0:
            return
        self._last_histogram = histogram
        channels = np.arange(len(histogram) + 1)
        self._hist_curve.setData(channels, histogram)
        if self._roi.isVisible() and not self._roi_dragging:
            self._update_roi_stats()

    def _update_statistics(self, rb: MCAReadback) -> None:
        self._last_elapsed_s = rb.elapsed_time / 10.0
        self.ui.lblCountRate.setText(str(rb.count_rate))
        self.ui.lblDeadTime.setText(f"{rb.pulse_deadtime_ms:.6f}")
        self.ui.lblElapsedTime.setText(f"{rb.elapsed_time / 10:.1f}")
        self.ui.lblEventsLost.setText("N/A" if rb.events_lost is None else str(rb.events_lost))
        self.ui.lblPulsePileup.setText(str(rb.pulse_pileup))
        self.ui.lblPulseOverrange.setText(str(rb.pulse_overrange))
        self.ui.lblEnergyOverrange.setText(str(rb.energy_overrange))
        self.ui.lblEnergyEstErr.setText(str(rb.energy_estimation_error))
        self.ui.lblThroughputErr.setText(str(rb.throughput_error))
