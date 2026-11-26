"""Dedicated offline Current capture browser; never opens a hardware client."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.current_capture import (
    MAX_MOVING_AVERAGE_KERNEL,
    CurrentCaptureInfo,
    CurrentChunk,
)
from nlab.analysis.sum_spectrum import (
    MAX_SEGMENT_LENGTH,
    MIN_SEGMENT_LENGTH,
    SumSpectrum,
    sum_interval_ns,
    window_mean_response,
)
from nlab.analysis.waveform_spectrum import (
    UNIT_LABELS,
    SpectrumUnit,
    Spur,
    as_density,
    find_spurs,
    format_frequency,
    line_threshold_db,
    parseval_check,
    plotted_bins,
    to_display_units,
)
from nlab.views import theme
from nlab.views.frequency_axis import set_frequency_axis
from nlab.views.numeric_controls import keep_enter_in_fields
from nlab.views.plot_viewbox import ModifierZoomViewBox
from nlab.views.spectrum_lines import SpectrumLinesTable
from nlab.views.window_flags import make_tool_window
from nlab.workers.current_capture_worker import CurrentCaptureWorker
from nlab.workers.sum_spectrum_worker import SumSpectrumWorker

_TITLE = "Current Capture Analysis"
_DEFAULT_SEGMENT_LENGTH = 256
_NOT_VALIDATED = "Software analysis; not validated on live hardware."


class CurrentCaptureDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        make_tool_window(self)
        self.setWindowTitle(_TITLE)
        self.resize(1180, 850)
        self._path: Path | None = None
        self._info: CurrentCaptureInfo | None = None
        self._chunk: CurrentChunk | None = None
        self._worker: CurrentCaptureWorker | None = None
        self._generation = 0
        self._spectrum_worker: SumSpectrumWorker | None = None
        self._spectrum_generation = 0
        self._spectrum_result: SumSpectrum | None = None
        root = QVBoxLayout(self)
        menu_bar = QMenuBar(self)
        root.setMenuBar(menu_bar)
        file_menu = menu_bar.addMenu("&File")
        self.open_action = QAction("&Open capture…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        file_menu.addAction(self.open_action)
        self.export_action = QAction("&Export configuration YAML…", self)
        self.export_action.setEnabled(False)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        self.close_action = QAction("&Close", self)
        self.close_action.setShortcut(QKeySequence.StandardKey.Close)
        file_menu.addAction(self.close_action)
        self.file_label = QLabel("Open a periodic-sum capture (File > Open) to begin.")
        self.file_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self.file_label)
        self.capture_status = QLabel()
        self.capture_status.setWordWrap(True)
        self.capture_status.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self.capture_status)
        navigation = QHBoxLayout()
        self.first_button = QPushButton("First")
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.last_button = QPushButton("Last")
        self.start_spin = self._time_spin(0, 0)
        self.width_spin = self._time_spin(0.008, 1e12)
        self.width_spin.setValue(1000)
        for button in (self.first_button, self.previous_button):
            navigation.addWidget(button)
        navigation.addWidget(QLabel("Start:"))
        navigation.addWidget(self.start_spin)
        navigation.addWidget(QLabel("View size:"))
        navigation.addWidget(self.width_spin)
        for button in (self.next_button, self.last_button):
            navigation.addWidget(button)
        navigation.addStretch()
        root.addLayout(navigation)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        data_page = QWidget()
        data_layout = QVBoxLayout(data_page)
        averaging_controls = QHBoxLayout()
        self.average_checkbox = QCheckBox("Moving average")
        self.average_checkbox.setChecked(True)
        self.average_kernel_spin = QSpinBox()
        self.average_kernel_spin.setLocale(QLocale.c())
        self.average_kernel_spin.setRange(1, MAX_MOVING_AVERAGE_KERNEL)
        self.average_kernel_spin.setValue(16)
        self.average_kernel_spin.setSuffix(" samples")
        self.average_kernel_spin.setKeyboardTracking(False)
        self.average_kernel_spin.setToolTip(
            "Trailing arithmetic mean of K consecutive recorded samples in this chunk. "
            "Each sample is one measured window mean. "
            "A full kernel is required; averaging resets at acquisition gaps. "
            "Calculated before preview sampling. Statistics use the raw data."
        )
        averaging_controls.addWidget(self.average_checkbox)
        averaging_controls.addWidget(QLabel("Kernel size:"))
        averaging_controls.addWidget(self.average_kernel_spin)
        averaging_controls.addWidget(QLabel("Trailing average; resets at gaps."))
        averaging_controls.addStretch()
        data_layout.addLayout(averaging_controls)
        self.trace_plot = pg.PlotWidget(viewBox=ModifierZoomViewBox())
        self.trace_plot.setBackground(theme.colors().plot_background)
        self.trace_plot.showGrid(x=True, y=True, alpha=0.2)
        self.trace_plot.setLabel("bottom", "Time from chunk start", units="µs")
        self.trace_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.trace_plot.setLabel("left", "Window mean", units="ADC codes")
        self.trace_plot.getAxis("left").enableAutoSIPrefix(False)
        self.trace_plot.addLegend()
        self.trace = self.trace_plot.plot(
            pen=pg.mkPen("#277da1", width=1),
            name="Raw window means",
        )
        self.average_trace = self.trace_plot.plot(
            pen=pg.mkPen("#e07800", width=2),
            name="Moving average",
        )
        plots = QHBoxLayout()
        plots.setSpacing(0)
        plots.addWidget(self.trace_plot, 3)
        data_layout.addLayout(plots, 1)
        stats = QGridLayout()
        self.stats: dict[str, QLabel] = {}
        for index, name in enumerate(
            (
                "Windows",
                "Average",
                "Std (population)",
                "RMS",
                "Minimum",
                "Maximum",
                "Peak to peak",
                "Adjacent noise",
                "Measured exposure",
                "Skipped windows",
            )
        ):
            row, col = divmod(index, 3)
            label = QLabel(f"{name}: —")
            label.setWordWrap(True)
            self.stats[name] = label
            stats.addWidget(label, row, col)
        self.stats["Adjacent noise"].setToolTip(
            "sqrt(mean((y[i+1] - y[i])²) / 2), using only consecutive scheduled windows. "
            "A noise estimate for stationary, uncorrelated window means; pulses, drift, "
            "and correlations affect it. It does not measure noise within an ADC window."
        )
        self.stats["Std (population)"].setToolTip(
            "sqrt(mean((y - mean(y))²)), ddof=0, over every window mean in this chunk."
        )
        data_layout.addLayout(stats)
        self.distribution_plot = pg.PlotWidget(viewBox=ModifierZoomViewBox())
        self.distribution_plot.setBackground(theme.colors().plot_background)
        self.distribution_plot.setLabel("bottom", "Counts")
        self.distribution_plot.hideAxis("left")
        self.distribution_plot.showGrid(x=True, y=True, alpha=0.2)
        self.distribution_plot.setYLink(self.trace_plot)
        self.distribution_plot.setToolTip(
            "Distribution of all raw window means in this chunk. "
            "Amplitude shares the signal plot's Y axis; Counts gives the number of samples."
        )
        self.distribution = pg.PlotCurveItem(
            fillLevel="enclosed",
            brush=pg.mkBrush(39, 125, 161, 60),
            pen=pg.mkPen("#277da1"),
        )
        self.distribution_plot.addItem(self.distribution)
        plots.addWidget(self.distribution_plot, 1)
        self.mean_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#c0392b", width=1.5, style=Qt.PenStyle.DashLine),
            label="Chunk mean: {value:.6g} ADC codes",
            labelOpts={
                "position": 0.98,
                "anchors": [(1, 0), (1, 1)],
                "color": "#c0392b",
                "fill": (248, 249, 250, 220),
            },
        )
        self.distribution_mean_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#c0392b", width=1.5, style=Qt.PenStyle.DashLine),
        )
        for plot, line in (
            (self.trace_plot, self.mean_line),
            (self.distribution_plot, self.distribution_mean_line),
        ):
            line.setZValue(20)
            plot.addItem(line, ignoreBounds=True)
            line.hide()
        note = QLabel(
            "Statistics and distribution use all window means in the chunk. "
            "Adjacent noise uses Δ/√2 and excludes missing windows. "
            "These aggregates cannot recover the noise of individual ADC samples."
        )
        note.setWordWrap(True)
        data_layout.addWidget(note)
        self.tabs.addTab(data_page, "Trace and distribution")
        self.tabs.addTab(self._build_spectrum_page(), "Spectrum")
        config_page = QWidget()
        config_layout = QVBoxLayout(config_page)
        self.configuration_status = QLabel("No capture loaded.")
        self.configuration_status.setWordWrap(True)
        config_layout.addWidget(self.configuration_status)
        self.configuration_text = QPlainTextEdit()
        self.configuration_text.setReadOnly(True)
        config_layout.addWidget(self.configuration_text)
        self.export_button = QPushButton("Export configuration YAML…")
        self.export_button.setEnabled(False)
        config_layout.addWidget(self.export_button)
        self.tabs.addTab(config_page, "Digitizer configuration")
        self.metadata_text = QPlainTextEdit()
        self.metadata_text.setReadOnly(True)
        self.tabs.addTab(self.metadata_text, "Capture metadata")
        self.status = QLabel("No capture loaded.")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.open_action.triggered.connect(self._choose_capture)
        self.close_action.triggered.connect(self.close)
        self.export_action.triggered.connect(self._export_configuration)
        self.export_button.clicked.connect(self._export_configuration)
        # Enter in the Start/View size spin boxes must only commit the value.
        keep_enter_in_fields(self)
        self.first_button.clicked.connect(lambda: self.start_spin.setValue(0))
        self.previous_button.clicked.connect(
            lambda: self.start_spin.setValue(self.start_spin.value() - self.width_spin.value())
        )
        self.next_button.clicked.connect(
            lambda: self.start_spin.setValue(self.start_spin.value() + self.width_spin.value())
        )
        self.last_button.clicked.connect(self._last_chunk)
        self.start_spin.valueChanged.connect(self._request_chunk)
        self.width_spin.valueChanged.connect(self._request_chunk)
        self.average_checkbox.toggled.connect(self._average_changed)
        self.average_kernel_spin.valueChanged.connect(self._average_changed)
        self._update_navigation()

    # ------------------------------------------------------------------
    # Spectrum of the window means (whole capture, background thread)
    # ------------------------------------------------------------------

    def _build_spectrum_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.segment_combo = QComboBox()
        length = MIN_SEGMENT_LENGTH
        while length <= MAX_SEGMENT_LENGTH:
            self.segment_combo.addItem(f"{length:,} windows", length)
            length *= 2
        self.segment_combo.setCurrentIndex(self.segment_combo.findData(_DEFAULT_SEGMENT_LENGTH))
        self.segment_combo.setToolTip(
            "Welch segment length L in windows. Δf = fs'/L, and every segment needs L "
            "contiguous windows: missing windows end a run."
        )
        self.resolution_label = QLabel()
        self.bridge_checkbox = QCheckBox("Bridge isolated single missing windows")
        self.bridge_checkbox.setToolTip(
            "Replace each single missing window by the mean of its neighbours so runs join. "
            "Invents data: exact far below fs'/2, wrong near it. Off by default; counted."
        )
        self.spectrum_unit_combo = QComboBox()
        for unit in SpectrumUnit:
            self.spectrum_unit_combo.addItem(UNIT_LABELS[unit], unit.value)
        self.spectrum_log_checkbox = QCheckBox("Log frequency")
        self.spectrum_button = QPushButton("Compute spectrum")
        self.spectrum_button.setEnabled(False)
        controls.addWidget(QLabel("Segment length:"))
        controls.addWidget(self.segment_combo)
        controls.addWidget(self.resolution_label)
        controls.addWidget(self.bridge_checkbox)
        controls.addWidget(QLabel("Units:"))
        controls.addWidget(self.spectrum_unit_combo)
        controls.addWidget(self.spectrum_log_checkbox)
        controls.addWidget(self.spectrum_button)
        controls.addStretch()
        layout.addLayout(controls)
        self.spectrum_plot = pg.PlotWidget(viewBox=ModifierZoomViewBox())
        self.spectrum_plot.setBackground(theme.colors().plot_background)
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.2)
        set_frequency_axis(self.spectrum_plot.getPlotItem(), log=False)
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen("#277da1", width=1))
        self.spectrum_markers = self.spectrum_plot.plot(
            pen=None, symbol="t", symbolSize=9,
            symbolBrush=pg.mkBrush("#c0563b"), symbolPen=None,
        )
        # Without a spectrum, explain why and offer the fix instead of empty axes.
        message_page = QWidget()
        message_layout = QVBoxLayout(message_page)
        message_layout.addStretch()
        self.spectrum_message = QLabel()
        self.spectrum_message.setWordWrap(True)
        self.spectrum_message.setTextFormat(Qt.TextFormat.PlainText)
        self.spectrum_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_layout.addWidget(self.spectrum_message)
        actions = QHBoxLayout()
        actions.addStretch()
        self.bridge_recompute_button = QPushButton("Bridge single gaps and recompute")
        self.bridge_recompute_button.setToolTip(self.bridge_checkbox.toolTip())
        self.shorter_segment_button = QPushButton()
        for button in (self.bridge_recompute_button, self.shorter_segment_button):
            button.hide()
            actions.addWidget(button)
        actions.addStretch()
        message_layout.addLayout(actions)
        message_layout.addStretch()
        self.spectrum_stack = QStackedWidget()
        self.spectrum_stack.addWidget(self.spectrum_plot)
        self.spectrum_stack.addWidget(message_page)
        self.spectrum_lines = SpectrumLinesTable()
        plot_row = QHBoxLayout()
        plot_row.addWidget(self.spectrum_stack, 1)
        plot_row.addWidget(self.spectrum_lines)
        layout.addLayout(plot_row, 3)
        self.response_plot = pg.PlotWidget(viewBox=ModifierZoomViewBox())
        self.response_plot.setBackground(theme.colors().plot_background)
        self.response_plot.showGrid(x=True, y=True, alpha=0.2)
        self.response_plot.setLabel("bottom", "Frequency", units="Hz")
        self.response_plot.setLabel("left", "Window-mean |H| [dB]")
        self.response_plot.setToolTip(
            "Each value is the mean of W samples (a 2W ns boxcar) once per interval. "
            "Content above fs'/2 folds back into the spectrum with this attenuation."
        )
        self.response_curve = self.response_plot.plot(pen=pg.mkPen("#6c757d", width=1))
        self.nyquist_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#c0392b", width=1.5, style=Qt.PenStyle.DashLine),
            label="fs'/2: content above aliases",
            labelOpts={"position": 0.15, "color": "#c0392b",
                       "fill": (248, 249, 250, 220)},
        )
        self.response_plot.addItem(self.nyquist_line, ignoreBounds=True)
        self.nyquist_line.hide()
        layout.addWidget(self.response_plot, 1)
        self.spectrum_summary = QLabel("Open a capture, then compute its spectrum.")
        self.spectrum_summary.setWordWrap(True)
        self.spectrum_summary.setTextFormat(Qt.TextFormat.PlainText)
        self.spectrum_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.spectrum_summary)
        self.segment_combo.currentIndexChanged.connect(self._update_spectrum_controls)
        self.spectrum_unit_combo.currentIndexChanged.connect(self._show_spectrum)
        self.spectrum_log_checkbox.toggled.connect(self._show_spectrum)
        self.spectrum_button.clicked.connect(self.compute_spectrum)
        self.bridge_recompute_button.clicked.connect(self._bridge_and_recompute)
        self.shorter_segment_button.clicked.connect(self._use_reachable_length)
        self._show_spectrum()
        return page

    def _bridge_and_recompute(self) -> None:
        self.bridge_checkbox.setChecked(True)
        self.compute_spectrum()

    def _use_reachable_length(self) -> None:
        result = self._spectrum_result
        length = result.largest_reachable_length if result is not None else None
        if length is not None:
            self.segment_combo.setCurrentIndex(self.segment_combo.findData(length))
            self.compute_spectrum()

    def _show_spectrum_message(self, text: str, result: SumSpectrum | None = None) -> None:
        """Replace the plot by ``text`` and the actions that would make it computable."""
        self.spectrum_message.setText(text)
        bridge = bool(result is not None and result.single_gaps and not result.bridge_single_gaps)
        length = result.largest_reachable_length if result is not None else None
        self.bridge_recompute_button.setVisible(bridge)
        self.shorter_segment_button.setVisible(length is not None)
        if result is not None and length is not None:
            self.shorter_segment_button.setText(
                f"Use L = {length:,} (Δf = {format_frequency(result.fs / length)}) "
                "and recompute"
            )
        self.spectrum_stack.setCurrentIndex(1)

    def _update_spectrum_controls(self, *_args: object) -> None:
        info = self._info
        geometry = info.geometry if info is not None else None
        self.spectrum_button.setEnabled(geometry is not None)
        if self._spectrum_result is None and self._spectrum_worker is None:
            self._show_spectrum()  # The capture may have loaded since the last message.
        if geometry is None:
            self.resolution_label.setText("")
            return
        fs = 1e9 / sum_interval_ns(geometry.frame_samples, geometry.gap_cycles)
        length = int(self.segment_combo.currentData())
        self.resolution_label.setText(
            f"Δf = {format_frequency(fs / length)} (fs' = {format_frequency(fs)})"
        )

    def compute_spectrum(self) -> None:
        """Start a whole-capture spectrum; a newer request replaces this one."""
        if self._info is None:
            return
        self._cancel_spectrum()
        self._spectrum_generation += 1
        worker = SumSpectrumWorker(
            self._spectrum_generation,
            self._info,
            segment_length=int(self.segment_combo.currentData()),
            bridge_single_gaps=self.bridge_checkbox.isChecked(),
        )
        worker.computed.connect(self._spectrum_computed)
        worker.failed.connect(self._spectrum_failed)
        worker.progressed.connect(self._spectrum_progressed)
        self._spectrum_worker = worker
        self.spectrum_summary.setText("Reading every window mean in the capture…")
        if self._spectrum_result is None or self._spectrum_result.spectrum is None:
            self._show_spectrum_message("Computing the spectrum…")
        worker.start()

    def _cancel_spectrum(self) -> None:
        worker = self._spectrum_worker
        self._spectrum_worker = None
        if worker is not None:
            worker.cancel()
            worker.wait()

    def _spectrum_progressed(self, generation: int, percent: int) -> None:
        if generation == self._spectrum_generation:
            self.spectrum_summary.setText(f"Reading every window mean in the capture… {percent}%")

    def _spectrum_failed(self, generation: int, message: str) -> None:
        if generation == self._spectrum_generation:
            self._spectrum_result = None
            self._show_spectrum()
            self.spectrum_summary.setText(message)

    def _spectrum_computed(self, generation: int, result: SumSpectrum) -> None:
        if generation == self._spectrum_generation:
            self._spectrum_result = result
            self._show_spectrum()

    def _clear_spectrum(self) -> None:
        self._cancel_spectrum()
        self._spectrum_generation += 1
        self._spectrum_result = None
        self._show_spectrum()
        self.spectrum_summary.setText("Open a capture, then compute its spectrum.")
        self._update_spectrum_controls()

    def _show_spectrum(self, *_args: object) -> None:
        unit = SpectrumUnit(self.spectrum_unit_combo.currentData())
        log_frequency = self.spectrum_log_checkbox.isChecked()
        self.spectrum_plot.setLogMode(x=log_frequency, y=unit != SpectrumUnit.DB)
        set_frequency_axis(self.spectrum_plot.getPlotItem(), log=log_frequency)
        quantity = "ASD" if unit == SpectrumUnit.ASD else "PSD"
        self.spectrum_plot.setLabel("left", f"{quantity} of window means [{UNIT_LABELS[unit]}]")
        result = self._spectrum_result
        spectrum = result.spectrum if result is not None else None
        self.spectrum_curve.clear()
        self.spectrum_markers.clear()
        if result is None:
            self.response_curve.clear()
            self.nyquist_line.hide()
            self._show_spectrum_message(
                "Choose a segment length and press Compute spectrum."
                if self._info is not None else "Open a capture, then compute its spectrum."
            )
            self.spectrum_lines.clear()
            return
        if spectrum is None:
            self._show_spectrum_message(result.message, result)
            self.spectrum_lines.clear("Lines: no spectrum for this segment length.")
        else:
            self.spectrum_stack.setCurrentIndex(0)
        frequencies = np.linspace(0, result.fs, 2001)
        response = window_mean_response(frequencies, result.window_samples)
        self.response_curve.setData(
            frequencies, 20 * np.log10(np.maximum(response, 1e-6))
        )
        self.nyquist_line.setValue(result.fs / 2)
        self.nyquist_line.show()
        spurs: list[Spur] = []
        if spectrum is not None:
            # Drawn as a density: the even-N Nyquist bin doubled, DC omitted.
            values = to_display_units(as_density(spectrum.psd, spectrum.n), unit)
            bins = plotted_bins(spectrum.n)
            shown = values[bins]
            if unit != SpectrumUnit.DB:
                shown = np.maximum(shown, np.finfo(np.float64).tiny)
            self.spectrum_curve.setData(spectrum.frequencies[bins], shown)
            threshold = line_threshold_db(spectrum.averages)
            spurs = find_spurs(
                spectrum, threshold_db_over_median=threshold, max_count=20,
            ) if spectrum.frames >= 2 else []
            self.spectrum_lines.show_lines(spurs, unit, threshold)
            if spurs:
                marks = values[[spur.bin for spur in spurs]]
                if unit != SpectrumUnit.DB:
                    marks = np.maximum(marks, np.finfo(np.float64).tiny)
                self.spectrum_markers.setData([spur.frequency for spur in spurs], marks)
            self.spectrum_plot.enableAutoRange()
        self.spectrum_summary.setText(self._spectrum_text(result))

    @staticmethod
    def _spectrum_text(result: SumSpectrum) -> str:
        nyquist_db = 20 * math.log10(max(result.nyquist_response, 1e-12))
        lines = [
            f"fs' = {format_frequency(result.fs)} (interval {result.interval_ns:g} ns: "
            f"{result.window_samples} samples + gap {result.gap_cycles} cycles) · "
            f"L = {result.segment_length:,} · Δf = {format_frequency(result.delta_f)} · "
            "Hann, 50% overlap",
            f"Window mean = {2 * result.window_samples} ns boxcar, duty "
            f"{result.duty_cycle:.3f}: |H| = {result.nyquist_response:.3f} "
            f"({nyquist_db:.1f} dB) at fs'/2; content above fs'/2 aliases.",
            f"{result.windows:,} recorded windows; {result.missing_windows:,} missing in "
            f"{result.gaps:,} gaps ({result.single_gaps:,} single); {result.runs:,} gapless "
            f"runs, longest {result.longest_run:,}.",
            (
                f"Bridged {result.bridged_windows:,} isolated single missing windows by "
                "linear interpolation (invented data; see Help)."
                if result.bridge_single_gaps else "Bridging off."
            ),
        ]
        spectrum = result.spectrum
        if spectrum is not None:
            lines.append(
                f"{result.segments:,} segments from {result.runs_used:,} runs · "
                f"{result.used_windows:,} windows used, {result.discarded_windows:,} "
                f"discarded · ENBW {format_frequency(spectrum.enbw)} · "
                f"±{100 * spectrum.relative_std:.1f}% per bin · "
                f"Parseval {parseval_check(spectrum):.3f}"
            )
        if result.message:
            lines.append(result.message)
        lines.append(_NOT_VALIDATED)
        return "\n".join(lines)

    def _set_export_enabled(self, enabled: bool) -> None:
        self.export_button.setEnabled(enabled)
        self.export_action.setEnabled(enabled)

    @staticmethod
    def _time_spin(minimum: float, maximum: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setLocale(QLocale.c())
        spin.setDecimals(3)
        spin.setRange(minimum, maximum)
        spin.setSuffix(" µs")
        spin.setSingleStep(1)
        spin.setKeyboardTracking(False)
        return spin

    def show_workspace(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _choose_capture(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Current capture",
            "",
            "Current sum capture (*.bin *.h5 *.hdf5);;All files (*)",
        )
        if filename:
            self.open_path(Path(filename))

    def open_path(self, path: Path) -> None:
        self._path = Path(path)
        self._info = self._chunk = None
        self.file_label.setText(str(path))
        self.setWindowTitle(f"{Path(path).name} — {_TITLE}")
        self.capture_status.clear()
        self.configuration_text.clear()
        self.metadata_text.clear()
        self.configuration_status.setText("Reading capture configuration…")
        self._set_export_enabled(False)
        self._clear_chunk_plots()
        self._clear_spectrum()
        self.start_spin.blockSignals(True)
        self.start_spin.setValue(0)
        self.start_spin.blockSignals(False)
        self._request_chunk()

    def _average_changed(self, *_args: object) -> None:
        enabled = self.average_checkbox.isChecked()
        self.average_kernel_spin.setEnabled(enabled)
        self.average_trace.clear()
        self.average_trace.setVisible(enabled)
        self._request_chunk()

    def _request_chunk(self, *_args: object) -> None:
        if self._path is None:
            return
        if self._worker is None:
            self._worker = CurrentCaptureWorker()
            self._worker.loaded.connect(self._loaded)
            self._worker.failed.connect(self._failed)
            self._worker.start()
        self._generation += 1
        self._worker.submit(
            self._generation,
            self._path,
            self.start_spin.value(),
            self.width_spin.value(),
            average_kernel=self.average_kernel_spin.value()
            if self.average_checkbox.isChecked()
            else 0,
        )
        self.status.setText("Loading selected chunk…")
        self._update_navigation()

    def _last_chunk(self) -> None:
        if self._info is not None:
            width = self.width_spin.value()
            page = max(0, math.ceil(self._info.duration_us / width) - 1)
            self.start_spin.setValue(page * width)

    def _update_navigation(self) -> None:
        start, width = self.start_spin.value(), self.width_spin.value()
        duration = self._info.duration_us if self._info else 0
        self.first_button.setEnabled(start > 0)
        self.previous_button.setEnabled(start > 0)
        self.next_button.setEnabled(start + width < duration)
        self.last_button.setEnabled(start + width < duration)

    def _loaded(self, generation: int, info: CurrentCaptureInfo, chunk: CurrentChunk) -> None:
        if generation != self._generation:
            return
        self._info, self._chunk = info, chunk
        self._update_spectrum_controls()
        self.start_spin.blockSignals(True)
        self.start_spin.setMaximum(max(0, info.duration_us - 0.008))
        self.start_spin.blockSignals(False)
        self.capture_status.setText(
            f"Measured periodic sums ({'binary' if info.container == 'nsum' else 'HDF5'}) · "
            f"source channel {info.metadata.get('channel', '?')} · "
            f"{info.windows:,} windows · {info.duration_us:,.3f} µs · "
            f"Complete={info.complete} · final hardware missed: {info.final_missed_windows:,}"
            + (f" · {info.cleanup_error}" if info.cleanup_error else "")
        )
        self.configuration_text.setPlainText(info.configuration_yaml)
        self.configuration_status.setText(
            "Full configuration saved before acquisition. View or export it here; "
            "readback does not apply settings to hardware."
            if info.configuration_yaml
            else "No embedded digitizer configuration. Older files remain readable."
        )
        self._set_export_enabled(bool(info.configuration_yaml))
        self.metadata_text.setPlainText(
            json.dumps(
                {
                    "capture": info.metadata,
                    "runtime": info.runtime_metadata,
                    "first_native_timestamp_8ns": info.first_timestamp,
                    "complete": info.complete,
                    "cleanup_error": info.cleanup_error,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        self.trace.setData(chunk.x_us, chunk.values, connect=chunk.connect)
        self.average_trace.setData(chunk.x_us, chunk.average_values, connect=chunk.connect)
        # A kernel as long as its run has one valid average: show its endpoint.
        finite_averages = int(np.count_nonzero(np.isfinite(chunk.average_values)))
        self.average_trace.setSymbol("o" if finite_averages == 1 else None)
        self.average_trace.setSymbolPen("#e07800")
        self.average_trace.setSymbolBrush("#e07800")
        self.average_trace.setVisible(bool(chunk.average_kernel))
        self.trace_plot.setXRange(0, self.width_spin.value(), padding=0)
        self.trace_plot.enableAutoRange(axis="y")
        self._set_distribution(chunk)
        self.distribution_plot.setXRange(
            0,
            max(1, int(chunk.histogram_counts.max())) * 1.05,
            padding=0,
        )
        for line in (self.mean_line, self.distribution_mean_line):
            line.setVisible(chunk.mean is not None)
            if chunk.mean is not None:
                line.setValue(chunk.mean)

        def value(number: float | None) -> str:
            return "—" if number is None else f"{number:.6g} ADC codes"

        fields = {
            "Windows": f"{chunk.count:,}",
            "Average": value(chunk.mean),
            "Std (population)": value(chunk.std),
            "RMS": value(chunk.rms),
            "Minimum": value(chunk.minimum),
            "Maximum": value(chunk.maximum),
            "Peak to peak": value(chunk.maximum - chunk.minimum)
            if chunk.maximum is not None and chunk.minimum is not None
            else "—",
            "Adjacent noise": value(chunk.adjacent_noise),
            "Measured exposure": f"{chunk.count * info.geometry.frame_samples * 0.002:.3f} µs"
            if info.geometry
            else "0 µs",
            "Skipped windows": f"{chunk.skipped_windows:,}",
        }
        for name, text in fields.items():
            self.stats[name].setText(f"{name}: {text}")
        self.status.setText(
            f"[{chunk.start_us:.3f}, {chunk.end_us:.3f}) µs · "
            f"{len(chunk.values):,} plotted / {chunk.count:,} measured windows · "
            f"{chunk.adjacent_pairs:,} adjacent pairs. "
            "Loaded packets checked against raw DMA data; other chunks not scanned."
        )
        if chunk.average_kernel and not finite_averages:
            self.status.setText(
                self.status.text()
                + f" Average needs {chunk.average_kernel:,} consecutive samples in this chunk."
            )
        self._update_navigation()

    def _set_distribution(self, chunk: CurrentChunk | None) -> None:
        if chunk is None or not chunk.count:
            self.distribution.clear()
        else:
            # One closed, filled step contour in (counts, amplitude) coordinates.
            # Adjacent bins share only the outline, without individual bar edges.
            edges = chunk.histogram_edges
            counts = np.concatenate(([0], np.repeat(chunk.histogram_counts, 2), [0, 0]))
            amplitudes = np.concatenate(
                (
                    [edges[0]],
                    np.repeat(edges, 2)[1:-1],
                    [edges[-1], edges[0]],
                )
            )
            self.distribution.setData(counts, amplitudes)

    def _clear_chunk_plots(self) -> None:
        self.trace.clear()
        self.average_trace.clear()
        self._set_distribution(None)
        self.mean_line.hide()
        self.distribution_mean_line.hide()

    def _failed(self, generation: int, message: str) -> None:
        if generation == self._generation:
            self.status.setText(message)
            self._clear_chunk_plots()
            for name, label in self.stats.items():
                label.setText(f"{name}: —")

    def export_configuration(self, path: Path) -> None:
        if self._info is None or not self._info.configuration_yaml:
            raise ValueError("This capture has no embedded configuration")
        Path(path).write_text(self._info.configuration_yaml, encoding="utf-8")

    def _export_configuration(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export recorded digitizer configuration",
            "configuration.yaml",
            "YAML (*.yaml)",
        )
        if filename:
            try:
                self.export_configuration(Path(filename))
            except OSError as exc:
                QMessageBox.critical(self, "Configuration export failed", str(exc))

    def _stop_sync(self) -> None:
        self._generation += 1
        self._spectrum_generation += 1
        self._cancel_spectrum()
        worker = self._worker
        if worker is not None:
            worker.stop()
            worker.wait()
            self._worker = None

    def close_without_prompt(self) -> None:
        self._stop_sync()
        self.close()

    def reject(self) -> None:
        self._stop_sync()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._stop_sync()
        super().closeEvent(event)
