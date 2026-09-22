"""Standalone waveform browser and offline charge/PSD workbench."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt, QThread, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.waveform_file import MappedWaveformFile, WaveformFileIndex, WaveformFrame
from nlab.analysis.waveform_psd import (
    BaselineMethod,
    WaveformPsdResult,
    WaveformPsdSettings,
    infer_waveform_polarity,
    integrate_waveform,
)
from nlab.views.offline_psd_plot import OfflinePsdPlot
from nlab.views.plot_viewbox import ModifierZoomViewBox
from nlab.workers.waveform_file_worker import WaveformFileIndexWorker
from nlab.workers.waveform_psd_worker import WaveformPsdWorker

log = logging.getLogger(__name__)
_MAX_QT_INDEX = 2_147_483_647
_AUTO_PREVIEW_STRIDE_MULTIPLIER = 20
_AUTO_PREVIEW_DEBOUNCE_MS = 120
AnalysisKind = Literal["manual", "sparse", "full"]


class WaveformAnalysisDialog(QDialog):
    """Browse saved waveforms and reconstruct short/long-gate PSD offline."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Waveform Analysis Workbench")
        self.setModal(False)
        self.resize(1420, 900)
        self._index: WaveformFileIndex | None = None
        self._reader: MappedWaveformFile | None = None
        self._frame: WaveformFrame | None = None
        self._index_thread: QThread | None = None
        self._index_worker: WaveformFileIndexWorker | None = None
        self._analysis_thread: QThread | None = None
        self._analysis_worker: WaveformPsdWorker | None = None
        self._active_analysis_kind: AnalysisKind | None = None
        self._pending_analysis_kind: AnalysisKind | None = None
        self._last_sample_period = 1.0
        self._updating_gates = False
        self._auto_preview_timer = QTimer(self)
        self._auto_preview_timer.setSingleShot(True)
        self._auto_preview_timer.setInterval(_AUTO_PREVIEW_DEBOUNCE_MS)
        self._build_ui()
        self._fit_initial_size_to_screen()

    def _fit_initial_size_to_screen(self) -> None:
        """Keep the complete workbench inside the screen's usable desktop."""
        parent = self.parentWidget()
        screen = parent.screen() if parent is not None else QApplication.primaryScreen()
        if screen is None:
            self.resize(1420, 900)
            return
        available = screen.availableGeometry()
        self.resize(
            min(1420, max(1, available.width() - 32)),
            min(900, max(1, available.height() - 48)),
        )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        note = QLabel(
            "Waveforms are read without changing live Scope or PSD acquisition. Drag the "
            "baseline region and the three gate markers, inspect individual events, then "
            "integrate the selected source in a background worker.",
            self,
        )
        note.setWordWrap(True)
        root.addWidget(note)

        toolbar = QHBoxLayout()
        self.load_button = QPushButton("Load waveform file...", self)
        self.close_file_button = QPushButton("Close file", self)
        self.close_file_button.setEnabled(False)
        toolbar.addWidget(self.load_button)
        toolbar.addWidget(self.close_file_button)
        toolbar.addWidget(QLabel("Source:", self))
        self.source_combo = QComboBox(self)
        self.source_combo.setMinimumWidth(190)
        self.source_combo.setEnabled(False)
        toolbar.addWidget(self.source_combo)
        toolbar.addWidget(QLabel("Frame:", self))
        self.frame_spin = QSpinBox(self)
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setEnabled(False)
        toolbar.addWidget(self.frame_spin)
        toolbar.addWidget(QLabel("Sample period:", self))
        self.sample_period = QDoubleSpinBox(self)
        self.sample_period.setRange(0.001, 1_000_000.0)
        self.sample_period.setDecimals(4)
        self.sample_period.setValue(2.0)
        self.sample_period.setSuffix(" ns")
        self.sample_period.setEnabled(False)
        toolbar.addWidget(self.sample_period)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        controls = QWidget(self)
        controls.setMinimumWidth(300)
        controls.setMaximumWidth(370)
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(self._build_preprocessing_group())
        controls_layout.addWidget(self._build_histogram_group())
        controls_layout.addStretch(1)
        splitter.addWidget(controls)

        self.plot_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.waveform_widget = pg.PlotWidget(self, viewBox=ModifierZoomViewBox())
        self.waveform_widget.setMinimumHeight(200)
        self.waveform_widget.setBackground("#f8f9fa")
        self.waveform_plot = self.waveform_widget.getPlotItem()
        self.waveform_plot.showAxis("top")
        self.waveform_plot.showAxis("right")
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.2)
        self.waveform_plot.setLabel("bottom", "Time", units="ns")
        self.waveform_plot.setLabel("left", "ADC amplitude", units="raw")
        self.waveform_plot.addLegend(offset=(10, 10))
        self.raw_curve = self.waveform_plot.plot(
            pen=pg.mkPen("#1597d4", width=1.2), name="Stored waveform"
        )
        self.oriented_curve = self.waveform_plot.plot(
            pen=pg.mkPen("#f28e2b", width=1.2), name="Polarity-normalized"
        )
        self.baseline_region = pg.LinearRegionItem(
            values=(0.0, 1.0),
            orientation="vertical",
            brush=pg.mkBrush(80, 140, 220, 35),
        )
        self.baseline_region.setZValue(20.0)
        self.waveform_plot.addItem(self.baseline_region)
        self.gate_start_line = self._gate_line("#2a9d8f")
        self.short_end_line = self._gate_line("#e9c46a")
        self.long_end_line = self._gate_line("#e76f51")
        self.plot_splitter.addWidget(self.waveform_widget)

        self.psd_plot = OfflinePsdPlot(self)
        self.psd_plot.setMinimumHeight(360)
        self.plot_splitter.addWidget(self.psd_plot)
        self.plot_splitter.setStretchFactor(0, 1)
        self.plot_splitter.setStretchFactor(1, 3)
        self.plot_splitter.setSizes([220, 520])
        splitter.addWidget(self.plot_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        status_row = QHBoxLayout()
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(230)
        status_row.addWidget(self.progress)
        self.status = QLabel("Select a waveform file to begin.", self)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        status_row.addWidget(self.status, 1)
        self.close_button = QPushButton("Close", self)
        status_row.addWidget(self.close_button)
        root.addLayout(status_row)

        self.load_button.clicked.connect(self._choose_file)
        self.close_file_button.clicked.connect(self.close_file)
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.frame_spin.valueChanged.connect(self._show_frame)
        self.sample_period.editingFinished.connect(self._sample_period_changed)
        self.polarity_combo.currentIndexChanged.connect(self._preview)
        self.baseline_method.currentIndexChanged.connect(self._preview)
        self.baseline_region.sigRegionChanged.connect(self._on_gate_moved)
        self.baseline_region.sigRegionChangeFinished.connect(
            self._on_gate_change_finished
        )
        for line in (self.gate_start_line, self.short_end_line, self.long_end_line):
            line.sigPositionChanged.connect(self._on_gate_moved)
            line.sigPositionChangeFinished.connect(self._on_gate_change_finished)
        self._auto_preview_timer.timeout.connect(self._request_sparse_analysis)
        self.analyze_button.clicked.connect(self._start_analysis)
        self.cancel_button.clicked.connect(self._cancel_analysis)
        self.close_button.clicked.connect(self.reject)

    def _build_preprocessing_group(self) -> QGroupBox:
        group = QGroupBox("Preprocessing and integration", self)
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.baseline_method = QComboBox(self)
        self.baseline_method.addItem("Median (robust)", "median")
        self.baseline_method.addItem("Mean", "mean")
        self.polarity_combo = QComboBox(self)
        self.polarity_combo.addItem("Auto from displayed frame", 0)
        self.polarity_combo.addItem("Negative-going pulse", -1)
        self.polarity_combo.addItem("Positive-going pulse", 1)
        form.addRow("Baseline estimator:", self.baseline_method)
        form.addRow("Pulse polarity:", self.polarity_combo)
        layout.addLayout(form)
        help_label = QLabel(
            "Blue region: baseline. Green: integration start. Yellow: short-gate end. "
            "Red: long-gate end. Gate positions are stored as sample indices.",
            self,
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.preview_status = QLabel("No frame loaded.", self)
        self.preview_status.setWordWrap(True)
        self.preview_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.preview_status)
        return group

    def _build_histogram_group(self) -> QGroupBox:
        group = QGroupBox("PSD reconstruction", self)
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.energy_bins = QSpinBox(self)
        self.energy_bins.setRange(16, 8192)
        self.energy_bins.setValue(512)
        self.energy_max = QDoubleSpinBox(self)
        self.energy_max.setRange(1.0, 1.0e15)
        self.energy_max.setDecimals(1)
        self.energy_max.setValue(1_000_000.0)
        self.ratio_bins = QSpinBox(self)
        self.ratio_bins.setRange(16, 4096)
        self.ratio_bins.setValue(256)
        self.ratio_min = QDoubleSpinBox(self)
        self.ratio_min.setRange(-10.0, 10.0)
        self.ratio_min.setDecimals(4)
        self.ratio_min.setValue(-1.0)
        self.ratio_max = QDoubleSpinBox(self)
        self.ratio_max.setRange(-10.0, 10.0)
        self.ratio_max.setDecimals(4)
        self.ratio_max.setValue(1.0)
        self.event_stride = QSpinBox(self)
        self.event_stride.setRange(1, 1_000_000)
        self.event_stride.setValue(1)
        self.maximum_events = QSpinBox(self)
        self.maximum_events.setRange(0, _MAX_QT_INDEX)
        self.maximum_events.setSpecialValueText("All")
        form.addRow("Energy bins:", self.energy_bins)
        form.addRow("Energy maximum:", self.energy_max)
        form.addRow("Ratio bins:", self.ratio_bins)
        form.addRow("Ratio minimum:", self.ratio_min)
        form.addRow("Ratio maximum:", self.ratio_max)
        form.addRow("Event stride:", self.event_stride)
        form.addRow("Maximum events:", self.maximum_events)
        layout.addLayout(form)
        self.auto_recalculate = QCheckBox(
            "Auto-recalculate while dragging (every 20th event)",
            self,
        )
        self.auto_recalculate.setChecked(True)
        self.auto_recalculate.setToolTip(
            "Gate motion is debounced and recalculated sparsely. Releasing a marker "
            "queues a full calculation using the configured event stride."
        )
        layout.addWidget(self.auto_recalculate)
        buttons = QHBoxLayout()
        self.analyze_button = QPushButton("Reconstruct PSD", self)
        self.analyze_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setEnabled(False)
        buttons.addWidget(self.analyze_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        return group

    def _gate_line(self, color: str) -> pg.InfiniteLine:
        line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen(color, width=2),
            hoverPen=pg.mkPen(color, width=4),
        )
        line.setZValue(30.0)
        self.waveform_plot.addItem(line)
        return line

    def _choose_file(self) -> None:
        folder = str(QSettings().value("analysis/waveform_folder", "measurements"))
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Open Waveform File",
            folder,
            "Waveform binaries (*.bin *.BIN);;All files (*)",
        )
        if path_text:
            self.open_path(Path(path_text))

    def open_path(self, path: Path) -> None:
        if self._index_thread is not None or self._analysis_thread is not None:
            QMessageBox.warning(self, "Waveform Analysis", "Cancel the current operation first.")
            return
        self.close_file()
        worker = WaveformFileIndexWorker(path)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._index_progress)
        worker.loaded.connect(self._index_loaded)
        worker.cancelled.connect(self._index_cancelled)
        worker.error.connect(self._index_failed)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._index_thread_finished)
        self._index_worker = worker
        self._index_thread = thread
        self._set_indexing(True)
        self.status.setText(f"Indexing {path.name}...")
        self.progress.setValue(0)
        QSettings().setValue("analysis/waveform_folder", str(path.parent))
        thread.start()

    def _index_progress(self, processed: int, total: int) -> None:
        self.progress.setValue(round(1000 * processed / total) if total else 0)
        self.status.setText(f"Indexing waveform file: {processed:,}/{total:,} bytes...")

    def _index_loaded(self, value: object) -> None:
        if not isinstance(value, WaveformFileIndex):
            self._index_failed("Waveform index worker returned an invalid result")
            return
        self._index = value
        try:
            self._reader = MappedWaveformFile(value)
        except OSError as exc:
            self._index_failed(str(exc))
            return
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        for source in value.channels:
            if source.board is None:
                label = f"Channel {source.channel} ({source.frame_count:,} frames)"
            else:
                label = (
                    f"Board {source.board} / Channel {source.channel} "
                    f"({source.frame_count:,} frames)"
                )
            self.source_combo.addItem(label)
        self.source_combo.blockSignals(False)
        period = value.sample_period_ns or 2.0
        self.sample_period.setValue(period)
        self._last_sample_period = period
        self.sample_period.setEnabled(value.sample_period_ns is None)
        self.source_combo.setEnabled(len(value.channels) > 1)
        self.frame_spin.setEnabled(True)
        self.close_file_button.setEnabled(True)
        self.analyze_button.setEnabled(True)
        self.progress.setValue(1000)
        self.status.setText(
            f"Loaded {value.path.name}: {value.frame_count:,} frames, {value.format_name}."
        )
        self._source_changed()

    def _index_failed(self, message: str) -> None:
        self.status.setText(f"Waveform load failed: {message}")
        QMessageBox.critical(self, "Waveform File Load Failed", message)

    def _index_cancelled(self) -> None:
        self.status.setText("Waveform indexing cancelled.")

    def _index_thread_finished(self) -> None:
        thread = self._index_thread
        self._index_worker = None
        self._index_thread = None
        self._set_indexing(False)
        if thread is not None:
            thread.deleteLater()

    def _set_indexing(self, indexing: bool) -> None:
        self.load_button.setEnabled(not indexing)
        if indexing:
            self.close_file_button.setEnabled(False)

    def _source_changed(self) -> None:
        if self._index is None or self.source_combo.currentIndex() < 0:
            return
        source = self._index.channels[self.source_combo.currentIndex()]
        self.frame_spin.blockSignals(True)
        self.frame_spin.setRange(0, min(source.frame_count - 1, _MAX_QT_INDEX))
        self.frame_spin.setValue(0)
        self.frame_spin.blockSignals(False)
        self._show_frame(0, reset_gates=True)

    def _show_frame(self, _value: int = 0, *, reset_gates: bool = False) -> None:
        reader = self._reader
        if reader is None or self.source_combo.currentIndex() < 0:
            return
        try:
            self._frame = reader.frame(self.frame_spin.value(), self.source_combo.currentIndex())
        except (IndexError, OSError, ValueError) as exc:
            self.status.setText(f"Could not read waveform: {exc}")
            return
        samples = self._frame.samples
        period = self.sample_period.value()
        times = np.arange(len(samples), dtype=np.float64) * period
        self.raw_curve.setData(times, samples)
        maximum_time = max(period, len(samples) * period)
        previous_update_state = self._updating_gates
        self._updating_gates = True
        try:
            for item in (
                self.baseline_region,
                self.gate_start_line,
                self.short_end_line,
                self.long_end_line,
            ):
                item.setBounds((0.0, maximum_time))
            if reset_gates:
                self._set_default_gates(samples)
        finally:
            self._updating_gates = previous_update_state
        self._preview()
        self.waveform_plot.autoRange()

    def _set_default_gates(self, samples: np.ndarray) -> None:
        count = len(samples)
        if count < 5:
            return
        probe = max(2, min(64, count // 5))
        baseline = float(np.median(samples[:probe]))
        deviations = np.abs(samples.astype(np.float64) - baseline)
        peak = int(np.argmax(deviations))
        start = max(probe, peak - max(2, count // 100))
        start = min(start, count - 3)
        short_end = min(count - 1, start + max(2, count // 20))
        long_end = min(count, max(short_end + 1, start + max(4, count // 4)))
        baseline_end = max(1, min(start, peak - max(3, count // 50)))
        baseline_start = max(0, baseline_end - probe)
        period = self.sample_period.value()
        previous_update_state = self._updating_gates
        self._updating_gates = True
        try:
            self.baseline_region.setRegion((baseline_start * period, baseline_end * period))
            self.gate_start_line.setPos(start * period)
            self.short_end_line.setPos(short_end * period)
            self.long_end_line.setPos(long_end * period)
        finally:
            self._updating_gates = previous_update_state

    def _sample_period_changed(self) -> None:
        new_period = self.sample_period.value()
        old_period = self._last_sample_period
        if old_period > 0 and new_period != old_period:
            scale = new_period / old_period
            previous_update_state = self._updating_gates
            self._updating_gates = True
            try:
                low, high = self.baseline_region.getRegion()
                self.baseline_region.setRegion((low * scale, high * scale))
                for line in (
                    self.gate_start_line,
                    self.short_end_line,
                    self.long_end_line,
                ):
                    line.setPos(line.value() * scale)
            finally:
                self._updating_gates = previous_update_state
        self._last_sample_period = new_period
        self._show_frame(self.frame_spin.value())

    def _polarity(self) -> Literal[-1, 1]:
        selected = int(self.polarity_combo.currentData())
        if selected in {-1, 1}:
            return cast(Literal[-1, 1], selected)
        frame = self._frame
        if frame is None:
            return 1
        baseline_start, baseline_end, gate_start, _, _ = self._gate_indices()
        return infer_waveform_polarity(
            frame.samples,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            search_start=gate_start,
        )

    def _gate_indices(self) -> tuple[int, int, int, int, int]:
        period = self.sample_period.value()
        baseline = sorted(round(value / period) for value in self.baseline_region.getRegion())
        return (
            baseline[0],
            baseline[1],
            round(self.gate_start_line.value() / period),
            round(self.short_end_line.value() / period),
            round(self.long_end_line.value() / period),
        )

    def _settings(self) -> WaveformPsdSettings:
        baseline_start, baseline_end, gate_start, short_end, long_end = self._gate_indices()
        method = cast(BaselineMethod, str(self.baseline_method.currentData()))
        return WaveformPsdSettings(
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            gate_start=gate_start,
            short_end=short_end,
            long_end=long_end,
            polarity=self._polarity(),
            baseline_method=method,
            energy_bins=self.energy_bins.value(),
            energy_range=(0.0, self.energy_max.value()),
            ratio_bins=self.ratio_bins.value(),
            ratio_range=(self.ratio_min.value(), self.ratio_max.value()),
        )

    def _preview(self) -> None:
        if self._updating_gates or self._frame is None:
            return
        try:
            settings = self._settings()
            charge = integrate_waveform(self._frame.samples, settings)
        except ValueError as exc:
            self.preview_status.setText(f"Gate configuration is invalid: {exc}")
            self.oriented_curve.clear()
            return
        if charge is None:
            self.preview_status.setText("Displayed waveform is shorter than the long gate.")
            self.oriented_curve.clear()
            return
        samples = self._frame.samples.astype(np.float64)
        oriented = charge.baseline + settings.polarity * (samples - charge.baseline)
        times = np.arange(len(samples), dtype=np.float64) * self.sample_period.value()
        self.oriented_curve.setData(times, oriented)
        stored = ""
        if self._frame.long_gate is not None and self._frame.short_gate is not None:
            stored = (
                f"; stored Qlong={self._frame.long_gate:,}, "
                f"Qshort={self._frame.short_gate:,}"
            )
        self.preview_status.setText(
            f"Baseline={charge.baseline:.2f}, RMS={charge.baseline_rms:.2f}; "
            f"Qshort={charge.short_charge:,.1f}, Qlong={charge.long_charge:,.1f}, "
            f"PSD={charge.ratio:.5f}{stored}"
        )

    def _on_gate_moved(self) -> None:
        if self._updating_gates:
            return
        self._preview()
        if self.auto_recalculate.isChecked() and self._index is not None:
            self._auto_preview_timer.start()

    def _on_gate_change_finished(self) -> None:
        if self._updating_gates:
            return
        self._preview()
        if self.auto_recalculate.isChecked() and self._index is not None:
            self._auto_preview_timer.stop()
            self._request_analysis("full")

    def _request_sparse_analysis(self) -> None:
        if self.auto_recalculate.isChecked():
            self._request_analysis("sparse")

    def _start_analysis(self) -> None:
        self._request_analysis("manual")

    def _request_analysis(self, kind: AnalysisKind) -> None:
        if self._index is None:
            return
        self._pending_analysis_kind = kind
        if self._analysis_thread is not None:
            if self._analysis_worker is not None:
                self._analysis_worker.stop()
            self.status.setText("Updating PSD calculation for the latest gate positions...")
            return
        self._launch_pending_analysis()

    def _launch_pending_analysis(self) -> None:
        kind = self._pending_analysis_kind
        self._pending_analysis_kind = None
        if kind is None or self._index is None:
            return
        try:
            settings = self._settings()
            stride_multiplier = (
                _AUTO_PREVIEW_STRIDE_MULTIPLIER if kind == "sparse" else 1
            )
            worker = WaveformPsdWorker(
                self._index,
                source_index=self.source_combo.currentIndex(),
                settings=settings,
                stride=self.event_stride.value() * stride_multiplier,
                maximum_events=self.maximum_events.value(),
            )
        except ValueError as exc:
            self.status.setText(f"Invalid PSD settings: {exc}")
            if kind == "manual":
                QMessageBox.critical(self, "Invalid PSD Settings", str(exc))
            return
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._analysis_progress)
        worker.loaded.connect(self._analysis_loaded)
        worker.cancelled.connect(self._analysis_cancelled)
        worker.error.connect(self._analysis_failed)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._analysis_thread_finished)
        self._analysis_worker = worker
        self._analysis_thread = thread
        self._active_analysis_kind = kind
        self._set_analysis_busy(True)
        self.progress.setValue(0)
        if kind == "sparse":
            self.status.setText("Updating sparse PSD preview (every 20th selected event)...")
        else:
            self.status.setText("Reconstructing full PSD from waveforms...")
        thread.start()

    def _analysis_progress(self, processed: int, total: int) -> None:
        self.progress.setValue(round(1000 * processed / total) if total else 0)
        prefix = "Sparse preview" if self._active_analysis_kind == "sparse" else "Full PSD"
        self.status.setText(f"{prefix}: integrating {processed:,}/{total:,} waveforms...")

    def _analysis_loaded(self, value: object) -> None:
        if not isinstance(value, WaveformPsdResult):
            self._analysis_failed("Waveform worker returned an invalid result")
            return
        if self._pending_analysis_kind is not None:
            return
        self.psd_plot.set_data(
            value.matrix,
            energy_range=value.energy_range,
            ratio_range=value.ratio_range,
        )
        stats = value.statistics
        comparison = ""
        if len(value.stored_long):
            nonzero = value.stored_long > 0
            if np.any(nonzero):
                scale = np.median(
                    value.calculated_long[nonzero] / value.stored_long[nonzero]
                )
                comparison = (
                    f" Stored/computed comparison: median Qlong scale={scale:.4g}."
                )
        self.progress.setValue(1000)
        summary = (
            f"accepted {stats.accepted:,}/{stats.received:,}; too short "
            f"{stats.too_short:,}; nonpositive Qlong {stats.nonpositive_long:,}; outside "
            f"view {stats.outside_range:,}.{comparison}"
        )
        if self._active_analysis_kind == "sparse":
            self.status.setText(
                f"Sparse PSD preview complete: {summary} Release a gate for a full update."
            )
        else:
            self.status.setText(f"PSD complete: {summary}")

    def _analysis_failed(self, message: str) -> None:
        self.status.setText(f"PSD reconstruction failed: {message}")
        QMessageBox.critical(self, "PSD Reconstruction Failed", message)

    def _analysis_cancelled(self) -> None:
        if self._pending_analysis_kind is None:
            self.status.setText("PSD reconstruction cancelled.")

    def _cancel_analysis(self) -> None:
        self._auto_preview_timer.stop()
        self._pending_analysis_kind = None
        if self._analysis_worker is not None:
            self._analysis_worker.stop()
            self.status.setText("Cancelling PSD reconstruction...")

    def _analysis_thread_finished(self) -> None:
        thread = self._analysis_thread
        self._analysis_worker = None
        self._analysis_thread = None
        self._active_analysis_kind = None
        self._set_analysis_busy(False)
        if thread is not None:
            thread.deleteLater()
        if self._pending_analysis_kind is not None and self._index is not None:
            QTimer.singleShot(0, self._launch_pending_analysis)

    def _set_analysis_busy(self, busy: bool) -> None:
        self.analyze_button.setEnabled(not busy and self._index is not None)
        self.cancel_button.setEnabled(busy)
        self.load_button.setEnabled(not busy and self._index_thread is None)
        self.close_file_button.setEnabled(not busy and self._index is not None)
        self.source_combo.setEnabled(
            not busy and self._index is not None and len(self._index.channels) > 1
        )

    def close_file(self) -> None:
        self._auto_preview_timer.stop()
        self._pending_analysis_kind = None
        self._stop_analysis_sync()
        self._stop_index_sync()
        if self._reader is not None:
            self._reader.close()
        self._reader = None
        self._index = None
        self._frame = None
        self.source_combo.clear()
        self.source_combo.setEnabled(False)
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setEnabled(False)
        self.sample_period.setEnabled(False)
        self.close_file_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.raw_curve.clear()
        self.oriented_curve.clear()
        self.preview_status.setText("No frame loaded.")

    def _stop_index_sync(self) -> None:
        if self._index_worker is not None:
            self._index_worker.stop()
        if self._index_thread is not None and self._index_thread.isRunning():
            if not self._index_thread.wait(5_000):
                log.warning("Waveform index worker did not stop within 5 seconds")

    def _stop_analysis_sync(self) -> None:
        self._auto_preview_timer.stop()
        self._pending_analysis_kind = None
        if self._analysis_worker is not None:
            self._analysis_worker.stop()
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            if not self._analysis_thread.wait(5_000):
                log.warning("Waveform PSD worker did not stop within 5 seconds")

    def show_workspace(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def close_without_prompt(self) -> None:
        self._stop_analysis_sync()
        self._stop_index_sync()
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self.close()

    def reject(self) -> None:
        self.close_without_prompt()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._stop_analysis_sync()
        self._stop_index_sync()
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        super().closeEvent(event)
