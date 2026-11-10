"""Dedicated offline Current capture browser; never opens a hardware client."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.current_capture import (
    MAX_MOVING_AVERAGE_KERNEL,
    CurrentCaptureInfo,
    CurrentChunk,
)
from nlab.views.plot_viewbox import ModifierZoomViewBox
from nlab.workers.current_capture_worker import CurrentCaptureWorker


class CurrentCaptureDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Current Capture Analysis")
        self.resize(1180, 850)
        self._path: Path | None = None
        self._info: CurrentCaptureInfo | None = None
        self._chunk: CurrentChunk | None = None
        self._worker: CurrentCaptureWorker | None = None
        self._generation = 0
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        self.open_button = QPushButton("Open capture…")
        self.file_label = QLabel("Open a periodic-sum HDF5 capture to begin.")
        self.file_label.setTextFormat(Qt.TextFormat.PlainText)
        header.addWidget(self.open_button)
        header.addWidget(self.file_label, 1)
        root.addLayout(header)
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
        self.trace_plot.setBackground("#f8f9fa")
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
        self.distribution_plot.setBackground("#f8f9fa")
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
        self.open_button.clicked.connect(self._choose_capture)
        self.export_button.clicked.connect(self._export_configuration)
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
            "Current sum capture (*.h5 *.hdf5)",
        )
        if filename:
            self.open_path(Path(filename))

    def open_path(self, path: Path) -> None:
        self._path = Path(path)
        self._info = self._chunk = None
        self.file_label.setText(str(path))
        self.capture_status.clear()
        self.configuration_text.clear()
        self.metadata_text.clear()
        self.configuration_status.setText("Reading capture configuration…")
        self.export_button.setEnabled(False)
        self._clear_chunk_plots()
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
        self.start_spin.blockSignals(True)
        self.start_spin.setMaximum(max(0, info.duration_us - 0.008))
        self.start_spin.blockSignals(False)
        self.capture_status.setText(
            f"Measured periodic sums · source channel {info.metadata.get('channel', '?')} · "
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
        self.export_button.setEnabled(bool(info.configuration_yaml))
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
