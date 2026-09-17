"""Offline split-pulse timing diagnostic for two native MCA captures."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt, QThread
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.timing_validation import TimingValidationResult
from nlab.workers.timing_validation_worker import TimingValidationWorker

_FILE_FILTER = "NLab MCA events (*.bin *.h5 *.hdf5 *.root);;All files (*)"
_OFFSET_SETTINGS_KEY = "timing/channel_b_offset_ns"


class TimingValidationDialog(QDialog):
    """Inspect two files without treating nearest neighbors as coincidences."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Validate Two-Channel Timing")
        self.resize(840, 630)
        self._worker: TimingValidationWorker | None = None
        self._thread: QThread | None = None
        self._close_when_done = False

        layout = QVBoxLayout(self)
        instructions = QLabel(
            "Record a known pulse split between two channels in the same board session. "
            "Select native MCA list-mode files with the lower-index channel in A; "
            "this checks timestamp order and "
            "shows candidate delay B − A. It does not prove a shared clock by itself."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        form = QFormLayout()
        self.path_a = QLineEdit(self)
        self.path_b = QLineEdit(self)
        form.addRow("Channel A file", self._file_row(self.path_a))
        form.addRow("Channel B file", self._file_row(self.path_b))
        self.offset_ns = QSpinBox(self)
        self.offset_ns.setRange(-1_000_000, 1_000_000)
        self.offset_ns.setSingleStep(8)
        self.offset_ns.setSuffix(" ns")
        self.offset_ns.setToolTip("Fixed offset added to channel B timestamps (8 ns steps).")
        stored_offset = cast(int, QSettings().value(_OFFSET_SETTINGS_KEY, 0, type=int))
        self.offset_ns.setValue(stored_offset if stored_offset % 8 == 0 else 0)
        self.offset_ns.valueChanged.connect(
            lambda value: QSettings().setValue(_OFFSET_SETTINGS_KEY, value)
        )
        form.addRow("B time offset", self.offset_ns)
        self.search_window_ns = QSpinBox(self)
        self.search_window_ns.setRange(8, 80_000)
        self.search_window_ns.setSingleStep(8)
        self.search_window_ns.setValue(800)
        self.search_window_ns.setSuffix(" ns")
        form.addRow("Search half-window", self.search_window_ns)
        layout.addLayout(form)

        controls = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze timestamps", self)
        self.analyze_button.clicked.connect(self._start)
        controls.addWidget(self.analyze_button)
        controls.addStretch()
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.reject)
        controls.addWidget(close_button)
        layout.addLayout(controls)

        self.status = QLabel("Select two recorded channel files.", self)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.plot = pg.PlotWidget(self)
        self.plot.setLabel("bottom", "Nearest B − A delay", units="ns")
        self.plot.setLabel("left", "Sampled A events")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.plot, 1)
        warning = QLabel(
            "Diagnostic nearest-neighbor pairs may reuse events and are not coincidence counts. "
            "An observed peak is meaningful only with a known split-pulse input."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)

    def _file_row(self, edit: QLineEdit) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit)
        browse = QPushButton("Browse...", row)
        browse.clicked.connect(lambda: self._browse(edit))
        layout.addWidget(browse)
        return row

    def _browse(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MCA Event File", str(Path(edit.text()).parent), _FILE_FILTER
        )
        if path:
            edit.setText(path)

    def _start(self) -> None:
        if self._thread is not None:
            return
        path_a = Path(self.path_a.text().strip())
        path_b = Path(self.path_b.text().strip())
        if not path_a.is_file() or not path_b.is_file():
            self.status.setText("Select two existing MCA event files.")
            return
        self.plot.clear()
        self.status.setText("Starting timestamp scan...")
        self.analyze_button.setEnabled(False)
        worker = TimingValidationWorker(
            path_a,
            path_b,
            offset_ns=self.offset_ns.value(),
            search_window_ns=self.search_window_ns.value(),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.status.setText)
        worker.result.connect(self._show_result)
        worker.error.connect(lambda message: self.status.setText(f"Timing check failed: {message}"))
        worker.cancelled.connect(lambda: self.status.setText("Timing check cancelled."))
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._worker = worker
        self._thread = thread
        thread.start()

    def _show_result(self, result: TimingValidationResult) -> None:
        self.plot.clear()
        centers = result.bin_centers_ns
        width = float(centers[1] - centers[0]) if len(centers) > 1 else 8.0
        self.plot.addItem(
            pg.BarGraphItem(
                x=centers,
                height=result.bin_counts,
                width=width,
                brush=pg.mkBrush("#4d91bd"),
                pen=None,
            )
        )
        peak = (
            f"strongest bin {result.peak_delay_ns:+.0f} ns ({result.peak_pairs:,} pairs)"
            if result.peak_delay_ns is not None
            else "no pairs inside the search window"
        )
        self.status.setText(
            f"Ch {result.channel_a.channel}: {result.channel_a.usable_events:,} usable; "
            f"Ch {result.channel_b.channel}: {result.channel_b.usable_events:,} usable. "
            f"Overlap {result.overlap_ns / 1e9:.3f} s; "
            f"sampled {result.sampled_a:,}/{result.sampled_b:,} events; "
            f"{result.matched_a:,} nearest pairs in window; {peak}. "
            f"Zero-timestamp exclusions (ambiguous padding or events): "
            f"{result.channel_a.zero_timestamps:,}/{result.channel_b.zero_timestamps:,}. "
            + (
                "Both captures identify the same device endpoint."
                if result.same_device_metadata
                else "Device identity unavailable in one or both captures."
            )
        )

    def _on_thread_finished(self) -> None:
        self._worker = None
        self._thread = None
        self.analyze_button.setEnabled(True)
        if self._close_when_done:
            super().reject()

    def reject(self) -> None:
        if self._worker is not None:
            self._close_when_done = True
            self._worker.stop()
            self.status.setText("Cancelling timestamp scan...")
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._worker is not None:
            self._close_when_done = True
            self._worker.stop()
            self.status.setText("Cancelling timestamp scan...")
            event.ignore()
            return
        super().closeEvent(event)
