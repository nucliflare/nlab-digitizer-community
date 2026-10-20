"""Standalone workbench for PSD reconstruction from saved event files."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.capture_metadata import capture_digitizer_info
from nlab.analysis.psd import PsdAccumulator
from nlab.analysis.psd_file import PsdEventFileInfo, inspect_psd_event_file
from nlab.analysis.psd_report import PsdReportRequest, export_psd_report
from nlab.views.offline_psd_plot import OfflinePsdPlot
from nlab.workers.psd_file_worker import PsdFileWorker

log = logging.getLogger(__name__)
_WINDOW_TITLE = "PSD Event Readback"


class PsdReadbackDialog(QDialog):
    """Analyze stored long/short-gate event records independently of live PSD."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_WINDOW_TITLE)
        self.setModal(False)
        self.resize(1180, 760)
        self._thread: QThread | None = None
        self._worker: PsdFileWorker | None = None
        self._pending_info: PsdEventFileInfo | None = None
        self._source_info: PsdEventFileInfo | None = None
        self._last_accumulator: PsdAccumulator | None = None
        self._processed_events = 0
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.setToolTip(
            "Read stored long/short-gate events without changing live acquisition. "
            "The PSD ratio is (Qlong - Qshort) / Qlong. Press F1 for parameter help."
        )

        settings = QGroupBox("File and histogram settings", self)
        settings.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        grid = QGridLayout(settings)
        grid.setContentsMargins(8, 6, 8, 6)
        self.load_button = QPushButton("Load event file...", self)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setEnabled(False)
        grid.addWidget(self.load_button, 0, 0)
        grid.addWidget(self.cancel_button, 0, 1)

        self.energy_bins = QSpinBox(self)
        self.energy_bins.setRange(16, 8192)
        self.energy_bins.setValue(1024)
        self.ratio_bins = QSpinBox(self)
        self.ratio_bins.setRange(16, 4096)
        self.ratio_bins.setValue(256)
        self.energy_shift = QSpinBox(self)
        self.energy_shift.setRange(0, 15)
        self.ratio_min = QDoubleSpinBox(self)
        self.ratio_min.setRange(-10.0, 10.0)
        self.ratio_min.setDecimals(4)
        self.ratio_min.setValue(-1.0)
        self.ratio_max = QDoubleSpinBox(self)
        self.ratio_max.setRange(-10.0, 10.0)
        self.ratio_max.setDecimals(4)
        self.ratio_max.setValue(1.0)
        for text, field, row, column in (
            ("Energy bins:", self.energy_bins, 0, 2),
            ("Ratio bins:", self.ratio_bins, 0, 4),
            ("Energy shift (bits):", self.energy_shift, 0, 6),
            ("Ratio minimum:", self.ratio_min, 1, 2),
            ("Ratio maximum:", self.ratio_max, 1, 4),
        ):
            label = QLabel(text, self)
            label.setBuddy(field)
            grid.addWidget(label, row, column)
            grid.addWidget(field, row, column + 1)
        self.current_file_label = QLabel("Current file: No file loaded", self)
        self.current_file_label.setTextFormat(Qt.TextFormat.PlainText)
        self.current_file_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.current_file_label.setWordWrap(True)
        grid.addWidget(self.current_file_label, 2, 0, 1, 9)
        grid.setColumnStretch(8, 1)
        root.addWidget(settings)

        self.plot = OfflinePsdPlot(self)
        root.addWidget(self.plot, 1)
        bottom = QHBoxLayout()
        self.status = QLabel("Select an event file to begin.", self)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        self.report_button = QPushButton("Download report...", self)
        self.report_button.setEnabled(False)
        self.report_button.setToolTip(
            "Save a PDF report and the plotted PSD/FOM data as CSV files."
        )
        bottom.addWidget(self.report_button)
        self.close_button = QPushButton("Close", self)
        bottom.addWidget(self.close_button)
        root.addLayout(bottom)

        self.load_button.clicked.connect(self._choose_file)
        self.cancel_button.clicked.connect(self._cancel)
        self.report_button.clicked.connect(self._download_report)
        self.plot.fom_widget.fit_changed.connect(self._fit_changed)
        self.close_button.clicked.connect(self.reject)

    def _choose_file(self) -> None:
        folder = str(QSettings().value("analysis/psd_event_folder", "measurements"))
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Open PSD Event File",
            folder,
            (
                "PSD event files (*.bin *.h5 *.hdf5 *.root);;"
                "Binary event files (*.bin);;HDF5 (*.h5 *.hdf5);;ROOT (*.root);;"
                "All files (*)"
            ),
        )
        if path_text:
            self.open_path(Path(path_text))

    def open_path(self, path: Path) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "PSD Readback", "Cancel the current load first.")
            return
        try:
            if self.ratio_min.value() >= self.ratio_max.value():
                raise ValueError("ratio minimum must be smaller than ratio maximum")
            info = inspect_psd_event_file(path)
            worker = PsdFileWorker(
                info,
                energy_bins=self.energy_bins.value(),
                ratio_bins=self.ratio_bins.value(),
                energy_right_shift=self.energy_shift.value(),
                ratio_range=(self.ratio_min.value(), self.ratio_max.value()),
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "PSD File Load Failed", str(exc))
            return
        QSettings().setValue("analysis/psd_event_folder", str(path.parent))
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._progress)
        worker.loaded.connect(self._loaded)
        worker.cancelled.connect(self._cancelled)
        worker.error.connect(self._failed)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        # Let the worker thread process its own deferred QObject deletion
        # before GUI-side cleanup releases the final Python reference.
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            self._thread_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker = worker
        self._thread = thread
        self._pending_info = info
        self._set_busy(True)
        self.status.setText(f"Loading {path.name} ({info.total_events:,} events)...")
        thread.start()

    def _progress(self, processed: int, total: int) -> None:
        self.status.setText(f"Loading events: {processed:,}/{total:,}...")

    def _loaded(self, accumulator: object, processed: int, source: str) -> None:
        if not isinstance(accumulator, PsdAccumulator):
            self._failed("PSD worker returned an invalid result")
            return
        self.plot.set_data(
            accumulator.matrix,
            energy_range=accumulator.energy_range,
            ratio_range=accumulator.ratio_range,
            energy_label="Long-gate energy",
        )
        self._source_info = self._pending_info
        self._pending_info = None
        if self._source_info is not None:
            self._set_current_file(self._source_info.path)
        self._last_accumulator = accumulator
        self._processed_events = processed
        stats = accumulator.statistics
        self.status.setText(
            f"Loaded {processed:,} events from {source}. Accepted {stats.accepted:,}; "
            f"zero energy {stats.zero_total:,}; outside view {stats.outside_range:,}."
        )

    def _failed(self, message: str) -> None:
        self._pending_info = None
        self.status.setText(f"File load failed: {message}")
        QMessageBox.critical(self, "PSD File Load Failed", message)

    def _cancelled(self) -> None:
        self._pending_info = None
        self.status.setText("File load cancelled.")

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.status.setText("Cancelling file load...")

    def _thread_finished(self) -> None:
        thread = self._thread
        self._worker = None
        self._thread = None
        self._set_busy(False)
        if thread is not None:
            thread.deleteLater()

    def _set_busy(self, busy: bool) -> None:
        self.load_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        for control in (
            self.energy_bins,
            self.ratio_bins,
            self.energy_shift,
            self.ratio_min,
            self.ratio_max,
        ):
            control.setEnabled(not busy)
        self._update_report_button()

    def _set_current_file(self, path: Path | None) -> None:
        if path is None:
            self.current_file_label.setText("Current file: No file loaded")
            self.current_file_label.setToolTip("")
            self.setWindowTitle(_WINDOW_TITLE)
            return
        path_text = str(path)
        self.current_file_label.setText(f"Current file: {path_text}")
        self.current_file_label.setToolTip(path_text)
        self.setWindowTitle(f"{_WINDOW_TITLE} — {path.name}")

    def _fit_changed(self, _fit: object) -> None:
        self._update_report_button()

    def _update_report_button(self) -> None:
        self.report_button.setEnabled(
            self._thread is None
            and self._source_info is not None
            and self._last_accumulator is not None
            and self.plot.fom_widget.last_fit is not None
        )

    def _download_report(self) -> None:
        info = self._source_info
        accumulator = self._last_accumulator
        fit = self.plot.fom_widget.last_fit
        if info is None or accumulator is None or fit is None:
            QMessageBox.warning(
                self,
                "PSD Report",
                "Load a PSD file and complete the FOM fit before creating a report.",
            )
            return
        default_folder = str(QSettings().value("analysis/psd_report_folder", str(info.path.parent)))
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select PSD Report Destination",
            default_folder,
        )
        if not selected:
            return
        state = self.plot.report_state()
        request = PsdReportRequest(
            source_path=info.path,
            title="PSD Event Readback Report",
            matrix=state.matrix,
            energy_range=state.energy_range,
            ratio_range=state.ratio_range,
            energy_label=state.energy_label,
            cut=state.cut,
            energy_roi=state.energy_roi,
            matrix_scale=state.matrix_scale,
            energy_log_y=state.energy_log_y,
            fit=fit,
            settings={
                "energy right shift": accumulator.energy_right_shift,
                "PSD ratio definition": "(Qlong - Qshort) / Qlong",
            },
            digitizer_info={
                **capture_digitizer_info(info.path),
                "source format": info.format_name,
                "source board": (
                    info.board if info.board is not None else "Not embedded in source file"
                ),
                "source channel": (
                    info.channel if info.channel is not None else "Not embedded in source file"
                ),
                "declared source events": info.total_events,
            },
            statistics={
                "processed events": self._processed_events,
                "accepted events": accumulator.statistics.accepted,
                "zero-energy events": accumulator.statistics.zero_total,
                "outside-view events": accumulator.statistics.outside_range,
            },
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            bundle = export_psd_report(request, Path(selected))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            log.exception("Could not create PSD readback report")
            QMessageBox.critical(self, "PSD Report Failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        QSettings().setValue("analysis/psd_report_folder", selected)
        self.status.setText(f"PSD report saved to {bundle.directory}.")
        QMessageBox.information(
            self,
            "PSD Report Saved",
            f"Report and data files were saved to:\n{bundle.directory}",
        )

    def show_workspace(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def close_without_prompt(self) -> None:
        self._stop_sync()
        self.close()

    def _stop_sync(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None and self._thread.isRunning() and not self._thread.wait(5_000):
            log.warning("PSD file worker did not stop within 5 seconds")

    def reject(self) -> None:
        self._stop_sync()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._stop_sync()
        super().closeEvent(event)
