"""Background selected-cluster export with cooperative cancellation and cleanup."""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.waveform_cluster_export import (
    ClusterExportCancelledError,
    ClusterExportResult,
    ClusterFormat,
    export_waveform_cluster,
)
from nlab.analysis.waveform_file import WaveformFileIndex

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterExportRequest:
    index: WaveformFileIndex
    destination: Path
    file_format: ClusterFormat
    source_index: int
    frames: np.ndarray
    energy: np.ndarray
    ratio: np.ndarray
    metadata: dict[str, Any]


class _ExportThread(QThread):
    progress = Signal(int)

    def __init__(self, request: ClusterExportRequest, parent: QWidget) -> None:
        super().__init__(parent)
        self.request = request
        self.stop = threading.Event()
        self.result: ClusterExportResult | None = None
        self.error = ""

    def run(self) -> None:
        request = self.request
        try:
            self.result = export_waveform_cluster(
                request.index, request.destination, file_format=request.file_format,
                source_index=request.source_index, frames=request.frames, energy=request.energy,
                ratio=request.ratio, metadata=request.metadata, progress=self.progress.emit,
                cancelled=self.stop.is_set,
            )
        except ClusterExportCancelledError:
            log.info("Selected-cluster export cancelled: %s", request.destination)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("Selected-cluster export failed: %s\n%s",
                      request.destination, traceback.format_exc())


class WaveformClusterExportDialog(QDialog):
    def __init__(self, request: ClusterExportRequest, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Selected Waveform Cluster")
        self.setMinimumWidth(440)
        self.status = QLabel(f"Exporting {len(request.frames):,} original waveforms...", self)
        self.status.setWordWrap(True)
        self.progress = QProgressBar(self)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.buttons)
        self.buttons.rejected.connect(self.reject)
        self._thread = _ExportThread(request, self)
        self._thread.progress.connect(self.progress.setValue)
        self._thread.finished.connect(self._finished, Qt.ConnectionType.QueuedConnection)
        self._running = True
        self._started = False
        QTimer.singleShot(0, self._start)

    def _start(self) -> None:
        if self._thread.stop.is_set():
            self._finished()
        else:
            self._started = True
            self._thread.start()

    def stop_sync(self) -> bool:
        """Protect parent shutdown from destroying an active writer thread."""
        self._thread.stop.set()
        return not self._started or self._thread.wait(5_000)

    @property
    def export_result(self) -> ClusterExportResult | None:
        return self._thread.result

    @property
    def error(self) -> str:
        return self._thread.error

    def _finished(self) -> None:
        self._running = False
        if self.export_result is not None:
            super().accept()
        else:
            super().reject()

    def reject(self) -> None:
        if self._running:
            self._thread.stop.set()
            self.status.setText("Cancelling; waiting for export cleanup...")
            self.buttons.setEnabled(False)
        else:
            super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self._running:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
