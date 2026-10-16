"""Responsive progress and cooperative cancellation for offline ROOT export."""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path

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

from nlab.utils.root_converter import (
    ConversionCancelledError,
    RootConversionResult,
    convert_binary_to_root,
)

log = logging.getLogger(__name__)


class _ConversionThread(QThread):
    progress = Signal(int)

    def __init__(self, src: Path, dst: Path, parent: QWidget) -> None:
        super().__init__(parent)
        self.src, self.dst = src, dst
        self.stop = threading.Event()
        self.result: RootConversionResult | None = None
        self.error = ""

    def run(self) -> None:
        log.info("ROOT conversion started: %s -> %s", self.src, self.dst)
        try:
            self.result = convert_binary_to_root(
                self.src, self.dst, progress=self.progress.emit, cancelled=self.stop.is_set,
            )
        except ConversionCancelledError:
            log.info("ROOT conversion cancelled: %s -> %s", self.src, self.dst)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            # Format the traceback here. Log handlers (including queued GUI and
            # session handlers) must not retain exception objects/file handles.
            detail = traceback.format_exc()
            log.error("ROOT conversion failed: %s -> %s\n%s", self.src, self.dst, detail)
        else:
            log.info(
                "ROOT conversion completed: %s -> %s (%s %s)",
                self.src, self.dst, self.result.count, self.result.kind,
            )


class RootConversionDialog(QDialog):
    def __init__(self, src: Path, dst: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Convert Binary to ROOT")
        self.setMinimumWidth(420)
        self.status = QLabel("Validating and converting the NDMA capture…", self)
        self.status.setWordWrap(True)
        self.progress = QProgressBar(self)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.buttons)
        self.buttons.rejected.connect(self.reject)
        self._thread = _ConversionThread(src, dst, self)
        self._thread.progress.connect(self.progress.setValue)
        self._thread.finished.connect(self._finished, Qt.ConnectionType.QueuedConnection)
        self._running = True
        QTimer.singleShot(0, self._thread.start)

    @property
    def conversion_result(self) -> RootConversionResult | None:
        return self._thread.result

    @property
    def error(self) -> str:
        return self._thread.error

    def _finished(self) -> None:
        self._running = False
        if self.conversion_result is not None:
            super().accept()
        else:
            super().reject()

    def reject(self) -> None:
        if self._running:
            self._thread.stop.set()
            self.status.setText("Cancelling; waiting for file cleanup…")
            self.buttons.setEnabled(False)
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self._running:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
