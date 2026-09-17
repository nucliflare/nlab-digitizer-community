"""Background two-file timestamp inspection for the timing dialog."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import Signal

from nlab.analysis.timing_validation import (
    TimingValidationCancelledError,
    validate_two_channel_timing,
)
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)


class TimingValidationWorker(BaseWorker):
    progress = Signal(str)
    result = Signal(object)
    cancelled = Signal()

    def __init__(
        self, path_a: Path, path_b: Path, *, offset_ns: int, search_window_ns: int
    ) -> None:
        super().__init__()
        self._path_a = path_a
        self._path_b = path_b
        self._offset_ns = offset_ns
        self._search_window_ns = search_window_ns
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            result = validate_two_channel_timing(
                self._path_a,
                self._path_b,
                offset_ns=self._offset_ns,
                search_window_ns=self._search_window_ns,
                cancelled=self._stop_event.is_set,
                progress=self.progress.emit,
            )
            self.result.emit(result)
        except TimingValidationCancelledError:
            self.cancelled.emit()
        except Exception as exc:
            log.exception("Two-channel timing validation failed")
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def stop(self) -> None:
        self._stop_event.set()
