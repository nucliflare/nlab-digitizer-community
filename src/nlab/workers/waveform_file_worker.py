"""Background index construction for memory-mapped waveform browsing."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import Signal

from nlab.analysis.waveform_file import (
    WaveformIndexCancelledError,
    build_waveform_file_index,
)
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)


class WaveformFileIndexWorker(BaseWorker):
    progress = Signal(object, object)
    loaded = Signal(object)
    cancelled = Signal()

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            index = build_waveform_file_index(
                self._path,
                progress=self.progress.emit,
                cancelled=self._stop_event.is_set,
            )
            self.loaded.emit(index)
        except WaveformIndexCancelledError:
            self.cancelled.emit()
        except Exception as exc:
            log.exception("Waveform file indexing failed for %s", self._path)
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def stop(self) -> None:
        self._stop_event.set()
