"""Background, batch-wise PSD reconstruction from saved event files."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import Signal

from nlab.analysis.psd import PsdAccumulator
from nlab.analysis.psd_file import PsdEventFileInfo, iter_psd_event_batches
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)


class PsdFileWorker(BaseWorker):
    # Python-object counts avoid Qt's signed 32-bit ``int`` limit for very
    # large list-mode captures.
    progress = Signal(object, object)
    loaded = Signal(object, object, str)
    cancelled = Signal()

    def __init__(
        self,
        info: PsdEventFileInfo,
        *,
        energy_bins: int,
        ratio_bins: int,
        energy_right_shift: int,
        ratio_range: tuple[float, float],
    ) -> None:
        super().__init__()
        self._info = info
        self._energy_bins = energy_bins
        self._ratio_bins = ratio_bins
        self._energy_right_shift = energy_right_shift
        self._ratio_range = ratio_range
        self._stop_event = threading.Event()

    def run(self) -> None:
        processed = 0
        try:
            accumulator = PsdAccumulator(
                energy_bins=self._energy_bins,
                ratio_bins=self._ratio_bins,
                energy_right_shift=self._energy_right_shift,
                ratio_range=self._ratio_range,
            )
            for events in iter_psd_event_batches(self._info.path):
                if self._stop_event.is_set():
                    self.cancelled.emit()
                    return
                accumulator.add_events(events)
                processed += len(events)
                self.progress.emit(processed, self._info.total_events)
            self.loaded.emit(accumulator, processed, str(self._info.path))
        except Exception as exc:
            log.exception("PSD file processing failed for %s", self._info.path)
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def stop(self) -> None:
        self._stop_event.set()
