"""Background PSD reconstruction from indexed waveform files."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import Signal

from nlab.analysis.waveform_file import MappedWaveformFile, WaveformFileIndex
from nlab.analysis.waveform_psd import (
    WaveformPsdAccumulator,
    WaveformPsdResult,
    WaveformPsdSettings,
)
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)

_NDMA_BATCH_EVENTS = 32_768
_CAEN_BATCH_TARGET_BYTES = 16 * 1024 * 1024


class WaveformPsdWorker(BaseWorker):
    """Integrate one indexed waveform source without retaining its waveforms."""

    progress = Signal(object, object)
    loaded = Signal(object)
    cancelled = Signal()

    def __init__(
        self,
        index: WaveformFileIndex,
        *,
        source_index: int,
        settings: WaveformPsdSettings,
        stride: int = 1,
        maximum_events: int = 0,
    ) -> None:
        super().__init__()
        if not 0 <= source_index < len(index.channels):
            raise ValueError("waveform source is out of range")
        if stride <= 0 or maximum_events < 0:
            raise ValueError("stride must be positive and maximum_events non-negative")
        self._index = index
        self._source_index = source_index
        self._settings = settings
        self._stride = stride
        self._maximum_events = maximum_events
        self._stop_event = threading.Event()

    def run(self) -> None:
        reader: MappedWaveformFile | None = None
        try:
            source = self._index.channels[self._source_index]
            available = (source.frame_count + self._stride - 1) // self._stride
            total = min(available, self._maximum_events) if self._maximum_events else available
            accumulator = WaveformPsdAccumulator(self._settings)
            reader = MappedWaveformFile(self._index)
            if self._index.caen_info is None:
                batch_events = _NDMA_BATCH_EVENTS
            else:
                bytes_per_waveform = max(1, self._settings.long_end * 2)
                batch_events = max(
                    1,
                    min(
                        _NDMA_BATCH_EVENTS,
                        _CAEN_BATCH_TARGET_BYTES // bytes_per_waveform,
                    ),
                )
            processed = 0
            while processed < total:
                if self._stop_event.is_set():
                    self.cancelled.emit()
                    return
                count = min(batch_events, total - processed)
                batch = reader.batch(
                    processed * self._stride,
                    count,
                    source_index=self._source_index,
                    sample_count=self._settings.long_end,
                    stride=self._stride,
                )
                accumulator.add_waveforms(
                    batch.samples,
                    complete=batch.complete,
                    stored_long=batch.long_gate,
                    stored_short=batch.short_gate,
                )
                # NDMA batches can be zero-copy mmap views. Release the final
                # view before ``reader.close()`` invalidates the mapping.
                del batch
                processed += count
                self.progress.emit(processed, total)
            result: WaveformPsdResult = accumulator.result()
            self.loaded.emit(result)
        except Exception as exc:
            log.exception("Waveform PSD reconstruction failed for %s", self._index.path)
            self.error.emit(str(exc))
        finally:
            if reader is not None:
                reader.close()
            self.finished.emit()

    def stop(self) -> None:
        self._stop_event.set()
