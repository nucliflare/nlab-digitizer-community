"""One cancellable noise/mean-pulse spectrum of recorded waveform frames."""

from __future__ import annotations

import logging
import threading

import numpy as np
from PySide6.QtCore import QThread, Signal

from nlab.analysis.waveform_file import MappedWaveformFile, WaveformFileIndex
from nlab.analysis.waveform_frame_spectrum import (
    FrameSpectrumAccumulator,
    FrameSpectrumRequest,
)

log = logging.getLogger(__name__)


class WaveformSpectrumWorker(QThread):
    """Read the selected frames once; report the result with its generation.

    ``frames`` (absolute frame indices) selects a population; ``None`` reads
    every ``stride``-th frame up to ``maximum_events`` (0 = all), as the PSD
    reconstruction does. A cancelled request reports nothing.
    """

    computed = Signal(int, object)
    failed = Signal(int, str)
    progressed = Signal(int, int)

    def __init__(
        self,
        generation: int,
        index: WaveformFileIndex,
        *,
        source_index: int,
        request: FrameSpectrumRequest,
        frames: np.ndarray | None = None,
        stride: int = 1,
        maximum_events: int = 0,
        population: str = "",
    ) -> None:
        super().__init__()
        if not 0 <= source_index < len(index.channels):
            raise ValueError("waveform source is out of range")
        if stride <= 0 or maximum_events < 0:
            raise ValueError("stride must be positive and maximum_events non-negative")
        self.generation = generation
        self._index = index
        self._source_index = source_index
        self._request = request
        self._frames = None if frames is None else np.asarray(frames, dtype=np.int64).copy()
        self._stride = stride
        self._maximum_events = maximum_events
        self._population = population
        self._cancel = threading.Event()
        self._percent = -1

    def cancel(self) -> None:
        self._cancel.set()

    def _progress(self, done: int, total: int) -> None:
        percent = 100 * done // max(1, total)
        if percent != self._percent:
            self._percent = percent
            self.progressed.emit(self.generation, percent)

    def run(self) -> None:
        reader: MappedWaveformFile | None = None
        try:
            reader = MappedWaveformFile(self._index)
            accumulator = FrameSpectrumAccumulator(self._request)
            if self._frames is None:
                self._read_strided(reader, accumulator)
            else:
                self._read_selected(reader, accumulator)
            if not self._cancel.is_set():
                self.computed.emit(self.generation, accumulator.result(self._population))
        except Exception as exc:
            log.exception("Waveform spectrum failed for %s", self._index.path)
            if not self._cancel.is_set():
                self.failed.emit(self.generation, f"{type(exc).__name__}: {exc}")
        finally:
            if reader is not None:
                reader.close()

    def _read_strided(
        self, reader: MappedWaveformFile, accumulator: FrameSpectrumAccumulator,
    ) -> None:
        source = self._index.channels[self._source_index]
        available = (source.frame_count + self._stride - 1) // self._stride
        total = min(available, self._maximum_events) if self._maximum_events else available
        processed = 0
        while processed < total and not self._cancel.is_set():
            count = min(self._request.batch_rows, total - processed)
            batch = reader.batch(
                processed * self._stride, count, source_index=self._source_index,
                sample_count=self._request.frame_samples, stride=self._stride,
            )
            accumulator.add(batch.samples, batch.complete)
            # NDMA batches are zero-copy views; release before the mapping closes.
            del batch
            processed += count
            self._progress(processed, total)

    def _read_selected(
        self, reader: MappedWaveformFile, accumulator: FrameSpectrumAccumulator,
    ) -> None:
        assert self._frames is not None
        length = self._request.frame_samples
        rows = self._request.batch_rows
        total = len(self._frames)
        for first in range(0, total, rows):
            if self._cancel.is_set():
                return
            chunk = self._frames[first:first + rows]
            block = np.zeros((len(chunk), length), dtype=np.float64)
            complete = np.zeros(len(chunk), dtype=np.bool_)
            for row, frame_index in enumerate(chunk):
                samples = reader.frame(int(frame_index), self._source_index).samples
                if samples.size >= length:
                    block[row] = samples[:length]
                    complete[row] = True
            accumulator.add(block, complete)
            self._progress(first + len(chunk), total)
