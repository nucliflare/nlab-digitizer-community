"""Read-only, bounded-memory averaging of an exact waveform cluster."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Signal

from nlab.analysis.waveform_file import MappedWaveformFile, WaveformFileIndex
from nlab.analysis.waveform_psd import WaveformPsdSettings
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WaveformAverage:
    """Mean baseline-subtracted, polarity-normalized samples (not peak aligned)."""

    samples: np.ndarray
    count: int


class WaveformAverageWorker(BaseWorker):
    loaded = Signal(object)

    def __init__(
        self, index: WaveformFileIndex, *, source_index: int,
        frames: np.ndarray, settings: WaveformPsdSettings,
    ) -> None:
        super().__init__()
        if frames.ndim != 1 or not len(frames) or frames.dtype.kind not in "iu":
            raise ValueError("Select at least one waveform to average")
        self._index = index
        self._source_index = source_index
        self._frames = frames.copy()
        self._settings = settings
        self._stop_event = threading.Event()

    def run(self) -> None:
        reader: MappedWaveformFile | None = None
        try:
            reader = MappedWaveformFile(self._index)
            total: np.ndarray | None = None
            settings = self._settings
            for frame_index in self._frames:
                if self._stop_event.is_set():
                    return
                samples = reader.frame(int(frame_index), self._source_index).samples.astype(float)
                if len(samples) < settings.long_end:
                    raise ValueError("A selected waveform is shorter than the reconstruction gates")
                region = samples[settings.baseline_start:settings.baseline_end]
                baseline = (
                    np.median(region) if settings.baseline_method == "median" else np.mean(region)
                )
                samples = settings.polarity * (samples - baseline)
                if total is None:
                    total = samples
                else:
                    # Variable-length CAEN records: every displayed sample must
                    # represent ALL selected events, never a changing tail subset.
                    length = min(len(total), len(samples))
                    total = total[:length]
                    total += samples[:length]
            if self._stop_event.is_set():
                return
            assert total is not None
            total /= len(self._frames)
            total.setflags(write=False)
            self.loaded.emit(WaveformAverage(total, len(self._frames)))
        except Exception as exc:
            log.exception("Cluster waveform averaging failed for %s", self._index.path)
            self.error.emit(str(exc))
        finally:
            if reader is not None:
                reader.close()
            self.finished.emit()

    def stop(self) -> None:
        self._stop_event.set()
