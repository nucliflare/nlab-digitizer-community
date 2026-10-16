"""Background gate search for waveform PSD figure-of-merit optimization."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
from PySide6.QtCore import Signal

from nlab.analysis.waveform_file import MappedWaveformFile, WaveformFileIndex
from nlab.analysis.waveform_fom_optimization import (
    WaveformFomOptimizationCancelledError,
    WaveformFomOptimizationRequest,
    WaveformGateCandidate,
    optimize_waveform_gate_ends,
)
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)

_TARGET_CHARGE_BYTES = 128 * 1024 * 1024
_BATCH_TARGET_BYTES = 16 * 1024 * 1024
_MAX_BATCH_EVENTS = 8192


class WaveformFomOptimizationWorker(BaseWorker):
    """Load a bounded waveform sample once, then search integer gate pairs."""

    status = Signal(str)
    progress = Signal(object, object, object, object, object)
    loaded = Signal(object)
    cancelled = Signal()

    def __init__(
        self,
        index: WaveformFileIndex,
        *,
        source_index: int,
        request: WaveformFomOptimizationRequest,
        stride: int,
        maximum_events: int,
    ) -> None:
        super().__init__()
        if not 0 <= source_index < len(index.channels):
            raise ValueError("waveform source is out of range")
        if stride <= 0 or maximum_events <= 0:
            raise ValueError("optimization stride and event limit must be positive")
        self._index = index
        self._source_index = source_index
        self._request = request
        self._stride = stride
        self._maximum_events = maximum_events
        self._stop_event = threading.Event()

    def run(self) -> None:
        reader: MappedWaveformFile | None = None
        try:
            request = self._request
            source = self._index.channels[self._source_index]
            available = (source.frame_count + self._stride - 1) // self._stride
            integration_width = request.long_bounds[1] - request.settings.gate_start
            memory_limit = max(1, _TARGET_CHARGE_BYTES // (integration_width * 8))
            total = min(available, self._maximum_events, memory_limit)
            if total <= 0:
                raise ValueError("waveform source contains no events for optimization")
            if total < min(available, self._maximum_events):
                self.status.emit(
                    f"Preparing {total:,} waveforms (bounded by the 128 MiB optimizer limit)..."
                )
            else:
                self.status.emit(f"Preparing {total:,} waveforms for gate optimization...")

            cumulative_charge = np.empty((total, integration_width), dtype=np.float64)
            max_long_end = request.long_bounds[1]
            bytes_per_waveform = max(1, max_long_end * 2)
            batch_events = max(
                1,
                min(_MAX_BATCH_EVENTS, _BATCH_TARGET_BYTES // bytes_per_waveform),
            )
            reader = MappedWaveformFile(self._index)
            processed = 0
            prepared = 0
            while processed < total:
                if self._stop_event.is_set():
                    raise WaveformFomOptimizationCancelledError
                count = min(batch_events, total - processed)
                batch = reader.batch(
                    processed * self._stride,
                    count,
                    source_index=self._source_index,
                    sample_count=max_long_end,
                    stride=self._stride,
                )
                complete = np.asarray(batch.complete, dtype=np.bool_)
                if np.any(complete):
                    values = batch.samples[complete].astype(np.float64)
                    settings = request.settings
                    baseline_samples = values[
                        :, settings.baseline_start : settings.baseline_end
                    ]
                    baseline = (
                        np.median(baseline_samples, axis=1)
                        if settings.baseline_method == "median"
                        else np.mean(baseline_samples, axis=1)
                    )
                    pulse = settings.polarity * (
                        values[:, settings.gate_start:max_long_end] - baseline[:, None]
                    )
                    next_prepared = prepared + len(pulse)
                    cumulative_charge[prepared:next_prepared] = np.cumsum(
                        pulse,
                        axis=1,
                        dtype=np.float64,
                    )
                    prepared = next_prepared
                del batch
                processed += count
            cumulative_charge = cumulative_charge[:prepared]
            if not len(cumulative_charge):
                raise ValueError("selected waveforms do not reach the gate search range")
            self.status.emit(
                f"Searching gate endpoints ({request.search_mode}) with "
                f"{len(cumulative_charge):,} complete waveforms..."
            )
            last_report_time = 0.0

            def report_progress(
                evaluated: int,
                valid: int,
                best: WaveformGateCandidate | None,
            ) -> None:
                nonlocal last_report_time
                now = time.monotonic()
                if (
                    request.search_mode == "exhaustive"
                    and evaluated not in (1, request.total_gate_pairs)
                    and now - last_report_time < 0.1
                ):
                    return
                last_report_time = now
                if best is None:
                    self.progress.emit(evaluated, valid, None, None, None)
                    return
                self.progress.emit(
                    evaluated,
                    valid,
                    best.fom,
                    best.short_end,
                    best.long_end,
                )

            result = optimize_waveform_gate_ends(
                cumulative_charge,
                request,
                progress=report_progress,
                cancelled=self._stop_event.is_set,
            )
            self.loaded.emit(result)
        except WaveformFomOptimizationCancelledError:
            self.cancelled.emit()
        except Exception as exc:
            log.exception("Waveform FOM gate optimization failed for %s", self._index.path)
            self.error.emit(str(exc))
        finally:
            if reader is not None:
                reader.close()
            self.finished.emit()

    def stop(self) -> None:
        self._stop_event.set()
