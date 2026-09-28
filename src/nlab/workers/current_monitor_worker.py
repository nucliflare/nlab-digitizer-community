from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, Signal, Slot

from nlab.hardware.digitizer.current_monitor import CurrentMonitorClient
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CurrentSample:
    """One current estimate and its timing/coverage metadata."""

    sequence: int
    timestamp_ns: int
    raw_code: float
    read_latency_ns: int
    coverage_ns: int = 0
    hardware_timestamp: int | None = None
    sample_count: int = 1


@dataclass(frozen=True)
class CurrentSampleBatch:
    samples: tuple[CurrentSample, ...]
    dropped_samples: int
    error: str | None


class CurrentSampleBuffer:
    """Bounded single-producer/single-consumer buffer for monitor samples."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("current sample buffer capacity must be positive")
        self._capacity = capacity
        self._samples: deque[CurrentSample] = deque(maxlen=capacity)
        self._dropped_samples = 0
        self._error: str | None = None
        self._lock = threading.Lock()

    def append(self, sample: CurrentSample) -> None:
        with self._lock:
            if len(self._samples) == self._capacity:
                self._dropped_samples += 1
            self._samples.append(sample)

    def set_error(self, message: str | None) -> None:
        with self._lock:
            self._error = message

    def drain(self) -> CurrentSampleBatch:
        with self._lock:
            samples = tuple(self._samples)
            self._samples.clear()
            dropped = self._dropped_samples
            self._dropped_samples = 0
            error = self._error
        return CurrentSampleBatch(samples, dropped, error)


class CurrentMonitorWorker(BaseWorker):
    """Continuously poll one FPGA IIR output on a worker-owned transport."""

    request_stop = Signal()

    def __init__(
        self,
        client_factory: Callable[[], CurrentMonitorClient],
        sample_buffer: CurrentSampleBuffer,
        target_hz: int = 1000,
        error_retry_ms: int = 100,
    ) -> None:
        super().__init__()
        if target_hz <= 0:
            raise ValueError("target_hz must be positive")
        self._client_factory = client_factory
        self._sample_buffer = sample_buffer
        self._target_period_ns = round(1_000_000_000 / target_hz)
        self._error_retry_ms = error_retry_ms
        self._client: CurrentMonitorClient | None = None
        self._timer: QTimer | None = None
        self._stop_requested = threading.Event()
        self._finished = False
        self._read_failing = False
        self._sequence = 0

    @Slot()
    def run(self) -> None:
        if self._stop_requested.is_set():
            self._finish()
            return
        try:
            # Construct the connection on the thread that will use it.  This
            # is mandatory for the direct-IIO backend and also keeps gRPC
            # completion-queue activity away from the GUI thread.
            self._client = self._client_factory()
        except Exception as exc:
            message = f"Could not open current monitor: {exc}"
            self._sample_buffer.set_error(message)
            log.exception(message)
            self.error.emit(message)
            self._finish()
            return

        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self.request_stop.connect(self._stop)
        self._timer.start(0)

    def request_shutdown(self) -> None:
        """Request stop even while a synchronous transport read is active."""
        self._stop_requested.set()
        self.request_stop.emit()

    @Slot()
    def _stop(self) -> None:
        self._stop_requested.set()
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.close()
            except Exception:
                log.exception("Closing the current-monitor transport failed")
        self.finished.emit()

    def _schedule(self, delay_ms: int) -> None:
        if self._stop_requested.is_set():
            self._finish()
            return
        timer = self._timer
        if timer is not None:
            timer.start(max(0, delay_ms))

    @Slot()
    def _tick(self) -> None:
        if self._stop_requested.is_set():
            self._finish()
            return
        client = self._client
        if client is None:
            self._finish()
            return

        started_ns = time.perf_counter_ns()
        try:
            raw_code = client.read_raw()
        except Exception as exc:
            if self._stop_requested.is_set():
                self._finish()
                return
            message = f"Current-monitor read failed: {exc}"
            self._sample_buffer.set_error(message)
            if not self._read_failing:
                self._read_failing = True
                log.exception(message)
                self.error.emit(message)
            self._schedule(self._error_retry_ms)
            return
        completed_ns = time.perf_counter_ns()

        if self._read_failing:
            log.info("Current-monitor reads recovered")
            self._read_failing = False
        self._sample_buffer.set_error(None)
        self._sequence += 1
        self._sample_buffer.append(
            CurrentSample(
                sequence=self._sequence,
                timestamp_ns=(started_ns + completed_ns) // 2,
                raw_code=int(raw_code),
                read_latency_ns=completed_ns - started_ns,
            )
        )

        elapsed_ns = completed_ns - started_ns
        remaining_ns = self._target_period_ns - elapsed_ns
        # If a synchronous read already consumed the target period, begin the
        # next one as soon as the worker event loop can service it. QTimer has
        # only millisecond resolution, so use the shutdown Event as an
        # interruptible sub-millisecond rate limiter before yielding back to
        # the event loop. This avoids turning a 0.2 ms read plus a rounded
        # 1 ms timer into an unintended ~800 Hz stream.
        if remaining_ns > 0:
            self._stop_requested.wait(remaining_ns / 1_000_000_000)
        self._schedule(0)
