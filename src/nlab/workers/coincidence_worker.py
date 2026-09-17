"""Background analysis of two independently recorded MCA list-mode streams."""

from __future__ import annotations

import logging
import threading

from nlab.analysis.coincidence import (
    CoincidenceAnalyzer,
    CoincidenceSettings,
    CoincidenceSnapshot,
)
from nlab.hardware.digitizer.dma import McaEventBuffer

log = logging.getLogger(__name__)


class CoincidenceAnalysisThread(threading.Thread):
    def __init__(
        self,
        ch0: McaEventBuffer,
        ch1: McaEventBuffer,
        settings: CoincidenceSettings,
    ) -> None:
        super().__init__(name="coincidence-analysis", daemon=False)
        self._buffers = (ch0, ch1)
        self._analyzer = CoincidenceAnalyzer(settings)
        self._snapshot = self._analyzer.snapshot()
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._pending_settings: CoincidenceSettings | None = None
        self._error: str | None = None

    def request_settings(self, settings: CoincidenceSettings) -> None:
        """Begin a fresh analysis generation without altering raw recording."""
        with self._lock:
            self._pending_settings = settings
        for buffer in self._buffers:
            buffer.clear()

    def request_stop(self) -> None:
        self._stop_requested.set()

    def result(self) -> tuple[CoincidenceSnapshot, str | None]:
        with self._lock:
            return self._snapshot, self._error

    def run(self) -> None:
        try:
            while True:
                with self._lock:
                    settings = self._pending_settings
                    self._pending_settings = None
                if settings is not None:
                    self._analyzer = CoincidenceAnalyzer(settings)
                batches = [buffer.drain() for buffer in self._buffers]
                changed = settings is not None or any(
                    channel_batches for channel_batches, _ in batches
                )
                for channel, (channel_batches, dropped) in enumerate(batches):
                    if dropped:
                        raise RuntimeError(
                            f"channel {channel} live queue dropped {dropped} records; "
                            "coincidence results are invalid"
                        )
                    for events in channel_batches:
                        self._analyzer.add_batch(channel, events)
                if changed:
                    with self._lock:
                        self._snapshot = self._analyzer.snapshot()
                if self._stop_requested.is_set() and not any(batch[0] for batch in batches):
                    self._analyzer.finish()
                    with self._lock:
                        self._snapshot = self._analyzer.snapshot()
                    return
                self._stop_requested.wait(0.02)
        except Exception as exc:
            log.exception("Coincidence analysis failed")
            with self._lock:
                self._error = str(exc)
