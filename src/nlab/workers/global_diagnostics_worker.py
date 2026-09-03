from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QTimer, Signal, Slot

from nlab.hardware.digitizer.digitizer import Digitizer
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)


class GlobalDiagnosticsWorker(BaseWorker):
    """Poll channel-independent digitizer sensors on a dedicated thread."""

    readback = Signal(object)
    change_interval = Signal(int)
    request_stop = Signal()

    def __init__(self, device: Digitizer, interval_ms: int = 1000) -> None:
        super().__init__()
        self._device = device
        self._interval_ms = interval_ms
        self._timer: QTimer | None = None
        self._stop_requested = threading.Event()
        self._finished = False

    def run(self) -> None:
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self.change_interval.connect(self._set_interval)
        self.request_stop.connect(self._stop)
        if self._stop_requested.is_set():
            self._stop()
            return
        self._tick()
        if not self._finished:
            self._timer.start(self._interval_ms)

    def request_shutdown(self) -> None:
        """Request shutdown from any thread, including during a blocking read."""
        self._stop_requested.set()
        self.request_stop.emit()

    @Slot()
    def _stop(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.finished.emit()

    @Slot(int)
    def _set_interval(self, interval_ms: int) -> None:
        if self._timer is not None:
            self._timer.setInterval(interval_ms)

    @Slot()
    def _tick(self) -> None:
        if self._stop_requested.is_set():
            self._stop()
            return
        try:
            readings = self._device.get_global_diagnostics()
        except Exception as exc:
            if self._stop_requested.is_set():
                self._stop()
                return
            log.exception("Global diagnostics readback failed")
            self.error.emit(str(exc))
            return
        # A queued request_stop signal cannot run while the synchronous IIO
        # read above occupies this thread.  Observe the thread-safe flag here
        # so shutdown completes immediately when that in-flight read returns.
        if self._stop_requested.is_set():
            self._stop()
            return
        self.readback.emit(readings)
