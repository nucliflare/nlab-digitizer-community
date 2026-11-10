"""One cancellable file reader; repeated requests replace the pending view."""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from nlab.analysis.current_capture import (
    CurrentCaptureInfo,
    CurrentReadCancelledError,
    read_current_capture_info,
    read_current_chunk,
)


class CurrentCaptureWorker(QThread):
    loaded = Signal(int, object, object)
    failed = Signal(int, str)

    def __init__(self) -> None:
        super().__init__()
        self._condition = threading.Condition()
        self._pending: tuple[int, Path, float, float, int] | None = None
        self._cancel = threading.Event()
        self._closing = False

    def submit(
        self,
        generation: int,
        path: Path,
        start_us: float,
        width_us: float,
        *,
        average_kernel: int = 0,
    ) -> None:
        with self._condition:
            self._cancel.set()
            self._pending = generation, path, start_us, width_us, average_kernel
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._closing = True
            self._cancel.set()
            self._pending = None
            self._condition.notify()

    def run(self) -> None:
        info: CurrentCaptureInfo | None = None
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closing or self._pending is not None)
                if self._closing:
                    return
                assert self._pending is not None
                generation, path, start_us, width_us, average_kernel = self._pending
                self._pending = None
                self._cancel = cancel = threading.Event()
            try:
                if info is None or info.path != path:
                    info = read_current_capture_info(path, cancelled=cancel.is_set)
                chunk = read_current_chunk(
                    info,
                    start_us,
                    width_us,
                    average_kernel=average_kernel,
                    cancelled=cancel.is_set,
                )
                if not cancel.is_set():
                    self.loaded.emit(generation, info, chunk)
            except CurrentReadCancelledError:
                pass
            except Exception as exc:
                info = None
                if not cancel.is_set():
                    self.failed.emit(generation, f"{type(exc).__name__}: {exc}")
