"""One cancellable whole-capture spectrum computation per request."""

from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal

from nlab.analysis.current_capture import CurrentCaptureInfo, CurrentReadCancelledError
from nlab.analysis.sum_spectrum import capture_sum_spectrum


class SumSpectrumWorker(QThread):
    """Read a recorded sum capture once and report its window-mean spectrum.

    ``computed`` and ``failed`` carry the request's generation; a cancelled
    request reports nothing. Progress is emitted in whole-percent steps.
    """

    computed = Signal(int, object)
    failed = Signal(int, str)
    progressed = Signal(int, int)

    def __init__(
        self,
        generation: int,
        info: CurrentCaptureInfo,
        *,
        segment_length: int,
        bridge_single_gaps: bool,
    ) -> None:
        super().__init__()
        self.generation = generation
        self._info = info
        self._segment_length = segment_length
        self._bridge = bridge_single_gaps
        self._cancel = threading.Event()
        self._percent = -1

    def cancel(self) -> None:
        self._cancel.set()

    def _progress(self, fraction: float) -> None:
        percent = int(100 * fraction)
        if percent != self._percent:
            self._percent = percent
            self.progressed.emit(self.generation, percent)

    def run(self) -> None:
        try:
            result = capture_sum_spectrum(
                self._info,
                segment_length=self._segment_length,
                bridge_single_gaps=self._bridge,
                cancelled=self._cancel.is_set,
                progress=self._progress,
            )
        except CurrentReadCancelledError:
            return
        except Exception as exc:
            if not self._cancel.is_set():
                self.failed.emit(self.generation, f"{type(exc).__name__}: {exc}")
            return
        if not self._cancel.is_set():
            self.computed.emit(self.generation, result)
