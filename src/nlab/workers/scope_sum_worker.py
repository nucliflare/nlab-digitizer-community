"""Bounded sum capture/replay worker shared by the Current view and Scope owner."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import h5py

from nlab.hardware.digitizer.scope_sums import ScopeSumAnalysis, ScopeSumCapture, read_sum_capture


class ScopeSumWorker:
    def __init__(
        self, *, capture: ScopeSumCapture | None = None, replay: Path | None = None
    ) -> None:
        if (capture is None) == (replay is None):
            raise ValueError("choose either live capture or recorded sums")
        self.capture = capture
        self.replay = replay
        self.analysis = capture.analysis if capture is not None else ScopeSumAnalysis()
        self.thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = False
        self.error = ""
        self.result = (
            "Acquiring periodic sums." if capture is not None else "Reading recorded sums."
        )

    @property
    def done(self) -> bool:
        return self._started and (self.thread is None or not self.thread.is_alive())

    @property
    def message(self) -> str:
        return self.error or self.result

    def start(self, output: Path | None = None, *, overwrite: bool = False) -> None:
        if self._started:
            raise RuntimeError("sum workers cannot be restarted")
        self.thread = threading.Thread(
            target=self._run, args=(output, overwrite), name="current-sums", daemon=False
        )
        self.thread.start()
        self._started = True

    def _run(self, output: Path | None, overwrite: bool) -> None:
        try:
            if self.capture is not None:
                result = self.capture.run(output, overwrite=overwrite)
                self.result = (
                    "Measurement stopped; tail drained and buffer closed. "
                    f"Final hardware missed windows: {result.get('final_missed_windows', 0)}."
                    if result.get("complete") else "Stopped before a complete capture."
                )
            else:
                assert self.replay is not None
                with h5py.File(self.replay, "r") as source:
                    complete = bool(source.attrs.get("complete", False))
                    cleanup = str(source.attrs.get("cleanup_error", ""))
                    metadata = json.loads(str(source.attrs.get("metadata_json", "{}")))
                for packet in read_sum_capture(self.replay):
                    if self._stop.is_set():
                        break
                    self.analysis.append(packet)
                state = "Replay stopped" if self._stop.is_set() else "Recorded measurement"
                self.result = (
                    f"{state}: {self.replay.name}; source channel {metadata.get('channel', '?')}. "
                    f"Complete={complete}. {cleanup}"
                )
        except Exception as exc:
            self.error = (self.error + "; " if self.error else "") + str(exc)

    def request_stop(self) -> None:
        self._stop.set()
        if self.capture is not None and not self.done:
            try:
                self.capture.request_stop()
            except Exception as exc:
                self.error = f"Stop request: {exc}"

    def join(self) -> None:
        """Join before releasing the hardware or destroying the view; idempotent."""
        if self.thread is not None:
            if self.thread.ident is not None:
                self.thread.join()
            self.thread = None
