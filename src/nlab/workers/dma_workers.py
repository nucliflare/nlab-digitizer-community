"""Background workers for IIO DMA streaming.

Follow the BaseWorker + QThread pattern. Because run() blocks in an IIO
refill loop the worker thread's Qt event loop never spins, so signal-based
stop does NOT work. Use worker.stop() directly instead.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Signal

from nlab.hardware.digitizer.dma import (
    IIOMcaDmaStreamer,
    IIOScopeDmaStreamer,
    McaEventBuffer,
    ScopeFrameBuffer,
)
from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode, McaRunSummary
from nlab.workers.base_worker import BaseWorker, is_fatal_transport_error

if TYPE_CHECKING:
    from nlab.hardware.digitizer.current_monitor import ScopeCurrentAccumulator

log = logging.getLogger(__name__)
_SCOPE_PROGRESS_INTERVAL_S = 0.1


def _finish_mca_run(
    *,
    channel: int,
    mode: McaDmaOutputMode,
    path: Path | None,
    started_utc: datetime,
    started_monotonic: float,
    records: int,
    diagnostics: dict[str, int | bool] | None,
    error: str | None,
) -> McaRunSummary:
    if error is not None or (diagnostics is not None and not diagnostics.get("continuity_valid")):
        continuity = "invalid"
    elif diagnostics is not None:
        continuity = "verified"
    else:
        continuity = "unverified"
    summary = McaRunSummary(
        channel=channel,
        mode=mode,
        path=path,
        started_utc=started_utc.isoformat(),
        finished_utc=datetime.now(UTC).isoformat(),
        duration_s=max(0.0, time.monotonic() - started_monotonic),
        records=records,
        continuity=continuity,
        diagnostics=diagnostics or {},
        error=error,
    )
    try:
        summary.write_sidecar()
    except OSError as exc:
        log.exception("Could not save MCA run summary for %s", path)
        summary = replace(summary, metadata_error=str(exc))
    return summary


class IIOScopeDmaWorker(BaseWorker):
    """Streams IIO Scope DMA frames to a file and/or live consumers.

    The first read_dma_raw_frame() call creates the DMA buffer and arms the
    hardware. ``ready`` reports that lifecycle transition to the controller;
    the controller must not issue a second scope.start().

    Stop with ``worker.stop()`` (direct call, not signal).
    """

    ready = Signal()
    progress = Signal(object)

    def __init__(
        self,
        streamer: IIOScopeDmaStreamer,
        filepath: Path | None,
        n_frames: int | None = None,
        frame_buffer: ScopeFrameBuffer | None = None,
        current_accumulator: ScopeCurrentAccumulator | None = None,
    ) -> None:
        super().__init__()
        self._streamer = streamer
        self._filepath = filepath
        self._n_frames = n_frames
        self._frame_buffer = frame_buffer
        self._current_accumulator = current_accumulator
        self._stop_event = threading.Event()

    def run(self) -> None:
        log.info("IIOScopeDmaWorker: starting, file=%s", self._filepath)
        last_progress: int | None = None
        last_reported: int | None = None
        last_report_at = float("-inf")

        def report_progress(value: int) -> None:
            nonlocal last_progress, last_reported, last_report_at
            last_progress = value
            now = time.monotonic()
            if now - last_report_at >= _SCOPE_PROGRESS_INTERVAL_S:
                self.progress.emit(value)
                last_reported = value
                last_report_at = now

        try:
            total = self._streamer.stream_to_file(
                filepath=self._filepath,
                stop_event=self._stop_event,
                n_frames=self._n_frames,
                on_ready=lambda: self.ready.emit(),
                on_progress=report_progress,
                frame_buffer=self._frame_buffer,
                current_accumulator=self._current_accumulator,
            )
            log.info("IIOScopeDmaWorker: completed, %d frames captured", total)
        except Exception:
            log.exception("IIOScopeDmaWorker: streaming failed")
            self.error.emit("IIO scope DMA streaming failed")
        finally:
            if last_progress is not None and last_progress != last_reported:
                self.progress.emit(last_progress)
            self.finished.emit()

    def stop(self) -> None:
        log.info("IIOScopeDmaWorker: stop requested")
        self._stop_event.set()
        self._streamer.request_stop()


class IIOMcaDmaWorker(BaseWorker):
    """Streams fixed-frame MCA list-mode records through IIO.

    The backend owns arm/start and stop/drain/close ordering. ``ready`` is
    retained for UI state and polling-worker startup, but the controller
    must not call mca.start() in response: the first blocking refill does
    that only after the reader is present.
    """

    ready = Signal()
    progress = Signal(object)
    summary = Signal(object)
    connection_lost = Signal(str)

    def __init__(
        self,
        streamer: IIOMcaDmaStreamer,
        filepath: Path | None = None,
        event_buffer: McaEventBuffer | tuple[list[np.ndarray], threading.Lock] | None = None,
        output_mode: McaDmaOutputMode = McaDmaOutputMode.BINARY,
        configuration_yaml: str = "",
        channel: int = 0,
        client_record_schema: str | None = None,
    ) -> None:
        super().__init__()
        self._streamer = streamer
        self._filepath = filepath
        self._event_buffer = event_buffer
        self._output_mode = output_mode
        self._configuration_yaml = configuration_yaml
        self._channel = channel
        self._client_record_schema = client_record_schema
        self._stop_event = threading.Event()

    def run(self) -> None:
        log.info("IIOMcaDmaWorker: starting, file=%s", self._filepath)
        started_utc = datetime.now(UTC)
        started_monotonic = time.monotonic()
        records = 0
        error: str | None = None

        def report_progress(count: int) -> None:
            nonlocal records
            records = count
            self.progress.emit(count)

        try:
            total = self._streamer.stream_events(
                stop_event=self._stop_event,
                filepath=self._filepath,
                event_buffer=self._event_buffer,
                on_ready=lambda: self.ready.emit(),
                on_progress=report_progress,
                output_mode=self._output_mode,
                configuration_yaml=self._configuration_yaml,
                client_record_schema=self._client_record_schema,
            )
            records = total
            log.info("IIOMcaDmaWorker: completed, %d records received", total)
        except Exception as exc:
            error = str(exc)
            log.exception("IIOMcaDmaWorker: streaming failed")
            if is_fatal_transport_error(exc):
                self.connection_lost.emit(error)
            self.error.emit(f"IIO MCA DMA streaming failed: {error}")
        finally:
            self.summary.emit(
                _finish_mca_run(
                    channel=self._channel,
                    mode=self._output_mode,
                    path=self._filepath,
                    started_utc=started_utc,
                    started_monotonic=started_monotonic,
                    records=records,
                    diagnostics=self._streamer.last_capture_diagnostics,
                    error=error,
                )
            )
            self.finished.emit()

    def stop(self) -> None:
        log.info("IIOMcaDmaWorker: stop requested")
        self._stop_event.set()
        # Stop the producer but deliberately keep the blocking refill and
        # buffer alive. vdpp_lm_frame completes the in-band final 16 KiB
        # frame; stream_events() then drains and closes it in driver order.
        self._streamer.request_stop()
