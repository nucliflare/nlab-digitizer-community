"""Background workers for DMA streaming.

Follow the BaseWorker + QThread pattern. Because run() blocks in a
ZMQ poll or IIO refill loop the worker thread's Qt event loop never spins, so
signal-based stop does NOT work.  Use worker.stop() (direct call to
threading.Event.set()) instead.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal

from nlab.hardware.digitizer.dma import (
    IIOMcaDmaStreamer,
    IIOScopeDmaStreamer,
    McaDmaStreamer,
    McaEventBuffer,
    ScopeDmaStreamer,
)
from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode, McaRunSummary
from nlab.workers.base_worker import BaseWorker

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


class ScopeDmaWorker(BaseWorker):
    """Streams scope DMA waveforms to a binary file.

    Emits ``ready`` once the ZMQ socket is connected and subscribed.
    The controller should enable DMA on the hardware in response so
    the StreamSTART sentinel is not missed.

    Stop with ``worker.stop()`` (direct call, not signal).
    """

    ready = Signal()
    progress = Signal(int)

    def __init__(
        self,
        streamer: ScopeDmaStreamer,
        filepath: Path,
        frame_samples: int,
    ) -> None:
        super().__init__()
        self._streamer = streamer
        self._filepath = filepath
        self._frame_samples = frame_samples
        self._stop_event = threading.Event()

    def run(self) -> None:
        log.info("ScopeDmaWorker: starting, file=%s", self._filepath)
        try:
            total = self._streamer.stream_to_file(
                filepath=self._filepath,
                frame_samples=self._frame_samples,
                stop_event=self._stop_event,
                on_ready=lambda: self.ready.emit(),
                on_progress=lambda n: self.progress.emit(n),
            )
            log.info("ScopeDmaWorker: completed, %d bytes written", total)
        except Exception:
            log.exception("ScopeDmaWorker: streaming failed")
            self.error.emit("Scope DMA streaming failed")
        finally:
            self.finished.emit()

    def stop(self) -> None:
        log.info("ScopeDmaWorker: stop requested")
        self._stop_event.set()


class IIOScopeDmaWorker(BaseWorker):
    """Streams full-resolution DMA frames to a binary file via the IIO
    backend's pull-based IIOScopeDmaStreamer.

    Not interchangeable with ScopeDmaWorker: IIOScopeDmaStreamer.
    stream_to_file() takes n_frames instead of frame_samples (it reads
    frame_samples itself from the backend at capture time) and has no
    connect-then-arm step -- the first read_dma_raw_frame() call both creates
    the DMA buffer and arms the hardware (see IIOScopeDmaStreamer's
    docstring). ``ready`` is kept for interface parity with ScopeDmaWorker
    but the controller must NOT call scope.start() in response to it for
    this streamer type -- see ScopeController._on_dma_ready().

    Stop with ``worker.stop()`` (direct call, not signal).
    """

    ready = Signal()
    progress = Signal(int)

    def __init__(
        self,
        streamer: IIOScopeDmaStreamer,
        filepath: Path,
        n_frames: int | None = None,
    ) -> None:
        super().__init__()
        self._streamer = streamer
        self._filepath = filepath
        self._n_frames = n_frames
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
            )
            log.info("IIOScopeDmaWorker: completed, %d frames written", total)
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


class McaDmaWorker(BaseWorker):
    """Streams MCA list-mode events via ZMQ.

    Emits ``ready`` once the ZMQ socket is connected and subscribed.
    The controller should enable DMA and start the measurement in
    response so the StreamSTART sentinel is not missed.

    Stop with ``worker.stop()`` (direct call, not signal).
    """

    ready = Signal()
    progress = Signal(object)
    summary = Signal(object)

    def __init__(
        self,
        streamer: McaDmaStreamer,
        filepath: Path | None = None,
        event_buffer: McaEventBuffer | tuple[list[np.ndarray], threading.Lock] | None = None,
        output_mode: McaDmaOutputMode = McaDmaOutputMode.BINARY,
        configuration_yaml: str = "",
        channel: int = 0,
    ) -> None:
        super().__init__()
        self._streamer = streamer
        self._filepath = filepath
        self._event_buffer = event_buffer
        self._output_mode = output_mode
        self._configuration_yaml = configuration_yaml
        self._channel = channel
        self._stop_event = threading.Event()

    def run(self) -> None:
        log.info("McaDmaWorker: starting, file=%s", self._filepath)
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
            )
            records = total
            log.info("McaDmaWorker: completed, %d events received", total)
        except Exception as exc:
            error = str(exc)
            log.exception("McaDmaWorker: streaming failed")
            self.error.emit(f"MCA DMA streaming failed: {error}")
        finally:
            self.summary.emit(
                _finish_mca_run(
                    channel=self._channel,
                    mode=self._output_mode,
                    path=self._filepath,
                    started_utc=started_utc,
                    started_monotonic=started_monotonic,
                    records=records,
                    diagnostics=None,
                    error=error,
                )
            )
            self.finished.emit()

    def stop(self) -> None:
        log.info("McaDmaWorker: stop requested")
        self._stop_event.set()


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

    def __init__(
        self,
        streamer: IIOMcaDmaStreamer,
        filepath: Path | None = None,
        event_buffer: McaEventBuffer | tuple[list[np.ndarray], threading.Lock] | None = None,
        output_mode: McaDmaOutputMode = McaDmaOutputMode.BINARY,
        configuration_yaml: str = "",
        channel: int = 0,
    ) -> None:
        super().__init__()
        self._streamer = streamer
        self._filepath = filepath
        self._event_buffer = event_buffer
        self._output_mode = output_mode
        self._configuration_yaml = configuration_yaml
        self._channel = channel
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
            )
            records = total
            log.info("IIOMcaDmaWorker: completed, %d records received", total)
        except Exception as exc:
            error = str(exc)
            log.exception("IIOMcaDmaWorker: streaming failed")
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
