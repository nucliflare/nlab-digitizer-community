"""Background workers for DMA streaming.

Follow the BaseWorker + QThread pattern. Because run() blocks in a
ZMQ poll or IIO refill loop the worker thread's Qt event loop never spins, so
signal-based stop does NOT work.  Use worker.stop() (direct call to
threading.Event.set()) instead.
"""

from __future__ import annotations

import logging
import threading
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
from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)


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
    connect-then-arm step -- the first read_dma_frame() call both creates
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
        try:
            total = self._streamer.stream_to_file(
                filepath=self._filepath,
                stop_event=self._stop_event,
                n_frames=self._n_frames,
                on_ready=lambda: self.ready.emit(),
                on_progress=lambda n: self.progress.emit(n),
            )
            log.info("IIOScopeDmaWorker: completed, %d frames written", total)
        except Exception:
            log.exception("IIOScopeDmaWorker: streaming failed")
            self.error.emit("IIO scope DMA streaming failed")
        finally:
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
    progress = Signal(int)

    def __init__(
        self,
        streamer: McaDmaStreamer,
        filepath: Path | None = None,
        event_buffer: McaEventBuffer | tuple[list[np.ndarray], threading.Lock] | None = None,
        output_mode: McaDmaOutputMode = McaDmaOutputMode.BINARY,
        configuration_yaml: str = "",
    ) -> None:
        super().__init__()
        self._streamer = streamer
        self._filepath = filepath
        self._event_buffer = event_buffer
        self._output_mode = output_mode
        self._configuration_yaml = configuration_yaml
        self._stop_event = threading.Event()

    def run(self) -> None:
        log.info("McaDmaWorker: starting, file=%s", self._filepath)
        try:
            total = self._streamer.stream_events(
                stop_event=self._stop_event,
                filepath=self._filepath,
                event_buffer=self._event_buffer,
                on_ready=lambda: self.ready.emit(),
                on_progress=lambda n: self.progress.emit(n),
                output_mode=self._output_mode,
                configuration_yaml=self._configuration_yaml,
            )
            log.info("McaDmaWorker: completed, %d events received", total)
        except Exception:
            log.exception("McaDmaWorker: streaming failed")
            self.error.emit("MCA DMA streaming failed")
        finally:
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
    progress = Signal(int)

    def __init__(
        self,
        streamer: IIOMcaDmaStreamer,
        filepath: Path | None = None,
        event_buffer: McaEventBuffer | tuple[list[np.ndarray], threading.Lock] | None = None,
        output_mode: McaDmaOutputMode = McaDmaOutputMode.BINARY,
        configuration_yaml: str = "",
    ) -> None:
        super().__init__()
        self._streamer = streamer
        self._filepath = filepath
        self._event_buffer = event_buffer
        self._output_mode = output_mode
        self._configuration_yaml = configuration_yaml
        self._stop_event = threading.Event()

    def run(self) -> None:
        log.info("IIOMcaDmaWorker: starting, file=%s", self._filepath)
        try:
            total = self._streamer.stream_events(
                stop_event=self._stop_event,
                filepath=self._filepath,
                event_buffer=self._event_buffer,
                on_ready=lambda: self.ready.emit(),
                on_progress=lambda n: self.progress.emit(n),
                output_mode=self._output_mode,
                configuration_yaml=self._configuration_yaml,
            )
            log.info("IIOMcaDmaWorker: completed, %d records received", total)
        except Exception:
            log.exception("IIOMcaDmaWorker: streaming failed")
            self.error.emit("IIO MCA DMA streaming failed")
        finally:
            self.finished.emit()

    def stop(self) -> None:
        log.info("IIOMcaDmaWorker: stop requested")
        self._stop_event.set()
        # Stop the producer but deliberately keep the blocking refill and
        # buffer alive. vdpp_lm_frame completes the in-band final 16 KiB
        # frame; stream_events() then drains and closes it in driver order.
        self._streamer.request_stop()
