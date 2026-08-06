"""ZMQ-based DMA streaming for scope waveforms and MCA list-mode events.

These classes manage ZMQ SUB socket connections to the digitizer's
streaming endpoints.  They are transport-layer components, analogous
to GrpcIDSBackend -- independent connections instantiated alongside
the main gRPC backend.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zmq

if TYPE_CHECKING:
    from .backends.iio_backend import IIODigitizerBackend

log = logging.getLogger(__name__)

SCOPE_DMA_BASE_PORT = 50152
MCA_DMA_BASE_PORT = 50162

STREAM_START = b"StreamSTART\x00"
STREAM_END = b"StreamEND\x00"

EVENT_STRUCT = struct.Struct("<BBHHHQ")
EVENT_SIZE = EVENT_STRUCT.size  # 14

FILE_HEADER_STRUCT = struct.Struct("<4sHBBdI4x")  # 24 bytes
FILE_MAGIC = b"NDMA"
FILE_VERSION = 1

# Each raw scope DMA frame is prefixed with a per-frame timestamp: the first
# 4 int16 slots (8 bytes) are a little-endian uint64, the remaining
# (frame_samples - SCOPE_TIMESTAMP_WORDS) slots are the int16 waveform.
SCOPE_TIMESTAMP_WORDS = 4

_EVENT_DTYPE = np.dtype([
    ("marker", np.uint8),
    ("zc_offset", np.uint8),
    ("zc_estimation", np.uint16),
    ("short_energy", np.uint16),
    ("energy", np.uint16),
    ("timestamp", np.uint64),
])


def _write_file_header(f, channel: int, frame_samples: int = 0) -> None:
    header = FILE_HEADER_STRUCT.pack(
        FILE_MAGIC, FILE_VERSION, channel, 0, time.time(), frame_samples,
    )
    f.write(header)
    f.flush()


class ScopeDmaStreamer:
    """Manages ZMQ SUB connection for scope DMA waveform streaming.

    Not a backend -- a standalone connection manager.  Create via
    Digitizer, use from a dedicated worker thread.
    """

    def __init__(self, channel: int, hostname: str = "192.168.10.20") -> None:
        self._channel = channel
        self._hostname = hostname
        self._port = SCOPE_DMA_BASE_PORT + channel - 1
        self._endpoint = f"tcp://{hostname}:{self._port}"

    def stream_to_file(
        self,
        filepath: Path,
        frame_samples: int,
        stop_event: threading.Event,
        on_ready: Callable[[], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> int:
        """Connect, signal ready, wait for StreamSTART, write frames until StreamEND or stop.

        *on_ready* is called after the ZMQ socket is connected and
        subscribed — the caller should enable DMA on the hardware at
        this point so the StreamSTART sentinel is not missed.

        Returns total bytes written.  Runs in caller's thread --
        creates and destroys its own ZMQ context (thread-local).
        """
        ctx = zmq.Context()
        socket = ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)      # don't block ctx.term() on unsent messages
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        log.debug("Scope DMA [worker 2a]: connecting ZMQ SUB to %s", self._endpoint)
        socket.connect(self._endpoint)
        time.sleep(0.2)
        log.debug("Scope DMA [worker 2b]: ZMQ connected, emitting ready "
                  "(controller will enable DMA + call start)")
        if on_ready is not None:
            on_ready()

        total_bytes = 0
        frame_count = 0
        started = False

        try:
            with open(filepath, "wb") as f:
                log.info("Scope DMA: recording to %s", filepath)
                _write_file_header(f, self._channel, frame_samples)
                log.debug("Scope DMA [worker 6/6]: polling for StreamSTART "
                          "(waiting for HW start_irq after scope.start())")

                while not stop_event.is_set():
                    events = dict(poller.poll(100))
                    if socket not in events:
                        continue

                    message = socket.recv()

                    if not started:
                        if message == STREAM_START:
                            log.info("Scope DMA: StreamSTART received "
                                     "(HW start_irq fired, DMA engine running)")
                            started = True
                            continue
                        if message != STREAM_END and len(message) > len(STREAM_START):
                            log.warning("Scope DMA: data arrived before StreamSTART "
                                        "(%d bytes) — server missed sentinel, "
                                        "treating as stream active", len(message))
                            started = True
                            f.write(message)
                            total_bytes += len(message)
                            frame_count += 1
                            continue
                        log.debug("Scope DMA: pre-start message, %d bytes (ignoring)",
                                  len(message))
                        continue

                    if message == STREAM_END:
                        log.info("Scope DMA: StreamEND received "
                                 "(HW stop_irq fired, scope.stop() was called)")
                        break

                    f.write(message)
                    total_bytes += len(message)
                    frame_count += 1
                    if frame_count == 1:
                        log.debug("Scope DMA: first data frame received, %d bytes "
                                  "(DMA transfers active)", len(message))

                    if on_progress is not None:
                        on_progress(total_bytes)

        except zmq.ZMQError:
            log.exception("Scope DMA: ZMQ error during streaming")
            raise
        finally:
            reason = "stop_event" if stop_event.is_set() else "StreamEND"
            log.info("Scope DMA: finished -- %d frames, %d bytes to %s (reason: %s)",
                     frame_count, total_bytes, filepath, reason)
            socket.close()
            ctx.term()

        return total_bytes


class IIOScopeDmaStreamer:
    """Full-resolution DMA-frame-to-file streaming for the IIO backend.

    Not a drop-in replacement for ScopeDmaStreamer: that class is a ZMQ SUB
    client fed by the gRPC Engine's own push-based DMA server, which
    continuously streams frames captured by hardware running free. The IIO
    scope core has no equivalent continuous-streaming path — it's a
    one-shot triggered-capture design (confirmed via ewt-scope-iio.c and
    extensive live testing: reusing a buffer across multiple refills
    returns corrupted data). This class instead *pulls* frames by looping
    IIODigitizerBackend.read_dma_frame() — each call does its own full
    arm/refill/read/destroy cycle — and writes each one to file as it
    arrives, rather than subscribing to a continuous push.

    Uses the same NDMA file header as ScopeDmaStreamer for tooling
    consistency, but the per-frame record layout is IIO's own (8-byte
    timestamp header immediately followed by the *full* frame_samples
    waveform) rather than gRPC's (timestamp overlaid into the first 4
    samples of the waveform array) — the two hardware frame formats
    genuinely differ, this doesn't try to force compatibility between them.
    """

    def __init__(self, backend: "IIODigitizerBackend", channel: int) -> None:
        self._backend = backend
        self._channel = channel

    def stream_to_file(
        self,
        filepath: Path,
        stop_event: threading.Event,
        n_frames: int | None = None,
        on_ready: Callable[[], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> int:
        """Repeatedly capture full-resolution frames and append them to file.

        Runs until *stop_event* is set or *n_frames* frames have been
        written (None = unbounded). *on_ready* is called once, right
        before the first capture attempt -- at that point no DMA buffer
        exists yet (it's created lazily by the first read_dma_frame()
        call), and that first call is what actually arms the scope (per
        vdpp-scope.c's postenable(), which sets both ENABLE and DMA_ENABLE
        when the buffer is enabled). Unlike the gRPC/ZMQ streamers, there
        is no separate hardware-arm step for a caller to perform in
        response to on_ready -- calling scope.start() here would be
        redundant at best and racy at worst (see ScopeController, which
        skips it for this streamer type).

        Returns the number of frames actually written — a failed capture
        (e.g. the known xilinx-vdma channel-stop issue, see references/...
        bug report) stops the loop and propagates the exception rather
        than silently skipping the frame, since a silently-incomplete file
        would be worse than a loud failure.

        *on_progress* is called with the total bytes written so far, not
        a frame count -- matching ScopeDmaStreamer's convention (which
        passes total_bytes), since ScopeController._on_dma_progress()
        treats the value as a byte count for its KB/MB display regardless
        of which streamer is in use. Passing frame_count here instead
        used to make the GUI display a wildly wrong size (e.g. "20 KB"
        for what was actually a 56 MB file, since 20,000-ish frames were
        being read as 20,000-ish bytes).

        Like ScopeDmaStreamer, the one-time file header written by
        _write_file_header() is not included in the count -- both
        streamers under-report the true file size by that fixed 24-byte
        header, consistently, not a new inconsistency introduced here.

        Always closes the DMA capture buffer before returning, success,
        stop_event, or exception alike -- per vdpp-scope.c's predisable(),
        that's what clears ENABLE/DMA_ENABLE back down for the DMA case;
        leaving the buffer open here would leave the hardware armed after
        the caller thinks the measurement has stopped.
        """
        frame_samples = self._backend.get_frame_samples()
        frame_count = 0
        total_bytes = 0

        try:
            with open(filepath, "wb") as f:
                log.info("IIO scope DMA: recording to %s", filepath)
                _write_file_header(f, self._channel, frame_samples)

                if on_ready is not None:
                    on_ready()

                while not stop_event.is_set():
                    if n_frames is not None and frame_count >= n_frames:
                        break

                    timestamp, samples = self._backend.read_dma_frame()
                    ts_bytes = struct.pack("<Q", timestamp)
                    payload_bytes = samples.tobytes()
                    f.write(ts_bytes)
                    f.write(payload_bytes)
                    f.flush()

                    frame_count += 1
                    total_bytes += len(ts_bytes) + len(payload_bytes)
                    if on_progress is not None:
                        on_progress(total_bytes)
        finally:
            self._backend.close_dma_capture()

        log.info("IIO scope DMA: finished -- %d frames to %s", frame_count, filepath)
        return frame_count


class McaDmaStreamer:
    """Manages ZMQ SUB connection for MCA list-mode DMA streaming.

    Not a backend -- a standalone connection manager.  Create via
    Digitizer, use from a dedicated worker thread.
    """

    def __init__(self, channel: int, hostname: str = "192.168.10.20") -> None:
        self._channel = channel
        self._hostname = hostname
        self._port = MCA_DMA_BASE_PORT + channel - 1
        self._endpoint = f"tcp://{hostname}:{self._port}"

    def stream_events(
        self,
        stop_event: threading.Event,
        filepath: Path | None = None,
        event_buffer: tuple[list, threading.Lock] | None = None,
        on_ready: Callable[[], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> int:
        """Connect, signal ready, wait for StreamSTART, read events until StreamEND or stop.

        *on_ready* is called after the ZMQ socket is connected and
        subscribed — the caller should enable DMA and start the
        measurement at this point so the StreamSTART sentinel is not missed.

        Events are simultaneously:
        - Written to *filepath* as raw binary (if provided)
        - Parsed and appended to *event_buffer* (if provided)

        Returns total event count.  Runs in caller's thread.
        """
        ctx = zmq.Context()
        socket = ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)      # don't block ctx.term() on unsent messages
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        log.debug("MCA DMA [worker 2a]: connecting ZMQ SUB to %s", self._endpoint)
        socket.connect(self._endpoint)
        time.sleep(0.2)
        log.debug("MCA DMA [worker 2b]: ZMQ connected, emitting ready "
                  "(controller will enable DMA + call start)")
        if on_ready is not None:
            on_ready()

        total_events = 0
        msg_count = 0
        started = False
        file_handle = None

        try:
            if filepath is not None:
                file_handle = open(filepath, "wb")
                log.info("MCA DMA: recording to %s", filepath)
                _write_file_header(file_handle, self._channel)

            log.debug("MCA DMA [worker 6/6]: polling for StreamSTART "
                      "(waiting for HW list_start_irq after mca.start())")

            while not stop_event.is_set():
                events = dict(poller.poll(100))
                if socket not in events:
                    continue

                message = socket.recv()

                if not started:
                    if message == STREAM_START:
                        log.info("MCA DMA: StreamSTART received "
                                 "(HW list_start_irq fired, DMA engine running)")
                        started = True
                        continue
                    if message != STREAM_END and len(message) > len(STREAM_START):
                        log.warning("MCA DMA: data arrived before StreamSTART "
                                    "(%d bytes) — server missed sentinel, "
                                    "treating as stream active", len(message))
                        started = True
                        # Fall through to process this message as data
                    else:
                        log.debug("MCA DMA: pre-start message, %d bytes (ignoring)",
                                  len(message))
                        continue

                if message == STREAM_END:
                    log.info("MCA DMA: StreamEND received "
                             "(HW list_stop_irq fired, mca.stop() was called)")
                    break

                if file_handle is not None:
                    file_handle.write(message)

                n_events = len(message) // EVENT_SIZE
                if len(message) % EVENT_SIZE != 0:
                    log.warning(
                        "MCA DMA: message size %d not aligned to event size %d, truncating",
                        len(message), EVENT_SIZE,
                    )

                total_events += n_events
                msg_count += 1
                if msg_count == 1:
                    log.debug("MCA DMA: first event frame received, %d bytes, %d events "
                              "(DMA transfers active)", len(message), n_events)

                if event_buffer is not None and n_events > 0:
                    parsed = self.parse_events(message[:n_events * EVENT_SIZE])
                    buf, lock = event_buffer
                    with lock:
                        buf.append(parsed)

                if on_progress is not None:
                    on_progress(total_events)

        except zmq.ZMQError:
            log.exception("MCA DMA: ZMQ error during streaming")
            raise
        finally:
            reason = "stop_event" if stop_event.is_set() else "StreamEND"
            log.info("MCA DMA: finished -- %d events from %d messages (reason: %s)",
                     total_events, msg_count, reason)
            if file_handle is not None:
                file_handle.close()
                log.info("MCA DMA: file closed: %s", filepath)
            socket.close()
            ctx.term()

        return total_events

    @staticmethod
    def parse_events(raw: bytes) -> np.ndarray:
        """Parse raw binary into structured numpy array."""
        return np.frombuffer(raw, dtype=_EVENT_DTYPE)

    @staticmethod
    def compute_psd(events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute PSD from structured event array.

        Returns (energy, psd_zc) arrays.
        """
        energy = events["energy"].astype(np.float64)
        psd_zc = events["zc_offset"].astype(np.float64) + (
            events["zc_estimation"].view(np.int16).astype(np.float64) / 2**14
        )
        return energy, psd_zc
