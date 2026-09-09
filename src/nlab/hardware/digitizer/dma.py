"""DMA streaming for scope waveforms and MCA list-mode events.

The legacy streamers manage ZMQ SUB connections to the gRPC Engine. The
IIO streamers pull complete blocks from IIODigitizerBackend on dedicated
worker threads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np
import zmq


class _IIOScopeBackend(Protocol):
    def prepare_dma_capture(self) -> None: ...

    def get_frame_samples(self) -> int: ...

    def read_dma_frame(self) -> tuple[int, np.ndarray]: ...

    def request_dma_stop(self) -> None: ...

    def close_dma_capture(self) -> None: ...


class _IIOMcaBackend(Protocol):
    def start_mca_dma_capture(
        self,
        on_started: Callable[[], None] | None = None,
    ) -> np.ndarray: ...

    def read_mca_dma_frame(self) -> np.ndarray: ...

    def mca_dma_measurement_in_progress(self) -> bool: ...

    def request_mca_dma_stop(self) -> None: ...

    def close_mca_dma_capture(
        self,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> int: ...

    def get_mca_dma_capture_diagnostics(self) -> tuple[int, int, int, int]: ...


log = logging.getLogger(__name__)

SCOPE_DMA_BASE_PORT = 50152
MCA_DMA_BASE_PORT = 50162

STREAM_START = b"StreamSTART\x00"
STREAM_END = b"StreamEND\x00"

EVENT_STRUCT = struct.Struct("<BBHHHQ")
EVENT_SIZE = EVENT_STRUCT.size  # 16 bytes

FILE_HEADER_STRUCT = struct.Struct("<4sHBBdI4x")  # 24 bytes
FILE_MAGIC = b"NDMA"
FILE_VERSION = 1
IIO_LM_FILE_VERSION = 2
IIO_LM_FRAME_RECORDS = 1024
IIO_LM_RECORD_BYTES = 16
IIO_LM_FRAME_BYTES = IIO_LM_FRAME_RECORDS * IIO_LM_RECORD_BYTES
IIO_LM_KERNEL_BUFFER_COUNT = 8
IIO_LM_CLIENT_SCHEMA = "vdpp-pulse-processor-event-v1"

# Each raw scope DMA frame is prefixed with a per-frame timestamp: the first
# 4 int16 slots (8 bytes) are a little-endian uint64, the remaining
# (frame_samples - SCOPE_TIMESTAMP_WORDS) slots are the int16 waveform.
SCOPE_TIMESTAMP_WORDS = 4

# Keep file-system stalls out of the time-critical IIO refill loop while still
# bounding memory use. At the largest legal scope frame this queue occupies
# roughly 4 MiB. If storage remains slower than acquisition after the queue is
# full, producer backpressure is intentional: silently dropping raw frames
# would make the capture file look valid while losing data.
_IIO_SCOPE_WRITE_QUEUE_FRAMES = 256
_IIO_SCOPE_WRITE_QUEUE_POLL_SECONDS = 0.050
_IIO_SCOPE_WRITER_STOP = object()

_EVENT_DTYPE = np.dtype([
    ("marker", np.uint8),
    ("zc_offset", np.uint8),
    ("zc_estimation", np.uint16),
    ("short_energy", np.uint16),
    ("energy", np.uint16),
    ("timestamp", np.uint64),
])

# vdpp-lm-frame.c v121: one unchanged 128-bit little-endian scan. This is
# deliberately separate from _EVENT_DTYPE above: the first four bytes have
# different field boundaries and semantics even though both records happen
# to total 16 bytes.
_LM_EVENT_DTYPE = np.dtype([
    ("flags", "<u2"),
    ("cfd_q2", "<u2"),
    ("charge_energy", "<u2"),
    ("trapezoid_energy", "<u2"),
    ("timestamp", "<u8"),
])


class McaEventBuffer:
    """Bounded hand-off queue for live consumers of MCA DMA batches.

    File recording remains synchronous in the streamer. If the GUI cannot
    drain this queue quickly enough, only the oldest display batch is
    discarded; raw capture continuity is unaffected.
    """

    def __init__(self, max_batches: int = 128) -> None:
        if max_batches <= 0:
            raise ValueError("max_batches must be positive")
        self._max_batches = max_batches
        self._batches: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._dropped_records = 0

    def append(self, events: np.ndarray) -> None:
        copied = events.copy()
        with self._lock:
            if len(self._batches) >= self._max_batches:
                self._dropped_records += len(self._batches.popleft())
            self._batches.append(copied)

    def clear(self) -> None:
        with self._lock:
            self._batches.clear()
            self._dropped_records = 0

    def drain(self) -> tuple[list[np.ndarray], int]:
        """Atomically remove pending batches and report cumulative drops."""
        with self._lock:
            batches = list(self._batches)
            self._batches.clear()
            return batches, self._dropped_records


def _append_mca_event_batch(
    target: McaEventBuffer | tuple[list[np.ndarray], threading.Lock],
    events: np.ndarray,
) -> None:
    """Support the bounded GUI queue and the legacy tuple test/API shape."""
    if isinstance(target, McaEventBuffer):
        target.append(events)
        return
    batches, lock = target
    with lock:
        batches.append(events.copy())


def _write_file_header(
    f: BinaryIO,
    channel: int,
    frame_samples: int = 0,
    version: int = FILE_VERSION,
) -> None:
    header = FILE_HEADER_STRUCT.pack(
        FILE_MAGIC, version, channel, 0, time.time(), frame_samples,
    )
    f.write(header)
    f.flush()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_iio_mca_metadata(
    path: Path,
    *,
    channel: int,
    started_utc: datetime,
    frame_count: int,
    dma_fault: int,
    dma_error_count: int,
    completed_frames: int,
    deadtime_records: int,
) -> Path:
    """Write the auditable sidecar used by the reference capture workflow."""
    metadata_path = path.with_suffix(".json")
    continuity_valid = (
        dma_fault == 0
        and completed_frames == frame_count
        and deadtime_records == 0
    )
    metadata = {
        "format": "nlab-iio-mca-ndma-v2",
        "capture": str(path.resolve()),
        "capture_sha256": _sha256_file(path),
        "channel_index": channel,
        "started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "ndma_header_bytes": FILE_HEADER_STRUCT.size,
        "record_layout": "opaque[16]",
        # The kernel intentionally promises only opaque[16]. Decoded PSD
        # fields belong to this explicitly named, replaceable client schema.
        "client_record_schema": IIO_LM_CLIENT_SCHEMA,
        "record_bytes": IIO_LM_RECORD_BYTES,
        "frame_records": IIO_LM_FRAME_RECORDS,
        "frame_bytes": IIO_LM_FRAME_BYTES,
        "kernel_buffers": IIO_LM_KERNEL_BUFFER_COUNT,
        "frames": frame_count,
        "records": frame_count * IIO_LM_FRAME_RECORDS,
        "driver_completed_frames": completed_frames,
        "driver_dma_fault": dma_fault,
        "driver_dma_error_count": dma_error_count,
        "list_deadtime_raw": deadtime_records,
        "continuity_valid": continuity_valid,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata_path


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
    scope core exposes a pull-based IIO buffer rather than the legacy ZMQ
    push stream. This class arms one buffer for the measurement, repeatedly
    refills it and hands copied, complete records to a dedicated file-writer
    thread. The refill loop therefore starts waiting for the next hardware
    frame as soon as the preceding record has entered the bounded queue; disk
    writes and ordinary file-system latency do not extend that interval.

    Uses the same NDMA file header as ScopeDmaStreamer for tooling
    consistency, but the per-frame record layout is IIO's own (8-byte
    timestamp header followed by ``frame_samples - 4`` waveform values).
    The timestamp occupies the first four 16-bit slots of the hardware frame,
    so each file record remains exactly ``frame_samples * 2`` bytes.
    """

    def __init__(self, backend: _IIOScopeBackend, channel: int) -> None:
        self._backend = backend
        self._channel = channel

    def request_stop(self) -> None:
        """Stop the acquisition gate and cancel a blocked backend refill."""
        self._backend.request_dma_stop()

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
        queued for writing (None = unbounded). *on_ready* is called once the
        writer has created and flushed the one-time NDMA header, right before
        the first capture attempt. At that point no DMA buffer exists yet
        (it's created lazily by the first read_dma_frame() call), and that
        first call arms DMA, starts its blocking reader and then writes
        ``enable=1`` in the backend-defined order. There is no separate
        hardware-arm operation for the controller.

        Returns the number of frames actually written — a failed capture
        (e.g. the known xilinx-vdma channel-stop issue, see references/...
        bug report) stops the loop and propagates the exception rather
        than silently skipping the frame, since a silently-incomplete file
        would be worse than a loud failure.

        A completed frame is copied into an immutable bytes record and put on
        a bounded queue. A separate writer consumes those records without a
        per-frame ``flush()``; normal file close flushes the complete stream.
        This lets the producer immediately issue the next blocking refill.
        The queue is bounded so a persistently slow or failed destination
        cannot consume unlimited memory, and writer failures cancel a blocked
        refill and propagate to the caller.

        *on_progress* is called by the writer thread with the total bytes
        accepted by Python's buffered file object so far, not
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
        self._backend.prepare_dma_capture()
        frame_samples = self._backend.get_frame_samples()
        expected_waveform_samples = frame_samples - SCOPE_TIMESTAMP_WORDS
        frame_count = 0
        write_queue: queue.Queue[bytes | object] = queue.Queue(
            maxsize=_IIO_SCOPE_WRITE_QUEUE_FRAMES,
        )
        writer_ready = threading.Event()
        writer_failed = threading.Event()
        queue_backpressure_reported = threading.Event()
        writer_errors: list[BaseException] = []
        written_frames: list[int] = [0]

        def writer() -> None:
            total_bytes = 0
            try:
                with open(filepath, "wb") as f:
                    log.info("IIO scope DMA: recording to %s", filepath)
                    # This deliberate, one-time flush makes the file/header
                    # visible before on_ready can start hardware. Frame writes
                    # below are not flushed individually.
                    _write_file_header(f, self._channel, frame_samples)
                    writer_ready.set()

                    while True:
                        record = write_queue.get()
                        if record is _IIO_SCOPE_WRITER_STOP:
                            break
                        if not isinstance(record, bytes):
                            raise TypeError(
                                "scope DMA writer received a non-bytes record"
                            )
                        bytes_written = f.write(record)
                        if bytes_written != len(record):
                            raise OSError(
                                "short scope DMA file write: "
                                f"wrote {bytes_written} of {len(record)} bytes"
                            )
                        written_frames[0] += 1
                        total_bytes += len(record)
                        if on_progress is not None:
                            on_progress(total_bytes)
            except BaseException as exc:
                writer_errors.append(exc)
                writer_failed.set()
                writer_ready.set()
                # A writer can fail while the producer is blocked in refill.
                # Cancel it so the worker can observe and report the real file
                # error instead of waiting forever for another trigger.
                try:
                    self._backend.request_dma_stop()
                except Exception:
                    log.exception(
                        "IIO scope DMA: failed to cancel capture after writer error"
                    )

        writer_thread = threading.Thread(
            target=writer,
            name=f"iio-scope-ch{self._channel}-file-writer",
            daemon=False,
        )
        writer_thread.start()

        def raise_writer_error() -> None:
            if writer_failed.is_set():
                raise writer_errors[0]

        def enqueue_record(record: bytes | object) -> bool:
            """Queue a record, noticing a dead writer while backpressured."""
            while writer_thread.is_alive():
                raise_writer_error()
                try:
                    write_queue.put(
                        record,
                        timeout=_IIO_SCOPE_WRITE_QUEUE_POLL_SECONDS,
                    )
                    return True
                except queue.Full:
                    if not queue_backpressure_reported.is_set():
                        queue_backpressure_reported.set()
                        log.warning(
                            "IIO scope DMA: the %d-frame file queue is full; "
                            "capture is now limited by storage throughput",
                            _IIO_SCOPE_WRITE_QUEUE_FRAMES,
                        )
                    continue
            raise_writer_error()
            return False

        def finish_writer() -> None:
            """Append the FIFO sentinel and always join the file owner."""
            while writer_thread.is_alive():
                try:
                    write_queue.put(
                        _IIO_SCOPE_WRITER_STOP,
                        timeout=_IIO_SCOPE_WRITE_QUEUE_POLL_SECONDS,
                    )
                    break
                except queue.Full:
                    continue
            writer_thread.join()

        try:
            writer_ready.wait()
            raise_writer_error()

            if on_ready is not None:
                on_ready()

            while not stop_event.is_set():
                if n_frames is not None and frame_count >= n_frames:
                    break

                raise_writer_error()
                try:
                    timestamp, samples = self._backend.read_dma_frame()
                except InterruptedError:
                    if writer_failed.is_set():
                        raise_writer_error()
                    if stop_event.is_set():
                        break
                    raise
                if samples.ndim != 1 or samples.size != expected_waveform_samples:
                    raise RuntimeError(
                        "scope DMA returned an incomplete waveform: "
                        f"{samples.size} samples, expected "
                        f"{expected_waveform_samples}"
                    )

                # struct.pack() and ndarray.tobytes() make this record wholly
                # independent of the IIO buffer before the next refill starts.
                record = struct.pack("<Q", timestamp) + samples.tobytes()
                if not enqueue_record(record):
                    raise RuntimeError("scope DMA writer stopped unexpectedly")
                frame_count += 1
        finally:
            try:
                self._backend.close_dma_capture()
            finally:
                finish_writer()

        raise_writer_error()
        if written_frames[0] != frame_count:
            raise RuntimeError(
                "scope DMA writer stopped before committing every queued frame: "
                f"wrote {written_frames[0]} of {frame_count}"
            )

        log.info("IIO scope DMA: finished -- %d frames to %s", frame_count, filepath)
        return frame_count


class IIOMcaDmaStreamer:
    """Pull fixed 16 KiB MCA list-mode blocks from the IIO backend.

    This is not interchangeable with McaDmaStreamer. The IIO driver has no
    ZMQ sentinels: the first backend read arms the single opaque u8[16] scan
    element, starts a blocking reader, and only then writes
    pulse_processor.enable=1.
    Shutdown is likewise backend-owned: enable=0, drain complete blocks for
    the documented one-second inactivity window, then destroy the buffer.
    """

    def __init__(self, backend: _IIOMcaBackend, channel: int) -> None:
        self._backend = backend
        self._channel = channel

    def request_stop(self) -> None:
        """Stop production while leaving the reader armed for tail drain."""
        self._backend.request_mca_dma_stop()

    def stream_events(
        self,
        stop_event: threading.Event,
        filepath: Path | None = None,
        event_buffer: McaEventBuffer | tuple[list[np.ndarray], threading.Lock] | None = None,
        on_ready: Callable[[], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> int:
        """Read complete 1024-record blocks until stopped.

        Raw IIO records are written unchanged. NDMA version 2 identifies
        their layout; version 1 remains the legacy gRPC/ZMQ layout. The
        count includes zero-padded slots in the final complete frame: v121
        publishes no valid-record count and mca-architecture.md forbids
        treating zero payloads as an end marker because a real event may
        also contain zero-valued fields.
        """
        total_records = 0
        frame_count = 0
        file_handle = None
        started_utc = datetime.now(UTC)

        if filepath is not None:
            file_handle = open(filepath, "wb", buffering=IIO_LM_FRAME_BYTES * 8)
            _write_file_header(
                file_handle,
                self._channel,
                version=IIO_LM_FILE_VERSION,
            )
            log.info("IIO MCA DMA: recording to %s", filepath)

        def consume(events: np.ndarray) -> None:
            nonlocal total_records, frame_count
            # Assert the backend and transport agree on the v121 ABI. This
            # catches an accidental legacy-dtype reuse before corrupting a
            # file or the GUI's shared event buffer.
            if events.dtype != _LM_EVENT_DTYPE:
                raise RuntimeError(
                    f"unexpected IIO MCA event dtype {events.dtype!r}; "
                    f"expected {_LM_EVENT_DTYPE!r}"
                )
            if file_handle is not None:
                file_handle.write(events.tobytes())
            if event_buffer is not None:
                _append_mca_event_batch(event_buffer, events)
            frame_count += 1
            total_records += len(events)
            if on_progress is not None:
                on_progress(total_records)

        capture_failed = False
        try:
            # Backend readiness is a hardware lifecycle boundary, not merely
            # worker-thread startup: arm all eight kernel blocks, enter the
            # first refill, enable the pulse processor, then notify the GUI.
            first = self._backend.start_mca_dma_capture(on_started=on_ready)
            if len(first):
                consume(first)
            while not stop_event.is_set():
                if not self._backend.mca_dma_measurement_in_progress():
                    break
                frame = self._backend.read_mca_dma_frame()
                if len(frame):
                    consume(frame)
                # Hardware time-limit completion can occur before the GUI
                # polling worker delivers its queued stop request. Once the
                # final complete frame has arrived, avoid entering another
                # refill that could block forever with the producer stopped.
                if not self._backend.mca_dma_measurement_in_progress():
                    break
        except BaseException:
            capture_failed = True
            raise
        finally:
            # Keep the file/event callback alive while close drains the
            # final complete blocks; destroying the buffer first would lose
            # that tail by construction.
            drained = 0
            try:
                drained = self._backend.close_mca_dma_capture(on_frame=consume)
            except BaseException:
                if capture_failed:
                    log.exception(
                        "IIO MCA DMA: buffer close also failed while handling "
                        "the capture error"
                    )
                else:
                    raise
            finally:
                if file_handle is not None:
                    file_handle.close()
            log.info(
                "IIO MCA DMA: finished -- %d records in %d frames "
                "(%d frame(s) received during close drain)",
                total_records, frame_count, drained,
            )

        dma_fault, dma_error_count, completed_frames, deadtime_records = (
            self._backend.get_mca_dma_capture_diagnostics()
        )
        if filepath is not None:
            metadata_path = _write_iio_mca_metadata(
                filepath,
                channel=self._channel,
                started_utc=started_utc,
                frame_count=frame_count,
                dma_fault=dma_fault,
                dma_error_count=dma_error_count,
                completed_frames=completed_frames,
                deadtime_records=deadtime_records,
            )
            log.info("IIO MCA DMA: capture metadata written to %s", metadata_path)
        if dma_fault:
            raise RuntimeError(
                f"MCA list-mode driver latched DMA fault reason {dma_fault}; "
                "the capture file was retained for diagnostics"
            )
        if completed_frames != frame_count:
            raise RuntimeError(
                f"MCA list-mode capture contains {frame_count} complete frames, "
                f"but the driver completed {completed_frames}; the retained "
                "capture is incomplete"
            )
        if deadtime_records:
            raise RuntimeError(
                f"MCA list-mode IP reports {deadtime_records} dropped records; "
                "the capture file was retained, but continuity is invalid"
            )
        log.info(
            "IIO MCA DMA: continuity verified -- %d driver frames, "
            "dma_fault=0, dma_error_count=%d, list_deadtime_raw=0",
            completed_frames, dma_error_count,
        )

        return total_records

    @staticmethod
    def parse_events(raw: bytes) -> np.ndarray:
        if len(raw) % _LM_EVENT_DTYPE.itemsize:
            raise ValueError(
                f"IIO list-mode payload is not aligned to "
                f"{_LM_EVENT_DTYPE.itemsize}-byte records"
            )
        return np.frombuffer(raw, dtype=_LM_EVENT_DTYPE)

    @staticmethod
    def compute_cfd_time(events: np.ndarray) -> np.ndarray:
        """Convert the Q2 CFD field to its physical legacy value."""
        return events["cfd_q2"].astype(np.float64) / 4.0


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
        event_buffer: McaEventBuffer | tuple[list[np.ndarray], threading.Lock] | None = None,
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
                    _append_mca_event_batch(event_buffer, parsed)

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
