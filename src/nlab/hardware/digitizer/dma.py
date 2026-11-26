"""IIO DMA streaming for scope waveforms and MCA list-mode events.

The streamers pull complete blocks from IIODigitizerBackend on dedicated
worker threads. Readers retain support for legacy NDMA v1 capture files.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import queue
import struct
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol

import numpy as np

from nlab.hardware.digitizer.iio_listmode import (
    IIO_LM_EVENT_DTYPE,
    MARKER_INPUT0,
    MARKER_INPUT1,
    VDPP_ZC_CALC_SCHEMA,
    cfd_interpolation_samples,
)
from nlab.hardware.digitizer.mca_capture import McaCaptureWriter, McaDmaOutputMode
from nlab.hardware.digitizer.scope_sums import (
    ScopeSumBackend,
    ScopeSumCapture,
    ScopeSumGeometry,
)

if TYPE_CHECKING:
    from nlab.hardware.digitizer.current_monitor import ScopeCurrentAccumulator
    from nlab.hardware.digitizer.listmode_process import ListModeProcessSession
    from nlab.hardware.digitizer.reader_process import ReaderSession


class _IIOScopeBackend(ScopeSumBackend, Protocol):
    def prepare_dma_capture(self) -> None: ...

    def get_frame_samples(self) -> int: ...

    # Periodic-sum mode 5 advertises ScopeSumGeometry; capture_geometry()
    # rejects it so structured sums never reach waveform/NDMA v1 consumers.
    def get_scope_dma_geometry(self) -> ScopeDmaGeometry | ScopeSumGeometry: ...

    def read_dma_raw_frame(self) -> bytes: ...

    def request_dma_stop(self) -> None: ...

    def close_dma_capture(self, on_frame: Callable[[bytes], None] | None = None) -> None: ...


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

FILE_HEADER_STRUCT = struct.Struct("<4sHBBdI4x")  # 24 bytes
FILE_MAGIC = b"NDMA"
FILE_VERSION = 1
IIO_LM_FILE_VERSION = 2
IIO_LM_FRAME_RECORDS = 1024
IIO_LM_RECORD_BYTES = 16
IIO_LM_FRAME_BYTES = IIO_LM_FRAME_RECORDS * IIO_LM_RECORD_BYTES
# Recorded in capture metadata; must equal iio_backend._LM_KERNEL_BUFFER_COUNT.
IIO_LM_KERNEL_BUFFER_COUNT = 64
IIO_LM_CLIENT_SCHEMA = VDPP_ZC_CALC_SCHEMA
IIO_LM_UNQUALIFIED_SCHEMA = "unqualified-opaque[16]"

# Each raw scope DMA frame is prefixed with a per-frame timestamp: the first
# 4 int16 slots (8 bytes) are a little-endian uint64, the remaining
# (frame_samples - SCOPE_TIMESTAMP_WORDS) slots are the int16 waveform.
SCOPE_TIMESTAMP_WORDS = 4
SCOPE_ADC_SAMPLE_PERIOD_NS = 2

# Keep file-system stalls out of the time-critical IIO refill loop while still
# bounding memory use. At the largest legal scope frame this queue occupies
# roughly 4 MiB. If storage remains slower than acquisition after the queue is
# full, producer backpressure is intentional: silently dropping raw frames
# would make the capture file look valid while losing data.
_IIO_SCOPE_WRITE_QUEUE_FRAMES = 256
_IIO_SCOPE_WRITE_QUEUE_POLL_SECONDS = 0.050
_IIO_SCOPE_WRITER_STOP = object()
# Producer-status poll period during MCA list mode. A per-frame status read
# cost 0.56 ms of a 1.86 ms frame live (v121, ~530 kcps), i.e. ~30% of the
# achievable record rate; 20 reads/s keep time-limit detection prompt.
_MCA_PRODUCER_POLL_SECONDS = 0.050
_MCA_PRODUCER_WATCH_JOIN_SECONDS = 2.0

_EVENT_DTYPE = np.dtype(
    [
        ("marker", np.uint8),
        ("zc_offset", np.uint8),
        ("zc_estimation", np.uint16),
        ("short_energy", np.uint16),
        ("energy", np.uint16),
        ("timestamp", np.uint64),
    ]
)

# Preserve the historical private name for callers of the NDMA v2 reader.
_LM_EVENT_DTYPE = IIO_LM_EVENT_DTYPE

# The first record of a clean list-mode measurement is the synthetic START
# record: marker/payload zero, timestamp 1. Records that precede it are left
# over from an earlier session whose tail was not drained (PetaLinux cde5fe1
# user-api.md "Stream carry-over between sessions", rule 2; confirmed live on
# .134 and .128). The bytes are the producer's, independent of client schema.
# The marker's logic-input bits are metadata, not record identity (PetaLinux
# coincidence-frontend-contract.md marker table: 0x40 input1, 0x80 input0);
# the producer stamps them on START too. Live on .128 (HW101/SW103025, both
# channels, 2026-11-26) START arrived as record 0 reading 80 00 .. 00, then
# timestamp 1, so the match ignores those two bits and nothing else.
_LM_START_TIMESTAMP = 1
_LM_START_IGNORED_MARKER_BITS = MARKER_INPUT0 | MARKER_INPUT1
# Carry-over is bounded by the external 2048-record FIFO plus in-flight
# blocks (.134 delivered two stale frames). If no START arrives within this
# many frames the held frames are kept, not discarded, and flagged.
_LM_START_SEARCH_FRAMES = 4


class ListModeStartGate:
    """Drop whole frames that precede the session's START record.

    Frames stay unchanged 16 KiB blocks. A START record later than record 0
    of its frame means that frame is partly stale (the producer's frame
    position carried over); it is delivered unchanged but reported, because
    trimming would break the complete-frame transport.
    """

    def __init__(self, deliver: Callable[[np.ndarray], None]) -> None:
        self._deliver = deliver
        self._held: list[np.ndarray] = []
        self.stale_frames = 0
        # None while searching; record index of START in its frame once found.
        self.start_index: int | None = None
        self.missing = False
        # Evidence for a missing START: the session's first record, and the
        # first record stamped 1 whatever its payload. A producer whose START
        # carries a nonzero payload is told apart from real carry-over by it.
        self.first_record: bytes | None = None
        self.timestamp_one: tuple[int, int, bytes] | None = None

    @property
    def searching(self) -> bool:
        return self.start_index is None and not self.missing

    def offer(self, frame: np.ndarray) -> None:
        if not self.searching:
            self._deliver(frame)
            return
        raw = frame.tobytes()
        if self.first_record is None:
            self.first_record = raw[:IIO_LM_RECORD_BYTES]
        index = _start_record_index(frame)
        if index is None:
            if self.timestamp_one is None:
                stamped = _timestamp_one_index(raw)
                if stamped is not None:
                    offset = stamped * IIO_LM_RECORD_BYTES
                    self.timestamp_one = (
                        len(self._held), stamped, raw[offset:offset + IIO_LM_RECORD_BYTES]
                    )
            self._held.append(frame)
            if len(self._held) >= _LM_START_SEARCH_FRAMES:
                self.finish()
            return
        self.stale_frames = len(self._held)
        self._held.clear()
        self.start_index = index
        self._deliver(frame)

    def finish(self) -> None:
        """Release held frames when the session ended or START never came."""
        if not self.searching:
            return
        if not self._held:
            return  # no frame at all; nothing to judge
        self.missing = True
        held, self._held = self._held, []
        for frame in held:
            self._deliver(frame)

    def diagnostics(self) -> dict[str, object]:
        if self.start_index == 0:
            state = "first"
        elif self.start_index is not None:
            state = "misaligned"
        else:
            state = "missing" if self.missing else "none"
        diagnostics: dict[str, object] = {
            "start_record": state,
            "start_record_index": self.start_index,
            "stale_frames_discarded": self.stale_frames,
        }
        if self.missing:
            diagnostics["first_record_hex"] = (
                self.first_record.hex(" ") if self.first_record is not None else None
            )
            diagnostics["timestamp_one_record"] = self._timestamp_one_text()
        return diagnostics

    def _timestamp_one_text(self) -> str | None:
        if self.timestamp_one is None:
            return None
        frame, record, payload = self.timestamp_one
        return f"frame {frame} record {record}: {payload.hex(' ')}"

    def problem(self) -> str | None:
        """Why this session's first records cannot be trusted, if they can't."""
        if self.start_index is not None and self.start_index > 0:
            return (
                f"the session's START record is record {self.start_index} of its "
                "first frame; the records before it are left over from an earlier "
                "undrained session"
            )
        if self.missing:
            first = self.first_record.hex(" ") if self.first_record is not None else "none"
            stamped = self._timestamp_one_text() or "none"
            return (
                "no START record arrived in the session's first frames; they may "
                "be left over from an earlier undrained session (first record: "
                f"{first}; first record with timestamp 1: {stamped})"
            )
        return None


def _start_record_index(frame: np.ndarray) -> int | None:
    # Each little-endian record is two words: bytes 0-7 (marker in the low
    # byte, then the payload) and the timestamp.
    words = np.frombuffer(frame.tobytes(), dtype="<u8").reshape(-1, 2)
    payload = words[:, 0] & ~np.uint64(_LM_START_IGNORED_MARKER_BITS)
    hits = np.flatnonzero((payload == 0) & (words[:, 1] == _LM_START_TIMESTAMP))
    return int(hits[0]) if len(hits) else None


def _timestamp_one_index(raw: bytes) -> int | None:
    """Index of the first record whose timestamp (bytes 8-15) is 1."""
    timestamps = np.frombuffer(raw, dtype="<u8")[1::2]
    hits = np.flatnonzero(timestamps == 1)
    return int(hits[0]) if len(hits) else None


@dataclass(frozen=True)
class ScopeDmaGeometry:
    """One advertised Scope DMA record layout.

    ``frame_samples`` remains the full-rate hardware capture length.  Newer
    periodic firmware can decimate the transported waveform and pad each DMA
    record to its required alignment, so neither the IIO buffer length nor the
    payload byte count may be inferred from that register.
    """

    frame_samples: int
    buffer_samples: int
    frame_bytes: int
    waveform_samples: int
    sample_decimation: int
    padding_bytes: int

    def __post_init__(self) -> None:
        payload_bytes = SCOPE_TIMESTAMP_WORDS * 2 + self.waveform_samples * 2
        if (
            self.frame_samples < SCOPE_TIMESTAMP_WORDS
            or self.buffer_samples <= 0
            or self.frame_bytes != self.buffer_samples * 2
            or self.waveform_samples < 0
            or self.sample_decimation <= 0
            or self.padding_bytes < 0
            or payload_bytes + self.padding_bytes != self.frame_bytes
        ):
            raise ValueError("invalid Scope DMA geometry")

    @classmethod
    def legacy(cls, frame_samples: int) -> ScopeDmaGeometry:
        return cls(
            frame_samples=frame_samples,
            buffer_samples=frame_samples,
            frame_bytes=frame_samples * 2,
            waveform_samples=frame_samples - SCOPE_TIMESTAMP_WORDS,
            sample_decimation=1,
            padding_bytes=0,
        )

    @property
    def sample_period_ns(self) -> int:
        return SCOPE_ADC_SAMPLE_PERIOD_NS * self.sample_decimation

    @property
    def capture_duration_ns(self) -> int:
        return self.waveform_samples * self.sample_period_ns

    @property
    def ndma_v1_compatible(self) -> bool:
        return (
            self.buffer_samples == self.frame_samples
            and self.waveform_samples == self.frame_samples - SCOPE_TIMESTAMP_WORDS
            and self.sample_decimation == 1
            and self.padding_bytes == 0
        )


def _is_immutable_bytes_view(array: np.ndarray) -> bool:
    """True when *array* is a read-only view whose memory is a ``bytes`` object."""
    if array.flags.writeable:
        return False
    base: object = array
    while isinstance(base, np.ndarray):
        base = base.base
    return isinstance(base, bytes)


@dataclass(frozen=True)
class ScopeDmaFrame:
    """A complete decoded Scope frame handed to live display consumers."""

    timestamp: int
    received_ns: int
    samples: np.ndarray
    geometry: ScopeDmaGeometry

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype="<i2").reshape(-1)
        if samples.size != self.geometry.waveform_samples:
            raise ValueError("Scope DMA frame waveform does not match its geometry")
        if _is_immutable_bytes_view(samples):
            # The streamer passes a read-only view into the backend's
            # immutable DMA record; sharing it is already safe.
            object.__setattr__(self, "samples", samples)
            return
        # Own one immutable byte string. Every bounded display consumer can
        # safely share the resulting read-only view after the next DMA refill.
        owned = bytes(samples.tobytes())
        object.__setattr__(self, "samples", np.frombuffer(owned, dtype="<i2"))


class ScopeFrameBuffer:
    """Bounded non-blocking hand-off queue for live Scope DMA consumers."""

    def __init__(self, max_frames: int = 256) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self._max_frames = max_frames
        self._frames: deque[ScopeDmaFrame] = deque()
        self._lock = threading.Lock()
        self._dropped_frames = 0
        self._subscribers: list[ScopeFrameBuffer] = []

    @staticmethod
    def _copy_frame(frame: ScopeDmaFrame) -> ScopeDmaFrame:
        # ScopeDmaFrame owns immutable backing bytes, so publication shares
        # one safe frame instead of copying the waveform for every subscriber.
        return frame

    def append(self, frame: ScopeDmaFrame) -> None:
        copied = self._copy_frame(frame)
        with self._lock:
            if len(self._frames) >= self._max_frames:
                self._frames.popleft()
                self._dropped_frames += 1
            self._frames.append(copied)
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber.append(frame)

    def append_records(
        self,
        records: Sequence[bytes],
        geometry: ScopeDmaGeometry,
        received_ns: Sequence[int],
    ) -> None:
        """Publish complete raw records; same result as append() for each in order.

        Only the newest ``max_frames`` records can survive one call, so older
        ones are counted as dropped without being decoded into frames.
        """
        count = len(records)
        keep = min(count, self._max_frames)
        skipped = count - keep
        frames = [
            ScopeDmaFrame(
                timestamp=int.from_bytes(record[:8], "little"),
                received_ns=int(stamp),
                samples=np.frombuffer(
                    record,
                    dtype="<i2",
                    count=geometry.waveform_samples,
                    offset=SCOPE_TIMESTAMP_WORDS * 2,
                ),
                geometry=geometry,
            )
            for record, stamp in zip(records[skipped:], received_ns[skipped:], strict=True)
        ]
        with self._lock:
            self._dropped_frames += skipped
            for _ in range(max(0, len(self._frames) + keep - self._max_frames)):
                self._frames.popleft()
                self._dropped_frames += 1
            self._frames.extend(frames)
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber.append_records(records, geometry, received_ns)

    def add_dropped(self, frames: int) -> None:
        """Account frames an upstream display queue (the reader process) dropped."""
        if frames <= 0:
            return
        with self._lock:
            self._dropped_frames += frames
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber.add_dropped(frames)

    def subscribe(self, subscriber: ScopeFrameBuffer) -> None:
        if subscriber is self:
            raise ValueError("a frame buffer cannot subscribe to itself")
        with self._lock:
            if subscriber not in self._subscribers:
                self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: ScopeFrameBuffer) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._dropped_frames = 0

    def drain(self) -> tuple[list[ScopeDmaFrame], int]:
        with self._lock:
            frames = list(self._frames)
            self._frames.clear()
            dropped = self._dropped_frames
            self._dropped_frames = 0
            return frames, dropped


class McaEventBuffer:
    """Bounded hand-off queue for live consumers of MCA DMA batches.

    File recording has its own bounded writer queue (McaCaptureWriter) and
    never depends on this display path. If the GUI cannot
    drain this queue quickly enough, only the oldest display batch is
    discarded; raw capture continuity is unaffected.
    """

    def __init__(self, max_batches: int = 128) -> None:
        if max_batches <= 0:
            raise ValueError("max_batches must be positive")
        self._max_batches = max_batches
        self._batches: deque[tuple[int, np.ndarray]] = deque()
        self._lock = threading.Lock()
        self._dropped_records = 0
        self._subscribers: list[McaEventBuffer] = []

    def append(self, events: np.ndarray) -> None:
        copied = events.copy()
        received_ns = time.monotonic_ns()
        with self._lock:
            if len(self._batches) >= self._max_batches:
                self._dropped_records += len(self._batches.popleft()[1])
            self._batches.append((received_ns, copied))
            subscribers = tuple(self._subscribers)
        # The producer never waits for a display consumer. Each subscriber
        # owns its own bounded queue; PSD and coincidence cannot steal batches
        # from one another by draining the same queue.
        for subscriber in subscribers:
            subscriber.append(events)

    def add_dropped(self, records: int) -> None:
        """Account records an upstream display queue (the reader process) dropped."""
        if records <= 0:
            return
        with self._lock:
            self._dropped_records += records
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber.add_dropped(records)

    def subscribe(self, subscriber: McaEventBuffer) -> None:
        if subscriber is self:
            raise ValueError("an event buffer cannot subscribe to itself")
        with self._lock:
            if subscriber not in self._subscribers:
                self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: McaEventBuffer) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def clear(self) -> None:
        with self._lock:
            self._batches.clear()
            self._dropped_records = 0

    def drain(self) -> tuple[list[np.ndarray], int]:
        """Atomically remove pending batches and report cumulative drops."""
        with self._lock:
            batches = [events for _received_ns, events in self._batches]
            self._batches.clear()
            return batches, self._dropped_records

    def drain_timed(self) -> tuple[list[tuple[int, np.ndarray]], int]:
        """Drain batches with their local monotonic publisher timestamps."""
        with self._lock:
            batches = list(self._batches)
            self._batches.clear()
            return batches, self._dropped_records


def _list_mode_loss(deadtime_after: int, deadtime_before: int | None) -> bool:
    """Whether list_deadtime_raw shows producer loss during this session.

    The counter survives arm and START and clears only at the first record
    after a completed end frame (L2), so an unchanged nonzero value carried in
    from an earlier session is not this session's loss, and a value cleared
    to zero is no loss. Without a baseline, any nonzero value counts.
    """
    if deadtime_after == 0:
        return False
    return deadtime_before is None or deadtime_after != deadtime_before


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
        FILE_MAGIC,
        version,
        channel,
        0,
        time.time(),
        frame_samples,
    )
    f.write(header)
    f.flush()


def _file_header_bytes(
    channel: int,
    frame_samples: int = 0,
    version: int = FILE_VERSION,
) -> bytes:
    return FILE_HEADER_STRUCT.pack(
        FILE_MAGIC,
        version,
        channel,
        0,
        time.time(),
        frame_samples,
    )


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
    client_record_schema: str,
    deadtime_before: int | None = None,
    stale_frames: int = 0,
    final_status: dict[str, object] | None = None,
    identity: dict[str, int | str] | None = None,
    capture_sha256: str | None = None,
) -> Path:
    """Write the auditable sidecar used by the reference capture workflow.

    *capture_sha256* is the digest the writer computed over every byte it
    wrote; without it (e.g. after a writer failure) the file is re-read.
    """
    metadata_path = path.with_suffix(".json")
    continuity_valid = (
        dma_fault == 0
        and completed_frames == frame_count + stale_frames
        and not _list_mode_loss(deadtime_records, deadtime_before)
    )
    metadata = {
        "format": "nlab-iio-mca-ndma-v2",
        "capture": str(path.resolve()),
        "capture_sha256": capture_sha256 or _sha256_file(path),
        "channel_index": channel,
        "started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "ndma_header_bytes": FILE_HEADER_STRUCT.size,
        "record_layout": "opaque[16]",
        # The kernel intentionally promises only opaque[16]. Decoded PSD
        # fields belong to this explicitly named, replaceable client schema.
        "client_record_schema": client_record_schema,
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
        "list_deadtime_before_raw": deadtime_before,
        "stale_frames_discarded": stale_frames,
        "continuity_valid": continuity_valid,
    }
    if final_status is not None:
        metadata["final_status"] = final_status
        metadata["continuity_valid"] = bool(final_status.get("continuity_valid", False))
    if identity is not None:
        metadata["producer_identity"] = identity
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata_path


class IIOScopeDmaStreamer:
    """Complete Scope DMA records for files and bounded live consumers.

    The IIO scope core exposes a pull-based IIO buffer. This class arms one
    buffer for the measurement, repeatedly refills it and publishes decoded
    complete records to live subscribers.
    Legacy-compatible records can additionally be handed to a dedicated file
    writer. New periodic v122 layouts can decimate and align the transport;
    their exact geometry is driver-advertised and is currently online-only
    because NDMA v1 cannot describe the decimation or padding.
    """

    def __init__(
        self,
        backend: _IIOScopeBackend,
        channel: int,
        *,
        process_uri: str | None = None,
    ) -> None:
        """*process_uri* runs each waveform/sum session in a reader process.

        See scope_process.py. Without it (tests, scripts, NLAB_DMA_IN_PROCESS=1)
        sessions run on the caller's thread against *backend* as before.
        """
        self._backend = backend
        self._channel = channel
        self._process_uri = process_uri
        self._session_lock = threading.Lock()
        self._session: ReaderSession | None = None

    def request_stop(self) -> None:
        """Stop the acquisition gate and cancel a blocked backend refill."""
        if self._process_uri is not None:
            # The session also forwards the caller's stop_event, so a stop
            # that precedes the spawn is delivered once the child exists.
            with self._session_lock:
                session = self._session
            if session is not None:
                session.request_stop()
            return
        self._backend.request_dma_stop()

    def create_sum_capture(self, *, configuration_yaml: str = "") -> ScopeSumCapture:
        """Create a distinct sum session with an optional full configuration snapshot."""
        return ScopeSumCapture(
            self._backend, self._channel, configuration_yaml=configuration_yaml,
            process_uri=self._process_uri,
        )

    def capture_geometry(self) -> ScopeDmaGeometry:
        """Return the configured record layout, with a v121 fallback."""
        reader = getattr(self._backend, "get_scope_dma_geometry", None)
        if callable(reader):
            geometry = reader()
            if not isinstance(geometry, ScopeDmaGeometry):
                raise TypeError("backend returned an invalid Scope DMA geometry")
            return geometry
        return ScopeDmaGeometry.legacy(self._backend.get_frame_samples())

    def stream_to_file(
        self,
        filepath: Path | None,
        stop_event: threading.Event,
        n_frames: int | None = None,
        on_ready: Callable[[], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
        frame_buffer: ScopeFrameBuffer | None = None,
        current_accumulator: ScopeCurrentAccumulator | None = None,
    ) -> int:
        """Capture complete frames for an optional file and live publisher.

        Runs until *stop_event* is set or *n_frames* complete frames have been
        accepted (None = unbounded). With file output, *on_ready* follows the
        flushed one-time NDMA header; online-only mode calls it immediately
        before the first capture attempt. At that point no DMA buffer exists yet
        (it's created lazily by the first read_dma_raw_frame() call), and that
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
        accepted by Python's buffered file object so far, not a frame count,
        because ScopeController._on_dma_progress() treats the value as a byte
        count for its KB/MB display. Passing frame_count here instead
        used to make the GUI display a wildly wrong size (e.g. "20 KB"
        for what was actually a 56 MB file, since 20,000-ish frames were
        being read as 20,000-ish bytes).

        The one-time file header written by _write_file_header() is not
        included in the count, so the displayed byte count excludes that
        fixed 24-byte header.

        Always closes the DMA capture buffer before returning, success,
        stop_event, or exception alike -- per vdpp-scope.c's predisable(),
        that's what clears ENABLE/DMA_ENABLE back down for the DMA case;
        leaving the buffer open here would leave the hardware armed after
        the caller thinks the measurement has stopped.
        """
        if self._process_uri is not None:
            return self._stream_in_reader_process(
                filepath, stop_event, n_frames, on_ready, on_progress,
                frame_buffer, current_accumulator,
            )
        self._backend.prepare_dma_capture()
        geometry = self.capture_geometry()
        frame_samples = geometry.frame_samples
        expected_frame_bytes = geometry.frame_bytes
        if filepath is not None and not geometry.ndma_v1_compatible:
            self._backend.close_dma_capture()
            raise RuntimeError(
                "this Scope DMA layout uses decimation/alignment padding and "
                "cannot be stored in the legacy NDMA v1 format; use online "
                "streaming or add a versioned file format"
            )
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
            assert filepath is not None
            total_bytes = 0
            try:
                # 1 MiB buffering batches small frames into large writes.
                with open(filepath, "wb", buffering=1024 * 1024) as f:
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
                            raise TypeError("scope DMA writer received a non-bytes record")
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
                    log.exception("IIO scope DMA: failed to cancel capture after writer error")

        writer_thread: threading.Thread | None = None
        if filepath is not None:
            writer_thread = threading.Thread(
                target=writer,
                name=f"iio-scope-ch{self._channel}-file-writer",
                daemon=False,
            )
            writer_thread.start()
        else:
            writer_ready.set()

        def raise_writer_error() -> None:
            if writer_failed.is_set():
                raise writer_errors[0]

        def enqueue_record(record: bytes | object) -> bool:
            """Queue a record, noticing a dead writer while backpressured."""
            if writer_thread is None:
                return True
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
            if writer_thread is None:
                return
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

        accumulator_error = False

        def deliver(record: bytes) -> None:
            """Publish one complete frame to every consumer, in order."""
            nonlocal frame_count, accumulator_error
            raise_writer_error()
            if len(record) != expected_frame_bytes:
                if current_accumulator is not None:
                    current_accumulator.note_rejected_frame()
                    accumulator_error = True
                raise RuntimeError(
                    "scope DMA returned an incomplete raw frame: "
                    f"{len(record)} bytes, expected {expected_frame_bytes}"
                )

            received_ns = time.perf_counter_ns()
            if current_accumulator is not None:
                try:
                    current_accumulator.append_frame(record, geometry, received_ns)
                except Exception:
                    accumulator_error = True
                    raise
                if frame_count == 0:
                    metadata_reader = getattr(
                        self._backend,
                        "get_scope_dma_runtime_metadata",
                        None,
                    )
                    if callable(metadata_reader):
                        metadata = metadata_reader()
                        current_accumulator.update_runtime_transport(
                            transport=str(metadata.get("transport", "unknown")),
                            kernel_buffers=int(metadata.get("kernel_buffers", 0)),
                            readbuf_batch_frames=int(metadata.get("readbuf_batch_frames", 0)),
                            queued_blocks=int(metadata.get("queued_blocks", 0)),
                            queue_high_watermark=int(metadata.get("queue_high_watermark", 0)),
                        )

            if frame_buffer is not None:
                timestamp = int.from_bytes(record[:8], "little")
                samples = np.frombuffer(
                    record,
                    dtype="<i2",
                    count=geometry.waveform_samples,
                    offset=SCOPE_TIMESTAMP_WORDS * 2,
                )
                frame_buffer.append(
                    ScopeDmaFrame(
                        timestamp=timestamp,
                        received_ns=received_ns,
                        samples=samples,
                        geometry=geometry,
                    )
                )

            # The backend has already copied this exact immutable frame
            # out of the transport buffer, so no decode/repack is needed
            # before the next refill starts.
            if not enqueue_record(record):
                raise RuntimeError("scope DMA writer stopped unexpectedly")
            frame_count += 1
            if filepath is None and on_progress is not None:
                on_progress(frame_count * expected_frame_bytes)

        healthy_stop = False
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
                    record = self._backend.read_dma_raw_frame()
                except InterruptedError:
                    if writer_failed.is_set():
                        raise_writer_error()
                    if stop_event.is_set():
                        break
                    raise
                deliver(record)
            # A frame-count limit ends the session itself; only a STOP has an
            # accepted tail that belongs to this capture.
            healthy_stop = n_frames is None or frame_count < n_frames
        except Exception:
            if (
                current_accumulator is not None
                and not accumulator_error
                and not stop_event.is_set()
            ):
                current_accumulator.note_protocol_error()
            raise
        finally:
            try:
                # STOP closes admission only: frames already accepted keep
                # arriving and belong to this capture (PetaLinux cde5fe1
                # user-api.md "STOP semantics"). A failed session's tail is
                # drained but not published.
                self._backend.close_dma_capture(on_frame=deliver if healthy_stop else None)
            finally:
                try:
                    finish_writer()
                finally:
                    if current_accumulator is not None:
                        current_accumulator.finish_session()

        raise_writer_error()
        if filepath is not None and written_frames[0] != frame_count:
            raise RuntimeError(
                "scope DMA writer stopped before committing every queued frame: "
                f"wrote {written_frames[0]} of {frame_count}"
            )

        destination = str(filepath) if filepath is not None else "live consumers"
        log.info("IIO scope DMA: finished -- %d frames to %s", frame_count, destination)
        return frame_count

    def _stream_in_reader_process(
        self,
        filepath: Path | None,
        stop_event: threading.Event,
        n_frames: int | None,
        on_ready: Callable[[], None] | None,
        on_progress: Callable[[int], None] | None,
        frame_buffer: ScopeFrameBuffer | None,
        current_accumulator: ScopeCurrentAccumulator | None,
    ) -> int:
        """Run stream_to_file() in a reader process; feed the GUI consumers here."""
        from nlab.hardware.digitizer.reader_process import (
            ReaderProcessCrashedError,
            ReaderSession,
            raise_child_error,
        )
        from nlab.hardware.digitizer.scope_process import (
            ScopeWaveformRequest,
            waveform_label,
            waveform_main,
        )

        assert self._process_uri is not None
        latched = getattr(self._backend, "dma_fault_is_latched", None)
        if callable(latched) and latched():
            raise RuntimeError(
                "DMA fault is latched from a previous session -- call "
                "acknowledge_dma_recovery() before capturing again"
            )
        request = ScopeWaveformRequest(
            uri=self._process_uri,
            channel=self._channel,
            filepath=str(filepath) if filepath is not None else None,
            n_frames=n_frames,
            want_frames=frame_buffer is not None or current_accumulator is not None,
            lossless=current_accumulator is not None,
            log_level=logging.getLogger().getEffectiveLevel(),
        )
        geometry: list[ScopeDmaGeometry] = []
        dropped_seen = 0

        def note(key: str, value: object = None) -> None:
            if key == "geometry":
                assert isinstance(value, ScopeDmaGeometry)
                geometry[:] = [value]
            elif current_accumulator is None:
                return
            elif key == "runtime":
                assert isinstance(value, dict)
                current_accumulator.update_runtime_transport(**value)
            elif key == "rejected":
                current_accumulator.note_rejected_frame()
            elif key == "protocol_error":
                current_accumulator.note_protocol_error()

        def data(items: list[tuple[bytes, int]], dropped: int) -> None:
            nonlocal dropped_seen
            if items:
                # One batch call per message: per-frame Python work here was
                # the Current DMA limit with a busy GUI thread.
                layout = geometry[0]
                records = [record for record, _ in items]
                stamps = [stamp for _, stamp in items]
                if current_accumulator is not None:
                    current_accumulator.append_frames(records, layout, stamps)
                if frame_buffer is not None:
                    frame_buffer.append_records(records, layout, stamps)
            if dropped > dropped_seen:
                if frame_buffer is not None:
                    frame_buffer.add_dropped(dropped - dropped_seen)
                dropped_seen = dropped

        handlers: dict[str, Callable[..., None]] = {"note": note, "data": data}
        if on_ready is not None:
            handlers["ready"] = on_ready
        if on_progress is not None:
            handlers["progress"] = on_progress
        session = ReaderSession(waveform_main, request, label=waveform_label(self._channel))
        with self._session_lock:
            self._session = session
        try:
            finished = session.run(stop_event, handlers)
        except ReaderProcessCrashedError:
            if current_accumulator is not None and not stop_event.is_set():
                current_accumulator.note_protocol_error()
            raise
        finally:
            with self._session_lock:
                self._session = None
            if current_accumulator is not None:
                current_accumulator.finish_session()
            adopt = getattr(self._backend, "adopt_scope_dma_session_state", None)
            if callable(adopt):
                report = session.finished
                adopt(None if report is None else bool(report.get("fault_latched")))
        raise_child_error(finished)
        frames = finished.get("frames")
        return frames if isinstance(frames, int) else 0


class IIOMcaDmaStreamer:
    """Pull fixed 16 KiB MCA list-mode blocks from the IIO backend.

    The first backend read arms the single opaque u8[16] scan element,
    starts a blocking reader, and only then writes
    pulse_processor.enable=1.
    Shutdown is likewise backend-owned: enable=0, drain complete blocks for
    the documented one-second inactivity window, then destroy the buffer.
    """

    def __init__(
        self,
        backend: _IIOMcaBackend,
        channel: int,
        *,
        process_uri: str | None = None,
    ) -> None:
        """*process_uri* runs each session in a dedicated reader process.

        See reader_process.py: a busy GUI thread starved the in-process
        reader of the interpreter lock and the FPGA dropped records. Without
        it (tests, scripts, NLAB_DMA_IN_PROCESS=1) the session runs on
        the caller's thread against *backend* exactly as before.
        """
        self._backend = backend
        self._channel = channel
        self._process_uri = process_uri
        self._session_lock = threading.Lock()
        self._session: ListModeProcessSession | None = None
        self.last_capture_diagnostics: dict[str, object] | None = None

    def transport_identity(self) -> dict[str, int | str]:
        """Return read-only list-mode identity available from the bound backend."""
        identity: dict[str, int | str] = {
            "channel_index": self._channel,
            "record_layout": "opaque[16]",
        }
        for key, getter_name in (
            ("ip_version", "get_lm_ip_version"),
            ("frame_records", "get_lm_frame_records"),
            ("frame_bytes", "get_lm_frame_bytes"),
        ):
            getter = getattr(self._backend, getter_name, None)
            if not callable(getter):
                continue
            try:
                value = getter()
            except Exception:
                log.debug("Could not read MCA DMA %s", getter_name, exc_info=True)
                continue
            if isinstance(value, (int, np.integer)):
                identity[key] = int(value)
        producer = getattr(self._backend, "mca_producer_identity", None)
        if callable(producer):
            identity.update(producer())
        return identity

    def qualified_record_schema(self) -> str | None:
        identity = self.transport_identity()
        value = identity.get("qualified_record_schema")
        return str(value) if value else None

    def request_stop(self) -> None:
        """Stop production while leaving the reader armed for tail drain."""
        if self._process_uri is not None:
            # The process session also forwards the caller's stop_event, so a
            # stop that precedes the spawn is delivered once it exists.
            with self._session_lock:
                session = self._session
            if session is not None:
                session.request_stop()
            return
        self._backend.request_mca_dma_stop()

    def stream_events(
        self,
        stop_event: threading.Event,
        filepath: Path | None = None,
        event_buffer: McaEventBuffer | tuple[list[np.ndarray], threading.Lock] | None = None,
        on_ready: Callable[[], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
        output_mode: McaDmaOutputMode = McaDmaOutputMode.BINARY,
        configuration_yaml: str = "",
        client_record_schema: str | None = None,
    ) -> int:
        """Read complete 1024-record blocks until stopped.

        Runs in the reader process when one is configured, otherwise on the
        calling thread (see ``_stream_events_local``).
        """
        if self._process_uri is None:
            return self._stream_events_local(
                stop_event, filepath, event_buffer, on_ready, on_progress,
                output_mode, configuration_yaml, client_record_schema,
            )
        return self._stream_events_in_reader_process(
            stop_event, filepath, event_buffer, on_ready, on_progress,
            output_mode, configuration_yaml, client_record_schema,
        )

    def _stream_events_in_reader_process(
        self,
        stop_event: threading.Event,
        filepath: Path | None,
        event_buffer: McaEventBuffer | tuple[list[np.ndarray], threading.Lock] | None,
        on_ready: Callable[[], None] | None,
        on_progress: Callable[[int], None] | None,
        output_mode: McaDmaOutputMode,
        configuration_yaml: str,
        client_record_schema: str | None,
    ) -> int:
        from nlab.hardware.digitizer.listmode_process import (
            ListModeProcessSession,
            ListModeRequest,
        )

        assert self._process_uri is not None
        recovery_required = getattr(self._backend, "mca_recovery_required", None)
        if callable(recovery_required) and recovery_required():
            raise RuntimeError(
                "MCA DMA fault is latched; acknowledge_mca_dma_recovery() required"
            )
        profile_reader = getattr(self._backend, "acquisition_profile", None)
        profile = profile_reader() if callable(profile_reader) else None
        request = ListModeRequest(
            uri=self._process_uri,
            channel=self._channel,
            profile=profile or None,
            filepath=str(filepath) if filepath is not None else None,
            output_mode=output_mode.value,
            configuration_yaml=configuration_yaml,
            client_record_schema=client_record_schema,
            want_events=event_buffer is not None,
            log_level=logging.getLogger().getEffectiveLevel(),
        )

        def on_events(events: np.ndarray) -> None:
            if event_buffer is not None:
                _append_mca_event_batch(event_buffer, events)

        def on_dropped(records: int) -> None:
            if isinstance(event_buffer, McaEventBuffer):
                event_buffer.add_dropped(records)

        session = ListModeProcessSession(request)
        self.last_capture_diagnostics = None
        with self._session_lock:
            self._session = session
        try:
            return session.run(
                stop_event,
                on_ready=on_ready,
                on_progress=on_progress,
                on_events=on_events,
                on_dropped=on_dropped,
            )
        finally:
            with self._session_lock:
                self._session = None
            self.last_capture_diagnostics = session.outcome.diagnostics
            adopt = getattr(self._backend, "adopt_mca_dma_session_state", None)
            if callable(adopt):
                adopt(session.outcome.recovery_required)

    def _stream_events_local(
        self,
        stop_event: threading.Event,
        filepath: Path | None = None,
        event_buffer: McaEventBuffer | tuple[list[np.ndarray], threading.Lock] | None = None,
        on_ready: Callable[[], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
        output_mode: McaDmaOutputMode = McaDmaOutputMode.BINARY,
        configuration_yaml: str = "",
        client_record_schema: str | None = None,
    ) -> int:
        """Run one session on the calling thread against the bound backend.

        Raw IIO records are written unchanged. NDMA version 2 identifies
        their layout; version 1 remains the legacy event-record layout. The
        count includes zero-padded slots in the final complete frame: v121
        publishes no valid-record count and mca-architecture.md forbids
        treating zero payloads as an end marker because a real event may
        also contain zero-valued fields.
        """
        total_records = 0
        frame_count = 0
        self.last_capture_diagnostics = None
        writer: McaCaptureWriter | None = None
        started_utc = datetime.now(UTC)
        capture_schema = client_record_schema or IIO_LM_UNQUALIFIED_SCHEMA
        producer = getattr(self._backend, "mca_producer_identity", None)
        if callable(producer):
            qualified = self.qualified_record_schema()
            if qualified is None:
                if (event_buffer is not None or output_mode is not McaDmaOutputMode.BINARY
                        or client_record_schema is not None):
                    raise RuntimeError("Unknown MCA producer permits opaque binary recording only")
            else:
                if client_record_schema is not None and client_record_schema != qualified:
                    raise ValueError("Selected MCA decoder differs from the identified producer")
                capture_schema = qualified

        provenance_reader = getattr(self._backend, "mca_capture_provenance", None)
        provenance = provenance_reader() if callable(provenance_reader) else {}

        if filepath is not None:
            writer = McaCaptureWriter(
                path=filepath,
                mode=output_mode,
                configuration_yaml=configuration_yaml,
                binary_header=_file_header_bytes(
                    self._channel,
                    version=IIO_LM_FILE_VERSION,
                ),
                client_record_schema=capture_schema,
            )
            log.info("IIO MCA DMA: recording %s to %s", output_mode.value, filepath)

        def deliver(events: np.ndarray) -> None:
            nonlocal total_records, frame_count
            # Assert the backend and transport agree on the v121 ABI. This
            # catches an accidental legacy-dtype reuse before corrupting a
            # file or the GUI's shared event buffer.
            if events.dtype != _LM_EVENT_DTYPE:
                raise RuntimeError(
                    f"unexpected IIO MCA event dtype {events.dtype!r}; expected {_LM_EVENT_DTYPE!r}"
                )
            if writer is not None:
                writer.append(events.tobytes(), events)
            if event_buffer is not None:
                _append_mca_event_batch(event_buffer, events)
            frame_count += 1
            total_records += len(events)
            if on_progress is not None:
                on_progress(total_records)

        gate = ListModeStartGate(deliver)
        consume = gate.offer

        failure: BaseException | None = None
        cleanup_errors: list[dict[str, int | str]] = []
        producer_done = threading.Event()
        watch_stop = threading.Event()
        watch_errors: list[tuple[int, str]] = []
        watcher: threading.Thread | None = None
        backend = self._backend

        def watch_producer() -> None:
            # Hardware time-limit completion can occur before the GUI's stop
            # request arrives, and a refill entered after the producer's final
            # frame would block forever. Poll the producer off the data path;
            # on completion end the loop and start the ordinary STOP, whose
            # watchdog cancels only a refill that then waits a full
            # inactivity window without data, so the in-band tail is kept.
            while not watch_stop.wait(_MCA_PRODUCER_POLL_SECONDS):
                try:
                    if backend.mca_dma_measurement_in_progress():
                        continue
                    producer_done.set()
                except BaseException as exc:
                    # Scalar error data only; see AGENTS.md threading rules.
                    watch_errors.append((getattr(exc, "errno", None) or errno.EIO, str(exc)))
                    producer_done.set()
                try:
                    backend.request_mca_dma_stop()
                except BaseException as exc:
                    watch_errors.append((getattr(exc, "errno", None) or errno.EIO, str(exc)))
                return

        def stop_watcher() -> None:
            watch_stop.set()
            if watcher is not None:
                watcher.join(_MCA_PRODUCER_WATCH_JOIN_SECONDS)
                if watcher.is_alive():
                    log.warning("IIO MCA DMA: producer-status watcher did not stop")

        def retain_failure(phase: str, error: BaseException) -> None:
            nonlocal failure
            if failure is None:
                failure = error
            else:
                cleanup_errors.append({
                    "phase": phase, "errno": getattr(error, "errno", None) or 0,
                    "message": str(error),
                })
                failure.add_note(f"{phase}: {error}")

        # list_deadtime_raw is not reset by arm or START (L2): an earlier
        # session's value can persist. Only a change during this session is
        # this session's loss (PetaLinux cde5fe1 user-api.md, list mode).
        deadtime_before: int | None = None
        baseline_reader = getattr(self._backend, "get_mca_dma_deadtime_baseline", None)
        if callable(baseline_reader):
            deadtime_before = int(baseline_reader())

        try:
            # Backend readiness is a hardware lifecycle boundary, not merely
            # worker-thread startup: arm all kernel blocks, enter the
            # first refill, enable the pulse processor, then notify the GUI.
            first = self._backend.start_mca_dma_capture(on_started=on_ready)
            if len(first):
                consume(first)
            # One synchronous check after the first frame (a short time limit
            # can already be complete), then the watcher above replaces the
            # former per-frame status round trip.
            if self._backend.mca_dma_measurement_in_progress():
                watcher = threading.Thread(
                    target=watch_producer,
                    name=f"vdpp-lm-frame-ch{self._channel}-producer-watch",
                    daemon=True,
                )
                watcher.start()
                while not (stop_event.is_set() or producer_done.is_set()):
                    frame = self._backend.read_mca_dma_frame()
                    if len(frame):
                        consume(frame)
        except BaseException as exc:
            retain_failure("Capture", exc)
        # The watcher may call request_mca_dma_stop(); it must be finished
        # before close starts the tail drain, or its watchdog could cancel it.
        stop_watcher()
        if watch_errors:
            retain_failure("Producer status", OSError(*watch_errors[0]))
        # Healthy STOP preserves accepted tail frames. Fault cleanup must not
        # refill a failed descriptor or expose its contents as new payload.
        try:
            abort = getattr(self._backend, "abort_mca_dma_capture", None)
            if failure is not None and callable(abort):
                abort()
            else:
                self._backend.close_mca_dma_capture(
                    on_frame=consume if failure is None else None
                )
        except BaseException as exc:
            retain_failure("DMA close", exc)
        try:
            gate.finish()
        except BaseException as exc:
            retain_failure("Capture", exc)

        dma_fault = dma_error_count = completed_frames = deadtime_records = 0
        diagnostics_available = False
        try:
            dma_fault, dma_error_count, completed_frames, deadtime_records = (
                self._backend.get_mca_dma_capture_diagnostics()
            )
            diagnostics_available = True
        except BaseException as exc:
            retain_failure("Diagnostics", exc)
        # vdpp-lm-frame.c exposes completed_frames, dma_fault,
        # dma_error_count (cumulative since driver initialization), and
        # list_deadtime_raw directly through its IIO attributes.
        # Preserve the final driver readback for every output mode, including
        # online-only, instead of treating a clean worker exit as proof of
        # continuity.
        self.last_capture_diagnostics = {
            "dma_fault": dma_fault,
            "dma_error_count": dma_error_count,
            "completed_frames": completed_frames,
            "streamed_frames": frame_count,
            "list_deadtime_raw": deadtime_records,
            "list_deadtime_before_raw": deadtime_before,
            **gate.diagnostics(),
        }
        if gate.stale_frames:
            log.warning(
                "IIO MCA DMA ch%d: discarded %d frame(s) left over from an earlier "
                "session before this session's START record",
                self._channel, gate.stale_frames,
            )
        if isinstance(provenance, dict):
            self.last_capture_diagnostics.update(provenance)
        status_reader = getattr(self._backend, "get_mca_dma_capture_status", None)
        if callable(status_reader):
            try:
                self.last_capture_diagnostics.update(status_reader())
            except BaseException as exc:
                retain_failure("Final status", exc)
        if dma_fault and failure is None:
            detail = (
                "the capture file was retained for diagnostics"
                if writer is not None
                else "online capture continuity is invalid"
            )
            retain_failure("DMA fault", RuntimeError(
                f"MCA list-mode driver latched DMA fault reason {dma_fault}; {detail}"
            ))
        received_frames = frame_count + gate.stale_frames
        if diagnostics_available and completed_frames != received_frames and failure is None:
            detail = (
                "the retained capture is incomplete"
                if writer is not None
                else "online capture continuity is invalid"
            )
            retain_failure("Frame accounting", RuntimeError(
                f"MCA list-mode capture received {received_frames} complete frames, "
                f"but the driver completed {completed_frames}; {detail}"
            ))
        start_problem = gate.problem()
        if start_problem is not None and failure is None:
            retain_failure("Session start", RuntimeError(
                f"MCA list-mode continuity is invalid: {start_problem}"
            ))
        if _list_mode_loss(deadtime_records, deadtime_before) and failure is None:
            detail = (
                "the capture file was retained, but continuity is invalid"
                if writer is not None
                else "online capture continuity is invalid"
            )
            retain_failure("Producer loss", RuntimeError(
                f"MCA list-mode IP reports {deadtime_records} dropped records; {detail}"
            ))
        status = self.last_capture_diagnostics
        if failure is None and any(status.get(key) for key in (
            "dma_poisoned", "reader_active", "recovery_required", "status_errors", "cleanup_errors"
        )):
            retain_failure(
                "Final status", RuntimeError("MCA cleanup/recovery status is incomplete")
            )
        continuity_valid = failure is None and diagnostics_available
        if writer is not None:
            try:
                writer.close(complete=continuity_valid)
            except BaseException as exc:
                retain_failure("File close", exc)
                continuity_valid = False
        status["continuity_valid"] = continuity_valid
        if failure is not None:
            status["primary_errno"] = getattr(failure, "errno", None) or 0
            status["primary_error"] = str(failure)
        if cleanup_errors:
            previous_errors = status.get("cleanup_errors", [])
            status["cleanup_errors"] = (
                previous_errors if isinstance(previous_errors, list) else []
            ) + cleanup_errors
        if filepath is not None and output_mode is McaDmaOutputMode.BINARY:
            try:
                _write_iio_mca_metadata(
                    filepath, channel=self._channel, started_utc=started_utc,
                    frame_count=frame_count, dma_fault=dma_fault,
                    dma_error_count=dma_error_count, completed_frames=completed_frames,
                    deadtime_records=deadtime_records, deadtime_before=deadtime_before,
                    stale_frames=gate.stale_frames, client_record_schema=capture_schema,
                    final_status=status, identity=self.transport_identity(),
                    capture_sha256=writer.binary_sha256 if writer is not None else None,
                )
            except BaseException as exc:
                retain_failure("Metadata", exc)
                status["continuity_valid"] = False
                status["metadata_error"] = str(exc)
        if failure is not None:
            raise failure
        return total_records

    @staticmethod
    def parse_events(raw: bytes) -> np.ndarray:
        if len(raw) % _LM_EVENT_DTYPE.itemsize:
            raise ValueError(
                f"IIO list-mode payload is not aligned to {_LM_EVENT_DTYPE.itemsize}-byte records"
            )
        return np.frombuffer(raw, dtype=_LM_EVENT_DTYPE)

    @staticmethod
    def compute_cfd_time(events: np.ndarray) -> np.ndarray:
        """Return the selected signed CFD interpolation in ADC-sample units."""
        return cfd_interpolation_samples(events)
