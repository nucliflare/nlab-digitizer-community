"""IP123/IP124 periodic-sum packets and lossless, versioned recordings.

Contract: PetaLinux bb8c57e, scope-periodic-sum-v123.md. Sums contain all N
original offset-binary ADC samples. They cannot recover waveforms or extrema.
"""

from __future__ import annotations

import errno
import json
import struct
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict

import h5py
import numpy as np

MASK64 = (1 << 64) - 1
MAGICS = {0x31534D5553504456: 123, 0x32534D5553504456: 124}


class ScopeSumSnapshot(TypedDict):
    points: tuple[tuple[float, float], ...]
    packets: int
    windows: int
    samples: int
    signed_sum: int
    coverage_ns: int
    missed: int
    skipped: int
    mean: float | None
    complete: NotRequired[bool]
    final_missed_windows: NotRequired[int]
    ip_version: NotRequired[int]
    samples_per_window: NotRequired[int]
    gap_cycles: NotRequired[int]
    interval_ns: NotRequired[int]


class ScopeSumBackend(Protocol):
    def prepare_dma_capture(self) -> None: ...
    def get_scope_dma_geometry(self) -> object: ...
    def read_dma_raw_frame(self) -> bytes: ...
    def request_scope_sum_stop(self) -> None: ...
    def close_scope_sum_capture(self) -> None: ...
    def stop_scope_sum_from_reader(self) -> None: ...
    def scope_sum_status(self) -> tuple[int, int]: ...
    def scope_sum_identity(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class ScopeSumGeometry:
    frame_samples: int
    ip_version: int
    gap_cycles: int
    buffer_samples: int = 2048
    frame_bytes: int = 4096
    waveform_samples: int = 0
    sample_decimation: int = 0
    padding_bytes: int = 0
    ndma_v1_compatible: bool = False

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int
                for value in (self.ip_version, self.frame_samples, self.gap_cycles)
            )
            or self.ip_version not in (123, 124)
            or not 4 <= self.frame_samples <= 8188
            or self.frame_samples % 4
            or not 0 <= self.gap_cycles <= 65535
            or (
                self.buffer_samples,
                self.frame_bytes,
                self.waveform_samples,
                self.sample_decimation,
                self.padding_bytes,
            )
            != (2048, 4096, 0, 0, 0)
        ):
            raise ValueError("Invalid periodic-sum geometry")

    @property
    def interval_ticks(self) -> int:
        return self.frame_samples // 4 + self.gap_cycles


@dataclass(frozen=True)
class ScopeSumPacket:
    ip_version: int
    sequence: int
    samples_per_window: int
    gap_cycles: int
    missed_windows: int
    stop_flush: bool
    records: tuple[tuple[int, int], ...]
    raw: bytes

    @property
    def signed_means(self) -> tuple[float, ...]:
        return tuple(total / self.samples_per_window - 32768.0 for _, total in self.records)


def decode_sum_packet(raw: bytes, geometry: ScopeSumGeometry | None = None) -> ScopeSumPacket:
    if len(raw) != 4096:
        raise ValueError("A periodic-sum packet must contain exactly 4096 bytes")
    first, magic, config, state = struct.unpack_from("<4Q", raw)
    version = MAGICS.get(magic)
    if version is None:
        raise ValueError("Unknown periodic-sum magic/ABI")
    n, gap, count = config & 0xFFFFFFFF, (config >> 32) & 0xFFFF, config >> 48
    actual = ScopeSumGeometry(n, version, gap)
    if geometry is not None and actual != geometry:
        raise ValueError("Sum packet differs from the armed IP/window/gap")
    capacity = 508 if version == 124 else 254
    stopped = bool(state >> 63)
    if not 1 <= count <= capacity or (count != capacity and not stopped):
        raise ValueError("Invalid sum valid_count or partial packet without STOP flush")
    if version == 124:
        sums = struct.unpack_from("<508Q", raw, 32)
        if any(value != sums[count - 1] for value in sums[count:]):
            raise ValueError("Invalid last-valid-value padding in IP124 packet")
        records = tuple(
            ((first + i * actual.interval_ticks) & MASK64, sums[i]) for i in range(count)
        )
    else:
        if any(raw[32 + count * 16 :]):
            raise ValueError("Nonzero IP123 packet padding")
        records = tuple(struct.unpack_from("<QQ", raw, 32 + i * 16) for i in range(count))
        if records[0][0] != first:
            raise ValueError("Header timestamp differs from the first sum window")
    previous = None
    for stamp, total in records:
        if total > n * 65535:
            raise ValueError("Sum exceeds N * full-scale ADC code")
        if previous is not None:
            delta = (stamp - previous) & MASK64
            if not 0 < delta < (1 << 63) or delta % actual.interval_ticks:
                raise ValueError("Nonmonotonic or off-grid sum timestamp")
        previous = stamp
    return ScopeSumPacket(
        version,
        state & 0xFFFFFFFF,
        n,
        gap,
        (state >> 32) & 0x7FFFFFFF,
        stopped,
        records,
        bytes(raw),
    )


class ScopeSumContinuity:
    """Validate modulo-u32 sequence and modulo-u64 timestamp continuity."""

    def __init__(self) -> None:
        self.previous: ScopeSumPacket | None = None
        self.skipped_windows = 0

    def append(self, packet: ScopeSumPacket) -> None:
        old = self.previous
        interval = packet.samples_per_window // 4 + packet.gap_cycles
        if old is not None:
            if old.stop_flush:
                raise ValueError("Packet received after terminal STOP flush")
            if ((packet.sequence - old.sequence) & 0xFFFFFFFF) != 1:
                raise ValueError("Sum packet sequence discontinuity")
            if (old.ip_version, old.samples_per_window, old.gap_cycles) != (
                packet.ip_version,
                packet.samples_per_window,
                packet.gap_cycles,
            ):
                raise ValueError("Sum geometry changed during acquisition")
            delta = (packet.records[0][0] - old.records[-1][0]) & MASK64
            if not 0 < delta < (1 << 63) or delta % interval:
                raise ValueError("Sum packet timestamps are nonmonotonic or off-grid")
            self.skipped_windows += delta // interval - 1
        for (before, _), (after, _) in zip(packet.records, packet.records[1:]):
            self.skipped_windows += ((after - before) & MASK64) // interval - 1
        self.previous = packet


class ScopeSumWriter:
    """Append original packets and exact u64 records; incomplete until clean close."""

    def __init__(self, path: Path, metadata: dict[str, object]) -> None:
        self._file = h5py.File(path, "x")
        self._file.attrs.update(
            format="nlab-scope-sums",
            format_version=1,
            signal_origin="measurement",
            created_utc=datetime.now(UTC).isoformat(),
            timestamp_unit="8 ns",
            sum_unit="offset-binary ADC code sum",
            complete=False,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        self._file.create_dataset(
            "packets", shape=(0, 4096), maxshape=(None, 4096), chunks=(16, 4096), dtype="u1"
        )
        for name in ("timestamps", "sums"):
            self._file.create_dataset(
                name, shape=(0,), maxshape=(None,), chunks=(8192,), dtype="<u8"
            )
        self._file.create_dataset(
            "valid_counts", shape=(0,), maxshape=(None,), chunks=(512,), dtype="<u2"
        )
        self._closed = False

    def append(self, packet: ScopeSumPacket) -> None:
        packets = self._file["packets"]
        row = packets.shape[0]
        packets.resize(row + 1, axis=0)
        packets[row] = np.frombuffer(packet.raw, dtype="u1")
        counts = self._file["valid_counts"]
        counts.resize(row + 1, axis=0)
        counts[row] = len(packet.records)
        for index, name in enumerate(("timestamps", "sums")):
            data = self._file[name]
            start = data.shape[0]
            data.resize(start + len(packet.records), axis=0)
            data[start:] = np.asarray([record[index] for record in packet.records], dtype="<u8")

    def close(self, *, complete: bool, error: str = "", missed_windows: int = 0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._file.attrs.update(
                complete=complete, cleanup_error=error, final_missed_windows=missed_windows
            )
            self._file.flush()
        finally:
            self._file.close()


def read_sum_capture(path: Path) -> Iterator[ScopeSumPacket]:
    """Reopen a sum recording; validate raw and decoded columns independently."""
    with h5py.File(path, "r") as source:
        if (
            source.attrs.get("format") != "nlab-scope-sums"
            or source.attrs.get("format_version") != 1
        ):
            raise ValueError("Unsupported Scope sum file")
        continuity = ScopeSumContinuity()
        offset = 0
        if len(source["packets"]) != len(source["valid_counts"]):
            raise ValueError("Truncated sum packet/count table")
        for index, row in enumerate(source["packets"]):
            packet = decode_sum_packet(row.tobytes())
            continuity.append(packet)
            count = len(packet.records)
            if int(source["valid_counts"][index]) != count:
                raise ValueError("Stored valid_count differs from raw packet")
            for col, name in enumerate(("timestamps", "sums")):
                expected = np.asarray([r[col] for r in packet.records], dtype="<u8")
                if not np.array_equal(source[name][offset : offset + count], expected):
                    raise ValueError(f"Stored {name} differs from raw packet")
            offset += count
            yield packet
        if len(source["timestamps"]) != offset or len(source["sums"]) != offset:
            raise ValueError("Trailing or truncated sum record columns")


class ScopeSumAnalysis:
    """Bounded display history plus exact, sample-weighted session totals."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._points: deque[tuple[float, float]] = deque(maxlen=8192)
        self._last_timestamp: int | None = None
        self._elapsed_ticks = 0
        self.packet_count = self.window_count = self.sample_count = self.signed_sum = 0
        self.missed_windows = self.coverage_ns = 0
        self.continuity = ScopeSumContinuity()

    def append(self, packet: ScopeSumPacket) -> None:
        with self._lock:
            self.continuity.append(packet)
            n = packet.samples_per_window
            for timestamp, total in packet.records:
                if self._last_timestamp is not None:
                    self._elapsed_ticks += (timestamp - self._last_timestamp) & MASK64
                self._last_timestamp = timestamp
                self._points.append((self._elapsed_ticks * 8e-9, total / n - 32768.0))
                self.signed_sum += total - n * 32768
            self.packet_count += 1
            self.window_count += len(packet.records)
            self.sample_count += n * len(packet.records)
            self.coverage_ns += n * 2 * len(packet.records)
            self.missed_windows = packet.missed_windows

    def snapshot(self) -> ScopeSumSnapshot:
        with self._lock:
            result: ScopeSumSnapshot = dict(
                points=tuple(self._points),
                packets=self.packet_count,
                windows=self.window_count,
                samples=self.sample_count,
                signed_sum=self.signed_sum,
                coverage_ns=self.coverage_ns,
                missed=self.missed_windows,
                skipped=self.continuity.skipped_windows,
                mean=self.signed_sum / self.sample_count if self.sample_count else None,
            )

            previous = self.continuity.previous
            if previous is not None:
                result["ip_version"] = previous.ip_version
                result["samples_per_window"] = previous.samples_per_window
                result["gap_cycles"] = previous.gap_cycles
                result["interval_ns"] = (previous.samples_per_window // 4 + previous.gap_cycles) * 8

            return result


class ScopeSumCapture:
    """Single-reader session; STOP drains through terminal timeout before close.

    The backend's dedicated DMA context owns refills. The GUI's stop hook only
    clears the acquisition gate; it never cancels a healthy tail drain.
    """

    def __init__(self, backend: ScopeSumBackend, channel: int) -> None:
        self.backend = backend
        self.channel = channel
        self.analysis = ScopeSumAnalysis()
        self._stop = threading.Event()
        self._started = False

    def request_stop(self) -> None:
        self._stop.set()
        if self._started:
            self.backend.request_scope_sum_stop()

    def run(
        self,
        path: Path | None = None,
        ready: Callable[[], None] | None = None,
        max_packets: int | None = None,
    ) -> ScopeSumSnapshot:
        writer = None
        error = ""
        complete = False
        missed = 0
        prepared = False
        try:
            self.backend.prepare_dma_capture()
            prepared = True
            geometry = self.backend.get_scope_dma_geometry()
            if not isinstance(geometry, ScopeSumGeometry):
                raise ValueError("Periodic-sum capture requires mode 5 on IP123/IP124")
            if path is not None:
                metadata: dict[str, object] = dict(
                    channel=self.channel,
                    ip_version=geometry.ip_version,
                    samples_per_window=geometry.frame_samples,
                    gap_cycles=geometry.gap_cycles,
                    adc_sample_period_ns=2,
                    timestamp_period_ns=8,
                    calibration=None,
                    current_conversion="unqualified; ADC codes only",
                    producer=self.backend.scope_sum_identity(),
                )
                writer = ScopeSumWriter(path, metadata)
            deadline = None
            self._started = True
            while True:
                if self._stop.is_set() and deadline is None:
                    if self.analysis.packet_count == 0:
                        break  # Never arm a new buffer after cancellation.
                    self.backend.stop_scope_sum_from_reader()
                    deadline = time.monotonic() + 5.0
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("Periodic-sum tail drain exceeded five seconds")
                try:
                    raw = self.backend.read_dma_raw_frame()
                except OSError as exc:
                    if self._stop.is_set() and exc.errno == errno.ECANCELED:
                        break
                    if self._stop.is_set() and exc.errno == errno.ETIMEDOUT:
                        status, missed = self.backend.scope_sum_status()
                        if status & (2 | 8):
                            raise RuntimeError(
                                f"Sum tail did not finish: status=0x{status:x}"
                            ) from None
                        complete = True
                        break
                    raise
                packet = decode_sum_packet(raw, geometry)
                self.analysis.append(packet)
                if writer is not None:
                    writer.append(packet)
                if ready is not None:
                    ready()
                    ready = None
                if max_packets is not None and self.analysis.packet_count >= max_packets:
                    self._stop.set()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if prepared:
                try:
                    self.backend.stop_scope_sum_from_reader()
                    self.backend.close_scope_sum_capture()
                except Exception as exc:
                    error = (error + "; " if error else "") + f"Cleanup: {exc}"
                    complete = False
            self._started = False
            if writer is not None:
                writer.close(complete=complete and not error, error=error, missed_windows=missed)
        if error:
            raise RuntimeError(error)
        return self.analysis.snapshot() | {"complete": complete, "final_missed_windows": missed}
