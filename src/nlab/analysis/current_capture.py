"""Lazy, hardware-independent browsing of measured periodic-sum recordings.

Reads both containers: HDF5 (raw packets plus decoded columns) and NSUM
binary (raw packets only, ``scope_sum_file``). Statistics describe window
means, not unrecorded ADC samples. Only packets in the requested view (plus
their neighbors) are validated, not the entire file.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import h5py
import numpy as np
import numpy.typing as npt

from nlab.hardware.digitizer import scope_sum_file as sum_file
from nlab.hardware.digitizer.scope_sums import (
    MASK64,
    ScopeSumContinuity,
    ScopeSumGeometry,
    ScopeSumPacket,
    decode_sum_packet,
)

_PREVIEW_POINTS = 100_000
_PACKET_BATCH = 32
MAX_MOVING_AVERAGE_KERNEL = 1_000_000


class CurrentReadCancelledError(Exception):
    """A newer view request superseded this read."""


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise CurrentReadCancelledError


@dataclass(frozen=True)
class CurrentCaptureInfo:
    path: Path
    fingerprint: tuple[int, int]
    metadata: dict[str, Any]
    runtime_metadata: dict[str, Any]
    configuration_yaml: str
    complete: bool
    cleanup_error: str
    final_missed_windows: int
    offsets: npt.NDArray[np.uint64]
    geometry: ScopeSumGeometry | None
    first_timestamp: int
    duration_ticks: int
    container: str = "hdf5"  # or "nsum"

    @property
    def windows(self) -> int:
        return int(self.offsets[-1])

    @property
    def duration_us(self) -> float:
        return self.duration_ticks * 0.008


@dataclass(frozen=True)
class CurrentChunk:
    start_us: float
    end_us: float
    count: int
    mean: float | None
    std: float | None
    rms: float | None
    minimum: float | None
    maximum: float | None
    adjacent_noise: float | None
    adjacent_pairs: int
    skipped_windows: int
    # Preview timestamps stay native uint64; x is relative to the requested start.
    timestamps: npt.NDArray[np.uint64]
    x_us: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    connect: npt.NDArray[np.bool_]
    histogram_counts: npt.NDArray[np.int64]
    histogram_edges: npt.NDArray[np.float64]
    average_kernel: int
    average_values: npt.NDArray[np.float64]


def _fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _json_attribute(source: h5py.File, name: str) -> dict[str, Any]:
    value = json.loads(str(source.attrs.get(name, "{}")))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a mapping")
    return value


class _Source(Protocol):
    def packets(self, start: int, stop: int) -> npt.NDArray[np.uint8]: ...
    def stored(self, name: str, begin: int, end: int) -> npt.NDArray[np.uint64] | None: ...
    def timestamp(self, info: CurrentCaptureInfo, window: int) -> int: ...
    def close(self) -> None: ...


class _Hdf5Source:
    """Raw packets plus the decoded columns stored beside them."""

    def __init__(self, path: Path | h5py.File) -> None:
        # Wrap an already-open file without taking ownership of it.
        self._owned = not isinstance(path, h5py.File)
        self._file: h5py.File = path if isinstance(path, h5py.File) else h5py.File(path, "r")

    def packets(self, start: int, stop: int) -> npt.NDArray[np.uint8]:
        return np.asarray(self._file["packets"][start:stop], dtype=np.uint8)

    def stored(self, name: str, begin: int, end: int) -> npt.NDArray[np.uint64] | None:
        return np.asarray(self._file[name][begin:end], dtype=np.uint64)

    def timestamp(self, info: CurrentCaptureInfo, window: int) -> int:
        return int(self._file["timestamps"][window])

    def close(self) -> None:
        if self._owned:
            self._file.close()


class _BinarySource:
    """NSUM raw packets; window timestamps come from decoding their packet."""

    def __init__(self, path: Path) -> None:
        self._layout = sum_file.read_layout(path)
        self._handle: BinaryIO = path.open("rb")
        self._decoded: tuple[int, ScopeSumPacket] | None = None

    def packets(self, start: int, stop: int) -> npt.NDArray[np.uint8]:
        data = sum_file.read_packet_bytes(self._handle, self._layout, start, stop)
        return np.frombuffer(data, dtype=np.uint8).reshape(-1, sum_file.PACKET_BYTES)

    def stored(self, name: str, begin: int, end: int) -> npt.NDArray[np.uint64] | None:
        return None  # Only the raw packets are stored.

    def timestamp(self, info: CurrentCaptureInfo, window: int) -> int:
        index = int(np.searchsorted(info.offsets, window, side="right")) - 1
        if self._decoded is None or self._decoded[0] != index:
            raw = self.packets(index, index + 1)[0].tobytes()
            self._decoded = (index, decode_sum_packet(raw, info.geometry))
        return int(self._decoded[1].timestamps[window - int(info.offsets[index])])

    def close(self) -> None:
        self._handle.close()


@contextmanager
def _open_source(info: CurrentCaptureInfo) -> Iterator[_Source]:
    source: _Source = (
        _BinarySource(info.path) if info.container == "nsum" else _Hdf5Source(info.path)
    )
    try:
        yield source
    finally:
        source.close()


def _packet(source: _Source, index: int, offsets: npt.NDArray[np.uint64]) -> ScopeSumPacket:
    packet = decode_sum_packet(source.packets(index, index + 1)[0].tobytes())
    start, end = int(offsets[index]), int(offsets[index + 1])
    if end - start != packet.timestamps.size:
        raise ValueError("Stored valid_count differs from raw packet")
    for name, expected in (("timestamps", packet.timestamps), ("sums", packet.sums)):
        stored = source.stored(name, start, end)
        if stored is not None and not np.array_equal(stored, expected):
            raise ValueError(f"Stored {name} differs from raw packet")
    return packet


def _span(
    first: ScopeSumPacket, last: ScopeSumPacket,
) -> tuple[ScopeSumGeometry, int, int]:
    geometry = ScopeSumGeometry(first.samples_per_window, first.ip_version, first.gap_cycles)
    if (last.samples_per_window, last.ip_version, last.gap_cycles) != (
        geometry.frame_samples,
        geometry.ip_version,
        geometry.gap_cycles,
    ):
        raise ValueError("Sum geometry changed during acquisition")
    first_stamp = int(first.timestamps[0])
    duration_ticks = (int(last.timestamps[-1]) - first_stamp) & MASK64
    if duration_ticks >= 1 << 63:
        raise ValueError("Nonmonotonic or ambiguous capture timestamp span")
    return geometry, first_stamp, duration_ticks + geometry.frame_samples // 4


def _binary_capture_info(
    path: Path, fingerprint: tuple[int, int], cancelled: Callable[[], bool],
) -> CurrentCaptureInfo:
    layout = sum_file.read_layout(path)
    header = layout.header
    if (header.get("format"), header.get("format_version")) != ("nlab-scope-sums", 1):
        raise ValueError("Choose an nlab-scope-sums version 1 capture")
    metadata = header.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("NSUM metadata must contain a mapping")
    trailer = layout.trailer
    if trailer is None:
        cleanup_error = "Recording did not close; packets up to the last whole block are shown"
        if layout.ignored_bytes:
            cleanup_error += f" ({layout.ignored_bytes} trailing bytes ignored)"
        runtime: Any = {}
    else:
        cleanup_error = str(trailer.get("cleanup_error", ""))
        runtime = trailer.get("runtime_metadata") or {}
        if not isinstance(runtime, dict):
            raise ValueError("runtime_metadata must contain a mapping")
    _check_cancelled(cancelled)
    counts = np.zeros(layout.packets, dtype=np.uint64)
    geometry, first_stamp, duration_ticks = None, 0, 0
    if layout.packets:
        source = _BinarySource(path)
        try:
            first = decode_sum_packet(source.packets(0, 1)[0].tobytes())
            last = decode_sum_packet(
                source.packets(layout.packets - 1, layout.packets)[0].tobytes()
            )
        finally:
            source.close()
        # The producer seals every packet full except a final STOP flush
        # (scope-periodic-sum-v123.md). A partial packet elsewhere fails decode
        # when viewed, like any other invalid packet.
        capacity = 508 if first.ip_version == 124 else 254
        if layout.packets > 1 and first.timestamps.size != capacity:
            raise ValueError("First sum packet is partial but not the last packet")
        counts[:] = capacity
        counts[-1] = last.timestamps.size
        geometry, first_stamp, duration_ticks = _span(first, last)
    offsets = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(counts, dtype=np.uint64)))
    return CurrentCaptureInfo(
        path,
        fingerprint,
        metadata,
        runtime,
        str(header.get("configuration_yaml", "") or ""),
        bool(trailer is not None and trailer.get("complete", False)),
        cleanup_error,
        int(trailer.get("final_missed_windows", 0)) if trailer is not None else 0,
        offsets,
        geometry,
        first_stamp,
        duration_ticks,
        "nsum",
    )


def read_current_capture_info(
    path: Path,
    *,
    cancelled: Callable[[], bool] = lambda: False,
) -> CurrentCaptureInfo:
    path = Path(path)
    fingerprint = _fingerprint(path)
    if sum_file.is_binary_sum_file(path):
        info = _binary_capture_info(path, fingerprint, cancelled)
        _check_cancelled(cancelled)
        if _fingerprint(path) != fingerprint:
            raise ValueError("Capture changed while opening; stop recording before browsing")
        return info
    with h5py.File(path, "r") as source:
        if (source.attrs.get("format"), source.attrs.get("format_version")) != (
            "nlab-scope-sums",
            1,
        ):
            raise ValueError("Choose an nlab-scope-sums version 1 HDF5 capture")
        for name, shape, size in (
            ("packets", (None, 4096), 1),
            ("valid_counts", (None,), 2),
            ("timestamps", (None,), 8),
            ("sums", (None,), 8),
        ):
            data = source[name]
            if (
                data.ndim != len(shape)
                or data.shape[1:] != shape[1:]
                or data.dtype.kind != "u"
                or data.dtype.itemsize != size
            ):
                raise ValueError(f"Invalid {name} dataset layout")
        _check_cancelled(cancelled)
        counts = source["valid_counts"][:]
        if len(counts) != len(source["packets"]) or np.any((counts < 1) | (counts > 508)):
            raise ValueError("Truncated or invalid packet/count table")
        offsets = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(counts, dtype=np.uint64)))
        if len(source["timestamps"]) != int(offsets[-1]) or len(source["sums"]) != offsets[-1]:
            raise ValueError("Trailing or truncated sum record columns")
        geometry: ScopeSumGeometry | None = None
        first_stamp, duration_ticks = 0, 0
        if len(counts):
            packets = _Hdf5Source(source)
            first = _packet(packets, 0, offsets)
            last = _packet(packets, len(counts) - 1, offsets)
            geometry, first_stamp, duration_ticks = _span(first, last)
        text = source["configuration_yaml"][()] if "configuration_yaml" in source else ""
        configuration = text.decode("utf-8") if isinstance(text, bytes) else str(text)
        info = CurrentCaptureInfo(
            path,
            fingerprint,
            _json_attribute(source, "metadata_json"),
            _json_attribute(source, "runtime_metadata_json"),
            configuration,
            bool(source.attrs.get("complete", False)),
            str(source.attrs.get("cleanup_error", "")),
            int(source.attrs.get("final_missed_windows", 0)),
            offsets,
            geometry,
            first_stamp,
            duration_ticks,
        )
    _check_cancelled(cancelled)
    if _fingerprint(path) != fingerprint:
        raise ValueError("Capture changed while opening; stop recording before browsing")
    return info


def _boundary_ticks(time_us: float) -> int:
    scaled = time_us * 125
    nearest = round(scaled)
    # Decimal UI values and repeated additions may land a few ulps above an
    # exact native tick. Do not discard the window on a half-open chunk edge.
    if abs(scaled - nearest) <= 4 * math.ulp(scaled):
        return nearest
    return math.ceil(scaled)


def _lower_bound(source: _Source, info: CurrentCaptureInfo, target: int) -> int:
    low, high = 0, info.windows
    while low < high:
        mid = (low + high) // 2
        ticks = (source.timestamp(info, mid) - info.first_timestamp) & MASK64
        if ticks < target:
            low = mid + 1
        else:
            high = mid
    return low


def _blocks(
    source: _Source,
    info: CurrentCaptureInfo,
    left: int,
    right: int,
    cancelled: Callable[[], bool],
) -> Iterator[tuple[int, npt.NDArray[np.uint64], npt.NDArray[np.float64]]]:
    if left == right:
        return
    p0 = max(0, int(np.searchsorted(info.offsets, left, side="right")) - 2)
    p1 = min(len(info.offsets) - 1, int(np.searchsorted(info.offsets, right - 1, side="right")) + 1)
    continuity = ScopeSumContinuity()
    for batch in range(p0, p1, _PACKET_BATCH):
        _check_cancelled(cancelled)
        # Read fixed-size batches; retain no open file or IIO handles in results.
        raw = source.packets(batch, min(batch + _PACKET_BATCH, p1))
        for j, row in enumerate(raw, batch):
            packet = decode_sum_packet(row.tobytes(), info.geometry)
            continuity.append(packet)
            begin, end = int(info.offsets[j]), int(info.offsets[j + 1])
            if end - begin != packet.timestamps.size:
                raise ValueError("Stored valid_count differs from raw packet")
            data = np.column_stack((packet.timestamps, packet.sums))
            for col, name in enumerate(("timestamps", "sums")):
                stored = source.stored(name, begin, end)
                if stored is not None and not np.array_equal(stored, data[:, col]):
                    raise ValueError(f"Stored {name} differs from raw packet")
            a, b = max(left, begin), min(right, end)
            if a < b:
                selected = data[a - begin : b - begin]
                ys = selected[:, 1].astype(float) / packet.samples_per_window - 32768
                yield a, selected[:, 0], ys


def iter_window_means(
    info: CurrentCaptureInfo,
    *,
    cancelled: Callable[[], bool] = lambda: False,
) -> Iterator[tuple[int, npt.NDArray[np.uint64], npt.NDArray[np.float64]]]:
    """Every recorded window mean in order: (first window index, timestamps, means).

    Packets are read in batches and checked as for a browsed chunk: decode,
    sequence/timestamp continuity and stored columns against the raw packets.
    Raises CurrentReadCancelledError once ``cancelled`` returns true.
    """
    if _fingerprint(info.path) != info.fingerprint:
        raise ValueError("Capture changed; reopen the finished file")
    with _open_source(info) as source:
        yield from _blocks(source, info, 0, info.windows, cancelled)
    if _fingerprint(info.path) != info.fingerprint:
        raise ValueError("Capture changed while reading; reopen the finished file")


class _MovingAverage:
    """Trailing full-kernel means with O(kernel) memory and O(samples) work.

    The first kernel-1 values in each contiguous run are NaN. State carries
    across I/O blocks within the viewed chunk, and resets at native tick gaps.
    """

    def __init__(self, kernel: int, interval_ticks: int) -> None:
        self.kernel = kernel
        self.interval_ticks = interval_ticks
        self._ring = np.empty(kernel, dtype=float)
        self._cursor = self._count = 0
        self._total = 0.0
        self._last_timestamp: int | None = None

    def _reset(self) -> None:
        self._cursor = self._count = 0
        self._total = 0.0

    def _append_run(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.kernel == 1:
            return values.copy()
        size = len(values)
        removed = np.zeros(size, dtype=float)
        # Unfilled ring positions contribute zero during initial warm-up.
        first = min(size, self.kernel - self._count)
        end = min(size, self.kernel)
        if first < end:
            positions = (self._cursor + np.arange(first, end)) % self.kernel
            removed[first:end] = self._ring[positions]
        if size > self.kernel:
            removed[self.kernel :] = values[: -self.kernel]
        totals = self._total + np.cumsum(values - removed)
        result = totals / self.kernel
        result[: max(0, self.kernel - self._count - 1)] = np.nan
        keep = min(size, self.kernel)
        positions = (self._cursor + np.arange(size - keep, size)) % self.kernel
        self._ring[positions] = values[-keep:]
        self._cursor = (self._cursor + size) % self.kernel
        self._count = min(self.kernel, self._count + size)
        self._total = float(totals[-1])
        return result

    def append(
        self,
        timestamps: npt.NDArray[np.uint64],
        values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        if not len(values):
            return values.copy()
        if (
            self._last_timestamp is not None
            and (int(timestamps[0]) - self._last_timestamp) & MASK64 != self.interval_ticks
        ):
            self._reset()
        breaks = np.flatnonzero(np.diff(timestamps) != self.interval_ticks) + 1
        result = np.empty(len(values), dtype=float)
        begin = 0
        for end in [*breaks, len(values)]:
            if begin:
                self._reset()
            result[begin:end] = self._append_run(values[begin:end])
            begin = end
        self._last_timestamp = int(timestamps[-1])
        return result


def read_current_chunk(
    info: CurrentCaptureInfo,
    start_us: float,
    width_us: float,
    *,
    average_kernel: int = 0,
    cancelled: Callable[[], bool] = lambda: False,
) -> CurrentChunk:
    if (
        not math.isfinite(start_us)
        or not math.isfinite(width_us)
        or start_us < 0
        or width_us < 0.008
    ):
        raise ValueError("Start must be nonnegative and view width at least 0.008 µs")
    if _fingerprint(info.path) != info.fingerprint:
        raise ValueError("Capture changed; reopen the finished file")
    if type(average_kernel) is not int or not 0 <= average_kernel <= MAX_MOVING_AVERAGE_KERNEL:
        raise ValueError(f"Average kernel must be an integer from 0 to {MAX_MOVING_AVERAGE_KERNEL}")
    end_us = min(start_us + width_us, info.duration_us)
    count, m2, mean, noise2, pairs, skipped = 0, 0.0, 0.0, 0.0, 0, 0
    minimum, maximum = math.inf, -math.inf
    previous_stamp, previous_value = None, 0.0
    stamps, values, row_numbers, averages = [], [], [], []
    interval = info.geometry.interval_ticks if info.geometry else 1
    average = _MovingAverage(average_kernel, interval) if average_kernel else None
    with _open_source(info) as source:
        start_tick = _boundary_ticks(start_us)
        end_tick = _boundary_ticks(max(start_us, end_us))
        left = _lower_bound(source, info, start_tick)
        right = _lower_bound(source, info, end_tick)
        stride = max(1, math.ceil((right - left - 1) / max(1, _PREVIEW_POINTS - 1)))
        for row, ts, ys in _blocks(source, info, left, right, cancelled):
            batch_mean = float(np.mean(ys))
            delta = batch_mean - mean
            total = count + len(ys)
            m2 += float(np.sum((ys - batch_mean) ** 2)) + delta * delta * count * len(ys) / total
            mean += delta * len(ys) / total
            count = total
            minimum, maximum = min(minimum, float(ys.min())), max(maximum, float(ys.max()))
            differences = np.diff(ys)
            ticks = np.diff(ts)  # uint64 subtraction preserves native counter wrap.
            adjacent = ticks == interval
            noise2 += float(np.sum(differences[adjacent] ** 2))
            pairs += int(np.count_nonzero(adjacent))
            if previous_stamp is not None:
                gap = (int(ts[0]) - previous_stamp) & MASK64
                if gap == interval:
                    pairs += 1
                    noise2 += (float(ys[0]) - previous_value) ** 2
            previous_stamp, previous_value = int(ts[-1]), float(ys[-1])
            indices = np.arange(row, row + len(ys))
            selected = ((indices - left) % stride == 0) | (indices == right - 1)
            smoothed = average.append(ts, ys) if average is not None else None
            if np.any(selected):
                stamps.append(ts[selected])
                values.append(ys[selected])
                row_numbers.append(indices[selected])
                if smoothed is not None:
                    averages.append(smoothed[selected])
        edges = (
            np.linspace(minimum, maximum, 129)
            if minimum < maximum
            else np.linspace(
                (minimum if count else 0) - 0.5,
                (maximum if count else 0) + 0.5,
                129,
            )
        )
        hist = np.zeros(128, dtype=np.int64)
        for _, _, ys in _blocks(source, info, left, right, cancelled):
            hist += np.histogram(ys, bins=edges)[0]
    scheduled = (end_tick + interval - 1) // interval - (start_tick + interval - 1) // interval
    skipped = max(0, scheduled - count)
    _check_cancelled(cancelled)
    if _fingerprint(info.path) != info.fingerprint:
        raise ValueError("Capture changed while reading; reopen the finished file")
    timestamps = np.concatenate(stamps) if stamps else np.empty(0, dtype=np.uint64)
    preview = np.concatenate(values) if values else np.empty(0, dtype=float)
    rows = np.concatenate(row_numbers) if row_numbers else np.empty(0, dtype=np.int64)
    connect = np.zeros(len(timestamps), dtype=bool)
    connect[:-1] = np.diff(timestamps) == np.diff(rows).astype(np.uint64) * interval
    # Subtract native integers before converting to display floats.
    relative = (timestamps - np.uint64(info.first_timestamp)).astype(float) * 0.008 - start_us
    std = math.sqrt(max(0.0, m2 / count)) if count else None
    return CurrentChunk(
        start_us,
        max(start_us, end_us),
        count,
        mean if count else None,
        std,
        math.hypot(mean, std) if std is not None else None,
        minimum if count else None,
        maximum if count else None,
        math.sqrt(noise2 / (2 * pairs)) if pairs else None,
        pairs,
        skipped,
        timestamps,
        relative,
        preview,
        connect,
        hist,
        edges,
        average_kernel,
        np.concatenate(averages) if averages else np.full(len(preview), np.nan),
    )
