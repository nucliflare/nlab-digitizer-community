"""Readers for native CAEN CoMPASS and legacy ``caen.py`` binaries."""

from __future__ import annotations

import logging
import mmap
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CAEN_SIGNATURE = 0xCAE0
CAEN_SIGNATURE_MASK = 0xFFF0
CAEN_LEGACY_HEADER = 0xDCAC
CAEN_FILE_HEADER_BYTES = 2

CAEN_RAW_ENERGY = 0x1
CAEN_CALIBRATED_ENERGY = 0x2
CAEN_SHORT_ENERGY = 0x4
CAEN_WAVEFORM = 0x8

_LEGACY_EVENT = struct.Struct("<HHQHHI")
_BASE_EVENT = struct.Struct("<HHQ")
_PSD_EVENT_DTYPE = np.dtype(
    [("timestamp", "<u8"), ("long_gate", "<u2"), ("short_gate", "<u2")]
)


class IncompleteCaenEventError(ValueError):
    """The file ended part-way through the next CAEN event."""


@dataclass(frozen=True)
class CaenFileInfo:
    path: Path
    format_name: str
    header: int
    options: int
    board: int
    channel: int
    total_events: int
    fixed_record_bytes: int | None

    @property
    def legacy(self) -> bool:
        return self.header == CAEN_LEGACY_HEADER

    @property
    def has_waveforms(self) -> bool:
        return not self.legacy and bool(self.options & CAEN_WAVEFORM)

    @property
    def psd_compatible(self) -> bool:
        return self.legacy or (
            bool(self.options & CAEN_RAW_ENERGY)
            and bool(self.options & CAEN_SHORT_ENERGY)
        )


@dataclass(frozen=True)
class CaenEvent:
    board: int
    channel: int
    timestamp: int
    long_gate: int | None
    short_gate: int | None
    flags: int | None
    waveform_code: int | None
    waveform_offset: int | None
    waveform_samples: int
    next_offset: int


def is_caen_header(header: int) -> bool:
    return header == CAEN_LEGACY_HEADER or (
        header & CAEN_SIGNATURE_MASK
    ) == CAEN_SIGNATURE


def _require_bytes(size: int, offset: int, count: int) -> None:
    if offset < 0 or count < 0 or offset + count > size:
        raise IncompleteCaenEventError(
            f"CAEN event at byte {offset} extends beyond the file"
        )


def _native_field_offsets(options: int) -> tuple[int | None, int | None, int | None, int]:
    cursor = _BASE_EVENT.size
    long_offset: int | None = None
    short_offset: int | None = None
    flags_offset: int | None = None
    if options & CAEN_RAW_ENERGY:
        long_offset = cursor
        cursor += 2
    if options & CAEN_CALIBRATED_ENERGY:
        cursor += 8
    if options & CAEN_SHORT_ENERGY:
        short_offset = cursor
        flags_offset = cursor + 2
        cursor += 6
    return long_offset, short_offset, flags_offset, cursor


def native_fixed_record_bytes(options: int) -> int | None:
    """Return packed event bytes, or ``None`` for waveform-bearing records."""
    if options & CAEN_WAVEFORM:
        return None
    return _native_field_offsets(options)[-1]


def parse_caen_event(data: mmap.mmap, offset: int, info: CaenFileInfo) -> CaenEvent:
    """Parse one event and locate, but do not materialize, its waveform."""
    size = len(data)
    if info.legacy:
        _require_bytes(size, offset, _LEGACY_EVENT.size)
        board, channel, timestamp, long_gate, short_gate, flags = (
            _LEGACY_EVENT.unpack_from(data, offset)
        )
        return CaenEvent(
            board=board,
            channel=channel,
            timestamp=timestamp,
            long_gate=long_gate,
            short_gate=short_gate,
            flags=flags,
            waveform_code=None,
            waveform_offset=None,
            waveform_samples=0,
            next_offset=offset + _LEGACY_EVENT.size,
        )

    _require_bytes(size, offset, _BASE_EVENT.size)
    board, channel, timestamp = _BASE_EVENT.unpack_from(data, offset)
    long_offset, short_offset, flags_offset, cursor = _native_field_offsets(info.options)
    _require_bytes(size, offset, cursor)
    long_gate = (
        struct.unpack_from("<H", data, offset + long_offset)[0]
        if long_offset is not None
        else None
    )
    short_gate = (
        struct.unpack_from("<H", data, offset + short_offset)[0]
        if short_offset is not None
        else None
    )
    flags = (
        struct.unpack_from("<I", data, offset + flags_offset)[0]
        if flags_offset is not None
        else None
    )
    waveform_code: int | None = None
    waveform_offset: int | None = None
    waveform_samples = 0
    if info.options & CAEN_WAVEFORM:
        _require_bytes(size, offset + cursor, 5)
        waveform_code = data[offset + cursor]
        waveform_samples = struct.unpack_from("<i", data, offset + cursor + 1)[0]
        if waveform_samples < 0:
            raise ValueError(
                f"CAEN event at byte {offset} has negative waveform length "
                f"{waveform_samples}"
            )
        waveform_offset = offset + cursor + 5
        _require_bytes(size, waveform_offset, waveform_samples * 2)
        cursor += 5 + waveform_samples * 2

    return CaenEvent(
        board=board,
        channel=channel,
        timestamp=timestamp,
        long_gate=long_gate,
        short_gate=short_gate,
        flags=flags,
        waveform_code=waveform_code,
        waveform_offset=waveform_offset,
        waveform_samples=waveform_samples,
        next_offset=offset + cursor,
    )


def inspect_caen_file(path: Path) -> CaenFileInfo:
    """Validate a CAEN signature and inspect its first event."""
    size = path.stat().st_size
    if size < CAEN_FILE_HEADER_BYTES + _BASE_EVENT.size:
        raise ValueError("File is too short to contain a CAEN event")
    with path.open("rb") as stream:
        raw_header = stream.read(CAEN_FILE_HEADER_BYTES)
        header = struct.unpack("<H", raw_header)[0]
        if not is_caen_header(header):
            raise ValueError(f"Unsupported CAEN binary header 0x{header:04X}")
        mapped = mmap.mmap(stream.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            if header == CAEN_LEGACY_HEADER:
                options = 0
                record_bytes: int | None = _LEGACY_EVENT.size
                format_name = "CAEN legacy coincidence binary"
            else:
                options = header & 0xF
                record_bytes = native_fixed_record_bytes(options)
                format_name = "CAEN CoMPASS binary"
            provisional = CaenFileInfo(
                path=path,
                format_name=format_name,
                header=header,
                options=options,
                board=0,
                channel=0,
                total_events=0,
                fixed_record_bytes=record_bytes,
            )
            first = parse_caen_event(mapped, CAEN_FILE_HEADER_BYTES, provisional)
        finally:
            mapped.close()

    total_events = (
        (size - CAEN_FILE_HEADER_BYTES) // record_bytes
        if record_bytes is not None
        else 0
    )
    return CaenFileInfo(
        path=path,
        format_name=format_name,
        header=header,
        options=options,
        board=first.board,
        channel=first.channel,
        total_events=total_events,
        fixed_record_bytes=record_bytes,
    )


def _fixed_event_dtype(info: CaenFileInfo) -> np.dtype:
    if info.legacy:
        return np.dtype(
            [
                ("board", "<u2"),
                ("channel", "<u2"),
                ("timestamp", "<u8"),
                ("long_gate", "<u2"),
                ("short_gate", "<u2"),
                ("flags", "<u4"),
            ]
        )
    names = ["board", "channel", "timestamp"]
    formats: list[str] = ["<u2", "<u2", "<u8"]
    if info.options & CAEN_RAW_ENERGY:
        names.append("long_gate")
        formats.append("<u2")
    if info.options & CAEN_CALIBRATED_ENERGY:
        names.append("calibrated_energy")
        formats.append("<f8")
    if info.options & CAEN_SHORT_ENERGY:
        names.extend(("short_gate", "flags"))
        formats.extend(("<u2", "<u4"))
    return np.dtype(list(zip(names, formats, strict=True)))


def _canonical_batch(events: np.ndarray) -> np.ndarray:
    batch = np.empty(len(events), dtype=_PSD_EVENT_DTYPE)
    batch["timestamp"] = events["timestamp"]
    batch["long_gate"] = events["long_gate"]
    batch["short_gate"] = events["short_gate"]
    return batch


def iter_caen_psd_batches(
    path: Path,
    *,
    target_batch_bytes: int,
) -> Iterator[np.ndarray]:
    """Yield canonical PSD batches without decoding waveform samples."""
    info = inspect_caen_file(path)
    if not info.psd_compatible:
        raise ValueError(
            "The CAEN file does not contain both raw Energy and Energy Short "
            "and cannot be used for PSD reconstruction"
        )
    with path.open("rb") as stream:
        mapped = mmap.mmap(stream.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            if info.fixed_record_bytes is not None:
                dtype = _fixed_event_dtype(info)
                payload_bytes = len(mapped) - CAEN_FILE_HEADER_BYTES
                usable_bytes = payload_bytes - payload_bytes % dtype.itemsize
                if usable_bytes != payload_bytes:
                    log.warning(
                        "PSD import ignores %d trailing bytes from incomplete CAEN data",
                        payload_bytes - usable_bytes,
                    )
                total = usable_bytes // dtype.itemsize
                batch_records = max(1, target_batch_bytes // dtype.itemsize)
                for first in range(0, total, batch_records):
                    count = min(batch_records, total - first)
                    raw = np.frombuffer(
                        mapped,
                        dtype=dtype,
                        count=count,
                        offset=CAEN_FILE_HEADER_BYTES + first * dtype.itemsize,
                    )
                    multiple_channels = bool(np.any(raw["channel"] != info.channel))
                    canonical = _canonical_batch(raw)
                    del raw
                    if multiple_channels:
                        raise ValueError("CAEN file contains events from multiple channels")
                    yield canonical
                return

            capacity = 65_536
            batch = np.empty(capacity, dtype=_PSD_EVENT_DTYPE)
            count = 0
            batch_start = CAEN_FILE_HEADER_BYTES
            offset = CAEN_FILE_HEADER_BYTES
            while offset < len(mapped):
                try:
                    event = parse_caen_event(mapped, offset, info)
                except IncompleteCaenEventError:
                    log.warning(
                        "PSD import ignores %d trailing bytes from an incomplete CAEN event",
                        len(mapped) - offset,
                    )
                    break
                if event.channel != info.channel:
                    raise ValueError("CAEN file contains events from multiple channels")
                assert event.long_gate is not None and event.short_gate is not None
                batch[count] = (event.timestamp, event.long_gate, event.short_gate)
                count += 1
                offset = event.next_offset
                if count == capacity or offset - batch_start >= target_batch_bytes:
                    yield batch[:count].copy()
                    batch = np.empty(capacity, dtype=_PSD_EVENT_DTYPE)
                    count = 0
                    batch_start = offset
            if count:
                yield batch[:count].copy()
        finally:
            mapped.close()
