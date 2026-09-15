"""Memory-mapped waveform access for NLab scope and CAEN binary files."""

from __future__ import annotations

import logging
import mmap
import struct
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nlab.analysis.caen_file import (
    CAEN_FILE_HEADER_BYTES,
    CaenFileInfo,
    IncompleteCaenEventError,
    inspect_caen_file,
    is_caen_header,
    parse_caen_event,
)
from nlab.hardware.digitizer.dma import FILE_HEADER_STRUCT, FILE_MAGIC, SCOPE_TIMESTAMP_WORDS
from nlab.utils.dma_converter import read_file_header

log = logging.getLogger(__name__)

_NDMA_SCOPE_SAMPLE_PERIOD_NS = 8.0


class WaveformIndexCancelledError(RuntimeError):
    pass


@dataclass(frozen=True)
class WaveformFileSummary:
    path: Path
    format_name: str
    channel: int


@dataclass(frozen=True)
class WaveformSourceChannel:
    board: int | None
    channel: int
    frame_count: int
    offsets: np.ndarray | None = None


@dataclass(frozen=True)
class WaveformFileIndex:
    path: Path
    format_name: str
    channels: tuple[WaveformSourceChannel, ...]
    sample_period_ns: float | None
    ndma_frame_samples: int = 0
    caen_info: CaenFileInfo | None = None

    @property
    def frame_count(self) -> int:
        """Return the total number of indexed frames across all sources."""
        return sum(source.frame_count for source in self.channels)

    @property
    def channel(self) -> int:
        """Return the sole or first source channel for routing compatibility."""
        return self.channels[0].channel


@dataclass(frozen=True)
class WaveformFrame:
    samples: np.ndarray
    timestamp: int
    board: int | None = None
    channel: int | None = None
    long_gate: int | None = None
    short_gate: int | None = None
    flags: int | None = None
    waveform_code: int | None = None


def _binary_signature(path: Path) -> tuple[bytes, int]:
    with path.open("rb") as stream:
        prefix = stream.read(4)
    if len(prefix) < 2:
        raise ValueError("File is too short to identify its binary format")
    return prefix, struct.unpack_from("<H", prefix)[0]


def inspect_waveform_file(path: Path) -> WaveformFileSummary:
    """Read only enough metadata to route a file to a Scope panel."""
    prefix, header_word = _binary_signature(path)
    if prefix == FILE_MAGIC:
        with path.open("rb") as stream:
            header = read_file_header(stream)
        if header["frame_samples"] <= SCOPE_TIMESTAMP_WORDS:
            raise ValueError("The selected NDMA file does not contain scope waveforms")
        return WaveformFileSummary(path, "NLab scope NDMA", header["channel"])
    if not is_caen_header(header_word):
        raise ValueError(
            "Unsupported waveform binary; expected NDMA or a CAEN 0xCAEx file"
        )
    info = inspect_caen_file(path)
    if not info.has_waveforms:
        raise ValueError("The selected CAEN file does not contain waveforms")
    return WaveformFileSummary(path, info.format_name, info.channel)


def build_waveform_file_index(
    path: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> WaveformFileIndex:
    """Inspect a waveform file and build only the index its format needs."""
    prefix, header_word = _binary_signature(path)
    if prefix == FILE_MAGIC:
        with path.open("rb") as stream:
            header = read_file_header(stream)
        frame_samples = header["frame_samples"]
        if frame_samples <= SCOPE_TIMESTAMP_WORDS:
            raise ValueError("The selected NDMA file does not contain scope waveforms")
        frame_bytes = frame_samples * np.dtype("<i2").itemsize
        payload = max(0, path.stat().st_size - FILE_HEADER_STRUCT.size)
        frame_count = payload // frame_bytes
        trailing = payload % frame_bytes
        if trailing:
            log.warning(
                "Waveform browser ignores %d trailing bytes from an incomplete NDMA frame",
                trailing,
            )
        if frame_count == 0:
            raise ValueError("The NDMA file contains no complete scope frames")
        if progress is not None:
            progress(path.stat().st_size, path.stat().st_size)
        return WaveformFileIndex(
            path=path,
            format_name="NLab scope NDMA",
            channels=(
                WaveformSourceChannel(
                    board=None,
                    channel=header["channel"],
                    frame_count=frame_count,
                ),
            ),
            sample_period_ns=_NDMA_SCOPE_SAMPLE_PERIOD_NS,
            ndma_frame_samples=frame_samples,
        )

    if not is_caen_header(header_word):
        raise ValueError(
            "Unsupported waveform binary; expected NDMA or a CAEN 0xCAEx file"
        )
    info = inspect_caen_file(path)
    if not info.has_waveforms:
        raise ValueError("The selected CAEN file does not contain waveforms")

    offsets_by_source: dict[tuple[int, int], array[int]] = {}
    size = path.stat().st_size
    with path.open("rb") as stream:
        mapped = mmap.mmap(stream.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            offset = CAEN_FILE_HEADER_BYTES
            last_progress = offset
            while offset < len(mapped):
                if cancelled is not None and cancelled():
                    raise WaveformIndexCancelledError
                try:
                    event = parse_caen_event(mapped, offset, info)
                except IncompleteCaenEventError:
                    log.warning(
                        "Waveform browser ignores %d trailing bytes from an incomplete "
                        "CAEN event",
                        len(mapped) - offset,
                    )
                    break
                offsets_by_source.setdefault(
                    (event.board, event.channel), array("Q")
                ).append(offset)
                offset = event.next_offset
                if progress is not None and offset - last_progress >= 4 * 1024 * 1024:
                    progress(offset, size)
                    last_progress = offset
            if progress is not None:
                progress(offset, size)
        finally:
            mapped.close()
    if not offsets_by_source:
        raise ValueError("The CAEN file contains no complete waveform events")
    channels = tuple(
        WaveformSourceChannel(
            board=board,
            channel=channel,
            frame_count=len(offsets),
            offsets=np.frombuffer(offsets, dtype=np.uint64).copy(),
        )
        for (board, channel), offsets in sorted(offsets_by_source.items())
    )
    return WaveformFileIndex(
        path=path,
        format_name=info.format_name,
        channels=channels,
        sample_period_ns=None,
        caen_info=info,
    )


class MappedWaveformFile:
    """Hold one file mapping and copy only the frame selected for display."""

    def __init__(self, index: WaveformFileIndex) -> None:
        self.index = index
        self._stream = index.path.open("rb")
        self._mapped = mmap.mmap(self._stream.fileno(), length=0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        self._mapped.close()
        self._stream.close()

    def frame(self, frame_index: int, source_index: int = 0) -> WaveformFrame:
        if not 0 <= source_index < len(self.index.channels):
            raise IndexError(f"waveform source {source_index} is out of range")
        source = self.index.channels[source_index]
        if not 0 <= frame_index < source.frame_count:
            raise IndexError(f"waveform frame {frame_index} is out of range")
        if self.index.caen_info is not None:
            offsets = source.offsets
            assert offsets is not None
            event = parse_caen_event(
                self._mapped,
                int(offsets[frame_index]),
                self.index.caen_info,
            )
            assert event.waveform_offset is not None
            samples = np.frombuffer(
                self._mapped,
                dtype="<i2",
                count=event.waveform_samples,
                offset=event.waveform_offset,
            ).copy()
            return WaveformFrame(
                samples=samples,
                timestamp=event.timestamp,
                board=event.board,
                channel=event.channel,
                long_gate=event.long_gate,
                short_gate=event.short_gate,
                flags=event.flags,
                waveform_code=event.waveform_code,
            )

        frame_samples = self.index.ndma_frame_samples
        frame_bytes = frame_samples * np.dtype("<i2").itemsize
        offset = FILE_HEADER_STRUCT.size + frame_index * frame_bytes
        timestamp = struct.unpack_from("<Q", self._mapped, offset)[0]
        samples = np.frombuffer(
            self._mapped,
            dtype="<i2",
            count=frame_samples - SCOPE_TIMESTAMP_WORDS,
            offset=offset + SCOPE_TIMESTAMP_WORDS * 2,
        ).copy()
        return WaveformFrame(samples=samples, timestamp=timestamp, channel=source.channel)
