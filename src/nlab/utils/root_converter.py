"""Bounded-memory, lossless NDMA-to-ROOT TTree conversion (no hardware access)."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nlab.hardware.digitizer.dma import (
    _EVENT_DTYPE,
    _LM_EVENT_DTYPE,
    FILE_HEADER_STRUCT,
    FILE_VERSION,
    IIO_LM_FILE_VERSION,
    IIO_LM_FRAME_BYTES,
    IIO_LM_FRAME_RECORDS,
    IIO_LM_UNQUALIFIED_SCHEMA,
    SCOPE_TIMESTAMP_WORDS,
)
from nlab.utils.dma_converter import _read_iio_capture_metadata, read_file_header

_BATCH_BYTES = 4 * 1024 * 1024
log = logging.getLogger(__name__)


class ConversionCancelledError(Exception):
    """Cooperative cancellation; the incomplete output is removed."""


@dataclass(frozen=True)
class RootConversionResult:
    kind: str
    count: int
    destination: Path


def convert_binary_to_root(
    src: Path,
    dst: Path,
    *,
    progress: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> RootConversionResult:
    """Convert native Scope v1 or MCA v1/v2 NDMA without overwriting files.

    Each raw field is retained at its original integer width, including padding
    records. Scope samples are indexed, not assigned a guessed sample period:
    NDMA v1 does not encode sampling/decimation settings. Optional IIO JSON
    metadata is authenticated exactly as in the existing HDF5 converter.
    """
    # Load the native ROOT dependency stack on the conversion worker, not while
    # importing the File-menu dialog. Missing frozen dependencies are then
    # reported through its normal error path instead of losing the menu click.
    import uproot

    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve() or (dst.exists() and src.samefile(dst)):
        raise ValueError("Source and destination must be different files")
    if dst.exists():
        raise FileExistsError("Destination already exists; choose a new ROOT filename")

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise ConversionCancelledError()

    last_progress = -1

    def report(value: int) -> None:
        nonlocal last_progress
        if progress is not None and value != last_progress:
            last_progress = value
            progress(value)

    check_cancelled()
    report(0)
    created = False
    try:
        with src.open("rb") as source:
            initial_stat = src.stat()
            header = read_file_header(source)
            payload_bytes = initial_stat.st_size - FILE_HEADER_STRUCT.size
            scope = header["frame_samples"] > 0
            version = header["version"]
            if scope:
                if version != FILE_VERSION:
                    raise ValueError(f"Unsupported Scope format version: {version}")
                samples = header["frame_samples"] - SCOPE_TIMESTAMP_WORDS
                if samples <= 0:
                    raise ValueError("Scope frame is too short for timestamp and waveform")
                dtype = np.dtype([("timestamp", "<u8"), ("samples", "<i2", (samples,))])
                alignment = dtype.itemsize
            elif version in (FILE_VERSION, IIO_LM_FILE_VERSION):
                dtype = _EVENT_DTYPE if version == FILE_VERSION else _LM_EVENT_DTYPE
                alignment = IIO_LM_FRAME_BYTES if version == IIO_LM_FILE_VERSION else dtype.itemsize
            else:
                raise ValueError(f"Unsupported MCA list-mode format version: {version}")
            if payload_bytes % alignment:
                raise ValueError("Payload is not aligned to complete records / DMA frames")
            count = payload_bytes // dtype.itemsize
            names = dtype.names or ()
            fields = dtype.fields
            assert fields is not None
            metadata: dict[str, Any] = {
                "format": "nlab-ndma-root-v1",
                "capture_kind": "scope" if scope else "listmode",
                "source_file": str(src),
                **header,
                "total_entries": count,
                "timestamp_unit": "8 ns ticks",
            }
            # Hash incrementally, permitting cancellation even during a large scan.
            source.seek(0)
            digest = hashlib.sha256()
            scanned = 0
            while block := source.read(_BATCH_BYTES):
                check_cancelled()
                digest.update(block)
                scanned += len(block)
                report(min(30, scanned * 30 // initial_stat.st_size))
            source_hash = digest.hexdigest()
            metadata["source_sha256"] = source_hash
            capture_metadata = None
            if not scope and version == IIO_LM_FILE_VERSION:
                capture_metadata = _read_iio_capture_metadata(src, source_hash)
                if capture_metadata is not None:
                    expected = {
                        "channel_index": header["channel"],
                        "record_bytes": dtype.itemsize,
                        "frame_records": IIO_LM_FRAME_RECORDS,
                        "frame_bytes": IIO_LM_FRAME_BYTES,
                        "frames": payload_bytes // IIO_LM_FRAME_BYTES,
                        "records": count,
                    }
                    if any(capture_metadata.get(key) != value for key, value in expected.items()):
                        raise ValueError("Capture metadata does not match the NDMA payload")
                metadata["client_record_schema"] = (
                    capture_metadata.get("client_record_schema", IIO_LM_UNQUALIFIED_SCHEMA)
                    if capture_metadata is not None else IIO_LM_UNQUALIFIED_SCHEMA
                )
                metadata.update(
                    frame_records=IIO_LM_FRAME_RECORDS,
                    frame_bytes=IIO_LM_FRAME_BYTES,
                    total_frames=payload_bytes // IIO_LM_FRAME_BYTES,
                )
            if scope:
                metadata.update(
                    total_frames=count,
                    samples_per_frame=samples,
                    sample_unit="ADC counts (int16)",
                    sample_period_ns=None,
                    sample_axis="integer sample index; sampling period is not encoded in NDMA v1",
                )
            else:
                metadata["total_events"] = count
            schema = {name: str(fields[name][0]) for name in names}
            schema["timestamp"] = "uint64, 8 ns ticks"
            if not scope and version == IIO_LM_FILE_VERSION:
                schema["client_record_schema"] = metadata["client_record_schema"]
                schema["zc_offset"] = "uint8, 2 ns ADC samples (client profile)"
                schema["zc_estimation"] = "int16, signed Q2.14 ADC-sample fraction (client profile)"
            config = src.with_suffix(".yaml")
            configuration = (
                config.read_text(encoding="utf-8") if config.is_file()
                else json.dumps({"measurement": {"channel": header["channel"]}})
            )
            check_cancelled()
            # Exclusive creation also protects aliases and files created after the
            # initial existence check. Only our own incomplete output is removed.
            with dst.open("x+b") as output:
                created = True
                with uproot.recreate(output) as root:
                    root["metadata_json"] = json.dumps(metadata, sort_keys=True)
                    root["configuration_yaml"] = configuration
                    root["waveform_schema" if scope else "event_schema"] = json.dumps(schema)
                    if capture_metadata is not None:
                        root["capture_metadata_json"] = json.dumps(capture_metadata, sort_keys=True)
                    # Explicit mktree is required: assignment defaults to RNTuple
                    # in uproot >=5.7. Keep compatibility with ROOT TTree tooling.
                    tree = root.mktree(
                        "waveforms" if scope else "events",
                        {name: fields[name][0] for name in names},
                        title="NLab Scope waveforms" if scope else "NLab MCA list-mode events",
                    )
                    written = 0
                    batch_bytes = max(alignment, _BATCH_BYTES // alignment * alignment)
                    payload_hash = hashlib.sha256()
                    source.seek(0)
                    payload_hash.update(source.read(FILE_HEADER_STRUCT.size))
                    while written < payload_bytes:
                        check_cancelled()
                        raw = source.read(min(batch_bytes, payload_bytes - written))
                        if not raw or len(raw) % alignment:
                            raise ValueError("Source changed or ended during conversion")
                        payload_hash.update(raw)
                        batch = np.frombuffer(raw, dtype=dtype)
                        tree.extend({name: batch[name] for name in names})
                        written += len(raw)
                        report(30 + written * 69 // max(1, payload_bytes))
                    final_stat = src.stat()
                    if (
                        source.read(1)
                        or payload_hash.hexdigest() != source_hash
                        or final_stat.st_size != initial_stat.st_size
                        or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
                    ):
                        raise ValueError("Source changed during conversion; stop recording first")
                    check_cancelled()
        report(100)
    except BaseException:
        if created:
            try:
                dst.unlink()
            except OSError:
                log.exception("Could not remove incomplete ROOT conversion: %s", dst)
        raise
    return RootConversionResult("scope frames" if scope else "list-mode events", count, dst)
