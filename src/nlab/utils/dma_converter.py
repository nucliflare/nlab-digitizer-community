"""Convert raw DMA binary files to HDF5 for scientific analysis."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, BinaryIO, TypedDict

import h5py
import numpy as np

from nlab.hardware.digitizer.dma import (
    _EVENT_DTYPE,
    _LM_EVENT_DTYPE,
    FILE_HEADER_STRUCT,
    FILE_MAGIC,
    FILE_VERSION,
    IIO_LM_FILE_VERSION,
    IIO_LM_FRAME_BYTES,
    IIO_LM_FRAME_RECORDS,
    IIO_LM_UNQUALIFIED_SCHEMA,
    SCOPE_TIMESTAMP_WORDS,
)
from nlab.hardware.digitizer.iio_listmode import cfd_interpolation_samples, cfd_valid

log = logging.getLogger(__name__)


class FileHeader(TypedDict):
    version: int
    channel: int
    timestamp: float
    frame_samples: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_iio_capture_metadata(src: Path, source_sha256: str) -> dict[str, Any] | None:
    """Read and authenticate the optional sidecar written by the IIO streamer."""
    metadata_path = src.with_suffix(".json")
    if not metadata_path.is_file():
        return None
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Capture metadata is not a JSON object: {metadata_path}")
    if metadata.get("format") != "nlab-iio-mca-ndma-v2":
        log.warning("Ignoring unrelated capture metadata sidecar: %s", metadata_path)
        return None
    expected_sha256 = metadata.get("capture_sha256")
    if expected_sha256 != source_sha256:
        raise ValueError(
            "Capture SHA-256 does not match its metadata sidecar: "
            f"{metadata_path}"
        )
    return metadata


def read_file_header(f: BinaryIO) -> FileHeader:
    raw = f.read(FILE_HEADER_STRUCT.size)
    if len(raw) < FILE_HEADER_STRUCT.size:
        raise ValueError("File too short for header")
    magic, version, channel, _, timestamp, frame_samples = FILE_HEADER_STRUCT.unpack(raw)
    if magic != FILE_MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}, expected {FILE_MAGIC!r}")
    return {
        "version": int(version),
        "channel": int(channel),
        "timestamp": float(timestamp),
        "frame_samples": int(frame_samples),
    }


def convert_listmode(src: Path, dst: Path) -> int:
    """Convert MCA listmode binary to HDF5. Returns event count."""
    with open(src, "rb") as f:
        header = read_file_header(f)
        raw_data = f.read()

    if header["version"] == FILE_VERSION:
        event_dtype = _EVENT_DTYPE
    elif header["version"] == IIO_LM_FILE_VERSION:
        event_dtype = _LM_EVENT_DTYPE
    else:
        raise ValueError(f"Unsupported MCA list-mode format version: {header['version']}")

    event_size = event_dtype.itemsize
    if len(raw_data) % event_size:
        raise ValueError(
            "List-mode payload is not aligned to complete records: "
            f"{len(raw_data)} bytes for {event_size}-byte records"
        )
    if (
        header["version"] == IIO_LM_FILE_VERSION
        and len(raw_data) % IIO_LM_FRAME_BYTES
    ):
        raise ValueError(
            "IIO list-mode payload is not aligned to complete DMA frames: "
            f"{len(raw_data)} bytes for {IIO_LM_FRAME_BYTES}-byte frames"
        )

    n_events = len(raw_data) // event_size
    n_frames = (
        n_events // IIO_LM_FRAME_RECORDS
        if header["version"] == IIO_LM_FILE_VERSION
        else None
    )
    events = np.frombuffer(raw_data, dtype=event_dtype)
    source_sha256 = _sha256_file(src)
    capture_metadata = (
        _read_iio_capture_metadata(src, source_sha256)
        if header["version"] == IIO_LM_FILE_VERSION
        else None
    )
    client_record_schema = (
        capture_metadata.get("client_record_schema")
        if capture_metadata is not None
        else IIO_LM_UNQUALIFIED_SCHEMA
    )
    if not isinstance(client_record_schema, str):
        client_record_schema = IIO_LM_UNQUALIFIED_SCHEMA

    if capture_metadata is not None:
        expected = {
            "channel_index": header["channel"],
            "record_bytes": event_size,
            "frame_records": IIO_LM_FRAME_RECORDS,
            "frame_bytes": IIO_LM_FRAME_BYTES,
            "frames": n_frames,
            "records": n_events,
        }
        mismatches = {
            name: (capture_metadata.get(name), value)
            for name, value in expected.items()
            if capture_metadata.get(name) != value
        }
        if mismatches:
            raise ValueError(
                "Capture metadata does not match the NDMA payload: "
                f"{mismatches}"
            )

    with h5py.File(dst, "w") as h5:
        h5.attrs["source_file"] = str(src)
        h5.attrs["format_version"] = header["version"]
        h5.attrs["channel"] = header["channel"]
        h5.attrs["recording_timestamp"] = header["timestamp"]
        h5.attrs["total_events"] = n_events
        h5.attrs["source_sha256"] = source_sha256
        if n_frames is not None:
            h5.attrs["transport_schema"] = "opaque[16]"
            h5.attrs["record_bytes"] = event_size
            h5.attrs["frame_records"] = IIO_LM_FRAME_RECORDS
            h5.attrs["frame_bytes"] = IIO_LM_FRAME_BYTES
            h5.attrs["total_frames"] = n_frames
        if capture_metadata is not None:
            h5.attrs["capture_metadata_json"] = json.dumps(
                capture_metadata, sort_keys=True,
            )
            for name in (
                "driver_completed_frames",
                "driver_dma_fault",
                "driver_dma_error_count",
                "list_deadtime_raw",
                "continuity_valid",
            ):
                h5.attrs[name] = capture_metadata[name]

        ds = h5.create_dataset("events", data=events, compression="gzip", compression_opts=4)
        ds.attrs["timestamp_unit"] = "8 ns ticks"
        h5.create_dataset(
            "timestamp", data=events["timestamp"], compression="gzip", compression_opts=4,
        )

        if header["version"] == FILE_VERSION:
            ds.attrs["fields"] = (
                "marker, zc_offset, zc_estimation, short_energy, energy, timestamp"
            )
            h5.create_dataset(
                "energy", data=events["energy"], compression="gzip", compression_opts=4,
            )
            psd_zc = events["zc_offset"].astype(np.float64) + (
                events["zc_estimation"].view(np.int16).astype(np.float64) / 2**14
            )
            h5.create_dataset("psd_zc", data=psd_zc, compression="gzip", compression_opts=4)
        else:
            h5.attrs["client_record_schema"] = client_record_schema
            ds.attrs["fields"] = (
                "marker, zc_offset, zc_estimation, charge_energy, "
                "trapezoid_energy, timestamp"
            )
            ds.attrs["zc_offset_format"] = "unsigned uint8, 2 ns ADC samples"
            ds.attrs["zc_estimation_format"] = "signed Q2.14 ADC-sample fraction"
            h5.create_dataset(
                "charge_energy",
                data=events["charge_energy"],
                compression="gzip",
                compression_opts=4,
            )
            h5.create_dataset(
                "trapezoid_energy",
                data=events["trapezoid_energy"],
                compression="gzip",
                compression_opts=4,
            )
            cfd_interpolation = h5.create_dataset(
                "cfd_interpolation_samples",
                data=cfd_interpolation_samples(events),
                compression="gzip",
                compression_opts=4,
            )
            cfd_interpolation.attrs["unit"] = "ADC samples (2 ns/sample)"
            cfd_interpolation.attrs["meaning"] = (
                "signed fractional-sample CFD term; reconstruct event time with the "
                "named client record schema"
            )
            h5.create_dataset(
                "cfd_valid", data=cfd_valid(events), compression="gzip", compression_opts=4,
            )

    log.info("Converted %d listmode events: %s -> %s", n_events, src, dst)
    return n_events


def convert_scope(src: Path, dst: Path) -> int:
    """Convert scope DMA binary to HDF5. Returns frame count."""
    with open(src, "rb") as f:
        header = read_file_header(f)
        raw_data = f.read()

    frame_samples = header["frame_samples"]
    if frame_samples == 0:
        raise ValueError("Scope file has frame_samples=0 in header, cannot determine frame size")
    if frame_samples <= SCOPE_TIMESTAMP_WORDS:
        raise ValueError(
            "Scope frame is too short to contain its 64-bit timestamp: "
            f"{frame_samples} samples"
        )

    # The NDMA header stores a count of int16 values, not a byte count.
    frame_bytes = frame_samples * np.dtype("<i2").itemsize

    if len(raw_data) % frame_bytes != 0:
        raise ValueError(
            "Scope payload is not aligned to complete frames: "
            f"{len(raw_data)} bytes for {frame_bytes}-byte frames"
        )
    n_frames = len(raw_data) // frame_bytes

    words_per_frame = frame_bytes // 2
    samples_per_frame = words_per_frame - SCOPE_TIMESTAMP_WORDS
    frame_dtype = np.dtype([("timestamp", "<u8"), ("samples", "<i2", (samples_per_frame,))])
    frames = np.frombuffer(raw_data[:n_frames * frame_bytes], dtype=frame_dtype)
    timestamps = frames["timestamp"]
    waveforms = frames["samples"]
    time_ns = np.arange(0, 8 * samples_per_frame, 8)

    with h5py.File(dst, "w") as h5:
        h5.attrs["source_file"] = str(src)
        h5.attrs["format_version"] = header["version"]
        h5.attrs["channel"] = header["channel"]
        h5.attrs["recording_timestamp"] = header["timestamp"]
        h5.attrs["frame_samples"] = header["frame_samples"]
        h5.attrs["total_frames"] = n_frames
        h5.attrs["samples_per_frame"] = samples_per_frame

        h5.create_dataset("waveforms", data=waveforms, compression="gzip", compression_opts=4)
        h5["waveforms"].attrs["unit"] = "ADC counts (int16)"
        h5.create_dataset("timestamps", data=timestamps, compression="gzip", compression_opts=4)
        h5["timestamps"].attrs["unit"] = "8 ns ticks"
        h5.create_dataset("time_ns", data=time_ns)
        h5["time_ns"].attrs["unit"] = "nanoseconds"

    log.info("Converted %d scope frames (%d samples each): %s -> %s",
             n_frames, samples_per_frame, src, dst)
    return n_frames
