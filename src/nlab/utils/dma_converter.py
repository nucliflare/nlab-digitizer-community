"""Convert raw DMA binary files to HDF5 for scientific analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from nlab.hardware.digitizer.dma import (
    _EVENT_DTYPE,
    _LM_EVENT_DTYPE,
    FILE_HEADER_STRUCT,
    FILE_MAGIC,
    FILE_VERSION,
    IIO_LM_FILE_VERSION,
    SCOPE_TIMESTAMP_WORDS,
)

log = logging.getLogger(__name__)


def read_file_header(f) -> dict:
    raw = f.read(FILE_HEADER_STRUCT.size)
    if len(raw) < FILE_HEADER_STRUCT.size:
        raise ValueError("File too short for header")
    magic, version, channel, _, timestamp, frame_samples = FILE_HEADER_STRUCT.unpack(raw)
    if magic != FILE_MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}, expected {FILE_MAGIC!r}")
    return {
        "version": version,
        "channel": channel,
        "timestamp": timestamp,
        "frame_samples": frame_samples,
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
    n_events = len(raw_data) // event_size
    if len(raw_data) % event_size != 0:
        log.warning("Listmode file size not aligned to event size, truncating %d trailing bytes",
                    len(raw_data) % event_size)
    events = np.frombuffer(raw_data[:n_events * event_size], dtype=event_dtype)

    with h5py.File(dst, "w") as h5:
        h5.attrs["source_file"] = str(src)
        h5.attrs["format_version"] = header["version"]
        h5.attrs["channel"] = header["channel"]
        h5.attrs["recording_timestamp"] = header["timestamp"]
        h5.attrs["total_events"] = n_events

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
            ds.attrs["fields"] = (
                "flags, cfd_q2, charge_energy, trapezoid_energy, timestamp"
            )
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
            cfd_time = events["cfd_q2"].astype(np.float64) / 4.0
            h5.create_dataset(
                "cfd_time", data=cfd_time, compression="gzip", compression_opts=4,
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
