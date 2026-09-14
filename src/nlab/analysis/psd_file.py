"""Bounded-memory readers for saved MCA event files."""

from __future__ import annotations

import logging
import mmap
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from nlab.hardware.digitizer.dma import (
    _EVENT_DTYPE,
    _LM_EVENT_DTYPE,
    FILE_HEADER_STRUCT,
    FILE_VERSION,
    IIO_LM_FILE_VERSION,
    IIO_LM_FRAME_BYTES,
    IIO_LM_FRAME_RECORDS,
)
from nlab.utils.dma_converter import read_file_header

log = logging.getLogger(__name__)

_TARGET_BATCH_BYTES = 16 * 1024 * 1024
_CANONICAL_DTYPE = np.dtype(
    [("timestamp", "<u8"), ("long_gate", "<u2"), ("short_gate", "<u2")]
)


@dataclass(frozen=True)
class PsdEventFileInfo:
    path: Path
    format_name: str
    channel: int | None
    total_events: int


def _channel_from_configuration(text: str) -> int | None:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        log.warning("Ignoring invalid embedded capture configuration", exc_info=True)
        return None
    if not isinstance(document, Mapping):
        return None
    measurement = document.get("measurement")
    if not isinstance(measurement, Mapping) or "channel" not in measurement:
        return None
    try:
        return int(measurement["channel"])
    except (TypeError, ValueError):
        return None


def _hdf5_configuration(file: h5py.File) -> str:
    if "configuration_yaml" not in file:
        return ""
    value = file["configuration_yaml"][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def inspect_psd_event_file(path: Path) -> PsdEventFileInfo:
    """Read only lightweight metadata needed to select the target PSD view."""
    suffix = path.suffix.lower()
    if suffix == ".bin":
        with path.open("rb") as stream:
            header = read_file_header(stream)
        if header["frame_samples"]:
            raise ValueError("The selected NDMA file contains scope frames, not MCA events")
        if header["version"] == FILE_VERSION:
            unit_bytes = _EVENT_DTYPE.itemsize
            format_name = "NDMA legacy list-mode"
        elif header["version"] == IIO_LM_FILE_VERSION:
            unit_bytes = IIO_LM_FRAME_BYTES
            format_name = "NDMA IIO list-mode"
        else:
            raise ValueError(f"Unsupported NDMA list-mode version {header['version']}")
        payload_bytes = max(0, path.stat().st_size - FILE_HEADER_STRUCT.size)
        complete_units = payload_bytes // unit_bytes
        total_events = (
            complete_units * IIO_LM_FRAME_RECORDS
            if header["version"] == IIO_LM_FILE_VERSION
            else complete_units
        )
        return PsdEventFileInfo(path, format_name, header["channel"], total_events)

    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r", swmr=True) as file:
            if "events" not in file or not isinstance(file["events"], h5py.Dataset):
                raise ValueError("HDF5 file has no /events dataset")
            events = file["events"]
            _validate_event_fields(events.dtype.names)
            committed = (
                int(file["committed_events"][()])
                if "committed_events" in file
                else len(events)
            )
            total_events = min(len(events), max(0, committed))
            channel_value = file.attrs.get("channel")
            channel = int(channel_value) if channel_value is not None else None
            if channel is None:
                channel = _channel_from_configuration(_hdf5_configuration(file))
        return PsdEventFileInfo(path, "HDF5 list-mode", channel, total_events)

    if suffix == ".root":
        import uproot

        with uproot.open(path) as file:
            if "events" not in file:
                raise ValueError("ROOT file has no events TTree")
            tree = file["events"]
            fields = set(tree.keys())
            _validate_event_fields(tuple(fields))
            configuration = str(file["configuration_yaml"]) if "configuration_yaml" in file else ""
            channel = _channel_from_configuration(configuration)
            total_events = int(tree.num_entries)
        return PsdEventFileInfo(path, "ROOT TTree", channel, total_events)

    raise ValueError("Unsupported PSD event file; select .bin, .h5, .hdf5, or .root")


def _validate_event_fields(names: tuple[str, ...] | None) -> None:
    fields = set(names or ())
    supported = (
        {"energy", "short_energy"},
        {"trapezoid_energy", "charge_energy"},
        {"long_gate", "short_gate"},
    )
    if "timestamp" not in fields or not any(required.issubset(fields) for required in supported):
        raise ValueError(
            "Event data must contain timestamp and a supported long/short energy pair"
        )


def _iter_ndma(path: Path) -> Iterator[np.ndarray]:
    with path.open("rb") as stream:
        header = read_file_header(stream)
        if header["frame_samples"]:
            raise ValueError("The selected NDMA file contains scope frames, not MCA events")
        if header["version"] == FILE_VERSION:
            dtype = _EVENT_DTYPE
            alignment = dtype.itemsize
        elif header["version"] == IIO_LM_FILE_VERSION:
            dtype = _LM_EVENT_DTYPE
            alignment = IIO_LM_FRAME_BYTES
        else:
            raise ValueError(f"Unsupported NDMA list-mode version {header['version']}")

        payload_bytes = max(0, path.stat().st_size - FILE_HEADER_STRUCT.size)
        usable_bytes = payload_bytes - payload_bytes % alignment
        if usable_bytes != payload_bytes:
            log.warning(
                "PSD import ignores %d trailing bytes from incomplete NDMA data",
                payload_bytes - usable_bytes,
            )
        total_records = usable_bytes // dtype.itemsize
        batch_records = max(1, _TARGET_BATCH_BYTES // dtype.itemsize)
        if header["version"] == IIO_LM_FILE_VERSION:
            batch_records = max(
                IIO_LM_FRAME_RECORDS,
                batch_records - batch_records % IIO_LM_FRAME_RECORDS,
            )

        mapped = mmap.mmap(stream.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            for first in range(0, total_records, batch_records):
                count = min(batch_records, total_records - first)
                # Copy only this bounded batch so the yielded array does not
                # keep a Windows mmap export alive when the generator closes.
                yield np.frombuffer(
                    mapped,
                    dtype=dtype,
                    count=count,
                    offset=FILE_HEADER_STRUCT.size + first * dtype.itemsize,
                ).copy()
        finally:
            mapped.close()


def _iter_hdf5(path: Path) -> Iterator[np.ndarray]:
    with h5py.File(path, "r", swmr=True) as file:
        events = file["events"]
        assert isinstance(events, h5py.Dataset)
        _validate_event_fields(events.dtype.names)
        committed = (
            int(file["committed_events"][()])
            if "committed_events" in file
            else len(events)
        )
        total = min(len(events), max(0, committed))
        batch_records = max(1, _TARGET_BATCH_BYTES // events.dtype.itemsize)
        for first in range(0, total, batch_records):
            yield np.asarray(events[first : min(total, first + batch_records)])


def _root_batch(arrays: Any, long_name: str, short_name: str) -> np.ndarray:
    if not isinstance(arrays, Mapping) and not (
        isinstance(arrays, np.ndarray) and arrays.dtype.names is not None
    ):
        raise ValueError(f"Unexpected ROOT event batch type {type(arrays).__name__}")
    batch = np.empty(len(arrays["timestamp"]), dtype=_CANONICAL_DTYPE)
    batch["timestamp"] = arrays["timestamp"]
    batch["long_gate"] = arrays[long_name]
    batch["short_gate"] = arrays[short_name]
    return batch


def _iter_root(path: Path) -> Iterator[np.ndarray]:
    import uproot

    with uproot.open(path) as file:
        tree = file["events"]
        fields = set(tree.keys())
        _validate_event_fields(tuple(fields))
        if {"long_gate", "short_gate"}.issubset(fields):
            long_name, short_name = "long_gate", "short_gate"
        elif {"trapezoid_energy", "charge_energy"}.issubset(fields):
            long_name, short_name = "trapezoid_energy", "charge_energy"
        else:
            long_name, short_name = "energy", "short_energy"
        for arrays in tree.iterate(
            expressions=["timestamp", long_name, short_name],
            step_size=f"{max(1, _TARGET_BATCH_BYTES // (1024 * 1024))} MB",
            library="np",
            how=dict,
        ):
            yield _root_batch(arrays, long_name, short_name)


def iter_psd_event_batches(path: Path) -> Iterator[np.ndarray]:
    """Yield bounded event batches without retaining prior batches in memory."""
    suffix = path.suffix.lower()
    if suffix == ".bin":
        yield from _iter_ndma(path)
    elif suffix in {".h5", ".hdf5"}:
        yield from _iter_hdf5(path)
    elif suffix == ".root":
        yield from _iter_root(path)
    else:
        raise ValueError("Unsupported PSD event file; select .bin, .h5, .hdf5, or .root")
