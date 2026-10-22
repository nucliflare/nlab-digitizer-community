"""Lossless waveform subsets with exact source-frame and PSD-cut provenance."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from nlab.analysis.waveform_file import MappedWaveformFile, WaveformFileIndex, WaveformFrame
from nlab.hardware.digitizer.dma import (
    FILE_HEADER_STRUCT,
    FILE_MAGIC,
    FILE_VERSION,
    SCOPE_TIMESTAMP_WORDS,
)

log = logging.getLogger(__name__)
ClusterFormat = Literal["bin", "root", "h5"]
_BATCH_BYTES = 4 * 1024 * 1024


class ClusterExportCancelledError(Exception):
    """Cancellation removes only files created by this export."""


@dataclass(frozen=True)
class ClusterExportResult:
    destination: Path
    count: int
    sidecar: Path | None = None


def export_waveform_cluster(
    index: WaveformFileIndex, destination: Path, *, file_format: ClusterFormat,
    source_index: int, frames: np.ndarray, energy: np.ndarray, ratio: np.ndarray,
    metadata: dict[str, Any], progress: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ClusterExportResult:
    """Export full stored waveforms, never processed/cropped/averaged samples.

    Native NDMA subsets retain the original header and exact complete frames.
    CAEN records use variable-length ROOT/HDF5 samples with unconverted native
    timestamps: their ABI cannot be represented losslessly as Scope NDMA v1.
    All outputs are exclusive-create; existing files and the source are untouched.
    Sample buffers are bounded independently of the selected cluster size.
    """
    destination = Path(destination)
    if file_format not in {"bin", "root", "h5"}:
        raise ValueError("Choose nlab binary, ROOT, or HDF5")
    if not 0 <= source_index < len(index.channels):
        raise ValueError("Invalid waveform source")
    source = index.channels[source_index]
    if (frames.ndim != 1 or not len(frames) or frames.dtype.kind not in "iu"
            or np.any(frames < 0) or np.any(frames >= source.frame_count)
            or np.any(frames[1:] <= frames[:-1])):
        raise ValueError("Selected source frames must be nonempty, sorted, unique, and in range")
    if (energy.shape != frames.shape or ratio.shape != frames.shape
            or not np.all(np.isfinite(energy)) or not np.all(np.isfinite(ratio))):
        raise ValueError("Selected PSD coordinates must be finite and match the source frames")
    if destination.resolve() == index.path.resolve():
        raise ValueError("Source and destination must be different files")
    if destination.exists():
        raise FileExistsError("Destination exists; choose a new filename (exports never overwrite)")
    native = index.caen_info is None
    if file_format == "bin" and not native:
        raise ValueError("nlab binary export requires an NDMA source. Use ROOT or HDF5 for CAEN "
                         "to preserve variable record lengths and native timestamp units.")
    sidecar = destination.with_suffix(".cluster.json") if file_format == "bin" else None
    if sidecar is not None and sidecar.exists():
        raise FileExistsError(f"Cluster metadata already exists: {sidecar}")
    initial = index.path.stat()
    header = b""
    samples_per_frame = index.ndma_frame_samples - SCOPE_TIMESTAMP_WORDS if native else 0
    if native:
        with index.path.open("rb") as stream:
            header = stream.read(FILE_HEADER_STRUCT.size)
        magic, version, _channel, _reserved, _time, frame_samples = (
            FILE_HEADER_STRUCT.unpack(header)
        )
        if (magic != FILE_MAGIC or version != FILE_VERSION
                or frame_samples != index.ndma_frame_samples or samples_per_frame <= 0):
            raise ValueError("Source is not a supported Scope NDMA v1 capture")
    details = {
        **metadata,
        "format": "nlab-waveform-cluster-v1",
        "source_file": str(index.path.resolve()), "source_format": index.format_name,
        "source_size": initial.st_size, "source_mtime_ns": initial.st_mtime_ns,
        "source_index": source_index, "board": source.board, "channel": source.channel,
        "total_frames": len(frames), "samples_per_frame": samples_per_frame if native else None,
        "timestamp_unit": "8 ns ticks" if native else "source-native CAEN ticks (unconverted)",
        "sample_unit": "ADC counts (int16)",
        "samples": "full original stored waveforms; no baseline, polarity, or averaging applied",
    }
    configuration = index.path.with_suffix(".yaml")
    if configuration.is_file():
        details["configuration_yaml"] = configuration.read_text(encoding="utf-8")

    def check() -> None:
        if cancelled is not None and cancelled():
            raise ClusterExportCancelledError()

    created: list[Path] = []
    count = len(frames)
    columns: dict[str, np.dtype[Any]] = {
        "timestamp": np.dtype("uint64"), "source_frame": np.dtype("uint64"),
        "energy": np.dtype("float64"), "psd": np.dtype("float64"),
    }
    if not native:
        columns.update(stored_long_gate=np.dtype("uint16"), stored_short_gate=np.dtype("uint16"),
                       has_stored_gates=np.dtype("bool"), flags=np.dtype("uint32"),
                       has_flags=np.dtype("bool"), waveform_code=np.dtype("uint8"),
                       has_waveform_code=np.dtype("bool"))
    try:
        check()
        if progress is not None:
            progress(0)
        with ExitStack() as stack:
            output = stack.enter_context(destination.open("x+b"))
            created.append(destination)
            reader = MappedWaveformFile(index)
            stack.callback(reader.close)
            tree: Any = None
            datasets: dict[str, Any] = {}
            if file_format == "bin":
                output.write(header)
            elif file_format == "root":
                import uproot

                root = stack.enter_context(uproot.recreate(output))
                root["metadata_json"] = json.dumps(details)
                schema: dict[str, Any] = dict(columns)
                schema["samples"] = (np.dtype(("int16", (samples_per_frame,)))
                                     if native else "var * int16")
                root["waveform_schema"] = json.dumps({name: str(dtype)
                                                     for name, dtype in schema.items()})
                tree = root.mktree("waveforms", schema, title="NLab selected waveform cluster")
            else:
                import h5py

                h5 = stack.enter_context(h5py.File(output, "w"))
                h5.attrs["metadata_json"] = json.dumps(details)
                for name, dtype in columns.items():
                    datasets[name] = h5.create_dataset(
                        "timestamps" if name == "timestamp" else name, (count,), dtype=dtype,
                        compression="gzip", compression_opts=4,
                    )
                datasets["samples"] = h5.create_dataset(
                    "waveforms", (count, samples_per_frame) if native else (count,),
                    dtype="int16" if native else h5py.vlen_dtype(np.dtype("int16")),
                    **({"compression": "gzip", "compression_opts": 4} if native else {}),
                )
                datasets["samples"].attrs["unit"] = details["sample_unit"]
                datasets["timestamp"].attrs["unit"] = details["timestamp_unit"]

            batch: list[WaveformFrame] = []
            batch_bytes = 0
            written = 0

            def flush() -> None:
                nonlocal written, batch_bytes
                if not batch:
                    return
                check()
                stop = written + len(batch)
                data: dict[str, Any] = {
                    "timestamp": np.asarray([event.timestamp for event in batch], dtype="uint64"),
                    "source_frame": frames[written:stop].astype("uint64"),
                    "energy": energy[written:stop].astype(float),
                    "psd": ratio[written:stop].astype(float),
                }
                if not native:
                    for name in columns.keys() - data.keys():
                        values = []
                        for event in batch:
                            value: Any
                            if name == "has_stored_gates":
                                value = event.long_gate is not None and event.short_gate is not None
                            elif name.startswith("has_"):
                                value = getattr(event, name[4:]) is not None
                            else:
                                attribute = {"stored_long_gate": "long_gate",
                                             "stored_short_gate": "short_gate"}.get(name, name)
                                value = getattr(event, attribute) or 0
                            values.append(value)
                        data[name] = np.asarray(values, dtype=columns[name])
                if native:
                    data["samples"] = np.stack([event.samples for event in batch])
                elif file_format == "root":
                    import awkward as ak

                    data["samples"] = ak.unflatten(
                        np.concatenate([event.samples for event in batch]),
                        [len(event.samples) for event in batch],
                    )
                if file_format == "root":
                    tree.extend(data)
                elif file_format == "h5":
                    for name in columns:
                        datasets[name][written:stop] = data[name]
                    if native:
                        datasets["samples"][written:stop] = data["samples"]
                    else:
                        for offset, event in enumerate(batch, written):
                            datasets["samples"][offset] = event.samples
                written = stop
                batch.clear()
                batch_bytes = 0
                if progress is not None:
                    progress(written * 99 // count)

            for frame_index in frames:
                check()
                event = reader.frame(int(frame_index), source_index)
                if native and len(event.samples) != samples_per_frame:
                    raise ValueError("Source waveform geometry changed during export")
                if file_format == "bin":
                    output.write(int(event.timestamp).to_bytes(8, "little"))
                    output.write(event.samples.astype("<i2", copy=False).tobytes())
                    written += 1
                    if progress is not None and written % 512 == 0:
                        progress(written * 99 // count)
                else:
                    batch.append(event)
                    batch_bytes += event.samples.nbytes
                    if batch_bytes >= _BATCH_BYTES or len(batch) >= 4096:
                        flush()
            flush()
            check()
            final = index.path.stat()
            if final.st_size != initial.st_size or final.st_mtime_ns != initial.st_mtime_ns:
                raise ValueError("Source changed during export; stop recording first")
            if sidecar is not None:
                check()
                with sidecar.open("x", encoding="utf-8") as metadata_stream:
                    created.append(sidecar)
                    json.dump({**details, "source_frame": frames.tolist(),
                               "energy": energy.tolist(), "psd": ratio.tolist()}, metadata_stream)
            check()
        if progress is not None:
            progress(100)
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                log.exception("Could not remove incomplete cluster export: %s", path)
        raise
    return ClusterExportResult(destination, count, sidecar)
