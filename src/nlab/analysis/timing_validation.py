"""Bounded-memory, offline check of two native MCA timestamp streams.

This diagnoses recorded time ordering and a candidate split-pulse delay. It
does not certify that two hardware channels share an epoch: the IIO list-mode
driver transports opaque records, so common-clock wiring and event timestamp
semantics must also be established by a known simultaneous input pulse.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import yaml

from nlab.analysis.psd_file import inspect_psd_event_file, iter_psd_event_batches
from nlab.hardware.digitizer.dma import IIO_LM_FILE_VERSION
from nlab.utils.dma_converter import read_file_header

_TICK_NS = 8  # NLab IIO list-mode client schema and HDF5/ROOT writer metadata.
_MAX_SAMPLE_EVENTS = 250_000
_MAX_SIGNED_TIMESTAMP = np.iinfo(np.int64).max


class TimingValidationCancelledError(Exception):
    """The user stopped the offline scan before it completed."""


@dataclass(frozen=True)
class TimingStreamScan:
    path: Path
    channel: int
    records: int
    usable_events: int
    zero_timestamps: int
    reversals: int
    first_tick: int
    last_tick: int


@dataclass(frozen=True)
class TimingValidationResult:
    channel_a: TimingStreamScan
    channel_b: TimingStreamScan
    applied_offset_ns: int
    search_window_ns: int
    overlap_ns: int
    sampled_a: int
    sampled_b: int
    matched_a: int
    peak_delay_ns: float | None
    peak_pairs: int
    same_device_metadata: bool | None
    bin_centers_ns: np.ndarray
    bin_counts: np.ndarray


def _require_native_file(path: Path) -> int:
    info = inspect_psd_event_file(path)
    if info.channel is None:
        raise ValueError(f"{path.name}: channel identity is missing")
    suffix = path.suffix.lower()
    if suffix == ".bin":
        if info.format_name != "NDMA IIO list-mode":
            raise ValueError(f"{path.name}: timing validation requires native IIO list-mode")
        with path.open("rb") as stream:
            header = read_file_header(stream)
        if header["version"] != IIO_LM_FILE_VERSION or header["frame_samples"]:
            raise ValueError(f"{path.name}: timing validation requires native IIO list-mode")
    elif suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r", swmr=True) as capture:
            events = capture["events"]
            if (
                capture.attrs.get("format") != "nlab-mca-listmode"
                or not isinstance(events, h5py.Dataset)
                or events.attrs.get("timestamp_unit") != "8 ns ticks"
            ):
                raise ValueError(f"{path.name}: native 8 ns timestamp metadata is missing")
    elif suffix == ".root":
        import uproot

        with uproot.open(path) as capture:
            if "event_schema" not in capture or "8 ns ticks" not in str(capture["event_schema"]):
                raise ValueError(f"{path.name}: native 8 ns timestamp metadata is missing")
    else:
        raise ValueError(f"{path.name}: select a native NDMA, HDF5, or ROOT event file")
    return info.channel


def _capture_device(path: Path) -> tuple[str, str] | None:
    """Read the saved connection identity, never infer it from a filename."""
    suffix = path.suffix.lower()
    if suffix == ".bin":
        config_path = path.with_suffix(".yaml")
        if not config_path.is_file():
            return None
        content = config_path.read_text(encoding="utf-8")
    elif suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r", swmr=True) as capture:
            if "configuration_yaml" not in capture:
                return None
            value = capture["configuration_yaml"][()]
            content = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    elif suffix == ".root":
        import uproot

        with uproot.open(path) as capture:
            if "configuration_yaml" not in capture:
                return None
            content = str(capture["configuration_yaml"])
    else:
        return None
    try:
        document = yaml.safe_load(content)
    except yaml.YAMLError:
        return None
    if not isinstance(document, Mapping):
        return None
    connection = document.get("connection")
    if not isinstance(connection, Mapping):
        return None
    backend, host = connection.get("backend"), connection.get("ip")
    if not isinstance(backend, str) or not isinstance(host, str) or not host:
        return None
    return backend, host


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise TimingValidationCancelledError


def _scan_stream(
    path: Path,
    channel: int,
    *,
    cancelled: Callable[[], bool] | None,
    progress: Callable[[str], None] | None,
) -> TimingStreamScan:
    records = usable = zeros = reversals = 0
    last_reported = 0
    first = last = previous = None
    for events in iter_psd_event_batches(path):
        _check_cancelled(cancelled)
        ticks = events["timestamp"]
        records += len(ticks)
        if progress is not None and records - last_reported >= 4_000_000:
            progress(f"Checking channel {channel}: {records:,} records scanned...")
            last_reported = records
        zeros += int(np.count_nonzero(ticks == 0))
        # v121's final frame is zero-padded but carries no valid-record
        # count. Zero-valued real events are possible, so report these as
        # ambiguous exclusions, never as a proven count of padding/loss.
        valid = ticks[ticks != 0]
        usable += len(valid)
        if not len(valid):
            continue
        if int(np.max(valid)) > _MAX_SIGNED_TIMESTAMP:
            raise ValueError(f"{path.name}: timestamp exceeds supported signed range")
        if previous is not None and int(valid[0]) < previous:
            reversals += 1
        reversals += int(np.count_nonzero(valid[1:] < valid[:-1]))
        if first is None:
            first = int(valid[0])
        last = previous = int(valid[-1])
    if first is None or last is None:
        raise ValueError(f"{path.name}: no usable nonzero timestamps")
    return TimingStreamScan(path, channel, records, usable, zeros, reversals, first, last)


def _sample_overlap(
    path: Path,
    *,
    lower: int,
    upper: int,
    shift_ticks: int,
    cancelled: Callable[[], bool] | None,
) -> np.ndarray:
    collected: list[np.ndarray] = []
    remaining = _MAX_SAMPLE_EVENTS
    for events in iter_psd_event_batches(path):
        _check_cancelled(cancelled)
        ticks = events["timestamp"]
        # The first pass has already checked the stream's order. Skip entire
        # batches outside the shared time interval without allocating them.
        if len(ticks) and int(ticks[0]) + shift_ticks > upper and ticks[0] != 0:
            break
        valid = ticks[ticks != 0].astype(np.int64, copy=False) + shift_ticks
        chosen = valid[(valid >= lower) & (valid <= upper)]
        if len(chosen):
            taken = chosen[:remaining]
            collected.append(taken.copy())
            remaining -= len(taken)
        if remaining == 0:
            break
    return np.concatenate(collected) if collected else np.empty(0, dtype=np.int64)


def validate_two_channel_timing(
    path_a: Path,
    path_b: Path,
    *,
    offset_ns: int = 0,
    search_window_ns: int = 800,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> TimingValidationResult:
    """Inspect both full streams, then histogram bounded overlap samples.

    The offset is applied to B. Nearest-neighbor pairs are diagnostic only:
    they may reuse a B event and must not be reported as coincidence counts.
    """
    if path_a.resolve() == path_b.resolve():
        raise ValueError("Choose two different channel files")
    if offset_ns % _TICK_NS or search_window_ns % _TICK_NS or search_window_ns < _TICK_NS:
        raise ValueError("Offset and search window must use 8 ns increments")
    channel_a = _require_native_file(path_a)
    channel_b = _require_native_file(path_b)
    if channel_a == channel_b:
        raise ValueError("Select files from two different channel indices")
    if channel_a > channel_b:
        raise ValueError("Put the lower-index channel in A and higher-index channel in B")
    device_a, device_b = _capture_device(path_a), _capture_device(path_b)
    for path, device in ((path_a, device_a), (path_b, device_b)):
        # NDMA v2 identifies the IIO source in its header. Structured files
        # have no versioned transport header, so require their saved IIO
        # connection identity before interpreting 8 ns event timestamps.
        if (path.suffix.lower() != ".bin" and device is None) or (
            device is not None and device[0] != "iio"
        ):
            raise ValueError(f"{path.name}: an IIO capture configuration is required")
    if device_a is not None and device_b is not None and device_a != device_b:
        raise ValueError("Files identify different digitizer endpoints; do not correlate them")
    same_device_metadata = True if device_a is not None and device_b is not None else None
    if progress is not None:
        progress(f"Checking channel {channel_a} timestamp order...")
    scan_a = _scan_stream(path_a, channel_a, cancelled=cancelled, progress=progress)
    if progress is not None:
        progress(f"Checking channel {channel_b} timestamp order...")
    scan_b = _scan_stream(path_b, channel_b, cancelled=cancelled, progress=progress)
    if scan_a.reversals or scan_b.reversals:
        raise ValueError(
            f"Timestamps go backwards: channel {channel_a}: {scan_a.reversals}, "
            f"channel {channel_b}: {scan_b.reversals}. Do not correlate these files."
        )
    shift = offset_ns // _TICK_NS
    lower = max(scan_a.first_tick, scan_b.first_tick + shift)
    upper = min(scan_a.last_tick, scan_b.last_tick + shift)
    if lower >= upper:
        raise ValueError("The channel timestamp ranges do not overlap after the offset")
    if progress is not None:
        progress("Sampling overlapping events and finding candidate delays...")
    sample_a = _sample_overlap(path_a, lower=lower, upper=upper, shift_ticks=0, cancelled=cancelled)
    sample_b = _sample_overlap(
        path_b, lower=lower, upper=upper, shift_ticks=shift, cancelled=cancelled
    )
    if not len(sample_a) or not len(sample_b):
        raise ValueError("No usable events occur in the overlapping time range")
    # Each channel may have a different rate. Restrict the comparison to the
    # time interval covered by both bounded samples, not just the full files.
    sample_end = min(int(sample_a[-1]), int(sample_b[-1]))
    sample_a = sample_a[sample_a <= sample_end]
    sample_b = sample_b[sample_b <= sample_end]
    if not len(sample_a) or not len(sample_b):
        raise ValueError("No sampled events share an overlapping time range")
    indices = np.searchsorted(sample_b, sample_a, side="left")
    before = np.clip(indices - 1, 0, len(sample_b) - 1)
    after = np.clip(indices, 0, len(sample_b) - 1)
    delta_before = sample_b[before] - sample_a
    delta_after = sample_b[after] - sample_a
    nearest = np.where(np.abs(delta_before) <= np.abs(delta_after), delta_before, delta_after)
    within = nearest[np.abs(nearest) <= search_window_ns // _TICK_NS] * _TICK_NS
    counts, edges = np.histogram(
        within,
        bins=min(2001, 2 * (search_window_ns // _TICK_NS) + 1),
        range=(-search_window_ns - 0.5, search_window_ns + 0.5),
    )
    centers = (edges[:-1] + edges[1:]) / 2
    peak_index = int(np.argmax(counts)) if len(within) else 0
    return TimingValidationResult(
        channel_a=scan_a,
        channel_b=scan_b,
        applied_offset_ns=offset_ns,
        search_window_ns=search_window_ns,
        overlap_ns=(upper - lower) * _TICK_NS,
        sampled_a=len(sample_a),
        sampled_b=len(sample_b),
        matched_a=len(within),
        peak_delay_ns=float(centers[peak_index]) if len(within) else None,
        peak_pairs=int(counts[peak_index]),
        same_device_metadata=same_device_metadata,
        bin_centers_ns=centers,
        bin_counts=counts,
    )
