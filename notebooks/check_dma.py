"""Inspect timestamps and frame integrity in an nLab scope DMA file.

Edit ``FILENAME`` below, then run from the repository root with:

    python notebooks/check_dma.py
"""

from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nlab.utils.dma_converter import read_file_header

# Change this path to inspect another scope DMA capture.
FILENAME = Path(r"D:\work\measurements\10_gap.bin")

# The scope timestamp is the raw 125 MHz datapath-clock counter: 8 ns/tick.
TIMESTAMP_TICK_NS = 8
SHOW_PLOTS = True
PREVIEW_FRAMES = 10


def _preview_indices(indices: np.ndarray, limit: int = 20) -> str:
    values = indices[:limit].tolist()
    suffix = " ..." if indices.size > limit else ""
    return f"{values}{suffix}"


def _print_frame_preview(
    timestamps: np.ndarray,
    waveforms: np.ndarray,
    count: int,
) -> None:
    print(f"\nFirst {min(count, len(timestamps))} frame(s):")
    for index in range(min(count, len(timestamps))):
        waveform = waveforms[index]
        print(
            f"  {index:6d}: timestamp={int(timestamps[index]):20d}, "
            f"min={int(waveform.min()):6d}, max={int(waveform.max()):6d}, "
            f"nonzero={np.count_nonzero(waveform):4d}, "
            f"mean={waveform.mean():10.3f}"
        )


def inspect_scope_dma(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        header = read_file_header(handle)
        header_bytes = handle.tell()

    frame_samples = header["frame_samples"]
    if frame_samples <= 4:
        raise ValueError(
            "This is not a valid scope capture: "
            f"frame_samples={frame_samples}; expected more than 4"
        )

    frame_bytes = frame_samples * np.dtype("<i2").itemsize
    waveform_samples = frame_samples - 4
    payload_bytes = file_bytes - header_bytes
    complete_frames, trailing_bytes = divmod(payload_bytes, frame_bytes)

    print(f"File:                  {path}")
    print(f"File size:             {file_bytes:,} bytes")
    print(f"Header:                {header}")
    print(f"Header size:           {header_bytes:,} bytes")
    print(f"Payload size:          {payload_bytes:,} bytes")
    print(f"Frame size:            {frame_bytes:,} bytes")
    print(f"Waveform samples:      {waveform_samples:,} per frame")
    print(f"Complete frames:       {complete_frames:,}")
    print(f"Trailing bytes:        {trailing_bytes:,}")

    if trailing_bytes:
        raise ValueError(
            f"Truncated final frame: payload has {trailing_bytes} trailing byte(s)"
        )
    if complete_frames == 0:
        print("No frames to analyze.")
        return

    frame_dtype = np.dtype(
        [
            ("timestamp", "<u8"),
            ("samples", "<i2", (waveform_samples,)),
        ]
    )
    frames = np.memmap(
        path,
        dtype=frame_dtype,
        mode="r",
        offset=header_bytes,
        shape=(complete_frames,),
    )
    timestamps_u64 = frames["timestamp"]
    if timestamps_u64.max() > np.iinfo(np.int64).max:
        raise ValueError("A timestamp exceeds the signed int64 range")
    timestamps = timestamps_u64.astype(np.int64)
    waveforms = frames["samples"]

    # A zero timestamp or all-zero waveform is an invalid-frame signature.
    zero_timestamp = timestamps == 0
    all_zero_waveform = np.all(waveforms == 0, axis=1)
    invalid = zero_timestamp | all_zero_waveform
    valid = ~invalid

    raw_deltas = np.diff(timestamps)
    nonpositive_delta_indices = np.flatnonzero(raw_deltas <= 0)
    zero_timestamp_indices = np.flatnonzero(zero_timestamp)
    all_zero_indices = np.flatnonzero(all_zero_waveform)

    _print_frame_preview(timestamps, waveforms, PREVIEW_FRAMES)
    print(f"\nLast {min(PREVIEW_FRAMES, len(timestamps))} timestamp(s):")
    print(" ", timestamps[-PREVIEW_FRAMES:].tolist())
    print(f"First {min(20, len(raw_deltas))} raw delta(s):")
    print(" ", raw_deltas[:20].tolist())

    print("\nIntegrity checks:")
    print(f"  Zero timestamp frames:       {zero_timestamp_indices.size:,}")
    print(f"  Their indices:               {_preview_indices(zero_timestamp_indices)}")
    print(f"  All-zero waveform frames:    {all_zero_indices.size:,}")
    print(f"  Their indices:               {_preview_indices(all_zero_indices)}")
    print(f"  Invalid frames (combined):   {np.count_nonzero(invalid):,}")
    print(f"  Nonpositive raw deltas:      {nonpositive_delta_indices.size:,}")
    print(f"  Delta start-frame indices:   {_preview_indices(nonpositive_delta_indices)}")

    valid_timestamps = timestamps[valid]
    if valid_timestamps.size < 2:
        print("Fewer than two valid timestamps; no timing statistics available.")
        return

    valid_deltas = np.diff(valid_timestamps)
    positive_deltas = valid_deltas[valid_deltas > 0]
    timestamp_span = int(valid_timestamps[-1] - valid_timestamps[0])
    span_seconds_8ns = timestamp_span * TIMESTAMP_TICK_NS * 1e-9
    span_seconds_1ps = timestamp_span * 1e-12
    header_time = datetime.fromtimestamp(header["timestamp"], UTC)
    mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    wall_duration = path.stat().st_mtime - header["timestamp"]

    print("\nTimestamp-unit check:")
    print(f"  Header time (UTC):           {header_time.isoformat()}")
    print(f"  File mtime (UTC):            {mtime.isoformat()}")
    print(f"  Header-to-mtime duration:    {wall_duration:.9f} s")
    print(f"  Raw timestamp span:          {timestamp_span:,} ticks")
    print(f"  Span interpreted as 8 ns:   {span_seconds_8ns:.9f} s")
    print(f"  Span interpreted as 1 ps:   {span_seconds_1ps:.9f} s")
    print(
        "  8 ns vs wall-time error:    "
        f"{abs(wall_duration - span_seconds_8ns):.9f} s"
    )

    print("\nValid-frame timing statistics:")
    print(f"  Valid frames:                {valid_timestamps.size:,}")
    print(f"  Strictly increasing:         {bool(np.all(valid_deltas > 0))}")
    if span_seconds_8ns > 0:
        rate = (valid_timestamps.size - 1) / span_seconds_8ns
        print(f"  Overall frame rate:          {rate:.6f} frames/s")

    if positive_deltas.size:
        median_ticks = float(np.median(positive_deltas))
        median_microseconds = median_ticks * TIMESTAMP_TICK_NS / 1_000
        print(f"  Median frame interval:       {median_ticks:.3f} ticks")
        print(f"                               {median_microseconds:.6f} us")
        print(f"                               {median_microseconds / 1_000:.9f} ms")
        quantiles = (0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1)
        print("  Positive delta quantiles:")
        print("    quantile       ticks              microseconds")
        for quantile in quantiles:
            ticks = float(np.quantile(positive_deltas, quantile))
            microseconds = ticks * TIMESTAMP_TICK_NS / 1_000
            print(f"    {quantile:8.2f}  {ticks:14.3f}  {microseconds:24.6f}")

    if SHOW_PLOTS:
        valid_indices = np.flatnonzero(valid)
        relative_seconds = (
            valid_timestamps - valid_timestamps[0]
        ) * TIMESTAMP_TICK_NS * 1e-9

        _, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
        axes[0].plot(valid_indices, relative_seconds, linewidth=1)
        if np.any(invalid):
            axes[0].scatter(
                np.flatnonzero(invalid),
                np.zeros(np.count_nonzero(invalid)),
                color="red",
                marker="x",
                label="invalid frame",
            )
            axes[0].legend()
        axes[0].set_title("Scope DMA timestamps")
        axes[0].set_xlabel("Frame index")
        axes[0].set_ylabel("Time from first valid frame (s)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            valid_indices[1:],
            valid_deltas * TIMESTAMP_TICK_NS / 1_000_000,
            linewidth=0.7,
        )
        axes[1].set_title("Interval between valid frames")
        axes[1].set_xlabel("Frame index")
        axes[1].set_ylabel("Interval (ms)")
        axes[1].grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    inspect_scope_dma(FILENAME)
