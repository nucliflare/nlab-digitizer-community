#!/usr/bin/env python
"""Profile bounded Current DMA rendering with synthetic receiver summaries.

This is a GUI-only diagnostic: it performs no hardware or network I/O. The
synthetic accumulator is filled before timing begins, then the production
Current widget repeatedly renders its immutable snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import struct
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from nlab.controllers.current_monitor_controller import (  # noqa: E402
    CurrentMonitorController,
)
from nlab.hardware.digitizer.current_monitor import (  # noqa: E402
    ScopeCurrentAccumulator,
    ScopeCurrentRuntime,
)
from nlab.hardware.digitizer.dma import ScopeDmaGeometry  # noqa: E402


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(np.ceil(fraction * len(ordered))) - 1)]


def profile(*, iterations: int, warmup: int) -> dict[str, object]:
    app = QApplication.instance() or QApplication([])
    geometry = ScopeDmaGeometry(
        frame_samples=8000,
        buffer_samples=2004,
        frame_bytes=4008,
        waveform_samples=1999,
        sample_decimation=4,
        padding_bytes=2,
    )
    accumulator = ScopeCurrentAccumulator(bin_width_ms=100, history_seconds=10)
    expected_ticks = 20_000  # 160 us / 6,250 synthetic frames/s
    accumulator.start_session(
        geometry,
        ScopeCurrentRuntime(0, 122, 18_000, expected_ticks),
    )
    waveform = np.arange(geometry.waveform_samples, dtype="<i2")
    payload = waveform.tobytes() + bytes(geometry.padding_bytes)
    frames = 62_500  # ten seconds of receiver-owned summaries
    for index in range(frames):
        timestamp = 1 + index * expected_ticks
        accumulator.append_frame(
            struct.pack("<Q", timestamp) + payload,
            geometry,
            1_000_000 + index * 160_000,
        )
    snapshot = accumulator.snapshot(now_ns=10_001_000_000)

    controller = CurrentMonitorController(
        object(),  # type: ignore[arg-type]
        channel=0,
        auto_start=False,
        scope_current_accumulator=accumulator,
    )
    controller.ui.comboMode.setCurrentIndex(1)

    for _ in range(warmup):
        controller._render_dma_snapshot(snapshot)
        app.processEvents()

    prepare_ms: list[float] = []
    complete_ms: list[float] = []
    for _ in range(iterations):
        started_ns = time.perf_counter_ns()
        controller._render_dma_snapshot(snapshot)
        rendered_ns = time.perf_counter_ns()
        app.processEvents()
        completed_ns = time.perf_counter_ns()
        prepare_ms.append((rendered_ns - started_ns) / 1_000_000)
        complete_ms.append((completed_ns - started_ns) / 1_000_000)

    curve_x, _curve_y = controller._curve.getData()
    controller.close()
    return {
        "synthetic_frames": frames,
        "scientific_bins": len(snapshot.bins),
        "plot_points": 0 if curve_x is None else len(curve_x),
        "iterations": iterations,
        "requested_display_fps": 30,
        "display_budget_ms": 1000 / 30,
        "render_median_ms": statistics.median(prepare_ms),
        "render_p95_ms": _percentile(prepare_ms, 0.95),
        "render_and_events_median_ms": statistics.median(complete_ms),
        "render_and_events_p95_ms": _percentile(complete_ms, 0.95),
        "received_frames": snapshot.received_frames,
        "analyzed_frames": snapshot.analyzed_frames,
        "discarded_analysis_frames": snapshot.discarded_analysis_frames,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args(argv)
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup non-negative")
    print(json.dumps(profile(iterations=args.iterations, warmup=args.warmup), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
