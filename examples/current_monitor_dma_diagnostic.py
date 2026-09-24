#!/usr/bin/env python
"""Measure the production Scope-DMA current accumulator, optionally rendering.

The command changes one idle Scope channel temporarily, receives a fixed
number of frames through the same streamer used by the application, prints a
JSON counter snapshot, and restores the original Scope settings.

Example::

    python examples/current_monitor_dma_diagnostic.py --host 192.168.10.135 \
        --frame-samples 8000 --gap-cycles 12500 --frames 100000
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import asdict, dataclass

from nlab.hardware.digitizer import Digitizer
from nlab.hardware.digitizer.current_monitor import (
    ScopeCurrentAccumulator,
    ScopeCurrentRuntime,
)
from nlab.hardware.digitizer.dma import IIOScopeDmaStreamer
from nlab.hardware.digitizer.scope import TriggerMode


@dataclass(frozen=True)
class Options:
    uri: str
    channel: int
    frame_samples: int
    gap_cycles: int
    frames: int
    bin_width_ms: int
    render_fps: int


def _capture(
    streamer: IIOScopeDmaStreamer,
    accumulator: ScopeCurrentAccumulator,
    frames: int,
    render_fps: int,
    channel: int,
) -> tuple[int, int]:
    """Capture synchronously or render production snapshots on the main thread."""
    stop_event = threading.Event()
    if not render_fps:
        captured = streamer.stream_to_file(
            None,
            stop_event,
            n_frames=frames,
            current_accumulator=accumulator,
        )
        return captured, 0

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from nlab.controllers.current_monitor_controller import CurrentMonitorController

    app = QApplication.instance() or QApplication([])
    controller = CurrentMonitorController(
        object(),  # type: ignore[arg-type]
        channel=channel,
        auto_start=False,
        scope_current_accumulator=accumulator,
    )
    controller.ui.comboMode.setCurrentIndex(1)
    result: dict[str, int | str] = {}

    def receive() -> None:
        try:
            result["frames"] = streamer.stream_to_file(
                None,
                stop_event,
                n_frames=frames,
                current_accumulator=accumulator,
            )
        except BaseException as exc:
            # Do not retain a traceback that may retain a native IIO buffer.
            result["error"] = f"{type(exc).__name__}: {exc}"

    receiver = threading.Thread(target=receive, name="current-dma-diagnostic")
    receiver.start()
    period_ns = round(1_000_000_000 / render_fps)
    next_render_ns = time.perf_counter_ns() + period_ns
    last_generation = -1
    try:
        while receiver.is_alive():
            now_ns = time.perf_counter_ns()
            if now_ns >= next_render_ns:
                snapshot = accumulator.snapshot(now_ns=now_ns)
                if snapshot.generation != last_generation:
                    render_started_ns = time.perf_counter_ns()
                    controller._render_dma_snapshot(snapshot)
                    accumulator.note_display_update(
                        snapshot.generation,
                        time.perf_counter_ns() - render_started_ns,
                    )
                    last_generation = snapshot.generation
                app.processEvents()
                next_render_ns += period_ns
                if next_render_ns <= now_ns:
                    next_render_ns = now_ns + period_ns
            else:
                time.sleep(min(0.002, (next_render_ns - now_ns) / 1_000_000_000))
    except BaseException:
        stop_event.set()
        try:
            streamer.request_stop()
        finally:
            receiver.join()
        raise
    finally:
        controller.close()
    receiver.join()
    if "error" in result:
        raise RuntimeError(f"Scope DMA failed: {result['error']}")
    return int(result.get("frames", 0)), render_fps


def run(options: Options) -> dict[str, object]:
    """Run one accumulation-only session and return JSON-safe diagnostics."""
    digitizer = Digitizer.from_iio(options.channel, options.uri, with_ids=False)
    scope = digitizer.scope
    streamer = digitizer.scope_dma
    if not isinstance(streamer, IIOScopeDmaStreamer):
        digitizer.close()
        raise RuntimeError("the selected endpoint does not expose IIO Scope DMA")

    owned = False
    original: dict[str, int | bool | TriggerMode] = {}
    try:
        original = {
            "enabled": scope.get_enable(),
            "dma_enabled": scope.get_dma_enable(),
            "frame_samples": scope.get_frame_samples(),
            "pretrigger_samples": scope.get_pretrigger_samples(),
            "gap_cycles": scope.get_frame_period_cycles(),
            "trigger_mode": scope.get_trigger_mode(),
        }
        if original["enabled"] or original["dma_enabled"]:
            raise RuntimeError("Scope is already active; refusing to alter its acquisition")
        if scope.dma_fault_is_latched():
            raise RuntimeError("Scope DMA fault is latched; recovery must be acknowledged first")
        owned = True

        scope.stop()
        scope.set_pretrigger_samples(0)
        scope.set_frame_samples(options.frame_samples)
        scope.set_trigger_mode(TriggerMode.TIMED)
        scope.set_frame_period_cycles(options.gap_cycles)

        geometry = streamer.capture_geometry()
        if geometry.waveform_samples <= 0:
            raise RuntimeError("advertised DMA geometry contains no waveform samples")
        accumulator = ScopeCurrentAccumulator(bin_width_ms=options.bin_width_ms)
        accumulator.start_session(
            geometry,
            ScopeCurrentRuntime(
                channel=options.channel,
                ip_version=scope.get_ip_version(),
                gap_cycles=options.gap_cycles,
                expected_interval_ticks=(
                    options.gap_cycles + options.frame_samples // 4
                ),
                viewer_state="disabled",
            ),
        )

        started_ns = time.perf_counter_ns()
        captured, rendered_fps = _capture(
            streamer,
            accumulator,
            options.frames,
            options.render_fps,
            options.channel,
        )
        stopped_ns = time.perf_counter_ns()
        snapshot = accumulator.snapshot(now_ns=stopped_ns)
        if captured != snapshot.received_frames:
            raise RuntimeError(
                f"streamer/accumulator mismatch: {captured} != {snapshot.received_frames}"
            )
        if snapshot.analyzed_frames != snapshot.received_frames:
            raise RuntimeError("not every validated frame reached scientific accumulation")

        elapsed_s = (stopped_ns - started_ns) / 1_000_000_000
        runtime = asdict(snapshot.runtime) if snapshot.runtime is not None else None
        mean = (
            snapshot.analyzed_raw_sum / snapshot.analyzed_sample_count
            if snapshot.analyzed_sample_count
            else None
        )
        return {
            "uri": options.uri,
            "channel": options.channel,
            "requested_frames": options.frames,
            "captured_frames": captured,
            "elapsed_s": elapsed_s,
            "whole_run_fps": captured / elapsed_s,
            "whole_run_payload_MB_s": snapshot.received_bytes / elapsed_s / 1e6,
            "rolling_received_fps": snapshot.received_fps,
            "rolling_payload_MB_s": snapshot.payload_mb_s,
            "geometry": asdict(geometry),
            "runtime": runtime,
            "bin_width_ms": snapshot.bin_width_ns // 1_000_000,
            "retained_bins": len(snapshot.bins),
            "received_frames": snapshot.received_frames,
            "received_bytes": snapshot.received_bytes,
            "analyzed_frames": snapshot.analyzed_frames,
            "analyzed_samples": snapshot.analyzed_sample_count,
            "sample_weighted_raw_mean": mean,
            "rejected_frames": snapshot.rejected_frames,
            "discarded_analysis_frames": snapshot.discarded_analysis_frames,
            "protocol_errors": snapshot.protocol_errors,
            "skipped_opportunities": snapshot.skipped_opportunities,
            "off_grid_intervals": snapshot.off_grid_intervals,
            "median_interval_us": snapshot.median_interval_ns / 1000,
            "maximum_interval_us": snapshot.maximum_interval_ns / 1000,
            "observed_coverage_percent": snapshot.observed_coverage_percent,
            "replaced_preview_frames": snapshot.replaced_preview_frames,
            "analysis_queue_depth": snapshot.analysis_queue_depth,
            "analysis_queue_high_water": snapshot.analysis_queue_high_water,
            "analysis_lag_ms": snapshot.analysis_lag_ns / 1_000_000,
            "maximum_analysis_lag_ms": snapshot.maximum_analysis_lag_ns / 1_000_000,
            "render_fps": rendered_fps,
            "display_updates": snapshot.display_updates,
            "replaced_display_generations": snapshot.replaced_display_generations,
            "display_p95_ms": snapshot.display_p95_duration_ns / 1_000_000,
        }
    finally:
        try:
            if owned:
                scope.stop()
                original_frame_samples = int(original["frame_samples"])
                if original_frame_samples >= 4:
                    scope.set_frame_samples(original_frame_samples)
                    scope.set_pretrigger_samples(int(original["pretrigger_samples"]))
                else:
                    scope.set_frame_samples(1024)
                    scope.set_pretrigger_samples(0)
                scope.set_frame_period_cycles(int(original["gap_cycles"]))
                scope.set_trigger_mode(TriggerMode(original["trigger_mode"]))
        finally:
            digitizer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.10.128")
    parser.add_argument("--port", type=int, default=30431)
    parser.add_argument("--channel", type=int, choices=(0, 1), default=0)
    parser.add_argument("--frame-samples", type=int, default=8000)
    parser.add_argument("--gap-cycles", type=int, default=12500)
    parser.add_argument("--frames", type=int, default=100000)
    parser.add_argument("--bin-width-ms", type=int, default=100)
    parser.add_argument(
        "--render-fps",
        type=int,
        default=0,
        help="render the production Current widget at 1..30 fps (default: disabled)",
    )
    args = parser.parse_args(argv)
    if args.frame_samples < 4 or args.frame_samples % 4:
        parser.error("--frame-samples must be a positive multiple of four")
    if args.gap_cycles < 0:
        parser.error("--gap-cycles must be non-negative")
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if not 0 <= args.render_fps <= 30:
        parser.error("--render-fps must be 0..30")
    options = Options(
        uri=f"ip:{args.host}:{args.port}",
        channel=args.channel,
        frame_samples=args.frame_samples,
        gap_cycles=args.gap_cycles,
        frames=args.frames,
        bin_width_ms=args.bin_width_ms,
        render_fps=args.render_fps,
    )
    report = run(options)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
