#!/usr/bin/env python
"""Record raw Scope NDMA frames over direct IIO (no GUI or legacy backend).

Example::

    python examples/dma_scope_periodic.py capture.bin --duration-s 10

Without --config, Scope Auto Setup calibrates the DAC/trigger threshold first.
With --config, a GUI v3 YAML or a single-channel ``scope:`` YAML supplies
those calibration values. Explicit CLI capture settings override YAML values.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nlab.hardware.digitizer import Digitizer
from nlab.hardware.digitizer.dma import FILE_HEADER_STRUCT
from nlab.hardware.digitizer.scope import TriggerMode
from nlab.utils.settings_io import (
    channel_entry,
    read_configuration,
    validate_configuration_version,
)
from nlab.workers.scope_auto_setup_worker import ScopeAutoSetupProcedure


@dataclass(frozen=True)
class CaptureOptions:
    output: Path
    uri: str
    channel: int
    duration_s: float
    frame_samples: int
    gap_cycles: int
    pretrigger_samples: int
    trigger_mode: TriggerMode
    calibration: dict[str, int] | None


def _size_label(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KiB"
    if size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MiB"
    return f"{size_bytes / 1024**3:.1f} GiB"


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _mode(value: Any) -> TriggerMode:
    if type(value) is int:
        try:
            return TriggerMode(value)
        except ValueError as exc:
            raise ValueError(f"invalid trigger mode: {value}") from exc
    if isinstance(value, str):
        name = value.upper().replace("-", "_")
        if name == "PERIODIC":
            name = "TIMED"
        try:
            return TriggerMode[name]
        except KeyError as exc:
            raise ValueError(f"invalid trigger mode: {value}") from exc
    raise ValueError(f"invalid trigger mode: {value!r}")


def _scope_yaml(path: Path | None, channel: int) -> dict[str, Any] | None:
    if path is None:
        return None
    document = read_configuration(path)
    validate_configuration_version(document)
    entry = channel_entry(document, channel) if "hardware" in document else document
    scope = entry.get("scope") if entry is not None else None
    if not isinstance(scope, dict):
        raise ValueError(f"{path}: no Scope settings for channel {channel}")
    if "dac_value" not in scope or "trigger_level" not in scope:
        raise ValueError(f"{path}: Scope YAML needs dac_value and trigger_level")
    return scope


def _options(args: argparse.Namespace) -> CaptureOptions:
    if not 0 < args.port <= 65535 or args.channel < 0:
        raise ValueError("port must be 1..65535 and channel must be nonnegative")
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        raise ValueError("duration-s must be a finite positive number")
    scope_yaml = _scope_yaml(args.config, args.channel)
    saved = scope_yaml or {}
    frame_ns = _integer(
        args.frame_ns if args.frame_ns is not None else saved.get("frame_ns", 16376),
        "frame-ns",
    )
    gap_ns = _integer(
        args.gap_ns if args.gap_ns is not None else saved.get("frame_gap_ns", 1000),
        "gap-ns",
    )
    pretrigger_ns = _integer(
        args.pretrigger_ns
        if args.pretrigger_ns is not None
        else saved.get("pretrigger_ns", 0),
        "pretrigger-ns",
    )
    # vdpp-scope.c v121: 2 ns/sample, four-sample beats, max 8,188 samples;
    # frame_period_cycles is a 16-bit count of 8 ns datapath clocks.
    if frame_ns < 8 or frame_ns > 16376 or frame_ns % 8:
        raise ValueError("frame-ns must be an 8 ns multiple in 8..16376")
    if gap_ns < 0 or gap_ns > 65535 * 8 or gap_ns % 8:
        raise ValueError("gap-ns must be an 8 ns multiple in 0..524280")
    if pretrigger_ns < 0 or pretrigger_ns > 2040 or pretrigger_ns % 8:
        raise ValueError("pretrigger-ns must be an 8 ns multiple in 0..2040")
    trigger_mode = _mode(
        args.trigger_mode if args.trigger_mode is not None else saved.get("edge_mode", "periodic")
    )
    calibration = None
    if scope_yaml is not None:
        calibration = {
            "dac_value": _integer(scope_yaml["dac_value"], "dac_value"),
            "trigger_level": _integer(scope_yaml["trigger_level"], "trigger_level"),
        }
    return CaptureOptions(
        output=args.output,
        uri=f"ip:{args.host}:{args.port}",
        channel=args.channel,
        duration_s=args.duration_s,
        frame_samples=frame_ns // 2,
        gap_cycles=gap_ns // 8,
        pretrigger_samples=pretrigger_ns // 2,
        trigger_mode=trigger_mode,
        calibration=calibration,
    )


def _record(digitizer: Digitizer, options: CaptureOptions) -> int:
    streamer = digitizer.scope_dma
    if streamer is None:
        raise RuntimeError("this IIO channel has no Scope DMA streamer")
    stop_event = threading.Event()
    ready = threading.Event()
    done = threading.Event()
    result: dict[str, int | str] = {}
    progress_lock = threading.Lock()
    progress_bytes = 0
    frame_bytes = options.frame_samples * 2
    interactive = sys.stdout.isatty()
    started: float | None = None

    def on_progress(payload_bytes: int) -> None:
        nonlocal progress_bytes
        # Called by the writer thread. Keep terminal I/O on the main thread.
        with progress_lock:
            progress_bytes = payload_bytes

    def show_progress(*, final: bool = False) -> None:
        if started is None:
            return
        elapsed = min(time.monotonic() - started, options.duration_s)
        fraction = min(1.0, elapsed / options.duration_s)
        filled = int(fraction * 30)
        with progress_lock:
            payload_bytes = progress_bytes
        frames = payload_bytes // frame_bytes
        size_label = _size_label(FILE_HEADER_STRUCT.size + payload_bytes)
        bar = f"[{'#' * filled}{'-' * (30 - filled)}]"
        line = (
            f"{bar} {fraction:5.1%}  {elapsed:5.1f}/{options.duration_s:.1f}s"
            f"  {frames:,} frames  {size_label:>10}"
        )
        if interactive:
            print(f"\r{line}", end="\n" if final else "", flush=True)
        else:
            print(line, flush=True)

    def capture() -> None:
        try:
            result["frames"] = streamer.stream_to_file(
                options.output,
                stop_event,
                on_ready=ready.set,
                on_progress=on_progress,
            )
        except BaseException as exc:
            # Keep only scalar error data; a traceback could retain the native buffer.
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            done.set()
            ready.set()

    worker = threading.Thread(target=capture, name="scope-dma-capture")
    worker.start()
    stop_errors: list[str] = []
    try:
        if not ready.wait(10):
            raise TimeoutError("DMA writer did not become ready within 10 seconds")
        if "error" not in result:
            started = time.monotonic()
            show_progress()
            refresh_s = 0.1 if interactive else 1.0
            while not done.is_set():
                remaining = options.duration_s - (time.monotonic() - started)
                if remaining <= 0:
                    break
                done.wait(min(refresh_s, remaining))
                show_progress()
    finally:
        # vdpp-scope.c: stop acquisition before cancel/drain/close of the IIO buffer.
        try:
            digitizer.scope.stop()
        except Exception as exc:
            stop_errors.append(f"scope stop: {type(exc).__name__}: {exc}")
        stop_event.set()
        try:
            streamer.request_stop()
        except Exception as exc:
            stop_errors.append(f"DMA cancel: {type(exc).__name__}: {exc}")
        worker.join()  # Native buffer must be destroyed before closing Digitizer.
        if started is not None:
            show_progress(final=True)
    if stop_errors:
        raise RuntimeError("; ".join(stop_errors))
    if "error" in result:
        raise RuntimeError(f"Scope DMA failed: {result['error']}")
    frames = int(result.get("frames", 0))
    if frames < 1:
        raise RuntimeError("Scope DMA stopped without a complete frame")
    expected_bytes = FILE_HEADER_STRUCT.size + frames * options.frame_samples * 2
    if options.output.stat().st_size != expected_bytes:
        raise RuntimeError("Scope DMA output has an unexpected size")
    return frames


def run(options: CaptureOptions) -> int:
    if options.output.exists():
        raise FileExistsError(f"output already exists: {options.output}")
    if not options.output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {options.output.parent}")

    digitizer = Digitizer.from_iio(options.channel, options.uri, with_ids=False)
    scope = digitizer.scope
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
            "trigger_level": scope.get_trigger_level(),
            "dac_value": scope.get_dac_value(),
        }
        if original["enabled"] or original["dma_enabled"]:
            raise RuntimeError("Scope is already active; refusing to alter another acquisition")
        if scope.dma_fault_is_latched():
            raise RuntimeError("Scope DMA fault is latched; acknowledge recovery before recording")
        owned = True

        # Some boards boot with frame_samples=0; Auto Setup needs valid geometry.
        if int(original["frame_samples"]) < 4:
            scope.set_frame_samples(1024)
            scope.set_pretrigger_samples(0)
        if options.calibration is None:
            print("Running Scope Auto Setup...", flush=True)
            auto = ScopeAutoSetupProcedure(scope).run()
            if not auto.verified:
                raise RuntimeError("Scope Auto Setup did not verify the input signal")
        else:
            scope.set_dac_value(options.calibration["dac_value"])
            scope.set_trigger_level(options.calibration["trigger_level"])

        scope.stop()
        scope.set_frame_samples(options.frame_samples)
        scope.set_pretrigger_samples(options.pretrigger_samples)
        scope.set_frame_period_cycles(options.gap_cycles)
        scope.set_trigger_mode(options.trigger_mode)
        print(f"Recording raw DMA frames to {options.output}...", flush=True)
        # Reserve the name atomically; the streamer will reopen this file for
        # its NDMA header, but a pre-existing user capture is never truncated.
        with options.output.open("xb"):
            pass
        frames = _record(digitizer, options)
        print(f"Saved {frames} complete frames ({options.output.stat().st_size} bytes)")
        return 0
    finally:
        try:
            if owned:
                scope.stop()
                if int(original["frame_samples"]) >= 4:
                    scope.set_frame_samples(int(original["frame_samples"]))
                    scope.set_pretrigger_samples(int(original["pretrigger_samples"]))
                else:
                    print("Initial frame length was invalid; leaving 1,024-sample geometry")
                    scope.set_frame_samples(1024)
                    scope.set_pretrigger_samples(0)
                scope.set_frame_period_cycles(int(original["gap_cycles"]))
                scope.set_trigger_level(int(original["trigger_level"]))
                scope.set_trigger_mode(TriggerMode(original["trigger_mode"]))
                scope.set_dac_value(int(original["dac_value"]))
        finally:
            digitizer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new NDMA .bin file (will not overwrite)")
    parser.add_argument("--host", default="192.168.10.128")
    parser.add_argument("--port", type=int, default=30431)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--config", type=Path, help="GUI v3 or single-channel Scope YAML")
    parser.add_argument(
        "--trigger-mode",
        help="periodic (default), any_above, any_below, falling_edge, rising_edge",
    )
    parser.add_argument("--frame-ns", type=int, help="frame duration; default 16376 ns")
    parser.add_argument("--gap-ns", type=int, help="periodic gap; default 1000 ns")
    parser.add_argument("--pretrigger-ns", type=int, help="pretrigger; default 0 ns")
    parser.add_argument(
        "--display-mode", choices=("raw",), default="raw", help="headless raw output only"
    )
    parser.add_argument("--dma", choices=("on",), default="on", help="DMA capture only")
    args = parser.parse_args(argv)
    try:
        options = _options(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        return run(options)
    except KeyboardInterrupt:
        print("Interrupted; DMA stopped and hardware restored")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
