#!/usr/bin/env python
"""Minimal example of continuous current monitoring with IIR or Scope DMA.

IIR reads one filtered value at a time. DMA receives waveform frames and keeps
only the newest frame for the callback. Both modes run without an added delay
and stop when you press Ctrl+C.

The values are raw device readings, not amperes. Converting them to current
requires calibration for the sensor and input circuit.

A callback is a function passed to another function. The measurement code
calls it whenever new data is ready. In this example, the IIR callback receives
one integer, while the DMA callback receives a NumPy array containing the
newest waveform. Its return value is ignored. A callback can print the data,
convert it to amperes, save it, update a plot, or place it in a queue for
another worker.

The callbacks in ``main()`` only print a short result. Replace them with your
own functions if needed. If a callback reports an error, the measurement stops
and the device is still closed safely.

Set ``USE_DMA = True`` to use DMA. Set it to ``False`` to use IIR.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from nlab.hardware.digitizer import Digitizer
from nlab.hardware.digitizer.dma import IIOScopeDmaStreamer, ScopeFrameBuffer
from nlab.hardware.digitizer.scope import TriggerMode

URI = "ip:192.168.10.128:30431"
CHANNEL = 0
USE_DMA = True


def measure_current_iir(
    digitizer: Digitizer,
    callback: Callable[[int], None] | None = None,
) -> None:
    """Read IIR values until Ctrl+C.

    Each loop reads one value. There is no ``sleep()``, so the next read starts
    as soon as the previous read and callback finish. If provided, ``callback``
    receives the newest raw value as an ``int``.

    Keep the callback short. Slow calculations, file writes, or network work
    will slow the reading loop. For heavier work, place the value in a queue
    and let another worker handle it.
    """
    while True:
        value = int(digitizer.mca.filters.lp.get_iir_average())
        if callback is not None:
            # For heavier work, put ``value`` in a queue here and process it
            # in another worker so the next reading can start quickly.
            callback(value)


def measure_current_dma(
    digitizer: Digitizer,
    callback: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Receive Scope DMA frames until Ctrl+C and pass on the newest frame.

    A background worker receives complete waveform frames. This function keeps
    only the newest available frame and gives it to ``callback`` as a NumPy
    ``int16`` array. If the callback is slow, older waiting frames may be
    replaced, but receiving continues.

    The callback can calculate a value, apply calibration, update a plot, or
    pass the frame to other code. This small example is meant for viewing the
    latest data, not for processing every frame.

    Cleanup stops Scope, ends any waiting read, and waits for the background
    worker before the device connection is closed.
    """
    streamer = digitizer.scope_dma
    if not isinstance(streamer, IIOScopeDmaStreamer):
        raise RuntimeError("Scope DMA requires the direct IIO backend")

    scope = digitizer.scope
    latest_only = ScopeFrameBuffer(max_frames=1)
    stop_event = threading.Event()
    frame_ready = threading.Event()
    errors: list[str] = []

    def capture() -> None:
        """Receive complete DMA frames in the background."""
        try:
            streamer.stream_to_file(
                None,
                stop_event,
                on_progress=lambda _bytes: frame_ready.set(),
                frame_buffer=latest_only,
            )
        except BaseException as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            frame_ready.set()

    worker: threading.Thread | None = None
    try:
        scope.stop()
        scope.set_pretrigger_samples(0)
        scope.set_frame_samples(8188)
        scope.set_trigger_mode(TriggerMode.TIMED)
        scope.set_frame_period_cycles(12_500)

        worker = threading.Thread(target=capture, name="current-dma")
        worker.start()
        while worker.is_alive():
            frame_ready.wait()
            frame_ready.clear()
            frames, _ = latest_only.drain()
            if frames and callback is not None:
                # The background worker keeps receiving data while this
                # callback handles the newest frame.
                callback(np.asarray(frames[-1].samples, dtype=np.int16))
            if errors:
                raise RuntimeError(f"Scope DMA failed: {errors[0]}")
    finally:
        try:
            scope.stop()
        finally:
            stop_event.set()
            if worker is not None:
                try:
                    streamer.request_stop()
                finally:
                    worker.join()


def main() -> None:
    """Connect to the device and run the selected mode until Ctrl+C."""
    digitizer = Digitizer.from_iio(CHANNEL, URI, with_ids=False)
    try:
        if USE_DMA:
            measure_current_dma(
                digitizer,
                callback=lambda frame: print(
                    f"DMA mean raw current: {frame.mean():.1f}"
                ),
            )
        else:
            measure_current_iir(
                digitizer,
                callback=lambda value: print(f"IIR raw current: {value}"),
            )
    except KeyboardInterrupt:
        print("\nMeasurement stopped.")
    finally:
        digitizer.close()


if __name__ == "__main__":
    main()
