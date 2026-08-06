from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QTimer, Signal, Slot

from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)


@dataclass
class MCAReadback:
    histogram: np.ndarray
    debug1: np.ndarray
    debug2: np.ndarray
    count_rate: int = 0
    pulse_deadtime: int = 0
    events_lost: int = 0
    elapsed_time: int = 0
    pulse_overrange: int = 0
    pulse_pileup: int = 0
    energy_overrange: int = 0
    energy_estimation_error: int = 0
    throughput_error: int = 0


class MCAWorker(BaseWorker):
    """Periodically reads histogram, debug waveforms, and statistics.

    Create, ``moveToThread``, connect ``thread.started`` to ``run``,
    then emit ``request_stop`` to shut down.
    """

    readback = Signal(object)
    measurement_done = Signal()
    request_stop = Signal()
    change_interval = Signal(int)

    def __init__(
        self,
        mca: MultiChannelAnalyzer,
        interval_ms: int = 200,
    ) -> None:
        super().__init__()
        self._mca = mca
        self._interval_ms = interval_ms
        self._timer: QTimer | None = None
        self._seen_running = False
        self._last_histogram: np.ndarray = np.empty(0, dtype=np.uint32)
        self._last_debug1: np.ndarray = np.empty(0, dtype=np.int16)
        self._last_debug2: np.ndarray = np.empty(0, dtype=np.int16)
        self._histogram_read_failing = False
        self._waveform_read_failing = False

    def run(self) -> None:
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self.request_stop.connect(self._stop)
        self.change_interval.connect(self._set_interval)
        self._timer.start(self._interval_ms)

    @Slot(int)
    def _set_interval(self, ms: int) -> None:
        if self._timer is not None:
            self._timer.setInterval(ms)

    @Slot()
    def _stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.finished.emit()

    @staticmethod
    def _stat(fn) -> int:
        """Some backends don't implement every statistic (e.g. the IIO
        backend's get_events_lost() -- vdpp-pulse-processor.c has no
        matching register). Without this, one missing stat would raise
        NotImplementedError here and abort the whole tick's readback
        (histogram/waveforms included), not just that one field.
        """
        try:
            return fn()
        except NotImplementedError:
            return 0

    def _acquire_histogram(self) -> np.ndarray:
        """Fall back to the last successful histogram on transport errors.

        The current vdpp-pulse-processor.c allows informational live
        snapshots while bins are changing and a stable snapshot after
        enable=0. RuntimeError is still expected on the deployed board
        tested so far because its histogram_data binary attribute is not
        discoverable through remote iiod at all.

        Specifically, confirmed live against a real board,
        IIODigitizerBackend.read_histogram() currently raises RuntimeError
        on *every* call, not just while running -- the histogram_data
        bin_attribute isn't discoverable at all over the IIO network
        transport on this board's firmware+iiod build (see that method's
        docstring). Same fallback applies; logged once on entry/exit
        rather than every tick, since this is expected to be a standing
        condition, not per-call transient noise.
        """
        try:
            self._last_histogram = self._mca.acquire_spectrum()
            if self._histogram_read_failing:
                log.info("MCA: histogram reads recovered")
                self._histogram_read_failing = False
        except OSError as e:
            if e.errno != 16:  # not EBUSY
                raise
        except RuntimeError:
            if not self._histogram_read_failing:
                log.warning(
                    "MCA: histogram reads are failing, falling back to the "
                    "last-known spectrum until this recovers", exc_info=True,
                )
                self._histogram_read_failing = True
        return self._last_histogram

    def _acquire_waveforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Same fallback reasoning as _acquire_histogram() -- see its
        docstring. debug_data has no EBUSY-while-running guard in the
        driver, but confirmed live it has the *same* bin_attribute
        discovery gap as histogram_data, so IIODigitizerBackend.
        read_waveform_banks() currently also raises RuntimeError on every
        call. Falls back to the last-known waveform banks (empty arrays
        until the first successful read) so statistics/measurement-state
        polling keeps working even while this is failing.
        """
        try:
            self._last_debug1, self._last_debug2 = self._mca.acquire_waveforms()
            if self._waveform_read_failing:
                log.info("MCA: waveform reads recovered")
                self._waveform_read_failing = False
        except (OSError, RuntimeError):
            if not self._waveform_read_failing:
                log.warning(
                    "MCA: waveform reads are failing, falling back to the "
                    "last-known waveforms until this recovers", exc_info=True,
                )
                self._waveform_read_failing = True
        return self._last_debug1, self._last_debug2

    def _tick(self) -> None:
        try:
            histogram = self._acquire_histogram()
            debug1, debug2 = self._acquire_waveforms()
            stats = self._mca.statistics

            rb = MCAReadback(
                histogram=histogram,
                debug1=debug1,
                debug2=debug2,
                count_rate=self._stat(stats.get_count_rate),
                pulse_deadtime=self._stat(stats.get_pulse_deadtime),
                events_lost=self._stat(stats.get_events_lost),
                elapsed_time=self._stat(stats.get_elapsed_time),
                pulse_overrange=self._stat(stats.get_pulse_overrange),
                pulse_pileup=self._stat(stats.get_pulse_pileup),
                energy_overrange=self._stat(stats.get_energy_overrange),
                energy_estimation_error=self._stat(stats.get_energy_estimation_error),
                throughput_error=self._stat(stats.get_throughput_error),
            )
        except Exception:
            log.exception("MCA readback failed")
            return
        self.readback.emit(rb)
        if self._mca.get_measurement_in_progress():
            self._seen_running = True
        elif self._seen_running:
            log.info("MCA: hardware measurement completed (time limit reached)")
            self._stop()
            self.measurement_done.emit()
