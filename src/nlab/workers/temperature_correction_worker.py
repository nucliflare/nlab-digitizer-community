from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QTimer, Signal, Slot

from nlab.hardware.digitizer.hv import HVSupply
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)

T_MAX = 67.0


@dataclass(frozen=True)
class TemperatureCorrectionReadback:
    temperature: float
    applied_coefficient: float
    applied_offset: int


class TemperatureCorrectionWorker(BaseWorker):
    """Apply ADS5407-based MCA correction on an independent Qt thread."""

    readback = Signal(object)
    change_interval = Signal(int)
    change_parameters = Signal(float, int)
    request_stop = Signal()

    def __init__(
        self,
        hv: HVSupply,
        mcas: Sequence[MultiChannelAnalyzer],
        coefficient: float,
        offset: int,
        interval_ms: int = 1000,
    ) -> None:
        super().__init__()
        if not mcas:
            raise ValueError("TemperatureCorrectionWorker requires at least one MCA")
        self._hv = hv
        self._mcas = tuple(mcas)
        self._coefficient = coefficient
        self._offset = offset
        self._interval_ms = interval_ms
        self._timer: QTimer | None = None
        self._stop_requested = threading.Event()
        self._finished = False

    def run(self) -> None:
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self.change_interval.connect(self._set_interval)
        self.change_parameters.connect(self._set_parameters)
        self.request_stop.connect(self._stop)
        if self._stop_requested.is_set():
            self._stop()
            return
        self._tick()
        if not self._finished:
            self._timer.start(self._interval_ms)

    def request_shutdown(self) -> None:
        """Request shutdown even while a synchronous hardware call is active."""
        self._stop_requested.set()
        self.request_stop.emit()

    @Slot()
    def _stop(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.finished.emit()

    @Slot(int)
    def _set_interval(self, interval_ms: int) -> None:
        self._interval_ms = interval_ms
        if self._timer is not None:
            self._timer.setInterval(interval_ms)

    @Slot(float, int)
    def _set_parameters(self, coefficient: float, offset: int) -> None:
        self._coefficient = coefficient
        self._offset = offset
        self._tick()

    @Slot()
    def _tick(self) -> None:
        if self._stop_requested.is_set():
            self._stop()
            return
        try:
            temperature = self._hv.get_ads_temp_for_correction()
            delta = temperature - T_MAX
            applied_coefficient = self._coefficient * delta
            applied_offset = int(self._offset * delta)
            for mca in self._mcas:
                mca.set_temperature_correction(
                    applied_coefficient,
                    applied_offset,
                )
        except Exception as exc:
            if self._stop_requested.is_set():
                self._stop()
                return
            log.exception("MCA temperature-correction cycle failed")
            self.error.emit(str(exc))
            return

        if self._stop_requested.is_set():
            self._stop()
            return
        self.readback.emit(TemperatureCorrectionReadback(
            temperature=temperature,
            applied_coefficient=applied_coefficient,
            applied_offset=applied_offset,
        ))
