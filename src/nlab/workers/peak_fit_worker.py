"""Cancellable background execution for MCA peak fitting."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import Signal

from nlab.analysis.peak_fitting import (
    PeakFitCancelledError,
    PeakFitSpec,
    fit_spectrum_peaks,
)
from nlab.analysis.spectrum import Spectrum
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)


class PeakFitWorker(BaseWorker):
    result = Signal(object)
    cancelled = Signal()

    def __init__(self, spectrum: Spectrum, spec: PeakFitSpec) -> None:
        super().__init__()
        self._spectrum = spectrum
        self._spec = spec
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            result = fit_spectrum_peaks(
                self._spectrum,
                self._spec,
                cancelled=self._stop_event.is_set,
            )
            self.result.emit(result)
        except PeakFitCancelledError:
            self.cancelled.emit()
        except Exception as exc:
            log.exception("MCA peak fit failed for %s", self._spectrum.label)
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def stop(self) -> None:
        self._stop_event.set()
