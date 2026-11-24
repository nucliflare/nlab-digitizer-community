"""Cancellable background execution for Klein–Nishina Compton-edge fits."""

from __future__ import annotations

import logging
import threading
import time

from PySide6.QtCore import Signal

from nlab.analysis.compton_edge import (
    ComptonEdgeFitCancelledError,
    ComptonEdgeFitProgress,
    ComptonEdgeFitSpec,
    fit_compton_edges,
)
from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)

PROGRESS_INTERVAL_S = 0.1
"""Minimum spacing of in-stage progress signals; stage changes are sent at once."""


class ComptonEdgeFitWorker(BaseWorker):
    result = Signal(object)
    progress = Signal(object)
    cancelled = Signal()

    def __init__(self, spec: ComptonEdgeFitSpec) -> None:
        super().__init__()
        self._spec = spec
        self._stop_event = threading.Event()
        self._last_progress: tuple[str, float] | None = None

    def run(self) -> None:
        try:
            result = fit_compton_edges(
                self._spec,
                cancelled=self._stop_event.is_set,
                progress=self._report_progress,
            )
            self.result.emit(result)
        except ComptonEdgeFitCancelledError:
            self.cancelled.emit()
        except Exception as exc:
            labels = ", ".join(dataset.label for dataset in self._spec.datasets)
            log.exception("Compton-edge fit failed for %s", labels)
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def _report_progress(self, update: ComptonEdgeFitProgress) -> None:
        """Forward stage changes and stage completion; throttle counts within a stage.

        The fit reports every gain candidate and residual evaluation. Throttling
        keeps the queued signals from flooding the GUI event loop.
        """
        now = time.monotonic()
        last = self._last_progress
        stage_complete = update.total is not None and update.done >= update.total
        if (
            last is not None
            and last[0] == update.stage
            and not stage_complete
            and now - last[1] < PROGRESS_INTERVAL_S
        ):
            return
        self._last_progress = (update.stage, now)
        self.progress.emit(update)

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()
