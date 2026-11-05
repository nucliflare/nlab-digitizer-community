"""Keep cyclic Qt wrapper cleanup on the GUI thread, outside numerical workers."""

from __future__ import annotations

import gc

from PySide6.QtCore import QObject, QThread, QTimer, Slot
from PySide6.QtWidgets import QApplication

_APPLICATION_ATTRIBUTE = "_nlab_gui_garbage_collector"


class GuiGarbageCollector(QObject):
    """Replace allocation-triggered collection with a GUI-owned timer.

    Numerical allocations can otherwise collect unreachable plot/widget cycles
    from a worker. PySide wrapper destruction then races GUI QObject cleanup;
    native CI stacks showed a Qt connection-mutex/GIL deadlock. The scheduling
    follows pyqtgraph.util.garbage_collector, with application ownership and
    explicit, idempotent restoration of the previous automatic-GC state.
    """

    def __init__(self, application: QApplication) -> None:
        super().__init__(application)
        self.closed = False
        self._restore_enabled = gc.isenabled()
        gc.disable()
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.check)
        application.aboutToQuit.connect(self.close)
        self.timer.start()

    @Slot()
    def check(self) -> None:
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("Cyclic GUI garbage collection requires the GUI thread")
        if self.closed:
            return
        counts = gc.get_count()
        thresholds = gc.get_threshold()
        if thresholds[0] > 0 and counts[0] > thresholds[0]:
            gc.collect(0)
            if counts[1] > thresholds[1]:
                gc.collect(1)
                if counts[2] > thresholds[2]:
                    gc.collect(2)

    @Slot()
    def close(self) -> None:
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("Restore GUI garbage collection from the GUI thread")
        if self.closed:
            return
        self.closed = True
        self.timer.stop()
        if self._restore_enabled:
            gc.enable()


def install_gui_garbage_collector() -> GuiGarbageCollector | None:
    application = QApplication.instance()
    if not isinstance(application, QApplication):
        return None
    if QThread.currentThread() is not application.thread():
        raise RuntimeError("Install GUI garbage collection from the GUI thread")
    existing = getattr(application, _APPLICATION_ATTRIBUTE, None)
    if isinstance(existing, GuiGarbageCollector) and not existing.closed:
        return existing
    collector = GuiGarbageCollector(application)
    setattr(application, _APPLICATION_ATTRIBUTE, collector)
    return collector
