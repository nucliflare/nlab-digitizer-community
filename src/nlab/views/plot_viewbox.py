from __future__ import annotations

from typing import Any

from pyqtgraph import PlotCurveItem, PlotWidget, ViewBox
from PySide6.QtCore import Qt, Signal


class ModifierZoomViewBox(ViewBox):
    """ViewBox with modifier-key axis-locked scrolling.

    * Shift + wheel  → horizontal zoom only
    * Ctrl  + wheel  → vertical zoom only
    * Plain wheel    → default pyqtgraph behaviour (zoom both)
    """

    def wheelEvent(self, ev, axis=None):  # noqa: N802
        mods = ev.modifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            axis = 0
        elif mods & Qt.KeyboardModifier.ControlModifier:
            axis = 1
        super().wheelEvent(ev, axis=axis)


class NLabPlotWidget(PlotWidget):
    """PlotWidget that uses ModifierZoomViewBox by default.

    Promote QWidget to this class in Qt Designer with header
    ``nlab.views.plot_viewbox``.
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent, viewBox=ModifierZoomViewBox())


class DraggableScopeCurve(PlotCurveItem):  # type: ignore[misc]
    """Raw Scope trace with a narrow grab area; ordinary plot drags still zoom/pan."""

    drag_started = Signal()
    drag_moved = Signal(float, float)
    drag_finished = Signal(bool)  # True when cancelled.

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.setClickable(True, width=14)
        self._drag_enabled = True
        self._drag_origin = None

    def setDragEnabled(self, enabled: bool) -> None:  # noqa: N802
        if not enabled:
            self.cancelDrag()
        self._drag_enabled = enabled
        self.setClickable(enabled, width=14)
        self.unsetCursor()

    def cancelDrag(self) -> None:  # noqa: N802
        if self._drag_origin is not None:
            self._drag_origin = None
            self.unsetCursor()
            self.drag_finished.emit(True)

    def hoverEvent(self, ev: Any) -> None:  # noqa: N802
        if self._drag_origin is not None:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif (
            self._drag_enabled
            and not ev.isExit()
            and self.mouseShape().contains(ev.pos())
            and ev.acceptDrags(Qt.MouseButton.LeftButton)
        ):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    def mouseDragEvent(self, ev: Any) -> None:  # noqa: N802
        if not self._drag_enabled or ev.button() != Qt.MouseButton.LeftButton:
            return
        if ev.isStart():
            if not self.mouseShape().contains(ev.buttonDownPos()):
                return
            self._drag_origin = ev.buttonDownPos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.drag_started.emit()
        if self._drag_origin is None:
            return
        ev.accept()
        delta = ev.pos() - self._drag_origin
        self.drag_moved.emit(float(delta.x()), float(delta.y()))
        if ev.isFinish():
            self._drag_origin = None
            self.unsetCursor()
            self.drag_finished.emit(False)

    def mouseClickEvent(self, ev: Any) -> None:  # noqa: N802
        if self._drag_origin is not None and ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self.cancelDrag()
            return
        super().mouseClickEvent(ev)
