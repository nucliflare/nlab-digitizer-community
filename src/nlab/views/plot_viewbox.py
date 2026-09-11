from __future__ import annotations

from pyqtgraph import PlotWidget, ViewBox
from PySide6.QtCore import Qt


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
