"""Readable plain-text tooltips without changing their meaning or F1 help."""

from __future__ import annotations

import textwrap

import shiboken6
from pyqtgraph import GraphicsView
from PySide6 import QtGui
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QCursor, QHelpEvent
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QTabBar, QToolTip, QWidget

from nlab.views.app_events import application_event_hub

TOOLTIP_LINE_LENGTH = 68
# A plot's tooltip covers the data under a resting cursor, so plots wait this
# many times Qt's tooltip delay (QStyle.SH_ToolTip_WakeUpDelay) before showing.
PLOT_TOOLTIP_DELAY_FACTOR = 2

# These show a tooltip per tab or menu entry before falling back to their own,
# and an application filter runs first: leave their tooltip events to Qt.
_PER_ITEM_TOOLTIP_WIDGETS = (QTabBar, QMenu)
_ROUTE_OWNER_NAME = "nlabTooltipWrapping"
_PLOT_DELAY_OWNER_NAME = "nlabPlotTooltipDelay"
# Qt drops a pending tooltip on these; so does the plot delay.
_TOOLTIP_CANCEL_EVENTS = (
    QEvent.Type.MouseButtonPress,
    QEvent.Type.MouseButtonDblClick,
    QEvent.Type.Wheel,
    QEvent.Type.KeyPress,
)


def wrap_tooltip(text: str) -> str:
    """Wrap prose and long paths, preserving explicit line/paragraph breaks.

    Keep plain text rather than HTML so F1 explanations do not expose markup.
    Character-based wrapping follows the user's font scaling without forcing a
    fixed-pixel tooltip size. Short hints and empty tooltips remain unchanged.
    """
    return "\n".join(
        textwrap.fill(line, width=TOOLTIP_LINE_LENGTH, break_on_hyphens=False,
                      replace_whitespace=False)
        for line in text.split("\n")
    )


def wrap_widget_tooltips(root: QWidget) -> None:
    """Apply at the end of UI construction; safe for already-wrapped children."""
    for widget in (root, *root.findChildren(QWidget)):
        if widget.toolTip():
            widget.setToolTip(wrap_tooltip(widget.toolTip()))


def _show_wrapped_tooltip(watched: QObject, event: QEvent) -> bool:
    """Show a long plain-text widget tooltip wrapped (application event hub).

    Qt word-wraps rich-text tooltips but keeps plain text on one line until it
    is wider than the whole screen. Only the displayed copy is wrapped: the
    widget keeps its tooltip, so F1 help and code read the original text.
    """
    if not isinstance(watched, QWidget) or not isinstance(event, QHelpEvent):
        return False
    if isinstance(watched, _PER_ITEM_TOOLTIP_WIDGETS):
        return False
    text = watched.toolTip()
    if not text or QtGui.Qt.mightBeRichText(text):
        return False
    wrapped = wrap_tooltip(text)
    if wrapped == text:
        return False
    QToolTip.showText(event.globalPos(), wrapped, watched, QRect(), watched.toolTipDuration())
    return True


def _is_plot(widget: QWidget) -> bool:
    """A pyqtgraph view, or the viewport Qt sends its tooltip events to."""
    return isinstance(widget, GraphicsView) or isinstance(widget.parentWidget(), GraphicsView)


def _pointer_rests_on(widget: QWidget, global_pos: QPoint) -> bool:
    """The cursor has not moved, no button is held, and ``widget`` is under it.

    Like Qt, only a widget in the active window shows a tooltip.
    """
    if QCursor.pos() != global_pos or QApplication.mouseButtons() != Qt.MouseButton.NoButton:
        return False
    if not widget.window().isActiveWindow():
        return False
    under = QApplication.widgetAt(global_pos)
    return under is not None and shiboken6.getCppPointer(under) == shiboken6.getCppPointer(widget)


class _PlotTooltipDelay(QObject):
    """Hold a plot's tooltip event for one more Qt tooltip delay.

    Qt has a single application-wide delay. Qt's own event is consumed but
    left ignored, so Qt does not start its short follow-up delay
    (SH_ToolTip_FallAsleepDelay) for a tooltip it never showed. Qt hands an
    ignored help event on to each parent up to the window; those copies are
    consumed too, or a container's own tooltip would appear at once. The help
    event is sent again once the cursor has rested on the same spot for the
    extra time. Moving restarts Qt's delay, and its next event restarts this one.
    """

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.setObjectName(_PLOT_DELAY_OWNER_NAME)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._release)
        self._target: QWidget | None = None
        self._local = QPoint()
        self._global = QPoint()
        self._releasing = False

    @property
    def pending(self) -> bool:
        return self._timer.isActive()

    def handle(self, watched: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.Type.ToolTip:
            self._cancel()
            return False
        if self._releasing or not isinstance(event, QHelpEvent):
            return False
        if not isinstance(watched, QWidget):
            return False
        if self._passes_up_held_event(watched, event):
            event.ignore()
            return True
        if not _is_plot(watched) or QToolTip.isVisible():
            return False  # already showing: follow the cursor as Qt does
        wake_ms = watched.style().styleHint(
            QStyle.StyleHint.SH_ToolTip_WakeUpDelay, None, watched
        )
        extra_ms = wake_ms * (PLOT_TOOLTIP_DELAY_FACTOR - 1)
        if extra_ms <= 0:
            return False
        self._target = watched
        self._local = QPoint(event.pos())
        self._global = QPoint(event.globalPos())
        event.ignore()
        self._timer.start(extra_ms)
        return True

    def _passes_up_held_event(self, widget: QWidget, event: QHelpEvent) -> bool:
        """``event`` is Qt's copy of the held event for a parent of the plot."""
        target = self._target
        return (
            self._timer.isActive()
            and target is not None
            and shiboken6.isValid(target)
            and event.globalPos() == self._global
            and widget.isAncestorOf(target)
        )

    def _cancel(self) -> None:
        self._timer.stop()
        self._target = None

    def _release(self) -> None:
        target, self._target = self._target, None
        if target is None or not shiboken6.isValid(target):
            return
        if not _pointer_rests_on(target, self._global):
            return
        event = QHelpEvent(QEvent.Type.ToolTip, self._local, self._global)
        self._releasing = True
        try:
            QApplication.sendEvent(target, event)
        finally:
            self._releasing = False


def install_tooltip_wrapping() -> None:
    """Wrap long tooltips of every existing and future widget when shown.

    Plot tooltips also wait ``PLOT_TOOLTIP_DELAY_FACTOR`` times Qt's delay.
    """
    application = QApplication.instance()
    event_hub = application_event_hub()
    if application is None or event_hub is None:
        return
    if application.findChild(QObject, _ROUTE_OWNER_NAME) is not None:
        return
    # Registered first: a held plot event must not reach the wrapping route.
    delay = _PlotTooltipDelay(application)
    # The hub holds its handler weakly; keep the Python object with the app.
    setattr(application, "_nlab_plot_tooltip_delay", delay)
    event_hub.register(
        delay, delay.handle, events=(QEvent.Type.ToolTip, *_TOOLTIP_CANCEL_EVENTS),
    )
    # The hub keeps one route per owner; the application itself already owns
    # the numeric-input route, so this route gets an owner of its own.
    owner = QObject(application)
    owner.setObjectName(_ROUTE_OWNER_NAME)
    event_hub.register(owner, _show_wrapped_tooltip, events=(QEvent.Type.ToolTip,))
