"""Panel messages that do not participate in acquisition-control layouts."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import QStatusBar, QWidget
from shiboken6 import isValid


class PanelStatus(QObject):
    """Retain a panel's latest message and optional diagnostic details."""

    changed = Signal()

    def __init__(self, parent: QObject, text: str = "") -> None:
        super().__init__(parent)
        self._text = text
        self._tooltip = ""

    def text(self) -> str:
        return self._text

    @Slot(str)
    def setText(self, text: str) -> None:  # noqa: N802 - mirrors Qt's text API
        if text != self._text:
            self._text = text
            self._tooltip = ""
            self.changed.emit()

    def toolTip(self) -> str:  # noqa: N802 - mirrors Qt's tooltip API
        return self._tooltip

    def setToolTip(self, text: str) -> None:  # noqa: N802 - mirrors Qt's tooltip API
        if text != self._tooltip:
            self._tooltip = text
            self.changed.emit()


class PanelStatusPresenter(QObject):
    """Present the visible panel's messages in the main window's status bar.

    Each panel keeps its own state while hidden. Returning to a tab restores
    that message, and background progress cannot overwrite the visible panel.
    The presenter is owned by the panel, including when its dock is floated.
    """

    def __init__(
        self,
        panel: QWidget,
        status: PanelStatus,
        bar: QStatusBar,
        source: str,
    ) -> None:
        super().__init__(panel)
        self._panel = panel
        self._status = status
        self._bar: QStatusBar | None = bar
        self._source = source
        self._active = True
        self._published_text: str | None = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        panel.installEventFilter(self)
        status.changed.connect(self.refresh)
        bar.destroyed.connect(self._on_bar_destroyed)

    @Slot()
    def _on_bar_destroyed(self) -> None:
        self._bar = None
        self._refresh_timer.stop()

    @Slot(bool)
    def set_active(self, active: bool) -> None:
        self._active = active
        if active:
            self._refresh_timer.start(0)
        else:
            self._clear_message()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if watched is self._panel and event.type() == QEvent.Type.Show:
            self._refresh_timer.start(0)
        elif watched is self._panel and event.type() == QEvent.Type.Hide:
            self._clear_message()
        return super().eventFilter(watched, event)

    def _clear_message(self) -> None:
        bar = self._bar
        if (
            bar is not None
            and isValid(bar)
            and self._published_text is not None
            and bar.currentMessage() == self._published_text
        ):
            bar.clearMessage()
            bar.setToolTip("")

    @Slot()
    def refresh(self) -> None:
        bar = self._bar
        if bar is None or not isValid(bar) or not self._active or not self._panel.isVisible():
            return
        message = self._status.text()
        text = f"{self._source}: {message}" if message else ""
        self._published_text = text
        bar.showMessage(text)
        # QStatusBar clips long messages; the tooltip retains the complete
        # message and run-health details without contributing a size hint.
        bar.setToolTip("\n".join(filter(None, (text, self._status.toolTip()))))
