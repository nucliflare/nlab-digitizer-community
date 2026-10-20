"""F1 click-to-explain mode, isolated from acquisition and controller writes."""

from __future__ import annotations

import logging
from typing import ClassVar

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget

from nlab.help.context import HelpContext, resolve_context
from nlab.views.help_dialog import HelpDialog

log = logging.getLogger(__name__)
_MOUSE_EVENTS = {
    QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
    QEvent.Type.MouseButtonDblClick, QEvent.Type.MouseMove,
    QEvent.Type.Wheel, QEvent.Type.ContextMenu,
    QEvent.Type.TouchBegin, QEvent.Type.TouchUpdate, QEvent.Type.TouchEnd,
}


class HelpController(QObject):
    """Consume selection clicks in this workspace, including its owned dialogs.

    Keep selection mode armed through release: consuming just the press can still
    deliver a release to a checkbox, plot, or acquisition button. Normal input is
    untouched outside this explicit mode. No hardware APIs are called here.
    """

    _active_controller: ClassVar[HelpController | None] = None

    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.active = False
        self._pressed = False
        self._disposed = False
        self._dialogs: dict[QWidget, HelpDialog] = {}
        application = QApplication.instance()
        if not isinstance(application, QApplication):
            raise RuntimeError("Context help requires QApplication")
        self._application = application
        self.guide_action = QAction("User Guide…", self)
        self.guide_action.setObjectName("actionUserGuide")
        self.guide_action.triggered.connect(lambda: self.open_topic(HelpContext("help")))
        self.context_action = QAction("Explain a Control (F1)", self)
        self.context_action.setObjectName("actionContextHelp")
        self.context_action.setCheckable(True)
        # Handle F1 at the application filter so modal child dialogs work too,
        # without an application-wide QAction shortcut stealing another window's F1.
        self.context_action.triggered.connect(self.set_active)
        application.installEventFilter(self)
        window.destroyed.connect(self.dispose)

    def owns(self, widget: QWidget) -> bool:
        current: QWidget | None = widget
        while current is not None:
            if current is self.window:
                return True
            current = current.parentWidget()
        return False

    def set_active(self, active: bool) -> None:
        if active == self.active:
            return
        if active and self._active_controller is not None:
            self._active_controller.set_active(False)
        self.active = active
        self._pressed = False
        self.context_action.setChecked(active)
        if active:
            HelpController._active_controller = self
            # Dialogs created since startup may install plot shortcut filters.
            # Qt moves an existing filter to the front when installed again.
            self._application.installEventFilter(self)
            QApplication.setOverrideCursor(Qt.CursorShape.WhatsThisCursor)
            self.window.statusBar().showMessage("Help mode: click a label or control. Esc cancels.")
        else:
            if self._active_controller is self:
                HelpController._active_controller = None
            QApplication.restoreOverrideCursor()
            # Do not erase an acquisition status posted since help mode began.
            status = self.window.statusBar()
            if status.currentMessage() == "Help mode: click a label or control. Esc cancels.":
                status.clearMessage()

    def open_topic(self, context: HelpContext, owner: QWidget | None = None) -> None:
        self.set_active(False)
        owner = owner or self.window
        if owner in self._dialogs.values():
            owner = owner.parentWidget() or self.window
        dialog = self._dialogs.get(owner)
        try:
            if dialog is None:
                dialog = HelpDialog(owner)
                self._dialogs[owner] = dialog
                owner.destroyed.connect(lambda: self._dialogs.pop(owner, None))
            dialog.show_topic(context)
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        except Exception:
            log.exception("Could not open offline help")
            QMessageBox.warning(owner, "Help unavailable", "The bundled help could not be loaded. "
                                "See System Log for details.")

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        # Also runs on QObject destruction, when the window's C++ wrapper may
        # already be invalid. Do not dereference widgets on this path.
        if self.active:
            self.active = False
            if self._active_controller is self:
                HelpController._active_controller = None
            QApplication.restoreOverrideCursor()
        self._application.removeEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.ApplicationDeactivate:
            self.set_active(False)
            return False
        if not isinstance(watched, QWidget):
            return False
        if not self.owns(watched):
            if self.active and event.type() == QEvent.Type.MouseButtonPress:
                self.set_active(False)
            return False
        kind = event.type()
        if kind == QEvent.Type.Close:
            self.set_active(False)
            return False
        if isinstance(event, QKeyEvent):
            f1 = (
                event.key() == Qt.Key.Key_F1
                and event.modifiers() == Qt.KeyboardModifier.NoModifier
            )
            if kind == QEvent.Type.ShortcutOverride and (self.active or f1):
                event.accept()
                return True
            if kind == QEvent.Type.KeyPress and f1:
                if not event.isAutoRepeat():
                    self.set_active(not self.active)
                return True
            if self.active and kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                if event.key() == Qt.Key.Key_Escape:
                    self.set_active(False)
                return True
        if not self.active:
            return False
        if kind in _MOUSE_EVENTS:
            if isinstance(event, QMouseEvent):
                if kind == QEvent.Type.MouseButtonPress:
                    self._pressed = True
                elif kind == QEvent.Type.MouseButtonRelease and self._pressed:
                    if event.button() == Qt.MouseButton.LeftButton:
                        target = QApplication.widgetAt(event.globalPosition().toPoint())
                        if target is None or not self.owns(target):
                            target = watched
                        # Disabled widgets may route their events to an enabled parent.
                        position = target.mapFromGlobal(event.globalPosition().toPoint())
                        child = target.childAt(position)
                        if child is not None:
                            target = child
                        self.open_topic(resolve_context(target), owner=target.window())
                    else:
                        self.set_active(False)
            return True
        return super().eventFilter(watched, event)
