"""Dock panels with standard native floating-window title-bar behavior."""

from __future__ import annotations

from typing import override

from PySide6.QtCore import QEvent
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QDockWidget


class PanelDockWidget(QDockWidget):
    @override
    def closeEvent(self, event: QCloseEvent) -> None:
        owner = self.parentWidget()
        if not self.isFloating() or owner is None:
            super().closeEvent(event)
            return
        # Application shutdown already stops the panel's workers. Let its
        # floating windows close without changing the saved dock layout.
        if QApplication.closingDown() or getattr(owner.window(), "_closing", False):
            event.accept()
            return
        # The floating caption's X means "dock back". Reject the actual close
        # so QWidget cannot hide or delete the panel after it has been docked.
        event.ignore()
        if self.isMaximized() or self.isMinimized():
            normal_geometry = self.normalGeometry()
            self.showNormal()
            # Restore the geometry before Qt records it for the next undock;
            # a window-manager resize may otherwise arrive after redocking.
            if normal_geometry.isValid():
                self.setGeometry(normal_geometry)
        self.setFloating(False)
        self.show()
        self.raise_()

    def event(self, event: QEvent) -> bool:
        if self.isFloating() and event.type() == QEvent.Type.NonClientAreaMouseButtonDblClick:
            # QDockWidget otherwise redocks on the same double-click that the
            # native window manager handles as maximize/restore. Leave the
            # native action alone and prevent the duplicate docking action.
            return True
        return super().event(event)
