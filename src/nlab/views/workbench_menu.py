"""Small shared menu builder for modeless offline analysis workspaces."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMenu, QMenuBar, QMessageBox, QWidget

from nlab.help.context import HelpContext
from nlab.views.help_dialog import HelpDialog

log = logging.getLogger(__name__)


class WorkbenchMenuBar(QMenuBar):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setNativeMenuBar(False)
        self._menus: list[QMenu] = []
        self._headers: list[QAction] = []
        self.help_dialog: HelpDialog | None = None

    def menu(self, title: str) -> QMenu:
        menu = self.addMenu(title)
        self._menus.append(menu)
        # Retain both wrappers for PySide's QAction.menu() ownership behavior.
        self._headers.append(menu.menuAction())
        return menu

    def command(
        self, menu: QMenu, title: str, *, enabled: bool = True, shortcut: str = "",
    ) -> QAction:
        action = menu.addAction(title)
        action.setEnabled(enabled)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        return action

    def add_guide(self, topic: str) -> None:
        menu = self.menu("&Help")
        self.guide_action = self.command(menu, "Workspace guide...")
        self.guide_action.triggered.connect(lambda: self._show_guide(topic))

    def _show_guide(self, topic: str) -> None:
        owner = self.parentWidget()
        try:
            if self.help_dialog is None:
                self.help_dialog = HelpDialog(owner)
            self.help_dialog.show_topic(HelpContext(topic))
            self.help_dialog.show()
            self.help_dialog.raise_()
            self.help_dialog.activateWindow()
        except Exception:
            log.exception("Could not open workbench help for %s", topic)
            QMessageBox.warning(owner, "Help unavailable", "The bundled help could not be loaded. "
                                "See System Log for details.")
