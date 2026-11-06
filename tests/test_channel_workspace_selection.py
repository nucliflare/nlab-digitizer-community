from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from pytestqt.qtbot import QtBot

from nlab.controllers.main_window_controller import MainWindowController

if TYPE_CHECKING:
    from nlab.app import MainAppWindow

_CHANNEL_WORKSPACES = ("Scope", "Current", "MCA", "PSD", "PSU")


@dataclass
class _Workspace:
    controller: MainWindowController
    tabs: QTabWidget
    hosts: dict[str, QMainWindow]
    pages: dict[str, QWidget]
    docks: dict[str, dict[int, QDockWidget]]

    def switch(self, name: str) -> None:
        self.tabs.setCurrentWidget(self.pages[name])
        QApplication.processEvents()

    def select_channel(self, name: str, channel: int) -> None:
        bar = self.hosts[name].findChild(QTabBar)
        assert bar is not None
        title = self.docks[name][channel].windowTitle()
        index = next(index for index in range(bar.count()) if bar.tabText(index) == title)
        QTest.mouseClick(bar, Qt.MouseButton.LeftButton, pos=bar.tabRect(index).center())
        self.assert_channel(name, channel)

    def assert_channel(self, name: str, channel: int) -> None:
        QApplication.processEvents()
        visible = [
            ch for ch, dock in self.docks[name].items() if not dock.visibleRegion().isEmpty()
        ]
        assert visible == [channel]


@pytest.fixture
def workspace_factory(qtbot: QtBot) -> Callable[..., _Workspace]:
    def create(*, initial_channel: int = 0, initial_workspace: str = "Scope") -> _Workspace:
        # Exercise real dock/tab signals without constructing hardware or workers.
        window = QMainWindow()
        qtbot.addWidget(window)
        tabs = QTabWidget()
        window.setCentralWidget(tabs)
        controller = object.__new__(MainWindowController)
        controller._window = cast("MainAppWindow", window)
        controller._selected_channel = None
        controller._active_channel_dock_host = None
        ui = SimpleNamespace(mainTabs=tabs)
        window.ui = ui  # type: ignore[attr-defined]
        hosts: dict[str, QMainWindow] = {}
        pages: dict[str, QWidget] = {}
        docks: dict[str, dict[int, QDockWidget]] = {}
        for name in (*_CHANNEL_WORKSPACES, "Global", "Coincidence", "External", "SystemLog"):
            page = QWidget()
            pages[name] = page
            setattr(ui, f"tab{name}", page)
            tabs.addTab(page, name)
            if name not in _CHANNEL_WORKSPACES:
                continue
            host = controller._make_dock_host()
            hosts[name] = host
            setattr(controller, f"_{name.lower()}_dock_host", host)
            layout = QVBoxLayout(page)
            layout.addWidget(host)
            docks[name] = {}
            for channel in range(4):
                # Two boards with CH0/CH1: IDs must distinguish the two CH1s.
                title = f"D{channel // 2 + 1}:CH{channel % 2}"
                docks[name][channel] = controller._make_dock(
                    f"{name.lower()}_ch{channel}", title, QWidget()
                )
            controller._populate_dock_host(host, list(docks[name].values()))
            docks[name][initial_channel if name == "Scope" else 0].raise_()
        tabs.setCurrentWidget(pages[initial_workspace])
        controller._connect_signals()
        window.resize(1000, 650)
        window.show()
        QApplication.processEvents()
        return _Workspace(controller, tabs, hosts, pages, docks)

    return create


@pytest.mark.parametrize("source", _CHANNEL_WORKSPACES)
@pytest.mark.parametrize("channel", (0, 1, 3))
def test_channel_follows_workspace_switches(
    workspace_factory: Callable[..., _Workspace],
    source: str,
    channel: int,
) -> None:
    workspace = workspace_factory()
    workspace.switch(source)
    workspace.select_channel(source, (channel + 1) % 4)
    workspace.select_channel(source, channel)

    for destination in (*_CHANNEL_WORKSPACES, source):
        workspace.switch(destination)
        workspace.assert_channel(destination, channel)


def test_destination_channel_changes_are_used_on_next_switch(
    workspace_factory: Callable[..., _Workspace],
) -> None:
    workspace = workspace_factory()
    workspace.select_channel("Scope", 1)
    workspace.switch("PSD")
    workspace.assert_channel("PSD", 1)
    workspace.select_channel("PSD", 0)
    workspace.switch("MCA")
    workspace.assert_channel("MCA", 0)
    workspace.switch("Scope")
    workspace.assert_channel("Scope", 0)


def test_restored_channel_is_followed_before_any_channel_click(
    workspace_factory: Callable[..., _Workspace],
) -> None:
    workspace = workspace_factory(initial_channel=1)
    workspace.switch("PSD")
    workspace.assert_channel("PSD", 1)


def test_first_channel_workspace_seeds_selection_after_starting_on_global(
    workspace_factory: Callable[..., _Workspace],
) -> None:
    workspace = workspace_factory(initial_channel=1, initial_workspace="Global")
    workspace.switch("Scope")
    workspace.assert_channel("Scope", 1)
    workspace.switch("PSD")
    workspace.assert_channel("PSD", 1)


def test_reordered_tabs_follow_channel_identity(
    workspace_factory: Callable[..., _Workspace],
) -> None:
    workspace = workspace_factory()
    bar = workspace.hosts["PSD"].findChild(QTabBar)
    assert bar is not None
    bar.moveTab(3, 0)
    workspace.select_channel("Scope", 3)

    workspace.switch("PSD")
    workspace.assert_channel("PSD", 3)
    workspace.select_channel("PSD", 0)
    workspace.switch("Current")
    workspace.assert_channel("Current", 0)


def test_missing_channel_keeps_view_and_remembers_requested_channel(
    workspace_factory: Callable[..., _Workspace],
) -> None:
    workspace = workspace_factory()
    workspace.switch("PSD")
    workspace.select_channel("PSD", 1)
    workspace.switch("Scope")
    for channel in (0, 2):
        dock = workspace.docks["PSD"].pop(channel)
        workspace.hosts["PSD"].removeDockWidget(dock)
        dock.setParent(None)
        dock.deleteLater()
    workspace.select_channel("Scope", 0)

    workspace.switch("PSD")
    workspace.assert_channel("PSD", 1)
    workspace.switch("MCA")
    workspace.assert_channel("MCA", 0)

    # An explicit choice in the limited workspace becomes the shared channel.
    workspace.switch("PSD")
    workspace.select_channel("PSD", 3)
    workspace.switch("Scope")
    workspace.assert_channel("Scope", 3)


def test_other_workspaces_and_background_docks_do_not_replace_selection(
    workspace_factory: Callable[..., _Workspace],
) -> None:
    workspace = workspace_factory()
    workspace.select_channel("Scope", 3)
    workspace.docks["MCA"][1].raise_()
    for name in ("Global", "Coincidence", "External", "SystemLog"):
        workspace.switch(name)
        workspace.docks["PSD"][0].raise_()
    workspace.switch("PSD")
    workspace.assert_channel("PSD", 3)


def test_disabled_workspace_does_not_prevent_following_other_views(
    workspace_factory: Callable[..., _Workspace],
) -> None:
    workspace = workspace_factory()
    workspace.tabs.setTabEnabled(workspace.tabs.indexOf(workspace.pages["PSU"]), False)
    workspace.select_channel("Scope", 1)
    workspace.switch("MCA")
    workspace.assert_channel("MCA", 1)
    workspace.switch("PSD")
    workspace.assert_channel("PSD", 1)
    assert not workspace.tabs.isTabEnabled(workspace.tabs.indexOf(workspace.pages["PSU"]))
