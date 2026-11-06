from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING, cast

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget
from pytestqt.qtbot import QtBot

from nlab.controllers.main_window_controller import MainWindowController

if TYPE_CHECKING:
    from nlab.app import MainAppWindow


def _workspace(qtbot: QtBot) -> tuple[QMainWindow, MainWindowController]:
    host = MainWindowController._make_dock_host()
    qtbot.addWidget(host)
    controller = object.__new__(MainWindowController)
    controller._window = cast("MainAppWindow", host)
    host.resize(800, 600)
    return host, controller


def _assert_window_controls(dock: QDockWidget) -> None:
    assert dock.isFloating()
    assert dock.windowFlags() & Qt.WindowType.WindowType_Mask == Qt.WindowType.Window
    assert dock.windowFlags() & Qt.WindowType.WindowMinimizeButtonHint
    assert dock.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint
    assert dock.windowFlags() & Qt.WindowType.CustomizeWindowHint
    assert dock.windowFlags() & Qt.WindowType.WindowCloseButtonHint


@pytest.mark.parametrize("panel", ("scope", "current", "mca", "psd", "psu", "global", "external"))
def test_floating_panels_get_window_controls_and_keep_content(qtbot: QtBot, panel: str) -> None:
    host, controller = _workspace(qtbot)
    content = QWidget()
    dock = controller._make_dock(f"{panel}_ch0", "D1:CH0", content)
    features = dock.features()
    host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    host.show()
    original_margins = dock.contentsMargins()
    assert dock.findChild(QWidget, "dockBackButton") is None

    dock.setFloating(True)
    QApplication.processEvents()

    _assert_window_controls(dock)
    assert dock.isVisible()
    assert dock.widget() is content
    assert dock.contentsMargins() == original_margins
    assert content.geometry() == dock.contentsRect()
    assert dock.features() == features
    assert dock.parentWidget() is host
    assert not dock.features() & QDockWidget.DockWidgetFeature.DockWidgetClosable


@pytest.mark.parametrize("maximized", (False, True))
def test_floating_close_returns_and_selects_original_tab(qtbot: QtBot, maximized: bool) -> None:
    host, controller = _workspace(qtbot)
    content = QWidget()
    first = controller._make_dock("scope_ch0", "D1:CH0", content)
    second = controller._make_dock("scope_ch1", "D1:CH1", QWidget())
    controller._populate_dock_host(host, [first, second])
    host.show()
    original_margins = first.contentsMargins()
    first.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    first.setFloating(True)
    first.setGeometry(80, 100, 480, 320)
    QApplication.processEvents()
    original_geometry = first.geometry()

    for _ in range(2):
        if maximized:
            first.showMaximized()
        QApplication.processEvents()
        # Reject the actual close so even WA_DeleteOnClose cannot destroy
        # the measurement panel after returning it to its tab group.
        assert not first.close()
        QApplication.processEvents()

        assert not first.isFloating()
        assert not first.isMaximized()
        assert first.isVisible()
        assert first.widget() is content
        assert first.contentsMargins() == original_margins
        assert first in host.tabifiedDockWidgets(second)
        assert not first.visibleRegion().isEmpty()
        assert second.visibleRegion().isEmpty()

        first.setFloating(True)
        QApplication.processEvents()
        _assert_window_controls(first)
        assert first.geometry() == original_geometry


def test_floating_close_docks_into_an_inactive_workspace(qtbot: QtBot) -> None:
    host, controller = _workspace(qtbot)
    dock = controller._make_dock("psd_ch0", "D1:CH0", QWidget())
    host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    host.show()
    dock.setFloating(True)
    host.hide()
    dock.show()
    QApplication.processEvents()

    assert not dock.close()
    QApplication.processEvents()

    assert not dock.isFloating()
    assert dock.parentWidget() is host
    assert not dock.isHidden()
    host.show()
    QApplication.processEvents()
    assert dock.isVisible()


def test_floating_close_allows_application_shutdown(qtbot: QtBot) -> None:
    host, controller = _workspace(qtbot)
    dock = controller._make_dock("mca_ch0", "D1:CH0", QWidget())
    host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    host.show()
    dock.setFloating(True)
    QApplication.processEvents()
    # MainAppWindow sets this flag before stopping acquisition and closing.
    host._closing = True  # type: ignore[attr-defined]

    assert dock.close()
    QApplication.processEvents()

    assert dock.isFloating()
    assert dock.isHidden()


def test_docked_panel_still_cannot_be_closed(qtbot: QtBot) -> None:
    host, controller = _workspace(qtbot)
    dock = controller._make_dock("scope_ch0", "D1:CH0", QWidget())
    host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    host.show()
    QApplication.processEvents()

    assert not dock.close()
    assert dock.isVisible()
    assert not dock.isFloating()


def test_minimize_maximize_restore_are_independent_of_host(qtbot: QtBot) -> None:
    host, controller = _workspace(qtbot)
    dock = controller._make_dock("scope_ch0", "D1:CH0", QWidget())
    host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    host.show()
    dock.setFloating(True)
    dock.setGeometry(80, 100, 480, 320)
    QApplication.processEvents()
    original = dock.geometry()

    dock.showMaximized()
    QApplication.processEvents()
    assert dock.isMaximized()
    assert not host.isMaximized()
    dock.showNormal()
    QApplication.processEvents()
    assert dock.geometry() == original

    dock.showMinimized()
    QApplication.processEvents()
    assert dock.isMinimized()
    assert not host.isMinimized()
    dock.showNormal()
    QApplication.processEvents()
    assert dock.geometry() == original
    _assert_window_controls(dock)


def test_docked_title_bar_double_click_still_undocks(qtbot: QtBot) -> None:
    host, controller = _workspace(qtbot)
    dock = controller._make_dock("psd_ch0", "D1:CH0", QWidget())
    host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    host.show()
    QApplication.processEvents()

    QTest.mouseDClick(dock, Qt.MouseButton.LeftButton, pos=QPoint(70, 10))
    QApplication.processEvents()

    _assert_window_controls(dock)
    assert dock.isVisible()


def test_native_title_bar_double_click_does_not_redock(qtbot: QtBot) -> None:
    host, controller = _workspace(qtbot)
    first = controller._make_dock("scope_ch0", "D1:CH0", QWidget())
    second = controller._make_dock("scope_ch1", "D1:CH1", QWidget())
    controller._populate_dock_host(host, [first, second])
    host.show()
    first.setFloating(True)
    first.setGeometry(80, 100, 480, 320)
    QApplication.processEvents()
    original = first.geometry()
    transitions: list[bool] = []
    first.topLevelChanged.connect(transitions.append)

    for maximized in (False, True, False):
        # The native window manager owns maximize/restore. Qt's duplicate
        # non-client event must never move the panel back into the dock host.
        if maximized:
            first.showMaximized()
        else:
            first.showNormal()
        QApplication.processEvents()
        position = QPoint(100, -10)
        event = QMouseEvent(
            QEvent.Type.NonClientAreaMouseButtonDblClick,
            QPointF(position),
            QPointF(first.mapToGlobal(position)),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(first, event)
        QApplication.processEvents()

        assert first.isFloating()
        assert first.isVisible()
        assert first.isMaximized() == maximized
        assert first.parentWidget() is host
        if not maximized:
            assert first.geometry() == original
    assert transitions == []

    first.setFloating(False)
    QApplication.processEvents()
    assert first in host.tabifiedDockWidgets(second)


def test_repeated_undocking_retains_controls_geometry_and_tab_group(qtbot: QtBot) -> None:
    host, controller = _workspace(qtbot)
    first = controller._make_dock("mca_ch0", "D1:CH0", QWidget())
    second = controller._make_dock("mca_ch1", "D1:CH1", QWidget())
    controller._populate_dock_host(host, [first, second])
    host.show()
    first.setFloating(True)
    expected = QRect(80, 100, 480, 320)
    first.setGeometry(expected)

    for _ in range(3):
        first.setFloating(False)
        QApplication.processEvents()
        assert not first.isFloating()
        assert first in host.tabifiedDockWidgets(second)
        first.setFloating(True)
        QApplication.processEvents()
        _assert_window_controls(first)
        assert first.isVisible()
        assert first.geometry() == expected


def test_saved_layout_restores_floating_window_controls(qtbot: QtBot) -> None:
    host, controller = _workspace(qtbot)
    dock = controller._make_dock("psd_ch0", "D1:CH0", QWidget())
    host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    host.show()
    dock.setFloating(True)
    dock.setGeometry(80, 100, 480, 320)
    QApplication.processEvents()
    expected = dock.geometry()
    state = host.saveState()
    dock.setFloating(False)

    assert host.restoreState(state)
    QApplication.processEvents()

    _assert_window_controls(dock)
    assert dock.isVisible()
    assert dock.geometry() == expected


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows caption styles")
def test_native_windows_controls_and_double_click_without_showing_a_window() -> None:
    # The offscreen plugin cannot expose native caption styles. Use a separate
    # Windows-platform process, with both widgets kept off screen throughout.
    script = """
import ctypes
from ctypes import wintypes
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget
import nlab.controllers.main_window_controller as module
from nlab.views.panel_dock import PanelDockWidget

class ProbeDock(PanelDockWidget):
    def nativeEvent(self, event_type, message):
        msg = wintypes.MSG.from_address(int(message))
        if msg.message == 0x0112 and (msg.wParam & 0xfff0) in (0xf030, 0xf120):
            command = msg.wParam & 0xfff0
            commands.append(command)
            if command == 0xf030:
                self.showMaximized()
            else:
                self.showNormal()
            # Qt's off-screen attribute suppresses native ShowWindow. Mirror
            # only the zoomed style so DefWindowProc can choose restore next.
            handle = int(self.winId())
            style = user32.GetWindowLongW(handle, -16)
            if command == 0xf030:
                style |= 0x01000000  # WS_MAXIMIZE
            else:
                style &= ~0x01000000
            user32.SetWindowLongW(handle, -16, style)
            return True, 0
        return super().nativeEvent(event_type, message)

commands = []
module.PanelDockWidget = ProbeDock
app = QApplication([])
host = module.MainWindowController._make_dock_host()
host.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
controller = object.__new__(module.MainWindowController)
controller._window = host
dock = controller._make_dock('scope_ch0', 'Hidden caption check', QWidget())
dock.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
host.show()
user32 = ctypes.WinDLL('user32', use_last_error=True)
user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.SendMessageW.restype = wintypes.LPARAM
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
user32.SetWindowLongW.restype = ctypes.c_long
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetSystemMenu.argtypes = [wintypes.HWND, wintypes.BOOL]
user32.GetSystemMenu.restype = wintypes.HMENU
user32.GetMenuState.argtypes = [wintypes.HMENU, wintypes.UINT, wintypes.UINT]
user32.GetMenuState.restype = wintypes.UINT
try:
    dock.setFloating(True)
    dock.setGeometry(100, 100, 480, 320)
    dock.show()
    app.processEvents()
    handle = int(dock.winId())
    style = user32.GetWindowLongW(handle, -16) & 0xffffffff
    assert style & 0x00020000, 'Missing WS_MINIMIZEBOX'
    assert style & 0x00010000, 'Missing WS_MAXIMIZEBOX'
    menu = user32.GetSystemMenu(handle, False)
    close_state = user32.GetMenuState(menu, 0xf060, 0)  # SC_CLOSE, MF_BYCOMMAND
    assert close_state != 0xffffffff and not close_state & 0x3, 'Caption X is disabled'
    for i in range(4):
        rect = wintypes.RECT()
        assert user32.GetWindowRect(handle, ctypes.byref(rect))
        point = ((rect.top + 10) & 0xffff) << 16 | ((rect.left + 100) & 0xffff)
        user32.SendMessageW(handle, 0x00a3, 2, point)  # WM_NCLBUTTONDBLCLK, HTCAPTION
        app.processEvents()
        assert commands == [0xf030, 0xf120, 0xf030, 0xf120][:i + 1]
        assert dock.isMaximized() == (i % 2 == 0)
        assert dock.isVisible()
        assert dock.isFloating()
        assert not user32.IsWindowVisible(handle), 'Test window must stay hidden'
    for maximized in (False, True):
        dock.setFloating(True)
        dock.setGeometry(100, 100, 480, 320)
        if maximized:
            dock.showMaximized()
        app.processEvents()
        handle = int(dock.winId())
        user32.SendMessageW(handle, 0x0112, 0xf060, 0)  # WM_SYSCOMMAND, SC_CLOSE
        app.processEvents()
        assert not dock.isFloating(), 'Caption X must dock back'
        assert dock.isVisible(), 'Docked panel must stay visible'
        assert not dock.isMaximized()
        assert not user32.IsWindowVisible(int(host.winId())), 'Test host must stay hidden'
finally:
    dock.close()
    host.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "QT_QPA_PLATFORM": "windows"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
