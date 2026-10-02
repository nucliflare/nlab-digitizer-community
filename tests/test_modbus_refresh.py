from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, call

import pytest
from PySide6.QtWidgets import QMainWindow
from pytestqt.qtbot import QtBot

from nlab import app as app_module
from nlab.app import MainAppWindow
from nlab.controllers.main_window_controller import MainWindowController
from nlab.hardware import modbus_devices as modbus_devices_module
from nlab.hardware.modbus_devices import ExternalDevices, ExternalDeviceScan
from nlab.ui.ui_main_window import Ui_MainWindow


def test_connection_menu_contains_reconnect_and_modbus_refresh(qtbot: QtBot) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.actionReconnectDevice in ui.menuConnection.actions()
    assert ui.actionRefreshModbus in ui.menuConnection.actions()
    assert ui.actionReconnectDevice not in ui.menuFile.actions()
    assert ui.menuConnection.title() == "Connection"


def test_scan_new_uses_fresh_manager_and_both_remote_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [object(), object()]
    manager = SimpleNamespace(
        all_devices=devices,
        scan_remote=Mock(),
        by_type=Mock(),
        close_all=Mock(),
    )
    manager_factory = Mock(return_value=manager)
    monkeypatch.setattr(modbus_devices_module, "DeviceManager", manager_factory)

    scan = ExternalDevices.scan_new("board.local")

    manager_factory.assert_called_once_with()
    assert manager.scan_remote.call_args_list == [
        call("board.local", 5001),
        call("board.local", 5002),
    ]
    assert scan.devices == tuple(devices)
    assert scan.manager._manager is manager


def test_apply_scan_stops_old_pollers_and_rebuilds_external_docks() -> None:
    controller = cast(MainWindowController, object.__new__(MainWindowController))
    events: list[str] = []
    old_controller = SimpleNamespace(
        configuration_id="SIPM:3",
        configuration_settings=Mock(return_value={"refresh_interval_ms": 2000}),
        request_polling_stop=Mock(side_effect=lambda: events.append("request")),
        stop_polling_sync=Mock(side_effect=lambda: events.append("stop")),
    )
    new_controller = SimpleNamespace(
        configuration_id="SIPM:3",
        apply_configuration_settings=Mock(),
    )
    old_owner = SimpleNamespace(close=Mock(side_effect=lambda: events.append("close-old")))
    new_owner = SimpleNamespace(close=Mock())
    widget = SimpleNamespace(deleteLater=Mock())
    dock = SimpleNamespace(widget=Mock(return_value=widget), setParent=Mock(), deleteLater=Mock())
    host = SimpleNamespace(findChildren=Mock(return_value=[dock]), removeDockWidget=Mock())
    discovered = (SimpleNamespace(), SimpleNamespace())
    scan = ExternalDeviceScan(cast(ExternalDevices, new_owner), discovered)

    controller._external_controllers = [old_controller]
    controller._external_devices = cast(ExternalDevices, old_owner)
    controller._external_dock_host = host
    tab_external = SimpleNamespace(setToolTip=Mock())
    main_tabs = SimpleNamespace(indexOf=Mock(return_value=9), setTabEnabled=Mock())
    controller._window = SimpleNamespace(
        ui=SimpleNamespace(mainTabs=main_tabs, tabExternal=tab_external)
    )

    def build(devices: object) -> None:
        assert devices is discovered
        events.append("build")
        controller._external_controllers = [new_controller]

    controller._build_external_docks = Mock(side_effect=build)

    controller.prepare_external_device_refresh()
    count = controller.apply_external_device_scan(scan)

    assert count == 2
    assert events == ["request", "stop", "close-old", "build"]
    assert controller._external_devices is new_owner
    main_tabs.setTabEnabled.assert_any_call(9, False)
    tab_external.setToolTip.assert_any_call("Refreshing external Modbus devices…")
    host.removeDockWidget.assert_called_once_with(dock)
    dock.setParent.assert_called_once_with(None)
    widget.deleteLater.assert_called_once_with()
    dock.deleteLater.assert_called_once_with()
    new_owner.close.assert_not_called()
    new_controller.apply_configuration_settings.assert_called_once_with(
        {"refresh_interval_ms": 2000}
    )


def test_refresh_action_starts_background_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    refresh_action = SimpleNamespace(setEnabled=Mock())
    reconnect_action = SimpleNamespace(setEnabled=Mock())
    status_bar = SimpleNamespace(showMessage=Mock())
    scan_target = Mock()
    controller = SimpleNamespace(prepare_external_device_refresh=Mock())
    window = SimpleNamespace(
        _modbus_refresh_running=False,
        _scan_modbus_in_background=scan_target,
        _controller=controller,
        ui=SimpleNamespace(
            actionRefreshModbus=refresh_action,
            actionReconnectDevice=reconnect_action,
        ),
        statusBar=Mock(return_value=status_bar),
    )
    thread = SimpleNamespace(start=Mock())
    thread_factory = Mock(return_value=thread)
    monkeypatch.setattr(app_module.threading, "Thread", thread_factory)

    MainAppWindow._on_refresh_modbus_devices(cast(MainAppWindow, window))

    assert window._modbus_refresh_running
    refresh_action.setEnabled.assert_called_once_with(False)
    reconnect_action.setEnabled.assert_called_once_with(False)
    controller.prepare_external_device_refresh.assert_called_once_with()
    status_bar.showMessage.assert_called_once_with("Scanning for external Modbus devices…")
    assert thread_factory.call_args.kwargs == {
        "target": scan_target,
        "name": "modbus-discovery",
        "daemon": True,
    }
    thread.start.assert_called_once_with()


def test_completed_refresh_is_applied_and_actions_are_reenabled() -> None:
    refresh_action = SimpleNamespace(setEnabled=Mock())
    reconnect_action = SimpleNamespace(setEnabled=Mock())
    status_bar = SimpleNamespace(showMessage=Mock())
    controller = SimpleNamespace(apply_external_device_scan=Mock(return_value=2))
    owner = SimpleNamespace(close=Mock())
    scan = ExternalDeviceScan(cast(ExternalDevices, owner), (SimpleNamespace(),) * 2)
    window = SimpleNamespace(
        _closing=False,
        _modbus_refresh_running=True,
        _controller=controller,
        ui=SimpleNamespace(
            actionRefreshModbus=refresh_action,
            actionReconnectDevice=reconnect_action,
        ),
        statusBar=Mock(return_value=status_bar),
    )

    MainAppWindow._on_modbus_refresh_finished(cast(MainAppWindow, window), scan, None)

    assert not window._modbus_refresh_running
    refresh_action.setEnabled.assert_called_once_with(True)
    reconnect_action.setEnabled.assert_called_once_with(True)
    controller.apply_external_device_scan.assert_called_once_with(scan)
    status_bar.showMessage.assert_called_once_with("Found 2 external Modbus devices.", 10000)
    owner.close.assert_not_called()


def test_completed_refresh_is_discarded_while_window_is_closing() -> None:
    owner = SimpleNamespace(close=Mock())
    scan = ExternalDeviceScan(cast(ExternalDevices, owner), ())
    controller = SimpleNamespace(apply_external_device_scan=Mock())
    window = SimpleNamespace(
        _closing=True,
        _modbus_refresh_running=True,
        _controller=controller,
    )

    MainAppWindow._on_modbus_refresh_finished(cast(MainAppWindow, window), scan, None)

    owner.close.assert_called_once_with()
    controller.apply_external_device_scan.assert_not_called()
