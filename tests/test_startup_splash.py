from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from nlab.app import MainAppWindow

main_module = importlib.import_module("nlab.main")
app_module = importlib.import_module("nlab.app")


def test_launch_and_connection_splashes_share_design(qtbot: QtBot) -> None:
    launch_splash = main_module._show_splash()
    splash = main_module._show_connection_splash()
    qtbot.addWidget(launch_splash)
    qtbot.addWidget(splash)

    assert launch_splash.isVisible()
    assert launch_splash.message() == "Launching application"
    assert splash.isVisible()
    assert splash.message() == "Connecting to device..."
    assert launch_splash.pixmap().toImage() == splash.pixmap().toImage()
    main_module._update_splash(splash, "Initializing channel 0 controls...")
    assert splash.message() == "Initializing channel 0 controls..."
    assert splash.pixmap().size().width() == 520
    launch_splash.close()
    splash.close()


def test_main_window_reports_setup_and_saved_settings_stages(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages: list[str] = []
    controller = SimpleNamespace(load_all_settings=Mock(), shutdown=Mock())
    factory = Mock(return_value=controller)
    monkeypatch.setattr(MainAppWindow, "_setup_ui", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_save_developer_settings", lambda _self: None)
    monkeypatch.setattr(app_module, "MainWindowController", factory)

    window = MainAppWindow(config_path=Path("settings.yaml"), on_progress=stages.append)
    qtbot.addWidget(window)

    assert stages == [
        "Preparing the main window...",
        "Applying saved settings...",
        "Restoring the workspace...",
    ]
    assert factory.call_args.kwargs["on_progress"] == stages.append
    controller.load_all_settings.assert_called_once_with(Path("settings.yaml"))


def test_offline_window_is_identified_and_hardware_actions_are_disabled(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = Mock()
    factory = Mock(return_value=controller)
    monkeypatch.setattr(app_module, "MainWindowController", factory)
    monkeypatch.setattr(MainAppWindow, "_apply_stored_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_persist_current_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", lambda _self: None)

    window = MainAppWindow(channels=2, offline=True)
    qtbot.addWidget(window)

    assert window.windowTitle() == "Nuclear Lab Digitizer — Offline"
    assert not window._offline_indicator.isHidden()
    assert window._offline_indicator.text() == "OFFLINE"
    assert window.ui.actionReplaySession.isEnabled()
    assert window.ui.actionConnectDevice.isEnabled()
    for action in (
        window.ui.actionStartSessionRecording,
        window.ui.actionSaveSettings,
        window.ui.actionLoadSettings,
        window.ui.actionDisconnectDevice,
        window.ui.actionReconnectDevice,
        window.ui.actionRefreshModbus,
        window.ui.actionRebootBoard,
        window.ui.actionShutdownBoard,
    ):
        assert not action.isEnabled()
    assert factory.call_args.kwargs["offline"] is True


def test_energy_calibration_status_badge_opens_calibration_tool(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = Mock()
    controller.energy_calibrated_channel_labels.return_value = ()
    monkeypatch.setattr(app_module, "MainWindowController", Mock(return_value=controller))
    monkeypatch.setattr(MainAppWindow, "_apply_stored_general_settings", lambda _self: None)
    monkeypatch.setattr(
        MainAppWindow,
        "_persist_current_general_settings",
        lambda _self: None,
    )
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", lambda _self: None)

    window = MainAppWindow(channels=2, offline=True)
    qtbot.addWidget(window)
    indicator = window._energy_calibration_indicator

    assert indicator.isHidden()
    assert indicator.text() == "ENERGY CALIBRATED"

    controller.energy_calibrated_channel_labels.return_value = ("D1:CH0", "D1:CH1")
    window.refresh_energy_calibration_indicator()
    window.show()

    assert indicator.isVisibleTo(window)
    assert "D1:CH0, D1:CH1" in indicator.toolTip()
    qtbot.mouseDClick(indicator, Qt.MouseButton.LeftButton)
    controller.show_energy_calibration.assert_called_once_with()

    controller.energy_calibrated_channel_labels.return_value = ()
    window.refresh_energy_calibration_indicator()
    assert indicator.isHidden()


def test_offline_window_can_connect_without_restarting(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = Mock(
        session_replay=None,
        session_replay_path=None,
        offline=False,
        channel_count=2,
        endpoint_descriptions=("digitizer.local:30431",),
        primary_endpoint=("digitizer.local", 30431),
    )
    factory = Mock(return_value=controller)
    apply_general = Mock()
    persist_general = Mock()
    apply_view = Mock()

    class FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)
        ip = "digitizer.local"
        port = 30431
        channels = 2
        offline = False

        def __init__(self, _parent: object, *, channels: int) -> None:
            assert channels == 2

        @staticmethod
        def exec() -> int:
            return 1

    monkeypatch.setattr(app_module, "MainWindowController", factory)
    monkeypatch.setattr(app_module, "ConnectionDialog", FakeDialog)
    monkeypatch.setattr(app_module, "auto_configuration_enabled", lambda: False)
    monkeypatch.setattr(MainAppWindow, "_apply_stored_general_settings", apply_general)
    monkeypatch.setattr(MainAppWindow, "_persist_current_general_settings", persist_general)
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", apply_view)
    information = Mock()
    monkeypatch.setattr(app_module.QMessageBox, "information", information)

    window = MainAppWindow(channels=2, offline=True)
    qtbot.addWidget(window)
    apply_general.reset_mock()
    persist_general.reset_mock()
    apply_view.reset_mock()

    window._on_connect_device()

    call_args = controller.connect_to_device.call_args
    assert call_args.args == ("digitizer.local", 30431, 2)
    assert callable(call_args.kwargs["on_progress"])
    assert not window._offline
    assert window._host == "digitizer.local"
    assert window.windowTitle().endswith("digitizer.local:30431")
    assert window._offline_indicator.isHidden()
    assert window.ui.actionConnectDevice.isEnabled()
    assert window.ui.actionConnectDevice.text() == "Add Digitizer..."
    assert window.ui.actionReconnectDevice.isEnabled()
    assert window.ui.actionStartSessionRecording.isEnabled()
    apply_general.assert_called_once_with()
    persist_general.assert_called_once_with()
    apply_view.assert_called_once_with()
    information.assert_called_once()


def test_live_connect_action_adds_another_digitizer(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = Mock(
        session_replay=None,
        session_replay_path=None,
        offline=False,
        channel_count=4,
        primary_endpoint=("board-1.local", 30431),
        endpoint_descriptions=("board-1.local:30431", "board-2.local:30431"),
    )
    controller.add_digitizer.return_value = 2

    class FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)
        ip = "board-2.local"
        port = 30431
        channels = 2
        offline = False

        def __init__(self, _parent: object, *, allow_offline: bool) -> None:
            assert not allow_offline

        @staticmethod
        def exec() -> int:
            return 1

    monkeypatch.setattr(app_module, "MainWindowController", Mock(return_value=controller))
    monkeypatch.setattr(app_module, "ConnectionDialog", FakeDialog)
    monkeypatch.setattr(app_module, "auto_configuration_enabled", lambda: False)
    monkeypatch.setattr(MainAppWindow, "_apply_stored_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_persist_current_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", lambda _self: None)
    information = Mock()
    monkeypatch.setattr(app_module.QMessageBox, "information", information)

    window = MainAppWindow(host="board-1.local", channels=2)
    qtbot.addWidget(window)
    window._on_connect_device()

    call_args = controller.add_digitizer.call_args
    assert call_args.args == ("board-2.local", 30431, 2)
    assert callable(call_args.kwargs["on_progress"])
    assert window._channels == 4
    assert window.ui.actionConnectDevice.text() == "Add Digitizer..."
    assert "D1 board-1.local:30431; D2 board-2.local:30431" in window.windowTitle()
    information.assert_called_once()


def test_live_disconnect_selects_a_digitizer_and_renumbers_remaining_board(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = Mock(
        session_replay=None,
        session_replay_path=None,
        offline=False,
        endpoint_descriptions=("board-1.local:30431", "board-2.local:30431"),
        channel_count=4,
        primary_endpoint=("board-1.local", 30431),
    )

    def disconnect(digitizer: int, *, on_progress: object) -> bool:
        assert digitizer == 1
        assert callable(on_progress)
        controller.endpoint_descriptions = ("board-1.local:30431",)
        controller.channel_count = 2
        controller.primary_endpoint = ("board-1.local", 30431)
        controller.offline = False
        return False

    controller.disconnect_digitizer.side_effect = disconnect
    monkeypatch.setattr(app_module, "MainWindowController", Mock(return_value=controller))
    monkeypatch.setattr(MainAppWindow, "_apply_stored_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_persist_current_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", lambda _self: None)
    monkeypatch.setattr(
        app_module.QInputDialog,
        "getItem",
        Mock(return_value=("D2 — board-2.local:30431", True)),
    )
    monkeypatch.setattr(
        app_module.QMessageBox,
        "question",
        Mock(return_value=app_module.QMessageBox.StandardButton.Yes),
    )
    information = Mock()
    monkeypatch.setattr(app_module.QMessageBox, "information", information)

    window = MainAppWindow(host="board-1.local", channels=4)
    qtbot.addWidget(window)
    window._on_disconnect_device()

    controller.disconnect_digitizer.assert_called_once()
    assert controller.disconnect_digitizer.call_args.args == (1,)
    assert window._channels == 2
    assert not window._offline
    assert window._host == "board-1.local"
    assert window.windowTitle().endswith("board-1.local:30431")
    assert window.ui.actionDisconnectDevice.isEnabled()
    information.assert_called_once()


def test_disconnecting_final_digitizer_returns_window_to_offline_mode(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = Mock(
        session_replay=None,
        session_replay_path=None,
        offline=False,
        endpoint_descriptions=("board.local:30431",),
        channel_count=2,
        primary_endpoint=("board.local", 30431),
    )

    def disconnect(digitizer: int, *, on_progress: object) -> bool:
        assert digitizer == 0
        assert callable(on_progress)
        controller.endpoint_descriptions = ()
        controller.primary_endpoint = None
        controller.offline = True
        return True

    controller.disconnect_digitizer.side_effect = disconnect
    monkeypatch.setattr(app_module, "MainWindowController", Mock(return_value=controller))
    monkeypatch.setattr(MainAppWindow, "_apply_stored_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_persist_current_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", lambda _self: None)
    item_dialog = Mock()
    monkeypatch.setattr(app_module.QInputDialog, "getItem", item_dialog)
    monkeypatch.setattr(
        app_module.QMessageBox,
        "question",
        Mock(return_value=app_module.QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(app_module.QMessageBox, "information", Mock())

    window = MainAppWindow(host="board.local", channels=2)
    qtbot.addWidget(window)
    window._on_disconnect_device()

    item_dialog.assert_not_called()
    assert window._offline
    assert window._host == ""
    assert window.windowTitle().endswith("Offline")
    assert not window._offline_indicator.isHidden()
    assert window.ui.actionConnectDevice.text() == "Connect..."
    assert not window.ui.actionDisconnectDevice.isEnabled()
    assert not window.ui.actionReconnectDevice.isEnabled()


def test_reconnect_uses_progress_and_restores_connection_actions(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = Mock(
        session_replay=None,
        session_replay_path=None,
        offline=False,
        channel_count=2,
        endpoint_descriptions=("board.local:30431",),
        primary_endpoint=("board.local", 30431),
    )
    action_state_during_reconnect: dict[str, bool] = {}
    window: MainAppWindow

    def reconnect(*, on_progress: object) -> None:
        assert callable(on_progress)
        action_state_during_reconnect.update(
            connect=window.ui.actionConnectDevice.isEnabled(),
            disconnect=window.ui.actionDisconnectDevice.isEnabled(),
            reconnect=window.ui.actionReconnectDevice.isEnabled(),
            refresh=window.ui.actionRefreshModbus.isEnabled(),
        )

    controller.reconnect.side_effect = reconnect
    monkeypatch.setattr(app_module, "MainWindowController", Mock(return_value=controller))
    monkeypatch.setattr(MainAppWindow, "_apply_stored_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_persist_current_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", lambda _self: None)
    monkeypatch.setattr(
        app_module.QMessageBox,
        "question",
        Mock(return_value=app_module.QMessageBox.StandardButton.Yes),
    )
    information = Mock()
    monkeypatch.setattr(app_module.QMessageBox, "information", information)

    window = MainAppWindow(host="board.local", channels=2)
    qtbot.addWidget(window)
    window._on_reconnect_device()

    assert action_state_during_reconnect == {
        "connect": False,
        "disconnect": False,
        "reconnect": False,
        "refresh": False,
    }
    controller.reconnect.assert_called_once()
    assert window.ui.mainTabs.isEnabled()
    assert window.ui.actionConnectDevice.isEnabled()
    assert window.ui.actionDisconnectDevice.isEnabled()
    assert window.ui.actionReconnectDevice.isEnabled()
    assert window.ui.actionRefreshModbus.isEnabled()
    information.assert_called_once_with(
        window,
        "Reconnect",
        "Device reconnected successfully.",
    )


def test_reconnect_failure_keeps_existing_live_presentation(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = Mock(
        session_replay=None,
        session_replay_path=None,
        offline=False,
        channel_count=2,
        endpoint_descriptions=("board.local:30431",),
        primary_endpoint=("board.local", 30431),
    )
    controller.reconnect.side_effect = RuntimeError("replacement unavailable")
    monkeypatch.setattr(app_module, "MainWindowController", Mock(return_value=controller))
    monkeypatch.setattr(MainAppWindow, "_apply_stored_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_persist_current_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", lambda _self: None)
    monkeypatch.setattr(
        app_module.QMessageBox,
        "question",
        Mock(return_value=app_module.QMessageBox.StandardButton.Yes),
    )
    critical = Mock()
    monkeypatch.setattr(app_module.QMessageBox, "critical", critical)

    window = MainAppWindow(host="board.local", channels=2)
    qtbot.addWidget(window)
    window._on_reconnect_device()

    assert not window._offline
    assert window.windowTitle().endswith("board.local:30431")
    assert window.ui.mainTabs.isEnabled()
    assert window.ui.actionReconnectDevice.isEnabled()
    critical.assert_called_once_with(
        window,
        "Reconnect Failed",
        "replacement unavailable",
    )


def test_failed_offline_connection_keeps_offline_actions_available(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = Mock(session_replay=None, session_replay_path=None)
    controller.connect_to_device.side_effect = RuntimeError("device unavailable")

    class FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)
        ip = "missing.local"
        port = 30431
        channels = 2
        offline = False

        def __init__(self, _parent: object, *, channels: int) -> None:
            assert channels == 2

        @staticmethod
        def exec() -> int:
            return 1

    monkeypatch.setattr(app_module, "MainWindowController", Mock(return_value=controller))
    monkeypatch.setattr(app_module, "ConnectionDialog", FakeDialog)
    monkeypatch.setattr(MainAppWindow, "_apply_stored_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_persist_current_general_settings", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", lambda _self: None)
    critical = Mock()
    monkeypatch.setattr(app_module.QMessageBox, "critical", critical)

    window = MainAppWindow(channels=2, offline=True)
    qtbot.addWidget(window)

    window._on_connect_device()

    assert window._offline
    assert window._host == ""
    assert window._offline_indicator.isVisibleTo(window)
    assert window.ui.actionConnectDevice.isEnabled()
    assert not window.ui.actionReconnectDevice.isEnabled()
    critical.assert_called_once_with(window, "Connection Failed", "device unavailable")


def _prepare_fake_main(monkeypatch: pytest.MonkeyPatch) -> tuple[list[object], Mock, Mock]:
    events: list[object] = []

    class FakeApp:
        def __init__(self, _argv: list[str]) -> None:
            events.append("app-created")

        @staticmethod
        def processEvents(*_args: object) -> None:  # noqa: N802
            events.append("events-processed")

        def setWindowIcon(self, _icon: object) -> None:  # noqa: N802
            pass

        def setApplicationName(self, _name: str) -> None:  # noqa: N802
            pass

        def setApplicationVersion(self, _version: str) -> None:  # noqa: N802
            pass

        def setOrganizationName(self, _name: str) -> None:  # noqa: N802
            pass

        def setOrganizationDomain(self, _domain: str) -> None:  # noqa: N802
            pass

        def exec(self) -> int:
            events.append("app-exec")
            return 0

    class FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)
        ip = "192.0.2.10"
        port = 30431
        channels = 2

        def __init__(self, **_kwargs: object) -> None:
            events.append("dialog-created")

        def exec(self) -> int:
            events.append("dialog-accepted")
            return 1

    first_splash = Mock()
    connection_splash = Mock()
    monkeypatch.setattr(main_module, "QApplication", FakeApp)
    monkeypatch.setattr(main_module, "ConnectionDialog", FakeDialog)
    monkeypatch.setattr(main_module, "_show_splash", lambda: first_splash)
    monkeypatch.setattr(main_module, "_show_connection_splash", lambda: connection_splash)
    monkeypatch.setattr(
        main_module,
        "_update_splash",
        lambda _splash, message: events.append(("progress", message)),
    )
    monkeypatch.setattr(main_module, "_set_windows_app_user_model_id", lambda: None)
    monkeypatch.setattr(main_module, "apply_taskbar_icon", lambda _window: None)
    monkeypatch.setattr(main_module, "QTimer", SimpleNamespace(singleShot=lambda *_args: None))
    monkeypatch.setattr(sys, "argv", ["nlab"])
    return events, first_splash, connection_splash


def test_connect_shows_second_splash_until_main_window_is_shown(
    monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    events, first_splash, connection_splash = _prepare_fake_main(monkeypatch)

    class FakeWindow:
        def show(self) -> None:
            events.append("window-shown")

    window = FakeWindow()

    def make_window(**kwargs: object) -> FakeWindow:
        events.append("window-construction-started")
        kwargs["on_progress"]("Connecting channel 1 of 2...")  # type: ignore[operator]
        return window

    monkeypatch.setattr(main_module, "MainAppWindow", make_window)

    with pytest.raises(SystemExit) as exited:
        main_module.main()

    assert exited.value.code == 0
    first_splash.finish.assert_called_once()
    connection_splash.finish.assert_called_once_with(window)
    assert events.index("dialog-accepted") < events.index("window-construction-started")
    assert events.index(("progress", "Connecting channel 1 of 2...")) < events.index("window-shown")
    assert events.index("window-shown") < events.index("app-exec")


def test_connection_failure_closes_second_splash_and_reports_error(
    monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    _events, _first_splash, connection_splash = _prepare_fake_main(monkeypatch)
    error_box = Mock()
    monkeypatch.setattr(main_module, "QMessageBox", SimpleNamespace(critical=error_box))

    def fail_window(**_kwargs: object) -> None:
        raise RuntimeError("device unavailable")

    monkeypatch.setattr(main_module, "MainAppWindow", fail_window)

    with pytest.raises(SystemExit) as exited:
        main_module.main()

    assert exited.value.code == 1
    connection_splash.close.assert_called_once_with()
    connection_splash.finish.assert_not_called()
    error_box.assert_called_once_with(None, "Connection Failed", "device unavailable")


def test_libiio_import_failure_is_reported_before_runtime_ui_loads(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    _events, first_splash, _connection_splash = _prepare_fake_main(monkeypatch)
    runtime_loader = Mock()
    error_dialog = Mock()
    monkeypatch.setattr(main_module, "_load_runtime_ui", runtime_loader)
    monkeypatch.setattr(main_module, "_show_libiio_runtime_error", error_dialog)
    monkeypatch.setattr(
        main_module,
        "_verify_libiio_runtime",
        Mock(side_effect=main_module.LibiioRuntimeError("missing libiio.dll")),
    )

    with pytest.raises(SystemExit) as exited:
        main_module.main()

    assert exited.value.code == 1
    runtime_loader.assert_not_called()
    first_splash.finish.assert_not_called()
    error_dialog.assert_called_once()
    assert str(error_dialog.call_args.args[0]) == "missing libiio.dll"


def test_libiio_runtime_check_wraps_native_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_name: str) -> None:
        raise OSError("missing libiio.dll")

    monkeypatch.setattr(main_module.importlib, "import_module", fail_import)

    with pytest.raises(main_module.LibiioRuntimeError, match="missing libiio.dll"):
        main_module._verify_libiio_runtime()


def test_libiio_runtime_check_rejects_v1_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(version=(1, 0, "test")),
    )

    with pytest.raises(main_module.LibiioRuntimeError, match="incompatible 1.x API"):
        main_module._verify_libiio_runtime()


def test_libiio_error_dialog_opens_pinned_release_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened_urls: list[str] = []
    captured: dict[str, object] = {}
    open_button = object()

    class FakeMessageBox:
        Icon = SimpleNamespace(Critical=1)
        ButtonRole = SimpleNamespace(ActionRole=2)
        StandardButton = SimpleNamespace(Close=3)

        def setIcon(self, value: object) -> None:  # noqa: N802 - Qt API
            captured["icon"] = value

        def setWindowTitle(self, value: str) -> None:  # noqa: N802 - Qt API
            captured["title"] = value

        def setText(self, value: str) -> None:  # noqa: N802 - Qt API
            captured["text"] = value

        def setInformativeText(self, value: str) -> None:  # noqa: N802 - Qt API
            captured["informative"] = value

        def setDetailedText(self, value: str) -> None:  # noqa: N802 - Qt API
            captured["details"] = value

        def addButton(self, text: str, _role: object) -> object:  # noqa: N802 - Qt API
            captured["button_text"] = text
            return open_button

        def setStandardButtons(self, value: object) -> None:  # noqa: N802 - Qt API
            captured["standard_buttons"] = value

        def setDefaultButton(self, value: object) -> None:  # noqa: N802 - Qt API
            captured["default_button"] = value

        def exec(self) -> None:
            pass

        def clickedButton(self) -> object:  # noqa: N802 - Qt API
            return open_button

    monkeypatch.setattr(main_module, "QMessageBox", FakeMessageBox)
    monkeypatch.setattr(
        main_module.QDesktopServices,
        "openUrl",
        lambda url: opened_urls.append(url.toString()) or True,
    )

    main_module._show_libiio_runtime_error(
        main_module.LibiioRuntimeError("missing libiio.dll")
    )

    assert captured["title"] == "libiio Runtime Required"
    assert "v0.26" in str(captured["informative"])
    assert "not yet a validated" in str(captured["informative"])
    assert "missing libiio.dll" in str(captured["details"])
    assert opened_urls == [main_module.LIBIIO_V026_RELEASE_URL]
