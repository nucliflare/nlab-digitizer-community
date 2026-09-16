from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
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
        backend = "iio"
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
