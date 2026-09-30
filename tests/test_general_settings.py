from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import QMainWindow, QWidget
from pytestqt.qtbot import QtBot

from nlab import app as app_module
from nlab.app import MainAppWindow
from nlab.ui.ui_mca_view import Ui_MCAView
from nlab.ui.ui_psu_view import Ui_PSUView
from nlab.ui.ui_scope_view import Ui_ScopeView
from nlab.views import general_settings_dialog as dialog_module
from nlab.views.general_settings_dialog import (
    GeneralSettingsDialog,
    auto_configuration_path,
)


def test_auto_configuration_paths_are_isolated_by_device_address(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(dialog_module, "_configuration_root", lambda: tmp_path)

    first = auto_configuration_path("192.0.2.10")
    second = auto_configuration_path("192.0.2.11")
    ipv6 = auto_configuration_path("[2001:db8::10]")

    assert first != second
    assert "192.0.2.10" in first.name
    assert first.parent == tmp_path / "device-configurations"
    assert ":" not in ipv6.name


def test_equivalent_hostname_addresses_share_one_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(dialog_module, "_configuration_root", lambda: tmp_path)

    assert auto_configuration_path(" DIGITIZER.local ") == auto_configuration_path(
        "digitizer.LOCAL"
    )


def test_general_settings_describes_complete_configuration_snapshot(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "last-configuration.yaml"
    monkeypatch.setattr(dialog_module, "auto_configuration_enabled", lambda: True)
    monkeypatch.setattr(
        dialog_module,
        "auto_configuration_path",
        lambda _device_address: snapshot,
    )

    dialog = GeneralSettingsDialog(
        "192.0.2.10",
        {
            "scope_display_mode": 1,
            "scope_persistence": 750,
            "scope_refresh_rate_hz": 20,
            "mca_refresh_rate_hz": 8,
            "psu_refresh_interval_ms": 500,
            "psu_plot_time_range_s": 120,
            "dma_save_folder": "captures",
            "mca_dma_output_mode": "hdf5",
            "show_roi": True,
            "log_y": True,
        },
    )
    qtbot.addWidget(dialog)

    assert dialog.auto_configuration_is_enabled is True
    assert "Scope, MCA, and Power Supply" in dialog.auto_configuration.toolTip()
    assert dialog.snapshot_path.text() == str(snapshot)
    assert dialog.settings == {
        "scope_display_mode": 1,
        "scope_persistence": 750,
        "scope_refresh_rate_hz": 20,
        "mca_refresh_rate_hz": 8,
        "psu_refresh_interval_ms": 500,
        "psu_plot_time_range_s": 120,
        "dma_save_folder": "captures",
        "mca_dma_output_mode": "hdf5",
        "show_roi": True,
        "log_y": True,
    }


def test_channel_panels_hide_controls_moved_to_general_settings(qtbot: QtBot) -> None:
    scope_widget = QWidget()
    mca_widget = QWidget()
    psu_widget = QWidget()
    qtbot.addWidget(scope_widget)
    qtbot.addWidget(mca_widget)
    qtbot.addWidget(psu_widget)
    scope = Ui_ScopeView()
    mca = Ui_MCAView()
    psu = Ui_PSUView()
    scope.setupUi(scope_widget)
    mca.setupUi(mca_widget)
    psu.setupUi(psu_widget)

    assert scope.groupDisplay.isHidden()
    assert mca.labelRefreshRate.isHidden()
    assert mca.spinRefreshRate.isHidden()
    assert psu.labelRefreshRate.isHidden()
    assert psu.spinRefreshRate.isHidden()
    assert psu.labelTimeRange.isHidden()
    assert psu.spinTimeRange.isHidden()


def test_app_applies_general_settings_to_controllers_and_global_options(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: dict[str, object] = {}

    class FakeSettings:
        def setValue(self, key: str, value: object) -> None:  # noqa: N802
            stored[key] = value

    owner = QWidget()
    qtbot.addWidget(owner)
    controller = SimpleNamespace(
        apply_general_configuration_settings=Mock(),
        refresh_dma_output_settings=Mock(),
    )
    save_preferences = Mock()
    window = SimpleNamespace(
        _controller=controller,
        _save_developer_settings=save_preferences,
        ui=SimpleNamespace(
            actionShowRoi=QAction(owner),
            actionLogY=QAction(owner),
        ),
    )
    window.ui.actionShowRoi.setCheckable(True)
    window.ui.actionLogY.setCheckable(True)
    monkeypatch.setattr(app_module, "QSettings", FakeSettings)
    values = {
        "scope_display_mode": 1,
        "scope_persistence": 800,
        "scope_refresh_rate_hz": 15,
        "mca_refresh_rate_hz": 6,
        "psu_refresh_interval_ms": 700,
        "psu_plot_time_range_s": 180,
        "dma_save_folder": "captures",
        "mca_dma_output_mode": "root",
        "show_roi": True,
        "log_y": True,
    }

    MainAppWindow._apply_general_settings(window, values)

    controller.apply_general_configuration_settings.assert_called_once_with(values)
    controller.refresh_dma_output_settings.assert_called_once_with()
    assert stored["dma/save_folder"] == "captures"
    assert stored["dma/mca_output_mode"] == "root"
    assert window.ui.actionShowRoi.isChecked()
    assert window.ui.actionLogY.isChecked()
    save_preferences.assert_called_once_with()


def test_enabling_auto_configuration_saves_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = SimpleNamespace(
        exec=lambda: True,
        auto_configuration_is_enabled=True,
        settings={"scope_refresh_rate_hz": 12},
    )
    save = Mock(return_value=True)
    apply = Mock()
    persist = Mock()
    status = SimpleNamespace(showMessage=Mock())
    window = SimpleNamespace(
        _host="192.0.2.10",
        _general_settings_values=lambda: {},
        _apply_general_settings=apply,
        _save_auto_configuration=save,
        statusBar=lambda: status,
    )
    monkeypatch.setattr(
        app_module,
        "GeneralSettingsDialog",
        lambda _device_address, _values, _parent: dialog,
    )
    monkeypatch.setattr(app_module, "set_auto_configuration_enabled", persist)
    monkeypatch.setattr(
        app_module,
        "auto_configuration_path",
        lambda _device_address: Path("last-configuration.yaml"),
    )

    MainAppWindow._on_general_settings(window)

    apply.assert_called_once_with(dialog.settings)
    save.assert_called_once_with(show_error=True)
    persist.assert_called_once_with(True)
    assert "Automatic configuration saved" in status.showMessage.call_args.args[0]


def test_failed_initial_auto_save_leaves_feature_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = SimpleNamespace(
        exec=lambda: True,
        auto_configuration_is_enabled=True,
        settings={},
    )
    persist = Mock()
    window = SimpleNamespace(
        _host="192.0.2.10",
        _general_settings_values=lambda: {},
        _apply_general_settings=Mock(),
        _save_auto_configuration=Mock(return_value=False),
    )
    monkeypatch.setattr(
        app_module,
        "GeneralSettingsDialog",
        lambda _device_address, _values, _parent: dialog,
    )
    monkeypatch.setattr(app_module, "set_auto_configuration_enabled", persist)

    MainAppWindow._on_general_settings(window)

    persist.assert_called_once_with(False)


def test_auto_configuration_save_creates_platform_config_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "nested" / "last-configuration.yaml"
    controller = SimpleNamespace(save_all_settings=Mock())
    window = SimpleNamespace(_controller=controller, _host="192.0.2.10")
    monkeypatch.setattr(
        app_module,
        "auto_configuration_path",
        lambda _device_address: snapshot,
    )

    saved = MainAppWindow._save_auto_configuration(window, show_error=False)

    assert saved is True
    assert snapshot.parent.is_dir()
    controller.save_all_settings.assert_called_once_with(snapshot)


def test_startup_restores_auto_configuration_without_explicit_config(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "last-configuration.yaml"
    snapshot.write_text("format_version: 3\n", encoding="utf-8")
    stages: list[str] = []
    controller = SimpleNamespace(
        load_all_settings=Mock(),
        save_all_settings=Mock(),
        shutdown=Mock(),
    )

    monkeypatch.setattr(MainAppWindow, "_setup_ui", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_save_developer_settings", lambda _self: None)
    monkeypatch.setattr(app_module, "MainWindowController", Mock(return_value=controller))
    monkeypatch.setattr(app_module, "auto_configuration_enabled", lambda: True)
    monkeypatch.setattr(
        app_module,
        "auto_configuration_path",
        lambda device_address: (
            snapshot if device_address == "192.0.2.10" else Path("wrong")
        ),
    )

    window = MainAppWindow(
        host="192.0.2.10",
        config_path=None,
        on_progress=stages.append,
    )
    qtbot.addWidget(window)

    controller.load_all_settings.assert_called_once_with(snapshot)
    assert "Restoring the last configuration..." in stages


def test_explicit_config_takes_precedence_over_auto_configuration(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = Path("explicit.yaml")
    controller = SimpleNamespace(
        load_all_settings=Mock(),
        save_all_settings=Mock(),
        shutdown=Mock(),
    )

    monkeypatch.setattr(MainAppWindow, "_setup_ui", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_save_developer_settings", lambda _self: None)
    monkeypatch.setattr(app_module, "MainWindowController", Mock(return_value=controller))
    monkeypatch.setattr(app_module, "auto_configuration_enabled", lambda: True)

    window = MainAppWindow(config_path=explicit)
    qtbot.addWidget(window)

    controller.load_all_settings.assert_called_once_with(explicit)


def test_close_saves_auto_configuration_before_hardware_shutdown(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class CloseTrackingWindow(MainAppWindow):
        def __init__(self) -> None:
            QMainWindow.__init__(self)
            self._controller = SimpleNamespace(shutdown=lambda: events.append("shutdown"))

        def _save_auto_configuration(self, *, show_error: bool) -> bool:
            assert show_error is False
            events.append("save")
            return True

        def _save_developer_settings(self) -> None:
            events.append("preferences")

    window = CloseTrackingWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(app_module, "auto_configuration_enabled", lambda: True)

    MainAppWindow.closeEvent(window, QCloseEvent())

    assert events == ["save", "preferences", "shutdown"]
