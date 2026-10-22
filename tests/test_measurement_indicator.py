from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QTabWidget
from pytestqt.qtbot import QtBot

from nlab.app import MainAppWindow
from nlab.controllers.main_window_controller import (
    DigitizerChannel,
    DigitizerEndpoint,
    MainWindowController,
)
from nlab.controllers.mca_controller import MCAController
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.offline import create_offline_digitizer


def test_badge_aggregates_controller_signals_across_tabs_and_core_types(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("_apply_stored_general_settings", "_persist_current_general_settings",
                 "_apply_view_state", "_save_developer_settings"):
        monkeypatch.setattr(MainAppWindow, name, lambda _self: None)
    window = MainAppWindow(channels=2, offline=True)
    qtbot.addWidget(window)
    controller = window._controller
    badge = window._measurement_indicator
    assert badge.isHidden() and not badge.text()
    # In-memory transports only: exercise the live presentation without ever
    # opening IIO. Real offline operation must suppress this badge (below).
    window._offline = controller._offline = False
    mca0, mca1 = controller._mca_controllers
    scope0 = controller._scope_controllers[0]
    window.show()
    initial_width = window.width()
    mca0._set_measurement_active(True)
    assert badge.isVisibleTo(window) and badge.text() == "Measuring: CH0"
    assert "#2e7d32" in badge.styleSheet()
    mca1._set_measurement_active(True)
    assert badge.text() == "Measuring: CH0, CH1"
    # Permanent status widget is independent of messages and active tabs.
    window.statusBar().showMessage("A diagnostic message")
    for tab in window.findChildren(QTabWidget):
        if tab.count() > 1:
            tab.setCurrentIndex(1)
            break
    assert badge.isVisibleTo(window) and badge.text() == "Measuring: CH0, CH1"
    scope0._set_measurement_active(True)
    mca0._set_measurement_active(False)
    assert badge.text() == "Measuring: CH0, CH1"  # Scope keeps CH0 active.
    scope0._set_measurement_active(False)
    assert badge.text() == "Measuring: CH1"
    mca1._set_measurement_active(False)
    assert badge.isHidden() and not badge.text()
    assert window.width() == initial_width
    mca0._set_measurement_active(True)
    window._offline = controller._offline = True
    window.refresh_measurement_indicator()
    assert badge.isHidden() and not badge.text()
    mca0._set_measurement_active(False)
    window.close()


def test_multi_board_channels_stay_qualified_and_offline_stays_empty() -> None:
    controller = object.__new__(MainWindowController)
    controller._offline = False
    controller._endpoints = [DigitizerEndpoint("a", 30431, 2), DigitizerEndpoint("b", 30431, 2)]
    controller._device_channels = [DigitizerChannel(0, 0), DigitizerChannel(0, 1),
                                   DigitizerChannel(1, 0), DigitizerChannel(1, 1)]
    controller._scope_controllers = [SimpleNamespace(channel=2, measurement_active=True)]
    controller._mca_controllers = [SimpleNamespace(channel=0, measurement_active=True),
                                   SimpleNamespace(channel=2, measurement_active=True)]
    assert controller.measuring_channel_labels() == ("D1:CH0", "D2:CH0")
    controller._offline = True
    assert controller.measuring_channel_labels() == ()


def test_mca_polling_dma_ready_timed_stop_and_connection_loss_report_state(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = create_offline_digitizer(0)
    mca = MCAController(device.mca)
    qtbot.addWidget(mca)
    monkeypatch.setattr(mca, "_start_worker", Mock())
    states: list[bool] = []
    mca.measurement_active_changed.connect(states.append)
    assert not mca.measurement_active
    mca._start_polling_only()
    assert mca.measurement_active and states == [True]
    mca._on_measurement_done()
    assert not mca.measurement_active and states == [True, False]
    # DMA is not active merely because Start is checked / setup is pending.
    mca.ui.btnStart.setChecked(True)
    assert not mca.measurement_active
    mca._on_dma_ready()
    assert mca.measurement_active
    mca._on_stop()
    assert not mca.measurement_active
    mca._on_dma_ready()
    mca._on_connection_lost("simulated transport loss")
    assert not mca.measurement_active
    assert states == [True, False, True, False, True, False]
    # An unsuccessful start must never leave an active badge behind.
    mca._connection_lost = False
    monkeypatch.setattr(
        device.mca, "start", Mock(side_effect=RuntimeError("simulated start failure")),
    )
    mca.ui.cbDmaEnable.setChecked(False)
    mca._on_start()
    assert not mca.measurement_active and states[-1] is False
    device.close()


def test_scope_run_does_not_flicker_with_viewer_requests_and_shutdown_clears_state(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = create_offline_digitizer(0)
    scope = ScopeController(device.scope)
    qtbot.addWidget(scope)
    states: list[bool] = []
    scope.measurement_active_changed.connect(states.append)
    assert not scope.measurement_active
    scope._start_polling_only()
    scope._refresh_timer.stop()  # No asynchronous viewer work in this unit test.
    assert scope.measurement_active
    scope._acquiring = True
    scope._on_viewer_frame_received(None)
    assert not scope._acquiring and scope.measurement_active
    assert states == [True]
    scope._on_measurement_timeout()
    assert not scope.measurement_active
    scope._dma_stopping = True
    scope._on_dma_ready()
    assert not scope.measurement_active  # Ignore queued readiness after Stop.
    scope._dma_stopping = False
    scope._on_dma_ready()
    scope._refresh_timer.stop()
    assert scope.measurement_active
    scope.stop_dma_sync()
    assert not scope.measurement_active
    assert states == [True, False, True, False]
    monkeypatch.setattr(device.scope, "start", Mock(side_effect=RuntimeError("start failed")))
    with pytest.raises(RuntimeError, match="start failed"):
        scope._start_polling_only()
    assert not scope.measurement_active
    device.close()


@pytest.mark.parametrize("kind", ["scope", "mca"])
def test_terminal_dma_completion_clears_active_state(qtbot: QtBot, kind: str) -> None:
    device = create_offline_digitizer(0)
    controller = ScopeController(device.scope) if kind == "scope" else MCAController(device.mca)
    qtbot.addWidget(controller)
    states: list[bool] = []
    controller.measurement_active_changed.connect(states.append)
    controller._set_measurement_active(True)
    thread = Mock()
    thread.wait.return_value = True
    controller._dma_thread = thread
    controller._dma_worker = Mock()
    controller._on_dma_finished()
    assert not controller.measurement_active and states == [True, False]
    assert controller._dma_thread is None
    thread.deleteLater.assert_called_once_with()
    device.close()
