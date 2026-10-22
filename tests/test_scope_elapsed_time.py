from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import Mock

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.offline import create_offline_digitizer


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    clock = Mock(return_value=100.0)
    monkeypatch.setattr("nlab.controllers.scope_controller.monotonic", clock)
    return clock


@pytest.fixture
def scope(qtbot: QtBot) -> Iterator[ScopeController]:
    device = create_offline_digitizer(0)
    controller = ScopeController(device.scope)
    qtbot.addWidget(controller)
    try:
        yield controller
    finally:
        controller._refresh_timer.stop()
        controller._measurement_timer.stop()
        controller.stop_dma_sync()
        device.close()


@pytest.mark.parametrize("limit", [0, 30])
def test_polling_elapsed_ticks_without_frames_and_freezes_on_stop(
    scope: ScopeController, clock: Mock, qtbot: QtBot, limit: int,
) -> None:
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 0.0 s"
    assert not scope._elapsed_timer.isActive()
    scope.ui.spinTime.setValue(limit)
    scope._start_polling_only()
    scope._refresh_timer.stop()  # No background viewer requests in this test.
    assert scope._elapsed_timer.isActive()
    assert scope._measurement_timer.isActive() == (limit > 0)
    clock.return_value = 112.3
    # Timer works even with no triggers/viewer frames and while the view is hidden.
    qtbot.waitUntil(lambda: scope.ui.labelElapsedTime.text() == "Elapsed: 12.3 s")
    scope._acquiring = True
    scope._on_viewer_frame_received(None)
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 12.3 s"
    clock.return_value = 114.6
    scope._on_stop()
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 14.6 s"
    assert not scope._elapsed_timer.isActive()
    clock.return_value = 150.0
    scope._update_elapsed_time()
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 14.6 s"
    scope._start_polling_only()
    scope._refresh_timer.stop()
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 0.0 s"
    clock.return_value = 151.7
    scope._on_measurement_timeout()
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 1.7 s"
    assert not scope._elapsed_timer.isActive()


def test_dma_ready_starts_clock_and_terminal_completion_stops_it(
    scope: ScopeController, clock: Mock,
) -> None:
    scope.ui.cbDmaEnable.setChecked(True)
    scope.ui.btnStart.setChecked(True)  # Arming alone must not start the clock.
    clock.return_value = 120.0
    scope._update_elapsed_time()
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 0.0 s"
    assert not scope._elapsed_timer.isActive()
    scope._on_dma_ready()
    scope._refresh_timer.stop()
    clock.return_value = 125.0
    scope._on_dma_ready()  # A repeated lifecycle notification must not reset it.
    scope._refresh_timer.stop()
    scope._update_elapsed_time()
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 5.0 s"
    scope._dma_thread = Mock()
    scope._dma_thread.wait.return_value = True
    scope._dma_worker = Mock()
    scope._on_dma_error("simulated transport failure")
    clock.return_value = 125.8
    scope._on_dma_finished()
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 5.8 s"
    assert not scope._elapsed_timer.isActive()
    scope._dma_stopping = True
    scope._on_dma_ready()  # Ignore late queued readiness after Stop.
    assert not scope._elapsed_timer.isActive()


def test_dma_stop_freezes_before_worker_teardown(scope: ScopeController, clock: Mock) -> None:
    scope._on_dma_ready()
    scope._refresh_timer.stop()
    worker = Mock()
    scope._dma_worker = worker
    clock.return_value = 103.4
    scope._on_stop()
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 3.4 s"
    assert not scope._elapsed_timer.isActive()
    worker.stop.assert_called_once_with()
    clock.return_value = 108.0
    scope._update_elapsed_time()
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 3.4 s"
    scope._dma_worker = None


def test_single_frame_replay_and_shutdown_do_not_leave_clock_running(
    scope: ScopeController, clock: Mock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scope._scope, "acquire_frame", Mock(return_value=np.zeros(256)))
    scope._on_acquire_frame()
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 0.0 s"
    assert not scope._elapsed_timer.isActive()
    scope._start_polling_only()
    scope._refresh_timer.stop()
    clock.return_value = 104.1
    scope.begin_session_replay(frame_count=1, duration_ns=1_000_000_000)
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 4.1 s"
    assert not scope._elapsed_timer.isActive()
    clock.return_value = 110.0
    scope.display_session_replay_frame(
        np.zeros(16, dtype=np.int16), sample_period_ns=2,
        frame_index=0, frame_count=1, elapsed_ns=1_000_000_000,
        duration_ns=1_000_000_000, source="simulated",
    )
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 4.1 s"
    scope.end_session_replay()
    scope._start_polling_only()
    scope._refresh_timer.stop()
    clock.return_value = 112.9
    scope.stop_dma_sync()
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 2.9 s"
    assert not scope._elapsed_timer.isActive()


def test_failed_start_does_not_reset_last_elapsed_time(
    scope: ScopeController, clock: Mock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope._start_polling_only()
    scope._refresh_timer.stop()
    clock.return_value = 103.2
    scope._on_stop()
    monkeypatch.setattr(scope._scope, "start", Mock(side_effect=RuntimeError("start failed")))
    with pytest.raises(RuntimeError, match="start failed"):
        scope._start_polling_only()
    assert not scope._elapsed_timer.isActive()
    assert scope.ui.labelElapsedTime.text() == "Elapsed: 3.2 s"
