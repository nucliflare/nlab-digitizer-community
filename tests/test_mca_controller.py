import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer


class _Backend:
    def __init__(self) -> None:
        self.enabled = True
        self.dma_enabled = False
        self.events: list[object] = []

    def get_global_enable(self) -> bool:
        return self.enabled

    def set_global_enable(self, enabled: bool) -> None:
        self.events.append(("enable", enabled))
        self.enabled = enabled

    def get_dpp_dma_enable(self) -> bool:
        return self.dma_enabled

    def get_measurement_in_progress(self) -> bool:
        self.events.append("status")
        return self.enabled


def test_live_reconfiguration_stops_writes_and_restarts() -> None:
    backend = _Backend()
    mca = MultiChannelAnalyzer(backend)  # type: ignore[arg-type]

    restarted = mca.reconfigure_while_running(
        lambda: backend.events.append("write")
    )

    assert restarted is True
    assert backend.events == [("enable", False), "write", ("enable", True)]
    assert backend.enabled is True


def test_live_reconfiguration_restarts_after_failed_write() -> None:
    backend = _Backend()
    mca = MultiChannelAnalyzer(backend)  # type: ignore[arg-type]

    def fail() -> None:
        backend.events.append("write")
        raise ValueError("bad setting")

    with pytest.raises(ValueError, match="bad setting"):
        mca.reconfigure_while_running(fail)

    assert backend.events == [("enable", False), "write", ("enable", True)]
    assert backend.enabled is True


def test_completion_poll_cannot_observe_reconfiguration_stop() -> None:
    backend = _Backend()
    mca = MultiChannelAnalyzer(backend)  # type: ignore[arg-type]
    write_started = threading.Event()
    allow_write = threading.Event()
    status_finished = threading.Event()

    def write() -> None:
        write_started.set()
        assert allow_write.wait(1)

    reconfigure_thread = threading.Thread(
        target=lambda: mca.reconfigure_while_running(write)
    )
    reconfigure_thread.start()
    assert write_started.wait(1)

    status_thread = threading.Thread(
        target=lambda: (mca.get_measurement_in_progress(), status_finished.set())
    )
    status_thread.start()
    assert not status_finished.wait(0.05)

    allow_write.set()
    reconfigure_thread.join(1)
    status_thread.join(1)

    assert not reconfigure_thread.is_alive()
    assert not status_thread.is_alive()
    assert status_finished.is_set()
    assert backend.events[-2:] == [("enable", True), "status"]


def test_controller_routes_running_write_through_reconfiguration() -> None:
    write = Mock()
    mca = Mock()
    mca.reconfigure_while_running.side_effect = lambda fn: (fn(), True)[1]
    histogram_curve = Mock()
    controller = SimpleNamespace(
        _dma_worker=None,
        _worker=object(),
        _channel=0,
        _mca=mca,
        _last_histogram=object(),
        _last_elapsed_s=12.0,
        _hist_curve=histogram_curve,
    )

    MCAController._apply_hardware_setting(controller, write)

    mca.reconfigure_while_running.assert_called_once_with(write)
    write.assert_called_once_with()
    assert controller._last_histogram is None
    assert controller._last_elapsed_s == 0.0
    histogram_curve.setData.assert_called_once_with([], [])
