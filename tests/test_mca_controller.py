import threading
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pyqtgraph as pg
import pytest

from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.dma import McaEventBuffer
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.workers.mca_worker import MCAReadback


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

    restarted = mca.reconfigure_while_running(lambda: backend.events.append("write"))

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

    reconfigure_thread = threading.Thread(target=lambda: mca.reconfigure_while_running(write))
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


def test_controller_initialization_unconditionally_clears_stale_enable() -> None:
    mca = Mock()
    controller = SimpleNamespace(_mca=mca, _channel=0)

    MCAController._disarm_before_initialization(controller)

    mca.stop.assert_called_once_with()
    mca.get_measurement_in_progress.assert_not_called()


def test_shutdown_stops_externally_armed_channel_even_when_not_in_progress() -> None:
    mca = Mock()
    mca.get_measurement_in_progress.return_value = False
    controller = SimpleNamespace(_mca=mca, _channel=0)

    MCAController._ensure_disarmed(controller, had_dma_worker=False)

    mca.stop.assert_called_once_with()
    mca.get_measurement_in_progress.assert_not_called()


def test_polling_start_clears_enable_before_writing_time_limit() -> None:
    events: list[object] = []
    mca = Mock()
    mca.stop.side_effect = lambda: events.append("stop")
    mca.set_time_limit.side_effect = lambda value: events.append(("limit", value))
    mca.start.side_effect = lambda: events.append("start")
    start_worker = Mock(side_effect=lambda: events.append("worker"))
    controller = SimpleNamespace(
        _mca=mca,
        _channel=0,
        _start_worker=start_worker,
        ui=SimpleNamespace(
            spinTimeLimit=SimpleNamespace(value=lambda: 30),
            btnStop=SimpleNamespace(setEnabled=Mock()),
        ),
    )

    MCAController._start_polling_only(controller)

    assert events == ["stop", ("limit", 30), "start", "worker"]


@pytest.mark.parametrize("charge_comparison_enabled", [False, True])
def test_psd_interception_requires_charge_comparison(
    charge_comparison_enabled: bool,
) -> None:
    capture = Mock()
    event_buffer = McaEventBuffer()
    controller = SimpleNamespace(
        _psd_capture=capture,
        _psd_capture_enabled=False,
        _event_buffer=event_buffer,
        ui=SimpleNamespace(
            cbCcEnable=SimpleNamespace(isChecked=lambda: charge_comparison_enabled),
        ),
    )

    result = MCAController._prepare_psd_capture(controller)

    assert (result is event_buffer) is charge_comparison_enabled
    assert controller._psd_capture_enabled is charge_comparison_enabled
    capture.begin_capture.assert_called_once()
    assert capture.begin_capture.call_args.args[0] is charge_comparison_enabled


def test_hardware_completion_releases_enable_before_rearming_gui() -> None:
    events: list[str] = []
    mca = Mock()
    mca.stop.side_effect = lambda: events.append("stop")
    controller = SimpleNamespace(
        _mca=mca,
        _mca_dma=None,
        _dma_worker=None,
        _channel=0,
        ui=SimpleNamespace(
            btnStart=SimpleNamespace(
                setChecked=Mock(),
                setEnabled=Mock(side_effect=lambda value: events.append(f"start:{value}")),
            ),
            btnStop=SimpleNamespace(setChecked=Mock(), setEnabled=Mock()),
            cbDmaEnable=SimpleNamespace(setEnabled=Mock()),
            btnDmaFile=SimpleNamespace(setEnabled=Mock()),
        ),
    )

    MCAController._on_measurement_done(controller)

    assert events[:2] == ["stop", "start:True"]


def test_roi_statistics_are_deferred_until_drag_finishes() -> None:
    update_stats = Mock()
    controller = SimpleNamespace(
        _roi_dragging=False,
        _update_roi_stats=update_stats,
    )

    MCAController._on_roi_region_changed(controller)

    assert controller._roi_dragging is True
    update_stats.assert_not_called()

    MCAController._on_roi_change_finished(controller)

    assert controller._roi_dragging is False
    update_stats.assert_called_once_with()


def test_installed_linear_region_item_exposes_connected_signals() -> None:
    assert hasattr(pg.LinearRegionItem, "sigRegionChanged")
    assert hasattr(pg.LinearRegionItem, "sigRegionChangeFinished")


def test_histogram_readback_skips_roi_statistics_during_drag() -> None:
    histogram = np.arange(16, dtype=np.uint32)
    update_stats = Mock()
    histogram_curve = Mock()
    controller = SimpleNamespace(
        _last_histogram=None,
        _hist_curve=histogram_curve,
        _roi=SimpleNamespace(isVisible=lambda: True),
        _roi_dragging=True,
        _update_roi_stats=update_stats,
    )

    MCAController._update_histogram(controller, histogram)

    np.testing.assert_array_equal(controller._last_histogram, histogram)
    histogram_curve.setData.assert_called_once()
    update_stats.assert_not_called()


def _readback(seed: int) -> MCAReadback:
    values = np.array([seed], dtype=np.int16)
    return MCAReadback(
        histogram=np.array([seed], dtype=np.uint32),
        debug1=values,
        debug2=values,
        elapsed_time=seed,
    )


def test_readbacks_are_coalesced_and_gui_render_rate_is_capped() -> None:
    timer = Mock()
    timer.isActive.side_effect = [False, True]
    first = _readback(1)
    latest = _readback(2)
    controller = SimpleNamespace(
        _pending_readback=None,
        _render_timer=timer,
        ui=SimpleNamespace(
            spinRefreshRate=SimpleNamespace(value=lambda: 30),
        ),
    )

    MCAController._on_readback(controller, first)
    MCAController._on_readback(controller, latest)

    assert controller._pending_readback is latest
    timer.start.assert_called_once_with(66)


def test_pending_readback_renders_only_latest_snapshot() -> None:
    latest = _readback(3)
    calls = Mock()
    controller = SimpleNamespace(
        _pending_readback=latest,
        _update_debug_plot=calls.debug,
        _update_statistics=calls.statistics,
        _update_histogram=calls.histogram,
    )

    MCAController._render_pending_readback(controller)

    assert controller._pending_readback is None
    assert calls.mock_calls == [
        call.debug(latest.debug1, latest.debug2),
        call.statistics(latest),
        call.histogram(latest.histogram),
    ]
