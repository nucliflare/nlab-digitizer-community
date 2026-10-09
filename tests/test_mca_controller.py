import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, call

import numpy as np
import pyqtgraph as pg
import pytest

from nlab.controllers import mca_controller as mca_controller_module
from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.dma import IIOMcaDmaStreamer, McaEventBuffer
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode, McaRunSummary
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

    def mca_dma_active(self) -> bool:
        return self.dma_enabled

    def get_measurement_in_progress(self) -> bool:
        self.events.append("status")
        return self.enabled


class _ThreadProbe:
    def __init__(self, events: list[str], *, exits: bool = True) -> None:
        self._events = events
        self._exits = exits

    def wait(self, timeout_ms: int) -> bool:
        self._events.append(f"wait:{timeout_ms}")
        return self._exits

    def deleteLater(self) -> None:  # noqa: N802 - mirrors the Qt API
        self._events.append("delete")


def test_live_reconfiguration_stops_writes_and_restarts() -> None:
    backend = _Backend()
    mca = MultiChannelAnalyzer(backend)  # type: ignore[arg-type]

    restarted = mca.reconfigure_while_running(lambda: backend.events.append("write"))

    assert restarted is True
    assert backend.events == [("enable", False), "write", ("enable", True)]
    assert backend.enabled is True


def test_dma_final_status_keeps_run_health_instead_of_stopped() -> None:
    label = Mock()
    events: list[str] = []
    thread = _ThreadProbe(events)
    controller = SimpleNamespace(
        _dma_summary=None,
        _dma_error=None,
        _dma_worker=object(),
        _dma_thread=thread,
        _channel=0,
        _connection_lost=False,
        _replay_hardware_interlocked=False,
        _set_controls_enabled=Mock(),
        ui=SimpleNamespace(
            lblDmaStatus=label,
            btnStart=Mock(),
            btnStop=Mock(),
            cbDmaEnable=Mock(),
            btnDmaFile=Mock(),
        ),
        _finish_psd_capture=Mock(),
    )
    summary = McaRunSummary(
        channel=0,
        mode=McaDmaOutputMode.ONLINE,
        path=None,
        started_utc="2026-09-17T10:00:00+00:00",
        finished_utc="2026-09-17T10:00:02+00:00",
        duration_s=2,
        records=2048,
        continuity="verified",
        diagnostics={"continuity_valid": True},
    )

    MCAController._on_dma_summary(controller, summary)
    MCAController._on_dma_finished(controller)

    label.setText.assert_called_once_with("Verified: 2,048 records, 1,024/s avg")
    controller._finish_psd_capture.assert_called_once_with()
    assert events == ["wait:2000", "delete"]
    assert controller._dma_thread is None
    assert controller._dma_worker is None
    controller._set_controls_enabled.assert_called_once_with(True)
    controller.ui.btnStart.setEnabled.assert_called_once_with(True)


def test_dma_finish_retries_without_releasing_live_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    thread = _ThreadProbe(events, exits=False)

    def callback() -> None:
        pass

    controller = SimpleNamespace(
        _dma_thread=thread,
        _dma_worker=object(),
        _channel=1,
        _on_dma_finished=callback,
    )
    retries: list[tuple[int, object]] = []
    monkeypatch.setattr(
        mca_controller_module,
        "QTimer",
        SimpleNamespace(singleShot=lambda delay, target: retries.append((delay, target))),
    )

    MCAController._on_dma_finished(controller)  # type: ignore[arg-type]

    assert events == ["wait:2000"]
    assert controller._dma_thread is thread
    assert controller._dma_worker is not None
    assert retries == [(100, callback)]


def test_polling_finish_waits_before_releasing_thread_wrappers() -> None:
    events: list[str] = []
    thread = _ThreadProbe(events)
    controller = SimpleNamespace(
        _worker_thread=thread,
        _worker=object(),
        _channel=0,
    )

    MCAController._on_worker_finished(controller)  # type: ignore[arg-type]

    assert events == ["wait:2000", "delete"]
    assert controller._worker_thread is None
    assert controller._worker is None


def test_dma_stop_keeps_start_interlocked_until_worker_finishes() -> None:
    worker = Mock()
    controller = SimpleNamespace(
        _channel=0,
        _coincidence_session=False,
        _dma_worker=worker,
        _cancel_debug_marker_drags=Mock(),
        _stop_worker=Mock(),
        _mca=Mock(),
        _set_controls_enabled=Mock(),
        ui=SimpleNamespace(
            btnStart=Mock(),
            btnStop=Mock(),
            cbDmaEnable=Mock(),
            btnDmaFile=Mock(),
        ),
    )

    MCAController._on_stop(controller)  # type: ignore[arg-type]

    worker.stop.assert_called_once_with()
    controller._mca.stop.assert_called_once_with()
    controller._set_controls_enabled.assert_called_once_with(False)
    controller.ui.btnStart.setEnabled.assert_called_once_with(False)
    controller.ui.cbDmaEnable.setEnabled.assert_called_once_with(False)


def test_start_is_ignored_while_dma_thread_is_closing() -> None:
    controller = SimpleNamespace(
        _replay_hardware_interlocked=False,
        _dma_worker=object(),
        _dma_thread=object(),
        _channel=0,
        _cancel_debug_marker_drags=Mock(),
    )

    MCAController._on_start(controller)  # type: ignore[arg-type]

    controller._cancel_debug_marker_drags.assert_not_called()


def test_connection_loss_keeps_controls_interlocked_after_dma_cleanup() -> None:
    label = Mock()
    controller = SimpleNamespace(
        _dma_summary=None,
        _dma_error="connection reset",
        _dma_worker=object(),
        _dma_thread=_ThreadProbe([]),
        _channel=0,
        _connection_lost=True,
        _replay_hardware_interlocked=False,
        _set_controls_enabled=Mock(),
        _finish_psd_capture=Mock(),
        ui=SimpleNamespace(
            lblDmaStatus=label,
            btnStart=Mock(),
            btnStop=Mock(),
            cbDmaEnable=Mock(),
            btnDmaFile=Mock(),
        ),
    )

    MCAController._on_dma_finished(controller)  # type: ignore[arg-type]

    controller._set_controls_enabled.assert_called_once_with(False)
    controller.ui.btnStart.setEnabled.assert_called_once_with(False)
    controller.ui.cbDmaEnable.setEnabled.assert_called_once_with(False)
    label.setText.assert_called_once_with(
        "Connection lost; use Connection -> Reconnect Device."
    )


def test_dma_finish_unlocks_file_button_after_psd_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MCAController,
        "_output_mode",
        staticmethod(lambda: McaDmaOutputMode.BINARY),
    )
    capture = Mock()
    controller = SimpleNamespace(
        _dma_summary=None,
        _dma_error=None,
        _dma_worker=object(),
        _dma_thread=_ThreadProbe([]),
        _channel=0,
        _connection_lost=False,
        _replay_hardware_interlocked=False,
        _set_controls_enabled=Mock(),
        _psd_capture=capture,
        _psd_capture_enabled=True,
        ui=SimpleNamespace(
            lblDmaStatus=Mock(),
            btnStart=Mock(),
            btnStop=Mock(),
            cbDmaEnable=Mock(),
            btnDmaFile=Mock(),
        ),
    )
    controller._finish_psd_capture = lambda: MCAController._finish_psd_capture(
        controller  # type: ignore[arg-type]
    )

    MCAController._on_dma_finished(controller)  # type: ignore[arg-type]

    capture.finish_capture.assert_called_once_with()
    assert controller._psd_capture_enabled is False
    controller.ui.btnDmaFile.setEnabled.assert_called_once_with(True)


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
    assert controller._display_histogram is None
    assert controller._differential_reference_histogram is None
    assert controller._display_interval_s == 0.0
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

    MCAController._ensure_disarmed(controller)

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


def test_coincidence_can_receive_online_events_without_feeding_disabled_psd() -> None:
    psd_buffer = McaEventBuffer()
    coincidence_buffer = McaEventBuffer()
    controller = SimpleNamespace(
        _psd_capture=Mock(),
        _psd_capture_enabled=False,
        _event_buffer=psd_buffer,
        _coincidence_buffer=coincidence_buffer,
        ui=SimpleNamespace(cbCcEnable=SimpleNamespace(isChecked=lambda: False)),
    )

    result = MCAController._prepare_psd_capture(controller)

    assert result is coincidence_buffer
    assert result is not psd_buffer


def test_session_capture_publishes_events_without_enabling_psd() -> None:
    event_buffer = McaEventBuffer()
    controller = SimpleNamespace(
        _psd_capture=Mock(),
        _psd_capture_enabled=False,
        _session_capture_enabled=True,
        _event_buffer=event_buffer,
        _coincidence_buffer=None,
        ui=SimpleNamespace(cbCcEnable=SimpleNamespace(isChecked=lambda: False)),
    )

    result = MCAController._prepare_psd_capture(controller)

    assert result is event_buffer
    assert controller._psd_capture_enabled is False


@pytest.mark.parametrize("mode", list(McaDmaOutputMode))
@pytest.mark.parametrize("charge_comparison_enabled", [False, True])
def test_dma_output_and_psd_interception_are_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: McaDmaOutputMode,
    charge_comparison_enabled: bool,
) -> None:
    worker = Mock()
    thread = Mock()
    worker_factory = Mock(return_value=worker)
    monkeypatch.setattr("nlab.controllers.mca_controller.IIOMcaDmaWorker", worker_factory)
    monkeypatch.setattr("nlab.controllers.mca_controller.QThread", Mock(return_value=thread))
    monkeypatch.setattr("nlab.utils.settings_io.write_configuration", Mock())

    capture = Mock()
    event_buffer = McaEventBuffer()
    destination = tmp_path / f"measurement{mode.extension or '.bin'}"
    label = Mock()
    controller = SimpleNamespace(
        _active_dma_mode=McaDmaOutputMode.BINARY,
        _dma_error=None,
        _dma_filepath=None,
        _dma_summary=None,
        _event_buffer=event_buffer,
        _channel=0,
        _mca=Mock(),
        _mca_dma=IIOMcaDmaStreamer(backend=Mock(), channel=0),
        _measurement_configuration=lambda: {},
        _generate_filepath=Mock(return_value=destination),
        _output_mode=lambda: mode,
        _psd_capture=capture,
        _psd_capture_enabled=False,
        _set_controls_enabled=Mock(),
        _on_dma_ready=Mock(),
        _on_dma_progress=Mock(),
        _on_connection_lost=Mock(),
        _on_dma_error=Mock(),
        _on_dma_summary=Mock(),
        _on_dma_finished=Mock(),
        ui=SimpleNamespace(
            cbCcEnable=SimpleNamespace(isChecked=lambda: charge_comparison_enabled),
            spinTimeLimit=SimpleNamespace(value=lambda: 30),
            lblDmaStatus=label,
        ),
    )
    controller._prepare_psd_capture = lambda: MCAController._prepare_psd_capture(
        cast(MCAController, controller)
    )

    MCAController._start_with_dma(cast(MCAController, controller))

    worker_kwargs = worker_factory.call_args.kwargs
    assert worker_kwargs["filepath"] == (
        None if mode is McaDmaOutputMode.ONLINE else destination
    )
    assert worker_kwargs["event_buffer"] is (
        event_buffer if charge_comparison_enabled else None
    )
    assert worker_kwargs["output_mode"] is mode
    thread.start.assert_called_once_with()
    if mode is McaDmaOutputMode.ONLINE:
        assert "no file" in label.setText.call_args.args[0]
    else:
        assert "file" in label.setText.call_args.args[0]


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
        _display_histogram=None,
        _differential_histogram_enabled=False,
        _differential_reference_histogram=None,
        _differential_reference_elapsed_s=0.0,
        _display_interval_s=0.0,
        _last_elapsed_s=1.0,
        _hist_curve=histogram_curve,
        _roi=SimpleNamespace(isVisible=lambda: True),
        _roi_dragging=True,
        _update_roi_stats=update_stats,
    )

    MCAController._update_histogram(controller, histogram)

    np.testing.assert_array_equal(controller._last_histogram, histogram)
    np.testing.assert_array_equal(controller._display_histogram, histogram)
    histogram_curve.setData.assert_called_once()
    update_stats.assert_not_called()


def test_differential_histogram_displays_non_negative_per_refresh_counts() -> None:
    histogram_curve = Mock()
    controller = SimpleNamespace(
        _last_histogram=None,
        _display_histogram=None,
        _differential_histogram_enabled=True,
        _differential_reference_histogram=None,
        _differential_reference_elapsed_s=0.0,
        _display_interval_s=0.0,
        _last_elapsed_s=1.0,
        _hist_curve=histogram_curve,
        _roi=SimpleNamespace(isVisible=lambda: False),
        _roi_dragging=False,
    )

    first = np.array([5, 7, 2], dtype=np.uint32)
    MCAController._update_histogram(controller, first)
    np.testing.assert_array_equal(controller._display_histogram, first)
    assert controller._display_interval_s == 1.0

    controller._last_elapsed_s = 2.5
    current = np.array([8, 6, 10], dtype=np.uint32)
    MCAController._update_histogram(controller, current)

    np.testing.assert_array_equal(controller._last_histogram, current)
    np.testing.assert_array_equal(controller._display_histogram, [3, 0, 8])
    np.testing.assert_array_equal(controller._differential_reference_histogram, current)
    assert controller._display_interval_s == 1.5
    np.testing.assert_array_equal(histogram_curve.setData.call_args.args[1], [3, 0, 8])


def test_histogram_display_switch_rebases_without_changing_accumulated_counts() -> None:
    histogram_curve = Mock()
    y_label = Mock()
    roi_stats = Mock()
    accumulated = np.array([10, 20], dtype=np.uint32)
    controller = SimpleNamespace(
        _last_histogram=accumulated,
        _display_histogram=accumulated,
        _differential_histogram_enabled=False,
        _differential_reference_histogram=np.array([8, 18], dtype=np.uint32),
        _differential_reference_elapsed_s=8.0,
        _display_interval_s=2.0,
        _last_elapsed_s=10.0,
        _hist_curve=histogram_curve,
        _histogram_y_label=y_label,
        _roi=SimpleNamespace(isVisible=lambda: True),
        _roi_dragging=False,
        _update_roi_stats=roi_stats,
    )

    MCAController.set_differential_histogram(controller, True)

    np.testing.assert_array_equal(controller._last_histogram, accumulated)
    assert controller._differential_histogram_enabled is True
    np.testing.assert_array_equal(controller._display_histogram, [0, 0])
    np.testing.assert_array_equal(
        controller._differential_reference_histogram,
        accumulated,
    )
    assert controller._display_interval_s == 0.0
    y_label.setText.assert_called_once_with("Counts / refresh")
    roi_stats.assert_called_once_with()

    MCAController.set_differential_histogram(controller, False)

    np.testing.assert_array_equal(controller._display_histogram, accumulated)
    assert controller._differential_histogram_enabled is False
    assert controller._display_interval_s == 10.0
    y_label.setText.assert_called_with("Counts")


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


def test_mca_debug_viewer_scales_time_axis_and_explains_sample_period() -> None:
    class DebugControllerStub:
        _update_debug_time_axis = MCAController._update_debug_time_axis

    label = Mock()
    plot = Mock()
    frame_control = Mock()
    frame_control.value.return_value = 2048
    debug1_curve = Mock()
    debug2_curve = Mock()
    debug1_curve.getData.return_value = (None, None)
    debug2_curve.getData.return_value = (None, None)
    controller = DebugControllerStub()
    controller.ui = SimpleNamespace(
        spinFrameSamples=frame_control,
        plotDebug=plot,
        sliderFrameSamples=SimpleNamespace(value=lambda: 2048),
    )
    controller._debug_time_axis_label = label
    controller._debug1_curve = debug1_curve
    controller._debug2_curve = debug2_curve
    controller._pretrigger_line_drag = None
    controller._sync_debug_pretrigger_line = Mock()

    MCAController._update_debug_time_axis(controller)
    label.setText.assert_called_with("Time [\N{MICRO SIGN}s]  (8 ns/debug sample)")
    plot.setToolTip.assert_not_called()
    assert "Debug sample period: 8 ns" in frame_control.setToolTip.call_args.args[0]

    raw = np.arange(300, dtype=np.int16)
    MCAController._update_debug_plot(controller, raw, raw)

    x_data, y_data = debug1_curve.setData.call_args.args
    assert len(y_data) == 256
    np.testing.assert_allclose(x_data[:3], [0.0, 0.008, 0.016])
