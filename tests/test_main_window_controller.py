from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from nlab.controllers import main_window_controller as main_window_module
from nlab.controllers.global_controller import GlobalController
from nlab.controllers.main_window_controller import MainWindowController
from nlab.session.replay import (
    CurrentReplaySample,
    McaReplayBatch,
    McaReplayReadback,
    ReplayStreamSummary,
    ScopeReplayFrame,
    TelemetryReplayEvent,
)
from nlab.workers.current_monitor_worker import CurrentSample


def test_startup_reports_connection_and_view_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    progress: list[str] = []
    connected: list[int] = []

    def connect(_self: MainWindowController, channel: int) -> object:
        connected.append(channel)
        return object()

    monkeypatch.setattr(MainWindowController, "_connect_channel", connect)
    monkeypatch.setattr(MainWindowController, "_make_dock_host", staticmethod(object))
    for name in (
        "_build_global_tab",
        "_build_channel_docks",
        "_build_coincidence_tab",
        "_build_external_docks",
        "_restore_dock_state",
        "_connect_signals",
    ):
        monkeypatch.setattr(MainWindowController, name, lambda _self: None)
    monkeypatch.setattr(main_window_module, "ExternalDevices", Mock())

    MainWindowController(
        SimpleNamespace(),
        host="192.0.2.10",
        port=30431,
        channels=2,
        on_progress=progress.append,
    )

    assert connected == [0, 1]
    assert progress == [
        "Connecting channel 1 of 2...",
        "Connecting channel 2 of 2...",
        "Preparing global controls...",
        "Preparing channel views...",
        "Preparing coincidence view...",
        "Discovering external devices...",
        "Restoring dock layout...",
    ]


def test_offline_startup_uses_in_memory_channels_and_interlocks_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress: list[str] = []
    devices = [object(), object()]
    factory = Mock(side_effect=devices)
    live_connect = Mock()
    interlock = Mock()
    monkeypatch.setattr(main_window_module, "create_offline_digitizer", factory)
    monkeypatch.setattr(MainWindowController, "_connect_channel", live_connect)
    monkeypatch.setattr(MainWindowController, "_make_dock_host", staticmethod(object))
    monkeypatch.setattr(MainWindowController, "_set_replay_hardware_interlock", interlock)
    for name in (
        "_build_global_tab",
        "_build_channel_docks",
        "_build_coincidence_tab",
        "_build_external_docks",
        "_restore_dock_state",
        "_connect_signals",
    ):
        monkeypatch.setattr(MainWindowController, name, lambda _self: None)
    monkeypatch.setattr(main_window_module, "ExternalDevices", Mock())

    controller = MainWindowController(
        SimpleNamespace(),
        channels=2,
        on_progress=progress.append,
        offline=True,
    )

    assert controller._devices == devices
    assert factory.call_args_list == [call(0), call(1)]
    live_connect.assert_not_called()
    interlock.assert_called_once_with(True, resume_background=False)
    assert progress[:2] == [
        "Preparing offline channel 1 of 2...",
        "Preparing offline channel 2 of 2...",
    ]


def test_offline_external_replay_panels_are_derived_from_recorded_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._offline = True
    controller._external_controllers = []
    panels = [
        SimpleNamespace(
            configuration_id="GEIGER:7",
            apply_configuration_settings=Mock(),
        ),
        SimpleNamespace(
            configuration_id="PMT:9",
            apply_configuration_settings=Mock(),
        ),
    ]

    def install(devices: object) -> None:
        assert list(devices) == ["recorded-device", "config-only-device"]  # type: ignore[arg-type]
        controller._external_controllers = panels

    controller._build_external_docks = Mock(side_effect=install)  # type: ignore[method-assign]
    device_factory = Mock(side_effect=["recorded-device", "config-only-device"])
    monkeypatch.setattr(
        main_window_module,
        "create_offline_external_device",
        device_factory,
    )
    replay = SimpleNamespace(
        configuration_snapshot=Mock(
            return_value={
                "hardware": {
                    "external_devices": {
                        "GEIGER:7": {"pwm_enable": 1},
                        "PMT:9": {"pwm_enable": 0},
                    },
                },
                "application": {
                    "external_devices": {
                        "GEIGER:7": {"plotted_registers": ["pulses_per_sec"]},
                    },
                },
            },
        ),
    )
    events = [
        TelemetryReplayEvent(
            sequence=0,
            host_monotonic_ns=10,
            source="external.GEIGER:7",
            payload={
                "worker_timestamp": 1.0,
                "values": {"pulses_per_sec": 42},
            },
        ),
    ]

    controller._install_offline_external_replay_controllers(replay, events)

    assert device_factory.call_args_list == [
        call("GEIGER:7", {"pwm_enable": 1}, {"pulses_per_sec": 42}),
        call("PMT:9", {"pwm_enable": 0}, {}),
    ]
    panels[0].apply_configuration_settings.assert_called_once_with(
        {"plotted_registers": ["pulses_per_sec"]}
    )
    panels[1].apply_configuration_settings.assert_called_once_with(None)


def test_waveform_file_opens_standalone_workbench(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._window = object()
    controller._waveform_analysis_dialog = None
    path = Path("channel-1.bin")
    dialog = SimpleNamespace(show_workspace=Mock(), open_path=Mock())
    factory = Mock(return_value=dialog)
    monkeypatch.setattr(main_window_module, "WaveformAnalysisDialog", factory)

    controller.load_waveform_file(path)
    controller.show_waveform_analysis()

    factory.assert_called_once_with(parent=controller._window)
    assert dialog.show_workspace.call_count == 2
    dialog.open_path.assert_called_once_with(path)


def test_psd_file_opens_standalone_readback_workbench(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._window = object()
    controller._psd_readback_dialog = None
    path = Path("events.h5")
    dialog = SimpleNamespace(show_workspace=Mock(), open_path=Mock())
    factory = Mock(return_value=dialog)
    monkeypatch.setattr(main_window_module, "PsdReadbackDialog", factory)

    controller.load_psd_events(path)
    controller.show_psd_event_readback()

    factory.assert_called_once_with(parent=controller._window)
    assert dialog.show_workspace.call_count == 2
    dialog.open_path.assert_called_once_with(path)


def _bare_controller() -> MainWindowController:
    controller = object.__new__(MainWindowController)
    controller._host = "board.local"
    controller._port = 30431
    controller._scope_controllers = []
    controller._current_monitor_controllers = []
    return controller


def test_session_recording_attaches_buffers_and_enables_mca_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._session_recorder = None
    controller._mca_event_buffers = {0: object()}  # type: ignore[dict-item]
    controller._scope_frame_buffers = {0: object()}  # type: ignore[dict-item]
    mca = SimpleNamespace(channel=0, set_session_capture_enabled=Mock(return_value=True))
    controller._mca_controllers = [mca]
    configuration = {"format_version": 4}
    controller.configuration_document = Mock(return_value=configuration)  # type: ignore[method-assign]
    recorder = SimpleNamespace(
        is_recording=True,
        record_command=Mock(),
        mark_integrity_issue=Mock(),
    )
    factory = Mock(return_value=recorder)
    monkeypatch.setattr(main_window_module, "SessionRecorder", factory)
    path = Path("measurement.nlab-session")

    assert controller.start_session_recording(path) == path

    factory.assert_called_once_with(
        path,
        configuration,
        mca_event_buffers=controller._mca_event_buffers,
        scope_frame_buffers=controller._scope_frame_buffers,
    )
    mca.set_session_capture_enabled.assert_called_once_with(True)
    assert controller._session_recorder is recorder


def test_session_stop_still_finalizes_when_final_configuration_fails() -> None:
    controller = _bare_controller()
    mca = SimpleNamespace(set_session_capture_enabled=Mock())
    controller._mca_controllers = [mca]
    controller.configuration_document = Mock(side_effect=OSError("transport lost"))  # type: ignore[method-assign]
    path = Path("measurement.nlab-session")
    recorder = SimpleNamespace(
        record_command=Mock(),
        mark_integrity_issue=Mock(),
        stop=Mock(return_value=path),
    )
    controller._session_recorder = recorder

    assert controller.stop_session_recording() == path

    recorder.mark_integrity_issue.assert_called_once_with(
        "final_configuration_snapshot_failed"
    )
    recorder.stop.assert_called_once_with(final_configuration=None)
    mca.set_session_capture_enabled.assert_called_once_with(False)
    assert controller._session_recorder is None


def test_replay_state_is_retained_until_explicitly_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._session_replay = None
    controller._session_replay_path = None
    controller._session_replay_spectrum_ids = set()
    spectra = [SimpleNamespace(spectrum_id="replay-a"), SimpleNamespace(spectrum_id="replay-b")]
    replay = SimpleNamespace(
        replay_mca_spectra=Mock(return_value=spectra),
        scope_frames=Mock(return_value=[]),
        manifest={"started_monotonic_ns": 1},
        integrity="verified",
    )
    monkeypatch.setattr(main_window_module, "SessionReplay", Mock(return_value=replay))
    dialog = SimpleNamespace(
        show_workspace=Mock(),
        add_spectra=Mock(),
        remove_spectra=Mock(),
    )
    controller._mca_peak_analysis_dialog = dialog
    path = Path("measurement.nlab-session")

    assert controller.replay_session(path) is replay
    assert controller.session_replay is replay
    assert controller.session_replay_path == path
    dialog.add_spectra.assert_called_once_with(spectra)
    dialog.show_workspace.assert_not_called()

    controller.clear_session_replay()

    dialog.remove_spectra.assert_called_once_with({"replay-a", "replay-b"})
    assert controller.session_replay is None
    assert controller.session_replay_path is None


def test_offline_replay_rejects_missing_channel_before_clearing_current_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    old_replay = object()
    controller._session_replay = old_replay
    controller._session_recorder = None
    controller._coincidence_controller = None
    controller._offline = True
    controller._channels = 1
    replay = SimpleNamespace(
        replay_mca_spectra=Mock(return_value=[]),
        recorded_channels=Mock(return_value=[0, 1]),
    )
    monkeypatch.setattr(main_window_module, "SessionReplay", Mock(return_value=replay))

    with pytest.raises(RuntimeError, match="at least 2 channels"):
        controller.replay_session(Path("two-channel.nlab-session"))

    assert controller._session_replay is old_replay


def test_scope_replay_uses_recorded_timing_and_holds_at_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._session_replay = None
    controller._session_replay_path = None
    controller._session_replay_spectrum_ids = set()
    timer = SimpleNamespace(start=Mock(), stop=Mock())
    controller._session_scope_replay_timer = timer  # type: ignore[assignment]
    scope = SimpleNamespace(
        channel=0,
        begin_session_replay=Mock(),
        display_session_replay_frame=Mock(),
        finish_session_replay=Mock(),
        end_session_replay=Mock(),
    )
    controller._scope_controllers = [scope]
    frames = [
        ScopeReplayFrame(
            channel=0,
            sequence=0,
            host_monotonic_ns=200,
            samples=np.array([1, 2], dtype=np.int16),
            sample_period_ns=8,
            source="viewer",
        ),
        ScopeReplayFrame(
            channel=0,
            sequence=1,
            host_monotonic_ns=300,
            samples=np.array([3, 4], dtype=np.int16),
            sample_period_ns=8,
            source="viewer",
        ),
    ]
    replay = SimpleNamespace(
        replay_mca_spectra=Mock(return_value=[]),
        scope_frames=Mock(return_value=frames),
        manifest={"started_monotonic_ns": 100},
        integrity="verified",
    )
    monkeypatch.setattr(main_window_module, "SessionReplay", Mock(return_value=replay))
    monkeypatch.setattr(
        main_window_module,
        "monotonic_ns",
        Mock(side_effect=[1_000, 1_000, 1_100, 1_200]),
    )

    controller.replay_session(Path("measurement.nlab-session"))

    scope.begin_session_replay.assert_called_once_with(frame_count=2, duration_ns=200)
    scope.display_session_replay_frame.assert_not_called()
    timer.start.assert_called_once_with(1)

    controller._advance_session_scope_replay()

    assert scope.display_session_replay_frame.call_count == 1
    first_call = scope.display_session_replay_frame.call_args
    np.testing.assert_array_equal(first_call.args[0], [1, 2])
    assert first_call.kwargs["frame_index"] == 1
    scope.finish_session_replay.assert_not_called()

    controller._advance_session_scope_replay()

    assert scope.display_session_replay_frame.call_count == 2
    scope.finish_session_replay.assert_called_once_with(
        frame_count=2,
        duration_ns=200,
    )

    controller.clear_session_replay()
    scope.end_session_replay.assert_called_once_with()


def test_telemetry_replay_uses_recorded_timing_for_psu_and_external(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._session_replay = object()
    controller._session_telemetry_replay_timer = SimpleNamespace(start=Mock(), stop=Mock())
    psu = SimpleNamespace(
        begin_session_replay=Mock(),
        display_session_replay_readback=Mock(),
    )
    external = SimpleNamespace(
        configuration_id="sipm:3",
        begin_session_replay=Mock(),
        display_session_replay_readback=Mock(),
    )
    controller._psu_controller_by_device = {0: psu}
    controller._external_controllers = [external]
    replay = SimpleNamespace(manifest={"started_monotonic_ns": 100})
    events = [
        TelemetryReplayEvent(
            sequence=1,
            host_monotonic_ns=200,
            source="psu.ch0",
            payload={"readback": {"timestamp": 1.0, "hv_voltage": 42.0}},
        ),
        TelemetryReplayEvent(
            sequence=2,
            host_monotonic_ns=300,
            source="external.sipm:3",
            payload={"worker_timestamp": 2.0, "values": {"cpu_temp": 31.5}},
        ),
    ]
    monkeypatch.setattr(
        main_window_module,
        "monotonic_ns",
        Mock(side_effect=[1_000, 1_000, 1_100, 1_200]),
    )

    controller._start_session_telemetry_replay(replay, events)

    psu.begin_session_replay.assert_called_once_with()
    external.begin_session_replay.assert_called_once_with()
    psu.display_session_replay_readback.assert_not_called()
    controller._advance_session_telemetry_replay()
    psu.display_session_replay_readback.assert_called_once_with(
        {"timestamp": 1.0, "hv_voltage": 42.0}
    )
    external.display_session_replay_readback.assert_not_called()
    controller._advance_session_telemetry_replay()
    external.display_session_replay_readback.assert_called_once_with(
        2.0,
        {"cpu_temp": 31.5},
    )


def test_current_replay_uses_recorded_timing_and_holds_at_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._session_replay = object()
    controller._session_current_replay_timer = SimpleNamespace(start=Mock(), stop=Mock())
    current = SimpleNamespace(
        channel=0,
        begin_session_replay=Mock(),
        display_session_replay_samples=Mock(),
        finish_session_replay=Mock(),
    )
    controller._current_monitor_controllers = [current]
    samples = [
        CurrentReplaySample(0, 200, CurrentSample(0, 200, 10.0, 50)),
        CurrentReplaySample(0, 300, CurrentSample(1, 300, 11.0, 60)),
    ]
    replay = SimpleNamespace(
        manifest={"started_monotonic_ns": 100},
        current_summary=Mock(return_value=ReplayStreamSummary({0: 2}, 200, 300)),
        iter_current_samples=Mock(return_value=iter(samples)),
        current_dropped_samples=Mock(return_value=3),
        configuration_snapshot=Mock(
            return_value={
                "application": {
                    "channels": {"0": {"current_monitor": {"unit": "nA"}}}
                }
            }
        ),
    )
    monkeypatch.setattr(
        main_window_module,
        "monotonic_ns",
        Mock(side_effect=[1_000, 1_000, 1_100, 1_200]),
    )

    controller._start_session_current_replay(replay)

    current.begin_session_replay.assert_called_once_with(
        settings={"unit": "nA"},
        dropped_samples=3,
    )
    current.display_session_replay_samples.assert_not_called()
    controller._advance_session_current_replay()
    assert current.display_session_replay_samples.call_args.args[0][0].raw_code == 10.0
    controller._advance_session_current_replay()
    assert current.display_session_replay_samples.call_count == 2
    current.finish_session_replay.assert_called_once_with()


def test_global_telemetry_is_dispatched_during_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._session_replay = object()
    controller._session_telemetry_replay_timer = SimpleNamespace(start=Mock(), stop=Mock())
    controller._psu_controller_by_device = {}
    controller._external_controllers = []
    global_controller = SimpleNamespace(
        begin_session_replay=Mock(),
        display_session_replay_temperature=Mock(),
        display_session_replay_diagnostics=Mock(),
        finish_session_replay=Mock(),
    )
    controller._global_controller = global_controller
    events = [
        TelemetryReplayEvent(
            0,
            100,
            "global.temperature_correction",
            {"readback": {"temperature": 68.0}},
        ),
        TelemetryReplayEvent(
            1,
            200,
            "global.diagnostics",
            {"readings": [{"key": "pll", "value": True}]},
        ),
    ]
    replay = SimpleNamespace(manifest={"started_monotonic_ns": 100})
    monkeypatch.setattr(
        main_window_module,
        "monotonic_ns",
        Mock(side_effect=[1_000, 1_000, 1_100]),
    )

    controller._start_session_telemetry_replay(replay, events)

    global_controller.begin_session_replay.assert_called_once_with()
    global_controller.display_session_replay_temperature.assert_called_once_with(
        {"temperature": 68.0}
    )
    controller._advance_session_telemetry_replay()
    global_controller.display_session_replay_diagnostics.assert_called_once_with(
        [{"key": "pll", "value": True}]
    )
    global_controller.finish_session_replay.assert_called_once_with()


def test_analysis_replay_feeds_recorded_events_to_psd_and_coincidence() -> None:
    controller = _bare_controller()
    psd = SimpleNamespace(channel=0, begin_session_replay=Mock())
    mca0 = SimpleNamespace(channel=0)
    mca1 = SimpleNamespace(channel=1)
    coincidence = SimpleNamespace(begin_session_replay=Mock())
    controller._psd_controllers = [psd]
    controller._mca_controllers = [mca0, mca1]
    controller._coincidence_controller = coincidence
    events = {
        0: np.array([1, 2], dtype=np.int16),
        1: np.array([3, 4], dtype=np.int16),
    }
    settings = object()
    replay = SimpleNamespace(
        mca_readbacks=Mock(return_value=[]),
        mca_event_batches=Mock(
            return_value=[
                McaReplayBatch(0, 0, 100, events[0]),
                McaReplayBatch(1, 0, 100, events[1]),
            ]
        ),
        configuration_snapshot=Mock(
            return_value={
                "application": {"channels": {"0": {"psd": {"energy_bins": 64}}}}
            }
        ),
        coincidence_settings=Mock(return_value=(settings, {"operator": "AND"})),
        manifest={"started_monotonic_ns": 100},
    )

    controller._start_session_analysis_replay(replay)

    psd.begin_session_replay.assert_called_once_with(settings={"energy_bins": 64})
    coincidence.begin_session_replay.assert_called_once()
    passed_settings, passed_display = coincidence.begin_session_replay.call_args.args
    assert passed_settings is settings
    assert passed_display == {"operator": "AND"}


def test_progressive_analysis_replay_updates_live_views_at_recorded_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._session_replay = object()
    controller._session_analysis_replay_timer = SimpleNamespace(start=Mock(), stop=Mock())
    mca = SimpleNamespace(
        channel=0,
        display_session_replay_readback=Mock(),
        finish_session_replay=Mock(),
    )
    psd = SimpleNamespace(
        channel=0,
        display_session_replay_events=Mock(),
        finish_session_replay=Mock(),
    )
    coincidence = SimpleNamespace(
        display_session_replay_events=Mock(),
        finish_session_replay=Mock(),
    )
    controller._mca_controllers = [mca]
    controller._psd_controllers = [psd]
    controller._coincidence_controller = coincidence
    controller._session_replay_psd_channels = {0}
    controller._session_replay_coincidence_active = True
    controller._session_analysis_replay_coincidence_channels = {0: 0}
    controller._session_analysis_replay_mca_channels = {0}
    batch = McaReplayBatch(0, 0, 200, np.array([1, 2], dtype=np.int16))
    readback_value = SimpleNamespace(histogram=np.array([3, 4]))
    readback = McaReplayReadback(0, 0, 300, readback_value)  # type: ignore[arg-type]
    controller._session_analysis_replay_events = [batch, readback]
    controller._session_analysis_replay_index = 0
    controller._session_analysis_replay_origin_ns = 100
    controller._session_analysis_replay_started_ns = 1_000
    monkeypatch.setattr(
        main_window_module,
        "monotonic_ns",
        Mock(side_effect=[1_100, 1_200]),
    )

    controller._advance_session_analysis_replay()

    psd.display_session_replay_events.assert_called_once()
    coincidence.display_session_replay_events.assert_called_once_with(0, batch.events)
    mca.display_session_replay_readback.assert_not_called()

    controller._advance_session_analysis_replay()

    mca.display_session_replay_readback.assert_called_once_with(readback_value)
    mca.finish_session_replay.assert_called_once_with()
    psd.finish_session_replay.assert_called_once_with()
    coincidence.finish_session_replay.assert_called_once_with()


def test_replay_hardware_interlock_reaches_every_hardware_controller() -> None:
    controller = _bare_controller()
    targets = [
        SimpleNamespace(set_replay_hardware_interlock=Mock())
        for _index in range(7)
    ]
    controller._coincidence_controller = targets[0]
    controller._current_monitor_controllers = [targets[1]]
    controller._scope_controllers = [targets[2]]
    controller._mca_controllers = [targets[3]]
    controller._global_controller = targets[4]
    controller._psu_controllers = [targets[5]]
    controller._external_controllers = [targets[6]]
    controller._replay_hardware_interlocked = False

    controller._set_replay_hardware_interlock(True)
    controller._set_replay_hardware_interlock(False)

    for target in targets:
        if target in (
            controller._global_controller,
            *controller._psu_controllers,
            *controller._external_controllers,
        ):
            assert target.set_replay_hardware_interlock.call_args_list == [
                call(True, resume_background=True),
                call(False, resume_background=True),
            ]
        else:
            assert target.set_replay_hardware_interlock.call_args_list == [
                call(True),
                call(False),
            ]


def test_general_settings_read_first_channel_and_apply_to_every_channel() -> None:
    controller = _bare_controller()
    scope0 = SimpleNamespace(
        configuration_settings=lambda: {
            "display_mode": 1,
            "persistence": 750,
            "refresh_rate_hz": 20,
        },
        apply_configuration_settings=Mock(),
    )
    scope1 = SimpleNamespace(
        configuration_settings=lambda: {},
        apply_configuration_settings=Mock(),
    )
    mca0 = SimpleNamespace(
        configuration_settings=lambda: {"refresh_rate_hz": 8},
        apply_configuration_settings=Mock(),
    )
    mca1 = SimpleNamespace(
        configuration_settings=lambda: {},
        apply_configuration_settings=Mock(),
    )
    psu0 = SimpleNamespace(
        configuration_settings=lambda: {
            "refresh_interval_ms": 500,
            "plot_time_range_s": 120,
        },
        apply_configuration_settings=Mock(),
    )
    psu1 = SimpleNamespace(
        configuration_settings=lambda: {},
        apply_configuration_settings=Mock(),
    )
    controller._scope_controllers = [scope0, scope1]
    controller._mca_controllers = [mca0, mca1]
    controller._psu_controllers = [psu0, psu1]

    settings = controller.general_configuration_settings()

    assert settings == {
        "scope_display_mode": 1,
        "scope_persistence": 750,
        "scope_refresh_rate_hz": 20,
        "mca_refresh_rate_hz": 8,
        "psu_refresh_interval_ms": 500,
        "psu_plot_time_range_s": 120,
    }

    controller.apply_general_configuration_settings(settings)

    for scope in (scope0, scope1):
        scope.apply_configuration_settings.assert_called_once_with(
            {"display_mode": 1, "persistence": 750, "refresh_rate_hz": 20}
        )
    for mca in (mca0, mca1):
        mca.apply_configuration_settings.assert_called_once_with(
            {"refresh_rate_hz": 8}
        )
    for psu in (psu0, psu1):
        psu.apply_configuration_settings.assert_called_once_with(
            {"refresh_interval_ms": 500, "plot_time_range_s": 120}
        )


def test_differential_histogram_view_mode_applies_to_every_mca_channel() -> None:
    controller = _bare_controller()
    channels = [
        SimpleNamespace(set_differential_histogram=Mock()),
        SimpleNamespace(set_differential_histogram=Mock()),
    ]
    controller._mca_controllers = channels

    controller.set_differential_histogram(True)

    for channel in channels:
        channel.set_differential_histogram.assert_called_once_with(True)


def test_mca_histogram_hover_is_reported_across_channels() -> None:
    controller = object.__new__(MainWindowController)
    controller._mca_controllers = [
        SimpleNamespace(histogram_under_mouse=Mock(return_value=False)),
        SimpleNamespace(histogram_under_mouse=Mock(return_value=True)),
    ]

    assert controller.mca_histogram_under_mouse()


def test_iio_startup_explicitly_requests_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _bare_controller()
    expected = object()
    factory = Mock(return_value=expected)
    monkeypatch.setattr(main_window_module.Digitizer, "from_iio", factory)

    result = controller._connect_channel(0)

    assert result is expected
    factory.assert_called_once_with(
        channel=0,
        uri="ip:board.local:30431",
        with_ids=True,
    )


def test_startup_builds_one_psu_controller_and_dock_per_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    supplies = [object(), object()]
    controller._devices = [
        SimpleNamespace(
            scope=object(),
            scope_dma=None,
            mca=object(),
            mca_dma=None,
            hv=supply,
            capabilities=SimpleNamespace(mca=False, mca_dma=False),
            mca_available=lambda: False,
        )
        for supply in supplies
    ]
    controller._scope_controllers = []
    controller._mca_controllers = []
    controller._psd_controllers = []
    controller._psd_controller_by_device = {}
    controller._psu_controllers = []
    controller._scope_dock_host = object()
    controller._current_dock_host = object()
    controller._mca_dock_host = object()
    controller._psd_dock_host = object()
    controller._psu_dock_host = object()

    tab_current = SimpleNamespace(setToolTip=Mock())
    tab_mca = SimpleNamespace(setToolTip=Mock())
    tab_psd = SimpleNamespace(setToolTip=Mock())
    tab_psu = SimpleNamespace(setToolTip=Mock())
    main_tabs = SimpleNamespace(
        indexOf=Mock(side_effect=lambda tab: 7 if tab is tab_mca else 8),
        setTabEnabled=Mock(),
    )
    controller._window = SimpleNamespace(
        ui=SimpleNamespace(
            layoutTabScope=SimpleNamespace(addWidget=Mock()),
            layoutTabCurrent=SimpleNamespace(addWidget=Mock()),
            layoutTabMCA=SimpleNamespace(addWidget=Mock()),
            layoutTabPSD=SimpleNamespace(addWidget=Mock()),
            layoutTabPSU=SimpleNamespace(addWidget=Mock()),
            mainTabs=main_tabs,
            tabCurrent=tab_current,
            tabMCA=tab_mca,
            tabPSD=tab_psd,
            tabPSU=tab_psu,
        )
    )

    scope_factory = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace())
    psu_factory = Mock(side_effect=lambda hv: SimpleNamespace(hv=hv))
    monkeypatch.setattr(main_window_module, "ScopeController", scope_factory)
    monkeypatch.setattr(main_window_module, "PSUController", psu_factory)

    made_docks: list[tuple[str, str, object]] = []
    controller._make_dock = Mock(
        side_effect=lambda name, title, widget: (
            made_docks.append((name, title, widget)) or SimpleNamespace()
        )
    )
    controller._populate_dock_host = Mock()

    controller._build_channel_docks()

    assert psu_factory.call_args_list == [call(supplies[0]), call(supplies[1])]
    assert len(controller._psu_controllers) == 2
    assert controller._psu_controller_by_device == {
        0: controller._psu_controllers[0],
        1: controller._psu_controllers[1],
    }
    assert [name for name, _, _ in made_docks if name.startswith("psu_")] == [
        "psu_ch0",
        "psu_ch1",
    ]
    controller._window.ui.layoutTabPSU.addWidget.assert_called_once_with(controller._psu_dock_host)
    main_tabs.setTabEnabled.assert_any_call(8, True)
    tab_psu.setToolTip.assert_called_once_with("")


def test_startup_disables_psu_tab_when_backend_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._devices = [
        SimpleNamespace(
            scope=object(),
            scope_dma=None,
            mca=object(),
            mca_dma=None,
            hv=None,
            capabilities=SimpleNamespace(mca=False, mca_dma=False),
            mca_available=lambda: False,
        )
    ]
    controller._scope_controllers = []
    controller._mca_controllers = []
    controller._psd_controllers = []
    controller._psd_controller_by_device = {}
    controller._psu_controllers = []
    controller._scope_dock_host = object()
    controller._current_dock_host = object()
    controller._mca_dock_host = object()
    controller._psd_dock_host = object()
    controller._psu_dock_host = object()
    tab_current = SimpleNamespace(setToolTip=Mock())
    tab_mca = SimpleNamespace(setToolTip=Mock())
    tab_psd = SimpleNamespace(setToolTip=Mock())
    tab_psu = SimpleNamespace(setToolTip=Mock())
    main_tabs = SimpleNamespace(
        indexOf=Mock(side_effect=lambda tab: 7 if tab is tab_mca else 8),
        setTabEnabled=Mock(),
    )
    controller._window = SimpleNamespace(
        ui=SimpleNamespace(
            layoutTabScope=SimpleNamespace(addWidget=Mock()),
            layoutTabCurrent=SimpleNamespace(addWidget=Mock()),
            layoutTabMCA=SimpleNamespace(addWidget=Mock()),
            layoutTabPSD=SimpleNamespace(addWidget=Mock()),
            layoutTabPSU=SimpleNamespace(addWidget=Mock()),
            mainTabs=main_tabs,
            tabCurrent=tab_current,
            tabMCA=tab_mca,
            tabPSD=tab_psd,
            tabPSU=tab_psu,
        )
    )
    monkeypatch.setattr(
        main_window_module,
        "ScopeController",
        Mock(return_value=SimpleNamespace()),
    )
    controller._make_dock = Mock(return_value=SimpleNamespace())
    controller._populate_dock_host = Mock()

    controller._build_channel_docks()

    assert controller._psu_controllers == []
    assert controller._psu_controller_by_device == {}
    main_tabs.setTabEnabled.assert_any_call(8, False)
    assert "no IDS/HV backend" in tab_psu.setToolTip.call_args.args[0]


def test_startup_builds_mca_features_only_for_capable_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    dma = object()
    controller._devices = [
        SimpleNamespace(
            scope=object(),
            scope_dma=None,
            mca=object(),
            mca_dma=dma,
            hv=None,
            capabilities=SimpleNamespace(mca=True, mca_dma=True),
            mca_available=lambda: True,
        ),
        SimpleNamespace(
            scope=object(),
            scope_dma=None,
            mca=object(),
            mca_dma=None,
            hv=None,
            capabilities=SimpleNamespace(mca=False, mca_dma=False),
            mca_available=lambda: False,
        ),
    ]
    controller._scope_controllers = []
    controller._mca_controllers = []
    controller._psd_controllers = []
    controller._psd_controller_by_device = {}
    controller._psu_controllers = []
    controller._scope_dock_host = object()
    controller._current_dock_host = object()
    controller._mca_dock_host = object()
    controller._psd_dock_host = object()
    controller._psu_dock_host = object()

    tab_current = SimpleNamespace(setToolTip=Mock())
    tab_mca = SimpleNamespace(setToolTip=Mock())
    tab_psd = SimpleNamespace(setToolTip=Mock())
    tab_psu = SimpleNamespace(setToolTip=Mock())
    main_tabs = SimpleNamespace(indexOf=Mock(return_value=1), setTabEnabled=Mock())
    controller._window = SimpleNamespace(
        ui=SimpleNamespace(
            layoutTabScope=SimpleNamespace(addWidget=Mock()),
            layoutTabCurrent=SimpleNamespace(addWidget=Mock()),
            layoutTabMCA=SimpleNamespace(addWidget=Mock()),
            layoutTabPSD=SimpleNamespace(addWidget=Mock()),
            layoutTabPSU=SimpleNamespace(addWidget=Mock()),
            mainTabs=main_tabs,
            tabCurrent=tab_current,
            tabMCA=tab_mca,
            tabPSD=tab_psd,
            tabPSU=tab_psu,
        )
    )

    scopes = [object(), object()]
    psd = object()
    mca = object()
    scope_factory = Mock(side_effect=scopes)
    current_factory = Mock(return_value=object())
    psd_factory = Mock(return_value=psd)
    mca_factory = Mock(return_value=mca)
    monkeypatch.setattr(main_window_module, "ScopeController", scope_factory)
    monkeypatch.setattr(main_window_module, "CurrentMonitorController", current_factory)
    monkeypatch.setattr(main_window_module, "PSDController", psd_factory)
    monkeypatch.setattr(main_window_module, "MCAController", mca_factory)
    controller._make_dock = Mock(return_value=SimpleNamespace())
    controller._populate_dock_host = Mock()

    controller._build_channel_docks()

    current_accumulator = scope_factory.call_args_list[0].kwargs["current_accumulator"]
    assert isinstance(
        current_accumulator, main_window_module.ScopeCurrentAccumulator
    )
    current_factory.assert_called_once_with(
        controller._devices[0].mca,
        channel=0,
        auto_start=False,
        scope_controller=scopes[0],
        scope_current_accumulator=current_accumulator,
    )
    assert controller._current_monitor_controllers == [current_factory.return_value]
    assert controller._current_monitor_controller_by_device == {
        0: current_factory.return_value
    }
    psd_buffer = psd_factory.call_args.kwargs["event_buffer"]
    assert isinstance(psd_buffer, main_window_module.McaEventBuffer)
    assert mca_factory.call_args.kwargs["event_buffer"] is psd_buffer
    assert mca_factory.call_args.kwargs["psd_capture"] is psd
    assert controller._psd_controllers == [psd]
    assert controller._psd_controller_by_device == {0: psd}
    assert controller._mca_controller_by_device == {0: mca}
    main_tabs.setTabEnabled.assert_any_call(1, True)
    assert "channel(s): 1" in tab_mca.setToolTip.call_args.args[0]


def test_startup_builds_one_global_controller_for_all_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._devices = [object(), object()]
    controller._global_dock_host = object()
    layout = SimpleNamespace(indexOf=Mock(return_value=-1), addWidget=Mock())
    controller._window = SimpleNamespace(
        ui=SimpleNamespace(layoutTabGlobal=layout),
    )
    expected = object()
    expected_dock = object()
    factory = Mock(return_value=expected)
    monkeypatch.setattr(main_window_module, "GlobalController", factory)
    controller._make_dock = Mock(return_value=expected_dock)
    controller._populate_dock_host = Mock()

    controller._build_global_tab()

    factory.assert_called_once_with(controller._devices, [0, 1])
    assert controller._global_controller is expected
    controller._make_dock.assert_called_once_with(
        "global_panel",
        "Global",
        expected,
    )
    controller._populate_dock_host.assert_called_once_with(
        controller._global_dock_host,
        [expected_dock],
    )
    layout.addWidget.assert_called_once_with(controller._global_dock_host)


def test_startup_discovers_external_modules_and_builds_one_dock_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    modules = [
        SimpleNamespace(
            device_type=SimpleNamespace(name="SIPM"),
            device_id=3,
        ),
        SimpleNamespace(
            device_type=SimpleNamespace(name="GEIGER"),
            device_id=7,
        ),
    ]
    controller._external_devices = SimpleNamespace(
        discover=Mock(return_value=modules),
    )
    controller._external_controllers = []
    controller._external_dock_host = object()

    tab_external = SimpleNamespace(setToolTip=Mock())
    layout = SimpleNamespace(indexOf=Mock(return_value=-1), addWidget=Mock())
    main_tabs = SimpleNamespace(indexOf=Mock(return_value=9), setTabEnabled=Mock())
    controller._window = SimpleNamespace(
        ui=SimpleNamespace(
            layoutTabExternal=layout,
            mainTabs=main_tabs,
            tabExternal=tab_external,
        ),
    )

    created_controllers = [object(), object()]
    factory = Mock(side_effect=created_controllers)
    monkeypatch.setattr(main_window_module, "ExternalDeviceController", factory)
    docks = [object(), object()]
    controller._make_dock = Mock(side_effect=docks)
    controller._populate_dock_host = Mock()

    controller._build_external_docks()

    controller._external_devices.discover.assert_called_once_with("board.local")
    assert factory.call_args_list == [call(modules[0]), call(modules[1])]
    assert controller._external_controllers == created_controllers
    assert controller._make_dock.call_args_list == [
        call("external_0", "Sipm #3", created_controllers[0]),
        call("external_1", "Geiger #7", created_controllers[1]),
    ]
    controller._populate_dock_host.assert_called_once_with(
        controller._external_dock_host,
        docks,
    )
    layout.addWidget.assert_called_once_with(controller._external_dock_host)
    main_tabs.setTabEnabled.assert_called_once_with(9, True)
    tab_external.setToolTip.assert_called_once_with("")


def test_external_tab_is_disabled_when_discovery_finds_no_modules() -> None:
    controller = _bare_controller()
    controller._external_devices = SimpleNamespace(discover=Mock(return_value=[]))
    controller._external_controllers = []
    controller._external_dock_host = object()

    tab_external = SimpleNamespace(setToolTip=Mock())
    layout = SimpleNamespace(indexOf=Mock(return_value=0), addWidget=Mock())
    main_tabs = SimpleNamespace(indexOf=Mock(return_value=9), setTabEnabled=Mock())
    controller._window = SimpleNamespace(
        ui=SimpleNamespace(
            layoutTabExternal=layout,
            mainTabs=main_tabs,
            tabExternal=tab_external,
        ),
    )
    controller._populate_dock_host = Mock()

    controller._build_external_docks()

    controller._external_devices.discover.assert_called_once_with("board.local")
    controller._populate_dock_host.assert_called_once_with(
        controller._external_dock_host,
        [],
    )
    layout.addWidget.assert_not_called()
    main_tabs.setTabEnabled.assert_called_once_with(9, False)
    assert "No external Modbus modules" in tab_external.setToolTip.call_args.args[0]


def test_shutdown_requests_all_pollers_before_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    events: list[str] = []

    def record(name: str) -> Mock:
        return Mock(side_effect=lambda *args: events.append(name))

    controller._global_controller = SimpleNamespace(
        request_polling_stop=record("request-global"),
        disarm_sync=record("disarm-global"),
        stop_polling_sync=record("wait-global"),
    )
    controller._scope_controllers = [
        SimpleNamespace(
            _refresh_timer=SimpleNamespace(stop=record("stop-scope-timer")),
            stop_dma_sync=record("wait-scope-dma"),
        )
    ]
    controller._mca_controllers = [
        SimpleNamespace(
            stop_dma_sync=record("wait-mca-dma"),
            stop_worker_sync=record("wait-mca-worker"),
        )
    ]
    controller._psd_controllers = [SimpleNamespace(stop_processing=record("stop-psd"))]
    controller._psu_controllers = [
        SimpleNamespace(
            request_monitor_stop=record("request-psu"),
            stop_monitor_sync=record("wait-psu"),
        )
    ]
    controller._external_controllers = [
        SimpleNamespace(
            request_polling_stop=record("request-external"),
            stop_polling_sync=record("wait-external"),
        )
    ]
    controller._current_monitor_controllers = [
        SimpleNamespace(
            request_monitor_stop=record("request-current"),
            stop_monitor_sync=record("wait-current"),
        )
    ]
    controller._thread = None
    thread_pool = SimpleNamespace(waitForDone=record("wait-thread-pool"))
    monkeypatch.setattr(
        main_window_module.QThreadPool,
        "globalInstance",
        Mock(return_value=thread_pool),
    )

    controller._stop_all_workers()

    assert events[:4] == [
        "request-global",
        "request-psu",
        "request-external",
        "request-current",
    ]
    first_wait = min(index for index, event in enumerate(events) if event.startswith("wait-"))
    assert all(
        events.index(request) < first_wait
        for request in (
            "request-global",
            "request-psu",
            "request-external",
            "request-current",
        )
    )


def test_global_stop_request_keeps_worker_alive_and_emits_only_once() -> None:
    request_shutdown = Mock()
    worker = SimpleNamespace(request_shutdown=request_shutdown)
    controller = SimpleNamespace(
        _temperature_worker=worker,
        _temperature_worker_stop_requested=False,
    )

    GlobalController._request_temperature_correction_stop(controller)  # type: ignore[arg-type]
    GlobalController._request_temperature_correction_stop(controller)  # type: ignore[arg-type]

    request_shutdown.assert_called_once_with()
    assert controller._temperature_worker is worker
    assert controller._temperature_worker_stop_requested


def test_global_stop_request_tolerates_already_deleted_qobject() -> None:
    controller = SimpleNamespace(
        _worker=SimpleNamespace(
            request_shutdown=Mock(side_effect=RuntimeError("Signal source has been deleted")),
        ),
        _worker_stop_requested=False,
    )

    GlobalController._request_diagnostics_stop(controller)  # type: ignore[arg-type]

    assert controller._worker_stop_requested


def test_peak_analysis_workbench_is_modeless_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller._mca_controllers = [object(), object()]
    controller._window = object()
    controller._mca_peak_analysis_dialog = None
    dialog = SimpleNamespace(show_workspace=Mock())
    factory = Mock(return_value=dialog)
    monkeypatch.setattr(main_window_module, "McaPeakAnalysisDialog", factory)

    controller.show_mca_peak_analysis()
    controller.show_mca_peak_analysis()

    factory.assert_called_once_with(controller._mca_controllers, parent=controller._window)
    assert dialog.show_workspace.call_count == 2
