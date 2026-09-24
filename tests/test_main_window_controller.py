from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from nlab.controllers import main_window_controller as main_window_module
from nlab.controllers.global_controller import GlobalController
from nlab.controllers.main_window_controller import MainWindowController


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
        backend="iio",
        host="192.0.2.10",
        port=30431,
        channels=2,
        on_progress=progress.append,
    )

    assert connected == [1, 2]
    assert progress == [
        "Connecting channel 1 of 2...",
        "Connecting channel 2 of 2...",
        "Preparing global controls...",
        "Preparing channel views...",
        "Preparing coincidence view...",
        "Discovering external devices...",
        "Restoring dock layout...",
    ]


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


def _bare_controller(*, backend: str = "iio") -> MainWindowController:
    controller = object.__new__(MainWindowController)
    controller._backend = backend
    controller._host = "board.local"
    controller._port = 30431 if backend == "iio" else 50050
    controller._current_monitor_controllers = []
    return controller


def test_iio_startup_explicitly_requests_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _bare_controller()
    expected = object()
    factory = Mock(return_value=expected)
    monkeypatch.setattr(main_window_module.Digitizer, "from_iio", factory)

    result = controller._connect_channel(1)

    assert result is expected
    factory.assert_called_once_with(
        channel=0,
        uri="ip:board.local:30431",
        with_ids=True,
    )


def test_grpc_startup_explicitly_requests_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _bare_controller(backend="grpc")
    expected = object()
    factory = Mock(return_value=expected)
    monkeypatch.setattr(main_window_module.Digitizer, "from_grpc", factory)

    result = controller._connect_channel(1)

    assert result is expected
    factory.assert_called_once_with(
        channel=1,
        hostname="board.local",
        port=50050,
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


def test_startup_shares_bounded_event_buffer_with_mca_and_psd(
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
            mca_available=lambda: True,
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

    scope = object()
    psd = object()
    mca = object()
    scope_factory = Mock(return_value=scope)
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

    scope_buffer = scope_factory.call_args.kwargs["dma_frame_buffer"]
    assert isinstance(scope_buffer, main_window_module.ScopeFrameBuffer)
    current_factory.assert_called_once_with(
        controller._devices[0].mca,
        channel=0,
        auto_start=False,
        scope_controller=scope,
        scope_frame_buffer=scope_buffer,
    )
    assert controller._current_monitor_controllers == [current_factory.return_value]
    psd_buffer = psd_factory.call_args.kwargs["event_buffer"]
    assert isinstance(psd_buffer, main_window_module.McaEventBuffer)
    assert mca_factory.call_args.kwargs["event_buffer"] is psd_buffer
    assert mca_factory.call_args.kwargs["psd_capture"] is psd
    assert controller._psd_controllers == [psd]
    assert controller._psd_controller_by_device == {0: psd}
    main_tabs.setTabEnabled.assert_any_call(1, True)


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
