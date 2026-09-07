from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from nlab.controllers import main_window_controller as main_window_module
from nlab.controllers.main_window_controller import MainWindowController


def _bare_controller(*, backend: str = "iio") -> MainWindowController:
    controller = object.__new__(MainWindowController)
    controller._backend = backend
    controller._host = "board.local"
    controller._port = 30431 if backend == "iio" else 50050
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
    controller._psu_controllers = []
    controller._scope_dock_host = object()
    controller._mca_dock_host = object()
    controller._psu_dock_host = object()

    tab_mca = SimpleNamespace(setToolTip=Mock())
    tab_psu = SimpleNamespace(setToolTip=Mock())
    main_tabs = SimpleNamespace(
        indexOf=Mock(side_effect=lambda tab: 7 if tab is tab_mca else 8),
        setTabEnabled=Mock(),
    )
    controller._window = SimpleNamespace(
        ui=SimpleNamespace(
            layoutTabScope=SimpleNamespace(addWidget=Mock()),
            layoutTabMCA=SimpleNamespace(addWidget=Mock()),
            layoutTabPSU=SimpleNamespace(addWidget=Mock()),
            mainTabs=main_tabs,
            tabMCA=tab_mca,
            tabPSU=tab_psu,
        )
    )

    scope_factory = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace())
    psu_factory = Mock(side_effect=lambda hv: SimpleNamespace(hv=hv))
    monkeypatch.setattr(main_window_module, "ScopeController", scope_factory)
    monkeypatch.setattr(main_window_module, "PSUController", psu_factory)

    made_docks: list[tuple[str, str, object]] = []
    controller._make_dock = Mock(
        side_effect=lambda name, title, widget: made_docks.append(
            (name, title, widget)
        ) or SimpleNamespace()
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
    controller._window.ui.layoutTabPSU.addWidget.assert_called_once_with(
        controller._psu_dock_host
    )
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
    controller._psu_controllers = []
    controller._scope_dock_host = object()
    controller._mca_dock_host = object()
    controller._psu_dock_host = object()
    tab_mca = SimpleNamespace(setToolTip=Mock())
    tab_psu = SimpleNamespace(setToolTip=Mock())
    main_tabs = SimpleNamespace(
        indexOf=Mock(side_effect=lambda tab: 7 if tab is tab_mca else 8),
        setTabEnabled=Mock(),
    )
    controller._window = SimpleNamespace(
        ui=SimpleNamespace(
            layoutTabScope=SimpleNamespace(addWidget=Mock()),
            layoutTabMCA=SimpleNamespace(addWidget=Mock()),
            layoutTabPSU=SimpleNamespace(addWidget=Mock()),
            mainTabs=main_tabs,
            tabMCA=tab_mca,
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
        "global_panel", "Global", expected,
    )
    controller._populate_dock_host.assert_called_once_with(
        controller._global_dock_host, [expected_dock],
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
