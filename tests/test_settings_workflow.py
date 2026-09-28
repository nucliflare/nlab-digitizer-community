from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QByteArray

from nlab.controllers.main_window_controller import MainWindowController
from nlab.utils import settings_io


def test_menu_save_writes_one_document_with_each_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = cast(Any, object.__new__(MainWindowController))
    controller._backend = "iio"
    controller._host = "board.local"
    controller._port = 30431
    controller._channels = 2
    controller._devices = [
        SimpleNamespace(
            scope=f"scope-{channel}",
            mca=f"mca-{channel}",
            hv=f"hv-{channel}",
            mca_available=lambda: True,
        )
        for channel in range(2)
    ]
    controller._scope_controllers = [
        SimpleNamespace(configuration_settings=lambda: {"refresh_rate_hz": 10}),
        SimpleNamespace(configuration_settings=lambda: {"refresh_rate_hz": 20}),
    ]
    controller._current_monitor_controllers = [
        SimpleNamespace(
            configuration_settings=lambda: {
                "zero_code": 12.5,
                "scale_per_code": 0.25,
                "unit": "nA",
            }
        ),
        SimpleNamespace(
            configuration_settings=lambda: {
                "zero_code": -3.0,
                "scale_per_code": 0.5,
                "unit": "nA",
            }
        ),
    ]
    controller._mca_controllers = [
        SimpleNamespace(
            hardware_configuration_settings=lambda: {"low_pass_preset": 0},
            configuration_settings=lambda: {"refresh_rate_hz": 5},
        ),
        SimpleNamespace(
            hardware_configuration_settings=lambda: {"low_pass_preset": 1},
            configuration_settings=lambda: {"refresh_rate_hz": 6},
        ),
    ]
    controller._psd_controllers = [
        SimpleNamespace(configuration_settings=lambda: {"ratio_cut": 0.2}),
        SimpleNamespace(configuration_settings=lambda: {"ratio_cut": 0.3}),
    ]
    controller._psd_controller_by_device = {
        0: controller._psd_controllers[0],
        1: controller._psd_controllers[1],
    }
    controller._psu_controllers = [
        SimpleNamespace(
            hardware_configuration_settings=lambda: {"hv_voltage": 100.0},
            configuration_settings=lambda: {"refresh_interval_ms": 1000},
        ),
        SimpleNamespace(
            hardware_configuration_settings=lambda: {"hv_voltage": 200.0},
            configuration_settings=lambda: {"refresh_interval_ms": 2000},
        ),
    ]
    controller._psu_controller_by_device = {
        0: controller._psu_controllers[0],
        1: controller._psu_controllers[1],
    }
    controller._global_controller = SimpleNamespace(
        hardware_configuration_settings=lambda: {"enabled": False},
        configuration_settings=lambda: {"diagnostics_interval_ms": 1000},
    )
    controller._external_controllers = []
    controller._window = SimpleNamespace(configuration_settings=lambda: {"show_roi": True})
    dock = SimpleNamespace(saveState=lambda: QByteArray(b"dock-state"))
    controller._scope_dock_host = dock
    controller._current_dock_host = dock
    controller._mca_dock_host = dock
    controller._psd_dock_host = dock
    controller._psu_dock_host = dock
    controller._global_dock_host = dock
    controller._external_dock_host = dock

    collect = Mock(side_effect=lambda scope, *args, **kwargs: {"source": scope})
    write = Mock()
    monkeypatch.setattr(settings_io, "collect_channel_hardware", collect)
    monkeypatch.setattr(settings_io, "write_configuration", write)
    path = tmp_path / "all-settings.yaml"

    controller.save_all_settings(path)

    write.assert_called_once()
    assert write.call_args.args[0] == Path(path)
    document = write.call_args.args[1]
    assert document["hardware"]["channels"] == {
        "0": {"source": "scope-0"},
        "1": {"source": "scope-1"},
    }
    assert set(document["application"]["channels"]) == {"0", "1"}
    assert document["application"]["channels"]["0"]["psd"] == {"ratio_cut": 0.2}
    assert document["application"]["channels"]["1"]["psd"] == {"ratio_cut": 0.3}
    assert document["application"]["channels"]["0"]["current_monitor"] == {
        "zero_code": 12.5,
        "scale_per_code": 0.25,
        "unit": "nA",
    }
    assert "current" in document["application"]["dock_layout"]
    assert "psd" in document["application"]["dock_layout"]
    assert document["connection"] == {
        "backend": "iio",
        "ip": "board.local",
        "port": 30431,
        "channels": 2,
    }
