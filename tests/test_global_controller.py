from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

from nlab.controllers.global_controller import GlobalController
from nlab.hardware.digitizer.diagnostics import GlobalDiagnosticReading


class _Sync:
    def __init__(self) -> None:
        self.enable = True
        self.source = 0
        self.software = 0
        self.writes: list[int] = []
        self.events: list[tuple[str, int | bool]] = []

    def get_enable(self) -> bool:
        return self.enable

    def get_trig_src(self) -> int:
        return self.source

    def get_sw_trig(self) -> int:
        return self.software

    def set_sw_trig(self, value: int) -> None:
        self.software = value
        self.writes.append(value)
        self.events.append(("software", value))

    def set_enable(self, value: bool) -> None:
        self.enable = value
        self.events.append(("enable", value))


class _MCA:
    def __init__(self, *, armed: bool = True, external: bool = True) -> None:
        self._armed = armed
        self._external = external

    def get_global_enable(self) -> bool:
        return self._armed

    def get_ext_trig_enable(self) -> bool:
        return self._external


def _controller(sync: _Sync, *mcas: _MCA) -> GlobalController:
    controller: Any = GlobalController.__new__(GlobalController)
    controller._sync = sync
    controller._devices = tuple(SimpleNamespace(mca=mca) for mca in mcas)
    controller._channel_labels = tuple(range(len(mcas)))
    controller.ui = SimpleNamespace(
        lblSoftwareState=SimpleNamespace(setText=Mock()),
        lblSyncStatus=SimpleNamespace(setText=Mock(), setStyleSheet=Mock()),
    )
    return cast(GlobalController, controller)


def test_software_start_fires_only_after_all_channels_are_armed() -> None:
    sync = _Sync()
    controller = _controller(sync, _MCA(), _MCA())

    controller._start_armed_channels()

    assert sync.writes == [1]
    ui = cast(Any, controller.ui)
    ui.lblSoftwareState.setText.assert_called_once_with("HIGH")


def test_software_start_reports_every_unready_channel_without_firing() -> None:
    sync = _Sync()
    controller = _controller(
        sync,
        _MCA(armed=False),
        _MCA(external=False),
    )

    controller._start_armed_channels()

    assert sync.writes == []
    ui = cast(Any, controller.ui)
    status = ui.lblSyncStatus.setText.call_args.args[0]
    assert "Ch 0 is not armed" in status
    assert "Ch 1 has External Trigger disabled" in status


def test_global_start_output_is_reset_to_safe_state_during_initialization() -> None:
    sync = _Sync()
    sync.software = 1
    controller = _controller(sync, _MCA(), _MCA())

    controller._reset_sync_for_initialization()

    assert sync.events == [("enable", False), ("software", 0)]
    assert sync.enable is False
    assert sync.software == 0


def test_diagnostic_value_formatting_respects_type_and_precision() -> None:
    assert GlobalController._format_diagnostic_value(
        GlobalDiagnosticReading("temp", "Temperature", 46.875, "°C", 1)
    ) == "46.9"
    assert GlobalController._format_diagnostic_value(
        GlobalDiagnosticReading("pll", "PLL", True, healthy=True)
    ) == "Yes"
    assert GlobalController._format_diagnostic_value(
        GlobalDiagnosticReading("clock", "Clock", 500_000_000, "Hz")
    ) == "500,000,000"


def test_temperature_parameter_edit_is_forwarded_to_worker() -> None:
    controller: Any = GlobalController.__new__(GlobalController)
    emit = Mock()
    controller._temperature_worker = SimpleNamespace(
        change_parameters=SimpleNamespace(emit=emit),
    )
    controller.ui = SimpleNamespace(
        spinTempCoeff=SimpleNamespace(value=Mock(return_value=-0.25)),
        spinTempOffset=SimpleNamespace(value=Mock(return_value=7)),
    )

    controller._send_temperature_parameters()

    emit.assert_called_once_with(-0.25, 7)


def test_temperature_readback_updates_global_panel() -> None:
    from nlab.workers.temperature_correction_worker import (
        TemperatureCorrectionReadback,
    )

    controller: Any = GlobalController.__new__(GlobalController)
    controller._devices = (object(), object())
    controller.ui = SimpleNamespace(
        lblAdsTemperature=SimpleNamespace(setText=Mock()),
        lblAppliedTempCoeff=SimpleNamespace(setText=Mock()),
        lblAppliedTempOffset=SimpleNamespace(setText=Mock()),
        lblTemperatureStatus=SimpleNamespace(setText=Mock(), setStyleSheet=Mock()),
    )

    controller._on_temperature_readback(
        TemperatureCorrectionReadback(69.0, -0.0005347, 14),
    )

    controller.ui.lblAdsTemperature.setText.assert_called_once_with("69 raw")
    controller.ui.lblAppliedTempCoeff.setText.assert_called_once_with("-0.000534700")
    controller.ui.lblAppliedTempOffset.setText.assert_called_once_with("14")
    assert "2 MCA channel" in controller.ui.lblTemperatureStatus.setText.call_args.args[0]
