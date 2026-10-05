from __future__ import annotations

from PySide6.QtWidgets import QCheckBox
from pytestqt.qtbot import QtBot

from nlab.controllers.external_device_controller import ExternalDeviceController
from nlab.hardware.digitizer.offline import (
    create_offline_digitizer,
    create_offline_external_device,
)


def test_offline_digitizer_exposes_replay_capabilities_without_transport() -> None:
    digitizer = create_offline_digitizer(0)

    assert digitizer.capabilities.mca
    assert digitizer.capabilities.mca_dma
    assert digitizer.capabilities.ids
    assert digitizer.hv is not None

    digitizer.scope.set_trigger_level(123)
    digitizer.mca.set_energy_bin(4)
    assert digitizer.scope.get_trigger_level() == 123
    assert digitizer.mca.get_energy_bin() == 4
    digitizer.close()


def test_recorded_external_module_is_read_only_and_accepts_replay_values(
    qtbot: QtBot,
) -> None:
    device = create_offline_external_device(
        "GEIGER:7",
        {"pwm_enable": 1},
        {"pulses_per_sec": 12, "dose_level_msvh": 0.25},
    )
    controller = ExternalDeviceController(device, auto_start=False)
    qtbot.addWidget(controller)
    controller.set_replay_hardware_interlock(True, resume_background=False)
    controller.begin_session_replay()

    controller.display_session_replay_readback(
        1.5,
        {"pulses_per_sec": 42, "dose_level_msvh": 0.5},
    )

    assert controller.configuration_id == "GEIGER:7"
    row = controller._row_by_name["telemetry:pulses_per_sec"]
    item = controller.ui.telemetryTable.item(row, 1)
    assert item is not None and item.text() == "42"
    assert not controller.ui.groupSettings.isEnabled()
    assert controller._worker_thread is None
    plotted = controller.ui.telemetryTable.cellWidget(row, 3)
    assert isinstance(plotted, QCheckBox)
