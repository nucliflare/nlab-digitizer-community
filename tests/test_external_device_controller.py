from types import SimpleNamespace
from typing import cast

from nlab_modbus.core.base_modbus_device import BaseModbusDevice
from nlab_modbus.maps.sipm_map import SIPM_REGISTER_MAP
from PySide6.QtWidgets import QScrollArea
from pytestqt.qtbot import QtBot

from nlab.controllers.external_device_controller import ExternalDeviceController


def _sipm_device() -> BaseModbusDevice:
    return cast(
        BaseModbusDevice,
        SimpleNamespace(
            REGISTER_MAP=SIPM_REGISTER_MAP,
            READOUT_START=3,
            READOUT_STOP=22,
            device_type=SimpleNamespace(name="SiPM"),
            device_id=1,
            connection_info=lambda: "test connection",
        ),
    )


def test_external_tables_fit_default_view_without_horizontal_scrolling(
    qtbot: QtBot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(ExternalDeviceController, "_load_settings", lambda self: None)
    monkeypatch.setattr(ExternalDeviceController, "_on_start_monitor", lambda self: None)
    controller = ExternalDeviceController(_sipm_device())
    qtbot.addWidget(controller)
    controller.resize(900, 600)
    controller.show()
    qtbot.waitExposed(controller)

    assert controller.ui.settingsTable.font().pointSizeF() < controller.font().pointSizeF()
    assert controller.ui.settingsTable.width() == controller.ui.telemetryTable.width()
    for table in (controller.ui.settingsTable, controller.ui.telemetryTable):
        last_column = table.columnCount() - 1
        last_column_right = (
            table.columnViewportPosition(last_column) + table.columnWidth(last_column)
        )
        assert last_column_right <= table.viewport().width()
        assert table.horizontalScrollBar().maximum() == 0
        assert table.width() < 400

    controls = controller.findChild(QScrollArea, "externalControlScroll")
    assert controls is not None
    assert controls.horizontalScrollBar().maximum() == 0


def test_long_external_register_labels_have_full_tooltips(
    qtbot: QtBot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(ExternalDeviceController, "_load_settings", lambda self: None)
    monkeypatch.setattr(ExternalDeviceController, "_on_start_monitor", lambda self: None)
    controller = ExternalDeviceController(_sipm_device())
    qtbot.addWidget(controller)

    row = controller._row_by_name["telemetry:ext_amp_supply_fault"]
    item = controller.ui.telemetryTable.item(row, 0)
    assert item is not None
    assert item.toolTip() == (
        "External Amplifier Supply Fault\nModbus register: ext_amp_supply_fault"
    )
