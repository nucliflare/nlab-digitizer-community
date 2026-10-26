from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from pytestqt.qtbot import QtBot

from nlab.controllers.psu_controller import PSUController
from nlab.hardware.digitizer.backends.base import IDSBackend
from nlab.hardware.digitizer.hv import HVSupply


@pytest.fixture
def psu(qtbot: QtBot) -> tuple[PSUController, MagicMock]:
    backend = MagicMock(spec=IDSBackend)
    backend.get_sipm_overload.return_value = -1
    backend.get_hv_adc_voltage.return_value = 500.0
    backend.get_hv_compens_output.return_value = 500.0
    backend.get_temp_analog.return_value = 20.0
    backend.get_temp_digital.return_value = 20.0
    backend.get_temp_digital_status.return_value = 1
    backend.get_ads_temp.return_value = 25.0
    controller = PSUController(HVSupply(backend))
    qtbot.addWidget(controller)
    controller.show()
    backend.reset_mock()
    return controller, backend


@pytest.mark.parametrize("finish", ["enter", "focus_out"])
@pytest.mark.parametrize(
    ("text", "expected"),
    [("0", 100.0), ("99", 100.0), ("100", 100.0), ("1250", 1250.0)],
)
def test_hv_voltage_edit_commits_a_supported_value(
    psu: tuple[PSUController, MagicMock],
    text: str,
    expected: float,
    finish: str,
) -> None:
    controller, backend = psu
    spinbox = controller.ui.spinHvVoltage
    QTest.mouseClick(spinbox, Qt.MouseButton.LeftButton)
    spinbox.selectAll()
    QTest.keyClicks(spinbox, text)

    if finish == "enter":
        QTest.keyClick(spinbox, Qt.Key.Key_Return)
    else:
        QTest.mouseClick(controller.ui.spinHvCompensCt, Qt.MouseButton.LeftButton)

    assert spinbox.value() == expected
    assert controller.hardware_configuration_settings()["hv_voltage"] == expected
    backend.set_hv_voltage.assert_called_once_with(expected)
    # ADC feedback is independent of the corrected request.
    assert controller.ui.lblHvVoltage.text() == "500.0"


def test_hv_voltage_cannot_step_below_minimum(
    psu: tuple[PSUController, MagicMock],
) -> None:
    controller, backend = psu
    spinbox = controller.ui.spinHvVoltage
    spinbox.setValue(100.0)

    QTest.keyClick(spinbox, Qt.Key.Key_Down)
    QTest.keyClick(spinbox, Qt.Key.Key_Return)

    assert spinbox.value() == 100.0
    backend.set_hv_voltage.assert_called_once_with(100.0)


@pytest.mark.parametrize("voltage", [0.0, 99.9, 1250.1])
def test_hv_api_rejects_voltage_outside_operating_range(voltage: float) -> None:
    backend = MagicMock(spec=IDSBackend)

    with pytest.raises(ValueError, match="hv_voltage: .* out of range"):
        HVSupply(backend).set_hv_voltage(voltage)

    backend.set_hv_voltage.assert_not_called()
