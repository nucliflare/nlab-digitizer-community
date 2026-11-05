"""Firmware unit integration with simulated Modbus I/O and offscreen Qt."""

from unittest.mock import Mock

import pytest
from nlab_modbus.devices.geiger import GeigerDevice
from nlab_modbus.devices.psu import PSUDevice
from nlab_modbus.devices.sipm import SiPMDevice
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QStyleOptionViewItem

from nlab.controllers.external_device_controller import ExternalDeviceController
from nlab.hardware.digitizer.offline import OfflineExternalDevice


def make_view(qtbot, device_class, input_words=None, holding_words=None):
    client = Mock()

    def response(words, *, address, count, device_id):
        return Mock(
            registers=[words.get(i, 0) for i in range(address, address + count)],
            isError=Mock(return_value=False),
        )

    client.read_input_registers.side_effect = lambda **kwargs: response(input_words or {}, **kwargs)
    client.read_holding_registers.side_effect = lambda **kwargs: response(
        holding_words or {}, **kwargs
    )
    client.write_register.return_value.isError.return_value = False
    device = device_class(client, 1)
    view = ExternalDeviceController(device, auto_start=False)
    qtbot.addWidget(view)
    client.write_register.assert_not_called()
    client.connect.assert_not_called()
    return view, device, client


@pytest.mark.parametrize(
    "device_class,input_words,register,expected,unit,scale",
    [
        (SiPMDevice, {4: 120}, "sipm_current_ua", 120, "uA", 1.0),
        (PSUDevice, {15: 2500}, "pmt_temp", 25, "degC", 0.01),
        (GeigerDevice, {4: 17}, "dose_level_msvh", 17, "mSv/h", 1.0),
    ],
)
def test_telemetry_plots_and_recording_metadata_use_firmware_units_once(
    qtbot, device_class, input_words, register, expected, unit, scale
):
    view, device, _client = make_view(qtbot, device_class, input_words)
    row = view._row_by_name[f"telemetry:{register}"]
    assert view.ui.telemetryTable.item(row, 2).text() == unit
    view.ui.telemetryTable.cellWidget(row, 3).setChecked(True)
    recorded = []
    view.session_readback.connect(lambda _id, _time, values: recorded.append(values))
    view._on_readback(0.5, device.read_snapshot())
    assert float(view.ui.telemetryTable.item(row, 1).text()) == expected
    assert list(view._plot_values[register]) == [expected]
    assert recorded[0][register] == expected
    document = next(r for r in view.replay_schema()["registers"] if r["name"] == register)
    assert document["unit"] == unit
    assert document["scale"] == scale


def test_sipm_settings_keep_voltage_fractions_and_encode_native_counts(qtbot):
    view, _device, client = make_view(qtbot, SiPMDevice, holding_words={3: 4567, 7: 2025, 9: 12})
    expected = {
        "vout_set": (45.67, "V"),
        "sipm_comp_tref": (20.25, "degC"),
        "sipm_comp_ct": (0, "mV/degC"),
    }
    for name, (value, unit) in expected.items():
        row = view._row_by_name[f"settings:{name}"]
        assert view.ui.settingsTable.item(row, 2).text() == unit
        assert view.ui.settingsTable.cellWidget(row, 1).value() == pytest.approx(value)
        option = QStyleOptionViewItem()
        option.initFrom(view.ui.settingsTable)
        option.font = view.ui.settingsTable.font()
        option.fontMetrics = QFontMetrics(option.font)
        index = view.ui.settingsTable.model().index(row, 2)
        needed = view.ui.settingsTable.itemDelegate().sizeHint(option, index).width()
        assert view.ui.settingsTable.columnWidth(2) >= needed
    row = view._row_by_name["settings:vout_set"]
    editor = view.ui.settingsTable.cellWidget(row, 1)
    assert (editor.minimum(), editor.maximum(), editor.singleStep()) == (15, 78, 0.01)
    editor.setValue(46.25)
    editor.editingFinished.emit()
    client.write_register.assert_called_once_with(address=3, value=4625, device_id=1)


def test_older_recording_keeps_its_recorded_ampere_values_and_units(qtbot):
    schema = {
        "registers": [
            {
                "name": "sipm_current_ua",
                "register_type": "input",
                "address": 4,
                "dtype": "int16",
                "min": -32768,
                "max": 32767,
                "scale": 1e-6,
                "unit": "A",
            }
        ]
    }
    device = OfflineExternalDevice("SIPM:1", {}, {"sipm_current_ua": 0.00012}, schema=schema)
    view = ExternalDeviceController(device, auto_start=False)
    qtbot.addWidget(view)
    row = view._row_by_name["telemetry:sipm_current_ua"]
    view.ui.telemetryTable.cellWidget(row, 3).setChecked(True)
    view.begin_session_replay()
    view.display_session_replay_readback(0.5, {"sipm_current_ua": 0.00012})
    assert view.ui.telemetryTable.item(row, 2).text() == "A"
    assert list(view._plot_values["sipm_current_ua"]) == [0.00012]
    assert view.replay_schema()["registers"][0]["scale"] == 1e-6
