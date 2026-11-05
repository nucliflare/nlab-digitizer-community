"""Firmware unit integration with simulated Modbus I/O and offscreen Qt."""

from unittest.mock import Mock

import pytest
from nlab_modbus.devices.geiger import GeigerDevice
from nlab_modbus.devices.psu import PSUDevice
from nlab_modbus.devices.sipm import SiPMDevice
from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDoubleSpinBox, QStyleOptionViewItem

from nlab.controllers.external_device_controller import ExternalDeviceController
from nlab.hardware.digitizer.offline import OfflineExternalDevice


@pytest.fixture(params=[QLocale.Language.Polish, QLocale.Language.German])
def comma_decimal_locale(request, qapp):
    previous = QLocale()
    locale = QLocale(request.param)
    QLocale.setDefault(locale)
    try:
        yield locale
    finally:
        QLocale.setDefault(previous)


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


@pytest.mark.parametrize(
    "recorded_unit,recorded_value,recorded_scale",
    [("A", 0.00012, 1e-6), ("mA", 0.12, 1e-3)],
)
def test_older_recording_keeps_native_values_while_plotting_microamps(
    qtbot, recorded_unit, recorded_value, recorded_scale
):
    schema = {
        "registers": [
            {
                "name": "sipm_current_ua",
                "register_type": "input",
                "address": 4,
                "dtype": "int16",
                "min": -32768,
                "max": 32767,
                "scale": recorded_scale,
                "unit": recorded_unit,
            }
        ]
    }
    device = OfflineExternalDevice("SIPM:1", {}, {"sipm_current_ua": recorded_value}, schema=schema)
    view = ExternalDeviceController(device, auto_start=False)
    qtbot.addWidget(view)
    row = view._row_by_name["telemetry:sipm_current_ua"]
    view.ui.telemetryTable.cellWidget(row, 3).setChecked(True)
    view.begin_session_replay()
    view.display_session_replay_readback(0.5, {"sipm_current_ua": recorded_value})
    assert view.ui.telemetryTable.item(row, 2).text() == recorded_unit
    assert list(view._plot_values["sipm_current_ua"]) == [recorded_value]
    document = view.replay_schema()["registers"][0]
    assert document["scale"] == recorded_scale
    assert document["unit"] == recorded_unit
    assert view._plot_curves["sipm_current_ua"].getData()[1] == pytest.approx([120])
    assert view.ui.plotWidget.getAxis("left").labelUnits == "uA"


@pytest.mark.parametrize("device_class", [SiPMDevice, GeigerDevice, PSUDevice])
def test_external_floats_use_dots_with_comma_system_locales(
    qtbot, comma_decimal_locale, device_class
):
    view, _device, _client = make_view(qtbot, device_class)
    editors = view.findChildren(QDoubleSpinBox)
    assert view.locale().decimalPoint() == "."
    if device_class is SiPMDevice:
        assert editors
    for editor in editors:
        editor.setValue(editor.minimum() + editor.singleStep())
        assert editor.locale().decimalPoint() == "."
        assert "." in editor.cleanText()
        assert "," not in editor.cleanText()
    view._on_readback(0.5, {"supply_voltage": 5.25})
    row = view._row_by_name["telemetry:supply_voltage"]
    assert view.ui.telemetryTable.item(row, 1).text() == "5.2500"
    for side in ("bottom", "left"):
        ticks = view.ui.plotWidget.getAxis(side).tickStrings([0, 0.5, 1], 1, 0.5)
        assert any("." in text for text in ticks)
        assert all("," not in text for text in ticks)
    # This policy is local to External, without changing other workspaces' locales.
    unrelated = QDoubleSpinBox()
    qtbot.addWidget(unrelated)
    assert unrelated.locale().decimalPoint() == comma_decimal_locale.decimalPoint() == ","


def test_sipm_dot_input_commits_the_correct_native_count(qtbot, comma_decimal_locale):
    view, _device, client = make_view(qtbot, SiPMDevice, holding_words={3: 4567})
    row = view._row_by_name["settings:vout_set"]
    editor = view.ui.settingsTable.cellWidget(row, 1)
    assert editor.cleanText() == "45.6700"
    editor.lineEdit().setText("46.25")
    QTest.keyClick(editor, Qt.Key.Key_Return)
    assert editor.value() == 46.25
    client.write_register.assert_called_once_with(address=3, value=4625, device_id=1)
    client.write_register.reset_mock()
    # A pasted comma must not become a thousands separator and clamp to 78 V.
    editor.lineEdit().setText("46,250")
    QTest.keyClick(editor, Qt.Key.Key_Return)
    assert editor.value() == 46.25
    assert all(call.kwargs["value"] == 4625 for call in client.write_register.call_args_list)


@pytest.mark.parametrize(
    "raw,expected", [(0, 0), (1, 1), (120, 120), (10000, 10000), (0xFFF1, -15)]
)
def test_sipm_current_plot_retains_microamps_through_ten_milliamps(qtbot, raw, expected):
    view, device, _client = make_view(qtbot, SiPMDevice, input_words={4: raw})
    name = "sipm_current_ua"
    row = view._row_by_name[f"telemetry:{name}"]
    view.ui.telemetryTable.cellWidget(row, 3).setChecked(True)
    view._on_readback(0.5, device.read_snapshot())
    curve = view._plot_curves[name]
    assert list(curve.getData()[1]) == [expected]
    assert curve.name() == "SiPM Output Current (uA)"
    assert float(view.ui.telemetryTable.item(row, 1).text()) == expected
    assert view.ui.telemetryTable.item(row, 2).text() == "uA"
    axis = view.ui.plotWidget.getAxis("left")
    assert axis.labelText == "SiPM Output Current"
    assert axis.labelUnits == "uA"
    assert not axis.autoSIPrefix


def test_external_plot_updates_axis_units_and_labels_mixed_traces(qtbot):
    view, _device, _client = make_view(qtbot, SiPMDevice)

    def select(name, enabled):
        row = view._row_by_name[f"telemetry:{name}"]
        view.ui.telemetryTable.cellWidget(row, 3).setChecked(enabled)

    axis = view.ui.plotWidget.getAxis("left")
    select("sipm_voltage_10mv", True)
    select("supply_voltage", True)
    assert axis.labelUnits == "V"
    select("sipm_current_ua", True)
    assert axis.labelText == "Value (units in legend)"
    assert axis.labelUnits == ""
    assert view._plot_curves["sipm_voltage_10mv"].name() == "SiPM Output Voltage (V)"
    assert view._plot_curves["sipm_current_ua"].name() == "SiPM Output Current (uA)"
    select("sipm_current_ua", False)
    assert axis.labelUnits == "V"
    select("sipm_voltage_10mv", False)
    select("supply_voltage", False)
    assert axis.labelText == "Value"
    assert axis.labelUnits == ""
