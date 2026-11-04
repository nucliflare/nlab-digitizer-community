"""Cross-controller input regressions with simulated devices, never live I/O."""

from unittest.mock import MagicMock

import pytest
from nlab_modbus.core.register_specs import RegisterSpec, RegisterType
from PySide6.QtCore import QLocale, Qt
from PySide6.QtTest import QTest

from nlab.controllers.current_monitor_controller import CurrentMonitorController
from nlab.controllers.external_device_controller import ExternalDeviceController
from nlab.controllers.psu_controller import PSUController
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.scope import PARAMETER_SPECS, Scope, ScopeParam, TriggerMode


@pytest.fixture
def scope_view(qtbot):
    scope = MagicMock(spec=Scope)
    scope.specs = PARAMETER_SPECS
    scope.get_trigger_level.return_value = 0
    scope.get_dac_value.return_value = 512
    scope.get_pretrigger_samples.return_value = 32
    scope.get_frame_samples.return_value = 1024
    scope.get_frame_period_cycles.return_value = 0
    scope.get_trigger_mode.return_value = TriggerMode.ANY_BELOW
    scope.frame_period_cycles_supported.return_value = True
    scope.dma_active.return_value = False
    scope.get_enable.return_value = False
    scope.get_ip_version.return_value = 122
    scope.get_viewer_frame_samples_limit.return_value = 2328
    view = ScopeController(scope, scope_dma=None, channel=0)
    qtbot.addWidget(view)
    scope.reset_mock()
    return view, scope


@pytest.mark.parametrize(
    "control,method,parameter,requested,displayed,raw",
    [
        ("spinPretrigger", "set_pretrigger_samples", ScopeParam.PRETRIGGER_SAMPLES, 409, 408, 204),
        ("spinPretrigger", "set_pretrigger_samples", ScopeParam.PRETRIGGER_SAMPLES, 412, 416, 208),
        ("spinPretrigger", "set_pretrigger_samples", ScopeParam.PRETRIGGER_SAMPLES, -99, 0, 0),
        ("spinFrameSamples", "set_frame_samples", ScopeParam.FRAME_SAMPLES, 409, 408, 204),
        ("spinFrameSamples", "set_frame_samples", ScopeParam.FRAME_SAMPLES, 99999, 16376, 8188),
        ("spinFrameGap", "set_frame_period_cycles", ScopeParam.FRAME_PERIOD_CYCLES, 12, 16, 2),
    ],
)
def test_scope_timing_snaps_in_display_units_before_one_hardware_write(
    scope_view, control, method, parameter, requested, displayed, raw
):
    view, scope = scope_view
    spinbox = getattr(view.ui, control)
    setter = getattr(scope, method)
    setter.side_effect = lambda value: PARAMETER_SPECS[parameter].validate(value)
    spinbox.setValue(requested)
    setter.assert_not_called()
    spinbox.editingFinished.emit()
    assert spinbox.value() == displayed
    setter.assert_called_once_with(raw)


def test_scope_marker_uses_same_halfway_rule_without_writing(scope_view):
    view, scope = scope_view
    view._pretrigger_line.setValue(68 / view._time_scale.ns_per_unit)
    assert view.ui.spinPretrigger.value() == 72
    assert view._pretrigger_line.value() == 72 / view._time_scale.ns_per_unit
    scope.set_pretrigger_samples.assert_not_called()


def test_scope_replay_interlock_still_prevents_writes(scope_view):
    view, scope = scope_view
    view._replay_hardware_interlocked = True
    view.ui.spinPretrigger.setValue(409)
    view.ui.spinPretrigger.editingFinished.emit()
    assert view.ui.spinPretrigger.value() == 408
    scope.set_pretrigger_samples.assert_not_called()


@pytest.mark.parametrize(
    "control,method,text,expected",
    [
        ("spinHvVoltage", "set_hv_voltage", "0", 100.0),
        ("spinHvVoltage", "set_hv_voltage", "9999", 1250.0),
        ("spinSipmVoltage", "set_sipm_voltage", "0", 30.0),
        ("spinSipmVoltage", "set_sipm_voltage", "9999", 70.0),
        ("spinHvCompensCt", "set_hv_compens_ct", "-9", -0.1),
        ("spinHvCompensCt", "set_hv_compens_ct", "0.015", 0.015),
        ("spinSipmCompensCt", "set_sipm_compens_ct", "9", 0.1),
        ("spinHvCompensTref", "set_hv_compens_tref", "-1", 0.0),
        ("spinSipmCompensTref", "set_sipm_compens_tref", "999", 100.0),
    ],
)
def test_psu_numeric_controls_clamp_before_writing(
    qtbot, monkeypatch, control, method, text, expected
):
    hv = MagicMock()
    hv.sipm_available.return_value = True
    hv.get_hv_adc_voltage.return_value = 500.0
    hv.get_sipm_adc_voltage.return_value = 40.0
    monkeypatch.setattr(PSUController, "_load_hardware_state", lambda self: None)
    view = PSUController(hv)
    qtbot.addWidget(view)
    hv.reset_mock()
    spinbox = getattr(view.ui, control)
    spinbox.setLocale(QLocale.c())
    spinbox.lineEdit().setText(text)
    QTest.keyClick(spinbox, Qt.Key.Key_Return)
    assert spinbox.value() == expected
    getattr(hv, method).assert_called_once_with(expected)


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("requested,expected", [(15.0049, 15.0), (15.005, 15.01), (99, 78.0)])
def test_external_registers_snap_to_raw_resolution_before_writing(
    qtbot, replay, requested, expected
):
    spec = RegisterSpec(RegisterType.HOLDING, 3, "int16", 1500, 7800, scale=0.01)
    view = MagicMock()
    view._on_setting_changed = MagicMock()
    editor = ExternalDeviceController._make_editor(view, "vout_set", spec)
    qtbot.addWidget(editor)
    editor.setValue(requested)
    # Exercise the actual write/interlock handler with a mocked transport.
    device = MagicMock()
    view.device = device
    view._replay_hardware_interlocked = replay
    view._on_setting_changed.side_effect = lambda name, value: (
        ExternalDeviceController._on_setting_changed(view, name, value)
    )
    editor.editingFinished.emit()
    assert editor.value() == expected
    if replay:
        device.write.assert_not_called()
    else:
        device.write.assert_called_once_with("vout_set", expected)


@pytest.mark.parametrize(
    "requested,expected,cycles", [(0.004, 0.008, 1), (0.012, 0.016, 2), (999, 524.28, 65535)]
)
def test_current_dma_gap_uses_same_nearest_step(qtbot, requested, expected, cycles):
    view = CurrentMonitorController(MagicMock(), channel=0)
    qtbot.addWidget(view)
    view.ui.spinDmaGap.setValue(requested)
    view.ui.spinDmaGap.editingFinished.emit()
    assert view.ui.spinDmaGap.value() == expected
    assert view.configuration_settings()["dma_gap_cycles"] == cycles
