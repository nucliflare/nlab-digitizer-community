"""Controller input normalization using simulated hardware, with no live writes."""

from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox
from pytestqt.qtbot import QtBot

from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.mca import MCA_PARAMETER_SPECS, MCAParam
from nlab.hardware.digitizer.scope import RangeSpec


@pytest.fixture
def controller(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch):
    mca = MagicMock()
    mca.get_debug_signal_selectors.return_value = tuple(range(9))
    mca.edge_det_coeff_is_hardware_backed.return_value = False
    monkeypatch.setattr(MCAController, "_load_hardware_state", lambda self: None)
    view = MCAController(mca, mca_dma=None, channel=0)
    qtbot.addWidget(view)
    mca.reset_mock()
    yield view, mca
    view._worker = None
    view._dma_worker = None


@pytest.mark.parametrize("slider_release", [False, True])
@pytest.mark.parametrize(
    "spin_name,slider_name,method,parameter,requested,expected",
    [
        (
            "spinFrameSamples",
            "sliderFrameSamples",
            "set_frame_samples",
            MCAParam.FRAME_SAMPLES,
            409,
            408,
        ),
        (
            "spinPretrigger",
            "sliderPretrigger",
            "set_pretrigger_samples",
            MCAParam.PRETRIGGER_SAMPLES,
            409,
            408,
        ),
        (
            "spinCrrc2Cdelay",
            "sliderCrrc2Cdelay",
            "filters.crrc2.set_Cdelay",
            MCAParam.CRRC2_CDELAY,
            2,
            2,
        ),
        (
            "spinCrrc2Cdelay",
            "sliderCrrc2Cdelay",
            "filters.crrc2.set_Cdelay",
            MCAParam.CRRC2_CDELAY,
            7,
            6,
        ),
        ("spinTrapR", "sliderTrapR", "filters.trapezoid.set_R", MCAParam.TRAPEZ_R, 19, 20),
        ("spinTrapR", "sliderTrapR", "filters.trapezoid.set_R", MCAParam.TRAPEZ_R, 20, 20),
        (
            "spinTriggerLevel",
            "sliderTriggerLevel",
            "set_trigger_level",
            MCAParam.TRIGGER_LEVEL,
            -409,
            -409,
        ),
        (
            "spinFrameSamples",
            "sliderFrameSamples",
            "set_frame_samples",
            MCAParam.FRAME_SAMPLES,
            -20,
            4,
        ),
        (
            "spinFrameSamples",
            "sliderFrameSamples",
            "set_frame_samples",
            MCAParam.FRAME_SAMPLES,
            99999,
            4096,
        ),
    ],
)
def test_setting_commit_snaps_and_synchronizes_once(
    controller, slider_release, spin_name, slider_name, method, parameter, requested, expected
):
    view, mca = controller
    spinbox, slider = getattr(view.ui, spin_name), getattr(view.ui, slider_name)
    setter = mca
    for name in method.split("."):
        setter = getattr(setter, name)
    setter.side_effect = lambda value: MCA_PARAMETER_SPECS[parameter].validate(value)
    (slider if slider_release else spinbox).setValue(requested)
    setter.assert_not_called()
    (slider.sliderReleased if slider_release else spinbox.editingFinished).emit()
    setter.assert_called_once_with(expected)
    assert spinbox.value() == slider.value() == expected


@pytest.mark.parametrize(
    "spin_name,method,parameter,requested,expected",
    [
        ("spinCcTime", "filters.charge_comparison.set_time", MCAParam.CC_TIME, 409, 408),
        ("spinCfdTwLow", "filters.cfd.set_time_window_low", MCAParam.CFD_TW_LOW, 409, 408),
        ("spinPsdZcHigh", "filters.psd_zc.set_time_window_high", MCAParam.PSD_ZC_HIGH, 409, 408),
        ("spinTrapT", "filters.trapezoid.set_T", MCAParam.TRAPEZ_T, 67, 67),
        ("spinTrapT", "filters.trapezoid.set_T", MCAParam.TRAPEZ_T, 68, 68),
    ],
)
def test_standalone_and_double_spinbox_timings_snap_before_writing(
    controller, spin_name, method, parameter, requested, expected
):
    view, mca = controller
    setter = mca
    for name in method.split("."):
        setter = getattr(setter, name)
    setter.side_effect = lambda value: MCA_PARAMETER_SPECS[parameter].validate(value)
    spinbox = getattr(view.ui, spin_name)
    spinbox.setValue(requested)
    setter.assert_not_called()
    spinbox.editingFinished.emit()
    setter.assert_called_once_with(expected)
    assert spinbox.value() == expected


def test_running_setting_snaps_before_one_stop_write_restart(controller):
    view, mca = controller
    view._worker = object()
    mca.reconfigure_while_running.side_effect = lambda write: (write(), True)[1]
    mca.set_frame_samples.side_effect = lambda value: MCA_PARAMETER_SPECS[
        MCAParam.FRAME_SAMPLES
    ].validate(value)
    view.ui.spinFrameSamples.setValue(409)
    view.ui.spinFrameSamples.editingFinished.emit()
    mca.reconfigure_while_running.assert_called_once()
    mca.set_frame_samples.assert_called_once_with(408)
    assert view.ui.sliderFrameSamples.value() == 408


@pytest.mark.parametrize("interlock", ["replay", "dma"])
def test_snapping_preserves_replay_and_dma_write_interlocks(controller, interlock):
    view, mca = controller
    if interlock == "replay":
        view._replay_hardware_interlocked = True
    else:
        view._dma_worker = object()
    view.ui.spinFrameSamples.setValue(409)
    view.ui.spinFrameSamples.editingFinished.emit()
    assert view.ui.spinFrameSamples.value() == 408
    mca.set_frame_samples.assert_not_called()
    mca.reconfigure_while_running.assert_not_called()


def test_fractional_coefficient_keeps_existing_numeric_mapping(controller):
    view, mca = controller
    spinbox = view.ui.spinCfdFactor
    spinbox.setValue(0.41234)
    displayed = spinbox.value()
    spinbox.editingFinished.emit()
    assert spinbox.value() == displayed
    mca.filters.cfd.set_factor.assert_called_once_with(displayed)


@pytest.mark.parametrize("widget_class", [QSpinBox, QDoubleSpinBox])
@pytest.mark.parametrize("requested,expected", [(-10, 3), (5, 7), (8, 7), (12, 11), (99, 11)])
def test_snapping_uses_minimum_and_last_valid_grid_point(qtbot, widget_class, requested, expected):
    spinbox = widget_class()
    qtbot.addWidget(spinbox)
    spinbox.setRange(3, 12)
    spinbox.setSingleStep(4)
    MCAController._snap_integer_control(spinbox, requested)
    assert spinbox.value() == expected


@pytest.mark.parametrize("position,expected", [(27.9, 12), (28.0, 16), (28.1, 16)])
def test_pretrigger_marker_uses_same_nearest_step_without_double_rounding(
    controller, position, expected
):
    view, mca = controller
    view._debug_pretrigger_line.setValue(position)
    assert view.ui.spinPretrigger.value() == expected
    assert view.ui.sliderPretrigger.value() == expected
    assert view._debug_pretrigger_line.value() == expected * 2
    mca.set_pretrigger_samples.assert_not_called()


def test_hardware_api_still_rejects_misaligned_input():
    spec = MCA_PARAMETER_SPECS[MCAParam.FRAME_SAMPLES]
    assert isinstance(spec, RangeSpec)
    with pytest.raises(ValueError, match="not aligned"):
        spec.validate(409, "frame_samples")
