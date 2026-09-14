from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QSizePolicy,
    QWidget,
)
from pytestqt.qtbot import QtBot

from nlab.controllers.mca_controller import MCAController
from nlab.controllers.scope_controller import DisplayMode, ScopeController
from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.mca import MCA_PARAMETER_SPECS, MCAParam
from nlab.hardware.digitizer.scope import (
    PARAMETER_SPECS,
    RangeSpec,
    Scope,
    ScopeParam,
    TriggerMode,
)
from nlab.ui.ui_mca_view import Ui_MCAView
from nlab.ui.ui_psd_view import Ui_PSDView
from nlab.ui.ui_scope_view import Ui_ScopeView


def _assert_all_spinboxes_and_combos_have_tooltips(widget: QWidget) -> None:
    controls = [*widget.findChildren(QAbstractSpinBox), *widget.findChildren(QComboBox)]
    missing = sorted(control.objectName() for control in controls if not control.toolTip().strip())
    assert not missing, f"controls without tooltips: {missing}"


def test_scope_v121_hardware_ranges() -> None:
    pretrigger = PARAMETER_SPECS[ScopeParam.PRETRIGGER_SAMPLES]
    frame = PARAMETER_SPECS[ScopeParam.FRAME_SAMPLES]
    frame_gap = PARAMETER_SPECS[ScopeParam.FRAME_PERIOD_CYCLES]
    dac = PARAMETER_SPECS[ScopeParam.DAC_VALUE]
    assert isinstance(pretrigger, RangeSpec)
    assert isinstance(frame, RangeSpec)
    assert isinstance(frame_gap, RangeSpec)
    assert isinstance(dac, RangeSpec)

    assert (pretrigger.min_val, pretrigger.max_val, pretrigger.step) == (0, 1020, 4)
    assert (frame.min_val, frame.max_val, frame.step) == (4, 8188, 4)
    assert (frame_gap.min_val, frame_gap.max_val, frame_gap.step) == (0, 65535, 1)
    assert (dac.min_val, dac.max_val, dac.step) == (0, 1023, 1)


def test_scope_widgets_are_driven_from_hardware_specs(qapp: QApplication) -> None:
    widget = QWidget()
    ui = Ui_ScopeView()
    ui.setupUi(widget)
    controller = SimpleNamespace(
        ui=ui,
        _scope=SimpleNamespace(specs=PARAMETER_SPECS),
        _apply_range_to_spinbox=ScopeController._apply_range_to_spinbox,
        _apply_scaled_range_to_spinbox=ScopeController._apply_scaled_range_to_spinbox,
        _apply_range_to_slider=ScopeController._apply_range_to_slider,
        _SAMPLE_PERIOD_NS=ScopeController._SAMPLE_PERIOD_NS,
        _VIEWER_POINT_PERIOD_NS=ScopeController._VIEWER_POINT_PERIOD_NS,
    )

    ScopeController._apply_parameter_specs(controller)

    assert (ui.spinPretrigger.minimum(), ui.spinPretrigger.maximum()) == (0, 2040)
    assert ui.spinPretrigger.singleStep() == 8
    assert ui.spinPretrigger.suffix() == " ns"
    assert (ui.spinFrameSamples.minimum(), ui.spinFrameSamples.maximum()) == (8, 16376)
    assert ui.spinFrameSamples.singleStep() == 8
    assert ui.spinFrameSamples.suffix() == " ns"
    assert (ui.spinFrameGap.minimum(), ui.spinFrameGap.maximum()) == (0, 524280)
    assert ui.spinFrameGap.singleStep() == 8
    assert ui.spinFrameGap.suffix() == " ns"
    assert ui.labelPretrigger.text() == "Pretrigger:"
    assert ui.labelFrameSamples.text() == "Frame:"
    assert ui.labelFrameGap.text() == "Periodic gap:"
    assert (ui.spinDacValue.minimum(), ui.spinDacValue.maximum()) == (0, 1023)
    assert [ui.comboTriggerMode.itemText(i) for i in range(5)] == [
        "Any above",
        "Any below",
        "Falling edge",
        "Rising edge",
        "Periodic",
    ]
    assert ui.labelDacValue.text() == "DAC baseline:"
    assert ui.btnAutoSetup.text() == "Auto Setup"


def _scope_model_for_controller() -> MagicMock:
    scope = MagicMock(spec=Scope)
    scope.specs = PARAMETER_SPECS
    scope.get_trigger_level.return_value = 0
    scope.get_dac_value.return_value = 512
    scope.get_pretrigger_samples.return_value = 32
    scope.get_frame_samples.return_value = 1024
    scope.get_frame_period_cycles.return_value = 0
    scope.get_trigger_mode.return_value = TriggerMode.ANY_BELOW
    scope.frame_period_cycles_supported.return_value = True
    scope.get_dma_enable.return_value = False
    scope.get_viewer_frame_samples_limit.return_value = 2328
    return scope


def test_scope_viewer_scales_time_axis_and_explains_sample_period(qtbot: QtBot) -> None:
    scope = _scope_model_for_controller()
    controller = ScopeController(scope, scope_dma=None, channel=0)
    qtbot.addWidget(controller)

    assert controller._time_scale.unit == "\N{MICRO SIGN}s"
    assert "Time [\N{MICRO SIGN}s]" in controller._time_axis_label.text
    assert "2 ns/ADC sample" in controller._time_axis_label.text
    assert "8 ns/viewer point" in controller._time_axis_label.text
    assert controller.ui.plotWaveform.toolTip() == ""
    assert controller.ui.spinPretrigger.value() == 64
    assert controller.ui.spinFrameSamples.value() == 2048
    assert "1024 ADC samples at 2 ns/sample" in controller.ui.spinFrameSamples.toolTip()
    _assert_all_spinboxes_and_combos_have_tooltips(controller)

    controller._set_display_mode(DisplayMode.RAW)
    controller._on_frame_received(
        [np.array([0, 8, 16]), np.array([10, 20, 30], dtype=np.int16)]
    )
    x_data, _ = controller._raw_curve.getData()
    np.testing.assert_allclose(x_data, [0.0, 0.008, 0.016])

    controller.ui.spinFrameSamples.setValue(512)
    controller._on_frame_samples_changed()

    assert controller._time_scale.unit == "ns"
    assert "Time [ns]" in controller._time_axis_label.text
    assert "Sets the frame length: 512 ns (256 ADC samples at 2 ns/sample)" in (
        controller.ui.spinFrameSamples.toolTip()
    )
    scope.set_frame_samples.assert_called_with(256)
    x_data, _ = controller._raw_curve.getData()
    np.testing.assert_allclose(x_data, [0.0, 8.0, 16.0])


def test_scope_timing_controls_convert_nanoseconds_to_hardware_units(
    qtbot: QtBot,
) -> None:
    scope = _scope_model_for_controller()
    controller = ScopeController(scope, scope_dma=None, channel=0)
    qtbot.addWidget(controller)
    scope.reset_mock()

    controller.ui.spinPretrigger.setValue(80)
    controller.ui.spinPretrigger.editingFinished.emit()
    controller.ui.spinFrameSamples.setValue(512)
    controller._on_frame_samples_changed()
    controller.ui.spinFrameGap.setValue(24)
    controller.ui.spinFrameGap.editingFinished.emit()

    scope.set_pretrigger_samples.assert_called_once_with(40)
    scope.set_frame_samples.assert_called_once_with(256)
    scope.set_frame_period_cycles.assert_called_once_with(3)


def test_scope_status_message_cannot_widen_controls_panel(qtbot: QtBot) -> None:
    controller = ScopeController(_scope_model_for_controller(), scope_dma=None, channel=0)
    qtbot.addWidget(controller)

    status = controller.ui.lblRecordingStatus
    assert status.wordWrap()
    assert status.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored

    status.setText(controller._viewer_limit_message())
    assert "truncated above 4656 ns (2328 ADC samples)" in status.text()


def test_scope_frame_gap_is_enabled_only_for_periodic_trigger(qtbot: QtBot) -> None:
    scope = _scope_model_for_controller()
    controller = ScopeController(scope, scope_dma=None, channel=0)
    qtbot.addWidget(controller)

    assert not controller.ui.spinFrameGap.isHidden()
    assert not controller.ui.spinFrameGap.isEnabled()
    assert not controller.ui.labelFrameGap.isEnabled()

    controller.ui.comboTriggerMode.setCurrentIndex(TriggerMode.TIMED)

    scope.set_trigger_mode.assert_called_with(TriggerMode.TIMED)
    assert controller.ui.spinFrameGap.isEnabled()
    assert controller.ui.labelFrameGap.isEnabled()

    controller.ui.comboTriggerMode.setCurrentIndex(TriggerMode.RISING_EDGE)

    assert not controller.ui.spinFrameGap.isEnabled()
    assert not controller.ui.labelFrameGap.isEnabled()


def test_iio_viewer_limit_is_separate_from_dma_hardware_limit() -> None:
    backend = object.__new__(IIODigitizerBackend)
    scope = Scope(backend)

    backend._scope = SimpleNamespace(attrs={})
    assert scope.get_viewer_frame_samples_limit() == 2328
    backend._scope = SimpleNamespace(attrs={"viewer_data_raw": object()})
    assert scope.get_viewer_frame_samples_limit() is None
    frame_spec = PARAMETER_SPECS[ScopeParam.FRAME_SAMPLES]
    assert isinstance(frame_spec, RangeSpec)
    assert frame_spec.max_val == 8188


def test_scope_allows_truncated_long_viewer_only_start(qtbot: QtBot) -> None:
    scope = _scope_model_for_controller()
    controller = ScopeController(scope, scope_dma=None, channel=0)
    qtbot.addWidget(controller)
    scope.reset_mock()
    controller.ui.spinFrameSamples.setValue(8192)

    controller._on_start()

    scope.start.assert_called_once_with()
    assert controller.ui.btnStart.isChecked()
    assert controller._refresh_timer.isActive()
    assert "Live preview is truncated" in controller.ui.lblRecordingStatus.text()
    assert "Enable Record DMA frames" in controller.ui.lblRecordingStatus.text()
    controller._on_stop()


def test_binary_viewer_allows_long_viewer_only_start(qtbot: QtBot) -> None:
    scope = _scope_model_for_controller()
    scope.get_viewer_frame_samples_limit.return_value = None
    controller = ScopeController(scope, scope_dma=None, channel=0)
    qtbot.addWidget(controller)
    scope.reset_mock()
    controller.ui.spinFrameSamples.setValue(8192)

    controller._on_start()

    scope.start.assert_called_once_with()
    assert controller._refresh_timer.isActive()
    controller._on_stop()


def test_scope_accepts_long_frame_change_during_viewer_readout(qtbot: QtBot) -> None:
    scope = _scope_model_for_controller()
    controller = ScopeController(scope, scope_dma=None, channel=0)
    qtbot.addWidget(controller)
    scope.reset_mock()
    scope.get_frame_samples.return_value = 1024
    controller._refresh_timer.start(1000)
    controller.ui.spinFrameSamples.setValue(8192)

    controller._on_frame_samples_changed()

    assert controller.ui.spinFrameSamples.value() == 8192
    scope.set_frame_samples.assert_called_once_with(4096)
    assert "Live preview is truncated" in controller.ui.lblRecordingStatus.text()
    controller._refresh_timer.stop()


def test_long_dma_capture_keeps_truncated_text_viewer_polling(qtbot: QtBot) -> None:
    scope = _scope_model_for_controller()
    controller = ScopeController(scope, scope_dma=None, channel=0)
    qtbot.addWidget(controller)
    controller._scope_dma = object()  # type: ignore[assignment]
    controller.ui.spinFrameSamples.setValue(8192)

    controller._on_dma_ready()

    assert controller._refresh_timer.isActive()
    assert "live preview is truncated" in controller.ui.lblRecordingStatus.text()
    controller._on_stop()


def test_scope_displays_partial_frame_with_explicit_warning(qtbot: QtBot) -> None:
    scope = _scope_model_for_controller()
    controller = ScopeController(scope, scope_dma=None, channel=0)
    qtbot.addWidget(controller)
    controller.ui.spinFrameSamples.setValue(8192)
    controller._set_display_mode(DisplayMode.RAW)

    controller._on_frame_received(
        [np.array([0, 8, 16]), np.array([10, 20, 30], dtype=np.int16)]
    )

    x_data, y_data = controller._raw_curve.getData()
    np.testing.assert_allclose(x_data, [0.0, 0.008, 0.016])
    np.testing.assert_array_equal(y_data, [10, 20, 30])
    assert "showing 3 of 1024 points" in controller.ui.lblRecordingStatus.text()
    assert "Enable Record DMA frames" in controller.ui.lblRecordingStatus.text()

    controller._persistence_buffer.fill(0)
    controller._rasterize_frame(np.array([10, 20, 30], dtype=np.int16))
    occupied_x = np.nonzero(controller._persistence_buffer)[0]
    assert occupied_x.max() <= 2


def test_mca_widgets_and_enums_match_v101_iio_metadata(qapp: QApplication) -> None:
    widget = QWidget()
    ui = Ui_MCAView()
    ui.setupUi(widget)
    mca = SimpleNamespace(
        edge_det_coeff_is_hardware_backed=lambda: False,
        get_debug_signal_selectors=lambda: tuple(range(9)),
    )
    controller = SimpleNamespace(
        ui=ui,
        _mca=mca,
        _apply_range_to_spinbox=MCAController._apply_range_to_spinbox,
        _apply_range_to_slider=MCAController._apply_range_to_slider,
        _apply_range_to_double_spinbox=MCAController._apply_range_to_double_spinbox,
    )

    MCAController._populate_combos(controller)
    MCAController._apply_parameter_specs(controller)
    MCAController._apply_control_tooltips(controller)

    integer_controls = {
        MCAParam.TRIGGER_LEVEL: (ui.spinTriggerLevel, ui.sliderTriggerLevel),
        MCAParam.PRETRIGGER_SAMPLES: (ui.spinPretrigger, ui.sliderPretrigger),
        MCAParam.FRAME_SAMPLES: (ui.spinFrameSamples, ui.sliderFrameSamples),
        MCAParam.CRRC2_CDELAY: (ui.spinCrrc2Cdelay, ui.sliderCrrc2Cdelay),
        MCAParam.CRRC2_FDELAY: (ui.spinCrrc2Fdelay, ui.sliderCrrc2Fdelay),
        MCAParam.CRRC2_PZC: (ui.spinCrrc2Pzc, ui.sliderCrrc2Pzc),
        MCAParam.CFD_DELAY: (ui.spinCfdDelay, ui.sliderCfdDelay),
        MCAParam.TRAPEZ_R: (ui.spinTrapR, ui.sliderTrapR),
        MCAParam.TRAPEZ_M: (ui.spinTrapM, ui.sliderTrapM),
        MCAParam.TRAPEZ_E: (ui.spinTrapE, ui.sliderTrapE),
        MCAParam.PILEUP_WINDOW: (ui.spinPileupWindow,),
        MCAParam.TIME_LIMIT: (ui.spinTimeLimit,),
        MCAParam.CFD_TW_LOW: (ui.spinCfdTwLow,),
        MCAParam.CFD_TW_HIGH: (ui.spinCfdTwHigh,),
        MCAParam.CC_TIME: (ui.spinCcTime,),
        MCAParam.PSD_ZC_LOW: (ui.spinPsdZcLow,),
        MCAParam.PSD_ZC_HIGH: (ui.spinPsdZcHigh,),
    }
    for parameter, controls in integer_controls.items():
        spec = MCA_PARAMETER_SPECS[parameter]
        assert isinstance(spec, RangeSpec)
        for control in controls:
            assert (control.minimum(), control.maximum(), control.singleStep()) == (
                int(spec.min_val),
                int(spec.max_val),
                int(spec.step),
            )

    double_controls = {
        MCAParam.CFD_FACTOR: ui.spinCfdFactor,
        MCAParam.TRAPEZ_T: ui.spinTrapT,
        MCAParam.EDGE_DET_COEFF: ui.spinEdgeDetCoeff,
    }
    for parameter, control in double_controls.items():
        spec = MCA_PARAMETER_SPECS[parameter]
        assert isinstance(spec, RangeSpec)
        assert control.minimum() == pytest.approx(spec.min_val)
        assert control.maximum() == pytest.approx(spec.max_val)
        assert control.singleStep() == pytest.approx(spec.step)
    assert ui.spinCfdFactor.decimals() == 15

    assert [ui.comboTriggerSource.itemText(i) for i in range(3)] == [
        "Threshold",
        "CR-RC2",
        "CR2-RC2",
    ]
    assert [ui.comboPulsePolarity.itemText(i) for i in range(2)] == ["Negative", "Positive"]
    assert [ui.comboBaseline.itemText(i) for i in range(7)] == [
        "8 ns",
        "16 ns",
        "32 ns",
        "64 ns",
        "128 ns",
        "256 ns",
        "512 ns",
    ]
    assert [ui.comboDebug1.itemText(i) for i in range(9)] == [
        "Input signal",
        "Trigger signal",
        "Trapezoid signal",
        "Trapezoid energy",
        "CFD signal",
        "CFD window",
        "Charge comparison window",
        "PSD ZC window",
        "Logic trigger",
    ]
    assert [ui.comboDebug1.itemData(i) for i in range(9)] == list(range(9))

    for control in (
        ui.spinPretrigger,
        ui.spinFrameSamples,
        ui.spinCrrc2Cdelay,
        ui.spinCrrc2Fdelay,
        ui.spinCfdDelay,
        ui.spinCfdTwLow,
        ui.spinCfdTwHigh,
        ui.spinTrapR,
        ui.spinTrapM,
        ui.spinTrapT,
        ui.spinTrapE,
        ui.spinCcTime,
        ui.spinPsdZcLow,
        ui.spinPsdZcHigh,
    ):
        assert control.suffix() == " ns"
    assert ui.labelTrapT.text() == "Pole-zero time:"
    assert [ui.comboBinning.itemText(i) for i in range(10)] == [
        "1",
        "2",
        "4",
        "8",
        "16",
        "32",
        "64",
        "128",
        "256",
        "512",
    ]
    assert [ui.comboLpPreset.itemText(i) for i in range(3)] == [
        "200 MHz",
        "70 MHz",
        "Moving average",
    ]
    assert [ui.comboTrapFt.itemText(i) for i in range(7)] == [
        "8 ns",
        "16 ns",
        "32 ns",
        "64 ns",
        "128 ns",
        "256 ns",
        "512 ns",
    ]
    assert ui.spinEdgeDetCoeff.isHidden()
    _assert_all_spinboxes_and_combos_have_tooltips(widget)


def test_iio_mca_raw_registers_are_exposed_as_physical_nanoseconds() -> None:
    backend = object.__new__(IIODigitizerBackend)
    values = {
        "crrc2_cdelay": "16",
        "trapezoid_beta_raw": "1975780336",
    }
    backend._pp_attr_get = lambda name: values[name]  # type: ignore[method-assign]
    backend._pp_attr_set = (  # type: ignore[method-assign]
        lambda name, value: values.__setitem__(name, value)
    )

    assert backend.get_crrc2_Cdelay() == 128
    backend.set_crrc2_Cdelay(64)
    assert values["crrc2_cdelay"] == "8"

    assert backend.get_trapez_T() == 96
    backend.set_trapez_T(96)
    assert values["trapezoid_beta_raw"] == "1975780336"
    backend.set_trapez_T(0)
    assert values["trapezoid_beta_raw"] == "0"


def test_iio_mca_reads_debug_selector_capability_from_driver() -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._pp = SimpleNamespace(
        attrs={
            "debug_signal1_available": SimpleNamespace(
                value=(
                    "0 input 1 trigger 2 trapezoid 3 trapezoid-energy "
                    "4 cfd 5 cfd-window 6 cc-window 7 psd-zc-window "
                    "8 logic-trigger"
                )
            )
        }
    )

    assert backend.get_debug_signal_selectors() == tuple(range(9))


def test_old_debug_capability_uses_corrected_hardware_labels(qapp: QApplication) -> None:
    widget = QWidget()
    ui = Ui_MCAView()
    ui.setupUi(widget)
    controller = SimpleNamespace(
        ui=ui,
        _mca=SimpleNamespace(get_debug_signal_selectors=lambda: tuple(range(8))),
    )

    MCAController._populate_combos(controller)

    assert [ui.comboDebug1.itemText(i) for i in range(8)][-2:] == [
        "Charge comparison window",
        "PSD ZC window",
    ]


def test_psd_widgets_reflect_unsigned_16_bit_event_energy(qapp: QApplication) -> None:
    widget = QWidget()
    ui = Ui_PSDView()
    ui.setupUi(widget)

    assert (ui.spinEnergyShift.minimum(), ui.spinEnergyShift.maximum()) == (0, 15)
    assert ui.spinEnergyShift.singleStep() == 1
    assert ui.spinEnergyShift.value() == 0
    assert ui.labelEnergyShift.text() == "Energy shift (bits)"
    assert "display-only" in ui.labelEnergyShift.toolTip()
