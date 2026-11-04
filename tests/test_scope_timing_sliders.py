"""Scope timing controls tested offscreen against simulated hardware only."""

from unittest.mock import Mock

import pytest

from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.current_monitor import ScopeCurrentAccumulator
from nlab.hardware.digitizer.dma import ScopeDmaGeometry
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.hardware.digitizer.scope import TriggerMode


@pytest.fixture
def scope(qtbot):
    device = create_offline_digitizer(0)
    view = ScopeController(device.scope)
    qtbot.addWidget(view)
    try:
        yield view
    finally:
        view._refresh_timer.stop()
        view._measurement_timer.stop()
        view.stop_dma_sync()
        device.close()


@pytest.mark.parametrize("name", ["Pretrigger", "FrameSamples", "FrameGap"])
def test_timing_sliders_match_spinbox_ranges_units_and_initial_values(scope, name):
    spinbox, slider = getattr(scope.ui, f"spin{name}"), getattr(scope.ui, f"slider{name}")
    assert (slider.minimum(), slider.maximum(), slider.singleStep()) == (
        spinbox.minimum(),
        spinbox.maximum(),
        spinbox.singleStep(),
    )
    assert slider.value() == spinbox.value()
    assert slider.toolTip() == spinbox.toolTip()


@pytest.mark.parametrize("source", ["slider", "spinbox"])
@pytest.mark.parametrize(
    "name,method,requested,displayed,raw",
    [
        ("Pretrigger", "set_pretrigger_samples", 409, 408, 204),
        ("Pretrigger", "set_pretrigger_samples", 412, 416, 208),
        ("FrameSamples", "set_frame_samples", 409, 408, 204),
        ("FrameSamples", "set_frame_samples", 99999, 16376, 8188),
        ("FrameGap", "set_frame_period_cycles", 12, 16, 2),
    ],
)
def test_timing_changes_snap_and_commit_once_in_native_hardware_units(
    scope, monkeypatch, source, name, method, requested, displayed, raw
):
    spinbox, slider = getattr(scope.ui, f"spin{name}"), getattr(scope.ui, f"slider{name}")
    setter = Mock(wraps=getattr(scope._scope, method))
    monkeypatch.setattr(scope._scope, method, setter)
    if source == "slider":
        slider.setSliderDown(True)
        slider.setValue(requested)
    else:
        spinbox.setValue(requested)
    assert spinbox.value() == slider.value()
    setter.assert_not_called()
    if source == "slider":
        slider.setSliderDown(False)
    else:
        spinbox.editingFinished.emit()
    assert spinbox.value() == slider.value() == displayed
    setter.assert_called_once_with(raw)


def test_acquisition_time_spinbox_preserves_loaded_configuration(scope):
    scope.ui.spinTime.setValue(30)
    assert scope.configuration_settings()["measurement_time_s"] == 30
    scope.apply_configuration_settings({"measurement_time_s": 0})
    assert scope.ui.spinTime.value() == 0


def test_marker_hardware_readback_and_default_reset_update_sliders(scope):
    scope._pretrigger_line.setValue(68 / scope._time_scale.ns_per_unit)
    assert scope.ui.sliderPretrigger.value() == scope.ui.spinPretrigger.value() == 72
    scope._scope.set_pretrigger_samples(128)
    scope._scope.set_frame_samples(512)
    scope._scope.set_frame_period_cycles(12)
    scope._load_hardware_state()
    assert scope.ui.sliderPretrigger.value() == 256
    assert scope.ui.sliderFrameSamples.value() == 1024
    assert scope.ui.sliderFrameGap.value() == 96
    scope.reset_defaults()
    assert scope.ui.sliderPretrigger.value() == 64
    assert scope.ui.sliderFrameSamples.value() == 2048
    assert scope.ui.sliderFrameGap.value() == 0


@pytest.mark.parametrize("supported", [False, True])
def test_periodic_gap_slider_follows_trigger_mode_and_capability(scope, monkeypatch, supported):
    monkeypatch.setattr(scope._scope, "frame_period_cycles_supported", lambda: supported)
    scope.ui.comboTriggerMode.setCurrentIndex(TriggerMode.TIMED)
    assert scope.ui.sliderFrameGap.isEnabled() == supported
    assert scope.ui.spinFrameGap.isEnabled() == supported
    scope.ui.comboTriggerMode.setCurrentIndex(TriggerMode.RISING_EDGE)
    assert not scope.ui.sliderFrameGap.isEnabled()


@pytest.mark.parametrize("replay", [False, True])
def test_timing_sliders_follow_existing_acquisition_and_replay_interlocks(scope, replay):
    if replay:
        scope.set_replay_hardware_interlock(True)
    else:
        scope._set_controls_enabled(False)
    for name in ("Pretrigger", "FrameSamples", "FrameGap"):
        assert not getattr(scope.ui, f"slider{name}").isEnabled()


def test_current_dma_preset_updates_sliders_without_duplicate_writes(scope, monkeypatch):
    monkeypatch.setattr(scope._scope, "get_ip_version", lambda: 122)
    monkeypatch.setattr(scope._scope, "frame_period_cycles_supported", lambda: True)
    setters = {}
    for method in ("set_pretrigger_samples", "set_frame_samples", "set_frame_period_cycles"):
        setters[method] = Mock(wraps=getattr(scope._scope, method))
        monkeypatch.setattr(scope._scope, method, setters[method])
    scope._scope_dma = Mock()
    scope._scope_dma.capture_geometry.return_value = ScopeDmaGeometry.legacy(8188)
    scope._current_accumulator = ScopeCurrentAccumulator()
    monkeypatch.setattr(scope, "_on_start", lambda: setattr(scope, "_dma_worker", Mock()))
    try:
        scope.start_current_dma_monitor(gap_cycles=10_453)
        assert scope.ui.sliderPretrigger.value() == scope.ui.spinPretrigger.value() == 0
        assert scope.ui.sliderFrameSamples.value() == scope.ui.spinFrameSamples.value() == 16376
        assert scope.ui.sliderFrameGap.value() == scope.ui.spinFrameGap.value() == 83624
        setters["set_pretrigger_samples"].assert_called_once_with(0)
        setters["set_frame_samples"].assert_called_once_with(8188)
        setters["set_frame_period_cycles"].assert_called_once_with(10_453)
    finally:
        scope._dma_worker = None
        scope._dma_online_current = False
        scope._scope_dma = None
        scope._current_accumulator.finish_session()
