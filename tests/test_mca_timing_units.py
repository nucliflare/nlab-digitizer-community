"""GUI nanoseconds versus native ABI2 samples, using only simulated hardware."""

from unittest.mock import Mock

import numpy as np
import pytest

from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.offline import create_offline_digitizer


@pytest.fixture
def panel(qtbot):
    device = create_offline_digitizer(0)
    controller = MCAController(device.mca)
    qtbot.addWidget(controller)
    yield controller, device.mca
    device.close()


@pytest.mark.parametrize("control,path,display_ns,native,next_native", [
    ("spinPretrigger", "set_pretrigger_samples", 80, 40, 44),
    ("spinFrameSamples", "set_frame_samples", 512, 256, 260),
    ("spinCrrc2Fdelay", "filters.crrc2.set_Fdelay", 64, 32, 36),
    ("spinCfdDelay", "filters.cfd.set_delay", 26, 13, 14),
    ("spinCfdTwLow", "filters.cfd.set_time_window_low", 24, 12, 16),
    ("spinCfdTwHigh", "filters.cfd.set_time_window_high", 80, 40, 44),
    ("spinTrapR", "filters.trapezoid.set_R", 72, 36, 40),
    ("spinTrapM", "filters.trapezoid.set_M", 40, 20, 24),
    ("spinTrapE", "filters.trapezoid.set_E", 136, 68, 72),
    ("spinCcTime", "filters.charge_comparison.set_time", 48, 24, 28),
    ("spinPsdZcLow", "filters.psd_zc.set_time_window_low", 16, 8, 12),
    ("spinPsdZcHigh", "filters.psd_zc.set_time_window_high", 80, 40, 44),
])
def test_ns_edit_writes_native_samples_and_readback_restores_ns(
    panel, monkeypatch, control, path, display_ns, native, next_native
):
    controller, mca = panel
    owner = mca
    parts = path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    original = getattr(owner, parts[-1])
    getter = getattr(owner, parts[-1].replace("set_", "get_", 1))
    write = Mock(wraps=original)
    monkeypatch.setattr(owner, parts[-1], write)
    widget = getattr(controller.ui, control)

    assert widget.suffix() == " ns"
    widget.setValue(display_ns)
    write.assert_not_called()
    widget.editingFinished.emit()
    write.assert_called_once_with(native)
    assert getter() == native

    original(next_native)
    write.reset_mock()
    controller._load_hardware_state()
    assert widget.value() == next_native * 2
    write.assert_not_called()


def test_ns_edit_keeps_calibration_fingerprint_in_native_units(panel):
    controller, mca = panel
    assert controller.ui.spinTrapR.value() == 64
    assert controller.energy_calibration_fingerprint()["trapezoid_r_samples"] == 32
    controller.ui.spinTrapR.setValue(72)
    controller.ui.spinTrapR.editingFinished.emit()
    fingerprint = controller.energy_calibration_fingerprint()
    assert mca.filters.trapezoid.get_R() == 36
    assert fingerprint["parameter_units"] == "adc_samples"
    assert fingerprint["trapezoid_r_samples"] == 36
    assert fingerprint["trapezoid_m_samples"] == mca.filters.trapezoid.get_M()
    assert fingerprint["trapezoid_start_offset_samples"] == mca.filters.trapezoid.get_E()


def test_debug_trace_spans_ns_window_at_native_debug_spacing(panel):
    controller, _ = panel
    controller.ui.spinFrameSamples.setValue(512)
    waveform = np.arange(128, dtype=np.int16)
    controller._update_debug_plot(waveform, waveform)
    for curve in (controller._debug1_curve, controller._debug2_curve):
        time, amplitude = curve.getData()
        assert len(amplitude) == 64
        np.testing.assert_array_equal(time, np.arange(0, 512, 8))
        np.testing.assert_array_equal(amplitude, waveform[:64])
