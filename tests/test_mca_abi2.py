"""ABI2 admission and native units, checked against the documented contract."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.mca import McaGateTiming, MultiChannelAnalyzer
from nlab.hardware.digitizer.mca_parameters import (
    DEFAULT_PARAMETERS,
    validate_document,
    validate_parameters,
)
from nlab.hardware.digitizer.offline import create_offline_digitizer


def backend(version="2", units="adc_samples", ack="0"):
    result = object.__new__(IIODigitizerBackend)
    values = DEFAULT_PARAMETERS | {
        "parameter_abi_version": version,
        "parameter_units": units,
        "parameter_abi_ack": ack,
        "enable": "0",
        "list_buffer_active": "0",
        "debug_signal1": "0",
        "debug_signal2": "1",
        "measurement_in_progress": "1",
        "cfd_factor_scale": str(1 / 32768),
    }
    result._pp = SimpleNamespace(
        attrs={name: SimpleNamespace(value=str(value)) for name, value in values.items()}
    )
    result._input_filter = object()
    result._lm_frame = None
    result.mca_dma_active = lambda: False
    return result


@pytest.mark.parametrize("version,units", [("1", "ns"), ("3", "adc_samples"), ("2", "ns")])
def test_unknown_contract_refuses_before_configuration(version, units):
    b = backend(version, units)
    assert not b.mca_hardware_present()
    with pytest.raises(RuntimeError, match="ABI2"):
        b.prepare_mca_configuration(dict(DEFAULT_PARAMETERS))
    assert b._pp.attrs["parameter_abi_ack"].value == "0"
    assert b._pp.attrs["frame_samples"].value == "128"
    b.set_global_enable(False)  # STOP and diagnostic muxes remain available.
    b.set_mem1_sig_select(2)
    assert b._pp.attrs["debug_signal1"].value == "2"


def test_ack_requires_idle_ownership_not_progress_status():
    b = backend()
    b.prepare_mca_configuration(dict(DEFAULT_PARAMETERS))
    assert b._pp.attrs["parameter_abi_ack"].value == "2"
    assert b._pp.attrs["measurement_in_progress"].value == "1"
    b._pp.attrs["enable"].value = "1"
    with pytest.raises(RuntimeError, match="Stop"):
        b.prepare_mca_configuration(dict(DEFAULT_PARAMETERS))


def test_validate_before_ack_and_native_encoding_owned_by_kernel():
    b = backend()
    invalid = DEFAULT_PARAMETERS | {"cfd_factor_raw": 32768}
    with pytest.raises(ValueError):
        b.prepare_mca_configuration(invalid)
    assert b._pp.attrs["parameter_abi_ack"].value == "0"
    b.prepare_mca_configuration(dict(DEFAULT_PARAMETERS))
    b.set_crrc2_Cdelay(3)  # 64 ns averaging; selector is exponent 3, never 8.
    b.set_trapez_R(16)  # 16 ADC samples, never register encoding or old ns.
    b.set_cfd_delay(5)  # Kernel handles the phase-dependent encoding.
    b.set_trapez_T(0xFFFFFFFF)
    assert b._pp.attrs["crrc2_cdelay"].value == "3"
    assert b._pp.attrs["trapezoid_r_samples"].value == "16"
    assert b._pp.attrs["cfd_delay_samples"].value == "5"
    assert b.get_trapez_T() == 0xFFFFFFFF
    b.set_cfd_factor(32767 / 32768)
    assert b._pp.attrs["cfd_factor_raw"].value == "32767"
    with pytest.raises(ValueError):
        b.set_cfd_factor(1.0)


@pytest.mark.parametrize(
    "patch",
    [
        {"frame_samples": 409},
        {"frame_samples": 128.0},
        {"frame_samples": True},
        {"cfd_delay_samples": 4},
        {"trapezoid_r_samples": 4},
        {"crrc2_cdelay": 7},
        {"trapezoid_beta_raw": 1 << 32},
        {"cfd_factor_raw": 32768},
        {"psd_zero_crossing_enable": 1, "psd_window_high_samples": 128},
        {"cfd_enable": 1, "cfd_window_low_samples": 36, "cfd_window_high_samples": 32},
        {"charge_comparison_enable": 1, "charge_comparison_samples": 132},
        {"trapezoid_enable": 1, "trapezoid_start_offset_samples": 128},
        {"pileup_window_raw": 8},
    ],
)
def test_invalid_settings_are_rejected_without_snapping(patch):
    with pytest.raises(ValueError):
        validate_parameters(DEFAULT_PARAMETERS | patch, complete=True)


def test_inclusive_windows_pileup_boundary_and_independent_pretrigger():
    values = DEFAULT_PARAMETERS | dict(
        frame_samples=16,
        pretrigger_samples=2044,
        cfd_enable=1,
        cfd_window_high_samples=12,
        psd_zero_crossing_enable=1,
        psd_window_high_samples=12,
        pileup_window_raw=84,
    )
    validate_parameters(values, complete=True)  # (16/4-1)*(84+1)=255 exactly.
    with pytest.raises(ValueError):
        validate_parameters(values | {"pileup_window_raw": 85}, complete=True)


def test_unversioned_presets_and_fractional_values_are_not_migrated():
    with pytest.raises(ValueError, match="review older"):
        validate_document({"signal": {"frame_ns": 512}})
    document = {
        "parameter_abi_version": 2,
        "parameter_units": "adc_samples",
        "parameters": DEFAULT_PARAMETERS | {"frame_samples": 128.5},
    }
    with pytest.raises(ValueError):
        validate_document(document)


def test_non_dma_abi2_edit_stops_configures_and_restarts():
    b = backend(ack="2")
    b._pp.attrs["enable"].value = "1"
    mca = MultiChannelAnalyzer(b)

    def write():
        assert not b.get_global_enable()
        b.set_dpp_frame_samples(256)

    assert mca.reconfigure_while_running(write) is True
    assert b.get_global_enable()
    assert b.get_dpp_frame_samples() == 256


def test_abi2_dma_edit_is_rejected_before_stop_or_write():
    b = backend(ack="2")
    b._pp.attrs["enable"].value = "1"
    b.mca_dma_active = lambda: True
    mca = MultiChannelAnalyzer(b)
    write = Mock()
    with pytest.raises(RuntimeError, match="Stop list-mode DMA"):
        mca.reconfigure_while_running(write)
    write.assert_not_called()
    assert b.get_global_enable()


def test_failed_abi2_live_edit_stays_stopped():
    b = backend(ack="2")
    b._pp.attrs["enable"].value = "1"
    mca = MultiChannelAnalyzer(b)
    with pytest.raises(ValueError):
        mca.reconfigure_while_running(lambda: (_ for _ in ()).throw(ValueError("injected")))
    assert not b.get_global_enable()


def test_histogram_holds_shared_lock_across_complete_read(monkeypatch):
    b = backend(ack="2")
    _ = b.mca_configuration_lock
    entered, release, written = threading.Event(), threading.Event(), threading.Event()

    def read():
        entered.set()
        assert release.wait(2)
        return "completed snapshot"

    monkeypatch.setattr(b, "_read_histogram_serialized", read)
    reader = threading.Thread(target=b.read_histogram)
    writer = threading.Thread(target=lambda: (b.set_dpp_frame_samples(256), written.set()))
    reader.start()
    assert entered.wait(1)
    writer.start()
    assert not written.wait(0.05)
    release.set()
    reader.join(2)
    writer.join(2)
    assert written.is_set() and not reader.is_alive() and not writer.is_alive()


def test_gui_ns_timing_and_native_driver_derived_reciprocal(qtbot):
    device = create_offline_digitizer(0)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    try:
        assert view.ui.spinFrameSamples.suffix() == " ns"
        assert view.ui.spinFrameSamples.minimum() == 8
        assert view.ui.spinFrameSamples.singleStep() == 8
        view.ui.spinTrapR.setValue(32)
        view.ui.spinTrapR.editingFinished.emit()
        assert device.mca.filters.trapezoid.get_reciprocal_raw() == 8192
        assert view.ui.spinCrrc2Cdelay.value() == 3
        assert view.ui.spinTrapT.maximum() == 0xFFFFFFFF
        device.mca.apply_gate_timing(McaGateTiming(24, 256, 16, True))
        assert device.mca.get_pretrigger_samples() == 12
        assert device.mca.get_frame_samples() == 128
        assert device.mca.filters.charge_comparison.get_time() == 8
        assert device.mca.gate_timing() == McaGateTiming(24, 256, 16, True)
        assert view.energy_calibration_fingerprint()["parameter_units"] == "adc_samples"
    finally:
        device.close()


def test_poisoned_list_channel_is_rejected_before_geometry_or_buffer(monkeypatch):
    b = backend(ack="2")
    b._mca_dma_pp = b._pp
    b._mca_dma_control_lock = threading.RLock()
    b._lm_frame = SimpleNamespace(attrs={"dma_poisoned": SimpleNamespace(value="1")})
    with pytest.raises(RuntimeError, match="poisoned"):
        b._validate_lm_geometry()


def test_lm_close_preserves_drain_errno_and_releases_buffer(monkeypatch):
    import errno

    import iio

    b = backend(ack="2")
    b._mca_dma_lifecycle_lock = threading.RLock()
    b._mca_dma_cancel_timer = None
    native = object()
    buffer = SimpleNamespace(_buffer=native)
    b._mca_dma_buf = buffer
    channel = SimpleNamespace(enabled=True)
    b._lm_frame = SimpleNamespace(channels=[channel], attrs={})
    calls = []
    monkeypatch.setattr(b, "_mca_dma_pp_attr_set", lambda *args: calls.append("stop"))

    def fail_drain(*args):
        raise OSError(errno.EIO, "injected drain failure")

    monkeypatch.setattr(b, "_drain_mca_for_close", fail_drain)
    monkeypatch.setattr(iio, "_buffer_destroy", lambda handle: calls.append("destroy"))

    def fail_release():
        calls.append("release")
        raise RuntimeError("injected release failure")

    monkeypatch.setattr(b, "_wait_for_mca_dma_release", fail_release)
    with pytest.raises(OSError) as failure:
        b._close_mca_dma_buffer()
    assert failure.value.errno == errno.EIO
    assert "Drain:" in str(failure.value) and "Release:" in str(failure.value)
    assert calls == ["stop", "destroy", "release"]
    assert b._mca_dma_buf is None and buffer._buffer is None and not channel.enabled


@pytest.mark.parametrize("cancelled", [True, False])
def test_first_scope_reader_is_joined_after_cancel_or_failed_enable(monkeypatch, cancelled):
    import errno

    import iio

    from nlab.hardware.digitizer.scope_sums import ScopeSumGeometry

    b = object.__new__(IIODigitizerBackend)
    b._ch = 0
    b._dma_capture_geometry = ScopeSumGeometry(16, 124, 3)
    b._sum_draining = cancelled
    b._dma_stop_requested = threading.Event()
    released, exited = threading.Event(), threading.Event()
    buffer = SimpleNamespace(_buffer=object(), cancel=released.set)

    def refill(handle):
        assert released.wait(2)
        exited.set()
        raise OSError(errno.EBADF, "cancelled native read")

    monkeypatch.setattr(iio, "_buffer_refill", refill)
    monkeypatch.setattr(b, "_dma_get_enable", lambda: False)
    enable = Mock(side_effect=OSError(errno.EBUSY, "enable refused"))
    monkeypatch.setattr(b, "_dma_set_enable", enable)
    with pytest.raises(OSError) as failure:
        b._start_reader_then_enable(buffer)
    assert failure.value.errno == (errno.ECANCELED if cancelled else errno.EBUSY)
    assert released.is_set() and exited.is_set()
    assert enable.call_count == (0 if cancelled else 1)
