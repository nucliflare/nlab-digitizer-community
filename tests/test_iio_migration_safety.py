"""Current PetaLinux contract regressions; controlled doubles, no hardware I/O."""

import errno
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import iio
import pytest

from nlab.hardware.digitizer.acquisition_profile import AcquisitionProfile
from nlab.hardware.digitizer.backends import iio_backend as module
from nlab.hardware.digitizer.backends.iio_backend import DmaReaderActiveError, IIODigitizerBackend
from nlab.hardware.digitizer.dma import IIOMcaDmaStreamer
from nlab.hardware.digitizer.iio_listmode import VDPP_ZC_CALC_SCHEMA
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.hardware.digitizer.mca_parameters import DEFAULT_PARAMETERS


@pytest.fixture(autouse=True)
def forbid_hardware(monkeypatch):
    monkeypatch.setattr(iio, "Context", Mock(side_effect=AssertionError("No hardware access")))


@pytest.fixture
def backend(monkeypatch):
    b = object.__new__(IIODigitizerBackend)
    b._ch = 0
    values = DEFAULT_PARAMETERS | {
        "parameter_abi_version": "2", "parameter_units": "adc_samples",
        "parameter_abi_ack": "2", "enable": "0", "list_buffer_active": "0",
        "measurement_in_progress": "0", "psd_zero_crossing_mode_available":
        "0 rc_cr2 1 rc_cr 2 trapezoid",
        "hw_version": 101, "sw_version": 103023,
    }
    b._pp = SimpleNamespace(attrs={key: SimpleNamespace(value=str(value))
                                   for key, value in values.items()})
    b._mca_dma_pp = b._pp
    b._mca_dma_buf = None
    b._mca_dma_control_lock = threading.RLock()
    b._mca_dma_lifecycle_lock = threading.RLock()
    b._mca_dma_cancel_timer = None
    b._mca_dma_cancel_explained = False
    b._mca_dma_stop_requested = threading.Event()
    b._lm_frame = SimpleNamespace(attrs={
        "ip_version": SimpleNamespace(value="121"), "buffer_active": SimpleNamespace(value="0"),
    }, channels=[])
    b._dma_scope = SimpleNamespace(attrs={})
    b._dma_buf = None
    b._dma_stop_requested = threading.Event()
    b._dma_fault_latched = False
    b._scope_dma_reader = None
    b._mca_dma_reader = None
    monkeypatch.setattr(module, "_DMA_CLOSE_DRAIN_SECONDS", 0.01)
    monkeypatch.setattr(module, "_DMA_CLOSE_JOIN_SECONDS", 0.02)
    monkeypatch.setattr(module, "_LM_CLOSE_JOIN_SECONDS", 0.02)
    monkeypatch.setattr(module, "_MCA_STOP_STATUS_SECONDS", 0.02)
    return b


@pytest.mark.parametrize("cancel_raises", [False, True])
def test_scope_retains_native_owner_until_cancelled_reader_exits(
    backend, monkeypatch, cancel_raises
):
    release, entered = threading.Event(), threading.Event()
    native = object()
    def cancel():
        if cancel_raises:
            raise OSError(errno.EIO, "cancel failed")
    buf = SimpleNamespace(_buffer=native, cancel=cancel)
    backend._dma_buf = buf
    def refill(_buf):
        entered.set()
        assert release.wait(2)
        raise OSError(errno.EBADF, "cancelled")
    monkeypatch.setattr(backend, "_refill_scope_buffer", refill)
    monkeypatch.setattr(backend, "_dma_get_enable", lambda: False)
    monkeypatch.setattr(backend, "_wait_for_dma_release", lambda: None)
    destroy = Mock()
    monkeypatch.setattr(iio, "_buffer_destroy", destroy)
    try:
        with pytest.raises(DmaReaderActiveError):
            backend._close_dma_buffer()
        assert entered.is_set() and backend._scope_dma_reader.is_alive()
        assert backend._dma_buf is buf and buf._buffer is native
        assert backend._dma_fault_latched
        destroy.assert_not_called()
    finally:
        release.set()
        backend._scope_dma_reader.join(2)
    buf.cancel = lambda: None
    backend._close_dma_buffer(drain=False)
    destroy.assert_called_once_with(native)
    assert backend._dma_buf is None


def test_mca_failed_start_preserves_errno_and_retains_live_reader(backend, monkeypatch):
    release, entered = threading.Event(), threading.Event()
    native = object()
    buf = SimpleNamespace(_buffer=native, cancel=lambda: None)
    monkeypatch.setattr(backend, "_create_mca_dma_buffer",
                        lambda: setattr(backend, "_mca_dma_buf", buf))
    def refill(_buf):
        entered.set()
        assert release.wait(2)
        raise OSError(errno.EBADF, "controlled cancellation")
    monkeypatch.setattr(backend, "_refill_mca_dma_buffer", refill)
    def enable(name, value):
        if value == "1":
            raise OSError(errno.EPROTO, "START refused")
    monkeypatch.setattr(backend, "_mca_dma_pp_attr_set", enable)
    monkeypatch.setattr(backend, "_wait_for_mca_dma_release", lambda: None)
    destroy = Mock()
    monkeypatch.setattr(iio, "_buffer_destroy", destroy)
    try:
        with pytest.raises(OSError) as caught:
            backend.start_mca_dma_capture()
        assert caught.value.errno == errno.EPROTO
        assert entered.is_set() and backend._mca_dma_reader.is_alive()
        assert backend._mca_dma_buf is buf and buf._buffer is native
        assert backend.mca_recovery_required()
        destroy.assert_not_called()
    finally:
        release.set()
        backend._mca_dma_reader.join(2)
    backend._close_mca_dma_buffer(drain=False)
    destroy.assert_called_once_with(native)


def test_scope_drain_eio_is_propagated_after_safe_destruction(backend, monkeypatch):
    backend._dma_buf = SimpleNamespace(_buffer=object(), cancel=lambda: None)
    monkeypatch.setattr(backend, "_dma_get_enable", lambda: False)
    monkeypatch.setattr(backend, "_wait_for_dma_release", lambda: None)
    monkeypatch.setattr(
        backend, "_refill_scope_buffer", Mock(side_effect=OSError(errno.EIO, "drain"))
    )
    destroy = Mock()
    monkeypatch.setattr(iio, "_buffer_destroy", destroy)
    with pytest.raises(OSError) as caught:
        backend._close_dma_buffer()
    assert caught.value.errno == errno.EIO
    assert backend._dma_buf is None and backend._dma_fault_latched
    destroy.assert_called_once()


def test_dma_stop_serializes_with_the_whole_histogram(backend):
    written, attempted = threading.Event(), threading.Event()
    backend._pp.attrs["enable"].value = "1"
    def stop():
        attempted.set()
        backend._mca_dma_pp_attr_set("enable", "0")
        written.set()
    with backend.mca_configuration_lock:
        worker = threading.Thread(target=stop)
        worker.start()
        assert attempted.wait(1)
        assert not written.wait(0.03)
        assert backend._pp.attrs["enable"].value == "1"
    worker.join(2)
    assert written.is_set() and not worker.is_alive()


def test_persistent_progress_blocks_live_edit_and_requires_explicit_recovery(backend):
    backend._pp.attrs["enable"].value = "1"
    backend._pp.attrs["measurement_in_progress"].value = "1"
    analyzer = MultiChannelAnalyzer(backend)
    with pytest.raises(RuntimeError, match="measurement_in_progress=1"):
        analyzer.reconfigure_while_running(lambda: backend.set_dpp_frame_samples(256))
    assert backend._pp.attrs["enable"].value == "0"
    assert backend._pp.attrs["frame_samples"].value == "128"
    assert analyzer.recovery_required()
    backend._pp.attrs["measurement_in_progress"].value = "0"
    with pytest.raises(RuntimeError, match="recovery required"):
        analyzer.start()
    analyzer.acknowledge_dma_recovery()
    assert not analyzer.recovery_required()
    analyzer.start()
    assert backend._pp.attrs["enable"].value == "1"


def test_psd2_needs_advertisement_before_ack_and_keeps_integration_energy(backend):
    candidate = DEFAULT_PARAMETERS | {
        "psd_zero_crossing_mode": 2, "psd_zero_crossing_enable": 1,
        "trapezoid_enable": 0, "trapezoid_start_offset_samples": 8188,
    }
    backend._pp.attrs["parameter_abi_ack"].value = "0"
    available = backend._pp.attrs.pop("psd_zero_crossing_mode_available")
    with pytest.raises(ValueError, match="not advertised"):
        backend.prepare_mca_configuration(candidate)
    assert backend._pp.attrs["parameter_abi_ack"].value == "0"
    backend._pp.attrs["psd_zero_crossing_mode_available"] = available
    backend.configure_mca_parameters(candidate)
    assert backend._pp.attrs["trapezoid_enable"].value == "0"
    assert backend._pp.attrs["trapezoid_r_samples"].value == "32"
    assert backend._pp.attrs["psd_zero_crossing_mode"].value == "2"


def test_standard_decoder_needs_no_manual_metadata_and_records_unknown_image(backend):
    identity = backend.mca_producer_identity()
    assert identity["qualified_record_schema"] == VDPP_ZC_CALC_SCHEMA
    assert identity["firmware_id"] == "unknown"
    assert "application default" in identity["decoder_selection"]
    assert "unverified" in identity["decoder_selection"]
    assert IIOMcaDmaStreamer(backend, channel=0).qualified_record_schema() == VDPP_ZC_CALC_SCHEMA
    backend._pp.attrs["sw_version"].value = "999999"
    assert "qualified_record_schema" not in backend.mca_producer_identity()
    backend._acquisition_profile = AcquisitionProfile(
        dac_output=1, firmware_id="reviewed-image-sha256", record_schema=VDPP_ZC_CALC_SCHEMA)
    identity = backend.mca_producer_identity()
    assert identity["qualified_record_schema"] == VDPP_ZC_CALC_SCHEMA
    assert identity["firmware_id"] == "reviewed-image-sha256"
    assert "unverified" in identity["decoder_selection"]


@pytest.mark.parametrize("values", [
    {"record_schema": VDPP_ZC_CALC_SCHEMA}, {"dac_output": True}, {"dac_output": 2},
    {"record_schema": "guessed-schema", "firmware_id": "image"},
])
def test_invalid_wiring_or_producer_provenance_is_rejected(values):
    with pytest.raises(ValueError):
        AcquisitionProfile.from_mapping(values)


def test_capture_keeps_primary_errno_cleanup_errors_and_poison_metadata(tmp_path):
    class FailingBackend:
        def start_mca_dma_capture(self, on_started=None):
            raise OSError(errno.EPIPE, "primary refill failure")
        def abort_mca_dma_capture(self):
            raise OSError(errno.EPERM, "close failure")
        def close_mca_dma_capture(self, on_frame=None):
            raise AssertionError("No healthy drain after a failed refill")
        def get_mca_dma_capture_diagnostics(self):
            return 0, 1, 0, 0
        def get_mca_dma_capture_status(self):
            return {"dma_poisoned": 1, "queued_blocks": 2, "dma_fault_kind": "lm_category"}
    streamer = IIOMcaDmaStreamer(FailingBackend(), channel=0)
    path = tmp_path / "failed.bin"
    with pytest.raises(OSError) as caught:
        streamer.stream_events(threading.Event(), path)
    assert caught.value.errno == errno.EPIPE
    status = streamer.last_capture_diagnostics
    assert status["primary_errno"] == errno.EPIPE and status["dma_poisoned"] == 1
    assert status["cleanup_errors"][0]["errno"] == errno.EPERM
    assert status["streamed_frames"] == 0 and not status["continuity_valid"]
    document = json.loads(path.with_suffix(".json").read_text())
    assert document["final_status"]["primary_errno"] == errno.EPIPE
    assert not document["continuity_valid"]


def test_unverified_dac_keeps_scope_viewer_available_without_baseline_writes(qtbot, monkeypatch):
    from nlab.controllers.scope_controller import ScopeController
    from nlab.hardware.digitizer.offline import create_offline_digitizer
    from nlab.utils.settings_io import collect_channel_hardware

    device = create_offline_digitizer(0)
    monkeypatch.setattr(device._backend, "supports_dac", lambda: False)
    forbidden = Mock(side_effect=AssertionError("Unverified baseline I/O"))
    monkeypatch.setattr(device._backend, "set_dac_value", forbidden)
    monkeypatch.setattr(device._backend, "get_dac_value", forbidden)
    view = ScopeController(device.scope)
    qtbot.addWidget(view)
    assert view.ui.btnStart.isEnabled()
    assert not view.ui.spinDacValue.isEnabled() and not view.ui.btnAutoSetup.isEnabled()
    view._on_auto_setup()
    saved = collect_channel_hardware(device.scope, None, None)
    assert "dac_value" not in saved["scope"]
    forbidden.assert_not_called()


@pytest.mark.parametrize("modes", [(0, 1), (0, 1, 2)])
def test_gui_psd2_matches_discovered_modes(qtbot, monkeypatch, modes):
    from nlab.controllers.mca_controller import MCAController
    from nlab.hardware.digitizer.offline import create_offline_digitizer

    device = create_offline_digitizer(0)
    monkeypatch.setattr(device._backend, "get_psd_zc_modes", lambda: modes)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    assert tuple(view.ui.comboPsdZcMode.itemData(i)
                 for i in range(view.ui.comboPsdZcMode.count())) == modes
    if 2 in modes:
        view.ui.comboPsdZcMode.setCurrentIndex(view.ui.comboPsdZcMode.findData(2))
        assert device.mca.filters.psd_zc.get_mode() == 2


def test_settings_preserve_explicit_profile_before_baseline_write(monkeypatch):
    from nlab.hardware.digitizer.offline import create_offline_digitizer
    from nlab.utils.settings_io import apply_channel_hardware, collect_channel_hardware

    device = create_offline_digitizer(0)
    profile = AcquisitionProfile(1, "image-id", VDPP_ZC_CALC_SCHEMA).to_dict()
    device.scope.set_frame_samples(128)
    device.scope.apply_acquisition_profile(profile)
    document = collect_channel_hardware(device.scope, None, None)
    assert document["acquisition_profile"] == profile
    target = create_offline_digitizer(1)
    def write_baseline(value):
        assert target.scope.acquisition_profile() == profile
    monkeypatch.setattr(target._backend, "set_dac_value", write_baseline)
    apply_channel_hardware(target.scope, None, None, document)


def test_scope_close_detects_poison_after_native_destroy(backend, monkeypatch):
    poison = SimpleNamespace(value="0")
    backend._dma_scope.attrs["dma_poisoned"] = poison
    backend._dma_buf = SimpleNamespace(_buffer=object(), cancel=lambda: None)
    monkeypatch.setattr(backend, "_dma_get_enable", lambda: False)
    monkeypatch.setattr(backend, "_wait_for_dma_release", lambda: None)
    monkeypatch.setattr(iio, "_buffer_destroy", lambda _native: setattr(poison, "value", "1"))
    with pytest.raises(OSError, match="reboot required"):
        backend._close_dma_buffer(drain=False)
    assert backend._dma_fault_latched and backend._dma_buf is None


def test_failed_hdf5_final_flush_retracts_completion_and_joins_writer(tmp_path, monkeypatch):
    import h5py
    import numpy as np

    from nlab.hardware.digitizer.mca_capture import McaCaptureWriter, McaDmaOutputMode

    path = tmp_path / "failed-flush.h5"
    original_flush = h5py.File.flush
    failed = []
    def fail_once(file):
        original_flush(file)
        if bool(file["capture_complete"][()]) and not failed:
            failed.append(True)
            raise OSError(errno.ENOSPC, "injected final flush failure")
    writer = McaCaptureWriter(
        path=path, mode=McaDmaOutputMode.HDF5, configuration_yaml="", binary_header=b""
    )
    monkeypatch.setattr(h5py.File, "flush", fail_once)
    with pytest.raises(RuntimeError) as caught:
        writer.close(complete=True)
    assert caught.value.errno == errno.ENOSPC
    assert not writer._thread.is_alive()
    assert isinstance(writer._error, tuple)
    with h5py.File(path, "r") as capture:
        assert not np.bool_(capture["capture_complete"][()])


def test_psd2_rejected_before_unrelated_hardware_writes(backend):
    from nlab.utils.settings_io import apply_channel_hardware

    backend._pp.attrs.pop("psd_zero_crossing_mode_available")
    scope = Mock()
    native = DEFAULT_PARAMETERS | {"psd_zero_crossing_mode": 2}
    with pytest.raises(ValueError, match="not advertised"):
        apply_channel_hardware(scope, MultiChannelAnalyzer(backend), None, {
            "scope": {"trigger_level": 42}, "mca": {
                "parameter_abi_version": 2, "parameter_units": "adc_samples",
                "parameters": native,
            },
        })
    assert not scope.mock_calls
