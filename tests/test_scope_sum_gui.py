"""Inline sum GUI and joined lifecycle using synthetic packets, not live validation."""

import struct
import threading
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog

from nlab.controllers.current_monitor_controller import CurrentMonitorController
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.hardware.digitizer.scope import TriggerMode
from nlab.hardware.digitizer.scope_sums import ScopeSumAnalysis, ScopeSumWriter, decode_sum_packet


def _packet():
    raw = bytearray(4096)
    struct.pack_into("<4Q", raw, 0, (1 << 60) + 1, 0x32534D5553504456, 16 | (1 << 48), 1 << 63)
    for index in range(508):
        struct.pack_into("<Q", raw, 32 + 8 * index, 16 * (32768 + 7))
    return decode_sum_packet(bytes(raw))


class _Capture:
    def __init__(self, *, fail=False):
        self.analysis = ScopeSumAnalysis()
        self.entered = threading.Event()
        self.cancelled = threading.Event()
        self.exited = threading.Event()
        self.output = None
        self.fail = fail

    def request_stop(self):
        self.cancelled.set()

    def run(self, path):
        self.output = path
        self.analysis.append(_packet())
        self.entered.set()
        try:
            if self.fail:
                raise RuntimeError("injected capture failure")
            assert self.cancelled.wait(3)
            if path is not None:
                writer = ScopeSumWriter(path, {
                    "channel": 0, "configuration_yaml": getattr(self, "configuration_yaml", ""),
                })
                writer.append(_packet())
                writer.close(complete=True)
            return {"complete": True}
        finally:
            self.exited.set()


def _views(qtbot, monkeypatch, capture=None):
    device = create_offline_digitizer(0)
    monkeypatch.setattr(device._backend, "get_scope_trigger_modes", lambda: tuple(range(6)))
    capture = capture or _Capture()
    streamer = Mock()
    streamer.create_sum_capture = Mock(return_value=capture)
    scope = ScopeController(device.scope, scope_dma=streamer)
    current = CurrentMonitorController(device.mca, 0, scope_controller=scope)
    qtbot.addWidget(scope)
    qtbot.addWidget(current)
    current.ui.comboMode.setCurrentIndex(2)
    return device, scope, current, capture


def _close(device, scope, current):
    current.stop_monitor_sync()
    scope.close_viewer_client()
    device.close()


def test_sums_use_existing_current_plot_and_restore_scope_after_stop(qtbot, monkeypatch):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        assert [scope.ui.comboTriggerMode.itemData(i) for i in range(5)] == list(range(5))
        assert scope.ui.comboTriggerMode.findData(5) == -1
        scope.ui.comboTriggerMode.setCurrentIndex(3)
        device.scope.set_pretrigger_samples(16)
        device.scope.set_frame_period_cycles(9)
        device.scope.set_frame_samples(2048)
        current.ui.spinSumSamples.setValue(256)
        current.ui.spinDmaGap.setValue(0.080)
        viewer = Mock(side_effect=AssertionError("structured sums are not waveforms"))
        monkeypatch.setattr(scope, "_viewer_scope_for_worker", viewer)
        dialogs = {id(w) for w in QApplication.topLevelWidgets() if isinstance(w, QDialog)}
        original_plot = current.ui.plotCurrent
        current.start_monitor()  # Returns immediately; no modal event loop.
        assert capture.entered.wait(1)
        qtbot.waitUntil(lambda: current._curve.yData is not None)
        assert current.ui.plotCurrent is original_plot
        assert current.ui.plotCurrent.isEnabled()
        remaining = {id(w) for w in QApplication.topLevelWidgets() if isinstance(w, QDialog)}
        assert not remaining - dialogs
        assert current._curve.yData[-1] == 7
        assert current._curve.xData[-1] == 0
        assert "7 ADC codes" in current.ui.lblCurrent.text()
        assert "sample-weighted" in current.ui.lblInterval.text()
        assert current.ui.btnStop.text() == "Stop and drain"
        assert not current.ui.comboMode.isEnabled()
        assert not current.ui.groupCalibration.isEnabled()
        assert device.scope.get_trigger_mode() == TriggerMode.PERIODIC_SUM
        assert device.scope.get_pretrigger_samples() == 0
        assert device.scope.get_frame_samples() == 256
        assert not current.ui.spinSumSamples.isEnabled()
        assert not current.ui.spinDmaGap.isEnabled()
        assert device.scope.get_frame_period_cycles() == 10
        assert not scope.ui.btnStart.isEnabled()
        assert not scope.ui.groupTrigger.isEnabled()
        scope._request_frame()
        scope._on_start()
        viewer.assert_not_called()
        current.request_monitor_stop()
        qtbot.waitUntil(lambda: not current._is_running())
        assert capture.exited.is_set()
        assert device.scope.get_trigger_mode() == TriggerMode.RISING_EDGE
        assert device.scope.get_pretrigger_samples() == 16
        assert device.scope.get_frame_period_cycles() == 9
        assert device.scope.get_frame_samples() == 2048
        assert current.ui.spinSumSamples.isEnabled()
        assert current.ui.spinDmaGap.isEnabled()
        assert scope.ui.btnStart.isEnabled()
        assert not scope.measurement_active
        assert current.ui.comboMode.isEnabled()
        assert current._sum_worker.thread is None
        assert "drained" in current.status.text()
        assert current._curve.yData[-1] == 7  # Final view remains available.
    finally:
        _close(device, scope, current)


@pytest.mark.parametrize("scope_first", [False, True])
def test_shutdown_joins_before_restoring_and_is_idempotent(qtbot, monkeypatch, scope_first):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        current.start_monitor()
        assert capture.entered.wait(1)
        worker, thread = current._sum_worker, current._sum_worker.thread
        if scope_first:
            scope.stop_dma_sync()
        current.stop_monitor_sync()
        current.stop_monitor_sync()
        assert capture.exited.is_set() and not thread.is_alive()
        assert worker.thread is None and scope._sum_worker is None
        assert not current._is_running()
        assert device.scope.get_trigger_mode() != TriggerMode.PERIODIC_SUM
        assert not scope.measurement_active
    finally:
        _close(device, scope, current)


def test_widget_close_stops_and_joins_sum_reader(qtbot, monkeypatch):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        current.start_monitor()
        assert capture.entered.wait(1)
        current.close()
        assert capture.exited.is_set()
        assert current._sum_worker.thread is None
        assert not current._is_running()
    finally:
        _close(device, scope, current)


def test_capture_failure_restores_controls_and_reports_error(qtbot, monkeypatch):
    device, scope, current, capture = _views(qtbot, monkeypatch, _Capture(fail=True))
    try:
        current.start_monitor()
        qtbot.waitUntil(lambda: not current._is_running())
        assert "injected capture failure" in current.status.text()
        assert device.scope.get_trigger_mode() != TriggerMode.PERIODIC_SUM
        assert current.ui.btnStart.isEnabled() and scope.ui.btnStart.isEnabled()
        assert current._sum_worker.thread is None
    finally:
        _close(device, scope, current)


def test_current_sums_are_capability_gated_and_respect_replay(qtbot, monkeypatch):
    device = create_offline_digitizer(0)
    scope = ScopeController(device.scope)
    current = CurrentMonitorController(device.mca, 0, scope_controller=scope)
    qtbot.addWidget(scope)
    qtbot.addWidget(current)
    try:
        assert current.ui.comboMode.findText("Periodic sums") == -1
        with pytest.raises(ValueError, match="IP123/IP124"):
            current.apply_configuration_settings({"mode": "periodic_sum"})
        monkeypatch.setattr(scope, "current_sum_supported", lambda: True)
        scope._replay_hardware_interlocked = True
        with pytest.raises(RuntimeError, match="replay"):
            scope.start_current_sum_monitor(gap_cycles=0)
        scope._replay_hardware_interlocked = False
    finally:
        _close(device, scope, current)


def test_sums_cannot_preempt_scope_and_configuration_failure_restores(qtbot, monkeypatch):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        scope._set_measurement_active(True)
        current.start_monitor()
        assert "stop the active Scope" in current.status.text()
        scope._set_measurement_active(False)
        old = device.scope.get_frame_period_cycles()
        write = device.scope.set_frame_period_cycles
        monkeypatch.setattr(
            device.scope, "set_frame_period_cycles",
            Mock(side_effect=[RuntimeError("injected configuration failure"), write(old)]),
        )
        current.start_monitor()
        assert "injected configuration failure" in current.status.text()
        assert scope._sum_worker is None
        assert device.scope.get_trigger_mode() != TriggerMode.PERIODIC_SUM
        assert not capture.entered.is_set()
    finally:
        _close(device, scope, current)


def test_recording_opens_in_dedicated_tool_without_changing_live_plot(qtbot, monkeypatch, tmp_path):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        current.ui.cbRecordSums.setChecked(True)
        current.start_monitor()
        assert "Choose a new HDF5 file" in current.status.text()
        assert not capture.entered.is_set()
        path = tmp_path / "sums.h5"
        current.ui.editSumFile.setText(str(path))
        current.start_monitor()
        assert capture.entered.wait(1)
        assert not current.ui.editSumFile.isEnabled()
        assert current.ui.btnOpenSums.isEnabled()
        previous_path = tmp_path / "previous-capture.h5"
        writer = ScopeSumWriter(previous_path, {"channel": 1})
        writer.append(_packet())
        writer.close(complete=True)
        active_worker = current._sum_worker
        current.open_sum_capture(previous_path)
        qtbot.waitUntil(lambda: current._capture_dialog._chunk is not None)
        assert current._is_running()
        assert current._sum_worker is active_worker
        assert not capture.cancelled.is_set()
        current.stop_monitor_sync()
        assert capture.output == path and path.exists()
        current._curve.setData([0, 1], [2, 3])
        current.open_sum_capture(path)
        qtbot.waitUntil(lambda: current._capture_dialog._chunk is not None)
        assert current._capture_dialog._chunk.mean == 7
        assert current._curve.yData[-1] == 3
        assert not current._is_running()
        assert current.ui.btnStart.isEnabled()
    finally:
        if current._capture_dialog is not None:
            current._capture_dialog.close_without_prompt()
        _close(device, scope, current)


def test_replay_preserves_incomplete_status_without_hardware(qtbot, tmp_path):
    path = tmp_path / "incomplete.h5"
    writer = ScopeSumWriter(path, {"channel": 1})
    writer.append(_packet())
    writer.close(complete=False, error="injected incomplete capture")
    mca = Mock()
    current = CurrentMonitorController(mca, 0)
    qtbot.addWidget(current)
    try:
        current.open_sum_capture(path)
        dialog = current._capture_dialog
        qtbot.waitUntil(lambda: dialog._chunk is not None)
        assert dialog._chunk.mean == 7
        assert "Complete=False" in dialog.capture_status.text()
        assert "source channel 1" in dialog.capture_status.text()
        assert "incomplete capture" in dialog.capture_status.text()
        mca.assert_not_called()
    finally:
        current.close()


def test_mode_change_clears_old_trace_and_sum_settings_remain_saved(qtbot, monkeypatch):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        current._curve.setData([0, 1], [100, 200])
        current.ui.comboMode.setCurrentIndex(0)
        assert current._curve.yData is None or len(current._curve.yData) == 0
        current.apply_configuration_settings({"mode": "periodic_sum", "dma_gap_cycles": 17})
        assert current.configuration_settings()["mode"] == "periodic_sum"
        assert current.configuration_settings()["dma_gap_cycles"] == 17
        current.apply_configuration_settings({"sum_frame_samples": 512})
        assert current.configuration_settings()["sum_frame_samples"] == 512
        assert current.ui.plotCurrent.isEnabled()
        assert current.ui.groupReading.isEnabled()
        assert not capture.entered.is_set()
    finally:
        _close(device, scope, current)


def test_scope_stop_drains_inline_sum_capture(qtbot, monkeypatch):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        current.start_monitor()
        assert capture.entered.wait(1)
        scope._on_stop()
        qtbot.waitUntil(lambda: not current._is_running())
        assert capture.cancelled.is_set() and capture.exited.is_set()
        assert "drained" in current.status.text()
        assert current._sum_worker.thread is None
    finally:
        _close(device, scope, current)


@pytest.mark.parametrize("samples,gap", [(4, 0), (8188, 65535), (1024, 100)])
def test_sum_timing_controls_have_native_units_and_preview(qtbot, monkeypatch, samples, gap):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        spin = current.ui.spinSumSamples
        assert (spin.minimum(), spin.maximum(), spin.singleStep()) == (4, 8188, 4)
        assert spin.suffix() == " samples"
        assert not current.ui.sumTimingControls.isHidden()
        assert current.ui.labelDmaGap.text() == "Window gap:"
        assert current.ui.spinDmaGap.locale().decimalPoint() == "."
        spin.setValue(samples)
        current._set_dma_gap_cycles(gap)
        expected = (
            f"Window: {samples * 0.002:.3f} us; "
            f"start interval: {(samples * 2 + gap * 8) / 1000:.3f} us"
        )
        assert current.ui.labelSumTiming.text() == expected
        assert current._sum_frame_samples() == samples
        assert current._dma_gap_cycles() == gap
        assert device.scope.get_frame_samples() == 1024  # Edits apply only on Start.
        assert device.scope.get_frame_period_cycles() == 0
        assert not capture.entered.is_set()
    finally:
        _close(device, scope, current)


@pytest.mark.parametrize("typed,expected", [(409, 408), (410, 412), (1, 4), (9999, 8188)])
def test_sum_window_edits_clamp_to_nearest_supported_size(qtbot, monkeypatch, typed, expected):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        current.ui.spinSumSamples.setValue(typed)
        current.ui.spinSumSamples.editingFinished.emit()
        assert current.ui.spinSumSamples.value() == expected
        assert current.configuration_settings()["sum_frame_samples"] == expected
        current.set_replay_hardware_interlock(True)
        assert not current.ui.spinSumSamples.isEnabled()
        assert not current.ui.spinDmaGap.isEnabled()
        current.set_replay_hardware_interlock(False)
        assert current.ui.spinSumSamples.isEnabled()
    finally:
        _close(device, scope, current)


@pytest.mark.parametrize("samples", [0, 409, 8192])
def test_invalid_sum_window_is_rejected_before_hardware_writes(qtbot, monkeypatch, samples):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    writes = Mock()
    try:
        for name in (
            "set_pretrigger_samples", "set_frame_samples", "set_edge", "set_frame_period_cycles"
        ):
            monkeypatch.setattr(device._backend, name, getattr(writes, name))
        with pytest.raises(ValueError, match="frame_samples"):
            scope.start_current_sum_monitor(frame_samples=samples, gap_cycles=0)
        assert writes.mock_calls == []
        assert scope._sum_previous_settings is None
        assert not capture.entered.is_set()
    finally:
        _close(device, scope, current)


def test_sum_plot_breaks_at_missing_windows_including_timestamp_wrap(qtbot, monkeypatch):
    device, scope, current, _ = _views(qtbot, monkeypatch)
    try:
        analysis = ScopeSumAnalysis()
        first = (1 << 64) - 10
        for i in range(2):
            raw = bytearray(_packet().raw)
            struct.pack_into(
                "<4Q", raw, 0, (first + i * 510 * 4) & ((1 << 64) - 1),
                0x32534D5553504456, 16 | ((3 if i else 508) << 48), i | (i << 63),
            )
            analysis.append(decode_sum_packet(bytes(raw)))
        current._sum_worker = Mock(analysis=analysis, capture=object(), done=False)
        current._render_sum_snapshot()
        connections = current._curve.opts["connect"]
        expected = np.ones(511, dtype=bool)
        expected[507] = expected[-1] = False
        np.testing.assert_array_equal(connections, expected)
        assert len(current._curve.yData) == 511
        assert "gaps (timestamps): 2" in current.status.text()
    finally:
        current._sum_worker = None
        _close(device, scope, current)


def test_configuration_snapshot_after_sum_setup_before_worker(qtbot, monkeypatch, tmp_path):
    import yaml
    device, scope, current, capture = _views(qtbot, monkeypatch)
    snapshots = []

    def snapshot():
        assert not capture.entered.is_set()
        assert device.scope.get_trigger_mode() == TriggerMode.PERIODIC_SUM
        assert device.scope.get_frame_samples() == 2048
        assert device.scope.get_pretrigger_samples() == 0
        document = {"format_version": 5, "hardware": {"channels": {
            "0": {"scope": {"edge_mode": device.scope.get_trigger_mode().value}},
            "1": {"mca": {"fixture": "other channel"}},
        }}, "connection": {"ip": "fixture"}, "application": {"fixture": True}}
        snapshots.append(document)
        return document

    scope._measurement_configuration = snapshot
    factory = scope._scope_dma.create_sum_capture

    def create(**kwargs):
        capture.configuration_yaml = kwargs["configuration_yaml"]
        return factory(**kwargs)

    monkeypatch.setattr(scope._scope_dma, "create_sum_capture", create)
    try:
        current.ui.spinSumSamples.setValue(2048)
        current.ui.cbRecordSums.setChecked(True)
        path = tmp_path / "configured.h5"
        current.ui.editSumFile.setText(str(path))
        current.start_monitor()
        assert capture.entered.wait(1)
        current.stop_monitor_sync()
        with h5py.File(path, "r") as source:
            assert yaml.safe_load(source["configuration_yaml"][()].decode()) == snapshots[0]
        assert len(snapshots) == 1
    finally:
        _close(device, scope, current)


def test_configuration_read_failure_never_arms_and_restores_scope(qtbot, monkeypatch, tmp_path):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    scope._measurement_configuration = Mock(side_effect=RuntimeError("snapshot failed"))
    try:
        original = device.scope.get_trigger_mode()
        current.ui.cbRecordSums.setChecked(True)
        current.ui.editSumFile.setText(str(tmp_path / "failed.h5"))
        current.start_monitor()
        assert "snapshot failed" in current.status.text()
        assert not capture.entered.is_set()
        assert device.scope.get_trigger_mode() == original
        assert scope._sum_worker is None
    finally:
        _close(device, scope, current)
