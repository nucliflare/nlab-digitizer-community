"""Inline sum GUI and joined lifecycle using synthetic packets, not live validation."""

import errno
import struct
import threading
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from nlab.controllers import current_monitor_controller as current_module
from nlab.controllers.current_monitor_controller import CurrentMonitorController
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer import scope_sums
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.hardware.digitizer.scope import TriggerMode
from nlab.hardware.digitizer.scope_sums import (
    ScopeSumAnalysis,
    ScopeSumCapture,
    ScopeSumGeometry,
    ScopeSumWriter,
    decode_sum_packet,
)


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

    def run(self, path, *, overwrite=False):
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
                }, overwrite=overwrite)
                writer.append(_packet())
                writer.close(complete=True)
            return {"complete": True}
        finally:
            self.exited.set()


def _views(qtbot, monkeypatch, capture=None, *, channel=0):
    device = create_offline_digitizer(channel)
    monkeypatch.setattr(device._backend, "get_scope_trigger_modes", lambda: tuple(range(6)))
    capture = capture or _Capture()
    streamer = Mock()
    streamer.create_sum_capture = Mock(return_value=capture)
    scope = ScopeController(device.scope, scope_dma=streamer)
    current = CurrentMonitorController(device.mca, channel, scope_controller=scope)
    qtbot.addWidget(scope)
    qtbot.addWidget(current)
    current.ui.comboMode.setCurrentIndex(2)
    return device, scope, current, capture


def _close(device, scope, current):
    current.stop_monitor_sync()
    scope.close_viewer_client()
    device.close()


@pytest.fixture
def sum_save_settings(monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("dma/save_folder", str(tmp_path / "measurements"))
    monkeypatch.setattr(current_module, "QSettings", lambda: settings)
    return settings


@pytest.mark.parametrize("channel", [0, 1])
@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_sum_filename_generates_fresh_channel_path_on_start(
    qtbot, monkeypatch, tmp_path, sum_save_settings, channel, blank,
):
    clock = Mock(wraps=current_module.time)
    clock.strftime.return_value = "20261113_140506"
    monkeypatch.setattr(current_module, "time", clock)
    question = Mock(side_effect=AssertionError("Automatic names must not replace existing files"))
    monkeypatch.setattr(QMessageBox, "question", question)
    device, scope, current, _ = _views(qtbot, monkeypatch, channel=channel)
    folder = tmp_path / "measurements"
    first_contents = None
    try:
        current.ui.cbRecordSums.setChecked(True)
        current.ui.editSumFile.setText(blank)
        assert not folder.exists()
        for suffix in ("", "_001"):
            capture = _Capture()
            scope._scope_dma.create_sum_capture.return_value = capture
            current.ui.btnStart.click()
            assert capture.entered.wait(1), current.status.text()
            current.stop_monitor_sync()
            expected = folder / f"ch{channel}_current_20261113_140506{suffix}.h5"
            assert capture.output == expected and expected.is_file()
            assert current.ui.editSumFile.text() == blank
            with h5py.File(expected, "r") as source:
                assert source.attrs["complete"]
            first_path = folder / f"ch{channel}_current_20261113_140506.h5"
            if first_contents is None:
                first_contents = first_path.read_bytes()
            assert first_path.read_bytes() == first_contents
        question.assert_not_called()
    finally:
        _close(device, scope, current)


@pytest.mark.parametrize(
    "filename,expected_name",
    [
        ("detector", "detector.h5"),
        ("detector.h5", "detector.h5"),
        ("detector.H5", "detector.H5"),
        ("nested/detector.hdf5", "nested/detector.hdf5"),
    ],
)
def test_named_sum_recording_uses_latest_measurement_folder(
    qtbot, monkeypatch, tmp_path, sum_save_settings, filename, expected_name,
):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    folder = tmp_path / "changed-folder"
    sum_save_settings.setValue("dma/save_folder", str(folder))
    try:
        current.ui.cbRecordSums.setChecked(True)
        current.ui.editSumFile.setText(filename)
        current.start_monitor()
        assert capture.entered.wait(1)
        current.stop_monitor_sync()
        assert capture.output == folder / expected_name
        assert capture.output.is_file()
        assert current.ui.editSumFile.text() == filename
        assert not (tmp_path / "measurements").exists()
    finally:
        _close(device, scope, current)


def test_explicit_sum_path_keeps_selected_location(qtbot, monkeypatch, tmp_path, sum_save_settings):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    selected = tmp_path / "elsewhere" / "custom.h5"
    try:
        current.ui.cbRecordSums.setChecked(True)
        current.ui.editSumFile.setText(str(selected))
        current.start_monitor()
        assert capture.entered.wait(1)
        current.stop_monitor_sync()
        assert capture.output == selected and selected.is_file()
        assert not (tmp_path / "measurements").exists()
    finally:
        _close(device, scope, current)


@pytest.mark.parametrize("selected", ["chosen", ""])
def test_sum_browse_starts_in_measurement_folder(
    qtbot, monkeypatch, tmp_path, sum_save_settings, selected,
):
    chosen = str(tmp_path / "elsewhere" / selected) if selected else ""
    dialog = Mock(return_value=(chosen, ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", dialog)
    device, scope, current, _ = _views(qtbot, monkeypatch)
    try:
        current.ui.btnSumFile.click()
        assert dialog.call_args.args[2] == str(tmp_path / "measurements")
        assert dialog.call_args.kwargs["options"] == QFileDialog.Option.DontConfirmOverwrite
        assert current.ui.cbRecordSums.isChecked() == bool(selected)
        assert current.ui.editSumFile.text() == (chosen + ".h5" if selected else "")
    finally:
        _close(device, scope, current)


def test_sum_monitor_without_recording_does_not_create_folder(
    qtbot, monkeypatch, tmp_path, sum_save_settings,
):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        current.start_monitor()
        assert capture.entered.wait(1)
        current.stop_monitor_sync()
        assert capture.output is None
        assert not (tmp_path / "measurements").exists()
    finally:
        _close(device, scope, current)


def test_sum_output_folder_failure_does_not_enter_capture(
    qtbot, monkeypatch, tmp_path, sum_save_settings,
):
    folder = tmp_path / "blocked-folder"
    folder.write_bytes(b"preserve existing file")
    sum_save_settings.setValue("dma/save_folder", str(folder))
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        current.ui.cbRecordSums.setChecked(True)
        current.start_monitor()
        qtbot.waitUntil(lambda: not current._is_running())
        assert not capture.entered.is_set()
        assert current._sum_worker.error
        assert current._sum_worker.thread is None
        assert scope._sum_worker is None
        assert current.ui.btnStart.isEnabled()
        assert not device.scope.get_enable()
        assert folder.read_bytes() == b"preserve existing file"
    finally:
        _close(device, scope, current)


def test_sum_output_folder_is_created_off_gui_thread(
    qtbot, monkeypatch, tmp_path, sum_save_settings,
):
    folder = tmp_path / "measurements"
    mkdir = Path.mkdir
    gui_thread = threading.get_ident()
    entered, release = threading.Event(), threading.Event()

    def held_mkdir(path, *args, **kwargs):
        if path == folder:
            assert threading.get_ident() != gui_thread
            entered.set()
            assert release.wait(5)
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", held_mkdir)
    device, scope, current, capture = _views(qtbot, monkeypatch)
    ticks = []
    heartbeat = QTimer(current)
    heartbeat.setInterval(10)
    heartbeat.timeout.connect(lambda: ticks.append(True))
    try:
        current.ui.cbRecordSums.setChecked(True)
        heartbeat.start()
        current.ui.btnStart.click()
        qtbot.waitUntil(entered.is_set)
        qtbot.waitUntil(lambda: len(ticks) >= 3)
        assert not capture.entered.is_set()
        release.set()
        assert capture.entered.wait(1)
        current.stop_monitor_sync()
        assert capture.output.is_file()
    finally:
        release.set()
        _close(device, scope, current)


def test_sums_use_existing_current_plot_and_restore_scope_after_stop(qtbot, monkeypatch):
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        assert [scope.ui.comboTriggerMode.itemData(i) for i in range(4)] == list(range(4))
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


@pytest.mark.parametrize("phase", ["open", "write", "close"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_sum_saving_keeps_gui_responsive_until_disk_finishes(
    qtbot, monkeypatch, tmp_path, phase, overwrite,
):
    entered, release = threading.Event(), threading.Event()
    read_first, stopped, dma_closed = threading.Event(), threading.Event(), threading.Event()
    gui_thread = threading.get_ident()
    io_threads = []

    def disk_operation(name):
        io_threads.append(threading.get_ident())
        assert threading.get_ident() != gui_thread
        if name == phase:
            entered.set()
            assert release.wait(10), "Timed out waiting to release the simulated slow disk"

    class HeldWriter(ScopeSumWriter):
        def __init__(self, *args, **kwargs):
            disk_operation("open")
            super().__init__(*args, **kwargs)

        def append_many(self, packets):
            disk_operation("write")
            super().append_many(packets)

        def close(self, **kwargs):
            disk_operation("close")
            super().close(**kwargs)

    monkeypatch.setattr(scope_sums, "ScopeSumWriter", HeldWriter)
    raw = bytearray(_packet().raw)
    struct.pack_into("<QQ", raw, 16, 16 | (508 << 48), 0)

    def read_frame():
        if not read_first.is_set():
            read_first.set()
            return bytes(raw)
        assert stopped.wait(10)
        raise OSError(errno.ETIMEDOUT, "Simulated terminal drain timeout")

    backend = Mock()
    backend.get_scope_dma_geometry.return_value = ScopeSumGeometry(16, 124, 0)
    backend.scope_sum_identity.return_value = {"uri": "synthetic GUI responsiveness test"}
    backend.scope_sum_status.return_value = (0, 0)
    backend.get_scope_dma_runtime_metadata.return_value = {}
    backend.read_dma_raw_frame.side_effect = read_frame
    backend.request_scope_sum_stop.side_effect = stopped.set
    backend.stop_scope_sum_from_reader.side_effect = stopped.set
    backend.close_scope_sum_capture.side_effect = dma_closed.set
    capture = ScopeSumCapture(backend, 0)
    device, scope, current, _ = _views(qtbot, monkeypatch, capture)
    path = tmp_path / "slow-recording.h5"
    if overwrite:
        path.write_bytes(b"original recording")
    question = Mock(return_value=QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "question", question)
    heartbeats = []
    heartbeat = QTimer(current)
    heartbeat.setInterval(10)
    heartbeat.timeout.connect(lambda: heartbeats.append(True))
    try:
        current.ui.spinSumSamples.setValue(16)
        current.ui.cbRecordSums.setChecked(True)
        current.ui.editSumFile.setText(str(path))
        heartbeat.start()
        current.ui.btnStart.click()
        if phase == "close":
            qtbot.waitUntil(read_first.is_set)
            current.ui.btnStop.click()
        qtbot.waitUntil(entered.is_set)
        ticks = len(heartbeats)
        qtbot.waitUntil(lambda: len(heartbeats) >= ticks + 5, timeout=2000)
        assert not release.is_set()
        assert current._is_running() and scope.measurement_active
        if phase == "open":
            backend.read_dma_raw_frame.assert_not_called()
            if overwrite:
                assert path.read_bytes() == b"original recording"
        if phase != "close":
            current.ui.btnStop.click()
        if phase != "open":
            qtbot.waitUntil(dma_closed.is_set)
        # The Stop slot and GUI timers keep working while the disk is still held.
        ticks = len(heartbeats)
        qtbot.waitUntil(lambda: len(heartbeats) >= ticks + 5, timeout=2000)
        assert current._sum_worker.thread.is_alive()
        assert not current.ui.btnStart.isEnabled()
        assert not current.ui.comboMode.isEnabled()
        assert not scope.ui.btnStart.isEnabled()
        assert not current.ui.btnStop.isEnabled()
        assert question.call_count == int(overwrite)
        release.set()
        qtbot.waitUntil(lambda: not current._is_running())
        assert current._sum_worker.thread is None
        assert current.ui.btnStart.isEnabled() and scope.ui.btnStart.isEnabled()
        assert not current._sum_worker.error
        assert len(set(io_threads)) == 1
        with h5py.File(path, "r") as source:
            assert bool(source.attrs["complete"]) == (phase != "open")
            assert len(source["packets"]) == (0 if phase == "open" else 1)
    finally:
        heartbeat.stop()
        release.set()
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


@pytest.mark.parametrize(
    "answer", [QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.No],
)
@pytest.mark.parametrize("relative", [False, True])
def test_declining_sum_overwrite_preserves_file_and_plot(
    qtbot, monkeypatch, tmp_path, sum_save_settings, answer, relative,
):
    path = tmp_path / "existing.h5"
    path.write_bytes(b"original measurement")
    sum_save_settings.setValue("dma/save_folder", str(tmp_path))
    question = Mock(return_value=answer)
    monkeypatch.setattr(QMessageBox, "question", question)
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        current.ui.cbRecordSums.setChecked(True)
        current.ui.editSumFile.setText(path.stem if relative else str(path))
        current._curve.setData([0, 1], [2, 3])
        trigger_mode = device.scope.get_trigger_mode()
        current.start_monitor()
        question.assert_called_once()
        assert str(path) in question.call_args.args[2]
        assert question.call_args.args[-1] == QMessageBox.StandardButton.Cancel
        assert path.read_bytes() == b"original measurement"
        assert current._curve.yData[-1] == 3
        assert not capture.entered.is_set()
        assert not current._is_running()
        assert scope._sum_previous_settings is None
        assert device.scope.get_trigger_mode() == trigger_mode
        assert current.ui.btnStart.isEnabled()
    finally:
        _close(device, scope, current)


def test_sum_overwrite_requires_confirmation_for_each_start(qtbot, monkeypatch, tmp_path):
    path = tmp_path / "existing.h5"
    path.write_bytes(b"original measurement")
    question = Mock(side_effect=[QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.Cancel])
    monkeypatch.setattr(QMessageBox, "question", question)
    device, scope, current, capture = _views(qtbot, monkeypatch)
    run = Mock(wraps=capture.run)
    monkeypatch.setattr(capture, "run", run)
    try:
        current.ui.cbRecordSums.setChecked(True)
        current.ui.editSumFile.setText(str(path))
        current.start_monitor()
        assert capture.entered.wait(1)
        # Confirmation does not delete the file; only the writer may replace it.
        assert path.read_bytes() == b"original measurement"
        current.stop_monitor_sync()
        run.assert_called_once_with(path, overwrite=True)
        with h5py.File(path, "r") as source:
            assert source.attrs["complete"]
            assert source["packets"].shape == (1, 4096)
        recorded = path.read_bytes()
        current.start_monitor()
        assert question.call_count == 2
        assert run.call_count == 1
        assert not current._is_running()
        assert path.read_bytes() == recorded
    finally:
        _close(device, scope, current)


@pytest.mark.parametrize("record", [False, True])
def test_sum_start_without_file_conflict_does_not_prompt(qtbot, monkeypatch, tmp_path, record):
    path = tmp_path / "sums.h5"
    if not record:
        path.write_bytes(b"existing file with recording disabled")
    question = Mock(side_effect=AssertionError("No overwrite confirmation is needed"))
    monkeypatch.setattr(QMessageBox, "question", question)
    device, scope, current, capture = _views(qtbot, monkeypatch)
    try:
        current.ui.cbRecordSums.setChecked(record)
        current.ui.editSumFile.setText(str(path))
        current.start_monitor()
        assert capture.entered.wait(1)
        current.stop_monitor_sync()
        question.assert_not_called()
        assert capture.output == (path if record else None)
        if record:
            with h5py.File(path, "r") as source:
                assert source.attrs["complete"]
        else:
            assert path.read_bytes() == b"existing file with recording disabled"
    finally:
        _close(device, scope, current)


def test_replay_started_during_sum_confirmation_blocks_recording(qtbot, monkeypatch, tmp_path):
    path = tmp_path / "existing.h5"
    path.write_bytes(b"original measurement")
    device, scope, current, capture = _views(qtbot, monkeypatch)

    def confirm(*_args):
        current.set_replay_hardware_interlock(True)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", confirm)
    try:
        current.ui.cbRecordSums.setChecked(True)
        current.ui.editSumFile.setText(str(path))
        current.start_monitor()
        assert path.read_bytes() == b"original measurement"
        assert not capture.entered.is_set()
        assert not current._is_running()
        assert scope._sum_previous_settings is None
    finally:
        _close(device, scope, current)


def test_scope_busy_after_sum_confirmation_preserves_existing_file(qtbot, monkeypatch, tmp_path):
    path = tmp_path / "existing.h5"
    path.write_bytes(b"original measurement")
    device, scope, current, capture = _views(qtbot, monkeypatch)
    monkeypatch.setattr(QMessageBox, "question", Mock(return_value=QMessageBox.StandardButton.Yes))
    try:
        current.ui.cbRecordSums.setChecked(True)
        current.ui.editSumFile.setText(str(path))
        device.scope.set_enable(True)
        current.start_monitor()
        assert "stop the active Scope acquisition" in current.status.text()
        assert path.read_bytes() == b"original measurement"
        assert not capture.entered.is_set()
        assert not current._is_running()
    finally:
        device.scope.set_enable(False)
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
