from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtWidgets import QDialog, QMessageBox
from pytestqt.qtbot import QtBot

from nlab.analysis.waveform_mca_gates import WaveformGateSource
from nlab.app import MainAppWindow
from nlab.controllers.main_window_controller import MainWindowController
from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.dma import FILE_HEADER_STRUCT, FILE_MAGIC, FILE_VERSION
from nlab.hardware.digitizer.mca import McaGateTiming, MultiChannelAnalyzer
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.help.context import resolve_context
from nlab.views.waveform_mca_transfer_dialog import McaGateTransferTarget, WaveformMcaTransferDialog


def test_mapping_uses_baseline_end_to_green_gap_and_upward_steps() -> None:
    source = WaveformGateSource(7, 10, 19, 47, 2.5)
    proposal = source.propose(True)
    assert proposal.desired_ns == (7.5, 92.5, 22.5)
    assert proposal.timing == McaGateTiming(8, 94, 24, True)
    assert proposal.boundaries_ns == (25.5, 49.5, 119.5)
    assert source == WaveformGateSource(7, 10, 19, 47, 2.5)  # No mutation of offline gates.
    # Exact decimal steps must not acquire a spurious extra step from floating arithmetic.
    assert WaveformGateSource(0, 0, 20, 40, .1).propose(False).timing == McaGateTiming(
        0, 4, 2, False,
    )
    assert WaveformGateSource(0, 0, 1, 2, 2.000001).propose(True).timing.cc_time_ns == 4


def test_changing_baseline_end_changes_only_pretrigger_and_zero_gap_is_valid() -> None:
    first = WaveformGateSource(5, 10, 20, 40, 8).propose(True)
    second = WaveformGateSource(7, 10, 20, 40, 8).propose(True)
    zero = WaveformGateSource(10, 10, 20, 40, 8).propose(True)
    assert (first.timing.pretrigger_ns, second.timing.pretrigger_ns, zero.timing.pretrigger_ns) == (
        40, 24, 0,
    )
    assert first.timing.frame_ns == second.timing.frame_ns == zero.timing.frame_ns == 240
    assert first.timing.cc_time_ns == second.timing.cc_time_ns == zero.timing.cc_time_ns == 80
    assert first.boundaries_ns == second.boundaries_ns == zero.boundaries_ns == (80, 160, 320)


def test_driver_range_boundaries_are_not_double_converted() -> None:
    assert WaveformGateSource(0, 2047, 3047, 6142, 2).propose(True).timing == McaGateTiming(
        4094, 8190, 2000, True,
    )
    assert WaveformGateSource(0, 0, 1, 1, 2).propose(False).timing == McaGateTiming(0, 2, 2, False)
    # CC_TIME's driver maximum is 65534 ns, but a transferable short gate must fit in the window.
    McaGateTiming(0, 8190, 8190, True).validate()


@pytest.mark.parametrize("source", [
    WaveformGateSource(0, 0, 1, 2, float("nan")),
    WaveformGateSource(0, 0, 1, 2, 0),
    WaveformGateSource(0, 0, 1, 2, float("inf")),
    WaveformGateSource(11, 10, 20, 40, 8),  # Baseline ends after integration start.
    WaveformGateSource(-1, 10, 20, 40, 8),
    WaveformGateSource(0, 2048, 2049, 5000, 2),  # Pretrigger exceeds driver range.
    WaveformGateSource(0, 0, 1, 4096, 2),
    WaveformGateSource(0, 0, 2, 1, 2),
    WaveformGateSource(0, 2, 3, 4, 2),  # Pretrigger is not shorter than the window.
])
def test_invalid_mapping_is_rejected_without_clipping(source: WaveformGateSource) -> None:
    with pytest.raises(ValueError):
        source.propose(True)


def _backend() -> Mock:
    backend = Mock()
    values: dict[str, int | bool] = {"pre": 24, "frame": 256, "cc": 8, "enable": False}
    for getter, setter, key in (
        ("get_dpp_pretrigger_samples", "set_dpp_pretrigger_samples", "pre"),
        ("get_dpp_frame_samples", "set_dpp_frame_samples", "frame"),
        ("get_cc_time", "set_cc_time", "cc"),
        ("get_cc_enable", "set_cc_enable", "enable"),
    ):
        getattr(backend, getter).side_effect = lambda key=key: values[key]
        getattr(backend, setter).side_effect = lambda value, key=key: values.__setitem__(key, value)
    backend.get_global_enable.return_value = False
    backend.get_measurement_in_progress.return_value = False
    backend.mca_dma_active.return_value = False
    return backend


def test_idle_batch_writes_physical_ns_only_and_never_starts_acquisition() -> None:
    backend = _backend()
    mca = MultiChannelAnalyzer(backend)
    candidate = McaGateTiming(32, 500, 50, True)
    mca.apply_gate_timing(candidate)
    assert mca.gate_timing() == candidate
    backend.set_dpp_pretrigger_samples.assert_called_once_with(32)
    backend.set_dpp_frame_samples.assert_called_once_with(500)
    backend.set_cc_time.assert_called_once_with(50)
    backend.set_cc_enable.assert_called_once_with(True)
    backend.set_global_enable.assert_not_called()


@pytest.mark.parametrize("timing", [McaGateTiming(0, 2, 2, False),
                                   McaGateTiming(4094, 8190, 8190, True)])
def test_idle_batch_accepts_driver_boundaries_unchanged(timing: McaGateTiming) -> None:
    backend = _backend()
    mca = MultiChannelAnalyzer(backend)
    mca.apply_gate_timing(timing)
    assert mca.gate_timing() == timing
    backend.set_global_enable.assert_not_called()


def test_snapshot_failure_aborts_before_any_write() -> None:
    backend = _backend()
    backend.get_dpp_frame_samples.side_effect = OSError("read unavailable")
    with pytest.raises(OSError, match="read unavailable"):
        MultiChannelAnalyzer(backend).apply_gate_timing(McaGateTiming(32, 500, 50, True))
    assert not any(call[0].startswith("set_") for call in backend.mock_calls)


@pytest.mark.parametrize("busy", [
    "get_global_enable", "get_measurement_in_progress", "mca_dma_active",
])
def test_hardware_busy_checks_perform_no_writes(busy: str) -> None:
    backend = _backend()
    getattr(backend, busy).return_value = True
    with pytest.raises(RuntimeError, match="Stop MCA"):
        MultiChannelAnalyzer(backend).apply_gate_timing(McaGateTiming(32, 500, 50, True))
    assert not any(call[0].startswith("set_") for call in backend.mock_calls)


@pytest.mark.parametrize("field", ["set_dpp_pretrigger_samples", "set_dpp_frame_samples",
                                    "set_cc_time", "set_cc_enable"])
def test_each_partial_write_failure_restores_previous_settings(field: str) -> None:
    backend = _backend()
    mca = MultiChannelAnalyzer(backend)
    before = mca.gate_timing()
    setter = getattr(backend, field)
    write = setter.side_effect

    def fail_first(value: int | bool) -> None:
        if setter.call_count == 1:
            raise OSError("simulated write failure")
        write(value)

    setter.side_effect = fail_first
    with pytest.raises(RuntimeError, match="previous settings restored"):
        mca.apply_gate_timing(McaGateTiming(32, 500, 50, True))
    assert mca.gate_timing() == before
    backend.set_global_enable.assert_not_called()


def test_rollback_failure_is_reported_and_remaining_restoration_is_attempted() -> None:
    backend = _backend()
    backend.set_dpp_frame_samples.side_effect = OSError("transport unavailable")
    with pytest.raises(RuntimeError, match="rollback incomplete"):
        MultiChannelAnalyzer(backend).apply_gate_timing(McaGateTiming(32, 500, 50, True))
    assert backend.set_cc_time.call_count == 1
    backend.set_cc_enable.assert_called_once_with(False)
    backend.set_global_enable.assert_not_called()


@pytest.mark.parametrize("timing", [
    McaGateTiming(1, 256, 8, True), McaGateTiming(0, 8192, 8, True),
    McaGateTiming(0, 256, 258, True), McaGateTiming(0, 256, 0, True),
])
def test_invalid_presets_never_read_or_write_backend(timing: McaGateTiming) -> None:
    backend = _backend()
    with pytest.raises(ValueError):
        MultiChannelAnalyzer(backend).apply_gate_timing(timing)
    assert backend.mock_calls == []


@pytest.mark.parametrize("flag", ["_connection_lost", "_replay_hardware_interlocked",
                                  "_session_replay_active", "_worker", "_dma_worker",
                                  "_worker_thread", "_dma_thread", "_coincidence_session",
                                  "_coincidence_stopping", "_measurement_active"])
def test_controller_interlocks_block_transfer(qtbot: QtBot, flag: str) -> None:
    device = create_offline_digitizer(0)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    before = device.mca.gate_timing()
    previous = getattr(view, flag)
    setattr(view, flag, True)
    try:
        with pytest.raises(RuntimeError, match="busy, disconnected"):
            view.apply_gate_timing(McaGateTiming(32, 500, 50, True))
        assert device.mca.gate_timing() == before
    finally:
        setattr(view, flag, previous)


def test_controller_refreshes_both_sliders_without_duplicate_hardware_writes(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = create_offline_digitizer(0)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    before = device.mca.get_trigger_level(), device.mca.get_baseline_window()
    writes = []
    for name in ("set_pretrigger_samples", "set_frame_samples"):
        spy = Mock(wraps=getattr(device.mca, name))
        monkeypatch.setattr(device.mca, name, spy)
        writes.append(spy)
    view.apply_gate_timing(McaGateTiming(32, 500, 50, True))
    assert [spy.call_count for spy in writes] == [1, 1]
    assert view.ui.spinPretrigger.value() == view.ui.sliderPretrigger.value() == 32
    assert view.ui.spinFrameSamples.value() == view.ui.sliderFrameSamples.value() == 500
    assert view.ui.spinCcTime.value() == 50 and view.ui.cbCcEnable.isChecked()
    assert (device.mca.get_trigger_level(), device.mca.get_baseline_window()) == before
    assert not device.mca.get_global_enable()


def _target(label: str = "D1:CH0") -> McaGateTransferTarget:
    return McaGateTransferTarget(label, Mock(return_value=McaGateTiming(24, 256, 8, False)), Mock())


def test_dialog_previews_baseline_gap_ceilings_and_targets_selected_channel(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [_target(), _target("D2:CH1")]
    dialog = WaveformMcaTransferDialog(WaveformGateSource(7, 10, 19, 47, 2.5), targets)
    qtbot.addWidget(dialog)
    previews = []
    dialog.preview_changed.connect(previews.append)
    dialog.refresh()
    assert previews[-1].boundaries_ns == (25.5, 49.5, 119.5)
    assert dialog.table.item(0, 2).text() == "7.5"
    assert dialog.table.item(0, 3).text() == "8"
    assert resolve_context(dialog.target_combo).topic == "waveform-mca-transfer"
    assert not hasattr(dialog, "trigger_position")
    assert not hasattr(dialog, "trigger_confirmed")
    assert dialog.apply_button.isEnabled()
    dialog.target_combo.setCurrentIndex(1)
    dialog.enable_cc.setChecked(False)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog.apply_button.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    targets[0].apply.assert_not_called()
    targets[1].apply.assert_called_once_with(McaGateTiming(8, 94, 24, False))


def test_dialog_cancel_invalid_range_and_missing_targets_never_write(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    dialog = WaveformMcaTransferDialog(WaveformGateSource(7, 10, 20, 40, 8), [target])
    qtbot.addWidget(dialog)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    dialog.apply_button.click()
    target.apply.assert_not_called()
    invalid = WaveformMcaTransferDialog(WaveformGateSource(11, 10, 20, 40, 8), [target])
    qtbot.addWidget(invalid)
    assert not invalid.apply_button.isEnabled() and invalid.proposal is None
    empty = WaveformMcaTransferDialog(WaveformGateSource(7, 10, 20, 40, 8), [])
    qtbot.addWidget(empty)
    assert not empty.apply_button.isEnabled()
    assert "No live MCA" in empty.status.text()
    assert empty.proposal is not None  # Offline preview remains useful, but cannot write.
    assert empty.table.item(1, 1).text() == "—"
    assert empty.table.item(1, 3).text() == "240"


def test_dialog_reports_failure_without_silent_success(qtbot: QtBot,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    target = _target()
    target.apply.side_effect = RuntimeError("Target became busy")
    dialog = WaveformMcaTransferDialog(WaveformGateSource(7, 10, 20, 40, 8), [target])
    qtbot.addWidget(dialog)
    warning = Mock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog.apply_button.click()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "Target became busy" in dialog.status.text()
    warning.assert_called_once()


def test_target_callbacks_recheck_offline_replay_reconnection_and_scope() -> None:
    controller = object.__new__(MainWindowController)
    controller._offline = False
    controller._replay_hardware_interlocked = False
    target = SimpleNamespace(channel=0, channel_label="D1:CH0", gate_transfer_current=Mock(),
                             apply_gate_timing=Mock())
    controller._mca_controllers = [target]
    controller._scope_controllers = []
    option = controller._waveform_mca_targets()[0]
    controller._offline = True
    assert controller._waveform_mca_targets() == []
    with pytest.raises(RuntimeError, match="offline"):
        option.apply(McaGateTiming(0, 100, 20, True))
    controller._offline = False
    controller._replay_hardware_interlocked = True
    assert controller._waveform_mca_targets() == []
    with pytest.raises(RuntimeError, match="replay"):
        option.read()
    controller._replay_hardware_interlocked = False
    controller._mca_controllers = []
    with pytest.raises(RuntimeError, match="no longer connected"):
        option.read()
    controller._mca_controllers = [target]
    controller._scope_controllers = [SimpleNamespace(channel=0, measurement_active=True)]
    with pytest.raises(RuntimeError, match="Stop Scope"):
        option.apply(McaGateTiming(0, 100, 20, True))
    target.apply_gate_timing.assert_not_called()


def test_replay_started_during_confirmation_prevents_any_write(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = object.__new__(MainWindowController)
    controller._offline = controller._replay_hardware_interlocked = False
    target = SimpleNamespace(channel=0, channel_label="D1:CH0", apply_gate_timing=Mock(),
                             gate_transfer_current=Mock(
                                 return_value=McaGateTiming(24, 256, 8, False),
                             ))
    controller._mca_controllers, controller._scope_controllers = [target], []
    dialog = WaveformMcaTransferDialog(WaveformGateSource(7, 10, 20, 40, 8),
                                      controller._waveform_mca_targets())
    qtbot.addWidget(dialog)

    def begin_replay(*args: object, **kwargs: object) -> QMessageBox.StandardButton:
        controller._replay_hardware_interlocked = True
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", begin_replay)
    monkeypatch.setattr(QMessageBox, "warning", Mock())
    dialog.apply_button.click()
    target.apply_gate_timing.assert_not_called()
    assert "Transfer failed" in dialog.status.text() and "replay" in dialog.status.text()


@pytest.mark.parametrize("apply_preset", [True, False])
def test_workbench_file_menu_routes_transfer_and_removes_only_preview_markers(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apply_preset: bool,
) -> None:
    for name in ("_apply_stored_general_settings", "_persist_current_general_settings",
                 "_apply_view_state", "_save_developer_settings"):
        monkeypatch.setattr(MainAppWindow, name, lambda _self: None)
    window = MainAppWindow(channels=2, offline=True)
    qtbot.addWidget(window)
    controller = window._controller
    assert controller._waveform_mca_targets() == []
    # Simulated routing through in-memory transports; no IIO connections/writes.
    controller._offline = False
    controller._set_replay_hardware_interlock(False, resume_background=False)
    controller.show_waveform_analysis()
    workbench = controller._waveform_analysis_dialog
    assert not workbench.transfer_gates_action.isEnabled()
    path = tmp_path / "source.bin"
    samples = np.full(100, 1000, dtype="<i2")
    samples[25:35] -= 100
    path.write_bytes(FILE_HEADER_STRUCT.pack(FILE_MAGIC, FILE_VERSION, 0, 0, 0., 104)
                     + struct.pack("<Q", 1) + samples.tobytes())
    workbench.open_path(path)
    qtbot.waitUntil(lambda: workbench._index_thread is None)
    workbench.auto_recalculate.setChecked(False)
    assert workbench.transfer_gates_action.isEnabled()
    original = workbench._gate_indices()
    items = set(workbench.waveform_plot.items)
    before = controller._mca_controllers[0]._mca.gate_timing()
    timing = None

    def confirm(dialog: WaveformMcaTransferDialog) -> int:
        nonlocal timing
        assert len(set(workbench.waveform_plot.items) - items) == 3
        dialog.target_combo.setCurrentIndex(1)
        assert dialog.apply_button.isEnabled(), dialog.status.text()
        timing = dialog.proposal.timing
        assert dialog.source.baseline_end == original[1]
        assert timing.pretrigger_ns == (original[2] - original[1]) * 8
        if not apply_preset:
            return QDialog.DialogCode.Rejected
        dialog.apply_button.click()
        return dialog.result()

    monkeypatch.setattr(WaveformMcaTransferDialog, "exec", confirm)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    warning = Mock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    workbench.transfer_gates_action.trigger()
    assert not warning.called, warning.call_args
    if apply_preset:
        assert "transferred" in workbench.status.text()
    assert set(workbench.waveform_plot.items) == items
    assert workbench._gate_indices() == original
    assert controller._mca_controllers[0]._mca.gate_timing() == before
    assert controller._mca_controllers[1]._mca.gate_timing() == (timing if apply_preset else before)
    controller._offline = True
    workbench.close_file_action.trigger()
    assert not workbench.transfer_gates_action.isEnabled()
    window.close()
