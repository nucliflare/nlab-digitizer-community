"""Non-DMA GUI reconfiguration with simulated ABI2 hardware, without live I/O."""

from unittest.mock import Mock

import numpy as np
import pytest

from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.offline import create_offline_digitizer


@pytest.fixture
def running_panel(qtbot, monkeypatch):
    device = create_offline_digitizer(0)
    panel = MCAController(device.mca)
    qtbot.addWidget(panel)
    # Exercise the real Start/UI path without starting a polling thread.
    monkeypatch.setattr(panel, "_start_worker", lambda: setattr(panel, "_worker", Mock()))
    panel.ui.cbDmaEnable.setChecked(False)
    panel._on_start()
    yield panel, device.mca
    panel._worker = None
    panel._dma_worker = None
    panel._dma_thread = None
    device.close()


def parameter_widgets(panel):
    return (
        panel.ui.comboPulsePolarity,
        panel.ui.comboBaseline,
        panel.ui.spinPileupWindow,
        panel.ui.comboBinning,
        panel.ui.cbExtTrigger,
        panel.ui.groupSignal,
        panel.ui.tabFilters,
        panel.ui.spinTimeLimit,
        panel.ui.spinRefreshRate,
    )


def test_polling_start_keeps_parameters_and_markers_editable(running_panel):
    panel, mca = running_panel
    assert mca.parameter_contract()["parameter_abi_version"] == 2
    assert mca.get_global_enable() and panel.measurement_active
    assert all(widget.isEnabled() for widget in parameter_widgets(panel))
    assert panel._debug_threshold_line.movable
    assert panel._debug_pretrigger_line.movable
    assert not panel.ui.cbDmaEnable.isEnabled()
    assert not panel.ui.btnStart.isEnabled()
    assert panel.ui.btnStop.isEnabled()


@pytest.mark.parametrize("control,backend_method,display,native", [
    ("spinFrameSamples", "set_dpp_frame_samples", 512, 256),
    ("spinTriggerLevel", "set_dpp_trigger_level", -123, -123),
    ("spinTrapR", "set_trapez_R", 72, 36),
])
def test_running_edit_stops_before_write_and_restarts_fresh(
    running_panel, monkeypatch, control, backend_method, display, native
):
    panel, mca = running_panel
    backend = mca._b
    events = []
    set_enable = backend.set_global_enable
    setter = getattr(backend, backend_method)

    def enable(value):
        events.append(("enable", value))
        set_enable(value)

    def write(value):
        assert not mca.get_global_enable(), "driver requires stopped measurement writes"
        events.append(("write", value))
        setter(value)

    monkeypatch.setattr(backend, "set_global_enable", enable)
    monkeypatch.setattr(backend, backend_method, write)
    panel._last_histogram = np.array([1, 2], dtype=np.uint32)
    panel._display_histogram = panel._last_histogram
    panel._differential_reference_histogram = panel._last_histogram
    panel._last_elapsed_s = 20.0
    worker = panel._worker
    widget = getattr(panel.ui, control)
    widget.setValue(display)
    assert events == []
    widget.editingFinished.emit()

    assert events == [("enable", False), ("write", native), ("enable", True)]
    assert panel._worker is worker
    assert mca.get_global_enable() and panel.measurement_active
    assert panel._last_histogram is None
    assert panel._display_histogram is None
    assert panel._differential_reference_histogram is None
    assert panel._last_elapsed_s == 0
    assert all(widget.isEnabled() for widget in parameter_widgets(panel))
    assert not panel.ui.cbDmaEnable.isEnabled()


def test_running_marker_edit_uses_the_same_stopped_write_path(running_panel, monkeypatch):
    panel, mca = running_panel
    backend = mca._b
    set_pretrigger = backend.set_dpp_pretrigger_samples
    states = []

    def write(value):
        states.append((mca.get_global_enable(), value))
        set_pretrigger(value)

    monkeypatch.setattr(backend, "set_dpp_pretrigger_samples", write)
    line = panel._debug_pretrigger_line
    line.moving = True
    line.setValue(80)
    line.moving = False
    line.sigPositionChangeFinished.emit(line)

    assert states == [(False, 40)]
    assert mca.get_global_enable()
    assert panel.ui.spinPretrigger.value() == 80


@pytest.mark.parametrize("owner", ["_dma_worker", "_dma_thread"])
def test_dma_arming_and_draining_lock_ui_and_programmatic_writes(
    running_panel, monkeypatch, owner
):
    panel, mca = running_panel
    setattr(panel, owner, object())
    panel._set_controls_enabled(True)
    assert all(not widget.isEnabled() for widget in parameter_widgets(panel))
    assert not panel._debug_threshold_line.movable
    assert not panel._debug_pretrigger_line.movable
    assert panel.ui.comboDebug1.isEnabled() and panel.ui.comboDebug2.isEnabled()
    setter = Mock(wraps=mca.set_frame_samples)
    monkeypatch.setattr(mca, "set_frame_samples", setter)
    panel.ui.spinFrameSamples.setValue(512)
    panel.ui.spinFrameSamples.editingFinished.emit()
    setter.assert_not_called()
    assert mca.get_global_enable()

    setattr(panel, owner, None)
    panel._set_controls_enabled(True)
    assert all(widget.isEnabled() for widget in parameter_widgets(panel))
    assert not panel.ui.cbDmaEnable.isEnabled()


def test_driver_owned_dma_blocks_edits_without_local_dma_worker(running_panel, monkeypatch):
    panel, mca = running_panel
    monkeypatch.setattr(mca._b, "mca_dma_active", lambda: True)
    setter = Mock(wraps=mca.set_frame_samples)
    monkeypatch.setattr(mca, "set_frame_samples", setter)
    panel.ui.spinFrameSamples.setValue(512)
    panel.ui.spinFrameSamples.editingFinished.emit()
    setter.assert_not_called()
    assert mca.get_global_enable()
    assert panel.ui.spinFrameSamples.value() == 256
    assert "Stop list-mode DMA" in panel.status.text()


@pytest.mark.parametrize("interlock", ["_replay_hardware_interlocked", "_connection_lost"])
def test_replay_and_connection_loss_still_block_edits(running_panel, monkeypatch, interlock):
    panel, mca = running_panel
    setattr(panel, interlock, True)
    panel._set_controls_enabled(True)
    assert all(not widget.isEnabled() for widget in parameter_widgets(panel))
    assert not panel.ui.comboDebug1.isEnabled()
    setter = Mock(wraps=mca.set_frame_samples)
    monkeypatch.setattr(mca, "set_frame_samples", setter)
    panel.ui.spinFrameSamples.setValue(512)
    panel.ui.spinFrameSamples.editingFinished.emit()
    setter.assert_not_called()
    assert mca.get_global_enable()


def test_rejected_running_edit_stays_stopped_and_restores_retryable_ui(
    running_panel, monkeypatch
):
    panel, mca = running_panel
    worker = panel._worker
    enable = Mock(wraps=mca._b.set_global_enable)
    monkeypatch.setattr(mca._b, "set_global_enable", enable)

    def fail(value):
        assert not mca.get_global_enable()
        raise ValueError("injected rejected setting")

    monkeypatch.setattr(mca, "set_frame_samples", fail)
    panel.ui.spinFrameSamples.setValue(512)
    panel.ui.spinFrameSamples.editingFinished.emit()

    assert not mca.get_global_enable()
    assert not panel.measurement_active
    assert all(call.args != (True,) for call in enable.call_args_list)
    worker.request_stop.emit.assert_called_once_with()
    assert panel.ui.btnStart.isEnabled()
    assert not panel.ui.btnStop.isEnabled()
    assert panel.ui.cbDmaEnable.isEnabled()
    assert panel.ui.spinFrameSamples.value() == 256
    assert "Setting not applied: injected rejected setting" in panel.status.text()
