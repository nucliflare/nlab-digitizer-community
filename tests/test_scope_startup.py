"""Scope startup against driver-derived guards; no live hardware implied."""

import errno
from unittest.mock import Mock

import pytest

from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.hardware.digitizer.scope import TriggerMode


def _guarded_device(monkeypatch, mode, *, enabled=False):
    device = create_offline_digitizer(0)
    scope = device.scope
    backend = device._backend
    monkeypatch.setattr(backend, "get_scope_trigger_modes", lambda: tuple(range(6)))
    scope.set_pretrigger_samples(0)
    scope.set_trigger_mode(mode)
    scope.set_enable(enabled)
    set_edge, set_pretrigger = backend.set_edge, backend.set_pretrigger_samples

    def write_mode(value):
        # vdpp-scope.c: trigger_mode_store() refuses armed sum transitions.
        if scope.get_enable() and (scope.get_trigger_mode() == 5 or value == 5):
            raise OSError(errno.EBUSY, "sum mode is armed")
        set_edge(value)

    def write_pretrigger(value):
        # vdpp-scope.c: pretrigger_samples_store() requires zero in mode 5.
        if scope.get_trigger_mode() == 5 and value:
            raise OSError(errno.EINVAL, "sum pretrigger must be zero")
        set_pretrigger(value)

    monkeypatch.setattr(backend, "set_edge", write_mode)
    monkeypatch.setattr(backend, "set_pretrigger_samples", write_pretrigger)
    return device


@pytest.mark.parametrize("mode", [TriggerMode.PERIODIC_SUM, TriggerMode.TIMED])
@pytest.mark.parametrize("enabled", [False, True])
def test_startup_leaves_previous_mode_before_applying_waveform_defaults(
    qtbot, monkeypatch, mode, enabled
):
    device = _guarded_device(monkeypatch, mode, enabled=enabled)
    view = None
    try:
        view = ScopeController(device.scope)
        qtbot.addWidget(view)
        assert not device.scope.get_enable()
        assert device.scope.get_trigger_mode() == TriggerMode.ANY_BELOW
        assert device.scope.get_pretrigger_samples() == 32
        assert view.ui.comboTriggerMode.currentData() == int(TriggerMode.ANY_BELOW)
        assert view.ui.spinPretrigger.value() == 64
    finally:
        if view is not None:
            view.stop_dma_sync()
            view.close_viewer_client()
        device.close()


def test_reset_defaults_can_leave_stopped_sum_mode(qtbot, monkeypatch):
    device = _guarded_device(monkeypatch, TriggerMode.ANY_BELOW)
    view = ScopeController(device.scope)
    qtbot.addWidget(view)
    try:
        device.scope.set_pretrigger_samples(0)
        device.scope.set_trigger_mode(TriggerMode.PERIODIC_SUM)
        view.reset_defaults()
        assert device.scope.get_trigger_mode() == TriggerMode.ANY_BELOW
        assert device.scope.get_pretrigger_samples() == 32
        assert not device.scope.get_enable()
    finally:
        view.stop_dma_sync()
        view.close_viewer_client()
        device.close()


def test_startup_does_not_modify_a_channel_owned_by_dma(qtbot, monkeypatch):
    device = _guarded_device(monkeypatch, TriggerMode.PERIODIC_SUM, enabled=True)
    writes = Mock()
    for name in (
        "set_enable", "set_edge", "set_pretrigger_samples", "set_frame_samples",
        "set_frame_period_cycles", "set_dac_value", "set_trigger_level",
    ):
        monkeypatch.setattr(device._backend, name, getattr(writes, name))
    monkeypatch.setattr(device._backend, "scope_dma_active", lambda: True)
    try:
        with pytest.raises(RuntimeError, match="stop.*Scope DMA"):
            ScopeController(device.scope)
        assert writes.mock_calls == []
    finally:
        device.close()
