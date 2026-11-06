"""Read-only 1/R ABI and UI behavior using fake IIO and offline devices."""

from __future__ import annotations

import errno
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QLabel
from pytestqt.qtbot import QtBot

from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.backends.base import MCABackend
from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.mca import MCAParam, MultiChannelAnalyzer
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.help.context import resolve_context
from nlab.utils.settings_io import apply_mca_hardware, collect_channel_hardware


@pytest.mark.parametrize("raw", [0, 64, 4681, 16384, 32768])
def test_read_only_iio_attribute_preserves_actual_driver_value(raw: int) -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._pp = SimpleNamespace(attrs={"trapezoid_1r_raw": SimpleNamespace(value=str(raw))})
    trapezoid = MultiChannelAnalyzer(backend).filters.trapezoid
    assert trapezoid.supports_reciprocal_raw()
    assert trapezoid.get_reciprocal_raw() == raw
    assert not hasattr(trapezoid, "set_reciprocal_raw")
    assert not hasattr(backend, "set_trapezoid_reciprocal_raw")


def test_old_driver_has_no_calculated_readback_fallback() -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._pp = SimpleNamespace(attrs={"trapezoid_r": SimpleNamespace(value="64")})
    trapezoid = MultiChannelAnalyzer(backend).filters.trapezoid
    assert not trapezoid.supports_reciprocal_raw()
    with pytest.raises(NotImplementedError):
        trapezoid.get_reciprocal_raw()
    with pytest.raises(NotImplementedError, match="update vdpp-mca"):
        backend.get_trapezoid_reciprocal_raw()
    backend._pp = None
    assert not trapezoid.supports_reciprocal_raw()
    assert MCABackend.supports_trapezoid_reciprocal_raw(backend) is False


@pytest.mark.parametrize("rise,expected", [(16, 32768), (24, 16384), (64, 4681), (4088, 64)])
def test_offline_register_dependency_matches_documented_driver_mapping(
    rise: int, expected: int,
) -> None:
    device = create_offline_digitizer(0)
    trapezoid = device.mca.filters.trapezoid
    assert trapezoid.get_reciprocal_raw() == 0  # Probe/reset state, before R is applied.
    trapezoid.set_R(rise)
    assert trapezoid.get_R() == rise
    assert trapezoid.get_reciprocal_raw() == expected
    snapshot = {item["id"]: item["value"] for item in device.mca.get_settings()}
    assert snapshot[int(MCAParam.TRAPEZ_RECIPROCAL)] == expected
    hardware = collect_channel_hardware(device.scope, device.mca, None)
    assert "reciprocal_raw" not in hardware["mca"]["trapezoid"]


@pytest.mark.parametrize("supported", [False, True])
def test_gui_loads_read_only_value_and_supports_old_drivers(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, supported: bool,
) -> None:
    device = create_offline_digitizer(0)
    monkeypatch.setattr(device._backend, "supports_trapezoid_reciprocal_raw", lambda: supported)
    read = Mock(wraps=device._backend.get_trapezoid_reciprocal_raw)
    monkeypatch.setattr(device._backend, "get_trapezoid_reciprocal_raw", read)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    label = view.ui.lblTrapReciprocal
    assert isinstance(label, QLabel)
    assert resolve_context(label).topic == "mca-trapezoid"
    assert resolve_context(view.ui.labelTrapReciprocal).topic == "mca-trapezoid"
    snapshot = {item["id"]: item["value"] for item in device.mca.get_settings()}
    if supported:
        assert label.text() == "4681"
        assert snapshot[int(MCAParam.TRAPEZ_RECIPROCAL)] == 4681
        assert "driver shadow" in label.toolTip()
    else:
        assert label.text() == "Unavailable"
        assert "Update vdpp-mca" in label.toolTip()
        assert int(MCAParam.TRAPEZ_RECIPROCAL) not in snapshot
        read.assert_not_called()
        view.ui.spinTrapR.setValue(24)
        view.ui.spinTrapR.editingFinished.emit()
        assert device.mca.filters.trapezoid.get_R() == 24
        assert label.text() == "Unavailable"
        read.assert_not_called()


@pytest.mark.parametrize("running", [False, True])
def test_gui_refreshes_after_spin_and_slider_commit(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, running: bool,
) -> None:
    device = create_offline_digitizer(0)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    transaction = Mock(wraps=device.mca.reconfigure_while_running)
    monkeypatch.setattr(device.mca, "reconfigure_while_running", transaction)
    if running:
        view._worker = Mock()  # Simulate polling ownership without starting a thread.
        device.mca.start()
    try:
        spin, slider, label = view.ui.spinTrapR, view.ui.sliderTrapR, view.ui.lblTrapReciprocal
        spin.setValue(24)
        assert label.text() == "4681"  # Still the applied R, not an edit preview.
        spin.editingFinished.emit()
        assert label.text() == "16384"
        assert transaction.call_count == int(running)
        slider.setValue(4088)
        assert label.text() == "16384"
        slider.sliderReleased.emit()
        assert label.text() == "64"
        assert transaction.call_count == 2 * int(running)
        assert device._backend.get_global_enable() is running
    finally:
        view._worker = None
        device.mca.stop()


def test_settings_reload_refreshes_the_derived_value(qtbot: QtBot) -> None:
    device = create_offline_digitizer(0)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    apply_mca_hardware(device.mca, {"mca": {"trapezoid": {"rise_ns": 16}}})
    view._load_hardware_state()
    assert view.ui.spinTrapR.value() == 16
    assert view.ui.lblTrapReciprocal.text() == "32768"


def test_read_failure_does_not_leave_a_stale_value(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = create_offline_digitizer(0)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    monkeypatch.setattr(
        device._backend, "get_trapezoid_reciprocal_raw",
        Mock(side_effect=OSError(errno.EIO, "read failed")),
    )
    with pytest.raises(OSError):
        view._refresh_trapezoid_reciprocal()
    assert view.ui.lblTrapReciprocal.text() == "Unavailable"


@pytest.mark.parametrize("interlock", ["replay", "dma", "disconnected"])
def test_interlocked_r_edit_does_not_change_derived_readout(qtbot: QtBot, interlock: str) -> None:
    device = create_offline_digitizer(0)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    if interlock == "replay":
        view._replay_hardware_interlocked = True
    elif interlock == "disconnected":
        view._connection_lost = True
    else:
        view._dma_worker = object()
    try:
        view.ui.spinTrapR.setValue(24)
        view.ui.spinTrapR.editingFinished.emit()
        assert view.ui.lblTrapReciprocal.text() == "4681"
        assert device.mca.filters.trapezoid.get_R() == 64
    finally:
        view._dma_worker = None
