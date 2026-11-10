"""Read-only 1/R ABI and UI behavior using fake IIO and offline devices."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QWidget
from pytestqt.qtbot import QtBot

from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.backends.base import MCABackend
from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.mca import MCAParam, MultiChannelAnalyzer
from nlab.hardware.digitizer.mca_parameters import DEFAULT_PARAMETERS
from nlab.hardware.digitizer.offline import create_offline_digitizer
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


@pytest.mark.parametrize("rise,expected", [(8, 16384), (16, 8192), (64, 2048), (2044, 64)])
def test_offline_register_dependency_matches_documented_driver_mapping(
    rise: int, expected: int,
) -> None:
    device = create_offline_digitizer(0)
    trapezoid = device.mca.filters.trapezoid
    assert trapezoid.get_reciprocal_raw() == 4096  # Explicit offline defaults use R=32 samples.
    trapezoid.set_R(rise)
    assert trapezoid.get_R() == rise
    assert trapezoid.get_reciprocal_raw() == expected
    snapshot = {item["id"]: item["value"] for item in device.mca.get_settings()}
    assert snapshot[int(MCAParam.TRAPEZ_RECIPROCAL)] == expected
    hardware = collect_channel_hardware(device.scope, device.mca, None)
    assert "reciprocal_raw" not in hardware["mca"]["parameters"]


@pytest.mark.parametrize("supported", [False, True])
def test_gui_omits_automatic_1r_for_both_driver_capabilities(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, supported: bool,
) -> None:
    device = create_offline_digitizer(0)
    monkeypatch.setattr(device._backend, "supports_trapezoid_reciprocal_raw", lambda: supported)
    read = Mock(side_effect=OSError("no coefficient read is needed by the GUI"))
    monkeypatch.setattr(device._backend, "get_trapezoid_reciprocal_raw", read)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    assert view.findChild(QWidget, "labelTrapReciprocal") is None
    assert view.findChild(QWidget, "lblTrapReciprocal") is None
    read.assert_not_called()


@pytest.mark.parametrize("running", [False, True])
def test_gui_r_edits_keep_driver_derived_normalization(
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
        spin, slider = view.ui.spinTrapR, view.ui.sliderTrapR
        spin.setValue(48)
        assert device.mca.filters.trapezoid.get_R() == 32  # Edit is not yet committed.
        spin.editingFinished.emit()
        assert device.mca.filters.trapezoid.get_R() == 24
        assert device.mca.filters.trapezoid.get_reciprocal_raw() == 5461
        assert transaction.call_count == int(running)
        slider.setValue(4088)
        assert device.mca.filters.trapezoid.get_R() == 24
        slider.sliderReleased.emit()
        assert device.mca.filters.trapezoid.get_R() == 2044
        assert device.mca.filters.trapezoid.get_reciprocal_raw() == 64
        assert transaction.call_count == 2 * int(running)
        assert device._backend.get_global_enable() is running
    finally:
        view._worker = None
        device.mca.stop()


def test_settings_reload_does_not_read_the_hidden_derived_coefficient(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = create_offline_digitizer(0)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    read = Mock(side_effect=OSError("optional raw readback unavailable"))
    monkeypatch.setattr(device._backend, "get_trapezoid_reciprocal_raw", read)
    apply_mca_hardware(
        device.mca,
        {
            "mca": {
                "parameter_abi_version": 2,
                "parameter_units": "adc_samples",
                "parameters": DEFAULT_PARAMETERS | {"trapezoid_r_samples": 16},
            }
        },
    )
    view._load_hardware_state()
    assert view.ui.spinTrapR.value() == 32
    read.assert_not_called()


@pytest.mark.parametrize("interlock", ["replay", "dma", "disconnected"])
def test_interlocked_r_edit_preserves_the_driver_derived_value(qtbot: QtBot, interlock: str):
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
        view.ui.spinTrapR.setValue(48)
        view.ui.spinTrapR.editingFinished.emit()
        assert device.mca.filters.trapezoid.get_R() == 32
        assert device.mca.filters.trapezoid.get_reciprocal_raw() == 4096
    finally:
        view._dma_worker = None
