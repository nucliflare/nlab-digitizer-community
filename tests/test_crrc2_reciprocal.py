"""Raw CR-RC2 register access with fake IIO attributes and in-memory devices."""

from __future__ import annotations

import errno
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pytestqt.qtbot import QtBot

from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.mca import MCAParam, MultiChannelAnalyzer
from nlab.hardware.digitizer.mca_parameters import DEFAULT_PARAMETERS
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.help.context import resolve_context
from nlab.utils.settings_io import (
    FORMAT_VERSION,
    apply_mca_hardware,
    collect_channel_hardware,
    read_configuration,
    write_configuration,
)


def _backend(*, supported: bool = True) -> IIODigitizerBackend:
    backend = object.__new__(IIODigitizerBackend)
    attributes = {"crrc2_cdelay": SimpleNamespace(value="3"),
                  "parameter_abi_version": SimpleNamespace(value="2"),
                  "parameter_units": SimpleNamespace(value="adc_samples"),
                  "parameter_abi_ack": SimpleNamespace(value="2")}
    if supported:
        attributes["crrc2_1cdelay_raw"] = SimpleNamespace(value="0")
    backend._pp = SimpleNamespace(attrs=attributes)
    return backend


@pytest.mark.parametrize("value", [0, 1, 32768, 65535])
def test_iio_register_and_public_api_preserve_all_raw_bits(value: int) -> None:
    backend = _backend()
    shape = MultiChannelAnalyzer(backend).filters.crrc2
    assert shape.supports_reciprocal_raw()

    shape.set_reciprocal_raw(value)

    assert backend._pp.attrs["crrc2_1cdelay_raw"].value == str(value)
    assert shape.get_reciprocal_raw() == value
    assert backend._pp.attrs["crrc2_cdelay"].value == "3"


@pytest.mark.parametrize("value", [-1, 65536, 1.5])
def test_invalid_raw_values_never_reach_iio(value: int) -> None:
    backend = _backend()
    shape = MultiChannelAnalyzer(backend).filters.crrc2
    with pytest.raises(ValueError):
        shape.set_reciprocal_raw(value)
    with pytest.raises(ValueError):
        backend.set_crrc2_reciprocal_raw(value)
    assert backend._pp.attrs["crrc2_1cdelay_raw"].value == "0"


def test_older_driver_is_reported_as_unsupported_without_a_shadow_fallback() -> None:
    backend = _backend(supported=False)
    shape = MultiChannelAnalyzer(backend).filters.crrc2
    assert not shape.supports_reciprocal_raw()
    for read in (shape.get_reciprocal_raw, backend.get_crrc2_reciprocal_raw):
        with pytest.raises(NotImplementedError):
            read()
    for write in (shape.set_reciprocal_raw, backend.set_crrc2_reciprocal_raw):
        with pytest.raises(NotImplementedError):
            write(1)


def test_driver_busy_error_is_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend()
    write = Mock(side_effect=OSError(errno.EBUSY, "measurement owns MCA"))
    monkeypatch.setattr(backend, "_pp_attr_set", write)
    with pytest.raises(OSError) as result:
        backend.set_crrc2_reciprocal_raw(4096)
    assert result.value.errno == errno.EBUSY
    write.assert_called_once_with("crrc2_1cdelay_raw", "4096")


def test_settings_round_trip_and_flat_snapshot_keep_raw_coefficient(tmp_path: Path) -> None:
    source = create_offline_digitizer(0)
    target = create_offline_digitizer(1)
    MCAController._send_defaults(SimpleNamespace(_mca=source.mca, _channel=0))
    source.mca.filters.crrc2.set_reciprocal_raw(54321)
    channel = collect_channel_hardware(source.scope, source.mca, None)
    path = tmp_path / "settings.yaml"
    write_configuration(path, {
        "format_version": FORMAT_VERSION, "hardware": {"channels": {"0": channel}},
    })
    restored = read_configuration(path)["hardware"]["channels"]["0"]

    apply_mca_hardware(target.mca, restored)

    assert restored["mca"]["optional_parameters"]["crrc2_1cdelay_raw"] == 54321
    assert target.mca.filters.crrc2.get_reciprocal_raw() == 54321
    settings = {entry["id"]: entry["value"] for entry in target.mca.get_settings()}
    assert settings[int(MCAParam.CRRC2_RECIPROCAL)] == 54321


@pytest.mark.parametrize("value", [-1, 65536, 1.5, "4096"])
def test_invalid_saved_raw_values_are_rejected_without_truncation(value: object) -> None:
    device = create_offline_digitizer(0)
    with pytest.raises(ValueError):
        apply_mca_hardware(device.mca, {"mca": {"crrc2": {"reciprocal_raw": value}}})
    assert device.mca.filters.crrc2.get_reciprocal_raw() == 0


def test_legacy_settings_require_review_and_leave_register_unchanged() -> None:
    device = create_offline_digitizer(0)
    device.mca.filters.crrc2.set_reciprocal_raw(4096)
    with pytest.raises(ValueError, match="review older"):
        apply_mca_hardware(device.mca, {"mca": {"crrc2": {"cdelay_ns": 128}}})
    assert device.mca.filters.crrc2.get_reciprocal_raw() == 4096

def test_saved_reciprocal_requires_driver_support(monkeypatch: pytest.MonkeyPatch) -> None:
    device = create_offline_digitizer(0)
    monkeypatch.setattr(device._backend, "supports_crrc2_reciprocal_raw", lambda: False)
    with pytest.raises(NotImplementedError, match="does not advertise"):
        apply_mca_hardware(device.mca, {"mca": {
            "parameter_abi_version": 2, "parameter_units": "adc_samples",
            "parameters": dict(DEFAULT_PARAMETERS),
            "optional_parameters": {"crrc2_1cdelay_raw": 1}}})


@pytest.mark.parametrize("supported", [False, True])
def test_gui_initialization_and_driver_capability(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, supported: bool,
) -> None:
    device = create_offline_digitizer(0)
    monkeypatch.setattr(device._backend, "supports_crrc2_reciprocal_raw", lambda: supported)
    write = Mock(wraps=device._backend.set_crrc2_reciprocal_raw)
    monkeypatch.setattr(device._backend, "set_crrc2_reciprocal_raw", write)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    spinbox = view.ui.spinCrrc2Reciprocal
    assert (spinbox.minimum(), spinbox.maximum(), spinbox.singleStep()) == (0, 65535, 1)
    assert spinbox.suffix() == ""
    assert spinbox.isEnabled() is supported
    assert view.ui.sliderCrrc2Reciprocal.isEnabled() is supported
    assert view.ui.formCrRc2.isRowVisible(view.ui.labelCrrc2Reciprocal) is supported
    assert view.ui.labelCrrc2Reciprocal.isHidden() is not supported
    assert spinbox.isHidden() is not supported
    assert view.ui.sliderCrrc2Reciprocal.isHidden() is not supported
    assert resolve_context(spinbox).topic == "mca-crrc"
    hardware = collect_channel_hardware(device.scope, device.mca, None)
    snapshot = {entry["id"]: entry["value"] for entry in device.mca.get_settings()}
    if supported:
        write.assert_called_once_with(0)
        assert hardware["mca"]["optional_parameters"]["crrc2_1cdelay_raw"] == 0
        assert snapshot[int(MCAParam.CRRC2_RECIPROCAL)] == 0
    else:
        assert "does not expose" in spinbox.toolTip()
        write.assert_not_called()
        assert "optional_parameters" not in hardware["mca"]
        assert int(MCAParam.CRRC2_RECIPROCAL) not in snapshot


@pytest.mark.parametrize("running", [False, True])
def test_gui_commits_spinbox_and_slider_without_linking_cdelay(
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
        spinbox, slider = view.ui.spinCrrc2Reciprocal, view.ui.sliderCrrc2Reciprocal
        spinbox.setValue(4096)
        assert slider.value() == 4096
        assert device.mca.filters.crrc2.get_reciprocal_raw() == 0
        spinbox.editingFinished.emit()
        assert transaction.call_count == int(running)
        assert device.mca.filters.crrc2.get_reciprocal_raw() == 4096
        view.ui.spinCrrc2Cdelay.setValue(4)
        view.ui.spinCrrc2Cdelay.editingFinished.emit()
        assert device.mca.filters.crrc2.get_reciprocal_raw() == 4096
        slider.setValue(65535)
        slider.sliderReleased.emit()
        assert spinbox.value() == 65535
        assert device.mca.filters.crrc2.get_reciprocal_raw() == 65535
        assert device._backend.get_global_enable() is running
    finally:
        view._worker = None
        device.mca.stop()


@pytest.mark.parametrize("interlock", ["replay", "dma", "disconnected"])
def test_gui_reciprocal_retains_write_interlocks(qtbot: QtBot, interlock: str) -> None:
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
        view.ui.spinCrrc2Reciprocal.setValue(1024)
        view.ui.spinCrrc2Reciprocal.editingFinished.emit()
        assert device.mca.filters.crrc2.get_reciprocal_raw() == 0
    finally:
        view._dma_worker = None
