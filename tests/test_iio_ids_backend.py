from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import iio
import pytest

from nlab.hardware.digitizer.backends import iio_ids_backend
from nlab.hardware.digitizer.backends.base import DigitizerBackend, IDSBackend
from nlab.hardware.digitizer.backends.iio_ids_backend import (
    IIOIDSBackend,
    IIOIDSUnavailableError,
)
from nlab.hardware.digitizer.hv import HVSupply


class _Attr:
    def __init__(self, value: str | int | float) -> None:
        self.value = str(value)


class _Channel:
    def __init__(
        self,
        channel_id: str,
        *,
        output: bool = False,
        **attrs: str | int | float,
    ) -> None:
        self.id = channel_id
        self.output = output
        self.attrs = {name: _Attr(value) for name, value in attrs.items()}


class _Device:
    def __init__(
        self,
        name: str,
        channels: list[_Channel],
        *,
        label: str | None = None,
        **attrs: str | int | float,
    ) -> None:
        self.name = name
        self.label = label
        self.channels = channels
        self.attrs = {attr_name: _Attr(value) for attr_name, value in attrs.items()}

    def find_channel(self, name: str, is_output: bool = False) -> _Channel | None:
        return next(
            (
                channel
                for channel in self.channels
                if channel.id == name and channel.output == is_output
            ),
            None,
        )


class _Context:
    def __init__(self, devices: list[_Device]) -> None:
        self.devices = devices

    def find_device(self, name: str) -> _Device | None:
        return next((device for device in self.devices if device.name == name), None)


def _make_context() -> _Context:
    # Deliberately shuffle same-name TMP devices and MCP channels: selection
    # must use the labels documented by the source clients, not probe order.
    adc = _Device(
        "mcp3564r",
        [
            _Channel("voltage1", label="CHB_HV_get_Vout", raw=200_000),
            _Channel("temp", label="temperature", raw=87_433),
            _Channel("voltage0", label="CHA_HV_get_Vout", raw=100_000),
        ],
        scale=0.001,
    )
    dac = _Device(
        "ad5686r",
        [
            _Channel("voltage0", output=True, raw=0, scale=0.04),
            _Channel("voltage1", output=True, raw=2500, scale=0.04),
            _Channel("voltage2", output=True, raw=0, scale=0.04),
            _Channel("voltage3", output=True, raw=0, scale=0.04),
        ],
    )
    temp_b = _Device(
        "tmp117",
        [_Channel("temp", raw=4000, scale=7.8125)],
        label="chb_temp",
    )
    temp_hat = _Device(
        "tmp117",
        [_Channel("temp", raw=6000, scale=7.8125)],
        label="HAT_temp",
    )
    temp_a = _Device(
        "tmp117",
        [_Channel("temp", raw=3000, scale=7.8125)],
        label="cha_temp",
    )
    ads = _Device(
        "ads5407",
        [_Channel("temp", raw=71)],
        sampling_frequency=500_000_000,
    )
    xadc = _Device(
        "xadc",
        [
            _Channel("temp0", raw=2750, offset=-2219, scale=123.040771484),
            _Channel("voltage0", label="vccint", raw=1342, scale=0.732421875),
        ],
    )
    clock = _Device("ltc6951", [], pll_locked=1, ref_ok=1, pll_unlock=0)
    return _Context([temp_b, dac, temp_hat, adc, ads, temp_a, xadc, clock])


def _without_channel_temperatures() -> _Context:
    context = _make_context()
    context.devices = [
        device
        for device in context.devices
        if device.label not in ("cha_temp", "chb_temp")
    ]
    return context


def _without_any_tmp117() -> _Context:
    context = _make_context()
    context.devices = [device for device in context.devices if device.name != "tmp117"]
    return context


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> IIOIDSBackend:
    contexts: list[_Context] = []

    def context_factory(uri: str) -> _Context:
        assert uri == "ip:test"
        context = _make_context()
        contexts.append(context)
        return context

    monkeypatch.setattr(iio, "Context", context_factory)
    result = IIOIDSBackend(1, "ip:test")
    assert len(contexts) == 3
    return result


def test_labelled_discovery_and_scaled_readback(backend: IIOIDSBackend) -> None:
    assert backend.get_hv_adc_voltage() == pytest.approx(100.0)
    assert backend.get_temp_digital() == pytest.approx(31.25)
    assert backend.get_temp_analog() == pytest.approx(25.0017)
    assert backend.get_ads_temp() == pytest.approx(71.0)
    assert backend.get_temp_digital_status() == 1
    assert backend.get_ads_temp_for_correction() == pytest.approx(71.0)


def test_channel_tmp117_is_optional_and_hat_sensor_is_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(iio, "Context", lambda uri: _without_channel_temperatures())

    backend = IIOIDSBackend(1, "ip:test")

    assert backend._devices.digital_temp_label == "HAT_temp"
    assert backend.get_temp_digital_status() == 1
    assert backend.get_temp_digital() == pytest.approx(46.875)


def test_psu_remains_available_without_any_tmp117(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(iio, "Context", lambda uri: _without_any_tmp117())

    backend = IIOIDSBackend(0, "ip:test")

    assert backend.get_temp_digital_status() == 0
    assert math.isnan(backend.get_temp_digital())
    assert backend.get_hv_adc_voltage() == pytest.approx(50.0)
    with pytest.raises(RuntimeError, match="requires.*TMP117"):
        backend.set_hv_compens_mode(1)


def test_global_diagnostics_use_dedicated_context_and_physical_units(
    backend: IIOIDSBackend,
) -> None:
    readings = {reading.key: reading for reading in backend.get_global_diagnostics()}

    assert readings["hat_temperature"].value == pytest.approx(46.875)
    assert readings["fpga_temperature"].value == pytest.approx(65.33465)
    assert readings["ads_temperature_raw"].value == 71
    assert readings["sampling_frequency"].value == 500_000_000
    assert readings["pll_locked"].healthy is True
    assert readings["ref_ok"].healthy is True
    assert readings["pll_unlock"].healthy is True
    assert readings["rail_vccint"].value == pytest.approx(0.98291015625)


def test_hv_setpoint_uses_channel_dac_scale(backend: IIOIDSBackend) -> None:
    backend.set_hv_voltage(100.0)

    assert backend._devices.hv_output.id == "voltage1"
    assert backend._devices.hv_output.attrs["raw"].value == "5000"

    backend.set_hv_voltage(0.0)
    assert backend._devices.hv_output.attrs["raw"].value == "0"
    backend.set_hv_voltage(1250.0)
    assert backend._devices.hv_output.attrs["raw"].value == "62500"


def test_hv_temperature_compensation_is_applied_during_poll(
    backend: IIOIDSBackend,
) -> None:
    backend.set_hv_voltage(100.0)
    backend.set_hv_compens_ct(0.1)
    backend.set_hv_compens_tref(20.0)
    backend.set_hv_compens_mode(1)

    # Digital temperature is 31.25 C, so the target is 101.125 V. The DAC
    # resolution is 0.02 V/code and therefore rounds to 101.12 V.
    assert backend.get_hv_compens_output() == pytest.approx(101.12)
    assert backend._monitor_devices.hv_output.attrs["raw"].value == "5056"


def test_removed_sipm_is_unavailable(backend: IIOIDSBackend) -> None:
    assert backend.get_sipm_overload() == -1
    with pytest.raises(NotImplementedError, match="not present"):
        backend.set_sipm_voltage(50.0)


def test_undocumented_compensation_mode_is_not_guessed(
    backend: IIOIDSBackend,
) -> None:
    with pytest.raises(NotImplementedError, match="mode 3"):
        backend.set_hv_compens_mode(3)


def test_tmp117_enable_is_compatibility_state_only(backend: IIOIDSBackend) -> None:
    backend.set_temp_digital_enable(2)
    assert backend._temp_digital_mode == 2
    with pytest.raises(ValueError, match="0, 1 or 2"):
        backend.set_temp_digital_enable(3)


def test_iio_safe_shutdown_voltage_is_real_dac_off(backend: IIOIDSBackend) -> None:
    assert backend.get_safe_shutdown_voltage() == 0.0
    backend.set_hv_voltage(100.0)

    HVSupply(backend).safe_shutdown()

    assert backend._devices.hv_output.attrs["raw"].value == "0"


def test_factory_can_disable_iio_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    from nlab.hardware.digitizer import digitizer
    from nlab.hardware.digitizer.backends import iio_backend

    fake_backend = cast(
        DigitizerBackend,
        SimpleNamespace(mca_dma_hardware_present=lambda: False),
    )
    monkeypatch.setattr(
        iio_backend,
        "IIODigitizerBackend",
        lambda channel, uri: fake_backend,
    )
    monkeypatch.setattr(
        iio_ids_backend,
        "IIOIDSBackend",
        lambda channel, uri: pytest.fail("IDS backend should not be constructed"),
    )

    result = digitizer.Digitizer.from_iio(0, "ip:test", with_ids=False)

    assert result.hv is None


def test_factory_enables_iio_ids_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from nlab.hardware.digitizer import digitizer
    from nlab.hardware.digitizer.backends import iio_backend

    fake_backend = SimpleNamespace(mca_dma_hardware_present=lambda: False)
    fake_ids = cast(IDSBackend, SimpleNamespace())
    monkeypatch.setattr(
        iio_backend,
        "IIODigitizerBackend",
        lambda channel, uri: fake_backend,
    )
    monkeypatch.setattr(
        iio_ids_backend,
        "IIOIDSBackend",
        lambda channel, uri: fake_ids,
    )

    result = digitizer.Digitizer.from_iio(1, "ip:test")

    assert result.hv is not None
    assert result.hv._b is fake_ids


def test_factory_continues_when_channel_ids_devices_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nlab.hardware.digitizer import digitizer
    from nlab.hardware.digitizer.backends import iio_backend

    fake_backend = cast(
        DigitizerBackend,
        SimpleNamespace(mca_dma_hardware_present=lambda: False),
    )
    monkeypatch.setattr(
        iio_backend,
        "IIODigitizerBackend",
        lambda channel, uri: fake_backend,
    )

    def unavailable(channel: int, uri: str) -> IDSBackend:
        raise IIOIDSUnavailableError(
            "IIO IDS backend: no mcp3564/mcp3564r device found"
        )

    monkeypatch.setattr(iio_ids_backend, "IIOIDSBackend", unavailable)

    result = digitizer.Digitizer.from_iio(0, "ip:test")

    assert result.hv is None
    assert result.scope._b is fake_backend
