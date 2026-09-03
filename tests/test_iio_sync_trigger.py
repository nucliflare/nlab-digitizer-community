from __future__ import annotations

import errno
from types import SimpleNamespace
from typing import Any

import pytest

from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend


class _Attr:
    def __init__(self, value: str, error: OSError | None = None) -> None:
        self._value = value
        self._error = error

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, value: str) -> None:
        if self._error is not None:
            raise self._error
        self._value = value


def _backend(**values: str) -> IIODigitizerBackend:
    backend = object.__new__(IIODigitizerBackend)
    backend._sync_trigger = SimpleNamespace(
        attrs={name: _Attr(value) for name, value in values.items()}
    )
    return backend


def test_sync_interface_maps_to_vdpp_sync_trigger_attributes() -> None:
    backend = _backend(
        start_enable="1",
        software_start_state="0",
        start_source="1",
        timestamp_raw="18446744073709551614",
    )

    assert backend.get_sync_enable() is True
    assert backend.get_sync_sw_trig() == 0
    assert backend.get_sync_trig_src() == 1
    assert backend.get_sync_timestamp() == 18446744073709551614

    backend.set_sync_enable(False)
    backend.set_sync_sw_trig(1)
    backend.set_sync_trig_src(0)

    sync_trigger = backend._sync_trigger
    assert sync_trigger is not None
    attrs: dict[str, Any] = sync_trigger.attrs
    assert attrs["start_enable"].value == "0"
    assert attrs["software_start_state"].value == "1"
    assert attrs["start_source"].value == "0"


def test_sync_source_busy_error_is_not_hidden() -> None:
    backend = _backend(start_source="0")
    sync_trigger = backend._sync_trigger
    assert sync_trigger is not None
    sync_trigger.attrs["start_source"] = _Attr(
        "0", OSError(errno.EBUSY, "Device or resource busy")
    )

    with pytest.raises(OSError) as exc_info:
        backend.set_sync_trig_src(1)

    assert exc_info.value.errno == errno.EBUSY


def test_sync_methods_report_missing_optional_core() -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._sync_trigger = None

    with pytest.raises(RuntimeError, match="vdpp_sync_trigger"):
        backend.get_sync_enable()


def test_worker_temperature_correction_uses_dedicated_filter() -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._temperature_correction_ctx = object()
    backend._temperature_correction_filter = SimpleNamespace(attrs={
        "temperature_coefficient_scale": _Attr("0.000030517578125"),
        "temperature_coefficient_raw": _Attr("0"),
        "temperature_offset_raw": _Attr("0"),
    })

    backend.set_temperature_correction_from_worker(-0.0005347, 14)

    attrs = backend._temperature_correction_filter.attrs
    assert attrs["temperature_coefficient_raw"].value == "-18"
    assert attrs["temperature_offset_raw"].value == "14"
