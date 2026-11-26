"""Per-channel IIO devices are paired by channel_index, not by iio:deviceN order.

PetaLinux 03f312d adds read-only ``channel_index`` (0 = input A, 1 = input B)
to vdpp_scope and vdpp_input_filter, matching the pulse-processor and
list-mode drivers; user-api.md asks clients to reject missing, duplicate or
invalid identities. On .128 the scopes are iio:device11/12 and the input
filters iio:device15/16 (read-only iio_info, 2026-11-26); the fakes below
reverse the numbering so sorted order would pick the wrong device.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from nlab.hardware.digitizer.backends import iio_backend as iio_backend_module
from nlab.hardware.digitizer.backends.iio_backend import (
    IIODigitizerBackend,
    _device_for_channel,
)


def _device(name: str, number: int, index: str | None, **attrs: str) -> SimpleNamespace:
    values = dict(attrs)
    if index is not None:
        values["channel_index"] = index
    return SimpleNamespace(
        name=name,
        id=f"iio:device{number}",
        attrs={key: SimpleNamespace(value=value) for key, value in values.items()},
        find_channel=Mock(return_value=SimpleNamespace(enabled=False)),
    )


def _context(*devices: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(devices=list(devices))


def test_channel_index_wins_over_device_numbers() -> None:
    context = _context(_device("vdpp_scope", 11, "1"), _device("vdpp_scope", 12, "0"))
    assert _device_for_channel(context, "vdpp_scope", 0).id == "iio:device12"
    assert _device_for_channel(context, "vdpp_scope", 1).id == "iio:device11"


def test_older_images_without_channel_index_keep_sorted_order() -> None:
    context = _context(_device("vdpp_scope", 16, None), _device("vdpp_scope", 9, None))
    assert _device_for_channel(context, "vdpp_scope", 0).id == "iio:device9"
    assert _device_for_channel(context, "vdpp_scope", 2) is None


@pytest.mark.parametrize("indices,match", [
    (("0", None), "only 1 of 2"),
    (("0", "0"), r"duplicate channel_index \[0\]"),
    (("0", "-1"), "invalid channel_index '-1'"),
    (("0", "B"), "invalid channel_index 'B'"),
    (("1", "2"), r"no 'vdpp_input_filter' device with channel_index=0; found \[1, 2\]"),
])
def test_missing_duplicate_or_invalid_identity_is_refused(
    indices: tuple[str | None, str | None], match: str,
) -> None:
    context = _context(
        _device("vdpp_input_filter", 15, indices[0]),
        _device("vdpp_input_filter", 16, indices[1]),
    )
    with pytest.raises(RuntimeError, match=match):
        _device_for_channel(context, "vdpp_input_filter", 0)


def test_scope_gui_and_dma_contexts_select_the_scope_by_channel_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts: list[SimpleNamespace] = []

    def context_factory(_uri: str) -> SimpleNamespace:
        context = _context(_device("vdpp_scope", 11, "1"), _device("vdpp_scope", 12, "0"))
        contexts.append(context)
        return context

    monkeypatch.setattr(iio_backend_module.iio, "Context", context_factory)
    backend = IIODigitizerBackend(0, "ip:board:30431", dac_name=None, _scope_dma_only=True)

    assert len(contexts) == 2  # GUI/control and DMA contexts, one each.
    assert backend._scope.id == "iio:device12"
    assert backend._dma_scope.id == "iio:device12"


def test_scope_without_a_matching_identity_does_not_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        iio_backend_module.iio, "Context",
        lambda _uri: _context(_device("vdpp_scope", 11, "1"), _device("vdpp_scope", 12, "1")),
    )
    with pytest.raises(RuntimeError, match="duplicate channel_index"):
        IIODigitizerBackend(0, "ip:board:30431", dac_name=None, _scope_only=True)


def test_temperature_correction_context_selects_the_filter_by_channel_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filters = {
        index: _device(
            "vdpp_input_filter", number, str(index),
            temperature_coefficient_scale="0.001",
            temperature_coefficient_raw="0", temperature_offset_raw="0",
        )
        for index, number in ((1, 15), (0, 16))
    }
    monkeypatch.setattr(
        iio_backend_module.iio, "Context", lambda _uri: _context(filters[1], filters[0]),
    )
    backend: Any = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._uri = "ip:board:30431"
    backend._temperature_correction_ctx = None
    backend._temperature_correction_filter = None

    backend.set_temperature_correction_from_worker(0.0, 7)

    assert backend._temperature_correction_filter is filters[0]
    assert filters[0].attrs["temperature_offset_raw"].value == "7"
    assert filters[1].attrs["temperature_offset_raw"].value == "0"
