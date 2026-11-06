"""Signed Q1.15 boundaries using fake IIO attributes, with no hardware access."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QWidget
from pytestqt.qtbot import QtBot

from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.mca import MCA_PARAMETER_SPECS, MCAParam, MultiChannelAnalyzer
from nlab.ui.ui_global_view import Ui_GlobalView
from nlab.workers.temperature_correction_worker import TemperatureCorrectionWorker

SCALE = 2**-15
MAXIMUM = 32767 * SCALE


def _attributes(scale: float = SCALE) -> dict[str, SimpleNamespace]:
    return {
        "temperature_coefficient_scale": SimpleNamespace(value=str(scale)),
        "temperature_coefficient_raw": SimpleNamespace(value="123"),
        "temperature_offset_raw": SimpleNamespace(value="456"),
    }


def _backend(scale: float = SCALE) -> IIODigitizerBackend:
    backend = object.__new__(IIODigitizerBackend)
    backend._input_filter = SimpleNamespace(attrs=_attributes(scale))
    backend._temperature_correction_ctx = object()
    backend._temperature_correction_filter = SimpleNamespace(attrs=_attributes(scale))
    return backend


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("value,raw", [
    (-1.0, -32768), (MAXIMUM, 32767), (0.0, 0), (-0.0005347, -18),
])
def test_signed_endpoints_and_normal_rounding(value: float, raw: int, worker: bool) -> None:
    backend = _backend()
    if worker:
        backend.set_temperature_correction_from_worker(value, 14)
        attrs = backend._temperature_correction_filter.attrs
        assert attrs["temperature_offset_raw"].value == "14"
        assert backend._input_filter.attrs["temperature_coefficient_raw"].value == "123"
    else:
        backend.set_temp_coeff(value)
        attrs = backend._input_filter.attrs
        worker_attrs = backend._temperature_correction_filter.attrs
        assert worker_attrs["temperature_coefficient_raw"].value == "123"
    assert attrs["temperature_coefficient_raw"].value == str(raw)
    if not worker:
        assert backend.get_temp_coeff() == raw * SCALE


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("value", [
    1.0, math.nextafter(MAXIMUM, math.inf), math.nextafter(-1.0, -math.inf),
    math.nan, math.inf, -math.inf,
])
def test_invalid_applied_coefficient_never_writes_either_term(value: float, worker: bool) -> None:
    backend = _backend()
    with pytest.raises(ValueError, match="temperature coefficient"):
        if worker:
            backend.set_temperature_correction_from_worker(value, 14)
        else:
            backend.set_temp_coeff(value)
    for device in (backend._input_filter, backend._temperature_correction_filter):
        assert device.attrs["temperature_coefficient_raw"].value == "123"
        assert device.attrs["temperature_offset_raw"].value == "456"


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("scale", [0.0, -SCALE, math.nan, math.inf])
def test_invalid_driver_scale_is_reported_before_writes(scale: float, worker: bool) -> None:
    backend = _backend(scale)
    with pytest.raises(ValueError, match="scale must be finite and positive"):
        if worker:
            backend.set_temperature_correction_from_worker(0.25, 14)
        else:
            backend.set_temp_coeff(0.25)
    assert backend._input_filter.attrs["temperature_coefficient_raw"].value == "123"
    worker_attrs = backend._temperature_correction_filter.attrs
    assert worker_attrs["temperature_coefficient_raw"].value == "123"
    assert backend._temperature_correction_filter.attrs["temperature_offset_raw"].value == "456"


@pytest.mark.parametrize("combined", [False, True])
def test_public_api_rejects_positive_one_before_invoking_backend(combined: bool) -> None:
    backend = Mock()
    mca = MultiChannelAnalyzer(backend)
    with pytest.raises(ValueError, match="temperature_coefficient"):
        if combined:
            mca.set_temperature_correction(1.0, 7)
        else:
            mca.set_temp_coeff(1.0)
    backend.set_temp_coeff.assert_not_called()
    backend.set_temperature_correction_from_worker.assert_not_called()
    backend.set_temp_offset.assert_not_called()
    spec = MCA_PARAMETER_SPECS[MCAParam.TEMP_COEFF]
    assert spec.max_val == MAXIMUM
    assert spec.step == SCALE


def test_base_calibration_precision_is_preserved_until_applied(qtbot: QtBot) -> None:
    view = QWidget()
    ui = Ui_GlobalView()
    ui.setupUi(view)
    qtbot.addWidget(view)
    base = -0.00026735
    ui.spinTempCoeff.setValue(base)
    assert ui.spinTempCoeff.value() == base
    backend = _backend()
    mca = MultiChannelAnalyzer(backend)
    hv = Mock()
    hv.get_ads_temp_for_correction.return_value = 69.0
    worker = TemperatureCorrectionWorker(hv, [mca], ui.spinTempCoeff.value(), 7)
    worker._tick()
    worker_attrs = backend._temperature_correction_filter.attrs
    assert worker_attrs["temperature_coefficient_raw"].value == "-18"
    assert backend._temperature_correction_filter.attrs["temperature_offset_raw"].value == "14"


def test_worker_reports_invalid_applied_value_without_updating_hardware() -> None:
    backend = _backend()
    mca = MultiChannelAnalyzer(backend)
    hv = Mock()
    hv.get_ads_temp_for_correction.return_value = 68.0
    worker = TemperatureCorrectionWorker(hv, [mca], coefficient=1.0, offset=7)
    errors: list[str] = []
    readbacks: list[object] = []
    worker.error.connect(errors.append)
    worker.readback.connect(readbacks.append)
    worker._tick()
    assert len(errors) == 1 and "temperature_coefficient" in errors[0]
    assert readbacks == []
    worker_attrs = backend._temperature_correction_filter.attrs
    assert worker_attrs["temperature_coefficient_raw"].value == "123"
    assert backend._temperature_correction_filter.attrs["temperature_offset_raw"].value == "456"
