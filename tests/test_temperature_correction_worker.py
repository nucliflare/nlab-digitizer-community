from __future__ import annotations

from unittest.mock import Mock, call

from nlab.workers.temperature_correction_worker import (
    T_MAX,
    TemperatureCorrectionReadback,
    TemperatureCorrectionWorker,
)


def test_temperature_cycle_applies_values_relative_to_t_max() -> None:
    hv = Mock()
    hv.get_ads_temp_for_correction.return_value = 69.0
    mcas = [Mock(), Mock()]
    worker = TemperatureCorrectionWorker(
        hv,
        mcas,
        coefficient=-0.00026735,
        offset=7,
    )
    readbacks: list[TemperatureCorrectionReadback] = []
    worker.readback.connect(readbacks.append)

    worker._tick()

    assert T_MAX == 67.0
    for mca in mcas:
        mca.set_temperature_correction.assert_called_once_with(-0.0005347, 14)
    assert readbacks == [TemperatureCorrectionReadback(69.0, -0.0005347, 14)]


def test_parameter_change_recalculates_immediately() -> None:
    hv = Mock()
    hv.get_ads_temp_for_correction.return_value = 65.0
    mca = Mock()
    worker = TemperatureCorrectionWorker(hv, [mca], coefficient=1.0, offset=1)

    worker._set_parameters(0.25, 3)

    assert mca.set_temperature_correction.call_args_list == [call(-0.5, -6)]


def test_cycle_error_is_reported_without_readback() -> None:
    hv = Mock()
    hv.get_ads_temp_for_correction.side_effect = OSError("sensor unavailable")
    worker = TemperatureCorrectionWorker(hv, [Mock()], coefficient=1.0, offset=1)
    errors: list[str] = []
    readbacks: list[TemperatureCorrectionReadback] = []
    worker.error.connect(errors.append)
    worker.readback.connect(readbacks.append)

    worker._tick()

    assert errors == ["sensor unavailable"]
    assert readbacks == []
