"""Separate elapsed status units from the programmed-duration register scale."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.workers.mca_worker import MCAReadback, MCAWorker


@pytest.mark.parametrize("raw", [0, 1, 9, 10, 999, 1000, 2**32 - 1])
def test_elapsed_register_is_already_deciseconds(
    monkeypatch: pytest.MonkeyPatch, raw: int,
) -> None:
    backend = object.__new__(IIODigitizerBackend)
    read = Mock(return_value=str(raw))
    write = Mock()
    monkeypatch.setattr(backend, "_mca_pp_attr_get", read)
    monkeypatch.setattr(backend, "_pp_attr_set", write)

    assert backend.get_elapsed_time() == raw
    read.assert_called_once_with("enable_time_raw")  # Never read measurement_time_scale.
    write.assert_not_called()


def test_100_second_setting_and_elapsed_counter_use_different_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Values independently read with iio_info on CH0, hw101/sw103023. This
    # regression replays that snapshot; it does not open a device or run a capture.
    values = {"measurement_time_scale": "0.134217728", "measurement_time_raw": "745"}
    backend = object.__new__(IIODigitizerBackend)
    monkeypatch.setattr(backend, "_pp_attr_get", lambda name: values[name])
    write = Mock(side_effect=lambda name, value: values.__setitem__(name, value))
    monkeypatch.setattr(backend, "_pp_attr_set", write)
    monkeypatch.setattr(backend, "_mca_pp_attr_get", lambda name: {"enable_time_raw": "999"}[name])

    backend.set_time_limit(100)

    write.assert_called_once_with("measurement_time_raw", "745")
    assert backend.get_time_limit() == 100
    assert backend.get_elapsed_time() / 10 == 99.9  # Was 134.1 s before the fix.


def test_elapsed_read_errors_are_not_silently_replaced_with_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(IIODigitizerBackend)
    monkeypatch.setattr(backend, "_mca_pp_attr_get", Mock(side_effect=OSError("read failed")))
    with pytest.raises(OSError, match="read failed"):
        backend.get_elapsed_time()


def test_corrected_elapsed_reaches_gui_roi_rate_and_analysis_snapshots(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(IIODigitizerBackend)
    monkeypatch.setattr(backend, "_mca_pp_attr_get", lambda name: {"enable_time_raw": "999"}[name])
    device = create_offline_digitizer(0)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    monkeypatch.setattr(device.mca.statistics, "get_elapsed_time", backend.get_elapsed_time)
    monkeypatch.setattr(
        device.mca.statistics, "get_pulse_deadtime_ms", lambda: 36_048.994304,
    )
    histogram = np.zeros(16_384, dtype=np.uint32)
    histogram[1060] = 630_018
    monkeypatch.setattr(device.mca, "acquire_spectrum", lambda: histogram)
    worker = MCAWorker(device.mca)
    readbacks: list[MCAReadback] = []
    worker.readback.connect(readbacks.append)
    worker._tick()  # In-memory diagnostics only; no QThread or IIO connection.
    assert len(readbacks) == 1 and readbacks[0].elapsed_time == 999

    view.show()
    view._roi.setRegion((801, 1380))
    view._roi.setVisible(True)
    view._on_readback(readbacks[0])
    view._render_pending_readback()
    view._update_roi_stats()

    assert view.ui.lblElapsedTime.text() == "99.9"
    assert view.ui.lblDeadTime.text() == "36048.994304"
    assert view.ui.lblDeadTimePercent.text() == "36.09"
    assert view._last_elapsed_s == 99.9
    assert "6306.49" in view.ui.lblRoiStats.text()  # 630018 / 99.9, not / 134.1.
    assert view.spectrum_snapshot().elapsed_s == 99.9
    dose = view.dose_calibration_snapshot()
    assert dose is not None and dose.acquisition_elapsed_s == 99.9


@pytest.mark.parametrize(
    ("dead_ms", "elapsed_ticks", "expected"),
    [
        (0.0, 0, "N/A"),
        (1000.0, 0, "N/A"),
        (1000.0, -1, "N/A"),
        (0.0, 1000, "0.00"),
        (50.0, 1, "50.00"),  # 50 ms / 0.1 s.
        (36_048.994304, 1000, "36.05"),
        (100_000.0, 1000, "100.00"),
        (110_000.0, 1000, "110.00"),  # Do not hide inconsistent snapshots.
    ],
)
def test_dead_time_percentage_uses_consistent_units(
    qtbot: QtBot, dead_ms: float, elapsed_ticks: int, expected: str,
) -> None:
    device = create_offline_digitizer(0)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    try:
        assert view.ui.lblDeadTimePercent.text() == "N/A"
        readback = MCAReadback(
            histogram=np.zeros(16_384, dtype=np.uint32),
            debug1=np.empty(0, dtype=np.int16), debug2=np.empty(0, dtype=np.int16),
            pulse_deadtime_ms=dead_ms, elapsed_time=elapsed_ticks,
        )
        view._update_statistics(readback)
        assert view.ui.lblDeadTimePercent.text() == expected
        assert view.ui.lblDeadTime.text() == f"{dead_ms:.6f}"
        assert view.ui.lblElapsedTime.text() == f"{elapsed_ticks / 10:.1f}"
        # Replay clears stale values and uses the same calculation as live polls.
        view.begin_session_replay()
        assert view.ui.lblDeadTimePercent.text() == "N/A"
        view.display_session_replay_readback(readback)
        assert view.ui.lblDeadTimePercent.text() == expected
        view.end_session_replay()
    finally:
        device.close()
