from __future__ import annotations

import numpy as np

from nlab.analysis.psd import PsdAccumulator
from nlab.hardware.digitizer.dma import _EVENT_DTYPE, _LM_EVENT_DTYPE, McaEventBuffer


def test_iio_events_accumulate_ratio_matrix_and_projections() -> None:
    events = np.zeros(5, dtype=_LM_EVENT_DTYPE)
    events["trapezoid_energy"] = [8192, 24576, 40960, 57344, 0]
    events["charge_energy"] = [8192, 12288, 40960, 0, 0]
    accumulator = PsdAccumulator(
        energy_bins=4,
        ratio_bins=4,
        energy_right_shift=0,
        ratio_range=(-1.0, 1.0),
    )

    stats = accumulator.add_events(events)

    assert stats.received == 5
    assert stats.accepted == 4
    assert stats.zero_total == 1
    assert stats.outside_range == 0
    assert accumulator.matrix.sum() == 4

    below, above = accumulator.energy_projections(0.25)
    np.testing.assert_array_equal(below, [1, 0, 1, 0])
    np.testing.assert_array_equal(above, [0, 1, 0, 1])
    np.testing.assert_array_equal(
        accumulator.ratio_projection((0.0, 32768.0)),
        [0, 0, 1, 1],
    )


def test_legacy_events_use_energy_and_short_energy() -> None:
    events = np.zeros(2, dtype=_EVENT_DTYPE)
    events["energy"] = [1000, 2000]
    events["short_energy"] = [500, 2000]
    accumulator = PsdAccumulator(
        energy_bins=64,
        ratio_bins=20,
        energy_right_shift=0,
        ratio_range=(-1.0, 1.0),
    )

    accumulator.add_events(events)

    assert accumulator.statistics.accepted == 2
    assert accumulator.matrix.sum() == 2


def test_display_event_buffer_is_bounded_and_reports_drops() -> None:
    buffer = McaEventBuffer(max_batches=2)
    first = np.zeros(3, dtype=_LM_EVENT_DTYPE)
    second = np.zeros(4, dtype=_LM_EVENT_DTYPE)
    third = np.zeros(5, dtype=_LM_EVENT_DTYPE)

    buffer.append(first)
    buffer.append(second)
    buffer.append(third)
    batches, dropped = buffer.drain()

    assert [len(batch) for batch in batches] == [4, 5]
    assert dropped == 3
    assert buffer.drain() == ([], 3)

    buffer.clear()
    assert buffer.drain() == ([], 0)
