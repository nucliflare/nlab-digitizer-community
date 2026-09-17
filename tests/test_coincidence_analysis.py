"""Deterministic two-channel list-mode logic and bounded fan-out checks."""

from __future__ import annotations

import numpy as np
import pytest

from nlab.analysis.coincidence import CoincidenceAnalyzer, CoincidenceSettings
from nlab.hardware.digitizer.dma import McaEventBuffer

_DTYPE = np.dtype(
    [
        ("flags", "<u2"),
        ("cfd_q2", "<u2"),
        ("charge_energy", "<u2"),
        ("trapezoid_energy", "<u2"),
        ("timestamp", "<u8"),
    ]
)


def _events(*rows: tuple[int, int, int]) -> np.ndarray:
    result = np.zeros(len(rows), dtype=_DTYPE)
    for index, (tick, energy, flags) in enumerate(rows):
        result[index]["timestamp"] = tick
        result[index]["trapezoid_energy"] = energy
        result[index]["flags"] = flags
    return result


def _analyze(
    settings: CoincidenceSettings,
    ch0: np.ndarray,
    ch1: np.ndarray,
) -> CoincidenceAnalyzer:
    analyzer = CoincidenceAnalyzer(settings)
    analyzer.add_batch(0, ch0)
    analyzer.add_batch(1, ch1)
    analyzer.finish()
    return analyzer


def test_and_uses_signed_delay_and_one_to_one_nearest_pairing() -> None:
    analyzer = _analyze(
        CoincidenceSettings(low_tick=-6, high_tick=6),
        _events((100, 160, 0), (200, 240, 0)),
        _events((98, 164, 0), (103, 168, 0), (206, 244, 0)),
    )
    snap = analyzer.snapshot()
    assert snap.pairs == 2
    assert snap.accepted_ch0 == snap.accepted_ch1 == 2
    assert snap.delay_counts[4] == 1  # -2 ticks relative to CH0
    assert snap.delay_counts[12] == 1  # +6 inclusive boundary
    assert snap.ambiguous == 1
    assert snap.energy_ch1[41] == 1
    assert snap.energy_ch1[42] == 0  # nearest CH1=98, not 103


def test_roi_uses_fixed_dma_to_mca_scale_independent_of_binning() -> None:
    analyzer = _analyze(
        CoincidenceSettings(
            roi_ch0=(99, 101),
            roi_ch1=None,
            energy_bin_ch0=7,
            energy_bin_ch1=6,
        ),
        _events((100, 400, 0), (200, 600, 0)),
        _events((101, 2000, 0), (201, 2000, 0)),
    )
    snap = analyzer.snapshot()
    assert snap.pairs == 1
    assert snap.energy_ch0[100] == 1
    assert snap.energy_ch1[500] == 1
    assert snap.outside_roi == 1


def test_recorded_photopeak_codes_land_inside_mca_rois() -> None:
    analyzer = _analyze(
        CoincidenceSettings(
            roi_ch0=(5055, 6353),
            roi_ch1=(9611, 12201),
            energy_bin_ch0=7,
            energy_bin_ch1=6,
        ),
        _events((100, 22758, 0)),
        _events((100, 43959, 0)),
    )
    snap = analyzer.snapshot()
    assert snap.pairs == 1
    assert snap.energy_ch0[5689] == 1
    assert snap.energy_ch1[10989] == 1
    assert snap.outside_roi == 0


def test_full_u16_dma_energy_maps_to_full_mca_histogram() -> None:
    analyzer = _analyze(
        CoincidenceSettings(operator="OR", energy_bin_ch0=9),
        _events((100, 0, 0), (200, 65535, 0)),
        _events(),
    )
    snap = analyzer.snapshot()
    assert snap.energy_ch0[0] == 1
    assert snap.energy_ch0[16383] == 1
    assert snap.energy_overflow == 0


def test_asymmetric_window_and_channel_offset_use_signed_ticks() -> None:
    analyzer = _analyze(
        CoincidenceSettings(low_tick=2, high_tick=6, offset_ch1_tick=-4),
        _events((100, 40, 0)),
        _events((110, 50, 0)),
    )
    snap = analyzer.snapshot()
    assert snap.pairs == 1
    assert snap.delay_counts[-1] == 1  # CH1 110 - 4 - CH0 100 = +6 ticks


@pytest.mark.parametrize(
    ("not_ch0", "not_ch1", "expected_ch0", "expected_ch1"),
    [(True, False, 0, 1), (False, True, 1, 0)],
)
def test_veto_requires_absence_of_roi_qualified_opposite_event(
    not_ch0: bool,
    not_ch1: bool,
    expected_ch0: int,
    expected_ch1: int,
) -> None:
    analyzer = _analyze(
        CoincidenceSettings(not_ch0=not_ch0, not_ch1=not_ch1),
        _events((100, 40, 0), (200, 42, 0)),
        _events((103, 50, 0), (300, 52, 0)),
    )
    snap = analyzer.snapshot()
    assert (snap.accepted_ch0, snap.accepted_ch1) == (expected_ch0, expected_ch1)
    assert snap.pairs == 0


def test_or_counts_all_qualified_singles_and_xor_counts_only_unpaired() -> None:
    ch0 = _events((100, 160, 0), (200, 168, 0))
    ch1 = _events((103, 200, 0), (300, 208, 0))
    union = _analyze(CoincidenceSettings(operator="OR"), ch0, ch1).snapshot()
    exclusive = _analyze(CoincidenceSettings(operator="XOR"), ch0, ch1).snapshot()
    assert (union.accepted_ch0, union.accepted_ch1) == (2, 2)
    assert (exclusive.accepted_ch0, exclusive.accepted_ch1) == (1, 1)
    assert exclusive.energy_ch0[42] == 1
    assert exclusive.energy_ch1[52] == 1


def test_zero_timestamp_and_input_markers_do_not_create_false_pairs() -> None:
    analyzer = _analyze(
        CoincidenceSettings(),
        _events((0, 40, 0), (100, 40, 0), (101, 40, 0x2000)),
        _events((101, 50, 0)),
    )
    snap = analyzer.snapshot()
    assert snap.pairs == 1
    assert snap.zero_timestamps == 1
    assert snap.input_markers == 1


def test_reversed_timestamps_fail_instead_of_producing_false_results() -> None:
    analyzer = CoincidenceAnalyzer(CoincidenceSettings())
    with pytest.raises(ValueError, match="reversed"):
        analyzer.add_batch(0, _events((200, 1, 0), (100, 1, 0)))


def test_unresolved_event_limit_fails_instead_of_silently_losing_matches() -> None:
    analyzer = CoincidenceAnalyzer(CoincidenceSettings(), max_pending=1)
    with pytest.raises(RuntimeError, match="pending-event limit"):
        analyzer.add_batch(0, _events((100, 1, 0), (200, 1, 0)))


def test_psd_and_coincidence_have_independent_bounded_queues() -> None:
    psd = McaEventBuffer(max_batches=2)
    coincidence = McaEventBuffer(max_batches=2)
    psd.subscribe(coincidence)
    psd.append(_events((100, 40, 0)))
    psd_batches, _ = psd.drain()
    coincidence_batches, _ = coincidence.drain()
    assert len(psd_batches) == len(coincidence_batches) == 1
    psd.unsubscribe(coincidence)
    psd.append(_events((200, 40, 0)))
    assert coincidence.drain()[0] == []


def test_fanout_consumers_report_independent_queue_overflow() -> None:
    psd = McaEventBuffer(max_batches=2)
    coincidence = McaEventBuffer(max_batches=1)
    psd.subscribe(coincidence)
    for tick in (100, 200, 300):
        psd.append(_events((tick, 40, 0)))
    assert psd.drain()[1] == 1
    assert coincidence.drain()[1] == 2
