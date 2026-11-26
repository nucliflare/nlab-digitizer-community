"""Deterministic two-channel list-mode logic and bounded fan-out checks."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import ndtr

from nlab.analysis.coincidence import (
    COINCIDENCE_MATRIX_CHANNELS_PER_BIN,
    FINE_BIN_NS,
    PEAK_FIT_MIN_PAIRS,
    CoincidenceAnalyzer,
    CoincidenceSettings,
    fit_coincidence_peak,
)
from nlab.hardware.digitizer.dma import McaEventBuffer
from nlab.hardware.digitizer.iio_listmode import (
    COARSE_TICK_Q,
    IIO_LM_EVENT_DTYPE,
    TIME_Q_PER_NS,
    cfd_interpolation_samples,
)

_DTYPE = IIO_LM_EVENT_DTYPE


def _events(*rows: tuple[int, int, int]) -> np.ndarray:
    result = np.zeros(len(rows), dtype=_DTYPE)
    for index, (tick, energy, marker) in enumerate(rows):
        result[index]["timestamp"] = tick
        result[index]["trapezoid_energy"] = energy
        result[index]["marker"] = marker
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


def test_and_emits_all_pairs_inside_inclusive_gate() -> None:
    analyzer = _analyze(
        CoincidenceSettings(low_q=-6 * COARSE_TICK_Q, high_q=6 * COARSE_TICK_Q),
        _events((100, 160, 0), (200, 240, 0)),
        _events((98, 164, 0), (103, 168, 0), (206, 244, 0)),
    )
    snap = analyzer.snapshot()
    assert snap.pairs == 3
    assert snap.accepted_ch0 == 2
    assert snap.accepted_ch1 == 3
    assert snap.delay_counts[4] == 1  # -2 ticks relative to CH0
    assert snap.delay_counts[9] == 1  # +3 ticks relative to CH0
    assert snap.delay_counts[12] == 1  # +6 inclusive boundary
    assert snap.ambiguous == 1
    assert snap.energy_ch1[41] == 1
    assert snap.energy_ch1[42] == 1


def test_prompt_and_delayed_random_matrices_preserve_paired_energies() -> None:
    analyzer = _analyze(
        CoincidenceSettings(low_q=-COARSE_TICK_Q, high_q=COARSE_TICK_Q),
        _events((100, 100 * 4, 0)),
        _events(
            (97, 300 * 4, 0),  # lower delayed sideband
            (100, 200 * 4, 0),  # prompt
            (103, 400 * 4, 0),  # upper delayed sideband
        ),
    )

    snapshot = analyzer.snapshot()

    assert snapshot.pairs == 1
    assert snapshot.random_pairs == 2
    x = 100 // COINCIDENCE_MATRIX_CHANNELS_PER_BIN
    assert snapshot.prompt_matrix[200 // COINCIDENCE_MATRIX_CHANNELS_PER_BIN, x] == 1
    assert snapshot.random_matrix[300 // COINCIDENCE_MATRIX_CHANNELS_PER_BIN, x] == 1
    assert snapshot.random_matrix[400 // COINCIDENCE_MATRIX_CHANNELS_PER_BIN, x] == 1
    assert snapshot.prompt_matrix.sum() == 1
    assert snapshot.random_matrix.sum() == 2


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
        CoincidenceSettings(
            low_q=2 * COARSE_TICK_Q,
            high_q=6 * COARSE_TICK_Q,
            channel_delay_q=4 * COARSE_TICK_Q,
        ),
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


def test_zero_timestamp_and_input_tags_do_not_discard_real_events() -> None:
    analyzer = _analyze(
        CoincidenceSettings(),
        _events((0, 40, 0), (100, 160, 0x82)),
        _events((101, 200, 0xC8)),
    )
    snap = analyzer.snapshot()
    assert snap.pairs == 1
    assert snap.zero_timestamps == 1
    assert snap.cfd_valid == 1
    assert snap.psd_zc_valid == 1


def test_negative_cfd_offset_byte_is_not_an_input_marker() -> None:
    ch0 = _events((100, 160, 0x82))
    ch0["zc_offset"] = 0xFD
    ch0["zc_estimation"] = -4096
    ch1 = _events((97, 200, 0x82))
    ch1["zc_offset"] = 0xFE
    analyzer = _analyze(CoincidenceSettings(), ch0, ch1)
    snap = analyzer.snapshot()
    assert snap.pairs == 1
    assert snap.cfd_valid == 2
    np.testing.assert_allclose(cfd_interpolation_samples(ch0), [-0.25])


def test_psd_result_takes_priority_over_cfd_in_shared_zero_crossing_field() -> None:
    events = _events((100, 160, 0x8A), (101, 160, 0x82), (102, 160, 0x80))
    events["zc_estimation"] = [-8192, -4096, 0]
    result = cfd_interpolation_samples(events)
    assert np.isnan(result[0])
    assert result[1] == -0.25
    assert np.isnan(result[2])


def test_fine_cfd_time_uses_exact_62_5_ps_bins() -> None:
    ch0 = _events((100, 160, 0x82))
    ch0["zc_estimation"] = -8192  # -0.5 sample = -1 ns
    ch1 = _events((103, 200, 0x82))
    ch1["zc_estimation"] = -4096  # -0.25 sample = -0.5 ns

    analyzer = _analyze(CoincidenceSettings(fine_timing=True), ch0, ch1)
    snapshot = analyzer.snapshot()

    assert snapshot.pairs == 1
    assert FINE_BIN_NS == 0.0625
    assert len(snapshot.delay_counts) == 1537
    assert snapshot.delay_counts[1160] == 1  # -48 ns + 1160 * 0.0625 ns = +24.5 ns
    assert snapshot.cfd_skipped == 0


def test_petalinux_acceptance_vector_uses_unsigned_two_ns_sample_offset() -> None:
    ch0 = _events((100, 160, 0x02))
    ch0["zc_offset"] = 18
    ch1 = _events((100, 200, 0x02))
    ch1["zc_offset"] = 19
    ch1["zc_estimation"] = -4096
    settings = CoincidenceSettings(
        low_q=-TIME_Q_PER_NS,
        high_q=TIME_Q_PER_NS,
        channel_delay_q=TIME_Q_PER_NS // 2,
        fine_timing=True,
    )

    snapshot = _analyze(settings, ch0, ch1).snapshot()

    assert snapshot.pairs == 1
    assert snapshot.delay_counts[32] == 1  # calibrated +1 ns inclusive upper edge


def test_uint8_offset_boundary_is_reported_without_signed_unwrap() -> None:
    ch0 = _events((100, 160, 0x02))
    ch1 = _events((100, 200, 0x02))
    ch0["zc_offset"] = 0xFF
    ch1["zc_offset"] = 0
    settings = CoincidenceSettings(
        low_q=-600 * TIME_Q_PER_NS,
        high_q=600 * TIME_Q_PER_NS,
        fine_timing=True,
    )

    snapshot = _analyze(settings, ch0, ch1).snapshot()

    assert snapshot.pairs == 1
    assert snapshot.offset_boundary_pairs == 1
    assert snapshot.delay_counts[1440] == 1  # -510 ns, not a heuristic +2 ns unwrap


def test_fine_mode_excludes_events_without_selected_cfd_result() -> None:
    ch0 = _events((100, 160, 0x82), (200, 160, 0x80))
    ch1 = _events((103, 200, 0x82), (203, 200, 0x8A))

    snapshot = _analyze(CoincidenceSettings(fine_timing=True), ch0, ch1).snapshot()

    assert snapshot.pairs == 1
    assert snapshot.cfd_skipped == 2
    assert snapshot.cfd_valid == 2
    assert snapshot.psd_zc_valid == 1


def test_fine_mode_rejects_fraction_outside_minus_one_to_zero_samples() -> None:
    ch0 = _events((100, 160, 0x02), (200, 160, 0x02))
    ch1 = _events((100, 200, 0x02), (200, 200, 0x02))
    ch0["zc_estimation"] = [-16385, -16384]

    snapshot = _analyze(CoincidenceSettings(fine_timing=True), ch0, ch1).snapshot()

    assert snapshot.pairs == 1
    assert snapshot.fine_out_of_range == 1
    assert snapshot.cfd_skipped == 1


def test_fine_correction_reorders_close_events_across_dma_batches() -> None:
    settings = CoincidenceSettings(low_q=-1, high_q=1, fine_timing=True)
    analyzer = CoincidenceAnalyzer(settings)
    ch0_first = _events((100, 160, 0x82))
    ch0_first["zc_offset"] = 2
    ch0_late = _events((100, 164, 0x82))
    analyzer.add_batch(0, ch0_first)
    analyzer.add_batch(0, ch0_late)
    ch1 = _events((100, 200, 0x82), (100, 204, 0x82))
    ch1[1]["zc_offset"] = 2
    analyzer.add_batch(1, ch1)
    analyzer.finish()

    snapshot = analyzer.snapshot()
    assert snapshot.pairs == 2
    assert snapshot.delay_counts[0] == 2  # both fine delays are exactly zero


def test_fine_match_uses_opposite_event_time_watermark() -> None:
    analyzer = CoincidenceAnalyzer(CoincidenceSettings(fine_timing=True))
    analyzer.add_batch(0, _events((100, 160, 0x82)))
    analyzer.add_batch(1, _events((100, 200, 0x82), (1000, 200, 0x82)))

    assert analyzer.snapshot().pairs == 1


def test_exact_integer_timing_preserves_adjacent_ticks_above_two_to_53() -> None:
    origin = 2**53 + 10
    settings = CoincidenceSettings(
        low_q=0,
        high_q=COARSE_TICK_Q,
        fine_timing=True,
    )
    snapshot = _analyze(
        settings,
        _events((origin, 160, 0x02)),
        _events((origin + 1, 200, 0x02)),
    ).snapshot()

    assert snapshot.pairs == 1
    assert snapshot.delay_counts[128] == 1


def test_gaussian_peak_fit_reports_sub_nanosecond_fwhm() -> None:
    settings = CoincidenceSettings(
        low_q=-4 * TIME_Q_PER_NS,
        high_q=4 * TIME_Q_PER_NS,
        fine_timing=True,
    )
    centers = (
        settings.low_ns
        + (np.arange((settings.high_q - settings.low_q) // settings.bin_width_q + 1) + 0.5)
        * settings.bin_width_ns
    )
    sigma_ns = 0.2
    expected = 4.0 + 500.0 * np.exp(-0.5 * ((centers - 0.25) / sigma_ns) ** 2)

    fit = fit_coincidence_peak(np.rint(expected).astype(np.uint64), settings)

    assert fit is not None
    assert fit.center_ns == pytest.approx(0.25, abs=0.01)
    assert fit.fwhm_ns == pytest.approx(0.47096, abs=0.02)
    assert fit.signal_counts > 1_000
    assert fit.reduced_chi_square < 1.0


def test_peak_fit_rejects_sparse_and_coarse_histograms() -> None:
    fine = CoincidenceSettings(fine_timing=True)
    sparse = np.zeros((fine.high_q - fine.low_q) // fine.bin_width_q + 1, dtype=np.uint64)
    sparse[len(sparse) // 2] = PEAK_FIT_MIN_PAIRS - 1

    assert fit_coincidence_peak(sparse, fine) is None
    assert (
        fit_coincidence_peak(np.array([0, 200, 0], dtype=np.uint64), CoincidenceSettings()) is None
    )


def _fine_centers(settings: CoincidenceSettings) -> np.ndarray:
    count = (settings.high_q - settings.low_q) // settings.bin_width_q + 1
    return settings.low_ns + (np.arange(count) + 0.5) * settings.bin_width_ns


def _binned_peak(centers: np.ndarray, total: float, center: float, sigma: float) -> np.ndarray:
    upper = (centers + 0.5 * FINE_BIN_NS - center) / sigma
    lower = (centers - 0.5 * FINE_BIN_NS - center) / sigma
    return np.asarray(total * (ndtr(upper) - ndtr(lower)), dtype=np.float64)


def test_single_bin_peak_reports_its_centre_and_an_fwhm_upper_limit() -> None:
    fine = CoincidenceSettings(fine_timing=True)
    counts = np.zeros(len(_fine_centers(fine)), dtype=np.uint64)
    counts[len(counts) // 2] = 500

    fit = fit_coincidence_peak(counts, fine)

    assert fit is not None and not fit.width_resolved
    assert fit.center_ns == pytest.approx(_fine_centers(fine)[len(counts) // 2], abs=0.01)
    assert fit.signal_counts == pytest.approx(500, rel=0.01)
    assert fit.fwhm_upper_limit_ns == pytest.approx(2.3548 * 0.5 * FINE_BIN_NS, rel=1e-3)
    assert math.isnan(fit.fwhm_uncertainty_ns)


def test_narrow_peak_is_fitted_beside_broad_structure() -> None:
    # 2026-11-26 generator run on .128 (fine CFD, +/-48 ns): a line one or two
    # bins wide at -21.9 ns next to broad humps at -23, -15 and -12 ns. The
    # fixed +/-4 ns window fitted the -23 ns hump instead (FWHM 1421 ps).
    settings = CoincidenceSettings(
        low_q=-48 * TIME_Q_PER_NS, high_q=48 * TIME_Q_PER_NS, fine_timing=True,
    )
    centers = _fine_centers(settings)
    expected = 0.5 + sum(
        _binned_peak(centers, total, center, fwhm / 2.3548)
        for total, center, fwhm in ((16_300, -23.06, 1.42), (6_000, -15.0, 0.6),
                                    (5_000, -12.0, 2.5))
    )
    true_center = -21.9 + 0.3 * FINE_BIN_NS
    expected += _binned_peak(centers, 45_000, true_center, 0.015)
    counts = np.random.default_rng(3).poisson(expected).astype(np.uint64)

    fit = fit_coincidence_peak(counts, settings)

    assert fit is not None
    assert fit.center_ns == pytest.approx(true_center, abs=0.005)
    assert not fit.width_resolved and fit.fwhm_upper_limit_ns < 0.075
    assert fit.signal_counts == pytest.approx(45_000, rel=0.02)
    assert -22.7 < fit.fit_low_ns and fit.fit_high_ns < -21.1  # The hump stays outside.


def test_peak_about_one_bin_wide_is_resolved_without_bin_centre_bias() -> None:
    settings = CoincidenceSettings(
        low_q=-4 * TIME_Q_PER_NS, high_q=4 * TIME_Q_PER_NS, fine_timing=True,
    )
    centers = _fine_centers(settings)
    sigma = 0.8 * FINE_BIN_NS
    expected = 2.0 + _binned_peak(centers, 20_000, 0.37, sigma)
    counts = np.random.default_rng(5).poisson(expected).astype(np.uint64)

    fit = fit_coincidence_peak(counts, settings)

    assert fit is not None and fit.width_resolved
    assert fit.fwhm_ns == pytest.approx(2.3548 * sigma, rel=0.05)
    assert fit.center_ns == pytest.approx(0.37, abs=0.003)
    assert fit.reduced_chi_square < 2.0


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


@pytest.mark.parametrize(("operator", "fits"), [("AND", True), ("OR", False)])
def test_analysis_thread_fits_the_drained_snapshot_off_the_caller_thread(
    monkeypatch: pytest.MonkeyPatch, operator: str, fits: bool,
) -> None:
    import threading

    from nlab.workers import coincidence_worker
    from nlab.workers.coincidence_worker import CoincidenceAnalysisThread

    fit_threads: list[str] = []
    sentinel = object()

    def fake_fit(delay_counts: np.ndarray, settings: CoincidenceSettings) -> object:
        fit_threads.append(threading.current_thread().name)
        return sentinel

    monkeypatch.setattr(coincidence_worker, "fit_coincidence_peak", fake_fit)
    ch0, ch1 = McaEventBuffer(), McaEventBuffer()
    ch0.append(_events((100, 40, 0), (900, 40, 0)))
    ch1.append(_events((101, 40, 0), (901, 40, 0)))
    worker = CoincidenceAnalysisThread(ch0, ch1, CoincidenceSettings(operator=operator))
    worker.start()
    worker.request_stop()
    worker.join(timeout=10)

    snapshot, error = worker.result()
    assert error is None
    if fits:
        assert snapshot.pairs == 2
        assert fit_threads and set(fit_threads) == {"coincidence-analysis"}
        assert worker.timing_fit() is sentinel
    else:
        assert fit_threads == []
        assert worker.timing_fit() is None
