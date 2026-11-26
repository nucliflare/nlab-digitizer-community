"""Waveform PSD normalisation, windows, averaging and resets (no Qt)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import periodogram

from nlab.analysis.waveform_spectrum import (
    SCALLOPING_LOSS_DB,
    AccumulatedSpectrum,
    SpectrumAveraging,
    SpectrumUnit,
    SpectrumWindow,
    compensated_psd,
    decimation_factor,
    find_spurs,
    format_frequency,
    frame_psd,
    mean_decimation_response,
    parseval_check,
    to_display_units,
    tone_amplitude,
    write_spectrum_csv,
)

PERIOD_NS = 8.0  # Scope viewer mean4 points
FS = 1e9 / PERIOD_NS


def _tone(n: int, cycles: float, amplitude: float, phase: float = 0.3) -> np.ndarray:
    return amplitude * np.cos(2 * np.pi * cycles * np.arange(n) / n + phase)


@pytest.mark.parametrize("window", list(SpectrumWindow))
@pytest.mark.parametrize("n", [2048, 1023])
def test_frame_psd_equals_scipy_periodogram_density(window: SpectrumWindow, n: int) -> None:
    samples = np.random.default_rng(5).normal(100, 3, n)
    spectrum = frame_psd(samples, PERIOD_NS, window=window)
    frequencies, expected = periodogram(
        samples, FS, window=window.value, scaling="density",
        detrend="constant", return_onesided=True,
    )
    np.testing.assert_allclose(spectrum.frequencies, frequencies)
    np.testing.assert_allclose(spectrum.psd, expected, rtol=1e-10, atol=1e-18)
    assert spectrum.fs == FS
    assert spectrum.delta_f == pytest.approx(FS / n)
    assert spectrum.enbw == pytest.approx(FS * spectrum.s2 / spectrum.s1**2)


def test_window_metadata_matches_published_noise_bandwidths() -> None:
    samples = np.zeros(4096)
    samples[0] = 1
    expected = {"boxcar": 1.0, "hann": 1.5, "blackmanharris": 2.0044, "flattop": 3.7702}
    for window, bins in expected.items():
        assert frame_psd(samples, PERIOD_NS, window=window).enbw_bins == pytest.approx(
            bins, abs=1e-3
        )


@pytest.mark.parametrize("cycles", [100.0, 100.5, 100.25])
def test_flattop_reads_tone_amplitude_within_a_tenth_of_a_db(cycles: float) -> None:
    n, amplitude = 2048, 1000.0
    spectrum = frame_psd(_tone(n, cycles, amplitude), PERIOD_NS, window="flattop")
    measured = tone_amplitude(spectrum, cycles * FS / n)
    assert abs(20 * math.log10(measured / amplitude)) < 0.1


def test_hann_scalloping_stays_within_its_documented_loss() -> None:
    n, amplitude = 2048, 1000.0
    centred = frame_psd(_tone(n, 100.0, amplitude), PERIOD_NS, window="hann")
    assert tone_amplitude(centred, 100 * FS / n) == pytest.approx(amplitude, rel=1e-6)
    off_bin = frame_psd(_tone(n, 100.5, amplitude), PERIOD_NS, window="hann")
    loss_db = -20 * math.log10(tone_amplitude(off_bin, 100.5 * FS / n) / amplitude)
    assert 1.0 < loss_db <= SCALLOPING_LOSS_DB[SpectrumWindow.HANN] + 0.02
    assert SCALLOPING_LOSS_DB[SpectrumWindow.FLATTOP] < 0.1


def test_white_noise_level_and_parseval_ratio() -> None:
    sigma, n, frames = 3.0, 2048, 200
    rng = np.random.default_rng(7)
    accumulated = AccumulatedSpectrum(average_count=0)
    for _ in range(frames):
        assert accumulated.add(rng.normal(0, sigma, n), source="test", sample_period_ns=PERIOD_NS)
    mean = accumulated.result("mean")
    assert mean is not None and mean.frames == frames
    level = mean.psd[2:-1].mean()
    assert level == pytest.approx(2 * sigma**2 / FS, rel=0.05)
    assert parseval_check(mean) == pytest.approx(1.0, abs=0.05)
    assert accumulated.relative_std == pytest.approx(1 / math.sqrt(frames))


def test_linear_power_average_avoids_the_db_averaging_bias() -> None:
    # Keep this: averaging log power reads ~2.5 dB low for noise (Euler's gamma
    # times 10 log10 e for a two-degree-of-freedom bin), so never "simplify" it.
    sigma, n, frames = 3.0, 1024, 400
    rng = np.random.default_rng(11)
    accumulated = AccumulatedSpectrum(average_count=0)
    db_sum = np.zeros(n // 2 + 1)
    for _ in range(frames):
        samples = rng.normal(0, sigma, n)
        accumulated.add(samples, source="test", sample_period_ns=PERIOD_NS)
        db_sum += to_display_units(frame_psd(samples, PERIOD_NS).psd, "db")
    expected_db = 10 * math.log10(2 * sigma**2 / FS)
    mean = accumulated.result("mean")
    assert mean is not None
    linear_db = 10 * math.log10(mean.psd[2:-1].mean())
    averaged_db = float((db_sum / frames)[2:-1].mean())
    assert linear_db == pytest.approx(expected_db, abs=0.2)
    assert expected_db - averaged_db == pytest.approx(10 * math.log10(math.e) * 0.5772, abs=0.2)


def test_length_period_source_and_trigger_changes_restart_with_a_reason() -> None:
    accumulated = AccumulatedSpectrum()
    noise = np.random.default_rng(1).normal(0, 1, 512)
    accumulated.add(noise, source="viewer", sample_period_ns=8, trigger_mode="Periodic")
    accumulated.add(noise, source="viewer", sample_period_ns=8, trigger_mode="Periodic")
    assert accumulated.frames == 2
    accumulated.add(noise[:256], source="viewer", sample_period_ns=8, trigger_mode="Periodic")
    assert accumulated.frames == 1
    assert "frame length 512 → 256" in accumulated.reset_reason
    accumulated.add(noise[:256], source="viewer", sample_period_ns=2, trigger_mode="Periodic")
    assert accumulated.frames == 1 and "sample period 8 → 2 ns" in accumulated.reset_reason
    accumulated.add(noise[:256], source="DMA", sample_period_ns=2, trigger_mode="Periodic")
    assert accumulated.frames == 1 and "source viewer → DMA" in accumulated.reset_reason
    accumulated.add(noise[:256], source="DMA", sample_period_ns=2, trigger_mode="Rising edge")
    assert accumulated.frames == 1 and "trigger mode" in accumulated.reset_reason
    result = accumulated.result()
    assert result is not None and result.n == 256 and result.sample_period_ns == 2
    accumulated.clear()
    assert accumulated.result() is None and accumulated.frames == 0


def test_max_and_min_hold_track_linear_power_per_bin() -> None:
    n = 1024
    accumulated = AccumulatedSpectrum(averaging="max")
    singles = []
    for amplitude in (10.0, 100.0, 30.0):
        samples = _tone(n, 64, amplitude) + np.random.default_rng(int(amplitude)).normal(0, 1, n)
        accumulated.add(samples, source="test", sample_period_ns=PERIOD_NS)
        singles.append(frame_psd(samples, PERIOD_NS).psd)
    maximum, minimum, mean = (accumulated.result(trace) for trace in ("max", "min", "mean"))
    assert maximum is not None and minimum is not None and mean is not None
    np.testing.assert_allclose(maximum.psd, np.max(singles, axis=0))
    np.testing.assert_allclose(minimum.psd, np.min(singles, axis=0))
    np.testing.assert_allclose(mean.psd, np.mean(singles, axis=0))
    display = accumulated.result()
    assert display is not None
    np.testing.assert_array_equal(display.psd, maximum.psd)
    accumulated.set_averaging("min")
    assert accumulated.frames == 3  # Same linear mean: no restart.
    display = accumulated.result()
    assert display is not None
    np.testing.assert_array_equal(display.psd, minimum.psd)


def test_exponential_average_uses_alpha_one_over_n_after_warm_up() -> None:
    n = 512
    first, second = _tone(n, 32, 100.0), _tone(n, 32, 300.0)
    psd_first = frame_psd(first, PERIOD_NS).psd
    psd_second = frame_psd(second, PERIOD_NS).psd
    accumulated = AccumulatedSpectrum(averaging="exponential", average_count=4)
    for _ in range(6):
        accumulated.add(first, source="test", sample_period_ns=PERIOD_NS)
    accumulated.add(second, source="test", sample_period_ns=PERIOD_NS)
    mean = accumulated.result("mean")
    assert mean is not None
    np.testing.assert_allclose(mean.psd, 0.75 * psd_first + 0.25 * psd_second)
    assert accumulated.effective_averages == 7  # min(K, 2N - 1)
    accumulated.set_averaging("linear")
    assert accumulated.frames == 0 and "averaging" in accumulated.reset_reason


def test_linear_average_holds_after_n_frames() -> None:
    accumulated = AccumulatedSpectrum(average_count=3)
    noise = np.random.default_rng(3).normal(0, 1, (5, 256))
    added = [accumulated.add(row, source="test", sample_period_ns=PERIOD_NS) for row in noise]
    assert added == [True, True, True, False, False]
    assert accumulated.frames == 3 and accumulated.complete
    accumulated.set_average_count(10)
    assert accumulated.frames == 0 and not accumulated.complete


@pytest.mark.parametrize("frequency,expected", [
    (0.0, 1.0), (31.25e6, 0.9061), (62.5e6, 0.6533),
])
def test_mean4_response_at_dc_quarter_rate_and_nyquist(frequency: float, expected: float) -> None:
    assert float(mean_decimation_response(frequency, 4, 2.0)[()]) == pytest.approx(
        expected, abs=1e-4
    )
    assert decimation_factor(8.0) == 4 and decimation_factor(2.0) == 1
    assert decimation_factor(3.0) == 1


def test_compensation_never_divides_where_the_response_is_weak() -> None:
    raw = frame_psd(np.random.default_rng(2).normal(0, 1, 1024), 2.0)  # 2 ns, to 250 MHz
    response = mean_decimation_response(raw.frequencies, 4, 2.0)
    corrected = compensated_psd(raw, 4, 2.0)
    weak = response < 0.1
    assert weak.any()  # The mean4 zero at 125 MHz lies on this grid.
    np.testing.assert_array_equal(corrected[weak], raw.psd[weak])
    strong = ~weak
    np.testing.assert_allclose(corrected[strong], raw.psd[strong] / response[strong] ** 2)
    assert np.isfinite(corrected).all()


@pytest.mark.parametrize("samples,period", [
    (np.array([]), 8.0),
    (np.zeros(15), 8.0),
    (np.r_[np.zeros(20), np.nan], 8.0),
    (np.r_[np.zeros(20), np.inf], 8.0),
    (np.zeros(64), 0.0),
    (np.zeros(64), -8.0),
    (np.zeros(64), math.nan),
    (np.zeros(64), math.inf),
])
def test_invalid_frames_and_periods_are_rejected(samples: np.ndarray, period: float) -> None:
    with pytest.raises(ValueError):
        frame_psd(samples, period)
    accumulated = AccumulatedSpectrum()
    assert not accumulated.add(samples, source="test", sample_period_ns=period)
    assert accumulated.frames == 0 and accumulated.rejected_frames == 1
    assert accumulated.last_rejection


def test_settings_changes_restart_the_average() -> None:
    accumulated = AccumulatedSpectrum()
    noise = np.random.default_rng(4).normal(0, 1, 256)
    for change in (
        lambda: accumulated.set_window("flattop"),
        lambda: accumulated.set_detrend("linear"),
        lambda: accumulated.set_average_count(7),
    ):
        accumulated.add(noise, source="test", sample_period_ns=PERIOD_NS)
        change()
        assert accumulated.frames == 0 and accumulated.reset_reason
    with pytest.raises(ValueError):
        accumulated.set_average_count(-1)


def test_spurs_are_found_above_the_median_floor() -> None:
    n = 4096
    rng = np.random.default_rng(9)
    samples = rng.normal(0, 2, n) + _tone(n, 410.3, 40.0) + _tone(n, 1201.0, 8.0) + 500
    spurs = find_spurs(frame_psd(samples, PERIOD_NS), threshold_db_over_median=10, max_count=5)
    assert len(spurs) >= 2
    delta_f = FS / n
    assert spurs[0].frequency == pytest.approx(410.3 * delta_f, abs=0.5 * delta_f)
    assert any(abs(s.frequency - 1201 * delta_f) < 0.5 * delta_f for s in spurs)
    assert spurs[0].excess_db > spurs[-1].excess_db or len(spurs) == 1
    assert all(spur.bin > 0 and spur.width_bins <= 8 for spur in spurs)
    assert find_spurs(frame_psd(rng.normal(0, 2, n), PERIOD_NS), threshold_db_over_median=20) == []


def test_display_units_and_frequency_format() -> None:
    psd = np.array([4.0, 0.0, 1e-4])
    np.testing.assert_allclose(to_display_units(psd, SpectrumUnit.PSD), psd)
    np.testing.assert_allclose(to_display_units(psd, SpectrumUnit.ASD), [2.0, 0.0, 1e-2])
    decibels = to_display_units(psd, SpectrumUnit.DB)
    assert decibels[0] == pytest.approx(10 * math.log10(4))
    assert np.isfinite(decibels).all()
    assert format_frequency(61_035.15625) == "61.04 kHz"
    assert format_frequency(125e6) == "125 MHz"
    assert format_frequency(50) == "50 Hz"


def test_csv_export_has_metadata_header_and_four_columns(tmp_path: Path) -> None:
    accumulated = AccumulatedSpectrum(averaging=SpectrumAveraging.LINEAR)
    for seed in range(3):
        accumulated.add(
            np.random.default_rng(seed).normal(0, 1, 128), source="Scope viewer (8 ns mean4)",
            sample_period_ns=PERIOD_NS, trigger_mode="Periodic",
        )
    path = tmp_path / "spectrum.csv"
    write_spectrum_csv(path, accumulated, {"channel": 1, "app_version": "test"})
    lines = path.read_text(encoding="utf-8").splitlines()
    header = {line[2:].split(": ", 1)[0]: line[2:].split(": ", 1)[1]
              for line in lines if line.startswith("# ") and ": " in line}
    for key in ("source", "sample_rate_hz", "points_n", "window", "enbw_hz", "frames_k",
                "trigger_mode", "channel", "app_version"):
        assert key in header
    assert header["frames_k"] == "3" and header["trigger_mode"] == "Periodic"
    rows = list(csv.reader(line for line in lines if not line.startswith("#")))
    assert rows[0] == ["frequency_hz", "mean_psd", "max_psd", "min_psd"]
    values = np.array(rows[1:], dtype=float)
    assert values.shape == (65, 4)
    assert (values[:, 2] >= values[:, 1] * (1 - 1e-6)).all()
    assert (values[:, 3] <= values[:, 1] * (1 + 1e-6)).all()
    with pytest.raises(ValueError):
        write_spectrum_csv(tmp_path / "empty.csv", AccumulatedSpectrum(), {})
