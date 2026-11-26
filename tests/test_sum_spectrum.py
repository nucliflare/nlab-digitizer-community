"""Spectrum of periodic-sum window means: runs, Welch average, bridging, files."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from nlab.analysis import current_capture as reader
from nlab.analysis.sum_spectrum import (
    SumSpectrum,
    SumWelch,
    capture_sum_spectrum,
    sum_interval_ns,
    window_mean_response,
    windows_for_resolution,
)
from nlab.analysis.waveform_spectrum import (
    SCALLOPING_LOSS_DB,
    SpectrumWindow,
    find_spurs,
    parseval_check,
    tone_amplitude,
)
from nlab.hardware.digitizer.scope_sums import MASK64, decode_sum_packet, open_sum_writer
from nlab.help.context import resolve_context
from nlab.views.current_capture_dialog import CurrentCaptureDialog

WINDOW = 256  # ADC samples per window; gap 0 -> 64 ticks = 512 ns
INTERVAL_TICKS = WINDOW // 4
FS = 1e9 / 512


def _welch(length: int, *, bridge: bool = False, gap: int = 0) -> SumWelch:
    return SumWelch(window_samples=WINDOW, gap_cycles=gap, segment_length=length,
                    bridge_single_gaps=bridge)


def _stamps(count: int, missing_after: dict[int, int] | None = None) -> np.ndarray:
    """Native timestamps of ``count`` recorded windows; skip windows after given indices."""
    steps = np.full(count, INTERVAL_TICKS, dtype=np.int64)
    steps[0] = 0
    for index, missing in (missing_after or {}).items():
        steps[index + 1] += missing * INTERVAL_TICKS
    return ((1 << 60) + np.cumsum(steps)).astype(np.uint64)


def _feed(welch: SumWelch, stamps: np.ndarray, values: np.ndarray, block: int = 508) -> None:
    for start in range(0, stamps.size, block):
        welch.append(stamps[start:start + block], values[start:start + block])


def test_interval_response_and_resolution_requirement() -> None:
    assert sum_interval_ns(256, 0) == 512
    assert sum_interval_ns(256, 12_500) == (64 + 12_500) * 8
    # A boxcar covering the whole interval (D = 1): sinc(1/2) at fs'/2.
    assert float(window_mean_response(FS / 2, WINDOW)[()]) == pytest.approx(2 / math.pi, rel=1e-3)
    assert float(window_mean_response(FS, WINDOW)[()]) == pytest.approx(0, abs=1e-9)
    assert windows_for_resolution(10, 512) == 195_313  # 50 Hz needs ~10 Hz bins
    assert windows_for_resolution(FS / 4096, 512) == 4096


def test_injected_sinusoid_is_recovered_at_its_frequency_and_amplitude() -> None:
    count, length = 200_000, 4096
    rng = np.random.default_rng(3)
    t = np.arange(count) * 512e-9
    frequency, amplitude = 50_000.0, 3.0
    values = 100 + amplitude * np.sin(2 * np.pi * frequency * t) + rng.normal(0, 1, count)
    welch = _welch(length)
    _feed(welch, _stamps(count), values)
    result = welch.finish()
    spectrum = result.spectrum
    assert spectrum is not None
    assert result.delta_f == pytest.approx(FS / length)
    spurs = find_spurs(spectrum, threshold_db_over_median=20)
    assert spurs and spurs[0].frequency == pytest.approx(frequency, abs=result.delta_f / 2)
    loss = -20 * math.log10(tone_amplitude(spectrum, frequency) / amplitude)
    assert -0.05 < loss <= SCALLOPING_LOSS_DB[SpectrumWindow.HANN] + 0.05
    assert result.segments == 1 + (count - length) // (length // 2)
    assert result.runs == result.runs_used == 1


def test_white_noise_level_parseval_and_overlap_averages() -> None:
    sigma, count, length = 2.0, 300_000, 1024
    values = np.random.default_rng(8).normal(0, sigma, count)
    welch = _welch(length)
    _feed(welch, _stamps(count), values)
    result = welch.finish()
    spectrum = result.spectrum
    assert spectrum is not None
    assert spectrum.psd[2:-1].mean() == pytest.approx(2 * sigma**2 / FS, rel=0.05)
    assert parseval_check(spectrum) == pytest.approx(1, abs=0.05)
    # 50 % Hann overlap: correlated neighbours make K_eff a little below K.
    assert 0.9 * result.segments < spectrum.averages < result.segments


def test_runs_split_at_missed_windows_and_short_runs_are_discarded() -> None:
    lengths, gaps = (1000, 300, 2000), (1, 3)
    count = sum(lengths)
    stamps = _stamps(count, {999: gaps[0], 1299: gaps[1]})
    welch = _welch(512)
    _feed(welch, stamps, np.random.default_rng(1).normal(0, 1, count))
    result = welch.finish()
    assert (result.runs, result.runs_used, result.longest_run) == (3, 2, 2000)
    assert (result.gaps, result.missing_windows, result.single_gaps) == (2, 4, 1)
    # 1000 -> 2 segments covering 768; 300 -> none; 2000 -> 6 covering 1792.
    assert result.segments == 8
    assert result.used_windows == 768 + 1792
    assert result.discarded_windows == 232 + 300 + 208
    assert result.windows == count and result.bridged_windows == 0
    assert result.message == ""


def test_bridging_counts_flags_and_matches_the_complete_series() -> None:
    count = 3000
    ramp = np.arange(count, dtype=float) * 0.25 + np.sin(np.arange(count) / 40)
    removed = (700, 1600, 2400)  # isolated single windows
    kept = np.setdiff1d(np.arange(count), removed)
    full_stamps = _stamps(count)
    stamps, values = full_stamps[kept], ramp[kept]
    off = _welch(256)
    _feed(off, stamps, values)
    split = off.finish()
    assert split.runs == 4 and split.bridged_windows == 0 and split.single_gaps == 3
    on = _welch(256, bridge=True)
    _feed(on, stamps, values, block=97)  # Gaps fall inside and between blocks.
    bridged = on.finish()
    assert bridged.runs == 1 and bridged.bridged_windows == 3 and bridged.missing_windows == 3
    np.testing.assert_array_equal(bridged.bridged_timestamps, full_stamps[list(removed)])
    # A linear interpolation of a slow signal reproduces it closely.
    complete = _welch(256)
    _feed(complete, full_stamps, ramp)
    reference = complete.finish().spectrum
    assert reference is not None and bridged.spectrum is not None
    np.testing.assert_allclose(bridged.spectrum.psd, reference.psd, rtol=2e-3, atol=1e-12)


def test_multi_window_gaps_are_never_bridged() -> None:
    welch = _welch(16, bridge=True)
    _feed(welch, _stamps(100, {49: 2}), np.zeros(100))
    result = welch.finish()
    assert result.runs == 2 and result.bridged_windows == 0 and result.missing_windows == 2


def test_unreachable_resolution_explains_the_limit_and_bridging() -> None:
    # The IP124 gap-0 pattern: one window skipped per 508-window packet.
    count = 508 * 6
    stamps = _stamps(count, {index: 1 for index in range(507, count - 1, 508)})
    values = np.random.default_rng(2).normal(0, 1, count)
    welch = _welch(1024)
    _feed(welch, stamps, values)
    result = welch.finish()
    assert result.spectrum is None and result.longest_run == 508
    assert "longest gapless run has 508" in result.message
    assert "L = 256" in result.message and "Bridging the 5" in result.message
    bridged = _welch(1024, bridge=True)
    _feed(bridged, stamps, values)
    joined = bridged.finish()
    assert joined.spectrum is not None and joined.runs == 1 and joined.bridged_windows == 5


@pytest.mark.parametrize("length", [0, 15, 100, 1 << 25])
def test_segment_length_must_be_a_power_of_two_in_range(length: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        _welch(length)


def test_off_grid_or_nonfinite_input_is_rejected() -> None:
    welch = _welch(16)
    with pytest.raises(ValueError, match="off the interval grid"):
        welch.append(np.array([100, 100 + INTERVAL_TICKS + 1], dtype=np.uint64), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        _welch(16).append(_stamps(2), np.array([0.0, np.nan]))
    wrapped = _welch(16)
    start = MASK64 - 2 * INTERVAL_TICKS + 1
    stamps = ((start + np.arange(64, dtype=object) * INTERVAL_TICKS) & MASK64).astype(np.uint64)
    _feed(wrapped, stamps, np.zeros(64))
    assert wrapped.finish().runs == 1  # The native counter wrap is not a gap.


def _write_capture(path: Path, values: np.ndarray, skip_per_packet: int) -> None:
    """IP124 packets of 508 windows; ``skip_per_packet`` windows lost between packets."""
    writer = open_sum_writer(path, {"channel": 0})
    stamp, sequence, n = (1 << 60) + 7, 0, 16
    try:
        for row in range(0, values.size, 508):
            size = min(508, values.size - row)
            raw = bytearray(4096)
            struct.pack_into(
                "<4Q", raw, 0, stamp & MASK64, 0x32534D5553504456, n | size << 48,
                sequence | skip_per_packet * sequence << 32 | int(row + size == values.size) << 63,
            )
            for j in range(508):
                total = round((values[row + min(j, size - 1)] + 32768) * n)
                struct.pack_into("<Q", raw, 32 + 8 * j, total)
            writer.append(decode_sum_packet(bytes(raw)))
            stamp += (size + skip_per_packet) * 4
            sequence += 1
    finally:
        writer.close(complete=True, missed_windows=skip_per_packet * sequence)


@pytest.mark.parametrize("suffix", [".h5", ".bin"])
def test_recorded_capture_with_packet_skips_recovers_a_low_frequency_line(
    tmp_path: Path, suffix: str,
) -> None:
    # N = 16 windows at gap 0: 32 ns interval, fs' = 31.25 MHz. Values are
    # quantised to 1/16 code by the integer sums.
    count = 508 * 40
    t = np.arange(count) * 32e-9
    values = 2.0 * np.sin(2 * np.pi * 200e3 * t) + np.random.default_rng(4).normal(0, 0.5, count)
    path = tmp_path / f"capture{suffix}"
    _write_capture(path, values, skip_per_packet=1)
    info = reader.read_current_capture_info(path)
    progress: list[float] = []
    split = capture_sum_spectrum(info, segment_length=256, progress=progress.append)
    assert split.runs == 40 and split.missing_windows == 39 and split.longest_run == 508
    assert progress[-1] == pytest.approx(1.0)
    assert capture_sum_spectrum(info, segment_length=1024).spectrum is None
    joined = capture_sum_spectrum(info, segment_length=4096, bridge_single_gaps=True)
    assert joined.runs == 1 and joined.bridged_windows == 39
    assert joined.spectrum is not None
    spurs = find_spurs(joined.spectrum, threshold_db_over_median=20)
    assert spurs[0].frequency == pytest.approx(200e3, abs=joined.delta_f)


def _open_dialog(qtbot: QtBot, path: Path) -> CurrentCaptureDialog:
    dialog = CurrentCaptureDialog()
    qtbot.addWidget(dialog)
    dialog.open_path(path)
    qtbot.waitUntil(lambda: dialog._info is not None, timeout=5000)
    return dialog


def _compute(
    qtbot: QtBot, dialog: CurrentCaptureDialog, length: int, *, bridge: bool = False,
) -> SumSpectrum:
    dialog.segment_combo.setCurrentIndex(dialog.segment_combo.findData(length))
    dialog.bridge_checkbox.setChecked(bridge)
    dialog.spectrum_button.click()
    qtbot.waitUntil(
        lambda: dialog._spectrum_result is not None
        and dialog._spectrum_result.segment_length == length
        and dialog._spectrum_result.bridge_single_gaps == bridge,
        timeout=10000,
    )
    assert dialog._spectrum_result is not None
    return dialog._spectrum_result


def test_dialog_spectrum_tab_reports_runs_limits_and_bridging(
    qtbot: QtBot, tmp_path: Path,
) -> None:
    count = 508 * 8
    t = np.arange(count) * 32e-9
    values = 2.0 * np.sin(2 * np.pi * 1e6 * t) + np.random.default_rng(6).normal(0, 0.5, count)
    path = tmp_path / "capture.h5"
    _write_capture(path, values, skip_per_packet=1)
    dialog = _open_dialog(qtbot, path)
    assert dialog.spectrum_button.isEnabled()
    assert "fs' = 31.25 MHz" in dialog.resolution_label.text()
    assert dialog.spectrum_stack.currentIndex() == 1  # A message, not empty axes.
    assert "press Compute spectrum" in dialog.spectrum_message.text()

    result = _compute(qtbot, dialog, 256)
    assert dialog.spectrum_stack.currentIndex() == 0
    x, _ = dialog.spectrum_curve.getData()
    assert x is not None and x.size == 127  # Without DC and the un-doubled Nyquist bin.
    assert x[-1] < result.fs / 2
    text = dialog.spectrum_summary.text()
    for part in ("Δf = 122.1 kHz", "8 gapless runs, longest 508", "7 missing in 7 gaps",
                 "Bridging off.", "Parseval", "not validated on live hardware"):
        assert part in text
    assert dialog.nyquist_line.isVisible()
    assert dialog.nyquist_line.value() == pytest.approx(result.fs / 2)
    assert dialog.response_curve.getData()[0] is not None

    _compute(qtbot, dialog, 1024)
    assert dialog.spectrum_curve.getData()[0] is None
    assert dialog.spectrum_stack.currentIndex() == 1
    assert "longest gapless run has 508" in dialog.spectrum_message.text()
    assert "Bridging the 7" in dialog.spectrum_summary.text()
    assert not dialog.bridge_recompute_button.isHidden()
    assert "Use L = 256" in dialog.shorter_segment_button.text()

    dialog.bridge_recompute_button.click()
    qtbot.waitUntil(
        lambda: dialog._spectrum_result is not None
        and dialog._spectrum_result.bridge_single_gaps, timeout=10000,
    )
    bridged = dialog._spectrum_result
    assert bridged is not None and bridged.spectrum is not None and bridged.bridged_windows == 7
    assert bridged.segment_length == 1024 and dialog.bridge_checkbox.isChecked()
    assert dialog.spectrum_stack.currentIndex() == 0
    assert "Bridged 7 isolated single missing windows" in dialog.spectrum_summary.text()
    dialog.spectrum_log_checkbox.setChecked(True)
    axis = dialog.spectrum_plot.getPlotItem().getAxis("bottom")
    assert axis._tickLevels is not None and (3.0, "1 kHz") in axis._tickLevels[0]
    dialog.spectrum_unit_combo.setCurrentIndex(dialog.spectrum_unit_combo.findData("psd"))
    assert dialog._spectrum_result is bridged  # Display changes do not recompute.
    assert dialog.spectrum_plot.getPlotItem().getAxis("left").logMode

    assert resolve_context(dialog.segment_combo).topic == "current-spectrum"
    assert resolve_context(dialog.width_spin).topic == "current"
    other = tmp_path / "other.bin"
    _write_capture(other, values[:600], skip_per_packet=0)
    dialog.open_path(other)
    assert dialog._spectrum_result is None
    assert dialog.spectrum_curve.getData()[0] is None


def test_closing_the_dialog_cancels_a_running_spectrum(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "capture.bin"
    _write_capture(path, np.zeros(508 * 50), skip_per_packet=0)
    dialog = _open_dialog(qtbot, path)
    dialog.compute_spectrum()
    worker = dialog._spectrum_worker
    assert worker is not None
    dialog.close_without_prompt()
    assert dialog._spectrum_worker is None and worker.isFinished()
