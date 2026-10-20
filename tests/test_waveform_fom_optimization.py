"""Waveform PSD gate-optimization tests."""

from __future__ import annotations

import struct
import weakref
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QWidget
from pytestqt.qtbot import QtBot

from nlab.analysis import waveform_fom_optimization as optimization_module
from nlab.analysis.psd_fom import PsdFomFit
from nlab.analysis.waveform_file import build_waveform_file_index
from nlab.analysis.waveform_fom_optimization import (
    WaveformFomOptimizationCancelledError,
    WaveformFomOptimizationRequest,
    WaveformFomOptimizationResult,
    WaveformGateCandidate,
    optimize_waveform_gate_ends,
)
from nlab.analysis.waveform_psd import WaveformPsdSettings
from nlab.hardware.digitizer.dma import FILE_HEADER_STRUCT, FILE_MAGIC, FILE_VERSION
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog
from nlab.workers.waveform_fom_optimization_worker import WaveformFomOptimizationWorker


def _synthetic_pulses(events: int = 6000) -> np.ndarray:
    random = np.random.default_rng(3)
    population = np.repeat((0, 1), events // 2)
    amplitude = random.normal(120.0, 8.0, events)
    pulse = random.normal(0.0, 2.0, (events, 14))
    pulse[:, :3] += amplitude[:, None] * np.asarray((30.0, 55.0, 35.0)) / 120.0
    tail = np.exp(-np.arange(11, dtype=np.float64) / 2.5)
    pulse[:, 3:] += amplitude[:, None] * tail * (
        0.035 + 0.22 * population[:, None]
    )
    return pulse


def _settings() -> WaveformPsdSettings:
    return WaveformPsdSettings(
        baseline_start=0,
        baseline_end=2,
        gate_start=2,
        short_end=5,
        long_end=12,
        polarity=1,
        energy_bins=64,
        energy_range=(0.0, 250.0),
        ratio_bins=128,
        ratio_range=(-0.1, 0.8),
    )


def _request() -> WaveformFomOptimizationRequest:
    return WaveformFomOptimizationRequest(
        settings=_settings(),
        energy_roi=(80.0, 200.0),
        fit_roi=(-0.03, 0.6),
        centroid_seeds=(0.08, 0.35),
        short_bounds=(3, 9),
        long_bounds=(7, 16),
        initial_step=4,
    )


def _write_synthetic_waveforms(path: Path, events: int) -> None:
    pulse = _synthetic_pulses(events)
    samples = np.concatenate(
        (
            np.full((len(pulse), 2), 1000.0),
            1000.0 + pulse,
        ),
        axis=1,
    ).round().astype("<i2")
    frame_samples = samples.shape[1] + 4
    with path.open("wb") as stream:
        stream.write(
            FILE_HEADER_STRUCT.pack(
                FILE_MAGIC,
                FILE_VERSION,
                0,
                0,
                0.0,
                frame_samples,
            )
        )
        for timestamp, waveform in enumerate(samples):
            stream.write(struct.pack("<Q", timestamp))
            stream.write(waveform.tobytes())


def test_discrete_gate_search_maximizes_trustworthy_local_fom() -> None:
    cumulative_charge = np.cumsum(_synthetic_pulses(), axis=1)

    result = optimize_waveform_gate_ends(cumulative_charge, _request())

    assert result.best.short_end == 5
    assert result.best.long_end == 8
    assert result.best.fom > 2.5
    assert result.best.fit.poor_convergence is False
    assert result.evaluated_candidates > result.valid_candidates > 0
    assert result.sampled_waveforms == 6000


def test_gate_search_honours_cancellation() -> None:
    cumulative_charge = np.cumsum(_synthetic_pulses(1000), axis=1)

    with pytest.raises(WaveformFomOptimizationCancelledError):
        optimize_waveform_gate_ends(
            cumulative_charge,
            _request(),
            cancelled=lambda: True,
        )


@pytest.mark.parametrize("short_bounds,long_bounds", [
    ((3, 9), (7, 16)), ((3, 9), (3, 12)), ((3, 6), (5, 12)),
    ((5, 5), (12, 12)), ((3, 9), (12, 12)),
])
def test_exhaustive_pair_count_matches_strict_short_before_long_grid(
    short_bounds: tuple[int, int], long_bounds: tuple[int, int],
) -> None:
    request = replace(_request(), short_bounds=short_bounds, long_bounds=long_bounds)
    expected = sum(
        short < long
        for short in range(short_bounds[0], short_bounds[1] + 1)
        for long in range(long_bounds[0], long_bounds[1] + 1)
    )
    assert request.total_gate_pairs == expected


def test_pair_count_does_not_enumerate_large_ranges() -> None:
    request = replace(_request(), short_bounds=(5, 1_000_000_000), long_bounds=(12, 12))
    assert request.total_gate_pairs == 7


def test_exhaustive_scan_visits_every_pair_once_and_finds_peak_missed_by_local_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    candidate_refs: list[weakref.ReferenceType[WaveformGateCandidate]] = []

    def evaluate(
        integrals: np.ndarray, request: WaveformFomOptimizationRequest,
        short: int, long: int,
    ) -> WaveformGateCandidate | None:
        calls.append((short, long))
        if (short, long) == (9, 10):
            return None  # An untrustworthy fit must not stop the full scan.
        fom = 10.0 if (short, long) == (3, 7) else 1.0 if (short, long) == (5, 12) else 0.5
        candidate = WaveformGateCandidate(short, long, fom, 1000, Mock(spec=PsdFomFit))
        candidate_refs.append(weakref.ref(candidate))
        return candidate

    monkeypatch.setattr(optimization_module, "_evaluate_candidate", evaluate)
    charges = np.ones((10, 14))
    local = optimize_waveform_gate_ends(charges, _request())
    assert (local.best.short_end, local.best.long_end) == (5, 12)
    calls.clear()
    candidate_refs.clear()
    request = replace(_request(), search_mode="exhaustive")
    progress: list[tuple[int, int]] = []
    result = optimize_waveform_gate_ends(
        charges, request,
        progress=lambda evaluated, valid, best: progress.append((evaluated, valid)),
    )
    expected = {
        (short, long)
        for short in range(3, 10)
        for long in range(7, 17)
        if short < long
    }
    assert len(calls) == len(set(calls)) == request.total_gate_pairs == len(expected)
    assert set(calls) == expected
    assert (result.best.short_end, result.best.long_end) == (3, 7)
    assert result.search_mode == "exhaustive"
    assert result.evaluated_candidates == len(expected)
    assert result.valid_candidates == len(expected) - 1
    assert progress[-1] == (len(expected), len(expected) - 1)
    assert sum(ref() is not None for ref in candidate_refs) == 1  # Only the best fit is retained.


def test_exhaustive_scan_keeps_original_gate_tie_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def evaluate(
        integrals: np.ndarray, request: WaveformFomOptimizationRequest,
        short: int, long: int,
    ) -> WaveformGateCandidate:
        return WaveformGateCandidate(short, long, 1.0, 100, Mock(spec=PsdFomFit))

    monkeypatch.setattr(optimization_module, "_evaluate_candidate", evaluate)
    result = optimize_waveform_gate_ends(
        np.ones((10, 14)), replace(_request(), search_mode="exhaustive"),
    )
    assert (result.best.short_end, result.best.long_end) == (5, 12)


@pytest.mark.parametrize("cancel_at_end", [False, True])
def test_exhaustive_scan_can_cancel_between_fits_and_after_the_final_fit(
    monkeypatch: pytest.MonkeyPatch, cancel_at_end: bool,
) -> None:
    request = replace(_request(), search_mode="exhaustive")
    evaluated = 0

    def evaluate(
        integrals: np.ndarray, request: WaveformFomOptimizationRequest,
        short: int, long: int,
    ) -> WaveformGateCandidate:
        nonlocal evaluated
        evaluated += 1
        return WaveformGateCandidate(short, long, 1.0, 100, Mock(spec=PsdFomFit))

    monkeypatch.setattr(optimization_module, "_evaluate_candidate", evaluate)
    limit = request.total_gate_pairs if cancel_at_end else 5
    with pytest.raises(WaveformFomOptimizationCancelledError):
        optimize_waveform_gate_ends(
            np.ones((10, 14)), request, cancelled=lambda: evaluated >= limit,
        )
    assert evaluated == limit


def test_exhaustive_scan_reports_no_trustworthy_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optimization_module, "_evaluate_candidate", lambda *args: None)
    with pytest.raises(ValueError, match="No gate pair"):
        optimize_waveform_gate_ends(
            np.ones((10, 14)), replace(_request(), search_mode="exhaustive"),
        )


@pytest.mark.parametrize("exhaustive", [False, True])
def test_optimization_worker_reads_waveforms_once_and_returns_best_gates(
    tmp_path: Path, exhaustive: bool,
) -> None:
    path = tmp_path / "optimization.bin"
    _write_synthetic_waveforms(path, 2000)
    index = build_waveform_file_index(path)
    results: list[object] = []
    errors: list[str] = []
    request = replace(_request(), search_mode="exhaustive" if exhaustive else "local")
    worker = WaveformFomOptimizationWorker(
        index,
        source_index=0,
        request=request,
        stride=1,
        maximum_events=2000,
    )
    worker.loaded.connect(results.append)
    worker.error.connect(errors.append)

    worker.run()

    assert errors == []
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, WaveformFomOptimizationResult)
    if exhaustive:
        assert result.evaluated_candidates == request.total_gate_pairs
        assert result.best.fom > 2.5
    else:
        assert result.best.short_end == 5
        assert result.best.long_end in {8, 9}
    assert result.sampled_waveforms == 2000
    assert result.search_mode == request.search_mode


@pytest.mark.parametrize("mode", ["local", "exhaustive", "decline"])
def test_waveform_workbench_applies_and_confirms_optimized_gates(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    path = tmp_path / "workbench-optimization.bin"
    _write_synthetic_waveforms(path, 2000)
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    dialog.open_path(path)
    qtbot.waitUntil(lambda: dialog._index_thread is None, timeout=5000)

    dialog.auto_recalculate.setChecked(False)
    period = dialog.sample_period.value()
    dialog.baseline_region.setRegion((0.0, 2 * period))
    dialog.gate_start_line.setPos(2 * period)
    dialog.short_end_line.setPos(5 * period)
    dialog.long_end_line.setPos(12 * period)
    dialog.energy_bins.setValue(64)
    dialog.energy_max.setValue(250.0)
    dialog.ratio_bins.setValue(128)
    dialog.ratio_min.setValue(-0.1)
    dialog.ratio_max.setValue(0.8)
    dialog.optimization_events.setValue(2000)
    dialog.optimization_radius.setValue(4)
    dialog._start_analysis()
    qtbot.waitUntil(lambda: dialog._analysis_thread is None, timeout=5000)

    dialog.psd_plot.energy_region.setRegion((80.0, 200.0))
    dialog.psd_plot._render_projections()
    fit_widget = dialog.psd_plot.fom_widget
    fit_widget.fit_region.setRegion((-0.03, 0.6))
    fit_widget.centroid_lines[0].setPos(0.08)
    fit_widget.centroid_lines[1].setPos(0.35)
    assert fit_widget.fit_current_projection() is not None
    assert dialog.optimize_button.isEnabled()

    assert not dialog.exhaustive_search.isChecked()
    dialog.exhaustive_search.setChecked(mode != "local")
    assert "long time" in dialog.exhaustive_search.toolTip()
    warnings: list[tuple[str, tuple[QMessageBox.StandardButton, ...]]] = []

    def confirm(
        parent: QWidget, title: str, message: str, *buttons: QMessageBox.StandardButton,
    ) -> QMessageBox.StandardButton:
        warnings.append((message, buttons))
        return (
            QMessageBox.StandardButton.Cancel if mode == "decline"
            else QMessageBox.StandardButton.Yes
        )

    monkeypatch.setattr(QMessageBox, "warning", confirm)

    qtbot.mouseClick(  # type: ignore[no-untyped-call]  # pytest-qt has no typed mouseClick.
        dialog.optimize_button, Qt.MouseButton.LeftButton,
    )
    if mode != "local":
        assert len(warnings) == 1
        assert "all 60 valid" in warnings[0][0]
        assert "long time" in warnings[0][0]
        assert warnings[0][1][-1] == QMessageBox.StandardButton.Cancel
    else:
        assert warnings == [], dialog.status.text()
    if mode == "decline":
        assert dialog._optimization_thread is None
        assert dialog._optimization_worker is None
        assert dialog._gate_indices()[3:] == (5, 12)
        dialog.close_file()
        return
    assert dialog._optimization_thread is not None
    assert not dialog.exhaustive_search.isEnabled()
    if mode == "exhaustive":
        assert dialog._optimization_total_pairs == 60
        dialog._optimization_progress(30, 20, None, None, None)
        assert dialog.progress.value() == 500
        assert "30/60" in dialog.status.text()
    qtbot.waitUntil(
        lambda: (
            dialog._optimization_thread is None
            and dialog._analysis_thread is None
            and "FOM gate optimization complete" in dialog.status.text()
        ),
        timeout=20_000,
    )

    _baseline_start, _baseline_end, gate_start, short_end, long_end = dialog._gate_indices()
    assert gate_start == 2
    assert short_end == 5
    assert long_end in {8, 9}
    assert "FOM gate optimization complete" in dialog.status.text()
    assert "confirmed full-data FOM" in dialog.status.text()
    assert dialog.psd_plot.fom_widget.last_fit is not None
    assert dialog._last_report_context is not None
    assert dialog._last_report_context.optimization is not None
    assert dialog._last_report_context.optimization.search_mode == mode
    if mode == "exhaustive":
        assert dialog._last_report_context.optimization.evaluated_candidates == 60
    assert dialog.exhaustive_search.isEnabled()
    dialog.close_file()
