"""Waveform PSD gate-optimization tests."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

import nlab.views.waveform_analysis_dialog as waveform_dialog_module
from nlab.analysis.waveform_file import build_waveform_file_index
from nlab.analysis.waveform_fom_optimization import (
    WaveformFomOptimizationCancelledError,
    WaveformFomOptimizationRequest,
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


def test_optimization_worker_reads_waveforms_once_and_returns_best_gates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "optimization.bin"
    _write_synthetic_waveforms(path, 2000)
    index = build_waveform_file_index(path)
    results: list[object] = []
    errors: list[str] = []
    worker = WaveformFomOptimizationWorker(
        index,
        source_index=0,
        request=_request(),
        stride=1,
        maximum_events=2000,
    )
    worker.loaded.connect(results.append)
    worker.error.connect(errors.append)

    worker.run()

    assert errors == []
    assert len(results) == 1
    result = results[0]
    assert result.best.short_end == 5
    assert result.best.long_end in {8, 9}
    assert result.sampled_waveforms == 2000


def test_waveform_workbench_applies_and_confirms_optimized_gates(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
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

    warnings: list[str] = []
    monkeypatch.setattr(
        waveform_dialog_module.QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message),
    )

    qtbot.mouseClick(dialog.optimize_button, Qt.MouseButton.LeftButton)
    assert warnings == [], dialog.status.text()
    assert dialog._optimization_thread is not None
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
    dialog.close_file()
