"""Waveform-derived PSD integration and histogram checks."""

from __future__ import annotations

import struct
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from PySide6.QtCore import Signal
from pytestqt.qtbot import QtBot

import nlab.views.waveform_analysis_dialog as waveform_dialog_module
from nlab.analysis.waveform_file import build_waveform_file_index
from nlab.analysis.waveform_psd import (
    WaveformPsdAccumulator,
    WaveformPsdResult,
    WaveformPsdSettings,
    infer_waveform_polarity,
    integrate_waveform,
)
from nlab.hardware.digitizer.dma import FILE_HEADER_STRUCT, FILE_MAGIC, FILE_VERSION
from nlab.views.offline_psd_plot import OfflinePsdPlot
from nlab.views.plot_viewbox import ModifierZoomViewBox
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog
from nlab.workers.base_worker import BaseWorker
from nlab.workers.waveform_psd_worker import WaveformPsdWorker


def _settings(*, polarity: Literal[-1, 1] = -1) -> WaveformPsdSettings:
    return WaveformPsdSettings(
        baseline_start=0,
        baseline_end=4,
        gate_start=4,
        short_end=7,
        long_end=10,
        polarity=polarity,
        energy_bins=10,
        energy_range=(0.0, 100.0),
        ratio_bins=10,
        ratio_range=(0.0, 1.0),
    )


def test_negative_pulse_is_baselined_inverted_and_integrated() -> None:
    samples = np.asarray([100, 100, 100, 100, 90, 80, 70, 80, 90, 100])

    charge = integrate_waveform(samples, _settings())

    assert charge is not None
    assert charge.baseline == 100
    assert charge.baseline_rms == 0
    assert charge.short_charge == 60
    assert charge.long_charge == 90
    assert charge.ratio == pytest.approx(1 / 3)


def test_polarity_inference_uses_larger_excursion_after_baseline() -> None:
    negative = np.asarray([100, 100, 100, 100, 98, 70, 95])
    positive = np.asarray([100, 100, 100, 100, 102, 140, 105])

    assert infer_waveform_polarity(
        negative, baseline_start=0, baseline_end=4, search_start=4
    ) == -1
    assert infer_waveform_polarity(
        positive, baseline_start=0, baseline_end=4, search_start=4
    ) == 1


def test_accumulator_tracks_rejections_and_stored_gate_comparison() -> None:
    accumulator = WaveformPsdAccumulator(_settings())
    valid = np.asarray([100, 100, 100, 100, 90, 80, 70, 80, 90, 100])
    nonpositive = np.full(10, 100)

    accumulator.add_waveform(valid, stored_long=88, stored_short=58)
    accumulator.add_waveform(nonpositive)
    accumulator.add_waveform(valid[:8])
    result = accumulator.result()

    assert result.statistics.received == 3
    assert result.statistics.accepted == 1
    assert result.statistics.nonpositive_long == 1
    assert result.statistics.too_short == 1
    assert result.matrix.sum() == 1
    np.testing.assert_allclose(result.calculated_long, [90])
    np.testing.assert_allclose(result.stored_long, [88])
    np.testing.assert_allclose(result.calculated_short, [60])
    np.testing.assert_allclose(result.stored_short, [58])


def test_vectorized_batch_matches_per_waveform_accumulation() -> None:
    settings = _settings()
    valid = np.asarray([100, 100, 100, 100, 90, 80, 70, 80, 90, 100])
    nonpositive = np.full(10, 100)
    placeholder = np.zeros(10, dtype=np.int16)

    scalar = WaveformPsdAccumulator(settings)
    scalar.add_waveform(valid, stored_long=88, stored_short=58)
    scalar.add_waveform(nonpositive, stored_long=0, stored_short=0)
    scalar.add_waveform(placeholder[:8], stored_long=0, stored_short=0)

    vectorized = WaveformPsdAccumulator(settings)
    vectorized.add_waveforms(
        np.stack((valid, nonpositive, placeholder)),
        complete=np.asarray((True, True, False)),
        stored_long=np.asarray((88, 0, 0)),
        stored_short=np.asarray((58, 0, 0)),
    )

    scalar_result = scalar.result()
    vectorized_result = vectorized.result()
    assert vectorized_result.statistics == scalar_result.statistics
    np.testing.assert_array_equal(vectorized_result.matrix, scalar_result.matrix)
    np.testing.assert_array_equal(
        vectorized_result.calculated_long,
        scalar_result.calculated_long,
    )
    np.testing.assert_array_equal(vectorized_result.stored_long, scalar_result.stored_long)


def test_settings_require_nested_nonoverlapping_regions() -> None:
    with pytest.raises(ValueError, match="baseline"):
        WaveformPsdSettings(0, 5, 4, 7, 10, -1)
    with pytest.raises(ValueError, match="integration gates"):
        WaveformPsdSettings(0, 4, 4, 3, 10, -1)


def test_offline_psd_plot_keeps_requested_matrix_range(qtbot: QtBot) -> None:
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    plot.resize(1000, 500)
    plot.show()

    plot.set_data(
        np.ones((512, 256), dtype=np.uint64),
        energy_range=(0.0, 1_000_000.0),
        ratio_range=(-1.0, 1.0),
    )
    plot.matrix_plot.autoRange()

    x_range, y_range = plot.matrix_plot.viewRange()
    assert x_range == pytest.approx([0.0, 1_000_000.0])
    assert y_range == pytest.approx([-1.0, 1.0])


def test_offline_psd_plots_use_modifier_zoom(qtbot: QtBot) -> None:
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)

    for plot_item in (plot.matrix_plot, plot.ratio_plot, plot.energy_plot):
        assert isinstance(plot_item.getViewBox(), ModifierZoomViewBox)


def test_worker_reconstructs_indexed_ndma_without_retaining_frames(tmp_path: Path) -> None:
    path = tmp_path / "waveforms.bin"
    frame_samples = 12
    header = FILE_HEADER_STRUCT.pack(
        FILE_MAGIC,
        FILE_VERSION,
        0,
        0,
        0.0,
        frame_samples,
    )
    first = np.asarray((100, 100, 100, 90, 70, 80, 90, 100), dtype="<i2")
    second = np.asarray((100, 100, 100, 80, 40, 60, 80, 100), dtype="<i2")
    path.write_bytes(
        header
        + struct.pack("<Q", 1)
        + first.tobytes()
        + struct.pack("<Q", 2)
        + second.tobytes()
    )
    index = build_waveform_file_index(path)
    settings = WaveformPsdSettings(
        baseline_start=0,
        baseline_end=3,
        gate_start=3,
        short_end=5,
        long_end=8,
        polarity=-1,
        energy_bins=32,
        energy_range=(0.0, 1000.0),
        ratio_bins=32,
        ratio_range=(-0.1, 1.0),
    )
    results: list[WaveformPsdResult] = []
    worker = WaveformPsdWorker(index, source_index=0, settings=settings)
    worker.loaded.connect(results.append)

    worker.run()

    assert len(results) == 1
    assert results[0].statistics.received == 2
    assert results[0].statistics.accepted == 2
    assert int(results[0].matrix.sum()) == 2


def test_waveform_workbench_indexes_and_reconstructs_ndma(
    tmp_path: Path,
    qtbot: QtBot,
) -> None:
    path = tmp_path / "workbench.bin"
    frame_samples = 12
    header = FILE_HEADER_STRUCT.pack(
        FILE_MAGIC,
        FILE_VERSION,
        0,
        0,
        0.0,
        frame_samples,
    )
    samples = np.asarray((100, 100, 100, 90, 60, 70, 90, 100), dtype="<i2")
    path.write_bytes(header + struct.pack("<Q", 1) + samples.tobytes())
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)

    dialog.open_path(path)
    qtbot.waitUntil(lambda: dialog._index_thread is None, timeout=5000)

    assert dialog._reader is not None
    assert dialog.frame_spin.maximum() == 0
    assert "Qlong" in dialog.preview_status.text()
    assert dialog.waveform_widget.minimumHeight() >= 200
    assert dialog.psd_plot.minimumHeight() >= 360
    assert dialog.ratio_min.value() == -1.0
    assert "20th" in dialog.auto_recalculate.text()
    assert isinstance(dialog.waveform_widget.getViewBox(), ModifierZoomViewBox)

    dialog._start_analysis()
    qtbot.waitUntil(lambda: dialog._analysis_thread is None, timeout=5000)

    assert int(dialog.psd_plot._matrix.sum()) == 1
    assert "PSD complete" in dialog.status.text()

    dialog._request_sparse_analysis()
    qtbot.waitUntil(lambda: dialog._analysis_thread is None, timeout=5000)
    assert "Sparse PSD preview complete" in dialog.status.text()

    dialog._on_gate_moved()
    assert dialog._auto_preview_timer.isActive()
    dialog._on_gate_change_finished()
    assert not dialog._auto_preview_timer.isActive()
    qtbot.waitUntil(lambda: dialog._analysis_thread is None, timeout=5000)
    assert "PSD complete" in dialog.status.text()


def test_waveform_worker_is_deleted_by_its_finished_thread(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "worker-lifetime.bin"
    frame_samples = 12
    header = FILE_HEADER_STRUCT.pack(
        FILE_MAGIC,
        FILE_VERSION,
        0,
        0,
        0.0,
        frame_samples,
    )
    samples = np.asarray((100, 100, 100, 90, 60, 70, 90, 100), dtype="<i2")
    path.write_bytes(header + struct.pack("<Q", 1) + samples.tobytes())

    class BlockingWorker(BaseWorker):
        progress = Signal(object, object)
        loaded = Signal(object)
        cancelled = Signal()

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__()
            self.release = threading.Event()

        def run(self) -> None:
            assert self.release.wait(5)
            self.finished.emit()

        def stop(self) -> None:
            self.release.set()

    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    dialog.open_path(path)
    qtbot.waitUntil(lambda: dialog._index_thread is None, timeout=5000)
    monkeypatch.setattr(waveform_dialog_module, "WaveformPsdWorker", BlockingWorker)

    dialog._start_analysis()
    worker = dialog._analysis_worker
    assert isinstance(worker, BlockingWorker)
    destroyed: list[bool] = []
    worker.destroyed.connect(lambda: destroyed.append(True))
    worker.release.set()

    qtbot.waitUntil(lambda: dialog._analysis_thread is None, timeout=5000)

    assert destroyed == [True]
