from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from nlab.analysis.psd_selection import points_in_polygon
from nlab.analysis.waveform_file import build_waveform_file_index
from nlab.analysis.waveform_psd import WaveformPsdAccumulator, WaveformPsdSettings
from nlab.hardware.digitizer.dma import FILE_HEADER_STRUCT, FILE_MAGIC
from nlab.help.context import resolve_context
from nlab.views.offline_psd_plot import OfflinePsdPlot
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog
from nlab.views.waveform_cluster_selection import WaveformClusterSelection
from nlab.workers import waveform_average_worker as average_module
from nlab.workers import waveform_psd_worker as worker_module
from nlab.workers.waveform_average_worker import WaveformAverage, WaveformAverageWorker
from nlab.workers.waveform_psd_worker import WaveformPsdWorker


def _settings() -> WaveformPsdSettings:
    return WaveformPsdSettings(0, 4, 4, 7, 10, -1, energy_range=(0, 100), ratio_range=(0, 1))


def _pulse() -> np.ndarray:
    return np.asarray([100, 100, 100, 100, 90, 80, 70, 80, 90, 100], dtype="<i2")


def _capture(path: Path, samples: np.ndarray) -> None:
    records = np.zeros(len(samples), dtype=[("time", "<u8"), ("samples", "<i2", samples.shape[1:])])
    records["time"] = np.arange(len(samples), dtype=np.uint64) + 2**63
    records["samples"] = samples
    path.write_bytes(FILE_HEADER_STRUCT.pack(FILE_MAGIC, 1, 0, 0, 0.0, samples.shape[1] + 4)
                     + records.tobytes())


def test_polygon_selection_includes_edges_vertices_concavity_and_transforms() -> None:
    polygon = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4], [0, 0]])
    x = np.array([0, 4, 0.5, 3, 1, 2, np.nan, 8])
    y = np.array([0, 0, 3, 0.5, 1, 2, 0, 8])
    expected = [True, True, True, True, True, False, False, False]
    for vertices in (polygon, polygon[::-1]):
        np.testing.assert_array_equal(points_in_polygon(x, y, vertices), expected)
        # Energy and ratio axes have very different physical scales.
        np.testing.assert_array_equal(points_in_polygon(
            1e8 + x * 1e6, -0.5 + y * 0.01, [1e8, -0.5] + vertices * [1e6, 0.01],
        ), expected)


def test_indexed_accumulation_retains_exact_coordinates_only_for_plotted_events() -> None:
    valid, zero = _pulse(), np.full(10, 100)
    endpoint = np.array([100] * 7 + [90, 90, 100])  # PSD exactly 1
    raw = np.stack([valid, zero, valid, 100 + 2 * (valid - 100), endpoint])
    frames = np.array([10, 20, 30, 40, 50])
    accumulator = WaveformPsdAccumulator(_settings())
    accumulator.add_waveforms(raw, complete=np.array([1, 1, 0, 1, 1], dtype=bool),
                              frame_indices=frames)
    result = accumulator.result()
    np.testing.assert_array_equal(result.event_frames, [10, 50])
    np.testing.assert_allclose(result.event_energy, [90, 20])
    np.testing.assert_allclose(result.event_ratio, [1/3, 1])
    assert len(result.event_frames) == result.statistics.accepted == result.matrix.sum()
    assert not result.event_frames.flags.writeable
    assert not result.event_energy.flags.writeable
    assert not result.event_ratio.flags.writeable
    # Histogram-only/FOM callers do not retain event columns.
    ordinary = WaveformPsdAccumulator(_settings())
    ordinary.add_waveforms(raw)
    assert len(ordinary.result().event_frames) == 0
    scalar = WaveformPsdAccumulator(_settings())
    scalar.add_waveform(valid, frame_index=123)
    assert scalar.result().event_frames.tolist() == [123]


@pytest.mark.parametrize("maximum,expected", [(0, [0, 4]), (2, [0])])
def test_worker_preserves_original_frame_indices_through_stride_limits_and_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, maximum: int, expected: list[int],
) -> None:
    raw = np.tile(_pulse(), (6, 1))
    raw[2] = 100  # Rejected source frame; subsequent indices must not shift.
    path = tmp_path / "strided.bin"
    _capture(path, raw)
    monkeypatch.setattr(worker_module, "_NDMA_BATCH_EVENTS", 1)
    worker = WaveformPsdWorker(build_waveform_file_index(path), source_index=0,
                               settings=_settings(), stride=2, maximum_events=maximum)
    results = []
    worker.loaded.connect(results.append)
    worker.run()
    assert len(results) == 1
    assert results[0].event_frames.tolist() == expected


def test_editable_rectangle_polygon_clear_and_navigation(qtbot: QtBot) -> None:
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    selection = WaveformClusterSelection(plot.matrix_plot, plot)
    assert resolve_context(selection.polygon_button).topic == "waveform-clusters"
    selection.set_events(np.array([0, 2, 5, 9]), np.array([10., 20, 30, 40]),
                         np.array([-.2, 0, .2, .4]))
    requested = []
    selection.frame_requested.connect(requested.append)
    selection.add_region("rectangle")
    selection._roi.setPos((10, -.2))
    selection._roi.setSize((20, .4))
    selection.apply_selection()
    assert selection._selected.tolist() == [0, 2, 5]
    assert requested[-1] == 0
    selection.next_button.click()
    assert requested[-1] == 2
    selection.match_spin.setValue(3)
    assert requested[-1] == 5
    assert not selection.next_button.isEnabled()
    selection.add_region("polygon")
    selection._roi.setPoints([(0, -.3), (20, -.3), (20, .1), (0, .1)], closed=True)
    selection._roi.setPos((10, 0))  # Test transformed polygon handles, not local coordinates.
    selection.apply_selection()
    assert selection._selected.tolist() == [0, 2]
    selection.clear_selection()
    assert selection._selected is None
    assert selection._roi is None
    selection.invalidate()
    assert not selection.rectangle_button.isEnabled()


def test_dialog_browses_cluster_frames_handles_empty_selection_and_invalidates_changes(
    tmp_path: Path, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.tile(_pulse(), (6, 1))
    raw[1::2, 7:9] = 110  # Short=60, long=40, PSD=-0.5.
    path = tmp_path / "clusters.bin"
    _capture(path, raw)
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    dialog.auto_recalculate.setChecked(False)
    dialog.open_path(path)
    qtbot.waitUntil(lambda: dialog._index_thread is None, timeout=5000)
    dialog.polarity_combo.setCurrentIndex(dialog.polarity_combo.findData(-1))
    period = dialog.sample_period.value()
    dialog._updating_gates = True
    dialog.baseline_region.setRegion((0, 4 * period))
    for item, sample in ((dialog.gate_start_line, 4), (dialog.short_end_line, 7),
                         (dialog.long_end_line, 10)):
        item.setPos(sample * period)
    dialog._updating_gates = False
    dialog.energy_max.setValue(1000)
    dialog._start_analysis()
    qtbot.waitUntil(lambda: dialog._analysis_thread is None, timeout=5000)
    selection = dialog.cluster_selection
    assert selection.has_events
    selection.add_region("rectangle")
    selection._roi.setPos((0, -.6))
    selection._roi.setSize((1000, .5))
    selection.apply_selection()
    assert selection._selected.tolist() == [1, 3, 5]
    assert dialog.frame_spin.value() == 1
    assert dialog.frame_spin.isReadOnly()
    dialog.waveform_plot.setXRange(0, 50, padding=0)
    selection.next_button.click()
    assert dialog.frame_spin.value() == 3
    assert "PSD=-0.50000" in dialog.preview_status.text()
    assert dialog.waveform_plot.viewRange()[0] == pytest.approx([0, 50])
    selected_frames = selection._selected.copy()
    dialog.psd_plot.roi_checkbox.setChecked(False)
    assert not dialog.psd_plot.energy_region.isVisible()
    assert not dialog.psd_plot.cut_line.isVisible()
    assert selection._roi.isVisible()
    dialog.waveform_selection_checkbox.setChecked(False)
    assert selection.isHidden()
    assert not selection._roi.isVisible() and not selection._point.isVisible()
    assert dialog.frame_spin.value() == 3
    assert dialog.frame_spin.isReadOnly()
    assert selection.match_spin.value() == 2
    np.testing.assert_array_equal(selection._selected, selected_frames)
    dialog.psd_plot.roi_checkbox.setChecked(True)
    assert not selection._roi.isVisible()  # Independent visibility switches.
    dialog.waveform_selection_checkbox.setChecked(True)
    assert not selection.isHidden()
    assert selection._roi.isVisible() and selection._point.isVisible()
    assert selection.match_spin.value() == 2
    assert selection.average_button.isEnabled()
    selection.average_button.click()
    qtbot.waitUntil(lambda: dialog._average_thread is None, timeout=5000)
    assert dialog._average.count == 3
    # Overlay is baseline corrected event-by-event, then placed on the current
    # individual pulse's baseline. Sample indices and amplitudes are untouched.
    expected = 100 - raw[[1, 3, 5]].astype(float).mean(axis=0)
    np.testing.assert_allclose(dialog._average.samples, expected)
    x, y = dialog.average_curve.getData()
    np.testing.assert_allclose(x, np.arange(10) * period)
    np.testing.assert_allclose(y, 100 + expected)
    assert dialog.average_curve.isVisible()
    assert len(dialog.waveform_plot.legend.items) == 3
    assert dialog.waveform_plot.viewRange()[0] == pytest.approx([0, 50])
    selection.next_button.click()
    assert dialog.frame_spin.value() == 5
    assert dialog.average_curve.isVisible()
    selection.average_button.click()
    assert not dialog.average_curve.isVisible()
    assert len(dialog.waveform_plot.legend.items) == 2
    selection.average_button.click()
    qtbot.waitUntil(lambda: dialog._average_thread is None, timeout=5000)
    assert dialog.average_curve.isVisible()
    selection.average_button.click()

    def delayed_result(worker: WaveformAverageWorker) -> None:
        # Even a completion already queued at cancellation must not restore an
        # average for a now-stale selection (nor touch closed source state).
        worker._stop_event.wait(5)
        worker.loaded.emit(WaveformAverage(np.ones(10), 3))
        worker.finished.emit()

    monkeypatch.setattr(WaveformAverageWorker, "run", delayed_result)
    selection.average_button.click()
    # Moving the region removes the active filter until Apply is pressed again.
    selection._roi.setPos((10000, 0))
    qtbot.waitUntil(lambda: dialog._average_thread is None, timeout=5000)
    assert dialog._average is None and not dialog.average_curve.isVisible()
    assert not selection.average_button.isChecked()
    selection.apply_selection()
    assert selection._selected.tolist() == []
    assert dialog._frame is None
    assert "No waveforms" in dialog.preview_status.text()
    assert not dialog.frame_spin.isEnabled()
    assert not selection.average_button.isEnabled()
    selection.clear_selection()
    assert dialog._frame is not None
    assert dialog.frame_spin.isEnabled() and not dialog.frame_spin.isReadOnly()
    dialog.event_stride.setValue(2)
    assert not selection.has_events
    dialog._start_analysis()
    qtbot.waitUntil(lambda: dialog._analysis_thread is None, timeout=5000)
    assert dialog._last_psd_result.event_frames.tolist() == [0, 2, 4]
    dialog._request_analysis("sparse")
    qtbot.waitUntil(lambda: dialog._analysis_thread is None, timeout=5000)
    assert selection.has_events
    assert dialog._last_psd_result.event_frames.tolist() == [0]
    dialog.short_end_line.setPos(6 * period)
    assert not selection.has_events
    dialog._start_analysis()
    qtbot.waitUntil(lambda: dialog._analysis_thread is None, timeout=5000)
    selection.add_region("rectangle")
    selection._roi.setPos((0, -1))
    selection._roi.setSize((1000, 2))
    selection.apply_selection()
    selection.average_button.click()
    assert dialog._average_thread is not None
    dialog.close_file()
    qtbot.waitUntil(lambda: dialog._average_thread is None, timeout=5000)
    assert dialog._average is None and not dialog.average_curve.isVisible()
    assert not selection.has_events


def test_auto_polarity_is_frozen_while_browsing_a_reconstructed_cluster(
    qtbot: QtBot,
) -> None:
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    # A selected pulse may have a larger opposite-sign feature OUTSIDE the gates.
    # The reconstruction's polarity must survive browsing that waveform.
    context = type("Context", (), {"settings": replace(_settings(), polarity=-1)})()
    dialog._last_report_context = context
    dialog._selected_frame_indices = np.array([1])
    assert dialog._polarity() == -1


@pytest.mark.parametrize("shape", ["rectangle", "polygon"])
def test_black_cluster_outline_and_visibility_survive_updates(
    qtbot: QtBot, shape: str,
) -> None:
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    selection = WaveformClusterSelection(plot.matrix_plot, plot)
    data = (np.arange(10), np.arange(10, dtype=float), np.linspace(-.3, .3, 10))
    selection.set_events(*data)
    selection.add_region(shape)
    assert selection._roi.pen.color().name() == "#000000"
    assert selection._roi.hoverPen.color().name() == "#000000"
    if shape == "polygon":
        assert all(seg.pen.color().name() == "#000000" for seg in selection._roi.segments)
    selection.apply_selection()
    original = selection._selected
    selection.set_tools_visible(False)
    selection.set_busy(True)
    selection.set_busy(False)
    assert not selection._roi.isVisible()
    assert selection._selected is original
    selection.set_tools_visible(True)
    assert selection._roi.isVisible()
    selection.set_tools_visible(False)
    selection.set_events(*data)
    selection.add_region(shape)
    assert selection.isHidden()
    assert not selection._roi.isVisible()


def test_psd_roi_visibility_preserves_values_projections_and_fit_guide_setting(
    qtbot: QtBot,
) -> None:
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    matrix = np.ones((8, 8), dtype=np.uint64)
    plot.set_data(matrix, energy_range=(0, 80), ratio_range=(-1, 1))
    plot.energy_region.setRegion((20, 60))
    plot.cut_spin.setValue(.25)
    state = plot.report_state()
    guides = plot.fom_widget.guides_checkbox.isChecked()
    plot.roi_checkbox.setChecked(False)
    assert not plot.energy_region.isVisible() and not plot.cut_line.isVisible()
    assert plot.energy_region.getRegion() == state.energy_roi
    assert plot.cut_spin.value() == state.cut
    assert plot.fom_widget.guides_checkbox.isChecked() == guides
    np.testing.assert_array_equal(plot.report_state().matrix, state.matrix)
    plot.set_data(matrix * 2, energy_range=(0, 80), ratio_range=(-1, 1))
    assert not plot.energy_region.isVisible() and not plot.cut_line.isVisible()
    plot.roi_checkbox.setChecked(True)
    assert plot.energy_region.isVisible() and plot.cut_line.isVisible()


@pytest.mark.parametrize("method", ["mean", "median"])
def test_cluster_average_exact_frames_baselines_and_read_only_capture(
    tmp_path: Path, method: str,
) -> None:
    raw = np.tile(_pulse(), (4, 1))
    raw[1] += 12000
    raw[3] += 20000
    raw[3, :4] += [0, 0, 4, 12]
    raw[3, 4:] -= 20
    path = tmp_path / "average.bin"
    _capture(path, raw)
    before = path.read_bytes()
    settings = replace(_settings(), baseline_method=method)
    worker = WaveformAverageWorker(build_waveform_file_index(path), source_index=0,
                                   frames=np.array([1, 3]), settings=settings)
    results, errors, finished = [], [], []
    worker.loaded.connect(results.append)
    worker.error.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))
    worker.run()
    assert len(results) == 1 and not errors and finished == [True]
    selected = raw[[1, 3]].astype(float)
    baseline = getattr(np, method)(selected[:, :4], axis=1)
    expected = -(selected - baseline[:, None]).mean(axis=0)
    np.testing.assert_allclose(results[0].samples, expected)
    assert results[0].count == 2
    assert not results[0].samples.flags.writeable
    assert path.read_bytes() == before


def test_cluster_average_cancel_error_and_variable_length_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "average.bin"
    _capture(path, np.tile(_pulse(), (2, 1)))
    index = build_waveform_file_index(path)
    worker = WaveformAverageWorker(index, source_index=0, frames=np.array([0, 1]),
                                   settings=_settings())
    results, errors, finished = [], [], []
    worker.loaded.connect(results.append)
    worker.error.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))
    worker.stop()
    worker.run()
    assert not results and not errors and finished == [True]
    # The independent reader is already closed even after cancellation.
    path.rename(tmp_path / "moved.bin")
    path = tmp_path / "moved.bin"
    index = build_waveform_file_index(path)
    worker = WaveformAverageWorker(index, source_index=0, frames=np.array([99]),
                                   settings=_settings())
    worker.error.connect(errors.append)
    worker.run()
    assert len(errors) == 1 and "out of range" in errors[0]

    class Reader:
        closed = False

        def __init__(self, _index: object) -> None:
            pass

        def frame(self, frame: int, _source: int) -> object:
            # Model varying CAEN record lengths with the real reader contract.
            samples = np.r_[_pulse(), [95, 95]] if frame == 0 else _pulse()
            return type("Frame", (), {"samples": samples})()

        def close(self) -> None:
            Reader.closed = True

    monkeypatch.setattr(average_module, "MappedWaveformFile", Reader)
    worker = WaveformAverageWorker(index, source_index=0, frames=np.array([0, 1]),
                                   settings=_settings())
    worker.loaded.connect(results.append)
    worker.run()
    np.testing.assert_allclose(results[0].samples, 100 - _pulse())
    assert Reader.closed and results[0].count == 2
