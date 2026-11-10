"""Synthetic wire fixtures validate aggregate statistics, not detector noise."""

from __future__ import annotations

import math
import struct
import threading
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from nlab.analysis import current_capture as reader
from nlab.analysis.capture_metadata import capture_digitizer_info
from nlab.hardware.digitizer.scope_sums import MASK64, ScopeSumWriter, decode_sum_packet
from nlab.views.current_capture_dialog import CurrentCaptureDialog
from nlab.workers import current_capture_worker as worker_module


@pytest.fixture
def capture_file(tmp_path):
    def create(
        *,
        version=124,
        count=1100,
        first=(1 << 60) + 7,
        gap_windows=0,
        constant=None,
        configuration="",
        complete=True,
    ):
        path = tmp_path / f"capture-{len(list(tmp_path.iterdir()))}.h5"
        capacity = 508 if version == 124 else 254
        means = (np.arange(count) % 101 - 50).astype(float)
        if constant is not None:
            means[:] = constant
        writer = ScopeSumWriter(path, {"channel": 1, "configuration_yaml": configuration})
        try:
            for row in range(0, count, capacity):
                size = min(capacity, count - row)
                stamp = (first + (row + (gap_windows if row else 0)) * 4) & MASK64
                raw = bytearray(4096)
                magic = 0x32534D5553504456 if version == 124 else 0x31534D5553504456
                struct.pack_into(
                    "<4Q",
                    raw,
                    0,
                    stamp,
                    magic,
                    16 | size << 48,
                    row // capacity | gap_windows << 32 | int(row + size == count) << 63,
                )
                for j in range(capacity if version == 124 else size):
                    total = int((means[row + min(j, size - 1)] + 32768) * 16)
                    if version == 124:
                        struct.pack_into("<Q", raw, 32 + 8 * j, total)
                    else:
                        struct.pack_into("<QQ", raw, 32 + 16 * j, (stamp + j * 4) & MASK64, total)
                writer.append(decode_sum_packet(bytes(raw)))
        finally:
            writer.close(
                complete=complete,
                error="" if complete else "injected incomplete capture",
                missed_windows=gap_windows,
            )
        return path, means

    return create


@pytest.mark.parametrize("version", [123, 124])
@pytest.mark.parametrize("first", [(1 << 60) + 7, MASK64 - 3])
def test_half_open_microsecond_chunks_preserve_native_counter_and_statistics(
    capture_file, version, first
):
    path, means = capture_file(version=version, first=first)
    info = reader.read_current_capture_info(path)
    first_chunk = reader.read_current_chunk(info, 0, 0.32)
    next_chunk = reader.read_current_chunk(info, 0.32, 0.32)
    assert first_chunk.count == next_chunk.count == 10
    np.testing.assert_array_equal(
        first_chunk.timestamps, [(first + i * 4) & MASK64 for i in range(10)]
    )
    np.testing.assert_array_equal(next_chunk.values, means[10:20])
    np.testing.assert_allclose(first_chunk.x_us, np.arange(10) * 0.032)
    assert first_chunk.mean == pytest.approx(np.mean(means[:10]))
    assert first_chunk.std == pytest.approx(np.std(means[:10], ddof=0))
    assert first_chunk.rms == pytest.approx(np.sqrt(np.mean(means[:10] ** 2)))
    assert first_chunk.adjacent_noise == pytest.approx(math.sqrt(0.5))
    assert first_chunk.histogram_counts.sum() == 10
    assert first_chunk.adjacent_pairs == 9


def test_gap_is_not_noise_and_trace_does_not_bridge_it(capture_file):
    path, _ = capture_file(count=510, gap_windows=100, constant=0)
    # Make the two late windows a different constant level, without changing the gap.
    with h5py.File(path, "r+") as source:
        row = bytearray(source["packets"][1].tobytes())
        for j in range(508):
            struct.pack_into("<Q", row, 32 + j * 8, 16 * (32768 + 1000))
        source["packets"][1] = np.frombuffer(row, dtype="u1")
        source["sums"][508:] = 16 * (32768 + 1000)
    info = reader.read_current_capture_info(path)
    chunk = reader.read_current_chunk(info, 0, 100)
    assert chunk.adjacent_noise == 0
    assert chunk.adjacent_pairs == 508
    assert chunk.skipped_windows == 100
    assert not chunk.connect[507]
    assert chunk.std > 0
    empty = reader.read_current_chunk(info, 16.256, 1)
    assert empty.count == 0
    assert empty.mean is empty.std is empty.adjacent_noise is None
    assert empty.histogram_counts.sum() == 0
    assert empty.skipped_windows == 32


def test_preview_is_bounded_but_statistics_and_histogram_use_all_windows(capture_file, monkeypatch):
    path, means = capture_file()
    monkeypatch.setattr(reader, "_PREVIEW_POINTS", 10)
    info = reader.read_current_capture_info(path)
    chunk = reader.read_current_chunk(info, 0, 1000)
    assert len(chunk.values) <= 10
    assert chunk.count == len(means)
    assert chunk.histogram_counts.sum() == len(means)
    assert chunk.mean == pytest.approx(np.mean(means))
    assert chunk.std == pytest.approx(np.std(means), abs=1e-12)
    assert chunk.adjacent_noise == pytest.approx(np.sqrt(np.mean(np.diff(means) ** 2) / 2))


def test_reader_only_loads_requested_packets_and_rejects_corrupt_loaded_data(capture_file):
    path, _ = capture_file(count=2600)
    with h5py.File(path, "r+") as source:
        source["packets"][3, 8] ^= np.uint8(1)
    info = reader.read_current_capture_info(path)
    assert reader.read_current_chunk(info, 0, 0.32).count == 10
    with pytest.raises(ValueError, match="magic"):
        reader.read_current_chunk(info, 3 * 508 * 0.032, 1)


def test_decoded_columns_are_checked_against_raw_packets(capture_file):
    path, _ = capture_file(count=2600)
    with h5py.File(path, "r+") as source:
        source["sums"][3 * 508] += np.uint64(1)
    info = reader.read_current_capture_info(path)
    with pytest.raises(ValueError, match="differs from raw"):
        reader.read_current_chunk(info, 3 * 508 * 0.032, 1)


def test_embedded_large_full_configuration_is_retrievable(capture_file):
    document = {
        "format_version": 5,
        "connection": {"ip": "fixture", "port": 30431},
        "hardware": {"channels": {"0": {"scope": {"edge_mode": 5}}, "1": {"mca": {"a": 3}}}},
        "application": {"note": "µ" * 80_000},
    }
    text = yaml.safe_dump(document, allow_unicode=True)
    path, _ = capture_file(configuration=text)
    info = reader.read_current_capture_info(path)
    assert yaml.safe_load(info.configuration_yaml) == document
    assert capture_digitizer_info(path)["device endpoint"] == "fixture:30431"
    with h5py.File(path, "r") as source:
        assert "configuration_yaml" not in source.attrs["metadata_json"]


@pytest.mark.parametrize("count", [0, 1, 15])
def test_empty_single_and_constant_chunks_have_explicit_statistics(capture_file, count):
    path, _ = capture_file(count=count, constant=7)
    chunk = reader.read_current_chunk(reader.read_current_capture_info(path), 0, 1000)
    assert chunk.count == count
    assert chunk.mean == (7 if count else None)
    assert chunk.std == (0 if count else None)
    assert chunk.adjacent_noise == (0 if count > 1 else None)
    assert chunk.histogram_counts.sum() == count


def test_cancellation_and_changed_file_are_not_accepted(capture_file):
    path, _ = capture_file()
    with pytest.raises(reader.CurrentReadCancelledError):
        reader.read_current_capture_info(path, cancelled=lambda: True)
    info = reader.read_current_capture_info(path)
    with pytest.raises(reader.CurrentReadCancelledError):
        reader.read_current_chunk(info, 0, 1000, cancelled=lambda: True)
    with h5py.File(path, "r+") as source:
        source.attrs["cleanup_error"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        reader.read_current_chunk(info, 0, 1000)


@pytest.mark.parametrize("start,width", [(-1, 10), (0, 0), (0, float("inf")), (float("nan"), 1)])
def test_invalid_time_ranges_are_rejected(capture_file, start, width):
    path, _ = capture_file()
    with pytest.raises(ValueError, match="Start"):
        reader.read_current_chunk(reader.read_current_capture_info(path), start, width)


def test_dialog_navigation_configuration_export_and_close(qtbot, capture_file, tmp_path):
    document = {"format_version": 5, "hardware": {"channels": {"0": {"scope": {"edge_mode": 5}}}}}
    path, means = capture_file(configuration=yaml.safe_dump(document))
    dialog = CurrentCaptureDialog()
    qtbot.addWidget(dialog)
    dialog.width_spin.setValue(10)
    dialog.show_workspace()
    dialog.open_path(path)
    try:
        qtbot.waitUntil(lambda: dialog._chunk is not None)
        assert dialog._chunk.start_us == 0
        assert dialog._chunk.count == 313
        assert dialog.start_spin.locale().decimalPoint() == "."
        assert dialog.width_spin.suffix() == " µs"
        assert "Complete=True" in dialog.capture_status.text()
        assert "Average:" in dialog.stats["Average"].text()
        assert dialog.next_button.isEnabled()
        dialog.next_button.click()
        qtbot.waitUntil(lambda: dialog._chunk.start_us == 10)
        np.testing.assert_array_equal(dialog._chunk.values, means[313:625])
        dialog.last_button.click()
        qtbot.waitUntil(lambda: dialog._chunk.start_us == 30)
        assert not dialog.next_button.isEnabled()
        dialog.previous_button.click()
        qtbot.waitUntil(lambda: dialog._chunk.start_us == 20)
        dialog.first_button.click()
        qtbot.waitUntil(lambda: dialog._chunk.start_us == 0)
        exported = tmp_path / "retrieved.yaml"
        dialog.export_configuration(exported)
        assert yaml.safe_load(exported.read_text(encoding="utf-8")) == document
        assert not dialog.grab().isNull()
    finally:
        worker = dialog._worker
        dialog.close_without_prompt()
        dialog.close_without_prompt()
        assert worker is not None and not worker.isRunning()
    dialog.open_path(path)
    qtbot.waitUntil(lambda: dialog._chunk is not None and dialog._worker is not None)
    dialog.close_without_prompt()


def test_dialog_keeps_legacy_and_incomplete_captures_readable(qtbot, capture_file):
    path, _ = capture_file(complete=False)
    dialog = CurrentCaptureDialog()
    qtbot.addWidget(dialog)
    dialog.open_path(path)
    try:
        qtbot.waitUntil(lambda: dialog._info is not None)
        assert "Complete=False" in dialog.capture_status.text()
        assert "injected incomplete capture" in dialog.capture_status.text()
        assert "no embedded" in dialog.configuration_status.text().lower()
        assert not dialog.export_button.isEnabled()
        with pytest.raises(ValueError, match="no embedded"):
            dialog.export_configuration(Path("unused.yaml"))
    finally:
        dialog.close_without_prompt()


def test_worker_replaces_pending_requests_and_joins(qtbot, capture_file, monkeypatch):
    path, _ = capture_file()
    entered, release = threading.Event(), threading.Event()
    calls, results = [], []
    original = reader.read_current_chunk

    def read(info, start, width, *, cancelled, average_kernel=0):
        calls.append(start)
        if start == 0:
            entered.set()
            assert release.wait(3)
            if cancelled():
                raise reader.CurrentReadCancelledError
        return original(info, start, width, average_kernel=average_kernel, cancelled=cancelled)

    monkeypatch.setattr(worker_module, "read_current_chunk", read)
    worker = worker_module.CurrentCaptureWorker()
    worker.loaded.connect(lambda generation, info, chunk: results.append((generation, chunk)))
    worker.start()
    try:
        worker.submit(1, path, 0, 1, average_kernel=3)
        assert entered.wait(2)
        worker.submit(2, path, 1, 1, average_kernel=7)
        worker.submit(3, path, 2, 1, average_kernel=11)
        release.set()
        qtbot.waitUntil(lambda: bool(results))
        assert calls == [0, 2]
        assert [generation for generation, _ in results] == [3]
        assert results[0][1].start_us == 2
        assert results[0][1].average_kernel == 11
    finally:
        release.set()
        worker.stop()
        assert worker.wait(3000)


def test_dialog_close_cancels_inflight_read_before_widget_destruction(
    qtbot, capture_file, monkeypatch
):
    path, _ = capture_file()
    entered = threading.Event()

    def read(info, start, width, *, cancelled, average_kernel=0):
        entered.set()
        while not cancelled():
            threading.Event().wait(0.001)
        raise reader.CurrentReadCancelledError

    monkeypatch.setattr(worker_module, "read_current_chunk", read)
    dialog = CurrentCaptureDialog()
    qtbot.addWidget(dialog)
    dialog.open_path(path)
    assert entered.wait(2)
    worker = dialog._worker
    dialog.close()
    assert not worker.isRunning()
    assert dialog._worker is None


def test_chunk_edge_roundoff_does_not_drop_exact_native_window(capture_file):
    path, _ = capture_file()
    info = reader.read_current_capture_info(path)
    start_us = 36 * 0.008  # Float multiplication lands above an exact tick.
    assert math.ceil(start_us * 125) == 37
    chunk = reader.read_current_chunk(info, start_us, 0.032)
    assert chunk.count == 1
    assert int(chunk.timestamps[0]) == info.first_timestamp + 36
    assert chunk.skipped_windows == 0


@pytest.mark.parametrize("version", [123, 124])
@pytest.mark.parametrize("kernel", [1, 2, 7, 254, 508, 509, 2000])
def test_moving_average_matches_full_resolution_trailing_oracle(capture_file, version, kernel):
    path, means = capture_file(version=version)
    info = reader.read_current_capture_info(path)
    chunk = reader.read_current_chunk(info, 0, 1000, average_kernel=kernel)
    expected = np.full(len(means), np.nan)
    if kernel <= len(means):
        expected[kernel - 1 :] = np.convolve(means, np.ones(kernel) / kernel, mode="valid")
    assert chunk.average_kernel == kernel
    np.testing.assert_allclose(chunk.average_values, expected, atol=1e-12)
    np.testing.assert_array_equal(chunk.values, means)
    assert chunk.mean == pytest.approx(np.mean(means))
    assert chunk.std == pytest.approx(np.std(means), abs=1e-12)
    assert chunk.histogram_counts.sum() == len(means)


@pytest.mark.parametrize("version,capacity", [(123, 254), (124, 508)])
@pytest.mark.parametrize("gap", [0, 10])
def test_moving_average_preserves_wrap_and_resets_only_at_native_gaps(
    capture_file,
    version,
    capacity,
    gap,
):
    path, means = capture_file(
        version=version,
        count=capacity + 20,
        first=MASK64 - 3,
        gap_windows=gap,
    )
    info = reader.read_current_capture_info(path)
    chunk = reader.read_current_chunk(info, 0, 1000, average_kernel=5)
    expected = np.full(len(means), np.nan)
    runs = [means[:capacity], means[capacity:]] if gap else [means]
    offset = 0
    for run in runs:
        expected[offset + 4 : offset + len(run)] = np.convolve(run, np.ones(5) / 5, mode="valid")
        offset += len(run)
    np.testing.assert_allclose(chunk.average_values, expected, atol=1e-12)
    assert np.isfinite(chunk.average_values[5])  # Native wrap did not reset the kernel.
    assert np.isnan(chunk.average_values[capacity]) == bool(gap)


def test_moving_average_resets_at_gap_inside_ip123_packet(capture_file):
    path, means = capture_file(version=123, count=20)
    with h5py.File(path, "r+") as source:
        raw = bytearray(source["packets"][0].tobytes())
        for index in range(5, 20):
            stamp = (int(source["timestamps"][index]) + 20) & MASK64
            struct.pack_into("<Q", raw, 32 + index * 16, stamp)
            source["timestamps"][index] = stamp
        source["packets"][0] = np.frombuffer(raw, dtype="u1")
    chunk = reader.read_current_chunk(
        reader.read_current_capture_info(path), 0, 100, average_kernel=3
    )
    expected = np.full(20, np.nan)
    expected[2:5] = [np.mean(means[i - 2 : i + 1]) for i in range(2, 5)]
    expected[7:] = [np.mean(means[i - 2 : i + 1]) for i in range(7, 20)]
    np.testing.assert_allclose(chunk.average_values, expected, atol=1e-12)


def test_moving_average_precedes_preview_sampling_and_includes_last_window(
    capture_file, monkeypatch
):
    path, means = capture_file()
    info = reader.read_current_capture_info(path)
    monkeypatch.setattr(reader, "_PREVIEW_POINTS", 10)
    chunk = reader.read_current_chunk(info, 0, 1000, average_kernel=7)
    expected = np.full(len(means), np.nan)
    expected[6:] = [np.mean(means[i - 6 : i + 1]) for i in range(6, len(means))]
    indices = ((chunk.timestamps - np.uint64(info.first_timestamp)) // 4).astype(int)
    assert len(chunk.values) <= 10
    assert indices[0] == 0 and indices[-1] == len(means) - 1
    np.testing.assert_allclose(chunk.average_values, expected[indices], atol=1e-12)
    np.testing.assert_array_equal(chunk.values, means[indices])
    assert np.all(chunk.connect[:-1])


def test_moving_average_uses_only_viewed_chunk_and_requires_full_kernel(capture_file):
    path, means = capture_file()
    info = reader.read_current_capture_info(path)
    chunk = reader.read_current_chunk(info, 0.32, 0.32, average_kernel=3)
    assert np.isnan(chunk.average_values[:2]).all()
    np.testing.assert_allclose(
        chunk.average_values[2:], [np.mean(means[i - 2 : i + 1]) for i in range(12, 20)]
    )
    too_large = reader.read_current_chunk(info, 0.32, 0.32, average_kernel=1_000_000)
    assert np.isnan(too_large.average_values).all()
    assert too_large.count == 10


@pytest.mark.parametrize("kernel", [-1, 1.5, True, 1_000_001])
def test_invalid_moving_average_kernels_are_rejected(capture_file, kernel):
    path, _ = capture_file()
    with pytest.raises(ValueError, match="Average kernel"):
        reader.read_current_chunk(
            reader.read_current_capture_info(path), 0, 1000, average_kernel=kernel
        )


def test_dialog_overlays_average_and_updates_kernel_and_chunk(qtbot, capture_file):
    path, _ = capture_file()
    dialog = CurrentCaptureDialog()
    qtbot.addWidget(dialog)
    dialog.width_spin.setValue(10)
    dialog.open_path(path)
    try:
        qtbot.waitUntil(lambda: dialog._chunk is not None)
        assert dialog.average_checkbox.isChecked()
        assert dialog._chunk.average_kernel == 16
        assert dialog.average_kernel_spin.suffix() == " samples"
        assert dialog.average_trace.isVisible()
        assert dialog.average_trace.getViewBox() is dialog.trace.getViewBox()
        assert dialog.average_trace in dialog.trace_plot.getPlotItem().listDataItems()
        raw = dialog._chunk.values.copy()
        raw_stats = {name: label.text() for name, label in dialog.stats.items()}
        dialog.average_kernel_spin.setValue(2)
        qtbot.waitUntil(lambda: dialog._chunk.average_kernel == 2)
        np.testing.assert_array_equal(dialog.trace.yData, raw)
        np.testing.assert_allclose(dialog.average_trace.yData[1:], (raw[1:] + raw[:-1]) / 2)
        assert {name: label.text() for name, label in dialog.stats.items()} == raw_stats
        dialog.average_checkbox.setChecked(False)
        qtbot.waitUntil(lambda: dialog._chunk.average_kernel == 0)
        assert not dialog.average_trace.isVisible()
        assert not dialog.average_kernel_spin.isEnabled()
        np.testing.assert_array_equal(dialog.trace.yData, raw)
        dialog.average_checkbox.setChecked(True)
        dialog.next_button.click()
        qtbot.waitUntil(lambda: dialog._chunk.start_us == 10 and dialog._chunk.average_kernel == 2)
        next_values = dialog._chunk.values
        np.testing.assert_allclose(
            dialog.average_trace.yData[1:],
            (next_values[1:] + next_values[:-1]) / 2,
        )
        dialog.average_kernel_spin.setValue(1_000_000)
        qtbot.waitUntil(lambda: dialog._chunk.average_kernel == 1_000_000)
        assert "consecutive samples" in dialog.status.text()
    finally:
        dialog.close_without_prompt()


def test_distribution_is_horizontal_on_right_with_shared_amplitude_axis(qtbot, capture_file):
    path, _ = capture_file()
    dialog = CurrentCaptureDialog()
    qtbot.addWidget(dialog)
    dialog.width_spin.setValue(10)
    dialog.show_workspace()
    dialog.open_path(path)
    try:
        qtbot.waitUntil(lambda: dialog._chunk is not None)
        trace_view = dialog.trace_plot.getViewBox()
        distribution_view = dialog.distribution_plot.getViewBox()
        qtbot.waitUntil(
            lambda: np.allclose(trace_view.viewRange()[1], distribution_view.viewRange()[1])
        )
        trace_rect = trace_view.screenGeometry()
        distribution_rect = distribution_view.screenGeometry()
        assert distribution_rect.left() > trace_rect.right()
        assert abs(distribution_rect.top() - trace_rect.top()) <= 1
        assert abs(distribution_rect.bottom() - trace_rect.bottom()) <= 1
        counts, amplitudes = dialog.distribution.getData()
        np.testing.assert_array_equal(counts[1:-2:2], dialog._chunk.histogram_counts)
        np.testing.assert_array_equal(amplitudes[1:-2:2], dialog._chunk.histogram_edges[:-1])
        np.testing.assert_array_equal(amplitudes[2:-2:2], dialog._chunk.histogram_edges[1:])
        assert counts[0] == counts[-1] == 0
        assert amplitudes[0] == amplitudes[-1]
        assert np.all((np.diff(counts) == 0) | (np.diff(amplitudes) == 0))
        assert dialog.distribution.opts["fillLevel"] == "enclosed"
        assert dialog.distribution_plot.getAxis("bottom").labelText == "Counts"

        dialog.trace_plot.setYRange(-70, 50, padding=0)
        qtbot.waitUntil(lambda: np.allclose(distribution_view.viewRange()[1], [-70, 50]))
        dialog.distribution_plot.setYRange(-40, 30, padding=0)
        qtbot.waitUntil(lambda: np.allclose(trace_view.viewRange()[1], [-40, 30]))
        counts_range = distribution_view.viewRange()[0]
        dialog.trace_plot.setXRange(0, 5, padding=0)
        np.testing.assert_allclose(distribution_view.viewRange()[0], counts_range)
        assert not dialog.grab().isNull()
    finally:
        dialog.close_without_prompt()


def test_chunk_mean_reference_uses_raw_population_and_clears_on_empty_or_error(
    qtbot,
    capture_file,
    monkeypatch,
):
    path, means = capture_file()
    monkeypatch.setattr(reader, "_PREVIEW_POINTS", 10)
    dialog = CurrentCaptureDialog()
    qtbot.addWidget(dialog)
    dialog.open_path(path)
    try:
        qtbot.waitUntil(lambda: dialog._chunk is not None)
        assert not np.isclose(np.mean(dialog._chunk.values), np.mean(means))
        for line in (dialog.mean_line, dialog.distribution_mean_line):
            assert line.isVisible()
            assert line.angle == 0
            assert not line.movable
            assert line.value() == pytest.approx(np.mean(means))
        assert "Chunk mean:" in dialog.mean_line.label.textItem.toPlainText()
        assert "ADC codes" in dialog.mean_line.label.textItem.toPlainText()
        dialog.average_checkbox.setChecked(False)
        qtbot.waitUntil(lambda: dialog._chunk.average_kernel == 0)
        assert dialog.mean_line.value() == pytest.approx(np.mean(means))
        dialog.width_spin.setValue(10)
        qtbot.waitUntil(lambda: dialog._chunk.end_us == 10)
        dialog.next_button.click()
        qtbot.waitUntil(lambda: dialog._chunk.start_us == 10)
        assert dialog.mean_line.value() == pytest.approx(np.mean(means[313:625]))
        assert dialog.distribution_mean_line.value() == dialog.mean_line.value()

        empty_path, _ = capture_file(count=0)
        dialog.open_path(empty_path)
        qtbot.waitUntil(lambda: dialog._chunk is not None)
        assert not dialog.mean_line.isVisible()
        assert not dialog.distribution_mean_line.isVisible()
        assert dialog.distribution.getData()[0] is None
        dialog.open_path(path)
        qtbot.waitUntil(lambda: dialog._chunk is not None)
        assert dialog.mean_line.isVisible()
        dialog.open_path(path.with_name("missing-current-capture.h5"))
        qtbot.waitUntil(lambda: "FileNotFoundError" in dialog.status.text())
        assert not dialog.mean_line.isVisible()
        assert not dialog.distribution_mean_line.isVisible()
        assert dialog.distribution.getData()[0] is None
    finally:
        dialog.close_without_prompt()
