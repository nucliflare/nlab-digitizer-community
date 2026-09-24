from __future__ import annotations

import struct
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import QObject, Qt, QThread, Signal
from pytestqt.qtbot import QtBot

from nlab.controllers.current_monitor_controller import CurrentMonitorController
from nlab.hardware.digitizer.backends import iio_backend as iio_backend_module
from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.current_monitor import (
    CurrentMonitorClient,
    ScopeCurrentAccumulator,
    ScopeCurrentRuntime,
)
from nlab.hardware.digitizer.dma import ScopeDmaGeometry
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.workers.current_monitor_worker import (
    CurrentMonitorWorker,
    CurrentSample,
    CurrentSampleBuffer,
)


class _FakeCurrentClient(CurrentMonitorClient):
    def __init__(self) -> None:
        self.reads = 0
        self.closed = False

    def read_raw(self) -> int:
        self.reads += 1
        return self.reads

    def close(self) -> None:
        self.closed = True


def test_sample_buffer_retains_latest_values_and_reports_overflow() -> None:
    buffer = CurrentSampleBuffer(capacity=2)
    for sequence in range(1, 4):
        buffer.append(CurrentSample(sequence, sequence * 1000, sequence * 10, 50))

    batch = buffer.drain()

    assert [sample.sequence for sample in batch.samples] == [2, 3]
    assert batch.dropped_samples == 1
    assert buffer.drain().samples == ()


def test_worker_opens_reads_and_closes_transport_on_its_thread(qtbot: QtBot) -> None:
    client = _FakeCurrentClient()
    factory_thread: list[QThread] = []

    def factory() -> CurrentMonitorClient:
        factory_thread.append(QThread.currentThread())
        return client

    buffer = CurrentSampleBuffer()
    worker = CurrentMonitorWorker(factory, buffer, target_hz=500)
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
    thread.finished.connect(worker.deleteLater)
    thread.start()

    try:
        qtbot.waitUntil(lambda: client.reads >= 4, timeout=2000)
    finally:
        worker.request_shutdown()
        assert thread.wait(2000)

    batch = buffer.drain()
    assert factory_thread == [thread]
    assert client.closed
    assert len(batch.samples) >= 4
    assert [sample.sequence for sample in batch.samples] == list(range(1, len(batch.samples) + 1))
    assert all(sample.read_latency_ns >= 0 for sample in batch.samples)


def test_controller_renders_calibrated_latest_and_interval_values(qtbot: QtBot) -> None:
    mca = cast(MultiChannelAnalyzer, SimpleNamespace())
    controller = CurrentMonitorController(mca, channel=0, auto_start=False)
    qtbot.addWidget(controller)
    controller._display_timer.stop()
    controller.ui.spinZero.setValue(100.0)
    controller.ui.spinScale.setValue(0.5)
    controller.ui.comboUnit.setCurrentText("nA")

    now_ns = time.perf_counter_ns()
    for sequence, raw_code in enumerate((100, 110, 120), start=1):
        controller._sample_buffer.append(
            CurrentSample(
                sequence=sequence,
                timestamp_ns=now_ns - (3 - sequence) * 1_000_000,
                raw_code=raw_code,
                read_latency_ns=200_000,
            )
        )

    controller._render_pending()
    controller._display_timer.stop()

    assert controller.ui.lblCurrent.text() == "10 nA"
    assert controller.ui.lblRaw.text() == "120"
    assert "mean 5" in controller.ui.lblInterval.text()
    assert "min 0" in controller.ui.lblInterval.text()
    assert "max 10 nA" in controller.ui.lblInterval.text()
    assert "1,000.0 Hz" in controller.ui.lblAcquisition.text()
    assert "median read 0.200 ms" in controller.ui.lblAcquisition.text()

    controller._set_zero_from_recent()
    assert controller.ui.spinZero.value() == pytest.approx(110.0)
    assert controller.configuration_settings() == {
        "mode": "iir",
        "zero_code": 110.0,
        "scale_per_code": 0.5,
        "unit": "nA",
        "display_fps": 30,
        "analysis_bin_ms": 100,
    }
    controller.apply_configuration_settings(
        {"zero_code": -4.0, "scale_per_code": 0.125, "unit": "uA"}
    )
    assert controller.configuration_settings() == {
        "mode": "iir",
        "zero_code": -4.0,
        "scale_per_code": 0.125,
        "unit": "uA",
        "display_fps": 30,
        "analysis_bin_ms": 100,
    }
    controller._on_worker_finished()
    qtbot.wait(60)
    assert not controller._display_timer.isActive()
    assert controller.ui.lblStatus.text() == "Monitor stopped."


def test_controller_starts_stopped_by_default(qtbot: QtBot) -> None:
    mca = cast(MultiChannelAnalyzer, SimpleNamespace())

    controller = CurrentMonitorController(mca, channel=0)
    qtbot.addWidget(controller)

    assert controller._worker is None
    assert controller._worker_thread is None
    assert not controller._dma_active
    assert controller.ui.btnStart.isEnabled()
    assert not controller.ui.btnStop.isEnabled()
    assert controller.ui.lblStatus.text() == "Monitor stopped."


class _FakeScopeController(QObject):
    current_dma_state_changed = Signal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        self.starts = 0
        self.stops = 0

    def current_dma_supported(self) -> bool:
        return True

    def start_current_dma_monitor(self) -> None:
        self.starts += 1

    def stop_current_dma_monitor(self, *, wait: bool = False) -> None:
        self.stops += 1
        self.current_dma_state_changed.emit(False, "Scope DMA stopped.")


def _scope_record(
    timestamp: int,
    values: np.ndarray,
    geometry: ScopeDmaGeometry,
) -> bytes:
    return (
        struct.pack("<Q", timestamp)
        + np.asarray(values, dtype="<i2").tobytes()
        + bytes(geometry.padding_bytes)
    )


def test_scope_current_accumulator_preserves_weighted_sums_and_bounded_bins() -> None:
    geometry = ScopeDmaGeometry.legacy(8)
    accumulator = ScopeCurrentAccumulator(bin_width_ms=1, history_seconds=1)
    accumulator.start_session(
        geometry,
        ScopeCurrentRuntime(
            channel=0,
            ip_version=121,
            gap_cycles=24_998,
            expected_interval_ticks=25_000,
        ),
    )
    accumulator.append_frame(
        _scope_record(1, np.full(4, 0, dtype="<i2"), geometry), geometry, 1_000
    )
    accumulator.append_frame(
        _scope_record(50_001, np.full(4, 10, dtype="<i2"), geometry),
        geometry,
        401_000,
    )
    accumulator.append_frame(
        _scope_record(125_001, np.full(4, 100, dtype="<i2"), geometry),
        geometry,
        1_001_000,
    )

    snapshot = accumulator.snapshot(now_ns=1_101_000)

    assert snapshot.received_frames == snapshot.analyzed_frames == 3
    assert snapshot.analyzed_raw_sum == 440
    assert snapshot.analyzed_sample_count == 12
    assert snapshot.discarded_analysis_frames == 0
    assert len(snapshot.bins) == 2
    assert snapshot.bins[0].raw_mean == pytest.approx(5.0)
    assert snapshot.bins[1].raw_mean == pytest.approx(100.0)
    assert sum(item.raw_sum for item in snapshot.bins) / sum(
        item.sample_count for item in snapshot.bins
    ) == pytest.approx(110 / 3)
    assert snapshot.skipped_opportunities == 3
    assert snapshot.off_grid_intervals == 0
    assert snapshot.analysis_queue_depth == 0
    assert snapshot.analysis_queue_high_water == 0


def test_scope_current_accumulator_rejects_nonzero_padding() -> None:
    geometry = ScopeDmaGeometry(
        frame_samples=16,
        buffer_samples=8,
        frame_bytes=16,
        waveform_samples=3,
        sample_decimation=4,
        padding_bytes=2,
    )
    accumulator = ScopeCurrentAccumulator()
    accumulator.start_session(
        geometry,
        ScopeCurrentRuntime(0, 122, 10, 14),
    )
    bad = struct.pack("<Q", 1) + np.arange(3, dtype="<i2").tobytes() + b"\x00\x01"

    with pytest.raises(ValueError, match="padding"):
        accumulator.append_frame(bad, geometry, 100)

    snapshot = accumulator.snapshot(now_ns=100)
    assert snapshot.rejected_frames == 1
    assert snapshot.protocol_errors == 1
    assert snapshot.analyzed_frames == 0


def test_scope_current_accumulator_gui_pause_keeps_all_analysis() -> None:
    geometry = ScopeDmaGeometry.legacy(8)
    accumulator = ScopeCurrentAccumulator(bin_width_ms=100, history_seconds=1)
    accumulator.start_session(
        geometry,
        ScopeCurrentRuntime(0, 121, 8, 10),
    )
    record_values = np.array([-32768, -1, 0, 32767], dtype="<i2")

    for index in range(750):
        accumulator.append_frame(
            _scope_record(1 + index * 10, record_values, geometry),
            geometry,
            1_000 + index * 80,
        )

    snapshot = accumulator.snapshot(now_ns=1_000_000)
    accumulator.note_display_update(snapshot.generation, 250_000_000)
    accumulator.finish_session()
    final = accumulator.snapshot(now_ns=1_000_000)

    assert final.received_frames == final.analyzed_frames == 750
    assert final.analyzed_raw_sum == -2 * 750
    assert final.analyzed_sample_count == 4 * 750
    assert final.discarded_analysis_frames == 0
    assert sum(item.raw_sum for item in final.bins) == -2 * 750
    assert sum(item.sample_count for item in final.bins) == 4 * 750
    assert final.replaced_preview_frames == 749
    assert final.display_updates == 1
    assert final.last_display_duration_ns == 250_000_000
    assert not final.active


def test_controller_scope_dma_mode_renders_accumulated_summaries(qtbot: QtBot) -> None:
    mca = cast(MultiChannelAnalyzer, SimpleNamespace())
    scope = _FakeScopeController()
    accumulator = ScopeCurrentAccumulator()
    controller = CurrentMonitorController(
        mca,
        channel=0,
        auto_start=False,
        scope_controller=cast(Any, scope),
        scope_current_accumulator=accumulator,
    )
    qtbot.addWidget(controller)
    controller.ui.comboMode.setCurrentIndex(1)

    controller.start_monitor()

    assert scope.starts == 1
    assert not controller.ui.comboMode.isEnabled()
    geometry = ScopeDmaGeometry(
        frame_samples=8188,
        buffer_samples=2052,
        frame_bytes=4104,
        waveform_samples=2046,
        sample_decimation=4,
        padding_bytes=4,
    )
    accumulator.start_session(
        geometry,
        ScopeCurrentRuntime(0, 122, 515, 46_125, "iiod-batched", 4, 32),
    )
    received_ns = time.perf_counter_ns() - 20_000_000
    for timestamp, host_offset_ns in ((1234, 0), (47_359, 10_000_000)):
        accumulator.append_frame(
            _scope_record(timestamp, np.arange(2046, dtype="<i2"), geometry),
            geometry,
            received_ns + host_offset_ns,
        )
    controller._render_pending()
    controller._display_timer.stop()

    assert controller.ui.lblRaw.text() == "1,022.5"
    assert "median spacing 0.369 ms" in controller.ui.lblAcquisition.text()
    assert "window 16.368 us" in controller.ui.lblAcquisition.text()
    assert "Analyzed 2/2 received frames" in controller.ui.lblStatus.text()
    assert "iiod-batched, buffers=4, READBUF x32" in controller.ui.lblStatus.text()
    assert "queue=0/high 0" in controller.ui.lblStatus.text()
    assert controller.configuration_settings()["mode"] == "scope_dma"

    controller.request_monitor_stop()
    assert scope.stops == 1
    assert controller.ui.comboMode.isEnabled()
    assert controller.ui.lblStatus.text() == "Scope DMA stopped."


def test_iio_monitor_client_uses_a_new_context_and_channel_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_attr = SimpleNamespace(value="-1234")
    voltage = SimpleNamespace(attrs={"raw": raw_attr})

    def device(index: int, number: int) -> SimpleNamespace:
        return SimpleNamespace(
            name="vdpp_input_filter",
            id=f"iio:device{number}",
            attrs={"channel_index": SimpleNamespace(value=str(index))},
            find_channel=Mock(return_value=voltage),
        )

    context = SimpleNamespace(devices=[device(0, 8), device(1, 3)])
    context_factory = Mock(return_value=context)
    monkeypatch.setattr(iio_backend_module.iio, "Context", context_factory)
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 1
    backend._uri = "ip:board.local:30431"

    client = backend.create_current_monitor_client()

    context_factory.assert_called_once_with("ip:board.local:30431")
    assert client.read_raw() == -1234
    client.close()
    assert getattr(client, "_context") is None
