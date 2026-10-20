from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import fields
from time import monotonic, monotonic_ns, sleep
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QTimer
from pytestqt.qtbot import QtBot

from nlab.analysis.coincidence import CoincidenceAnalyzer, CoincidenceSettings
from nlab.analysis.psd import PsdAccumulator
from nlab.hardware.digitizer.dma import IIO_LM_EVENT_DTYPE
from nlab.session.replay import McaReplayBatch, McaReplayReadback
from nlab.workers import session_replay_worker as module
from nlab.workers.session_replay_worker import (
    SessionAnalysisReplayThread,
    SessionAnalysisSnapshot,
)


def _events(tick: int) -> np.ndarray:
    events = np.zeros(4, dtype=IIO_LM_EVENT_DTYPE)
    events["timestamp"] = np.arange(tick, tick + 4)
    events["trapezoid_energy"] = [1000, 2000, 3000, 4000]
    events["charge_energy"] = [500, 1000, 1500, 2000]
    return events


def _result(worker: SessionAnalysisReplayThread) -> SessionAnalysisSnapshot:
    deadline = monotonic() + 5
    while monotonic() < deadline:
        snapshot = worker.take_snapshot()
        if snapshot is not None and (snapshot.complete or snapshot.error is not None):
            return snapshot
        sleep(0.005)
    pytest.fail("replay did not publish a final snapshot")


def _stop(worker: SessionAnalysisReplayThread) -> None:
    worker.request_stop()
    worker.join(2)
    assert not worker.is_alive()


def test_replay_preserves_every_batch_and_interval_but_coalesces_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "monotonic_ns", lambda: 100_000_000)
    batches = [
        McaReplayBatch(channel, sequence, sequence, _events(100 * sequence + channel + 1))
        for sequence in range(50) for channel in (0, 1)
    ]
    readbacks = [
        McaReplayReadback(0, sequence, sequence, SimpleNamespace())  # type: ignore[arg-type]
        for sequence in (10, 40)
    ]
    accesses: list[int] = []
    closed: list[int] = []

    def stream(items: list) -> Iterator:
        try:
            for item in items:
                accesses.append(threading.get_ident())
                yield item
        finally:
            closed.append(threading.get_ident())

    settings = CoincidenceSettings()
    expected = CoincidenceAnalyzer(settings)
    expected_psd = PsdAccumulator(energy_bins=64, ratio_bins=32)
    for batch in batches:
        if 10 <= batch.host_monotonic_ns <= 40:
            expected.add_batch(batch.channel, batch.events)
        if batch.channel == 0:
            expected_psd.add_events(batch.events)
    expected.finish()
    published: list[int] = []
    original = SessionAnalysisReplayThread._publish

    def publish(self: SessionAnalysisReplayThread, **kwargs: object) -> None:
        published.append(self._processed)
        original(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(SessionAnalysisReplayThread, "_publish", publish)
    worker = SessionAnalysisReplayThread(
        (stream(batches), stream(readbacks)), origin_ns=0, started_ns=0,
        psd={0: PsdAccumulator(energy_bins=64, ratio_bins=32)},
        coincidence_settings=settings, coincidence_channels={0: 0, 1: 1},
        coincidence_intervals=[(10, 40)],
    )
    worker.start()
    try:
        result = _result(worker)
        assert result.error is None
        assert result.processed == 102
        assert result.readbacks[0] is readbacks[-1]
        assert len(published) == 2  # first progress and EOF, not every batch
        assert set(accesses) == {worker.ident}
        assert closed == [worker.ident, worker.ident]
        assert result.psd[0].statistics == expected_psd.statistics
        np.testing.assert_array_equal(result.psd[0].matrix, expected_psd.matrix)
        assert not result.psd[0].matrix.flags.writeable
        for field in fields(expected.snapshot()):
            actual = getattr(result.coincidence, field.name)
            wanted = getattr(expected.snapshot(), field.name)
            np.testing.assert_equal(actual, wanted)
            if isinstance(actual, np.ndarray):
                assert not actual.flags.writeable
        assert worker.take_snapshot() is None  # bounded, consuming mailbox

        # EOF closes HDF5 but retains raw PSD batches for asynchronous re-binning.
        worker.request_psd_configuration(0, PsdAccumulator(energy_bins=32, ratio_bins=16))
        rebinned = _result(worker)
        assert rebinned.psd[0].matrix.shape == (32, 16)
        assert rebinned.psd[0].statistics == expected_psd.statistics
        assert rebinned.processed == result.processed
    finally:
        _stop(worker)


def test_timestamp_wait_is_interruptible_and_closes_readers_on_worker() -> None:
    loaded, closed = threading.Event(), []

    def stream() -> Iterator[McaReplayBatch]:
        try:
            loaded.set()
            yield McaReplayBatch(0, 0, 60_000_000_000, _events(1))
        finally:
            closed.append(threading.get_ident())

    worker = SessionAnalysisReplayThread(
        (stream(),), origin_ns=0, started_ns=monotonic_ns(), psd={},
    )
    worker.start()
    assert loaded.wait(2)
    assert worker.take_snapshot() is None  # future batches are not replayed early
    before = monotonic()
    _stop(worker)
    assert monotonic() - before < 0.5
    assert closed == [worker.ident]


def test_reader_error_is_a_visible_scalar_result_and_closes_stream() -> None:
    closed = threading.Event()

    def stream() -> Iterator[McaReplayBatch]:
        try:
            raise OSError("broken HDF5 stream")
            yield  # pragma: no cover
        finally:
            closed.set()

    worker = SessionAnalysisReplayThread((stream(),), origin_ns=0, started_ns=0, psd={})
    worker.start()
    try:
        result = _result(worker)
        assert result.error == "OSError: broken HDF5 stream"
        assert not result.complete
        assert closed.wait(2)
    finally:
        _stop(worker)


def test_short_burst_is_published_before_the_next_distant_batch() -> None:
    batches = [
        McaReplayBatch(0, 0, 0, _events(1)),
        McaReplayBatch(0, 1, 0, _events(10)),
        McaReplayBatch(0, 2, 60_000_000_000, _events(20)),
    ]
    worker = SessionAnalysisReplayThread(
        (iter(batches),), origin_ns=0, started_ns=monotonic_ns(),
        psd={0: PsdAccumulator(energy_bins=32, ratio_bins=16)},
    )
    worker.start()
    try:
        deadline = monotonic() + 2
        while monotonic() < deadline:
            snapshot = worker.take_snapshot()
            if snapshot is not None and snapshot.processed == 2:
                assert not snapshot.complete
                assert snapshot.psd[0].statistics.received == 8
                break
            sleep(0.005)
        else:
            pytest.fail("burst result was not flushed while waiting for the next timestamp")
    finally:
        _stop(worker)


def test_slow_replay_analysis_does_not_block_qt_and_stop_cancels_backlog(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = threading.Event(), threading.Event()
    original = PsdAccumulator.add_events

    def slow(self: PsdAccumulator, events: np.ndarray) -> object:
        entered.set()
        assert release.wait(3)
        return original(self, events)

    monkeypatch.setattr(PsdAccumulator, "add_events", slow)
    batches = [McaReplayBatch(0, seq, 0, _events(1)) for seq in range(100)]
    worker = SessionAnalysisReplayThread(
        (iter(batches),), origin_ns=0, started_ns=0,
        psd={0: PsdAccumulator(energy_bins=32, ratio_bins=16)},
    )
    worker.start()
    try:
        qtbot.waitUntil(entered.is_set)
        fired: list[bool] = []
        QTimer.singleShot(0, lambda: fired.append(True))
        qtbot.waitUntil(lambda: bool(fired), timeout=500)
        worker.request_stop()
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert worker._processed == 1  # stop does not drain the recorded backlog
    finally:
        release.set()
        _stop(worker)
