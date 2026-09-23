from __future__ import annotations

import threading
from pathlib import Path

import h5py
import numpy as np
import pytest
from PySide6.QtCore import QThread, Signal
from pytestqt.qtbot import QtBot

import nlab.views.psd_readback_dialog as psd_readback_module
from nlab.analysis import psd_file
from nlab.analysis.psd import PsdAccumulator
from nlab.analysis.psd_file import inspect_psd_event_file, iter_psd_event_batches
from nlab.hardware.digitizer.dma import (
    _LM_EVENT_DTYPE,
    FILE_HEADER_STRUCT,
    FILE_MAGIC,
    IIO_LM_FILE_VERSION,
)
from nlab.hardware.digitizer.mca_capture import McaCaptureWriter, McaDmaOutputMode
from nlab.views.psd_readback_dialog import PsdReadbackDialog
from nlab.workers.base_worker import BaseWorker


def _events(count: int) -> np.ndarray:
    events = np.zeros(count, dtype=_LM_EVENT_DTYPE)
    events["timestamp"] = np.arange(count)
    events["trapezoid_energy"] = np.arange(count) % 1000 + 100
    events["charge_energy"] = events["trapezoid_energy"] // 2
    return events


def _header(channel: int = 0) -> bytes:
    return FILE_HEADER_STRUCT.pack(FILE_MAGIC, IIO_LM_FILE_VERSION, channel, 0, 0.0, 0)


def test_ndma_reader_uses_bounded_complete_frame_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.bin"
    events = _events(2048)
    writer = McaCaptureWriter(
        path=path,
        mode=McaDmaOutputMode.BINARY,
        configuration_yaml="",
        binary_header=_header(),
    )
    writer.append(events[:1024].tobytes(), events[:1024])
    writer.append(events[1024:].tobytes(), events[1024:])
    writer.close(complete=True)
    monkeypatch.setattr(psd_file, "_TARGET_BATCH_BYTES", 16 * 1024)

    info = inspect_psd_event_file(path)
    batches = list(iter_psd_event_batches(path))

    assert info.channel == 0
    assert info.total_events == 2048
    assert [len(batch) for batch in batches] == [1024, 1024]
    np.testing.assert_array_equal(np.concatenate(batches), events)


def test_hdf5_reader_honours_committed_rows_and_configuration_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.h5"
    events = _events(20)
    with h5py.File(path, "w") as file:
        file.create_dataset("events", data=events)
        file.create_dataset("committed_events", data=np.uint64(13))
        file.create_dataset(
            "configuration_yaml",
            data="measurement:\n  channel: 1\n",
            dtype=h5py.string_dtype("utf-8"),
        )
    monkeypatch.setattr(psd_file, "_TARGET_BATCH_BYTES", 5 * events.dtype.itemsize)

    info = inspect_psd_event_file(path)
    batches = list(iter_psd_event_batches(path))

    assert info.channel == 1
    assert info.total_events == 13
    assert [len(batch) for batch in batches] == [5, 5, 3]


def test_earlier_three_field_hdf5_capture_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "earlier-events.h5"
    old_dtype = np.dtype(
        [("timestamp", "<u8"), ("long_gate", "<u2"), ("short_gate", "<u2")]
    )
    events = np.zeros(3, dtype=old_dtype)
    events["timestamp"] = [1, 2, 3]
    events["long_gate"] = [100, 200, 300]
    events["short_gate"] = [25, 50, 75]
    with h5py.File(path, "w") as capture:
        capture.attrs["format_version"] = 1
        capture.create_dataset("events", data=events)

    batches = list(iter_psd_event_batches(path))
    assert inspect_psd_event_file(path).total_events == 3
    np.testing.assert_array_equal(np.concatenate(batches)["long_gate"], [100, 200, 300])


def test_native_hdf5_capture_feeds_psd_accumulator(tmp_path: Path) -> None:
    path = tmp_path / "native-events.h5"
    events = _events(100)
    writer = McaCaptureWriter(
        path=path,
        mode=McaDmaOutputMode.HDF5,
        configuration_yaml="measurement:\n  channel: 1\n",
        binary_header=b"",
    )
    writer.append(b"", events)
    writer.close(complete=True)

    accumulator = PsdAccumulator()
    for batch in iter_psd_event_batches(path):
        accumulator.add_events(batch)

    assert accumulator.statistics.received == 100
    assert accumulator.statistics.accepted == 100


def test_standalone_psd_readback_loads_hdf5(
    tmp_path: Path,
    qtbot: QtBot,
) -> None:
    path = tmp_path / "readback.h5"
    events = _events(10)
    with h5py.File(path, "w") as file:
        file.create_dataset("events", data=events)
    dialog = PsdReadbackDialog()
    qtbot.addWidget(dialog)

    dialog.open_path(path)
    qtbot.waitUntil(lambda: dialog._thread is None, timeout=5000)

    assert int(dialog.plot._matrix.sum()) == 10
    assert "Loaded 10 events" in dialog.status.text()


def test_psd_readback_worker_is_deleted_by_its_finished_thread(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "worker-lifetime.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("events", data=_events(1))

    class BlockingWorker(BaseWorker):
        progress = Signal(object, object)
        loaded = Signal(object, object, str)
        cancelled = Signal()

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__()
            self.release = threading.Event()

        def run(self) -> None:
            assert self.release.wait(5)
            self.finished.emit()

        def stop(self) -> None:
            self.release.set()

    finish_threads: list[QThread] = []

    class TrackingDialog(PsdReadbackDialog):
        def _thread_finished(self) -> None:
            finish_threads.append(QThread.currentThread())
            super()._thread_finished()

    monkeypatch.setattr(psd_readback_module, "PsdFileWorker", BlockingWorker)
    dialog = TrackingDialog()
    qtbot.addWidget(dialog)
    dialog.open_path(path)
    worker = dialog._worker
    assert isinstance(worker, BlockingWorker)
    destroyed: list[bool] = []
    worker.destroyed.connect(lambda: destroyed.append(True))
    worker.release.set()

    qtbot.waitUntil(lambda: dialog._thread is None, timeout=5000)

    assert destroyed == [True]
    assert finish_threads == [dialog.thread()]


def test_root_reader_iterates_tree_and_feeds_psd_accumulator(tmp_path: Path) -> None:
    pytest.importorskip("uproot")
    path = tmp_path / "events.root"
    events = _events(100)
    writer = McaCaptureWriter(
        path=path,
        mode=McaDmaOutputMode.ROOT,
        configuration_yaml="measurement:\n  channel: 0\n",
        binary_header=b"",
    )
    writer.append(b"", events)
    writer.close(complete=True)

    info = inspect_psd_event_file(path)
    accumulator = PsdAccumulator()
    for batch in iter_psd_event_batches(path):
        accumulator.add_events(batch)

    assert info.format_name == "ROOT TTree"
    assert info.channel == 0
    assert accumulator.statistics.received == 100
    assert accumulator.statistics.accepted == 100
