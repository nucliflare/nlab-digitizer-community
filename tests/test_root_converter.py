from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
import uproot
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox
from pytestqt.qtbot import QtBot

from nlab.app import MainAppWindow
from nlab.hardware.digitizer.dma import (
    _EVENT_DTYPE,
    _LM_EVENT_DTYPE,
    FILE_HEADER_STRUCT,
    FILE_MAGIC,
    IIO_LM_FRAME_RECORDS,
)
from nlab.ui.ui_main_window import Ui_MainWindow
from nlab.utils import root_converter
from nlab.utils.root_converter import ConversionCancelledError, convert_binary_to_root
from nlab.views import root_conversion_dialog
from nlab.views.root_conversion_dialog import RootConversionDialog


def _capture(path: Path, payload: bytes, *, version: int = 1, frame_samples: int = 0) -> None:
    path.write_bytes(FILE_HEADER_STRUCT.pack(FILE_MAGIC, version, 1, 0, 123.5, frame_samples)
                     + payload)


def _events(version: int, count: int) -> np.ndarray:
    events = np.zeros(count, dtype=_EVENT_DTYPE if version == 1 else _LM_EVENT_DTYPE)
    events["timestamp"] = np.arange(count, dtype=np.uint64) + 2**63
    events["marker"] = 255
    events["zc_offset"] = 200
    events["zc_estimation"] = 65535 if version == 1 else -4096
    events["short_energy" if version == 1 else "charge_energy"] = 65535
    events["energy" if version == 1 else "trapezoid_energy"] = 32768
    return events


def _sidecar(src: Path, records: int) -> dict[str, object]:
    metadata: dict[str, object] = {
        "format": "nlab-iio-mca-ndma-v2",
        "capture_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "channel_index": 1,
        "record_bytes": 16,
        "frame_records": 1024,
        "frame_bytes": 16384,
        "frames": records // 1024,
        "records": records,
        "client_record_schema": "vdpp-zc-calc-q2.14-v1",
        "continuity_valid": False,
        "list_deadtime_raw": 7,
    }
    src.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


@pytest.mark.parametrize("version,count", [(1, 7), (2, IIO_LM_FRAME_RECORDS * 2), (1, 0), (2, 0)])
def test_listmode_root_preserves_raw_fields_and_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int, count: int,
) -> None:
    src, dst = tmp_path / "events.bin", tmp_path / "events.root"
    events = _events(version, count)
    if count:
        events[-1] = np.zeros((), dtype=events.dtype)  # padding must not be dropped
    _capture(src, events.tobytes(), version=version)
    monkeypatch.setattr(root_converter, "_BATCH_BYTES", 16384)
    progress: list[int] = []
    result = convert_binary_to_root(src, dst, progress=progress.append)
    assert result.count == count
    assert result.kind == "list-mode events"
    assert progress[0] == 0 and progress[-1] == 100
    assert progress == sorted(set(progress))
    with uproot.open(dst) as root:
        tree = root["events"]
        assert tree.classname == "TTree"
        assert tree.num_entries == count
        assert set(tree.keys()) == set(events.dtype.names or ())
        for name in events.dtype.names or ():
            values = tree[name].array(library="np")
            assert values.dtype.kind == events[name].dtype.kind
            assert values.dtype.itemsize == events[name].dtype.itemsize
            np.testing.assert_array_equal(values, events[name])
        metadata = json.loads(str(root["metadata_json"]))
        assert metadata["version"] == version
        assert metadata["channel"] == 1
        assert metadata["timestamp"] == 123.5
        assert metadata["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
        assert "8 ns ticks" in str(root["event_schema"])
        if version == 2:
            assert metadata["client_record_schema"] == "unqualified-opaque[16]"


@pytest.mark.parametrize("count", [0, 3])
def test_waveform_root_preserves_fixed_arrays_and_uint64_timestamps(
    tmp_path: Path, count: int,
) -> None:
    src, dst = tmp_path / "scope.bin", tmp_path / "scope.root"
    frames = np.zeros(count, dtype=[("timestamp", "<u8"), ("samples", "<i2", (5,))])
    frames["timestamp"] = np.arange(count, dtype=np.uint64) + 2**63
    frames["samples"] = [-32768, -1, 0, 1, 32767]
    _capture(src, frames.tobytes(), frame_samples=9)
    assert convert_binary_to_root(src, dst).count == count
    with uproot.open(dst) as root:
        tree = root["waveforms"]
        assert tree.classname == "TTree"
        assert tree.num_entries == count
        np.testing.assert_array_equal(tree["samples"].array(library="np"), frames["samples"])
        np.testing.assert_array_equal(tree["timestamp"].array(library="np"), frames["timestamp"])
        metadata = json.loads(str(root["metadata_json"]))
        assert metadata["samples_per_frame"] == 5
        assert metadata["sample_period_ns"] is None


def test_root_embeds_verified_iio_diagnostics_and_configuration(tmp_path: Path) -> None:
    src, dst = tmp_path / "events.bin", tmp_path / "events.root"
    _capture(src, _events(2, 1024).tobytes(), version=2)
    metadata = _sidecar(src, 1024)
    config = "connection:\n  ip: 10.0.0.1\nmeasurement:\n  channel: 1\n"
    src.with_suffix(".yaml").write_text(config, encoding="utf-8")
    convert_binary_to_root(src, dst)
    with uproot.open(dst) as root:
        assert json.loads(str(root["capture_metadata_json"])) == metadata
        assert str(root["configuration_yaml"]) == config
        assert json.loads(str(root["metadata_json"]))["client_record_schema"] == (
            "vdpp-zc-calc-q2.14-v1"
        )
    # The converted raw branches remain usable by the existing PSD reader.
    from nlab.analysis.psd_file import inspect_psd_event_file, iter_psd_event_batches

    assert inspect_psd_event_file(dst).channel == 1
    assert sum(len(batch) for batch in iter_psd_event_batches(dst)) == 1024


@pytest.mark.parametrize("fault", ["hash", "geometry", "json"])
def test_reject_invalid_sidecars_before_creating_output(tmp_path: Path, fault: str) -> None:
    src, dst = tmp_path / "events.bin", tmp_path / "events.root"
    _capture(src, _events(2, 1024).tobytes(), version=2)
    metadata = _sidecar(src, 1024)
    metadata["capture_sha256" if fault == "hash" else "records"] = "wrong"
    src.with_suffix(".json").write_text(
        "not JSON" if fault == "json" else json.dumps(metadata), encoding="utf-8",
    )
    with pytest.raises(ValueError):
        convert_binary_to_root(src, dst)
    assert not dst.exists()


@pytest.mark.parametrize("version,frame_samples,payload", [
    (3, 0, b""), (2, 8, bytes(16)), (1, 4, b""),
    (1, 0, bytes(17)), (2, 0, bytes(16)), (1, 9, bytes(19)),
])
def test_reject_bad_versions_geometry_and_partial_frames(
    tmp_path: Path, version: int, frame_samples: int, payload: bytes,
) -> None:
    src, dst = tmp_path / "bad.bin", tmp_path / "bad.root"
    _capture(src, payload, version=version, frame_samples=frame_samples)
    with pytest.raises(ValueError):
        convert_binary_to_root(src, dst)
    assert not dst.exists()


@pytest.mark.parametrize("raw", [b"", bytes(24)])
def test_reject_short_or_foreign_header(tmp_path: Path, raw: bytes) -> None:
    src, dst = tmp_path / "bad.bin", tmp_path / "bad.root"
    src.write_bytes(raw)
    with pytest.raises(ValueError):
        convert_binary_to_root(src, dst)
    assert not dst.exists()


def test_never_overwrites_source_or_existing_destination(tmp_path: Path) -> None:
    src, dst = tmp_path / "source.bin", tmp_path / "existing.root"
    _capture(src, _events(1, 1).tobytes())
    original = src.read_bytes()
    with pytest.raises(ValueError, match="different"):
        convert_binary_to_root(src, src)
    dst.write_bytes(b"existing data")
    with pytest.raises(FileExistsError):
        convert_binary_to_root(src, dst)
    assert src.read_bytes() == original
    assert dst.read_bytes() == b"existing data"


@pytest.mark.parametrize("cancel_after", [0, 1, 50])
def test_cancel_removes_only_its_incomplete_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_after: int,
) -> None:
    src, dst = tmp_path / "source.bin", tmp_path / "cancelled.root"
    _capture(src, _events(1, 10000).tobytes())
    monkeypatch.setattr(root_converter, "_BATCH_BYTES", 16384)
    stop = threading.Event()

    def progress(percent: int) -> None:
        if percent >= cancel_after:
            stop.set()

    with pytest.raises(ConversionCancelledError):
        convert_binary_to_root(src, dst, progress=progress, cancelled=stop.is_set)
    assert not dst.exists()
    assert src.exists()


def test_changed_source_is_rejected_and_partial_export_removed(tmp_path: Path) -> None:
    src, dst = tmp_path / "source.bin", tmp_path / "changed.root"
    _capture(src, _events(1, 1).tobytes())

    def change_source(percent: int) -> None:
        if percent == 30:
            src.write_bytes(src.read_bytes() + bytes(16))

    with pytest.raises(ValueError, match="Source changed"):
        convert_binary_to_root(src, dst, progress=change_source)
    assert not dst.exists()


def test_writer_error_removes_partial_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    src, dst = tmp_path / "source.bin", tmp_path / "failed.root"
    _capture(src, _events(1, 1).tobytes())
    monkeypatch.setattr(uproot, "recreate", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        convert_binary_to_root(src, dst)
    assert not dst.exists()


def test_conversion_dialog_completes_real_conversion(qtbot: QtBot, tmp_path: Path) -> None:
    src, dst = tmp_path / "source.bin", tmp_path / "source.root"
    _capture(src, _events(1, 2).tobytes())
    dialog = RootConversionDialog(src, dst)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: not dialog._running)
    assert dialog.error == ""
    assert dialog.conversion_result is not None
    assert dialog.conversion_result.count == 2
    assert not dialog._thread.isRunning()


def test_dialog_cancellation_keeps_event_loop_responsive_and_waits_for_worker(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gui_thread = threading.get_ident()
    worker_threads: list[int] = []

    def slow_convert(src: Path, dst: Path, *, progress: object, cancelled: object) -> None:
        worker_threads.append(threading.get_ident())
        assert callable(progress) and callable(cancelled)
        progress(50)
        while not cancelled():
            threading.Event().wait(0.01)
        raise ConversionCancelledError()

    monkeypatch.setattr(root_conversion_dialog, "convert_binary_to_root", slow_convert)
    dialog = RootConversionDialog(tmp_path / "source.bin", tmp_path / "cancelled.root")
    qtbot.addWidget(dialog)
    dialog._thread.progress.connect(lambda _value: dialog.reject())
    dialog.show()
    qtbot.waitUntil(lambda: not dialog._running)
    assert worker_threads and worker_threads[0] != gui_thread
    assert dialog.conversion_result is None
    assert dialog.error == ""
    assert not dialog._thread.isRunning()


def test_file_menu_and_handler_convert_without_hardware(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    ui = Ui_MainWindow()
    ui.setupUi(window)  # type: ignore[no-untyped-call]  # Generated Qt form.
    assert ui.actionConvertToRoot in ui.menuFile.actions()
    src, dst = tmp_path / "source.bin", tmp_path / "export.root"
    _capture(src, _events(1, 2).tobytes())
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(src), ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(dst), ""))
    information = Mock()
    monkeypatch.setattr(QMessageBox, "information", information)
    ui.actionConvertToRoot.triggered.connect(
        lambda: MainAppWindow._on_convert_to_root(cast(MainAppWindow, window)),
    )
    ui.actionConvertToRoot.trigger()
    assert dst.exists()
    information.assert_called_once()
    assert "2 list-mode events" in information.call_args.args[2]


@pytest.mark.parametrize("stage", ["source", "destination", "invalid"])
def test_menu_cancel_and_conversion_error_paths(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    src, dst = tmp_path / "invalid.bin", tmp_path / "export.root"
    src.write_bytes(b"not an NDMA file")
    open_dialog = Mock(return_value=("" if stage == "source" else str(src), ""))
    save_dialog = Mock(return_value=("" if stage == "destination" else str(dst), ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", open_dialog)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", save_dialog)
    critical, information = Mock(), Mock()
    monkeypatch.setattr(QMessageBox, "critical", critical)
    monkeypatch.setattr(QMessageBox, "information", information)
    MainAppWindow._on_convert_to_root(cast(MainAppWindow, window))
    assert not dst.exists()
    assert critical.call_count == (1 if stage == "invalid" else 0)
    information.assert_not_called()
    if stage == "source":
        save_dialog.assert_not_called()


@pytest.mark.parametrize("close", [False, True])
def test_cancelling_dialog_before_worker_starts_is_safe(
    qtbot: QtBot, tmp_path: Path, close: bool,
) -> None:
    src, dst = tmp_path / "source.bin", tmp_path / "cancelled.root"
    _capture(src, _events(1, 1).tobytes())
    dialog = RootConversionDialog(src, dst)
    qtbot.addWidget(dialog)
    dialog.show()
    if close:
        dialog.close()
    else:
        dialog.reject()
    qtbot.waitUntil(lambda: not dialog._running)
    assert not dialog._thread.isRunning()
    assert dialog.error == ""
    assert dialog.conversion_result is None
    assert not dst.exists()
