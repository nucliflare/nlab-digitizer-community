from __future__ import annotations

import builtins
import json
import struct
import threading
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import uproot
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox
from pytestqt.qtbot import QtBot

from nlab.analysis import waveform_cluster_export as exporter
from nlab.analysis.waveform_cluster_export import (
    ClusterExportCancelledError,
    ClusterFormat,
    export_waveform_cluster,
)
from nlab.analysis.waveform_file import MappedWaveformFile, build_waveform_file_index
from nlab.hardware.digitizer.dma import FILE_HEADER_STRUCT, FILE_MAGIC
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog
from nlab.views.waveform_cluster_export_dialog import (
    ClusterExportRequest,
    WaveformClusterExportDialog,
)


def _capture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    samples = np.tile([100, 100, 100, 100, 90, 80, 70, 80, 90, 100], (6, 1))
    samples[1::2, 7:9] = 110
    samples = samples.astype("<i2")
    samples[4, -1] = -32768
    timestamps = np.arange(6, dtype=np.uint64) + 2**63
    records = np.zeros(6, dtype=[("timestamp", "<u8"), ("samples", "<i2", (10,))])
    records["timestamp"], records["samples"] = timestamps, samples
    path.write_bytes(FILE_HEADER_STRUCT.pack(FILE_MAGIC, 1, 1, 0, 123.5, 14) + records.tobytes())
    return samples, timestamps


def _export(path: Path, destination: Path, file_format: ClusterFormat, **kwargs: object) -> object:
    return export_waveform_cluster(
        build_waveform_file_index(path), destination, file_format=file_format, source_index=0,
        frames=np.array([1, 4, 5]), energy=np.array([40., 90., 40.]),
        ratio=np.array([-.5, 1/3, -.5]),
        metadata={"settings": {"baseline_start": 0, "baseline_end": 4},
                  "sample_period_ns": 8., "selection_vertices": [[0, -1], [100, 1], [0, 1]]},
        **kwargs,
    )


@pytest.mark.parametrize("file_format", ["bin", "root", "h5"])
def test_native_cluster_export_roundtrip_exact_samples_timestamps_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_format: ClusterFormat,
) -> None:
    path = tmp_path / "source.bin"
    raw, timestamps = _capture(path)
    before = path.read_bytes()
    path.with_suffix(".yaml").write_text("measurement: {channel: 1}", encoding="utf-8")
    # Force more than one output batch even for this tiny capture.
    monkeypatch.setattr(exporter, "_BATCH_BYTES", 1)
    destination = tmp_path / f"cluster.{file_format}"
    progress: list[int] = []
    result = _export(path, destination, file_format, progress=progress.append)
    assert result.count == 3 and progress[0] == 0 and progress[-1] == 100
    selected = np.array([1, 4, 5])
    if file_format == "bin":
        assert destination.read_bytes()[:24] == before[:24]
        with destination.with_suffix(".cluster.json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        np.testing.assert_array_equal(metadata["source_frame"], selected)
        np.testing.assert_allclose(metadata["psd"], [-.5, 1/3, -.5])
        reader = MappedWaveformFile(build_waveform_file_index(destination))
        try:
            for number, original in enumerate(selected):
                event = reader.frame(number)
                assert event.timestamp == timestamps[original]
                np.testing.assert_array_equal(event.samples, raw[original])
        finally:
            reader.close()
    elif file_format == "root":
        with uproot.open(destination) as root:
            assert root["waveforms"].classname == "TTree"
            arrays = root["waveforms"].arrays(library="np")
            np.testing.assert_array_equal(arrays["samples"], raw[selected])
            np.testing.assert_array_equal(arrays["timestamp"], timestamps[selected])
            np.testing.assert_array_equal(arrays["source_frame"], selected)
            np.testing.assert_allclose(arrays["psd"], [-.5, 1/3, -.5])
            assert arrays["timestamp"].dtype.kind == "u"
            assert arrays["timestamp"].dtype.itemsize == 8
            metadata = json.loads(str(root["metadata_json"]))
    else:
        with h5py.File(destination, "r") as h5:
            np.testing.assert_array_equal(h5["waveforms"][:], raw[selected])
            np.testing.assert_array_equal(h5["timestamps"][:], timestamps[selected])
            np.testing.assert_array_equal(h5["source_frame"][:], selected)
            np.testing.assert_allclose(h5["psd"][:], [-.5, 1/3, -.5])
            assert h5["timestamps"].dtype == np.dtype("uint64")
            metadata = json.loads(h5.attrs["metadata_json"])
    assert metadata["source_file"] == str(path.resolve())
    assert metadata["total_frames"] == 3
    assert metadata["sample_period_ns"] == 8.
    assert metadata["timestamp_unit"] == "8 ns ticks"
    assert metadata["configuration_yaml"] == "measurement: {channel: 1}"
    assert metadata["settings"]["baseline_end"] == 4
    assert path.read_bytes() == before


def _caen(path: Path) -> list[np.ndarray]:
    waves = [np.arange(10, dtype="<i2"), np.arange(15, dtype="<i2"),
             np.arange(12, dtype="<i2")]
    payload = struct.pack("<H", 0xCAED)  # raw/short charge + waveform
    for number, samples in enumerate(waves):
        payload += struct.pack("<HHQHHIBi", 3, 2, 2**63 + number, 65535, 32768,
                               0xF0000000 + number, 7, len(samples)) + samples.tobytes()
    path.write_bytes(payload)
    return waves


@pytest.mark.parametrize("file_format", ["root", "h5"])
def test_variable_length_caen_cluster_preserves_full_records(
    tmp_path: Path, file_format: ClusterFormat,
) -> None:
    path = tmp_path / "caen.bin"
    waves = _caen(path)
    destination = tmp_path / f"caen-cluster.{file_format}"
    export_waveform_cluster(
        build_waveform_file_index(path), destination, file_format=file_format, source_index=0,
        frames=np.array([0, 2]), energy=np.array([20., 30.]), ratio=np.array([.1, .2]), metadata={},
    )
    if file_format == "root":
        with uproot.open(destination) as root:
            arrays = root["waveforms"].arrays(library="np")
            for row, original in enumerate([0, 2]):
                np.testing.assert_array_equal(arrays["samples"][row], waves[original])
            assert arrays["timestamp"].tolist() == [2**63, 2**63 + 2]
            assert arrays["flags"].tolist() == [0xF0000000, 0xF0000002]
            assert arrays["stored_long_gate"].tolist() == [65535, 65535]
            assert arrays["stored_short_gate"].tolist() == [32768, 32768]
            assert arrays["has_stored_gates"].all()
            assert arrays["waveform_code"].tolist() == [7, 7]
    else:
        with h5py.File(destination) as h5:
            for row, original in enumerate([0, 2]):
                np.testing.assert_array_equal(h5["waveforms"][row], waves[original])
            assert h5["timestamps"][:].tolist() == [2**63, 2**63 + 2]
            assert h5["flags"][:].tolist() == [0xF0000000, 0xF0000002]
            assert h5["stored_long_gate"][:].tolist() == [65535, 65535]
            assert h5["has_flags"][:].all()
    with pytest.raises(ValueError, match="NDMA source"):
        export_waveform_cluster(
            build_waveform_file_index(path), tmp_path / "unsupported.bin", file_format="bin",
            source_index=0, frames=np.array([0, 2]), energy=np.ones(2), ratio=np.ones(2),
            metadata={},
        )
    assert not (tmp_path / "unsupported.bin").exists()


@pytest.mark.parametrize("file_format", ["bin", "root", "h5"])
def test_cancellation_source_change_and_existing_file_protection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_format: ClusterFormat,
) -> None:
    path = tmp_path / "source.bin"
    _capture(path)
    before = path.read_bytes()
    destination = tmp_path / f"cluster.{file_format}"
    destination.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError):
        _export(path, destination, file_format)
    assert destination.read_bytes() == b"do not overwrite"
    with pytest.raises(ValueError, match="different"):
        _export(path, path, file_format)
    destination = tmp_path / f"new-cluster.{file_format}"
    stop = threading.Event()
    monkeypatch.setattr(exporter, "_BATCH_BYTES", 1)

    def report(value: int) -> None:
        if value > 0:
            stop.set()

    # Binary progress is less frequent, so a cancellation callback also observes
    # the file creation itself. Other formats cancel after their first batch.
    def cancelled() -> bool:
        return destination.exists() if file_format == "bin" else stop.is_set()

    with pytest.raises(ClusterExportCancelledError):
        _export(path, destination, file_format, cancelled=cancelled, progress=report)
    assert not destination.exists() and not destination.with_suffix(".cluster.json").exists()
    assert path.read_bytes() == before

    original_frame = exporter.MappedWaveformFile.frame

    def changed(reader: MappedWaveformFile, *args: object) -> object:
        frame = original_frame(reader, *args)
        import os
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return frame

    monkeypatch.setattr(exporter.MappedWaveformFile, "frame", changed)
    with pytest.raises(ValueError, match="changed"):
        _export(path, destination, file_format)
    assert not destination.exists()


def test_binary_sidecar_collision_and_invalid_selection_do_not_create_outputs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.bin"
    _capture(path)
    destination = tmp_path / "cluster.bin"
    sidecar = destination.with_suffix(".cluster.json")
    sidecar.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _export(path, destination, "bin")
    assert sidecar.read_text(encoding="utf-8") == "existing" and not destination.exists()
    for frames in (np.array([], dtype=int), np.array([1, 1]), np.array([5, 1]), np.array([99])):
        with pytest.raises(ValueError, match="source frames"):
            export_waveform_cluster(
                build_waveform_file_index(path), tmp_path / "invalid.root", file_format="root",
                source_index=0, frames=frames, energy=np.ones(len(frames)),
                ratio=np.ones(len(frames)),
                metadata={},
            )
    assert not (tmp_path / "invalid.root").exists()


@pytest.mark.parametrize("extension", ["bin", "root", "h5"])
def test_workbench_file_menu_exports_applied_cluster_and_invalidates_availability(
    tmp_path: Path, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, extension: str,
) -> None:
    path = tmp_path / "source.bin"
    _capture(path)
    destination = tmp_path / f"from-menu.{extension}"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", Mock(return_value=(str(destination), "")))
    warning = Mock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    assert dialog.export_cluster_action in dialog.file_menu.actions()
    assert not dialog.export_cluster_action.isEnabled()
    dialog.auto_recalculate.setChecked(False)
    dialog.open_path(path)
    qtbot.waitUntil(lambda: dialog._index_thread is None, timeout=5000)
    dialog.polarity_combo.setCurrentIndex(dialog.polarity_combo.findData(-1))
    period = dialog.sample_period.value()
    dialog._updating_gates = True
    dialog.baseline_region.setRegion((0, 4 * period))
    for line, sample in ((dialog.gate_start_line, 4), (dialog.short_end_line, 7),
                         (dialog.long_end_line, 10)):
        line.setPos(sample * period)
    dialog._updating_gates = False
    dialog.energy_max.setValue(1000)
    dialog._start_analysis()
    qtbot.waitUntil(lambda: dialog._analysis_thread is None, timeout=5000)
    cluster = dialog.cluster_selection
    cluster.add_region("rectangle")
    cluster._roi.setPos((0, -.6))
    cluster._roi.setSize((1000, .5))
    cluster.apply_selection()
    assert cluster._selected.tolist() == [1, 3, 5]
    assert dialog.export_cluster_action.isEnabled()
    dialog.waveform_selection_checkbox.setChecked(False)
    assert dialog.export_cluster_action.isEnabled()
    dialog.export_cluster_action.trigger()
    assert destination.exists() and "Exported 3 waveforms" in dialog.status.text()
    assert not warning.called
    assert cluster._selected.tolist() == [1, 3, 5]
    # Existing outputs fail visibly, without modifying the valid export.
    before = destination.read_bytes()
    dialog.export_cluster_action.trigger()
    assert warning.call_count == 1 and "exists" in str(warning.call_args)
    assert destination.read_bytes() == before
    cluster.clear_selection()
    assert not dialog.export_cluster_action.isEnabled()
    dialog.close_file()


def test_export_dialog_cancellation_before_start_is_safe(tmp_path: Path, qtbot: QtBot) -> None:
    path = tmp_path / "source.bin"
    _capture(path)
    request = ClusterExportRequest(
        build_waveform_file_index(path), tmp_path / "cancelled.bin", "bin", 0,
        np.array([1]), np.ones(1), np.ones(1), {},
    )
    dialog = WaveformClusterExportDialog(request)
    qtbot.addWidget(dialog)
    dialog.reject()
    assert dialog.stop_sync()
    QTimer.singleShot(0, lambda: None)
    qtbot.waitUntil(lambda: not dialog._running)
    assert not request.destination.exists() and dialog.export_result is None


def test_missing_root_dependency_is_reported_and_partial_file_removed(
    tmp_path: Path, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.bin"
    _capture(path)
    request = ClusterExportRequest(
        build_waveform_file_index(path), tmp_path / "missing.root", "root", 0,
        np.array([1]), np.ones(1), np.ones(1), {},
    )
    original_import = builtins.__import__

    def missing(name: str, *args: object, **kwargs: object) -> object:
        if name == "uproot":
            raise ImportError("simulated missing frozen uproot dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    dialog = WaveformClusterExportDialog(request)
    qtbot.addWidget(dialog)
    dialog.exec()
    assert "ImportError" in dialog.error and "uproot" in dialog.error
    assert dialog.export_result is None and not request.destination.exists()


def test_cancellation_while_writing_cleans_up_before_dialog_finishes(
    tmp_path: Path, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.bin"
    _capture(path)
    request = ClusterExportRequest(
        build_waveform_file_index(path), tmp_path / "cancelled.h5", "h5", 0,
        np.array([1, 2, 3]), np.ones(3), np.ones(3), {},
    )
    monkeypatch.setattr(exporter, "_BATCH_BYTES", 1)
    entered = threading.Event()
    original = exporter.MappedWaveformFile.frame
    dialog = WaveformClusterExportDialog(request)
    qtbot.addWidget(dialog)

    def pause(reader: MappedWaveformFile, *args: object) -> object:
        entered.set()
        assert dialog._thread.stop.wait(5)
        return original(reader, *args)

    monkeypatch.setattr(exporter.MappedWaveformFile, "frame", pause)
    dialog.show()
    qtbot.waitUntil(entered.is_set, timeout=5000)
    assert request.destination.exists()
    dialog.close()  # Must request cancellation, not destroy its active thread.
    qtbot.waitUntil(lambda: not dialog._running, timeout=5000)
    assert not dialog._thread.isRunning()
    assert not dialog.error and dialog.export_result is None
    assert not request.destination.exists()
