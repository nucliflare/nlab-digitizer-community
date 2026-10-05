from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLabel, QMainWindow
from pytestqt.qtbot import QtBot

from nlab.app import MainAppWindow
from nlab.hardware.digitizer.dma import (
    McaEventBuffer,
    ScopeDmaFrame,
    ScopeDmaGeometry,
    ScopeFrameBuffer,
)
from nlab.hardware.digitizer.iio_listmode import IIO_LM_EVENT_DTYPE
from nlab.session.recorder import SessionRecorder
from nlab.session.replay import SessionReplay
from nlab.ui.ui_main_window import Ui_MainWindow
from nlab.workers.current_monitor_worker import CurrentSample, CurrentSampleBatch
from nlab.workers.mca_worker import MCAReadback


def _events() -> np.ndarray:
    events = np.zeros(5, dtype=IIO_LM_EVENT_DTYPE)
    # The first event is valid despite timestamp zero.  The last record is
    # intentionally all-zero DMA padding and must not enter the spectrum.
    events[0]["marker"] = 1
    events[0]["trapezoid_energy"] = 40
    events[1]["timestamp"] = 1
    events[1]["trapezoid_energy"] = 40
    events[2]["timestamp"] = 2
    events[2]["trapezoid_energy"] = 44
    events[3]["timestamp"] = 3
    events[3]["trapezoid_energy"] = 65535
    return events


def test_session_actions_are_available_from_file_menu(qtbot: QtBot) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    ui = Ui_MainWindow()
    ui.setupUi(window)

    actions = ui.menuFile.actions()
    assert ui.actionStartSessionRecording in actions
    assert ui.actionStopSessionRecording in actions
    assert ui.actionReplaySession in actions
    assert ui.actionExitReplay in actions
    assert ui.actionStartSessionRecording.isEnabled()
    assert not ui.actionStopSessionRecording.isEnabled()
    assert not ui.actionExitReplay.isEnabled()


def test_replay_indicator_is_persistent_and_warns_that_hardware_is_live(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    class IndicatorWindow(MainAppWindow):
        def __init__(self) -> None:
            QMainWindow.__init__(self)
            self._base_window_title = "Nuclear Lab Digitizer - test"
            actions = {}
            for name in (
                "actionExitReplay",
                "actionStartSessionRecording",
                "actionReconnectDevice",
                "actionRefreshModbus",
                "actionLoadSettings",
                "actionGeneralSettings",
                "actionRebootBoard",
                "actionShutdownBoard",
            ):
                action = QAction(self)
                action.setObjectName(name)
                actions[name] = action
            self.ui = SimpleNamespace(**actions)
            self._replay_indicator = QLabel(self)

    window = IndicatorWindow()
    qtbot.addWidget(window)
    replay = SimpleNamespace(
        capture_complete=True,
        integrity="verified",
        manifest={"session_id": "session-123"},
    )
    path = tmp_path / "measurement.nlab-session"

    window._set_replay_indicator(replay, path)  # type: ignore[arg-type]

    assert window.ui.actionExitReplay.isEnabled()
    assert not window.ui.actionLoadSettings.isEnabled()
    assert not window.ui.actionReconnectDevice.isEnabled()
    assert not window.ui.actionStartSessionRecording.isEnabled()
    assert not window._replay_indicator.isHidden()
    assert "REPLAY" in window._replay_indicator.text()
    assert "Recorded Scope, MCA, PSU, and external-module updates play once" in (
        window._replay_indicator.toolTip()
    )
    assert "all hardware-setting controls are locked" in (
        window._replay_indicator.toolTip()
    )
    assert window.windowTitle().endswith("[REPLAY]")

    window._set_replay_indicator(None, None)

    assert not window.ui.actionExitReplay.isEnabled()
    assert window.ui.actionLoadSettings.isEnabled()
    assert window.ui.actionReconnectDevice.isEnabled()
    assert window.ui.actionStartSessionRecording.isEnabled()
    assert window._replay_indicator.isHidden()
    assert window.windowTitle() == window._base_window_title


def test_session_records_streams_and_replays_mca_deterministically(tmp_path: Path) -> None:
    mca_publisher = McaEventBuffer()
    scope_publisher = ScopeFrameBuffer()
    scope_dma_only_publisher = ScopeFrameBuffer()
    path = tmp_path / "measurement.nlab-session"
    recorder = SessionRecorder(
        path,
        {"format_version": 4, "connection": {"backend": "iio"}},
        mca_event_buffers={0: mca_publisher},
        scope_frame_buffers={0: scope_publisher, 1: scope_dma_only_publisher},
        application_version="test",
    )
    recording_manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert recording_manifest["capture_complete"] is False
    assert recording_manifest["integrity"] == "recording"

    recorder.record_timeline("coincidence_started", "coincidence", {"session_id": "test"})
    mca_publisher.append(_events())
    recorder.record_timeline("coincidence_finished", "coincidence", {"session_id": "test"})
    recorder.record_mca_readback(
        0,
        MCAReadback(
            histogram=np.arange(8, dtype=np.uint32),
            debug1=np.array([-2, 3], dtype=np.int16),
            debug2=np.array([4], dtype=np.int16),
            count_rate=12,
            events_lost=None,
        ),
    )
    geometry = ScopeDmaGeometry.legacy(8)
    scope_publisher.append(
        ScopeDmaFrame(
            timestamp=123,
            received_ns=time.monotonic_ns(),
            samples=np.array([1, -2, 3, -4], dtype=np.int16),
            geometry=geometry,
        )
    )
    scope_dma_only_publisher.append(
        ScopeDmaFrame(
            timestamp=124,
            received_ns=time.monotonic_ns(),
            samples=np.array([7, -8, 9, -10], dtype=np.int16),
            geometry=geometry,
        )
    )
    recorder.record_scope_viewer(0, np.array([5, 6], dtype=np.int16), sample_period_ns=8)
    recorder.record_current_samples(
        0,
        CurrentSampleBatch(
            samples=(
                CurrentSample(
                    sequence=7,
                    timestamp_ns=99,
                    raw_code=12.5,
                    read_latency_ns=1000,
                    hardware_timestamp=42,
                ),
            ),
            dropped_samples=0,
            error=None,
        ),
    )
    recorder.record_command("mca.ch0", "start", {"mode": "list"})
    recorder.record_telemetry(
        "psu.ch0",
        {"readback": {"timestamp": 1.0, "hv_voltage": 42.5}},
    )
    recorder.record_telemetry(
        "external.sipm:3",
        {"worker_timestamp": 2.0, "values": {"cpu_temp": 31.5}},
    )
    recorder.stop(final_configuration={"format_version": 4})

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["capture_complete"] is True
    assert manifest["integrity"] == "verified"
    assert manifest["stream_counts"]["mca_ch0_list_mode_events"] == 5
    assert set(manifest["sha256"]) == {
        "configuration.yaml",
        "streams.h5",
        "timeline.jsonl",
    }
    assert manifest["mutable_files"] == ["annotations.yaml"]

    replay = SessionReplay(path)
    assert replay.annotations()["label_status"] == "unlabeled"
    first = replay.mca_event_spectrum(0)
    second = replay.mca_event_spectrum(0)
    assert np.array_equal(first, second)
    assert first.dtype == np.uint64
    assert first[10] == 2
    assert first[11] == 1
    assert first[16383] == 1
    assert int(first.sum()) == 4
    assert np.array_equal(replay.mca_checkpoint(0), np.arange(8, dtype=np.uint32))
    replay_batches = replay.mca_event_batches()
    assert len(replay_batches) == 1
    assert replay_batches[0].channel == 0
    np.testing.assert_array_equal(replay_batches[0].events, _events())
    replay_readbacks = replay.mca_readbacks()
    assert len(replay_readbacks) == 1
    np.testing.assert_array_equal(
        replay_readbacks[0].readback.histogram,
        np.arange(8, dtype=np.uint32),
    )
    assert replay.replay_mca_spectra()[0].metadata["replay_source"] == "list_mode_events"
    scope_frames = replay.scope_frames()
    assert len(scope_frames) == 2
    scope_by_channel = {frame.channel: frame for frame in scope_frames}
    assert scope_by_channel[0].source == "viewer"
    assert scope_by_channel[0].sample_period_ns == 8
    np.testing.assert_array_equal(scope_by_channel[0].samples, [5, 6])
    assert scope_by_channel[1].source == "dma"
    assert scope_by_channel[1].sample_period_ns == 2
    np.testing.assert_array_equal(scope_by_channel[1].samples, [7, -8, 9, -10])

    with h5py.File(path / "streams.h5", "r") as h5:
        assert int(h5["capture_complete"][()]) == 1
        assert len(h5["scope/ch0/dma/frames"]) == 1
        assert len(h5["scope/ch0/viewer/frames"]) == 1
        assert len(h5["scope/ch1/dma/frames"]) == 1
        assert len(h5["current/ch0/samples"]) == 1
        assert h5["mca/ch0/list_mode/events"].attrs["committed_records"] == 5
        assert h5["mca/ch0/list_mode/batches"].attrs["committed_records"] == 1

    timeline = list(replay.timeline())
    assert [record["sequence"] for record in timeline] == list(range(len(timeline)))
    assert any(record["kind"] == "command" for record in timeline)
    assert any(record["kind"] == "configuration_snapshot" for record in timeline)
    assert replay.configuration_snapshot() == {"format_version": 4}
    telemetry = replay.telemetry_events()
    assert [event.source for event in telemetry] == ["psu.ch0", "external.sipm:3"]
    assert telemetry[0].payload["readback"]["hv_voltage"] == 42.5
    assert replay.has_explicit_mca_batch_timing()
    coincidence_intervals = replay.coincidence_intervals()
    assert len(coincidence_intervals) == 1
    assert coincidence_intervals[0][1] is not None


def test_existing_session_directory_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "existing.nlab-session"
    path.mkdir()
    sentinel = path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        SessionRecorder(path, {})

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_replay_rejects_tampered_finalized_member(tmp_path: Path) -> None:
    path = tmp_path / "tampered.nlab-session"
    recorder = SessionRecorder(path, {})
    recorder.stop()
    with (path / "timeline.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")

    with pytest.raises(ValueError, match="integrity check failed"):
        SessionReplay(path)


def test_mutable_annotations_do_not_invalidate_capture_hashes(tmp_path: Path) -> None:
    path = tmp_path / "annotations.nlab-session"
    recorder = SessionRecorder(path, {})
    recorder.stop()
    (path / "annotations.yaml").write_text(
        "format: nlab-session-annotations\n"
        "format_version: 1\n"
        "label_status: verified\n"
        "isotopes:\n"
        "  - nuclide: Cs-137\n",
        encoding="utf-8",
    )

    replay = SessionReplay(path)

    assert replay.annotations()["isotopes"] == [{"nuclide": "Cs-137"}]


def test_replay_rejects_unknown_session_schema(tmp_path: Path) -> None:
    path = tmp_path / "future.nlab-session"
    recorder = SessionRecorder(path, {})
    recorder.stop()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported session format_version"):
        SessionReplay(path, verify_hashes=False)
