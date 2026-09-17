from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest

from nlab.hardware.digitizer.mca_capture import (
    McaCaptureWriter,
    McaDmaOutputMode,
    McaRunSummary,
)
from nlab.workers.dma_workers import IIOMcaDmaWorker, McaDmaWorker, _finish_mca_run

_IIO_EVENTS = np.dtype(
    [
        ("flags", "<u2"),
        ("cfd_q2", "<u2"),
        ("charge_energy", "<u2"),
        ("trapezoid_energy", "<u2"),
        ("timestamp", "<u8"),
    ]
)


def _events(offset: int) -> np.ndarray:
    events = np.zeros(4, dtype=_IIO_EVENTS)
    events["timestamp"] = np.arange(offset, offset + 4)
    events["trapezoid_energy"] = np.arange(100 + offset, 104 + offset)
    events["charge_energy"] = np.arange(10 + offset, 14 + offset)
    return events


def test_binary_writer_appends_raw_batches_without_changing_payload(tmp_path: Path) -> None:
    path = tmp_path / "capture.bin"
    first = _events(0)
    second = _events(4)
    writer = McaCaptureWriter(
        path=path,
        mode=McaDmaOutputMode.BINARY,
        configuration_yaml="ignored: true\n",
        binary_header=b"header",
    )
    writer.append(first.tobytes(), first)
    writer.append(second.tobytes(), second)
    writer.close(complete=True)

    assert path.read_bytes() == b"header" + first.tobytes() + second.tobytes()


def test_binary_writer_never_overwrites_an_existing_measurement(tmp_path: Path) -> None:
    path = tmp_path / "capture.bin"
    path.write_bytes(b"existing")

    with pytest.raises(RuntimeError, match="File exists"):
        McaCaptureWriter(
            path=path,
            mode=McaDmaOutputMode.BINARY,
            configuration_yaml="",
            binary_header=b"header",
        )

    assert path.read_bytes() == b"existing"


def test_hdf5_writer_appends_events_and_embeds_configuration(tmp_path: Path) -> None:
    path = tmp_path / "capture.h5"
    writer = McaCaptureWriter(
        path=path,
        mode=McaDmaOutputMode.HDF5,
        configuration_yaml="format_version: 3\n",
        binary_header=b"",
    )
    writer.append(b"", _events(0))
    writer.append(b"", _events(4))
    writer.close(complete=True)

    with h5py.File(path, "r", swmr=True) as capture:
        assert capture["configuration_yaml"].asstr()[()] == "format_version: 3\n"
        assert int(capture["committed_events"][()]) == 8
        assert bool(capture["capture_complete"][()])
        np.testing.assert_array_equal(capture["events"]["timestamp"], np.arange(8))
        np.testing.assert_array_equal(
            capture["events"]["long_gate"], np.arange(100, 108)
        )
        np.testing.assert_array_equal(
            capture["events"]["short_gate"], np.arange(10, 18)
        )


def test_root_writer_appends_to_real_ttree_and_embeds_configuration(tmp_path: Path) -> None:
    uproot = pytest.importorskip("uproot")
    path = tmp_path / "capture.root"
    writer = McaCaptureWriter(
        path=path,
        mode=McaDmaOutputMode.ROOT,
        configuration_yaml="format_version: 3\n",
        binary_header=b"",
    )
    writer.append(b"", _events(0))
    writer.append(b"", _events(4))
    writer.close(complete=True)

    with uproot.open(path) as capture:
        tree = capture["events"]
        assert tree.classname == "TTree"
        assert tree.num_entries == 8
        assert set(tree.keys()) == {"timestamp", "long_gate", "short_gate"}
        np.testing.assert_array_equal(tree["timestamp"].array(library="np"), np.arange(8))
        assert str(capture["configuration_yaml"]) == "format_version: 3\n"


@pytest.mark.parametrize(
    "mode", [McaDmaOutputMode.BINARY, McaDmaOutputMode.HDF5, McaDmaOutputMode.ROOT]
)
def test_run_summary_is_written_beside_each_capture_format(
    tmp_path: Path, mode: McaDmaOutputMode
) -> None:
    path = tmp_path / f"capture{mode.extension}"
    path.write_bytes(b"capture")
    summary = McaRunSummary(
        channel=1,
        mode=mode,
        path=path,
        started_utc="2026-09-17T10:00:00+00:00",
        finished_utc="2026-09-17T10:00:02+00:00",
        duration_s=2.0,
        records=2048,
        continuity="verified",
        diagnostics={"continuity_valid": True, "list_deadtime_raw": 0},
    )

    sidecar = summary.write_sidecar()

    assert sidecar == path.with_suffix(".run.json")
    assert sidecar is not None
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    assert document["format"] == "nlab-mca-run-summary-v1"
    assert document["average_rate_hz"] == 1024
    assert document["continuity"] == "verified"
    with pytest.raises(FileExistsError):
        summary.write_sidecar()


def test_online_summary_has_no_sidecar_and_legacy_is_unverified() -> None:
    summary = _finish_mca_run(
        channel=0,
        mode=McaDmaOutputMode.ONLINE,
        path=None,
        started_utc=datetime.now(UTC),
        started_monotonic=time.monotonic() - 2,
        records=100,
        diagnostics=None,
        error=None,
    )

    assert summary.continuity == "unverified"
    assert summary.sidecar_path is None
    assert summary.write_sidecar() is None


def test_iio_dropped_records_mark_run_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "capture.h5"
    path.write_bytes(b"capture")
    summary = _finish_mca_run(
        channel=1,
        mode=McaDmaOutputMode.HDF5,
        path=path,
        started_utc=datetime.now(UTC),
        started_monotonic=time.monotonic() - 1,
        records=1024,
        diagnostics={"continuity_valid": False, "list_deadtime_raw": 3},
        error="3 dropped records",
    )

    assert summary.continuity == "invalid"
    assert "Incomplete" in summary.status_text()
    assert "3 dropped" in summary.status_text()
    assert summary.sidecar_path is not None
    assert json.loads(summary.sidecar_path.read_text(encoding="utf-8"))["diagnostics"][
        "list_deadtime_raw"
    ] == 3


def test_iio_worker_emits_verified_summary_for_online_run() -> None:
    class FakeStreamer:
        last_capture_diagnostics = {
            "continuity_valid": True,
            "completed_frames": 2,
            "list_deadtime_raw": 0,
        }

        def stream_events(self, **kwargs: object) -> int:
            kwargs["on_progress"](2048)  # type: ignore[operator]
            return 2048

    worker = IIOMcaDmaWorker(
        streamer=FakeStreamer(),  # type: ignore[arg-type]
        output_mode=McaDmaOutputMode.ONLINE,
        channel=1,
    )
    summaries: list[McaRunSummary] = []
    worker.summary.connect(summaries.append)

    worker.run()

    assert len(summaries) == 1
    assert summaries[0].records == 2048
    assert summaries[0].continuity == "verified"
    assert summaries[0].sidecar_path is None


def test_legacy_worker_reports_unverified_without_loss_counters() -> None:
    class FakeStreamer:
        def stream_events(self, **kwargs: object) -> int:
            kwargs["on_progress"](123)  # type: ignore[operator]
            return 123

    worker = McaDmaWorker(
        streamer=FakeStreamer(),  # type: ignore[arg-type]
        output_mode=McaDmaOutputMode.ONLINE,
        channel=1,
    )
    summaries: list[McaRunSummary] = []
    worker.summary.connect(summaries.append)

    worker.run()

    assert summaries[0].records == 123
    assert summaries[0].continuity == "unverified"
