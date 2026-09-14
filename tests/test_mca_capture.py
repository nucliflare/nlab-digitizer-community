from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from nlab.hardware.digitizer.mca_capture import McaCaptureWriter, McaDmaOutputMode

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
