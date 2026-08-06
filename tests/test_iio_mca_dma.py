from __future__ import annotations

import ctypes
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import h5py
import iio
import numpy as np

from nlab.hardware.digitizer.backends.iio_backend import (
    _LM_EVENT_DTYPE as _BACKEND_LM_EVENT_DTYPE,
)
from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.dma import (
    _LM_EVENT_DTYPE,
    FILE_HEADER_STRUCT,
    IIO_LM_FILE_VERSION,
    IIOMcaDmaStreamer,
)
from nlab.utils.dma_converter import convert_listmode, read_file_header


def _frame(seed: int) -> np.ndarray:
    events = np.zeros(1024, dtype=_LM_EVENT_DTYPE)
    events["flags"] = seed
    events["cfd_q2"] = 4 * seed
    events["charge_energy"] = 100 + seed
    events["trapezoid_energy"] = 200 + seed
    events["timestamp"] = np.arange(1024, dtype=np.uint64) + 1000 * seed
    return events


class _FakeBackend:
    def __init__(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event
        self.first = _frame(1)
        self.tail = _frame(2)

    def read_mca_dma_frame(self) -> np.ndarray:
        self._stop_event.set()
        return self.first

    def mca_dma_measurement_in_progress(self) -> bool:
        return False

    def close_mca_dma_capture(
        self,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> int:
        if on_frame is not None:
            on_frame(self.tail)
        return 1


class _FakeAutoStopBackend(_FakeBackend):
    def read_mca_dma_frame(self) -> np.ndarray:
        return self.first


def test_iio_event_dtype_matches_v121_layout() -> None:
    assert _LM_EVENT_DTYPE == _BACKEND_LM_EVENT_DTYPE
    assert _LM_EVENT_DTYPE.itemsize == 16
    fields = _LM_EVENT_DTYPE.fields
    assert fields is not None
    assert {name: info[1] for name, info in fields.items()} == {
        "flags": 0,
        "cfd_q2": 2,
        "charge_energy": 4,
        "trapezoid_energy": 6,
        "timestamp": 8,
    }


def test_binary_attribute_read_reserves_libiio_terminator(monkeypatch) -> None:
    payload = bytes(range(256)) * 32

    def read_attr(device, name, buf, capacity):
        assert name == b"debug_data"
        assert capacity == 2 * len(payload) + 1
        ctypes.memmove(buf, payload, len(payload))
        return len(payload) + 1

    monkeypatch.setattr(iio, "_d_read_attr", read_attr)
    backend = object.__new__(IIODigitizerBackend)
    device = SimpleNamespace(_device=object())

    assert backend._read_large_pp_attr(device, "debug_data", len(payload)) == payload


def test_binary_attribute_read_accepts_older_unterminated_payload(monkeypatch) -> None:
    payload = bytes(range(256)) * 32

    def read_attr(device, name, buf, capacity):
        ctypes.memmove(buf, payload, len(payload))
        return len(payload)

    monkeypatch.setattr(iio, "_d_read_attr", read_attr)
    backend = object.__new__(IIODigitizerBackend)
    device = SimpleNamespace(_device=object())

    assert backend._read_large_pp_attr(device, "debug_data", len(payload)) == payload


def test_histogram_reassembles_four_transport_chunks(monkeypatch) -> None:
    names = tuple(f"histogram_data{index}" for index in range(4))
    chunks = {
        name: np.full(4096, index, dtype="<u4").tobytes()
        for index, name in enumerate(names)
    }
    backend = object.__new__(IIODigitizerBackend)
    backend._mca_pp = SimpleNamespace(attrs=dict.fromkeys(names), _device=object())

    def read_large(device, name, size):
        assert size == 16384
        return chunks[name]

    monkeypatch.setattr(backend, "_read_large_pp_attr", read_large)

    histogram = backend.read_histogram()

    assert histogram.shape == (16384,)
    for index in range(4):
        np.testing.assert_array_equal(histogram[index * 4096 : (index + 1) * 4096], index)


def test_iio_streamer_writes_versioned_frames_and_preserves_tail(tmp_path: Path) -> None:
    stop_event = threading.Event()
    backend = _FakeBackend(stop_event)
    streamer = IIOMcaDmaStreamer(backend, channel=0)
    event_chunks: list[np.ndarray] = []
    event_lock = threading.Lock()
    progress: list[int] = []
    output = tmp_path / "listmode.bin"

    total = streamer.stream_events(
        stop_event=stop_event,
        filepath=output,
        event_buffer=(event_chunks, event_lock),
        on_progress=progress.append,
    )

    assert total == 2048
    assert progress == [1024, 2048]
    assert len(event_chunks) == 2
    with output.open("rb") as file:
        header = read_file_header(file)
        payload = file.read()
    assert header["version"] == IIO_LM_FILE_VERSION
    assert len(payload) == 2 * 1024 * _LM_EVENT_DTYPE.itemsize
    parsed = np.frombuffer(payload, dtype=_LM_EVENT_DTYPE)
    np.testing.assert_array_equal(parsed[:1024], backend.first)
    np.testing.assert_array_equal(parsed[1024:], backend.tail)
    assert output.stat().st_size == FILE_HEADER_STRUCT.size + len(payload)


def test_converter_understands_iio_listmode_version(tmp_path: Path) -> None:
    stop_event = threading.Event()
    backend = _FakeBackend(stop_event)
    source = tmp_path / "listmode.bin"
    destination = tmp_path / "listmode.h5"
    IIOMcaDmaStreamer(backend, channel=0).stream_events(stop_event, source)

    assert convert_listmode(source, destination) == 2048
    with h5py.File(destination, "r") as h5:
        assert h5.attrs["format_version"] == IIO_LM_FILE_VERSION
        assert h5["events"].dtype == _LM_EVENT_DTYPE
        np.testing.assert_array_equal(
            h5["trapezoid_energy"][:1024], backend.first["trapezoid_energy"]
        )
        np.testing.assert_allclose(h5["cfd_time"][:1024], backend.first["cfd_q2"] / 4.0)


def test_streamer_does_not_refill_after_hardware_time_limit() -> None:
    stop_event = threading.Event()
    backend = _FakeAutoStopBackend(stop_event)

    total = IIOMcaDmaStreamer(backend, channel=0).stream_events(stop_event)

    assert total == 2048
    assert not stop_event.is_set()
