from __future__ import annotations

import ctypes
import errno
import gc
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import h5py
import iio
import numpy as np
import pytest

from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.dma import (
    FILE_HEADER_STRUCT,
    FILE_MAGIC,
    FILE_VERSION,
    IIOScopeDmaStreamer,
)
from nlab.utils.dma_converter import convert_scope


def test_binary_viewer_returns_complete_4096_sample_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_entries = 1024
    expected = np.arange(expected_entries, dtype="<i2")
    payload = expected.tobytes()

    def read_attr(device: object, name: bytes, buffer: object, capacity: int) -> int:
        assert name == b"viewer_data_raw"
        assert capacity == 2 * len(payload) + 1
        ctypes.memmove(buffer, payload, len(payload))
        return len(payload) + 1

    monkeypatch.setattr(iio, "_d_read_attr", read_attr)
    backend = object.__new__(IIODigitizerBackend)
    backend._scope = SimpleNamespace(
        attrs={"viewer_data_raw": object()},
        _device=object(),
    )
    monkeypatch.setattr(backend, "get_frame_samples", lambda: 4096)
    monkeypatch.setattr(backend, "get_mem_frame_size", lambda: 2048)

    frame = backend.read_frame()

    np.testing.assert_array_equal(frame, expected)


class _BlockingScopeBackend:
    def __init__(self) -> None:
        self.reading = threading.Event()
        self.release = threading.Event()
        self.prepared = 0
        self.stop_requests = 0
        self.closed = 0

    def prepare_dma_capture(self) -> None:
        self.prepared += 1

    def get_frame_samples(self) -> int:
        return 8

    def read_dma_frame(self) -> tuple[int, np.ndarray]:
        self.reading.set()
        assert self.release.wait(2)
        raise InterruptedError(errno.ECANCELED, "test stop")

    def request_dma_stop(self) -> None:
        self.stop_requests += 1
        self.release.set()

    def close_dma_capture(self) -> None:
        self.closed += 1


class _IncompleteScopeBackend(_BlockingScopeBackend):
    def read_dma_frame(self) -> tuple[int, np.ndarray]:
        return 123, np.zeros(3, dtype="<i2")


def test_streamer_stop_interrupts_blocked_refill(tmp_path: Path) -> None:
    backend = _BlockingScopeBackend()
    streamer = IIOScopeDmaStreamer(backend, channel=0)
    stop_event = threading.Event()
    result: list[int] = []
    output = tmp_path / "stopped.bin"

    worker = threading.Thread(
        target=lambda: result.append(streamer.stream_to_file(output, stop_event))
    )
    worker.start()
    assert backend.reading.wait(2)

    stop_event.set()
    streamer.request_stop()
    worker.join(2)

    assert not worker.is_alive()
    assert result == [0]
    assert backend.prepared == 1
    assert backend.stop_requests == 1
    assert backend.closed == 1
    assert output.stat().st_size == FILE_HEADER_STRUCT.size


def test_streamer_rejects_incomplete_waveform(tmp_path: Path) -> None:
    backend = _IncompleteScopeBackend()
    streamer = IIOScopeDmaStreamer(backend, channel=0)
    output = tmp_path / "incomplete.bin"

    with pytest.raises(RuntimeError, match="incomplete waveform"):
        streamer.stream_to_file(output, threading.Event(), n_frames=1)

    assert backend.closed == 1
    assert output.stat().st_size == FILE_HEADER_STRUCT.size


def test_backend_releases_faulted_buffer_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Buffer:
        pass

    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._dma_fault_latched = False
    backend._dma_stop_requested = threading.Event()
    backend._dma_buf_frame_samples = 8
    buffer = Buffer()
    buffer_ref = weakref.ref(buffer)
    backend._dma_buf = buffer
    del buffer

    monkeypatch.setattr(backend, "_dma_get_frame_samples", lambda: 8)

    def failed_refill(buf: object, first: bool) -> int:
        raise OSError(errno.EIO, "test DMA fault")

    monkeypatch.setattr(backend, "_refill_once", failed_refill)
    cleanup: list[bool] = []

    def close_buffer(*, drain: bool = True) -> None:
        backend._dma_buf = None
        gc.collect()
        assert buffer_ref() is None
        cleanup.append(drain)

    monkeypatch.setattr(backend, "_close_dma_buffer", close_buffer)

    with pytest.raises(OSError) as caught:
        backend._refill_dma_buffer()

    assert caught.value.errno == errno.EIO
    assert backend._dma_fault_latched
    assert cleanup == [False]


def test_backend_rejects_short_refill_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._dma_fault_latched = False
    backend._dma_stop_requested = threading.Event()
    backend._dma_buf_frame_samples = 8
    backend._dma_buf = object()

    monkeypatch.setattr(backend, "_dma_get_frame_samples", lambda: 8)
    monkeypatch.setattr(backend, "_refill_once", lambda buf, first: 8)
    cleanup: list[bool] = []

    def close_buffer(*, drain: bool = True) -> None:
        backend._dma_buf = None
        cleanup.append(drain)

    monkeypatch.setattr(backend, "_close_dma_buffer", close_buffer)

    with pytest.raises(OSError, match="returned 8 bytes, expected 16") as caught:
        backend._refill_dma_buffer()

    assert caught.value.errno == errno.EIO
    assert backend._dma_fault_latched
    assert cleanup == [False]


def _write_scope_file(path: Path, *, frames: int, trailing: bytes = b"") -> None:
    header = FILE_HEADER_STRUCT.pack(FILE_MAGIC, FILE_VERSION, 0, 0, 0.0, 8)
    frame = np.arange(8, dtype="<i2").tobytes()
    path.write_bytes(header + frame * frames + trailing)


def test_scope_converter_uses_int16_sample_count(tmp_path: Path) -> None:
    source = tmp_path / "scope.bin"
    destination = tmp_path / "scope.h5"
    _write_scope_file(source, frames=2)

    assert convert_scope(source, destination) == 2
    with h5py.File(destination, "r") as h5:
        assert h5.attrs["total_frames"] == 2
        assert h5.attrs["samples_per_frame"] == 4
        assert h5["waveforms"].shape == (2, 4)


def test_scope_converter_rejects_partial_frame(tmp_path: Path) -> None:
    source = tmp_path / "partial.bin"
    destination = tmp_path / "partial.h5"
    _write_scope_file(source, frames=2, trailing=b"12345678")

    with pytest.raises(ValueError, match="not aligned"):
        convert_scope(source, destination)
