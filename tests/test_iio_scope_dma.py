from __future__ import annotations

import ctypes
import errno
import gc
import struct
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import h5py
import iio
import numpy as np
import pytest

import nlab.hardware.digitizer.dma as dma_module
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


class _TwoFrameScopeBackend(_BlockingScopeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.read_calls = 0
        self.second_refill_started = threading.Event()

    def read_dma_frame(self) -> tuple[int, np.ndarray]:
        self.read_calls += 1
        if self.read_calls == 2:
            self.second_refill_started.set()
        return self.read_calls, np.arange(4, dtype="<i2")


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


def test_streamer_refills_while_separate_writer_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _TwoFrameScopeBackend()
    streamer = IIOScopeDmaStreamer(backend, channel=0)
    output = tmp_path / "pipelined.bin"
    first_frame_write_started = threading.Event()
    release_writer = threading.Event()
    flush_calls: list[None] = []
    real_open = open

    class DelayedFile:
        def __init__(self, path: Path, mode: str) -> None:
            self._file = real_open(path, mode)

        def __enter__(self) -> DelayedFile:
            return self

        def __exit__(self, *args: object) -> None:
            self._file.close()

        def write(self, data: bytes) -> int:
            # The header is 24 bytes; each test frame is 8 timestamp bytes
            # plus four int16 waveform samples, for a 16-byte record.
            if len(data) == 16 and not first_frame_write_started.is_set():
                first_frame_write_started.set()
                assert release_writer.wait(2)
            return self._file.write(data)

        def flush(self) -> None:
            flush_calls.append(None)
            self._file.flush()

    def delayed_open(path: Path, mode: str) -> DelayedFile:
        assert path == output
        assert mode == "wb"
        return DelayedFile(path, mode)

    monkeypatch.setattr(dma_module, "open", delayed_open, raising=False)
    result: list[int] = []
    errors: list[BaseException] = []

    def capture() -> None:
        try:
            result.append(
                streamer.stream_to_file(
                    output,
                    threading.Event(),
                    n_frames=2,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=capture)
    worker.start()
    assert first_frame_write_started.wait(2)

    # The file writer still owns and is blocked on frame 1, while the
    # producer has already issued the next backend refill.
    assert backend.second_refill_started.wait(2)
    release_writer.set()
    worker.join(2)

    assert not worker.is_alive()
    assert errors == []
    assert result == [2]
    assert backend.read_calls == 2
    assert backend.closed == 1
    assert output.stat().st_size == FILE_HEADER_STRUCT.size + 2 * 16
    records = output.read_bytes()[FILE_HEADER_STRUCT.size:]
    assert struct.unpack_from("<Q", records, 0) == (1,)
    assert struct.unpack_from("<Q", records, 16) == (2,)
    np.testing.assert_array_equal(
        np.frombuffer(records[8:16], dtype="<i2"),
        np.arange(4, dtype="<i2"),
    )
    # Only _write_file_header() flushes. Frame writes rely on buffered I/O
    # and the final close, so no capture-time disk flush stalls the refill.
    assert len(flush_calls) == 1


def test_backend_selects_one_kernel_buffer_before_scope_buffer_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._dma_fault_latched = False
    backend._dma_stop_requested = threading.Event()
    backend._dma_buf = None
    order: list[str] = []

    backend._dma_scope = SimpleNamespace(
        set_kernel_buffers_count=lambda count: order.append(f"kernel:{count}"),
    )
    monkeypatch.setattr(backend, "_close_dma_buffer", lambda: order.append("close"))
    monkeypatch.setattr(backend, "_dma_get_enable", lambda: False)
    monkeypatch.setattr(backend, "_dma_get_dma_enable", lambda: False)
    monkeypatch.setattr(backend, "_dma_attr_get", lambda name: "0")

    def make_buffer(device: object, length: int, cyclic: bool) -> object:
        assert device is backend._dma_scope
        assert length == 8
        assert not cyclic
        order.append("buffer")
        return object()

    monkeypatch.setattr(iio, "Buffer", make_buffer)

    backend._create_dma_buffer(8)

    assert order == ["close", "kernel:1", "buffer"]
    assert backend._dma_buf_frame_samples == 8


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
