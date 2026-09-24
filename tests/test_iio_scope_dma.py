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

import nlab.hardware.digitizer.backends.iio_backend as iio_backend_module
import nlab.hardware.digitizer.dma as dma_module
from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.current_monitor import (
    ScopeCurrentAccumulator,
    ScopeCurrentRuntime,
)
from nlab.hardware.digitizer.dma import (
    FILE_HEADER_STRUCT,
    FILE_MAGIC,
    FILE_VERSION,
    IIOScopeDmaStreamer,
    ScopeDmaGeometry,
    ScopeFrameBuffer,
)
from nlab.utils.dma_converter import convert_scope
from nlab.workers.dma_workers import IIOScopeDmaWorker, ScopeDmaWorker


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


def test_legacy_text_viewer_returns_available_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._scope = SimpleNamespace(attrs={})
    monkeypatch.setattr(backend, "get_frame_samples", lambda: 4096)
    monkeypatch.setattr(backend, "get_mem_frame_size", lambda: 2048)
    monkeypatch.setattr(backend, "_read_large_attr", lambda _name: b"10 20 30 \x00")

    frame = backend.read_frame()

    np.testing.assert_array_equal(frame, np.array([10, 20, 30], dtype=np.int16))


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

    def read_dma_raw_frame(self) -> bytes:
        self.reading.set()
        assert self.release.wait(2)
        raise InterruptedError(errno.ECANCELED, "test stop")

    def request_dma_stop(self) -> None:
        self.stop_requests += 1
        self.release.set()

    def close_dma_capture(self) -> None:
        self.closed += 1


class _IncompleteScopeBackend(_BlockingScopeBackend):
    def read_dma_raw_frame(self) -> bytes:
        return b"incomplete"


class _TwoFrameScopeBackend(_BlockingScopeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.read_calls = 0
        self.second_refill_started = threading.Event()

    def read_dma_raw_frame(self) -> bytes:
        self.read_calls += 1
        if self.read_calls == 2:
            self.second_refill_started.set()
        return struct.pack("<Q", self.read_calls) + np.arange(4, dtype="<i2").tobytes()


class _OneDynamicFrameScopeBackend(_BlockingScopeBackend):
    def get_scope_dma_geometry(self) -> ScopeDmaGeometry:
        return ScopeDmaGeometry(
            frame_samples=512,
            buffer_samples=132,
            frame_bytes=264,
            waveform_samples=127,
            sample_decimation=4,
            padding_bytes=2,
        )

    def read_dma_raw_frame(self) -> bytes:
        waveform = np.arange(127, dtype="<i2").tobytes()
        return struct.pack("<Q", 1234) + waveform + b"\x00\x00"


def test_iio_scope_worker_coalesces_frame_progress(tmp_path: Path) -> None:
    def stream_to_file(*, on_ready, on_progress, **_kwargs):
        on_ready()
        for value in range(1000):
            on_progress(value)
        return 1000

    worker = IIOScopeDmaWorker(
        SimpleNamespace(stream_to_file=stream_to_file),
        tmp_path / "simulated.bin",
    )
    updates: list[int] = []
    worker.progress.connect(updates.append)

    worker.run()

    assert updates == [0, 999]


@pytest.mark.parametrize("worker_kind", ["legacy", "iio"])
def test_scope_worker_preserves_progress_beyond_eight_gibibytes(
    tmp_path: Path,
    worker_kind: str,
) -> None:
    expected = 8 * 1024**3 + 123_456

    def stream_to_file(*, on_ready, on_progress, **_kwargs):
        on_ready()
        on_progress(expected)
        return expected

    streamer = SimpleNamespace(
        stream_to_file=stream_to_file,
        request_stop=lambda: None,
    )
    if worker_kind == "legacy":
        worker = ScopeDmaWorker(streamer, tmp_path / "legacy.bin", frame_samples=8)
    else:
        worker = IIOScopeDmaWorker(streamer, tmp_path / "iio.bin")
    updates: list[int] = []
    worker.progress.connect(updates.append)

    worker.run()

    assert updates == [expected]


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


def test_streamer_publishes_dynamic_frames_without_creating_a_file() -> None:
    backend = _OneDynamicFrameScopeBackend()
    streamer = IIOScopeDmaStreamer(backend, channel=0)
    frame_buffer = ScopeFrameBuffer()

    captured = streamer.stream_to_file(
        None,
        threading.Event(),
        n_frames=1,
        frame_buffer=frame_buffer,
    )

    frames, dropped = frame_buffer.drain()
    assert captured == 1
    assert dropped == 0
    assert len(frames) == 1
    assert frames[0].timestamp == 1234
    assert frames[0].geometry.capture_duration_ns == 1016
    np.testing.assert_array_equal(frames[0].samples, np.arange(127, dtype="<i2"))
    assert backend.closed == 1


def test_streamer_accumulates_current_before_replaceable_display() -> None:
    backend = _OneDynamicFrameScopeBackend()
    streamer = IIOScopeDmaStreamer(backend, channel=0)
    geometry = backend.get_scope_dma_geometry()
    accumulator = ScopeCurrentAccumulator()
    accumulator.start_session(
        geometry,
        ScopeCurrentRuntime(0, 122, 100, 228),
    )

    captured = streamer.stream_to_file(
        None,
        threading.Event(),
        n_frames=1,
        current_accumulator=accumulator,
    )
    snapshot = accumulator.snapshot()

    assert captured == 1
    assert snapshot.received_frames == snapshot.analyzed_frames == 1
    assert snapshot.latest_frame_mean == pytest.approx(63.0)
    assert snapshot.received_bytes == geometry.frame_bytes
    assert not snapshot.active
    assert backend.closed == 1


def test_streamer_counts_incomplete_current_frame_as_rejected_protocol_data() -> None:
    backend = _IncompleteScopeBackend()
    streamer = IIOScopeDmaStreamer(backend, channel=0)
    geometry = ScopeDmaGeometry.legacy(8)
    accumulator = ScopeCurrentAccumulator()
    accumulator.start_session(
        geometry,
        ScopeCurrentRuntime(0, 121, 8, 10),
    )

    with pytest.raises(RuntimeError, match="incomplete raw frame"):
        streamer.stream_to_file(
            None,
            threading.Event(),
            current_accumulator=accumulator,
        )

    snapshot = accumulator.snapshot()
    assert snapshot.rejected_frames == 1
    assert snapshot.protocol_errors == 1
    assert snapshot.received_frames == snapshot.analyzed_frames == 0
    assert not snapshot.active


def test_scope_frame_subscribers_drain_independently() -> None:
    geometry = ScopeDmaGeometry.legacy(8)
    publisher = ScopeFrameBuffer()
    subscriber = ScopeFrameBuffer()
    publisher.subscribe(subscriber)
    frame = dma_module.ScopeDmaFrame(
        timestamp=1,
        received_ns=2,
        samples=np.arange(4, dtype="<i2"),
        geometry=geometry,
    )

    publisher.append(frame)
    published, _ = publisher.drain()
    subscribed, _ = subscriber.drain()

    assert len(published) == len(subscribed) == 1
    assert published[0] is subscribed[0]
    assert not published[0].samples.flags.writeable
    with pytest.raises(ValueError):
        published[0].samples[0] = 99


def test_streamer_rejects_incomplete_waveform(tmp_path: Path) -> None:
    backend = _IncompleteScopeBackend()
    streamer = IIOScopeDmaStreamer(backend, channel=0)
    output = tmp_path / "incomplete.bin"

    with pytest.raises(RuntimeError, match="incomplete raw frame"):
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


def test_backend_uses_queue_capability_before_scope_buffer_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._dma_fault_latched = False
    backend._dma_stop_requested = threading.Event()
    backend._dma_buf = None
    order: list[str] = []

    backend._dma_scope = SimpleNamespace(
        attrs={
            "dma_queue_mode": SimpleNamespace(value="dmaengine"),
            "dma_kernel_buffers_max": SimpleNamespace(value="4"),
            "dma_kernel_buffers_recommended": SimpleNamespace(value="4"),
        },
        id="iio:device15",
        set_kernel_buffers_count=lambda count: order.append(f"kernel:{count}"),
    )
    backend._uri = "local:"
    monkeypatch.setattr(backend, "_close_dma_buffer", lambda: order.append("close"))
    monkeypatch.setattr(backend, "_dma_get_enable", lambda: False)
    monkeypatch.setattr(backend, "_dma_get_dma_enable", lambda: False)
    monkeypatch.setattr(backend, "_dma_get_frame_samples", lambda: 8)
    capability_values = {
        "dma_queue_mode": "dmaengine",
        "dma_kernel_buffers_max": "4",
        "dma_kernel_buffers_recommended": "4",
    }
    monkeypatch.setattr(
        backend,
        "_dma_attr_get",
        lambda name: capability_values.get(name, "0"),
    )

    def make_buffer(device: object, length: int, cyclic: bool) -> object:
        assert device is backend._dma_scope
        assert length == 8
        assert not cyclic
        order.append("buffer")
        return object()

    monkeypatch.setattr(iio, "Buffer", make_buffer)

    backend._create_dma_buffer(8)

    assert order == ["close", "kernel:4", "buffer"]
    assert backend._dma_buf_frame_samples == 8


def test_backend_uses_batched_iiod_for_qualified_remote_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._uri = "ip:192.168.10.128:30431"
    backend._dma_fault_latched = False
    backend._dma_stop_requested = threading.Event()
    backend._dma_buf = None
    selected: list[int] = []
    backend._dma_scope = SimpleNamespace(
        id="iio:device15",
        attrs={
            "dma_queue_mode": object(),
            "dma_kernel_buffers_max": object(),
            "dma_kernel_buffers_recommended": object(),
            "queued_blocks": object(),
            "queue_high_watermark": object(),
        },
        set_kernel_buffers_count=selected.append,
    )
    values = {
        "dma_queue_mode": "dmaengine",
        "dma_kernel_buffers_max": "4",
        "dma_kernel_buffers_recommended": "4",
        "dma_buffer_active": "0",
        "ip_version": "122",
        "queued_blocks": "3",
        "queue_high_watermark": "4",
    }
    monkeypatch.setattr(backend, "_close_dma_buffer", lambda: None)
    monkeypatch.setattr(backend, "_dma_get_enable", lambda: False)
    monkeypatch.setattr(backend, "_dma_get_dma_enable", lambda: False)
    monkeypatch.setattr(backend, "_dma_get_frame_samples", lambda: 1024)
    monkeypatch.setattr(backend, "_dma_attr_get", values.__getitem__)
    created: list[tuple[str, str, int, int, int]] = []
    fake_buffer = object()

    def create_stream(
        uri: str,
        device: str,
        samples: int,
        buffers: int,
        batch_frames: int,
    ) -> object:
        created.append((uri, device, samples, buffers, batch_frames))
        return fake_buffer

    monkeypatch.setattr(iio_backend_module, "IiodScopeStream", create_stream)

    backend._create_dma_buffer(1024)

    assert selected == [4]
    assert created == [
        ("ip:192.168.10.128:30431", "iio:device15", 1024, 4, 32),
    ]
    assert backend._dma_buf is fake_buffer
    assert backend._dma_capture_transport == "iiod-batched"
    assert backend._dma_capture_kernel_buffers == 4
    assert backend.get_scope_dma_runtime_metadata() == {
        "uri": "ip:192.168.10.128:30431",
        "device": "iio:device15",
        "channel": 0,
        "ip_version": 122,
        "transport": "iiod-batched",
        "kernel_buffers": 4,
        "queued_blocks": 3,
        "queue_high_watermark": 4,
        "readbuf_batch_frames": 32,
    }


def test_backend_allocates_v122_advertised_periodic_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._uri = "local:"
    backend._dma_fault_latched = False
    backend._dma_stop_requested = threading.Event()
    backend._dma_buf = None
    backend._dma_scope = SimpleNamespace(
        id="iio:device15",
        attrs={
            "dma_frame_bytes": object(),
            "dma_waveform_samples": object(),
            "dma_sample_decimation": object(),
        },
        set_kernel_buffers_count=lambda _count: None,
    )
    values = {
        "dma_frame_bytes": "264",
        "dma_waveform_samples": "127",
        "dma_sample_decimation": "4",
        "dma_buffer_active": "0",
    }
    monkeypatch.setattr(backend, "_close_dma_buffer", lambda: None)
    monkeypatch.setattr(backend, "_dma_get_enable", lambda: False)
    monkeypatch.setattr(backend, "_dma_get_dma_enable", lambda: False)
    monkeypatch.setattr(backend, "_dma_get_frame_samples", lambda: 512)
    monkeypatch.setattr(backend, "_dma_attr_get", values.__getitem__)
    lengths: list[int] = []

    def make_buffer(_device: object, length: int, _cyclic: bool) -> object:
        lengths.append(length)
        return object()

    monkeypatch.setattr(iio, "Buffer", make_buffer)

    backend._create_dma_buffer(512)

    assert lengths == [132]
    assert backend._dma_buf_frame_samples == 132
    assert backend._dma_capture_geometry == ScopeDmaGeometry(
        frame_samples=512,
        buffer_samples=132,
        frame_bytes=264,
        waveform_samples=127,
        sample_decimation=4,
        padding_bytes=2,
    )


def test_backend_keeps_one_block_without_queue_capability() -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._dma_scope = SimpleNamespace(attrs={})

    assert backend._scope_dma_kernel_buffers() == 1


def test_backend_rejects_invalid_queue_capability() -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._dma_scope = SimpleNamespace(
        attrs={
            "dma_queue_mode": object(),
            "dma_kernel_buffers_max": object(),
            "dma_kernel_buffers_recommended": object(),
        }
    )
    values = {
        "dma_queue_mode": "dmaengine",
        "dma_kernel_buffers_max": "8",
        "dma_kernel_buffers_recommended": "8",
    }
    backend._dma_attr_get = values.__getitem__

    with pytest.raises(RuntimeError, match="invalid Scope DMA queue capability"):
        backend._scope_dma_kernel_buffers()


def test_backend_rejects_uninitialized_scope_geometry_before_arm() -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._dma_fault_latched = False
    backend._dma_stop_requested = threading.Event()

    with pytest.raises(ValueError, match="invalid Scope DMA frame geometry 0"):
        backend._create_dma_buffer(0)


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


def test_backend_latches_batched_protocol_error_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._dma_fault_latched = False
    backend._dma_stop_requested = threading.Event()
    backend._dma_buf_frame_samples = 8
    backend._dma_buf = object.__new__(iio_backend_module.IiodScopeStream)
    monkeypatch.setattr(backend, "_dma_get_frame_samples", lambda: 8)
    monkeypatch.setattr(
        backend,
        "_refill_once",
        lambda _buf, _first: (_ for _ in ()).throw(
            OSError(errno.EPROTO, "bad batch boundary")
        ),
    )
    cleanup: list[bool] = []

    def close_buffer(*, drain: bool = True) -> None:
        backend._dma_buf = None
        cleanup.append(drain)

    monkeypatch.setattr(backend, "_close_dma_buffer", close_buffer)

    with pytest.raises(OSError) as caught:
        backend._refill_dma_buffer()

    assert caught.value.errno == errno.EPROTO
    assert backend._dma_fault_latched
    assert cleanup == [False]


def test_backend_uses_armed_geometry_without_per_frame_attribute_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._dma_fault_latched = False
    backend._dma_stop_requested = threading.Event()
    backend._dma_buf_frame_samples = 8
    backend._dma_last_timestamp = None
    raw = struct.pack("<Q", 123) + np.arange(4, dtype="<i2").tobytes()
    buffer = object.__new__(iio_backend_module.IiodScopeStream)
    buffer.read = lambda: raw
    backend._dma_buf = buffer
    monkeypatch.setattr(
        backend,
        "_dma_get_frame_samples",
        lambda: pytest.fail("frame_samples was read again while the buffer was armed"),
    )
    monkeypatch.setattr(backend, "_refill_once", lambda _buf, _first: len(raw))

    assert backend._refill_dma_buffer() == raw


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
