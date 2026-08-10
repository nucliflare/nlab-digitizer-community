from __future__ import annotations

import ctypes
import io
import json
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import iio
import numpy as np
import pytest

from nlab.hardware.digitizer.backends.iio_backend import (
    _LM_EVENT_DTYPE as _BACKEND_LM_EVENT_DTYPE,
)
from nlab.hardware.digitizer.backends.iio_backend import (
    _LM_KERNEL_BUFFER_COUNT,
    IIODigitizerBackend,
)
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
        self.stop_requests = 0
        self.diagnostics = (0, 0, 2, 0)

    def start_mca_dma_capture(
        self,
        on_started: Callable[[], None] | None = None,
    ) -> np.ndarray:
        if on_started is not None:
            on_started()
        self._stop_event.set()
        return self.first

    def read_mca_dma_frame(self) -> np.ndarray:
        raise AssertionError("streamer refilled after stop was requested")

    def mca_dma_measurement_in_progress(self) -> bool:
        return False

    def request_mca_dma_stop(self) -> None:
        self.stop_requests += 1

    def close_mca_dma_capture(
        self,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> int:
        if on_frame is not None:
            on_frame(self.tail)
        return 1

    def get_mca_dma_capture_diagnostics(self) -> tuple[int, int, int, int]:
        return self.diagnostics


class _FakeAutoStopBackend(_FakeBackend):
    def start_mca_dma_capture(
        self,
        on_started: Callable[[], None] | None = None,
    ) -> np.ndarray:
        if on_started is not None:
            on_started()
        return self.first


class _FakeEmptyBackend(_FakeBackend):
    def start_mca_dma_capture(
        self,
        on_started: Callable[[], None] | None = None,
    ) -> np.ndarray:
        if on_started is not None:
            on_started()
        self._stop_event.set()
        return np.empty(0, dtype=_LM_EVENT_DTYPE)

    def close_mca_dma_capture(
        self,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> int:
        return 0


class _FakeCloseErrorBackend(_FakeBackend):
    def close_mca_dma_capture(
        self,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> int:
        super().close_mca_dma_capture(on_frame)
        raise RuntimeError("synthetic close failure")


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


def test_binary_attribute_read_reserves_libiio_terminator(monkeypatch: Any) -> None:
    payload = bytes(range(256)) * 32

    def read_attr(device: Any, name: bytes, buf: Any, capacity: int) -> int:
        assert name == b"debug_data"
        assert capacity == 2 * len(payload) + 1
        ctypes.memmove(buf, payload, len(payload))
        return len(payload) + 1

    monkeypatch.setattr(iio, "_d_read_attr", read_attr)
    backend = object.__new__(IIODigitizerBackend)
    device = SimpleNamespace(_device=object())

    assert backend._read_large_pp_attr(device, "debug_data", len(payload)) == payload


def test_binary_attribute_read_accepts_older_unterminated_payload(monkeypatch: Any) -> None:
    payload = bytes(range(256)) * 32

    def read_attr(device: Any, name: bytes, buf: Any, capacity: int) -> int:
        ctypes.memmove(buf, payload, len(payload))
        return len(payload)

    monkeypatch.setattr(iio, "_d_read_attr", read_attr)
    backend = object.__new__(IIODigitizerBackend)
    device = SimpleNamespace(_device=object())

    assert backend._read_large_pp_attr(device, "debug_data", len(payload)) == payload


def test_histogram_reassembles_four_transport_chunks(monkeypatch: Any) -> None:
    names = tuple(f"histogram_data{index}" for index in range(4))
    chunks = {
        name: np.full(4096, index, dtype="<u4").tobytes()
        for index, name in enumerate(names)
    }
    backend = object.__new__(IIODigitizerBackend)
    backend._mca_pp = SimpleNamespace(attrs=dict.fromkeys(names), _device=object())

    def read_large(device: Any, name: str, size: int) -> bytes:
        assert size == 16384
        return chunks[name]

    monkeypatch.setattr(backend, "_read_large_pp_attr", read_large)

    histogram = backend.read_histogram()

    assert histogram.shape == (16384,)
    for index in range(4):
        np.testing.assert_array_equal(histogram[index * 4096 : (index + 1) * 4096], index)


def test_mca_buffer_arm_stops_then_selects_eight_kernel_blocks(monkeypatch: Any) -> None:
    def attr(value: int) -> SimpleNamespace:
        return SimpleNamespace(value=str(value))

    order: list[str] = []
    lm_frame = SimpleNamespace(
        attrs={
            "ip_version": attr(121),
            "frame_records": attr(1024),
            "frame_bytes": attr(16384),
            "channel_index": attr(0),
            "record_layout": SimpleNamespace(value="opaque[16]"),
            "buffer_active": attr(0),
            "dma_fault": attr(0),
            "dma_error_count": attr(0),
            "queued_blocks": attr(0),
            "kernel_buffer_blocks": attr(0),
        },
        channels=[SimpleNamespace(enabled=False)],
        sample_size=16,
    )

    def set_kernel_buffers_count(count: int) -> None:
        assert count == _LM_KERNEL_BUFFER_COUNT
        assert pulse.attrs["enable"].value == "0"
        order.append("kernel-buffers")

    lm_frame.set_kernel_buffers_count = set_kernel_buffers_count
    pulse = SimpleNamespace(
        attrs={
            "channel_index": attr(0),
            "enable": attr(1),
            "list_buffer_active": attr(0),
        }
    )
    fake_buffer = SimpleNamespace(_buffer=object())

    def create_buffer(device: Any, records: int, cyclic: bool) -> SimpleNamespace:
        assert order == ["kernel-buffers"]
        assert device is lm_frame
        assert records == 1024
        assert cyclic is False
        assert lm_frame.channels[0].enabled
        lm_frame.attrs["buffer_active"].value = "1"
        lm_frame.attrs["kernel_buffer_blocks"].value = str(_LM_KERNEL_BUFFER_COUNT)
        order.append("buffer")
        return fake_buffer

    monkeypatch.setattr(iio, "Buffer", create_buffer)
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._mca_dma_buf = None
    backend._lm_frame = lm_frame
    backend._mca_dma_pp = pulse
    backend._mca_dma_control_lock = threading.RLock()
    backend._mca_dma_stop_requested = threading.Event()
    backend._mca_dma_lifecycle_lock = threading.RLock()
    backend._mca_dma_cancel_timer = None

    backend._create_mca_dma_buffer()

    assert order == ["kernel-buffers", "buffer"]
    assert backend._mca_dma_buf is fake_buffer


def test_mca_rejects_superseded_five_channel_scan_abi() -> None:
    def attr(value: int | str) -> SimpleNamespace:
        return SimpleNamespace(value=str(value))

    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._lm_frame = SimpleNamespace(
        attrs={
            "ip_version": attr(121),
            "frame_records": attr(1024),
            "frame_bytes": attr(16384),
            "channel_index": attr(0),
            "record_layout": attr(
                "le16 flags; le16 cfd_time_q2; le16 charge_energy; "
                "le16 energy; le64 timestamp"
            ),
        },
        channels=[SimpleNamespace(enabled=False) for _ in range(5)],
        sample_size=16,
    )
    backend._mca_dma_pp = SimpleNamespace(attrs={"channel_index": attr(0)})
    backend._mca_dma_control_lock = threading.RLock()

    with pytest.raises(RuntimeError, match="superseded Linux 5.15 ABI"):
        backend._validate_lm_geometry()


def test_mca_ready_fires_after_reader_entry_and_enable(monkeypatch: Any) -> None:
    refill_entered = threading.Event()
    release_refill = threading.Event()
    pulse = SimpleNamespace(attrs={"enable": SimpleNamespace(value="0")})
    buffer = SimpleNamespace(_buffer=object(), cancel=lambda: None)
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._mca_dma_pp = pulse
    backend._mca_dma_control_lock = threading.RLock()

    def refill(native_buffer: Any) -> int:
        assert native_buffer is buffer._buffer
        refill_entered.set()
        assert release_refill.wait(1.0)
        return 16384

    monkeypatch.setattr(iio, "_buffer_refill", refill)
    ready: list[None] = []

    def on_started() -> None:
        assert refill_entered.wait(1.0)
        assert pulse.attrs["enable"].value == "1"
        ready.append(None)
        release_refill.set()

    assert backend._start_mca_reader_then_enable(buffer, on_started) == 16384
    assert ready == [None]


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
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["format"] == "nlab-iio-mca-ndma-v2"
    assert metadata["frames"] == 2
    assert metadata["records"] == 2048
    assert metadata["driver_completed_frames"] == 2
    assert metadata["driver_dma_error_count"] == 0
    assert metadata["continuity_valid"] is True


def test_converter_understands_iio_listmode_version(tmp_path: Path) -> None:
    stop_event = threading.Event()
    backend = _FakeBackend(stop_event)
    source = tmp_path / "listmode.bin"
    destination = tmp_path / "listmode.h5"
    IIOMcaDmaStreamer(backend, channel=0).stream_events(stop_event, source)

    assert convert_listmode(source, destination) == 2048
    with h5py.File(destination, "r") as h5:
        assert h5.attrs["format_version"] == IIO_LM_FILE_VERSION
        assert h5.attrs["total_frames"] == 2
        assert h5.attrs["frame_records"] == 1024
        assert h5.attrs["frame_bytes"] == 16384
        assert h5.attrs["driver_completed_frames"] == 2
        assert bool(h5.attrs["continuity_valid"])
        assert h5.attrs["source_sha256"]
        assert h5["events"].dtype == _LM_EVENT_DTYPE
        np.testing.assert_array_equal(
            h5["trapezoid_energy"][:1024], backend.first["trapezoid_energy"]
        )
        np.testing.assert_allclose(h5["cfd_time"][:1024], backend.first["cfd_q2"] / 4.0)


def test_converter_rejects_partial_iio_listmode_frame(tmp_path: Path) -> None:
    stop_event = threading.Event()
    backend = _FakeBackend(stop_event)
    source = tmp_path / "partial.bin"
    destination = tmp_path / "partial.h5"
    IIOMcaDmaStreamer(backend, channel=0).stream_events(stop_event, source)
    source.write_bytes(source.read_bytes()[:-_LM_EVENT_DTYPE.itemsize])

    with pytest.raises(ValueError, match="complete DMA frames"):
        convert_listmode(source, destination)

    assert not destination.exists()


def test_converter_rejects_capture_changed_after_metadata(tmp_path: Path) -> None:
    stop_event = threading.Event()
    backend = _FakeBackend(stop_event)
    source = tmp_path / "changed.bin"
    destination = tmp_path / "changed.h5"
    IIOMcaDmaStreamer(backend, channel=0).stream_events(stop_event, source)
    capture = bytearray(source.read_bytes())
    capture[-1] ^= 0x01
    source.write_bytes(capture)

    with pytest.raises(ValueError, match="SHA-256"):
        convert_listmode(source, destination)

    assert not destination.exists()


def test_streamer_does_not_refill_after_hardware_time_limit() -> None:
    stop_event = threading.Event()
    backend = _FakeAutoStopBackend(stop_event)

    total = IIOMcaDmaStreamer(backend, channel=0).stream_events(stop_event)

    assert total == 2048
    assert not stop_event.is_set()


def test_streamer_accepts_clean_zero_frame_capture() -> None:
    stop_event = threading.Event()
    backend = _FakeEmptyBackend(stop_event)
    backend.diagnostics = (0, 0, 0, 0)

    total = IIOMcaDmaStreamer(backend, channel=0).stream_events(stop_event)

    assert total == 0


def test_streamer_stop_request_stops_only_the_producer() -> None:
    backend = _FakeBackend(threading.Event())

    IIOMcaDmaStreamer(backend, channel=0).request_stop()

    assert backend.stop_requests == 1


def test_streamer_closes_file_when_backend_close_fails(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    stop_event = threading.Event()
    backend = _FakeCloseErrorBackend(stop_event)
    tracked_file = io.BytesIO()
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: tracked_file)

    with pytest.raises(RuntimeError, match="synthetic close failure"):
        IIOMcaDmaStreamer(backend, channel=0).stream_events(
            stop_event, tmp_path / "close-error.bin",
        )

    assert tracked_file.closed


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        ((2, 0, 2, 0), "DMA fault reason 2"),
        ((0, 0, 1, 0), "driver completed 1"),
        ((0, 0, 2, 7), "7 dropped records"),
    ],
)
def test_streamer_rejects_invalid_post_capture_diagnostics(
    diagnostics: tuple[int, int, int, int],
    message: str,
) -> None:
    stop_event = threading.Event()
    backend = _FakeBackend(stop_event)
    backend.diagnostics = diagnostics

    with pytest.raises(RuntimeError, match=message):
        IIOMcaDmaStreamer(backend, channel=0).stream_events(stop_event)


def test_streamer_retains_invalid_continuity_metadata(tmp_path: Path) -> None:
    stop_event = threading.Event()
    backend = _FakeBackend(stop_event)
    backend.diagnostics = (0, 4, 2, 7)
    output = tmp_path / "deadtime.bin"

    with pytest.raises(RuntimeError, match="7 dropped records"):
        IIOMcaDmaStreamer(backend, channel=0).stream_events(stop_event, output)

    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["driver_dma_error_count"] == 4
    assert metadata["list_deadtime_raw"] == 7
    assert metadata["continuity_valid"] is False
