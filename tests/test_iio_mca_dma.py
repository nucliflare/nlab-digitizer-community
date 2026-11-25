from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import h5py
import iio
import numpy as np
import pytest

from nlab.hardware.digitizer.backends.iio_backend import (
    _LIBIIO_TIMEOUT_ERRNOS,
    _LM_KERNEL_BUFFER_COUNT,
    _REMOTE_ETIMEDOUT,
    IIODigitizerBackend,
    _disable_mca_dma_client_timeout,
)
from nlab.hardware.digitizer.backends.iio_backend import (
    _LM_EVENT_DTYPE as _BACKEND_LM_EVENT_DTYPE,
)
from nlab.hardware.digitizer.dma import (
    _LM_EVENT_DTYPE,
    FILE_HEADER_STRUCT,
    IIO_LM_FILE_VERSION,
    IIO_LM_UNQUALIFIED_SCHEMA,
    IIOMcaDmaStreamer,
    McaEventBuffer,
)
from nlab.hardware.digitizer.iio_listmode import cfd_interpolation_samples
from nlab.hardware.digitizer.mca_parameters import DEFAULT_PARAMETERS
from nlab.utils.dma_converter import convert_listmode, read_file_header
from nlab.workers.dma_workers import IIOMcaDmaWorker


def test_iio_event_bytes_match_zero_crossing_hls_output_layout() -> None:
    raw = bytes.fromhex("82 fd 00 c0 34 12 78 56 08 07 06 05 04 03 02 01")
    events = np.frombuffer(raw, dtype=_LM_EVENT_DTYPE)

    assert _LM_EVENT_DTYPE.itemsize == 16
    assert {name: field[1] for name, field in _LM_EVENT_DTYPE.fields.items()} == {
        "marker": 0,
        "zc_offset": 1,
        "zc_estimation": 2,
        "charge_energy": 4,
        "trapezoid_energy": 6,
        "timestamp": 8,
    }
    assert events["marker"][0] == 0x82
    assert events["zc_offset"][0] == 0xFD
    assert events["zc_estimation"][0] == -16384
    assert events["charge_energy"][0] == 0x1234
    assert events["trapezoid_energy"][0] == 0x5678
    assert events["timestamp"][0] == 0x0102030405060708
    np.testing.assert_array_equal(cfd_interpolation_samples(events), [-1.0])


def _frame(seed: int) -> np.ndarray:
    events = np.zeros(1024, dtype=_LM_EVENT_DTYPE)
    events["marker"] = 0x82
    events["zc_offset"] = seed
    events["zc_estimation"] = -4096 * seed
    events["charge_energy"] = 100 + seed
    events["trapezoid_energy"] = 200 + seed
    events["timestamp"] = np.arange(1024, dtype=np.uint64) + 1000 * seed
    return events


def _session_first_frame(seed: int) -> np.ndarray:
    # A clean session's first record is the synthetic START record
    # (zero marker/payload, timestamp 1), as delivered by the hardware.
    events = _frame(seed)
    events[0] = np.zeros(1, dtype=_LM_EVENT_DTYPE)[0]
    events["timestamp"][0] = 1
    return events


class _FakeBackend:
    def __init__(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event
        self.first = _session_first_frame(1)
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


def test_iio_event_dtype_matches_supplied_hls_output_layout() -> None:
    assert _LM_EVENT_DTYPE == _BACKEND_LM_EVENT_DTYPE
    assert _LM_EVENT_DTYPE.itemsize == 16
    fields = _LM_EVENT_DTYPE.fields
    assert fields is not None
    assert {name: info[1] for name, info in fields.items()} == {
        "marker": 0,
        "zc_offset": 1,
        "zc_estimation": 2,
        "charge_energy": 4,
        "trapezoid_energy": 6,
        "timestamp": 8,
    }


def test_mca_dma_client_stream_context_disables_socket_timeout() -> None:
    context = SimpleNamespace(set_timeout=Mock())

    _disable_mca_dma_client_timeout(context)

    # pylibiio/libiio 0.x defines zero as no timeout. The separate control
    # context retains its default timeout and Buffer.cancel() stops refill.
    context.set_timeout.assert_called_once_with(0)


@pytest.mark.parametrize("timeout_errno", sorted(_LIBIIO_TIMEOUT_ERRNOS))
def test_mca_dma_refill_retries_timeout_while_waiting_for_trigger(
    monkeypatch: Any,
    timeout_errno: int,
) -> None:
    native_buffer = object()
    buffer = SimpleNamespace(_buffer=native_buffer)
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._mca_dma_stop_requested = threading.Event()
    refill = Mock(side_effect=[
        OSError(timeout_errno, "host unreachable"),
        16384,
    ])
    monkeypatch.setattr(iio, "_buffer_refill", refill)

    assert backend._refill_mca_dma_buffer(buffer) == 16384
    assert refill.call_args_list == [call(native_buffer), call(native_buffer)]


def test_mca_dma_refill_does_not_retry_timeout_after_stop(monkeypatch: Any) -> None:
    native_buffer = object()
    buffer = SimpleNamespace(_buffer=native_buffer)
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._mca_dma_stop_requested = threading.Event()
    backend._mca_dma_stop_requested.set()
    refill = Mock(side_effect=OSError(_REMOTE_ETIMEDOUT, "host unreachable"))
    monkeypatch.setattr(iio, "_buffer_refill", refill)

    with pytest.raises(OSError) as exc_info:
        backend._refill_mca_dma_buffer(buffer)

    assert exc_info.value.errno == _REMOTE_ETIMEDOUT
    refill.assert_called_once_with(native_buffer)


def test_mca_dma_cancel_explains_native_ebadf_once(caplog: Any) -> None:
    buffer = SimpleNamespace(cancel=Mock())
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._mca_dma_lifecycle_lock = threading.RLock()
    backend._mca_dma_cancel_explained = False

    with caplog.at_level(logging.INFO, logger="nlab.hardware.digitizer.backends.iio_backend"):
        backend._cancel_mca_dma_refill(buffer, "test stop")
        backend._cancel_mca_dma_refill(buffer, "duplicate cleanup")

    assert buffer.cancel.call_count == 2
    messages = [record.getMessage() for record in caplog.records]
    matching = [message for message in messages if "READ LINE: -9" in message]
    assert len(matching) == 1
    assert "expected EBADF" in matching[0]
    assert "test stop" in matching[0]


def test_dead_time_scale_is_read_once_and_raw_every_call() -> None:
    reads: list[str] = []
    values = {"dead_time_raw": "10", "dead_time_scale": "0.5"}
    backend = object.__new__(IIODigitizerBackend)
    backend._mca_pp_attr_get = lambda name: reads.append(name) or values[name]  # type: ignore[method-assign]

    assert backend.get_pulse_deadtime_ms() == 5.0
    values["dead_time_raw"] = "20"
    assert backend.get_pulse_deadtime_ms() == 10.0

    assert reads == ["dead_time_raw", "dead_time_scale", "dead_time_raw"]


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


def _armable_mca_backend(
    order: list[str], parameters: dict[str, int] | None = None,
) -> tuple[Any, SimpleNamespace]:
    """Backend whose list-mode arm reaches iio.Buffer(), and its lm_frame fake."""

    def attr(value: int) -> SimpleNamespace:
        return SimpleNamespace(value=str(value))

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
            **{
                name: attr(value)
                for name, value in (DEFAULT_PARAMETERS | (parameters or {})).items()
            },
            "parameter_abi_version": attr(2),
            "parameter_units": SimpleNamespace(value="adc_samples"),
            "parameter_abi_ack": attr(2),
            "channel_index": attr(0),
            "enable": attr(1),
            "list_buffer_active": attr(0),
        }
    )
    backend: Any = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._mca_dma_buf = None
    backend._lm_frame = lm_frame
    backend._mca_dma_pp = pulse
    backend._mca_dma_control_lock = threading.RLock()
    backend._mca_dma_stop_requested = threading.Event()
    backend._mca_dma_lifecycle_lock = threading.RLock()
    backend._mca_dma_cancel_timer = None
    return backend, lm_frame


def test_mca_buffer_arm_stops_then_selects_eight_kernel_blocks(monkeypatch: Any) -> None:
    order: list[str] = []
    backend, lm_frame = _armable_mca_backend(order)
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

    backend._create_mca_dma_buffer()

    assert order == ["kernel-buffers", "buffer"]
    assert backend._mca_dma_buf is fake_buffer


def test_mca_arm_names_rejected_pileup_before_buffer_creation(monkeypatch: Any) -> None:
    # Live 2026-11-25 readback, ch0: 1424 ns window, one pile-up, CC enabled.
    # lm_buffer_preenable() rejects it (vdpp_pulse_validate_config) as EINVAL.
    order: list[str] = []
    backend, _ = _armable_mca_backend(order, {
        "frame_samples": 712, "pileup_window_raw": 1,
        "charge_comparison_enable": 1, "charge_comparison_samples": 196,
    })
    create_buffer = Mock()
    monkeypatch.setattr(iio, "Buffer", create_buffer)

    with pytest.raises(ValueError, match=r"Pile-ups 1 with a 1424 ns window .* = 354 > 255"):
        backend._create_mca_dma_buffer()

    create_buffer.assert_not_called()
    assert backend._mca_dma_buf is None


def test_mca_arm_failure_reports_driver_errno(monkeypatch: Any) -> None:
    order: list[str] = []
    backend, _ = _armable_mca_backend(order)
    monkeypatch.setattr(
        iio, "Buffer", Mock(side_effect=OSError(errno.EINVAL, "Invalid argument")),
    )

    with pytest.raises(RuntimeError, match=r"errno 22 \(Invalid argument\) \(buffer_active=0"):
        backend._create_mca_dma_buffer()


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
    backend._mca_dma_pp = SimpleNamespace(attrs={"channel_index": attr(0),
        "parameter_abi_version": attr(2), "parameter_abi_ack": attr(2),
        "parameter_units": attr("adc_samples")})
    backend._mca_dma_control_lock = threading.RLock()

    with pytest.raises(RuntimeError, match="superseded Linux 5.15 ABI"):
        backend._validate_lm_geometry()


def test_mca_ready_fires_after_reader_entry_and_enable(monkeypatch: Any) -> None:
    refill_entered = threading.Event()
    release_refill = threading.Event()
    pulse = SimpleNamespace(attrs={"enable": SimpleNamespace(value="0"),
        "parameter_abi_version": SimpleNamespace(value="2"),
        "parameter_units": SimpleNamespace(value="adc_samples"),
        "parameter_abi_ack": SimpleNamespace(value="2")})
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
    assert metadata["client_record_schema"] == IIO_LM_UNQUALIFIED_SCHEMA
    assert streamer.last_capture_diagnostics == {
        "dma_fault": 0,
        "dma_error_count": 0,
        "completed_frames": 2,
        "streamed_frames": 2,
        "list_deadtime_raw": 0,
        "list_deadtime_before_raw": None,
        "start_record": "first",
        "start_record_index": 0,
        "stale_frames_discarded": 0,
        "continuity_valid": True,
    }


def test_binary_sidecar_checksum_comes_from_writer_not_a_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reread_forbidden(_path: Path) -> str:
        raise AssertionError("a clean capture must not be re-read for its checksum")

    monkeypatch.setattr("nlab.hardware.digitizer.dma._sha256_file", reread_forbidden)
    stop_event = threading.Event()
    output = tmp_path / "hashed.bin"

    IIOMcaDmaStreamer(_FakeBackend(stop_event), channel=0).stream_events(stop_event, output)

    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["capture_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_mca_worker_coalesces_frame_progress() -> None:
    def stream_events(*, on_ready, on_progress, **_kwargs):
        on_ready()
        for frame in range(1, 1001):
            on_progress(frame * 1024)
        return 1000 * 1024

    streamer = SimpleNamespace(stream_events=stream_events, last_capture_diagnostics=None)
    worker = IIOMcaDmaWorker(streamer)
    updates: list[int] = []
    summaries: list[Any] = []
    worker.progress.connect(updates.append)
    worker.summary.connect(summaries.append)

    worker.run()

    assert updates == [1024, 1000 * 1024]
    assert summaries[0].records == 1000 * 1024


def test_iio_streamer_forwards_initial_and_drained_frames_to_bounded_buffer() -> None:
    stop_event = threading.Event()
    backend = _FakeBackend(stop_event)
    event_buffer = McaEventBuffer()

    total = IIOMcaDmaStreamer(backend, channel=0).stream_events(
        stop_event=stop_event,
        event_buffer=event_buffer,
    )

    batches, dropped = event_buffer.drain()
    assert total == 2048
    assert dropped == 0
    assert len(batches) == 2
    np.testing.assert_array_equal(batches[0], backend.first)
    np.testing.assert_array_equal(batches[1], backend.tail)


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
        assert h5.attrs["client_record_schema"] == IIO_LM_UNQUALIFIED_SCHEMA
        assert h5["events"].dtype == _LM_EVENT_DTYPE
        np.testing.assert_array_equal(
            h5["trapezoid_energy"][:1024], backend.first["trapezoid_energy"]
        )
        assert "cfd_interpolation_samples" not in h5
        assert "cfd_valid" not in h5
        assert h5["events"].attrs["zc_estimation_format"] == (
            "signed Q2.14; marker-selected units"
        )


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


def test_online_drop_error_does_not_claim_that_a_file_was_retained() -> None:
    stop_event = threading.Event()
    backend = _FakeBackend(stop_event)
    backend.diagnostics = (0, 0, 2, 7)

    with pytest.raises(RuntimeError, match="online capture continuity is invalid") as exc:
        IIOMcaDmaStreamer(backend, channel=0).stream_events(stop_event)

    assert "file" not in str(exc.value)


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


def test_unknown_producer_refuses_decoded_consumers_before_arm():
    backend = Mock()
    backend.mca_producer_identity.return_value = {"pulse_hw_version": 999}
    streamer = IIOMcaDmaStreamer(backend, channel=0)
    with pytest.raises(RuntimeError, match="opaque binary"):
        streamer.stream_events(threading.Event(), event_buffer=McaEventBuffer())
    backend.start_mca_dma_capture.assert_not_called()


class _FakeFastStreamBackend(_FakeBackend):
    """Delivers frames as fast as possible; counts producer-status reads."""

    def __init__(self, stop_event: threading.Event, *, frames: int,
                 stop_after: bool, status_error: bool = False) -> None:
        super().__init__(stop_event)
        self.frames_left = frames
        self.stop_after = stop_after
        self.status_error = status_error
        self.status_reads = 0
        self.producer_running = True
        self.unblock = threading.Event()
        self.diagnostics = (0, 0, frames + 2, 0)

    def start_mca_dma_capture(
        self,
        on_started: Callable[[], None] | None = None,
    ) -> np.ndarray:
        if on_started is not None:
            on_started()
        return self.first

    def read_mca_dma_frame(self) -> np.ndarray:
        if self.frames_left:
            self.frames_left -= 1
            if not self.frames_left:
                if self.stop_after:
                    self._stop_event.set()
                else:
                    self.producer_running = False  # hardware time limit
            return _frame(3)
        # A refill after the producer's final frame blocks until the STOP
        # watchdog cancels it (modelled by request_mca_dma_stop).
        assert self.unblock.wait(2.0), "blocked refill was never released"
        return np.empty(0, dtype=_LM_EVENT_DTYPE)

    def mca_dma_measurement_in_progress(self) -> bool:
        self.status_reads += 1
        if self.status_error and self.status_reads > 1:
            raise OSError(5, "status read failed")
        return self.producer_running

    def request_mca_dma_stop(self) -> None:
        super().request_mca_dma_stop()
        self.unblock.set()


def test_streamer_does_not_read_producer_status_per_frame(monkeypatch: Any) -> None:
    import nlab.hardware.digitizer.dma as dma_module

    monkeypatch.setattr(dma_module, "_MCA_PRODUCER_POLL_SECONDS", 0.5)
    stop_event = threading.Event()
    backend = _FakeFastStreamBackend(stop_event, frames=2000, stop_after=True)
    streamer = IIOMcaDmaStreamer(backend, channel=0)

    total = streamer.stream_events(stop_event=stop_event)

    assert total == (1 + 2000 + 1) * 1024
    # Live, one status round trip per frame cost ~30% of the record rate.
    assert backend.status_reads <= 3


def test_streamer_ends_on_hardware_time_limit_without_gui_stop(monkeypatch: Any) -> None:
    import nlab.hardware.digitizer.dma as dma_module

    monkeypatch.setattr(dma_module, "_MCA_PRODUCER_POLL_SECONDS", 0.01)
    stop_event = threading.Event()
    backend = _FakeFastStreamBackend(stop_event, frames=50, stop_after=False)
    streamer = IIOMcaDmaStreamer(backend, channel=0)

    total = streamer.stream_events(stop_event=stop_event)

    assert not stop_event.is_set()
    assert backend.stop_requests == 1
    assert total == (1 + 50 + 1) * 1024  # first, streamed and drained tail
    assert streamer.last_capture_diagnostics["continuity_valid"] is True


def test_streamer_fails_when_producer_status_read_fails(monkeypatch: Any) -> None:
    import nlab.hardware.digitizer.dma as dma_module

    monkeypatch.setattr(dma_module, "_MCA_PRODUCER_POLL_SECONDS", 0.01)
    stop_event = threading.Event()
    backend = _FakeFastStreamBackend(
        stop_event, frames=0, stop_after=False, status_error=True,
    )
    streamer = IIOMcaDmaStreamer(backend, channel=0)

    with pytest.raises(OSError, match="status read failed"):
        streamer.stream_events(stop_event=stop_event)
    assert backend.stop_requests == 1


def test_capture_metadata_records_the_kernel_block_count_the_backend_arms() -> None:
    from nlab.hardware.digitizer.dma import IIO_LM_KERNEL_BUFFER_COUNT

    # Live v121 stalls of 20-50 ms emptied eight blocks; 64 absorbed them.
    assert IIO_LM_KERNEL_BUFFER_COUNT == _LM_KERNEL_BUFFER_COUNT == 64


class _FakeCarryOverBackend(_FakeBackend):
    """Delivers *stale* frames from an undrained session before START."""

    def __init__(self, stop_event: threading.Event, stale: list[np.ndarray],
                 session: list[np.ndarray], deadtime: tuple[int, int] = (0, 0)) -> None:
        super().__init__(stop_event)
        self._frames = [*stale, *session]
        self.diagnostics = (0, 0, len(self._frames), deadtime[1])
        self.deadtime_before = deadtime[0]

    def get_mca_dma_deadtime_baseline(self) -> int:
        return self.deadtime_before

    def start_mca_dma_capture(
        self,
        on_started: Callable[[], None] | None = None,
    ) -> np.ndarray:
        if on_started is not None:
            on_started()
        self._stop_event.set()
        return self._frames.pop(0)

    def close_mca_dma_capture(
        self,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> int:
        frames, self._frames = self._frames, []
        for frame in frames:
            assert on_frame is not None
            on_frame(frame)
        return len(frames)


def test_frames_before_the_start_record_are_discarded(tmp_path: Path) -> None:
    stop_event = threading.Event()
    stale = [_frame(7), _frame(8)]
    session = [_session_first_frame(1), _frame(2)]
    backend = _FakeCarryOverBackend(stop_event, stale, session)
    output = tmp_path / "carry.bin"
    streamer = IIOMcaDmaStreamer(backend, channel=0)

    assert streamer.stream_events(stop_event, output) == 2 * 1024

    with output.open("rb") as file:
        file.seek(FILE_HEADER_STRUCT.size)
        parsed = np.frombuffer(file.read(), dtype=_LM_EVENT_DTYPE)
    np.testing.assert_array_equal(parsed, np.concatenate(session))
    diagnostics = streamer.last_capture_diagnostics
    assert diagnostics["stale_frames_discarded"] == 2
    assert diagnostics["start_record"] == "first"
    assert diagnostics["continuity_valid"] is True
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["stale_frames_discarded"] == 2
    assert metadata["driver_completed_frames"] == 4
    assert metadata["continuity_valid"] is True


def test_start_record_inside_the_first_frame_invalidates_continuity() -> None:
    stop_event = threading.Event()
    misaligned = _frame(1)
    misaligned[6] = np.zeros(1, dtype=_LM_EVENT_DTYPE)[0]
    misaligned["timestamp"][6] = 1
    backend = _FakeCarryOverBackend(stop_event, [], [misaligned, _frame(2)])
    streamer = IIOMcaDmaStreamer(backend, channel=0)

    with pytest.raises(RuntimeError, match="START record is record 6"):
        streamer.stream_events(stop_event)

    diagnostics = streamer.last_capture_diagnostics
    assert diagnostics["start_record"] == "misaligned"
    assert diagnostics["streamed_frames"] == 2  # delivered unchanged
    assert diagnostics["continuity_valid"] is False


def test_missing_start_record_keeps_the_frames_and_invalidates_continuity() -> None:
    stop_event = threading.Event()
    backend = _FakeCarryOverBackend(stop_event, [], [_frame(1), _frame(2)])
    events = McaEventBuffer()
    streamer = IIOMcaDmaStreamer(backend, channel=0)

    with pytest.raises(RuntimeError, match="no START record"):
        streamer.stream_events(stop_event, event_buffer=events)

    batches, _ = events.drain()
    assert len(batches) == 2
    assert streamer.last_capture_diagnostics["start_record"] == "missing"


@pytest.mark.parametrize("before,after,loss", [
    (0, 0, False),
    (30001, 30001, False),  # carried over from an earlier session (L2)
    (30001, 0, False),      # cleared by this session's first record
    (30001, 30005, True),
    (0, 7, True),
])
def test_deadtime_loss_is_judged_against_the_pre_arm_baseline(
    before: int, after: int, loss: bool,
) -> None:
    stop_event = threading.Event()
    backend = _FakeCarryOverBackend(
        stop_event, [], [_session_first_frame(1)], deadtime=(before, after),
    )
    streamer = IIOMcaDmaStreamer(backend, channel=0)

    if loss:
        with pytest.raises(RuntimeError, match="dropped records"):
            streamer.stream_events(stop_event)
    else:
        streamer.stream_events(stop_event)
    diagnostics = streamer.last_capture_diagnostics
    assert diagnostics["list_deadtime_before_raw"] == before
    assert diagnostics["continuity_valid"] is (not loss)
