from __future__ import annotations

import errno
import io
import json
import socket
import struct
import threading
from collections.abc import Callable
from types import SimpleNamespace

import h5py
import pytest

from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.iio_scope_stream import IiodScopeStream
from nlab.hardware.digitizer.scope_sums import (
    MASK64,
    ScopeSumCapture,
    ScopeSumGeometry,
    read_sum_capture,
)


class _FakeSocket:
    def __init__(self, responses: bytes) -> None:
        self.input = io.BytesIO(responses)
        self.sent: list[bytes] = []
        self.options: list[tuple[int, int, int]] = []
        self.shutdown_calls: list[int] = []
        self.closed = False

    def setsockopt(self, level: int, option: int, value: int) -> None:
        self.options.append((level, option, value))

    def makefile(self, mode: str) -> io.BytesIO:
        assert mode == "rb"
        return self.input

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def shutdown(self, how: int) -> None:
        self.shutdown_calls.append(how)

    def close(self) -> None:
        self.closed = True


def _patch_connection(
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakeSocket,
) -> list[tuple[tuple[str, int], float | None]]:
    calls: list[tuple[tuple[str, int], float | None]] = []

    def connect(address: tuple[str, int], timeout: float | None = None) -> _FakeSocket:
        calls.append((address, timeout))
        return fake

    monkeypatch.setattr(socket, "create_connection", connect)
    return calls


def _open_responses() -> bytes:
    return b"0.25.0\r\n0\r\n0\r\n0\r\n"


def test_batched_stream_opens_four_blocks_and_reuses_one_readbuf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = bytes(range(16))
    second = bytes(range(16, 32))
    fake = _FakeSocket(
        _open_responses()
        + b"16\r\n00000001\r\n"
        + first
        + b"16\r\n"
        + second
    )
    calls = _patch_connection(monkeypatch, fake)

    stream = IiodScopeStream(
        "ip:192.168.10.128:30431",
        "iio:device15",
        samples=8,
        buffers=4,
        batch_frames=2,
    )
    assert stream.refill() == 16
    assert stream.read() == first
    assert stream.refill() == 16
    assert stream.read() == second

    assert calls == [(('192.168.10.128', 30431), 4.0)]
    assert fake.sent == [
        b"VERSION\r\n",
        b"TIMEOUT 3000\r\n",
        b"SET iio:device15 BUFFERS_COUNT 4\r\n",
        b"OPEN iio:device15 8 00000001\r\n",
        b"READBUF iio:device15 32\r\n",
    ]
    assert fake.options == [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]

    stream.close()
    assert fake.shutdown_calls == [socket.SHUT_RDWR]
    assert fake.closed


def test_batch_preserves_complete_prefix_before_server_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = b"x" * 16
    fake = _FakeSocket(
        _open_responses() + b"16\r\n00000001\r\n" + frame + b"-110\r\n"
    )
    _patch_connection(monkeypatch, fake)
    stream = IiodScopeStream(
        "ip:board",
        "iio:device9",
        samples=8,
        buffers=2,
        batch_frames=2,
    )

    assert stream.refill() == 16
    assert stream.read() == frame
    with pytest.raises(OSError) as caught:
        stream.refill()
    assert caught.value.errno == 110
    with pytest.raises(OSError) as reused:
        stream.refill()
    assert reused.value.errno == errno.EPIPE


def test_stream_rejects_partial_frame_and_cannot_be_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSocket(_open_responses() + b"8\r\n")
    _patch_connection(monkeypatch, fake)
    stream = IiodScopeStream(
        "ip:board",
        "iio:device9",
        samples=8,
        batch_frames=1,
    )

    with pytest.raises(OSError, match="short/misaligned") as caught:
        stream.refill()
    assert caught.value.errno == errno.EPROTO
    with pytest.raises(OSError) as reused:
        stream.refill()
    assert reused.value.errno == errno.EPIPE


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ("usb:", "iio:device1", 8, 1, 1, 3000),
        lambda: ("ip:board", "vdpp_scope", 8, 1, 1, 3000),
        lambda: ("ip:board", "iio:device1", 6, 1, 1, 3000),
        lambda: ("ip:board", "iio:device1", 8, 3, 1, 3000),
    ],
)
def test_stream_rejects_unsupported_protocol_parameters(
    factory: Callable[[], tuple[str, str, int, int, int, int]],
) -> None:
    with pytest.raises(ValueError):
        IiodScopeStream(*factory())


def _sum_packet(sequence, *, count=508, stopped=False, missed=0):
    # Native uint64 wrap plus two skipped windows before packet 1.
    first = (MASK64 - 50 + (sequence * 508 + (2 if sequence else 0)) * 7) & MASK64
    raw = bytearray(4096)
    struct.pack_into(
        "<4Q", raw, 0, first, 0x32534D5553504456, 16 | (3 << 32) | (count << 48),
        sequence | (missed << 32) | (int(stopped) << 63),
    )
    for i in range(508):
        struct.pack_into("<Q", raw, 32 + 8 * i, 16 * 32768 + min(i, count - 1))
    return bytes(raw)


@pytest.mark.parametrize("batch_frames", [2, 64])
@pytest.mark.parametrize("ending,status_value", [
    ("server-timeout", 16), ("server-timeout", 2), ("server-timeout", 8),
    ("socket-timeout-linux", 16), ("socket-timeout-windows", 16),
    ("server-eio", 16), ("bad-reply", 16), ("truncated-frame", 16),
])
def test_sum_capture_retains_batched_tail_and_only_server_timeout_completes(
    tmp_path, monkeypatch, batch_frames, ending, status_value,
):
    packets = [
        _sum_packet(0), _sum_packet(1, missed=2),
        _sum_packet(2, count=3, stopped=True, missed=2),
    ]
    response = _open_responses()
    for i, packet in enumerate(packets):
        response += b"4096\r\n" + (b"00000001\r\n" if i % batch_frames == 0 else b"") + packet
    response += {
        "server-timeout": b"-110\r\n", "server-eio": b"-5\r\n",
        "bad-reply": b"not-a-length\r\n", "socket-timeout-linux": b"",
        "socket-timeout-windows": b"", "truncated-frame": b"4096\r\n" + b"x" * 128,
    }[ending]
    fake = _FakeSocket(response)
    if ending.startswith("socket-timeout"):
        class TimedInput(io.BytesIO):
            def readline(self, size=-1):
                if self.tell() == len(self.getbuffer()):
                    # Linux's local errno 110 equals the server errno, while
                    # Windows commonly reports WSAETIMEDOUT 10060.
                    code = 110 if ending.endswith("linux") else 10060
                    raise TimeoutError(code, "injected socket timeout")
                return super().readline(size)

        fake.input = TimedInput(response)
    _patch_connection(monkeypatch, fake)
    stream = IiodScopeStream("ip:board", "iio:device13", 2048, 4, batch_frames)
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._dma_buf = stream
    backend._dma_buf_frame_samples = 2048
    backend._dma_capture_geometry = ScopeSumGeometry(16, 124, 3)
    backend._dma_fault_latched = False
    backend._dma_stop_requested = threading.Event()
    backend._sum_draining = False
    backend._dma_scope = SimpleNamespace(attrs={})
    events = []
    monkeypatch.setattr(backend, "prepare_dma_capture", lambda: events.append("prepare"))
    monkeypatch.setattr(backend, "get_scope_dma_geometry", lambda: ScopeSumGeometry(16, 124, 3))
    monkeypatch.setattr(backend, "scope_sum_identity", lambda: {"uri": "fixture"})
    monkeypatch.setattr(backend, "_dma_get_enable", lambda: False)
    monkeypatch.setattr(backend, "_dma_set_enable", lambda val: events.append(("enable", val)))
    monkeypatch.setattr(backend, "_wait_for_dma_release", lambda: events.append("release"))

    def status(name):
        events.append(name)
        return {"sum_status": str(status_value), "sum_missed_windows": "2"}[name]

    monkeypatch.setattr(backend, "_dma_attr_get", status)
    runtime = {"transport": "iiod-batched", "kernel_buffers": 4,
               "readbuf_batch_frames": batch_frames}
    monkeypatch.setattr(backend, "get_scope_dma_runtime_metadata", lambda: runtime)
    path = tmp_path / "sum-tail.h5"
    capture = ScopeSumCapture(backend, 0)
    complete = ending == "server-timeout" and status_value == 16
    if complete:
        result = capture.run(path, max_packets=1)
        assert result["complete"] and result["final_missed_windows"] == 2
        assert result["windows"] == 1019 and result["skipped"] == 2
        assert events.index("sum_status") < events.index("release")
        assert not backend._dma_fault_latched
        assert stream.server_timed_out
    else:
        with pytest.raises(RuntimeError):
            capture.run(path, max_packets=1)
        assert backend._dma_fault_latched == (ending != "server-timeout")
        assert ("sum_status" in events) == (ending == "server-timeout")
        assert stream.server_timed_out == (ending == "server-timeout")
    assert stream.failed and stream.closed and fake.closed
    assert backend._dma_buf is None
    assert events[-1] in ("release", ("enable", False))
    assert [p.raw for p in read_sum_capture(path)] == packets
    with h5py.File(path, "r") as source:
        assert bool(source.attrs["complete"]) == complete
        assert bool(source.attrs["cleanup_error"]) == (not complete)
        if complete:
            assert json.loads(source.attrs["runtime_metadata_json"]) == runtime
        else:
            assert "runtime_metadata_json" not in source.attrs
    requests = [cmd for cmd in fake.sent if cmd.startswith(b"READBUF")]
    assert len(requests) == (2 if batch_frames == 2 else 1)
    assert requests[0] == f"READBUF iio:device13 {4096 * batch_frames}\r\n".encode()
    assert fake.sent[3] == b"OPEN iio:device13 2048 00000001\r\n"
    sent = len(fake.sent)
    with pytest.raises(OSError) as caught:
        stream.refill()
    assert caught.value.errno == errno.EPIPE and len(fake.sent) == sent
