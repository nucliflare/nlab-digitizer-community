from __future__ import annotations

import errno
import io
import socket
from collections.abc import Callable

import pytest

from nlab.hardware.digitizer.iio_scope_stream import IiodScopeStream


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
