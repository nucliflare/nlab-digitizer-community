# SPDX-License-Identifier: MIT
"""Batched iiod 0.25 transport for exact-frame Scope DMA reads.

The normal libiio buffer API sends one synchronous ``READBUF`` request for
each Scope frame.  The current PetaLinux Scope driver keeps one hardware frame
per IIO block but supports four queued blocks, so a bounded multi-frame
``READBUF`` can amortize network request latency without changing DMA geometry.

This protocol client follows the reference implementation shipped with the
authoritative PetaLinux tree.  It returns complete prefix frames one at a time;
an error later in a batch is raised on the corresponding refill and permanently
faults the stream.
"""

from __future__ import annotations

import errno
import re
import socket
from typing import BinaryIO


class IiodScopeStream:
    """One exact-frame IIO buffer read through batched iiod ``READBUF``."""

    def __init__(
        self,
        uri: str,
        device: str,
        samples: int,
        buffers: int = 1,
        batch_frames: int = 32,
        timeout_ms: int = 3000,
    ) -> None:
        match = re.fullmatch(r"ip:([A-Za-z0-9.-]+)(?::([0-9]+))?", uri)
        if match is None or re.fullmatch(r"iio:device[0-9]+", device) is None:
            raise ValueError("requires ip:host[:port] and a mapped IIO device ID")
        if (
            type(samples) is not int
            or not 4 <= samples <= 8188
            or samples % 4
            or type(buffers) is not int
            or buffers not in (1, 2, 4)
            or type(batch_frames) is not int
            or not 1 <= batch_frames <= 64
            or type(timeout_ms) is not int
            or not 1 <= timeout_ms <= 60000
        ):
            raise ValueError("invalid scope geometry, depth, batch or timeout")

        self.device = device
        self.frame_bytes = samples * 2
        self.batch_frames = batch_frames
        self.remaining = 0
        self.first_chunk = False
        self.failed = False
        self.closed = False
        self.payload: bytes | None = None
        host = match.group(1)
        port = int(match.group(2) or 30431)
        self.sock = socket.create_connection(
            (host, port),
            timeout=timeout_ms / 1000 + 1,
        )
        self.input: BinaryIO | None = None
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.input = self.sock.makefile("rb")
            self.sock.sendall(b"VERSION\r\n")
            self.version = self._line().decode("ascii")
            if not self.version.startswith("0.25."):
                raise ValueError("only the verified iiod 0.25 protocol is supported")
            self._command(f"TIMEOUT {timeout_ms}")
            self._command(f"SET {device} BUFFERS_COUNT {buffers}")
            self._command(f"OPEN {device} {samples} 00000001")
        except BaseException:
            self.close()
            raise

    def _line(self) -> bytes:
        if self.input is None:
            raise OSError(errno.EPIPE, "iiod stream is closed")
        line = self.input.readline(65)
        if not line.endswith(b"\n") or len(line) > 64:
            raise OSError(errno.EPROTO, "missing or oversized iiod protocol line")
        return line.rstrip(b"\r\n")

    def _integer(self) -> int:
        line = self._line()
        if re.fullmatch(rb"-?[0-9]+", line) is None:
            raise OSError(errno.EPROTO, "invalid iiod byte count/status")
        return int(line)

    def _command(self, command: str) -> None:
        self.sock.sendall(command.encode("ascii") + b"\r\n")
        result = self._integer()
        if result:
            raise OSError(-result if result < 0 else errno.EPROTO, command)

    def refill(self) -> int:
        """Receive one complete frame, retaining the rest of its batch."""
        if self.failed or self.closed:
            raise OSError(errno.EPIPE, "stream is closed or faulted; close and rearm")
        self.payload = None
        try:
            if not self.remaining:
                self.remaining = self.batch_frames
                self.first_chunk = True
                request_bytes = self.frame_bytes * self.remaining
                self.sock.sendall(
                    f"READBUF {self.device} {request_bytes}\r\n".encode("ascii")
                )
            size = self._integer()
            if size < 0:
                raise OSError(
                    -size,
                    "iiod READBUF (earlier complete frames remain valid)",
                )
            if size != self.frame_bytes:
                raise OSError(errno.EPROTO, f"short/misaligned batch chunk: {size}")
            if self.first_chunk and self._line() != b"00000001":
                raise OSError(errno.EPROTO, "unexpected scan mask")
            self.first_chunk = False
            if self.input is None:
                raise OSError(errno.EPIPE, "iiod stream is closed")
            payload = self.input.read(size)
            if len(payload) != size:
                raise OSError(errno.EPROTO, "connection ended within a frame")
            self.payload = payload
            self.remaining -= 1
            return size
        except BaseException:
            self.failed = True
            raise

    def read(self) -> bytes:
        """Return the payload from the most recent successful refill."""
        if self.payload is None:
            raise RuntimeError("no successfully refilled frame")
        return self.payload

    def cancel(self) -> None:
        """Interrupt a blocked refill; the stream cannot be reused."""
        self.failed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def close(self) -> None:
        """Close the transport and let iiod destroy its IIO buffer."""
        if self.closed:
            return
        self.closed = True
        self.cancel()
        try:
            if self.input is not None:
                self.input.close()
        finally:
            self.sock.close()
