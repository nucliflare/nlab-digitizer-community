"""Independent wire fixtures; these checks do not qualify detector response."""

import errno
import struct

import h5py
import pytest

from nlab.hardware.digitizer.scope_sums import (
    MASK64,
    ScopeSumAnalysis,
    ScopeSumCapture,
    ScopeSumContinuity,
    ScopeSumGeometry,
    ScopeSumWriter,
    decode_sum_packet,
    read_sum_capture,
)


def packet(
    version=124,
    n=16,
    gap=3,
    count=None,
    first=(1 << 60) + 7,
    sequence=0,
    total=0,
    stopped=False,
    missed=0,
):
    capacity = 508 if version == 124 else 254
    count = capacity if count is None else count
    magic = 0x32534D5553504456 if version == 124 else 0x31534D5553504456
    raw = bytearray(4096)
    struct.pack_into(
        "<4Q",
        raw,
        0,
        first,
        magic,
        n | gap << 32 | count << 48,
        sequence | missed << 32 | int(stopped) << 63,
    )
    for i in range(capacity if version == 124 else count):
        if version == 124:
            struct.pack_into("<Q", raw, 32 + 8 * i, total)
        else:
            struct.pack_into("<QQ", raw, 32 + 16 * i, (first + i * (n // 4 + gap)) & MASK64, total)
    return bytes(raw)


@pytest.mark.parametrize("version,capacity", [(123, 254), (124, 508)])
@pytest.mark.parametrize("total,mean", [(0, -32768.0), (16 * 32768, 0.0), (16 * 65535, 32767.0)])
def test_packet_formats_exclude_padding_and_keep_native_ticks(version, capacity, total, mean):
    raw = packet(version, total=total)
    decoded = decode_sum_packet(raw, ScopeSumGeometry(16, version, 3))
    assert len(decoded.records) == capacity
    assert decoded.records[0][0] == (1 << 60) + 7
    assert decoded.records[-1] == ((1 << 60) + 7 + (capacity - 1) * 7, total)
    assert decoded.signed_means == (mean,) * capacity


@pytest.mark.parametrize("version", [123, 124])
def test_partial_stop_and_timestamp_wrap(version):
    raw = packet(version, count=3, first=MASK64 - 3, stopped=True)
    decoded = decode_sum_packet(raw)
    assert [t for t, _ in decoded.records] == [MASK64 - 3, 3, 10]
    assert decoded.stop_flush
    with pytest.raises(ValueError, match="STOP"):
        decode_sum_packet(packet(version, count=3))


@pytest.mark.parametrize("mutation", ["length", "magic", "count", "padding", "sum", "geometry"])
def test_corrupt_packets_are_rejected(mutation):
    raw = bytearray(packet(count=2, total=10, stopped=True))
    if mutation == "length":
        raw.pop()
    elif mutation == "magic":
        raw[8] ^= 1
    elif mutation == "count":
        raw[22:24] = b"\0\0"
    elif mutation == "padding":
        raw[-1] = 1
    elif mutation == "sum":
        struct.pack_into("<Q", raw, 32, 16 * 65535 + 1)
    elif mutation == "geometry":
        struct.pack_into("<I", raw, 16, 17)
    with pytest.raises(ValueError):
        decode_sum_packet(bytes(raw))


def test_armed_geometry_is_checked_and_sums_are_not_waveforms():
    geometry = ScopeSumGeometry(16, 124, 3)
    assert (geometry.frame_bytes, geometry.buffer_samples) == (4096, 2048)
    assert (geometry.waveform_samples, geometry.sample_decimation) == (0, 0)
    assert not geometry.ndma_v1_compatible
    with pytest.raises(ValueError, match="armed"):
        decode_sum_packet(packet(), ScopeSumGeometry(20, 124, 3))


def test_sequence_wrap_gaps_and_terminal_flush():
    check = ScopeSumContinuity()
    check.append(decode_sum_packet(packet(sequence=0xFFFFFFFF, first=10)))
    check.append(decode_sum_packet(packet(sequence=0, first=10 + 510 * 7, count=1, stopped=True)))
    assert check.skipped_windows == 2
    with pytest.raises(ValueError, match="terminal"):
        check.append(decode_sum_packet(packet(sequence=1)))
    check = ScopeSumContinuity()
    check.append(decode_sum_packet(packet()))
    with pytest.raises(ValueError, match="sequence"):
        check.append(decode_sum_packet(packet(sequence=2)))


def test_mean_weights_windows_and_measured_samples_not_packet_count():
    analysis = ScopeSumAnalysis()
    first = 100
    analysis.append(decode_sum_packet(packet(first=first, total=16 * 32768)))
    analysis.append(
        decode_sum_packet(
            packet(first=first + 508 * 7, sequence=1, count=1, stopped=True, total=16 * 65535)
        )
    )
    result = analysis.snapshot()
    assert result["windows"] == 509
    assert result["samples"] == 509 * 16
    assert result["mean"] == pytest.approx(32767 / 509)
    assert result["coverage_ns"] == 509 * 16 * 2
    assert len(result["points"]) == 509


def test_lossless_recording_roundtrip_and_incomplete_status(tmp_path):
    path = tmp_path / "sums.h5"
    original = decode_sum_packet(packet(count=2, stopped=True, total=123456))
    writer = ScopeSumWriter(path, {"channel": 1})
    writer.append(original)
    writer.close(complete=False, error="injected release fault")
    writer.close(complete=True)  # Idempotent close cannot rewrite completion.
    assert list(read_sum_capture(path)) == [original]
    with h5py.File(path, "r+") as source:
        assert source["timestamps"][0] == (1 << 60) + 7
        assert not source.attrs["complete"]
        assert source.attrs["signal_origin"] == "measurement"
        source["sums"][0] = 123457
    with pytest.raises(ValueError, match="differs"):
        list(read_sum_capture(path))


class Backend:
    def __init__(self, raws, status=0, close_error=False):
        self.raws = iter(raws)
        self.events = []
        self.status = status
        self.close_error = close_error

    def prepare_dma_capture(self):
        self.events.append("prepare")

    def get_scope_dma_geometry(self):
        return ScopeSumGeometry(16, 124, 3)

    def read_dma_raw_frame(self):
        self.events.append("read")
        try:
            return next(self.raws)
        except StopIteration:
            raise OSError(errno.ETIMEDOUT, "end") from None

    def request_scope_sum_stop(self):
        self.events.append("stop")

    def stop_scope_sum_from_reader(self):
        self.events.append("stop")

    def scope_sum_status(self):
        self.events.append("status")
        return self.status, 5

    def close_scope_sum_capture(self):
        self.events.append("close")
        if self.close_error:
            raise RuntimeError("release fault")

    def scope_sum_identity(self):
        return {"uri": "fixture", "channel": 0}


def test_stop_keeps_tail_until_terminal_timeout_then_closes(tmp_path):
    backend = Backend(
        [packet(first=10), packet(sequence=1, first=10 + 508 * 7, count=1, stopped=True)]
    )
    capture = ScopeSumCapture(backend, 0)
    path = tmp_path / "tail.h5"
    result = capture.run(path, max_packets=1)
    assert result["windows"] == 509 and result["complete"]
    assert backend.events[-2:] == ["stop", "close"]
    assert backend.events.index("status") < backend.events.index("close")
    assert len(list(read_sum_capture(path))) == 2


@pytest.mark.parametrize("status,close_error", [(2, False), (8, False), (0, True)])
def test_faults_keep_recorded_prefix_incomplete(tmp_path, status, close_error):
    backend = Backend([packet()], status, close_error)
    path = tmp_path / "partial.h5"
    with pytest.raises(RuntimeError):
        ScopeSumCapture(backend, 0).run(path, max_packets=1)
    assert backend.events[-1] == "close"
    with h5py.File(path, "r") as source:
        assert not source.attrs["complete"]
        assert len(source["packets"]) == 1
        assert source.attrs["cleanup_error"]


def test_unsolicited_timeout_is_not_a_successful_drain():
    backend = Backend([])
    with pytest.raises(RuntimeError, match="Timeout|timed|end"):
        ScopeSumCapture(backend, 0).run()
    assert "status" not in backend.events
    assert backend.events[-1] == "close"


def test_cancelled_before_arm_never_requests_a_frame():
    backend = Backend([])
    capture = ScopeSumCapture(backend, 0)
    capture.request_stop()
    result = capture.run()
    assert result["complete"] is False
    assert "read" not in backend.events and backend.events[-1] == "close"


def test_cancel_before_arm_still_reports_cleanup_failure():
    backend = Backend([], close_error=True)
    capture = ScopeSumCapture(backend, 0)
    capture.request_stop()
    with pytest.raises(RuntimeError, match="Cleanup: release fault"):
        capture.run()
    assert "read" not in backend.events
