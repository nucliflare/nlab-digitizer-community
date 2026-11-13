"""Independent wire fixtures; these checks do not qualify detector response."""

import errno
import json
import struct
import threading

import h5py
import pytest

from nlab.hardware.digitizer import scope_sums
from nlab.hardware.digitizer.scope_sums import (
    MASK64,
    ScopeSumAnalysis,
    ScopeSumCapture,
    ScopeSumContinuity,
    ScopeSumGeometry,
    ScopeSumRecorder,
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

    def get_scope_dma_runtime_metadata(self):
        self.events.append("runtime")
        return {"transport": "fixture", "kernel_buffers": 4, "readbuf_batch_frames": 64}


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
    assert backend.events.index("status") < backend.events.index("runtime")
    assert backend.events.index("runtime") < backend.events.index("close")
    with h5py.File(path, "r") as source:
        assert json.loads(source.attrs["runtime_metadata_json"]) == {
            "transport": "fixture", "kernel_buffers": 4, "readbuf_batch_frames": 64,
        }


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


@pytest.mark.parametrize("version,capacity", [(123, 254), (124, 508)])
def test_batch_writer_preserves_wrap_partial_tail_and_exact_columns(tmp_path, version, capacity):
    first = MASK64 - 100
    originals = [
        decode_sum_packet(packet(
            version, first=(first + i * capacity * 7) & MASK64,
            sequence=(0xFFFFFFFE + i) & 0xFFFFFFFF,
            count=3 if i == 2 else capacity, stopped=i == 2, total=123456 + i,
        ))
        for i in range(3)
    ]
    path = tmp_path / "batch.h5"
    writer = ScopeSumWriter(path, {"channel": 0})
    writer.append_many(originals[:2])
    writer.append_many([])
    writer.append_many(originals[2:])
    writer.close(complete=True)
    assert list(read_sum_capture(path)) == originals
    with pytest.raises(RuntimeError, match="closed"):
        writer.append(originals[0])


def test_recorder_keeps_hdf5_off_producer_and_joins_queued_tail(tmp_path, monkeypatch):
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()
    io_threads, batches = [], []
    original = ScopeSumWriter

    class HeldWriter(original):
        def __init__(self, *args, **kwargs):
            io_threads.append(threading.get_ident())
            super().__init__(*args, **kwargs)

        def append_many(self, packets):
            io_threads.append(threading.get_ident())
            batches.append(len(packets))
            entered.set()
            assert release.wait(5)
            super().append_many(packets)

        def close(self, **kwargs):
            io_threads.append(threading.get_ident())
            super().close(**kwargs)

    monkeypatch.setattr(scope_sums, "ScopeSumWriter", HeldWriter)
    path = tmp_path / "async.h5"
    recorder = ScopeSumRecorder(path, {}, queue_packets=8, batch_packets=2)
    packets = [
        decode_sum_packet(packet(first=10 + i * 508 * 7, sequence=i,
                                 count=3 if i == 5 else 508, stopped=i == 5, total=i))
        for i in range(6)
    ]
    closer = None
    errors = []

    def close():
        closing.set()
        try:
            recorder.close(complete=True, missed_windows=7)
        except Exception as exc:
            errors.append(str(exc))

    try:
        for p in packets[:2]:
            recorder.append(p)
        assert entered.wait(2)
        # Producer continues while the HDF5 sink is deliberately blocked.
        for p in packets[2:]:
            recorder.append(p)
        closer = threading.Thread(target=close)
        closer.start()
        assert closing.wait(2) and closer.is_alive()
    finally:
        release.set()
        if closer:
            closer.join(5)
        recorder.close(complete=False)
    assert not errors and not closer.is_alive() and not recorder._thread.is_alive()
    assert len(set(io_threads)) == 1 and io_threads[0] != threading.get_ident()
    assert batches == [2, 2, 2]
    assert list(read_sum_capture(path)) == packets
    with h5py.File(path, "r") as source:
        assert source.attrs["complete"] and source.attrs["final_missed_windows"] == 7
    with pytest.raises(RuntimeError, match="closed"):
        recorder.append(packets[0])


def test_low_rate_recorder_flushes_partial_batch_before_stop(tmp_path, monkeypatch):
    written = threading.Event()

    class NotifyingWriter(ScopeSumWriter):
        def append_many(self, packets):
            super().append_many(packets)
            written.set()

    monkeypatch.setattr(scope_sums, "ScopeSumWriter", NotifyingWriter)
    recorder = ScopeSumRecorder(tmp_path / "low-rate.h5", {}, batch_seconds=0.01)
    try:
        recorder.append(decode_sum_packet(packet()))
        assert written.wait(2)
    finally:
        recorder.close(complete=False)


def test_recording_queue_overflow_stops_capture_keeps_prefix_and_joins(tmp_path, monkeypatch):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class HeldWriter(ScopeSumWriter):
        def append_many(self, packets):
            entered.set()
            assert release.wait(5)
            super().append_many(packets)

    class SmallRecorder(ScopeSumRecorder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, queue_packets=1, batch_packets=1)

    class ObservedBackend(Backend):
        def read_dma_raw_frame(self):
            if "read" in self.events:
                assert entered.wait(2)
            return super().read_dma_raw_frame()

        def close_scope_sum_capture(self):
            super().close_scope_sum_capture()
            closed.set()

    monkeypatch.setattr(scope_sums, "ScopeSumWriter", HeldWriter)
    monkeypatch.setattr(scope_sums, "ScopeSumRecorder", SmallRecorder)
    raws = [packet(sequence=i, first=10 + i * 508 * 7) for i in range(3)]
    backend = ObservedBackend(raws)
    capture = ScopeSumCapture(backend, 0)
    path, errors = tmp_path / "overflow.h5", []

    def run():
        try:
            capture.run(path)
        except RuntimeError as exc:
            errors.append(str(exc))

    reader = threading.Thread(target=run)
    reader.start()
    try:
        assert closed.wait(2)
        assert reader.is_alive()  # Capture cannot finish while its writer is held.
    finally:
        release.set()
        reader.join(5)
    assert not reader.is_alive() and len(errors) == 1 and "queue is full" in errors[0]
    assert backend.events[-2:] == ["stop", "close"]
    assert [p.raw for p in read_sum_capture(path)] == raws[:2]
    with h5py.File(path, "r") as source:
        assert not source.attrs["complete"]
        assert "queue is full" in source.attrs["cleanup_error"]


def test_async_write_failure_is_reported_and_file_is_incomplete(tmp_path, monkeypatch):
    class FailingWriter(ScopeSumWriter):
        def append_many(self, packets):
            super().append_many(packets)
            raise OSError("injected disk failure")

    monkeypatch.setattr(scope_sums, "ScopeSumWriter", FailingWriter)
    path = tmp_path / "failed.h5"
    recorder = ScopeSumRecorder(path, {}, batch_packets=1)
    original = decode_sum_packet(packet())
    recorder.append(original)
    recorder._thread.join(2)
    assert not recorder._thread.is_alive()
    with pytest.raises(RuntimeError, match="injected disk failure"):
        recorder.append(original)
    with pytest.raises(RuntimeError, match="injected disk failure"):
        recorder.close(complete=True)
    assert list(read_sum_capture(path)) == [original]
    with h5py.File(path, "r") as source:
        assert not source.attrs["complete"]
        assert "injected disk failure" in source.attrs["cleanup_error"]


def test_capture_propagates_async_close_failure_after_dma_cleanup(tmp_path, monkeypatch):
    class FailingCloseWriter(ScopeSumWriter):
        def close(self, **kwargs):
            super().close(**(kwargs | {"complete": False, "error": "injected flush failure"}))
            raise OSError("injected flush failure")

    monkeypatch.setattr(scope_sums, "ScopeSumWriter", FailingCloseWriter)
    backend = Backend([packet()])
    path = tmp_path / "close-failure.h5"
    with pytest.raises(RuntimeError, match="injected flush failure"):
        ScopeSumCapture(backend, 0).run(path, max_packets=1)
    assert backend.events[-2:] == ["stop", "close"]
    with h5py.File(path, "r") as source:
        assert not source.attrs["complete"]


def test_recorder_open_failure_is_joined_and_existing_file_is_preserved(tmp_path):
    path = tmp_path / "existing.h5"
    path.write_bytes(b"original measurement")
    threads = {t.ident for t in threading.enumerate()}
    with pytest.raises(RuntimeError, match="HDF5 recording"):
        ScopeSumRecorder(path, {})
    assert path.read_bytes() == b"original measurement"
    assert {t.ident for t in threading.enumerate()} == threads


def test_capture_replaces_existing_recording_only_when_overwrite_is_explicit(tmp_path):
    path = tmp_path / "existing.h5"
    old = ScopeSumWriter(path, {"channel": 9})
    old.append(decode_sum_packet(packet(sequence=99)))
    old.close(complete=True)
    original = path.read_bytes()
    backend = Backend([packet()])
    with pytest.raises(RuntimeError, match="HDF5 recording"):
        ScopeSumCapture(backend, 0).run(path, max_packets=1)
    assert path.read_bytes() == original
    assert "read" not in backend.events
    assert backend.events[-2:] == ["stop", "close"]

    backend = Backend([packet()])
    result = ScopeSumCapture(backend, 0).run(path, max_packets=1, overwrite=True)
    assert result["complete"]
    assert [p.raw for p in read_sum_capture(path)] == [packet()]
    assert backend.events[-2:] == ["stop", "close"]
    with h5py.File(path, "r") as source:
        assert source.attrs["complete"]
        assert json.loads(source.attrs["metadata_json"])["channel"] == 0


def test_capture_setup_failure_does_not_replace_file_even_with_overwrite(tmp_path):
    class FailingBackend(Backend):
        def prepare_dma_capture(self):
            raise RuntimeError("injected prepare failure")

    path = tmp_path / "existing.h5"
    path.write_bytes(b"original measurement")
    with pytest.raises(RuntimeError, match="injected prepare failure"):
        ScopeSumCapture(FailingBackend([]), 0).run(path, overwrite=True)
    assert path.read_bytes() == b"original measurement"


def test_flush_failure_is_reported_and_completion_is_retracted(tmp_path, monkeypatch):
    class FailingFlushWriter(ScopeSumWriter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            flush = self._file.flush
            self.calls = 0

            def failing_flush():
                self.calls += 1
                # Fail the completion-marker flush after the data flush succeeded.
                if self.calls == 2:
                    raise OSError("injected completion flush failure")
                flush()

            self._file.flush = failing_flush

    monkeypatch.setattr(scope_sums, "ScopeSumWriter", FailingFlushWriter)
    backend = Backend([packet()])
    path = tmp_path / "flush-failure.h5"
    with pytest.raises(RuntimeError, match="injected completion flush failure"):
        ScopeSumCapture(backend, 0).run(path, max_packets=1)
    assert backend.events[-2:] == ["stop", "close"]
    assert len(list(read_sum_capture(path))) == 1
    with h5py.File(path, "r") as source:
        assert not source.attrs["complete"]
        assert "injected completion flush failure" in source.attrs["cleanup_error"]


def test_full_configuration_roundtrips_through_async_capture_before_first_read(tmp_path):
    configuration = "format_version: 5\nhardware:\n  channels:\n    '0': {scope: {edge_mode: 5}}\n"
    backend = Backend([packet()])
    capture = ScopeSumCapture(backend, 0, configuration_yaml=configuration)
    path = tmp_path / "full-config.h5"
    assert capture.run(path, max_packets=1)["complete"]
    with h5py.File(path, "r") as source:
        assert source["configuration_yaml"][()].decode("utf-8") == configuration
        metadata = json.loads(source.attrs["metadata_json"])
        assert metadata["configuration_snapshot"] == "before acquisition"
        assert "configuration_yaml" not in metadata
    assert list(read_sum_capture(path)) == [decode_sum_packet(packet())]
