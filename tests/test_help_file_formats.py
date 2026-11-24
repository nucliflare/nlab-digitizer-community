"""The binary-format help topics describe the files the application writes.

Each topic's example reader and usage code run unchanged against files written
by the application's own writers, and must agree with its readers.
"""

from __future__ import annotations

import gc
import re
import shutil
import struct
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nlab.hardware.digitizer.dma import IIOMcaDmaStreamer, IIOScopeDmaStreamer
from nlab.hardware.digitizer.iio_listmode import (
    IIO_LM_EVENT_DTYPE,
    VDPP_ZC_CALC_SCHEMA,
    cfd_time_q,
)
from nlab.hardware.digitizer.scope_sums import (
    ScopeSumRecorder,
    decode_sum_packet,
    read_sum_capture,
)
from nlab.help.content import load_topics

_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
_IP123_MAGIC = 0x31534D5553504456
_IP124_MAGIC = 0x32534D5553504456


@pytest.fixture
def run_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[str], dict[str, Any]]]:
    """Run a topic's reader, then its usage example in the capture folder."""
    namespaces: list[dict[str, Any]] = []

    def run(topic: str) -> dict[str, Any]:
        reader, usage = _PYTHON_BLOCK.findall(load_topics()[topic].markdown)
        namespace: dict[str, Any] = {"__name__": f"help_{topic}"}
        namespaces.append(namespace)
        exec(compile(reader, f"<help {topic} reader>", "exec"), namespace)
        monkeypatch.chdir(tmp_path)
        exec(compile(usage, f"<help {topic} example>", "exec"), namespace)
        return namespace

    yield run
    # The examples memory-map captures; release them so Windows can delete.
    for namespace in namespaces:
        namespace.clear()
    gc.collect()


# ---- Scope waveforms -------------------------------------------------------


class _ScopeFrames:
    """Scope DMA backend delivering known v121 raw frames to the real writer."""

    def __init__(self, frame_samples: int, frames: list[bytes]) -> None:
        self._frame_samples = frame_samples
        self._frames = list(frames)

    def prepare_dma_capture(self) -> None:
        pass

    def get_frame_samples(self) -> int:
        return self._frame_samples

    def read_dma_raw_frame(self) -> bytes:
        return self._frames.pop(0)

    def request_dma_stop(self) -> None:
        pass

    def close_dma_capture(self, on_frame: object = None) -> None:
        pass


def test_scope_waveform_example_reads_a_recorded_capture(
    tmp_path: Path, run_example: Callable[[str], dict[str, Any]],
) -> None:
    frame_samples = 64
    ticks = np.array([2**60, 2**60 + 125, 2**60 + 1000], dtype=np.uint64)
    waves = (np.arange(3 * 60, dtype=np.int64).reshape(3, 60) * 377 % 65536
             - 32768).astype(np.int16)
    frames = [struct.pack("<Q", int(t)) + w.astype("<i2").tobytes()
              for t, w in zip(ticks, waves, strict=True)]
    path = tmp_path / "scope.bin"
    streamer = IIOScopeDmaStreamer(_ScopeFrames(frame_samples, frames), channel=1)
    assert streamer.stream_to_file(path, threading.Event(), n_frames=3) == 3

    example = run_example("format-waveforms")

    assert example["header"]["channel"] == 1
    assert example["header"]["frame_samples"] == frame_samples
    np.testing.assert_array_equal(example["ticks"], ticks)
    np.testing.assert_array_equal(example["waves"], waves)
    np.testing.assert_array_equal(example["spacing_ns"], [1000, 7000])
    assert example["t_ns"][-1] == 118.0

    # A cut-short copy keeps every whole frame and ignores the partial one.
    shutil.copyfile(path, tmp_path / "cut.bin")
    with (tmp_path / "cut.bin").open("ab") as handle:
        handle.write(b"\x01" * 50)
    _, cut_ticks, cut_waves = example["read_scope_ndma"](tmp_path / "cut.bin")
    np.testing.assert_array_equal(cut_ticks, ticks)
    np.testing.assert_array_equal(cut_waves, waves)


# ---- MCA list mode ---------------------------------------------------------


def _events(first_tick: int) -> np.ndarray:
    events = np.zeros(1024, dtype=IIO_LM_EVENT_DTYPE)
    index = np.arange(1024)
    # CFD, PSD-precedence, untimed and out-of-range-fraction records.
    events["marker"] = np.array([0x02, 0x0A, 0x00, 0x02, 0x82])[index % 5]
    events["zc_offset"] = index % 256
    events["zc_estimation"] = np.where(index % 5 == 3, 100, -((index * 16) % 16385))
    events["charge_energy"] = 100 + index
    events["trapezoid_energy"] = 4000 + 7 * index
    events["timestamp"] = first_tick + 50 * index.astype(np.uint64)
    return events


class _ListModeFrames:
    """MCA list-mode backend delivering a START frame and a padded tail."""

    def __init__(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event
        self.first = _events(2**40)
        self.first[0] = np.zeros(1, dtype=IIO_LM_EVENT_DTYPE)[0]
        self.first["timestamp"][0] = 1
        self.tail = _events(2**40 + 60_000)
        self.tail[700:] = np.zeros(324, dtype=IIO_LM_EVENT_DTYPE)

    def start_mca_dma_capture(self, on_started: Callable[[], None] | None = None) -> np.ndarray:
        if on_started is not None:
            on_started()
        self._stop_event.set()
        return self.first

    def read_mca_dma_frame(self) -> np.ndarray:
        raise AssertionError("refill after stop")

    def mca_dma_measurement_in_progress(self) -> bool:
        return False

    def request_mca_dma_stop(self) -> None:
        pass

    def close_mca_dma_capture(
        self, on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> int:
        if on_frame is not None:
            on_frame(self.tail)
        return 1

    def get_mca_dma_capture_diagnostics(self) -> tuple[int, int, int, int]:
        return (0, 0, 2, 0)


def test_listmode_example_reads_a_recorded_capture(
    tmp_path: Path, run_example: Callable[[str], dict[str, Any]],
) -> None:
    stop_event = threading.Event()
    backend = _ListModeFrames(stop_event)
    IIOMcaDmaStreamer(backend, channel=1).stream_events(
        stop_event, tmp_path / "run.bin", client_record_schema=VDPP_ZC_CALC_SCHEMA,
    )

    example = run_example("format-listmode")

    assert example["RECORD"] == IIO_LM_EVENT_DTYPE
    assert example["check"]["client_record_schema"] == VDPP_ZC_CALC_SCHEMA
    assert example["check"]["continuity_valid"] is True
    assert example["header"]["channel"] == 1
    written = np.concatenate([backend.first, backend.tail])
    np.testing.assert_array_equal(np.asarray(example["records"]), written)
    expected = np.concatenate([backend.first[1:], backend.tail[:700]])
    events = example["events"]
    np.testing.assert_array_equal(events, expected)
    assert example["spectrum"].sum() == len(expected)
    np.testing.assert_array_equal(
        np.flatnonzero(example["spectrum"]),
        np.unique(expected["trapezoid_energy"] >> 2),
    )
    np.testing.assert_allclose(
        example["ratio"], 1 - expected["charge_energy"] / expected["trapezoid_energy"],
    )

    first = int(expected["timestamp"][0])
    marker = expected["marker"]
    fine = expected["zc_estimation"]
    timed = ((marker & 0x08) == 0) & ((marker & 0x02) != 0) & (fine >= -16384) & (fine <= 0)
    reference = [
        (cfd_time_q(int(t), int(o), int(f)) - cfd_time_q(first, 0, 0)) / 8192
        for t, o, f in zip(
            expected["timestamp"][timed], expected["zc_offset"][timed], fine[timed],
            strict=True,
        )
    ]
    times = example["times"]
    np.testing.assert_array_equal(np.isnan(times), ~timed)
    np.testing.assert_allclose(times[timed], reference, rtol=0, atol=1e-9)
    # Two in five records are CFD-timed; the rest exercise each NaN rule.
    assert np.count_nonzero(timed) > 0.35 * len(expected)
    assert np.count_nonzero(~timed) > 0.55 * len(expected)


def test_listmode_example_decodes_the_driver_test_vector(
    tmp_path: Path, run_example: Callable[[str], dict[str, Any]],
) -> None:
    stop_event = threading.Event()
    IIOMcaDmaStreamer(_ListModeFrames(stop_event), channel=0).stream_events(
        stop_event, tmp_path / "run.bin", client_record_schema=VDPP_ZC_CALC_SCHEMA,
    )
    example = run_example("format-listmode")
    vector = "02 13 00 f0 34 12 78 56 64 00 00 00 00 00 00 00"
    assert vector in load_topics()["format-listmode"].markdown

    record = np.frombuffer(bytes.fromhex(vector), dtype=example["RECORD"])

    assert record[0].tolist() == (2, 19, -4096, 0x1234, 0x5678, 100)
    assert example["cfd_time_ns"](record, reference_ticks=0)[0] == 837.5
    assert cfd_time_q(100, 19, -4096) == 837.5 * 8192


def test_listmode_example_drops_records_before_a_late_start(
    tmp_path: Path, run_example: Callable[[str], dict[str, Any]],
) -> None:
    stop_event = threading.Event()
    IIOMcaDmaStreamer(_ListModeFrames(stop_event), channel=0).stream_events(
        stop_event, tmp_path / "run.bin", client_record_schema=VDPP_ZC_CALC_SCHEMA,
    )
    example = run_example("format-listmode")
    records = _events(5000)[:6]
    records[2] = np.zeros(1, dtype=IIO_LM_EVENT_DTYPE)[0]
    records["timestamp"][2] = 1

    kept = example["events_only"](records)

    np.testing.assert_array_equal(kept, records[3:])


# ---- Current periodic sums -------------------------------------------------


def _ip124(sequence: int, first: int, sums: np.ndarray, *, missed: int = 0,
           stop: bool = False) -> bytes:
    raw = bytearray(4096)
    struct.pack_into("<QQIHHII", raw, 0, first, _IP124_MAGIC, 256, 2, len(sums),
                     sequence, missed | (int(stop) << 31))
    words = np.frombuffer(raw, dtype="<u8", count=508, offset=32)
    words[: len(sums)] = sums
    words[len(sums):] = sums[-1]
    return bytes(raw)


def _ip123(sequence: int, first: int, sums: np.ndarray, *, stop: bool = False) -> bytes:
    raw = bytearray(4096)
    struct.pack_into("<QQIHHII", raw, 0, first, _IP123_MAGIC, 8, 0, len(sums),
                     sequence, int(stop) << 31)
    pairs = np.frombuffer(raw, dtype="<u8", count=2 * len(sums), offset=32)
    pairs[0::2] = first + 2 * np.arange(len(sums), dtype=np.uint64)
    pairs[1::2] = sums
    return bytes(raw)


def _record_sums(path: Path, raws: list[bytes], missed: int) -> list[Any]:
    packets = [decode_sum_packet(raw) for raw in raws]
    recorder = ScopeSumRecorder(path, {"channel": 0, "configuration_yaml": "k: 1"})
    for packet in packets:
        recorder.append(packet)
    recorder.close(complete=True, missed_windows=missed)
    return packets


def test_current_example_reads_an_ip124_recording(
    tmp_path: Path, run_example: Callable[[str], dict[str, Any]],
) -> None:
    interval = 256 // 4 + 2
    base = 256 * 32768
    sums = [base + np.arange(508, dtype=np.uint64) * 3,
            base - np.arange(508, dtype=np.uint64),
            base + np.arange(7, dtype=np.uint64)]
    first = 2**62
    second = first + 507 * interval + 4 * interval  # three windows missed
    third = second + 508 * interval
    raws = [_ip124(0, first, sums[0]),
            _ip124(1, second, sums[1], missed=3),
            _ip124(2, third, sums[2], missed=3, stop=True)]
    packets = _record_sums(tmp_path / "current.bin", raws, missed=3)

    example = run_example("format-current")

    capture = example["capture"]
    np.testing.assert_array_equal(
        capture["timestamps"], np.concatenate([p.timestamps for p in read_sum_capture(
            tmp_path / "current.bin")]),
    )
    np.testing.assert_array_equal(capture["sums"], np.concatenate(sums))
    np.testing.assert_array_equal(
        capture["timestamps"], np.concatenate([p.timestamps for p in packets]),
    )
    assert (capture["samples_per_window"], capture["gap_cycles"]) == (256, 2)
    assert capture["missed_windows"] == 3
    assert capture["header"]["format"] == "nlab-scope-sums"
    assert capture["header"]["configuration_yaml"] == "k: 1"
    assert capture["trailer"]["packets"] == 3 and example["complete"] is True
    assert example["lost"] == 3
    np.testing.assert_allclose(example["mean_code"], np.concatenate(sums) / 256 - 32768)
    assert example["t_s"][-1] == pytest.approx((third + 6 * interval - first) * 8e-9)


def test_current_example_reads_ip123_and_an_unclosed_recording(
    tmp_path: Path, run_example: Callable[[str], dict[str, Any]],
) -> None:
    sums = [8 * 32768 + np.arange(254, dtype=np.uint64), 8 * 32768 + np.arange(5, dtype=np.uint64)]
    raws = [_ip123(0, 1000, sums[0]), _ip123(1, 1000 + 2 * 254, sums[1], stop=True)]
    packets = _record_sums(tmp_path / "current.bin", raws, missed=0)
    example = run_example("format-current")
    read_nsum = example["read_nsum"]
    timestamps = np.concatenate([p.timestamps for p in packets])
    np.testing.assert_array_equal(example["capture"]["timestamps"], timestamps)
    np.testing.assert_array_equal(example["capture"]["sums"], np.concatenate(sums))
    assert example["lost"] == 0

    # Without its trailer, and with a partial block, the packets stay readable.
    data = (tmp_path / "current.bin").read_bytes()
    (tmp_path / "crashed.bin").write_bytes(data[:-4096] + b"\x07" * 100)
    crashed = read_nsum(tmp_path / "crashed.bin")
    assert crashed["trailer"] is None
    np.testing.assert_array_equal(crashed["timestamps"], timestamps)
    np.testing.assert_array_equal(crashed["sums"], np.concatenate(sums))
