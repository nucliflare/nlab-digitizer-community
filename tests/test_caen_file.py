from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from nlab.analysis.caen_file import (
    CAEN_LEGACY_HEADER,
    inspect_caen_file,
    iter_caen_psd_batches,
)
from nlab.analysis.psd_file import inspect_psd_event_file, iter_psd_event_batches
from nlab.analysis.waveform_file import MappedWaveformFile, build_waveform_file_index
from nlab.hardware.digitizer.dma import (
    FILE_HEADER_STRUCT,
    FILE_MAGIC,
    FILE_VERSION,
)


def _native_event(
    *,
    options: int,
    timestamp: int,
    long_gate: int = 0,
    short_gate: int = 0,
    samples: tuple[int, ...] = (),
    board: int = 0,
    channel: int = 0,
    flags: int = 0x4000,
) -> bytes:
    event = bytearray(struct.pack("<HHQ", board, channel, timestamp))
    if options & 0x1:
        event.extend(struct.pack("<H", long_gate))
    if options & 0x2:
        event.extend(struct.pack("<d", float(long_gate)))
    if options & 0x4:
        event.extend(struct.pack("<HI", short_gate, flags))
    if options & 0x8:
        event.extend(struct.pack("<Bi", 1, len(samples)))
        event.extend(np.asarray(samples, dtype="<i2").tobytes())
    return bytes(event)


def _native_file(path: Path, options: int, events: list[bytes]) -> None:
    path.write_bytes(struct.pack("<H", 0xCAE0 | options) + b"".join(events))


def _extracted_event(
    *,
    timestamp: int,
    long_gate: int,
    short_gate: int,
    board: int = 0,
    channel: int = 0,
    flags: int = 0x4040,
    reserved: int = 0,
) -> bytes:
    return struct.pack(
        "<HHQHHII",
        board,
        channel,
        timestamp,
        long_gate,
        short_gate,
        flags,
        reserved,
    )


def test_native_caen_waveforms_feed_psd_without_materializing_samples(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compass.BIN"
    _native_file(
        path,
        0xD,
        [
            _native_event(
                options=0xD,
                timestamp=10,
                long_gate=100,
                short_gate=25,
                samples=(1, 2, 3),
            ),
            _native_event(
                options=0xD,
                timestamp=20,
                long_gate=200,
                short_gate=80,
                samples=(-4, 5),
            ),
        ],
    )

    info = inspect_psd_event_file(path)
    batches = list(iter_psd_event_batches(path))
    waveform_index = build_waveform_file_index(path)
    reader = MappedWaveformFile(waveform_index)
    try:
        waveform_batch = reader.batch(0, 2, sample_count=3)
    finally:
        reader.close()

    assert info.format_name == "CAEN CoMPASS binary"
    assert info.channel == 0
    assert info.total_events == 0  # Variable records are counted during processing.
    assert [len(batch) for batch in batches] == [2]
    np.testing.assert_array_equal(batches[0]["timestamp"], [10, 20])
    np.testing.assert_array_equal(batches[0]["long_gate"], [100, 200])
    np.testing.assert_array_equal(batches[0]["short_gate"], [25, 80])
    np.testing.assert_array_equal(waveform_batch.complete, [True, False])
    np.testing.assert_array_equal(waveform_batch.samples[0], [1, 2, 3])
    np.testing.assert_array_equal(waveform_batch.long_gate, [100, 200])
    np.testing.assert_array_equal(waveform_batch.short_gate, [25, 80])


def test_legacy_caen_file_is_detected_as_fixed_records(tmp_path: Path) -> None:
    path = tmp_path / "coincidences.BIN"
    events = b"".join(
        struct.pack("<HHQHHI", 0, 6, timestamp, energy, 0, 0x4000)
        for timestamp, energy in ((1, 123), (2, 456))
    )
    path.write_bytes(struct.pack("<H", CAEN_LEGACY_HEADER) + events)

    info = inspect_caen_file(path)
    batches = list(iter_caen_psd_batches(path, target_batch_bytes=20))

    assert info.channel == 6
    assert info.total_events == 2
    assert [len(batch) for batch in batches] == [1, 1]
    assert batches[1]["long_gate"][0] == 456
    assert batches[1]["short_gate"][0] == 0


def test_caen_without_short_energy_is_rejected_for_psd(tmp_path: Path) -> None:
    path = tmp_path / "energy-only.bin"
    _native_file(
        path,
        0x1,
        [_native_event(options=0x1, timestamp=1, long_gate=100)],
    )

    with pytest.raises(ValueError, match="Energy Short"):
        inspect_psd_event_file(path)


def test_caen_raw_and_calibrated_energy_layout_uses_raw_gate(tmp_path: Path) -> None:
    path = tmp_path / "calibrated.bin"
    _native_file(
        path,
        0x7,
        [
            _native_event(
                options=0x7,
                timestamp=7,
                long_gate=900,
                short_gate=300,
            )
        ],
    )

    info = inspect_psd_event_file(path)
    batch = next(iter_psd_event_batches(path))

    assert info.total_events == 1
    assert batch["long_gate"][0] == 900
    assert batch["short_gate"][0] == 300


def test_headerless_extracted_caen_channel_feeds_psd(tmp_path: Path) -> None:
    path = tmp_path / "Data_CH0@DT5730_666.bin"
    path.write_bytes(
        b"".join(
            [
                _extracted_event(timestamp=10, long_gate=452, short_gate=28),
                _extracted_event(timestamp=20, long_gate=534, short_gate=30),
                _extracted_event(timestamp=30, long_gate=614, short_gate=35),
            ]
        )
    )

    info = inspect_psd_event_file(path)
    batches = list(iter_psd_event_batches(path))

    assert info.format_name == "CAEN extracted channel binary"
    assert info.channel == 0
    assert info.total_events == 3
    assert [len(batch) for batch in batches] == [3]
    np.testing.assert_array_equal(batches[0]["timestamp"], [10, 20, 30])
    np.testing.assert_array_equal(batches[0]["long_gate"], [452, 534, 614])
    np.testing.assert_array_equal(batches[0]["short_gate"], [28, 30, 35])


def test_extracted_caen_detection_rejects_arbitrary_aligned_binary(tmp_path: Path) -> None:
    path = tmp_path / "not-caen.bin"
    path.write_bytes(bytes(48))

    with pytest.raises(ValueError, match="Unsupported CAEN binary header"):
        inspect_psd_event_file(path)


def test_extracted_caen_iterator_validates_every_reserved_word(tmp_path: Path) -> None:
    path = tmp_path / "corrupt-extraction.bin"
    records = [
        _extracted_event(
            timestamp=index + 1,
            long_gate=100,
            short_gate=20,
            reserved=1 if index == 1 else 0,
        )
        for index in range(200)
    ]
    path.write_bytes(b"".join(records))

    info = inspect_psd_event_file(path)

    assert info.total_events == 200
    with pytest.raises(ValueError, match="nonzero reserved field"):
        list(iter_psd_event_batches(path))


def test_caen_waveform_index_supports_variable_sample_counts(tmp_path: Path) -> None:
    path = tmp_path / "waveforms.bin"
    first_samples = (1, -2, 3)
    second_samples = (10, 20, 30, 40, 50)
    _native_file(
        path,
        0xD,
        [
            _native_event(
                options=0xD,
                timestamp=100,
                long_gate=300,
                short_gate=75,
                samples=first_samples,
            ),
            _native_event(
                options=0xD,
                timestamp=200,
                long_gate=500,
                short_gate=125,
                samples=second_samples,
            ),
        ],
    )

    index = build_waveform_file_index(path)
    reader = MappedWaveformFile(index)
    try:
        first = reader.frame(0)
        second = reader.frame(1)
    finally:
        reader.close()

    assert index.frame_count == 2
    assert index.sample_period_ns is None
    np.testing.assert_array_equal(first.samples, first_samples)
    np.testing.assert_array_equal(second.samples, second_samples)
    assert second.timestamp == 200
    assert second.long_gate == 500
    assert second.short_gate == 125


def test_caen_waveform_index_groups_interleaved_board_channels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "multi-channel.bin"
    _native_file(
        path,
        0xD,
        [
            _native_event(
                options=0xD,
                timestamp=10,
                samples=(20, 21),
                board=0,
                channel=2,
            ),
            _native_event(
                options=0xD,
                timestamp=11,
                samples=(0, 1),
                board=0,
                channel=0,
            ),
            _native_event(
                options=0xD,
                timestamp=12,
                samples=(10, 11),
                board=1,
                channel=0,
            ),
            _native_event(
                options=0xD,
                timestamp=13,
                samples=(22, 23),
                board=0,
                channel=2,
            ),
        ],
    )

    index = build_waveform_file_index(path)
    reader = MappedWaveformFile(index)
    try:
        board_zero_channel_two = reader.frame(1, source_index=1)
        board_one_channel_zero = reader.frame(0, source_index=2)
    finally:
        reader.close()

    assert index.frame_count == 4
    assert [
        (source.board, source.channel, source.frame_count)
        for source in index.channels
    ] == [(0, 0, 1), (0, 2, 2), (1, 0, 1)]
    assert board_zero_channel_two.timestamp == 13
    np.testing.assert_array_equal(board_zero_channel_two.samples, [22, 23])
    assert board_one_channel_zero.board == 1
    assert board_one_channel_zero.channel == 0


def test_truncated_caen_waveform_tail_salvages_complete_events(tmp_path: Path) -> None:
    path = tmp_path / "truncated.bin"
    first = _native_event(
        options=0xD,
        timestamp=1,
        long_gate=100,
        short_gate=20,
        samples=(1, 2, 3),
    )
    incomplete = _native_event(
        options=0xD,
        timestamp=2,
        long_gate=200,
        short_gate=40,
        samples=(4, 5, 6, 7),
    )[:-3]
    _native_file(path, 0xD, [first, incomplete])

    batches = list(iter_psd_event_batches(path))
    index = build_waveform_file_index(path)

    assert sum(map(len, batches)) == 1
    assert index.frame_count == 1


def test_unknown_binary_signature_is_not_treated_as_ndma_or_caen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unknown.bin"
    path.write_bytes(b"NOPE" + bytes(32))

    with pytest.raises(ValueError, match="Unsupported CAEN binary header"):
        inspect_psd_event_file(path)


def test_ndma_scope_frames_use_direct_mmap_offsets(tmp_path: Path) -> None:
    path = tmp_path / "scope.bin"
    frame_samples = 8  # Four timestamp words and four waveform samples.
    header = FILE_HEADER_STRUCT.pack(
        FILE_MAGIC,
        FILE_VERSION,
        1,
        0,
        0.0,
        frame_samples,
    )
    frames = b"".join(
        struct.pack("<Q", timestamp) + np.asarray(samples, dtype="<i2").tobytes()
        for timestamp, samples in (
            (11, (1, 2, 3, 4)),
            (22, (-1, -2, -3, -4)),
            (33, (5, 6, 7, 8)),
        )
    )
    path.write_bytes(header + frames)

    index = build_waveform_file_index(path)
    reader = MappedWaveformFile(index)
    try:
        second = reader.frame(1)
        batch = reader.batch(0, 3, sample_count=4)
        np.testing.assert_array_equal(batch.complete, [True, True, True])
        np.testing.assert_array_equal(
            batch.samples,
            [[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]],
        )
        del batch
        strided = reader.batch(0, 2, sample_count=4, stride=2)
        np.testing.assert_array_equal(
            strided.samples,
            [[1, 2, 3, 4], [5, 6, 7, 8]],
        )
        del strided
    finally:
        reader.close()

    assert index.format_name == "NLab scope NDMA"
    assert index.frame_count == 3
    assert index.sample_period_ns == 8.0
    assert second.timestamp == 22
    np.testing.assert_array_equal(second.samples, [-1, -2, -3, -4])
