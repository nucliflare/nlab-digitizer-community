from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np
import pytest

from nlab.hardware.digitizer.dma import FILE_HEADER_STRUCT, FILE_MAGIC, FILE_VERSION
from notebooks import check_dma


def test_sample_fill_factor_uses_total_elapsed_time() -> None:
    # Two 16 ns stored waveforms cover 120 ns between three trigger stamps.
    # Averaging the two per-gap percentages instead would incorrectly give 30%.
    timestamps = np.array([100, 105, 115], dtype=np.int64)

    assert check_dma._sample_fill_factor(timestamps, 8) == pytest.approx(32 / 120)


@pytest.mark.parametrize(
    "timestamps",
    [
        np.array([100], dtype=np.int64),
        np.array([100, 100], dtype=np.int64),
        np.array([110, 100], dtype=np.int64),
        np.array([110, 100], dtype=np.uint64),
    ],
)
def test_sample_fill_factor_requires_ordered_pair(timestamps: np.ndarray) -> None:
    assert check_dma._sample_fill_factor(timestamps, 8) is None


def test_scope_dma_report_labels_nominal_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "periodic.bin"
    samples = np.arange(1, 9, dtype="<i2").tobytes()
    header = FILE_HEADER_STRUCT.pack(FILE_MAGIC, FILE_VERSION, 0, 0, time.time(), 12)
    records = b"".join(struct.pack("<Q", stamp) + samples for stamp in (100, 105, 115))
    path.write_bytes(header + records)
    monkeypatch.setattr(check_dma, "SHOW_PLOTS", False)
    monkeypatch.setattr(check_dma, "PREVIEW_FRAMES", 2)
    monkeypatch.setattr(check_dma, "TRIGGER_MODE_LABEL", "periodic")

    check_dma.inspect_scope_dma(path)

    output = capsys.readouterr().out
    assert "Trigger mode label:    periodic" in output
    assert "Stored waveform duration:    0.016000 us/frame" in output
    assert "Mean valid-frame interval:   0.060000 us" in output
    assert "Sample fill factor:          26.667%" in output


def test_scope_dma_report_excludes_invalid_waveform_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "with_invalid_frame.bin"
    samples = np.arange(1, 9, dtype="<i2").tobytes()
    header = FILE_HEADER_STRUCT.pack(FILE_MAGIC, FILE_VERSION, 0, 0, time.time(), 12)
    records = b"".join(
        struct.pack("<Q", stamp) + waveform
        for stamp, waveform in (
            (100, samples),
            (105, bytes(len(samples))),
            (115, samples),
        )
    )
    path.write_bytes(header + records)
    monkeypatch.setattr(check_dma, "SHOW_PLOTS", False)
    monkeypatch.setattr(check_dma, "PREVIEW_FRAMES", 1)

    check_dma.inspect_scope_dma(path)

    output = capsys.readouterr().out
    assert "Invalid frames (combined):   1" in output
    assert "Valid frames:                2" in output
    assert "Sample fill factor:          13.333%" in output
