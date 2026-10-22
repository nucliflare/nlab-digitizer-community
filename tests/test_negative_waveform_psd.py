from pathlib import Path
from runpy import run_path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nlab.analysis.waveform_psd import WaveformPsdSettings, integrate_waveform
from nlab.hardware.digitizer.dma import FILE_HEADER_STRUCT, FILE_MAGIC

SCRIPT = Path(__file__).resolve().parents[1] / "notebooks" / "inspect_negative_waveform_psd.py"


def test_signed_waveform_tail_produces_negative_psd_and_plots(tmp_path: Path) -> None:
    script = run_path(str(SCRIPT))
    data = np.zeros(3, dtype=[("timestamp", "<u8"), ("samples", "<i2", (64,))])
    data["timestamp"] = np.arange(3, dtype=np.uint64) + 2**63
    data["samples"] = 1000
    data["samples"][:2, 10] = 900
    data["samples"][:2, 11] = 950
    data["samples"][0, 12:18] = 1010  # opposite-sign tail area: -60
    data["samples"][1, 12:18] = 990   # same-sign tail area: +60
    path = tmp_path / "waveforms.bin"
    original = FILE_HEADER_STRUCT.pack(FILE_MAGIC, 1, 0, 0, 0.0, 68) + data.tobytes()
    path.write_bytes(original)

    analysis = script["reconstruct"](
        path, baseline=(0, 8), search=(8, 16), short_samples=2, long_samples=(8, 16),
    )
    result = analysis["charges"]["median", 8]
    np.testing.assert_array_equal(analysis["onset"][:2], [10, 10])
    np.testing.assert_array_equal(result["short"], [150, 150, 0])
    np.testing.assert_array_equal(result["long"], [90, 210, 0])
    np.testing.assert_allclose(result["psd"], [-2/3, 2/7, np.nan], equal_nan=True)
    assert not analysis["triggered"][2]
    # Cross-check the vectorized sums against the existing workbench integrator.
    expected = integrate_waveform(data["samples"][0], WaveformPsdSettings(
        baseline_start=0, baseline_end=8, gate_start=10, short_end=12,
        long_end=18, polarity=-1,
    ))
    assert expected is not None
    assert result["psd"][0] == pytest.approx(expected.ratio)
    assert all(row["negative"] == 1 and row["valid"] == 2
               for row in script["summarize"](analysis))
    fig, frames = script["show_negative_pulses"](analysis, frames=[0], long_samples=8)
    try:
        fig.canvas.draw()
        assert frames.tolist() == [0]
    finally:
        plt.close(fig)
    assert path.read_bytes() == original
