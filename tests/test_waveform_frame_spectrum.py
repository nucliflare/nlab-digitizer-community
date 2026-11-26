"""Noise and mean-pulse spectra of recorded frames, and the workbench Spectrum tab."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from nlab.analysis.waveform_file import build_waveform_file_index
from nlab.analysis.waveform_frame_spectrum import (
    FrameSpectrumAccumulator,
    FrameSpectrumRequest,
    FrameSpectrumResult,
    SpectrumRegion,
)
from nlab.analysis.waveform_spectrum import find_spurs, line_threshold_db, parseval_check
from nlab.hardware.digitizer.dma import FILE_HEADER_STRUCT, FILE_MAGIC
from nlab.help.context import resolve_context
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog
from nlab.workers.waveform_spectrum_worker import WaveformSpectrumWorker

PERIOD_NS = 2.0
FS = 1e9 / PERIOD_NS
FRAME = 1024
BASELINE = 128


def _pulse(n: int = FRAME, amplitude: float = 800.0) -> np.ndarray:
    """A negative pulse starting after the baseline: fast rise, 80 ns decay."""
    t = np.arange(n) - 160.0
    shape = np.where(t >= 0, np.exp(-t / 40.0) - np.exp(-t / 4.0), 0.0)
    return -amplitude * shape / shape.max()


def _frames(
    count: int, *, sigma: float = 3.0, tone_mhz: float = 0.0, seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = 1000 + _pulse() + rng.normal(0, sigma, (count, FRAME))
    if tone_mhz:
        t = np.arange(FRAME) * PERIOD_NS * 1e-9
        phases = rng.uniform(0, 2 * np.pi, (count, 1))
        frames += 5.0 * np.cos(2 * np.pi * tone_mhz * 1e6 * t + phases)
    return frames


def _request(region: str = "pretrigger", **changes: Any) -> FrameSpectrumRequest:
    request = FrameSpectrumRequest(
        sample_period_ns=PERIOD_NS, frame_samples=FRAME, region=SpectrumRegion(region),
        baseline_start=0, baseline_end=BASELINE, polarity=-1, baseline_method="median",
    )
    return replace(request, **changes)  # replace() validates again.


def _result(frames: np.ndarray, region: str = "pretrigger") -> FrameSpectrumResult:
    accumulator = FrameSpectrumAccumulator(_request(region))
    accumulator.add(frames)
    return accumulator.result("test")


def test_pretrigger_noise_level_excludes_the_pulse() -> None:
    sigma = 3.0
    frames = _frames(400, sigma=sigma)
    pretrigger = _result(frames).noise
    assert pretrigger is not None and pretrigger.n == BASELINE
    assert pretrigger.psd[2:-1].mean() == pytest.approx(2 * sigma**2 / FS, rel=0.06)
    assert parseval_check(pretrigger) == pytest.approx(1, abs=0.05)
    whole = _result(frames, "frame").noise
    assert whole is not None and whole.n == FRAME
    # The whole frame carries the pulse: far more low-frequency power than noise.
    assert whole.psd[1:6].mean() > 100 * pretrigger.psd[1:3].mean()


def test_mean_pulse_spectrum_follows_the_known_pulse_shape() -> None:
    result = _result(_frames(300, sigma=3.0))
    assert result.pulse_frequencies is not None and result.pulse_db is not None
    expected = np.abs(np.fft.rfft(-_pulse())) ** 2  # Polarity-normalised: positive.
    expected_db = 10 * np.log10(expected / expected.max())
    strong = expected_db > -30
    assert result.pulse_db.max() == pytest.approx(0.0)
    np.testing.assert_allclose(result.pulse_db[strong], expected_db[strong], atol=0.3)
    np.testing.assert_allclose(result.pulse_frequencies[1], FS / FRAME)


def test_a_line_in_the_baseline_noise_is_found() -> None:
    result = _result(_frames(200, sigma=2.0, tone_mhz=50.0))
    noise = result.noise
    assert noise is not None
    spurs = find_spurs(noise, threshold_db_over_median=line_threshold_db(noise.averages))
    assert spurs and spurs[0].frequency == pytest.approx(50e6, abs=noise.delta_f)


def test_short_and_nonfinite_frames_are_skipped_and_counted() -> None:
    frames = _frames(5)
    frames[2, 10] = np.nan
    accumulator = FrameSpectrumAccumulator(_request())
    accumulator.add(frames, complete=np.array([True, False, True, True, True]))
    accumulator.add(frames[:2, :500])  # Too short for the frame.
    assert (accumulator.frames, accumulator.skipped) == (3, 4)
    assert FrameSpectrumAccumulator(_request()).result().message.startswith("No frame had")


@pytest.mark.parametrize("changes,match", [
    ({"baseline_end": 8}, "has 8 samples"),
    ({"baseline_end": FRAME + 4}, "inside the frame"),
    ({"sample_period_ns": 0.0}, "sample period"),
    ({"polarity": 0}, "polarity"),
])
def test_invalid_requests_are_rejected(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _request(**changes)


def test_messages_name_the_coarse_resolution_and_the_pulse_content() -> None:
    frames = _frames(3)
    assert "Δf = 3.906 MHz" in _result(frames).message
    assert "pulse plus noise" in _result(frames, "frame").message


def _capture(path: Path, frames: np.ndarray) -> None:
    samples = np.round(frames).astype("<i2")
    records = np.zeros(len(samples), dtype=[("time", "<u8"), ("samples", "<i2", (FRAME,))])
    records["time"] = np.arange(len(samples), dtype=np.uint64) + 2**63
    records["samples"] = samples
    path.write_bytes(FILE_HEADER_STRUCT.pack(FILE_MAGIC, 1, 0, 0, 0.0, FRAME + 4)
                     + records.tobytes())


def _run(worker: WaveformSpectrumWorker) -> FrameSpectrumResult:
    results: list[FrameSpectrumResult] = []
    failures: list[str] = []
    worker.computed.connect(lambda _generation, result: results.append(result))
    worker.failed.connect(lambda _generation, message: failures.append(message))
    worker.run()  # Synchronously, in this thread.
    assert not failures and len(results) == 1
    return results[0]


def test_worker_reads_strided_frames_and_a_selected_population(tmp_path: Path) -> None:
    path = tmp_path / "frames.bin"
    _capture(path, _frames(30, tone_mhz=25.0))
    index = build_waveform_file_index(path)
    strided = _run(WaveformSpectrumWorker(
        1, index, source_index=0, request=_request(), stride=2, maximum_events=10,
    ))
    assert strided.frames == 10 and strided.skipped == 0
    selected = _run(WaveformSpectrumWorker(
        2, index, source_index=0, request=_request(), frames=np.array([3, 5, 7]),
        population="Selected",
    ))
    assert selected.frames == 3 and selected.population == "Selected"
    assert selected.noise is not None and selected.noise.n == BASELINE


def test_workbench_spectrum_tab_computes_and_explains(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "frames.bin"
    _capture(path, _frames(40, tone_mhz=25.0))
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    dialog.auto_recalculate.setChecked(False)
    panel = dialog.spectrum_panel
    assert dialog.analysis_tabs.tabText(1) == "Spectrum"
    panel.compute()
    assert panel.stack.currentIndex() == 1 and "Load a waveform file" in panel.message.text()

    dialog.open_path(path)
    qtbot.waitUntil(lambda: dialog._index_thread is None and dialog._frame is not None,
                    timeout=5000)
    dialog.polarity_combo.setCurrentIndex(dialog.polarity_combo.findData(-1))
    dialog._updating_gates = True
    dialog.baseline_region.setRegion((0, BASELINE * PERIOD_NS))
    dialog._updating_gates = False
    panel.population_combo.setCurrentIndex(panel.population_combo.findData("selected"))
    panel.compute()
    assert "Select a population" in panel.message.text()

    panel.population_combo.setCurrentIndex(panel.population_combo.findData("all"))
    panel.compute()
    qtbot.waitUntil(lambda: panel.result is not None, timeout=10000)
    assert panel.stack.currentIndex() == 0
    assert panel.result is not None and panel.result.frames == 40
    x, _ = panel.noise_curve.getData()
    assert x is not None and x.size == BASELINE // 2  # Without DC; Nyquist kept.
    assert panel.pulse_curve.getData()[0] is not None
    assert panel._plot.getAxis("right").isVisible() or not panel.isVisible()
    text = panel.summary.text()
    for part in ("40 frames used", "Pretrigger samples 0–127", "Δf = 3.906 MHz",
                 "energy spectrum", "not validated on live hardware"):
        assert part in text
    marked, _ = panel.markers.getData()
    assert marked is not None and np.any(np.abs(marked - 25e6) < 4e6)
    panel.log_checkbox.setChecked(True)
    log_x, _ = panel.pulse_curve.getData()
    assert log_x is not None and log_x.max() == pytest.approx(math.log10(FS / 2), abs=0.01)
    assert resolve_context(panel.region_combo).topic == "waveform-spectrum"
    dialog.close_file()
    assert panel.result is None
