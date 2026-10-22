"""Reconstruct signed gate integrals and show actual negative-PSD waveforms.

Jupyter: %run notebooks/inspect_negative_waveform_psd.py
The waveform file has no saved gates. The explicit choices below are an offline
diagnostic, NOT a reproduction of the separate hardware list-mode measurement.
Indices are samples, gates are half-open, and charges are ADC-count samples.
The reader assumes 8 ns/sample for this legacy format; that period is not stored
in its header, so plots deliberately use sample indices.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from nlab.analysis.waveform_file import MappedWaveformFile, build_waveform_file_index

FILENAME = Path(r"D:\collaborations\LS_EWT_PSD\Cs137_0cm_Pube_waveforms.bin")
OUTPUT = Path(__file__).resolve().parents[1] / "audit" / "negative_waveform_psd"
BASELINE = (0, 24)
SEARCH = (24, 40)
SHORT_SAMPLES = 5
LONG_SAMPLES = (32, 64, 128)


def reconstruct(path=FILENAME, baseline=BASELINE, search=SEARCH,
                short_samples=SHORT_SAMPLES, long_samples=LONG_SAMPLES):
    """Scan every complete waveform, with bounded input/scratch memory.

    Both baseline methods use the SAME onset, found in the search region using
    max(10% of the local peak, 5 baseline RMS). There is no hidden energy ROI,
    pile-up rejection, negative-area clipping, or smoothing.
    """
    if not 0 <= baseline[0] < baseline[1] <= search[0] < search[1]:
        raise ValueError("Baseline must precede the pulse search region")
    if not long_samples or not 0 < short_samples <= min(long_samples):
        raise ValueError("Gate widths must satisfy 0 < short <= long")
    index = build_waveform_file_index(Path(path))
    if len(index.channels) != 1:
        raise ValueError("Select a single-channel waveform capture")
    count = index.frame_count
    prefix = search[1] + max(long_samples)
    onset = np.zeros(count, dtype=np.int32)
    triggered = np.zeros(count, dtype=bool)
    rms = np.zeros(count)
    peak = np.zeros(count)
    baselines = {method: np.zeros(count) for method in ("median", "mean")}
    charges = {
        (method, width): {key: np.zeros(count) for key in ("short", "long", "psd")}
        for method in baselines for width in long_samples
    }
    mapped = MappedWaveformFile(index)
    try:
        for first in range(0, count, 8192):
            size = min(8192, count - first)
            batch = mapped.batch(first, size, sample_count=prefix)
            if not np.all(batch.complete):
                raise ValueError("Waveform is shorter than the requested gates")
            raw = batch.samples.astype(np.float64)
            region = slice(first, first + size)
            before = raw[:, baseline[0]:baseline[1]]
            bmedian = np.median(before, axis=1)
            brms = np.sqrt(np.mean((before - bmedian[:, None]) ** 2, axis=1))
            pulse = bmedian[:, None] - raw[:, search[0]:search[1]]
            height = pulse.max(axis=1)
            crossed = pulse >= np.maximum(0.1 * height, 5 * brms)[:, None]
            crossed &= (height > 0)[:, None]
            start = search[0] + crossed.argmax(axis=1)
            onset[region], triggered[region] = start, crossed.any(axis=1)
            rms[region], peak[region] = brms, height
            # Sum raw ADC samples first; subtract baseline with each gate width.
            sums = np.pad(np.cumsum(raw, axis=1), ((0, 0), (1, 0)))
            rows = np.arange(size)
            origin = sums[rows, start]
            for method, values in baselines.items():
                b = bmedian if method == "median" else np.mean(before, axis=1)
                values[region] = b
                qs = b * short_samples - (sums[rows, start + short_samples] - origin)
                for width in long_samples:
                    ql = b * width - (sums[rows, start + width] - origin)
                    ratio = np.full(size, np.nan)
                    np.divide(ql - qs, ql, out=ratio, where=crossed.any(axis=1) & (ql > 0))
                    result = charges[method, width]
                    result["short"][region], result["long"][region] = qs, ql
                    result["psd"][region] = ratio
    finally:
        mapped.close()
    return dict(index=index, onset=onset, triggered=triggered, rms=rms, peak=peak,
                baselines=baselines, charges=charges, baseline_region=baseline,
                short_samples=short_samples, long_samples=long_samples)


def summarize(analysis):
    rows = []
    for (method, width), result in analysis["charges"].items():
        valid = np.isfinite(result["psd"])
        negative = valid & (result["psd"] < 0)
        rows.append(dict(baseline=method, short_samples=analysis["short_samples"],
                         long_samples=width, valid=int(valid.sum()),
                         negative=int(negative.sum()),
                         negative_percent=100 * float(negative.sum()) / max(1, int(valid.sum()))))
    return rows


def show_negative_pulses(analysis, frames=None, long_samples=32, examples=3):
    """Show negative examples, preferring events negative under BOTH baselines."""
    median = analysis["charges"]["median", long_samples]
    mean = analysis["charges"]["mean", long_samples]
    candidates = np.flatnonzero(
        (median["psd"] < -0.02) & (mean["psd"] < 0)
        & (median["long"] > 1000) & (analysis["rms"] < 100)
    )
    if frames is None:
        frames = candidates[:examples]
    if not len(frames):
        raise ValueError("No negative examples pass the stated display criteria")
    fig, axes = plt.subplots(len(frames), 2, figsize=(13, 3.4 * len(frames)),
                             squeeze=False, constrained_layout=True)
    mapped = MappedWaveformFile(analysis["index"])
    try:
        for row, frame in enumerate(frames):
            frame = int(frame)
            raw = mapped.frame(frame).samples.astype(np.float64)
            start = int(analysis["onset"][frame])
            short_end, long_end = start + analysis["short_samples"], start + long_samples
            b = analysis["baselines"]["median"][frame]
            y = b - raw
            x = np.arange(len(y))
            left, right = axes[row]
            left.step(x, y, where="post", color="tab:blue", linewidth=0.9)
            left.axhline(0, color="black", linewidth=0.7)
            left.axvspan(*analysis["baseline_region"], color="tab:blue", alpha=0.10)
            left.axvspan(start, short_end, color="green", alpha=0.12)
            left.axvspan(short_end, long_end, color="orange", alpha=0.15)
            tail = np.arange(short_end, long_end + 1)
            negative_area = np.r_[np.minimum(y[short_end:long_end], 0), 0]
            left.fill_between(tail, negative_area, 0,
                              step="post", color="red", alpha=0.45)
            left.set(xlim=(max(0, start - 25), long_end + 10),
                     ylabel="B - ADC (median baseline)", xlabel="Sample index")
            left.set_title(f"Frame {frame}: Qshort={median['short'][frame]:,.0f}, "
                           f"Qlong={median['long'][frame]:,.0f}; "
                           f"tail={median['long'][frame] - median['short'][frame]:,.0f}")
            for method, color in (("median", "tab:blue"), ("mean", "tab:orange")):
                baseline = analysis["baselines"][method][frame]
                running = np.r_[0, np.cumsum(baseline - raw[start:long_end])]
                ratio = analysis["charges"][method, long_samples]["psd"][frame]
                right.step(np.arange(start, long_end + 1), running, where="post",
                           color=color, label=f"{method}: B={baseline:.2f}; PSD={ratio:.4f}")
            for ax in (left, right):
                for location, color in ((start, "green"), (short_end, "goldenrod"),
                                        (long_end, "red")):
                    ax.axvline(location, color=color, linestyle="--", linewidth=0.9)
                ax.grid(alpha=0.25)
            right.set(xlabel="Sample index", ylabel="Cumulative charge (ADC samples)")
            right.legend(fontsize=8, loc="upper right")
    finally:
        mapped.close()
    fig.suptitle("Actual negative-PSD waveforms: green=short; orange=extra long-gate tail; "
                 "red shading=negative tail samples\n"
                 "Explicit offline gates, not event-matched to the separate ROOT capture")
    return fig, np.asarray(frames, dtype=int)


def show_psd_maps(analysis, long_samples=32):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, method in zip(axes, ("median", "mean"), strict=True):
        result = analysis["charges"][method, long_samples]
        valid = np.isfinite(result["psd"])
        energy, ratio = result["long"][valid], result["psd"][valid]
        image = ax.hist2d(energy, ratio, bins=(512, 256), cmin=1, norm=LogNorm(),
                          range=((0, np.percentile(energy, 99.5)), (-0.4, 0.6)))
        fig.colorbar(image[3], ax=ax, label="Waveforms/bin")
        ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
        ax.set(xlabel="Reconstructed long charge (ADC-count samples)", ylabel="PSD",
               title=f"{method} baseline: {np.count_nonzero(ratio < 0):,}/{len(ratio):,} "
                     f"negative ({100 * np.mean(ratio < 0):.3f}%)")
    fig.suptitle(f"Waveform PSD, short={analysis['short_samples']}, long={long_samples} samples\n"
                 "Display limits: 99.5th energy percentile and PSD [-0.4, 0.6]")
    return fig


if __name__ == "__main__":
    analysis = reconstruct()
    summary = summarize(analysis)
    print(f"Reconstructed {analysis['index'].frame_count:,} complete waveforms")
    print(json.dumps(summary, indent=2))
    fig, negative_frames = show_negative_pulses(analysis)
    psd_figure = show_psd_maps(analysis)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "negative_pulses.png", dpi=160)
    psd_figure.savefig(OUTPUT / "psd_maps.png", dpi=160)
    report = dict(source=str(FILENAME), frames=analysis["index"].frame_count,
                  baseline_samples=BASELINE, search_samples=SEARCH,
                  units="Sample indices; charges in ADC-count samples",
                  onset_rule="First sample >= max(10% local peak, 5 pretrigger RMS)",
                  summary=summary, displayed_frames=negative_frames.tolist(),
                  caveat="Offline gates are explicit assumptions; no event matching to ROOT.")
    report["examples"] = [
        dict(frame=int(frame), gate_start=int(analysis["onset"][frame]),
             results={
                 method: dict(baseline=float(analysis["baselines"][method][frame]),
                              **{key: float(values[frame]) for key, values in
                                 analysis["charges"][method, 32].items()})
                 for method in ("median", "mean")
             })
        for frame in negative_frames
    ]
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Plots/report: {OUTPUT}")
    plt.show()
