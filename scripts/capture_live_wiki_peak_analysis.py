"""Capture a real MCA peak-analysis screenshot from the .128 reference board."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from scipy.signal import find_peaks

from nlab.analysis.spectrum import Spectrum
from nlab.app import MainAppWindow
from nlab.views.mca_peak_analysis_dialog import McaPeakAnalysisDialog


def _log(message: str) -> None:
    print(message, flush=True)


def _wait_until(predicate: object, timeout_s: float, description: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():  # type: ignore[operator]
            return
        QTest.qWait(100)
    raise TimeoutError(f"timed out waiting for {description}")


def _set_control(control: object, method: str, value: object) -> None:
    control.blockSignals(True)  # type: ignore[attr-defined]
    getattr(control, method)(value)
    control.blockSignals(False)  # type: ignore[attr-defined]


def _save_dialog(dialog: McaPeakAnalysisDialog, path: Path) -> None:
    dialog.resize(1560, 1000)
    dialog.show_workspace()
    QApplication.processEvents()
    QTest.qWait(300)
    if not dialog.grab().save(str(path), "PNG"):
        raise RuntimeError(f"could not save {path}")
    _log(f"Saved {path}")


def run_capture(args: argparse.Namespace) -> dict[str, object]:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    QSettings().setValue("dma/mca_output_mode", "online")
    window: MainAppWindow | None = None
    dialog: McaPeakAnalysisDialog | None = None
    mca = None
    try:
        window = MainAppWindow(
            backend="iio",
            host=args.host,
            port=args.port,
            channels=2,
            on_progress=lambda message: _log(f"GUI: {message}"),
        )
        window.resize(1560, 1000)
        window.show()
        QApplication.processEvents()
        mca = window._controller._mca_controllers[args.channel]
        hardware = mca._mca
        hardware.stop()
        hardware.set_pulse_polarity(0)
        hardware.set_baseline_window(3)
        hardware.set_trigger_level(args.threshold)
        hardware.set_frame_samples(256)
        hardware.set_pretrigger_samples(24)
        hardware.set_trg_source(0)
        hardware.set_pileup_window(0)
        hardware.set_energy_bin(args.binning_index)
        hardware.set_mem1_sig_select(0)
        hardware.set_mem2_sig_select(1)
        hardware.set_ext_trig_enable(False)

        ui = mca.ui
        for control, method, value in (
            (ui.comboPulsePolarity, "setCurrentIndex", 0),
            (ui.comboBaseline, "setCurrentIndex", 3),
            (ui.spinTriggerLevel, "setValue", args.threshold),
            (ui.spinFrameSamples, "setValue", 256),
            (ui.spinPretrigger, "setValue", 24),
            (ui.comboTriggerSource, "setCurrentIndex", 0),
            (ui.spinPileupWindow, "setValue", 0),
            (ui.comboBinning, "setCurrentIndex", args.binning_index),
            (ui.comboDebug1, "setCurrentIndex", 0),
            (ui.comboDebug2, "setCurrentIndex", 1),
            (ui.cbExtTrigger, "setChecked", False),
            (ui.cbDmaEnable, "setChecked", True),
            (ui.spinTimeLimit, "setValue", 0),
            (ui.spinRefreshRate, "setValue", 5),
        ):
            _set_control(control, method, value)

        window.ui.mainTabs.setCurrentWidget(window.ui.tabMCA)
        QTest.mouseClick(ui.btnStart, Qt.MouseButton.LeftButton)
        _wait_until(lambda: mca._dma_worker is not None, 15, "MCA DMA start")
        _wait_until(
            lambda: mca._last_histogram is not None
            and float(np.max(mca._last_histogram, initial=0)) >= args.minimum_peak_count,
            args.timeout,
            f"a live peak with at least {args.minimum_peak_count} counts",
        )
        QTest.mouseClick(ui.btnStop, Qt.MouseButton.LeftButton)
        _wait_until(lambda: mca._dma_thread is None, 30, "MCA DMA stop and drain")

        raw_histogram = np.asarray(mca._last_histogram, dtype=np.uint32).copy()
        histogram = raw_histogram.astype(np.float64)
        _log(f"Frozen live spectrum contains {int(histogram.sum()):,} counts")
        search = histogram.copy()
        search[: args.minimum_channel] = 0
        peaks, properties = find_peaks(
            search,
            distance=128,
            width=3,
            prominence=max(5.0, 0.03 * float(search.max(initial=0))),
        )
        if not len(peaks):
            raise RuntimeError("live MCA spectrum has no suitable peak for the workbench")
        peak = int(peaks[int(np.argmax(properties["prominences"]))])
        half_width = args.fit_half_width
        fit_low = float(max(0, peak - half_width))
        fit_high = float(min(len(histogram) - 1, peak + half_width))

        _log(f"Opening workbench around MCA channel {peak}")
        dialog = McaPeakAnalysisDialog([])
        dialog._add_spectrum(
            Spectrum.create(
                label=f"Live .128 CH{args.channel} stopped spectrum",
                x=np.arange(len(raw_histogram), dtype=np.float64),
                counts=raw_histogram,
                axis_unit="channel",
                source="mca",
                metadata={
                    "endpoint": f"{args.host}:{args.port}",
                    "channel": args.channel,
                    "trigger_level": args.threshold,
                    "binning": 1 << args.binning_index,
                    "live_hardware": True,
                    "stopped_snapshot": True,
                },
                poisson_counts=True,
            )
        )
        dialog.fit_region.setRegion((fit_low, fit_high))
        dialog.peak_count.setValue(1)
        dialog.background_combo.setCurrentIndex(dialog.background_combo.findData("linear"))
        dialog._suggest_peaks()
        dialog.spectrum_plot.setXRange(fit_low, fit_high, padding=0.03)
        _save_dialog(dialog, args.output_dir / "mca-peak-analysis-live.png")

        return {
            "endpoint": f"{args.host}:{args.port}",
            "channel": args.channel,
            "trigger_level": args.threshold,
            "binning": 1 << args.binning_index,
            "fit_range": [fit_low, fit_high],
            "suggested_peak_channel": peak,
            "spectrum_total_counts": int(histogram.sum(dtype=np.float64)),
        }
    finally:
        if dialog is not None:
            dialog.close_without_prompt()
        if mca is not None:
            mca.stop_dma_sync()
            mca._mca.stop()
        if window is not None:
            for controller in window._controller._mca_controllers:
                controller.stop_dma_sync()
                controller.stop_worker_sync()
                controller._mca.stop()
            for controller in window._controller._scope_controllers:
                controller.stop_dma_sync()
                controller._scope.stop()
            window.close()
            QApplication.processEvents()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.10.128")
    parser.add_argument("--port", type=int, default=30_431)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--threshold", type=int, default=-96)
    parser.add_argument("--binning-index", type=int, default=5)
    parser.add_argument("--minimum-channel", type=int, default=512)
    parser.add_argument("--minimum-peak-count", type=int, default=200)
    parser.add_argument("--fit-half-width", type=int, default=320)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nlab-live-wiki-peak-") as settings_dir:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, settings_dir)
        result = run_capture(args)
    _log(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
