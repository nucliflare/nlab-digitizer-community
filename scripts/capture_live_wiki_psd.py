"""Capture the live CH0 Scope Auto Setup, MCA settings, and PSD wiki images."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDockWidget
from scipy.signal import find_peaks

from nlab.app import MainAppWindow


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


def _save_window(window: MainAppWindow, path: Path) -> None:
    window.resize(1560, 1000)
    window.show()
    QApplication.processEvents()
    QTest.qWait(300)
    if not window.grab().save(str(path), "PNG"):
        raise RuntimeError(f"could not save {path}")
    _log(f"Saved {path}")


def _raise_dock(window: MainAppWindow, object_name: str) -> None:
    dock = window.findChild(QDockWidget, object_name)
    if dock is None:
        raise RuntimeError(f"could not find dock {object_name}")
    dock.raise_()
    QApplication.processEvents()
    QTest.qWait(250)


def _set_control(control: object, method: str, value: object) -> None:
    control.blockSignals(True)  # type: ignore[attr-defined]
    getattr(control, method)(value)
    control.blockSignals(False)  # type: ignore[attr-defined]


def _ratio_peaks(matrix: np.ndarray, ratio_centers: np.ndarray) -> list[dict[str, float]]:
    counts = np.asarray(matrix, dtype=np.uint64).sum(axis=0, dtype=np.uint64)
    peaks, properties = find_peaks(
        counts.astype(np.float64),
        distance=max(1, len(counts) // 8),
        prominence=max(10.0, 0.05 * float(counts.max(initial=0))),
    )
    ranked = sorted(
        zip(peaks, properties["prominences"], strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:4]
    return [
        {
            "ratio": float(ratio_centers[index]),
            "counts": int(counts[index]),
            "prominence": float(prominence),
        }
        for index, prominence in ranked
    ]


def run_capture(args: argparse.Namespace) -> dict[str, object]:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    QSettings().setValue("dma/mca_output_mode", "online")
    window: MainAppWindow | None = None
    scope = None
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
        controller = window._controller
        scope = controller._scope_controllers[0]
        mca = controller._mca_controllers[0]
        psd = controller._psd_controller_by_device[0]

        scope.stop_dma_sync()
        _set_control(scope.ui.cbDmaEnable, "setChecked", False)
        window.ui.mainTabs.setCurrentWidget(window.ui.tabScope)
        _raise_dock(window, "scope_ch0")
        QTest.mouseClick(scope.ui.btnAutoSetup, Qt.MouseButton.LeftButton)
        _wait_until(lambda: scope._auto_setup_thread is not None, 5, "Scope Auto Setup start")
        _wait_until(lambda: scope._auto_setup_thread is None, 90, "Scope Auto Setup completion")
        if scope._auto_setup_result is None:
            raise RuntimeError(f"Scope Auto Setup failed: {scope._auto_setup_error}")
        auto = scope._auto_setup_result
        _save_window(window, args.output_dir / "psd-ch0-scope-auto-setup-live.png")
        scope._scope.stop()

        hw = mca._mca
        hw.stop()
        hw.set_pulse_polarity(0)
        hw.set_baseline_window(3)
        hw.set_trigger_level(-96)
        hw.set_frame_samples(256)
        hw.set_pretrigger_samples(24)
        hw.set_trg_source(0)
        hw.set_pileup_window(0)
        hw.set_energy_bin(5)
        hw.set_mem1_sig_select(0)
        hw.set_mem2_sig_select(1)
        hw.set_ext_trig_enable(False)
        hw.filters.charge_comparison.set_time(64)
        hw.filters.charge_comparison.set_enable(True)
        hw.filters.psd_zc.set_enable(False)
        hw.filters.psd_zc.set_mode(0)
        hw.filters.psd_zc.set_time_window_low(8)
        hw.filters.psd_zc.set_time_window_high(16)

        ui = mca.ui
        for control, method, value in (
            (ui.comboPulsePolarity, "setCurrentIndex", 0),
            (ui.comboBaseline, "setCurrentIndex", 3),
            (ui.spinTriggerLevel, "setValue", -96),
            (ui.spinFrameSamples, "setValue", 256),
            (ui.spinPretrigger, "setValue", 24),
            (ui.comboTriggerSource, "setCurrentIndex", 0),
            (ui.spinPileupWindow, "setValue", 0),
            (ui.comboBinning, "setCurrentIndex", 5),
            (ui.comboDebug1, "setCurrentIndex", 0),
            (ui.comboDebug2, "setCurrentIndex", 1),
            (ui.cbExtTrigger, "setChecked", False),
            (ui.cbCcEnable, "setChecked", True),
            (ui.spinCcTime, "setValue", 64),
            (ui.cbPsdZcEnable, "setChecked", False),
            (ui.comboPsdZcMode, "setCurrentIndex", 0),
            (ui.spinPsdZcLow, "setValue", 8),
            (ui.spinPsdZcHigh, "setValue", 16),
            (ui.spinTimeLimit, "setValue", 0),
            (ui.cbDmaEnable, "setChecked", True),
        ):
            _set_control(control, method, value)

        psd.apply_configuration_settings(
            {
                "energy_bins": 1024,
                "ratio_bins": 256,
                "energy_right_shift": 0,
                "ratio_range": [-1.0, 1.0],
                "ratio_cut": 0.5834,
                "energy_roi": [0.0, 16384.0],
                "energy_log_y": False,
            }
        )
        window.setFocus(Qt.FocusReason.OtherFocusReason)
        QApplication.processEvents()

        window.ui.mainTabs.setCurrentWidget(window.ui.tabMCA)
        _raise_dock(window, "mca_ch0")
        QTest.mouseClick(ui.btnStart, Qt.MouseButton.LeftButton)
        _wait_until(lambda: mca._dma_worker is not None and psd._capturing, 15, "PSD DMA arm")
        window.ui.mainTabs.setCurrentWidget(window.ui.tabPSD)
        _raise_dock(window, "psd_ch0")
        _wait_until(
            lambda: psd._accumulator.statistics.accepted >= args.minimum_events,
            args.timeout,
            f"{args.minimum_events:,} accepted PSD events",
        )
        ui.btnStop.click()
        _wait_until(lambda: mca._dma_thread is None, 30, "PSD DMA stop and drain")
        psd.process_pending_events()

        stats = psd._accumulator.statistics
        below, above = psd._accumulator.energy_projections(psd.ui.spinCut.value())
        ratio_peaks = _ratio_peaks(
            psd._accumulator.matrix, psd._accumulator.ratio_centers
        )
        psd_status = psd.ui.lblStatus.text()
        _log(f"PSD after drain: {psd_status}")
        window.ui.mainTabs.setCurrentWidget(window.ui.tabPSD)
        _raise_dock(window, "psd_ch0")
        _save_window(window, args.output_dir / "psd-ch0-live.png")

        window.ui.mainTabs.setCurrentWidget(window.ui.tabMCA)
        _raise_dock(window, "mca_ch0")
        mca.set_roi_visible(True)
        mca._roi.setRegion((100, 200))
        mca._on_roi_change_finished()
        _save_window(window, args.output_dir / "psd-ch0-mca-settings-live.png")

        summary = mca._dma_summary
        if summary is None:
            raise RuntimeError("PSD run completed without a DMA summary")
        return {
            "endpoint": f"{args.host}:{args.port}",
            "channel": 0,
            "auto_setup": {
                "dac": auto.dac_value,
                "trigger_level": auto.trigger_level,
                "trigger_mode": auto.trigger_mode.name,
                "baseline": auto.baseline,
                "noise_sigma": auto.noise_sigma,
                "pulse_amplitude": auto.pulse_amplitude,
                "verified": auto.verified,
            },
            "mca": {
                "trigger_level": -96,
                "baseline_ns": 64,
                "window_ns": 256,
                "pretrigger_ns": 24,
                "binning": 32,
                "charge_comparison_ns": 64,
                "zero_crossing_enabled": False,
            },
            "psd": {
                "cut": psd.ui.spinCut.value(),
                "received": stats.received,
                "accepted": stats.accepted,
                "zero_energy": stats.zero_total,
                "outside_view": stats.outside_range,
                "below_cut": int(below.sum()),
                "above_cut": int(above.sum()),
                "ratio_peaks": ratio_peaks,
                "status": psd_status,
            },
            "dma": {
                "continuity": summary.continuity,
                "records": summary.records,
                "diagnostics": summary.diagnostics,
                "status": mca.ui.lblDmaStatus.text(),
            },
        }
    finally:
        if mca is not None:
            mca.stop_dma_sync()
            mca._mca.stop()
        if scope is not None:
            scope.stop_dma_sync()
            scope._scope.stop()
        if window is not None:
            window.close()
            QApplication.processEvents()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.10.128")
    parser.add_argument("--port", type=int, default=30_431)
    parser.add_argument("--minimum-events", type=int, default=200_000)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nlab-live-wiki-psd-") as settings_dir:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, settings_dir)
        result = run_capture(args)
    _log(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
