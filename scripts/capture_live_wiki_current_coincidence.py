"""Temporary live GUI probe/capture for Current and Coincidence wiki pages."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDockWidget
from scipy.signal import find_peaks

from nlab.analysis.coincidence import fit_coincidence_peak
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


def _wait(seconds: float, label: str) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.processEvents()
        remaining = max(0, math.ceil(deadline - time.monotonic()))
        if remaining % 2 == 0:
            _log(f"{label}: about {remaining} s remaining")
        QTest.qWait(250)


def _coarse_peaks(values: np.ndarray) -> list[tuple[int, float]]:
    histogram = np.asarray(values, dtype=np.float64)
    coarse = histogram.reshape(-1, 64).sum(axis=1)
    peaks, properties = find_peaks(
        coarse,
        distance=5,
        prominence=max(2.0, 0.02 * float(coarse.max(initial=0))),
    )
    ranked = sorted(
        zip(peaks, properties["prominences"], strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    return [(int(peak * 64 + 32), float(prominence)) for peak, prominence in ranked]


def _save_window(window: MainAppWindow, path: Path) -> None:
    window.resize(1500, 900)
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


def run_current_capture(args: argparse.Namespace) -> dict[str, object]:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    window: MainAppWindow | None = None
    current = None
    try:
        window = MainAppWindow(
            backend="iio",
            host=args.host,
            port=args.port,
            channels=2,
            on_progress=lambda message: _log(f"GUI: {message}"),
        )
        window.resize(1500, 900)
        window.show()
        QApplication.processEvents()
        controller = window._controller
        current = controller._current_monitor_controllers[1]
        window.ui.mainTabs.setCurrentWidget(window.ui.tabCurrent)
        _raise_dock(window, "current_ch1")
        current.ui.comboUnit.setCurrentText("raw")

        current.ui.comboMode.setCurrentIndex(0)
        QTest.mouseClick(current.ui.btnStart, Qt.MouseButton.LeftButton)
        _wait_until(lambda: current._latest is not None, 10, "a live IIR sample")
        _wait(4, "Current IIR")
        _save_window(window, args.output_dir / "current-monitor-iir-live.png")
        QTest.mouseClick(current.ui.btnStop, Qt.MouseButton.LeftButton)
        _wait_until(lambda: not current._is_running(), 10, "IIR monitor cleanup")
        iir_summary = {
            "value": current.ui.lblCurrent.text(),
            "raw": current.ui.lblRaw.text(),
            "acquisition": current.ui.lblAcquisition.text(),
        }

        current.ui.comboMode.setCurrentIndex(1)
        QTest.mouseClick(current.ui.btnStart, Qt.MouseButton.LeftButton)
        _wait_until(lambda: current._dma_active, 15, "Scope DMA current monitor")
        accumulator = current._scope_current_accumulator
        if accumulator is None:
            raise RuntimeError("Current DMA accumulator is unavailable")
        _wait_until(
            lambda: accumulator.snapshot(now_ns=time.perf_counter_ns()).received_frames >= 20,
            15,
            "Current DMA frames",
        )
        _wait(4, "Current DMA")
        _save_window(window, args.output_dir / "current-monitor-dma-live.png")
        dma_snapshot = accumulator.snapshot(now_ns=time.perf_counter_ns())
        dma_summary = {
            "received_frames": dma_snapshot.received_frames,
            "analyzed_frames": dma_snapshot.analyzed_frames,
            "protocol_errors": dma_snapshot.protocol_errors,
            "skipped_opportunities": dma_snapshot.skipped_opportunities,
            "coverage_percent": dma_snapshot.observed_coverage_percent,
            "status": current.ui.lblStatus.text(),
        }
        QTest.mouseClick(current.ui.btnStop, Qt.MouseButton.LeftButton)
        _wait_until(lambda: not current._is_running(), 20, "Current DMA cleanup")
        QTest.mouseClick(current.ui.btnResetScopeDefaults, Qt.MouseButton.LeftButton)
        return {"channel": 1, "iir": iir_summary, "dma": dma_summary}
    finally:
        if current is not None:
            current.stop_monitor_sync()
        if window is not None:
            for scope in window._controller._scope_controllers:
                scope._scope.stop()
            for mca in window._controller._mca_controllers:
                mca._mca.stop()
            window.close()
            QApplication.processEvents()


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    QSettings().setValue("dma/mca_output_mode", "online")
    window: MainAppWindow | None = None
    coincidence = None
    try:
        window = MainAppWindow(
            backend="iio",
            host=args.host,
            port=args.port,
            channels=2,
            on_progress=lambda message: _log(f"GUI: {message}"),
        )
        window.resize(1500, 900)
        window.show()
        QApplication.processEvents()
        controller = window._controller
        for view in controller._mca_controllers:
            view._mca.stop()
            view._mca.set_pulse_polarity(0)
            view._mca.set_trigger_level(args.threshold)
            view._mca.set_energy_bin(args.binning_index)
            view.ui.comboPulsePolarity.setCurrentIndex(0)
            view.ui.spinTriggerLevel.setValue(args.threshold)
            view.ui.comboBinning.setCurrentIndex(args.binning_index)

        coincidence = controller._coincidence_controller
        if coincidence is None:
            raise RuntimeError("Coincidence workspace is unavailable")
        for checkbox in coincidence.use_roi:
            checkbox.setChecked(False)
        coincidence.timing_mode.setCurrentIndex(0)
        coincidence.low.setValue(-1000.0)
        coincidence.high.setValue(1000.0)
        coincidence.duration.setValue(0)

        window.ui.mainTabs.setCurrentWidget(window.ui.tabCoincidence)
        QApplication.processEvents()
        QTest.mouseClick(coincidence.btnStart, Qt.MouseButton.LeftButton)
        _wait_until(
            lambda: coincidence._state == "running",
            15,
            "both coincidence DMA readers to arm",
        )
        _wait(args.seconds, "Broad coincidence probe")
        QTest.mouseClick(coincidence.btnStop, Qt.MouseButton.LeftButton)
        _wait_until(lambda: coincidence._state == "idle", 30, "coincidence cleanup")
        snapshot = coincidence._last_rendered_snapshot
        if snapshot is None:
            raise RuntimeError("coincidence run produced no snapshot")
        delay_counts = np.asarray(snapshot.delay_counts)
        delay_bin = int(np.argmax(delay_counts)) if delay_counts.size else -1
        delay_ns = -1000.0 + delay_bin * 8.0 if delay_bin >= 0 else math.nan
        matrix_index = np.unravel_index(
            int(np.argmax(snapshot.prompt_matrix)),
            snapshot.prompt_matrix.shape,
        )
        return {
            "pairs": int(snapshot.pairs),
            "random_pairs": int(snapshot.random_pairs),
            "accepted_ch0": int(snapshot.accepted_ch0),
            "accepted_ch1": int(snapshot.accepted_ch1),
            "delay_peak_ns": delay_ns,
            "delay_peak_counts": int(delay_counts[delay_bin]) if delay_bin >= 0 else 0,
            "energy_peaks_ch0": _coarse_peaks(snapshot.energy_ch0),
            "energy_peaks_ch1": _coarse_peaks(snapshot.energy_ch1),
            "matrix_max_raw_channels": [
                int(matrix_index[1] * 32 + 16),
                int(matrix_index[0] * 32 + 16),
            ],
            "status": coincidence.status.text(),
        }
    finally:
        if coincidence is not None and coincidence._state in {"arming", "running"}:
            coincidence._begin_stop("probe cleanup")
            try:
                _wait_until(lambda: coincidence._state == "idle", 30, "cleanup")
            except Exception as exc:
                _log(f"Coincidence cleanup warning: {exc}")
        if window is not None:
            window.close()
            QApplication.processEvents()


def _set_widget_value(widget: object, value: object) -> None:
    widget.blockSignals(True)  # type: ignore[attr-defined]
    if hasattr(widget, "setChecked"):
        widget.setChecked(value)  # type: ignore[attr-defined]
    elif hasattr(widget, "setCurrentIndex"):
        widget.setCurrentIndex(value)  # type: ignore[attr-defined]
    else:
        widget.setValue(value)  # type: ignore[attr-defined]
    widget.blockSignals(False)  # type: ignore[attr-defined]


def run_coincidence_comparison(args: argparse.Namespace) -> dict[str, object]:
    """Capture live coarse and CFD timing views for the prepared Na-22 setup."""
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    QSettings().setValue("dma/mca_output_mode", "online")
    window: MainAppWindow | None = None
    coincidence = None
    roi_by_channel = (tuple(args.roi_ch0), tuple(args.roi_ch1))
    runs: list[dict[str, object]] = []
    try:
        window = MainAppWindow(
            backend="iio",
            host=args.host,
            port=args.port,
            channels=2,
            on_progress=lambda message: _log(f"GUI: {message}"),
        )
        window.resize(1500, 900)
        window.show()
        QApplication.processEvents()
        controller = window._controller

        for channel, view in enumerate(controller._mca_controllers):
            view._mca.stop()
            view._mca.set_pulse_polarity(0)
            view._mca.set_trigger_level(args.threshold)
            view._mca.set_energy_bin(args.binning_index)
            cfd = view._mca.filters.cfd
            cfd.set_enable(False)
            cfd.set_factor(0.4)
            cfd.set_delay(2)
            cfd.set_time_window_low(8)
            cfd.set_time_window_high(64)
            _set_widget_value(view.ui.comboPulsePolarity, 0)
            _set_widget_value(view.ui.spinTriggerLevel, args.threshold)
            _set_widget_value(view.ui.comboBinning, args.binning_index)
            _set_widget_value(view.ui.cbCfdEnable, False)
            _set_widget_value(view.ui.spinCfdFactor, 0.4)
            _set_widget_value(view.ui.spinCfdDelay, 2)
            _set_widget_value(view.ui.spinCfdTwLow, 8)
            _set_widget_value(view.ui.spinCfdTwHigh, 64)
            view.set_roi_visible(True)
            view._roi.setRegion(roi_by_channel[channel])
            view._on_roi_change_finished()

        coincidence = controller._coincidence_controller
        if coincidence is None:
            raise RuntimeError("Coincidence workspace is unavailable")
        for checkbox in coincidence.use_roi:
            checkbox.setChecked(True)
        coincidence.operator.setCurrentText("AND")
        coincidence.not_ch0.setChecked(False)
        coincidence.not_ch1.setChecked(False)
        coincidence.low.setValue(args.timing_low)
        coincidence.high.setValue(args.timing_high)
        coincidence.offset.setValue(0.0)
        coincidence.duration.setValue(0)
        coincidence.random_sidebands.setChecked(args.random_sidebands)
        coincidence.random_gap.setValue(0.0)

        window.ui.mainTabs.setCurrentWidget(window.ui.tabCoincidence)
        coincidence.result_tabs.setCurrentIndex(0)
        QApplication.processEvents()

        run_specs = (
            ("coarse", False, "coincidence-na22-511-coarse-live.png"),
            ("cfd", True, "coincidence-na22-511-cfd-live.png"),
        )
        if args.comparison_run != "both":
            run_specs = tuple(spec for spec in run_specs if spec[0] == args.comparison_run)
        for name, fine, filename in run_specs:
            for view in controller._mca_controllers:
                view._mca.filters.cfd.set_enable(fine)
                _set_widget_value(view.ui.cbCfdEnable, fine)
            coincidence.timing_mode.setCurrentIndex(1 if fine else 0)
            QApplication.processEvents()
            QTest.mouseClick(coincidence.btnStart, Qt.MouseButton.LeftButton)
            _wait_until(
                lambda: coincidence._state == "running",
                15,
                f"both {name} coincidence DMA readers to arm",
            )
            _wait_until(
                lambda: coincidence._last_rendered_snapshot is not None
                and coincidence._last_rendered_snapshot.pairs >= args.minimum_pairs,
                20,
                f"at least {args.minimum_pairs} {name} coincidence pairs",
            )
            _wait(args.seconds, f"Na-22 511 keV {name}")
            snapshot = coincidence._last_rendered_snapshot
            if snapshot is None:
                raise RuntimeError(f"{name} coincidence run produced no snapshot")
            settings = coincidence._current_settings
            delay_counts = np.asarray(snapshot.delay_counts)
            peak_index = int(np.argmax(delay_counts))
            peak_ns = settings.low_ns + (peak_index + 0.5) * settings.bin_width_ns
            fit = fit_coincidence_peak(delay_counts, settings)
            live_status = coincidence.status.text()
            matrix_index = np.unravel_index(
                int(np.argmax(snapshot.prompt_matrix)), snapshot.prompt_matrix.shape
            )
            if args.capture_matrix:
                coincidence.matrix_mode.setCurrentIndex(0)
                coincidence.matrix_scale.setCurrentIndex(1)
                coincidence.matrix_gate_ch0.setRegion(
                    (roi_by_channel[0][0], roi_by_channel[0][1] + 1)
                )
                coincidence.matrix_gate_ch1.setRegion(
                    (roi_by_channel[1][0], roi_by_channel[1][1] + 1)
                )
                coincidence.result_tabs.setCurrentIndex(coincidence._matrix_tab_index)
                coincidence._render_matrix(snapshot)
                coincidence.matrix_plot.setXRange(
                    roi_by_channel[0][0] - 100, roi_by_channel[0][1] + 100, padding=0
                )
                coincidence.matrix_plot.setYRange(
                    roi_by_channel[1][0] - 100, roi_by_channel[1][1] + 100, padding=0
                )
                coincidence.matrix_projection0_plot.setXRange(
                    roi_by_channel[0][0] - 100, roi_by_channel[0][1] + 100, padding=0
                )
                coincidence.matrix_projection1_plot.setXRange(
                    roi_by_channel[1][0] - 100, roi_by_channel[1][1] + 100, padding=0
                )
                filename = "coincidence-na22-511-matrix-live.png"
                QApplication.processEvents()
            _save_window(window, args.output_dir / filename)

            QTest.mouseClick(coincidence.btnStop, Qt.MouseButton.LeftButton)
            _wait_until(lambda: coincidence._state == "idle", 30, f"{name} cleanup")
            summaries = []
            for channel, view in enumerate(controller._mca_controllers):
                summary = view.coincidence_run_summary
                summaries.append(
                    {
                        "channel": channel,
                        "continuity": summary.continuity if summary is not None else None,
                        "records": summary.records if summary is not None else None,
                        "diagnostics": summary.diagnostics if summary is not None else None,
                    }
                )
            runs.append(
                {
                    "mode": name,
                    "cfd_enabled": fine,
                    "pairs": int(snapshot.pairs),
                    "random_pairs": int(snapshot.random_pairs),
                    "peak_bin_center_ns": peak_ns,
                    "peak_bin_counts": int(delay_counts[peak_index]),
                    "matrix_max_raw_channels": [
                        int(matrix_index[1] * 32 + 16),
                        int(matrix_index[0] * 32 + 16),
                    ],
                    "matrix_max_counts": int(snapshot.prompt_matrix[matrix_index]),
                    "fit": (
                        {
                            "center_ns": fit.center_ns,
                            "fwhm_ns": fit.fwhm_ns,
                            "fwhm_uncertainty_ns": fit.fwhm_uncertainty_ns,
                            "signal_counts": fit.signal_counts,
                            "reduced_chi_square": fit.reduced_chi_square,
                        }
                        if fit is not None
                        else None
                    ),
                    "live_status": live_status,
                    "final_status": coincidence.status.text(),
                    "summaries": summaries,
                }
            )

        final_deadtime = [
            int(device.mca._b.get_list_deadtime_raw()) for device in controller._devices
        ]
        return {
            "endpoint": f"{args.host}:{args.port}",
            "threshold": args.threshold,
            "binning_index": args.binning_index,
            "rois": roi_by_channel,
            "timing_gate_ns": [args.timing_low, args.timing_high],
            "runs": runs,
            "final_list_deadtime_raw": final_deadtime,
        }
    finally:
        if coincidence is not None and coincidence._state in {"arming", "running"}:
            coincidence._begin_stop("comparison cleanup")
            try:
                _wait_until(lambda: coincidence._state == "idle", 30, "comparison cleanup")
            except Exception as exc:
                _log(f"Coincidence cleanup warning: {exc}")
        if window is not None:
            for mca in window._controller._mca_controllers:
                mca.stop_dma_sync()
                mca._mca.stop()
            window.close()
            QApplication.processEvents()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.10.128")
    parser.add_argument("--port", type=int, default=30_431)
    parser.add_argument("--threshold", type=int, default=-1024)
    parser.add_argument("--binning-index", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images"))
    parser.add_argument("--comparison-run", choices=("both", "coarse", "cfd"), default="both")
    parser.add_argument("--roi-ch0", nargs=2, type=int, default=(1932, 2666))
    parser.add_argument("--roi-ch1", nargs=2, type=int, default=(1321, 1961))
    parser.add_argument("--timing-low", type=float, default=-48.0)
    parser.add_argument("--timing-high", type=float, default=48.0)
    parser.add_argument("--minimum-pairs", type=int, default=500)
    parser.add_argument("--random-sidebands", action="store_true")
    parser.add_argument("--capture-matrix", action="store_true")
    parser.add_argument(
        "--mode", choices=("probe", "current", "coincidence-compare"), default="probe"
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nlab-live-wiki-") as settings_dir:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, settings_dir)
        if args.mode == "current":
            result = run_current_capture(args)
        elif args.mode == "coincidence-compare":
            result = run_coincidence_comparison(args)
        else:
            result = run_probe(args)
    _log(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
