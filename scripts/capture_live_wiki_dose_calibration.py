"""Capture the dose logger with live MCA data from a hardware test signal."""

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


def run_capture(args: argparse.Namespace, temporary_directory: Path) -> dict[str, object]:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    window: MainAppWindow | None = None
    dialog = None
    mca = None
    try:
        window = MainAppWindow(
            host=args.host,
            port=args.port,
            channels=2,
            on_progress=lambda message: _log(f"GUI: {message}"),
        )
        window.resize(1560, 1000)
        window.show()
        QApplication.processEvents()
        mca = window._controller._mca_controllers[args.channel]
        mca.stop_dma_sync()
        mca._mca.stop()
        mca.ui.cbDmaEnable.setChecked(False)
        mca.ui.spinTimeLimit.setValue(0)
        mca.ui.spinRefreshRate.setValue(5)

        window.ui.mainTabs.setCurrentWidget(window.ui.tabMCA)
        QTest.mouseClick(mca.ui.btnStart, Qt.MouseButton.LeftButton)
        _wait_until(lambda: mca._worker is not None, 15, "MCA polling start")

        def spectrum_is_ready() -> bool:
            histogram = mca._last_histogram
            return bool(
                histogram is not None
                and int(np.sum(histogram, dtype=np.uint64)) >= args.minimum_counts
            )

        _wait_until(spectrum_is_ready, args.timeout, "live MCA dose-logger input")

        window._controller.show_dose_calibration()
        dialog = window._controller._dose_calibration_dialog
        if dialog is None:
            raise RuntimeError("dose calibration dialog did not open")
        dialog.channel_combo.setCurrentIndex(args.channel)
        dialog.source_label.setText("CAEN Na-22 generator (functional test; no PSU)")
        dialog.dose_rate.setValue(0.0)
        dialog.dose_unit.setCurrentText("\N{MICRO SIGN}Sv/h")
        dialog.save_mca_statistics.setChecked(True)
        dialog.save_mca_differential.setChecked(True)
        dialog.save_psu_voltage.setChecked(False)
        dialog.save_temperatures.setChecked(False)
        dialog.save_psd_integrals.setChecked(False)
        dialog.interval_ms.setValue(args.sample_wait_ms)
        csv_path = temporary_directory / "dose_calibration_na22_generator.csv"
        dialog.output_path.setText(str(csv_path))
        dialog._commit_reference_fields()
        dialog._start_recording()
        if dialog._writer is None:
            raise RuntimeError("dose calibration CSV recording did not start")

        _wait_until(
            lambda: dialog._row_count >= args.samples,
            args.samples * args.sample_wait_ms / 1_000.0 + 5.0,
            f"{args.samples} dose-logger intervals",
        )

        plot_index = dialog.plot_data_combo.findData("mca_count_rate_raw")
        if plot_index < 0:
            raise RuntimeError("MCA count-rate plot is unavailable")
        dialog.plot_data_combo.setCurrentIndex(plot_index)
        dialog._stop_recording()
        dialog.resize(1400, 760)
        dialog.show_workspace()
        QApplication.processEvents()
        QTest.qWait(300)
        output_path = args.output_dir / "dose-calibration-na22-generator-live.png"
        if not dialog.grab().save(str(output_path), "PNG"):
            raise RuntimeError(f"could not save {output_path}")
        _log(f"Saved {output_path}")

        plotted_elapsed, plotted_values = dialog.plot_curve.getData()
        histogram = np.asarray(mca._last_histogram, dtype=np.uint32)
        return {
            "endpoint": f"{args.host}:{args.port}",
            "channel": args.channel,
            "signal": "CAEN Na-22 generator",
            "physical_dose_reference": False,
            "rows": dialog._row_count,
            "spectrum_total_counts": int(histogram.sum(dtype=np.uint64)),
            "plotted_elapsed_s": [] if plotted_elapsed is None else plotted_elapsed.tolist(),
            "plotted_count_rate_raw": (
                [] if plotted_values is None else plotted_values.tolist()
            ),
        }
    finally:
        if dialog is not None:
            dialog.close_without_prompt()
        if mca is not None:
            mca.stop_dma_sync()
            try:
                mca.stop_worker_sync()
            except RuntimeError:
                pass
            mca._mca.stop()
        if window is not None:
            for controller in window._controller._mca_controllers:
                controller.stop_dma_sync()
                try:
                    controller.stop_worker_sync()
                except RuntimeError:
                    pass
                controller._mca.stop()
            for controller in window._controller._scope_controllers:
                controller.stop_dma_sync()
                controller._scope.stop()
            window.close()
            QApplication.processEvents()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.10.135")
    parser.add_argument("--port", type=int, default=30_431)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--minimum-counts", type=int, default=20_000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sample-wait-ms", type=int, default=750)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nlab-live-wiki-dose-") as temporary:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, temporary)
        result = run_capture(args, Path(temporary))
    _log(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
