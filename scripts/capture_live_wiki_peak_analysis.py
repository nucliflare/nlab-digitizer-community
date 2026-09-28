"""Capture live Na-22 MCA calibration and 511 keV peak-fit wiki images."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
from scipy.signal import find_peaks

from nlab.analysis.energy_calibration import (
    EnergyCalibration,
    FingerprintValue,
    SpectrumSnapshot,
)
from nlab.analysis.peak_fitting import PeakFitResult, PeakFitSpec, fit_spectrum_peaks
from nlab.analysis.psd_file import inspect_psd_event_file, iter_psd_event_batches
from nlab.analysis.spectrum import Spectrum
from nlab.app import MainAppWindow
from nlab.views.energy_calibration_dialog import EnergyCalibrationDialog
from nlab.views.mca_peak_analysis_dialog import McaPeakAnalysisDialog


@dataclass
class _FrozenMcaController:
    """Small adapter that exposes one stopped hardware spectrum to the dialog."""

    channel: int
    snapshot: SpectrumSnapshot
    energy_calibration: EnergyCalibration | None = None

    def spectrum_snapshot(self) -> SpectrumSnapshot:
        return self.snapshot

    def energy_calibration_fingerprint(self) -> dict[str, object]:
        return dict(self.snapshot.fingerprint)

    def apply_energy_calibration(self, calibration: EnergyCalibration) -> None:
        self.energy_calibration = calibration

    def clear_energy_calibration(self) -> None:
        self.energy_calibration = None


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


def _save_widget(widget: QWidget, path: Path, size: tuple[int, int]) -> None:
    widget.resize(*size)
    show_workspace = getattr(widget, "show_workspace", None)
    if callable(show_workspace):
        show_workspace()
    else:
        widget.show()
    QApplication.processEvents()
    QTest.qWait(300)
    if not widget.grab().save(str(path), "PNG"):
        raise RuntimeError(f"could not save {path}")
    _log(f"Saved {path}")


def _range_maximum(histogram: np.ndarray, bounds: tuple[int, int]) -> float:
    low, high = bounds
    return float(np.max(histogram[low : high + 1], initial=0))


def _peak_guess(histogram: np.ndarray, bounds: tuple[int, int]) -> int:
    low, high = bounds
    values = np.asarray(histogram[low : high + 1], dtype=np.float64)
    peaks, properties = find_peaks(
        values,
        distance=64,
        width=3,
        prominence=max(5.0, 0.03 * float(values.max(initial=0))),
    )
    if not len(peaks):
        return low + int(np.argmax(values))
    return low + int(peaks[int(np.argmax(properties["prominences"]))])


def _fit_reference_peak(
    spectrum: Spectrum,
    center: int,
    half_width: int,
) -> PeakFitResult:
    return fit_spectrum_peaks(
        spectrum,
        PeakFitSpec(
            peak_count=1,
            background="linear",
            statistic="poisson",
            fit_min=float(center - half_width),
            fit_max=float(center + half_width),
            peak_centers=(float(center),),
        ),
    )


def _load_recorded_histogram(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Build the displayed MCA histogram from a retained IIO list-mode capture."""
    info = inspect_psd_event_file(path)
    if info.format_name != "NDMA IIO list-mode":
        raise ValueError(f"expected an IIO list-mode NDMA file, got {info.format_name}")
    histogram: np.ndarray = np.zeros(16_384, dtype=np.uint64)
    for batch in iter_psd_event_batches(path):
        # Coincidence/MCA display channels use the firmware energy word shifted by two.
        channels = (batch["trapezoid_energy"] >> 2).astype(np.int64)
        histogram += np.bincount(channels, minlength=16_384).astype(np.uint64)
    if not int(histogram.sum(dtype=np.uint64)):
        raise ValueError(f"{path} contains no MCA events")
    if int(histogram.max(initial=0)) > np.iinfo(np.uint32).max:
        raise ValueError("recorded histogram exceeds the uint32 MCA count range")

    summary_path = path.with_suffix(".run.json")
    summary: dict[str, object] = {}
    if summary_path.exists():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            summary = loaded
    return histogram.astype(np.uint32), {
        "capture_file": path.name,
        "format": info.format_name,
        "records": info.total_events,
        "continuity": summary.get("continuity"),
        "duration_s": summary.get("duration_s"),
    }


def _render_analysis(
    args: argparse.Namespace,
    raw_histogram: np.ndarray,
    *,
    label: str,
    elapsed_s: float,
    fingerprint: dict[str, FingerprintValue],
    readout: str,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Render calibration and peak-fit dialogs from one stopped MCA spectrum."""
    calibration_dialog: EnergyCalibrationDialog | None = None
    peak_dialog: McaPeakAnalysisDialog | None = None
    try:
        total_counts = int(raw_histogram.sum(dtype=np.uint64))
        _log(f"Frozen live Na-22 spectrum contains {total_counts:,} counts")
        raw_spectrum = Spectrum.create(
            label=label,
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
                "source": "Na-22",
                "readout": readout,
                **(provenance or {}),
            },
            poisson_counts=True,
        )

        range_511 = tuple(args.range_511)
        range_1274 = tuple(args.range_1274)
        guess_511 = _peak_guess(raw_histogram, range_511)
        guess_1274 = _peak_guess(raw_histogram, range_1274)
        fit_511_reference = _fit_reference_peak(
            raw_spectrum,
            guess_511,
            args.fit_half_width_511,
        )
        fit_1274_reference = _fit_reference_peak(
            raw_spectrum,
            guess_1274,
            args.fit_half_width_1274,
        )
        channel_511 = fit_511_reference.peaks[0].center
        channel_1274 = fit_1274_reference.peaks[0].center
        _log(
            "Reference centres: "
            f"511 keV at {channel_511:.3f}, 1274.5 keV at {channel_1274:.3f}"
        )

        snapshot = SpectrumSnapshot.create(
            channel=args.channel,
            counts=raw_histogram,
            label=label,
            elapsed_s=elapsed_s,
            live=False,
            fingerprint=fingerprint,
        )
        frozen_controller = _FrozenMcaController(args.channel, snapshot)
        calibration_dialog = EnergyCalibrationDialog([frozen_controller])
        calibration_dialog._add_point(channel_511)
        calibration_dialog._add_point(channel_1274)
        workspace = calibration_dialog._workspace()
        if workspace is None or len(workspace.points) != 2:
            raise RuntimeError("could not create calibration reference points")
        workspace.points[0].label = "511"
        workspace.points[0].energy_kev = args.energy_511
        workspace.points[1].label = "1274.5"
        workspace.points[1].energy_kev = args.energy_1274
        calibration_dialog._rebuild_table()
        calibration_dialog._refit()
        calibration = workspace.fit
        if calibration is None:
            raise RuntimeError(f"energy calibration failed: {calibration_dialog.fit_status.text()}")
        calibration_dialog.plot.setXRange(
            max(0.0, channel_511 - 600.0),
            channel_1274 + 600.0,
            padding=0.02,
        )
        _save_widget(
            calibration_dialog,
            args.output_dir / "mca-energy-calibration-na22-live.png",
            (1560, 900),
        )

        x: np.ndarray = np.arange(len(raw_histogram), dtype=np.float64)
        calibrated_spectrum = Spectrum.create(
            label=label,
            x=x,
            counts=raw_histogram,
            axis_unit="channel",
            source="mca",
            metadata=raw_spectrum.metadata,
            energy_kev=np.asarray(
                calibration.energy_at_binning(x, args.binning_index),
                dtype=np.float64,
            ),
            poisson_counts=True,
        )
        peak_dialog = McaPeakAnalysisDialog([])
        peak_dialog._add_spectrum(calibrated_spectrum)
        fit_low = float(channel_511 - args.fit_half_width_511)
        fit_high = float(channel_511 + args.fit_half_width_511)
        peak_dialog.fit_region.setRegion((fit_low, fit_high))
        peak_dialog.peak_count.setValue(1)
        peak_dialog.background_combo.setCurrentIndex(
            peak_dialog.background_combo.findData("linear")
        )
        peak_dialog.statistic_combo.setCurrentIndex(
            peak_dialog.statistic_combo.findData("poisson")
        )
        peak_dialog._suggest_peaks()
        peak_spec = PeakFitSpec(
            peak_count=1,
            background="linear",
            statistic="poisson",
            fit_min=fit_low,
            fit_max=fit_high,
            peak_centers=(channel_511,),
        )
        peak_fit = fit_spectrum_peaks(calibrated_spectrum, peak_spec)
        peak_dialog._fit_spectrum_id = calibrated_spectrum.spectrum_id
        peak_dialog._last_fit_spec = peak_spec
        peak_dialog._fit_completed(peak_fit)
        peak_dialog.side_tabs.setCurrentIndex(2)
        peak_dialog.spectrum_plot.setXRange(fit_low, fit_high, padding=0.03)
        _save_widget(
            peak_dialog,
            args.output_dir / "mca-peak-fit-na22-511-live.png",
            (1560, 1000),
        )

        peak = peak_fit.peaks[0]
        return {
            "endpoint": f"{args.host}:{args.port}",
            "channel": args.channel,
            "source": "Na-22",
            "trigger_level": args.threshold,
            "binning": 1 << args.binning_index,
            "readout": readout,
            "spectrum_total_counts": total_counts,
            "provenance": provenance or {},
            "calibration": {
                "reference_channels": [channel_511, channel_1274],
                "reference_energies_kev": [args.energy_511, args.energy_1274],
                "coefficients_kev": calibration.coefficients_kev,
                "rms_residual_kev": calibration.rms_residual_kev,
                "maximum_residual_kev": calibration.max_residual_kev,
            },
            "peak_fit_511": {
                "range": [fit_low, fit_high],
                "center_channel": peak.center,
                "center_kev": peak.energy_kev,
                "fwhm_channel": peak.fwhm,
                "fwhm_kev": peak.energy_fwhm_kev,
                "resolution_percent": peak.resolution_percent,
                "reduced_chi_square": peak_fit.reduced_chi_square,
                "statistic": peak_fit.statistic,
                "warnings": peak_fit.warnings,
            },
        }
    finally:
        if peak_dialog is not None:
            peak_dialog.close_without_prompt()
        if calibration_dialog is not None:
            calibration_dialog.close_without_prompt()


def run_capture(args: argparse.Namespace) -> dict[str, object]:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    if args.input_bin is not None:
        histogram, provenance = _load_recorded_histogram(args.input_bin)
        if provenance.get("continuity") != "verified":
            raise ValueError("the retained NDMA run must have verified continuity")
        duration = provenance.get("duration_s")
        elapsed_s = (
            float(duration)
            if isinstance(duration, (int, float)) and not isinstance(duration, bool)
            else 0.0
        )
        return _render_analysis(
            args,
            histogram,
            label=f"Recorded live Na-22 .128 CH{args.channel} spectrum",
            elapsed_s=elapsed_s,
            fingerprint={
                "binning": args.binning_index,
                "pulse_polarity": 0,
                "low_pass_preset": 0,
                "trapezoid_enabled": False,
                "trapezoid_r_ns": 64,
                "trapezoid_m_ns": 64,
                "trapezoid_t_ns": 96.0,
                "trapezoid_e_ns": 256,
                "trapezoid_ft": 0,
            },
            readout="recorded IIO list-mode NDMA",
            provenance=provenance,
        )
    window: MainAppWindow | None = None
    calibration_dialog: EnergyCalibrationDialog | None = None
    peak_dialog: McaPeakAnalysisDialog | None = None
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
            (ui.cbDmaEnable, "setChecked", False),
            (ui.spinTimeLimit, "setValue", 0),
            (ui.spinRefreshRate, "setValue", 5),
        ):
            _set_control(control, method, value)

        range_511 = tuple(args.range_511)
        range_1274 = tuple(args.range_1274)
        window.ui.mainTabs.setCurrentWidget(window.ui.tabMCA)
        QTest.mouseClick(ui.btnStart, Qt.MouseButton.LeftButton)
        _wait_until(lambda: mca._worker is not None, 15, "MCA polling start")
        last_progress = 0.0

        def spectrum_is_ready() -> bool:
            nonlocal last_progress
            histogram = mca._last_histogram
            now = time.monotonic()
            if histogram is not None and now - last_progress >= 10.0:
                _log(
                    "Live spectrum: "
                    f"511-window max={_range_maximum(histogram, range_511):.0f}, "
                    f"1274-window max={_range_maximum(histogram, range_1274):.0f}, "
                    f"total={int(np.sum(histogram, dtype=np.uint64)):,}"
                )
                last_progress = now
            return bool(
                histogram is not None
                and _range_maximum(histogram, range_511) >= args.minimum_511_count
                and _range_maximum(histogram, range_1274) >= args.minimum_1274_count
            )

        _wait_until(
            spectrum_is_ready,
            args.timeout,
            "Na-22 511 and 1274.5 keV peaks",
        )
        QTest.mouseClick(ui.btnStop, Qt.MouseButton.LeftButton)
        _wait_until(lambda: mca._worker_thread is None, 10, "MCA polling stop")

        raw_histogram = np.asarray(mca._last_histogram, dtype=np.uint32).copy()
        total_counts = int(raw_histogram.sum(dtype=np.uint64))
        _log(f"Frozen live Na-22 spectrum contains {total_counts:,} counts")
        raw_spectrum = Spectrum.create(
            label=f"Live Na-22 .128 CH{args.channel} stopped spectrum",
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
                "source": "Na-22",
            },
            poisson_counts=True,
        )

        guess_511 = _peak_guess(raw_histogram, range_511)
        guess_1274 = _peak_guess(raw_histogram, range_1274)
        fit_511_reference = _fit_reference_peak(
            raw_spectrum,
            guess_511,
            args.fit_half_width_511,
        )
        fit_1274_reference = _fit_reference_peak(
            raw_spectrum,
            guess_1274,
            args.fit_half_width_1274,
        )
        channel_511 = fit_511_reference.peaks[0].center
        channel_1274 = fit_1274_reference.peaks[0].center
        _log(
            "Reference centres: "
            f"511 keV at {channel_511:.3f}, 1274.5 keV at {channel_1274:.3f}"
        )

        fingerprint = mca.energy_calibration_fingerprint()
        snapshot = SpectrumSnapshot.create(
            channel=args.channel,
            counts=raw_histogram,
            label=f"Live Na-22 CH{args.channel}, stopped",
            elapsed_s=float(mca._last_elapsed_s),
            live=False,
            fingerprint=fingerprint,
        )
        frozen_controller = _FrozenMcaController(args.channel, snapshot)
        calibration_dialog = EnergyCalibrationDialog([frozen_controller])
        calibration_dialog._add_point(channel_511)
        calibration_dialog._add_point(channel_1274)
        workspace = calibration_dialog._workspace()
        if workspace is None or len(workspace.points) != 2:
            raise RuntimeError("could not create calibration reference points")
        workspace.points[0].label = "511"
        workspace.points[0].energy_kev = args.energy_511
        workspace.points[1].label = "1274.5"
        workspace.points[1].energy_kev = args.energy_1274
        calibration_dialog._rebuild_table()
        calibration_dialog._refit()
        calibration = workspace.fit
        if calibration is None:
            raise RuntimeError(f"energy calibration failed: {calibration_dialog.fit_status.text()}")
        calibration_dialog.plot.setXRange(
            max(0.0, channel_511 - 600.0),
            channel_1274 + 600.0,
            padding=0.02,
        )
        _save_widget(
            calibration_dialog,
            args.output_dir / "mca-energy-calibration-na22-live.png",
            (1560, 900),
        )

        x: np.ndarray = np.arange(len(raw_histogram), dtype=np.float64)
        calibrated_spectrum = Spectrum.create(
            label=f"Live Na-22 .128 CH{args.channel} stopped spectrum",
            x=x,
            counts=raw_histogram,
            axis_unit="channel",
            source="mca",
            metadata=raw_spectrum.metadata,
            energy_kev=np.asarray(
                calibration.energy_at_binning(x, args.binning_index),
                dtype=np.float64,
            ),
            poisson_counts=True,
        )
        peak_dialog = McaPeakAnalysisDialog([])
        peak_dialog._add_spectrum(calibrated_spectrum)
        fit_low = float(channel_511 - args.fit_half_width_511)
        fit_high = float(channel_511 + args.fit_half_width_511)
        peak_dialog.fit_region.setRegion((fit_low, fit_high))
        peak_dialog.peak_count.setValue(1)
        peak_dialog.background_combo.setCurrentIndex(
            peak_dialog.background_combo.findData("linear")
        )
        peak_dialog.statistic_combo.setCurrentIndex(
            peak_dialog.statistic_combo.findData("poisson")
        )
        peak_dialog._suggest_peaks()
        peak_spec = PeakFitSpec(
            peak_count=1,
            background="linear",
            statistic="poisson",
            fit_min=fit_low,
            fit_max=fit_high,
            peak_centers=(channel_511,),
        )
        peak_fit = fit_spectrum_peaks(calibrated_spectrum, peak_spec)
        peak_dialog._fit_spectrum_id = calibrated_spectrum.spectrum_id
        peak_dialog._last_fit_spec = peak_spec
        peak_dialog._fit_completed(peak_fit)
        peak_dialog.side_tabs.setCurrentIndex(2)
        peak_dialog.spectrum_plot.setXRange(fit_low, fit_high, padding=0.03)
        _save_widget(
            peak_dialog,
            args.output_dir / "mca-peak-fit-na22-511-live.png",
            (1560, 1000),
        )

        peak = peak_fit.peaks[0]
        return {
            "endpoint": f"{args.host}:{args.port}",
            "channel": args.channel,
            "source": "Na-22",
            "trigger_level": args.threshold,
            "binning": 1 << args.binning_index,
            "readout": "IIO histogram polling",
            "spectrum_total_counts": total_counts,
            "calibration": {
                "reference_channels": [channel_511, channel_1274],
                "reference_energies_kev": [args.energy_511, args.energy_1274],
                "coefficients_kev": calibration.coefficients_kev,
                "rms_residual_kev": calibration.rms_residual_kev,
                "maximum_residual_kev": calibration.max_residual_kev,
            },
            "peak_fit_511": {
                "range": [fit_low, fit_high],
                "center_channel": peak.center,
                "center_kev": peak.energy_kev,
                "fwhm_channel": peak.fwhm,
                "fwhm_kev": peak.energy_fwhm_kev,
                "resolution_percent": peak.resolution_percent,
                "reduced_chi_square": peak_fit.reduced_chi_square,
                "statistic": peak_fit.statistic,
                "warnings": peak_fit.warnings,
            },
        }
    finally:
        if peak_dialog is not None:
            peak_dialog.close_without_prompt()
        if calibration_dialog is not None:
            calibration_dialog.close_without_prompt()
        if mca is not None:
            mca.stop_dma_sync()
            mca._mca.stop()
        if window is not None:
            for controller in window._controller._mca_controllers:
                controller.stop_dma_sync()
                try:
                    controller.stop_worker_sync()
                except RuntimeError:
                    # Qt may already have deleted the worker during window shutdown.
                    pass
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
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--threshold", type=int, default=-512)
    parser.add_argument("--binning-index", type=int, default=5)
    parser.add_argument("--range-511", nargs=2, type=int, default=(1200, 2200))
    parser.add_argument("--range-1274", nargs=2, type=int, default=(3200, 4600))
    parser.add_argument("--minimum-511-count", type=int, default=1000)
    parser.add_argument("--minimum-1274-count", type=int, default=150)
    parser.add_argument("--fit-half-width-511", type=int, default=320)
    parser.add_argument("--fit-half-width-1274", type=int, default=420)
    parser.add_argument("--energy-511", type=float, default=511.0)
    parser.add_argument("--energy-1274", type=float, default=1274.537)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--input-bin", type=Path)
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
