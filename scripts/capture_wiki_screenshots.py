#!/usr/bin/env python
"""Capture reproducible GUI screenshots for the project wiki.

The script runs Qt offscreen and never opens a hardware or network connection.
Plots are populated with deterministic synthetic data so the images remain
useful in documentation without being mistaken for live measurements.

Run ``python scripts/build_ui.py`` first when the generated Qt modules are not
present, then run this script from the project environment.
"""

from __future__ import annotations

import argparse
import os
import struct
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QSettings, Signal  # noqa: E402
from PySide6.QtGui import QFont, QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication, QCheckBox, QSpinBox, QWidget  # noqa: E402

from nlab.analysis.coincidence import (  # noqa: E402
    CoincidenceAnalyzer,
    CoincidenceSettings,
)
from nlab.analysis.spectrum import Spectrum  # noqa: E402
from nlab.controllers.coincidence_controller import CoincidenceController  # noqa: E402
from nlab.controllers.current_monitor_controller import (  # noqa: E402
    CurrentMonitorController,
)
from nlab.hardware.digitizer.current_monitor import (  # noqa: E402
    ScopeCurrentAccumulator,
    ScopeCurrentRuntime,
)
from nlab.hardware.digitizer.dma import ScopeDmaGeometry  # noqa: E402
from nlab.views.connection_dialog import ConnectionDialog  # noqa: E402
from nlab.views.mca_peak_analysis_dialog import McaPeakAnalysisDialog  # noqa: E402


class _FakeScopeController(QObject):
    current_dma_state_changed = Signal(bool, str)

    def current_dma_supported(self) -> bool:
        return True

    def reset_defaults(self) -> None:
        pass


class _FakeSync:
    pass


class _FakeMcaView(QObject):
    roi_changed = Signal()
    roi_preview_changed = Signal()
    coincidence_ready = Signal(int)
    coincidence_finished = Signal(int)
    coincidence_error = Signal(int, str)
    coincidence_stop_requested = Signal()

    def __init__(self, channel: int) -> None:
        super().__init__()
        self.channel = channel
        self.coincidence_energy_bin = 2
        self.energy_calibration = None
        self.ui = SimpleNamespace(
            cbDmaEnable=QCheckBox(),
            cbExtTrigger=QCheckBox(),
            cbCfdEnable=QCheckBox(),
            spinTimeLimit=QSpinBox(),
        )

    def coincidence_roi(self) -> tuple[int, int]:
        return (2_400, 10_800) if self.channel == 0 else (3_000, 11_500)

    def energy_calibration_is_stale(self) -> bool:
        return False


def _save_widget(app: QApplication, widget: QWidget, path: Path, size: tuple[int, int]) -> None:
    widget.resize(*size)
    widget.show()
    app.processEvents()
    image = widget.grab()
    if not image.save(str(path), "PNG"):
        raise RuntimeError(f"could not save {path}")
    widget.close()
    app.processEvents()


def _load_documentation_font(app: QApplication) -> None:
    """Load one explicit font when an offscreen platform finds no system fonts."""
    candidates = (
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(candidate))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 9))
            return
    if not QFontDatabase.families():
        raise RuntimeError("Qt could not discover a usable font for screenshots")


def _capture_connection(app: QApplication, output_dir: Path) -> None:
    dialog = ConnectionDialog(
        ip="192.168.10.128",
        port=30_431,
        backend="iio",
        channels=2,
    )
    _save_widget(app, dialog, output_dir / "connection-dialog.png", (560, 300))


def _capture_current(app: QApplication, output_dir: Path) -> None:
    geometry = ScopeDmaGeometry(
        frame_samples=8_000,
        buffer_samples=2_004,
        frame_bytes=4_008,
        waveform_samples=1_999,
        sample_decimation=4,
        padding_bytes=2,
    )
    accumulator = ScopeCurrentAccumulator(bin_width_ms=100, history_seconds=10)
    expected_ticks = 20_000
    accumulator.start_session(
        geometry,
        ScopeCurrentRuntime(
            channel=0,
            ip_version=122,
            gap_cycles=18_000,
            expected_interval_ticks=expected_ticks,
            transport="IIO network",
            kernel_buffers=4,
            readbuf_batch_frames=64,
            queue_high_watermark=2,
        ),
    )
    phase = np.linspace(0.0, 2.0 * np.pi, geometry.waveform_samples, endpoint=False)
    waveform = np.rint(820 + 115 * np.sin(5 * phase) + 22 * np.sin(17 * phase)).astype("<i2")
    payload = waveform.tobytes() + bytes(geometry.padding_bytes)
    frame_period_ns = 160_000
    frames = 6_250
    first_received_ns = time.perf_counter_ns() - frames * frame_period_ns
    for index in range(frames):
        accumulator.append_frame(
            struct.pack("<Q", 1 + index * expected_ticks) + payload,
            geometry,
            first_received_ns + index * frame_period_ns,
        )
    snapshot = accumulator.snapshot(
        now_ns=first_received_ns + (frames - 1) * frame_period_ns + 1_000_000
    )
    snapshot = replace(
        snapshot,
        data_age_ns=1_000_000,
        analysis_lag_ns=180_000,
        maximum_analysis_lag_ns=310_000,
        display_p95_duration_ns=0.0,
    )

    controller = CurrentMonitorController(
        object(),  # type: ignore[arg-type]
        channel=0,
        auto_start=False,
        scope_controller=_FakeScopeController(),  # type: ignore[arg-type]
        scope_current_accumulator=accumulator,
    )
    controller.ui.comboMode.setCurrentIndex(1)
    controller.ui.spinZero.setValue(800.0)
    controller.ui.spinScale.setValue(0.025)
    controller.ui.comboUnit.setCurrentText("nA")
    controller._last_dma_plot_update_ns = 0
    controller._render_dma_snapshot(snapshot)
    controller.ui.lblStatus.setText(
        "Synthetic documentation data; no hardware or network I/O. "
        + controller.ui.lblStatus.text()
    )
    _save_widget(app, controller, output_dir / "current-monitor-dma.png", (1500, 850))


def _capture_coincidence(app: QApplication, output_dir: Path) -> None:
    sync = _FakeSync()
    devices = [
        SimpleNamespace(mca=SimpleNamespace(sync=sync)),
        SimpleNamespace(mca=SimpleNamespace(sync=sync)),
    ]
    views = [_FakeMcaView(0), _FakeMcaView(1)]
    global_view = SimpleNamespace(set_coincidence_locked=lambda _locked: None)
    controller = CoincidenceController(  # type: ignore[arg-type]
        devices,
        views,
        global_view,
    )
    controller._display_timer.stop()

    settings = CoincidenceSettings(random_sidebands=True)
    snapshot = CoincidenceAnalyzer(settings).snapshot()
    rng = np.random.default_rng(4_020)
    snapshot.prompt_matrix[:] = rng.poisson(0.08, snapshot.prompt_matrix.shape)
    snapshot.random_matrix[:] = rng.poisson(0.025, snapshot.random_matrix.shape)
    yy, xx = np.indices(snapshot.prompt_matrix.shape)
    spot = 90.0 * np.exp(-0.5 * (((xx - 185) / 11) ** 2 + ((yy - 318) / 14) ** 2))
    diagonal = 26.0 * np.exp(-0.5 * ((yy - (0.78 * xx + 120)) / 7.0) ** 2)
    snapshot.prompt_matrix[:] += np.rint(spot + diagonal).astype(snapshot.prompt_matrix.dtype)
    snapshot = replace(snapshot, pairs=184_320, random_pairs=21_840)
    controller._current_settings = settings
    controller.matrix_mode.setCurrentIndex(controller.matrix_mode.findData("corrected"))
    controller.matrix_scale.setCurrentIndex(controller.matrix_scale.findData("log"))
    controller.matrix_gate_ch0.setRegion((4_800, 7_200))
    controller.matrix_gate_ch1.setRegion((8_600, 11_400))
    controller.result_tabs.setCurrentIndex(controller._matrix_tab_index)
    controller._render_matrix(snapshot)
    controller.matrix_status.setText(
        "Synthetic documentation data; no hardware or network I/O. "
        + controller.matrix_status.text()
    )
    _save_widget(app, controller, output_dir / "coincidence-matrix.png", (1500, 900))


def _synthetic_spectrum(
    rng: np.random.Generator,
    *,
    label: str,
    scale: float,
    shift: float,
) -> Spectrum:
    x = np.arange(2_048, dtype=np.float64)
    expected = (
        18.0
        + 1_350.0 * scale * np.exp(-0.5 * ((x - (640.0 + shift)) / 18.0) ** 2)
        + 820.0 * scale * np.exp(-0.5 * ((x - (1_320.0 + shift)) / 28.0) ** 2)
        + 260.0 * scale * np.exp(-x / 900.0)
    )
    counts = rng.poisson(expected).astype(np.float64)
    return Spectrum.create(
        label=label,
        x=x,
        counts=counts,
        axis_unit="channel",
        source="mca",
        metadata={"elapsed_s": 60.0},
    )


def _capture_peak_analysis(app: QApplication, output_dir: Path) -> None:
    dialog = McaPeakAnalysisDialog([])
    rng = np.random.default_rng(1_337)
    dialog._add_spectrum(
        _synthetic_spectrum(rng, label="Synthetic CH0 spectrum", scale=1.0, shift=0.0)
    )
    dialog._add_spectrum(
        _synthetic_spectrum(
            rng,
            label="Synthetic CH1 spectrum",
            scale=0.72,
            shift=24.0,
        ),
        select=False,
    )
    dialog.fit_region.setRegion((560.0, 720.0))
    dialog._render()
    _save_widget(app, dialog, output_dir / "mca-peak-analysis.png", (1500, 850))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "images",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nlab-wiki-screenshots-") as settings_dir:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(
            QSettings.Format.IniFormat,
            QSettings.Scope.UserScope,
            settings_dir,
        )
        app = QApplication.instance() or QApplication([])
        app.setStyle("Fusion")
        _load_documentation_font(app)
        for capture in (
            _capture_connection,
            _capture_current,
            _capture_coincidence,
            _capture_peak_analysis,
        ):
            capture(app, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
