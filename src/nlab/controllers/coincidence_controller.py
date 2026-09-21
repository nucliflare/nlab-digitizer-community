"""Two-channel, software-synchronized MCA coincidence measurement view."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.coincidence import TICK_NS, CoincidenceSettings, CoincidenceSnapshot
from nlab.controllers.global_controller import GlobalController
from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.digitizer import Digitizer
from nlab.hardware.digitizer.dma import IIOMcaDmaStreamer, McaEventBuffer
from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode
from nlab.utils.settings_io import write_configuration
from nlab.views.plot_viewbox import ModifierZoomViewBox
from nlab.workers.coincidence_worker import CoincidenceAnalysisThread

log = logging.getLogger(__name__)
_CHANNELS = (0, 1)


class CoincidenceController(QWidget):
    """Own both MCA DMA sessions and the shared start until both are drained."""

    def __init__(
        self,
        devices: list[Digitizer],
        mca_views: list[MCAController],
        global_view: GlobalController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if len(devices) < 2 or len(mca_views) < 2:
            raise ValueError("coincidence requires two MCA channels")
        self._devices = devices[:2]
        self._mca_views = mca_views[:2]
        self._global_view = global_view
        self._sync = devices[0].mca.sync
        self._state = "idle"
        self._run_error: str | None = None
        self._session_id: str | None = None
        self._session_manifest: Path | None = None
        self._run_mode = McaDmaOutputMode.BINARY
        self._files: tuple[Path | None, Path | None] = (None, None)
        self._buffers: tuple[McaEventBuffer, McaEventBuffer] | None = None
        self._analysis: CoincidenceAnalysisThread | None = None
        self._last_rendered_snapshot: CoincidenceSnapshot | None = None
        self._ready: set[int] = set()
        self._finished: set[int] = set()
        self._prior_ext: tuple[bool, bool] | None = None
        self._prior_source: int | None = None
        self._prior_dma_check: tuple[bool, bool] | None = None
        self._current_settings = CoincidenceSettings()
        self._analysis_epochs: list[dict[str, object]] = []
        self._build_ui()
        self._connect_signals()
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(100)
        self._display_timer.timeout.connect(self._render)
        self._display_timer.start()
        self._arm_timer = QTimer(self)
        self._arm_timer.setSingleShot(True)
        self._arm_timer.timeout.connect(
            lambda: self._begin_stop("Timed out waiting for both DMA readers")
        )
        self._finish_timer = QTimer(self)
        self._finish_timer.setInterval(50)
        self._finish_timer.timeout.connect(self._finish_if_ready)
        self.refresh_dma_output_settings()
        self._update_rule()
        self._render_roi_labels()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        splitter = QSplitter()
        root.addWidget(splitter)
        controls = QWidget()
        controls.setMinimumWidth(285)
        controls.setMaximumWidth(380)
        left = QVBoxLayout(controls)
        splitter.addWidget(controls)

        run_box = QGroupBox("Synchronized acquisition")
        run_layout = QVBoxLayout(run_box)
        buttons = QHBoxLayout()
        self.btnStart = QPushButton("Start coincidence")
        self.btnStop = QPushButton("Stop")
        self.btnStop.setEnabled(False)
        buttons.addWidget(self.btnStart)
        buttons.addWidget(self.btnStop)
        run_layout.addLayout(buttons)
        self.duration = QSpinBox()
        self.duration.setRange(0, 86_400)
        self.duration.setSuffix(" s")
        self.duration.setToolTip("Common MCA acquisition limit; 0 runs until stopped.")
        run_layout.addWidget(QLabel("Duration (both channels):"))
        run_layout.addWidget(self.duration)
        self.status = QLabel("Ready; both MCA channels must be idle.")
        self.status.setWordWrap(True)
        run_layout.addWidget(self.status)
        left.addWidget(run_box)

        gate_box = QGroupBox("Energy gates from MCA histograms")
        gate_layout = QVBoxLayout(gate_box)
        self.use_roi = (QCheckBox("Use CH0 MCA ROI"), QCheckBox("Use CH1 MCA ROI"))
        self.roi_label = (QLabel(), QLabel())
        for check, label in zip(self.use_roi, self.roi_label, strict=True):
            check.setChecked(True)
            gate_layout.addWidget(check)
            gate_layout.addWidget(label)
        hint = QLabel(
            "A hidden MCA ROI means the full energy range. Gates use MCA histogram channels, "
            "not calibrated keV."
        )
        hint.setWordWrap(True)
        gate_layout.addWidget(hint)
        left.addWidget(gate_box)

        logic_box = QGroupBox("Event logic")
        logic_form = QFormLayout(logic_box)
        self.operator = QComboBox()
        self.operator.addItems(["AND", "OR", "XOR"])
        self.not_ch0 = QCheckBox("NOT CH0 (veto)")
        self.not_ch1 = QCheckBox("NOT CH1 (veto)")
        self.expression = QLabel()
        self.expression.setWordWrap(True)
        logic_form.addRow("Operator:", self.operator)
        logic_form.addRow(self.not_ch0)
        logic_form.addRow(self.not_ch1)
        logic_form.addRow(self.expression)
        left.addWidget(logic_box)

        timing_box = QGroupBox("Timing")
        timing_form = QFormLayout(timing_box)
        self.timing_mode = QComboBox()
        self.timing_mode.addItem("Auto (CFD when both enabled)", "auto")
        self.timing_mode.addItem("Coarse (8 ns)", "coarse")
        self.timing_mode.addItem("CFD fine (1 ns, experimental)", "cfd")
        self.timing_mode.setToolTip(
            "CFD mode uses timestamp + signed zero-crossing offset + Q2.14 "
            "interpolation. The upstream timestamp anchor is unverified; "
            "compare against coarse timing during live validation."
        )
        timing_form.addRow("Precision:", self.timing_mode)
        self.low = QSpinBox()
        self.high = QSpinBox()
        for spin in (self.low, self.high):
            spin.setRange(-1_000, 1_000)
            spin.setSingleStep(TICK_NS)
            spin.setSuffix(" ns")
        self.low.setValue(-48)
        self.high.setValue(48)
        self.offset = QSpinBox()
        self.offset.setRange(-100_000, 100_000)
        self.offset.setSingleStep(TICK_NS)
        self.offset.setSuffix(" ns")
        self.offset.setToolTip("Offset added to CH1 timestamps; 8 ns resolution.")
        timing_form.addRow("Lower Δt:", self.low)
        timing_form.addRow("Upper Δt:", self.high)
        timing_form.addRow("CH1 offset:", self.offset)
        self.timing_hint = QLabel()
        self.timing_hint.setWordWrap(True)
        timing_form.addRow(self.timing_hint)
        left.addWidget(timing_box)

        output_box = QGroupBox("Raw DMA recording")
        output_layout = QVBoxLayout(output_box)
        self.output_label = QLabel()
        self.output_label.setWordWrap(True)
        output_layout.addWidget(self.output_label)
        self.btnFolder = QPushButton("Measurement location…")
        output_layout.addWidget(self.btnFolder)
        note = QLabel(
            "Each run records both complete raw streams in the selected MCA DMA format. "
            "ROI and logic affect only the live analysis."
        )
        note.setWordWrap(True)
        output_layout.addWidget(note)
        left.addWidget(output_box)
        left.addStretch(1)

        plots = QWidget()
        plot_layout = QVBoxLayout(plots)
        self.delay_plot, self.delay_curve = self._plot(
            "Signed CH1 − CH0 delay", "Δt", "ns", "#7353a6"
        )
        self.energy_plot0, self.energy_curve0 = self._plot(
            "Accepted CH0 events", "MCA channel", None, "#1f77b4"
        )
        self.energy_plot1, self.energy_curve1 = self._plot(
            "Accepted CH1 events", "MCA channel", None, "#d28b38"
        )
        self._roi_regions = (
            self._add_roi_region(self.energy_plot0),
            self._add_roi_region(self.energy_plot1),
        )
        for plot in (self.delay_plot, self.energy_plot0, self.energy_plot1):
            plot_layout.addWidget(plot, 1)
        self.counts_label = QLabel("Pairs: 0  •  CH0: 0  •  CH1: 0")
        self.counts_label.setWordWrap(True)
        plot_layout.addWidget(self.counts_label)
        splitter.addWidget(plots)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    @staticmethod
    def _plot(title: str, axis: str, unit: str | None, color: str) -> tuple[Any, Any]:
        plot = pg.PlotWidget(viewBox=ModifierZoomViewBox())
        plot.setBackground("#f8f9fa")
        plot.setTitle(title)
        plot.setLabel("left", "Counts")
        plot.setLabel("bottom", axis, units=unit)
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.showAxis("top")
        plot.showAxis("right")
        curve = plot.plot(pen=pg.mkPen(color, width=1.5), stepMode="center")
        return plot, curve

    @staticmethod
    def _add_roi_region(plot: Any) -> pg.LinearRegionItem:
        region = pg.LinearRegionItem(values=(0, 1), movable=False)
        region.setZValue(5)
        plot.addItem(region, ignoreBounds=True)
        region.setVisible(False)
        return region

    @staticmethod
    def _style_roi_region(region: pg.LinearRegionItem, color: str, active: bool) -> None:
        shade = pg.mkColor(color)
        shade.setAlpha(36 if active else 10)
        region.setBrush(pg.mkBrush(shade))
        boundary = pg.mkColor(color)
        boundary.setAlpha(185 if active else 90)
        pen = pg.mkPen(boundary, width=1.5, style=Qt.PenStyle.DashLine)
        for line in region.lines:
            line.setPen(pen)

    def _connect_signals(self) -> None:
        self.btnStart.clicked.connect(self.start)
        self.btnStop.clicked.connect(lambda: self._begin_stop(None))
        self.btnFolder.clicked.connect(self._choose_folder)
        self.timing_mode.currentIndexChanged.connect(self._analysis_settings_changed)
        self.operator.currentTextChanged.connect(self._update_rule)
        self.not_ch0.toggled.connect(lambda checked: self._on_not_changed(0, checked))
        self.not_ch1.toggled.connect(lambda checked: self._on_not_changed(1, checked))
        for control in (*self.use_roi, self.low, self.high, self.offset):
            if isinstance(control, QCheckBox):
                control.toggled.connect(self._analysis_settings_changed)
            else:
                control.valueChanged.connect(self._analysis_settings_changed)
        for spin in (self.low, self.high, self.offset):
            spin.editingFinished.connect(
                lambda selected=spin: selected.setValue(round(selected.value() / TICK_NS) * TICK_NS)
            )
        for channel, mca in enumerate(self._mca_views):
            mca.ui.cbCfdEnable.toggled.connect(self._analysis_settings_changed)
            mca.roi_changed.connect(self._analysis_settings_changed)
            mca.roi_preview_changed.connect(self._render_roi_labels)
            mca.coincidence_ready.connect(self._on_channel_ready)
            mca.coincidence_finished.connect(self._on_channel_finished)
            mca.coincidence_error.connect(self._on_channel_error)
            mca.coincidence_stop_requested.connect(lambda ch=channel: self._begin_stop(None))

    def _on_not_changed(self, channel: int, checked: bool) -> None:
        if checked:
            other = self.not_ch1 if channel == 0 else self.not_ch0
            other.setChecked(False)
        self._update_rule()

    def _update_rule(self, *_args: object) -> None:
        is_and = self.operator.currentText() == "AND"
        if not is_and:
            self.not_ch0.setChecked(False)
            self.not_ch1.setChecked(False)
        self.not_ch0.setEnabled(is_and)
        self.not_ch1.setEnabled(is_and)
        ch0 = "NOT CH0" if self.not_ch0.isChecked() else "CH0"
        ch1 = "NOT CH1" if self.not_ch1.isChecked() else "CH1"
        self.expression.setText(f"{ch0} {self.operator.currentText()} {ch1}")
        self.delay_plot.setTitle(
            "Signed CH1 − CH0 delay"
            if is_and and not (self.not_ch0.isChecked() or self.not_ch1.isChecked())
            else "Accepted events over elapsed time"
        )
        self.delay_plot.setLabel(
            "bottom",
            "Δt"
            if is_and and not (self.not_ch0.isChecked() or self.not_ch1.isChecked())
            else "Elapsed time",
            units="ns"
            if is_and and not (self.not_ch0.isChecked() or self.not_ch1.isChecked())
            else "s",
        )
        self._analysis_settings_changed()

    def _settings(self) -> CoincidenceSettings:
        low = self.low.value()
        high = self.high.value()
        offset = self.offset.value()
        if low >= high:
            raise ValueError("Lower timing boundary must be below upper boundary")
        if any(value % TICK_NS for value in (low, high, offset)):
            raise ValueError("Timing bounds and CH1 offset must be multiples of 8 ns")
        mode = self.timing_mode.currentData()
        cfd_ready = all(mca.ui.cbCfdEnable.isChecked() for mca in self._mca_views)
        if mode == "cfd" and not cfd_ready:
            raise ValueError("Enable CFD on both MCA channels for fine timing")
        fine_timing = mode == "cfd" or (mode == "auto" and cfd_ready)
        rois = tuple(
            mca.coincidence_roi() if check.isChecked() else None
            for mca, check in zip(self._mca_views, self.use_roi, strict=True)
        )
        return CoincidenceSettings(
            operator=self.operator.currentText(),
            not_ch0=self.not_ch0.isChecked(),
            not_ch1=self.not_ch1.isChecked(),
            low_tick=low // TICK_NS,
            high_tick=high // TICK_NS,
            offset_ch1_tick=offset // TICK_NS,
            roi_ch0=rois[0],
            roi_ch1=rois[1],
            energy_bin_ch0=self._mca_views[0].coincidence_energy_bin,
            energy_bin_ch1=self._mca_views[1].coincidence_energy_bin,
            fine_timing=fine_timing,
        )

    def _render_roi_labels(self) -> None:
        for channel, (mca, use, label, region, color) in enumerate(
            zip(
                self._mca_views,
                self.use_roi,
                self.roi_label,
                self._roi_regions,
                ("#1f77b4", "#d28b38"),
                strict=True,
            )
        ):
            roi = mca.coincidence_roi()
            if roi is None:
                region.setVisible(False)
                label.setText(f"CH{channel}: full MCA-channel range (MCA ROI hidden)")
                continue
            # The ROI is inclusive in MCA-bin coordinates; step histograms use
            # bin edges, so the right-hand boundary is the edge after high.
            region.setRegion((roi[0], roi[1] + 1))
            self._style_roi_region(region, color, use.isChecked())
            region.setVisible(True)
            state = "active gate" if use.isChecked() else "reference only; gate off"
            label.setText(f"CH{channel}: bins {roi[0]}–{roi[1]} ({state})")

    def _analysis_settings_changed(self, *_args: object) -> None:
        self._render_roi_labels()
        try:
            settings = self._settings()
        except ValueError as exc:
            self.status.setText(str(exc))
            self.timing_hint.setText("Fine timing requires valid CFD on both channels.")
            return
        self._current_settings = settings
        self.timing_hint.setText(
            "CFD fine timing: provisional timestamp correction; 1 ns Δt bins. "
            "Events without valid CFD are excluded from live analysis."
            if settings.fine_timing
            else "Coarse timestamps; 8 ns Δt bins."
        )
        if self._analysis is not None:
            self._last_rendered_snapshot = self._analysis.result()[0]
        if self._analysis is not None and self._state in {"arming", "running"}:
            self._analysis_epochs.append(
                {"changed_utc": datetime.now(UTC).isoformat(), "settings": self._describe(settings)}
            )
            self._analysis.request_settings(settings)
            self.status.setText(
                "Analysis reset for updated ROI, rule, or timing window; raw recording continues."
            )
        self.delay_curve.setData([], [])
        self.energy_curve0.setData([], [])
        self.energy_curve1.setData([], [])

    def refresh_dma_output_settings(self) -> None:
        settings = QSettings()
        mode = MCAController._output_mode()
        folder = str(settings.value("dma/save_folder", "measurements"))
        self.output_label.setText(
            f"{mode.value.upper()} • {folder}"
            if mode is not McaDmaOutputMode.ONLINE
            else "Online only • no files"
        )

    def _choose_folder(self) -> None:
        current = str(QSettings().value("dma/save_folder", "measurements"))
        chosen = QFileDialog.getExistingDirectory(self, "Coincidence measurement location", current)
        if chosen:
            QSettings().setValue("dma/save_folder", chosen)
            self.refresh_dma_output_settings()

    def _prepare_paths(self, mode: McaDmaOutputMode) -> tuple[Path | None, Path | None]:
        if mode is McaDmaOutputMode.ONLINE:
            self._session_manifest = None
            return None, None
        folder = Path(str(QSettings().value("dma/save_folder", "measurements")))
        folder.mkdir(parents=True, exist_ok=True)
        assert self._session_id is not None
        stem = f"coincidence_{self._session_id}"
        paths = tuple(folder / f"{stem}_ch{ch}{mode.extension}" for ch in _CHANNELS)
        manifest = folder / f"{stem}_session.yaml"
        if manifest.exists() or any(path.exists() for path in paths):
            raise FileExistsError("Coincidence output name already exists; retry Start")
        self._session_manifest = manifest
        return paths  # type: ignore[return-value]

    def _manifest(self, *, status: str) -> dict[str, object]:
        return {
            "kind": "two_channel_coincidence_session",
            "session_id": self._session_id,
            "status": status,
            "format": self._run_mode.value,
            "files": {
                f"ch{ch}": str(path.name) if path is not None else None
                for ch, path in enumerate(self._files)
            },
            "analysis": self._describe(self._current_settings),
            "analysis_epochs": self._analysis_epochs,
            "channels": (
                {
                    f"ch{channel}": {
                        "continuity": summary.continuity,
                        "records": summary.records,
                        "diagnostics": summary.diagnostics,
                        "error": summary.error,
                    }
                    for channel, view in enumerate(self._mca_views)
                    if (summary := view.coincidence_run_summary) is not None
                }
                if status != "arming"
                else {}
            ),
            "error": self._run_error,
        }

    @staticmethod
    def _describe(settings: CoincidenceSettings) -> dict[str, object]:
        return {
            "operator": settings.operator,
            "not_ch0": settings.not_ch0,
            "not_ch1": settings.not_ch1,
            "window_ns": [settings.low_tick * TICK_NS, settings.high_tick * TICK_NS],
            "offset_ch1_ns": settings.offset_ch1_tick * TICK_NS,
            "timing_source": "cfd_provisional_v1" if settings.fine_timing else "coarse",
            "delta_t_bin_ns": settings.bin_width_ns,
            "cfd_correction": (
                "timestamp + int8(zc_offset) + signed_q2_14(zc_estimation)"
                if settings.fine_timing else None
            ),
            "roi_bins": {"ch0": settings.roi_ch0, "ch1": settings.roi_ch1},
            "energy_bin": {"ch0": settings.energy_bin_ch0, "ch1": settings.energy_bin_ch1},
            "dma_energy_to_mca_channel": "trapezoid_energy >> 2",
            "pairing_policy": "A-ordered nearest available B, one-to-one",
        }

    def start(self) -> None:
        if self._state != "idle":
            return
        try:
            settings = self._settings()
            if settings.fine_timing and not all(
                device.mca.filters.cfd.get_enable() for device in self._devices
            ):
                raise RuntimeError("CFD fine timing requires hardware CFD enabled on both MCAs")
            self._current_settings = settings
            self._analysis_epochs = [
                {"changed_utc": datetime.now(UTC).isoformat(), "settings": self._describe(settings)}
            ]
            if any(mca.coincidence_busy for mca in self._mca_views):
                raise RuntimeError(
                    "Stop both existing MCA measurements before starting coincidence"
                )
            if any(not isinstance(device.mca_dma, IIOMcaDmaStreamer) for device in self._devices):
                raise RuntimeError("Both channels need IIO list-mode DMA")
            self._session_id = (
                datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:6]
            )
            mode = MCAController._output_mode()
            self._run_mode = mode
            self._files = self._prepare_paths(mode)
            self._run_error = None
            self._ready.clear()
            self._finished.clear()
            self._prior_ext = tuple(device.mca.get_ext_trig_enable() for device in self._devices)  # type: ignore[assignment]
            self._prior_dma_check = tuple(
                view.ui.cbDmaEnable.isChecked() for view in self._mca_views
            )  # type: ignore[assignment]
            self._prior_source = self._sync.get_trig_src()
            self._state = "arming"
            self.btnStart.setEnabled(False)
            self.btnStop.setEnabled(True)
            self.btnFolder.setEnabled(False)
            self._global_view.set_coincidence_locked(True)
            self.status.setText("Preparing shared software start…")
            # PetaLinux b08ad28 vdpp-sync-trigger.c: this is a level-sensitive
            # common start controller, not an IIO trigger provider.
            # Source writes require the gate off; keep LOW until *both* IIO
            # reader threads report buffer armed and pulse processor enabled.
            self._sync.set_enable(False)
            self._sync.set_sw_trig(0)
            self._sync.set_trig_src(0)
            for device, view in zip(self._devices, self._mca_views, strict=True):
                device.mca.stop()
                device.mca.set_ext_trig_enable(True)
                view.ui.cbExtTrigger.blockSignals(True)
                view.ui.cbExtTrigger.setChecked(True)
                view.ui.cbExtTrigger.blockSignals(False)
                view.ui.spinTimeLimit.setValue(self.duration.value())
            self._sync.set_enable(True)
            buffers = (McaEventBuffer(), McaEventBuffer())
            self._buffers = buffers
            self._analysis = CoincidenceAnalysisThread(*buffers, settings)
            self._last_rendered_snapshot = None
            self._analysis.start()
            metadata = self._manifest(status="arming")
            if self._session_manifest is not None:
                write_configuration(self._session_manifest, metadata)
            for channel, view in enumerate(self._mca_views):
                view.start_coincidence_capture(buffers[channel], self._files[channel], metadata)
            self._arm_timer.start(10_000)
            self.status.setText("Waiting for both DMA readers to arm…")
        except Exception as exc:
            log.exception("Could not start coincidence session")
            if self._state == "idle":
                self.status.setText(f"Start blocked: {exc}")
            else:
                self._begin_stop(str(exc))

    def _on_channel_ready(self, channel: int) -> None:
        if self._state != "arming":
            return
        self._ready.add(channel)
        self.status.setText(f"DMA readers ready: {len(self._ready)}/2")
        if len(self._ready) != 2:
            return
        try:
            if (
                self._sync.get_trig_src() != 0
                or not self._sync.get_enable()
                or self._sync.get_sw_trig()
            ):
                raise RuntimeError("Shared software start is not gated LOW")
            for device in self._devices:
                if not device.mca.get_global_enable() or not device.mca.get_ext_trig_enable():
                    raise RuntimeError("An MCA was not fully armed for external start")
            self._sync.set_sw_trig(1)
            self._arm_timer.stop()
            self._state = "running"
            self.status.setText("Both channels started together; live coincidence analysis active.")
        except Exception as exc:
            log.exception("Coincidence shared start failed")
            self._begin_stop(str(exc))

    def _on_channel_error(self, channel: int, message: str) -> None:
        detail = f"CH{channel}: {message}"
        if self._state == "stopping":
            self._run_error = f"{self._run_error}; {detail}" if self._run_error else detail
        else:
            self._begin_stop(detail)

    def _on_channel_finished(self, channel: int) -> None:
        self._finished.add(channel)
        if self._state == "arming":
            self._begin_stop(f"CH{channel} DMA ended before synchronized start")
        elif self._state == "running":
            # A configured time limit may stop one producer first. Its run
            # summary still decides whether the paired session is healthy.
            self._begin_stop(None)
        if self._state == "stopping" and len(self._finished) == 2:
            self._analysis_stop_after_dma()

    def _begin_stop(self, error: str | None) -> None:
        if self._state not in {"arming", "running"}:
            return
        self._state = "stopping"
        self._arm_timer.stop()
        if error is not None:
            self._run_error = error
        self.status.setText("Stopping both channels and draining DMA…")
        operations: tuple[Callable[[], None], ...] = (
            lambda: self._sync.set_enable(False),
            lambda: self._sync.set_sw_trig(0),
        )
        for operation in operations:
            try:
                operation()
            except Exception as exc:
                self._run_error = f"{self._run_error or 'Stop'}; shared trigger: {exc}"
                log.exception("Could not disarm coincidence shared trigger")
        for channel, view in enumerate(self._mca_views):
            if view.coincidence_active:
                try:
                    view.stop_coincidence_capture()
                except Exception as exc:
                    self._run_error = f"{self._run_error or 'Stop'}; CH{channel}: {exc}"
                    log.exception("Could not stop coincidence CH%d", channel)
            else:
                self._finished.add(channel)
        if len(self._finished) == 2:
            self._analysis_stop_after_dma()

    def _analysis_stop_after_dma(self) -> None:
        if self._analysis is not None:
            self._analysis.request_stop()
        self._finish_timer.start()

    def _finish_if_ready(self) -> None:
        if self._analysis is not None and self._analysis.is_alive():
            return
        self._finish_timer.stop()
        if self._analysis is not None:
            _, analysis_error = self._analysis.result()
            if analysis_error is not None:
                self._run_error = f"{self._run_error or 'Analysis'}; {analysis_error}"
        for channel, view in enumerate(self._mca_views):
            summary = view.coincidence_run_summary
            if summary is None or summary.continuity != "verified":
                detail = (
                    f"CH{channel} continuity {summary.continuity if summary else 'unavailable'}"
                )
                self._run_error = f"{self._run_error}; {detail}" if self._run_error else detail
        for channel, device in enumerate(self._devices):
            try:
                device.mca.stop()
                if self._prior_ext is not None:
                    device.mca.set_ext_trig_enable(self._prior_ext[channel])
                    trigger_check = self._mca_views[channel].ui.cbExtTrigger
                    trigger_check.blockSignals(True)
                    trigger_check.setChecked(self._prior_ext[channel])
                    trigger_check.blockSignals(False)
                if self._prior_dma_check is not None:
                    checkbox = self._mca_views[channel].ui.cbDmaEnable
                    checkbox.blockSignals(True)
                    checkbox.setChecked(self._prior_dma_check[channel])
                    checkbox.blockSignals(False)
            except Exception as exc:
                self._run_error = f"{self._run_error or 'Cleanup'}; CH{channel}: {exc}"
                log.exception("Could not restore coincidence CH%d controls", channel)
        try:
            self._sync.set_enable(False)
            self._sync.set_sw_trig(0)
            if self._prior_source is not None:
                self._sync.set_trig_src(self._prior_source)
        except Exception as exc:
            self._run_error = f"{self._run_error or 'Cleanup'}; {exc}"
            log.exception("Coincidence cleanup could not restore shared trigger")
        self._global_view.set_coincidence_locked(False)
        if self._session_manifest is not None:
            try:
                write_configuration(
                    self._session_manifest,
                    self._manifest(status="failed" if self._run_error else "complete"),
                )
            except OSError as exc:
                self._run_error = f"{self._run_error or 'Manifest'}; {exc}"
                log.exception("Could not finish coincidence session manifest")
        self._state = "idle"
        self._buffers = None
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.btnFolder.setEnabled(True)
        self.status.setText(
            f"Coincidence run failed: {self._run_error}"
            if self._run_error
            else "Coincidence run complete."
        )

    def _render(self) -> None:
        analysis = self._analysis
        if analysis is None:
            return
        snapshot, error = analysis.result()
        if error is not None and self._state in {"arming", "running"}:
            self._begin_stop(error)
        if snapshot is self._last_rendered_snapshot:
            return
        try:
            settings = self._settings()
        except ValueError:
            return
        pair_mode = settings.operator == "AND" and not (settings.not_ch0 or settings.not_ch1)
        if pair_mode:
            left_ns = settings.low_tick * TICK_NS
            self.delay_curve.setData(
                left_ns + np.arange(len(snapshot.delay_counts) + 1) * settings.bin_width_ns,
                snapshot.delay_counts,
            )
        else:
            if len(snapshot.rate_seconds):
                edges = np.append(snapshot.rate_seconds, snapshot.rate_seconds[-1] + 1)
                self.delay_curve.setData(edges, snapshot.rate_counts)
            else:
                self.delay_curve.setData([], [])
        self.energy_curve0.setData(np.arange(16_385), snapshot.energy_ch0)
        self.energy_curve1.setData(np.arange(16_385), snapshot.energy_ch1)
        self.counts_label.setText(
            f"Pairs: {snapshot.pairs:,}  •  CH0: {snapshot.accepted_ch0:,}  •  "
            f"CH1: {snapshot.accepted_ch1:,}  •  Ambiguous: {snapshot.ambiguous:,}  •  "
            f"Zero timestamps: {snapshot.zero_timestamps:,}  •  "
            f"CFD valid: {snapshot.cfd_valid:,}  •  "
            f"PSD ZC valid: {snapshot.psd_zc_valid:,}  •  "
            f"CFD skipped: {snapshot.cfd_skipped:,}  •  "
            f"ROI rejects: {snapshot.outside_roi:,}  •  "
            f"Energy beyond plot: {snapshot.energy_overflow:,}"
        )
        self._last_rendered_snapshot = snapshot

    def configuration_settings(self) -> dict[str, object]:
        return {
            "operator": self.operator.currentText(),
            "not_ch0": self.not_ch0.isChecked(),
            "not_ch1": self.not_ch1.isChecked(),
            "use_roi": [control.isChecked() for control in self.use_roi],
            "window_ns": [self.low.value(), self.high.value()],
            "offset_ch1_ns": self.offset.value(),
            "timing_mode": self.timing_mode.currentData(),
            "duration_s": self.duration.value(),
        }

    @property
    def active(self) -> bool:
        return self._state != "idle"

    def apply_configuration_settings(self, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        operator = str(settings.get("operator", "AND"))
        if operator in {"AND", "OR", "XOR"}:
            self.operator.setCurrentText(operator)
        if operator == "AND":
            self.not_ch0.setChecked(bool(settings.get("not_ch0", False)))
            self.not_ch1.setChecked(bool(settings.get("not_ch1", False)))
        roi = settings.get("use_roi")
        if isinstance(roi, list) and len(roi) == 2:
            for control, value in zip(self.use_roi, roi, strict=True):
                control.setChecked(bool(value))
        window = settings.get("window_ns")
        if isinstance(window, list) and len(window) == 2:
            self.low.setValue(int(window[0]))
            self.high.setValue(int(window[1]))
        self.offset.setValue(int(settings.get("offset_ch1_ns", 0)))
        timing_mode = str(settings.get("timing_mode", "auto"))
        mode_index = self.timing_mode.findData(timing_mode)
        if mode_index >= 0:
            self.timing_mode.setCurrentIndex(mode_index)
        self.duration.setValue(int(settings.get("duration_s", 0)))

    def request_shutdown(self) -> None:
        """Gate off both producers before main-window worker teardown."""
        self._arm_timer.stop()
        self._finish_timer.stop()
        self._display_timer.stop()
        if self._state in {"arming", "running"}:
            self._begin_stop(None)

    def finish_shutdown_sync(self) -> None:
        """Called after both MCA DMA workers have been synchronously drained."""
        if self._analysis is not None:
            self._analysis.request_stop()
            self._analysis.join(timeout=10)
            if self._analysis.is_alive():
                log.error("Coincidence analysis did not finish before shutdown")
                return
        if self._state == "stopping":
            self._finish_if_ready()
