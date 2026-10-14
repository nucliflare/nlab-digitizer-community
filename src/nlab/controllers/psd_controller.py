from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QRectF, Qt, QTimer
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QWidget

from nlab.analysis.dose_calibration import PsdDoseCalibrationSnapshot
from nlab.analysis.psd import PsdAccumulator
from nlab.hardware.digitizer.dma import McaEventBuffer
from nlab.ui.ui_psd_view import Ui_PSDView
from nlab.views.image_display import transparent_sparse_bins
from nlab.views.plot_viewbox import style_plot_widget
from nlab.views.psd_fom_widget import PsdFomWidget
from nlab.views.responsive_layout import configure_psd_layout

log = logging.getLogger(__name__)

_DISPLAY_INTERVAL_MS = 100
_MATRIX_COLUMN_LEFT_AXIS_WIDTH = 84
_MATRIX_COLUMN_RIGHT_AXIS_WIDTH = 66


class PSDController(QWidget):
    """Live PSD matrix and projections for one MCA channel.

    The controller drains decoded event batches already produced by the MCA
    DMA worker. It never owns an IIO context or DMA buffer, so plotting cannot
    alter the stop/drain/close sequence in the hardware backend.
    """

    def __init__(
        self,
        event_buffer: McaEventBuffer,
        channel: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._event_buffer = event_buffer
        self._channel = channel
        self._capturing = False
        self._capture_failed = False
        self._capture_note = "Waiting for DMA with Charge Comparison enabled."
        self._last_dropped_records = 0
        self._energy_log_y = False
        self._session_replay_batches: list[np.ndarray] | None = None
        self._session_replay_previous_settings: dict[str, object] | None = None
        self._session_replay_timer_was_active = False

        self.ui = Ui_PSDView()
        self.ui.setupUi(self)  # type: ignore[no-untyped-call]
        configure_psd_layout(self, self.ui)
        self._accumulator = self._make_accumulator()
        self._setup_plots()
        self._connect_signals()
        self._application = QApplication.instance()
        if self._application is not None:
            self._application.installEventFilter(self)

        self._timer = QTimer(self)
        self._timer.setInterval(_DISPLAY_INTERVAL_MS)
        self._timer.timeout.connect(self.process_pending_events)
        self._timer.start()
        self._reset_display_ranges()
        self._render()

    def _make_accumulator(self) -> PsdAccumulator:
        return PsdAccumulator(
            energy_bins=self.ui.spinEnergyBins.value(),
            ratio_bins=self.ui.spinRatioBins.value(),
            energy_right_shift=self.ui.spinEnergyShift.value(),
            ratio_range=(self.ui.spinRatioMin.value(), self.ui.spinRatioMax.value()),
        )

    def _setup_plots(self) -> None:
        for widget in (self.ui.plotPsd, self.ui.plotRatio, self.ui.plotEnergy):
            style_plot_widget(widget)

        self._psd_plot = self.ui.plotPsd.getPlotItem()
        self._psd_plot.showAxis("right")
        self._psd_plot.showAxis("top")
        self._psd_plot.getViewBox().setDefaultPadding(0.0)
        self._psd_plot.setLabel("bottom", "Trapezoid energy", units="raw")
        self._psd_plot.setLabel("left", "1 - charge / trapezoid")
        self._psd_image = pg.ImageItem(axisOrder="row-major")
        self._psd_image.setLookupTable(pg.colormap.get("CET-L9").getLookupTable())
        self._psd_plot.addItem(self._psd_image)

        self._cut_line = pg.InfiniteLine(
            angle=0,
            movable=True,
            pen=pg.mkPen("#e63946", width=2),
            hoverPen=pg.mkPen("#ff6b6b", width=3),
        )
        self._cut_line.setPos(self.ui.spinCut.value())
        self._psd_plot.addItem(self._cut_line)

        self._energy_roi = pg.LinearRegionItem(
            values=(0.0, self._accumulator.energy_range[1] / 4.0),
            orientation="vertical",
            brush=pg.mkBrush(80, 140, 220, 35),
        )
        self._psd_plot.addItem(self._energy_roi)

        self._ratio_plot = self.ui.plotRatio.getPlotItem()
        self._ratio_plot.showAxis("right")
        self._ratio_plot.showAxis("top")
        self._ratio_plot.showGrid(x=True, y=True, alpha=0.2)
        self._ratio_plot.setLabel("bottom", "Counts")
        self._ratio_plot.setLabel("left", "PSD ratio")
        self._ratio_curve = self._ratio_plot.plot(pen=pg.mkPen("#6a4c93", width=1.5))
        self._ratio_cut_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#e63946", width=1.5),
        )
        self._ratio_plot.addItem(self._ratio_cut_line)

        self._energy_plot = self.ui.plotEnergy.getPlotItem()
        self._energy_plot.showAxis("right")
        self._energy_plot.showAxis("top")
        self._energy_plot.showGrid(x=True, y=True, alpha=0.2)
        self._energy_plot.setLabel("bottom", "Trapezoid energy", units="raw")
        self._energy_plot.setLabel("left", "Counts")
        self._energy_plot.addLegend(offset=(10, 10))
        for plot in (self._psd_plot, self._energy_plot):
            plot.getAxis("left").setWidth(_MATRIX_COLUMN_LEFT_AXIS_WIDTH)
            plot.getAxis("right").setWidth(_MATRIX_COLUMN_RIGHT_AXIS_WIDTH)
        self.ui.plotEnergy.setToolTip(
            "Hover here and press L to toggle logarithmic Y scale. "
            "Shift+wheel zooms X; Ctrl+wheel zooms Y."
        )
        self._below_curve = self._energy_plot.plot(
            pen=pg.mkPen("#277da1", width=1.5), name="Below cut"
        )
        self._above_curve = self._energy_plot.plot(
            pen=pg.mkPen("#f8961e", width=1.5), name="Above cut"
        )

        # The Designer form keeps the generated UI simple. Re-place the energy
        # spectrum here so the shared fit panel occupies the requested lower-right
        # cell beside it.
        self.ui.plotsLayout.removeWidget(self.ui.plotEnergy)
        self.ui.plotsLayout.addWidget(self.ui.plotEnergy, 1, 0, 1, 1)
        self._fom_widget = PsdFomWidget(self._ratio_plot, self)
        self._fom_widget.setObjectName("psdFomWidget")
        self._fom_widget.set_ratio_range(self._accumulator.ratio_range)
        self.ui.plotsLayout.addWidget(self._fom_widget, 1, 1, 1, 1)

    def _connect_signals(self) -> None:
        for spinbox in (
            self.ui.spinEnergyBins,
            self.ui.spinRatioBins,
            self.ui.spinEnergyShift,
            self.ui.spinRatioMin,
            self.ui.spinRatioMax,
        ):
            spinbox.editingFinished.connect(self._on_analysis_settings_changed)
        self.ui.spinCut.valueChanged.connect(self._on_cut_spin_changed)
        self.ui.btnClear.clicked.connect(self.clear)
        self._cut_line.sigPositionChanged.connect(self._on_cut_line_changed)
        self._energy_roi.sigRegionChangeFinished.connect(self._render_projections)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Handle the local log shortcut without stealing a global key."""
        if (
            event.type() == QEvent.Type.KeyPress
            and self.isVisible()
            and self.ui.plotEnergy.underMouse()
            and isinstance(event, QKeyEvent)
            and event.key() == Qt.Key.Key_L
            and event.modifiers()
            in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier)
            and not event.isAutoRepeat()
        ):
            self.set_energy_log_y(not self._energy_log_y)
            return True
        return super().eventFilter(watched, event)

    def set_energy_log_y(self, enabled: bool) -> None:
        """Set logarithmic Y display on the bottom energy histogram."""
        self._energy_log_y = bool(enabled)
        self._energy_plot.setLogMode(y=self._energy_log_y)

    def begin_capture(self, enabled: bool, note: str = "") -> None:
        """Reset the live view and declare whether this capture feeds PSD."""
        self._event_buffer.clear()
        self._accumulator.reset()
        self._last_dropped_records = 0
        self._capturing = enabled
        self._capture_failed = False
        self._fom_widget.clear_fit()
        self._capture_note = note or (
            "Recording live PSD."
            if enabled
            else "PSD inactive: enable Charge Comparison before starting DMA."
        )
        self._render()

    def finish_capture(self) -> None:
        """Consume the DMA tail and mark the current matrix complete."""
        self.process_pending_events()
        self._capturing = False
        if self._accumulator.statistics.received and not self._capture_failed:
            self._capture_note = "Capture complete."
        self._update_status()

    def set_capture_error(self, message: str) -> None:
        self._capturing = False
        self._capture_failed = True
        self._capture_note = f"DMA error: {message}"
        self._update_status()

    def clear(self) -> None:
        self._event_buffer.clear()
        self._accumulator.reset()
        self._last_dropped_records = 0
        self._fom_widget.clear_fit()
        self._render()

    @property
    def channel(self) -> int:
        return self._channel

    def dose_calibration_snapshot(self) -> PsdDoseCalibrationSnapshot:
        """Return cumulative below/above-cut totals for the full view and ROI."""
        cut = float(self.ui.spinCut.value())
        below, above = self._accumulator.energy_projections(cut)
        roi = tuple(sorted(float(value) for value in self._energy_roi.getRegion()))
        edges = self._accumulator.energy_edges
        first = int(
            np.clip(
                np.searchsorted(edges, roi[0], side="right") - 1,
                0,
                self._accumulator.energy_bins,
            )
        )
        last = int(
            np.clip(
                np.searchsorted(edges, roi[1], side="left"),
                0,
                self._accumulator.energy_bins,
            )
        )
        return PsdDoseCalibrationSnapshot(
            cut=cut,
            roi=(roi[0], roi[1]),
            below_cut_total_counts=int(np.sum(below, dtype=np.uint64)),
            above_cut_total_counts=int(np.sum(above, dtype=np.uint64)),
            below_cut_roi_counts=int(np.sum(below[first:last], dtype=np.uint64)),
            above_cut_roi_counts=int(np.sum(above[first:last], dtype=np.uint64)),
        )

    def process_pending_events(self) -> None:
        if self._session_replay_batches is not None:
            return
        batches, dropped_records = self._event_buffer.drain()
        self._last_dropped_records = dropped_records
        if not batches:
            self._update_status()
            return
        try:
            for events in batches:
                self._accumulator.add_events(events)
        except (TypeError, ValueError):
            log.exception("PSD ch%d: failed to analyze a DMA event batch", self._channel)
            self._capture_note = "PSD analysis error; raw DMA recording continues."
        self._render()

    def _on_analysis_settings_changed(self) -> None:
        ratio_range = (self.ui.spinRatioMin.value(), self.ui.spinRatioMax.value())
        if ratio_range[0] >= ratio_range[1]:
            self.ui.lblStatus.setText("Ratio minimum must be smaller than ratio maximum.")
            return
        self._event_buffer.clear()
        self._accumulator = self._make_accumulator()
        self._last_dropped_records = 0
        self.ui.spinCut.setRange(*ratio_range)
        self.ui.spinCut.setValue(float(np.clip(self.ui.spinCut.value(), *ratio_range)))
        self._energy_roi.setBounds(self._accumulator.energy_range)
        self._energy_roi.setRegion((0.0, self._accumulator.energy_range[1] / 4.0))
        self._fom_widget.set_ratio_range(self._accumulator.ratio_range)
        if self._session_replay_batches is not None:
            try:
                for events in self._session_replay_batches:
                    self._accumulator.add_events(events)
            except (TypeError, ValueError):
                log.exception("PSD ch%d: session replay analysis failed", self._channel)
                self._capture_note = "Session replay analysis failed."
        self._reset_display_ranges()
        self._render()

    def begin_session_replay(
        self,
        events: np.ndarray | None = None,
        settings: object = None,
    ) -> None:
        """Prepare progressive PSD accumulation from recorded event batches."""
        self._session_replay_previous_settings = self.configuration_settings()
        self._session_replay_timer_was_active = self._timer.isActive()
        self._timer.stop()
        self._event_buffer.clear()
        if isinstance(settings, dict):
            self.apply_configuration_settings(settings)
        self._session_replay_batches = []
        self._accumulator = self._make_accumulator()
        self._last_dropped_records = 0
        self._capturing = False
        self._capture_failed = False
        self._capture_note = "Session replay: waiting for recorded events."
        if events is not None:
            self.display_session_replay_events(events)
        self._reset_display_ranges()
        self._render()

    def display_session_replay_events(self, events: np.ndarray) -> None:
        """Add one captured DMA batch to the replay PSD accumulator."""
        if self._session_replay_batches is None:
            return
        copied = np.asarray(events).reshape(-1).copy()
        self._session_replay_batches.append(copied)
        try:
            self._accumulator.add_events(copied)
        except (TypeError, ValueError):
            self._capture_note = "Session replay analysis failed."
            raise
        self._capture_note = "Session replay in progress."
        self._render()

    def finish_session_replay(self) -> None:
        if self._session_replay_batches is None:
            return
        self._capture_note = "Session replay complete; final PSD is held."
        self._update_status()

    def end_session_replay(self) -> None:
        """Remove replay products and restore the pre-replay analysis controls."""
        if self._session_replay_batches is None:
            return
        previous = self._session_replay_previous_settings
        restart_timer = self._session_replay_timer_was_active
        self._session_replay_batches = None
        self._session_replay_previous_settings = None
        self._session_replay_timer_was_active = False
        if previous is not None:
            self.apply_configuration_settings(previous)
        else:
            self._accumulator = self._make_accumulator()
            self._render()
        self._capture_note = "Waiting for DMA with Charge Comparison enabled."
        self._update_status()
        if restart_timer:
            self._timer.start()

    def _on_cut_spin_changed(self, value: float) -> None:
        self._cut_line.blockSignals(True)
        self._cut_line.setPos(value)
        self._cut_line.blockSignals(False)
        self._ratio_cut_line.setPos(value)
        self._render_projections()

    def _on_cut_line_changed(self) -> None:
        value = float(np.clip(self._cut_line.value(), *self._accumulator.ratio_range))
        self.ui.spinCut.setValue(value)

    def _render(self) -> None:
        matrix = self._accumulator.matrix.T
        image = np.log1p(matrix.astype(np.float64))
        high = max(1.0, float(image.max(initial=0.0)))
        displayed = transparent_sparse_bins(image, matrix)
        self._psd_image.setImage(displayed, autoLevels=False, levels=(0.0, high))
        energy_low, energy_high = self._accumulator.energy_range
        ratio_low, ratio_high = self._accumulator.ratio_range
        self._psd_image.setRect(
            QRectF(energy_low, ratio_low, energy_high - energy_low, ratio_high - ratio_low)
        )
        self._render_projections()
        self._update_status()

    def _render_projections(self) -> None:
        cut = self.ui.spinCut.value()
        below, above = self._accumulator.energy_projections(cut)
        energy = self._accumulator.energy_centers
        self._below_curve.setData(energy, below)
        self._above_curve.setData(energy, above)

        ratio_counts = self._accumulator.ratio_projection(self._energy_roi.getRegion())
        self._ratio_curve.setData(ratio_counts, self._accumulator.ratio_centers)
        self._fom_widget.set_projection(self._accumulator.ratio_centers, ratio_counts)
        self._ratio_cut_line.setPos(cut)

    def _update_status(self) -> None:
        stats = self._accumulator.statistics
        state = "Recording" if self._capturing else self._capture_note
        self.ui.lblStatus.setText(
            f"{state}  Accepted: {stats.accepted:,}/{stats.received:,}; "
            f"zero energy: {stats.zero_total:,}; outside view: {stats.outside_range:,}; "
            f"display drops: {self._last_dropped_records:,}."
        )

    def _reset_display_ranges(self) -> None:
        self._ratio_plot.setYRange(*self._accumulator.ratio_range, padding=0.0)
        self._energy_plot.setXRange(*self._accumulator.energy_range, padding=0.0)
        self._psd_plot.setRange(
            xRange=self._accumulator.energy_range,
            yRange=self._accumulator.ratio_range,
            padding=0.0,
        )

    def reset_zoom(self) -> None:
        self._reset_display_ranges()
        self._ratio_plot.autoRange()
        self._ratio_plot.setYRange(*self._accumulator.ratio_range, padding=0.0)
        self._energy_plot.autoRange()
        self._energy_plot.setXRange(*self._accumulator.energy_range, padding=0.0)
        self._psd_plot.setRange(
            xRange=self._accumulator.energy_range,
            yRange=self._accumulator.ratio_range,
            padding=0.0,
        )

    def configuration_settings(self) -> dict[str, object]:
        return {
            "energy_bins": self.ui.spinEnergyBins.value(),
            "ratio_bins": self.ui.spinRatioBins.value(),
            "energy_right_shift": self.ui.spinEnergyShift.value(),
            "ratio_range": [self.ui.spinRatioMin.value(), self.ui.spinRatioMax.value()],
            "ratio_cut": self.ui.spinCut.value(),
            "energy_roi": [float(value) for value in self._energy_roi.getRegion()],
            "energy_log_y": self._energy_log_y,
            "fom_fit_roi": [float(value) for value in self._fom_widget.fit_region.getRegion()],
            "fom_centroid_seeds": [float(line.value()) for line in self._fom_widget.centroid_lines],
            "fom_guides_visible": self._fom_widget.guides_checkbox.isChecked(),
        }

    def apply_configuration_settings(self, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        controls = (
            self.ui.spinEnergyBins,
            self.ui.spinRatioBins,
            self.ui.spinEnergyShift,
            self.ui.spinRatioMin,
            self.ui.spinRatioMax,
            self.ui.spinCut,
        )
        for control in controls:
            control.blockSignals(True)
        try:
            self.ui.spinEnergyBins.setValue(int(settings.get("energy_bins", 1024)))
            self.ui.spinRatioBins.setValue(int(settings.get("ratio_bins", 256)))
            self.ui.spinEnergyShift.setValue(int(settings.get("energy_right_shift", 0)))
            ratio_range = settings.get("ratio_range", [-1.0, 1.0])
            if isinstance(ratio_range, list) and len(ratio_range) == 2:
                self.ui.spinRatioMin.setValue(float(ratio_range[0]))
                self.ui.spinRatioMax.setValue(float(ratio_range[1]))
            self.ui.spinCut.setValue(float(settings.get("ratio_cut", 0.0)))
        finally:
            for control in controls:
                control.blockSignals(False)
        self._on_analysis_settings_changed()
        energy_roi = settings.get("energy_roi")
        if isinstance(energy_roi, list) and len(energy_roi) == 2:
            self._energy_roi.setRegion((float(energy_roi[0]), float(energy_roi[1])))
        fom_fit_roi = settings.get("fom_fit_roi")
        if isinstance(fom_fit_roi, list) and len(fom_fit_roi) == 2:
            self._fom_widget.fit_region.setRegion((float(fom_fit_roi[0]), float(fom_fit_roi[1])))
        fom_centroid_seeds = settings.get("fom_centroid_seeds")
        if isinstance(fom_centroid_seeds, list) and len(fom_centroid_seeds) == 2:
            for line, value in zip(
                self._fom_widget.centroid_lines,
                fom_centroid_seeds,
                strict=True,
            ):
                line.setPos(float(value))
        self._fom_widget.guides_checkbox.setChecked(bool(settings.get("fom_guides_visible", True)))
        self.set_energy_log_y(bool(settings.get("energy_log_y", False)))
        self._render_projections()

    def stop_processing(self) -> None:
        self._timer.stop()
        if self._application is not None:
            self._application.removeEventFilter(self)
            self._application = None
