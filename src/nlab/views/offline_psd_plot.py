"""Reusable PSD matrix and projection display for offline analysis tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QRectF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from nlab.views.image_display import transparent_sparse_bins
from nlab.views.plot_viewbox import NLabPlotWidget, style_plot_widget
from nlab.views.psd_fom_widget import PsdFomWidget
from nlab.views.tooltips import wrap_widget_tooltips

_MATRIX_COLUMN_LEFT_AXIS_WIDTH = 84
_MATRIX_COLUMN_RIGHT_AXIS_WIDTH = 66


@dataclass(frozen=True)
class OfflinePsdReportState:
    """Current matrix, selections, and display state for report generation."""

    matrix: np.ndarray
    energy_range: tuple[float, float]
    ratio_range: tuple[float, float]
    energy_label: str
    cut: float
    energy_roi: tuple[float, float]
    matrix_scale: Literal["linear", "log"]
    energy_log_y: bool


class OfflinePsdPlot(QWidget):
    """Display an energy/PSD matrix with linked ratio and energy projections."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._matrix = np.zeros((1, 1), dtype=np.uint64)
        self._energy_range = (0.0, 1.0)
        self._ratio_range = (0.0, 1.0)
        self._energy_label = "Long-gate charge"
        self._has_data = False
        self._energy_log_y = False

        root = QVBoxLayout(self)
        # Keep display controls together so a host can place them in a sidebar
        # without duplicating controls or changing their plotting connections.
        self.controls_widget = QWidget(self)
        controls = QHBoxLayout(self.controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        self.controls_layout = controls
        scale_label = QLabel("Matrix scale:", self)
        scale_label.setProperty("nlabHelpTopic", "display")
        controls.addWidget(scale_label)
        self.scale_combo = QComboBox(self)
        self.scale_combo.setProperty("nlabHelpTopic", "display")
        scale_label.setBuddy(self.scale_combo)
        self.scale_combo.addItem("Logarithmic", "log")
        self.scale_combo.addItem("Linear", "linear")
        controls.addWidget(self.scale_combo)
        cut_label = QLabel("PSD cut:", self)
        cut_label.setProperty("nlabHelpTopic", "psd-cut")
        controls.addWidget(cut_label)
        self.cut_spin = QDoubleSpinBox(self)
        self.cut_spin.setProperty("nlabHelpTopic", "psd-cut")
        cut_label.setBuddy(self.cut_spin)
        self.cut_spin.setDecimals(4)
        self.cut_spin.setSingleStep(0.01)
        controls.addWidget(self.cut_spin)
        self.roi_checkbox = QCheckBox("Show PSD ROI", self)
        self.roi_checkbox.setChecked(True)
        self.roi_checkbox.setProperty("nlabHelpTopic", "psd-roi")
        self.roi_checkbox.setToolTip(
            "Show/hide the energy ROI and PSD cut line on the matrix without changing "
            "their values or projections. FOM fit guides have their own visibility switch."
        )
        controls.addWidget(self.roi_checkbox)
        controls.addStretch(1)
        root.addWidget(self.controls_widget)

        plots_widget = QWidget(self)
        plots = QGridLayout(plots_widget)
        plots.setContentsMargins(0, 0, 0, 0)
        plots.setSpacing(4)

        self._matrix_widget = NLabPlotWidget(plots_widget)
        style_plot_widget(self._matrix_widget)
        self.matrix_plot = self._matrix_widget.getPlotItem()
        self.matrix_plot.showAxis("top")
        self.matrix_plot.showAxis("right")
        self.matrix_plot.getViewBox().setDefaultPadding(0.0)
        self.matrix_plot.setLabel("left", "PSD ratio")
        self.matrix_plot.setLabel("bottom", "Long-gate charge", units="raw")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.image.setLookupTable(pg.colormap.get("CET-L9").getLookupTable())
        self.matrix_plot.addItem(self.image)

        self.energy_region = pg.LinearRegionItem(
            values=(0.0, 1.0),
            orientation="vertical",
            brush=pg.mkBrush(80, 140, 220, 35),
        )
        self.energy_region.setZValue(10.0)
        self.matrix_plot.addItem(self.energy_region)
        self.cut_line = pg.InfiniteLine(
            angle=0,
            movable=True,
            pen=pg.mkPen("#e63946", width=2),
            hoverPen=pg.mkPen("#ff6b6b", width=3),
        )
        self.cut_line.setZValue(20.0)
        self.matrix_plot.addItem(self.cut_line)

        self._ratio_widget = NLabPlotWidget(plots_widget)
        style_plot_widget(self._ratio_widget)
        self._ratio_widget.setMaximumWidth(300)
        self.ratio_plot = self._ratio_widget.getPlotItem()
        self.ratio_plot.setYLink(self.matrix_plot)
        self.ratio_plot.showAxis("top")
        self.ratio_plot.showAxis("right")
        self.ratio_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ratio_plot.setLabel("bottom", "Counts")
        self.ratio_plot.setLabel("left", "PSD ratio")
        self.ratio_curve = self.ratio_plot.plot(pen=pg.mkPen("#6a4c93", width=1.5))

        self._energy_widget = NLabPlotWidget(plots_widget)
        style_plot_widget(self._energy_widget)
        self._energy_widget.setMaximumHeight(240)
        self.energy_plot = self._energy_widget.getPlotItem()
        self.energy_plot.setXLink(self.matrix_plot)
        self.energy_plot.showAxis("top")
        self.energy_plot.showAxis("right")
        self.energy_plot.showGrid(x=True, y=True, alpha=0.2)
        self.energy_plot.setLabel("left", "Counts")
        self.energy_plot.setLabel("bottom", "Long-gate charge", units="raw")
        for plot in (self.matrix_plot, self.energy_plot):
            plot.getAxis("left").setWidth(_MATRIX_COLUMN_LEFT_AXIS_WIDTH)
            plot.getAxis("right").setWidth(_MATRIX_COLUMN_RIGHT_AXIS_WIDTH)
        self._energy_widget.setToolTip(
            "Hover here and press L to toggle logarithmic Y scale. "
            "Shift+wheel zooms X; Ctrl+wheel zooms Y."
        )
        self.energy_plot.addLegend(offset=(10, 10))
        self.below_curve = self.energy_plot.plot(
            pen=pg.mkPen("#277da1", width=1.5), name="Below cut"
        )
        self.above_curve = self.energy_plot.plot(
            pen=pg.mkPen("#f8961e", width=1.5), name="Above cut"
        )
        self.fom_widget = PsdFomWidget(self.ratio_plot, plots_widget)
        self.fom_widget.setMaximumWidth(300)

        plots.addWidget(self._matrix_widget, 0, 0)
        plots.addWidget(self._ratio_widget, 0, 1)
        plots.addWidget(self._energy_widget, 1, 0)
        plots.addWidget(self.fom_widget, 1, 1)
        plots.setRowStretch(0, 3)
        plots.setRowStretch(1, 2)
        plots.setColumnStretch(0, 3)
        plots.setColumnStretch(1, 2)
        root.addWidget(plots_widget, 1)

        self.scale_combo.currentIndexChanged.connect(self._render_image)
        self.cut_spin.valueChanged.connect(self._cut_spin_changed)
        self.cut_line.sigPositionChanged.connect(self._cut_line_changed)
        self.energy_region.sigRegionChangeFinished.connect(self._render_projections)
        self.roi_checkbox.toggled.connect(self._set_roi_visible)
        self._application = QApplication.instance()
        if self._application is not None:
            self._application.installEventFilter(self)
        wrap_widget_tooltips(self)

    def _set_roi_visible(self, visible: bool) -> None:
        self.energy_region.setVisible(visible)
        self.cut_line.setVisible(visible)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Toggle energy-projection log scale only while that plot is hovered."""
        if (
            event.type() == QEvent.Type.KeyPress
            and self.isVisible()
            and self._energy_widget.underMouse()
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
        """Set logarithmic Y display on the bottom energy projection."""
        self._energy_log_y = bool(enabled)
        self.energy_plot.setLogMode(y=self._energy_log_y)

    def set_data(
        self,
        matrix: np.ndarray,
        *,
        energy_range: tuple[float, float],
        ratio_range: tuple[float, float],
        energy_label: str = "Long-gate charge",
    ) -> None:
        values = np.asarray(matrix, dtype=np.uint64)
        if values.ndim != 2 or 0 in values.shape:
            raise ValueError("PSD matrix must be a non-empty two-dimensional array")
        previous_energy_region = self.energy_region.getRegion()
        self._matrix = values
        self._energy_range = energy_range
        self._ratio_range = ratio_range
        self._energy_label = energy_label
        self.fom_widget.set_ratio_range(ratio_range)
        self.fom_widget.clear_fit()
        self.matrix_plot.setLabel("bottom", energy_label, units="raw")
        self.energy_plot.setLabel("bottom", energy_label, units="raw")

        self.cut_spin.blockSignals(True)
        self.cut_spin.setRange(*ratio_range)
        self.cut_spin.setValue(float(np.clip(self.cut_spin.value(), *ratio_range)))
        self.cut_spin.blockSignals(False)
        self.cut_line.setBounds(ratio_range)
        self.cut_line.setPos(self.cut_spin.value())
        self.energy_region.setBounds(energy_range)
        if self._has_data:
            low, high = sorted(map(float, previous_energy_region))
            low = float(np.clip(low, *energy_range))
            high = float(np.clip(high, *energy_range))
            if high <= low:
                low = energy_range[0]
                high = energy_range[0] + (energy_range[1] - energy_range[0]) / 4.0
            self.energy_region.setRegion((low, high))
        else:
            self.energy_region.setRegion(
                (energy_range[0], energy_range[0] + (energy_range[1] - energy_range[0]) / 4.0)
            )
        self._has_data = True
        # Prime the linked projections before applying the authoritative matrix
        # range.  Setting a linked projection last makes pyqtgraph compensate for
        # the plots' unequal axis geometry and can crop the matrix (for example,
        # a requested -1..1 ratio appeared as roughly -0.5..1.2).
        self.ratio_plot.setYRange(*ratio_range, padding=0.0)
        self.energy_plot.setXRange(*energy_range, padding=0.0)
        self._render_image()
        self._render_projections()
        self.matrix_plot.setRange(xRange=energy_range, yRange=ratio_range, padding=0.0)

    def _render_image(self) -> None:
        matrix = self._matrix.T
        image = matrix.astype(np.float64)
        if self.scale_combo.currentData() == "log":
            image = np.log1p(image)
        high = max(1.0, float(image.max(initial=0.0)))
        displayed = transparent_sparse_bins(image, matrix)
        self.image.setImage(displayed, autoLevels=False, levels=(0.0, high))
        energy_low, energy_high = self._energy_range
        ratio_low, ratio_high = self._ratio_range
        self.image.setRect(
            QRectF(energy_low, ratio_low, energy_high - energy_low, ratio_high - ratio_low)
        )

    def _cut_spin_changed(self, value: float) -> None:
        self.cut_line.blockSignals(True)
        self.cut_line.setPos(value)
        self.cut_line.blockSignals(False)
        self._render_projections()

    def _cut_line_changed(self) -> None:
        self.cut_spin.setValue(float(np.clip(self.cut_line.value(), *self._ratio_range)))

    def _render_projections(self) -> None:
        energy_edges = np.linspace(*self._energy_range, self._matrix.shape[0] + 1)
        ratio_edges = np.linspace(*self._ratio_range, self._matrix.shape[1] + 1)
        energy_centers = (energy_edges[:-1] + energy_edges[1:]) * 0.5
        ratio_centers = (ratio_edges[:-1] + ratio_edges[1:]) * 0.5

        split = int(
            np.clip(
                np.searchsorted(ratio_edges, self.cut_spin.value()),
                0,
                len(ratio_centers),
            )
        )
        below = self._matrix[:, :split].sum(axis=1, dtype=np.uint64)
        above = self._matrix[:, split:].sum(axis=1, dtype=np.uint64)
        self.below_curve.setData(energy_centers, below)
        self.above_curve.setData(energy_centers, above)

        low, high = sorted(self.energy_region.getRegion())
        first = int(
            np.clip(
                np.searchsorted(energy_edges, low, side="right") - 1,
                0,
                len(energy_centers),
            )
        )
        last = int(
            np.clip(
                np.searchsorted(energy_edges, high, side="left"),
                0,
                len(energy_centers),
            )
        )
        ratio_counts = self._matrix[first:last].sum(axis=0, dtype=np.uint64)
        self.ratio_curve.setData(ratio_counts, ratio_centers)
        self.fom_widget.set_projection(ratio_centers, ratio_counts)

    def report_state(self) -> OfflinePsdReportState:
        """Return an immutable snapshot of the currently displayed report inputs."""
        if not self._has_data:
            raise ValueError("No PSD data are available for a report")
        matrix = self._matrix.copy()
        matrix.flags.writeable = False
        energy_roi = tuple(sorted(map(float, self.energy_region.getRegion())))
        return OfflinePsdReportState(
            matrix=matrix,
            energy_range=self._energy_range,
            ratio_range=self._ratio_range,
            energy_label=self._energy_label,
            cut=float(self.cut_spin.value()),
            energy_roi=(energy_roi[0], energy_roi[1]),
            matrix_scale=cast(Literal["linear", "log"], str(self.scale_combo.currentData())),
            energy_log_y=self._energy_log_y,
        )
