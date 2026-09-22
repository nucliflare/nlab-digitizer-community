"""Reusable PSD matrix and projection display for offline analysis tools."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from nlab.views.plot_viewbox import ModifierZoomViewBox


class OfflinePsdPlot(QWidget):
    """Display an energy/PSD matrix with linked ratio and energy projections."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._matrix = np.zeros((1, 1), dtype=np.uint64)
        self._energy_range = (0.0, 1.0)
        self._ratio_range = (0.0, 1.0)

        root = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Matrix scale:", self))
        self.scale_combo = QComboBox(self)
        self.scale_combo.addItem("Logarithmic", "log")
        self.scale_combo.addItem("Linear", "linear")
        controls.addWidget(self.scale_combo)
        controls.addWidget(QLabel("PSD cut:", self))
        self.cut_spin = QDoubleSpinBox(self)
        self.cut_spin.setDecimals(4)
        self.cut_spin.setSingleStep(0.01)
        controls.addWidget(self.cut_spin)
        controls.addStretch(1)
        root.addLayout(controls)

        graphics = pg.GraphicsLayoutWidget(self)
        graphics.setBackground("#f8f9fa")
        self.matrix_plot = graphics.addPlot(
            row=0,
            col=0,
            viewBox=ModifierZoomViewBox(),
        )
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

        self.ratio_plot = graphics.addPlot(
            row=0,
            col=1,
            viewBox=ModifierZoomViewBox(),
        )
        self.ratio_plot.setMaximumWidth(300)
        self.ratio_plot.setYLink(self.matrix_plot)
        self.ratio_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ratio_plot.setLabel("bottom", "Counts")
        self.ratio_plot.setLabel("left", "PSD ratio")
        self.ratio_curve = self.ratio_plot.plot(pen=pg.mkPen("#6a4c93", width=1.5))

        self.energy_plot = graphics.addPlot(
            row=1,
            col=0,
            viewBox=ModifierZoomViewBox(),
        )
        self.energy_plot.setMaximumHeight(240)
        self.energy_plot.setXLink(self.matrix_plot)
        self.energy_plot.showGrid(x=True, y=True, alpha=0.2)
        self.energy_plot.setLabel("left", "Counts")
        self.energy_plot.setLabel("bottom", "Long-gate charge", units="raw")
        self.energy_plot.addLegend(offset=(10, 10))
        self.below_curve = self.energy_plot.plot(
            pen=pg.mkPen("#277da1", width=1.5), name="Below cut"
        )
        self.above_curve = self.energy_plot.plot(
            pen=pg.mkPen("#f8961e", width=1.5), name="Above cut"
        )
        root.addWidget(graphics, 1)

        self.scale_combo.currentIndexChanged.connect(self._render_image)
        self.cut_spin.valueChanged.connect(self._cut_spin_changed)
        self.cut_line.sigPositionChanged.connect(self._cut_line_changed)
        self.energy_region.sigRegionChangeFinished.connect(self._render_projections)

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
        self._matrix = values
        self._energy_range = energy_range
        self._ratio_range = ratio_range
        self.matrix_plot.setLabel("bottom", energy_label, units="raw")
        self.energy_plot.setLabel("bottom", energy_label, units="raw")

        self.cut_spin.blockSignals(True)
        self.cut_spin.setRange(*ratio_range)
        self.cut_spin.setValue(float(np.clip(self.cut_spin.value(), *ratio_range)))
        self.cut_spin.blockSignals(False)
        self.cut_line.setBounds(ratio_range)
        self.cut_line.setPos(self.cut_spin.value())
        self.energy_region.setBounds(energy_range)
        self.energy_region.setRegion(
            (energy_range[0], energy_range[0] + (energy_range[1] - energy_range[0]) / 4.0)
        )
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
        image = self._matrix.T.astype(np.float64)
        if self.scale_combo.currentData() == "log":
            image = np.log1p(image)
        high = max(1.0, float(image.max(initial=0.0)))
        self.image.setImage(image, autoLevels=False, levels=(0.0, high))
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
