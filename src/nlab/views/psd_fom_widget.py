"""Interactive controls and overlays for PSD figure-of-merit fitting."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.psd_fom import PsdFomFit, fit_psd_fom_lm


class PsdFomWidget(QGroupBox):
    """Fit panel plus draggable guides attached to a ratio-projection plot."""

    fit_completed = Signal(object)

    def __init__(self, ratio_plot: pg.PlotItem, parent: QWidget | None = None) -> None:
        super().__init__("PSD figure of merit", parent)
        self._ratio_plot = ratio_plot
        self._ratio = np.empty(0, dtype=np.float64)
        self._counts = np.empty(0, dtype=np.float64)
        self._guide_range: tuple[float, float] | None = None
        self.last_fit: PsdFomFit | None = None

        self.fit_region = pg.LinearRegionItem(
            values=(0.1, 0.9),
            orientation="horizontal",
            brush=pg.mkBrush(42, 157, 143, 35),
            pen=pg.mkPen("#2a9d8f", width=1.5),
            hoverPen=pg.mkPen("#2a9d8f", width=2.5),
        )
        self.fit_region.setZValue(12.0)
        ratio_plot.addItem(self.fit_region)
        self.centroid_lines = (
            pg.InfiniteLine(
                angle=0,
                movable=True,
                pen=pg.mkPen("#277da1", width=2),
                hoverPen=pg.mkPen("#4ea8de", width=3),
                label="μ1",
                labelOpts={"position": 0.08, "color": "#277da1"},
            ),
            pg.InfiniteLine(
                angle=0,
                movable=True,
                pen=pg.mkPen("#f8961e", width=2),
                hoverPen=pg.mkPen("#ffb347", width=3),
                label="μ2",
                labelOpts={"position": 0.08, "color": "#f8961e"},
            ),
        )
        for line in self.centroid_lines:
            line.setZValue(20.0)
            ratio_plot.addItem(line)

        self._first_curve = ratio_plot.plot(
            pen=pg.mkPen("#277da1", width=1.25, style=Qt.PenStyle.DashLine)
        )
        self._second_curve = ratio_plot.plot(
            pen=pg.mkPen("#f8961e", width=1.25, style=Qt.PenStyle.DashLine)
        )
        self._fit_curve = ratio_plot.plot(pen=pg.mkPen("#2a9d8f", width=2.5))

        self.guides_checkbox = QCheckBox("Show draggable fit guides", self)
        self.guides_checkbox.setChecked(True)
        self.guides_checkbox.setToolTip(
            "Show or hide the two centroid seeds and the ratio fit-ROI band."
        )
        self.fit_button = QPushButton("Fit double Gaussian", self)
        self.fit_button.setToolTip(
            "Run a bounded Levenberg-Marquardt fit on the current projected PSD counts."
        )

        button_row = QHBoxLayout()
        button_row.addWidget(self.guides_checkbox)
        button_row.addStretch(1)
        button_row.addWidget(self.fit_button)

        self.first_label = QLabel("—", self)
        self.second_label = QLabel("—", self)
        self.fom_label = QLabel("—", self)
        self.chi_square_label = QLabel("—", self)
        results = QFormLayout()
        results.setContentsMargins(0, 0, 0, 0)
        results.addRow("Population 1 μ / FWHM:", self.first_label)
        results.addRow("Population 2 μ / FWHM:", self.second_label)
        results.addRow("FOM:", self.fom_label)
        results.addRow("Reduced χ²:", self.chi_square_label)

        self.status_label = QLabel(
            "Drag the ROI and μ markers, then press Fit double Gaussian.", self
        )
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(button_row)
        layout.addLayout(results)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.guides_checkbox.toggled.connect(self.set_guides_visible)
        self.fit_button.clicked.connect(self.fit_current_projection)

    def set_projection(self, ratio: np.ndarray, counts: np.ndarray) -> None:
        """Store the currently displayed ratio projection for the next button press."""
        ratio_values = np.asarray(ratio, dtype=np.float64)
        count_values = np.asarray(counts, dtype=np.float64)
        if ratio_values.ndim != 1 or ratio_values.shape != count_values.shape:
            raise ValueError("PSD ratio projection coordinates and counts must match")
        self._ratio = ratio_values.copy()
        self._counts = count_values.copy()

    def set_ratio_range(self, ratio_range: tuple[float, float]) -> None:
        """Bound and, when the display range changes, reset the fit guides."""
        low, high = sorted(map(float, ratio_range))
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("PSD ratio range must contain two increasing finite values")
        changed = self._guide_range != (low, high)
        self._guide_range = (low, high)
        self.fit_region.setBounds((low, high))
        for line in self.centroid_lines:
            line.setBounds((low, high))
        if changed:
            span = high - low
            self.fit_region.setRegion((low + 0.1 * span, low + 0.9 * span))
            self.centroid_lines[0].setPos(low + 0.35 * span)
            self.centroid_lines[1].setPos(low + 0.65 * span)
            self.clear_fit()

    def set_guides_visible(self, visible: bool) -> None:
        """Toggle all draggable fit objects without hiding fitted curves."""
        self.fit_region.setVisible(visible)
        for line in self.centroid_lines:
            line.setVisible(visible)

    def clear_fit(self) -> None:
        """Remove a previous fit result and its curves."""
        self.last_fit = None
        self._fit_curve.setData([], [])
        self._first_curve.setData([], [])
        self._second_curve.setData([], [])
        self.first_label.setText("—")
        self.second_label.setText("—")
        self.fom_label.setText("—")
        self.chi_square_label.setText("—")
        self.status_label.setStyleSheet("")
        self.status_label.setText("Drag the ROI and μ markers, then press Fit double Gaussian.")

    def fit_current_projection(self) -> PsdFomFit | None:
        """Fit the current projection snapshot and update the result panel."""
        try:
            region = self.fit_region.getRegion()
            centroid_seeds = (
                float(self.centroid_lines[0].value()),
                float(self.centroid_lines[1].value()),
            )
            fit = fit_psd_fom_lm(
                self._ratio,
                self._counts,
                roi=(float(region[0]), float(region[1])),
                centroid_seeds=centroid_seeds,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            self.clear_fit()
            self.status_label.setStyleSheet("color: #b42318; font-weight: bold;")
            self.status_label.setText(f"Fit unavailable: {error}")
            return None

        self.last_fit = fit
        self._first_curve.setData(fit.first_component, fit.x)
        self._second_curve.setData(fit.second_component, fit.x)
        self._fit_curve.setData(fit.best_fit, fit.x)
        for line, centroid in zip(self.centroid_lines, fit.centroids, strict=True):
            line.setPos(centroid)
        self.first_label.setText(_peak_text(fit.centroids[0], fit.fwhms[0]))
        self.second_label.setText(_peak_text(fit.centroids[1], fit.fwhms[1]))
        self.fom_label.setText(f"{fit.fom:.4g}")
        self.chi_square_label.setText(f"{fit.reduced_chi_square:.3g}")
        if fit.poor_convergence:
            details = "; ".join(fit.warnings) or fit.message
            self.status_label.setStyleSheet("color: #b54708; font-weight: bold;")
            self.status_label.setText(f"Warning: poor convergence — {details}.")
        else:
            self.status_label.setStyleSheet("color: #067647;")
            self.status_label.setText(
                f"LM fit converged in {fit.evaluations} evaluations; result is a snapshot."
            )
        self.fit_completed.emit(fit)
        return fit


def _peak_text(centroid: float, fwhm: float) -> str:
    return f"{centroid:.5g} / {fwhm:.5g}"
