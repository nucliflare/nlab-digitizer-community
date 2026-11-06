"""Interactive controls and overlays for PSD figure-of-merit fitting."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.psd_fom import PsdFomFit, fit_psd_fom_lm, fit_psd_fom_separate_lm
from nlab.views.tooltips import wrap_widget_tooltips


class PsdFomWidget(QGroupBox):
    """Fit panel plus draggable guides attached to a ratio-projection plot."""

    fit_completed = Signal(object)
    fit_changed = Signal(object)

    def __init__(self, ratio_plot: pg.PlotItem, parent: QWidget | None = None) -> None:
        super().__init__("PSD figure of merit", parent)
        self._ratio_plot = ratio_plot
        self._ratio = np.empty(0, dtype=np.float64)
        self._counts = np.empty(0, dtype=np.float64)
        self._guide_range: tuple[float, float] | None = None
        self.last_fit: PsdFomFit | None = None
        self._fit_dirty = True
        self._peak_regions_initialized = False
        self._continuous_timer = QTimer(self)
        self._continuous_timer.setInterval(500)
        self._continuous_timer.timeout.connect(self._fit_latest_projection)

        self.fit_region = pg.LinearRegionItem(
            values=(0.1, 0.9),
            orientation="horizontal",
            brush=pg.mkBrush(42, 157, 143, 35),
            pen=pg.mkPen("#2a9d8f", width=1.5),
            hoverPen=pg.mkPen("#2a9d8f", width=2.5),
        )
        self.fit_region.setZValue(12.0)
        ratio_plot.addItem(self.fit_region)
        self.peak_regions = tuple(
            pg.LinearRegionItem(
                values=values, orientation="horizontal",
                brush=pg.mkBrush(*color, 35), pen=pg.mkPen(color, width=1.5),
                hoverPen=pg.mkPen(color, width=2.5),
            )
            for values, color in (((0.1, 0.45), (39, 125, 161)),
                                  ((0.55, 0.9), (248, 150, 30)))
        )
        for region in self.peak_regions:
            region.setZValue(12.0)
            region.hide()
            ratio_plot.addItem(region)
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

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem("Double Gaussian", "double")
        self.mode_combo.addItem("Separate Gaussians", "separate")
        self.mode_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.mode_combo.setMinimumContentsLength(10)
        self.mode_combo.setToolTip(
            "Use a joint two-peak fit, or independently fit a Gaussian plus local background "
            "inside each colored range for well-separated populations."
        )
        self.background_checkbox = QCheckBox("Use y0 offset", self)
        self.background_checkbox.setChecked(True)
        self.background_checkbox.setToolTip(
            "Fit a non-negative constant count offset y0; separate mode fits one per range. "
            "Uncheck to fix the offset to zero. Press F1 for the model formulas."
        )
        self.continuous_checkbox = QCheckBox("Continuous fit", self)
        self.continuous_checkbox.setToolTip(
            "Refit changed projections and fit guides using the latest data, "
            "at most twice per second."
        )
        self.guides_checkbox = QCheckBox("Fit guides", self)
        self.guides_checkbox.setChecked(True)
        self.guides_checkbox.setToolTip(
            "Show or hide the centroid seeds and the active fit range(s)."
        )
        self.fit_button = QPushButton("Fit", self)
        self.fit_button.setToolTip(
            "Fit the selected Gaussian model to the current projected PSD counts "
            "using bounded Levenberg-Marquardt optimization."
        )
        self.copy_button = QPushButton("Copy table", self)
        self.copy_button.setEnabled(False)
        self.copy_button.setToolTip(
            "Copy the fitted populations and overall FOM diagnostics as a tab-separated table."
        )

        button_row = QHBoxLayout()
        button_row.addWidget(self.guides_checkbox)
        button_row.addStretch(1)
        button_row.addWidget(self.fit_button)

        self.first_label = QLabel("—", self)
        self.second_label = QLabel("—", self)
        self.first_label.setToolTip(
            "Population 1: fitted centroid μ / full width at half maximum."
        )
        self.second_label.setToolTip(
            "Population 2: fitted centroid μ / full width at half maximum."
        )
        self.fom_label = QLabel("—", self)
        self.fom_label.setToolTip(
            "Centroid separation divided by the sum of the two FWHMs. "
            "Press F1 for the Gaussian/FOM formulas and fit-quality implications."
        )
        self.deviance_label = QLabel("—", self)
        self.chi_square_label = self.deviance_label
        results = QFormLayout()
        results.setContentsMargins(0, 0, 0, 0)
        results.addRow("μ₁ / FWHM₁:", self.first_label)
        results.addRow("μ₂ / FWHM₂:", self.second_label)
        results.addRow("FOM:", self.fom_label)
        results.addRow("Reduced deviance:", self.deviance_label)

        copy_row = QHBoxLayout()
        copy_row.addStretch(1)
        copy_row.addWidget(self.copy_button)

        self.status_label = QLabel("No fit yet.", self)
        self.status_label.setToolTip(
            "Drag the ratio fit ROI and μ centroid markers, then press Fit. "
            "A manual fit is a snapshot; enable Continuous fit to follow changes. "
            "Separate mode uses blue population-1 and orange population-2 fit bands. "
            "Press F1 for fit-quality guidance."
        )
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        model_row = QFormLayout()
        model_row.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        model_row.addRow("Fit model:", self.mode_combo)
        layout.addLayout(model_row)
        layout.addWidget(self.background_checkbox)
        layout.addWidget(self.continuous_checkbox)
        layout.addLayout(button_row)
        layout.addLayout(results)
        layout.addLayout(copy_row)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.background_checkbox.toggled.connect(self.clear_fit)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.continuous_checkbox.toggled.connect(self._continuous_changed)
        self.guides_checkbox.toggled.connect(self.set_guides_visible)
        self.fit_button.clicked.connect(self.fit_current_projection)
        self.copy_button.clicked.connect(self.copy_results_table)
        self.fit_region.sigRegionChangeFinished.connect(self.clear_fit)
        for region in self.peak_regions:
            region.sigRegionChangeFinished.connect(self.clear_fit)
        for line in self.centroid_lines:
            line.sigPositionChangeFinished.connect(self.clear_fit)
        wrap_widget_tooltips(self)

    def set_projection(self, ratio: np.ndarray, counts: np.ndarray) -> None:
        """Store the latest ratio projection for manual or continuous fitting."""
        ratio_values = np.asarray(ratio, dtype=np.float64)
        count_values = np.asarray(counts, dtype=np.float64)
        if ratio_values.ndim != 1 or ratio_values.shape != count_values.shape:
            raise ValueError("PSD ratio projection coordinates and counts must match")
        changed = not (
            np.array_equal(self._ratio, ratio_values) and np.array_equal(self._counts, count_values)
        )
        self._ratio = ratio_values.copy()
        self._counts = count_values.copy()
        if changed:
            self._fit_dirty = True
            if self.last_fit is not None:
                if self.continuous_checkbox.isChecked():
                    # Keep the previous numbers visible between live updates,
                    # but invalidate exports until the latest projection fits.
                    self.last_fit = None
                    self.copy_button.setEnabled(False)
                    self.status_label.setStyleSheet("")
                    self.status_label.setText("Updating fit...")
                    for curve in (self._fit_curve, self._first_curve, self._second_curve):
                        curve.setOpacity(0.4)
                    self.fit_changed.emit(None)
                else:
                    self.clear_fit()

    def set_ratio_range(self, ratio_range: tuple[float, float]) -> None:
        """Initialize guides once, then preserve positions within new bounds."""
        low, high = sorted(map(float, ratio_range))
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("PSD ratio range must contain two increasing finite values")
        if self._guide_range == (low, high):
            return
        if self._guide_range is None:
            span = high - low
            region = (low + 0.1 * span, low + 0.9 * span)
            seeds: tuple[float, ...] = (low + 0.35 * span, low + 0.65 * span)
        else:
            region = tuple(np.clip(self.fit_region.getRegion(), low, high))
            seeds = tuple(float(np.clip(line.value(), low, high)) for line in self.centroid_lines)
        for peak_region, fractions in zip(
            self.peak_regions, ((0.1, 0.45), (0.55, 0.9)), strict=True,
        ):
            values = (
                tuple(low + fraction * (high - low) for fraction in fractions)
                if self._guide_range is None
                else tuple(np.clip(peak_region.getRegion(), low, high))
            )
            with QSignalBlocker(peak_region):
                peak_region.setBounds((low, high))
                peak_region.setRegion(values)
        self._guide_range = (low, high)
        with QSignalBlocker(self.fit_region):
            self.fit_region.setBounds((low, high))
            self.fit_region.setRegion(region)
        for line, seed in zip(self.centroid_lines, seeds, strict=True):
            with QSignalBlocker(line):
                line.setBounds((low, high))
                line.setPos(seed)
        self.clear_fit()

    def set_guides_visible(self, visible: bool) -> None:
        """Toggle all draggable fit objects without hiding fitted curves."""
        separate = self.mode_combo.currentData() == "separate"
        self.fit_region.setVisible(visible and not separate)
        for region in self.peak_regions:
            region.setVisible(visible and separate)
        for line in self.centroid_lines:
            line.setVisible(visible)

    def clear_fit(self) -> None:
        """Remove a previous fit result and its curves."""
        self.last_fit = None
        self._fit_dirty = True
        self._fit_curve.setData([], [])
        self._first_curve.setData([], [])
        self._second_curve.setData([], [])
        self.first_label.setText("—")
        self.second_label.setText("—")
        self.fom_label.setText("—")
        self.deviance_label.setText("—")
        self.copy_button.setEnabled(False)
        self.status_label.setStyleSheet("")
        self.status_label.setText("No fit yet.")
        self.fit_changed.emit(None)

    def fit_current_projection(self, *, _continuous: bool = False) -> PsdFomFit | None:
        """Fit the current projection snapshot and update the result panel."""
        self._fit_dirty = False
        try:
            region = self.fit_region.getRegion()
            centroid_seeds = (
                float(self.centroid_lines[0].value()),
                float(self.centroid_lines[1].value()),
            )
            budget = 400 if _continuous else 10_000
            if self.mode_combo.currentData() == "separate":
                values = [item.getRegion() for item in self.peak_regions]
                rois = ((float(values[0][0]), float(values[0][1])),
                        (float(values[1][0]), float(values[1][1])))
                fit = fit_psd_fom_separate_lm(
                    self._ratio, self._counts,
                    rois=(rois[0], rois[1]), centroid_seeds=centroid_seeds, max_nfev=budget,
                    use_background=self.background_checkbox.isChecked(),
                )
            else:
                fit = fit_psd_fom_lm(
                    self._ratio, self._counts,
                    roi=(float(region[0]), float(region[1])),
                    centroid_seeds=centroid_seeds, max_nfev=budget,
                    use_background=self.background_checkbox.isChecked(),
                )
        except (RuntimeError, TypeError, ValueError) as error:
            self.clear_fit()
            self._fit_dirty = False
            self.status_label.setStyleSheet("color: #b42318; font-weight: bold;")
            self.status_label.setText(f"Fit unavailable: {error}")
            return None

        self.last_fit = fit
        self._show_fit_curves(fit)
        if not _continuous or not fit.poor_convergence:
            for line, centroid in zip(self.centroid_lines, fit.centroids, strict=True):
                line.setPos(centroid)
        self.first_label.setText(_peak_text(fit.centroids[0], fit.fwhms[0]))
        self.second_label.setText(_peak_text(fit.centroids[1], fit.fwhms[1]))
        self.fom_label.setText(f"{fit.fom:.4g}")
        self.deviance_label.setText(f"{fit.reduced_deviance:.3g}")
        self.copy_button.setEnabled(True)
        if fit.poor_convergence:
            details = "; ".join(fit.warnings) or fit.message
            self.status_label.setStyleSheet("color: #b54708; font-weight: bold;")
            self.status_label.setText(f"Warning: poor convergence — {details}.")
        else:
            self.status_label.setStyleSheet("color: #067647;")
            self.status_label.setText(
                f"Fit converged ({fit.evaluations} evaluations)."
            )
        self.fit_completed.emit(fit)
        self.fit_changed.emit(fit)
        return fit

    def _mode_changed(self) -> None:
        if self.mode_combo.currentData() == "separate" and not self._peak_regions_initialized:
            low, high = sorted(map(float, self.fit_region.getRegion()))
            seeds = sorted(float(line.value()) for line in self.centroid_lines)
            split = float(np.clip(sum(seeds) / 2.0, low, high))
            gap = 0.02 * (high - low)
            for region, values in zip(
                self.peak_regions, ((low, split - gap), (split + gap, high)), strict=True,
            ):
                with QSignalBlocker(region):
                    region.setRegion(values)
            self._peak_regions_initialized = True
        self.set_guides_visible(self.guides_checkbox.isChecked())
        self.clear_fit()

    def _continuous_changed(self, enabled: bool) -> None:
        if enabled:
            self._fit_dirty = True
            self._continuous_timer.start()
        else:
            self._continuous_timer.stop()
            if self.last_fit is None:
                self.clear_fit()

    def _fit_latest_projection(self) -> None:
        # A single GUI timer coalesces arrivals without queuing fit jobs. Fits
        # use a smaller iteration budget; no worker owns widgets or survives
        # their destruction. Manual Fit retains the full evaluation budget.
        if self._fit_dirty and self.continuous_checkbox.isChecked() and self.fit_button.isEnabled():
            self.fit_current_projection(_continuous=True)

    def _show_fit_curves(self, fit: PsdFomFit) -> None:
        x = fit.x
        first, second, total = fit.first_component, fit.second_component, fit.best_fit
        if fit.fit_mode == "separate":
            split = int(np.searchsorted(x, fit.fit_regions[1][0]))
            first = np.where(x <= fit.fit_regions[0][1], first, np.nan)
            second = np.where(x >= fit.fit_regions[1][0], second, np.nan)
            x = np.insert(x, split, np.nan)
            first = np.insert(first, split, np.nan)
            second = np.insert(second, split, np.nan)
            total = np.insert(total, split, np.nan)
        for curve in (self._fit_curve, self._first_curve, self._second_curve):
            curve.setOpacity(1.0)
        self._first_curve.setData(first, x, connect="finite")
        self._second_curve.setData(second, x, connect="finite")
        self._fit_curve.setData(total, x, connect="finite")

    def copy_results_table(self) -> str | None:
        """Copy the latest fit as a spreadsheet-friendly TSV table."""
        fit = self.last_fit
        if fit is None:
            return None
        table = format_psd_fom_table(fit)
        QApplication.clipboard().setText(table)
        return table


def _peak_text(centroid: float, fwhm: float) -> str:
    return f"{centroid:.5g} / {fwhm:.5g}"


def format_psd_fom_table(fit: PsdFomFit) -> str:
    """Format both fitted populations and shared diagnostics as TSV."""
    headers = (
        "Population",
        "Centroid",
        "Centroid uncertainty",
        "FWHM",
        "FWHM uncertainty",
        "FOM",
        "Reduced Poisson deviance",
        "Fit status",
        "Evaluations",
        "Warnings",
    )
    status = "Poor convergence" if fit.poor_convergence else "Converged"
    warnings = "; ".join(fit.warnings)
    rows = [headers]
    for index in range(2):
        rows.append(
            (
                str(index + 1),
                _table_number(fit.centroids[index]),
                _table_optional_number(fit.centroid_uncertainties[index]),
                _table_number(fit.fwhms[index]),
                _table_optional_number(fit.fwhm_uncertainties[index]),
                _table_number(fit.fom),
                _table_number(fit.reduced_deviance),
                status,
                str(fit.evaluations),
                warnings,
            )
        )
    return "\n".join("\t".join(row) for row in rows)


def _table_number(value: float) -> str:
    return f"{value:.12g}"


def _table_optional_number(value: float | None) -> str:
    return "" if value is None else _table_number(value)
