"""Modeless multi-spectrum MCA peak-analysis and arithmetic workbench."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt, QThread
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.peak_fitting import (
    BackgroundModel,
    FitStatistic,
    PeakFitResult,
    PeakFitSpec,
    export_peak_fit_result,
    suggest_peak_centers,
)
from nlab.analysis.spectrum import (
    NormalizationMode,
    Spectrum,
    combine_spectra,
    crop_spectrum,
    normalize_spectrum,
    rebin_spectrum,
    scale_spectrum,
    subtract_background,
)
from nlab.analysis.spectrum_io import export_spectrum_csv, load_spectra
from nlab.views.plot_viewbox import ModifierZoomViewBox
from nlab.workers.peak_fit_worker import PeakFitWorker

if TYPE_CHECKING:
    from nlab.controllers.mca_controller import MCAController

log = logging.getLogger(__name__)
_COLORS = ("#1f77b4", "#d28b38", "#648b71", "#7353a6", "#a66f6f", "#2a9d8f")
_FIT_REGION_Z = 20.0
_PEAK_MARKER_Z = 30.0


class _SpectrumEnergyAxis(pg.AxisItem):  # type: ignore[misc]
    """Top axis interpolating the selected spectrum's retained keV coordinates."""

    def __init__(self) -> None:
        super().__init__(orientation="top")
        self.enableAutoSIPrefix(False)
        self._channel: np.ndarray | None = None
        self._energy: np.ndarray | None = None

    def set_spectrum(self, spectrum: Spectrum | None) -> None:
        if spectrum is None or spectrum.axis_unit == "keV" or spectrum.energy_kev is None:
            self._channel = None
            self._energy = None
            self.setLabel(text="")
            self.setToolTip("No secondary energy calibration is available.")
        else:
            self._channel = spectrum.x
            self._energy = spectrum.energy_kev
            stale = bool(spectrum.metadata.get("calibration_stale", False))
            self.setLabel(text="Energy (stale)" if stale else "Energy", units="keV")
            self.setToolTip(
                "Calibration settings no longer match the captured MCA configuration."
                if stale
                else "Energy coordinates retained with the selected spectrum."
            )
        self.picture = None
        self.update()

    def tickStrings(  # noqa: N802 - pyqtgraph API
        self,
        values: list[float],
        scale: float,
        spacing: float,
    ) -> list[str]:
        if self._channel is None or self._energy is None:
            return [str(value) for value in super().tickStrings(values, scale, spacing)]
        energies = np.interp(values, self._channel, self._energy)
        if values:
            first = float(values[0])
            mapped_spacing = abs(
                float(np.interp(first + spacing, self._channel, self._energy))
                - float(np.interp(first, self._channel, self._energy))
            )
        else:
            mapped_spacing = 1.0
        decimals = (
            2
            if not math.isfinite(mapped_spacing) or mapped_spacing <= 0
            else max(0, min(6, int(math.ceil(-math.log10(mapped_spacing))) + 1))
        )
        return [f"{float(energy):.{decimals}f}" for energy in energies]


class McaPeakAnalysisDialog(QDialog):
    """Fit and combine frozen live or file-backed MCA spectra."""

    def __init__(self, controllers: list[MCAController], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MCA Peak Analysis Workbench")
        self.resize(1500, 850)
        self.setMinimumSize(1120, 680)
        self.setModal(False)
        self._controllers = {controller.channel: controller for controller in controllers}
        self._spectra: dict[str, Spectrum] = {}
        self._order: list[str] = []
        self._visible: dict[str, bool] = {}
        self._live_ids: dict[int, str] = {}
        self._fit_result: PeakFitResult | None = None
        self._fit_spectrum_id: str | None = None
        self._last_fit_spec: PeakFitSpec | None = None
        self._fit_thread: QThread | None = None
        self._fit_worker: PeakFitWorker | None = None
        self._peak_lines: list[pg.InfiniteLine] = []
        self._rebuilding_sources = False
        self._allow_close = False
        self._build_ui()
        self._connect_signals()
        self._refresh_all(show_errors=False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        instructions = QLabel(
            "Live MCA data are copied as frozen snapshots. File and derived spectra use the "
            "same fitting pipeline; acquisition is never reconfigured by this workbench.",
            self,
        )
        instructions.setWordWrap(True)
        root.addWidget(instructions)

        toolbar = QHBoxLayout()
        self.load_button = QPushButton("Load spectrum files...", self)
        self.refresh_selected_button = QPushButton("Refresh selected MCA", self)
        self.refresh_all_button = QPushButton("Refresh all MCA spectra", self)
        self.remove_button = QPushButton("Remove selected", self)
        self.export_spectrum_button = QPushButton("Export spectrum...", self)
        for button in (
            self.load_button,
            self.refresh_selected_button,
            self.refresh_all_button,
            self.remove_button,
            self.export_spectrum_button,
        ):
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        toolbar.addWidget(QLabel("Counts:", self))
        self.scale_combo = QComboBox(self)
        self.scale_combo.addItem("Linear", "linear")
        self.scale_combo.addItem("Logarithmic", "log")
        toolbar.addWidget(self.scale_combo)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        sources_panel = QWidget(self)
        sources_panel.setMinimumWidth(260)
        sources_panel.setMaximumWidth(360)
        sources_layout = QVBoxLayout(sources_panel)
        sources_layout.addWidget(QLabel("Spectra", self))
        self.source_list = QListWidget(self)
        self.source_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        sources_layout.addWidget(self.source_list, 1)
        self.source_status = QLabel("No spectra loaded.", self)
        self.source_status.setWordWrap(True)
        self.source_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sources_layout.addWidget(self.source_status)
        splitter.addWidget(sources_panel)

        plot_widget = pg.GraphicsLayoutWidget(self)
        plot_widget.setBackground("#f8f9fa")
        self.energy_axis = _SpectrumEnergyAxis()
        self.spectrum_plot = plot_widget.addPlot(
            row=0,
            col=0,
            viewBox=ModifierZoomViewBox(),
            axisItems={"top": self.energy_axis},
        )
        self.spectrum_plot.showAxis("top")
        self.spectrum_plot.showAxis("right")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.2)
        self.spectrum_plot.setLabel("left", "Counts")
        self.legend = self.spectrum_plot.addLegend()
        self.fit_region = pg.LinearRegionItem(values=(0.0, 1.0), movable=True)
        self.fit_region.setZValue(_FIT_REGION_Z)
        self.spectrum_plot.addItem(self.fit_region)
        self.residual_plot = plot_widget.addPlot(
            row=1,
            col=0,
            viewBox=ModifierZoomViewBox(),
        )
        self.residual_plot.setMaximumHeight(210)
        self.residual_plot.setXLink(self.spectrum_plot)
        self.residual_plot.showGrid(x=True, y=True, alpha=0.2)
        self.residual_plot.setLabel("left", "Residual")
        self.residual_plot.setLabel("bottom", "Coordinate")
        self.residual_curve = self.residual_plot.plot(pen=pg.mkPen("#555555", width=1))
        splitter.addWidget(plot_widget)

        self.side_tabs = QTabWidget(self)
        self.side_tabs.setMinimumWidth(390)
        self.side_tabs.setMaximumWidth(520)
        self.side_tabs.addTab(self._build_fit_tab(), "Peak fit")
        self.side_tabs.addTab(self._build_operations_tab(), "Operations")
        self.side_tabs.addTab(self._build_results_tab(), "Results")
        splitter.addWidget(self.side_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([290, 800, 420])
        root.addWidget(splitter, 1)

        close_row = QHBoxLayout()
        self.workbench_status = QLabel("Ready.", self)
        self.workbench_status.setWordWrap(True)
        self.workbench_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        close_row.addWidget(self.workbench_status, 1)
        self.close_button = QPushButton("Close", self)
        close_row.addWidget(self.close_button)
        root.addLayout(close_row)

    def _build_fit_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.peak_count = QSpinBox(self)
        self.peak_count.setRange(1, 3)
        self.background_combo = QComboBox(self)
        self.background_combo.addItem("None", "none")
        self.background_combo.addItem("Constant", "constant")
        self.background_combo.addItem("Linear", "linear")
        self.background_combo.addItem("Exponential", "exponential")
        self.background_combo.addItem("Compton step (erfc)", "compton")
        self.background_combo.addItem("Fermi step (logistic)", "fermi")
        self.background_combo.setCurrentIndex(self.background_combo.findData("linear"))
        self.statistic_combo = QComboBox(self)
        self.statistic_combo.addItem("Auto (count-aware)", "auto")
        self.statistic_combo.addItem("Poisson deviance", "poisson")
        self.statistic_combo.addItem("Weighted least squares", "weighted")
        self.statistic_combo.addItem("Unweighted least squares", "unweighted")
        form.addRow("Gaussian peaks:", self.peak_count)
        form.addRow("Background:", self.background_combo)
        form.addRow("Statistic:", self.statistic_combo)
        layout.addLayout(form)

        marker_row = QHBoxLayout()
        self.suggest_button = QPushButton("Suggest peak positions", self)
        self.fit_button = QPushButton("Fit selected spectrum", self)
        self.cancel_fit_button = QPushButton("Cancel", self)
        self.cancel_fit_button.setEnabled(False)
        marker_row.addWidget(self.suggest_button)
        marker_row.addWidget(self.fit_button)
        marker_row.addWidget(self.cancel_fit_button)
        layout.addLayout(marker_row)
        help_label = QLabel(
            "Drag the shaded fit range and dashed peak markers. Marker positions provide "
            "bounded initial centres; the source spectrum remains unchanged.",
            self,
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.fit_status = QLabel("Select a spectrum to configure a fit.", self)
        self.fit_status.setWordWrap(True)
        self.fit_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.fit_status)
        layout.addStretch(1)
        return tab

    def _build_operations_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        explanation = QLabel(
            "Operations create a new immutable spectrum and propagate variance. Arithmetic "
            "requires identical coordinate grids; no interpolation is performed.",
            self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        form = QFormLayout()
        self.operation_combo = QComboBox(self)
        self.operation_combo.addItem("Scale", "scale")
        self.operation_combo.addItem("Normalize by area", "normalize_area")
        self.operation_combo.addItem("Normalize by maximum", "normalize_maximum")
        self.operation_combo.addItem("Normalize by acquisition time", "normalize_elapsed")
        self.operation_combo.addItem("Add spectrum", "add")
        self.operation_combo.addItem("Subtract spectrum", "subtract")
        self.operation_combo.addItem("Subtract background", "background")
        self.operation_combo.addItem("Integer rebin", "rebin")
        self.operation_combo.addItem("Crop to fit range", "crop")
        self.other_spectrum_combo = QComboBox(self)
        self.operation_factor = QDoubleSpinBox(self)
        self.operation_factor.setDecimals(8)
        self.operation_factor.setRange(-1.0e9, 1.0e9)
        self.operation_factor.setValue(1.0)
        self.rebin_factor = QSpinBox(self)
        self.rebin_factor.setRange(1, 16_384)
        self.rebin_factor.setValue(2)
        self.auto_time_scale = QCheckBox("Use live/elapsed-time ratio", self)
        self.auto_time_scale.setChecked(True)
        form.addRow("Operation:", self.operation_combo)
        form.addRow("Other spectrum:", self.other_spectrum_combo)
        form.addRow("Scale/factor:", self.operation_factor)
        form.addRow("Rebin factor:", self.rebin_factor)
        form.addRow("Background scaling:", self.auto_time_scale)
        layout.addLayout(form)
        self.apply_operation_button = QPushButton("Create derived spectrum", self)
        layout.addWidget(self.apply_operation_button)
        self.operation_status = QLabel("", self)
        self.operation_status.setWordWrap(True)
        self.operation_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.operation_status)
        layout.addStretch(1)
        return tab

    def _build_results_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        self.peak_table = QTableWidget(0, 7, self)
        self.peak_table.setHorizontalHeaderLabels(
            [
                "Peak",
                "Centre",
                "E\n(keV)",
                "FWHM",
                "Area",
                "Res.\n(%)",
                "Centre\nSE",
            ]
        )
        self.peak_table.verticalHeader().setVisible(False)
        self.peak_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        peak_header = self.peak_table.horizontalHeader()
        header_font = peak_header.font()
        if header_font.pointSizeF() > 0:
            header_font.setPointSizeF(max(7.0, header_font.pointSizeF() - 1.0))
        elif header_font.pixelSize() > 0:
            header_font.setPixelSize(max(9, header_font.pixelSize() - 1))
        peak_header.setFont(header_font)
        peak_header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        peak_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header_tooltips = (
            "Peak component identifier",
            "Fitted centre in the spectrum coordinate unit",
            "Calibrated peak energy in keV",
            "Full width at half maximum",
            "Integrated Gaussian peak area",
            "FWHM divided by the calibrated energy",
            "Standard error of the fitted centre",
        )
        for column, tooltip in enumerate(header_tooltips):
            item = self.peak_table.horizontalHeaderItem(column)
            if item is not None:
                item.setToolTip(tooltip)
        layout.addWidget(self.peak_table)
        self.parameter_table = QTableWidget(0, 5, self)
        self.parameter_table.setHorizontalHeaderLabels(
            ["Parameter", "Value", "Std. error", "Bounds", "Constraint"]
        )
        self.parameter_table.verticalHeader().setVisible(False)
        self.parameter_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.parameter_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.parameter_table)
        self.fit_summary = QLabel("No fit result.", self)
        self.fit_summary.setWordWrap(True)
        self.fit_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.fit_summary)
        self.export_fit_button = QPushButton("Export fit results...", self)
        self.export_fit_button.setEnabled(False)
        layout.addWidget(self.export_fit_button)
        return tab

    def _connect_signals(self) -> None:
        self.load_button.clicked.connect(self._load_files)
        self.refresh_selected_button.clicked.connect(self._refresh_selected)
        self.refresh_all_button.clicked.connect(self._refresh_all)
        self.remove_button.clicked.connect(self._remove_selected)
        self.export_spectrum_button.clicked.connect(self._export_selected_spectrum)
        self.scale_combo.currentIndexChanged.connect(self._render)
        self.source_list.currentItemChanged.connect(self._selection_changed)
        self.source_list.itemChanged.connect(self._visibility_changed)
        self.peak_count.valueChanged.connect(self._suggest_peaks)
        self.suggest_button.clicked.connect(self._suggest_peaks)
        self.fit_button.clicked.connect(self._start_fit)
        self.cancel_fit_button.clicked.connect(self._cancel_fit)
        self.operation_combo.currentIndexChanged.connect(self._update_operation_controls)
        self.apply_operation_button.clicked.connect(self._apply_operation)
        self.export_fit_button.clicked.connect(self._export_fit_result)
        self.close_button.clicked.connect(self.reject)

    def _current_id(self) -> str | None:
        item = self.source_list.currentItem()
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        return str(value) if value else None

    def _current_spectrum(self) -> Spectrum | None:
        spectrum_id = self._current_id()
        return self._spectra.get(spectrum_id) if spectrum_id is not None else None

    def _add_spectrum(self, spectrum: Spectrum, *, select: bool = True) -> None:
        self._spectra[spectrum.spectrum_id] = spectrum
        if spectrum.spectrum_id not in self._order:
            self._order.append(spectrum.spectrum_id)
        self._visible.setdefault(spectrum.spectrum_id, True)
        self._rebuild_source_list(select_id=spectrum.spectrum_id if select else self._current_id())

    def _rebuild_source_list(self, *, select_id: str | None = None) -> None:
        current = select_id or self._current_id()
        self._rebuilding_sources = True
        self.source_list.clear()
        selected_item: QListWidgetItem | None = None
        for spectrum_id in self._order:
            spectrum = self._spectra.get(spectrum_id)
            if spectrum is None:
                continue
            item = QListWidgetItem(f"{spectrum.label}  [{spectrum.source}]", self.source_list)
            item.setData(Qt.ItemDataRole.UserRole, spectrum_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if self._visible.get(spectrum_id, True)
                else Qt.CheckState.Unchecked
            )
            if spectrum_id == current:
                selected_item = item
        self._rebuilding_sources = False
        if selected_item is not None:
            self.source_list.setCurrentItem(selected_item)
        elif self.source_list.count():
            self.source_list.setCurrentRow(0)
        self._update_other_spectra()
        self._update_source_status()

    def _selection_changed(self, *_args: object) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            self.fit_button.setEnabled(False)
            self.export_spectrum_button.setEnabled(False)
            self._render()
            return
        self.fit_button.setEnabled(self._fit_thread is None)
        self.export_spectrum_button.setEnabled(True)
        self.fit_region.setRegion((float(spectrum.x[0]), float(spectrum.x[-1])))
        self._fit_result = None
        self._fit_spectrum_id = None
        self.export_fit_button.setEnabled(False)
        self._suggest_peaks()
        self._update_other_spectra()
        self._update_source_status()
        self._render()

    def _visibility_changed(self, item: QListWidgetItem) -> None:
        if self._rebuilding_sources:
            return
        spectrum_id = str(item.data(Qt.ItemDataRole.UserRole))
        self._visible[spectrum_id] = item.checkState() == Qt.CheckState.Checked
        self._render()

    def _render(self, *_args: object) -> None:
        selected = self._current_spectrum()
        self.spectrum_plot.clear()
        self.legend.clear()
        self.spectrum_plot.addItem(self.fit_region)
        selected_unit = selected.axis_unit if selected is not None else "channel"
        plotted = 0
        incompatible = 0
        for index, spectrum_id in enumerate(self._order):
            spectrum = self._spectra.get(spectrum_id)
            if spectrum is None or not self._visible.get(spectrum_id, True):
                continue
            if spectrum.axis_unit != selected_unit:
                incompatible += 1
                continue
            color = _COLORS[index % len(_COLORS)]
            width = 1.8 if spectrum is selected else 1.0
            self.spectrum_plot.plot(
                spectrum.x,
                spectrum.counts,
                pen=pg.mkPen(color, width=width),
                name=spectrum.label,
            )
            plotted += 1
        for line in self._peak_lines:
            self.spectrum_plot.addItem(line)
        logarithmic = self.scale_combo.currentData() == "log"
        self.spectrum_plot.setLogMode(y=logarithmic)
        self.spectrum_plot.setLabel(
            "bottom",
            "Energy" if selected_unit == "keV" else "MCA channel",
            units="keV" if selected_unit == "keV" else None,
        )
        self.residual_plot.setLabel(
            "bottom",
            "Energy" if selected_unit == "keV" else "MCA channel",
            units="keV" if selected_unit == "keV" else None,
        )
        self._refresh_energy_axis(selected)
        self.residual_curve.setData([], [])
        if self._fit_result is not None and self._fit_spectrum_id == self._current_id():
            fit = self._fit_result
            self.spectrum_plot.plot(fit.x, fit.best_fit, pen=pg.mkPen("#d62728", width=2.0))
            component_colors = ("#9467bd", "#2ca02c", "#17becf", "#8c564b")
            for index, values in enumerate(fit.components.values()):
                self.spectrum_plot.plot(
                    fit.x,
                    values,
                    pen=pg.mkPen(component_colors[index % len(component_colors)], width=1.0),
                )
            self.residual_curve.setData(fit.x, fit.residual)
        suffix = (
            f"; {incompatible} visible spectrum/spectra use another axis"
            if incompatible
            else ""
        )
        if selected is not None:
            self.workbench_status.setText(f"Displayed {plotted} spectrum/spectra{suffix}.")

    def _refresh_energy_axis(self, spectrum: Spectrum | None) -> None:
        self.energy_axis.set_spectrum(spectrum)

    def _snapshot_spectrum(self, controller: MCAController, *, spectrum_id: str | None) -> Spectrum:
        snapshot = controller.spectrum_snapshot()
        x = np.arange(len(snapshot.counts), dtype=np.float64)
        calibration = controller.energy_calibration
        binning = controller.coincidence_energy_bin
        energy = (
            np.asarray(calibration.energy_at_binning(x, binning), dtype=np.float64)
            if calibration is not None
            else None
        )
        return Spectrum.create(
            spectrum_id=spectrum_id,
            label=snapshot.label,
            x=x,
            counts=snapshot.counts,
            axis_unit="channel",
            source="mca",
            metadata={
                "channel": snapshot.channel,
                "created_utc": snapshot.created_utc,
                "elapsed_s": snapshot.elapsed_s,
                "live_non_atomic": snapshot.live,
                "settings_fingerprint": snapshot.fingerprint,
                "binning": binning,
                "energy_calibration": calibration.to_dict() if calibration is not None else None,
                "calibration_stale": controller.energy_calibration_is_stale(),
            },
            energy_kev=energy,
            poisson_counts=True,
        )

    def _refresh_all(self, _checked: bool = False, *, show_errors: bool = True) -> None:
        errors: list[str] = []
        selected = self._current_id()
        for channel, controller in sorted(self._controllers.items()):
            spectrum_id = self._live_ids.get(channel)
            try:
                spectrum = self._snapshot_spectrum(controller, spectrum_id=spectrum_id)
            except RuntimeError as exc:
                errors.append(str(exc))
                continue
            if spectrum_id is None:
                self._live_ids[channel] = spectrum.spectrum_id
                self._order.append(spectrum.spectrum_id)
                self._visible[spectrum.spectrum_id] = True
            self._spectra[spectrum.spectrum_id] = spectrum
        self._rebuild_source_list(select_id=selected)
        if errors:
            self.source_status.setText("; ".join(errors))
            if show_errors:
                QMessageBox.information(self, "MCA Peak Analysis", "\n".join(errors))

    def _refresh_selected(self) -> None:
        spectrum_id = self._current_id()
        if spectrum_id is None:
            return
        channel = next(
            (channel for channel, live_id in self._live_ids.items() if live_id == spectrum_id),
            None,
        )
        if channel is None:
            self.source_status.setText("The selected file or derived spectrum cannot be refreshed.")
            return
        try:
            spectrum = self._snapshot_spectrum(
                self._controllers[channel],
                spectrum_id=spectrum_id,
            )
        except RuntimeError as exc:
            QMessageBox.information(self, "MCA Peak Analysis", str(exc))
            return
        self._spectra[spectrum_id] = spectrum
        self._rebuild_source_list(select_id=spectrum_id)
        self._selection_changed()

    def _load_files(self) -> None:
        folder = str(QSettings().value("analysis/spectrum_folder", "measurements"))
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load MCA Spectra",
            folder,
            "Spectrum files (*.csv *.CSV *.spe *.SPE *.wdm *.WDM *.txt *.txt3 *.tsv "
            "*.root *.ROOT);;All files (*)",
        )
        if not paths:
            return
        loaded = 0
        failures: list[str] = []
        for raw_path in paths:
            path = Path(raw_path)
            try:
                spectra = load_spectra(path)
                for spectrum in spectra:
                    self._add_spectrum(spectrum, select=True)
                loaded += len(spectra)
            except (OSError, ValueError) as exc:
                failures.append(f"{path.name}: {exc}")
        QSettings().setValue("analysis/spectrum_folder", str(Path(paths[0]).parent))
        if self.source_list.count() and self.source_list.currentRow() < 0:
            self.source_list.setCurrentRow(0)
        if failures:
            QMessageBox.warning(self, "Spectrum Load", "\n".join(failures))
        self.source_status.setText(f"Loaded {loaded} spectrum file(s).")

    def _remove_selected(self) -> None:
        spectrum_id = self._current_id()
        if spectrum_id is None:
            return
        self._spectra.pop(spectrum_id, None)
        self._visible.pop(spectrum_id, None)
        if spectrum_id in self._order:
            self._order.remove(spectrum_id)
        for channel, live_id in tuple(self._live_ids.items()):
            if live_id == spectrum_id:
                del self._live_ids[channel]
        self._fit_result = None
        self._fit_spectrum_id = None
        self._rebuild_source_list()
        self._render()

    def _update_source_status(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            self.source_status.setText("No spectrum selected.")
            return
        state = "Poisson counts" if spectrum.poisson_counts else "derived/weighted data"
        timing = (
            f", normalization time {spectrum.normalization_time_s:.3g} s"
            if spectrum.normalization_time_s is not None
            else ""
        )
        live = "; live, non-atomic snapshot" if spectrum.metadata.get("live_non_atomic") else ""
        history = f"; {len(spectrum.history)} operation(s)" if spectrum.history else ""
        self.source_status.setText(
            f"{len(spectrum.counts):,} bins, axis {spectrum.axis_unit}, "
            f"{state}{timing}{live}{history}."
        )

    def _suggest_peaks(self, *_args: object) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        low, high = sorted(map(float, self.fit_region.getRegion()))
        mask = (spectrum.x >= low) & (spectrum.x <= high)
        if np.count_nonzero(mask) < self.peak_count.value():
            self.fit_status.setText("The fit range is too narrow for peak suggestions.")
            return
        try:
            centres = suggest_peak_centers(
                spectrum.x[mask],
                spectrum.counts[mask],
                self.peak_count.value(),
            )
        except ValueError as exc:
            self.fit_status.setText(str(exc))
            return
        for line in self._peak_lines:
            try:
                self.spectrum_plot.removeItem(line)
            except RuntimeError:
                pass
        self._peak_lines = []
        for index, centre in enumerate(centres, start=1):
            line = pg.InfiniteLine(
                pos=centre,
                angle=90,
                movable=True,
                pen=pg.mkPen("#a66f6f", width=1.5, style=Qt.PenStyle.DashLine),
                hoverPen=pg.mkPen("#c17c7c", width=2),
                label=f"P{index}",
                labelOpts={"color": "#a66f6f", "position": 0.92},
            )
            line.setBounds((low, high))
            line.setZValue(_PEAK_MARKER_Z)
            self._peak_lines.append(line)
        self.fit_status.setText("Peak positions suggested; drag markers or start the fit.")
        self._render()

    def _start_fit(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None or self._fit_thread is not None:
            return
        if len(self._peak_lines) != self.peak_count.value():
            self._suggest_peaks()
        low, high = sorted(map(float, self.fit_region.getRegion()))
        spec = PeakFitSpec(
            peak_count=self.peak_count.value(),
            background=cast(BackgroundModel, str(self.background_combo.currentData())),
            statistic=cast(FitStatistic, str(self.statistic_combo.currentData())),
            fit_min=low,
            fit_max=high,
            peak_centers=tuple(sorted(float(line.value()) for line in self._peak_lines)),
        )
        thread = QThread(self)
        worker = PeakFitWorker(spectrum, spec)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result.connect(self._fit_completed)
        worker.error.connect(self._fit_failed)
        worker.cancelled.connect(lambda: self.fit_status.setText("Peak fit cancelled."))
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            self._fit_thread_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        thread.finished.connect(thread.deleteLater)
        self._fit_thread = thread
        self._fit_worker = worker
        self._fit_spectrum_id = spectrum.spectrum_id
        self._last_fit_spec = spec
        self.fit_button.setEnabled(False)
        self.cancel_fit_button.setEnabled(True)
        self.fit_status.setText("Fitting...")
        thread.start()

    def _fit_completed(self, result: object) -> None:
        if not isinstance(result, PeakFitResult):
            return
        self._fit_result = result
        self.fit_status.setText(
            f"Fit complete using {result.statistic}; {result.evaluations} evaluations."
            if result.success
            else f"Fit stopped without convergence: {result.message}"
        )
        self._populate_results(result)
        self.export_fit_button.setEnabled(True)
        self._render()

    def _fit_failed(self, message: str) -> None:
        self.fit_status.setText(f"Fit failed: {message}")
        self.workbench_status.setText(f"Peak fit failed: {message}")
        QMessageBox.critical(self, "Peak Fit Failed", message)

    def _fit_thread_finished(self) -> None:
        self._fit_thread = None
        self._fit_worker = None
        self.fit_button.setEnabled(self._current_spectrum() is not None)
        self.cancel_fit_button.setEnabled(False)

    def _cancel_fit(self) -> None:
        if self._fit_worker is not None:
            self._fit_worker.stop()
            self.cancel_fit_button.setEnabled(False)
            self.fit_status.setText("Cancelling fit...")

    def _populate_results(self, result: PeakFitResult) -> None:
        self.peak_table.setRowCount(len(result.peaks))
        for row, peak in enumerate(result.peaks):
            values = (
                f"P{peak.index}",
                f"{peak.center:.8g}",
                "" if peak.energy_kev is None else f"{peak.energy_kev:.8g}",
                (
                    f"{peak.energy_fwhm_kev:.8g} keV"
                    if peak.energy_fwhm_kev is not None
                    else f"{peak.fwhm:.8g}"
                ),
                f"{peak.area:.8g}",
                "" if peak.resolution_percent is None else f"{peak.resolution_percent:.5g}",
                "" if peak.center_stderr is None else f"{peak.center_stderr:.5g}",
            )
            for column, value in enumerate(values):
                self.peak_table.setItem(row, column, QTableWidgetItem(value))
        self.parameter_table.setRowCount(len(result.parameters))
        for row, parameter in enumerate(result.parameters):
            bounds = f"[{parameter.minimum:.5g}, {parameter.maximum:.5g}]"
            parameter_values = (
                parameter.name,
                f"{parameter.value:.10g}",
                "" if parameter.stderr is None else f"{parameter.stderr:.5g}",
                bounds,
                parameter.expression or ("vary" if parameter.varying else "fixed"),
            )
            for column, value in enumerate(parameter_values):
                self.parameter_table.setItem(row, column, QTableWidgetItem(value))
        warnings = " ".join(result.warnings) if result.warnings else "No fit warnings."
        self.fit_summary.setText(
            f"Statistic: {result.statistic}; chi-square {result.chi_square:.6g}; "
            f"reduced chi-square {result.reduced_chi_square:.6g}; AIC {result.aic:.6g}; "
            f"BIC {result.bic:.6g}. {warnings}"
        )

    def _update_other_spectra(self) -> None:
        current = self._current_id()
        previous = self.other_spectrum_combo.currentData()
        self.other_spectrum_combo.clear()
        for spectrum_id in self._order:
            if spectrum_id == current or spectrum_id not in self._spectra:
                continue
            self.other_spectrum_combo.addItem(self._spectra[spectrum_id].label, spectrum_id)
        index = self.other_spectrum_combo.findData(previous)
        if index >= 0:
            self.other_spectrum_combo.setCurrentIndex(index)
        self._update_operation_controls()

    def _update_operation_controls(self, *_args: object) -> None:
        operation = str(self.operation_combo.currentData())
        needs_other = operation in {"add", "subtract", "background"}
        self.other_spectrum_combo.setEnabled(needs_other)
        self.operation_factor.setEnabled(operation in {"scale", "add", "subtract", "background"})
        self.rebin_factor.setEnabled(operation == "rebin")
        self.auto_time_scale.setEnabled(operation == "background")
        self.apply_operation_button.setEnabled(
            self._current_spectrum() is not None
            and (not needs_other or self.other_spectrum_combo.count() > 0)
        )

    def _apply_operation(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        operation = str(self.operation_combo.currentData())
        try:
            if operation == "scale":
                derived = scale_spectrum(spectrum, self.operation_factor.value())
            elif operation.startswith("normalize_"):
                mode = operation.removeprefix("normalize_")
                derived = normalize_spectrum(
                    spectrum,
                    cast(NormalizationMode, mode),
                )
            elif operation in {"add", "subtract", "background"}:
                other_id = str(self.other_spectrum_combo.currentData())
                other = self._spectra[other_id]
                if operation == "background":
                    scale = (
                        None
                        if self.auto_time_scale.isChecked()
                        else self.operation_factor.value()
                    )
                    derived = subtract_background(spectrum, other, scale=scale)
                else:
                    derived = combine_spectra(
                        spectrum,
                        other,
                        operation=cast(Literal["add", "subtract"], operation),
                        right_scale=self.operation_factor.value(),
                    )
            elif operation == "rebin":
                derived = rebin_spectrum(spectrum, self.rebin_factor.value())
            elif operation == "crop":
                low, high = self.fit_region.getRegion()
                derived = crop_spectrum(spectrum, float(low), float(high))
            else:
                raise ValueError(f"unsupported operation {operation!r}")
        except (KeyError, ValueError) as exc:
            self.operation_status.setText(f"Operation failed: {exc}")
            return
        self._add_spectrum(derived)
        self.operation_status.setText(
            f"Created {derived.label}; variance and {len(derived.history)} operation(s) preserved."
        )

    def _export_selected_spectrum(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        folder = str(QSettings().value("analysis/spectrum_folder", "measurements"))
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Export Spectrum",
            str(Path(folder) / f"{_safe_stem(spectrum.label)}.csv"),
            "CSV files (*.csv)",
        )
        if not path_text:
            return
        try:
            export_spectrum_csv(Path(path_text), spectrum)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Spectrum Export Failed", str(exc))
            return
        QSettings().setValue("analysis/spectrum_folder", str(Path(path_text).parent))
        self.workbench_status.setText(f"Spectrum exported to {path_text}")

    def _export_fit_result(self) -> None:
        result = self._fit_result
        spectrum = self._spectra.get(self._fit_spectrum_id or "")
        if result is None or spectrum is None:
            return
        folder = str(QSettings().value("analysis/spectrum_folder", "measurements"))
        path_text, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Peak Fit Results",
            str(Path(folder) / f"{_safe_stem(spectrum.label)}_fit.json"),
            "JSON (*.json);;CSV (*.csv)",
        )
        if not path_text:
            return
        path = Path(path_text)
        try:
            if not path.suffix:
                path = path.with_suffix(".csv" if selected_filter.startswith("CSV") else ".json")
            if self._last_fit_spec is None:
                raise ValueError("the fit model description is unavailable")
            export_peak_fit_result(
                path,
                spectrum=spectrum,
                spec=self._last_fit_spec,
                result=result,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Fit Export Failed", str(exc))
            return
        self.workbench_status.setText(f"Fit results exported to {path}")

    def show_workspace(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def close_without_prompt(self) -> None:
        self._allow_close = True
        self._stop_fit_sync()
        self.close()

    def _stop_fit_sync(self) -> None:
        worker = self._fit_worker
        thread = self._fit_thread
        if worker is not None:
            worker.stop()
        if thread is not None and thread.isRunning() and not thread.wait(5_000):
            log.warning("MCA peak-fit worker did not stop within 5 seconds")

    def reject(self) -> None:
        self._stop_fit_sync()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if not self._allow_close:
            self._stop_fit_sync()
        super().closeEvent(event)


def _safe_stem(label: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in label
    )
    return cleaned.strip("_") or "spectrum"
