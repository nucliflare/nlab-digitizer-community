"""Modeless multi-spectrum MCA peak-analysis and arithmetic workbench."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QMetaMethod, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QActionGroup, QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nlab import settings_keys
from nlab.analysis import dietze_klein_table
from nlab.analysis.compton_edge import (
    COMPTON_EDGE_ENERGY_UNIT,
    COMPTON_PRESET_NUCLIDES,
    AmplitudeMode,
    ComptonBackground,
    ComptonEdgeDataset,
    ComptonEdgeFitProgress,
    ComptonEdgeFitResult,
    ComptonEdgeFitSpec,
    ComptonLine,
    ResolutionTermName,
    SimulatedResponse,
    compton_edge_calibration_points,
    compton_edge_summary_rows,
    export_compton_edge_result,
    load_simulated_response,
    nuclide_compton_lines,
    parse_custom_energies,
)
from nlab.analysis.energy_calibration import CalibrationPointTransfer, FingerprintValue
from nlab.analysis.n42 import export_n42_peak_fit, export_n42_spectrum
from nlab.analysis.peak_fitting import (
    BackgroundModel,
    FitStatistic,
    PeakFitResult,
    PeakFitSpec,
    export_peak_fit_result,
    suggest_peak_centers,
)
from nlab.analysis.radionuclide_lines import (
    BUILTIN_RADIONUCLIDE_CATALOG,
    RadionuclideCandidate,
    RadionuclideLine,
    canonical_nuclide,
    coordinate_for_energy,
    energy_range_for_coordinates,
    has_usable_energy_axis,
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
from nlab.views import theme
from nlab.views.plot_viewbox import ModifierZoomViewBox, set_frame_only, show_frame_axes
from nlab.views.tooltips import wrap_tooltip, wrap_widget_tooltips
from nlab.views.window_flags import make_tool_window
from nlab.views.workbench_menu import WorkbenchMenuBar
from nlab.workers.compton_edge_fit_worker import ComptonEdgeFitWorker
from nlab.workers.peak_fit_worker import PeakFitWorker

if TYPE_CHECKING:
    from nlab.controllers.mca_controller import MCAController

log = logging.getLogger(__name__)
_COLORS = ("#1f77b4", "#d28b38", "#648b71", "#7353a6", "#a66f6f", "#2a9d8f")
_FIT_REGION_Z = 20.0
_REFERENCE_LINE_Z = 25.0
_PEAK_MARKER_Z = 30.0
_COMPTON_EDGE_Z = 28.0
_COMPTON_COLORS = ("#9467bd", "#2ca02c", "#17becf", "#8c564b", "#e377c2", "#bcbd22")


@dataclass
class _ComptonDatasetConfig:
    """Per-spectrum Compton-edge settings kept while the workbench is open."""

    include: bool = False
    nuclides: list[str] = field(default_factory=list)
    custom_energies: str = ""
    background: ComptonBackground = "exponential"
    fit_range: tuple[float, float] | None = None
    responses: dict[float, SimulatedResponse] = field(default_factory=dict)
_REFERENCE_COLORS = ("#c23b22", "#0072b2", "#009e73", "#cc79a7", "#e69f00", "#6f4e7c")
_REFERENCE_FAVOURITES_KEY = settings_keys.ANALYSIS_RADIONUCLIDE_FAVOURITES
_DEFAULT_REFERENCE_FAVOURITES = (
    "Cs-137",
    "Na-22",
    "Bi-207",
    "Mn-54",
    "Cr-51",
    "Co-57",
    "Cd-109",
    "Am-241",
    "Pm-145",
    "Mo-93",
    "Ba-133",
)


class _SpectrumEnergyAxis(pg.AxisItem):  # type: ignore[misc]
    """Top axis interpolating the selected spectrum's retained keV coordinates."""

    def __init__(self) -> None:
        super().__init__(orientation="top")
        self.enableAutoSIPrefix(False)
        self._channel: np.ndarray | None = None
        self._energy: np.ndarray | None = None
        # Without retained keV coordinates the axis only mirrors the bottom one.
        set_frame_only(self, True)

    def set_spectrum(self, spectrum: Spectrum | None) -> None:
        available = not (
            spectrum is None or spectrum.axis_unit == "keV" or spectrum.energy_kev is None
        )
        if spectrum is None or not available:
            self._channel = None
            self._energy = None
            self.setLabel(text="")
            set_frame_only(self, True)
            self.setToolTip("No secondary energy calibration is available.")
        else:
            set_frame_only(self, False)
            self._channel = spectrum.x
            self._energy = spectrum.energy_kev
            stale = bool(spectrum.metadata.get("calibration_stale", False))
            unit = spectrum.energy_unit
            quantity = "Light output" if unit == "keVee" else "Energy"
            self.setLabel(text=f"{quantity} (stale)" if stale else quantity, units=unit)
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

    compton_calibration_requested = Signal(object)
    """Emits a :class:`CalibrationPointTransfer` with fitted Compton edges in keVee."""

    def __init__(self, controllers: list[MCAController], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        make_tool_window(self)
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
        self._reference_lines: list[pg.InfiniteLine] = []
        self._reference_candidates: dict[str, RadionuclideCandidate] = {}
        self._reference_roi_energy_range: tuple[float, float] | None = None
        self._reference_favourites: list[str] = []
        self._updating_reference_items = False
        self._rebuilding_sources = False
        self._allow_close = False
        self._compton_configs: dict[str, _ComptonDatasetConfig] = {}
        self._compton_result: ComptonEdgeFitResult | None = None
        self._compton_spec: ComptonEdgeFitSpec | None = None
        self._pending_compton_spec: ComptonEdgeFitSpec | None = None
        self._compton_thread: QThread | None = None
        self._compton_worker: ComptonEdgeFitWorker | None = None
        self._updating_compton = False
        self._programmatic_region = False
        self._build_ui()
        self._connect_signals()
        self._refresh_all(show_errors=False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.menu_bar = WorkbenchMenuBar(self)
        root.setMenuBar(self.menu_bar)
        self.file_menu = self.menu_bar.menu("&File")
        self.analysis_menu = self.menu_bar.menu("&Analysis")
        self.reference_menu = self.menu_bar.menu("&References")
        self.view_menu = self.menu_bar.menu("&View")
        self.menu_bar.add_guide("peak-analysis")
        instructions = QLabel(
            "Live MCA data are copied as frozen snapshots. File and derived spectra use the "
            "same fitting pipeline; acquisition is never reconfigured by this workbench.",
            self,
        )
        instructions.setWordWrap(True)
        root.addWidget(instructions)

        counts_row = QHBoxLayout()
        self.load_action = self.menu_bar.command(
            self.file_menu, "Load spectrum files...", shortcut="Ctrl+O",
        )
        self.export_spectrum_action = self.menu_bar.command(
            self.file_menu, "Export spectrum...", enabled=False,
        )
        self.file_menu.addSeparator()
        self.refresh_selected_action = self.menu_bar.command(
            self.analysis_menu, "Refresh selected MCA",
        )
        self.refresh_all_action = self.menu_bar.command(
            self.analysis_menu, "Refresh all MCA spectra",
        )
        self.remove_action = self.menu_bar.command(self.analysis_menu, "Remove selected spectrum")
        self.analysis_menu.addSeparator()
        counts_row.addWidget(QLabel("Counts:", self))
        self.scale_combo = QComboBox(self)
        self.scale_combo.addItem("Linear", "linear")
        self.scale_combo.addItem("Logarithmic", "log")
        counts_row.addWidget(self.scale_combo)
        counts_row.addStretch(1)
        root.addLayout(counts_row)

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
        plot_widget.setBackground(theme.colors().plot_background)
        self.energy_axis = _SpectrumEnergyAxis()
        self.spectrum_plot = plot_widget.addPlot(
            row=0,
            col=0,
            viewBox=ModifierZoomViewBox(),
            axisItems={"top": self.energy_axis},
        )
        self.spectrum_plot.showAxis("top")  # Energy axis: values only when available.
        show_frame_axes(self.spectrum_plot, "right")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.2)
        self.spectrum_plot.setLabel("left", "Counts")
        self.legend = self.spectrum_plot.addLegend()
        self.fit_region = pg.LinearRegionItem(
            values=(0.0, 1.0), movable=True, **theme.region_style()
        )
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
        self.compton_tab = self._build_compton_tab()
        self.side_tabs.addTab(self.compton_tab, "Compton edge")
        self.side_tabs.addTab(self._build_reference_tab(), "Reference lines")
        self.side_tabs.addTab(self._build_operations_tab(), "Operations")
        self.side_tabs.addTab(self._build_results_tab(), "Results")
        self.compton_export_action = self.menu_bar.command(
            self.file_menu, "Export Compton-edge results...", enabled=False,
        )
        self.compton_copy_action = self.menu_bar.command(
            self.file_menu, "Copy Compton-edge table", enabled=False,
        )
        splitter.addWidget(self.side_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([290, 800, 420])
        root.addWidget(splitter, 1)

        status_row = QHBoxLayout()
        # Shown only while a background fit runs: determinate for stages with a
        # known length, a busy indicator otherwise.
        self.workbench_progress = QProgressBar(self)
        self.workbench_progress.setMaximumWidth(230)
        self.workbench_progress.setVisible(False)
        status_row.addWidget(self.workbench_progress)
        self.workbench_status = QLabel("Ready.", self)
        self.workbench_status.setWordWrap(True)
        self.workbench_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        status_row.addWidget(self.workbench_status, 1)
        root.addLayout(status_row)
        self.file_menu.addSeparator()
        self.close_action = self.menu_bar.command(
            self.file_menu, "Close workspace", shortcut="Ctrl+W",
        )
        self._build_view_menu()
        wrap_widget_tooltips(self)

    def _build_view_menu(self) -> None:
        self._scale_action_group = QActionGroup(self)
        self.linear_action = self.menu_bar.command(self.view_menu, "Linear counts")
        self.log_action = self.menu_bar.command(self.view_menu, "Logarithmic counts")
        for index, action in enumerate((self.linear_action, self.log_action)):
            action.setCheckable(True)
            self._scale_action_group.addAction(action)
            action.triggered.connect(lambda _checked, index=index:
                                     self.scale_combo.setCurrentIndex(index))
        self.scale_combo.currentIndexChanged.connect(self._sync_view_menu)
        self.view_menu.addSeparator()
        self._tab_action_group = QActionGroup(self)
        self.tab_actions = []
        for index in range(self.side_tabs.count()):
            action = self.menu_bar.command(self.view_menu, self.side_tabs.tabText(index))
            action.setCheckable(True)
            self._tab_action_group.addAction(action)
            action.triggered.connect(lambda _checked, index=index:
                                     self.side_tabs.setCurrentIndex(index))
            self.tab_actions.append(action)
        self.side_tabs.currentChanged.connect(self._sync_view_menu)
        self.view_menu.addSeparator()
        self.reset_zoom_action = self.menu_bar.command(self.view_menu, "Reset plot zoom")
        self.reset_zoom_action.triggered.connect(self._reset_plot_zoom)
        self._sync_view_menu()

    def _sync_view_menu(self, *_args: object) -> None:
        self.linear_action.setChecked(self.scale_combo.currentIndex() == 0)
        self.log_action.setChecked(self.scale_combo.currentIndex() == 1)
        for index, action in enumerate(self.tab_actions):
            action.setChecked(self.side_tabs.currentIndex() == index)

    def _reset_plot_zoom(self) -> None:
        self.spectrum_plot.autoRange()
        self.residual_plot.autoRange()

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

        self.suggest_action = self.menu_bar.command(self.analysis_menu, "Suggest peak positions")
        self.fit_action = self.menu_bar.command(
            self.analysis_menu, "Fit selected spectrum", enabled=False,
        )
        self.cancel_fit_action = self.menu_bar.command(
            self.analysis_menu, "Cancel fit", enabled=False,
        )
        self.analysis_menu.addSeparator()
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

    def _build_compton_tab(self) -> QWidget:
        tab = QWidget(self)
        # Every control in this tab resolves F1 help to the Compton-edge topic.
        tab.setProperty("nlabHelpTopic", "compton-edge")
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        explanation = QLabel(
            "Forward fit of the Klein–Nishina Compton-electron spectrum folded with the "
            "detector resolution, for organic scintillators without photopeaks. Checked "
            "spectra are fitted jointly with a shared gain, offset and resolution. Edges "
            "are in electron-equivalent keV (keVee).",
            self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        layout.addWidget(QLabel("Datasets (checked spectra are fitted jointly):", self))
        self.compton_dataset_list = QListWidget(self)
        self.compton_dataset_list.setMaximumHeight(110)
        self.compton_dataset_list.setToolTip(
            "Check spectra of the same detector and MCA settings, for example Cs-137 "
            "alone and Cs-137 + Co-60 + Na-22. Selecting a row selects that spectrum."
        )
        layout.addWidget(self.compton_dataset_list)
        self.compton_dataset_heading = QLabel("Selected spectrum: none", self)
        self.compton_dataset_heading.setWordWrap(True)
        layout.addWidget(self.compton_dataset_heading)

        dataset_form = QFormLayout()
        self.compton_nuclide_list = QListWidget(self)
        self.compton_nuclide_list.setMaximumHeight(120)
        for nuclide in COMPTON_PRESET_NUCLIDES:
            try:
                lines = nuclide_compton_lines(nuclide)
            except KeyError:
                continue
            if not lines:
                continue
            energies = ", ".join(f"{line.energy_kev:g}" for line in lines)
            item = QListWidgetItem(f"{nuclide}  ({energies} keV)")
            item.setData(Qt.ItemDataRole.UserRole, nuclide)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setToolTip(
                "Catalogue gamma lines at or above 150 keV with an emission probability of "
                "at least 5 %; Na-22 includes the 511 keV annihilation line."
            )
            self.compton_nuclide_list.addItem(item)
        self.compton_custom_energies = QLineEdit(self)
        self.compton_custom_energies.setPlaceholderText("e.g. 834.8, 1460.8")
        self.compton_custom_energies.setToolTip(
            "Additional gamma energies in keV, separated by commas or spaces."
        )
        self.compton_background_combo = QComboBox(self)
        for label, value in (
            ("None", "none"),
            ("Constant", "constant"),
            ("Linear", "linear"),
            ("Exponential", "exponential"),
        ):
            self.compton_background_combo.addItem(label, value)
        self.compton_background_combo.setCurrentIndex(
            self.compton_background_combo.findData("exponential")
        )
        self.compton_range_label = QLabel("Full spectrum", self)
        self.compton_range_label.setToolTip(
            "Drag the shaded region on the plot while this tab is open to set the fit "
            "range of the selected spectrum."
        )
        dataset_form.addRow("Sources:", self.compton_nuclide_list)
        dataset_form.addRow("Custom lines (keV):", self.compton_custom_energies)
        dataset_form.addRow("Background:", self.compton_background_combo)
        dataset_form.addRow("Fit range:", self.compton_range_label)
        layout.addLayout(dataset_form)
        self.compton_response_label = QLabel(self)
        self.compton_response_label.setWordWrap(True)
        layout.addWidget(self.compton_response_label)

        global_form = QFormLayout()
        self.compton_offset_free = QCheckBox("Fit offset (otherwise fixed at 0)", self)
        self.compton_offset_free.setChecked(True)
        self.compton_offset_free.setToolTip(
            "x = g·E + o. A free offset needs at least two distinct edge energies."
        )
        resolution_grid = QGridLayout()
        self.compton_fit_a = QCheckBox("a", self)
        self.compton_fit_a.setChecked(True)
        self.compton_a = QDoubleSpinBox(self)
        self.compton_a.setRange(0.0, 0.5)
        self.compton_a.setDecimals(4)
        self.compton_a.setSingleStep(0.005)
        self.compton_a.setValue(0.05)
        self.compton_fit_b = QCheckBox("b", self)
        self.compton_fit_b.setChecked(True)
        self.compton_b = QDoubleSpinBox(self)
        self.compton_b.setRange(0.0, 20.0)
        self.compton_b.setDecimals(3)
        self.compton_b.setSingleStep(0.1)
        self.compton_b.setValue(1.0)
        self.compton_fit_c = QCheckBox("c", self)
        self.compton_c = QDoubleSpinBox(self)
        self.compton_c.setRange(0.0, 200.0)
        self.compton_c.setDecimals(2)
        self.compton_c.setValue(0.0)
        resolution_tooltip = (
            "σ_E/E = sqrt(a² + b²/E + c²/E²) with E in keVee (b in keV^½, c in keV). "
            "Checked terms are fitted; unchecked terms stay fixed at the entered value. "
            "The Dietze & Klein FWHM terms α, β, γ are 2.3548 × these σ terms."
        )
        for column, (box, spin) in enumerate(
            (
                (self.compton_fit_a, self.compton_a),
                (self.compton_fit_b, self.compton_b),
                (self.compton_fit_c, self.compton_c),
            )
        ):
            box.setToolTip(resolution_tooltip)
            spin.setToolTip(resolution_tooltip)
            resolution_grid.addWidget(box, 0, 2 * column)
            resolution_grid.addWidget(spin, 0, 2 * column + 1)
            resolution_grid.setColumnStretch(2 * column + 1, 1)
        self.compton_statistic_combo = QComboBox(self)
        self.compton_statistic_combo.addItem("Auto (count-aware)", "auto")
        self.compton_statistic_combo.addItem("Poisson deviance", "poisson")
        self.compton_statistic_combo.addItem("Weighted least squares", "weighted")
        self.compton_statistic_combo.addItem("Unweighted least squares", "unweighted")
        self.compton_amplitude_combo = QComboBox(self)
        self.compton_amplitude_combo.addItem("Free per line (reference)", "free")
        self.compton_amplitude_combo.addItem(
            "Linked by emission yield (approximation)", "linked"
        )
        self.compton_amplitude_combo.setToolTip(
            "Linked amplitudes fix the ratios within a nuclide to catalogue emission "
            "yields. This ignores the energy dependence of the detection efficiency; the "
            "free mode is the reference."
        )
        self.compton_initial_gain = QDoubleSpinBox(self)
        self.compton_initial_gain.setRange(0.0, 1.0e6)
        self.compton_initial_gain.setDecimals(5)
        self.compton_initial_gain.setSpecialValueText("Automatic")
        self.compton_initial_gain.setValue(0.0)
        self.compton_initial_gain.setToolTip(
            "Starting gain in x units per keVee. Automatic scans the gain at the starting "
            "resolution before the fit."
        )
        self.compton_dk_combo = QComboBox(self)
        self.compton_dk_combo.addItem("Off", None)
        for detector in dietze_klein_table.DETECTORS:
            self.compton_dk_combo.addItem(detector.description, detector.number)
        self.compton_dk_combo.setToolTip(
            "Optional independent edge estimate from the Dietze & Klein (1982) Table 3 "
            "ratios for Cs-137 and the Na-22 511 keV line. Indicative only for other "
            "scintillators and detector sizes."
        )
        global_form.addRow("Offset:", self.compton_offset_free)
        layout.addLayout(global_form)
        resolution_label = QLabel(
            "Resolution σ/E = sqrt(a² + b²/E + c²/E²); checked terms are fitted:", self
        )
        resolution_label.setWordWrap(True)
        resolution_label.setToolTip(resolution_tooltip)
        layout.addWidget(resolution_label)
        layout.addLayout(resolution_grid)
        options_form = QFormLayout()
        options_form.addRow("Statistic:", self.compton_statistic_combo)
        options_form.addRow("Amplitudes:", self.compton_amplitude_combo)
        options_form.addRow("Starting gain:", self.compton_initial_gain)
        options_form.addRow("Dietze–Klein size:", self.compton_dk_combo)
        layout.addLayout(options_form)

        self.compton_fit_action = self.menu_bar.command(
            self.analysis_menu, "Fit Compton edges", enabled=False,
        )
        self.compton_cancel_action = self.menu_bar.command(
            self.analysis_menu, "Cancel Compton-edge fit", enabled=False,
        )
        self.compton_response_action = self.menu_bar.command(
            self.analysis_menu, "Load simulated response...",
        )
        self.compton_response_action.setToolTip(
            "Replace the single-scatter shape of one line with a deposited-energy "
            "histogram (CSV: energy_kev, weight), for example from Geant4 or OpenGATE."
        )
        self.compton_clear_responses_action = self.menu_bar.command(
            self.analysis_menu, "Clear simulated responses",
        )
        self.compton_calibration_action = self.menu_bar.command(
            self.analysis_menu, "Open Compton edges in Energy Calibration...", enabled=False,
        )
        self.compton_calibration_action.setToolTip(
            "Open MCA Energy Calibration with the fitted edges as fixed keVee points. "
            "Nothing is applied until you review the fit there and choose Apply calibration."
        )
        self.analysis_menu.addSeparator()

        self.compton_status = QLabel(
            "Check one or more spectra and select their source lines.", self
        )
        self.compton_status.setWordWrap(True)
        self.compton_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.compton_status)
        self.compton_result_table = QTableWidget(0, 4, self)
        self.compton_result_table.setHorizontalHeaderLabels(
            ["Quantity", "Value", "Uncertainty", "Unit / note"]
        )
        self.compton_result_table.verticalHeader().setVisible(False)
        self.compton_result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.compton_result_table.setMinimumHeight(220)
        result_header = self.compton_result_table.horizontalHeader()
        for column in range(3):
            result_header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        result_header.setStretchLastSection(True)
        layout.addWidget(self.compton_result_table)
        self.compton_summary = QLabel("No Compton-edge fit result.", self)
        self.compton_summary.setWordWrap(True)
        self.compton_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.compton_summary)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        self._update_compton_response_label()
        return tab

    def _build_reference_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        explanation = QLabel(
            "Overlay evaluated decay-photon energies on the selected calibrated spectrum. "
            "Emission probability controls marker visibility; it does not predict measured "
            "peak height.",
            self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.reference_search = QLineEdit(self)
        self.reference_search.setPlaceholderText("Filter nuclides, for example Na-22")
        layout.addWidget(self.reference_search)

        self.reference_roi_filter = QCheckBox("Only nuclides with lines in fit ROI", self)
        self.reference_roi_filter.setEnabled(False)
        self.reference_nuclide_list = QListWidget(self)
        self.reference_nuclide_list.setMaximumHeight(150)
        self._reference_summaries = {
            summary.nuclide: summary for summary in BUILTIN_RADIONUCLIDE_CATALOG.summaries()
        }
        self._reference_favourites = self._load_reference_favourites()
        self.reference_favourite_action = self.menu_bar.command(
            self.reference_menu, "Add to favourites", enabled=False,
        )
        self.reference_favourite_action.setToolTip(
            "Pin the highlighted nuclide above the rest of the current list."
        )
        self.restore_reference_favourites_action = self.menu_bar.command(
            self.reference_menu, "Restore common defaults",
        )
        self.restore_reference_favourites_action.setToolTip(
            "Restore the built-in common laboratory source favourites."
        )
        self._populate_reference_nuclides(
            self._ordered_reference_nuclides(tuple(self._reference_summaries))
        )
        layout.addWidget(self.reference_nuclide_list)

        controls = QFormLayout()
        self.reference_probability = QDoubleSpinBox(self)
        self.reference_probability.setRange(0.0, 200.0)
        self.reference_probability.setDecimals(3)
        self.reference_probability.setSingleStep(0.5)
        self.reference_probability.setValue(1.0)
        self.reference_probability.setSuffix(" %")
        controls.addRow("Minimum emission probability:", self.reference_probability)
        controls.addRow("", self.reference_roi_filter)
        layout.addLayout(controls)

        self.reference_filter_status = QLabel(
            "ROI candidate filtering requires a current energy calibration.",
            self,
        )
        self.reference_filter_status.setWordWrap(True)
        layout.addWidget(self.reference_filter_status)

        self.reference_menu.addSeparator()
        self.clear_reference_action = self.menu_bar.command(self.reference_menu, "Clear selection")

        self.reference_table = QTableWidget(0, 4, self)
        self.reference_table.setHorizontalHeaderLabels(
            ["Nuclide", "Energy (keV)", "Emission (%)", "Radiation"]
        )
        self.reference_table.verticalHeader().setVisible(False)
        self.reference_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.reference_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.reference_table, 1)

        self.reference_status = QLabel("Select a radionuclide to display its photon lines.", self)
        self.reference_status.setWordWrap(True)
        self.reference_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.reference_status)
        metadata = BUILTIN_RADIONUCLIDE_CATALOG.metadata
        source = QLabel(
            f'Offline catalogue v{BUILTIN_RADIONUCLIDE_CATALOG.version}: '
            f'{len(BUILTIN_RADIONUCLIDE_CATALOG.nuclides()):,} parent nuclides and '
            f'{int(metadata["recommended_line_count"]):,} evaluated photon lines from the '
            '<a href="https://www.nndc.bnl.gov/ensdf/">NNDC ENSDF</a> snapshot dated '
            f'{metadata["source_data_generated_at"]}; Na-22 annihilation from '
            '<a href="https://www-nds.iaea.org/IRDFF/Recomm-DecayData-IRDFF-II.pdf">'
            "DDEP/IRDFF-II</a>.",
            self,
        )
        source.setOpenExternalLinks(True)
        source.setWordWrap(True)
        layout.addWidget(source)
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
        self.apply_operation_action = self.menu_bar.command(
            self.analysis_menu, "Create derived spectrum", enabled=False,
        )
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
        self.export_fit_action = self.menu_bar.command(
            self.file_menu, "Export fit results...", enabled=False,
        )
        self.report_fit_action = self.menu_bar.command(
            self.file_menu, "Download report...", enabled=False, shortcut="Ctrl+S",
        )
        self.report_fit_action.setToolTip(wrap_tooltip(
            "Save a PDF report and complete fitted spectrum, peak, and parameter data."
        ))
        return tab

    def _connect_signals(self) -> None:
        self.load_action.triggered.connect(self._load_files)
        self.refresh_selected_action.triggered.connect(self._refresh_selected)
        self.refresh_all_action.triggered.connect(self._refresh_all)
        self.remove_action.triggered.connect(self._remove_selected)
        self.export_spectrum_action.triggered.connect(self._export_selected_spectrum)
        self.scale_combo.currentIndexChanged.connect(self._render)
        self.source_list.currentItemChanged.connect(self._selection_changed)
        self.source_list.itemChanged.connect(self._visibility_changed)
        self.peak_count.valueChanged.connect(self._suggest_peaks)
        self.suggest_action.triggered.connect(self._suggest_peaks)
        self.fit_action.triggered.connect(self._start_fit)
        self.cancel_fit_action.triggered.connect(self._cancel_fit)
        self.reference_search.textChanged.connect(self._filter_reference_nuclides)
        self.reference_nuclide_list.itemChanged.connect(self._reference_selection_changed)
        self.reference_nuclide_list.currentItemChanged.connect(
            self._reference_current_item_changed
        )
        self.reference_favourite_action.triggered.connect(self._toggle_reference_favourite)
        self.restore_reference_favourites_action.triggered.connect(
            self._restore_reference_favourites
        )
        self.reference_probability.valueChanged.connect(self._reference_probability_changed)
        self.reference_roi_filter.toggled.connect(self._reference_roi_filter_changed)
        self.fit_region.sigRegionChangeFinished.connect(self._reference_roi_changed)
        self.clear_reference_action.triggered.connect(self._clear_reference_selection)
        self.operation_combo.currentIndexChanged.connect(self._update_operation_controls)
        self.apply_operation_action.triggered.connect(self._apply_operation)
        self.export_fit_action.triggered.connect(self._export_fit_result)
        self.report_fit_action.triggered.connect(self._download_fit_report)
        self.close_action.triggered.connect(self.reject)
        self.compton_dataset_list.itemChanged.connect(self._compton_dataset_toggled)
        self.compton_dataset_list.currentItemChanged.connect(self._compton_dataset_selected)
        self.compton_nuclide_list.itemChanged.connect(self._compton_editor_changed)
        self.compton_custom_energies.editingFinished.connect(self._compton_editor_changed)
        self.compton_background_combo.currentIndexChanged.connect(self._compton_editor_changed)
        self.fit_region.sigRegionChangeFinished.connect(self._compton_region_changed)
        self.side_tabs.currentChanged.connect(self._side_tab_changed)
        self.compton_fit_action.triggered.connect(self._start_compton_fit)
        self.compton_cancel_action.triggered.connect(self._cancel_compton_fit)
        self.compton_response_action.triggered.connect(self._load_compton_response)
        self.compton_clear_responses_action.triggered.connect(self._clear_compton_responses)
        self.compton_calibration_action.triggered.connect(self._send_compton_calibration_points)
        self.compton_export_action.triggered.connect(self._export_compton_result)
        self.compton_copy_action.triggered.connect(self._copy_compton_table)

    def _load_reference_favourites(self) -> list[str]:
        settings = QSettings()
        configured = settings.contains(_REFERENCE_FAVOURITES_KEY)
        if configured:
            stored = settings.value(_REFERENCE_FAVOURITES_KEY, [])
            if isinstance(stored, str):
                raw_values = [stored]
            elif isinstance(stored, (list, tuple)):
                raw_values = list(stored)
            else:
                raw_values = []
        else:
            raw_values = list(_DEFAULT_REFERENCE_FAVOURITES)
        favourites: list[str] = []
        for raw_value in raw_values:
            try:
                nuclide = canonical_nuclide(str(raw_value))
            except ValueError:
                continue
            if nuclide in self._reference_summaries and nuclide not in favourites:
                favourites.append(nuclide)
        return favourites

    def _save_reference_favourites(self) -> None:
        QSettings().setValue(_REFERENCE_FAVOURITES_KEY, self._reference_favourites)

    def _ordered_reference_nuclides(self, nuclides: tuple[str, ...]) -> tuple[str, ...]:
        available = set(nuclides)
        favourites = [
            nuclide for nuclide in self._reference_favourites if nuclide in available
        ]
        favourite_set = set(favourites)
        return tuple(favourites + [nuclide for nuclide in nuclides if nuclide not in favourite_set])

    def _populate_reference_nuclides(self, nuclides: tuple[str, ...]) -> None:
        checked = set(self._selected_reference_nuclides())
        current = self.reference_nuclide_list.currentItem()
        current_nuclide = (
            str(current.data(Qt.ItemDataRole.UserRole)) if current is not None else None
        )
        restored_current: QListWidgetItem | None = None
        favourites = set(self._reference_favourites)
        self._updating_reference_items = True
        try:
            self.reference_nuclide_list.clear()
            for nuclide in nuclides:
                summary = self._reference_summaries[nuclide]
                candidate = self._reference_candidates.get(nuclide)
                if candidate is not None:
                    count = candidate.matching_line_count
                    label = (
                        f"{nuclide}  ({count} ROI line{'s' if count != 1 else ''}; "
                        f"strongest {candidate.strongest_probability_percent:g}%)"
                    )
                elif self.reference_roi_filter.isChecked() and nuclide in checked:
                    label = f"{nuclide}  (selected; no ROI match)"
                else:
                    label = f"{nuclide}  ({summary.line_count} line(s))"
                if nuclide in favourites:
                    label = f"★ {label}"
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, nuclide)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if nuclide in checked else Qt.CheckState.Unchecked
                )
                state = (
                    f"\nParent-state energy: {summary.state_energy_kev:g} keV"
                    if summary.state_energy_kev > 0
                    else ""
                )
                favourite = "Favourite reference\n" if nuclide in favourites else ""
                item.setToolTip(
                    f"{favourite}{BUILTIN_RADIONUCLIDE_CATALOG.reference}{state}"
                )
                self.reference_nuclide_list.addItem(item)
                if nuclide == current_nuclide:
                    restored_current = item
            if restored_current is not None:
                self.reference_nuclide_list.setCurrentItem(restored_current)
        finally:
            self._updating_reference_items = False
        self._update_reference_favourite_action()

    def _reference_current_item_changed(self, *_args: object) -> None:
        self._update_reference_favourite_action()

    def _update_reference_favourite_action(self) -> None:
        item = self.reference_nuclide_list.currentItem()
        if item is None:
            self.reference_favourite_action.setText("Add to favourites")
            self.reference_favourite_action.setEnabled(False)
            return
        nuclide = str(item.data(Qt.ItemDataRole.UserRole))
        is_favourite = nuclide in self._reference_favourites
        self.reference_favourite_action.setText(
            "Remove from favourites" if is_favourite else "Add to favourites"
        )
        self.reference_favourite_action.setEnabled(True)

    def _toggle_reference_favourite(self) -> None:
        item = self.reference_nuclide_list.currentItem()
        if item is None:
            return
        nuclide = str(item.data(Qt.ItemDataRole.UserRole))
        if nuclide in self._reference_favourites:
            self._reference_favourites.remove(nuclide)
        else:
            self._reference_favourites.append(nuclide)
        self._save_reference_favourites()
        self._update_reference_candidate_filter(force_rebuild=True)

    def _restore_reference_favourites(self) -> None:
        self._reference_favourites = [
            nuclide
            for nuclide in _DEFAULT_REFERENCE_FAVOURITES
            if nuclide in self._reference_summaries
        ]
        self._save_reference_favourites()
        self._update_reference_candidate_filter(force_rebuild=True)

    def _filter_reference_nuclides(self, _text: str = "") -> None:
        self._apply_reference_nuclide_visibility()

    def _apply_reference_nuclide_visibility(self) -> None:
        normalized = self.reference_search.text().strip().lower().replace("-", "")
        roi_filter = self.reference_roi_filter.isChecked()
        for row in range(self.reference_nuclide_list.count()):
            item = self.reference_nuclide_list.item(row)
            nuclide = str(item.data(Qt.ItemDataRole.UserRole))
            text_matches = normalized in nuclide.lower().replace("-", "")
            roi_matches = (
                not roi_filter
                or nuclide in self._reference_candidates
                or item.checkState() == Qt.CheckState.Checked
            )
            item.setHidden(not text_matches or not roi_matches)

    def _update_reference_candidate_filter(self, *, force_rebuild: bool = False) -> None:
        spectrum = self._current_spectrum()
        low, high = sorted(map(float, self.fit_region.getRegion()))
        energy_range = (
            energy_range_for_coordinates(spectrum, low, high) if spectrum is not None else None
        )
        self.reference_roi_filter.setEnabled(energy_range is not None)
        unavailable_tooltip = (
            "The fit ROI must overlap the selected spectrum at a non-negative energy."
            if spectrum is not None and has_usable_energy_axis(spectrum)
            else "A selected spectrum with a current energy calibration is required."
        )
        self.reference_roi_filter.setToolTip(
            "Filter and rank parent nuclides by evaluated lines inside the fit ROI."
            if energy_range is not None
            else unavailable_tooltip
        )
        if energy_range is None:
            if self.reference_roi_filter.isChecked():
                self.reference_roi_filter.blockSignals(True)
                self.reference_roi_filter.setChecked(False)
                self.reference_roi_filter.blockSignals(False)
            rebuild = force_rebuild or bool(self._reference_candidates) or (
                self.reference_nuclide_list.count() != len(self._reference_summaries)
            )
            self._reference_candidates = {}
            self._reference_roi_energy_range = None
            if rebuild:
                self._populate_reference_nuclides(
                    self._ordered_reference_nuclides(tuple(self._reference_summaries))
                )
            self._apply_reference_nuclide_visibility()
            if spectrum is not None and bool(spectrum.metadata.get("calibration_stale", False)):
                reason = "The selected spectrum's calibration is stale."
            elif spectrum is not None and has_usable_energy_axis(spectrum):
                reason = (
                    "The fit ROI does not overlap the selected spectrum at a non-negative "
                    "energy."
                )
            else:
                reason = "A current energy calibration is required."
            self.reference_filter_status.setText(f"ROI candidate filter unavailable: {reason}")
            return

        self._reference_roi_energy_range = energy_range
        if not self.reference_roi_filter.isChecked():
            rebuild = force_rebuild or bool(self._reference_candidates) or (
                self.reference_nuclide_list.count() != len(self._reference_summaries)
            )
            self._reference_candidates = {}
            if rebuild:
                self._populate_reference_nuclides(
                    self._ordered_reference_nuclides(tuple(self._reference_summaries))
                )
            self._apply_reference_nuclide_visibility()
            self.reference_filter_status.setText(
                f"Fit ROI spans {energy_range[0]:g}–{energy_range[1]:g} keV; enable the "
                "candidate filter to search it."
            )
            return

        candidates = BUILTIN_RADIONUCLIDE_CATALOG.candidates(
            energy_range[0],
            energy_range[1],
            minimum_probability_percent=self.reference_probability.value(),
        )
        self._reference_candidates = {candidate.nuclide: candidate for candidate in candidates}
        selected = set(self._selected_reference_nuclides())
        matching = self._ordered_reference_nuclides(
            tuple(candidate.nuclide for candidate in candidates)
        )
        selected_nonmatches = self._ordered_reference_nuclides(
            tuple(
                nuclide
                for nuclide in self._reference_summaries
                if nuclide in selected and nuclide not in self._reference_candidates
            )
        )
        self._populate_reference_nuclides(matching + selected_nonmatches)
        self._apply_reference_nuclide_visibility()
        self.reference_filter_status.setText(
            f"Found {len(candidates):,} candidate parent nuclide(s) with evaluated lines at or "
            f"above {self.reference_probability.value():g}% in "
            f"{energy_range[0]:g}–{energy_range[1]:g} keV. A match is not an identification."
        )

    def _reference_probability_changed(self, *_args: object) -> None:
        self._update_reference_candidate_filter()
        self._render()

    def _reference_roi_filter_changed(self, _checked: bool) -> None:
        self._update_reference_candidate_filter()

    def _reference_roi_changed(self, *_args: object) -> None:
        self._update_reference_candidate_filter()

    def _reference_selection_changed(self, item: QListWidgetItem) -> None:
        if not self._updating_reference_items:
            if (
                self.reference_roi_filter.isChecked()
                and item.checkState() != Qt.CheckState.Checked
            ):
                self._apply_reference_nuclide_visibility()
            self._render()

    def _clear_reference_selection(self) -> None:
        self._updating_reference_items = True
        try:
            for row in range(self.reference_nuclide_list.count()):
                self.reference_nuclide_list.item(row).setCheckState(Qt.CheckState.Unchecked)
        finally:
            self._updating_reference_items = False
        self._render()

    def _selected_reference_nuclides(self) -> tuple[str, ...]:
        return tuple(
            str(item.data(Qt.ItemDataRole.UserRole))
            for row in range(self.reference_nuclide_list.count())
            if (item := self.reference_nuclide_list.item(row)).checkState()
            == Qt.CheckState.Checked
        )

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

    def add_spectra(self, spectra: list[Spectrum]) -> None:
        """Add externally reconstructed spectra without coupling them to file loaders."""
        for index, spectrum in enumerate(spectra):
            self._add_spectrum(spectrum, select=index == len(spectra) - 1)

    def remove_spectra(self, spectrum_ids: set[str]) -> None:
        """Remove a related set of externally managed spectra in one UI rebuild."""
        if not spectrum_ids:
            return
        for spectrum_id in spectrum_ids:
            self._spectra.pop(spectrum_id, None)
            self._visible.pop(spectrum_id, None)
        self._order = [
            spectrum_id for spectrum_id in self._order if spectrum_id not in spectrum_ids
        ]
        for channel, live_id in tuple(self._live_ids.items()):
            if live_id in spectrum_ids:
                del self._live_ids[channel]
        if self._fit_spectrum_id in spectrum_ids:
            self._fit_result = None
            self._fit_spectrum_id = None
            self.export_fit_action.setEnabled(False)
        self._forget_compton_spectra(spectrum_ids)
        self._rebuild_source_list()
        self._render()

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
        self._rebuild_compton_datasets()

    def _selection_changed(self, *_args: object) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            self.fit_action.setEnabled(False)
            self.compton_fit_action.setEnabled(False)
            self.export_spectrum_action.setEnabled(False)
            self._update_reference_candidate_filter()
            self._load_compton_editor()
            self._render()
            return
        self.fit_action.setEnabled(self._fit_thread is None)
        self.compton_fit_action.setEnabled(self._compton_thread is None)
        self.export_spectrum_action.setEnabled(True)
        self._programmatic_region = True
        try:
            self.fit_region.setRegion((float(spectrum.x[0]), float(spectrum.x[-1])))
            compton_range = self._compton_range_for(spectrum.spectrum_id)
            if compton_range is not None and self._compton_tab_active():
                self.fit_region.setRegion(compton_range)
        finally:
            self._programmatic_region = False
        self._fit_result = None
        self._fit_spectrum_id = None
        self.export_fit_action.setEnabled(False)
        self.report_fit_action.setEnabled(False)
        self._suggest_peaks()
        self._update_other_spectra()
        self._update_source_status()
        self._update_reference_candidate_filter()
        self._load_compton_editor()
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
        self._reference_lines = []
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
        if not self._compton_tab_active():
            for line in self._peak_lines:
                self.spectrum_plot.addItem(line)
        self._render_reference_lines(selected)
        logarithmic = self.scale_combo.currentData() == "log"
        self.spectrum_plot.setLogMode(y=logarithmic)
        axis_unit = selected.energy_unit if selected is not None else "keV"
        axis_quantity = "Light output" if axis_unit == "keVee" else "Energy"
        for plot in (self.spectrum_plot, self.residual_plot):
            plot.setLabel(
                "bottom",
                axis_quantity if selected_unit == "keV" else "MCA channel",
                units=axis_unit if selected_unit == "keV" else None,
            )
        self._refresh_energy_axis(selected)
        self.residual_curve.setData([], [])
        compton_shown = self._render_compton_overlay(selected)
        if (
            self._fit_result is not None
            and self._fit_spectrum_id == self._current_id()
            and not compton_shown
        ):
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
        # A running Compton-edge fit owns the status line for its progress text.
        if selected is not None and self._compton_thread is None:
            self.workbench_status.setText(f"Displayed {plotted} spectrum/spectra{suffix}.")

    def _render_reference_lines(self, spectrum: Spectrum | None) -> None:
        nuclides = self._selected_reference_nuclides()
        minimum = self.reference_probability.value()
        selected_lines: list[tuple[str, RadionuclideLine, str]] = []
        for nuclide_index, nuclide in enumerate(nuclides):
            color = _REFERENCE_COLORS[nuclide_index % len(_REFERENCE_COLORS)]
            selected_lines.extend(
                (nuclide, line, color)
                for line in BUILTIN_RADIONUCLIDE_CATALOG.lines(
                    nuclide,
                    minimum_probability_percent=minimum,
                )
            )

        self.reference_table.setRowCount(0)
        if not nuclides:
            self.reference_status.setText(
                "Select a radionuclide to display its evaluated photon lines."
            )
            return
        if spectrum is None:
            self.reference_status.setText("Select a spectrum before displaying reference lines.")
            return
        if bool(spectrum.metadata.get("calibration_stale", False)):
            self.reference_status.setText(
                "The selected spectrum's energy calibration is stale; reference lines are hidden."
            )
            return
        if not has_usable_energy_axis(spectrum):
            self.reference_status.setText(
                "Apply an energy calibration, or load a spectrum with a keV axis, to display "
                "reference lines."
            )
            return

        plotted: list[tuple[str, RadionuclideLine, str, float]] = []
        for nuclide, reference, color in selected_lines:
            coordinate = coordinate_for_energy(spectrum, reference.energy_kev)
            if coordinate is None:
                continue
            marker = pg.InfiniteLine(
                pos=coordinate,
                angle=90,
                movable=False,
                pen=pg.mkPen(color, width=1.25, style=Qt.PenStyle.DashLine),
                hoverPen=pg.mkPen(color, width=2.25),
                label=f"{nuclide}  {reference.energy_kev:g} keV",
                labelOpts={
                    "position": 0.96 - 0.08 * (len(plotted) % 4),
                    "color": color,
                    "fill": (255, 255, 255, 190),
                    "movable": False,
                },
            )
            marker.setZValue(_REFERENCE_LINE_Z)
            kind = "annihilation" if reference.kind == "annihilation" else "gamma"
            note = f"\n{reference.note}" if reference.note else ""
            marker.setToolTip(
                f"{nuclide}: {reference.energy_kev:g} keV\n"
                f"Emission probability: {reference.emission_probability_percent:g}%\n"
                f"Radiation: {kind}\n"
                f"Decay: {reference.decay_mode or 'not specified'}\n"
                f"Source dataset: {reference.source_dataset or 'catalogue supplement'}{note}"
            )
            self.spectrum_plot.addItem(marker)
            self._reference_lines.append(marker)
            plotted.append((nuclide, reference, color, coordinate))

        self.reference_table.setRowCount(len(plotted))
        for row, (nuclide, reference, color, _coordinate) in enumerate(plotted):
            radiation = "Annihilation" if reference.kind == "annihilation" else "Gamma"
            values = (
                nuclide,
                f"{reference.energy_kev:.8g}",
                f"{reference.emission_probability_percent:.8g}",
                radiation,
            )
            source_document = BUILTIN_RADIONUCLIDE_CATALOG.line_document(reference)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setForeground(pg.mkBrush(theme.legible_text(color)))
                item.setToolTip(
                    f"{source_document['reference']}\n"
                    f"{reference.source_dataset or 'catalogue supplement'}\n"
                    f"{source_document['reference_url']}"
                )
                self.reference_table.setItem(row, column, item)

        outside = len(selected_lines) - len(plotted)
        if plotted:
            suffix = f"; {outside} outside this spectrum's energy range" if outside else ""
            self.reference_status.setText(
                f"Displayed {len(plotted)} evaluated line(s) for {len(nuclides)} "
                f"nuclide(s){suffix}. Marker height is not a predicted peak height."
            )
        else:
            self.reference_status.setText(
                f"No selected lines at or above {minimum:g}% fall inside this spectrum's "
                "calibrated energy range."
            )

    def _reference_selection_document(self) -> dict[str, object] | None:
        nuclides = self._selected_reference_nuclides()
        if not nuclides:
            return None
        minimum = self.reference_probability.value()
        document = BUILTIN_RADIONUCLIDE_CATALOG.selection_document(
            nuclides,
            minimum_probability_percent=minimum,
        )
        spectrum = self._current_spectrum()
        displayed = []
        if spectrum is not None:
            displayed = [
                BUILTIN_RADIONUCLIDE_CATALOG.line_document(line)
                for nuclide in nuclides
                for line in BUILTIN_RADIONUCLIDE_CATALOG.lines(
                    nuclide,
                    minimum_probability_percent=minimum,
                )
                if coordinate_for_energy(spectrum, line.energy_kev) is not None
            ]
        document["lines"] = displayed
        document["spectrum_id"] = spectrum.spectrum_id if spectrum is not None else None
        document["candidate_filter"] = {
            "enabled": self.reference_roi_filter.isChecked(),
            "fit_roi_energy_kev": (
                list(self._reference_roi_energy_range)
                if self.reference_roi_filter.isChecked()
                and self._reference_roi_energy_range is not None
                else None
            ),
        }
        return document

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
                "live_time_s": snapshot.live_time_s,
                "dead_time_s": (
                    max(snapshot.elapsed_s - snapshot.live_time_s, 0.0)
                    if snapshot.live_time_s is not None
                    else None
                ),
                "live_non_atomic": snapshot.live,
                "settings_fingerprint": snapshot.fingerprint,
                "binning": binning,
                "energy_calibration": calibration.to_dict() if calibration is not None else None,
                "calibration_stale": controller.energy_calibration_is_stale(),
                "n42_instrument": snapshot.n42_instrument or {},
                "n42_detector": snapshot.n42_detector or {},
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
        folder = str(QSettings().value(settings_keys.ANALYSIS_SPECTRUM_FOLDER, "measurements"))
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load MCA Spectra",
            folder,
            "Spectrum files (*.csv *.CSV *.spe *.SPE *.wdm *.WDM *.txt *.txt3 *.tsv "
            "*.root *.ROOT *.n42 *.N42 *.xml *.XML);;All files (*)",
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
        QSettings().setValue(settings_keys.ANALYSIS_SPECTRUM_FOLDER, str(Path(paths[0]).parent))
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
        self._forget_compton_spectra({spectrum_id})
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
        self.fit_action.setEnabled(False)
        self.cancel_fit_action.setEnabled(True)
        self.report_fit_action.setEnabled(False)
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
        self.export_fit_action.setEnabled(True)
        self.report_fit_action.setEnabled(True)
        self._render()

    def _fit_failed(self, message: str) -> None:
        self.fit_status.setText(f"Fit failed: {message}")
        self.workbench_status.setText(f"Peak fit failed: {message}")
        QMessageBox.critical(self, "Peak Fit Failed", message)

    def _fit_thread_finished(self) -> None:
        self._fit_thread = None
        self._fit_worker = None
        self.fit_action.setEnabled(self._current_spectrum() is not None)
        self.cancel_fit_action.setEnabled(False)

    def _cancel_fit(self) -> None:
        if self._fit_worker is not None:
            self._fit_worker.stop()
            self.cancel_fit_action.setEnabled(False)
            self.fit_status.setText("Cancelling fit...")

    def _populate_results(self, result: PeakFitResult) -> None:
        fitted = self._spectra.get(self._fit_spectrum_id or "")
        energy_header = self.peak_table.horizontalHeaderItem(2)
        if energy_header is not None:
            unit = fitted.energy_unit if fitted is not None else "keV"
            energy_header.setText(f"E\n({unit})")
            energy_header.setToolTip(
                "Calibrated light output in keVee"
                if unit == "keVee"
                else "Calibrated peak energy in keV"
            )
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
        self.apply_operation_action.setEnabled(
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
        folder = str(QSettings().value(settings_keys.ANALYSIS_SPECTRUM_FOLDER, "measurements"))
        path_text, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Spectrum",
            str(Path(folder) / f"{_safe_stem(spectrum.label)}.csv"),
            "CSV files (*.csv);;ANSI/IEEE N42.42 (*.n42 *.xml)",
        )
        if not path_text:
            return
        try:
            path = Path(path_text)
            if selected_filter.startswith("ANSI") and path.suffix.lower() not in {".n42", ".xml"}:
                path = path.with_suffix(".n42")
            elif not path.suffix:
                path = path.with_suffix(".csv")
            if path.suffix.lower() in {".n42", ".xml"}:
                export_n42_spectrum(path, spectrum)
            else:
                export_spectrum_csv(path, spectrum)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Spectrum Export Failed", str(exc))
            return
        QSettings().setValue(settings_keys.ANALYSIS_SPECTRUM_FOLDER, str(Path(path_text).parent))
        self.workbench_status.setText(f"Spectrum exported to {path}")

    def _export_fit_result(self) -> None:
        result = self._fit_result
        spectrum = self._spectra.get(self._fit_spectrum_id or "")
        if result is None or spectrum is None:
            return
        folder = str(QSettings().value(settings_keys.ANALYSIS_SPECTRUM_FOLDER, "measurements"))
        path_text, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Peak Fit Results",
            str(Path(folder) / f"{_safe_stem(spectrum.label)}_fit.json"),
            "JSON (*.json);;CSV (*.csv);;ANSI/IEEE N42.42 (*.n42 *.xml)",
        )
        if not path_text:
            return
        path = Path(path_text)
        try:
            if selected_filter.startswith("ANSI") and path.suffix.lower() not in {".n42", ".xml"}:
                path = path.with_suffix(".n42")
            elif selected_filter.startswith("CSV") and path.suffix.lower() != ".csv":
                path = path.with_suffix(".csv")
            elif not path.suffix:
                path = path.with_suffix(".json")
            if self._last_fit_spec is None:
                raise ValueError("the fit model description is unavailable")
            if path.suffix.lower() in {".n42", ".xml"}:
                export_n42_peak_fit(
                    path,
                    spectrum=spectrum,
                    spec=self._last_fit_spec,
                    result=result,
                )
            else:
                export_peak_fit_result(
                    path,
                    spectrum=spectrum,
                    spec=self._last_fit_spec,
                    result=result,
                    reference_overlay=self._reference_selection_document(),
                )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Fit Export Failed", str(exc))
            return
        self.workbench_status.setText(f"Fit results exported to {path}")

    def _download_fit_report(self) -> None:
        # Deferred: matplotlib + reportlab cost ~0.45 s at application start.
        from nlab.analysis.mca_report import PeakFitReportRequest, export_peak_fit_report

        result = self._fit_result
        spectrum = self._spectra.get(self._fit_spectrum_id or "")
        spec = self._last_fit_spec
        if result is None or spectrum is None or spec is None:
            QMessageBox.information(
                self,
                "Peak-Fitting Report",
                "Complete a peak fit before creating a report.",
            )
            return
        default_folder = str(
            QSettings().value(settings_keys.ANALYSIS_PEAK_FIT_REPORT_FOLDER, "measurements")
        )
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Peak-Fitting Report Destination",
            default_folder,
        )
        if not selected:
            return
        request = PeakFitReportRequest(
            spectrum=spectrum,
            spec=spec,
            result=result,
            reference_overlay=self._reference_selection_document(),
            y_scale=cast(Literal["linear", "log"], str(self.scale_combo.currentData())),
        )
        try:
            bundle = export_peak_fit_report(request, Path(selected))
        except (OSError, ValueError) as exc:
            log.exception("Could not create MCA peak-fitting report")
            QMessageBox.critical(self, "Peak-Fitting Report Failed", str(exc))
            return
        QSettings().setValue(settings_keys.ANALYSIS_PEAK_FIT_REPORT_FOLDER, selected)
        self.workbench_status.setText(f"Peak-fitting report saved to {bundle.directory}.")
        QMessageBox.information(
            self,
            "Peak-Fitting Report Saved",
            f"Report and data files were saved to:\n{bundle.directory}",
        )

    # ------------------------------------------------------------------
    # Compton-edge (Klein–Nishina) fits for organic scintillators

    def add_compton_datasets(self, spectra: list[Spectrum]) -> None:
        """Add spectra, include them in the Compton-edge fit and show that tab."""
        for spectrum in spectra:
            self._add_spectrum(spectrum, select=False)
            self._compton_config(spectrum.spectrum_id).include = True
        self.side_tabs.setCurrentWidget(self.compton_tab)
        if spectra:
            self._rebuild_source_list(select_id=spectra[-1].spectrum_id)
            self._selection_changed()
        self.compton_status.setText(
            f"Added {len(spectra)} spectrum/spectra to the Compton-edge fit. Select their "
            "source lines and fit ranges, then choose Analysis → Fit Compton edges."
        )

    def _compton_tab_active(self) -> bool:
        return self.side_tabs.currentWidget() is self.compton_tab

    def _compton_config(self, spectrum_id: str) -> _ComptonDatasetConfig:
        return self._compton_configs.setdefault(spectrum_id, _ComptonDatasetConfig())

    def _compton_range_for(self, spectrum_id: str) -> tuple[float, float] | None:
        config = self._compton_configs.get(spectrum_id)
        return config.fit_range if config is not None else None

    def _compton_dataset_text(
        self, spectrum: Spectrum, config: _ComptonDatasetConfig,
    ) -> str:
        sources = list(config.nuclides)
        if config.custom_energies.strip():
            sources.append(f"custom {config.custom_energies.strip()} keV")
        lines = ", ".join(sources) if sources else "no lines"
        if config.fit_range is None:
            window = "full range"
        else:
            window = f"{config.fit_range[0]:.6g}–{config.fit_range[1]:.6g}"
        return f"{spectrum.label}  [{lines}; {config.background}; {window}]"

    def _rebuild_compton_datasets(self) -> None:
        current = self._current_id()
        self._updating_compton = True
        try:
            self.compton_dataset_list.clear()
            for spectrum_id in self._order:
                spectrum = self._spectra.get(spectrum_id)
                if spectrum is None:
                    continue
                config = self._compton_configs.get(spectrum_id, _ComptonDatasetConfig())
                text = self._compton_dataset_text(spectrum, config)
                item = QListWidgetItem(text)
                item.setToolTip(text)
                item.setData(Qt.ItemDataRole.UserRole, spectrum_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if config.include else Qt.CheckState.Unchecked
                )
                self.compton_dataset_list.addItem(item)
                if spectrum_id == current:
                    self.compton_dataset_list.setCurrentItem(item)
        finally:
            self._updating_compton = False

    def _compton_dataset_toggled(self, item: QListWidgetItem) -> None:
        if self._updating_compton:
            return
        spectrum_id = str(item.data(Qt.ItemDataRole.UserRole))
        self._compton_config(spectrum_id).include = item.checkState() == Qt.CheckState.Checked

    def _compton_dataset_selected(self, current: QListWidgetItem | None, *_args: object) -> None:
        if self._updating_compton or current is None:
            return
        spectrum_id = str(current.data(Qt.ItemDataRole.UserRole))
        if spectrum_id == self._current_id():
            return
        for row in range(self.source_list.count()):
            source = self.source_list.item(row)
            if str(source.data(Qt.ItemDataRole.UserRole)) == spectrum_id:
                self.source_list.setCurrentItem(source)
                return

    def _load_compton_editor(self) -> None:
        spectrum = self._current_spectrum()
        self._updating_compton = True
        try:
            if spectrum is None:
                self.compton_dataset_heading.setText("Selected spectrum: none")
                config = _ComptonDatasetConfig()
            else:
                config = self._compton_configs.get(spectrum.spectrum_id, _ComptonDatasetConfig())
                self.compton_dataset_heading.setText(
                    f"Selected spectrum: {spectrum.label} (x axis: {spectrum.axis_unit})"
                )
            for row in range(self.compton_nuclide_list.count()):
                item = self.compton_nuclide_list.item(row)
                nuclide = str(item.data(Qt.ItemDataRole.UserRole))
                item.setCheckState(
                    Qt.CheckState.Checked if nuclide in config.nuclides else Qt.CheckState.Unchecked
                )
            self.compton_custom_energies.setText(config.custom_energies)
            self.compton_background_combo.setCurrentIndex(
                self.compton_background_combo.findData(config.background)
            )
            self._update_compton_range_label(config)
            for row in range(self.compton_dataset_list.count()):
                item = self.compton_dataset_list.item(row)
                if spectrum is not None and item.data(Qt.ItemDataRole.UserRole) == (
                    spectrum.spectrum_id
                ):
                    self.compton_dataset_list.setCurrentItem(item)
                    break
        finally:
            self._updating_compton = False
        self._update_compton_response_label()

    def _compton_editor_changed(self, *_args: object) -> None:
        if self._updating_compton:
            return
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        config = self._compton_config(spectrum.spectrum_id)
        text = self.compton_custom_energies.text().strip()
        try:
            parse_custom_energies(text)
        except ValueError as exc:
            self.compton_status.setText(f"Invalid custom line: {exc}")
            return
        config.nuclides = [
            str(item.data(Qt.ItemDataRole.UserRole))
            for row in range(self.compton_nuclide_list.count())
            if (item := self.compton_nuclide_list.item(row)).checkState()
            == Qt.CheckState.Checked
        ]
        config.custom_energies = text
        config.background = cast(
            ComptonBackground, str(self.compton_background_combo.currentData())
        )
        if config.nuclides or config.custom_energies:
            config.include = True
        self._rebuild_compton_datasets()
        self._update_compton_response_label()

    def _compton_region_changed(self, *_args: object) -> None:
        if self._programmatic_region or not self._compton_tab_active():
            return
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        config = self._compton_config(spectrum.spectrum_id)
        low, high = sorted(map(float, self.fit_region.getRegion()))
        config.fit_range = (low, high)
        self._update_compton_range_label(config)
        self._rebuild_compton_datasets()

    def _side_tab_changed(self, _index: int) -> None:
        spectrum = self._current_spectrum()
        if spectrum is not None and self._compton_tab_active():
            compton_range = self._compton_range_for(spectrum.spectrum_id)
            if compton_range is not None:
                self._programmatic_region = True
                try:
                    self.fit_region.setRegion(compton_range)
                finally:
                    self._programmatic_region = False
        self._render()

    def _update_compton_range_label(self, config: _ComptonDatasetConfig) -> None:
        if config.fit_range is None:
            self.compton_range_label.setText("Full spectrum (drag the shaded region to limit)")
        else:
            low, high = config.fit_range
            self.compton_range_label.setText(f"{low:.6g} – {high:.6g}")

    def _update_compton_response_label(self) -> None:
        spectrum = self._current_spectrum()
        config = (
            self._compton_configs.get(spectrum.spectrum_id) if spectrum is not None else None
        )
        if config is None or not config.responses:
            self.compton_response_label.setText(
                "Response: Klein–Nishina single scatter for every line. Use Analysis → "
                "Load simulated response... for detectors where multiple scattering matters."
            )
            return
        described = "; ".join(
            f"{energy:g} keV: {response.label}"
            for energy, response in sorted(config.responses.items())
        )
        self.compton_response_label.setText(f"Simulated responses: {described}.")

    def _compton_lines(
        self, config: _ComptonDatasetConfig, *, with_responses: bool = True,
    ) -> tuple[ComptonLine, ...]:
        lines: list[ComptonLine] = []
        for nuclide in config.nuclides:
            lines.extend(nuclide_compton_lines(nuclide))
        for energy in parse_custom_energies(config.custom_energies):
            lines.append(ComptonLine(energy, label=f"{energy:g} keV (custom)"))
        unique: dict[float, ComptonLine] = {}
        for line in lines:
            key = round(line.energy_kev, 3)
            if key in unique:
                continue
            response = config.responses.get(key) if with_responses else None
            unique[key] = line.with_response(response) if response is not None else line
        return tuple(unique.values())

    def _compton_fit_spec(self) -> ComptonEdgeFitSpec:
        included = [
            spectrum_id
            for spectrum_id in self._order
            if spectrum_id in self._spectra
            and (config := self._compton_configs.get(spectrum_id)) is not None
            and config.include
        ]
        if not included:
            current = self._current_id()
            if current is None:
                raise ValueError("select a spectrum to fit")
            self._compton_config(current).include = True
            included = [current]
            self._rebuild_compton_datasets()
        units = {self._spectra[spectrum_id].axis_unit for spectrum_id in included}
        if len(units) > 1:
            raise ValueError("jointly fitted spectra must share one x axis (channel or keV)")
        datasets: list[ComptonEdgeDataset] = []
        for spectrum_id in included:
            spectrum = self._spectra[spectrum_id]
            config = self._compton_configs[spectrum_id]
            lines = self._compton_lines(config)
            if not lines:
                raise ValueError(f"select at least one source or custom line for {spectrum.label}")
            low, high = config.fit_range if config.fit_range is not None else (None, None)
            datasets.append(
                ComptonEdgeDataset(spectrum, lines, low, high, background=config.background)
            )
        terms = tuple(
            cast(ResolutionTermName, name)
            for name, box in (
                ("a", self.compton_fit_a),
                ("b", self.compton_fit_b),
                ("c", self.compton_fit_c),
            )
            if box.isChecked()
        )
        gain = self.compton_initial_gain.value()
        detector = self.compton_dk_combo.currentData()
        return ComptonEdgeFitSpec(
            datasets=tuple(datasets),
            offset_free=self.compton_offset_free.isChecked(),
            resolution_terms=terms,
            statistic=cast(FitStatistic, str(self.compton_statistic_combo.currentData())),
            amplitude_mode=cast(AmplitudeMode, str(self.compton_amplitude_combo.currentData())),
            initial_gain=gain if gain > 0 else None,
            initial_offset=0.0,
            initial_resolution=(
                self.compton_a.value(),
                self.compton_b.value(),
                self.compton_c.value(),
            ),
            dietze_klein_detector=int(detector) if detector is not None else None,
        )

    def _start_compton_fit(self) -> None:
        if self._compton_thread is not None:
            return
        try:
            spec = self._compton_fit_spec()
        except (KeyError, ValueError) as exc:
            self.compton_status.setText(f"Cannot fit Compton edges: {exc}")
            return
        thread = QThread(self)
        worker = ComptonEdgeFitWorker(spec)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result.connect(self._compton_fit_completed)
        worker.progress.connect(self._compton_fit_progress)
        worker.error.connect(self._compton_fit_failed)
        worker.cancelled.connect(self._compton_fit_cancelled)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            self._compton_fit_thread_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        thread.finished.connect(thread.deleteLater)
        self._compton_thread = thread
        self._compton_worker = worker
        self._pending_compton_spec = spec
        self.compton_fit_action.setEnabled(False)
        self.compton_cancel_action.setEnabled(True)
        self.compton_status.setText(
            f"Fitting {len(spec.datasets)} dataset(s) with "
            f"{sum(len(dataset.lines) for dataset in spec.datasets)} line(s)..."
        )
        self._set_compton_busy(True)
        thread.start()

    def _set_compton_busy(self, busy: bool) -> None:
        """Show the bottom progress bar and a busy cursor while the fit runs.

        BusyCursor (arrow with spinner) rather than WaitCursor: the window stays
        usable, including Cancel Compton-edge fit. Only this window's cursor changes.
        """
        self.workbench_progress.setRange(0, 0)
        self.workbench_progress.setVisible(busy)
        if busy:
            self.setCursor(Qt.CursorShape.BusyCursor)
            self.workbench_status.setText("Compton-edge fit: starting...")
        else:
            self.unsetCursor()

    def _compton_fit_progress(self, update: object) -> None:
        worker = self._compton_worker
        if (
            not isinstance(update, ComptonEdgeFitProgress)
            or worker is None
            or worker.stop_requested
        ):
            return
        bar = self.workbench_progress
        if update.total:
            bar.setRange(0, update.total)
            bar.setValue(min(update.done, update.total))
        else:
            bar.setRange(0, 0)
        if update.stage == "prepare":
            message = "preparing the detector-response model..."
        elif update.stage == "gain_scan":
            message = f"scanning the starting gain ({update.done}/{update.total})..."
        elif update.stage == "optimize":
            message = f"optimizing, {update.done} evaluation(s)..."
        else:
            message = "computing uncertainties and diagnostics..."
        self.workbench_status.setText(f"Compton-edge fit: {message}")

    def _compton_fit_completed(self, result: object) -> None:
        if not isinstance(result, ComptonEdgeFitResult):
            return
        self._compton_result = result
        self._compton_spec = self._pending_compton_spec
        self.compton_status.setText(
            f"Compton-edge fit complete using {result.statistic}; "
            f"{result.evaluations} evaluations ({result.evaluation_path} folding)."
            if result.success
            else f"Compton-edge fit stopped without convergence: {result.message}"
        )
        self._populate_compton_results(result)
        self.compton_export_action.setEnabled(True)
        self.compton_copy_action.setEnabled(True)
        self.compton_calibration_action.setEnabled(True)
        self._render()
        self.workbench_status.setText(
            "Compton-edge fit complete."
            if result.success
            else "Compton-edge fit stopped without convergence."
        )

    def _compton_fit_failed(self, message: str) -> None:
        self.compton_status.setText(f"Compton-edge fit failed: {message}")
        self.workbench_status.setText(f"Compton-edge fit failed: {message}")
        QMessageBox.critical(self, "Compton-Edge Fit Failed", message)

    def _compton_fit_cancelled(self) -> None:
        self.compton_status.setText("Compton-edge fit cancelled.")
        self.workbench_status.setText("Compton-edge fit cancelled.")

    def _compton_fit_thread_finished(self) -> None:
        self._compton_thread = None
        self._compton_worker = None
        self._set_compton_busy(False)
        self.compton_fit_action.setEnabled(self._current_spectrum() is not None)
        self.compton_cancel_action.setEnabled(False)

    def _cancel_compton_fit(self) -> None:
        if self._compton_worker is not None:
            self._compton_worker.stop()
            self.compton_cancel_action.setEnabled(False)
            self.compton_status.setText("Cancelling Compton-edge fit...")
            self.workbench_progress.setRange(0, 0)
            self.workbench_status.setText("Cancelling Compton-edge fit...")

    def _populate_compton_results(self, result: ComptonEdgeFitResult) -> None:
        rows = compton_edge_summary_rows(result)
        self.compton_result_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            if row.value is None:
                value = ""
            elif row.unit == "of maximum":
                value = f"{row.value:.3f}"
            elif row.unit == "relative" or row.quantity.endswith(
                ("σ/E at edge", "FWHM/E at edge", "(FWHM, DK rule)")
            ):
                value = f"{100.0 * row.value:+.3g} %" if row.unit == "relative" else (
                    f"{100.0 * row.value:.4g} %"
                )
            else:
                value = f"{row.value:.8g}"
            uncertainty = "" if row.stderr is None else f"{row.stderr:.3g}"
            note = row.unit if not row.note else (
                f"{row.unit}; {row.note}" if row.unit else row.note
            )
            if row.unit in {"of maximum", "relative"}:
                note = row.note or row.unit
            for column, text in enumerate((row.quantity, value, uncertainty, note)):
                item = QTableWidgetItem(text)
                if column == 3 and text:
                    item.setToolTip(text)
                self.compton_result_table.setItem(row_index, column, item)
        warnings = " ".join(result.warnings) if result.warnings else "No fit warnings."
        low, high = result.edge_span_kev
        self.compton_summary.setText(
            f"Statistic: {result.statistic}; reduced deviance {result.reduced_deviance:.4g} "
            f"({result.degrees_of_freedom} degrees of freedom); AIC {result.aic:.6g}; "
            f"BIC {result.bic:.6g}. Edges span {low:.4g}–{high:.4g} keVee; a keVee scale "
            f"outside that span is an extrapolation. {warnings}"
        )

    def _render_compton_overlay(self, selected: Spectrum | None) -> bool:
        result = self._compton_result
        if result is None or selected is None:
            return False
        dataset = next(
            (item for item in result.datasets if item.spectrum_id == selected.spectrum_id),
            None,
        )
        if dataset is None:
            return False
        peak_fit_here = (
            self._fit_result is not None and self._fit_spectrum_id == selected.spectrum_id
        )
        if peak_fit_here and not self._compton_tab_active():
            return False
        self.spectrum_plot.plot(
            dataset.x,
            dataset.best_fit,
            pen=pg.mkPen("#d62728", width=2.0),
            name="Compton-edge model",
        )
        for index, (name, values) in enumerate(dataset.components.items()):
            color = _COMPTON_COLORS[index % len(_COMPTON_COLORS)]
            style = Qt.PenStyle.DotLine if name == "background" else Qt.PenStyle.SolidLine
            self.spectrum_plot.plot(
                dataset.x,
                values,
                pen=pg.mkPen(color, width=1.0, style=style),
                name=name.strip(),
            )
        lines = [line for line in result.lines if line.dataset_index == dataset.index]
        for index, line in enumerate(lines):
            color = _COMPTON_COLORS[index % len(_COMPTON_COLORS)]
            marker = pg.InfiniteLine(
                pos=line.edge_x,
                angle=90,
                movable=False,
                pen=pg.mkPen(color, width=1.5, style=Qt.PenStyle.DashLine),
                label=f"edge {line.energy_kev:g} keV",
                labelOpts={
                    "position": 0.9 - 0.07 * (index % 4),
                    "color": color,
                    "fill": (255, 255, 255, 190),
                    "movable": False,
                },
            )
            error = "" if line.edge_x_stderr is None else f" ± {line.edge_x_stderr:.4g}"
            marker.setToolTip(
                f"{line.label}: Compton edge T_max = {line.edge_kev:.4f} keVee\n"
                f"x = {line.edge_x:.6g}{error}\n"
                f"σ/E at edge {100 * line.sigma_over_energy:.3g} %, "
                f"FWHM/E {100 * line.fwhm_over_energy:.3g} %\nResponse: {line.response}"
            )
            marker.setZValue(_COMPTON_EDGE_Z)
            self.spectrum_plot.addItem(marker)
        self.residual_curve.setData(dataset.x, dataset.residual)
        return True

    def _forget_compton_spectra(self, spectrum_ids: set[str]) -> None:
        for spectrum_id in spectrum_ids:
            self._compton_configs.pop(spectrum_id, None)
        result = self._compton_result
        if result is not None and any(
            dataset.spectrum_id in spectrum_ids for dataset in result.datasets
        ):
            self._compton_result = None
            self._compton_spec = None
            self.compton_result_table.setRowCount(0)
            self.compton_summary.setText("No Compton-edge fit result.")
            for action in (
                self.compton_export_action,
                self.compton_copy_action,
                self.compton_calibration_action,
            ):
                action.setEnabled(False)

    def _load_compton_response(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        config = self._compton_config(spectrum.spectrum_id)
        try:
            lines = self._compton_lines(config, with_responses=False)
        except ValueError as exc:
            self.compton_status.setText(str(exc))
            return
        if not lines:
            QMessageBox.information(
                self,
                "Simulated Response",
                "Select the source lines of the selected spectrum first.",
            )
            return
        labels = [line.label for line in lines]
        choice, accepted = QInputDialog.getItem(
            self, "Simulated Response", "Gamma line:", labels, 0, False
        )
        if not accepted:
            return
        line = lines[labels.index(choice)]
        folder = str(QSettings().value(settings_keys.ANALYSIS_SPECTRUM_FOLDER, "measurements"))
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            f"Load Simulated Response for {line.label}",
            folder,
            "Response tables (*.csv *.CSV *.txt *.tsv);;All files (*)",
        )
        if not path_text:
            return
        try:
            response = load_simulated_response(Path(path_text))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Simulated Response", str(exc))
            return
        config.responses[round(line.energy_kev, 3)] = response
        self._update_compton_response_label()
        self.compton_status.setText(
            f"{line.label} now uses {response.label}, up to {response.max_energy_kev:g} keV."
        )

    def _clear_compton_responses(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        config = self._compton_configs.get(spectrum.spectrum_id)
        if config is not None:
            config.responses.clear()
        self._update_compton_response_label()

    def _send_compton_calibration_points(self) -> None:
        result = self._compton_result
        spec = self._compton_spec
        if result is None or spec is None:
            return
        signal = QMetaMethod.fromSignal(self.compton_calibration_requested)
        if not self.isSignalConnected(signal):
            # Never report a transfer that nothing receives.
            QMessageBox.warning(
                self,
                "Compton-Edge Calibration",
                "MCA Energy Calibration is not connected to this workbench, so the "
                "edges could not be opened there.",
            )
            return
        if any(dataset.spectrum.axis_unit != "channel" for dataset in spec.datasets):
            QMessageBox.information(
                self,
                "Compton-Edge Calibration",
                "Calibration points need spectra on an MCA channel axis.",
            )
            return
        fingerprints: list[dict[str, FingerprintValue]] = []
        for dataset in spec.datasets:
            raw = dataset.spectrum.metadata.get("settings_fingerprint")
            fingerprint: dict[str, FingerprintValue] = {}
            if isinstance(raw, dict):
                fingerprint = {
                    str(key): value
                    for key, value in raw.items()
                    if isinstance(value, bool | int | float | str)
                }
            fingerprints.append(fingerprint)
        if len({json.dumps(item, sort_keys=True) for item in fingerprints}) > 1:
            QMessageBox.warning(
                self,
                "Compton-Edge Calibration",
                "The fitted spectra were acquired with different MCA energy settings. Fit "
                "spectra from one configuration before creating calibration points.",
            )
            return
        channels = {dataset.spectrum.metadata.get("channel") for dataset in spec.datasets}
        channel_value = next(iter(channels)) if len(channels) == 1 else None
        channel = (
            channel_value
            if isinstance(channel_value, int) and not isinstance(channel_value, bool)
            else None
        )
        points = compton_edge_calibration_points(result)
        low, high = result.edge_span_kev
        labels = ", ".join(dataset.label for dataset in spec.datasets)
        transfer = CalibrationPointTransfer(
            points=points,
            energy_unit=COMPTON_EDGE_ENERGY_UNIT,
            source_label=f"Compton-edge fit: {labels}",
            channel=channel,
            fingerprint=fingerprints[0],
            note=(
                f"Compton edges span {low:.4g}–{high:.4g} keVee; light output outside "
                "that span is extrapolated."
            ),
        )
        self.compton_calibration_requested.emit(transfer)
        self.compton_status.setText(
            f"Opened {len(points)} edge point(s) in keVee in MCA Energy Calibration; review "
            f"and apply the calibration there. Edges span {low:.4g}–{high:.4g} keVee; "
            "outside that span the keVee scale is extrapolated."
        )

    def _export_compton_result(self) -> None:
        result = self._compton_result
        spec = self._compton_spec
        if result is None or spec is None:
            return
        folder = str(QSettings().value(settings_keys.ANALYSIS_SPECTRUM_FOLDER, "measurements"))
        stem = _safe_stem(spec.datasets[0].label)
        path_text, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Compton-Edge Results",
            str(Path(folder) / f"{stem}_compton_edges.json"),
            "JSON (*.json);;CSV (*.csv)",
        )
        if not path_text:
            return
        path = Path(path_text)
        if selected_filter.startswith("CSV") and path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        elif not path.suffix:
            path = path.with_suffix(".json")
        try:
            export_compton_edge_result(path, spec=spec, result=result)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Compton-Edge Export Failed", str(exc))
            return
        self.workbench_status.setText(f"Compton-edge results exported to {path}")

    def _copy_compton_table(self) -> None:
        table = self.compton_result_table
        headers = [
            (header.text() if (header := table.horizontalHeaderItem(column)) else "")
            for column in range(table.columnCount())
        ]
        rows = ["\t".join(headers)]
        for row in range(table.rowCount()):
            cells = [
                (item.text() if (item := table.item(row, column)) else "")
                for column in range(table.columnCount())
            ]
            rows.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(rows))
        self.workbench_status.setText("Compton-edge result table copied to the clipboard.")

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
        compton_worker = self._compton_worker
        compton_thread = self._compton_thread
        if compton_worker is not None:
            compton_worker.stop()
        if (
            compton_thread is not None
            and compton_thread.isRunning()
            and not compton_thread.wait(5_000)
        ):
            log.warning("Compton-edge fit worker did not stop within 5 seconds")

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
