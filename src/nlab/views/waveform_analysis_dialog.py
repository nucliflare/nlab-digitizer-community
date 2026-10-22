"""Standalone waveform browser and offline charge/PSD workbench."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QThread, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.capture_metadata import capture_digitizer_info
from nlab.analysis.psd_report import PsdReportRequest, export_psd_report
from nlab.analysis.waveform_cluster_export import ClusterFormat
from nlab.analysis.waveform_file import MappedWaveformFile, WaveformFileIndex, WaveformFrame
from nlab.analysis.waveform_fom_optimization import (
    WaveformFomOptimizationRequest,
    WaveformFomOptimizationResult,
)
from nlab.analysis.waveform_mca_gates import GateTimingProposal, WaveformGateSource
from nlab.analysis.waveform_psd import (
    BaselineMethod,
    WaveformPsdResult,
    WaveformPsdSettings,
    infer_waveform_polarity,
    integrate_waveform,
)
from nlab.help.context import HelpContext
from nlab.views.help_dialog import HelpDialog
from nlab.views.offline_psd_plot import OfflinePsdPlot
from nlab.views.plot_viewbox import WAVEFORM_STEP_MODE, ModifierZoomViewBox, style_plot_widget
from nlab.views.time_axis import time_axis_scale
from nlab.views.tooltips import wrap_widget_tooltips
from nlab.views.waveform_cluster_export_dialog import (
    ClusterExportRequest,
    WaveformClusterExportDialog,
)
from nlab.views.waveform_cluster_selection import WaveformClusterSelection
from nlab.views.waveform_mca_transfer_dialog import McaGateTransferTarget, WaveformMcaTransferDialog
from nlab.workers.waveform_average_worker import WaveformAverage, WaveformAverageWorker
from nlab.workers.waveform_file_worker import WaveformFileIndexWorker
from nlab.workers.waveform_fom_optimization_worker import WaveformFomOptimizationWorker
from nlab.workers.waveform_psd_worker import WaveformPsdWorker

log = logging.getLogger(__name__)
_MAX_QT_INDEX = 2_147_483_647
_AUTO_PREVIEW_STRIDE_MULTIPLIER = 20
_AUTO_PREVIEW_DEBOUNCE_MS = 120
_WINDOW_TITLE = "Waveform Analysis Workbench"
AnalysisKind = Literal["manual", "sparse", "full", "optimized"]


@dataclass(frozen=True)
class _WaveformAnalysisContext:
    settings: WaveformPsdSettings
    source_index: int
    stride: int
    maximum_events: int
    analysis_kind: AnalysisKind
    sample_period_ns: float
    optimization: WaveformFomOptimizationResult | None = None


@dataclass(frozen=True)
class _FomOptimizationContext:
    fit_roi: tuple[float, float]
    centroid_seeds: tuple[float, float]
    result: WaveformFomOptimizationResult | None = None


class WaveformAnalysisDialog(QDialog):
    """Browse saved waveforms and reconstruct short/long-gate PSD offline."""

    def __init__(self, parent: QWidget | None = None, *,
                 gate_targets: Callable[[], Sequence[McaGateTransferTarget]] | None = None) -> None:
        super().__init__(parent)
        self._gate_targets = gate_targets
        self._set_current_file(None)
        self.setModal(False)
        self.resize(1420, 900)
        self._index: WaveformFileIndex | None = None
        self._reader: MappedWaveformFile | None = None
        self._frame: WaveformFrame | None = None
        self._index_thread: QThread | None = None
        self._index_worker: WaveformFileIndexWorker | None = None
        self._analysis_thread: QThread | None = None
        self._analysis_worker: WaveformPsdWorker | None = None
        self._optimization_thread: QThread | None = None
        self._optimization_worker: WaveformFomOptimizationWorker | None = None
        self._optimization_total_pairs: int | None = None
        self._active_analysis_kind: AnalysisKind | None = None
        self._pending_analysis_kind: AnalysisKind | None = None
        self._active_optimization_context: _FomOptimizationContext | None = None
        self._completed_optimization: _FomOptimizationContext | None = None
        self._pending_optimized_reconstruction: _FomOptimizationContext | None = None
        self._active_report_context: _WaveformAnalysisContext | None = None
        self._last_report_context: _WaveformAnalysisContext | None = None
        self._last_psd_result: WaveformPsdResult | None = None
        self._selected_frame_indices: np.ndarray | None = None
        self._average_thread: QThread | None = None
        self._average_worker: WaveformAverageWorker | None = None
        self._average_frames: np.ndarray | None = None
        self._average: WaveformAverage | None = None
        self._last_sample_period = 1.0
        self._updating_gates = False
        self._menu_button_actions: dict[QAbstractButton, QAction] = {}
        self._help_dialog: HelpDialog | None = None
        self._cluster_export_dialog: WaveformClusterExportDialog | None = None
        self._auto_preview_timer = QTimer(self)
        self._auto_preview_timer.setSingleShot(True)
        self._auto_preview_timer.setInterval(_AUTO_PREVIEW_DEBOUNCE_MS)
        self._build_ui()
        self._fit_initial_size_to_screen()

    def _fit_initial_size_to_screen(self) -> None:
        """Keep the complete workbench inside the screen's usable desktop."""
        parent = self.parentWidget()
        screen = parent.screen() if parent is not None else QApplication.primaryScreen()
        if screen is None:
            self.resize(1420, 900)
            return
        available = screen.availableGeometry()
        self.resize(
            min(1420, max(1, available.width() - 32)),
            min(900, max(1, available.height() - 48)),
        )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.setToolTip(
            "Waveforms are read without changing live Scope or PSD acquisition. Drag the "
            "baseline region and the three gate markers, inspect individual events, then "
            "integrate the selected source in a background worker. Press F1 for parameter help."
        )

        self.source_settings = QGroupBox("Waveform source", self)
        source_form = QFormLayout(self.source_settings)
        self.source_combo = QComboBox(self)
        self.source_combo.setMinimumContentsLength(12)
        self.source_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.source_combo.setEnabled(False)
        self.frame_spin = QSpinBox(self)
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setEnabled(False)
        self.sample_period = QDoubleSpinBox(self)
        self.sample_period.setRange(0.001, 1_000_000.0)
        self.sample_period.setDecimals(4)
        self.sample_period.setValue(2.0)
        self.sample_period.setSuffix(" ns")
        self.sample_period.setEnabled(False)
        source_form.addRow("Source:", self.source_combo)
        source_form.addRow("Frame:", self.frame_spin)
        source_form.addRow("Sample period:", self.sample_period)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.settings_panel = controls = QWidget(self)
        self.settings_scroll = QScrollArea(self)
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setMinimumWidth(320)
        self.settings_scroll.setMaximumWidth(390)
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(self.source_settings)
        controls_layout.addWidget(self._build_preprocessing_group())
        controls_layout.addWidget(self._build_histogram_group())
        controls_layout.addStretch(1)
        for form in controls.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for combo in controls.findChildren(QComboBox):
            combo.setMinimumContentsLength(12)
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
        self.settings_scroll.setWidget(controls)
        splitter.addWidget(self.settings_scroll)

        self.plot_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.waveform_widget = pg.PlotWidget(self, viewBox=ModifierZoomViewBox())
        self.waveform_widget.setMinimumHeight(200)
        style_plot_widget(self.waveform_widget)
        self.waveform_plot = self.waveform_widget.getPlotItem()
        self.waveform_plot.showAxis("top")
        self.waveform_plot.showAxis("right")
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.2)
        self.waveform_plot.setLabel("bottom", "Time", units="ns")
        for side in ("bottom", "top"):
            self.waveform_plot.getAxis(side).enableAutoSIPrefix(False)
        self.waveform_plot.getViewBox().sigXRangeChanged.connect(self._update_waveform_time_axis)
        self._update_waveform_time_axis()
        self.waveform_plot.setLabel("left", "ADC amplitude", units="raw")
        # Negative offsets anchor to the bottom-right and follow plot resizing.
        self.waveform_plot.addLegend(offset=(-10, -10))
        self.waveform_widget.setToolTip(
            "Blue region: baseline. Green: integration start. Yellow: short-gate end. "
            "Red: long-gate end. Gate positions are stored as sample indices. "
            "Press F1 and click this plot for gate help."
        )
        self.raw_curve = self.waveform_plot.plot(
            pen=pg.mkPen("#1597d4", width=1.2), name="Stored waveform",
            stepMode=WAVEFORM_STEP_MODE,
        )
        self.oriented_curve = self.waveform_plot.plot(
            pen=pg.mkPen("#f28e2b", width=1.2), name="Polarity-normalized",
            stepMode=WAVEFORM_STEP_MODE,
        )
        self.average_curve = self.waveform_plot.plot(
            pen=pg.mkPen("#8c2d91", width=2),
            stepMode=WAVEFORM_STEP_MODE,
        )
        self.average_curve.hide()
        self.baseline_region = pg.LinearRegionItem(
            values=(0.0, 1.0),
            orientation="vertical",
            brush=pg.mkBrush(80, 140, 220, 35),
        )
        self.baseline_region.setZValue(20.0)
        self.waveform_plot.addItem(self.baseline_region)
        self.gate_start_line = self._gate_line("#2a9d8f")
        self.short_end_line = self._gate_line("#e9c46a")
        self.long_end_line = self._gate_line("#e76f51")
        self.plot_splitter.addWidget(self.waveform_widget)

        self.psd_plot = OfflinePsdPlot(self)
        self.cluster_selection = WaveformClusterSelection(self.psd_plot.matrix_plot, self)
        cast(QVBoxLayout, self.psd_plot.layout()).insertWidget(1, self.cluster_selection)
        self.cluster_selection.selection_changed.connect(self._cluster_selection_changed)
        self.cluster_selection.frame_requested.connect(self._show_cluster_frame)
        self.cluster_selection.average_button.toggled.connect(self._toggle_cluster_average)
        self.waveform_selection_checkbox = QCheckBox("Waveform selection tools", self)
        self.waveform_selection_checkbox.setChecked(True)
        self.waveform_selection_checkbox.setProperty("nlabHelpTopic", "waveform-clusters")
        self.waveform_selection_checkbox.setToolTip(
            "Show/hide the cluster-selection controls, region, and current-event marker. "
            "The active waveform filter and selected match number are preserved."
        )
        self.psd_plot.controls_layout.insertWidget(
            self.psd_plot.controls_layout.count() - 1, self.waveform_selection_checkbox,
        )
        self.waveform_selection_checkbox.toggled.connect(self.cluster_selection.set_tools_visible)
        self.psd_plot.setMinimumHeight(360)
        self.plot_splitter.addWidget(self.psd_plot)
        self.plot_splitter.setStretchFactor(0, 1)
        self.plot_splitter.setStretchFactor(1, 3)
        self.plot_splitter.setSizes([220, 520])
        splitter.addWidget(self.plot_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        status_row = QHBoxLayout()
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(230)
        status_row.addWidget(self.progress)
        self.status = QLabel("Select a waveform file to begin.", self)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        status_row.addWidget(self.status, 1)
        self.report_button = QPushButton("Download report...", self)
        self.report_button.setEnabled(False)
        self.report_button.setToolTip(
            "Save a PDF report and all plotted PSD/FOM data and fit inputs as CSV files."
        )
        status_row.addWidget(self.report_button)
        self.close_button = QPushButton("Close", self)
        status_row.addWidget(self.close_button)
        root.addLayout(status_row)

        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.frame_spin.valueChanged.connect(self._show_frame)
        self.sample_period.editingFinished.connect(self._sample_period_changed)
        self.polarity_combo.currentIndexChanged.connect(self._preview)
        self.baseline_method.currentIndexChanged.connect(self._preview)
        self.baseline_region.sigRegionChanged.connect(self._on_gate_moved)
        self.baseline_region.sigRegionChangeFinished.connect(
            self._on_gate_change_finished
        )
        for line in (self.gate_start_line, self.short_end_line, self.long_end_line):
            line.sigPositionChanged.connect(self._on_gate_moved)
            line.sigPositionChangeFinished.connect(self._on_gate_change_finished)
        self._auto_preview_timer.timeout.connect(self._request_sparse_analysis)
        self.analyze_button.clicked.connect(self._start_analysis)
        self.optimize_button.clicked.connect(self._start_fom_optimization)
        self.cancel_button.clicked.connect(self._cancel_analysis)
        self.report_button.clicked.connect(self._download_report)
        self.psd_plot.fom_widget.fit_changed.connect(self._fit_changed)
        for control in (self.energy_bins, self.energy_max, self.ratio_bins,
                        self.ratio_min, self.ratio_max, self.event_stride, self.maximum_events):
            control.valueChanged.connect(self._invalidate_cluster_if_stale)
        self.close_button.clicked.connect(self.reject)
        wrap_widget_tooltips(self)
        self._build_menu_bar(root)

    def _update_waveform_time_axis(self, *_args: object) -> None:
        """Keep plot/gate coordinates in ns; scale only the displayed ticks."""
        low, high = self.waveform_plot.viewRange()[0]
        scale = time_axis_scale(high - low)
        # Applying an SI prefix to an already-prefixed "ns" produced "kns".
        # Select physical time units explicitly, identically on both axes.
        bottom = self.waveform_plot.getAxis("bottom")
        if bottom.labelUnits != scale.unit:
            bottom.setLabel("Time", units=scale.unit)
        for side in ("bottom", "top"):
            self.waveform_plot.getAxis(side).setScale(1.0 / scale.ns_per_unit)

    def _build_menu_bar(self, root: QVBoxLayout) -> None:
        self.menu_bar = QMenuBar(self)
        self.menu_bar.setNativeMenuBar(False)
        root.setMenuBar(self.menu_bar)
        self.file_menu = file_menu = self.menu_bar.addMenu("&File")
        self.load_action = file_menu.addAction("&Open waveform file...")
        self.load_action.setShortcut(QKeySequence("Ctrl+O"))
        self.load_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.load_action.triggered.connect(self._choose_file)
        self.close_file_action = file_menu.addAction("Close &file")
        self.close_file_action.setShortcut(QKeySequence("Ctrl+Shift+W"))
        self.close_file_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.close_file_action.setEnabled(False)
        self.close_file_action.triggered.connect(self.close_file)
        self.export_cluster_action = file_menu.addAction("&Export selected cluster...")
        self.export_cluster_action.setEnabled(False)
        self.export_cluster_action.triggered.connect(self._export_selected_cluster)
        self.transfer_gates_action = file_menu.addAction("Transfer &gates to MCA...")
        self.transfer_gates_action.setEnabled(False)
        self.transfer_gates_action.triggered.connect(self._transfer_gates_to_mca)
        self._menu_button(file_menu, "Download &report...", self.report_button, "Ctrl+S")
        file_menu.addSeparator()
        self._menu_button(file_menu, "&Close workbench", self.close_button, "Ctrl+W")

        self.analysis_menu = analysis_menu = self.menu_bar.addMenu("&Analysis")
        self._menu_button(analysis_menu, "&Reconstruct PSD", self.analyze_button, "Ctrl+R")
        self._menu_button(
            analysis_menu, "&Fit double Gaussian", self.psd_plot.fom_widget.fit_button,
        )
        self._menu_button(analysis_menu, "&Optimize FOM gates", self.optimize_button)
        self._menu_button(analysis_menu, "&Auto preview", self.auto_recalculate)
        analysis_menu.addSeparator()
        self._menu_button(analysis_menu, "&Cancel analysis", self.cancel_button)

        self.selection_menu = selection_menu = self.menu_bar.addMenu("&Selection")
        cluster = self.cluster_selection
        for label, button in (
            ("&Rectangle", cluster.rectangle_button), ("&Polygon", cluster.polygon_button),
            ("&Apply selection", cluster.apply_button), ("&Clear selection", cluster.clear_button),
        ):
            self._menu_button(selection_menu, label, button)
        selection_menu.addSeparator()
        self._menu_button(selection_menu, "&Previous waveform", cluster.previous_button)
        self._menu_button(selection_menu, "&Next waveform", cluster.next_button)
        self._menu_button(selection_menu, "Show a&verage waveform", cluster.average_button)

        self.view_menu = view_menu = self.menu_bar.addMenu("&View")
        self._menu_button(view_menu, "Show PSD &ROI", self.psd_plot.roi_checkbox)
        self._menu_button(view_menu, "Waveform &selection tools", self.waveform_selection_checkbox)
        self._menu_button(view_menu, "FOM fit &guides", self.psd_plot.fom_widget.guides_checkbox)
        view_menu.addSeparator()
        reset_action = view_menu.addAction("Reset &waveform zoom")
        reset_action.triggered.connect(lambda: self.waveform_plot.autoRange())

        self.help_menu = help_menu = self.menu_bar.addMenu("&Help")
        guide_action = help_menu.addAction("Workbench &guide...")
        guide_action.triggered.connect(self._show_workbench_help)
        # Keep menu-action wrappers alive as well as menus: PySide's temporary
        # QAction.menu() wrappers can otherwise invalidate their owned submenu.
        self._menu_headers = self.menu_bar.actions()

    def _menu_button(
        self, menu: QMenu, label: str, button: QAbstractButton, shortcut: str = "",
    ) -> None:
        """Route menu actions through existing controls, including their safety gates."""
        action = menu.addAction(label)
        action.setEnabled(button.isEnabled())
        action.setToolTip(button.toolTip())
        action.setCheckable(button.isCheckable())
        action.setChecked(button.isChecked())
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        action.triggered.connect(button.click)
        button.toggled.connect(action.setChecked)
        self._menu_button_actions[button] = action
        button.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if isinstance(watched, QAbstractButton) and event.type() == QEvent.Type.EnabledChange:
            action = self._menu_button_actions.get(watched)
            if action is not None:
                action.setEnabled(watched.isEnabled())
            if watched is self.cluster_selection.average_button:
                self._update_export_action()
        return super().eventFilter(watched, event)

    def _update_export_action(self) -> None:
        self._update_gate_transfer_action()
        action = getattr(self, "export_cluster_action", None)
        if action is not None:
            frames = self._selected_frame_indices
            action.setEnabled(
                frames is not None and len(frames) > 0 and self._index is not None
                and self._last_psd_result is not None and self._last_report_context is not None
                and self._analysis_thread is None and self._optimization_thread is None
                and self._index_thread is None and self._cluster_export_dialog is None
            )

    def _transfer_gates_to_mca(self) -> None:
        if (self._index is None or self._frame is None or self._index_thread is not None
                or self._analysis_thread is not None or self._optimization_thread is not None
                or self._cluster_export_dialog is not None
                or self._pending_analysis_kind is not None):
            return
        markers: list[pg.InfiniteLine] = []
        try:
            self._auto_preview_timer.stop()
            baseline_start, baseline_end, start, short, end = self._gate_indices()
            if not 0 <= baseline_start < baseline_end <= start < short <= end <= len(
                self._frame.samples,
            ):
                raise ValueError("Baseline and gates must be ordered inside the displayed waveform")
            source = WaveformGateSource(baseline_end, start, short, end, self.sample_period.value())
            targets = self._gate_targets() if self._gate_targets is not None else ()
            dialog = WaveformMcaTransferDialog(source, targets, self)
            for color in ("#2a9d8f", "#e9c46a", "#e76f51"):
                line = pg.InfiniteLine(
                    movable=False, pen=pg.mkPen(color, width=3, style=Qt.PenStyle.DashLine),
                )
                line.setZValue(40)
                self.waveform_plot.addItem(line, ignoreBounds=True)
                markers.append(line)

            def preview(value: object) -> None:
                for index, marker in enumerate(markers):
                    marker.setVisible(isinstance(value, GateTimingProposal))
                    if isinstance(value, GateTimingProposal):
                        marker.setPos(value.boundaries_ns[index])

            dialog.preview_changed.connect(preview)
            dialog.refresh()
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.status.setText(
                    "Candidate gate settings transferred to MCA; acquisition remains stopped."
                )
            dialog.deleteLater()
        except Exception as exc:
            log.exception("Could not open waveform-to-MCA gate transfer")
            QMessageBox.warning(self, "MCA gate transfer unavailable", str(exc))
        finally:
            for marker in markers:
                self.waveform_plot.removeItem(marker)

    def _export_selected_cluster(self) -> None:
        self._update_export_action()
        if not self.export_cluster_action.isEnabled():
            return
        index, context = self._index, self._last_report_context
        frames, result = self._selected_frame_indices, self._last_psd_result
        assert index is not None and context is not None
        assert frames is not None and result is not None
        filters = ["ROOT waveforms (*.root)", "HDF5 waveforms (*.h5 *.hdf5)"]
        if index.caen_info is None:
            filters.insert(0, "nlab waveform binary (*.bin)")
        suffix = ".bin" if index.caen_info is None else ".root"
        destination, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Selected Waveform Cluster",
            str(index.path.with_name(index.path.stem + "_cluster" + suffix)),
            ";;".join(filters), options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not destination:
            return
        try:
            path = Path(destination)
            if not path.suffix:
                suffix = {filters[0]: suffix, "ROOT waveforms (*.root)": ".root",
                          "HDF5 waveforms (*.h5 *.hdf5)": ".h5"}.get(selected_filter, suffix)
                path = path.with_suffix(suffix)
            formats: dict[str, ClusterFormat] = {
                ".bin": "bin", ".root": "root", ".h5": "h5", ".hdf5": "h5",
            }
            if path.suffix.lower() not in formats:
                raise ValueError("Choose a .bin, .root, .h5, or .hdf5 filename")
            if not self._cluster_context_matches():
                self.cluster_selection.invalidate()
                raise ValueError("Reconstruct PSD and reapply the selection with current settings")
            positions = np.searchsorted(result.event_frames, frames)
            if (np.any(positions >= len(result.event_frames))
                    or not np.array_equal(result.event_frames[positions], frames)):
                raise ValueError("Selected source frames do not match the reconstructed PSD")
            request = ClusterExportRequest(
                index=index, destination=path, file_format=formats[path.suffix.lower()],
                source_index=context.source_index, frames=frames.copy(),
                energy=result.event_energy[positions].copy(),
                ratio=result.event_ratio[positions].copy(),
                metadata={"settings": asdict(context.settings),
                          "sample_period_ns": context.sample_period_ns,
                          "stride": context.stride, "maximum_events": context.maximum_events,
                          "analysis_kind": context.analysis_kind,
                          "selection_vertices": self.cluster_selection.region_vertices().tolist()},
            )
            dialog = WaveformClusterExportDialog(request, self)
            self._cluster_export_dialog = dialog
            self._update_export_action()
            try:
                dialog.exec()
                if dialog.export_result is not None:
                    exported = dialog.export_result
                    extra = f" (metadata: {exported.sidecar.name})" if exported.sidecar else ""
                    self.status.setText(f"Exported {exported.count:,} waveforms to {path}{extra}")
                elif dialog.error:
                    self.status.setText(f"Cluster export failed: {dialog.error}")
                    QMessageBox.warning(self, "Cluster Export Failed", dialog.error)
                else:
                    self.status.setText("Cluster export cancelled; incomplete output removed.")
            finally:
                self._cluster_export_dialog = None
                dialog.deleteLater()
                self._update_export_action()
        except Exception as exc:
            log.exception("Could not export selected waveform cluster")
            self.status.setText(f"Cluster export failed: {exc}")
            QMessageBox.warning(self, "Cluster Export Failed", str(exc))

    def _show_workbench_help(self) -> None:
        try:
            if self._help_dialog is None:
                self._help_dialog = HelpDialog(self)
            self._help_dialog.show_topic(HelpContext("waveform"))
            self._help_dialog.show()
            self._help_dialog.raise_()
            self._help_dialog.activateWindow()
        except Exception:
            log.exception("Could not open waveform workbench help")
            QMessageBox.warning(self, "Help unavailable", "The bundled help could not be loaded. "
                                "See System Log for details.")

    def _build_preprocessing_group(self) -> QGroupBox:
        group = QGroupBox("Preprocessing", self)
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.baseline_method = QComboBox(self)
        self.baseline_method.addItem("Median (robust)", "median")
        self.baseline_method.addItem("Mean", "mean")
        self.polarity_combo = QComboBox(self)
        self.polarity_combo.addItem("Auto from displayed frame", 0)
        self.polarity_combo.addItem("Negative-going pulse", -1)
        self.polarity_combo.addItem("Positive-going pulse", 1)
        form.addRow("Baseline estimator:", self.baseline_method)
        form.addRow("Pulse polarity:", self.polarity_combo)
        layout.addLayout(form)
        self.preview_status = QLabel("No frame loaded.", self)
        self.preview_status.setWordWrap(True)
        self.preview_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.preview_status)
        return group

    def _build_histogram_group(self) -> QGroupBox:
        group = QGroupBox("PSD reconstruction", self)
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.energy_bins = QSpinBox(self)
        self.energy_bins.setRange(16, 8192)
        self.energy_bins.setValue(512)
        self.energy_max = QDoubleSpinBox(self)
        self.energy_max.setRange(1.0, 1.0e15)
        self.energy_max.setDecimals(1)
        self.energy_max.setValue(1_000_000.0)
        self.ratio_bins = QSpinBox(self)
        self.ratio_bins.setRange(16, 4096)
        self.ratio_bins.setValue(256)
        self.ratio_min = QDoubleSpinBox(self)
        self.ratio_min.setRange(-10.0, 10.0)
        self.ratio_min.setDecimals(4)
        self.ratio_min.setValue(-1.0)
        self.ratio_max = QDoubleSpinBox(self)
        self.ratio_max.setRange(-10.0, 10.0)
        self.ratio_max.setDecimals(4)
        self.ratio_max.setValue(1.0)
        self.event_stride = QSpinBox(self)
        self.event_stride.setRange(1, 1_000_000)
        self.event_stride.setValue(1)
        self.maximum_events = QSpinBox(self)
        self.maximum_events.setRange(0, _MAX_QT_INDEX)
        self.maximum_events.setSpecialValueText("All")
        self.optimization_events = QSpinBox(self)
        self.optimization_events.setRange(100, 1_000_000)
        self.optimization_events.setValue(20_000)
        self.optimization_events.setToolTip(
            "Maximum sampled waveforms used to compare candidate gate pairs."
        )
        self.optimization_radius = QSpinBox(self)
        self.optimization_radius.setRange(1, _MAX_QT_INDEX)
        self.optimization_radius.setValue(32)
        self.optimization_radius.setSuffix(" samples")
        self.optimization_radius.setToolTip(
            "Search this many samples on either side of both current gate endpoints."
        )
        self.exhaustive_search = QCheckBox("Exhaustive search", self)
        self.exhaustive_search.setToolTip(
            "Test every valid short/long endpoint pair within the search radius at "
            "one-sample steps. This can take a long time; confirmation is required "
            "before starting. Leave unchecked for the faster local search."
        )
        form.addRow("Energy bins:", self.energy_bins)
        form.addRow("Energy maximum:", self.energy_max)
        form.addRow("Ratio bins:", self.ratio_bins)
        form.addRow("Ratio minimum:", self.ratio_min)
        form.addRow("Ratio maximum:", self.ratio_max)
        form.addRow("Event stride:", self.event_stride)
        form.addRow("Maximum events:", self.maximum_events)
        form.addRow("Optimization events:", self.optimization_events)
        form.addRow("Gate search radius:", self.optimization_radius)
        form.addRow("Search mode:", self.exhaustive_search)
        layout.addLayout(form)
        self.auto_recalculate = QCheckBox("Auto preview", self)
        self.auto_recalculate.setChecked(True)
        self.auto_recalculate.setToolTip(
            "While dragging, gate motion is debounced and recalculated using every "
            "20th selected event. Releasing a marker "
            "queues a full calculation using the configured event stride."
        )
        layout.addWidget(self.auto_recalculate)
        buttons = QVBoxLayout()
        self.analyze_button = QPushButton("Reconstruct PSD", self)
        self.analyze_button.setEnabled(False)
        self.optimize_button = QPushButton("Optimize FOM gates", self)
        self.optimize_button.setEnabled(False)
        self.optimize_button.setToolTip(
            "Maximize FOM by varying the short- and long-gate endpoints while keeping "
            "the baseline, integration start, energy ROI, fit ROI, and seeds fixed."
        )
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setEnabled(False)
        buttons.addWidget(self.analyze_button)
        buttons.addWidget(self.optimize_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        return group

    def _gate_line(self, color: str) -> pg.InfiniteLine:
        line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen(color, width=2),
            hoverPen=pg.mkPen(color, width=4),
        )
        line.setZValue(30.0)
        self.waveform_plot.addItem(line)
        return line

    def _choose_file(self) -> None:
        folder = str(QSettings().value("analysis/waveform_folder", "measurements"))
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Open Waveform File",
            folder,
            "Waveform binaries (*.bin *.BIN);;All files (*)",
        )
        if path_text:
            self.open_path(Path(path_text))

    def open_path(self, path: Path) -> None:
        if (
            self._index_thread is not None
            or self._analysis_thread is not None
            or self._optimization_thread is not None
        ):
            QMessageBox.warning(self, "Waveform Analysis", "Cancel the current operation first.")
            return
        self.close_file()
        worker = WaveformFileIndexWorker(path)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._index_progress)
        worker.loaded.connect(self._index_loaded)
        worker.cancelled.connect(self._index_cancelled)
        worker.error.connect(self._index_failed)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        # Delete the worker only once QThread has stopped its event loop. If
        # deleteLater() is queued from worker.finished after the direct quit,
        # a long-running job can leave the QObject alive until the GUI drops
        # its final Python reference, causing cross-thread native destruction.
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            self._index_thread_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._index_worker = worker
        self._index_thread = thread
        self._set_indexing(True)
        self.status.setText(f"Indexing {path.name}...")
        self.progress.setValue(0)
        QSettings().setValue("analysis/waveform_folder", str(path.parent))
        thread.start()

    def _index_progress(self, processed: int, total: int) -> None:
        self.progress.setValue(round(1000 * processed / total) if total else 0)
        self.status.setText(f"Indexing waveform file: {processed:,}/{total:,} bytes...")

    def _index_loaded(self, value: object) -> None:
        if not isinstance(value, WaveformFileIndex):
            self._index_failed("Waveform index worker returned an invalid result")
            return
        self._index = value
        try:
            self._reader = MappedWaveformFile(value)
        except OSError as exc:
            self._index_failed(str(exc))
            return
        self._set_current_file(value.path)
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        for source in value.channels:
            if source.board is None:
                label = f"Channel {source.channel} ({source.frame_count:,} frames)"
            else:
                label = (
                    f"Board {source.board} / Channel {source.channel} "
                    f"({source.frame_count:,} frames)"
                )
            self.source_combo.addItem(label)
        self.source_combo.blockSignals(False)
        period = value.sample_period_ns or 2.0
        self.sample_period.setValue(period)
        self._last_sample_period = period
        self.sample_period.setEnabled(value.sample_period_ns is None)
        self.source_combo.setEnabled(len(value.channels) > 1)
        self.frame_spin.setEnabled(True)
        self.close_file_action.setEnabled(True)
        self.analyze_button.setEnabled(True)
        self.progress.setValue(1000)
        self.status.setText(
            f"Loaded {value.path.name}: {value.frame_count:,} frames, {value.format_name}."
        )
        self._source_changed()

    def _index_failed(self, message: str) -> None:
        self.status.setText(f"Waveform load failed: {message}")
        QMessageBox.critical(self, "Waveform File Load Failed", message)

    def _index_cancelled(self) -> None:
        self.status.setText("Waveform indexing cancelled.")

    def _index_thread_finished(self) -> None:
        thread = self._index_thread
        self._index_worker = None
        self._index_thread = None
        self._set_indexing(False)
        if thread is not None:
            thread.deleteLater()

    def _set_indexing(self, indexing: bool) -> None:
        self.load_action.setEnabled(not indexing)
        if indexing:
            self.close_file_action.setEnabled(False)
        self._update_optimize_button()
        self._update_gate_transfer_action()

    def _source_changed(self) -> None:
        if self._index is None or self.source_combo.currentIndex() < 0:
            return
        self.cluster_selection.invalidate()
        self._last_psd_result = None
        self._last_report_context = None
        self._update_optimize_button()
        source = self._index.channels[self.source_combo.currentIndex()]
        self.frame_spin.blockSignals(True)
        self.frame_spin.setRange(0, min(source.frame_count - 1, _MAX_QT_INDEX))
        self.frame_spin.setValue(0)
        self.frame_spin.blockSignals(False)
        self._show_frame(0, reset_gates=True)

    def _cluster_selection_changed(self, frames: object) -> None:
        self._toggle_cluster_average(False)
        self._selected_frame_indices = frames if isinstance(frames, np.ndarray) else None
        selected = self._selected_frame_indices
        active = selected is not None
        self.frame_spin.setReadOnly(active)
        self.frame_spin.setEnabled(
            self._index is not None and (selected is None or len(selected) > 0),
        )
        if selected is not None and not len(selected):
            self._frame = None
            self.raw_curve.clear()
            self.oriented_curve.clear()
            self.cluster_selection.set_current_frame(-1)
            self.preview_status.setText("No waveforms inside the selected PSD region.")
        elif not active and self._frame is None and self._reader is not None:
            self._show_frame()
        elif not active and self._frame is not None:
            self._preview()
        self._update_export_action()

    def _show_cluster_frame(self, frame: int) -> None:
        x_range = self.waveform_plot.viewRange()[0]
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(frame)
        self.frame_spin.blockSignals(False)
        self._show_frame()
        self.waveform_plot.setXRange(*x_range, padding=0)

    def _toggle_cluster_average(self, checked: bool) -> None:
        button = self.cluster_selection.average_button
        if not checked:
            self._average_frames = None
            self._average = None
            self.average_curve.clear()
            self.average_curve.hide()
            self.waveform_plot.legend.removeItem(self.average_curve)
            if self._average_worker is not None:
                self._average_worker.stop()
            button.setText("Show average waveform")
            return
        frames = self._selected_frame_indices
        context = self._last_report_context
        if (self._index is None or frames is None or not len(frames)
                or context is None or self._average_thread is not None):
            button.setChecked(False)
            return
        worker = WaveformAverageWorker(
            self._index, source_index=context.source_index,
            frames=frames, settings=context.settings,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(self._average_loaded)
        worker.error.connect(self._average_failed)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._average_thread_finished, Qt.ConnectionType.QueuedConnection)
        self._average_frames = frames
        self._average_worker, self._average_thread = worker, thread
        button.setText("Cancel average waveform")
        self.status.setText(f"Averaging {len(frames):,} selected waveforms...")
        thread.start()

    def _average_loaded(self, result: WaveformAverage) -> None:
        if (self._average_frames is None
                or self._average_frames is not self._selected_frame_indices):
            return  # Selection/settings changed or the operation was cancelled.
        self._average = result
        self.waveform_plot.legend.addItem(
            self.average_curve, "Cluster average (baseline-corrected)",
        )
        self.cluster_selection.average_button.setText("Hide average waveform")
        self._preview()
        self.status.setText(
            f"Cluster average: {result.count:,} waveforms, {len(result.samples):,} shared samples; "
            "per-event baseline removed; no peak alignment or amplitude scaling."
        )

    def _average_failed(self, message: str) -> None:
        if self._average_frames is not None:
            self.status.setText(f"Could not average selected waveforms: {message}")
            self.cluster_selection.average_button.setChecked(False)

    def _average_thread_finished(self) -> None:
        thread = self._average_thread
        self._average_worker = None
        self._average_thread = None
        if thread is not None:
            thread.deleteLater()

    def _cluster_context_matches(self) -> bool:
        context = self._last_report_context
        if context is None or context.source_index != self.source_combo.currentIndex():
            return False
        try:
            return (
                context.settings == self._settings()
                and context.sample_period_ns == self.sample_period.value()
                and context.stride == self.event_stride.value() * (
                    _AUTO_PREVIEW_STRIDE_MULTIPLIER if context.analysis_kind == "sparse" else 1
                )
                and context.maximum_events == self.maximum_events.value()
            )
        except ValueError:
            return False

    def _invalidate_cluster_if_stale(self, *_args: object) -> None:
        if self.cluster_selection.has_events and not self._cluster_context_matches():
            self.cluster_selection.invalidate()

    def _show_frame(self, _value: int = 0, *, reset_gates: bool = False) -> None:
        reader = self._reader
        if reader is None or self.source_combo.currentIndex() < 0:
            return
        try:
            self._frame = reader.frame(self.frame_spin.value(), self.source_combo.currentIndex())
        except (IndexError, OSError, ValueError) as exc:
            self.status.setText(f"Could not read waveform: {exc}")
            return
        samples = self._frame.samples
        self.optimization_radius.setMaximum(max(1, len(samples)))
        period = self.sample_period.value()
        times = np.arange(len(samples), dtype=np.float64) * period
        self.raw_curve.setData(times, samples)
        maximum_time = max(period, len(samples) * period)
        previous_update_state = self._updating_gates
        self._updating_gates = True
        try:
            for item in (
                self.baseline_region,
                self.gate_start_line,
                self.short_end_line,
                self.long_end_line,
            ):
                item.setBounds((0.0, maximum_time))
            if reset_gates:
                self._set_default_gates(samples)
        finally:
            self._updating_gates = previous_update_state
        self._preview()
        self.cluster_selection.set_current_frame(self.frame_spin.value())
        self.waveform_plot.autoRange()

    def _set_default_gates(self, samples: np.ndarray) -> None:
        count = len(samples)
        if count < 5:
            return
        probe = max(2, min(64, count // 5))
        baseline = float(np.median(samples[:probe]))
        deviations = np.abs(samples.astype(np.float64) - baseline)
        peak = int(np.argmax(deviations))
        start = max(probe, peak - max(2, count // 100))
        start = min(start, count - 3)
        short_end = min(count - 1, start + max(2, count // 20))
        long_end = min(count, max(short_end + 1, start + max(4, count // 4)))
        baseline_end = max(1, min(start, peak - max(3, count // 50)))
        baseline_start = max(0, baseline_end - probe)
        period = self.sample_period.value()
        previous_update_state = self._updating_gates
        self._updating_gates = True
        try:
            self.baseline_region.setRegion((baseline_start * period, baseline_end * period))
            self.gate_start_line.setPos(start * period)
            self.short_end_line.setPos(short_end * period)
            self.long_end_line.setPos(long_end * period)
        finally:
            self._updating_gates = previous_update_state

    def _sample_period_changed(self) -> None:
        new_period = self.sample_period.value()
        old_period = self._last_sample_period
        if old_period > 0 and new_period != old_period:
            scale = new_period / old_period
            previous_update_state = self._updating_gates
            self._updating_gates = True
            try:
                low, high = self.baseline_region.getRegion()
                self.baseline_region.setRegion((low * scale, high * scale))
                for line in (
                    self.gate_start_line,
                    self.short_end_line,
                    self.long_end_line,
                ):
                    line.setPos(line.value() * scale)
            finally:
                self._updating_gates = previous_update_state
        self._last_sample_period = new_period
        self._show_frame(self.frame_spin.value())

    def _polarity(self) -> Literal[-1, 1]:
        selected = int(self.polarity_combo.currentData())
        if selected in {-1, 1}:
            return cast(Literal[-1, 1], selected)
        # Auto polarity was resolved once for the reconstruction. Cluster
        # navigation must not silently change that generation's interpretation.
        if self._selected_frame_indices is not None and self._last_report_context is not None:
            return self._last_report_context.settings.polarity
        frame = self._frame
        if frame is None:
            return 1
        baseline_start, baseline_end, gate_start, _, _ = self._gate_indices()
        return infer_waveform_polarity(
            frame.samples,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            search_start=gate_start,
        )

    def _gate_indices(self) -> tuple[int, int, int, int, int]:
        period = self.sample_period.value()
        baseline = sorted(round(value / period) for value in self.baseline_region.getRegion())
        return (
            baseline[0],
            baseline[1],
            round(self.gate_start_line.value() / period),
            round(self.short_end_line.value() / period),
            round(self.long_end_line.value() / period),
        )

    def _settings(self) -> WaveformPsdSettings:
        baseline_start, baseline_end, gate_start, short_end, long_end = self._gate_indices()
        method = cast(BaselineMethod, str(self.baseline_method.currentData()))
        return WaveformPsdSettings(
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            gate_start=gate_start,
            short_end=short_end,
            long_end=long_end,
            polarity=self._polarity(),
            baseline_method=method,
            energy_bins=self.energy_bins.value(),
            energy_range=(0.0, self.energy_max.value()),
            ratio_bins=self.ratio_bins.value(),
            ratio_range=(self.ratio_min.value(), self.ratio_max.value()),
        )

    def _preview(self) -> None:
        if self._updating_gates:
            return
        self._update_gate_transfer_action()
        self._invalidate_cluster_if_stale()
        if self._frame is None:
            return
        try:
            settings = self._settings()
            charge = integrate_waveform(self._frame.samples, settings)
        except ValueError as exc:
            self.preview_status.setText(f"Gate configuration is invalid: {exc}")
            self.oriented_curve.clear()
            return
        if charge is None:
            self.preview_status.setText("Displayed waveform is shorter than the long gate.")
            self.oriented_curve.clear()
            return
        samples = self._frame.samples.astype(np.float64)
        oriented = charge.baseline + settings.polarity * (samples - charge.baseline)
        times = np.arange(len(samples), dtype=np.float64) * self.sample_period.value()
        self.oriented_curve.setData(times, oriented)
        if self._average is not None:
            average_times = np.arange(len(self._average.samples)) * self.sample_period.value()
            # Match the normalized individual pulse's vertical reference without
            # including pedestal drift in the cluster's mean pulse shape.
            self.average_curve.setData(average_times, charge.baseline + self._average.samples)
            self.average_curve.show()
        stored = ""
        if self._frame.long_gate is not None and self._frame.short_gate is not None:
            stored = (
                f"; stored Qlong={self._frame.long_gate:,}, "
                f"Qshort={self._frame.short_gate:,}"
            )
        self.preview_status.setText(
            f"Baseline={charge.baseline:.2f}, RMS={charge.baseline_rms:.2f}; "
            f"Qshort={charge.short_charge:,.1f}, Qlong={charge.long_charge:,.1f}, "
            f"PSD={charge.ratio:.5f}{stored}"
        )

    def _on_gate_moved(self) -> None:
        if self._updating_gates:
            return
        self._preview()
        self._update_optimize_button()
        if self.auto_recalculate.isChecked() and self._index is not None:
            self._auto_preview_timer.start()

    def _on_gate_change_finished(self) -> None:
        if self._updating_gates:
            return
        self._preview()
        if self.auto_recalculate.isChecked() and self._index is not None:
            self._auto_preview_timer.stop()
            self._request_analysis("full")

    def _request_sparse_analysis(self) -> None:
        if self.auto_recalculate.isChecked():
            self._request_analysis("sparse")

    def _start_analysis(self) -> None:
        self._request_analysis("manual")

    def _start_fom_optimization(self) -> None:
        index = self._index
        frame = self._frame
        report_context = self._last_report_context
        if index is None or frame is None or report_context is None:
            return
        if self._analysis_thread is not None or self._optimization_thread is not None:
            QMessageBox.warning(
                self,
                "FOM Gate Optimization",
                "Cancel the current operation before starting gate optimization.",
            )
            return
        try:
            settings = self._settings()
            if (
                report_context.settings != settings
                or report_context.source_index != self.source_combo.currentIndex()
            ):
                raise ValueError(
                    "Reconstruct the PSD with the current source and gate settings first"
                )
            fit_widget = self.psd_plot.fom_widget
            fit_roi_values = sorted(map(float, fit_widget.fit_region.getRegion()))
            centroid_seeds = (
                float(fit_widget.centroid_lines[0].value()),
                float(fit_widget.centroid_lines[1].value()),
            )
            fit = fit_widget.last_fit
            if fit is None:
                raise ValueError("Complete a valid two-population FOM fit first")
            if fit.poor_convergence:
                raise ValueError(
                    "The current FOM fit has convergence warnings; adjust the fit ROI, "
                    "centroid seeds, or event count before optimizing"
                )
            radius = self.optimization_radius.value()
            sample_count = len(frame.samples)
            short_bounds = (
                max(settings.gate_start + 1, settings.short_end - radius),
                min(sample_count - 1, settings.short_end + radius),
            )
            long_bounds = (
                max(settings.gate_start + 2, settings.long_end - radius),
                min(sample_count, settings.long_end + radius),
            )
            initial_step = 1 << (radius.bit_length() - 1)
            energy_roi_values = sorted(
                map(float, self.psd_plot.energy_region.getRegion())
            )
            request = WaveformFomOptimizationRequest(
                settings=settings,
                energy_roi=(energy_roi_values[0], energy_roi_values[1]),
                fit_roi=(fit_roi_values[0], fit_roi_values[1]),
                centroid_seeds=centroid_seeds,
                short_bounds=short_bounds,
                long_bounds=long_bounds,
                initial_step=initial_step,
                search_mode="exhaustive" if self.exhaustive_search.isChecked() else "local",
            )
            if request.search_mode == "exhaustive":
                answer = QMessageBox.warning(
                    self,
                    "Exhaustive FOM Gate Search",
                    f"This will test all {request.total_gate_pairs:,} valid short/long "
                    f"gate pairs within the ±{radius}-sample search radius, at one-sample "
                    f"steps, using up to {self.optimization_events.value():,} waveforms.\n\n"
                    "A double-Gaussian fit is attempted for each pair. This can take "
                    "a long time, especially with a large radius or waveform sample. "
                    "You can cancel between candidate fits.\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            worker = WaveformFomOptimizationWorker(
                index,
                source_index=self.source_combo.currentIndex(),
                request=request,
                stride=self.event_stride.value(),
                maximum_events=self.optimization_events.value(),
            )
        except ValueError as exc:
            self.status.setText(f"FOM gate optimization unavailable: {exc}.")
            QMessageBox.warning(self, "FOM Gate Optimization", str(exc))
            return

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status.connect(self._optimization_status)
        worker.progress.connect(self._optimization_progress)
        worker.loaded.connect(self._optimization_loaded)
        worker.cancelled.connect(self._optimization_cancelled)
        worker.error.connect(self._optimization_failed)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            self._optimization_thread_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._optimization_worker = worker
        self._optimization_total_pairs = (
            request.total_gate_pairs if request.search_mode == "exhaustive" else None
        )
        self._optimization_thread = thread
        self._active_optimization_context = _FomOptimizationContext(
            fit_roi=request.fit_roi,
            centroid_seeds=request.centroid_seeds,
        )
        self._completed_optimization = None
        self._set_optimization_busy(True)
        self.progress.setRange(0, 0)
        self.status.setText("Preparing waveform sample for FOM gate optimization...")
        thread.start()

    def _optimization_status(self, message: str) -> None:
        self.status.setText(message)

    def _optimization_progress(
        self,
        evaluated: int,
        valid: int,
        best_fom: float | None,
        short_end: int | None,
        long_end: int | None,
    ) -> None:
        total = self._optimization_total_pairs
        if total is not None:
            self.progress.setRange(0, 1000)
            self.progress.setValue(min(1000, evaluated * 1000 // total))
        search_label = "Exhaustive FOM gate search" if total is not None else "FOM gate search"
        evaluated_text = f"{evaluated:,}/{total:,}" if total is not None else str(evaluated)
        if best_fom is None:
            self.status.setText(
                f"{search_label}: {evaluated_text} candidates evaluated; "
                "no trustworthy fit yet."
            )
            return
        self.status.setText(
            f"{search_label}: {evaluated_text} evaluated, {valid} valid; "
            f"best FOM {best_fom:.5g} at short={short_end}, "
            f"long={long_end} samples."
        )

    def _optimization_loaded(self, value: object) -> None:
        context = self._active_optimization_context
        if not isinstance(value, WaveformFomOptimizationResult) or context is None:
            self._optimization_failed("FOM optimizer returned an invalid result")
            return
        self._completed_optimization = _FomOptimizationContext(
            fit_roi=context.fit_roi,
            centroid_seeds=context.centroid_seeds,
            result=value,
        )

    def _optimization_failed(self, message: str) -> None:
        self._completed_optimization = None
        self.status.setText(f"FOM gate optimization failed: {message}")
        QMessageBox.critical(self, "FOM Gate Optimization Failed", message)

    def _optimization_cancelled(self) -> None:
        self._completed_optimization = None
        self.status.setText("FOM gate optimization cancelled.")

    def _optimization_thread_finished(self) -> None:
        thread = self._optimization_thread
        completed = self._completed_optimization
        self._optimization_worker = None
        self._optimization_thread = None
        self._optimization_total_pairs = None
        self._active_optimization_context = None
        self._completed_optimization = None
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self._set_optimization_busy(False)
        if thread is not None:
            thread.deleteLater()
        if completed is not None and completed.result is not None:
            QTimer.singleShot(0, lambda: self._apply_optimized_gates(completed))

    def _apply_optimized_gates(self, context: _FomOptimizationContext) -> None:
        result = context.result
        if result is None:
            return
        period = self.sample_period.value()
        previous_update_state = self._updating_gates
        self._updating_gates = True
        try:
            self.short_end_line.setPos(result.best.short_end * period)
            self.long_end_line.setPos(result.best.long_end * period)
        finally:
            self._updating_gates = previous_update_state
        self._preview()
        self._pending_optimized_reconstruction = context
        self.status.setText(
            f"Best sampled FOM {result.best.fom:.5g} at short={result.best.short_end}, "
            f"long={result.best.long_end}; rebuilding the full PSD..."
        )
        self._request_analysis("optimized")

    def _request_analysis(self, kind: AnalysisKind) -> None:
        if self._index is None:
            return
        self._pending_analysis_kind = kind
        if self._analysis_thread is not None:
            if self._analysis_worker is not None:
                self._analysis_worker.stop()
            self.status.setText("Updating PSD calculation for the latest gate positions...")
            return
        self._launch_pending_analysis()

    def _launch_pending_analysis(self) -> None:
        kind = self._pending_analysis_kind
        self._pending_analysis_kind = None
        if kind is None or self._index is None:
            return
        try:
            settings = self._settings()
            stride_multiplier = (
                _AUTO_PREVIEW_STRIDE_MULTIPLIER if kind == "sparse" else 1
            )
            stride = self.event_stride.value() * stride_multiplier
            maximum_events = self.maximum_events.value()
            source_index = self.source_combo.currentIndex()
            worker = WaveformPsdWorker(
                self._index,
                source_index=source_index,
                settings=settings,
                stride=stride,
                maximum_events=maximum_events,
            )
        except ValueError as exc:
            self.status.setText(f"Invalid PSD settings: {exc}")
            if kind == "manual":
                QMessageBox.critical(self, "Invalid PSD Settings", str(exc))
            return
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._analysis_progress)
        worker.loaded.connect(self._analysis_loaded)
        worker.cancelled.connect(self._analysis_cancelled)
        worker.error.connect(self._analysis_failed)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            self._analysis_thread_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._analysis_worker = worker
        self._analysis_thread = thread
        self._active_analysis_kind = kind
        self.cluster_selection.invalidate()
        self._active_report_context = _WaveformAnalysisContext(
            settings=settings,
            source_index=source_index,
            stride=stride,
            maximum_events=maximum_events,
            analysis_kind=kind,
            sample_period_ns=self.sample_period.value(),
            optimization=(
                self._pending_optimized_reconstruction.result
                if kind == "optimized" and self._pending_optimized_reconstruction is not None
                else None
            ),
        )
        self._set_analysis_busy(True)
        self.progress.setValue(0)
        if kind == "sparse":
            self.status.setText("Updating sparse PSD preview (every 20th selected event)...")
        elif kind == "optimized":
            self.status.setText("Reconstructing the full PSD with optimized gates...")
        else:
            self.status.setText("Reconstructing full PSD from waveforms...")
        thread.start()

    def _analysis_progress(self, processed: int, total: int) -> None:
        self.progress.setValue(round(1000 * processed / total) if total else 0)
        if self._active_analysis_kind == "sparse":
            prefix = "Sparse preview"
        elif self._active_analysis_kind == "optimized":
            prefix = "Optimized PSD"
        else:
            prefix = "Full PSD"
        self.status.setText(f"{prefix}: integrating {processed:,}/{total:,} waveforms...")

    def _analysis_loaded(self, value: object) -> None:
        if not isinstance(value, WaveformPsdResult):
            self._analysis_failed("Waveform worker returned an invalid result")
            return
        if self._pending_analysis_kind is not None:
            return
        self.psd_plot.set_data(
            value.matrix,
            energy_range=value.energy_range,
            ratio_range=value.ratio_range,
        )
        self._last_psd_result = value
        self._last_report_context = self._active_report_context
        if self._cluster_context_matches():
            self.cluster_selection.set_events(
                value.event_frames, value.event_energy, value.event_ratio,
            )
            self.cluster_selection.set_current_frame(self.frame_spin.value())
        stats = value.statistics
        comparison = ""
        if len(value.stored_long):
            nonzero = value.stored_long > 0
            if np.any(nonzero):
                scale = np.median(
                    value.calculated_long[nonzero] / value.stored_long[nonzero]
                )
                comparison = (
                    f" Stored/computed comparison: median Qlong scale={scale:.4g}."
                )
        self.progress.setValue(1000)
        summary = (
            f"accepted {stats.accepted:,}/{stats.received:,}; too short "
            f"{stats.too_short:,}; nonpositive Qlong {stats.nonpositive_long:,}; outside "
            f"view {stats.outside_range:,}.{comparison}"
        )
        optimization = (
            self._pending_optimized_reconstruction
            if self._active_analysis_kind == "optimized"
            else None
        )
        if optimization is not None and optimization.result is not None:
            fit_widget = self.psd_plot.fom_widget
            fit_widget.fit_region.setRegion(optimization.fit_roi)
            for line, seed in zip(
                fit_widget.centroid_lines,
                optimization.centroid_seeds,
                strict=True,
            ):
                line.setPos(seed)
            final_fit = fit_widget.fit_current_projection()
            result = optimization.result
            self._pending_optimized_reconstruction = None
            if final_fit is None:
                fit_text = "the full-data confirmation fit was unavailable"
            elif final_fit.poor_convergence:
                fit_text = f"full-data FOM {final_fit.fom:.5g} has convergence warnings"
            else:
                fit_text = f"confirmed full-data FOM {final_fit.fom:.5g}"
            self.status.setText(
                f"FOM gate optimization complete: short={result.best.short_end}, "
                f"long={result.best.long_end} samples; sampled FOM "
                f"{result.best.fom:.5g}, {fit_text}; {summary}"
            )
        elif self._active_analysis_kind == "sparse":
            self.status.setText(
                f"Sparse PSD preview complete: {summary} Release a gate for a full update."
            )
        else:
            self.status.setText(f"PSD complete: {summary}")

    def _analysis_failed(self, message: str) -> None:
        if self._active_analysis_kind == "optimized":
            self._pending_optimized_reconstruction = None
        self.status.setText(f"PSD reconstruction failed: {message}")
        QMessageBox.critical(self, "PSD Reconstruction Failed", message)

    def _analysis_cancelled(self) -> None:
        if self._active_analysis_kind == "optimized":
            self._pending_optimized_reconstruction = None
        if self._pending_analysis_kind is None:
            self.status.setText("PSD reconstruction cancelled.")

    def _cancel_analysis(self) -> None:
        self._auto_preview_timer.stop()
        self._pending_analysis_kind = None
        if self._optimization_worker is not None:
            self._optimization_worker.stop()
            self.status.setText("Cancelling FOM gate optimization...")
        elif self._analysis_worker is not None:
            self._analysis_worker.stop()
            self.status.setText("Cancelling PSD reconstruction...")

    def _analysis_thread_finished(self) -> None:
        thread = self._analysis_thread
        self._analysis_worker = None
        self._analysis_thread = None
        self._active_analysis_kind = None
        self._active_report_context = None
        self._set_analysis_busy(False)
        if thread is not None:
            thread.deleteLater()
        if self._pending_analysis_kind is not None and self._index is not None:
            QTimer.singleShot(0, self._launch_pending_analysis)

    def _set_analysis_busy(self, busy: bool) -> None:
        self.cluster_selection.set_busy(busy)
        self.analyze_button.setEnabled(not busy and self._index is not None)
        self.cancel_button.setEnabled(busy)
        self.load_action.setEnabled(not busy and self._index_thread is None)
        self.close_file_action.setEnabled(not busy and self._index is not None)
        self.source_combo.setEnabled(
            not busy and self._index is not None and len(self._index.channels) > 1
        )
        self._update_report_button()
        self._update_optimize_button()

    def _set_optimization_busy(self, busy: bool) -> None:
        if busy:
            self.cluster_selection.invalidate()
        self.cluster_selection.set_busy(busy)
        available = not busy and self._index is not None
        self.load_action.setEnabled(not busy and self._index_thread is None)
        self.close_file_action.setEnabled(available)
        self.analyze_button.setEnabled(available)
        self.cancel_button.setEnabled(busy)
        self.frame_spin.setEnabled(available)
        self.source_combo.setEnabled(
            available and self._index is not None and len(self._index.channels) > 1
        )
        self.sample_period.setEnabled(
            available and self._index is not None and self._index.sample_period_ns is None
        )
        for control in (
            self.baseline_method,
            self.polarity_combo,
            self.energy_bins,
            self.energy_max,
            self.ratio_bins,
            self.ratio_min,
            self.ratio_max,
            self.event_stride,
            self.maximum_events,
            self.optimization_events,
            self.optimization_radius,
            self.exhaustive_search,
            self.auto_recalculate,
        ):
            control.setEnabled(not busy)
        for item in (
            self.baseline_region,
            self.gate_start_line,
            self.short_end_line,
            self.long_end_line,
            self.psd_plot.energy_region,
            self.psd_plot.fom_widget.fit_region,
            *self.psd_plot.fom_widget.centroid_lines,
        ):
            item.setMovable(not busy)
        self.psd_plot.fom_widget.fit_button.setEnabled(not busy)
        self.psd_plot.fom_widget.guides_checkbox.setEnabled(not busy)
        self._update_report_button()
        self._update_optimize_button()

    def _fit_changed(self, _fit: object) -> None:
        self._update_report_button()
        self._update_optimize_button()

    def _update_optimize_button(self) -> None:
        fit = self.psd_plot.fom_widget.last_fit
        context = self._last_report_context
        context_matches = False
        if context is not None and context.source_index == self.source_combo.currentIndex():
            try:
                context_matches = context.settings == self._settings()
            except ValueError:
                pass
        self.optimize_button.setEnabled(
            self._analysis_thread is None
            and self._optimization_thread is None
            and self._index_thread is None
            and self._index is not None
            and context_matches
            and fit is not None
            and not fit.poor_convergence
        )

    def _update_gate_transfer_action(self) -> None:
        self.transfer_gates_action.setEnabled(
            self._index is not None and self._frame is not None
            and self._analysis_thread is None and self._optimization_thread is None
            and self._index_thread is None and self._cluster_export_dialog is None
            and self._pending_analysis_kind is None
        )

    def _update_report_button(self) -> None:
        self._update_gate_transfer_action()
        self.report_button.setEnabled(
            self._analysis_thread is None
            and self._optimization_thread is None
            and self._index_thread is None
            and self._index is not None
            and self._last_psd_result is not None
            and self._last_report_context is not None
            and self.psd_plot.fom_widget.last_fit is not None
        )

    def _download_report(self) -> None:
        index = self._index
        result = self._last_psd_result
        context = self._last_report_context
        fit = self.psd_plot.fom_widget.last_fit
        if index is None or result is None or context is None or fit is None:
            QMessageBox.warning(
                self,
                "Waveform PSD Report",
                "Reconstruct the PSD and complete the FOM fit before creating a report.",
            )
            return
        if not 0 <= context.source_index < len(index.channels):
            QMessageBox.critical(
                self,
                "Waveform PSD Report Failed",
                "The analyzed waveform source is no longer available.",
            )
            return
        default_folder = str(
            QSettings().value("analysis/waveform_report_folder", str(index.path.parent))
        )
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Waveform PSD Report Destination",
            default_folder,
        )
        if not selected:
            return
        source = index.channels[context.source_index]
        settings = context.settings
        state = self.psd_plot.report_state()
        processing_settings: dict[str, object] = {
            "baseline estimator": settings.baseline_method,
            "baseline samples": (settings.baseline_start, settings.baseline_end),
            "integration start sample": settings.gate_start,
            "short-gate end sample": settings.short_end,
            "long-gate end sample": settings.long_end,
            "pulse polarity": settings.polarity,
            "sample period ns": context.sample_period_ns,
            "baseline time ns": (
                settings.baseline_start * context.sample_period_ns,
                settings.baseline_end * context.sample_period_ns,
            ),
            "integration start time ns": settings.gate_start * context.sample_period_ns,
            "short-gate end time ns": settings.short_end * context.sample_period_ns,
            "long-gate end time ns": settings.long_end * context.sample_period_ns,
            "event stride used": context.stride,
            "maximum events": (context.maximum_events if context.maximum_events else "All"),
            "analysis mode": context.analysis_kind,
            "PSD ratio definition": "(Qlong - Qshort) / Qlong",
        }
        if context.optimization is not None:
            optimization = context.optimization
            processing_settings.update(
                {
                    "optimization search mode": optimization.search_mode,
                    "optimization sampled FOM": optimization.best.fom,
                    "optimization sampled waveforms": optimization.sampled_waveforms,
                    "optimization candidates evaluated": optimization.evaluated_candidates,
                    "optimization valid candidates": optimization.valid_candidates,
                }
            )
        request = PsdReportRequest(
            source_path=index.path,
            title="Waveform PSD/FOM Analysis Report",
            matrix=state.matrix,
            energy_range=state.energy_range,
            ratio_range=state.ratio_range,
            energy_label=state.energy_label,
            cut=state.cut,
            energy_roi=state.energy_roi,
            matrix_scale=state.matrix_scale,
            energy_log_y=state.energy_log_y,
            fit=fit,
            settings=processing_settings,
            digitizer_info={
                **capture_digitizer_info(index.path),
                "source format": index.format_name,
                "board": source.board if source.board is not None else "Not embedded",
                "channel": source.channel,
                "source frames": source.frame_count,
                "sample period ns": context.sample_period_ns,
            },
            statistics={
                "received waveforms": result.statistics.received,
                "accepted waveforms": result.statistics.accepted,
                "too-short waveforms": result.statistics.too_short,
                "nonpositive Qlong": result.statistics.nonpositive_long,
                "outside-view waveforms": result.statistics.outside_range,
            },
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            bundle = export_psd_report(request, Path(selected))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            log.exception("Could not create waveform PSD report")
            QMessageBox.critical(self, "Waveform PSD Report Failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        QSettings().setValue("analysis/waveform_report_folder", selected)
        self.status.setText(f"Waveform PSD report saved to {bundle.directory}.")
        QMessageBox.information(
            self,
            "Waveform PSD Report Saved",
            f"Report and data files were saved to:\n{bundle.directory}",
        )

    def close_file(self) -> None:
        if not self._stop_cluster_export_sync():
            return
        self._stop_average_sync()
        self._auto_preview_timer.stop()
        self._pending_analysis_kind = None
        self._stop_analysis_sync()
        self._stop_optimization_sync()
        self._stop_index_sync()
        self.cluster_selection.invalidate()
        if self._reader is not None:
            self._reader.close()
        self._reader = None
        self._index = None
        self._frame = None
        self._active_report_context = None
        self._last_report_context = None
        self._last_psd_result = None
        self._active_optimization_context = None
        self._completed_optimization = None
        self._pending_optimized_reconstruction = None
        self._set_current_file(None)
        self.source_combo.clear()
        self.source_combo.setEnabled(False)
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setEnabled(False)
        self.sample_period.setEnabled(False)
        self.close_file_action.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.raw_curve.clear()
        self.oriented_curve.clear()
        self.preview_status.setText("No frame loaded.")
        self.psd_plot.fom_widget.clear_fit()
        self._update_report_button()
        self._update_optimize_button()

    def _set_current_file(self, path: Path | None) -> None:
        self.setWindowFilePath(str(path) if path is not None else "")
        filename = path.name if path is not None else "No file loaded"
        self.setWindowTitle(f"{_WINDOW_TITLE} — {filename}")

    def _stop_index_sync(self) -> None:
        if self._index_worker is not None:
            self._index_worker.stop()
        if self._index_thread is not None and self._index_thread.isRunning():
            if not self._index_thread.wait(5_000):
                log.warning("Waveform index worker did not stop within 5 seconds")

    def _stop_average_sync(self) -> None:
        self.cluster_selection.average_button.setChecked(False)
        self._toggle_cluster_average(False)
        if self._average_thread is not None and self._average_thread.isRunning():
            if not self._average_thread.wait(5_000):
                log.warning("Waveform average worker did not stop within 5 seconds")

    def _stop_cluster_export_sync(self) -> bool:
        dialog = self._cluster_export_dialog
        if dialog is not None and not dialog.stop_sync():
            log.warning("Cluster export is still stopping; keeping its parent open")
            self.status.setText("Waiting for cluster export cancellation and cleanup...")
            return False
        return True

    def _stop_analysis_sync(self) -> None:
        self._auto_preview_timer.stop()
        self._pending_analysis_kind = None
        if self._analysis_worker is not None:
            self._analysis_worker.stop()
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            if not self._analysis_thread.wait(5_000):
                log.warning("Waveform PSD worker did not stop within 5 seconds")

    def _stop_optimization_sync(self) -> None:
        if self._optimization_worker is not None:
            self._optimization_worker.stop()
        if self._optimization_thread is not None and self._optimization_thread.isRunning():
            if not self._optimization_thread.wait(5_000):
                log.warning("Waveform FOM optimization worker did not stop within 5 seconds")
        self._completed_optimization = None
        self._pending_optimized_reconstruction = None

    def show_workspace(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def close_without_prompt(self) -> None:
        if not self._stop_cluster_export_sync():
            return
        self._stop_average_sync()
        self._stop_analysis_sync()
        self._stop_optimization_sync()
        self._stop_index_sync()
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self.close()

    def reject(self) -> None:
        if not self._stop_cluster_export_sync():
            return
        self.close_without_prompt()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if not self._stop_cluster_export_sync():
            event.ignore()
            return
        self._stop_average_sync()
        self._stop_analysis_sync()
        self._stop_optimization_sync()
        self._stop_index_sync()
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        super().closeEvent(event)
