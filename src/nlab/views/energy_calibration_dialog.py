"""Modeless multi-spectrum MCA energy-calibration workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.energy_calibration import (
    CalibrationModel,
    CalibrationPeakFit,
    CalibrationPoint,
    EnergyCalibration,
    FingerprintValue,
    SpectrumSnapshot,
    fit_calibration_peak,
    fit_energy_calibration,
)
from nlab.analysis.mca_report import (
    EnergyCalibrationReportPoint,
    EnergyCalibrationReportRequest,
    export_energy_calibration_report,
)
from nlab.views.plot_viewbox import ModifierZoomViewBox
from nlab.views.tooltips import wrap_widget_tooltips
from nlab.views.workbench_menu import WorkbenchMenuBar

if TYPE_CHECKING:
    from nlab.controllers.mca_controller import MCAController

_COL_POINT = 0
_COL_LABEL = 1
_COL_SOURCE = 2
_COL_SEED = 3
_COL_CENTROID = 4
_COL_ENERGY = 5
_COL_RESIDUAL = 6
_COL_ENABLED = 7
_OVERLAY_COLORS = ("#1f77b4", "#d28b38", "#648b71", "#7353a6", "#a66f6f")
_FIT_COLORS = ("#2a9d8f", "#7c3aed", "#e76f51", "#0891b2", "#ca8a04")


@dataclass
class _DraftPoint:
    point_id: str
    seed_channel: float
    channel: float
    energy_kev: float | None
    label: str
    source: str
    fingerprint: dict[str, FingerprintValue]
    snapshot_id: str | None = None
    peak_fit: CalibrationPeakFit | None = None
    enabled: bool = True
    line: pg.InfiniteLine | None = None
    centroid_line: pg.InfiniteLine | None = None
    fit_curve: pg.PlotDataItem | None = None
    residual_curve: pg.PlotDataItem | None = None


@dataclass
class _Workspace:
    snapshots: list[SpectrumSnapshot] = field(default_factory=list)
    points: list[_DraftPoint] = field(default_factory=list)
    model: CalibrationModel = "linear"
    fit: EnergyCalibration | None = None
    dirty: bool = False
    show_peak_fits: bool = False


class EnergyCalibrationDialog(QDialog):
    """Fit one active channel-to-keV mapping from one or more spectra."""

    def __init__(
        self,
        controllers: list[MCAController],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("MCA Energy Calibration")
        self.resize(1320, 680)
        self.setMinimumSize(1050, 600)
        self.setModal(False)
        self._controllers = {controller.channel: controller for controller in controllers}
        self._workspaces: dict[int, _Workspace] = {}
        self._building_table = False
        self._allow_close = False
        self._build_ui()
        for channel in sorted(self._controllers):
            controller = self._controllers[channel]
            label = getattr(controller, "channel_label", f"CH{channel}")
            self.channel_combo.addItem(f"MCA {label}", channel)
            self._workspaces[channel] = self._workspace_from_applied(
                self._controllers[channel].energy_calibration
            )
        self._connect_signals()
        self._show_channel()
        if self._controllers:
            self._load_current(replace=True, show_error=False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.menu_bar = WorkbenchMenuBar(self)
        root.setMenuBar(self.menu_bar)
        self.file_menu = self.menu_bar.menu("&File")
        self.analysis_menu = self.menu_bar.menu("&Analysis")
        self.view_menu = self.menu_bar.menu("&View")
        self.menu_bar.add_guide("energy-calibration")
        instructions = QLabel(
            "Copy one or more spectra measured with the same MCA energy settings. "
            "Double-click a peak to add a draggable reference line, then enter its "
            "known energy. Applying calibration changes presentation and metadata only; "
            "raw histogram channels remain unchanged.",
            self,
        )
        instructions.setWordWrap(True)
        root.addWidget(instructions)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Calibrate:", self))
        self.channel_combo = QComboBox(self)
        source_row.addWidget(self.channel_combo)
        self.replace_action = self.menu_bar.command(self.file_menu, "Replace with current spectrum")
        self.overlay_action = self.menu_bar.command(self.file_menu, "Add current as overlay")
        self.file_menu.addSeparator()
        source_row.addStretch(1)
        root.addLayout(source_row)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.plot_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.plot = pg.PlotWidget(self.plot_splitter, viewBox=ModifierZoomViewBox())
        self.plot.setMinimumHeight(260)
        self.plot.setBackground("#f8f9fa")
        self.plot.setLabel("bottom", "MCA channel")
        self.plot.setLabel("left", "Counts")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.legend = self.plot.addLegend()
        self.plot_splitter.addWidget(self.plot)

        self.residual_plot = pg.PlotWidget(
            self.plot_splitter,
            viewBox=ModifierZoomViewBox(),
        )
        self.residual_plot.setMinimumHeight(120)
        self.residual_plot.setMaximumHeight(220)
        self.residual_plot.setBackground("#f8f9fa")
        self.residual_plot.setLabel("left", "Fit residual", units="counts")
        self.residual_plot.setLabel("bottom", "MCA channel")
        self.residual_plot.showGrid(x=True, y=True, alpha=0.2)
        self.residual_plot.getPlotItem().setXLink(self.plot.getPlotItem())
        self.residual_legend = self.residual_plot.addLegend()
        self._reset_residual_plot()
        self.plot_splitter.addWidget(self.residual_plot)
        self.plot_splitter.setStretchFactor(0, 4)
        self.plot_splitter.setStretchFactor(1, 1)
        self.plot_splitter.setSizes([480, 150])
        splitter.addWidget(self.plot_splitter)

        side = QWidget(self)
        side.setMinimumWidth(400)
        side.setMaximumWidth(500)
        side_layout = QVBoxLayout(side)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Fit model:", self))
        self.model_combo = QComboBox(self)
        self.model_combo.addItem("Linear", "linear")
        self.model_combo.addItem("Quadratic", "quadratic")
        model_row.addWidget(self.model_combo)
        model_row.addWidget(QLabel("Centroid half-width:", self))
        self.centroid_half_width = QDoubleSpinBox(self)
        self.centroid_half_width.setRange(6.0, 500.0)
        self.centroid_half_width.setDecimals(1)
        self.centroid_half_width.setValue(300.0)
        self.centroid_half_width.setSuffix(" ch")
        model_row.addWidget(self.centroid_half_width)
        model_row.addStretch(1)
        side_layout.addLayout(model_row)

        self.table = QTableWidget(0, 8, self)
        self.table.setHorizontalHeaderLabels(
            [
                "Point",
                "Label",
                "Spectrum",
                "Seed ch.",
                "Centroid ch.",
                "Energy [keV]",
                "Residual",
                "Use",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        for column in range(self.table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(_COL_SOURCE, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(_COL_POINT, 32)
        self.table.setColumnWidth(_COL_LABEL, 36)
        self.table.setColumnWidth(_COL_SEED, 50)
        self.table.setColumnWidth(_COL_CENTROID, 60)
        self.table.setColumnWidth(_COL_ENERGY, 60)
        self.table.setColumnWidth(_COL_RESIDUAL, 48)
        self.table.setColumnWidth(_COL_ENABLED, 28)
        side_layout.addWidget(self.table, 1)

        self.add_action = self.menu_bar.command(self.analysis_menu, "Add point")
        self.remove_action = self.menu_bar.command(self.analysis_menu, "Remove selected points")
        self.clear_points_action = self.menu_bar.command(self.analysis_menu, "Clear points")
        self.analysis_menu.addSeparator()

        self.fit_status = QLabel("Add at least two calibration points.", self)
        self.fit_status.setWordWrap(True)
        self.fit_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        side_layout.addWidget(self.fit_status)
        splitter.addWidget(side)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([820, 470])
        root.addWidget(splitter, 1)

        self.snapshot_status = QLabel("No spectrum copied.", self)
        self.snapshot_status.setWordWrap(True)
        root.addWidget(self.snapshot_status)
        self.apply_action = self.menu_bar.command(
            self.analysis_menu, "Apply calibration", enabled=False,
        )
        self.clear_calibration_action = self.menu_bar.command(
            self.analysis_menu, "Clear applied calibration",
        )
        self.report_action = self.menu_bar.command(
            self.file_menu, "Download report...", enabled=False, shortcut="Ctrl+S",
        )
        self.file_menu.addSeparator()
        self.close_action = self.menu_bar.command(
            self.file_menu, "Close workspace", shortcut="Ctrl+W",
        )
        self.reset_zoom_action = self.menu_bar.command(self.view_menu, "Reset plot zoom")
        self.reset_zoom_action.triggered.connect(self._reset_plot_zoom)
        wrap_widget_tooltips(self)

    def _reset_plot_zoom(self) -> None:
        self.plot.autoRange()
        self.residual_plot.autoRange()

    def _connect_signals(self) -> None:
        self.channel_combo.currentIndexChanged.connect(self._show_channel)
        self.replace_action.triggered.connect(lambda: self._load_current(replace=True))
        self.overlay_action.triggered.connect(lambda: self._load_current(replace=False))
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.centroid_half_width.valueChanged.connect(self._centroid_width_changed)
        self.add_action.triggered.connect(lambda: self._add_point())
        self.remove_action.triggered.connect(self._remove_selected)
        self.clear_points_action.triggered.connect(self._clear_points)
        self.table.cellChanged.connect(self._table_changed)
        self.apply_action.triggered.connect(self._apply)
        self.clear_calibration_action.triggered.connect(self._clear_applied)
        self.report_action.triggered.connect(self._download_report)
        self.close_action.triggered.connect(self.reject)
        self.plot.scene().sigMouseClicked.connect(self._plot_clicked)

    def _workspace_from_applied(self, calibration: EnergyCalibration | None) -> _Workspace:
        if calibration is None:
            return _Workspace()
        points = [
            _DraftPoint(
                point_id=uuid4().hex,
                seed_channel=point.channel,
                channel=point.channel,
                energy_kev=point.energy_kev,
                label=point.label,
                source=point.source or "Saved calibration",
                fingerprint=dict(calibration.fingerprint),
                enabled=point.enabled,
            )
            for point in calibration.points
        ]
        return _Workspace(points=points, model=calibration.model, fit=calibration)

    def _channel(self) -> int | None:
        value = self.channel_combo.currentData()
        return int(value) if value is not None else None

    def _workspace(self) -> _Workspace | None:
        channel = self._channel()
        return self._workspaces.get(channel) if channel is not None else None

    def _controller(self) -> MCAController | None:
        channel = self._channel()
        return self._controllers.get(channel) if channel is not None else None

    def _show_channel(self, *_args: object) -> None:
        workspace = self._workspace()
        enabled = workspace is not None
        for action in (self.replace_action, self.overlay_action, self.add_action):
            action.setEnabled(enabled)
        if workspace is None:
            self.plot.clear()
            self._reset_residual_plot()
            self.table.setRowCount(0)
            self.snapshot_status.setText("No MCA channel is available.")
            return
        model_index = self.model_combo.findData(workspace.model)
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(model_index)
        self.model_combo.blockSignals(False)
        self._render_workspace()
        self._refit()

    def _load_current(self, *, replace: bool, show_error: bool = True) -> None:
        controller = self._controller()
        workspace = self._workspace()
        if controller is None or workspace is None:
            return
        try:
            snapshot = controller.spectrum_snapshot()
        except RuntimeError as exc:
            self.snapshot_status.setText(str(exc))
            if show_error:
                QMessageBox.information(self, "Energy Calibration", str(exc))
            return
        if not replace and workspace.snapshots:
            reference = workspace.snapshots[0].fingerprint
            if snapshot.fingerprint != reference:
                QMessageBox.warning(
                    self,
                    "Incompatible Spectrum",
                    "The new spectrum used different MCA energy-processing settings and "
                    "cannot be overlaid with the current calibration spectra.",
                )
                return
        if replace:
            workspace.snapshots = [snapshot]
            for point in workspace.points:
                point.source = snapshot.label
                point.fingerprint = dict(snapshot.fingerprint)
                point.snapshot_id = snapshot.snapshot_id
                point.peak_fit = None
        else:
            workspace.snapshots.append(snapshot)
        workspace.show_peak_fits = False
        self._render_workspace()
        self._refit()
        state = "live, non-atomic snapshot" if snapshot.live else "stopped snapshot"
        self.snapshot_status.setText(
            f"{len(workspace.snapshots)} spectrum/spectra loaded; newest is a {state}, "
            f"elapsed {snapshot.elapsed_s:.1f} s. Existing reference points were preserved."
        )

    def _render_workspace(self) -> None:
        workspace = self._workspace()
        if workspace is None:
            return
        self.plot.clear()
        self.legend.clear()
        self._reset_residual_plot()
        for index, snapshot in enumerate(workspace.snapshots):
            color = _OVERLAY_COLORS[index % len(_OVERLAY_COLORS)]
            x = np.arange(len(snapshot.counts) + 1, dtype=np.float64)
            self.plot.plot(
                x,
                snapshot.counts,
                stepMode="center",
                pen=pg.mkPen(color, width=1.2),
                name=snapshot.label,
            )
        for index, point in enumerate(workspace.points):
            point.fit_curve = None
            point.residual_curve = None
            if workspace.show_peak_fits and point.peak_fit is not None:
                color = _FIT_COLORS[index % len(_FIT_COLORS)]
                point.fit_curve = self.plot.plot(
                    point.peak_fit.x,
                    point.peak_fit.best_fit,
                    pen=pg.mkPen(color, width=2.0),
                    name=f"P{index + 1} Gaussian fit",
                )
                point.residual_curve = self.residual_plot.plot(
                    point.peak_fit.x,
                    point.peak_fit.observed - point.peak_fit.best_fit,
                    pen=pg.mkPen(color, width=1.2),
                    name=f"P{index + 1}",
                )
            line = pg.InfiniteLine(
                pos=point.seed_channel,
                angle=90,
                movable=True,
                pen=pg.mkPen("#a66f6f", width=1.5, style=Qt.PenStyle.DashLine),
                hoverPen=pg.mkPen("#c17c7c", width=2),
                label=f"P{index + 1}",
                labelOpts={"color": "#a66f6f", "position": 0.95},
            )
            line.setBounds((0, 16_383))
            line.sigPositionChanged.connect(
                lambda moved, point_id=point.point_id: self._line_moved(
                    point_id, moved, refit=False
                )
            )
            line.sigPositionChangeFinished.connect(
                lambda moved, point_id=point.point_id: self._line_moved(
                    point_id, moved, refit=True
                )
            )
            self.plot.addItem(line)
            point.line = line
        self._rebuild_table()

    def _reset_residual_plot(self) -> None:
        self.residual_plot.clear()
        self.residual_legend.clear()
        self.residual_plot.addLine(y=0.0, pen=pg.mkPen("#64748b", width=1.0))

    def _plot_clicked(self, event: object) -> None:
        double = getattr(event, "double", None)
        if not callable(double) or not double():
            return
        scene_pos = event.scenePos()  # type: ignore[attr-defined]
        if not self.plot.plotItem.sceneBoundingRect().contains(scene_pos):
            return
        position = self.plot.plotItem.vb.mapSceneToView(scene_pos)
        self._add_point(float(position.x()))

    def _add_point(self, channel: float | None = None) -> None:
        workspace = self._workspace()
        if workspace is None:
            return
        snapshot = workspace.snapshots[-1] if workspace.snapshots else None
        if channel is None:
            if snapshot is not None and len(snapshot.counts):
                channel = float(np.argmax(snapshot.counts))
            else:
                channel = 0.0
        workspace.points.append(
            _DraftPoint(
                point_id=uuid4().hex,
                seed_channel=float(np.clip(channel, 0, 16_383)),
                channel=float(np.clip(channel, 0, 16_383)),
                energy_kev=None,
                label="",
                source=snapshot.label if snapshot is not None else "Manual",
                fingerprint=(
                    dict(snapshot.fingerprint)
                    if snapshot is not None
                    else self._controller().energy_calibration_fingerprint()  # type: ignore[union-attr]
                ),
                snapshot_id=snapshot.snapshot_id if snapshot is not None else None,
            )
        )
        workspace.dirty = True
        workspace.show_peak_fits = False
        self._render_workspace()
        self._refit()

    def _line_moved(
        self,
        point_id: str,
        line: pg.InfiniteLine,
        *,
        refit: bool,
    ) -> None:
        workspace = self._workspace()
        if workspace is None:
            return
        for row, point in enumerate(workspace.points):
            if point.point_id != point_id:
                continue
            point.seed_channel = float(np.clip(line.value(), 0, 16_383))
            point.channel = point.seed_channel
            point.peak_fit = None
            if point.centroid_line is not None:
                try:
                    self.plot.removeItem(point.centroid_line)
                except RuntimeError:
                    pass
                point.centroid_line = None
            if point.fit_curve is not None:
                try:
                    self.plot.removeItem(point.fit_curve)
                except RuntimeError:
                    pass
                point.fit_curve = None
            if point.residual_curve is not None:
                try:
                    self.residual_plot.removeItem(point.residual_curve)
                except RuntimeError:
                    pass
                point.residual_curve = None
            workspace.dirty = True
            workspace.show_peak_fits = False
            item = self.table.item(row, _COL_SEED)
            if item is not None:
                self._building_table = True
                item.setText(f"{point.seed_channel:.3f}")
                centroid_item = self.table.item(row, _COL_CENTROID)
                if centroid_item is not None:
                    centroid_item.setText("")
                self._building_table = False
            if refit:
                self._refit()
            return

    def _rebuild_table(self) -> None:
        workspace = self._workspace()
        if workspace is None:
            return
        self._building_table = True
        self.table.setRowCount(len(workspace.points))
        for row, point in enumerate(workspace.points):
            point_item = QTableWidgetItem(f"P{row + 1}")
            point_item.setFlags(point_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, _COL_POINT, point_item)
            self.table.setItem(row, _COL_LABEL, QTableWidgetItem(point.label))
            source_item = QTableWidgetItem(point.source)
            source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, _COL_SOURCE, source_item)
            self.table.setItem(row, _COL_SEED, QTableWidgetItem(f"{point.seed_channel:.3f}"))
            centroid = QTableWidgetItem(
                "" if point.peak_fit is None else f"{point.channel:.4f}"
            )
            centroid.setFlags(centroid.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, _COL_CENTROID, centroid)
            self.table.setItem(
                row,
                _COL_ENERGY,
                QTableWidgetItem("" if point.energy_kev is None else f"{point.energy_kev:.6g}"),
            )
            residual = QTableWidgetItem("")
            residual.setFlags(residual.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, _COL_RESIDUAL, residual)
            enabled = QTableWidgetItem()
            enabled.setFlags(
                (enabled.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                & ~Qt.ItemFlag.ItemIsEditable
            )
            enabled.setCheckState(
                Qt.CheckState.Checked if point.enabled else Qt.CheckState.Unchecked
            )
            self.table.setItem(row, _COL_ENABLED, enabled)
        self._building_table = False

    def _table_changed(self, row: int, column: int) -> None:
        if self._building_table:
            return
        workspace = self._workspace()
        if workspace is None or not 0 <= row < len(workspace.points):
            return
        point = workspace.points[row]
        item = self.table.item(row, column)
        if item is None:
            return
        try:
            if column == _COL_LABEL:
                point.label = item.text().strip()
            elif column == _COL_SEED:
                point.seed_channel = float(item.text())
                if not 0 <= point.seed_channel <= 16_383:
                    raise ValueError("channel must be within 0-16383")
                point.channel = point.seed_channel
                point.peak_fit = None
                if point.line is not None:
                    point.line.blockSignals(True)
                    point.line.setValue(point.seed_channel)
                    point.line.blockSignals(False)
            elif column == _COL_ENERGY:
                text = item.text().strip()
                point.energy_kev = None if not text else float(text)
                if point.energy_kev is not None and point.energy_kev < 0:
                    raise ValueError("energy cannot be negative")
            elif column == _COL_ENABLED:
                point.enabled = item.checkState() == Qt.CheckState.Checked
        except (TypeError, ValueError) as exc:
            self.fit_status.setText(f"Invalid calibration point: {exc}")
            self._rebuild_table()
            return
        workspace.dirty = True
        workspace.show_peak_fits = False
        self._refit()

    def _model_changed(self) -> None:
        workspace = self._workspace()
        if workspace is None:
            return
        workspace.model = str(self.model_combo.currentData())  # type: ignore[assignment]
        workspace.dirty = True
        workspace.show_peak_fits = False
        self._refit()

    def _centroid_width_changed(self) -> None:
        workspace = self._workspace()
        if workspace is not None:
            workspace.dirty = True
            workspace.show_peak_fits = False
        self._refit()

    def _calibration_points(self, workspace: _Workspace) -> tuple[CalibrationPoint, ...]:
        return tuple(
            CalibrationPoint(
                channel=point.channel,
                energy_kev=point.energy_kev,
                label=point.label,
                source=point.source,
                enabled=point.enabled,
            )
            for point in workspace.points
            if point.energy_kev is not None
        )

    def _refit(self) -> None:
        workspace = self._workspace()
        controller = self._controller()
        if workspace is None or controller is None:
            self.apply_action.setEnabled(False)
            self.report_action.setEnabled(False)
            return
        centroid_errors = self._fit_centroids(workspace)
        self._update_centroid_display(workspace)
        if centroid_errors:
            workspace.fit = None
            self.fit_status.setText(" ".join(centroid_errors))
            self.apply_action.setEnabled(False)
            self.report_action.setEnabled(False)
            self._update_residuals()
            return
        complete_enabled = [
            point for point in workspace.points if point.enabled and point.energy_kev is not None
        ]
        if any(point.enabled and point.energy_kev is None for point in workspace.points):
            workspace.fit = None
            self.fit_status.setText("Enter an expected energy for every enabled reference point.")
            self.apply_action.setEnabled(False)
            self.report_action.setEnabled(False)
            self._update_residuals()
            return
        fingerprints = {tuple(sorted(point.fingerprint.items())) for point in complete_enabled}
        if len(fingerprints) > 1:
            workspace.fit = None
            self.fit_status.setText(
                "Enabled points came from incompatible MCA settings; remove or disable "
                "points from one configuration."
            )
            self.apply_action.setEnabled(False)
            self.report_action.setEnabled(False)
            self._update_residuals()
            return
        fingerprint = dict(complete_enabled[0].fingerprint) if complete_enabled else {}
        try:
            workspace.fit = fit_energy_calibration(
                self._calibration_points(workspace),
                model=workspace.model,
                fingerprint=fingerprint,
            )
        except ValueError as exc:
            workspace.fit = None
            self.fit_status.setText(str(exc))
            self.apply_action.setEnabled(False)
            self.report_action.setEnabled(False)
            self._update_residuals()
            return
        calibration = workspace.fit
        coefficient_text = ", ".join(f"{value:.8g}" for value in calibration.coefficients_kev)
        stale = not calibration.settings_compatible(
            controller.energy_calibration_fingerprint(),
            allow_binning_rescale=True,
        )
        warning = " Settings differ from the source spectra; Apply is disabled." if stale else ""
        self.fit_status.setText(
            f"{calibration.model.capitalize()} fit coefficients [keV]: {coefficient_text}. "
            f"RMS residual {calibration.rms_residual_kev:.4g} keV; maximum "
            f"{calibration.max_residual_kev:.4g} keV.{warning}"
        )
        self.apply_action.setEnabled(not stale)
        self.report_action.setEnabled(True)
        self._update_residuals()

    def _fit_centroids(self, workspace: _Workspace) -> list[str]:
        snapshots = {snapshot.snapshot_id: snapshot for snapshot in workspace.snapshots}
        errors: list[str] = []
        half_width = self.centroid_half_width.value()
        for index, point in enumerate(workspace.points, start=1):
            point.channel = point.seed_channel
            snapshot = snapshots.get(point.snapshot_id or "")
            if snapshot is None or not point.enabled:
                point.peak_fit = None
                continue
            if (
                point.peak_fit is not None
                and point.peak_fit.half_width == half_width
                and (
                    point.peak_fit.seed_channel == point.seed_channel
                    or (
                        workspace.show_peak_fits
                        and point.peak_fit.centroid_channel == point.seed_channel
                    )
                )
            ):
                point.channel = point.peak_fit.centroid_channel
                continue
            try:
                point.peak_fit = fit_calibration_peak(
                    snapshot,
                    point.seed_channel,
                    half_width=half_width,
                )
            except ValueError as exc:
                errors.append(f"P{index} Gaussian centroid fit failed: {exc}")
                continue
            point.channel = point.peak_fit.centroid_channel
        return errors

    def _update_centroid_display(self, workspace: _Workspace) -> None:
        self._building_table = True
        for row, point in enumerate(workspace.points):
            item = self.table.item(row, _COL_CENTROID)
            if item is not None:
                item.setText("" if point.peak_fit is None else f"{point.channel:.4f}")
            if point.centroid_line is not None:
                try:
                    self.plot.removeItem(point.centroid_line)
                except RuntimeError:
                    pass
                point.centroid_line = None
            if not workspace.show_peak_fits and point.fit_curve is not None:
                try:
                    self.plot.removeItem(point.fit_curve)
                except RuntimeError:
                    pass
                point.fit_curve = None
            if not workspace.show_peak_fits and point.residual_curve is not None:
                try:
                    self.residual_plot.removeItem(point.residual_curve)
                except RuntimeError:
                    pass
                point.residual_curve = None
            if point.peak_fit is not None and not workspace.show_peak_fits:
                centroid_line = pg.InfiniteLine(
                    pos=point.channel,
                    angle=90,
                    movable=False,
                    pen=pg.mkPen("#2a9d8f", width=1.5),
                    label=f"C{row + 1}",
                    labelOpts={"color": "#2a9d8f", "position": 0.85},
                )
                self.plot.addItem(centroid_line)
                point.centroid_line = centroid_line
        self._building_table = False

    def _update_residuals(self) -> None:
        workspace = self._workspace()
        if workspace is None:
            return
        calibration = workspace.fit
        self._building_table = True
        for row, point in enumerate(workspace.points):
            item = self.table.item(row, _COL_RESIDUAL)
            if item is None:
                continue
            if calibration is None or point.energy_kev is None or not point.enabled:
                item.setText("")
            else:
                residual = point.energy_kev - float(calibration.energy(point.channel))
                item.setText(f"{residual:+.4g}")
        self._building_table = False

    def _remove_selected(self) -> None:
        workspace = self._workspace()
        if workspace is None:
            return
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            del workspace.points[row]
        if rows:
            workspace.dirty = True
            workspace.show_peak_fits = False
            self._render_workspace()
            self._refit()

    def _clear_points(self) -> None:
        workspace = self._workspace()
        if workspace is None or not workspace.points:
            return
        workspace.points.clear()
        workspace.fit = None
        workspace.dirty = True
        workspace.show_peak_fits = False
        self._render_workspace()
        self._refit()

    def _apply(self) -> None:
        workspace = self._workspace()
        controller = self._controller()
        if workspace is None or controller is None or workspace.fit is None:
            return
        if not workspace.fit.settings_compatible(
            controller.energy_calibration_fingerprint(),
            allow_binning_rescale=True,
        ):
            QMessageBox.warning(
                self,
                "Stale Calibration",
                "The MCA energy-processing settings changed after the source spectra "
                "were copied. Restore those settings or acquire new spectra.",
            )
            return
        controller.apply_energy_calibration(workspace.fit)
        for point in workspace.points:
            if point.peak_fit is not None:
                point.seed_channel = point.channel
        workspace.show_peak_fits = True
        workspace.dirty = False
        self._render_workspace()
        self._update_centroid_display(workspace)
        self.snapshot_status.setText(
            f"Applied {workspace.fit.model} energy calibration to MCA "
            f"{getattr(controller, 'channel_label', f'CH{controller.channel}')}. "
            "Raw channels were not modified."
        )

    def _clear_applied(self) -> None:
        controller = self._controller()
        workspace = self._workspace()
        if controller is None or workspace is None:
            return
        controller.clear_energy_calibration()
        workspace.show_peak_fits = False
        self._render_workspace()
        self._refit()
        self.snapshot_status.setText(
            "Cleared the applied calibration from MCA "
            f"{getattr(controller, 'channel_label', f'CH{controller.channel}')}."
        )

    def _download_report(self) -> None:
        workspace = self._workspace()
        channel = self._channel()
        if workspace is None or workspace.fit is None or channel is None:
            QMessageBox.information(
                self,
                "Energy Calibration Report",
                "Complete a valid energy calibration before creating a report.",
            )
            return
        default_folder = str(
            QSettings().value("analysis/energy_calibration_report_folder", "measurements")
        )
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Energy Calibration Report Destination",
            default_folder,
        )
        if not selected:
            return
        report_points = tuple(
            EnergyCalibrationReportPoint(
                point=CalibrationPoint(
                    channel=point.channel,
                    energy_kev=point.energy_kev,
                    label=point.label,
                    source=point.source,
                    enabled=point.enabled,
                ),
                seed_channel=(
                    point.peak_fit.seed_channel
                    if point.peak_fit is not None
                    else point.seed_channel
                ),
                peak_fit=point.peak_fit,
            )
            for point in workspace.points
            if point.energy_kev is not None
        )
        request = EnergyCalibrationReportRequest(
            channel=channel,
            snapshots=tuple(workspace.snapshots),
            calibration=workspace.fit,
            points=report_points,
            centroid_half_width=self.centroid_half_width.value(),
        )
        try:
            bundle = export_energy_calibration_report(request, Path(selected))
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Energy Calibration Report Failed", str(exc))
            return
        QSettings().setValue("analysis/energy_calibration_report_folder", selected)
        self.snapshot_status.setText(f"Energy calibration report saved to {bundle.directory}.")
        QMessageBox.information(
            self,
            "Energy Calibration Report Saved",
            f"Report and data files were saved to:\n{bundle.directory}",
        )

    def show_workspace(self) -> None:
        self._allow_close = False
        self.show()
        self.raise_()
        self.activateWindow()

    def close_without_prompt(self) -> None:
        self._allow_close = True
        self.close()

    def reject(self) -> None:
        dirty = any(workspace.dirty for workspace in self._workspaces.values())
        if not self._allow_close and dirty:
            answer = QMessageBox.question(
                self,
                "Discard Calibration Edits?",
                "Close the calibration tool and discard unapplied point edits?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Discard:
                return
        self._allow_close = True
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        dirty = any(workspace.dirty for workspace in self._workspaces.values())
        if not self._allow_close and dirty:
            event.ignore()
            self.reject()
            return
        super().closeEvent(event)
