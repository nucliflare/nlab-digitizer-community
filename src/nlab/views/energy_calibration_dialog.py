"""Modeless multi-spectrum MCA energy-calibration workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.energy_calibration import (
    CalibrationModel,
    CalibrationPoint,
    EnergyCalibration,
    FingerprintValue,
    SpectrumSnapshot,
    fit_energy_calibration,
)
from nlab.views.plot_viewbox import ModifierZoomViewBox

if TYPE_CHECKING:
    from nlab.controllers.mca_controller import MCAController

_COL_POINT = 0
_COL_LABEL = 1
_COL_SOURCE = 2
_COL_CHANNEL = 3
_COL_ENERGY = 4
_COL_RESIDUAL = 5
_COL_ENABLED = 6
_OVERLAY_COLORS = ("#1f77b4", "#d28b38", "#648b71", "#7353a6", "#a66f6f")


@dataclass
class _DraftPoint:
    point_id: str
    channel: float
    energy_kev: float | None
    label: str
    source: str
    fingerprint: dict[str, FingerprintValue]
    enabled: bool = True
    line: pg.InfiniteLine | None = None


@dataclass
class _Workspace:
    snapshots: list[SpectrumSnapshot] = field(default_factory=list)
    points: list[_DraftPoint] = field(default_factory=list)
    model: CalibrationModel = "linear"
    fit: EnergyCalibration | None = None
    dirty: bool = False


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
            self.channel_combo.addItem(f"MCA channel {channel}", channel)
            self._workspaces[channel] = self._workspace_from_applied(
                self._controllers[channel].energy_calibration
            )
        self._connect_signals()
        self._show_channel()
        if self._controllers:
            self._load_current(replace=True, show_error=False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
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
        self.replace_button = QPushButton("Replace with current spectrum", self)
        self.overlay_button = QPushButton("Add current as overlay", self)
        source_row.addWidget(self.replace_button)
        source_row.addWidget(self.overlay_button)
        source_row.addStretch(1)
        root.addLayout(source_row)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.plot = pg.PlotWidget(self, viewBox=ModifierZoomViewBox())
        self.plot.setBackground("#f8f9fa")
        self.plot.setLabel("bottom", "MCA channel")
        self.plot.setLabel("left", "Counts")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.legend = self.plot.addLegend()
        splitter.addWidget(self.plot)

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
        model_row.addStretch(1)
        side_layout.addLayout(model_row)

        self.table = QTableWidget(0, 7, self)
        self.table.setHorizontalHeaderLabels(
            ["Point", "Label", "Spectrum", "Channel", "Energy [keV]", "Residual", "Use"]
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
        self.table.setColumnWidth(_COL_POINT, 40)
        self.table.setColumnWidth(_COL_LABEL, 48)
        self.table.setColumnWidth(_COL_CHANNEL, 65)
        self.table.setColumnWidth(_COL_ENERGY, 82)
        self.table.setColumnWidth(_COL_RESIDUAL, 60)
        self.table.setColumnWidth(_COL_ENABLED, 36)
        side_layout.addWidget(self.table, 1)

        point_row = QHBoxLayout()
        self.add_button = QPushButton("Add point", self)
        self.remove_button = QPushButton("Remove selected", self)
        self.clear_points_button = QPushButton("Clear points", self)
        point_row.addWidget(self.add_button)
        point_row.addWidget(self.remove_button)
        point_row.addWidget(self.clear_points_button)
        side_layout.addLayout(point_row)

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

        action_row = QHBoxLayout()
        self.snapshot_status = QLabel("No spectrum copied.", self)
        self.snapshot_status.setWordWrap(True)
        action_row.addWidget(self.snapshot_status, 1)
        self.clear_calibration_button = QPushButton("Clear applied calibration", self)
        self.apply_button = QPushButton("Apply calibration", self)
        self.apply_button.setEnabled(False)
        close_button = QPushButton("Close", self)
        action_row.addWidget(self.clear_calibration_button)
        action_row.addWidget(self.apply_button)
        action_row.addWidget(close_button)
        root.addLayout(action_row)
        self._close_button = close_button

    def _connect_signals(self) -> None:
        self.channel_combo.currentIndexChanged.connect(self._show_channel)
        self.replace_button.clicked.connect(lambda: self._load_current(replace=True))
        self.overlay_button.clicked.connect(lambda: self._load_current(replace=False))
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.add_button.clicked.connect(lambda: self._add_point())
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_points_button.clicked.connect(self._clear_points)
        self.table.cellChanged.connect(self._table_changed)
        self.apply_button.clicked.connect(self._apply)
        self.clear_calibration_button.clicked.connect(self._clear_applied)
        self._close_button.clicked.connect(self.reject)
        self.plot.scene().sigMouseClicked.connect(self._plot_clicked)

    def _workspace_from_applied(self, calibration: EnergyCalibration | None) -> _Workspace:
        if calibration is None:
            return _Workspace()
        points = [
            _DraftPoint(
                point_id=uuid4().hex,
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
        for button in (self.replace_button, self.overlay_button, self.add_button):
            button.setEnabled(enabled)
        if workspace is None:
            self.plot.clear()
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
        else:
            workspace.snapshots.append(snapshot)
        self._render_workspace()
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
            line = pg.InfiniteLine(
                pos=point.channel,
                angle=90,
                movable=True,
                pen=pg.mkPen("#a66f6f", width=1.5, style=Qt.PenStyle.DashLine),
                hoverPen=pg.mkPen("#c17c7c", width=2),
                label=f"P{index + 1}",
                labelOpts={"color": "#a66f6f", "position": 0.95},
            )
            line.setBounds((0, 16_383))
            line.sigPositionChanged.connect(
                lambda moved, point_id=point.point_id: self._line_moved(point_id, moved)
            )
            self.plot.addItem(line)
            point.line = line
        self._rebuild_table()

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
                channel=float(np.clip(channel, 0, 16_383)),
                energy_kev=None,
                label="",
                source=snapshot.label if snapshot is not None else "Manual",
                fingerprint=(
                    dict(snapshot.fingerprint)
                    if snapshot is not None
                    else self._controller().energy_calibration_fingerprint()  # type: ignore[union-attr]
                ),
            )
        )
        workspace.dirty = True
        self._render_workspace()
        self._refit()

    def _line_moved(self, point_id: str, line: pg.InfiniteLine) -> None:
        workspace = self._workspace()
        if workspace is None:
            return
        for row, point in enumerate(workspace.points):
            if point.point_id != point_id:
                continue
            point.channel = float(np.clip(line.value(), 0, 16_383))
            workspace.dirty = True
            item = self.table.item(row, _COL_CHANNEL)
            if item is not None:
                self._building_table = True
                item.setText(f"{point.channel:.3f}")
                self._building_table = False
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
            self.table.setItem(row, _COL_CHANNEL, QTableWidgetItem(f"{point.channel:.3f}"))
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
            elif column == _COL_CHANNEL:
                point.channel = float(item.text())
                if not 0 <= point.channel <= 16_383:
                    raise ValueError("channel must be within 0-16383")
                if point.line is not None:
                    point.line.blockSignals(True)
                    point.line.setValue(point.channel)
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
        self._refit()

    def _model_changed(self) -> None:
        workspace = self._workspace()
        if workspace is None:
            return
        workspace.model = str(self.model_combo.currentData())  # type: ignore[assignment]
        workspace.dirty = True
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
            self.apply_button.setEnabled(False)
            return
        complete_enabled = [
            point for point in workspace.points if point.enabled and point.energy_kev is not None
        ]
        if any(point.enabled and point.energy_kev is None for point in workspace.points):
            workspace.fit = None
            self.fit_status.setText("Enter an expected energy for every enabled reference point.")
            self.apply_button.setEnabled(False)
            self._update_residuals()
            return
        fingerprints = {tuple(sorted(point.fingerprint.items())) for point in complete_enabled}
        if len(fingerprints) > 1:
            workspace.fit = None
            self.fit_status.setText(
                "Enabled points came from incompatible MCA settings; remove or disable "
                "points from one configuration."
            )
            self.apply_button.setEnabled(False)
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
            self.apply_button.setEnabled(False)
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
        self.apply_button.setEnabled(not stale)
        self._update_residuals()

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
            self._render_workspace()
            self._refit()

    def _clear_points(self) -> None:
        workspace = self._workspace()
        if workspace is None or not workspace.points:
            return
        workspace.points.clear()
        workspace.fit = None
        workspace.dirty = True
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
        workspace.dirty = False
        self.snapshot_status.setText(
            f"Applied {workspace.fit.model} energy calibration to MCA channel "
            f"{controller.channel}. Raw channels were not modified."
        )

    def _clear_applied(self) -> None:
        controller = self._controller()
        if controller is None:
            return
        controller.clear_energy_calibration()
        self.snapshot_status.setText(
            f"Cleared the applied calibration from MCA channel {controller.channel}."
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
