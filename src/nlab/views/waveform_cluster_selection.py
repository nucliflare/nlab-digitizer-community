"""Editable PSD regions and navigation through their original waveform frames."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from nlab.analysis.psd_selection import points_in_polygon
from nlab.views.tooltips import wrap_widget_tooltips


class WaveformClusterSelection(QWidget):
    selection_changed = Signal(object)  # None = unfiltered; empty array = no matches.
    frame_requested = Signal(int)

    def __init__(self, plot: pg.PlotItem, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("nlabHelpTopic", "waveform-clusters")
        self._plot = plot
        self._frames = np.empty(0, dtype=np.int64)
        self._energy = np.empty(0)
        self._ratio = np.empty(0)
        self._selected: np.ndarray | None = None
        self._roi: pg.ROI | None = None
        self._busy = False
        self._tools_visible = True
        self._point = pg.ScatterPlotItem(size=11, pen=pg.mkPen("black"), brush="white")
        self._point.setZValue(60)
        plot.addItem(self._point)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Waveform cluster:", self))
        self.rectangle_button = QPushButton("Rectangle", self)
        self.polygon_button = QPushButton("Polygon", self)
        self.apply_button = QPushButton("Apply selection", self)
        self.clear_button = QPushButton("Clear selection", self)
        self.average_button = QPushButton("Show average waveform", self)
        self.average_button.setCheckable(True)
        self.average_button.setToolTip(
            "Overlay the mean of all selected waveforms, with each event's baseline "
            "removed and reconstruction polarity applied. No peak alignment or amplitude "
            "scaling. Click again to hide/cancel. Variable-length records use their shared prefix."
        )
        for button in (self.rectangle_button, self.polygon_button,
                       self.apply_button, self.clear_button, self.average_button):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.previous_button = QPushButton("Previous", self)
        self.match_spin = QSpinBox(self)
        self.match_spin.setRange(0, 0)
        self.next_button = QPushButton("Next", self)
        for control in (self.previous_button, self.match_spin, self.next_button):
            row.addWidget(control)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        layout.addLayout(row)
        self.setToolTip(
            "Reconstruct PSD first. Add a rectangle or polygon, move/resize it over a cluster, "
            "then Apply selection. Polygon vertices can be dragged; click an edge to add a "
            "vertex. Previous/Next and the match number browse only the matching original "
            "frames. Boundaries are included. This is a geometric selection, not particle ID. "
            "Only events included by the reconstruction's stride, limit, and display ranges "
            "can be selected. Changing analysis settings invalidates the selection."
        )
        self.rectangle_button.clicked.connect(lambda: self.add_region("rectangle"))
        self.polygon_button.clicked.connect(lambda: self.add_region("polygon"))
        self.apply_button.clicked.connect(self.apply_selection)
        self.clear_button.clicked.connect(self.clear_selection)
        self.previous_button.clicked.connect(lambda: self.match_spin.stepDown())
        self.next_button.clicked.connect(lambda: self.match_spin.stepUp())
        self.match_spin.valueChanged.connect(self._show_match)
        self.invalidate()
        wrap_widget_tooltips(self)

    def set_events(self, frames: np.ndarray, energy: np.ndarray, ratio: np.ndarray) -> None:
        if frames.ndim != 1 or frames.shape != energy.shape or frames.shape != ratio.shape:
            raise ValueError("Cluster event columns must have matching one-dimensional shapes")
        self.invalidate()
        self._frames, self._energy, self._ratio = frames, energy, ratio
        self.status.setText(f"{len(frames):,} plotted events; add a region to select waveforms.")
        self._update_buttons()

    @property
    def has_events(self) -> bool:
        return bool(len(self._frames))

    def invalidate(self) -> None:
        self.clear_selection()
        self._frames = np.empty(0, dtype=np.int64)
        self._energy, self._ratio = np.empty(0), np.empty(0)
        self._point.clear()
        self.status.setText("Reconstruct PSD to select waveforms.")
        self._update_buttons()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        if self._roi is not None:
            self._roi.setEnabled(not busy)
        self._update_buttons()

    def set_tools_visible(self, visible: bool) -> None:
        """Hide controls/overlays without clearing the region or active filter."""
        self._tools_visible = visible
        self.setVisible(visible)
        if self._roi is not None:
            self._roi.setVisible(visible)
        self._point.setVisible(visible)

    def add_region(self, shape: str) -> None:
        if not len(self._frames) or self._busy:
            return
        self.clear_selection()
        low = np.array([np.quantile(self._energy, 0.2), np.quantile(self._ratio, 0.2)])
        high = np.array([np.quantile(self._energy, 0.8), np.quantile(self._ratio, 0.8)])
        width = np.maximum(high - low, [1.0, 0.01])
        style = dict(
            pen=pg.mkPen("black", width=2), hoverPen=pg.mkPen("black", width=2),
            handlePen=pg.mkPen("black"), handleHoverPen=pg.mkPen("black", width=2),
        )
        if shape == "rectangle":
            self._roi = pg.RectROI(low, width, sideScalers=True, **style)
        elif shape == "polygon":
            x, y = low
            w, h = width
            self._roi = pg.PolyLineROI(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                closed=True, **style,
            )
        else:
            raise ValueError("Unknown cluster shape")
        self._roi.setZValue(50)
        self._plot.addItem(self._roi)
        self._roi.setVisible(self._tools_visible)
        self._roi.sigRegionChangeFinished.connect(self._region_edited)
        self.status.setText("Move/resize the region, then Apply selection.")
        self._update_buttons()

    def _region_edited(self, *_args: object) -> None:
        self._clear_matches()
        self.status.setText("Region changed; Apply selection to update matching waveforms.")
        self._update_buttons()

    def region_vertices(self) -> np.ndarray:
        roi = self._roi
        if roi is None:
            raise ValueError("Add a rectangle or polygon first")
        if isinstance(roi, pg.PolyLineROI):
            points = [point for _name, point in roi.getLocalHandlePositions()]
        else:
            w, h = roi.size()
            points = [QPointF(0, 0), QPointF(w, 0), QPointF(w, h), QPointF(0, h)]
        points = [roi.mapToParent(point) for point in points]
        return np.asarray([(point.x(), point.y()) for point in points])

    def apply_selection(self) -> None:
        if self._roi is None or self._busy:
            return
        try:
            mask = points_in_polygon(self._energy, self._ratio, self.region_vertices())
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.average_button.setChecked(False)
        self._selected = self._frames[mask].copy()
        self._selected.setflags(write=False)
        count = len(self._selected)
        self.match_spin.blockSignals(True)
        self.match_spin.setRange(1 if count else 0, count)
        self.match_spin.setValue(1 if count else 0)
        self.match_spin.blockSignals(False)
        self.status.setText(f"{count:,} selected / {len(self._frames):,} plotted events.")
        self.selection_changed.emit(self._selected)
        self._show_match(self.match_spin.value())
        self._update_buttons()

    def _show_match(self, number: int) -> None:
        if self._selected is not None and 1 <= number <= len(self._selected):
            self.frame_requested.emit(int(self._selected[number - 1]))
        self._update_buttons()

    def _clear_matches(self) -> None:
        self.average_button.setChecked(False)
        had_selection = self._selected is not None
        self._selected = None
        self.match_spin.blockSignals(True)
        self.match_spin.setRange(0, 0)
        self.match_spin.blockSignals(False)
        if had_selection:
            self.selection_changed.emit(None)

    def clear_selection(self) -> None:
        if self._roi is not None:
            self._plot.removeItem(self._roi)
            self._roi = None
        self._clear_matches()
        self.status.setText("All frames available; add a region to filter waveforms.")
        self._update_buttons()

    def set_current_frame(self, frame: int) -> None:
        position = int(np.searchsorted(self._frames, frame))
        if position < len(self._frames) and self._frames[position] == frame:
            self._point.setData([self._energy[position]], [self._ratio[position]])
        else:
            self._point.clear()

    def _update_buttons(self) -> None:
        available = bool(len(self._frames)) and not self._busy
        self.rectangle_button.setEnabled(available)
        self.polygon_button.setEnabled(available)
        self.apply_button.setEnabled(available and self._roi is not None)
        self.clear_button.setEnabled(self._roi is not None or self._selected is not None)
        count = len(self._selected) if self._selected is not None else 0
        self.average_button.setEnabled(not self._busy and count > 0)
        self.match_spin.setEnabled(not self._busy and count > 0)
        self.previous_button.setEnabled(
            not self._busy and count > 0 and self.match_spin.value() > 1,
        )
        self.next_button.setEnabled(not self._busy and count > self.match_spin.value())
