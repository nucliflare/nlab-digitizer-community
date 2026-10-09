from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from nlab.views.plot_viewbox import NLabPlotWidget, style_plot_widget

_MAX_HISTORY_POINTS = 100_000


class DoubleClickableLinearRegionItem(pg.LinearRegionItem):  # type: ignore[misc]
    """Linear region that exposes a left-button double-click signal."""

    double_clicked = Signal()

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            self.double_clicked.emit()
            return
        super().mouseDoubleClickEvent(event)


class McaRoiHistoryDialog(QDialog):
    """Modeless history of counts integrated over one MCA ROI."""

    def __init__(self, channel: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._channel = int(channel)
        self._region: tuple[int, int] | None = None
        self._differential = False
        self._times_s: deque[float] = deque(maxlen=_MAX_HISTORY_POINTS)
        self._counts: deque[float] = deque(maxlen=_MAX_HISTORY_POINTS)

        self.setWindowTitle(f"MCA {self._channel} ROI Count History")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(760, 460)

        self.status_label = QLabel("Select an MCA ROI to begin monitoring.", self)
        self.plot_widget = NLabPlotWidget(self)
        style_plot_widget(self.plot_widget)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel("left", "Integrated counts")
        self.plot_widget.setLabel("bottom", "MCA elapsed time", units="s")
        self.curve = self.plot_widget.plot(pen=pg.mkPen("#1f77b4", width=2.0))
        self.curve.setDownsampling(auto=True, method="peak")
        self.curve.setClipToView(True)

        clear_button = QPushButton("Clear history", self)
        clear_button.clicked.connect(self.clear_history)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(clear_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.plot_widget, 1)
        layout.addLayout(button_row)

    @property
    def region(self) -> tuple[int, int] | None:
        return self._region

    def set_region(self, low_bin: int, high_bin: int) -> None:
        low, high = sorted((int(low_bin), int(high_bin)))
        region = (low, high)
        if region == self._region:
            return
        self._region = region
        self.clear_history()

    def set_differential(self, differential: bool) -> None:
        """Match the trace semantics to the MCA histogram display mode."""
        enabled = bool(differential)
        if enabled == self._differential:
            return
        self._differential = enabled
        self.plot_widget.setLabel(
            "left",
            "Counts / refresh" if enabled else "Integrated counts",
        )
        self.clear_history()

    def clear_history(self) -> None:
        self._times_s.clear()
        self._counts.clear()
        self.curve.setData([], [])
        self._refresh_status()

    def append_sample(self, elapsed_s: float, integrated_counts: int) -> None:
        elapsed = float(elapsed_s)
        counts = float(integrated_counts)
        if self._times_s and elapsed < self._times_s[-1]:
            self.clear_history()
        if self._times_s and elapsed == self._times_s[-1]:
            self._counts[-1] = counts
        else:
            self._times_s.append(elapsed)
            self._counts.append(counts)
        self.curve.setData(
            np.fromiter(self._times_s, dtype=np.float64, count=len(self._times_s)),
            np.fromiter(self._counts, dtype=np.float64, count=len(self._counts)),
        )
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self._region is None:
            self.status_label.setText("Select an MCA ROI to begin monitoring.")
            return
        low_bin, high_bin = self._region
        if self._counts:
            suffix = "counts in latest refresh" if self._differential else "accumulated counts"
            self.status_label.setText(
                f"MCA {self._channel}, bins {low_bin}-{high_bin}: "
                f"{int(self._counts[-1]):,} {suffix}"
            )
        else:
            self.status_label.setText(
                f"MCA {self._channel}, bins {low_bin}-{high_bin}: waiting for data"
            )
