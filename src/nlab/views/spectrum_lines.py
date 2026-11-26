"""Table of the lines found in a spectrum, shown beside the plot."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.waveform_spectrum import (
    UNIT_LABELS,
    SpectrumUnit,
    Spur,
    format_frequency,
    to_display_units,
)

_NUMERIC = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter


class SpectrumLinesTable(QWidget):
    """Marked lines in ascending frequency: frequency, level and excess over the floor."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel(self)
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self.table = QTableWidget(0, 3, self)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        # Share the panel width: no horizontal scrolling beside the plot.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.table, 1)
        self.setToolTip(
            "Lines that rise above the running-median noise floor by more than averaged "
            "noise could, in ascending frequency. Triangles mark them on the plot."
        )
        self.setMinimumWidth(230)
        self.setMaximumWidth(340)
        self.clear()

    def clear(self, text: str = "Lines: no spectrum yet.") -> None:
        self.table.setRowCount(0)
        self._set_headers(SpectrumUnit.DB)
        self.title.setText(text)

    def _set_headers(self, unit: SpectrumUnit) -> None:
        # Short headers keep three columns readable; the unit is in the title.
        self.table.setHorizontalHeaderLabels(["Frequency", "Level", "Above floor"])
        level = self.table.horizontalHeaderItem(1)
        if level is not None:
            level.setToolTip(f"Line level in {UNIT_LABELS[unit]}")

    def show_lines(self, spurs: Sequence[Spur], unit: SpectrumUnit, threshold_db: float) -> None:
        """List ``spurs`` by frequency, levels in ``unit``."""
        rows = sorted(spurs, key=lambda spur: spur.frequency)
        self._set_headers(unit)
        self.table.setRowCount(len(rows))
        levels = to_display_units(np.array([spur.level for spur in rows]), unit)
        for row, (spur, level) in enumerate(zip(rows, levels, strict=True)):
            for column, text in enumerate((
                format_frequency(spur.frequency), f"{level:.3g}", f"+{spur.excess_db:.1f} dB",
            )):
                item = QTableWidgetItem(text)
                item.setTextAlignment(_NUMERIC)
                self.table.setItem(row, column, item)
        self.title.setText(
            f"{len(rows)} line(s) more than {threshold_db:.1f} dB above the floor; "
            f"level in {UNIT_LABELS[unit]}"
            if rows else f"No line more than {threshold_db:.1f} dB above the floor"
        )
