"""The lines table beside spectrum plots: ascending frequency, units, empty states."""

from __future__ import annotations

import math

from pytestqt.qtbot import QtBot

from nlab.analysis.waveform_spectrum import SpectrumUnit, Spur
from nlab.views.spectrum_lines import SpectrumLinesTable


def _column(table: SpectrumLinesTable, column: int) -> list[str]:
    widget = table.table
    return [widget.item(row, column).text() for row in range(widget.rowCount())]


def test_lines_are_listed_by_frequency_in_the_selected_units(qtbot: QtBot) -> None:
    table = SpectrumLinesTable()
    qtbot.addWidget(table)
    spurs = [
        Spur(frequency=558.4, level=1.05, excess_db=10.6, width_bins=1, bin=75),
        Spur(frequency=140.1, level=0.65, excess_db=8.5, width_bins=1, bin=19),
        Spur(frequency=1117.2, level=0.30, excess_db=5.4, width_bins=1, bin=150),
    ]
    table.show_lines(spurs, SpectrumUnit.DB, 4.0)
    assert _column(table, 0) == ["140.1 Hz", "558.4 Hz", "1.117 kHz"]
    assert _column(table, 1)[0] == f"{10 * math.log10(0.65):.3g}"
    assert _column(table, 2) == ["+8.5 dB", "+10.6 dB", "+5.4 dB"]
    assert table.table.horizontalHeaderItem(1).text() == "Level"
    assert table.title.text() == (
        "3 line(s) more than 4.0 dB above the floor; level in dB re 1 code²/Hz"
    )
    table.show_lines(spurs, SpectrumUnit.PSD, 4.0)
    assert _column(table, 1)[0] == "0.65"
    assert table.title.text().endswith("level in codes²/Hz")
    assert "codes²/Hz" in table.table.horizontalHeaderItem(1).toolTip()
    table.show_lines([], SpectrumUnit.DB, 10.0)
    assert table.table.rowCount() == 0
    assert table.title.text() == "No line more than 10.0 dB above the floor"
    table.clear()
    assert table.table.rowCount() == 0 and "no spectrum" in table.title.text()
