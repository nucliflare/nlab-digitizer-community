"""Decade-labelled log frequency axes instead of overlapping SI minor labels."""

from __future__ import annotations

import math

import pyqtgraph as pg
from pytestqt.qtbot import QtBot

from nlab.views.frequency_axis import log_frequency_ticks, set_frequency_axis


def test_log_ticks_label_decades_and_halves_in_their_own_units() -> None:
    decades, halves, minor = log_frequency_ticks()
    labels = dict(decades)
    assert labels[1.0] == "10 Hz" and labels[3.0] == "1 kHz" and labels[6.0] == "1 MHz"
    assert dict(halves)[math.log10(2e3)] == "2 kHz"
    assert dict(halves)[math.log10(500)] == "500 Hz"
    assert all(text == "" for _, text in minor)
    assert len(minor) == 6 * len(decades)


def test_axis_switches_between_log_ticks_and_automatic_si(qtbot: QtBot) -> None:
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    plot = widget.getPlotItem()
    set_frequency_axis(plot, log=True)
    axis = plot.getAxis("bottom")
    assert axis._tickLevels is not None and not axis.autoSIPrefix
    assert axis.labelText == "Frequency" and not axis.labelUnits
    set_frequency_axis(plot, log=False)
    assert axis._tickLevels is None and axis.autoSIPrefix and axis.labelUnits == "Hz"
