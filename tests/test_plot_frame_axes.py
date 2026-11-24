"""Redundant top/right plot axes: frame only, energy axes only when calibrated."""

from __future__ import annotations

from unittest.mock import Mock

import pyqtgraph as pg
from pytestqt.qtbot import QtBot

from nlab.views.energy_axis import CalibratedEnergyAxis
from nlab.views.plot_viewbox import FRAME_AXIS_SIZE, show_frame_axes


def test_frame_axes_keep_the_frame_without_repeated_values(qtbot: QtBot) -> None:
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    plot = widget.getPlotItem()
    plot.setLabel("top", "Mirrored title")
    show_frame_axes(plot)

    for side in ("top", "right"):
        axis = plot.getAxis(side)
        assert axis.isVisible()
        assert not axis.style["showValues"]
        assert not axis.label.isVisible()
    # One fixed size on every plot is what keeps linked projections aligned.
    assert plot.getAxis("top").maximumHeight() == FRAME_AXIS_SIZE
    assert plot.getAxis("right").maximumWidth() == FRAME_AXIS_SIZE
    assert plot.getAxis("bottom").style["showValues"]
    assert plot.getAxis("left").style["showValues"]


def test_energy_axis_shows_values_only_with_a_calibration(qtbot: QtBot) -> None:
    widget = pg.PlotWidget(axisItems={"top": CalibratedEnergyAxis("top")})
    qtbot.addWidget(widget)
    axis = widget.getPlotItem().getAxis("top")
    assert isinstance(axis, CalibratedEnergyAxis)
    assert not axis.style["showValues"]
    assert not axis.label.isVisible()

    calibration = Mock()
    calibration.model = "linear"
    calibration.energy_unit = "keV"
    calibration.energy_quantity = "Energy"
    axis.set_calibration(calibration)
    assert axis.style["showValues"]
    assert axis.label.isVisible()
    assert axis.labelUnits == "keV"
    assert axis.labelText == "Energy"
    calibration.energy_unit = "keVee"
    calibration.energy_quantity = "Light output"
    axis.set_calibration(calibration)
    assert axis.labelUnits == "keVee"
    assert axis.labelText == "Light output"
    assert axis.fixedHeight is None  # Sized to its tick labels again.

    axis.set_calibration(None)
    assert not axis.style["showValues"]
    assert not axis.label.isVisible()
