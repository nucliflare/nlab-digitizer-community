"""Shared pyqtgraph interaction checks."""

from __future__ import annotations

from types import SimpleNamespace

import pyqtgraph as pg
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from nlab.views.plot_viewbox import ModifierZoomViewBox


@pytest.mark.parametrize(
    ("modifier", "expected_axis"),
    (
        (Qt.KeyboardModifier.NoModifier, None),
        (Qt.KeyboardModifier.ShiftModifier, 0),
        (Qt.KeyboardModifier.ControlModifier, 1),
    ),
)
def test_modifier_zoom_routes_wheel_to_requested_axis(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    modifier: Qt.KeyboardModifier,
    expected_axis: int | None,
) -> None:
    axes: list[int | None] = []

    def record_wheel(_viewbox: pg.ViewBox, _event: object, axis: int | None = None) -> None:
        axes.append(axis)

    monkeypatch.setattr(pg.ViewBox, "wheelEvent", record_wheel)
    event = SimpleNamespace(modifiers=lambda: modifier)

    ModifierZoomViewBox().wheelEvent(event)

    assert axes == [expected_axis]
