from __future__ import annotations

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QEvent, QObject, QPoint
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QApplication, QLineEdit, QSpinBox, QStyle, QTabBar, QWidget
from pytestqt.qtbot import QtBot

from nlab.help.context import resolve_context
from nlab.views import tooltips
from nlab.views.numeric_controls import install_numeric_input_policy
from nlab.views.offline_psd_plot import OfflinePsdPlot
from nlab.views.psd_readback_dialog import PsdReadbackDialog
from nlab.views.tooltips import (
    TOOLTIP_LINE_LENGTH,
    install_tooltip_wrapping,
    wrap_tooltip,
    wrap_widget_tooltips,
)
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog


def test_plain_text_tooltips_wrap_without_losing_explanations_or_paragraphs() -> None:
    text = "Signed charge is valid when Qlong > 0. " * 8
    wrapped = wrap_tooltip(text.rstrip())
    assert "\n" in wrapped
    assert max(map(len, wrapped.splitlines())) <= TOOLTIP_LINE_LENGTH
    assert wrapped.split() == text.split()
    assert wrap_tooltip(wrapped) == wrapped
    assert wrap_tooltip("Short hint.") == "Short hint."
    assert wrap_tooltip("") == ""
    assert wrap_tooltip("First line.\n\nSecond line.") == "First line.\n\nSecond line."
    path = "D:/" + "long_capture_name_" * 20 + ".bin"
    assert wrap_tooltip(path).replace("\n", "") == path
    assert max(map(len, wrap_tooltip(path).splitlines())) <= TOOLTIP_LINE_LENGTH


@pytest.mark.parametrize("view_type", [WaveformAnalysisDialog, PsdReadbackDialog, OfflinePsdPlot])
def test_waveform_and_psd_tooltips_are_multiline_and_keep_f1_plain_text(
    qtbot: QtBot, view_type: type[QWidget],
) -> None:
    view = view_type()
    qtbot.addWidget(view)
    tips = [widget.toolTip() for widget in (view, *view.findChildren(QWidget)) if widget.toolTip()]
    assert any("\n" in tip for tip in tips)
    assert all(len(line) <= TOOLTIP_LINE_LENGTH for tip in tips for line in tip.splitlines())
    before = tips.copy()
    wrap_widget_tooltips(view)
    assert before == [widget.toolTip() for widget in (view, *view.findChildren(QWidget))
                      if widget.toolTip()]
    if isinstance(view, WaveformAnalysisDialog):
        context = resolve_context(view.waveform_widget)
        assert context.topic == "waveform-gates"
        assert context.tooltip == view.waveform_widget.toolTip()
        assert "Blue region: baseline" in context.tooltip
        assert "<html" not in context.tooltip
        for button, action in view._menu_button_actions.items():
            assert action.toolTip() == button.toolTip() or not button.toolTip()


_LONG_TIP = (
    "Number of runs one Start performs while Settings > Enable Series is checked. "
    "Recordings are numbered <name>_001, <name>_002, where <name> is the chosen file."
)


def _shown_tooltips(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    shown: list[str] = []
    monkeypatch.setattr(
        tooltips.QToolTip, "showText", lambda _pos, text, *_args: shown.append(text),
    )
    return shown


def _hover(widget: QWidget) -> bool:
    """Deliver the help event Qt sends when the pointer rests on ``widget``."""
    event = QHelpEvent(QEvent.Type.ToolTip, QPoint(2, 2), widget.mapToGlobal(QPoint(2, 2)))
    QApplication.sendEvent(widget, event)
    return event.isAccepted()


def test_long_plain_tooltips_are_shown_wrapped_and_stored_unchanged(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_tooltip_wrapping()
    install_tooltip_wrapping()  # idempotent
    install_numeric_input_policy()  # shares the event hub; must keep its own route
    shown = _shown_tooltips(monkeypatch)
    # A widget created after installation, as in every view and dialog.
    parent = QWidget()
    qtbot.addWidget(parent)
    spinbox = QSpinBox(parent)
    spinbox.setToolTip(_LONG_TIP)

    # The pointer rests on the spin box's line edit; Qt hands the event up.
    line_edit = spinbox.findChild(QLineEdit)
    assert line_edit is not None and not line_edit.toolTip()
    _hover(line_edit)

    assert shown == [wrap_tooltip(_LONG_TIP)]
    lines = shown[0].splitlines()
    assert len(lines) > 1 and max(map(len, lines)) <= TOOLTIP_LINE_LENGTH
    assert "<name>_001" in shown[0]  # plain text: no markup interpretation
    assert spinbox.toolTip() == _LONG_TIP  # F1 help and code still read the original
    assert resolve_context(spinbox).tooltip == _LONG_TIP


def _help_event(widget: QWidget, position: QPoint) -> QHelpEvent:
    return QHelpEvent(QEvent.Type.ToolTip, position, widget.mapToGlobal(position))


@pytest.mark.parametrize(
    "tooltip",
    [
        "Short hint.",
        "<b>Rich text</b> is word-wrapped by Qt itself. " * 4,
        wrap_tooltip(_LONG_TIP),  # already wrapped at construction
    ],
)
def test_tooltips_qt_already_shows_well_are_left_to_qt(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tooltip: str,
) -> None:
    shown = _shown_tooltips(monkeypatch)
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setToolTip(tooltip)

    # Not consumed: Qt's own QWidget::event then shows the text as it is.
    assert not tooltips._show_wrapped_tooltip(widget, _help_event(widget, QPoint(2, 2)))
    assert shown == []


def _plot_delay() -> tooltips._PlotTooltipDelay:
    install_tooltip_wrapping()
    application = QApplication.instance()
    assert application is not None
    delay = application.findChild(QObject, "nlabPlotTooltipDelay")
    assert isinstance(delay, tooltips._PlotTooltipDelay)
    return delay


def _plot_in_container(qtbot: QtBot) -> tuple[QWidget, pg.PlotWidget]:
    container = QWidget()
    qtbot.addWidget(container)
    container.setToolTip("Container. " + _LONG_TIP)
    plot = pg.PlotWidget(container)
    plot.setToolTip(_LONG_TIP)
    return container, plot


@pytest.mark.parametrize("rests", [True, False])
def test_plot_tooltips_wait_twice_the_qt_delay_while_the_cursor_rests(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, rests: bool,
) -> None:
    delay = _plot_delay()
    shown = _shown_tooltips(monkeypatch)
    _container, plot = _plot_in_container(qtbot)

    # Qt's event after its own delay is held, and the copies Qt passes to
    # the plot and its container do not show their tooltips meanwhile.
    assert not _hover(plot.viewport())  # ignored: Qt starts no follow-up delay
    assert shown == []
    assert delay.pending
    wake_ms = plot.style().styleHint(QStyle.StyleHint.SH_ToolTip_WakeUpDelay, None, plot)
    assert delay._timer.interval() == wake_ms * (tooltips.PLOT_TOOLTIP_DELAY_FACTOR - 1)

    monkeypatch.setattr(tooltips, "_pointer_rests_on", lambda _widget, _pos: rests)
    delay._timer.stop()
    delay._release()

    # Released through the plot as Qt would deliver it; a moved cursor waits
    # for Qt's next event instead.
    assert shown == ([wrap_tooltip(_LONG_TIP)] if rests else [])


def test_clicks_wheel_and_keys_drop_a_held_plot_tooltip(qtbot: QtBot) -> None:
    delay = _plot_delay()
    _container, plot = _plot_in_container(qtbot)
    for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.Wheel, QEvent.Type.KeyPress):
        _hover(plot.viewport())
        assert delay.pending
        assert not delay.handle(plot.viewport(), QEvent(kind))
        assert not delay.pending


def test_other_widgets_show_their_tooltips_without_the_plot_delay(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    delay = _plot_delay()
    shown = _shown_tooltips(monkeypatch)
    container, _plot = _plot_in_container(qtbot)

    _hover(container)

    assert shown == [wrap_tooltip("Container. " + _LONG_TIP)]
    assert not delay.pending


def test_tab_bars_keep_their_per_tab_tooltips(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown = _shown_tooltips(monkeypatch)
    tabs = QTabBar()
    qtbot.addWidget(tabs)
    tabs.addTab("D1:CH0")
    tabs.setTabToolTip(0, "Channel 0")
    tabs.setToolTip(_LONG_TIP)

    # QTabBar::event picks the tab's tooltip; an application filter must not
    # answer first with the bar's own.
    event = _help_event(tabs, tabs.tabRect(0).center())
    assert not tooltips._show_wrapped_tooltip(tabs, event)
    assert shown == []
