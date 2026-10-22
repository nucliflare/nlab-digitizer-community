from __future__ import annotations

import pytest
from PySide6.QtWidgets import QWidget
from pytestqt.qtbot import QtBot

from nlab.help.context import resolve_context
from nlab.views.offline_psd_plot import OfflinePsdPlot
from nlab.views.psd_readback_dialog import PsdReadbackDialog
from nlab.views.tooltips import TOOLTIP_LINE_LENGTH, wrap_tooltip, wrap_widget_tooltips
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
