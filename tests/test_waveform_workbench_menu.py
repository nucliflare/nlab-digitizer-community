from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog


def test_menu_groups_and_action_availability_follow_existing_controls(qtbot: QtBot) -> None:
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    assert [action.text().replace("&", "") for action in dialog.menu_bar.actions()] == [
        "File", "Analysis", "Selection", "View", "Help",
    ]
    actions = dialog._menu_button_actions
    for button, action in actions.items():
        assert action.isEnabled() == button.isEnabled()
        assert action.isCheckable() == button.isCheckable()
        assert action.isChecked() == button.isChecked()
    assert not actions[dialog.analyze_button].isEnabled()
    assert dialog.load_action in dialog.file_menu.actions()
    assert dialog.close_file_action in dialog.file_menu.actions()
    assert not dialog.close_file_action.isEnabled()
    assert not actions[dialog.cluster_selection.average_button].isEnabled()
    dialog._set_indexing(True)
    assert not dialog.load_action.isEnabled()
    dialog._set_indexing(False)
    assert dialog.load_action.isEnabled()
    dialog._set_analysis_busy(True)
    assert actions[dialog.cancel_button].isEnabled()
    assert not dialog.load_action.isEnabled()
    dialog._set_analysis_busy(False)
    assert not actions[dialog.cancel_button].isEnabled()
    dialog._set_optimization_busy(True)
    assert not actions[dialog.psd_plot.fom_widget.fit_button].isEnabled()
    assert not actions[dialog.auto_recalculate].isEnabled()
    dialog._set_optimization_busy(False)
    assert actions[dialog.psd_plot.fom_widget.fit_button].isEnabled()


def test_menu_toggles_sync_both_ways_and_selection_works_with_tools_hidden(qtbot: QtBot) -> None:
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    for button in (dialog.psd_plot.roi_checkbox, dialog.waveform_selection_checkbox,
                   dialog.psd_plot.fom_widget.guides_checkbox, dialog.auto_recalculate):
        action = dialog._menu_button_actions[button]
        changes: list[bool] = []
        button.toggled.connect(changes.append)
        action.trigger()
        assert not button.isChecked() and not action.isChecked()
        assert changes == [False]  # No signal feedback loop / double toggling.
        button.setChecked(True)
        assert action.isChecked() and changes == [False, True]

    cluster = dialog.cluster_selection
    cluster.set_events(np.arange(4), np.arange(4, dtype=float), np.linspace(0, .3, 4))
    dialog._menu_button_actions[dialog.waveform_selection_checkbox].trigger()
    assert cluster.isHidden()
    dialog._menu_button_actions[cluster.rectangle_button].trigger()
    assert cluster._roi is not None and not cluster._roi.isVisible()
    dialog._menu_button_actions[cluster.apply_button].trigger()
    assert cluster._selected.tolist() == [1, 2]
    assert dialog._menu_button_actions[cluster.average_button].isEnabled()
    dialog._menu_button_actions[cluster.clear_button].trigger()
    assert cluster._selected is None
    assert not dialog._menu_button_actions[cluster.average_button].isEnabled()


def test_file_menu_and_window_scoped_shortcuts_use_existing_workflow(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    choose_file = Mock(return_value=("", ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", choose_file)
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.activateWindow()
    dialog.raise_()
    qtbot.waitUntil(dialog.isActiveWindow)
    action = dialog.load_action
    action.trigger()
    assert choose_file.call_count == 1
    qtbot.keyClick(dialog, Qt.Key.Key_O, Qt.KeyboardModifier.ControlModifier)
    assert choose_file.call_count == 2
    assert action.shortcutContext() == Qt.ShortcutContext.WindowShortcut
    dialog.load_action.setEnabled(False)
    qtbot.keyClick(dialog, Qt.Key.Key_O, Qt.KeyboardModifier.ControlModifier)
    assert choose_file.call_count == 2
    dialog._menu_button_actions[dialog.close_button].trigger()
    assert not dialog.isVisible()


def test_help_menu_opens_and_reuses_bundled_waveform_guide(qtbot: QtBot) -> None:
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    menu = dialog.menu_bar.actions()[-1].menu()
    menu.actions()[0].trigger()
    help_dialog = dialog._help_dialog
    assert help_dialog is not None and help_dialog.isVisible()
    assert help_dialog.parentWidget() is dialog
    assert help_dialog._history[-1].topic == "waveform"
    help_dialog.close()
    menu.actions()[0].trigger()
    assert dialog._help_dialog is help_dialog and help_dialog.isVisible()
