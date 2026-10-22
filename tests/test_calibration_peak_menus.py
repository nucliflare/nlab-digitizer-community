from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QPushButton, QToolBar
from pytestqt.qtbot import QtBot

from nlab.views.energy_calibration_dialog import EnergyCalibrationDialog
from nlab.views.mca_peak_analysis_dialog import McaPeakAnalysisDialog


@pytest.mark.parametrize("kind", ["calibration", "peaks"])
def test_workbench_commands_are_menu_actions_not_button_rows(qtbot: QtBot, kind: str) -> None:
    dialog = EnergyCalibrationDialog([]) if kind == "calibration" else McaPeakAnalysisDialog([])
    qtbot.addWidget(dialog)
    assert not dialog.findChildren(QPushButton)
    assert not dialog.findChildren(QToolBar)
    expected = ["File", "Analysis", "View", "Help"] if kind == "calibration" else [
        "File", "Analysis", "References", "View", "Help",
    ]
    assert [action.text().replace("&", "") for action in dialog.menu_bar.actions()] == expected
    assert dialog.close_action in dialog.file_menu.actions()
    assert dialog.close_action.shortcut().toString() == "Ctrl+W"
    assert dialog.close_action.shortcutContext() == Qt.ShortcutContext.WindowShortcut
    if isinstance(dialog, EnergyCalibrationDialog):
        assert dialog.replace_action in dialog.file_menu.actions()
        assert dialog.overlay_action in dialog.file_menu.actions()
        assert dialog.report_action in dialog.file_menu.actions()
        assert dialog.apply_action in dialog.analysis_menu.actions()
        assert not dialog.apply_action.isEnabled() and not dialog.report_action.isEnabled()
        assert not dialog.add_action.isEnabled()
    else:
        assert dialog.load_action in dialog.file_menu.actions()
        assert dialog.export_fit_action in dialog.file_menu.actions()
        assert dialog.report_fit_action in dialog.file_menu.actions()
        assert dialog.fit_action in dialog.analysis_menu.actions()
        assert dialog.apply_operation_action in dialog.analysis_menu.actions()
        assert dialog.reference_favourite_action in dialog.reference_menu.actions()
        assert not dialog.fit_action.isEnabled() and not dialog.cancel_fit_action.isEnabled()
        assert not dialog.export_fit_action.isEnabled() and not dialog.report_fit_action.isEnabled()
    dialog.reset_zoom_action.trigger()


def test_energy_menu_routes_commands_and_preserves_unsaved_close_confirmation(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    load, add, apply = Mock(), Mock(), Mock()
    monkeypatch.setattr(EnergyCalibrationDialog, "_load_current", load)
    monkeypatch.setattr(EnergyCalibrationDialog, "_add_point", add)
    monkeypatch.setattr(EnergyCalibrationDialog, "_apply", apply)
    controller = SimpleNamespace(channel=0, channel_label="CH0", energy_calibration=None)
    dialog = EnergyCalibrationDialog([controller])
    qtbot.addWidget(dialog)
    load.reset_mock()
    dialog.replace_action.trigger()
    load.assert_called_once_with(replace=True)
    dialog.overlay_action.trigger()
    assert load.call_args.kwargs == {"replace": False}
    dialog.add_action.trigger()
    add.assert_called_once_with()
    dialog.apply_action.setEnabled(True)
    dialog.apply_action.trigger()
    assert apply.call_count == 1
    dialog.show()
    dialog._workspace().dirty = True
    question = Mock(return_value=QMessageBox.StandardButton.Cancel)
    monkeypatch.setattr(QMessageBox, "question", question)
    dialog.close_action.trigger()
    assert question.call_count == 1 and dialog.isVisible()
    question.return_value = QMessageBox.StandardButton.Discard
    dialog.close_action.trigger()
    assert not dialog.isVisible()


def test_peak_view_menu_synchronizes_counts_scale_and_current_tab(qtbot: QtBot) -> None:
    dialog = McaPeakAnalysisDialog([])
    qtbot.addWidget(dialog)
    assert dialog.linear_action.isChecked() and not dialog.log_action.isChecked()
    dialog.log_action.trigger()
    assert dialog.scale_combo.currentData() == "log"
    assert dialog.log_action.isChecked() and not dialog.linear_action.isChecked()
    dialog.scale_combo.setCurrentIndex(0)
    assert dialog.linear_action.isChecked() and not dialog.log_action.isChecked()
    for index, action in enumerate(dialog.tab_actions):
        action.trigger()
        assert dialog.side_tabs.currentIndex() == index
        assert sum(item.isChecked() for item in dialog.tab_actions) == 1
    dialog.side_tabs.setCurrentIndex(0)
    assert dialog.tab_actions[0].isChecked()


def test_peak_menu_routes_file_operations_and_cancel_to_existing_handlers(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    load, operation, cancel = Mock(), Mock(), Mock()
    monkeypatch.setattr(McaPeakAnalysisDialog, "_load_files", load)
    monkeypatch.setattr(McaPeakAnalysisDialog, "_apply_operation", operation)
    monkeypatch.setattr(McaPeakAnalysisDialog, "_cancel_fit", cancel)
    dialog = McaPeakAnalysisDialog([])
    qtbot.addWidget(dialog)
    dialog.load_action.trigger()
    assert load.call_count == 1
    dialog.apply_operation_action.setEnabled(True)
    dialog.apply_operation_action.trigger()
    assert operation.call_count == 1
    dialog.cancel_fit_action.setEnabled(True)
    dialog.cancel_fit_action.trigger()
    assert cancel.call_count == 1
    dialog.show()
    dialog.activateWindow()
    dialog.raise_()
    qtbot.waitUntil(dialog.isActiveWindow)
    qtbot.keyClick(dialog, Qt.Key.Key_O, Qt.KeyboardModifier.ControlModifier)
    assert load.call_count == 2
    dialog.close_action.trigger()
    assert not dialog.isVisible()


@pytest.mark.parametrize("kind,topic", [("calibration", "energy-calibration"),
                                       ("peaks", "peak-analysis")])
def test_workbench_help_menu_opens_and_reuses_the_correct_guide(
    qtbot: QtBot, kind: str, topic: str,
) -> None:
    dialog = EnergyCalibrationDialog([]) if kind == "calibration" else McaPeakAnalysisDialog([])
    qtbot.addWidget(dialog)
    dialog.menu_bar.guide_action.trigger()
    guide = dialog.menu_bar.help_dialog
    assert guide is not None and guide.isVisible()
    assert guide._history[-1].topic == topic
    assert guide.parentWidget() is dialog
    guide.close()
    dialog.menu_bar.guide_action.trigger()
    assert dialog.menu_bar.help_dialog is guide and guide.isVisible()
