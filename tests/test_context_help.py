# pytest-qt's input helpers intentionally have no type annotations.
# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

import re
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeySequence, QShortcut, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pytestqt.qtbot import QtBot

from nlab.help.content import load_topics, parse_topics
from nlab.help.context import CLASS_TOPICS, FORM_TOPICS, HelpContext, resolve_context
from nlab.help.controller import HelpController
from nlab.views.help_dialog import HelpDialog
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog

type Workspace = tuple[QMainWindow, HelpController, QLabel, QSpinBox, QPushButton]


@pytest.fixture
def workspace(qtbot: QtBot) -> Generator[Workspace, None, None]:
    window = QMainWindow()
    window.resize(550, 350)
    qtbot.addWidget(window)
    form = QWidget(window)
    form.setObjectName("ScopeView")
    window.setCentralWidget(form)
    layout = QVBoxLayout(form)
    label = QLabel("Level (raw):", form)
    label.setObjectName("labelTriggerLevel")
    spin = QSpinBox(form)
    spin.setObjectName("spinTriggerLevel")
    spin.setToolTip("Scope threshold in raw ADC codes")
    button = QPushButton("Start", form)
    button.setObjectName("btnStart")
    layout.addWidget(label)
    layout.addWidget(spin)
    layout.addWidget(button)
    controller = HelpController(window)
    window.show()
    yield window, controller, label, spin, button
    controller.dispose()


def test_reference_has_no_broken_bindings_or_related_links() -> None:
    topics = load_topics()
    assert len(topics) >= 50
    for default, bindings in [*FORM_TOPICS.values(), *CLASS_TOPICS.values()]:
        assert default in topics
        assert set(bindings.values()) <= topics.keys()
    for topic in topics.values():
        links = re.findall(r"\]\(#([a-z0-9-]+)\)", topic.markdown)
        assert set(links) <= topics.keys()
        assert topic.title and topic.markdown
    source = Path(__file__).resolve().parents[1] / "docs/context_help.qmd"
    assert "include ../src/nlab/data/help/parameters.md" in source.read_text(encoding="utf-8")


def test_reference_parser_rejects_missing_home_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="home"):
        parse_topics("## Scope {#scope}\nText")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_topics("## Home {#help}\nText\n## Home again {#help}\nText")


def test_scope_and_mca_label_slider_and_internal_edit_are_scoped(qtbot: QtBot) -> None:
    for scope, expected in (("ScopeView", "scope-trigger"), ("MCAView", "mca-trigger")):
        root = QWidget()
        qtbot.addWidget(root)
        root.setObjectName(scope)
        label = QLabel("Trigger", root)
        label.setObjectName("labelTriggerLevel")
        spin = QSpinBox(root)
        spin.setObjectName("spinTriggerLevel")
        edit = spin.findChild(QLineEdit)
        assert edit is not None
        for item in (label, spin, edit):
            assert resolve_context(item).topic == expected


def test_programmatic_form_label_opens_field_topic(qtbot: QtBot) -> None:
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    labels = dialog.findChildren(QLabel)
    label = next(item for item in labels if item.text() == "Optimization events:")
    assert resolve_context(label).topic == "waveform-optimization"
    assert resolve_context(dialog.optimization_events).topic == "waveform-optimization"
    assert "Maximum sampled" in resolve_context(label).tooltip
    baseline = next(item for item in labels if item.text() == "Baseline estimator:")
    assert resolve_context(baseline).topic == "waveform-baseline"
    period = next(item for item in labels if item.text() == "Sample period:")
    assert resolve_context(period).topic == "waveform-source"
    assert resolve_context(dialog.psd_plot.fom_widget.fit_button).topic == "fom"


def test_explicit_topic_and_honest_unknown_fallback(qtbot: QtBot) -> None:
    root = QWidget()
    qtbot.addWidget(root)
    root.setObjectName("MCAView")
    label = QLabel("Unknown control", root)
    context = resolve_context(label)
    assert context.topic == "mca" and not context.specific
    label.setProperty("nlabHelpTopic", "mca-baseline")
    assert resolve_context(label) == HelpContext("mca-baseline", "Unknown control")


def test_f1_label_click_opens_topic_and_restores_cursor(workspace: Workspace, qtbot: QtBot) -> None:
    window, controller, label, spin, _button = workspace
    before = QApplication.overrideCursor()
    qtbot.keyClick(spin, Qt.Key.Key_F1)
    assert controller.active
    assert QApplication.overrideCursor().shape() == Qt.CursorShape.WhatsThisCursor
    qtbot.mouseClick(label, Qt.MouseButton.LeftButton)
    assert not controller.active
    dialog = controller._dialogs[window]
    assert dialog.isVisible() and dialog.current_topic == "scope-trigger"
    assert "MCA threshold" in dialog.browser.toPlainText()
    assert QApplication.overrideCursor() == before


def test_help_selection_does_not_click_start_but_normal_click_does(
    workspace: Workspace, qtbot: QtBot,
) -> None:
    _window, controller, _label, _spin, button = workspace
    clicked = Mock()
    button.clicked.connect(clicked)
    controller.set_active(True)
    qtbot.mousePress(button, Qt.MouseButton.LeftButton)
    assert controller.active
    clicked.assert_not_called()
    qtbot.mouseRelease(button, Qt.MouseButton.LeftButton)
    clicked.assert_not_called()
    controller._dialogs[_window].hide()
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    clicked.assert_called_once()


def test_keyboard_wheel_and_shortcuts_cannot_edit_during_selection(
    workspace: Workspace, qtbot: QtBot,
) -> None:
    _window, controller, _label, spin, button = workspace
    spin.setValue(12)
    changed = Mock()
    spin.valueChanged.connect(changed)
    shortcut = QShortcut(QKeySequence("Ctrl+J"), spin)
    triggered = Mock()
    shortcut.activated.connect(triggered)
    controller.set_active(True)
    qtbot.keyClick(spin, Qt.Key.Key_Up)
    qtbot.keyClick(spin, Qt.Key.Key_J, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClick(button, Qt.Key.Key_Space)
    wheel = QWheelEvent(
        QPointF(5, 5), QPointF(spin.mapToGlobal(QPoint(5, 5))), QPoint(), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    QApplication.sendEvent(spin, wheel)
    assert spin.value() == 12
    changed.assert_not_called()
    triggered.assert_not_called()
    assert controller.active
    qtbot.keyClick(spin, Qt.Key.Key_Escape)
    assert not controller.active
    qtbot.keyClick(spin, Qt.Key.Key_Up)
    assert spin.value() == 13


@pytest.mark.parametrize("cancel", ["f1", "right", "deactivate"])
def test_cancel_does_not_open_a_topic(workspace: Workspace, qtbot: QtBot, cancel: str) -> None:
    _window, controller, label, spin, _button = workspace
    controller.set_active(True)
    if cancel == "f1":
        qtbot.keyClick(spin, Qt.Key.Key_F1)
    elif cancel == "right":
        qtbot.mouseClick(label, Qt.MouseButton.RightButton)
    else:
        application = QApplication.instance()
        assert application is not None
        QApplication.sendEvent(application, QEvent(QEvent.Type.ApplicationDeactivate))
    assert not controller.active and not controller._dialogs


def test_disabled_control_can_be_explained_without_enabling_it(
    workspace: Workspace, qtbot: QtBot,
) -> None:
    window, controller, _label, spin, _button = workspace
    spin.setEnabled(False)
    controller.set_active(True)
    qtbot.mouseClick(spin, Qt.MouseButton.LeftButton)
    assert not controller.active
    assert controller._dialogs[window].current_topic == "scope-trigger"
    assert not spin.isEnabled() and spin.value() == 0


def test_f1_help_works_inside_owned_modal_dialog(workspace: Workspace, qtbot: QtBot) -> None:
    window, controller, _label, _spin, _button = workspace
    dialog = QDialog(window)
    qtbot.addWidget(dialog)
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setObjectName("MCAView")
    layout = QFormLayout(dialog)
    spin = QSpinBox(dialog)
    spin.setObjectName("spinCcTime")
    layout.addRow("Integration time:", spin)
    dialog.show()
    qtbot.keyClick(spin, Qt.Key.Key_F1)
    qtbot.mouseClick(spin, Qt.MouseButton.LeftButton)
    help_window = controller._dialogs[dialog]
    assert help_window.parentWidget() is dialog
    assert help_window.isVisible() and help_window.current_topic == "mca-charge"
    assert spin.value() == 0


def test_other_workspaces_do_not_steal_f1(workspace: Workspace, qtbot: QtBot) -> None:
    _window, controller, _label, _spin, _button = workspace
    other = QMainWindow()
    qtbot.addWidget(other)
    other.show()
    qtbot.keyClick(other, Qt.Key.Key_F1)
    assert not controller.active


def test_only_one_workspace_owns_the_help_cursor(workspace: Workspace, qtbot: QtBot) -> None:
    _window, controller, _label, _spin, _button = workspace
    other = QMainWindow()
    qtbot.addWidget(other)
    second = HelpController(other)
    controller.set_active(True)
    second.set_active(True)
    assert second.active and not controller.active
    second.dispose()
    assert QApplication.overrideCursor() is None


def test_closing_owned_dialog_cancels_selection(workspace: Workspace, qtbot: QtBot) -> None:
    window, controller, _label, _spin, _button = workspace
    dialog = QDialog(window)
    qtbot.addWidget(dialog)
    dialog.show()
    controller.set_active(True)
    dialog.close()
    assert not controller.active and QApplication.overrideCursor() is None


def test_search_history_links_and_unknown_topic(qtbot: QtBot) -> None:
    dialog = HelpDialog()
    qtbot.addWidget(dialog)
    dialog.show_topic(HelpContext("mca-charge", "Charge Comparison", "Example tooltip"))
    assert "Example tooltip" in dialog.selection.text()
    dialog.search.setText("pole-zero")
    visible = [dialog.contents.item(i).text() for i in range(dialog.contents.count())
               if not dialog.contents.item(i).isHidden()]
    assert any("CR-RC" in title for title in visible)
    assert not any("About" in title for title in visible)
    dialog.show_topic(HelpContext("psd-cut"))
    qtbot.mouseClick(dialog.back_button, Qt.MouseButton.LeftButton)
    assert dialog.current_topic == "mca-charge"
    qtbot.mouseClick(dialog.forward_button, Qt.MouseButton.LeftButton)
    assert dialog.current_topic == "psd-cut"
    dialog.show_topic(HelpContext("missing-topic"))
    assert dialog.current_topic == "help"
    assert "dedicated entry" in dialog.selection.text()


def test_f1_inside_help_reuses_the_existing_browser(workspace: Workspace) -> None:
    window, controller, _label, _spin, _button = workspace
    controller.open_topic(HelpContext("scope"))
    dialog = controller._dialogs[window]
    controller.open_topic(HelpContext("help"), owner=dialog)
    assert len(controller._dialogs) == 1 and dialog.current_topic == "help"


def test_help_load_failure_is_logged_and_visible(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _window, controller, _label, _spin, _button = workspace
    monkeypatch.setattr(
        "nlab.help.controller.HelpDialog", Mock(side_effect=OSError("missing data")),
    )
    warning = Mock()
    monkeypatch.setattr("nlab.help.controller.QMessageBox.warning", warning)
    controller.set_active(True)
    controller.open_topic(HelpContext("help"))
    assert not controller.active
    warning.assert_called_once()


def test_dispose_restores_cursor_and_removes_filter(workspace: Workspace, qtbot: QtBot) -> None:
    _window, controller, _label, spin, _button = workspace
    controller.set_active(True)
    controller.dispose()
    assert not controller.active and QApplication.overrideCursor() is None
    qtbot.keyClick(spin, Qt.Key.Key_F1)
    assert not controller.active


def test_main_window_help_menu_is_available_offline(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nlab.app import MainAppWindow

    hardware_controller = Mock()
    hardware_controller.energy_calibrated_channel_labels.return_value = ()
    monkeypatch.setattr("nlab.app.MainWindowController", Mock(return_value=hardware_controller))
    for method in (
        "_apply_stored_general_settings", "_persist_current_general_settings",
        "_apply_view_state", "_save_developer_settings",
    ):
        monkeypatch.setattr(MainAppWindow, method, lambda self: None)
    window = MainAppWindow(offline=True)
    qtbot.addWidget(window)
    window.show()
    help_controller = window._help_controller
    assert help_controller.guide_action in window.ui.menuHelp.actions()
    assert help_controller.context_action in window.ui.menuHelp.actions()
    help_controller.guide_action.trigger()
    assert help_controller._dialogs[window].current_topic == "help"
    window.close()
    hardware_controller.shutdown.assert_called_once()
    assert help_controller._disposed
