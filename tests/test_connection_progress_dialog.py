from __future__ import annotations

from pytestqt.qtbot import QtBot

from nlab.views.connection_progress_dialog import ConnectionProgressDialog


def test_progress_bar_spans_dialog_content_width(qtbot: QtBot) -> None:
    dialog = ConnectionProgressDialog("Adding Digitizer", "Preparing channel views...")
    qtbot.addWidget(dialog)
    dialog.resize(420, dialog.sizeHint().height())
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)
    assert dialog.layout() is not None
    dialog.layout().activate()

    content_rect = dialog.layout().contentsRect()
    bar_rect = dialog.progress_bar.geometry()
    assert bar_rect.left() == content_rect.left()
    assert bar_rect.right() == content_rect.right()
    assert dialog.minimumWidth() == 320


def test_progress_message_can_be_updated(qtbot: QtBot) -> None:
    dialog = ConnectionProgressDialog("Adding Digitizer", "Connecting...")
    qtbot.addWidget(dialog)

    dialog.setLabelText("Preparing channel views...")

    assert dialog.label.text() == "Preparing channel views..."
