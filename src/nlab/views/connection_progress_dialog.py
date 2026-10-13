"""Compact modal progress dialog used while rebuilding a live workspace."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout, QWidget


class ConnectionProgressDialog(QDialog):
    """Show connection status with an indeterminate full-width progress bar."""

    def __init__(
        self,
        title: str,
        message: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        self.label = QLabel(message, self)
        self.label.setObjectName("connectionProgressLabel")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setObjectName("connectionProgressBar")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)

    def setLabelText(self, message: str) -> None:  # noqa: N802 - Qt-compatible API
        self.label.setText(message)
