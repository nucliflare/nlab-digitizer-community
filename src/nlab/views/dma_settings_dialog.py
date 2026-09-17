"""Compact application-wide settings for MCA list-mode output."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode

DMA_FOLDER_KEY = "dma/save_folder"
MCA_DMA_OUTPUT_MODE_KEY = "dma/mca_output_mode"


class DmaSettingsDialog(QDialog):
    """Edit the destination folder and one-file-per-measurement MCA format."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("DMA Settings")
        self.setMinimumWidth(460)

        settings = QSettings()
        self._folder = QLineEdit(str(settings.value(DMA_FOLDER_KEY, "measurements")))
        self._folder.setReadOnly(True)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        folder_row = QHBoxLayout()
        folder_row.addWidget(self._folder, 1)
        folder_row.addWidget(browse)

        self._format = QComboBox()
        self._format.addItem("Binary NDMA + YAML settings", McaDmaOutputMode.BINARY.value)
        self._format.addItem("ROOT TTree", McaDmaOutputMode.ROOT.value)
        self._format.addItem("HDF5 (SWMR)", McaDmaOutputMode.HDF5.value)
        self._format.addItem("Online only (no file)", McaDmaOutputMode.ONLINE.value)
        stored_mode = str(
            settings.value(MCA_DMA_OUTPUT_MODE_KEY, McaDmaOutputMode.BINARY.value)
        )
        index = self._format.findData(stored_mode)
        self._format.setCurrentIndex(index if index >= 0 else 0)

        form = QFormLayout()
        form.addRow("Measurement location:", folder_row)
        form.addRow("MCA list-mode output:", self._format)

        note = QLabel(
            "Binary, ROOT, and HDF5 create a new file for every measurement, "
            "with or without Charge Comparison. When Charge Comparison is on, "
            "the live PSD view also receives the events when available. "
            "Online-only never writes a file; "
            "without Charge Comparison its events are discarded."
        )
        note.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Measurement Location",
            self._folder.text(),
        )
        if selected:
            self._folder.setText(selected)

    def accept(self) -> None:
        folder = self._folder.text().strip() or "measurements"
        settings = QSettings()
        settings.setValue(DMA_FOLDER_KEY, str(Path(folder)))
        settings.setValue(MCA_DMA_OUTPUT_MODE_KEY, str(self._format.currentData()))
        super().accept()
