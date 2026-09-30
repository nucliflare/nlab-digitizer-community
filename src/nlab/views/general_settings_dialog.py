"""Application-wide settings that do not belong to one hardware channel."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode

AUTO_CONFIGURATION_ENABLED_KEY = "configuration/auto_save_enabled"
AUTO_CONFIGURATION_DIRECTORY = "device-configurations"


def _integer_setting(
    values: Mapping[str, object],
    name: str,
    default: int,
) -> int:
    value = values.get(name, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            pass
    return default


def auto_configuration_enabled() -> bool:
    """Return whether the last complete configuration should be remembered."""
    return bool(QSettings().value(AUTO_CONFIGURATION_ENABLED_KEY, False, type=bool))


def set_auto_configuration_enabled(enabled: bool) -> None:
    """Persist the user's local auto-save preference."""
    QSettings().setValue(AUTO_CONFIGURATION_ENABLED_KEY, enabled)


def _configuration_root() -> Path:
    location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation
    )
    if not location:
        location = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppLocalDataLocation
        )
    return Path(location)


def auto_configuration_path(device_address: str) -> Path:
    """Return a stable, filesystem-safe snapshot path for one device address."""
    normalized = device_address.strip().casefold()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    normalized = normalized or "local-device"
    readable = re.sub(r"[^a-z0-9._-]+", "-", normalized).strip(".-_")
    readable = (readable or "device")[:48]
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    filename = f"{readable}-{digest}.yaml"
    return _configuration_root() / AUTO_CONFIGURATION_DIRECTORY / filename


class GeneralSettingsDialog(QDialog):
    """Edit settings whose scope is the whole application."""

    def __init__(
        self,
        device_address: str,
        values: Mapping[str, object],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("General Settings")
        self.setMinimumWidth(500)

        scope_group = QGroupBox("Scope display")
        scope_form = QFormLayout(scope_group)
        self.scope_display_mode = QComboBox()
        self.scope_display_mode.addItem("Persistence", 0)
        self.scope_display_mode.addItem("Raw", 1)
        display_index = self.scope_display_mode.findData(
            _integer_setting(values, "scope_display_mode", 0)
        )
        self.scope_display_mode.setCurrentIndex(display_index if display_index >= 0 else 0)
        self.scope_display_mode.setToolTip(
            "Applies the selected waveform display mode to every Scope channel."
        )
        self.scope_persistence = QDoubleSpinBox()
        self.scope_persistence.setRange(0.0, 0.999)
        self.scope_persistence.setDecimals(3)
        self.scope_persistence.setSingleStep(0.01)
        self.scope_persistence.setValue(
            _integer_setting(values, "scope_persistence", 900) / 1000.0
        )
        self.scope_persistence.setToolTip(
            "Controls waveform decay in Persistence mode for every Scope channel."
        )
        self.scope_refresh = QSpinBox()
        self.scope_refresh.setRange(1, 60)
        self.scope_refresh.setSuffix(" Hz")
        self.scope_refresh.setValue(
            _integer_setting(values, "scope_refresh_rate_hz", 10)
        )
        self.scope_refresh.setToolTip(
            "Sets the live waveform request and redraw rate for every Scope channel."
        )
        scope_form.addRow("Mode:", self.scope_display_mode)
        scope_form.addRow("Persistence:", self.scope_persistence)
        scope_form.addRow("Refresh:", self.scope_refresh)

        acquisition_group = QGroupBox("Acquisition and monitoring")
        acquisition_form = QFormLayout(acquisition_group)
        self.mca_refresh = QSpinBox()
        self.mca_refresh.setRange(1, 60)
        self.mca_refresh.setSuffix(" Hz")
        self.mca_refresh.setValue(_integer_setting(values, "mca_refresh_rate_hz", 5))
        self.mca_refresh.setToolTip(
            "Sets waveform, spectrum, and statistics updates for every MCA channel."
        )
        self.psu_refresh = QSpinBox()
        self.psu_refresh.setRange(100, 10_000)
        self.psu_refresh.setSingleStep(100)
        self.psu_refresh.setSuffix(" ms")
        self.psu_refresh.setValue(
            _integer_setting(values, "psu_refresh_interval_ms", 1000)
        )
        self.psu_refresh.setToolTip(
            "Sets the voltage and temperature polling interval for every Power Supply channel."
        )
        self.psu_history = QSpinBox()
        self.psu_history.setRange(10, 3600)
        self.psu_history.setSuffix(" s")
        self.psu_history.setValue(
            _integer_setting(values, "psu_plot_time_range_s", 60)
        )
        self.psu_history.setToolTip(
            "Sets the visible voltage-history duration for every Power Supply channel."
        )
        acquisition_form.addRow("MCA refresh:", self.mca_refresh)
        acquisition_form.addRow("PSU refresh:", self.psu_refresh)
        acquisition_form.addRow("PSU history:", self.psu_history)

        files_group = QGroupBox("Acquisition files")
        files_form = QFormLayout(files_group)
        self.dma_folder = QLineEdit(str(values.get("dma_save_folder", "measurements")))
        self.dma_folder.setReadOnly(True)
        self.dma_folder.setToolTip(
            "Default destination for Scope, MCA, PSD, and coincidence recordings."
        )
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_dma_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.dma_folder, 1)
        folder_row.addWidget(browse)
        self.mca_output = QComboBox()
        self.mca_output.addItem("Binary NDMA + YAML", McaDmaOutputMode.BINARY.value)
        self.mca_output.addItem("ROOT TTree", McaDmaOutputMode.ROOT.value)
        self.mca_output.addItem("HDF5 (SWMR)", McaDmaOutputMode.HDF5.value)
        self.mca_output.addItem("Online only", McaDmaOutputMode.ONLINE.value)
        output_mode = str(
            values.get("mca_dma_output_mode", McaDmaOutputMode.BINARY.value)
        )
        output_index = self.mca_output.findData(output_mode)
        self.mca_output.setCurrentIndex(output_index if output_index >= 0 else 0)
        self.mca_output.setToolTip(
            "Selects the application-wide MCA list-mode and coincidence output format."
        )
        files_form.addRow("Destination:", folder_row)
        files_form.addRow("MCA output:", self.mca_output)

        histogram_group = QGroupBox("Histograms")
        histogram_layout = QVBoxLayout(histogram_group)
        self.show_roi = QCheckBox("Show ROI")
        self.show_roi.setChecked(bool(values.get("show_roi", False)))
        self.show_roi.setToolTip(
            "Shows the ROI selector and statistics on every MCA histogram."
        )
        self.log_y = QCheckBox("Logarithmic Y axis")
        self.log_y.setChecked(bool(values.get("log_y", False)))
        self.log_y.setToolTip("Uses a logarithmic Y axis on every MCA histogram.")
        histogram_layout.addWidget(self.show_roi)
        histogram_layout.addWidget(self.log_y)

        configuration_group = QGroupBox("Configuration persistence")
        configuration_form = QFormLayout(configuration_group)

        self.auto_configuration = QCheckBox(
            "Remember device configuration"
        )
        self.auto_configuration.setChecked(auto_configuration_enabled())
        self.auto_configuration.setToolTip(
            "Save the current complete Scope, MCA, and Power Supply configuration "
            "immediately and whenever the application closes cleanly. Restore only "
            "the snapshot belonging to this device address at startup. An explicit "
            "--config file takes precedence, and restore failures do not prevent the "
            "application from opening."
        )
        snapshot_path = auto_configuration_path(device_address)
        self.snapshot_path = QLineEdit(str(snapshot_path))
        self.snapshot_path.setReadOnly(True)
        self.snapshot_path.setToolTip(str(snapshot_path))
        configuration_form.addRow(self.auto_configuration)
        configuration_form.addRow("Snapshot:", self.snapshot_path)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(scope_group)
        layout.addWidget(acquisition_group)
        layout.addWidget(files_group)
        layout.addWidget(histogram_group)
        layout.addWidget(configuration_group)
        layout.addWidget(buttons)

    @property
    def auto_configuration_is_enabled(self) -> bool:
        return self.auto_configuration.isChecked()

    @property
    def settings(self) -> dict[str, object]:
        return {
            "scope_display_mode": int(self.scope_display_mode.currentData()),
            "scope_persistence": round(self.scope_persistence.value() * 1000),
            "scope_refresh_rate_hz": self.scope_refresh.value(),
            "mca_refresh_rate_hz": self.mca_refresh.value(),
            "psu_refresh_interval_ms": self.psu_refresh.value(),
            "psu_plot_time_range_s": self.psu_history.value(),
            "dma_save_folder": self.dma_folder.text().strip() or "measurements",
            "mca_dma_output_mode": str(self.mca_output.currentData()),
            "show_roi": self.show_roi.isChecked(),
            "log_y": self.log_y.isChecked(),
        }

    def _browse_dma_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Measurement Location",
            self.dma_folder.text(),
        )
        if selected:
            self.dma_folder.setText(str(Path(selected)))
