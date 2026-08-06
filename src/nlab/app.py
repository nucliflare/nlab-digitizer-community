from __future__ import annotations

import logging

from PySide6.QtCore import QSettings
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from nlab import __version__
from nlab.controllers.main_window_controller import MainWindowController
from nlab.ui.ui_main_window import Ui_MainWindow
from nlab.views.license_dialog import LicenseDialog

_ABOUT_TEXT = f"""\
<b>Nuclear Lab Digitizer — Community Edition</b><br>
Version {__version__}<br>
<br>
A PySide6 desktop application for real-time data acquisition and<br>
pulse-shape analysis with EWT digitizer hardware.<br>
<br>
Organization: EWT<br>
License: MIT<br>
<br>
Source: <a href="https://github.com/ewt/nlab-community">github.com/ewt/nlab-community</a>
"""

_KEY_SHOW_LOG = "developer/show_system_log"
_KEY_DEBUG_MODE = "developer/debug_mode"
_KEY_DMA_FOLDER = "dma/save_folder"
_KEY_SHOW_ROI = "view/show_roi"
_KEY_LOG_Y = "view/log_y"


class MainAppWindow(QMainWindow):
    """Top-level application window. Owns the UI and its controller."""

    def __init__(
        self, backend: str = "grpc", host: str = "", port: int = 50050, channels: int = 2,
    ) -> None:
        super().__init__()
        self._setup_ui()
        self.setWindowIcon(QIcon(":/icons/ewt.ico"))
        # apply_taskbar_icon is intentionally deferred to after show() via
        # main.py's post-show callback — calling it before show() targets a
        # provisional HWND that QMainWindow replaces when it first settles its
        # dock layout on the screen, discarding the pre-show WM_SETICON.
        self.setWindowTitle(f"Nuclear Lab Digitizer — {backend}://{host}:{port}")
        self._controller = MainWindowController(
            self, backend=backend, host=host, port=port, channels=channels,
        )
        self._apply_view_state()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._save_developer_settings()
        self._controller.shutdown()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.resize(1024, 768)

        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionConvertToHdf5.triggered.connect(self._on_convert_to_hdf5)
        self.ui.actionReconnectDevice.triggered.connect(self._on_reconnect_device)
        self.ui.actionResetDocks.triggered.connect(self._on_reset_docks)
        self.ui.actionResetZoom.triggered.connect(self._on_reset_zoom)
        self.ui.actionShowRoi.toggled.connect(self._on_show_roi_toggled)
        self.ui.actionLogY.toggled.connect(self._on_log_y_toggled)
        self.ui.actionSaveSettings.triggered.connect(self._on_save_settings)
        self.ui.actionLoadSettings.triggered.connect(self._on_load_settings)
        self.ui.actionDmaSaveFolder.triggered.connect(self._on_dma_save_folder)
        self.ui.actionAbout.triggered.connect(self._on_about)
        self.ui.actionThirdPartyLicenses.triggered.connect(self._on_third_party_licenses)
        self.ui.actionShowSystemLog.toggled.connect(self._on_show_system_log_toggled)
        self.ui.actionDebugMode.toggled.connect(self._on_debug_mode_toggled)

        self._restore_developer_settings()

    # ------------------------------------------------------------------
    # QSettings persistence for developer panel state
    # ------------------------------------------------------------------

    def _restore_developer_settings(self) -> None:
        settings = QSettings()
        show_log = settings.value(_KEY_SHOW_LOG, False, type=bool)
        debug_mode = settings.value(_KEY_DEBUG_MODE, False, type=bool)
        show_roi = settings.value(_KEY_SHOW_ROI, False, type=bool)
        log_y = settings.value(_KEY_LOG_Y, False, type=bool)

        # Suppress intermediate toggled signals while restoring state.
        self.ui.actionShowSystemLog.blockSignals(True)
        self.ui.actionDebugMode.blockSignals(True)
        self.ui.actionShowRoi.blockSignals(True)
        self.ui.actionLogY.blockSignals(True)

        self.ui.actionShowSystemLog.setChecked(show_log)
        self.ui.actionDebugMode.setChecked(debug_mode)
        self.ui.actionShowRoi.setChecked(show_roi)
        self.ui.actionLogY.setChecked(log_y)
        self._set_log_tab_visible(show_log)
        if debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)

        self.ui.actionShowSystemLog.blockSignals(False)
        self.ui.actionDebugMode.blockSignals(False)
        self.ui.actionShowRoi.blockSignals(False)
        self.ui.actionLogY.blockSignals(False)

    def _save_developer_settings(self) -> None:
        settings = QSettings()
        settings.setValue(_KEY_SHOW_LOG, self.ui.actionShowSystemLog.isChecked())
        settings.setValue(_KEY_DEBUG_MODE, self.ui.actionDebugMode.isChecked())
        settings.setValue(_KEY_SHOW_ROI, self.ui.actionShowRoi.isChecked())
        settings.setValue(_KEY_LOG_Y, self.ui.actionLogY.isChecked())

    def _apply_view_state(self) -> None:
        """Re-apply persisted view toggles to the (re)built MCA controllers."""
        self._controller.set_roi_visible(self.ui.actionShowRoi.isChecked())
        self._controller.set_log_y(self.ui.actionLogY.isChecked())

    def _set_log_tab_visible(self, visible: bool) -> None:
        idx = self.ui.mainTabs.indexOf(self.ui.tabSystemLog)
        self.ui.mainTabs.setTabVisible(idx, visible)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_show_system_log_toggled(self, checked: bool) -> None:
        self._set_log_tab_visible(checked)

    def _on_debug_mode_toggled(self, checked: bool) -> None:
        level = logging.DEBUG if checked else logging.INFO
        logging.getLogger().setLevel(level)
        logging.getLogger(__name__).info("Debug mode %s (log level: %s)",
                                         "enabled" if checked else "disabled",
                                         logging.getLevelName(level))
        if checked and not self.ui.actionShowSystemLog.isChecked():
            self.ui.actionShowSystemLog.setChecked(True)

    def _on_convert_to_hdf5(self) -> None:
        from pathlib import Path
        from nlab.utils.dma_converter import convert_listmode, convert_scope, read_file_header

        src, _ = QFileDialog.getOpenFileName(
            self, "Select Binary DMA File", "", "Binary files (*.bin);;All files (*)",
        )
        if not src:
            return

        dst, _ = QFileDialog.getSaveFileName(
            self, "Save HDF5 File", str(Path(src).with_suffix(".h5")),
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if not dst:
            return

        try:
            with open(src, "rb") as f:
                header = read_file_header(f)

            if header["frame_samples"] > 0:
                n = convert_scope(Path(src), Path(dst))
                QMessageBox.information(self, "Conversion Complete",
                                        f"Converted {n} scope frames to:\n{dst}")
            else:
                n = convert_listmode(Path(src), Path(dst))
                QMessageBox.information(self, "Conversion Complete",
                                        f"Converted {n} listmode events to:\n{dst}")
        except Exception as e:
            logging.getLogger(__name__).exception("HDF5 conversion failed")
            QMessageBox.critical(self, "Conversion Failed", str(e))

    def _on_save_settings(self) -> None:
        from pathlib import Path
        from nlab.utils.settings_io import save_settings

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "", "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return

        try:
            self._controller.save_all_settings(Path(path))
        except Exception as e:
            logging.getLogger(__name__).exception("Failed to save settings")
            QMessageBox.critical(self, "Save Failed", str(e))

    def _on_load_settings(self) -> None:
        from pathlib import Path
        from nlab.utils.settings_io import load_settings

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return

        try:
            self._controller.load_all_settings(Path(path))
        except Exception as e:
            logging.getLogger(__name__).exception("Failed to load settings")
            QMessageBox.critical(self, "Load Failed", str(e))

    def _on_show_roi_toggled(self, checked: bool) -> None:
        self._controller.set_roi_visible(checked)

    def _on_log_y_toggled(self, checked: bool) -> None:
        self._controller.set_log_y(checked)

    def _on_reset_zoom(self) -> None:
        self._controller.reset_all_zoom()

    def _on_reconnect_device(self) -> None:
        reply = QMessageBox.question(
            self, "Reconnect Device",
            "This will stop all running measurements and re-establish the "
            "device connection. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self._controller.reconnect()
            self._apply_view_state()
            QMessageBox.information(self, "Reconnect", "Device reconnected successfully.")
        except Exception as e:
            logging.getLogger(__name__).exception("Reconnect failed")
            QMessageBox.critical(self, "Reconnect Failed", str(e))

    def _on_reset_docks(self) -> None:
        self._controller.reset_dock_layout()

    def _on_dma_save_folder(self) -> None:
        current = QSettings().value(_KEY_DMA_FOLDER, "measurements")
        folder = QFileDialog.getExistingDirectory(self, "DMA Save Folder", str(current))
        if folder:
            QSettings().setValue(_KEY_DMA_FOLDER, folder)
            logging.getLogger(__name__).info("DMA save folder set to: %s", folder)

    def _on_about(self) -> None:
        QMessageBox.about(self, "About Nuclear Lab Digitizer", _ABOUT_TEXT)

    def _on_third_party_licenses(self) -> None:
        LicenseDialog(self).exec()
