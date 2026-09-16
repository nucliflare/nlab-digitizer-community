from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QProcess, QSettings, QStandardPaths, QTimer
from PySide6.QtGui import QCloseEvent, QIcon, QScreen, QShowEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

from nlab import __version__
from nlab.controllers.main_window_controller import MainWindowController
from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode
from nlab.ui.ui_main_window import Ui_MainWindow
from nlab.utils.remote_board_power import (
    BoardPowerCommand,
    power_command_was_delivered,
    ssh_arguments,
)
from nlab.views.dma_settings_dialog import (
    DMA_FOLDER_KEY,
    MCA_DMA_OUTPUT_MODE_KEY,
    DmaSettingsDialog,
)
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
_KEY_DMA_FOLDER = DMA_FOLDER_KEY
_KEY_MCA_DMA_OUTPUT_MODE = MCA_DMA_OUTPUT_MODE_KEY
_KEY_SHOW_ROI = "view/show_roi"
_KEY_LOG_Y = "view/log_y"


class MainAppWindow(QMainWindow):
    """Top-level application window. Owns the UI and its controller."""

    def __init__(
        self,
        backend: str = "grpc",
        host: str = "",
        port: int = 50050,
        channels: int = 2,
        config_path: Path | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._host = host
        self._screen_change_connected = False
        self._board_power_process: QProcess | None = None
        self._board_power_command: BoardPowerCommand | None = None
        self._board_power_phase: str | None = None
        if on_progress is not None:
            on_progress("Preparing the main window...")
        self._setup_ui()
        self.setWindowIcon(QIcon(":/icons/ewt.ico"))
        # apply_taskbar_icon is intentionally deferred to after show() via
        # main.py's post-show callback — calling it before show() targets a
        # provisional HWND that QMainWindow replaces when it first settles its
        # dock layout on the screen, discarding the pre-show WM_SETICON.
        self.setWindowTitle(f"Nuclear Lab Digitizer — {backend}://{host}:{port}")
        self._controller = MainWindowController(
            self,
            backend=backend,
            host=host,
            port=port,
            channels=channels,
            on_progress=on_progress,
        )
        if config_path is not None:
            if on_progress is not None:
                on_progress("Applying saved settings...")
            self._controller.load_all_settings(config_path)
        if on_progress is not None:
            on_progress("Restoring the workspace...")
        self._apply_view_state()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._save_developer_settings()
        self._controller.shutdown()
        super().closeEvent(event)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        window = self.windowHandle()
        if window is not None and not self._screen_change_connected:
            window.screenChanged.connect(self._on_screen_changed)
            self._screen_change_connected = True

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._resize_for_available_screen()

        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionOpenPsdEvents.triggered.connect(self._on_open_psd_events)
        self.ui.actionOpenWaveformFile.triggered.connect(self._on_open_waveform_file)
        self.ui.actionConvertToHdf5.triggered.connect(self._on_convert_to_hdf5)
        self.ui.actionReconnectDevice.triggered.connect(self._on_reconnect_device)
        self.ui.actionResetDocks.triggered.connect(self._on_reset_docks)
        self.ui.actionResetZoom.triggered.connect(self._on_reset_zoom)
        self.ui.actionShowRoi.toggled.connect(self._on_show_roi_toggled)
        self.ui.actionLogY.toggled.connect(self._on_log_y_toggled)
        self.ui.actionSaveSettings.triggered.connect(self._on_save_settings)
        self.ui.actionLoadSettings.triggered.connect(self._on_load_settings)
        self.ui.actionDmaSaveFolder.triggered.connect(self._on_dma_settings)
        self.ui.actionAbout.triggered.connect(self._on_about)
        self.ui.actionThirdPartyLicenses.triggered.connect(self._on_third_party_licenses)
        self.ui.actionShowSystemLog.toggled.connect(self._on_show_system_log_toggled)
        self.ui.actionDebugMode.toggled.connect(self._on_debug_mode_toggled)
        self.ui.actionRebootBoard.triggered.connect(self._on_reboot_board)
        self.ui.actionShutdownBoard.triggered.connect(self._on_shutdown_board)

        power_actions_enabled = bool(self._host.strip())
        self.ui.actionRebootBoard.setEnabled(power_actions_enabled)
        self.ui.actionShutdownBoard.setEnabled(power_actions_enabled)

        self._restore_developer_settings()

    def _resize_for_available_screen(self, screen: QScreen | None = None) -> None:
        """Choose a useful initial size without exceeding the desktop.

        A fixed 1024x768 client window cannot fit on a 1280x720 desktop once
        the title bar and taskbar are accounted for. Keep the comfortable
        1280x800 target used on larger displays, but reserve a small margin on
        compact screens. Dense views provide their own control-panel scrolling
        rather than forcing the top-level window beyond this size.
        """
        screen = screen or self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1024, 640)
            return

        available = screen.availableGeometry()
        margin = 32
        width = max(800, min(1280, available.width() - margin))
        height = max(480, min(800, available.height() - margin))
        self.resize(width, height)

    def _on_screen_changed(self, screen: QScreen) -> None:
        if not self.isMaximized() and not self.isFullScreen():
            self._resize_for_available_screen(screen)
            # Child views receive the same screenChanged signal and may lower
            # their layout minimums in response. Retry on the next event-loop
            # turn so the old large-screen minimum cannot reject this resize.
            QTimer.singleShot(0, self._resize_after_screen_layout_change)

    def _resize_after_screen_layout_change(self) -> None:
        if not self.isMaximized() and not self.isFullScreen():
            self._resize_for_available_screen()

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

    def configuration_settings(self) -> dict[str, object]:
        """Return top-level controls that are owned only by the application."""
        return {
            "show_system_log": self.ui.actionShowSystemLog.isChecked(),
            "debug_mode": self.ui.actionDebugMode.isChecked(),
            "show_roi": self.ui.actionShowRoi.isChecked(),
            "log_y": self.ui.actionLogY.isChecked(),
            "dma_save_folder": str(QSettings().value(_KEY_DMA_FOLDER, "measurements")),
            "mca_dma_output_mode": str(
                QSettings().value(
                    _KEY_MCA_DMA_OUTPUT_MODE,
                    McaDmaOutputMode.BINARY.value,
                )
            ),
            "active_tab": self.ui.mainTabs.currentIndex(),
        }

    def apply_configuration_settings(self, settings: object) -> None:
        """Populate top-level GUI controls from the YAML application section."""
        if not isinstance(settings, dict):
            return
        actions = {
            "show_system_log": self.ui.actionShowSystemLog,
            "debug_mode": self.ui.actionDebugMode,
            "show_roi": self.ui.actionShowRoi,
            "log_y": self.ui.actionLogY,
        }
        for name, action in actions.items():
            if name in settings:
                action.setChecked(bool(settings[name]))
        if "dma_save_folder" in settings:
            QSettings().setValue(_KEY_DMA_FOLDER, str(settings["dma_save_folder"]))
        requested_mode = str(
            settings.get("mca_dma_output_mode", McaDmaOutputMode.BINARY.value)
        )
        try:
            mode = McaDmaOutputMode(requested_mode)
        except ValueError:
            logging.getLogger(__name__).warning(
                "Unknown MCA DMA output mode %r; using binary", requested_mode
            )
            mode = McaDmaOutputMode.BINARY
        QSettings().setValue(_KEY_MCA_DMA_OUTPUT_MODE, mode.value)
        if "active_tab" in settings:
            index = int(settings["active_tab"])
            if 0 <= index < self.ui.mainTabs.count():
                self.ui.mainTabs.setCurrentIndex(index)
        self._save_developer_settings()
        self._apply_view_state()
        self._controller.refresh_dma_output_settings()

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

    def _on_open_psd_events(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open PSD Event File",
            str(QSettings().value(_KEY_DMA_FOLDER, "measurements")),
            (
                "PSD event files (*.bin *.h5 *.hdf5 *.root);;"
                "Binary event files (*.bin);;HDF5 (*.h5 *.hdf5);;ROOT (*.root);;"
                "All files (*)"
            ),
        )
        if not path:
            return
        try:
            self._controller.load_psd_events(Path(path))
        except Exception as exc:
            logging.getLogger(__name__).exception("Failed to open PSD event file")
            QMessageBox.critical(self, "PSD File Load Failed", str(exc))

    def _on_open_waveform_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Waveform File",
            str(QSettings().value(_KEY_DMA_FOLDER, "measurements")),
            "Waveform binaries (*.bin *.BIN);;All files (*)",
        )
        if not path:
            return
        try:
            self._controller.load_waveform_file(Path(path))
        except Exception as exc:
            logging.getLogger(__name__).exception("Failed to open waveform file")
            QMessageBox.critical(self, "Waveform File Load Failed", str(exc))

    def _on_show_system_log_toggled(self, checked: bool) -> None:
        self._set_log_tab_visible(checked)

    def _on_debug_mode_toggled(self, checked: bool) -> None:
        level = logging.DEBUG if checked else logging.INFO
        logging.getLogger().setLevel(level)
        logging.getLogger(__name__).info(
            "Debug mode %s (log level: %s)",
            "enabled" if checked else "disabled",
            logging.getLevelName(level),
        )
        if checked and not self.ui.actionShowSystemLog.isChecked():
            self.ui.actionShowSystemLog.setChecked(True)

    def _on_reboot_board(self) -> None:
        self._request_board_power_action("reboot")

    def _on_shutdown_board(self) -> None:
        self._request_board_power_action("shutdown")

    def _request_board_power_action(self, command: BoardPowerCommand) -> None:
        if self._board_power_process is not None:
            QMessageBox.information(
                self,
                "Remote Board",
                "A remote board power request is already running.",
            )
            return

        if command == "reboot":
            title = "Reboot Remote Board"
            prompt = (
                f"Stop all measurements and reboot {self._host}?\n\n"
                "The application will remain disconnected until the board has "
                "booted and Reconnect Device is selected."
            )
        else:
            title = "Shut Down Remote Board"
            prompt = (
                f"Stop all measurements and power off {self._host}?\n\n"
                "The board will remain unavailable until its power is restored."
            )

        reply = QMessageBox.warning(
            self,
            title,
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        key_path = Path.home() / ".ssh" / "nlab_board_power_ed25519"
        if not key_path.is_file():
            QMessageBox.critical(
                self,
                title,
                f"SSH key not found:\n{key_path}\n\n"
                "Create the restricted nlab-reboot key before using this action.",
            )
            return

        ssh_program = QStandardPaths.findExecutable("ssh")
        if not ssh_program:
            QMessageBox.critical(
                self,
                title,
                "OpenSSH client 'ssh' was not found in PATH.",
            )
            return

        self._board_power_command = command
        self._board_power_phase = "check"
        self._set_board_power_actions_enabled(False)
        self.statusBar().showMessage(f"Checking remote power access to {self._host}...")
        self._start_board_ssh_process(ssh_program, key_path, "check")

    def _start_board_ssh_process(
        self,
        ssh_program: str,
        key_path: Path,
        remote_command: str,
    ) -> None:
        process = QProcess(self)
        process.setProgram(ssh_program)
        process.setArguments(ssh_arguments(self._host, key_path, remote_command))
        process.finished.connect(self._on_board_ssh_finished)
        process.errorOccurred.connect(self._on_board_ssh_error)
        self._board_power_process = process
        process.start()

    def _on_board_ssh_finished(
        self,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        process = self._board_power_process
        if process is None:
            return

        stdout = process.readAllStandardOutput().data().decode(errors="replace").strip()
        stderr = process.readAllStandardError().data().decode(errors="replace").strip()
        phase = self._board_power_phase
        command = self._board_power_command
        ssh_program = process.program()
        key_path = Path.home() / ".ssh" / "nlab_board_power_ed25519"
        process.deleteLater()
        self._board_power_process = None

        if phase == "check":
            check_ok = (
                exit_status == QProcess.ExitStatus.NormalExit
                and exit_code == 0
                and "nlab-power-command: READY" in stdout
            )
            if not check_ok or command is None:
                detail = stderr or stdout or f"ssh exited with code {exit_code}"
                self._finish_board_power_request(
                    error=f"Remote power access check failed:\n{detail}",
                )
                return

            try:
                self._controller.prepare_for_remote_power_action()
            except Exception as exc:
                logging.getLogger(__name__).exception(
                    "Failed to stop acquisition before remote board power action"
                )
                self._finish_board_power_request(
                    error=f"Could not stop active measurements safely:\n{exc}",
                )
                return

            self.ui.mainTabs.setEnabled(False)
            self._board_power_phase = "command"
            self.statusBar().showMessage(f"Sending {command} command to {self._host}...")
            self._start_board_ssh_process(ssh_program, key_path, command)
            return

        delivered = exit_status == QProcess.ExitStatus.NormalExit and power_command_was_delivered(
            exit_code, stderr
        )
        if not delivered or command is None:
            detail = stderr or stdout or f"ssh exited with code {exit_code}"
            self._finish_board_power_request(
                error=f"Remote board command failed:\n{detail}\n\n"
                "Acquisition is stopped; use Reconnect Device before continuing.",
            )
            return

        if command == "reboot":
            message = (
                "Reboot command was sent. Wait for the board to boot, then select "
                "File > Reconnect Device."
            )
        else:
            message = (
                "Shutdown command was sent. Restore board power before using Reconnect Device."
            )
        logging.getLogger(__name__).info("Remote board %s command delivered", command)
        self._finish_board_power_request(message=message)

    def _on_board_ssh_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart:
            return
        process = self._board_power_process
        detail = process.errorString() if process is not None else "unknown process error"
        if process is not None:
            process.deleteLater()
        self._board_power_process = None
        self._finish_board_power_request(error=f"Could not start ssh:\n{detail}")

    def _finish_board_power_request(
        self,
        *,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        command = self._board_power_command
        self._board_power_command = None
        self._board_power_phase = None
        self._set_board_power_actions_enabled(bool(self._host.strip()))
        self.statusBar().clearMessage()

        title = "Remote Board"
        if error is not None:
            logging.getLogger(__name__).error("Remote board power request failed: %s", error)
            QMessageBox.critical(self, title, error)
        elif message is not None:
            action = command or "power"
            self.statusBar().showMessage(f"Remote board {action} requested", 10000)
            QMessageBox.information(self, title, message)

    def _set_board_power_actions_enabled(self, enabled: bool) -> None:
        self.ui.actionRebootBoard.setEnabled(enabled)
        self.ui.actionShutdownBoard.setEnabled(enabled)

    def _on_convert_to_hdf5(self) -> None:
        from nlab.utils.dma_converter import convert_listmode, convert_scope, read_file_header

        src, _ = QFileDialog.getOpenFileName(
            self,
            "Select Binary DMA File",
            "",
            "Binary files (*.bin);;All files (*)",
        )
        if not src:
            return

        dst, _ = QFileDialog.getSaveFileName(
            self,
            "Save HDF5 File",
            str(Path(src).with_suffix(".h5")),
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if not dst:
            return

        try:
            with open(src, "rb") as f:
                header = read_file_header(f)

            if header["frame_samples"] > 0:
                n = convert_scope(Path(src), Path(dst))
                QMessageBox.information(
                    self, "Conversion Complete", f"Converted {n} scope frames to:\n{dst}"
                )
            else:
                n = convert_listmode(Path(src), Path(dst))
                QMessageBox.information(
                    self, "Conversion Complete", f"Converted {n} listmode events to:\n{dst}"
                )
        except Exception as e:
            logging.getLogger(__name__).exception("HDF5 conversion failed")
            QMessageBox.critical(self, "Conversion Failed", str(e))

    def _on_save_settings(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Settings",
            "",
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return

        try:
            self._controller.save_all_settings(Path(path))
        except Exception as e:
            logging.getLogger(__name__).exception("Failed to save settings")
            QMessageBox.critical(self, "Save Failed", str(e))

    def _on_load_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Settings",
            "",
            "YAML files (*.yaml *.yml);;All files (*)",
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
            self,
            "Reconnect Device",
            "This will stop all running measurements and re-establish the "
            "device connection. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self._controller.reconnect()
            self._apply_view_state()
            self.ui.mainTabs.setEnabled(True)
            QMessageBox.information(self, "Reconnect", "Device reconnected successfully.")
        except Exception as e:
            logging.getLogger(__name__).exception("Reconnect failed")
            QMessageBox.critical(self, "Reconnect Failed", str(e))

    def _on_reset_docks(self) -> None:
        self._controller.reset_dock_layout()

    def _on_dma_settings(self) -> None:
        if DmaSettingsDialog(self).exec():
            self._controller.refresh_dma_output_settings()
            settings = QSettings()
            logging.getLogger(__name__).info(
                "DMA settings updated: folder=%s, MCA output=%s",
                settings.value(_KEY_DMA_FOLDER, "measurements"),
                settings.value(_KEY_MCA_DMA_OUTPUT_MODE, McaDmaOutputMode.BINARY.value),
            )

    def _on_about(self) -> None:
        QMessageBox.about(self, "About Nuclear Lab Digitizer", _ABOUT_TEXT)

    def _on_third_party_licenses(self) -> None:
        LicenseDialog(self).exec()
