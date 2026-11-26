from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import cast

from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QIcon, QKeyEvent, QMouseEvent, QScreen, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
)

from nlab import __version__, settings_keys
from nlab.controllers.main_window_controller import MainWindowController
from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode
from nlab.hardware.modbus_devices import ExternalDeviceScan
from nlab.help.controller import HelpController
from nlab.session.replay import SessionReplay
from nlab.ui.ui_main_window import Ui_MainWindow
from nlab.utils import recording_paths
from nlab.utils.github_releases import GITHUB_REPOSITORY_URL
from nlab.views import theme
from nlab.views.about_dialog import AboutDialog
from nlab.views.app_events import application_event_hub
from nlab.views.connection_dialog import ConnectionDialog
from nlab.views.connection_progress_dialog import ConnectionProgressDialog
from nlab.views.dma_settings_dialog import (
    DMA_FOLDER_KEY,
    MCA_DMA_OUTPUT_MODE_KEY,
)
from nlab.views.general_settings_dialog import (
    GeneralSettingsDialog,
    auto_configuration_enabled,
    auto_configuration_path,
    set_auto_configuration_enabled,
)
from nlab.views.license_dialog import LicenseDialog
from nlab.views.numeric_controls import install_numeric_input_policy
from nlab.views.timing_validation_dialog import TimingValidationDialog
from nlab.views.tooltips import install_tooltip_wrapping

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
Source: <a href="{GITHUB_REPOSITORY_URL}">github.com/nucliflare/nlab-digitizer-community</a>
"""

_KEY_SHOW_LOG = settings_keys.DEVELOPER_SHOW_SYSTEM_LOG
_KEY_DEBUG_MODE = settings_keys.DEVELOPER_DEBUG_MODE
_KEY_DMA_FOLDER = DMA_FOLDER_KEY
_KEY_MCA_DMA_OUTPUT_MODE = MCA_DMA_OUTPUT_MODE_KEY
_KEY_TIMESTAMP_CHOSEN_NAMES = settings_keys.RECORDING_TIMESTAMP_CHOSEN_NAMES
_KEY_SHOW_ROI = settings_keys.VIEW_SHOW_ROI
_KEY_DIFFERENTIAL_HISTOGRAM = settings_keys.VIEW_DIFFERENTIAL_HISTOGRAM
_KEY_LOG_Y = settings_keys.VIEW_LOG_Y
_KEY_SCOPE_DISTRIBUTION = settings_keys.VIEW_SCOPE_DISTRIBUTION
_KEY_SCOPE_SPECTRUM = settings_keys.VIEW_SCOPE_SPECTRUM
_KEY_MEASUREMENT_SERIES = settings_keys.MEASUREMENT_SERIES_ENABLED
_KEY_TIMING_OFFSET_NS = settings_keys.TIMING_CHANNEL_B_OFFSET_NS
_KEY_SESSION_FOLDER = settings_keys.SESSION_LAST_FOLDER
_GENERAL_CONTROLLER_KEYS = {
    "scope_display_mode": "general/scope_display_mode",
    "scope_persistence": "general/scope_persistence",
    "scope_refresh_rate_hz": "general/scope_refresh_rate_hz",
    "mca_refresh_rate_hz": "general/mca_refresh_rate_hz",
    "psu_refresh_interval_ms": "general/psu_refresh_interval_ms",
    "psu_plot_time_range_s": "general/psu_plot_time_range_s",
}


class _EnergyCalibrationIndicator(QLabel):
    """Status-bar badge that opens the calibration workspace on double-click."""

    double_clicked = Signal()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class MainAppWindow(QMainWindow):
    """Top-level application window. Owns the UI and its controller."""

    _modbus_refresh_finished = Signal(object, object)

    def __init__(
        self,
        host: str = "",
        port: int = 30431,
        channels: int = 2,
        config_path: Path | None = None,
        on_progress: Callable[[str], None] | None = None,
        offline: bool = False,
    ) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._channels = channels
        self._config_path = config_path
        self._offline = bool(offline)
        self._closing = False
        self._connection_busy = False
        # One reconnect prompt per lost connection: every panel polling the
        # dead board reports the loss within a second or two.
        self._connection_loss_prompted = False
        self._modbus_refresh_running = False
        self._screen_change_connected = False
        self._timing_validation_dialog: TimingValidationDialog | None = None
        self._base_window_title = (
            "Nuclear Lab Digitizer — Offline"
            if self._offline
            else f"Nuclear Lab Digitizer — {host}:{port}"
        )
        if on_progress is not None:
            on_progress("Preparing the main window...")
        self._setup_ui()
        self.setWindowIcon(QIcon(":/icons/ewt.ico"))
        # apply_taskbar_icon is intentionally deferred to after show() via
        # main.py's post-show callback — calling it before show() targets a
        # provisional HWND that QMainWindow replaces when it first settles its
        # dock layout on the screen, discarding the pre-show WM_SETICON.
        self.setWindowTitle(self._base_window_title)
        self._controller = MainWindowController(
            self,
            host=host,
            port=port,
            channels=channels,
            on_progress=on_progress,
            offline=self._offline,
        )
        self.refresh_energy_calibration_indicator()
        self.refresh_measurement_indicator()
        event_hub = application_event_hub()
        if event_hub is not None:
            event_hub.register(self, self.eventFilter, events=(QEvent.Type.KeyPress,))
        self._apply_stored_general_settings()
        if config_path is not None and not self._offline:
            if on_progress is not None:
                on_progress("Applying saved settings...")
            self._controller.load_all_settings(config_path)
        elif not self._offline and auto_configuration_enabled():
            self._restore_auto_configuration(on_progress)
        self._persist_current_general_settings()
        if on_progress is not None:
            on_progress("Restoring the workspace...")
        self._apply_view_state()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._closing = True
        help_controller = getattr(self, "_help_controller", None)
        if help_controller is not None:
            help_controller.dispose()
        event_hub = application_event_hub()
        if event_hub is not None:
            event_hub.unregister(self)
        if not getattr(self, "_offline", False) and auto_configuration_enabled():
            self._save_auto_configuration(show_error=False)
        self._save_developer_settings()
        self._controller.shutdown()
        super().closeEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Route log shortcuts to the histogram under the pointer."""
        # Key-press handler on the application event hub (nlab.views.app_events).
        if event.type() != QEvent.Type.KeyPress:
            return False
        if self._try_scope_distribution_log_shortcut(event):
            return True
        if self._try_mca_log_shortcut(event):
            return True
        return super().eventFilter(watched, event)

    def _try_scope_distribution_log_shortcut(self, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.KeyPress
            and self.isVisible()
            and isinstance(event, QKeyEvent)
            and event.key() == Qt.Key.Key_L
            and event.modifiers()
            in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier)
            and not event.isAutoRepeat()
        ):
            return self._controller.toggle_scope_distribution_log_under_mouse()
        return False

    def _try_mca_log_shortcut(self, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.KeyPress
            and self.isVisible()
            and isinstance(event, QKeyEvent)
            and event.key() == Qt.Key.Key_L
            and event.modifiers()
            in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier)
            and not event.isAutoRepeat()
            and self._controller.mca_histogram_under_mouse()
        ):
            self.ui.actionLogY.setChecked(not self.ui.actionLogY.isChecked())
            return True
        return False

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
        install_numeric_input_policy()
        install_tooltip_wrapping()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._resize_for_available_screen()

        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionStartSessionRecording.triggered.connect(self._on_start_session_recording)
        self.ui.actionStopSessionRecording.triggered.connect(self._on_stop_session_recording)
        self.ui.actionReplaySession.triggered.connect(self._on_replay_session)
        self.ui.actionExitReplay.triggered.connect(self._on_exit_replay)
        self.ui.actionOpenPsdEvents.triggered.connect(self._on_open_psd_events)
        self.ui.actionOpenWaveformFile.triggered.connect(self._on_open_waveform_file)
        self.ui.actionOpenScopeSums.triggered.connect(self._on_open_scope_sums)
        self.ui.actionEnergyCalibration.triggered.connect(self._on_energy_calibration)
        self.ui.actionDoseCalibration.triggered.connect(self._on_dose_calibration)
        self.ui.actionMcaPeakAnalysis.triggered.connect(self._on_mca_peak_analysis)
        self.ui.actionValidateTiming.triggered.connect(self._on_validate_timing)
        self.ui.actionConvertToHdf5.triggered.connect(self._on_convert_to_hdf5)
        self.ui.actionConvertToRoot.triggered.connect(self._on_convert_to_root)
        self.ui.actionConnectDevice.triggered.connect(self._on_connect_device)
        self.ui.actionDisconnectDevice.triggered.connect(self._on_disconnect_device)
        self.ui.actionReconnectDevice.triggered.connect(self._on_reconnect_device)
        self.ui.actionRefreshModbus.triggered.connect(self._on_refresh_modbus_devices)
        self._modbus_refresh_finished.connect(self._on_modbus_refresh_finished)
        self.ui.actionResetDocks.triggered.connect(self._on_reset_docks)
        self.ui.actionResetZoom.triggered.connect(self._on_reset_zoom)
        self.ui.actionShowRoi.toggled.connect(self._on_show_roi_toggled)
        self.ui.actionDifferentialHistogram.toggled.connect(
            self._on_differential_histogram_toggled
        )
        self.ui.actionScopeDistribution.toggled.connect(
            self._on_scope_distribution_toggled
        )
        self.ui.actionClearScopeDistribution.triggered.connect(
            lambda: self._controller.clear_scope_distributions()
        )
        self.ui.actionScopeSpectrum.toggled.connect(self._on_scope_spectrum_toggled)
        self.ui.actionLogY.toggled.connect(self._on_log_y_toggled)
        self.ui.actionSaveSettings.triggered.connect(self._on_save_settings)
        self.ui.actionLoadSettings.triggered.connect(self._on_load_settings)
        self.ui.actionGeneralSettings.triggered.connect(self._on_general_settings)
        self.ui.actionEnableSeries.toggled.connect(self._on_measurement_series_toggled)
        self.ui.actionAbout.triggered.connect(self._on_about)
        self.ui.actionThirdPartyLicenses.triggered.connect(self._on_third_party_licenses)
        self.ui.actionShowSystemLog.toggled.connect(self._on_show_system_log_toggled)
        self.ui.actionDebugMode.toggled.connect(self._on_debug_mode_toggled)

        self._replay_indicator = QLabel(self)
        self._replay_indicator.setObjectName("replayModeIndicator")
        self._replay_indicator.setStyleSheet(theme.badge_style("replay"))
        self._replay_indicator.setAccessibleName("Session replay mode indicator")
        self._replay_indicator.hide()
        self.statusBar().addPermanentWidget(self._replay_indicator)

        self._offline_indicator = QLabel("OFFLINE", self)
        self._offline_indicator.setObjectName("offlineModeIndicator")
        self._offline_indicator.setStyleSheet(theme.badge_style("offline"))
        self._offline_indicator.setAccessibleName("Offline analysis mode indicator")
        self._offline_indicator.setToolTip(
            "No hardware transport is open. Analysis tools and deterministic session "
            "replay remain available; all hardware controls are locked."
        )
        self._offline_indicator.setVisible(self._offline)
        self.statusBar().addPermanentWidget(self._offline_indicator)

        self._measurement_indicator = QLabel(self)
        self._measurement_indicator.setObjectName("measurementActiveIndicator")
        self._measurement_indicator.setStyleSheet(theme.badge_style("active"))
        self._measurement_indicator.setAccessibleName("Active measurement channels")
        self._measurement_indicator.setTextFormat(Qt.TextFormat.PlainText)
        self._measurement_indicator.hide()
        self.statusBar().addPermanentWidget(self._measurement_indicator)

        self._energy_calibration_indicator = _EnergyCalibrationIndicator(
            "ENERGY CALIBRATED",
            self,
        )
        self._energy_calibration_indicator.setObjectName("energyCalibrationIndicator")
        self._energy_calibration_indicator.setStyleSheet(theme.badge_style("calibrated"))
        self._energy_calibration_indicator.setAccessibleName(
            "Energy calibration active indicator"
        )
        self._energy_calibration_indicator.setToolTip(
            "No MCA energy calibration is active."
        )
        self._energy_calibration_indicator.setCursor(Qt.CursorShape.PointingHandCursor)
        self._energy_calibration_indicator.double_clicked.connect(
            self._on_energy_calibration
        )
        self._energy_calibration_indicator.hide()
        self.statusBar().addPermanentWidget(self._energy_calibration_indicator)

        self._sync_connection_actions()

        self._restore_developer_settings()

        self._help_controller = HelpController(self)
        self.ui.menuHelp.insertAction(self.ui.actionAbout, self._help_controller.guide_action)
        self.ui.menuHelp.insertAction(self.ui.actionAbout, self._help_controller.context_action)
        self.ui.menuHelp.insertSeparator(self.ui.actionAbout)

    def _sync_connection_actions(self) -> None:
        """Reflect whether this window owns an offline or live controller set."""
        live = not self._offline
        available = not self._connection_busy and not self._modbus_refresh_running
        self.ui.actionConnectDevice.setEnabled(available)
        self.ui.actionConnectDevice.setText("Add Digitizer..." if live else "Connect...")
        self.ui.actionConnectDevice.setToolTip(
            "Connect another digitizer as additional channels."
            if live
            else "Connect this offline workspace to a digitizer."
        )
        self.ui.actionDisconnectDevice.setEnabled(live and available)
        self.ui.actionDisconnectDevice.setToolTip(
            "Stop measurements and disconnect one physical digitizer."
            if live
            else "Connect to a digitizer before using Disconnect Digitizer."
        )
        for action_name in (
            "actionStartSessionRecording",
            "actionSaveSettings",
            "actionLoadSettings",
            "actionReconnectDevice",
            "actionRefreshModbus",
        ):
            action = getattr(self.ui, action_name, None)
            if action is not None:
                action.setEnabled(live and available)
        self.ui.actionReconnectDevice.setToolTip(
            "Re-establish the current digitizer connection."
            if live
            else "Connect to a digitizer before using Reconnect Device."
        )

    def _set_connection_busy(self, busy: bool) -> None:
        """Interlock every connection entry point during a topology change."""
        self._connection_busy = busy
        self.ui.mainTabs.setEnabled(not busy)
        self._sync_connection_actions()

    def _sync_connection_presentation(self) -> None:
        """Refresh mirrored window state from the controller's topology."""
        # Fresh panels on fresh connections: a later loss deserves a new prompt.
        self._connection_loss_prompted = False
        self._offline = self._controller.offline
        self._channels = self._controller.channel_count
        endpoints = self._controller.endpoint_descriptions
        primary = self._controller.primary_endpoint
        if self._offline or primary is None:
            self._host = ""
            self._base_window_title = "Nuclear Lab Digitizer — Offline"
        else:
            self._host, self._port = primary
            endpoint_title = (
                endpoints[0]
                if len(endpoints) == 1
                else "; ".join(
                    f"D{index + 1} {endpoint}"
                    for index, endpoint in enumerate(endpoints)
                )
            )
            self._base_window_title = f"Nuclear Lab Digitizer — {endpoint_title}"
        self.setWindowTitle(self._base_window_title)
        self._offline_indicator.setVisible(self._offline)
        self.refresh_energy_calibration_indicator()
        self.refresh_measurement_indicator()
        self._sync_connection_actions()

    def refresh_measurement_indicator(self, *_args: object) -> None:
        """Show a permanent green badge for controller-confirmed live runs."""
        indicator = getattr(self, "_measurement_indicator", None)
        controller = getattr(self, "_controller", None)
        if indicator is None or controller is None:
            return
        getter = getattr(controller, "measuring_channel_labels", None)
        labels = getter() if callable(getter) else ()
        if (not isinstance(labels, tuple) or self._offline
                or getattr(self, "_closing", False)):
            labels = ()
        indicator.setVisible(bool(labels))
        indicator.setText(f"Measuring: {', '.join(labels)}" if labels else "")
        indicator.setToolTip(
            "Live Scope/MCA acquisition reported by this application.\n"
            "A channel is listed once even if both cores are active.\n"
            "Offline analysis and session replay are not measurements."
            if labels else "No active live measurement."
        )

    def refresh_energy_calibration_indicator(self, *_args: object) -> None:
        """Mirror the applied MCA energy-calibration state in the status bar."""
        indicator = getattr(self, "_energy_calibration_indicator", None)
        controller = getattr(self, "_controller", None)
        if indicator is None or controller is None:
            return
        labels_getter = getattr(controller, "energy_calibrated_channel_labels", None)
        labels = labels_getter() if callable(labels_getter) else ()
        if not isinstance(labels, tuple):
            labels = ()
        indicator.setVisible(bool(labels))
        if labels:
            channels = ", ".join(labels)
            indicator.setToolTip(
                f"Energy calibration is active for {channels}. "
                "Double-click to open MCA Energy Calibration."
            )
        else:
            indicator.setToolTip("No MCA energy calibration is active.")

    def _apply_post_connection_settings(
        self,
        *,
        restore_hardware_configuration: bool,
        on_progress: Callable[[str], None] | None,
    ) -> Exception | None:
        """Restore shared UI state after any successful topology transition."""
        try:
            self._apply_stored_general_settings()
            if restore_hardware_configuration and self._config_path is not None:
                if on_progress is not None:
                    on_progress("Applying saved settings...")
                self._controller.load_all_settings(self._config_path)
            elif restore_hardware_configuration and auto_configuration_enabled():
                self._restore_auto_configuration(on_progress)
            self._persist_current_general_settings()
            self._apply_view_state()
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "Connection changed, but post-connection settings could not be applied"
            )
            return exc
        return None

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
        show_log = cast(bool, settings.value(_KEY_SHOW_LOG, False, type=bool))
        debug_mode = cast(bool, settings.value(_KEY_DEBUG_MODE, False, type=bool))
        show_roi = cast(bool, settings.value(_KEY_SHOW_ROI, False, type=bool))
        differential_histogram = cast(
            bool,
            settings.value(
                _KEY_DIFFERENTIAL_HISTOGRAM,
                False,
                type=bool,
            ),
        )
        log_y = cast(bool, settings.value(_KEY_LOG_Y, False, type=bool))
        scope_distribution = cast(
            bool, settings.value(_KEY_SCOPE_DISTRIBUTION, False, type=bool)
        )
        scope_spectrum = cast(bool, settings.value(_KEY_SCOPE_SPECTRUM, False, type=bool))
        measurement_series = cast(
            bool, settings.value(_KEY_MEASUREMENT_SERIES, False, type=bool)
        )

        # Suppress intermediate toggled signals while restoring state.
        self.ui.actionShowSystemLog.blockSignals(True)
        self.ui.actionDebugMode.blockSignals(True)
        self.ui.actionShowRoi.blockSignals(True)
        self.ui.actionDifferentialHistogram.blockSignals(True)
        self.ui.actionLogY.blockSignals(True)
        self.ui.actionScopeDistribution.blockSignals(True)
        self.ui.actionScopeSpectrum.blockSignals(True)
        self.ui.actionEnableSeries.blockSignals(True)

        self.ui.actionShowSystemLog.setChecked(show_log)
        self.ui.actionDebugMode.setChecked(debug_mode)
        self.ui.actionShowRoi.setChecked(show_roi)
        self.ui.actionDifferentialHistogram.setChecked(differential_histogram)
        self.ui.actionLogY.setChecked(log_y)
        self.ui.actionScopeDistribution.setChecked(scope_distribution)
        self.ui.actionScopeSpectrum.setChecked(scope_spectrum)
        self.ui.actionEnableSeries.setChecked(measurement_series)
        self._set_log_tab_visible(show_log)
        if debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)

        self.ui.actionShowSystemLog.blockSignals(False)
        self.ui.actionDebugMode.blockSignals(False)
        self.ui.actionShowRoi.blockSignals(False)
        self.ui.actionDifferentialHistogram.blockSignals(False)
        self.ui.actionLogY.blockSignals(False)
        self.ui.actionScopeDistribution.blockSignals(False)
        self.ui.actionScopeSpectrum.blockSignals(False)
        self.ui.actionEnableSeries.blockSignals(False)

    def _save_developer_settings(self) -> None:
        settings = QSettings()
        settings.setValue(_KEY_SHOW_LOG, self.ui.actionShowSystemLog.isChecked())
        settings.setValue(_KEY_DEBUG_MODE, self.ui.actionDebugMode.isChecked())
        settings.setValue(_KEY_SHOW_ROI, self.ui.actionShowRoi.isChecked())
        settings.setValue(
            _KEY_DIFFERENTIAL_HISTOGRAM,
            self.ui.actionDifferentialHistogram.isChecked(),
        )
        settings.setValue(_KEY_LOG_Y, self.ui.actionLogY.isChecked())
        settings.setValue(_KEY_SCOPE_DISTRIBUTION, self.ui.actionScopeDistribution.isChecked())
        settings.setValue(_KEY_SCOPE_SPECTRUM, self.ui.actionScopeSpectrum.isChecked())
        settings.setValue(_KEY_MEASUREMENT_SERIES, self.ui.actionEnableSeries.isChecked())

    def configuration_settings(self) -> dict[str, object]:
        """Return top-level controls that are owned only by the application."""
        return {
            "show_system_log": self.ui.actionShowSystemLog.isChecked(),
            "debug_mode": self.ui.actionDebugMode.isChecked(),
            "show_roi": self.ui.actionShowRoi.isChecked(),
            "differential_histogram": self.ui.actionDifferentialHistogram.isChecked(),
            "log_y": self.ui.actionLogY.isChecked(),
            "scope_distribution": self.ui.actionScopeDistribution.isChecked(),
            "scope_spectrum": self.ui.actionScopeSpectrum.isChecked(),
            "measurement_series": self.ui.actionEnableSeries.isChecked(),
            "dma_save_folder": str(QSettings().value(_KEY_DMA_FOLDER, "measurements")),
            "mca_dma_output_mode": str(
                QSettings().value(
                    _KEY_MCA_DMA_OUTPUT_MODE,
                    McaDmaOutputMode.BINARY.value,
                )
            ),
            "timestamp_chosen_names": recording_paths.timestamp_chosen_names(),
            "timing_channel_b_offset_ns": cast(
                int, QSettings().value(_KEY_TIMING_OFFSET_NS, 0, type=int)
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
            "differential_histogram": self.ui.actionDifferentialHistogram,
            "log_y": self.ui.actionLogY,
            "scope_distribution": self.ui.actionScopeDistribution,
            "scope_spectrum": self.ui.actionScopeSpectrum,
            "measurement_series": self.ui.actionEnableSeries,
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
        if "timestamp_chosen_names" in settings:
            QSettings().setValue(
                _KEY_TIMESTAMP_CHOSEN_NAMES, bool(settings["timestamp_chosen_names"])
            )
        if "timing_channel_b_offset_ns" in settings:
            try:
                offset_ns = int(settings["timing_channel_b_offset_ns"])
            except (TypeError, ValueError):
                offset_ns = None
            if (
                offset_ns is not None
                and -1_000_000 <= offset_ns <= 1_000_000
                and offset_ns % 8 == 0
            ):
                QSettings().setValue(_KEY_TIMING_OFFSET_NS, offset_ns)
            else:
                logging.getLogger(__name__).warning(
                    "Ignoring invalid timing offset %r", settings["timing_channel_b_offset_ns"]
                )
        if "active_tab" in settings:
            index = int(settings["active_tab"])
            tabs = self.ui.mainTabs
            # Qt permits programmatic selection of a disabled/hidden tab.
            # Saved configurations may refer to hardware that is now absent.
            available = [
                i for i in range(tabs.count()) if tabs.isTabEnabled(i) and tabs.isTabVisible(i)
            ]
            if index in available:
                tabs.setCurrentIndex(index)
            elif available and tabs.currentIndex() not in available:
                tabs.setCurrentIndex(available[0])
        self._save_developer_settings()
        self._apply_view_state()
        self._controller.refresh_dma_output_settings()

    def _apply_view_state(self) -> None:
        """Re-apply persisted view toggles to the rebuilt plot controllers."""
        self._controller.set_roi_visible(self.ui.actionShowRoi.isChecked())
        self._controller.set_differential_histogram(
            self.ui.actionDifferentialHistogram.isChecked()
        )
        self._controller.set_log_y(self.ui.actionLogY.isChecked())
        self._controller.set_scope_distribution_enabled(
            self.ui.actionScopeDistribution.isChecked()
        )
        self._controller.set_scope_spectrum_enabled(self.ui.actionScopeSpectrum.isChecked())
        self._controller.set_measurement_series_enabled(
            self.ui.actionEnableSeries.isChecked()
        )

    def _general_settings_values(self) -> dict[str, object]:
        getter = getattr(self._controller, "general_configuration_settings", None)
        values: dict[str, object] = dict(getter()) if callable(getter) else {}
        settings = QSettings()
        values.update(
            {
                "dma_save_folder": str(settings.value(_KEY_DMA_FOLDER, "measurements")),
                "mca_dma_output_mode": str(
                    settings.value(
                        _KEY_MCA_DMA_OUTPUT_MODE,
                        McaDmaOutputMode.BINARY.value,
                    )
                ),
                "timestamp_chosen_names": recording_paths.timestamp_chosen_names(),
            }
        )
        return values

    def _apply_general_settings(self, values: dict[str, object]) -> None:
        settings = QSettings()
        for name, key in _GENERAL_CONTROLLER_KEYS.items():
            if name in values:
                settings.setValue(key, int(cast(int, values[name])))
        apply_controller = getattr(
            self._controller,
            "apply_general_configuration_settings",
            None,
        )
        if callable(apply_controller):
            apply_controller(values)

        settings.setValue(
            _KEY_DMA_FOLDER,
            str(values.get("dma_save_folder", "measurements")),
        )
        settings.setValue(
            _KEY_MCA_DMA_OUTPUT_MODE,
            str(values.get("mca_dma_output_mode", McaDmaOutputMode.BINARY.value)),
        )
        if "timestamp_chosen_names" in values:
            settings.setValue(
                _KEY_TIMESTAMP_CHOSEN_NAMES, bool(values["timestamp_chosen_names"])
            )
        self._save_developer_settings()
        self._controller.refresh_dma_output_settings()

    def _apply_stored_general_settings(self) -> None:
        getter = getattr(self._controller, "general_configuration_settings", None)
        apply_controller = getattr(
            self._controller,
            "apply_general_configuration_settings",
            None,
        )
        if not callable(getter) or not callable(apply_controller):
            return
        current = getter()
        settings = QSettings()
        values = {
            name: int(
                cast(
                    int,
                    settings.value(
                        key,
                        int(cast(int, current[name])),
                        type=int,
                    ),
                )
            )
            for name, key in _GENERAL_CONTROLLER_KEYS.items()
        }
        apply_controller(values)
        for name, key in _GENERAL_CONTROLLER_KEYS.items():
            settings.setValue(key, values[name])

    def _persist_current_general_settings(self) -> None:
        getter = getattr(self._controller, "general_configuration_settings", None)
        if not callable(getter):
            return
        values = getter()
        settings = QSettings()
        for name, key in _GENERAL_CONTROLLER_KEYS.items():
            if name in values:
                settings.setValue(key, int(values[name]))

    def _set_log_tab_visible(self, visible: bool) -> None:
        idx = self.ui.mainTabs.indexOf(self.ui.tabSystemLog)
        self.ui.mainTabs.setTabVisible(idx, visible)

    def _set_replay_indicator(
        self,
        replay: SessionReplay | None,
        path: Path | None,
    ) -> None:
        active = replay is not None and path is not None
        self.ui.actionExitReplay.setEnabled(active)
        hardware_action_names = (
            "actionStartSessionRecording",
            "actionConnectDevice",
            "actionDisconnectDevice",
            "actionReconnectDevice",
            "actionRefreshModbus",
            "actionLoadSettings",
            "actionGeneralSettings",
        )
        actions = [
            action
            for name in hardware_action_names
            if (action := getattr(self.ui, name, None)) is not None
        ]
        saved_states = getattr(self, "_replay_action_enabled_states", None)
        if active and saved_states is None:
            self._replay_action_enabled_states: dict[str, bool] | None = {
                action.objectName(): action.isEnabled() for action in actions
            }
            for action in actions:
                action.setEnabled(False)
        elif not active and saved_states is not None:
            for action in actions:
                action.setEnabled(bool(saved_states.get(action.objectName(), True)))
            self._replay_action_enabled_states = None
        if not active:
            self._replay_indicator.hide()
            self._replay_indicator.clear()
            self._replay_indicator.setToolTip("")
            self.setWindowTitle(self._base_window_title)
            return

        assert replay is not None and path is not None
        name = path.name
        if len(name) > 32:
            name = f"{name[:29]}..."
        warning = not replay.capture_complete or replay.integrity != "verified"
        self._replay_indicator.setText(f"REPLAY ! {name}" if warning else f"REPLAY | {name}")
        self._replay_indicator.setToolTip(
            "Offline session analysis replay is loaded.\n"
            f"Bundle: {path}\n"
            f"Session: {replay.manifest.get('session_id', 'unknown')}\n"
            f"Capture complete: {replay.capture_complete}\n"
            f"Integrity: {replay.integrity}\n\n"
            "Recorded Scope, Current, MCA, PSU, global, and external-module updates "
            "play once and hold their final values. MCA accumulation, PSD, and "
            "coincidence analysis build progressively from recorded checkpoints and "
            "event batches. Affected live acquisition and polling are stopped, and all "
            "hardware-setting controls are locked. Use File > Exit Replay Mode to "
            "unload replay."
        )
        self._replay_indicator.show()
        self.setWindowTitle(f"{self._base_window_title} [REPLAY]")

    def _sync_replay_indicator(self) -> None:
        self._set_replay_indicator(
            self._controller.session_replay,
            self._controller.session_replay_path,
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_open_psd_events(self) -> None:
        self._controller.show_psd_event_readback()

    def _on_start_session_recording(self) -> None:
        folder = Path(str(QSettings().value(_KEY_SESSION_FOLDER, "measurements")))
        suggested = folder / (f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nlab-session")
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Start Session Recording",
            str(suggested),
            "NLab session bundle (*.nlab-session)",
        )
        if not selected:
            return
        path = Path(selected)
        if not str(path).lower().endswith(".nlab-session"):
            path = Path(f"{path}.nlab-session")
        try:
            self._controller.start_session_recording(path)
        except Exception as exc:
            logging.getLogger(__name__).exception("Starting session recording failed")
            QMessageBox.critical(self, "Session Recording", str(exc))
            return
        QSettings().setValue(_KEY_SESSION_FOLDER, str(path.parent))
        self.ui.actionStartSessionRecording.setEnabled(False)
        self.ui.actionStopSessionRecording.setEnabled(True)
        self.statusBar().showMessage(f"Recording session to {path}")

    def _on_stop_session_recording(self) -> None:
        try:
            path = self._controller.stop_session_recording()
        except Exception as exc:
            logging.getLogger(__name__).exception("Finalizing session recording failed")
            QMessageBox.critical(self, "Session Recording", str(exc))
        else:
            if path is not None:
                self.statusBar().showMessage(f"Session finalized: {path}", 10_000)
        finally:
            active = self._controller.session_recording
            self.ui.actionStartSessionRecording.setEnabled(not active)
            self.ui.actionStopSessionRecording.setEnabled(active)

    def _on_replay_session(self) -> None:
        folder = str(QSettings().value(_KEY_SESSION_FOLDER, "measurements"))
        selected = QFileDialog.getExistingDirectory(
            self,
            "Replay Session Bundle",
            folder,
        )
        if not selected:
            return
        path = Path(selected)
        try:
            replay = self._controller.replay_session(path)
        except Exception as exc:
            logging.getLogger(__name__).exception("Loading session replay failed")
            QMessageBox.critical(self, "Session Replay", str(exc))
            self._sync_replay_indicator()
            return
        QSettings().setValue(_KEY_SESSION_FOLDER, str(path.parent))
        self._set_replay_indicator(replay, path)
        if not replay.capture_complete or replay.integrity != "verified":
            QMessageBox.warning(
                self,
                "Session Replay",
                "The session is incomplete or has recorded data loss. Its data remain "
                "available for investigation, but should not be treated as a complete "
                "measurement.",
            )
        self.statusBar().showMessage(
            f"Loaded session replay: {path} (integrity: {replay.integrity})",
            10_000,
        )

    def _on_exit_replay(self) -> None:
        self._controller.clear_session_replay()
        self._set_replay_indicator(None, None)
        self.statusBar().showMessage("Session replay unloaded", 5_000)

    def _on_open_scope_sums(self) -> None:
        self._controller.show_scope_sums()

    def _on_open_waveform_file(self) -> None:
        self._controller.show_waveform_analysis()

    def _on_validate_timing(self) -> None:
        # A modeless tool window like the other analysis tools (a modal one
        # with a minimize button would leave the main window locked). One
        # instance is reused, so the chosen files survive closing it.
        dialog = self._timing_validation_dialog
        if dialog is None:
            dialog = TimingValidationDialog(self)
            self._timing_validation_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_energy_calibration(self) -> None:
        self._controller.show_energy_calibration()

    def _on_dose_calibration(self) -> None:
        self._controller.show_dose_calibration()

    def _on_mca_peak_analysis(self) -> None:
        self._controller.show_mca_peak_analysis()

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

    def _on_convert_to_root(self) -> None:
        logging.getLogger(__name__).info("ROOT conversion requested from File menu")
        try:
            from nlab.views.root_conversion_dialog import RootConversionDialog
        except Exception as exc:
            logging.getLogger(__name__).exception("ROOT conversion dialog could not be loaded")
            QMessageBox.critical(
                self, "ROOT Conversion Unavailable",
                "ROOT conversion could not be loaded. If you are using the standalone "
                "application, install a rebuilt package with its ROOT dependencies."
                f"\n\n{type(exc).__name__}: {exc}",
            )
            return

        src, _ = QFileDialog.getOpenFileName(
            self, "Select Binary DMA File", "", "Binary files (*.bin);;All files (*)",
        )
        if not src:
            return
        dst, _ = QFileDialog.getSaveFileName(
            self, "Save ROOT File", str(Path(src).with_suffix(".root")),
            "ROOT files (*.root);;All files (*)",
        )
        if not dst:
            return
        destination = Path(dst)
        if not destination.suffix:
            destination = destination.with_suffix(".root")
        try:
            dialog = RootConversionDialog(Path(src), destination, self)
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "ROOT conversion dialog initialization failed: %s -> %s", src, destination,
            )
            QMessageBox.critical(
                self, "ROOT Conversion Failed", f"{type(exc).__name__}: {exc}",
            )
            return
        dialog.exec()
        if dialog.error:
            QMessageBox.critical(
                self, "Conversion Failed",
                f"{dialog.error}\n\nSee System Log for the full traceback.",
            )
        elif (result := dialog.conversion_result) is not None:
            QMessageBox.information(
                self, "Conversion Complete",
                f"Converted {result.count:,} {result.kind} to:\n{result.destination}",
            )
        dialog.deleteLater()

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

    def _on_general_settings(self) -> None:
        dialog = GeneralSettingsDialog(self._host, self._general_settings_values(), self)
        if not dialog.exec():
            return

        self._apply_general_settings(dialog.settings)
        enabled = dialog.auto_configuration_is_enabled
        if enabled and not self._save_auto_configuration(show_error=True):
            set_auto_configuration_enabled(False)
            return
        set_auto_configuration_enabled(enabled)
        if enabled:
            self.statusBar().showMessage(
                f"Automatic configuration saved to {auto_configuration_path(self._host)}",
                5000,
            )
        else:
            self.statusBar().showMessage(
                "General settings updated; automatic restore disabled",
                5000,
            )

    def _save_auto_configuration(self, *, show_error: bool) -> bool:
        path = auto_configuration_path(self._host)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._controller.save_all_settings(path)
        except Exception as exc:
            logging.getLogger(__name__).exception("Failed to save automatic configuration")
            if show_error:
                QMessageBox.critical(
                    self,
                    "Automatic Save Failed",
                    f"The automatic configuration could not be saved:\n{exc}",
                )
            return False
        return True

    def _restore_auto_configuration(
        self,
        on_progress: Callable[[str], None] | None,
    ) -> None:
        path = auto_configuration_path(self._host)
        if not path.is_file():
            return
        if on_progress is not None:
            on_progress("Restoring the last configuration...")
        try:
            self._controller.load_all_settings(path)
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "Failed to restore automatic configuration from %s", path
            )
            self.statusBar().showMessage(
                f"Automatic configuration was not restored: {exc}",
                10000,
            )

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
            self._persist_current_general_settings()
        except Exception as e:
            logging.getLogger(__name__).exception("Failed to load settings")
            QMessageBox.critical(self, "Load Failed", str(e))

    def _on_show_roi_toggled(self, checked: bool) -> None:
        self._controller.set_roi_visible(checked)

    def _on_differential_histogram_toggled(self, checked: bool) -> None:
        self._controller.set_differential_histogram(checked)

    def _on_scope_distribution_toggled(self, checked: bool) -> None:
        self._controller.set_scope_distribution_enabled(checked)

    def _on_scope_spectrum_toggled(self, checked: bool) -> None:
        self._controller.set_scope_spectrum_enabled(checked)

    def _on_log_y_toggled(self, checked: bool) -> None:
        self._controller.set_log_y(checked)

    def _on_measurement_series_toggled(self, checked: bool) -> None:
        self._controller.set_measurement_series_enabled(checked)
        QSettings().setValue(_KEY_MEASUREMENT_SERIES, checked)

    def _on_reset_zoom(self) -> None:
        self._controller.reset_all_zoom()

    def _on_connect_device(self) -> None:
        if self._connection_busy or self._modbus_refresh_running:
            return
        was_offline = self._offline
        dialog = (
            ConnectionDialog(self, channels=self._channels)
            if was_offline
            else ConnectionDialog(self, allow_offline=False)
        )
        if dialog.exec() != ConnectionDialog.DialogCode.Accepted or dialog.offline:
            return

        progress = ConnectionProgressDialog(
            "Connecting to Device" if was_offline else "Adding Digitizer",
            "Connecting to digitizer..." if was_offline else "Adding digitizer...",
            self,
        )
        progress.show()

        def update_progress(message: str) -> None:
            progress.setLabelText(message)
            QApplication.processEvents()

        self._set_connection_busy(True)
        try:
            if was_offline:
                self._controller.connect_to_device(
                    dialog.ip,
                    dialog.port,
                    dialog.channels,
                    on_progress=update_progress,
                )
                digitizer_number = 1
            else:
                digitizer_number = self._controller.add_digitizer(
                    dialog.ip,
                    dialog.port,
                    dialog.channels,
                    on_progress=update_progress,
                )
        except Exception as exc:
            logging.getLogger(__name__).exception("Digitizer connection failed")
            progress.close()
            self._set_connection_busy(False)
            self._sync_replay_indicator()
            QMessageBox.critical(self, "Connection Failed", str(exc))
            return

        self._sync_connection_presentation()
        settings_error = self._apply_post_connection_settings(
            restore_hardware_configuration=was_offline,
            on_progress=update_progress,
        )
        progress.close()
        self._set_connection_busy(False)
        self._sync_replay_indicator()
        self.statusBar().showMessage(
            (
                f"Connected to {self._host}:{self._port}"
                if was_offline
                else f"Added D{digitizer_number} at {dialog.ip}:{dialog.port}"
            ),
            10_000,
        )
        if settings_error is not None:
            QMessageBox.warning(
                self,
                "Connected with Settings Warning",
                "The digitizer is connected and measurement controls are enabled, but "
                f"some saved settings could not be applied:\n{settings_error}",
            )
        else:
            QMessageBox.information(
                self,
                "Connected" if was_offline else "Digitizer Added",
                (
                    f"Connected to {self._host}:{self._port}. "
                    "Measurement controls are enabled."
                    if was_offline
                    else f"Added D{digitizer_number} at {dialog.ip}:{dialog.port}. "
                    "Coincidence remains limited to channels on one digitizer."
                ),
            )

    def _on_disconnect_device(self) -> None:
        if self._connection_busy or self._modbus_refresh_running:
            return
        endpoints = self._controller.endpoint_descriptions
        if not endpoints:
            return

        digitizer = 0
        if len(endpoints) > 1:
            choices = [
                f"D{index + 1} — {endpoint}"
                for index, endpoint in enumerate(endpoints)
            ]
            selected, accepted = QInputDialog.getItem(
                self,
                "Disconnect Digitizer",
                "Digitizer:",
                choices,
                0,
                False,
            )
            if not accepted:
                return
            digitizer = choices.index(selected)

        board = f"D{digitizer + 1}"
        endpoint = endpoints[digitizer]
        final_endpoint = len(endpoints) == 1
        consequence = (
            "The application will return to offline analysis mode."
            if final_endpoint
            else "Remaining digitizers will be renumbered."
        )
        reply = QMessageBox.question(
            self,
            "Disconnect Digitizer",
            f"This will stop all running measurements and disconnect "
            f"{board} at {endpoint}. {consequence}\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        progress = ConnectionProgressDialog(
            "Disconnecting Digitizer",
            f"Disconnecting {board}...",
            self,
        )
        progress.show()

        def update_progress(message: str) -> None:
            progress.setLabelText(message)
            QApplication.processEvents()

        self._set_connection_busy(True)
        try:
            offline = self._controller.disconnect_digitizer(
                digitizer,
                on_progress=update_progress,
            )
        except Exception as exc:
            logging.getLogger(__name__).exception("Digitizer disconnect failed")
            progress.close()
            self._set_connection_busy(False)
            self._sync_replay_indicator()
            QMessageBox.critical(self, "Disconnect Failed", str(exc))
            return

        self._sync_connection_presentation()
        settings_error = self._apply_post_connection_settings(
            restore_hardware_configuration=False,
            on_progress=update_progress,
        )
        progress.close()
        self._set_connection_busy(False)
        self._sync_replay_indicator()
        self.statusBar().showMessage(f"Disconnected {board} at {endpoint}", 10_000)
        if settings_error is not None:
            QMessageBox.warning(
                self,
                "Disconnected with Settings Warning",
                f"Disconnected {board} at {endpoint}, but some saved UI settings "
                f"could not be applied:\n{settings_error}",
            )
        else:
            QMessageBox.information(
                self,
                "Digitizer Disconnected",
                (
                    f"Disconnected {board} at {endpoint}. Offline analysis mode is active."
                    if offline
                    else f"Disconnected {board} at {endpoint}. Remaining digitizers were "
                    "renumbered."
                ),
            )

    def notify_connection_lost(self, source: str, message: str) -> None:
        """Offer one reconnect when a panel reports that the board connection is gone."""
        if (
            self._offline
            or self._connection_busy
            or self._modbus_refresh_running
            or self._connection_loss_prompted
        ):
            return
        self._connection_loss_prompted = True
        logging.getLogger(__name__).warning(
            "Board connection lost (%s): %s", source, message,
        )
        # Leave the reporting panel's signal handler before a modal dialog runs.
        QTimer.singleShot(0, lambda: self._offer_reconnect_after_loss(f"{source}: {message}"))

    def _offer_reconnect_after_loss(self, detail: str) -> None:
        while self._ask_reconnect_after_loss(detail):
            error = self._reconnect_device()
            if error is None:
                return
            detail = f"Reconnect failed: {error}"

    def _ask_reconnect_after_loss(self, detail: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Connection Lost")
        box.setText("The connection to the digitizer was lost.")
        box.setInformativeText(
            f"{detail}\n\n"
            "Polling and acquisition on the lost connection have stopped. Reconnecting "
            "stops all running measurements and replaces every board connection. You "
            "can also reconnect later with Connection → Reconnect Device."
        )
        reconnect = box.addButton("Reconnect", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Not Now", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(reconnect)
        box.exec()
        return box.clickedButton() is reconnect

    def _on_reconnect_device(self) -> None:
        if self._connection_busy or self._modbus_refresh_running:
            return
        reply = QMessageBox.question(
            self,
            "Reconnect Device",
            "This will stop all running measurements and re-establish the "
            "device connection. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        error = self._reconnect_device()
        if error is not None:
            QMessageBox.critical(self, "Reconnect Failed", error)

    def _reconnect_device(self) -> str | None:
        """Replace every board connection; return the error message on failure."""
        if self._connection_busy or self._modbus_refresh_running:
            return "another connection operation is in progress"
        progress = ConnectionProgressDialog(
            "Reconnecting Digitizers",
            "Preparing replacement connections...",
            self,
        )
        progress.show()

        def update_progress(message: str) -> None:
            progress.setLabelText(message)
            QApplication.processEvents()

        self._set_connection_busy(True)
        try:
            self._controller.reconnect(on_progress=update_progress)
        except Exception as exc:
            logging.getLogger(__name__).exception("Reconnect failed")
            progress.close()
            self._set_connection_busy(False)
            self._sync_replay_indicator()
            return str(exc)

        self._sync_connection_presentation()
        settings_error = self._apply_post_connection_settings(
            restore_hardware_configuration=False,
            on_progress=update_progress,
        )
        progress.close()
        self._set_connection_busy(False)
        self._sync_replay_indicator()
        if settings_error is not None:
            QMessageBox.warning(
                self,
                "Reconnected with Settings Warning",
                "Digitizer transports were replaced, but some saved UI settings "
                f"could not be applied:\n{settings_error}",
            )
        else:
            QMessageBox.information(self, "Reconnect", "Device reconnected successfully.")
        return None

    def _on_refresh_modbus_devices(self) -> None:
        """Start a non-blocking scan of both digitizer ser2net bridges."""
        if self._modbus_refresh_running or self._connection_busy:
            return
        self._modbus_refresh_running = True
        self._sync_connection_actions()
        try:
            self._controller.prepare_external_device_refresh()
        except Exception as exc:
            self._modbus_refresh_running = False
            self._sync_connection_actions()
            logging.getLogger(__name__).exception("Failed to prepare Modbus refresh")
            self.statusBar().showMessage("Modbus refresh failed", 10000)
            QMessageBox.critical(self, "Modbus Refresh Failed", str(exc))
            return
        self.statusBar().showMessage("Scanning for external Modbus devices…")
        threading.Thread(
            target=self._scan_modbus_in_background,
            name="modbus-discovery",
            daemon=True,
        ).start()

    def _scan_modbus_in_background(self) -> None:
        scan: ExternalDeviceScan | None = None
        error: Exception | None = None
        try:
            scan = self._controller.scan_external_devices()
        except Exception as exc:
            error = exc
            logging.getLogger(__name__).exception("Modbus refresh scan failed")

        try:
            self._modbus_refresh_finished.emit(scan, error)
        except RuntimeError:
            # The window may have been destroyed while the bounded scan ran.
            if scan is not None:
                scan.manager.close()

    def _on_modbus_refresh_finished(self, scan_object: object, error_object: object) -> None:
        """Adopt a completed scan on the GUI thread and rebuild External docks."""
        self._modbus_refresh_running = False

        scan = scan_object if isinstance(scan_object, ExternalDeviceScan) else None
        error = error_object if isinstance(error_object, Exception) else None
        if self._closing:
            if scan is not None:
                scan.manager.close()
            return

        self._sync_connection_actions()
        if error is not None or scan is None:
            if scan is not None:
                scan.manager.close()
            message = str(error) if error is not None else "Invalid discovery result"
            self.statusBar().showMessage("Modbus refresh failed", 10000)
            QMessageBox.critical(self, "Modbus Refresh Failed", message)
            return

        try:
            count = self._controller.apply_external_device_scan(scan)
        except Exception as exc:
            logging.getLogger(__name__).exception("Failed to install refreshed Modbus devices")
            self.statusBar().showMessage("Modbus refresh failed", 10000)
            QMessageBox.critical(self, "Modbus Refresh Failed", str(exc))
            return

        if count:
            noun = "device" if count == 1 else "devices"
            message = f"Found {count} external Modbus {noun}."
        else:
            message = "No external Modbus devices found."
        self.statusBar().showMessage(message, 10000)

    def _on_reset_docks(self) -> None:
        self._controller.reset_dock_layout()

    def _on_about(self) -> None:
        controller = getattr(self, "_controller", None)
        digitizers = controller.digitizer_information() if controller is not None else ()
        AboutDialog(_ABOUT_TEXT, self, digitizers=digitizers).exec()

    def _on_third_party_licenses(self) -> None:
        LicenseDialog(self).exec()
