from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from time import monotonic_ns
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QByteArray, QSettings, Qt, QThread, QThreadPool, QTimer
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

from nlab.controllers.coincidence_controller import CoincidenceController
from nlab.controllers.current_monitor_controller import CurrentMonitorController
from nlab.controllers.external_device_controller import ExternalDeviceController
from nlab.controllers.global_controller import GlobalController
from nlab.controllers.mca_controller import MCAController
from nlab.controllers.psd_controller import PSDController
from nlab.controllers.psu_controller import PSUController
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.current_monitor import ScopeCurrentAccumulator
from nlab.hardware.digitizer.digitizer import Digitizer
from nlab.hardware.digitizer.dma import McaEventBuffer, ScopeFrameBuffer
from nlab.hardware.modbus_devices import ExternalDevices, ExternalDeviceScan
from nlab.session.recorder import SessionRecorder
from nlab.session.replay import (
    McaReplayBatch,
    McaReplayReadback,
    ScopeReplayFrame,
    SessionReplay,
    TelemetryReplayEvent,
)
from nlab.views.energy_calibration_dialog import EnergyCalibrationDialog
from nlab.views.mca_peak_analysis_dialog import McaPeakAnalysisDialog
from nlab.views.psd_readback_dialog import PsdReadbackDialog
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog

if TYPE_CHECKING:
    from nlab_modbus.core.base_modbus_device import BaseModbusDevice

    from nlab.app import MainAppWindow

log = logging.getLogger(__name__)


class MainWindowController:
    """Handles all business logic for MainAppWindow.

    Owns the DeviceManager and measurement thread lifecycle.
    Each channel's Scope and MCA views live in QDockWidgets so they can be
    floated onto a second monitor and re-docked freely.
    """

    def __init__(
        self,
        window: MainAppWindow,
        host: str = "",
        port: int = 30431,
        channels: int = 2,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self._window = window
        self._host = host
        self._port = port
        self._channels = channels
        self._devices: list[Digitizer] = []
        self._on_progress = on_progress

        log.info("Connecting via IIO to %s:%d, %d channel(s)", host, port, channels)
        for channel in range(self._channels):
            self._report_progress(f"Connecting channel {channel + 1} of {self._channels}...")
            self._devices.append(self._connect_channel(channel))
        log.info("All %d device(s) connected", len(self._devices))

        self._scope_controllers: list[ScopeController] = []
        self._current_monitor_controllers: list[CurrentMonitorController] = []
        self._current_monitor_controller_by_device: dict[int, CurrentMonitorController] = {}
        self._mca_controllers: list[MCAController] = []
        self._mca_controller_by_device: dict[int, MCAController] = {}
        self._psd_controllers: list[PSDController] = []
        self._psd_controller_by_device: dict[int, PSDController] = {}
        self._psu_controllers: list[PSUController] = []
        self._psu_controller_by_device: dict[int, PSUController] = {}
        self._external_controllers: list[ExternalDeviceController] = []
        self._external_refresh_application_settings: dict[str, dict[str, object]] = {}
        self._global_controller: GlobalController | None = None
        self._coincidence_controller: CoincidenceController | None = None
        self._energy_calibration_dialog: EnergyCalibrationDialog | None = None
        self._mca_peak_analysis_dialog: McaPeakAnalysisDialog | None = None
        self._psd_readback_dialog: PsdReadbackDialog | None = None
        self._waveform_analysis_dialog: WaveformAnalysisDialog | None = None
        self._external_devices = ExternalDevices()
        self._thread: QThread | None = None
        self._session_recorder: SessionRecorder | None = None
        self._session_replay: SessionReplay | None = None
        self._session_replay_path: Path | None = None
        self._session_replay_spectrum_ids: set[str] = set()
        self._session_replay_psd_channels: set[int] = set()
        self._session_replay_coincidence_active = False
        self._session_analysis_replay_events: list[McaReplayBatch | McaReplayReadback] = []
        self._session_analysis_replay_index = 0
        self._session_analysis_replay_origin_ns = 0
        self._session_analysis_replay_started_ns = 0
        self._session_analysis_replay_mca_channels: set[int] = set()
        self._session_analysis_replay_coincidence_channels: dict[int, int] = {}
        self._session_analysis_replay_coincidence_intervals: list[
            tuple[int, int | None]
        ] | None = None
        self._session_analysis_replay_timer = QTimer()
        self._session_analysis_replay_timer.setSingleShot(True)
        self._session_analysis_replay_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._session_analysis_replay_timer.timeout.connect(
            self._advance_session_analysis_replay
        )
        self._session_scope_replay_frames: list[ScopeReplayFrame] = []
        self._session_scope_replay_index = 0
        self._session_scope_replay_origin_ns = 0
        self._session_scope_replay_started_ns = 0
        self._session_scope_replay_channels: set[int] = set()
        self._session_scope_replay_frame_counts: dict[int, int] = {}
        self._session_scope_replay_delivered: dict[int, int] = {}
        self._session_scope_replay_timer = QTimer()
        self._session_scope_replay_timer.setSingleShot(True)
        self._session_scope_replay_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._session_scope_replay_timer.timeout.connect(
            self._advance_session_scope_replay
        )
        self._session_telemetry_replay_events: list[TelemetryReplayEvent] = []
        self._session_telemetry_replay_index = 0
        self._session_telemetry_replay_origin_ns = 0
        self._session_telemetry_replay_started_ns = 0
        self._session_telemetry_replay_psu_channels: set[int] = set()
        self._session_telemetry_replay_external_ids: set[str] = set()
        self._session_telemetry_replay_timer = QTimer()
        self._session_telemetry_replay_timer.setSingleShot(True)
        self._session_telemetry_replay_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._session_telemetry_replay_timer.timeout.connect(
            self._advance_session_telemetry_replay
        )
        self._replay_hardware_interlocked = False
        self._mca_event_buffers: dict[int, McaEventBuffer] = {}
        self._scope_frame_buffers: dict[int, ScopeFrameBuffer] = {}

        self._scope_dock_host = self._make_dock_host()
        self._current_dock_host = self._make_dock_host()
        self._mca_dock_host = self._make_dock_host()
        self._psd_dock_host = self._make_dock_host()
        self._psu_dock_host = self._make_dock_host()
        self._global_dock_host = self._make_dock_host()
        self._external_dock_host = self._make_dock_host()
        self._report_progress("Preparing global controls...")
        self._build_global_tab()
        self._report_progress("Preparing channel views...")
        self._build_channel_docks()
        self._report_progress("Preparing coincidence view...")
        self._build_coincidence_tab()
        self._report_progress("Discovering external devices...")
        self._build_external_docks()
        self._report_progress("Restoring dock layout...")
        self._restore_dock_state()
        self._connect_signals()
        log.info(
            "UI initialized, %d scope / %d current / %d MCA / %d PSD / %d PSU / "
            "%d external controllers",
            len(self._scope_controllers),
            len(self._current_monitor_controllers),
            len(self._mca_controllers),
            len(self._psd_controllers),
            len(self._psu_controllers),
            len(self._external_controllers),
        )

    def _report_progress(self, message: str) -> None:
        callback = getattr(self, "_on_progress", None)
        if callback is not None:
            callback(message)

    def _connect_channel(self, channel: int) -> Digitizer:
        """Connect one zero-based IIO hardware channel."""
        return Digitizer.from_iio(
            channel=channel,
            uri=f"ip:{self._host}:{self._port}",
            with_ids=True,
        )

    def _display_channel(self, idx: int) -> int:
        """Return the zero-based IIO channel number shown throughout the UI."""
        return idx

    # ------------------------------------------------------------------
    # Dock construction
    # ------------------------------------------------------------------

    @staticmethod
    def _make_dock_host() -> QMainWindow:
        """A QMainWindow used as an embedded dock-area panel."""
        host = QMainWindow()
        host.setWindowFlags(Qt.WindowType.Widget)
        host.setDockOptions(
            QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AnimatedDocks
        )
        return host

    @staticmethod
    def _connect_optional_signal(
        owner: object,
        name: str,
        slot: Callable[..., None],
    ) -> None:
        """Connect production Qt signals while keeping isolated factory tests simple."""
        signal = getattr(owner, name, None)
        if signal is not None:
            signal.connect(slot)

    @staticmethod
    def _make_dock(obj_name: str, title: str, widget: QWidget) -> QDockWidget:
        dock = QDockWidget(title)
        dock.setObjectName(obj_name)
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        return dock

    @staticmethod
    def _populate_dock_host(host: QMainWindow, docks: list[QDockWidget]) -> None:
        """Add docks to host, tabified, with the first tab raised.

        All docks must be registered via addDockWidget before tabifyDockWidget
        is called — Qt requires both arguments to already belong to the host.
        """
        if not docks:
            return
        for dock in docks:
            host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        for i in range(1, len(docks)):
            host.tabifyDockWidget(docks[i - 1], docks[i])
        docks[0].raise_()

    def _build_channel_docks(self) -> None:
        scope_docks: list[QDockWidget] = []
        current_docks: list[QDockWidget] = []
        mca_docks: list[QDockWidget] = []
        psd_docks: list[QDockWidget] = []
        psu_docks: list[QDockWidget] = []

        if not hasattr(self, "_mca_event_buffers"):
            self._mca_event_buffers = {}
        if not hasattr(self, "_scope_frame_buffers"):
            self._scope_frame_buffers = {}

        self._psu_controller_by_device = {}
        self._psd_controller_by_device = {}
        self._current_monitor_controller_by_device = {}
        self._mca_controller_by_device = {}
        missing_mca_channels: list[int] = []
        missing_psu_channels: list[int] = []

        # IIODigitizerBackend now implements MCABackend against
        # vdpp-pulse-processor.c/vdpp-input-filter.c (see iio_backend.py's
        # module docstring for the few methods that still raise
        # NotImplementedError -- none of them are on MCAController's
        # unconditional default-hydration path, so construction is safe).
        # Only skip MCA docks if this channel's firmware genuinely lacks
        # the pulse-processor/input-filter devices (older/scope-only
        # builds) -- MCAController.__init__ would otherwise crash on the
        # RuntimeError those methods raise instead. Capabilities are frozen
        # when each Digitizer is connected, so dock construction does not
        # repeat transport-specific discovery or inspect backend classes.
        for idx, device in enumerate(self._devices):
            ch = self._display_channel(idx)
            self._report_progress(f"Initializing channel {ch} controls...")
            ch_label = f"Ch {ch}"

            scope_current_accumulator = ScopeCurrentAccumulator()
            scope_frame_buffer = ScopeFrameBuffer()
            scope_ctrl = ScopeController(
                device.scope,
                scope_dma=device.scope_dma,
                channel=ch,
                dma_frame_buffer=scope_frame_buffer,
                current_accumulator=scope_current_accumulator,
            )
            self._connect_optional_signal(
                scope_ctrl,
                "session_viewer_frame",
                self._record_scope_viewer_frame,
            )
            self._scope_frame_buffers[ch] = scope_frame_buffer
            self._scope_controllers.append(scope_ctrl)
            scope_docks.append(self._make_dock(f"scope_ch{ch}", ch_label, scope_ctrl))

            if device.capabilities.mca:
                current_ctrl = CurrentMonitorController(
                    device.mca,
                    channel=ch,
                    auto_start=False,
                    scope_controller=scope_ctrl,
                    scope_current_accumulator=scope_current_accumulator,
                )
                self._connect_optional_signal(
                    current_ctrl,
                    "session_samples",
                    self._record_current_samples,
                )
                self._current_monitor_controllers.append(current_ctrl)
                self._current_monitor_controller_by_device[idx] = current_ctrl
                current_docks.append(
                    self._make_dock(f"current_ch{ch}", ch_label, current_ctrl)
                )
                event_buffer = None
                psd_ctrl = None
                if device.mca_dma is not None:
                    event_buffer = McaEventBuffer()
                    self._mca_event_buffers[ch] = event_buffer
                    psd_ctrl = PSDController(event_buffer=event_buffer, channel=ch)
                    self._psd_controllers.append(psd_ctrl)
                    self._psd_controller_by_device[idx] = psd_ctrl
                    psd_docks.append(self._make_dock(f"psd_ch{ch}", ch_label, psd_ctrl))
                mca_ctrl = MCAController(
                    device.mca,
                    mca_dma=device.mca_dma,
                    channel=ch,
                    event_buffer=event_buffer,
                    psd_capture=psd_ctrl,
                    measurement_configuration=self.configuration_document,
                )
                self._mca_controllers.append(mca_ctrl)
                self._connect_optional_signal(
                    mca_ctrl,
                    "session_readback",
                    self._record_mca_readback,
                )
                self._mca_controller_by_device[idx] = mca_ctrl
                mca_docks.append(self._make_dock(f"mca_ch{ch}", ch_label, mca_ctrl))
            else:
                missing_mca_channels.append(ch)

            hv = device.hv
            if hv is None:
                missing_psu_channels.append(ch)
                log.warning("PSU controls unavailable for channel %s", ch)
            else:
                psu_ctrl = PSUController(hv)
                self._connect_optional_signal(
                    psu_ctrl,
                    "session_readback",
                    partial(self._record_psu_readback, ch),
                )
                self._psu_controllers.append(psu_ctrl)
                self._psu_controller_by_device[idx] = psu_ctrl
                psu_docks.append(self._make_dock(f"psu_ch{ch}", ch_label, psu_ctrl))

        self._populate_dock_host(self._scope_dock_host, scope_docks)
        self._populate_dock_host(self._current_dock_host, current_docks)
        self._populate_dock_host(self._mca_dock_host, mca_docks)
        self._populate_dock_host(self._psd_dock_host, psd_docks)
        self._populate_dock_host(self._psu_dock_host, psu_docks)

        self._window.ui.layoutTabScope.addWidget(self._scope_dock_host)
        self._window.ui.layoutTabCurrent.addWidget(self._current_dock_host)
        self._window.ui.layoutTabMCA.addWidget(self._mca_dock_host)
        self._window.ui.layoutTabPSD.addWidget(self._psd_dock_host)
        self._window.ui.layoutTabPSU.addWidget(self._psu_dock_host)

        current_tab_index = self._window.ui.mainTabs.indexOf(self._window.ui.tabCurrent)
        self._window.ui.mainTabs.setTabEnabled(current_tab_index, bool(current_docks))
        if not current_docks:
            current_tooltip = "Current monitor is disabled: no input-filter device was found."
        elif missing_mca_channels:
            channels = ", ".join(str(channel) for channel in missing_mca_channels)
            current_tooltip = f"Current monitor unavailable for channel(s): {channels}."
        else:
            current_tooltip = ""
        self._window.ui.tabCurrent.setToolTip(current_tooltip)

        mca_tab_index = self._window.ui.mainTabs.indexOf(self._window.ui.tabMCA)
        self._window.ui.mainTabs.setTabEnabled(mca_tab_index, bool(mca_docks))
        if not mca_docks:
            mca_tooltip = "MCA is disabled: no pulse-processor/input-filter device was found."
        elif missing_mca_channels:
            channels = ", ".join(str(channel) for channel in missing_mca_channels)
            mca_tooltip = f"MCA controls unavailable for channel(s): {channels}."
        else:
            mca_tooltip = ""
        self._window.ui.tabMCA.setToolTip(mca_tooltip)

        psd_tab_index = self._window.ui.mainTabs.indexOf(self._window.ui.tabPSD)
        self._window.ui.mainTabs.setTabEnabled(psd_tab_index, bool(psd_docks))
        self._window.ui.tabPSD.setToolTip(
            ""
            if psd_docks
            else "PSD is disabled: MCA list-mode DMA is unavailable on all channels."
        )

        psu_tab_index = self._window.ui.mainTabs.indexOf(self._window.ui.tabPSU)
        self._window.ui.mainTabs.setTabEnabled(psu_tab_index, bool(psu_docks))
        if not psu_docks:
            psu_tooltip = "PSU is disabled: no IDS/HV backend connected."
        elif missing_psu_channels:
            channels = ", ".join(str(channel) for channel in missing_psu_channels)
            psu_tooltip = f"PSU controls unavailable for channel(s): {channels}."
        else:
            psu_tooltip = ""
        self._window.ui.tabPSU.setToolTip(psu_tooltip)

    def _build_global_tab(self) -> None:
        """Build the one floatable channel-independent digitizer panel."""
        labels = [self._display_channel(index) for index in range(len(self._devices))]
        self._global_controller = GlobalController(self._devices, labels)
        self._connect_optional_signal(
            self._global_controller,
            "session_temperature_readback",
            self._record_temperature_readback,
        )
        self._connect_optional_signal(
            self._global_controller,
            "session_diagnostics_readback",
            self._record_global_diagnostics,
        )
        dock = self._make_dock("global_panel", "Global", self._global_controller)
        self._populate_dock_host(self._global_dock_host, [dock])
        layout = self._window.ui.layoutTabGlobal
        if layout.indexOf(self._global_dock_host) < 0:
            layout.addWidget(self._global_dock_host)

    def _build_coincidence_tab(self) -> None:
        mca_pair = tuple(self._mca_controller_by_device.get(index) for index in range(2))
        available = (
            len(self._devices) >= 2
            and all(controller is not None for controller in mca_pair)
            and all(device.capabilities.mca_dma for device in self._devices[:2])
            and self._global_controller is not None
            and self._global_controller.sync_available
        )
        index = self._window.ui.mainTabs.indexOf(self._window.ui.tabCoincidence)
        self._window.ui.mainTabs.setTabEnabled(index, available)
        self._window.ui.tabCoincidence.setToolTip(
            ""
            if available
            else "Coincidence requires two IIO MCA list-mode channels and shared software start."
        )
        if available:
            assert self._global_controller is not None
            mca_a, mca_b = mca_pair
            assert mca_a is not None and mca_b is not None
            self._coincidence_controller = CoincidenceController(
                self._devices[:2], [mca_a, mca_b], self._global_controller
            )
            self._connect_optional_signal(
                self._coincidence_controller,
                "session_state",
                self._record_coincidence_state,
            )
            self._window.ui.layoutTabCoincidence.addWidget(self._coincidence_controller)

    def _build_external_docks(
        self,
        devices: Sequence[BaseModbusDevice] | None = None,
    ) -> None:
        """Discover or install Modbus devices and dock one tab for each.

        The SiPM bias board, Geiger-Mueller probe, and PMT HV supply share the
        digitizer's RS-485 bus via a ser2net TCP bridge — same host as the
        IIO digitizer connection. A device that doesn't respond (not present,
        or bus not bridged) is simply absent from the discovery results.
        """
        if devices is None:
            devices = self._external_devices.discover(self._host)
        docks: list[QDockWidget] = []
        for idx, device in enumerate(devices):
            ctrl = ExternalDeviceController(device)
            self._connect_optional_signal(
                ctrl,
                "session_readback",
                self._record_external_readback,
            )
            if getattr(self, "_replay_hardware_interlocked", False):
                ctrl.set_replay_hardware_interlock(True)
            self._external_controllers.append(ctrl)
            label = f"{device.device_type.name.title()} #{device.device_id}"
            docks.append(self._make_dock(f"external_{idx}", label, ctrl))
            log.info("External module panel created: %s", label)

        self._populate_dock_host(self._external_dock_host, docks)
        layout = self._window.ui.layoutTabExternal
        if layout.indexOf(self._external_dock_host) < 0:
            layout.addWidget(self._external_dock_host)

        tab_index = self._window.ui.mainTabs.indexOf(self._window.ui.tabExternal)
        self._window.ui.mainTabs.setTabEnabled(tab_index, bool(docks))
        self._window.ui.tabExternal.setToolTip(
            "" if docks else "No external Modbus modules were detected on the digitizer host."
        )

    def scan_external_devices(self) -> ExternalDeviceScan:
        """Perform a full Modbus scan using a fresh connection owner.

        This method touches no Qt widgets and is safe to call on a background
        thread. The result must later be adopted with
        :meth:`apply_external_device_scan` or explicitly closed.
        """
        return ExternalDevices.scan_new(self._host)

    def prepare_external_device_refresh(self) -> None:
        """Quiesce and close the current Modbus transport before rescanning.

        The board's ser2net configuration may permit only one useful client at
        a time. Keeping the polling connection open while a scanner connects
        can therefore disrupt both. This GUI-thread phase is short: it stops
        the bounded polling transactions, detaches the old docks, and leaves a
        valid empty owner in place for shutdown while discovery runs.
        """
        if self._external_controllers:
            self._external_refresh_application_settings = {
                controller.configuration_id: controller.configuration_settings()
                for controller in self._external_controllers
            }
        self._clear_external_docks()
        previous_owner = self._external_devices
        self._external_devices = ExternalDevices()
        previous_owner.close()

        tab_index = self._window.ui.mainTabs.indexOf(self._window.ui.tabExternal)
        self._window.ui.mainTabs.setTabEnabled(tab_index, False)
        self._window.ui.tabExternal.setToolTip("Refreshing external Modbus devices…")
        log.info("Modbus refresh: old polling stopped; starting a fresh scan")

    def apply_external_device_scan(self, scan: ExternalDeviceScan) -> int:
        """Replace the runtime Modbus device set with a completed scan.

        Existing polling workers are stopped before their shared clients are
        closed. Display preferences are restored by stable device type/address
        when the same module is rediscovered; hardware setpoints are only
        read, never rewritten, during refresh.
        """
        application_settings = self._external_refresh_application_settings
        placeholder_owner = self._external_devices
        self._external_devices = scan.manager
        placeholder_owner.close()

        self._build_external_docks(scan.devices)
        for controller in self._external_controllers:
            settings = application_settings.get(controller.configuration_id)
            if settings is not None:
                controller.apply_configuration_settings(settings)
        self._external_refresh_application_settings = {}

        count = len(scan.devices)
        log.info("Modbus refresh complete: %d external device(s)", count)
        return count

    def _clear_external_docks(self) -> None:
        """Stop external pollers and detach their dock widgets."""
        for controller in self._external_controllers:
            controller.request_polling_stop()
        for controller in self._external_controllers:
            controller.stop_polling_sync()

        for dock in self._external_dock_host.findChildren(QDockWidget):
            self._external_dock_host.removeDockWidget(dock)
            dock.setParent(None)
            widget = dock.widget()
            if widget is not None:
                widget.deleteLater()
            dock.deleteLater()
        self._external_controllers.clear()

    # ------------------------------------------------------------------
    # Dock state persistence
    # ------------------------------------------------------------------

    # Bump suffix when dock object names or topology change so stale layouts
    # are silently discarded rather than corrupting the initial tab arrangement.
    _DOCK_STATE_KEY_SCOPE = "docks/v2/scope"
    _DOCK_STATE_KEY_CURRENT = "docks/v1/current"
    _DOCK_STATE_KEY_MCA = "docks/v2/mca"
    _DOCK_STATE_KEY_PSD = "docks/v2/psd"
    _DOCK_STATE_KEY_PSU = "docks/v2/psu"
    _DOCK_STATE_KEY_GLOBAL = "docks/v2/global"
    _DOCK_STATE_KEY_EXTERNAL = "docks/v2/external"

    def _save_dock_state(self) -> None:
        settings = QSettings()
        settings.setValue(self._DOCK_STATE_KEY_SCOPE, self._scope_dock_host.saveState())
        settings.setValue(self._DOCK_STATE_KEY_CURRENT, self._current_dock_host.saveState())
        settings.setValue(self._DOCK_STATE_KEY_MCA, self._mca_dock_host.saveState())
        settings.setValue(self._DOCK_STATE_KEY_PSD, self._psd_dock_host.saveState())
        settings.setValue(self._DOCK_STATE_KEY_PSU, self._psu_dock_host.saveState())
        settings.setValue(self._DOCK_STATE_KEY_GLOBAL, self._global_dock_host.saveState())
        settings.setValue(self._DOCK_STATE_KEY_EXTERNAL, self._external_dock_host.saveState())

    def _restore_dock_state(self) -> None:
        settings = QSettings()
        if state := settings.value(self._DOCK_STATE_KEY_SCOPE):
            if not self._scope_dock_host.restoreState(state):
                settings.remove(self._DOCK_STATE_KEY_SCOPE)
        if state := settings.value(self._DOCK_STATE_KEY_CURRENT):
            if not self._current_dock_host.restoreState(state):
                settings.remove(self._DOCK_STATE_KEY_CURRENT)
        if state := settings.value(self._DOCK_STATE_KEY_MCA):
            if not self._mca_dock_host.restoreState(state):
                settings.remove(self._DOCK_STATE_KEY_MCA)
        if state := settings.value(self._DOCK_STATE_KEY_PSD):
            if not self._psd_dock_host.restoreState(state):
                settings.remove(self._DOCK_STATE_KEY_PSD)
        if state := settings.value(self._DOCK_STATE_KEY_PSU):
            if not self._psu_dock_host.restoreState(state):
                settings.remove(self._DOCK_STATE_KEY_PSU)
        if state := settings.value(self._DOCK_STATE_KEY_GLOBAL):
            if not self._global_dock_host.restoreState(state):
                settings.remove(self._DOCK_STATE_KEY_GLOBAL)
        if state := settings.value(self._DOCK_STATE_KEY_EXTERNAL):
            if not self._external_dock_host.restoreState(state):
                settings.remove(self._DOCK_STATE_KEY_EXTERNAL)

    def reset_dock_layout(self) -> None:
        """Clear saved dock state and re-tabify all channel docks."""
        settings = QSettings()
        settings.remove(self._DOCK_STATE_KEY_SCOPE)
        settings.remove(self._DOCK_STATE_KEY_CURRENT)
        settings.remove(self._DOCK_STATE_KEY_MCA)
        settings.remove(self._DOCK_STATE_KEY_PSD)
        settings.remove(self._DOCK_STATE_KEY_PSU)
        settings.remove(self._DOCK_STATE_KEY_GLOBAL)
        settings.remove(self._DOCK_STATE_KEY_EXTERNAL)

        dock_hosts = (
            self._scope_dock_host,
            self._current_dock_host,
            self._mca_dock_host,
            self._psd_dock_host,
            self._psu_dock_host,
            self._global_dock_host,
            self._external_dock_host,
        )
        for host in dock_hosts:
            docks = host.findChildren(QDockWidget)
            if not docks:
                continue
            for dock in docks:
                host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
            if len(docks) < 2:
                continue
            for i in range(1, len(docks)):
                host.tabifyDockWidget(docks[i - 1], docks[i])
            docks[0].raise_()

        log.info("Dock layout reset to default (tabbed)")

    def set_roi_visible(self, visible: bool) -> None:
        """Toggle the ROI selection tool + stats panel on every MCA histogram."""
        for ctrl in self._mca_controllers:
            ctrl.set_roi_visible(visible)
        log.info("ROI %s on all MCA histograms", "shown" if visible else "hidden")

    def set_log_y(self, enabled: bool) -> None:
        """Toggle logarithmic Y-axis on every MCA histogram."""
        for ctrl in self._mca_controllers:
            ctrl.set_log_y(enabled)
        log.info("Histogram Y-axis set to %s", "log" if enabled else "linear")

    def reset_all_zoom(self) -> None:
        """Auto-range/reset zoom on every live plot."""
        for scope_controller in self._scope_controllers:
            scope_controller.reset_zoom()
        for current_controller in self._current_monitor_controllers:
            current_controller.reset_zoom()
        for mca_controller in self._mca_controllers:
            mca_controller.reset_zoom()
        for psd_controller in self._psd_controllers:
            psd_controller.reset_zoom()
        log.info("Zoom reset on all plots")

    def refresh_dma_output_settings(self) -> None:
        """Refresh format-dependent controls after the DMA settings dialog."""
        for ctrl in self._mca_controllers:
            ctrl.refresh_dma_output_settings()
        if self._coincidence_controller is not None:
            self._coincidence_controller.refresh_dma_output_settings()

    def general_configuration_settings(self) -> dict[str, int]:
        """Return the application-wide controls now edited in General Settings."""
        scope = (
            self._scope_controllers[0].configuration_settings()
            if self._scope_controllers
            else {}
        )
        mca = (
            self._mca_controllers[0].configuration_settings()
            if self._mca_controllers
            else {}
        )
        psu = (
            self._psu_controllers[0].configuration_settings()
            if self._psu_controllers
            else {}
        )
        return {
            "scope_display_mode": int(scope.get("display_mode", 0)),
            "scope_persistence": int(scope.get("persistence", 900)),
            "scope_refresh_rate_hz": int(scope.get("refresh_rate_hz", 10)),
            "mca_refresh_rate_hz": int(cast(int, mca.get("refresh_rate_hz", 5))),
            "psu_refresh_interval_ms": int(psu.get("refresh_interval_ms", 1000)),
            "psu_plot_time_range_s": int(psu.get("plot_time_range_s", 60)),
        }

    def apply_general_configuration_settings(self, settings: object) -> None:
        """Apply one set of display/polling values to every applicable channel."""
        if not isinstance(settings, dict):
            return
        scope_settings = {
            "display_mode": settings.get("scope_display_mode", 0),
            "persistence": settings.get("scope_persistence", 900),
            "refresh_rate_hz": settings.get("scope_refresh_rate_hz", 10),
        }
        mca_settings = {"refresh_rate_hz": settings.get("mca_refresh_rate_hz", 5)}
        psu_settings = {
            "refresh_interval_ms": settings.get("psu_refresh_interval_ms", 1000),
            "plot_time_range_s": settings.get("psu_plot_time_range_s", 60),
        }
        for scope_ctrl in self._scope_controllers:
            scope_ctrl.apply_configuration_settings(scope_settings)
        for mca_ctrl in self._mca_controllers:
            mca_ctrl.apply_configuration_settings(mca_settings)
        for psu_ctrl in self._psu_controllers:
            psu_ctrl.apply_configuration_settings(psu_settings)

    def show_energy_calibration(self) -> None:
        """Open one modeless calibration workspace for all available MCAs."""
        if self._energy_calibration_dialog is None:
            self._energy_calibration_dialog = EnergyCalibrationDialog(
                self._mca_controllers,
                parent=self._window,
            )
        self._energy_calibration_dialog.show_workspace()

    def show_mca_peak_analysis(self) -> None:
        """Open one modeless peak-analysis workbench for live and saved spectra."""
        if self._mca_peak_analysis_dialog is None:
            self._mca_peak_analysis_dialog = McaPeakAnalysisDialog(
                self._mca_controllers,
                parent=self._window,
            )
        self._mca_peak_analysis_dialog.show_workspace()

    # ------------------------------------------------------------------
    # Versioned session recording and analysis replay
    # ------------------------------------------------------------------

    @property
    def session_recording(self) -> bool:
        recorder = getattr(self, "_session_recorder", None)
        return recorder is not None and recorder.is_recording

    def start_session_recording(self, path: Path) -> Path:
        """Passively record all data already observed by this application."""
        if self.session_replay is not None:
            raise RuntimeError("exit session replay before starting a recording")
        if getattr(self, "_session_recorder", None) is not None:
            raise RuntimeError("a session recording is already active")
        recorder = SessionRecorder(
            path,
            self.configuration_document(),
            mca_event_buffers=self._mca_event_buffers,
            scope_frame_buffers=self._scope_frame_buffers,
        )
        self._session_recorder = recorder
        coincidence = getattr(self, "_coincidence_controller", None)
        if coincidence is not None and coincidence.active:
            recorder.mark_integrity_issue("session_started_during_active_coincidence")
            recorder.record_timeline(
                "coincidence_observation",
                "coincidence",
                {"state": "already_active", "coverage": "partial"},
            )
        for controller in self._mca_controllers:
            if not controller.set_session_capture_enabled(True):
                recorder.mark_integrity_issue(
                    f"mca_ch{controller.channel}_session_started_during_unobserved_dma"
                )
        recorder.record_command(
            "application",
            "start_session_recording",
            {"path": str(path)},
        )
        log.info("Session recording started: %s", path)
        return path

    def stop_session_recording(self) -> Path | None:
        recorder = self._session_recorder
        if recorder is None:
            return None
        try:
            try:
                final_configuration = self.configuration_document()
            except Exception:
                log.exception("Capturing the final session configuration failed")
                recorder.mark_integrity_issue("final_configuration_snapshot_failed")
                final_configuration = None
            recorder.record_command("application", "stop_session_recording")
            path = recorder.stop(final_configuration=final_configuration)
        finally:
            for controller in self._mca_controllers:
                controller.set_session_capture_enabled(False)
            self._session_recorder = None
        log.info("Session recording finalized: %s", path)
        return path

    def replay_session(self, path: Path) -> SessionReplay:
        """Replay captured analysis inputs and monitored telemetry once."""
        if self.session_recording:
            raise RuntimeError("stop the active session recording before replay")
        coincidence = getattr(self, "_coincidence_controller", None)
        if coincidence is not None and coincidence.active:
            raise RuntimeError("stop the coincidence measurement before replay")
        replay = SessionReplay(path)
        spectra = replay.replay_mca_spectra()
        scope_frames = replay.scope_frames()
        telemetry_reader = getattr(replay, "telemetry_events", None)
        telemetry_events = telemetry_reader() if telemetry_reader is not None else []
        self.clear_session_replay()
        self._session_replay = replay
        self._session_replay_path = path
        try:
            self._set_replay_hardware_interlock(True)
            if spectra:
                if self._mca_peak_analysis_dialog is None:
                    self._mca_peak_analysis_dialog = McaPeakAnalysisDialog(
                        self._mca_controllers,
                        parent=self._window,
                    )
                assert self._mca_peak_analysis_dialog is not None
                self._mca_peak_analysis_dialog.add_spectra(spectra)
            self._session_replay_spectrum_ids = {
                spectrum.spectrum_id for spectrum in spectra
            }
            self._start_session_analysis_replay(replay)
            self._start_session_scope_replay(replay, scope_frames)
            self._start_session_telemetry_replay(replay, telemetry_events)
        except Exception:
            self.clear_session_replay()
            raise
        log.info(
            "Session replay loaded: %s (%d MCA spectra, %d Scope frames, "
            "%d telemetry updates, integrity=%s)",
            path,
            len(spectra),
            len(self._session_scope_replay_frames),
            len(self._session_telemetry_replay_events),
            replay.integrity,
        )
        return replay

    def _set_replay_hardware_interlock(
        self,
        enabled: bool,
        *,
        resume_background: bool = True,
    ) -> None:
        """Apply one app-wide hardware-write boundary for session replay."""
        enabled = bool(enabled)
        if enabled == getattr(self, "_replay_hardware_interlocked", False):
            return
        targets: list[object] = []
        coincidence = getattr(self, "_coincidence_controller", None)
        if coincidence is not None:
            targets.append(coincidence)
        targets.extend(getattr(self, "_current_monitor_controllers", []))
        targets.extend(getattr(self, "_scope_controllers", []))
        targets.extend(getattr(self, "_mca_controllers", []))
        global_controller = getattr(self, "_global_controller", None)
        if global_controller is not None:
            targets.append(global_controller)
        psu_controllers = getattr(self, "_psu_controllers", [])
        external_controllers = getattr(self, "_external_controllers", [])
        targets.extend(psu_controllers)
        targets.extend(external_controllers)
        background_targets = {
            id(target)
            for target in [global_controller, *psu_controllers, *external_controllers]
            if target is not None
        }

        applied: list[object] = []
        try:
            for target in targets:
                setter = getattr(target, "set_replay_hardware_interlock", None)
                if setter is not None:
                    if id(target) in background_targets:
                        setter(enabled, resume_background=resume_background)
                    else:
                        setter(enabled)
                    applied.append(target)
        except Exception:
            for target in reversed(applied):
                setter = getattr(target, "set_replay_hardware_interlock", None)
                if setter is not None:
                    try:
                        if id(target) in background_targets:
                            setter(not enabled, resume_background=False)
                        else:
                            setter(not enabled)
                    except Exception:
                        log.exception("Failed to roll back replay hardware interlock")
            raise
        self._replay_hardware_interlocked = enabled

    def _start_session_analysis_replay(self, replay: SessionReplay) -> None:
        """Replay MCA polls and list-mode batches over their captured timeline."""
        mca_controllers = {
            controller.channel: controller
            for controller in getattr(self, "_mca_controllers", [])
        }
        psd_controllers = getattr(self, "_psd_controllers", [])
        coincidence = getattr(self, "_coincidence_controller", None)
        if not mca_controllers and not psd_controllers and coincidence is None:
            return
        readbacks = [
            readback
            for readback in replay.mca_readbacks()
            if readback.channel in mca_controllers
        ]
        batches = replay.mca_event_batches()
        event_channels = {batch.channel for batch in batches}
        configuration = replay.configuration_snapshot()
        application = configuration.get("application")
        application = application if isinstance(application, dict) else {}
        channel_settings = application.get("channels")
        channel_settings = channel_settings if isinstance(channel_settings, dict) else {}
        hardware = configuration.get("hardware")
        hardware = hardware if isinstance(hardware, dict) else {}
        hardware_channels = hardware.get("channels")
        hardware_channels = hardware_channels if isinstance(hardware_channels, dict) else {}

        replayed_mca_channels = {readback.channel for readback in readbacks}
        for channel in replayed_mca_channels:
            mca_controllers[channel].begin_session_replay()
        self._session_analysis_replay_mca_channels = replayed_mca_channels

        replayed_psd: set[int] = set()
        for controller in psd_controllers:
            channel = controller.channel
            if channel not in event_channels:
                continue
            hardware_channel = hardware_channels.get(str(channel), {})
            hardware_channel = hardware_channel if isinstance(hardware_channel, dict) else {}
            mca_hardware = hardware_channel.get("mca", {})
            mca_hardware = mca_hardware if isinstance(mca_hardware, dict) else {}
            charge_comparison = mca_hardware.get("charge_comparison", {})
            if (
                isinstance(charge_comparison, dict)
                and charge_comparison.get("enable") is False
            ):
                continue
            saved_channel = channel_settings.get(str(channel), {})
            saved_channel = saved_channel if isinstance(saved_channel, dict) else {}
            try:
                controller.begin_session_replay(
                    settings=saved_channel.get("psd"),
                )
            except Exception:
                log.exception("PSD session replay failed for channel %d", channel)
            else:
                replayed_psd.add(channel)
        self._session_replay_psd_channels = replayed_psd

        ordered_mca = getattr(self, "_mca_controllers", [])
        coincidence_channels: dict[int, int] = {}
        interval_reader = getattr(replay, "coincidence_intervals", None)
        coincidence_intervals = interval_reader() if interval_reader is not None else []
        explicit_timing_reader = getattr(replay, "has_explicit_mca_batch_timing", None)
        explicit_timing = (
            explicit_timing_reader() if explicit_timing_reader is not None else False
        )
        allow_coincidence = bool(coincidence_intervals) or not explicit_timing
        if coincidence is not None and len(ordered_mca) >= 2 and allow_coincidence:
            channels = (ordered_mca[0].channel, ordered_mca[1].channel)
            if set(channels).issubset(event_channels):
                try:
                    settings, display_settings = replay.coincidence_settings(channels)
                    coincidence.begin_session_replay(settings, display_settings)
                except Exception as exc:
                    log.warning(
                        "Coincidence session replay is unavailable: %s",
                        exc,
                        exc_info=True,
                    )
                    status = getattr(coincidence, "status", None)
                    if status is not None:
                        status.setText(f"Session coincidence replay unavailable: {exc}")
                else:
                    self._session_replay_coincidence_active = True
                    coincidence_channels = {channels[0]: 0, channels[1]: 1}
        self._session_analysis_replay_coincidence_channels = coincidence_channels
        self._session_analysis_replay_coincidence_intervals = (
            coincidence_intervals if explicit_timing else None
        )

        used_batch_channels = replayed_psd | set(coincidence_channels)
        playable: list[McaReplayBatch | McaReplayReadback] = [
            *readbacks,
            *(batch for batch in batches if batch.channel in used_batch_channels),
        ]
        playable.sort(
            key=lambda item: (
                item.host_monotonic_ns,
                0 if isinstance(item, McaReplayBatch) else 1,
                item.channel,
                item.sequence,
            )
        )
        if not playable:
            return
        first_timestamp = playable[0].host_monotonic_ns
        manifest_origin = replay.manifest.get("started_monotonic_ns")
        origin = (
            int(manifest_origin)
            if isinstance(manifest_origin, int)
            and not isinstance(manifest_origin, bool)
            and 0 < manifest_origin <= first_timestamp
            else first_timestamp
        )
        self._session_analysis_replay_events = playable
        self._session_analysis_replay_index = 0
        self._session_analysis_replay_origin_ns = origin
        self._session_analysis_replay_started_ns = monotonic_ns()
        self._advance_session_analysis_replay()

    def _advance_session_analysis_replay(self) -> None:
        events = self._session_analysis_replay_events
        index = self._session_analysis_replay_index
        if getattr(self, "_session_replay", None) is None or index >= len(events):
            return
        elapsed_ns = max(0, monotonic_ns() - self._session_analysis_replay_started_ns)
        mca_controllers = {
            controller.channel: controller
            for controller in getattr(self, "_mca_controllers", [])
        }
        psd_controllers = {
            controller.channel: controller
            for controller in getattr(self, "_psd_controllers", [])
            if controller.channel in self._session_replay_psd_channels
        }
        coincidence = getattr(self, "_coincidence_controller", None)
        latest_readback: dict[int, McaReplayReadback] = {}
        while index < len(events):
            event = events[index]
            recorded_elapsed_ns = max(
                0,
                event.host_monotonic_ns - self._session_analysis_replay_origin_ns,
            )
            if recorded_elapsed_ns > elapsed_ns:
                break
            if isinstance(event, McaReplayReadback):
                latest_readback[event.channel] = event
            else:
                psd = psd_controllers.get(event.channel)
                if psd is not None:
                    try:
                        psd.display_session_replay_events(event.events)
                    except Exception:
                        log.exception("PSD replay batch failed for channel %d", event.channel)
                coincidence_channel = self._session_analysis_replay_coincidence_channels.get(
                    event.channel
                )
                intervals = getattr(
                    self,
                    "_session_analysis_replay_coincidence_intervals",
                    None,
                )
                inside_coincidence = intervals is None or any(
                    start <= event.host_monotonic_ns
                    and (end is None or event.host_monotonic_ns <= end)
                    for start, end in intervals
                )
                if (
                    coincidence is not None
                    and coincidence_channel is not None
                    and inside_coincidence
                ):
                    try:
                        coincidence.display_session_replay_events(
                            coincidence_channel,
                            event.events,
                        )
                    except Exception as exc:
                        log.warning(
                            "Progressive coincidence replay failed: %s",
                            exc,
                            exc_info=True,
                        )
                        self._session_analysis_replay_coincidence_channels = {}
                        status = getattr(coincidence, "status", None)
                        if status is not None:
                            status.setText(f"Session coincidence replay unavailable: {exc}")
            index += 1
        self._session_analysis_replay_index = index
        for channel, readback in latest_readback.items():
            controller = mca_controllers.get(channel)
            if controller is not None:
                controller.display_session_replay_readback(readback.readback)

        if index >= len(events):
            for channel in self._session_analysis_replay_mca_channels:
                controller = mca_controllers.get(channel)
                if controller is not None:
                    controller.finish_session_replay()
            for controller in psd_controllers.values():
                controller.finish_session_replay()
            if coincidence is not None and self._session_replay_coincidence_active:
                try:
                    coincidence.finish_session_replay()
                except Exception as exc:
                    log.warning("Finalizing coincidence replay failed: %s", exc, exc_info=True)
            log.info("MCA/PSD/coincidence session replay reached EOF; final views are held")
            return

        next_elapsed_ns = max(
            0,
            events[index].host_monotonic_ns - self._session_analysis_replay_origin_ns,
        )
        remaining_ns = max(0, next_elapsed_ns - elapsed_ns)
        delay_ms = max(1, (remaining_ns + 999_999) // 1_000_000)
        self._session_analysis_replay_timer.start(min(delay_ms, 2_147_000_000))

    def _start_session_scope_replay(
        self,
        replay: SessionReplay,
        frames: list[ScopeReplayFrame],
    ) -> None:
        controllers = {
            controller.channel: controller for controller in self._scope_controllers
        }
        playable = [frame for frame in frames if frame.channel in controllers]
        missing = sorted({frame.channel for frame in frames} - controllers.keys())
        if missing:
            log.warning(
                "Session has Scope replay data for unavailable channel(s): %s",
                ", ".join(str(channel) for channel in missing),
            )
        if not playable:
            return

        first_timestamp = playable[0].host_monotonic_ns
        manifest_origin = replay.manifest.get("started_monotonic_ns")
        origin = (
            int(manifest_origin)
            if isinstance(manifest_origin, int)
            and not isinstance(manifest_origin, bool)
            and 0 < manifest_origin <= first_timestamp
            else first_timestamp
        )
        frame_counts: dict[int, int] = {}
        for frame in playable:
            frame_counts[frame.channel] = frame_counts.get(frame.channel, 0) + 1
        channels = set(frame_counts)
        duration_ns = max(0, playable[-1].host_monotonic_ns - origin)

        self._session_scope_replay_frames = playable
        self._session_scope_replay_index = 0
        self._session_scope_replay_origin_ns = origin
        self._session_scope_replay_channels = channels
        self._session_scope_replay_frame_counts = frame_counts
        self._session_scope_replay_delivered = dict.fromkeys(channels, 0)
        for channel in sorted(channels):
            controllers[channel].begin_session_replay(
                frame_count=frame_counts[channel],
                duration_ns=duration_ns,
            )

        self._session_scope_replay_started_ns = monotonic_ns()
        self._advance_session_scope_replay()

    def _advance_session_scope_replay(self) -> None:
        frames = self._session_scope_replay_frames
        index = self._session_scope_replay_index
        if self._session_replay is None or index >= len(frames):
            return
        elapsed_ns = max(0, monotonic_ns() - self._session_scope_replay_started_ns)
        due_by_channel: dict[int, ScopeReplayFrame] = {}
        while index < len(frames):
            frame = frames[index]
            recorded_elapsed_ns = max(
                0,
                frame.host_monotonic_ns - self._session_scope_replay_origin_ns,
            )
            if recorded_elapsed_ns > elapsed_ns:
                break
            due_by_channel[frame.channel] = frame
            self._session_scope_replay_delivered[frame.channel] += 1
            index += 1
        self._session_scope_replay_index = index

        controllers = {
            controller.channel: controller for controller in self._scope_controllers
        }
        duration_ns = max(
            0,
            frames[-1].host_monotonic_ns - self._session_scope_replay_origin_ns,
        )
        for channel, frame in due_by_channel.items():
            controller = controllers.get(channel)
            if controller is None:
                continue
            controller.display_session_replay_frame(
                frame.samples,
                sample_period_ns=frame.sample_period_ns,
                frame_index=self._session_scope_replay_delivered[channel],
                frame_count=self._session_scope_replay_frame_counts[channel],
                elapsed_ns=max(
                    0,
                    frame.host_monotonic_ns - self._session_scope_replay_origin_ns,
                ),
                duration_ns=duration_ns,
                source=frame.source,
            )

        if index >= len(frames):
            for channel in sorted(self._session_scope_replay_channels):
                controller = controllers.get(channel)
                if controller is not None:
                    controller.finish_session_replay(
                        frame_count=self._session_scope_replay_frame_counts[channel],
                        duration_ns=duration_ns,
                    )
            log.info("Scope session replay reached EOF; final frames are held")
            return

        next_elapsed_ns = max(
            0,
            frames[index].host_monotonic_ns - self._session_scope_replay_origin_ns,
        )
        remaining_ns = max(0, next_elapsed_ns - elapsed_ns)
        delay_ms = max(1, (remaining_ns + 999_999) // 1_000_000)
        self._session_scope_replay_timer.start(min(delay_ms, 2_147_000_000))

    def _psu_replay_controllers(self) -> dict[int, PSUController]:
        result: dict[int, PSUController] = {}
        for device_index, controller in getattr(
            self,
            "_psu_controller_by_device",
            {},
        ).items():
            result[self._display_channel(device_index)] = controller
        return result

    def _start_session_telemetry_replay(
        self,
        replay: SessionReplay,
        events: list[TelemetryReplayEvent],
    ) -> None:
        psu_controllers = self._psu_replay_controllers()
        external_controllers = {
            controller.configuration_id: controller
            for controller in getattr(self, "_external_controllers", [])
        }
        playable: list[TelemetryReplayEvent] = []
        psu_channels: set[int] = set()
        external_ids: set[str] = set()
        for event in events:
            if event.source.startswith("psu.ch"):
                try:
                    channel = int(event.source.removeprefix("psu.ch"))
                except ValueError:
                    continue
                if channel in psu_controllers:
                    playable.append(event)
                    psu_channels.add(channel)
            elif event.source.startswith("external."):
                configuration_id = event.source.removeprefix("external.")
                if configuration_id in external_controllers:
                    playable.append(event)
                    external_ids.add(configuration_id)
        if not playable:
            return

        playable.sort(key=lambda event: (event.host_monotonic_ns, event.sequence))
        first_timestamp = playable[0].host_monotonic_ns
        manifest_origin = replay.manifest.get("started_monotonic_ns")
        origin = (
            int(manifest_origin)
            if isinstance(manifest_origin, int)
            and not isinstance(manifest_origin, bool)
            and 0 < manifest_origin <= first_timestamp
            else first_timestamp
        )
        self._session_telemetry_replay_events = playable
        self._session_telemetry_replay_index = 0
        self._session_telemetry_replay_origin_ns = origin
        self._session_telemetry_replay_started_ns = monotonic_ns()
        self._session_telemetry_replay_psu_channels = psu_channels
        self._session_telemetry_replay_external_ids = external_ids
        for channel in psu_channels:
            psu_controllers[channel].begin_session_replay()
        for configuration_id in external_ids:
            external_controllers[configuration_id].begin_session_replay()
        self._advance_session_telemetry_replay()

    def _advance_session_telemetry_replay(self) -> None:
        events = self._session_telemetry_replay_events
        index = self._session_telemetry_replay_index
        if self._session_replay is None or index >= len(events):
            return
        elapsed_ns = max(0, monotonic_ns() - self._session_telemetry_replay_started_ns)
        psu_controllers = self._psu_replay_controllers()
        external_controllers = {
            controller.configuration_id: controller
            for controller in getattr(self, "_external_controllers", [])
        }
        while index < len(events):
            event = events[index]
            recorded_elapsed_ns = max(
                0,
                event.host_monotonic_ns - self._session_telemetry_replay_origin_ns,
            )
            if recorded_elapsed_ns > elapsed_ns:
                break
            if event.source.startswith("psu.ch"):
                channel = int(event.source.removeprefix("psu.ch"))
                controller = psu_controllers.get(channel)
                readback = event.payload.get("readback")
                if controller is not None and isinstance(readback, dict):
                    controller.display_session_replay_readback(readback)
            elif event.source.startswith("external."):
                configuration_id = event.source.removeprefix("external.")
                controller = external_controllers.get(configuration_id)
                timestamp = event.payload.get("worker_timestamp")
                values = event.payload.get("values")
                if controller is not None and isinstance(timestamp, int | float):
                    controller.display_session_replay_readback(float(timestamp), values)
            index += 1
        self._session_telemetry_replay_index = index
        if index >= len(events):
            log.info("Session telemetry replay reached EOF; final readbacks are held")
            return
        next_elapsed_ns = max(
            0,
            events[index].host_monotonic_ns - self._session_telemetry_replay_origin_ns,
        )
        remaining_ns = max(0, next_elapsed_ns - elapsed_ns)
        delay_ms = max(1, (remaining_ns + 999_999) // 1_000_000)
        self._session_telemetry_replay_timer.start(min(delay_ms, 2_147_000_000))

    @property
    def session_replay(self) -> SessionReplay | None:
        return getattr(self, "_session_replay", None)

    @property
    def session_replay_path(self) -> Path | None:
        return getattr(self, "_session_replay_path", None)

    def clear_session_replay(self, *, resume_background: bool = True) -> None:
        """Leave replay mode without resuming any stopped live acquisition."""
        timer = getattr(self, "_session_scope_replay_timer", None)
        if timer is not None:
            timer.stop()
        analysis_timer = getattr(self, "_session_analysis_replay_timer", None)
        if analysis_timer is not None:
            analysis_timer.stop()
        telemetry_timer = getattr(self, "_session_telemetry_replay_timer", None)
        if telemetry_timer is not None:
            telemetry_timer.stop()
        replay_channels: set[int] = getattr(
            self,
            "_session_scope_replay_channels",
            set(),
        )
        for controller in getattr(self, "_scope_controllers", []):
            if controller.channel in replay_channels:
                try:
                    controller.end_session_replay()
                except Exception:
                    log.exception(
                        "Failed to leave Scope replay on channel %d",
                        controller.channel,
                    )
        mca_channels: set[int] = getattr(
            self,
            "_session_analysis_replay_mca_channels",
            set(),
        )
        for controller in getattr(self, "_mca_controllers", []):
            if controller.channel in mca_channels:
                try:
                    controller.end_session_replay()
                except Exception:
                    log.exception("Failed to leave MCA replay on channel %d", controller.channel)
        psd_channels: set[int] = getattr(self, "_session_replay_psd_channels", set())
        for controller in getattr(self, "_psd_controllers", []):
            if controller.channel in psd_channels:
                try:
                    controller.end_session_replay()
                except Exception:
                    log.exception("Failed to leave PSD replay on channel %d", controller.channel)
        coincidence = getattr(self, "_coincidence_controller", None)
        if coincidence is not None and getattr(
            self,
            "_session_replay_coincidence_active",
            False,
        ):
            try:
                coincidence.end_session_replay()
            except Exception:
                log.exception("Failed to leave coincidence replay")
        psu_controllers = self._psu_replay_controllers()
        for channel in getattr(self, "_session_telemetry_replay_psu_channels", set()):
            controller = psu_controllers.get(channel)
            if controller is not None:
                controller.end_session_replay()
        external_controllers = {
            controller.configuration_id: controller
            for controller in getattr(self, "_external_controllers", [])
        }
        for configuration_id in getattr(
            self,
            "_session_telemetry_replay_external_ids",
            set(),
        ):
            controller = external_controllers.get(configuration_id)
            if controller is not None:
                controller.end_session_replay()
        spectrum_ids: set[str] = getattr(
            self,
            "_session_replay_spectrum_ids",
            set(),
        )
        dialog = getattr(self, "_mca_peak_analysis_dialog", None)
        if dialog is not None and spectrum_ids:
            dialog.remove_spectra(spectrum_ids)
        self._session_replay = None
        self._session_replay_path = None
        self._session_replay_spectrum_ids = set()
        self._session_replay_psd_channels = set()
        self._session_replay_coincidence_active = False
        self._session_analysis_replay_events = []
        self._session_analysis_replay_index = 0
        self._session_analysis_replay_origin_ns = 0
        self._session_analysis_replay_started_ns = 0
        self._session_analysis_replay_mca_channels = set()
        self._session_analysis_replay_coincidence_channels = {}
        self._session_analysis_replay_coincidence_intervals = None
        self._session_scope_replay_frames = []
        self._session_scope_replay_index = 0
        self._session_scope_replay_origin_ns = 0
        self._session_scope_replay_started_ns = 0
        self._session_scope_replay_channels = set()
        self._session_scope_replay_frame_counts = {}
        self._session_scope_replay_delivered = {}
        self._session_telemetry_replay_events = []
        self._session_telemetry_replay_index = 0
        self._session_telemetry_replay_origin_ns = 0
        self._session_telemetry_replay_started_ns = 0
        self._session_telemetry_replay_psu_channels = set()
        self._session_telemetry_replay_external_ids = set()
        self._set_replay_hardware_interlock(
            False,
            resume_background=resume_background,
        )

    def _record_mca_readback(self, channel: int, readback: object) -> None:
        recorder = self._session_recorder
        if recorder is not None:
            recorder.record_mca_readback(channel, readback)

    def _record_scope_viewer_frame(
        self,
        channel: int,
        samples: object,
        sample_period_ns: int,
    ) -> None:
        recorder = self._session_recorder
        if recorder is not None:
            recorder.record_scope_viewer(
                channel,
                samples,
                sample_period_ns=sample_period_ns,
            )

    def _record_current_samples(self, channel: int, batch: object) -> None:
        recorder = self._session_recorder
        if recorder is not None:
            recorder.record_current_samples(channel, batch)

    def _record_psu_readback(self, channel: int, readback: object) -> None:
        recorder = self._session_recorder
        if recorder is not None:
            recorder.record_telemetry(f"psu.ch{channel}", {"readback": readback})

    def _record_temperature_readback(self, readback: object) -> None:
        recorder = self._session_recorder
        if recorder is not None:
            recorder.record_telemetry(
                "global.temperature_correction",
                {"readback": readback},
            )

    def _record_global_diagnostics(self, readings: object) -> None:
        recorder = self._session_recorder
        if recorder is not None:
            recorder.record_telemetry("global.diagnostics", {"readings": readings})

    def _record_external_readback(
        self,
        configuration_id: str,
        timestamp: float,
        snapshot: object,
    ) -> None:
        recorder = self._session_recorder
        if recorder is not None:
            recorder.record_telemetry(
                f"external.{configuration_id}",
                {"worker_timestamp": timestamp, "values": snapshot},
            )

    def _record_coincidence_state(self, state: str, values: object) -> None:
        recorder = self._session_recorder
        if recorder is not None:
            payload = values if isinstance(values, dict) else {"value": values}
            recorder.record_timeline(
                f"coincidence_{state}",
                "coincidence",
                payload,
            )

    def show_psd_event_readback(self, path: Path | None = None) -> None:
        """Open the hardware-independent saved-event PSD workbench."""
        if self._psd_readback_dialog is None:
            self._psd_readback_dialog = PsdReadbackDialog(parent=self._window)
        self._psd_readback_dialog.show_workspace()
        if path is not None:
            self._psd_readback_dialog.open_path(path)

    def show_waveform_analysis(self, path: Path | None = None) -> None:
        """Open the waveform browser and offline PSD reconstruction workbench."""
        if self._waveform_analysis_dialog is None:
            self._waveform_analysis_dialog = WaveformAnalysisDialog(parent=self._window)
        self._waveform_analysis_dialog.show_workspace()
        if path is not None:
            self._waveform_analysis_dialog.open_path(path)

    def load_psd_events(self, path: Path) -> None:
        """Compatibility wrapper opening the standalone PSD readback tool."""
        self.show_psd_event_readback(path)

    def load_waveform_file(self, path: Path) -> None:
        """Compatibility wrapper opening the standalone waveform workbench."""
        self.show_waveform_analysis(path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare_for_remote_power_action(self) -> None:
        """Stop acquisition cleanly before the board is rebooted or powered off.

        Device handles deliberately remain open until reconnect/shutdown. With
        all polling and DMA threads stopped they cannot issue requests while
        Dropbear executes the power command, and ``reconnect()`` can dispose of
        the old handles through its normal teardown path after the board boots.
        """
        log.info("Remote board power action: stopping all acquisition workers")
        self._save_dock_state()
        for ctrl in self._scope_controllers:
            ctrl.save_display_settings()
        self._stop_all_workers()
        log.info("Remote board power action: acquisition stopped")

    def _stop_all_workers(self) -> None:
        """Stop all running timers/workers (blocking). Devices stay open."""
        coincidence = getattr(self, "_coincidence_controller", None)
        if coincidence is not None:
            coincidence.request_shutdown()
        # Broadcast the cheap stop requests before waiting for any individual
        # worker. Independent IIO/Modbus calls then finish concurrently rather
        # than making shutdown pay every transport timeout in series.
        if self._global_controller is not None:
            self._global_controller.request_polling_stop()
        for psu_controller in self._psu_controllers:
            psu_controller.request_monitor_stop()
        for external_controller in self._external_controllers:
            external_controller.request_polling_stop()
        for current_controller in self._current_monitor_controllers:
            current_controller.request_monitor_stop()

        if self._global_controller is not None:
            self._global_controller.disarm_sync()
            self._global_controller.stop_polling_sync()

        # 1. Stop scope timers — no new workers will be submitted
        for scope_controller in self._scope_controllers:
            scope_controller._refresh_timer.stop()

        # 2. Stop DMA workers (blocking)
        for scope_controller in self._scope_controllers:
            scope_controller.stop_dma_sync()
        for mca_controller in self._mca_controllers:
            mca_controller.stop_dma_sync()

        # 3. Stop MCA polling workers (blocking)
        for mca_controller in self._mca_controllers:
            mca_controller.stop_worker_sync()

        # 4. Current monitors own independent direct-IIO connections.
        for current_controller in self._current_monitor_controllers:
            current_controller.stop_monitor_sync()

        if coincidence is not None:
            coincidence.finish_shutdown_sync()

        # PSD timers consume only already-decoded display batches. Stop them
        # after MCA DMA tail drain has completed.
        for psd_controller in self._psd_controllers:
            psd_controller.stop_processing()

        # 5. Wait for in-flight scope workers to finish
        viewer_pool_idle = QThreadPool.globalInstance().waitForDone(3000)
        if viewer_pool_idle:
            for scope_controller in self._scope_controllers:
                scope_controller.close_viewer_client()
        else:
            # An in-flight worker still owns its isolated context. Do not
            # close that context out from under a native libiio read.
            log.warning("Scope viewer workers did not stop within 3 seconds")

        # 6. Stop PSU workers (blocking)
        for psu_controller in self._psu_controllers:
            psu_controller.stop_monitor_sync()

        # 7. Stop external Modbus device workers (blocking)
        for external_controller in self._external_controllers:
            external_controller.stop_polling_sync()

        # Producers are now quiet, so session subscribers can drain their
        # final complete DMA records before the manifest is finalized.
        if getattr(self, "_session_recorder", None) is not None:
            try:
                self.stop_session_recording()
            except Exception:
                log.exception("Finalizing the active session during shutdown failed")

        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None

    def shutdown(self) -> None:
        """Persist dock layout, stop all workers, close all devices."""
        log.info("Shutdown: persisting state")
        self.clear_session_replay(resume_background=False)
        if self._energy_calibration_dialog is not None:
            self._energy_calibration_dialog.close_without_prompt()
            self._energy_calibration_dialog = None
        if self._mca_peak_analysis_dialog is not None:
            self._mca_peak_analysis_dialog.close_without_prompt()
            self._mca_peak_analysis_dialog = None
        if self._psd_readback_dialog is not None:
            self._psd_readback_dialog.close_without_prompt()
            self._psd_readback_dialog = None
        if self._waveform_analysis_dialog is not None:
            self._waveform_analysis_dialog.close_without_prompt()
            self._waveform_analysis_dialog = None
        self._save_dock_state()
        for ctrl in self._scope_controllers:
            ctrl.save_display_settings()

        self._stop_all_workers()

        log.info("Shutdown: closing hardware connections")
        for device in self._devices:
            device.close()
        self._devices.clear()
        self._external_devices.close()
        log.info("Shutdown complete")

    def reconnect(self) -> None:
        """Tear down all devices/controllers and reconnect from scratch.

        Keeps the window and dock layout positions; rebuilds the channel
        docks and their controllers since they hold direct references to
        the (now-closed) backend connections.
        """
        log.info("Reconnect: stopping workers and closing current devices")
        self.clear_session_replay(resume_background=False)
        if self._energy_calibration_dialog is not None:
            self._energy_calibration_dialog.close_without_prompt()
            self._energy_calibration_dialog = None
        if self._mca_peak_analysis_dialog is not None:
            self._mca_peak_analysis_dialog.close_without_prompt()
            self._mca_peak_analysis_dialog = None
        self._save_dock_state()
        for ctrl in self._scope_controllers:
            ctrl.save_display_settings()

        self._stop_all_workers()

        self._global_controller = None

        for device in self._devices:
            device.close()
        self._devices.clear()
        self._external_devices.close()
        self._external_devices = ExternalDevices()

        dock_hosts = (
            self._scope_dock_host,
            self._current_dock_host,
            self._mca_dock_host,
            self._psd_dock_host,
            self._psu_dock_host,
            self._global_dock_host,
            self._external_dock_host,
        )
        for host in dock_hosts:
            for dock in host.findChildren(QDockWidget):
                host.removeDockWidget(dock)
                widget = dock.widget()
                if widget is not None:
                    widget.deleteLater()
                dock.deleteLater()

        self._scope_controllers.clear()
        self._current_monitor_controllers.clear()
        self._mca_controllers.clear()
        self._psd_controllers.clear()
        self._psd_controller_by_device.clear()
        self._psu_controllers.clear()
        self._psu_controller_by_device.clear()
        self._external_controllers.clear()
        self._mca_event_buffers.clear()
        self._scope_frame_buffers.clear()

        log.info(
            "Reconnect: connecting via IIO to %s:%d, %d channel(s)",
            self._host,
            self._port,
            self._channels,
        )
        for channel in range(self._channels):
            self._devices.append(self._connect_channel(channel))
        log.info("Reconnect: all %d device(s) connected", len(self._devices))

        self._build_global_tab()
        self._build_channel_docks()
        self._build_external_docks()
        self._restore_dock_state()
        log.info(
            "Reconnect complete, %d scope / %d current / %d MCA / %d PSD / %d PSU / "
            "%d external controllers",
            len(self._scope_controllers),
            len(self._current_monitor_controllers),
            len(self._mca_controllers),
            len(self._psd_controllers),
            len(self._psu_controllers),
            len(self._external_controllers),
        )

    def save_all_settings(self, path: Path) -> None:
        """Save settings for all channels to a single YAML file."""
        from nlab.utils.settings_io import write_configuration

        write_configuration(path, self.configuration_document())
        log.info("All channel and application settings saved to %s", path)

    @staticmethod
    def _encoded_dock_state(host: QMainWindow) -> str:
        encoded = cast(bytes, host.saveState().toBase64().data())
        return encoded.decode("ascii")

    def configuration_document(self) -> dict[str, object]:
        """Snapshot the same complete document used by Save Settings."""
        from nlab.utils.settings_io import FORMAT_VERSION, collect_channel_hardware

        hardware_channels: dict[str, object] = {}
        application_channels: dict[str, object] = {}
        for idx, device in enumerate(self._devices):
            channel = self._display_channel(idx)
            mca_ctrl = self._mca_controller_by_device.get(idx)
            current_ctrl = self._current_monitor_controller_by_device.get(idx)
            psu_ctrl = self._psu_controller_by_device.get(idx)
            lp_preset = None
            if mca_ctrl is not None:
                lp_preset = mca_ctrl.hardware_configuration_settings()["low_pass_preset"]
            hardware_channels[str(channel)] = collect_channel_hardware(
                device.scope,
                device.mca if device.mca_available() else None,
                device.hv,
                mca_lp_preset=lp_preset,
                psu_settings=(
                    psu_ctrl.hardware_configuration_settings() if psu_ctrl is not None else None
                ),
            )
            application_channel: dict[str, object] = {
                "scope": self._scope_controllers[idx].configuration_settings(),
            }
            if current_ctrl is not None:
                application_channel["current_monitor"] = current_ctrl.configuration_settings()
            if psu_ctrl is not None:
                application_channel["psu"] = psu_ctrl.configuration_settings()
            if mca_ctrl is not None:
                application_channel["mca"] = mca_ctrl.configuration_settings()
            psd_ctrl = self._psd_controller_by_device.get(idx)
            if psd_ctrl is not None:
                application_channel["psd"] = psd_ctrl.configuration_settings()
            application_channels[str(channel)] = application_channel

        external_hardware = {
            ctrl.configuration_id: ctrl.hardware_configuration_settings()
            for ctrl in self._external_controllers
        }
        external_application = {
            ctrl.configuration_id: ctrl.configuration_settings()
            for ctrl in self._external_controllers
        }
        shared_hardware = (
            self._global_controller.hardware_configuration_settings()
            if self._global_controller is not None
            else {}
        )
        global_application = (
            self._global_controller.configuration_settings()
            if self._global_controller is not None
            else {}
        )
        coincidence = getattr(self, "_coincidence_controller", None)
        layout = {
            "scope": self._encoded_dock_state(self._scope_dock_host),
            "current": self._encoded_dock_state(self._current_dock_host),
            "mca": self._encoded_dock_state(self._mca_dock_host),
            "psd": self._encoded_dock_state(self._psd_dock_host),
            "psu": self._encoded_dock_state(self._psu_dock_host),
            "global": self._encoded_dock_state(self._global_dock_host),
            "external": self._encoded_dock_state(self._external_dock_host),
        }
        document = {
            "format_version": FORMAT_VERSION,
            "connection": {
                "ip": self._host,
                "port": self._port,
                "channels": self._channels,
            },
            "hardware": {
                "channels": hardware_channels,
                "shared_trigger": shared_hardware,
                "external_devices": external_hardware,
            },
            "application": {
                "main_window": self._window.configuration_settings(),
                "global": global_application,
                "coincidence": (
                    coincidence.configuration_settings() if coincidence is not None else {}
                ),
                "channels": application_channels,
                "external_devices": external_application,
                "dock_layout": layout,
            },
        }
        return document

    def load_all_settings(self, path: Path) -> None:
        """Load settings from YAML and apply to hardware, then refresh UI."""
        if self.session_replay is not None:
            raise RuntimeError("exit session replay before loading hardware settings")
        if self._coincidence_controller is not None and self._coincidence_controller.active:
            raise RuntimeError("Stop the coincidence measurement before loading settings")
        from nlab.utils.settings_io import (
            apply_channel_hardware,
            channel_application_entry,
            channel_entry,
            read_configuration,
            validate_configuration_version,
        )

        document = read_configuration(path)
        validate_configuration_version(document)
        if "hardware" not in document:
            # Original files contained one channel directly at the root.
            apply_channel_hardware(
                self._devices[0].scope,
                self._devices[0].mca if self._devices[0].mca_available() else None,
                self._devices[0].hv,
                document,
            )
            log.warning("Loaded legacy settings into the first connected channel")
        else:
            for idx, device in enumerate(self._devices):
                channel = self._display_channel(idx)
                hardware = channel_entry(document, channel)
                if hardware is None:
                    log.warning("No settings found for channel %s", channel)
                    continue
                apply_channel_hardware(
                    device.scope,
                    device.mca if device.mca_available() else None,
                    device.hv,
                    hardware,
                )

        for idx, device in enumerate(self._devices):
            self._scope_controllers[idx]._load_hardware_state()
            mca_ctrl = self._mca_controller_by_device.get(idx)
            if mca_ctrl is not None:
                mca_ctrl._load_hardware_state()
            channel = self._display_channel(idx)
            hardware = channel_entry(document, channel)
            if hardware is not None:
                mca_settings = hardware.get("mca", {})
                if mca_ctrl is not None and isinstance(mca_settings, dict):
                    low_pass = mca_settings.get("low_pass", {})
                    if isinstance(low_pass, dict) and "preset" in low_pass:
                        mca_ctrl.populate_hardware_configuration_settings(
                            {"low_pass_preset": low_pass["preset"]}
                        )
                psu_ctrl = self._psu_controller_by_device.get(idx)
                if psu_ctrl is not None:
                    psu_settings = hardware.get("psu", {})
                    psu_ctrl.populate_hardware_configuration_settings(psu_settings)

            app_channel = channel_application_entry(document, channel)
            if app_channel is not None:
                self._scope_controllers[idx].apply_configuration_settings(app_channel.get("scope"))
                current_ctrl = self._current_monitor_controller_by_device.get(idx)
                if current_ctrl is not None:
                    current_ctrl.apply_configuration_settings(
                        app_channel.get("current_monitor")
                    )
                psu_ctrl = self._psu_controller_by_device.get(idx)
                if psu_ctrl is not None:
                    psu_ctrl.apply_configuration_settings(app_channel.get("psu"))
                if mca_ctrl is not None:
                    mca_ctrl.apply_configuration_settings(app_channel.get("mca"))
                psd_ctrl = self._psd_controller_by_device.get(idx)
                if psd_ctrl is not None:
                    psd_ctrl.apply_configuration_settings(app_channel.get("psd"))

        if self._coincidence_controller is not None:
            application = document.get("application", {})
            if isinstance(application, dict):
                self._coincidence_controller.apply_configuration_settings(
                    application.get("coincidence")
                )

        hardware_root = document.get("hardware", {})
        application = document.get("application", {})
        if isinstance(hardware_root, dict) and self._global_controller is not None:
            self._global_controller.apply_hardware_configuration_settings(
                hardware_root.get("shared_trigger")
            )
        if self._global_controller is not None:
            if isinstance(application, dict):
                self._global_controller.apply_configuration_settings(application.get("global"))
            self._global_controller.refresh_temperature_state()
        if isinstance(hardware_root, dict):
            external = hardware_root.get("external_devices", {})
            if isinstance(external, dict):
                for ctrl in self._external_controllers:
                    ctrl.apply_hardware_configuration_settings(external.get(ctrl.configuration_id))
        if isinstance(application, dict):
            self._window.apply_configuration_settings(application.get("main_window"))
            external = application.get("external_devices", {})
            if isinstance(external, dict):
                for ctrl in self._external_controllers:
                    ctrl.apply_configuration_settings(external.get(ctrl.configuration_id))
            self._restore_yaml_dock_layout(application.get("dock_layout"))
        log.info("All hardware and application settings loaded from %s", path)

    def _restore_yaml_dock_layout(self, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        hosts = {
            "scope": self._scope_dock_host,
            "current": self._current_dock_host,
            "mca": self._mca_dock_host,
            "psd": self._psd_dock_host,
            "psu": self._psu_dock_host,
            "global": self._global_dock_host,
            "external": self._external_dock_host,
        }
        for name, host in hosts.items():
            encoded = settings.get(name)
            if isinstance(encoded, str):
                host.restoreState(QByteArray.fromBase64(encoded.encode("ascii")))

    def _connect_signals(self) -> None:
        self._window.ui.mainTabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int) -> None:
        tab_name = self._window.ui.mainTabs.tabText(index)
        log.info("Active tab: %s", tab_name)
