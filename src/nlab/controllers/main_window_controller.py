from __future__ import annotations

import logging
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from ipaddress import IPv6Address, ip_address
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QByteArray, QSettings, Qt, QThread, QThreadPool
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from nlab import settings_keys
from nlab.analysis.energy_calibration import CalibrationPointTransfer
from nlab.analysis.spectrum import Spectrum
from nlab.controllers.coincidence_controller import CoincidenceController
from nlab.controllers.current_monitor_controller import CurrentMonitorController
from nlab.controllers.external_device_controller import ExternalDeviceController
from nlab.controllers.global_controller import GlobalController
from nlab.controllers.interlock import ReplayInterlock
from nlab.controllers.mca_controller import MCAController
from nlab.controllers.psd_controller import PSDController
from nlab.controllers.psu_controller import PSUController
from nlab.controllers.scope_controller import DisplayMode, ScopeController
from nlab.controllers.session_replay_coordinator import SessionReplayCoordinator
from nlab.hardware.digitizer.current_monitor import ScopeCurrentAccumulator
from nlab.hardware.digitizer.device_information import DigitizerInformation, InformationSection
from nlab.hardware.digitizer.digitizer import Digitizer
from nlab.hardware.digitizer.dma import McaEventBuffer, ScopeFrameBuffer
from nlab.hardware.digitizer.mca import McaGateTiming
from nlab.hardware.digitizer.offline import (
    create_offline_digitizer,
    create_offline_external_device,
)
from nlab.hardware.modbus_devices import ExternalDevices, ExternalDeviceScan
from nlab.session.recorder import SessionRecorder
from nlab.session.replay import (
    SessionReplay,
    TelemetryReplayEvent,
)
from nlab.views.current_capture_dialog import CurrentCaptureDialog
from nlab.views.dose_calibration_dialog import (
    DoseCalibrationChannelSource,
    DoseCalibrationDialog,
)
from nlab.views.energy_calibration_dialog import EnergyCalibrationDialog
from nlab.views.mca_peak_analysis_dialog import McaPeakAnalysisDialog
from nlab.views.panel_dock import PanelDockWidget
from nlab.views.panel_status import PanelStatus, PanelStatusPresenter
from nlab.views.psd_readback_dialog import PsdReadbackDialog
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog
from nlab.views.waveform_mca_transfer_dialog import McaGateTransferTarget
from nlab.views.window_flags import TOOL_WINDOW_FLAGS

if TYPE_CHECKING:
    from nlab_modbus.core.base_modbus_device import BaseModbusDevice

    from nlab.app import MainAppWindow

log = logging.getLogger(__name__)



@dataclass(frozen=True)
class DigitizerEndpoint:
    """One physical digitizer and the channels opened from its IIO endpoint."""

    host: str
    port: int
    channels: int


@dataclass(frozen=True)
class DigitizerChannel:
    """Physical address for one entry in the flat controller/device lists."""

    digitizer: int
    channel: int


@dataclass(frozen=True)
class _ConnectionTopology:
    """Validated device ownership and addressing for one complete workspace."""

    offline: bool
    endpoints: tuple[DigitizerEndpoint, ...]
    devices: tuple[Digitizer, ...]
    device_channels: tuple[DigitizerChannel, ...]
    offline_port: int

    def __post_init__(self) -> None:
        if len(self.devices) != len(self.device_channels):
            raise ValueError("every digitizer channel requires one physical address")
        if self.offline:
            if self.endpoints:
                raise ValueError("offline topology cannot contain hardware endpoints")
            return
        if not self.endpoints:
            raise ValueError("live topology requires at least one hardware endpoint")
        expected_channels = tuple(
            DigitizerChannel(digitizer, channel)
            for digitizer, endpoint in enumerate(self.endpoints)
            for channel in range(endpoint.channels)
        )
        if self.device_channels != expected_channels:
            raise ValueError("digitizer devices and endpoint channels are inconsistent")

    @property
    def host(self) -> str:
        return "" if self.offline else self.endpoints[0].host

    @property
    def port(self) -> int:
        return self.offline_port if self.offline else self.endpoints[0].port


@dataclass(frozen=True)
class _ConnectionRuntime:
    """Rollback snapshot for a topology and its per-board UI state."""

    topology: _ConnectionTopology
    coincidence_settings: dict[int, dict[str, object]]
    selected_coincidence_digitizer: int
    replay_hardware_interlocked: bool


@dataclass(frozen=True)
class CoincidenceDigitizerOption:
    """One same-board pair eligible for the coincidence workspace."""

    digitizer: int
    device_indices: tuple[int, int]


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
        offline: bool = False,
    ) -> None:
        self._window = window
        self._host = host
        self._port = port
        self._channels = channels
        self._offline = bool(offline)
        self._devices: list[Digitizer] = []
        self._endpoints: list[DigitizerEndpoint] = (
            [] if self._offline else [DigitizerEndpoint(host, port, channels)]
        )
        self._device_channels = [DigitizerChannel(0, channel) for channel in range(channels)]
        self._on_progress = on_progress

        if self._offline:
            log.info("Preparing offline analysis workspace with %d channel(s)", channels)
            for channel in range(self._channels):
                self._report_progress(
                    f"Preparing offline channel {channel + 1} of {self._channels}..."
                )
                self._devices.append(create_offline_digitizer(channel))
        else:
            log.info("Connecting via IIO to %s:%d, %d channel(s)", host, port, channels)
            for channel in range(self._channels):
                self._report_progress(f"Connecting channel {channel + 1} of {self._channels}...")
                self._devices.append(self._connect_channel(channel))
            log.info("All %d device(s) connected", len(self._devices))

        self._selected_channel: int | None = None
        self._active_channel_dock_host: QMainWindow | None = None
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
        self._global_controllers: list[GlobalController] = []
        self._global_controller_by_digitizer: dict[int, GlobalController] = {}
        self._coincidence_controller: CoincidenceController | None = None
        self._coincidence_channels: tuple[int, int] | None = None
        self._coincidence_options: list[CoincidenceDigitizerOption] = []
        self._coincidence_selector: QComboBox | None = None
        self._coincidence_selector_row: QWidget | None = None
        self._selected_coincidence_digitizer = 0
        self._coincidence_settings_by_digitizer: dict[int, dict[str, object]] = {}
        self._energy_calibration_dialog: EnergyCalibrationDialog | None = None
        self._dose_calibration_dialog: DoseCalibrationDialog | None = None
        self._mca_peak_analysis_dialog: McaPeakAnalysisDialog | None = None
        self._psd_readback_dialog: PsdReadbackDialog | None = None
        self._waveform_analysis_dialog: WaveformAnalysisDialog | None = None
        self._current_capture_dialog: CurrentCaptureDialog | None = None
        self._external_devices = ExternalDevices()
        self._thread: QThread | None = None
        self._session_recorder: SessionRecorder | None = None
        self._session_replayer = SessionReplayCoordinator(self)
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
        if self._offline:
            self._set_replay_hardware_interlock(True, resume_background=False)
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
        return self._connect_endpoint_channel(self._host, self._port, channel)

    @staticmethod
    def _connect_endpoint_channel(host: str, port: int, channel: int) -> Digitizer:
        """Open one channel at an explicit endpoint without mutating controller state."""
        return Digitizer.from_iio(
            channel=channel,
            uri=f"ip:{host}:{port}",
            with_ids=True,
        )

    def _display_channel(self, idx: int) -> int:
        """Return the stable flat channel ID used in settings and session files."""
        return idx

    def _physical_channel(self, idx: int) -> DigitizerChannel:
        """Return a channel's board-local address, including for legacy test fixtures."""
        addresses = cast(
            Sequence[DigitizerChannel],
            getattr(self, "_device_channels", ()),
        )
        if idx < len(addresses):
            return addresses[idx]
        return DigitizerChannel(0, idx)

    def _channel_label(self, idx: int) -> str:
        address = self._physical_channel(idx)
        return f"D{address.digitizer + 1}:CH{address.channel}"

    @staticmethod
    def _set_controller_channel_label(controller: object, label: str) -> None:
        """Attach a display label while tolerating immutable lightweight test doubles."""
        try:
            setattr(controller, "display_channel_label", label)
        except AttributeError:
            pass

    def _devices_for_digitizer(self) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = {}
        for index in range(len(self._devices)):
            grouped.setdefault(self._physical_channel(index).digitizer, []).append(index)
        return grouped

    @property
    def digitizer_count(self) -> int:
        return len(getattr(self, "_endpoints", ()))

    @property
    def offline(self) -> bool:
        return bool(self._offline)

    @property
    def endpoint_descriptions(self) -> tuple[str, ...]:
        return tuple(f"{endpoint.host}:{endpoint.port}" for endpoint in self._endpoints)

    @property
    def primary_endpoint(self) -> tuple[str, int] | None:
        endpoints = getattr(self, "_endpoints", ())
        if not endpoints:
            return None
        return endpoints[0].host, endpoints[0].port

    @property
    def channel_count(self) -> int:
        return len(self._devices)

    def digitizer_information(self) -> tuple[tuple[str, DigitizerInformation], ...]:
        """Supply About with cached scalar metadata, never live IIO handles."""
        if self._offline:
            return ()
        result = []
        for index, device in enumerate(self._devices):
            information = device.get_device_information()
            if information is None:
                address = self._physical_channel(index)
                endpoint = self._endpoints[address.digitizer]
                information = DigitizerInformation(
                    f"ip:{endpoint.host}:{endpoint.port}", address.channel,
                    (InformationSection("Connection", (
                        ("Device metadata", "Unknown / unavailable"),
                    )),),
                    (),
                )
            result.append((self._channel_label(index), information))
        return tuple(result)

    def energy_calibrated_channel_labels(self) -> tuple[str, ...]:
        """Return display labels for MCAs with an applied energy calibration."""
        return tuple(
            controller.channel_label
            for controller in self._mca_controllers
            if controller.energy_calibration is not None
        )

    def measuring_channel_labels(self) -> tuple[str, ...]:
        """Summarize controller run states without IIO reads or extra timers.

        Scope and MCA on one channel count once. Keep board-qualified names
        when multiple digitizers are connected so CH0 cannot be ambiguous.
        Replay/analysis does not start these live lifecycle flags.
        """
        if getattr(self, "_offline", False):
            return ()
        channels = sorted({
            controller.channel
            for controller in (
                *getattr(self, "_scope_controllers", []),
                *getattr(self, "_mca_controllers", []),
            )
            if controller.measurement_active
        })
        labels = tuple(self._channel_label(channel) for channel in channels)
        if len(getattr(self, "_endpoints", ())) <= 1:
            return tuple(label.split(":", 1)[-1] for label in labels)
        return labels

    def _measurement_state_changed(self, *_args: object) -> None:
        refresh = getattr(self._window, "refresh_measurement_indicator", None)
        if callable(refresh):
            refresh()

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

    def _report_connection_lost(self, source: str, message: str) -> None:
        """Let the main window offer a reconnect when a panel loses its board."""
        notify = getattr(self._window, "notify_connection_lost", None)
        if callable(notify):
            notify(source, message)

    def _make_dock(self, obj_name: str, title: str, widget: QWidget) -> QDockWidget:
        dock = PanelDockWidget(title)
        dock.setObjectName(obj_name)
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.topLevelChanged.connect(partial(self._on_dock_floating_changed, dock))
        panel_name = obj_name.split("_", 1)[0]
        panel_name = panel_name.upper() if panel_name in {"mca", "psd"} else panel_name.title()
        self._bind_panel_status(widget, f"{panel_name} {title}", dock)
        if self._dock_channel(dock) is not None:
            dock.visibilityChanged.connect(partial(self._on_channel_dock_activated, dock))
        return dock

    @staticmethod
    def _on_dock_floating_changed(dock: QDockWidget, floating: bool) -> None:
        if not floating:
            return
        # Qt restores tool-window flags on each undock. Use a normal window
        # so the native title bar supports independent minimize/maximize.
        # Mask integer values: enum inversion can discard high hint bits.
        flags = (
            Qt.WindowType(int(dock.windowFlags()) & ~int(Qt.WindowType.WindowType_Mask))
            | TOOL_WINDOW_FLAGS
        )
        if flags == dock.windowFlags():
            return
        visible = dock.isVisible()
        geometry = dock.geometry()
        dock.setWindowFlags(flags)
        dock.setGeometry(geometry)
        # Changing window flags hides a widget; retain its previous visibility.
        if visible:
            dock.show()

    def _bind_panel_status(
        self,
        widget: QWidget,
        source: str,
        dock: QDockWidget | None = None,
    ) -> None:
        status = getattr(widget, "status", None)
        if not isinstance(status, PanelStatus):
            return
        presenter = PanelStatusPresenter(widget, status, self._window.statusBar(), source)
        if dock is not None:
            presenter.set_active(dock.isVisible())
            dock.visibilityChanged.connect(presenter.set_active)

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
            ch_label = self._channel_label(idx)
            self._report_progress(f"Initializing {ch_label} controls...")

            scope_current_accumulator = ScopeCurrentAccumulator()
            scope_frame_buffer = ScopeFrameBuffer()
            scope_ctrl = ScopeController(
                device.scope,
                scope_dma=device.scope_dma,
                channel=ch,
                dma_frame_buffer=scope_frame_buffer,
                current_accumulator=scope_current_accumulator,
                measurement_configuration=self.configuration_document,
            )
            self._set_controller_channel_label(scope_ctrl, ch_label)
            self._connect_optional_signal(
                scope_ctrl,
                "session_viewer_frame",
                self._record_scope_viewer_frame,
            )
            self._scope_frame_buffers[ch] = scope_frame_buffer
            self._scope_controllers.append(scope_ctrl)
            self._connect_optional_signal(
                scope_ctrl, "measurement_active_changed", self._measurement_state_changed,
            )
            self._connect_optional_signal(
                scope_ctrl,
                "connection_lost",
                partial(self._report_connection_lost, f"Scope {ch_label}"),
            )
            scope_docks.append(self._make_dock(f"scope_ch{ch}", ch_label, scope_ctrl))

            if device.capabilities.mca:
                current_ctrl = CurrentMonitorController(
                    device.mca,
                    channel=ch,
                    auto_start=False,
                    scope_controller=scope_ctrl,
                    scope_current_accumulator=scope_current_accumulator,
                    capture_viewer=self.show_scope_sums,
                )
                self._set_controller_channel_label(current_ctrl, ch_label)
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
                    self._set_controller_channel_label(psd_ctrl, ch_label)
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
                self._set_controller_channel_label(mca_ctrl, ch_label)
                self._mca_controllers.append(mca_ctrl)
                self._connect_optional_signal(
                    mca_ctrl, "measurement_active_changed", self._measurement_state_changed,
                )
                self._connect_optional_signal(
                    mca_ctrl,
                    "connection_lost",
                    partial(self._report_connection_lost, f"MCA {ch_label}"),
                )
                calibration_indicator_slot = getattr(
                    self._window,
                    "refresh_energy_calibration_indicator",
                    None,
                )
                if callable(calibration_indicator_slot):
                    self._connect_optional_signal(
                        mca_ctrl,
                        "energy_calibration_changed",
                        calibration_indicator_slot,
                    )
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
                self._set_controller_channel_label(psu_ctrl, ch_label)
                self._connect_optional_signal(
                    psu_ctrl,
                    "session_readback",
                    partial(self._record_psu_readback, ch),
                )
                self._connect_optional_signal(
                    psu_ctrl,
                    "connection_lost",
                    partial(self._report_connection_lost, f"PSU {ch_label}"),
                )
                self._psu_controllers.append(psu_ctrl)
                self._psu_controller_by_device[idx] = psu_ctrl
                psu_docks.append(self._make_dock(f"psu_ch{ch}", ch_label, psu_ctrl))

        # A Current capture may ask the MCA, PSU and Global pollers to pause
        # so they do not compete with the capture on the board's iiod.
        for current_ctrl in self._current_monitor_controllers:
            self._connect_optional_signal(
                current_ctrl,
                "readback_pause_changed",
                partial(
                    self._set_board_readbacks_paused,
                    f"Current {getattr(current_ctrl, 'channel', id(current_ctrl))}",
                ),
            )
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
            channels = ", ".join(
                self._channel_label(channel) for channel in missing_mca_channels
            )
            current_tooltip = f"Current monitor unavailable for channel(s): {channels}."
        else:
            current_tooltip = ""
        self._window.ui.tabCurrent.setToolTip(current_tooltip)

        mca_tab_index = self._window.ui.mainTabs.indexOf(self._window.ui.tabMCA)
        self._window.ui.mainTabs.setTabEnabled(mca_tab_index, bool(mca_docks))
        if not mca_docks:
            mca_tooltip = "MCA is disabled: no pulse-processor/input-filter device was found."
        elif missing_mca_channels:
            channels = ", ".join(
                self._channel_label(channel) for channel in missing_mca_channels
            )
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
            channels = ", ".join(
                self._channel_label(channel) for channel in missing_psu_channels
            )
            psu_tooltip = f"PSU controls unavailable for channel(s): {channels}."
        else:
            psu_tooltip = ""
        self._window.ui.tabPSU.setToolTip(psu_tooltip)

    def _build_global_tab(self) -> None:
        """Build one independent global-control panel per physical digitizer."""
        self._global_controllers = []
        self._global_controller_by_digitizer = {}
        docks: list[QDockWidget] = []
        for digitizer, indices in self._devices_for_digitizer().items():
            devices = [self._devices[index] for index in indices]
            labels = [self._channel_label(index) for index in indices]
            controller = GlobalController(devices, labels)
            self._global_controllers.append(controller)
            self._connect_optional_signal(
                controller,
                "connection_lost",
                partial(self._report_connection_lost, f"Global D{digitizer + 1}"),
            )
            self._global_controller_by_digitizer[digitizer] = controller
            if digitizer == 0:
                self._connect_optional_signal(
                    controller,
                    "session_temperature_readback",
                    self._record_temperature_readback,
                )
                self._connect_optional_signal(
                    controller,
                    "session_diagnostics_readback",
                    self._record_global_diagnostics,
                )
            else:
                self._connect_optional_signal(
                    controller,
                    "session_temperature_readback",
                    partial(self._record_digitizer_temperature_readback, digitizer),
                )
                self._connect_optional_signal(
                    controller,
                    "session_diagnostics_readback",
                    partial(self._record_digitizer_diagnostics, digitizer),
                )
            board_label = f"D{digitizer + 1}"
            docks.append(
                self._make_dock(
                    "global_panel" if digitizer == 0 else f"global_panel_d{digitizer + 1}",
                    board_label,
                    controller,
                )
            )
        self._global_controller = self._global_controllers[0] if self._global_controllers else None
        self._populate_dock_host(self._global_dock_host, docks)
        layout = self._window.ui.layoutTabGlobal
        if layout.indexOf(self._global_dock_host) < 0:
            layout.addWidget(self._global_dock_host)

    def _build_coincidence_tab(self) -> None:
        self._coincidence_options = []
        for digitizer, indices in self._devices_for_digitizer().items():
            eligible = [
                index
                for index in indices
                if self._mca_controller_by_device.get(index) is not None
                and self._devices[index].capabilities.mca_dma
            ]
            global_controller = self._global_controller_by_digitizer.get(digitizer)
            if (
                len(eligible) >= 2
                and global_controller is not None
                and global_controller.sync_available
            ):
                self._coincidence_options.append(
                    CoincidenceDigitizerOption(
                        digitizer=digitizer,
                        device_indices=(eligible[0], eligible[1]),
                    )
                )

        available = bool(self._coincidence_options)
        index = self._window.ui.mainTabs.indexOf(self._window.ui.tabCoincidence)
        self._window.ui.mainTabs.setTabEnabled(index, available)
        if available:
            boards = ", ".join(
                f"D{option.digitizer + 1}" for option in self._coincidence_options
            )
            tooltip = (
                f"Choose a same-board coincidence pair from {boards}. Cross-digitizer "
                "coincidence is disabled because board synchronization has not been validated."
            )
        else:
            tooltip = (
                "Coincidence requires two IIO MCA list-mode channels on the same digitizer "
                "and shared software start. Cross-digitizer coincidence is disabled."
            )
        self._window.ui.tabCoincidence.setToolTip(tooltip)
        if not available:
            self._coincidence_channels = None
            return

        row = QWidget()
        row.setObjectName("coincidenceDigitizerRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(QLabel("Digitizer:", row))
        selector = QComboBox(row)
        selector.setObjectName("coincidenceDigitizerSelector")
        for option in self._coincidence_options:
            pair = option.device_indices
            selector.addItem(
                f"D{option.digitizer + 1} — "
                f"{self._channel_label(pair[0])} + {self._channel_label(pair[1])}",
                option.digitizer,
            )
        row_layout.addWidget(selector)
        row_layout.addStretch(1)
        self._coincidence_selector = selector
        self._coincidence_selector_row = row

        selected_index = selector.findData(
            getattr(self, "_selected_coincidence_digitizer", 0)
        )
        if selected_index < 0:
            selected_index = 0
        selector.setCurrentIndex(selected_index)
        selector.currentIndexChanged.connect(self._on_coincidence_digitizer_changed)
        selected = selector.currentData()
        assert isinstance(selected, int)
        self._install_coincidence_digitizer(selected)

    def _install_coincidence_digitizer(self, digitizer: int) -> None:
        option = next(
            (
                candidate
                for candidate in self._coincidence_options
                if candidate.digitizer == digitizer
            ),
            None,
        )
        if option is None:
            return
        current = self._coincidence_controller
        selected_digitizer = getattr(self, "_selected_coincidence_digitizer", 0)
        if current is not None and digitizer == selected_digitizer:
            self._sync_coincidence_selector_enabled()
            return
        if current is not None:
            if current.active:
                return
            settings_by_digitizer = getattr(
                self, "_coincidence_settings_by_digitizer", None
            )
            if settings_by_digitizer is None:
                settings_by_digitizer = {}
                self._coincidence_settings_by_digitizer = settings_by_digitizer
            settings_by_digitizer[
                selected_digitizer
            ] = current.configuration_settings()
            detach_selector = getattr(
                current, "set_digitizer_selector_widget", None
            )
            if callable(detach_selector):
                detach_selector(None)
            self._window.ui.layoutTabCoincidence.removeWidget(current)
            current.setParent(None)
            current.deleteLater()

        pair = option.device_indices
        mca_a, mca_b = (self._mca_controller_by_device[index] for index in pair)
        global_controller = self._global_controller_by_digitizer[digitizer]
        controller = CoincidenceController(
            [self._devices[index] for index in pair],
            [mca_a, mca_b],
            global_controller,
            channel_labels=[self._channel_label(index) for index in pair],
        )
        self._bind_panel_status(
            controller, f"Coincidence {' + '.join(self._channel_label(index) for index in pair)}"
        )
        saved = getattr(self, "_coincidence_settings_by_digitizer", {}).get(
            digitizer
        )
        if saved is not None:
            controller.apply_configuration_settings(saved)
        if getattr(self, "_replay_hardware_interlocked", False):
            controller.set_replay_hardware_interlock(True)
        self._coincidence_controller = controller
        self._coincidence_channels = pair
        self._selected_coincidence_digitizer = digitizer
        self._connect_optional_signal(
            controller,
            "session_state",
            self._on_coincidence_session_state,
        )
        attach_selector = getattr(
            controller, "set_digitizer_selector_widget", None
        )
        if callable(attach_selector):
            attach_selector(self._coincidence_selector_row)
        self._window.ui.layoutTabCoincidence.addWidget(controller)
        self._sync_coincidence_selector_enabled()

    def _on_coincidence_digitizer_changed(self, index: int) -> None:
        selector = getattr(self, "_coincidence_selector", None)
        if selector is None or index < 0:
            return
        current = self._coincidence_controller
        if current is not None and current.active:
            previous = selector.findData(self._selected_coincidence_digitizer)
            selector.blockSignals(True)
            selector.setCurrentIndex(previous)
            selector.blockSignals(False)
            return
        digitizer = selector.itemData(index)
        if isinstance(digitizer, int):
            self._install_coincidence_digitizer(digitizer)

    def _on_coincidence_session_state(self, state: str, values: object) -> None:
        self._record_coincidence_state(state, values)
        self._sync_coincidence_selector_enabled()

    def _sync_coincidence_selector_enabled(self) -> None:
        selector = getattr(self, "_coincidence_selector", None)
        if selector is None:
            return
        controller = self._coincidence_controller
        selector.setEnabled(
            len(self._coincidence_options) > 1
            and not getattr(self, "_replay_hardware_interlocked", False)
            and (controller is None or not getattr(controller, "active", False))
        )

    @staticmethod
    def _saved_coincidence_digitizer(settings: object) -> int | None:
        if not isinstance(settings, dict):
            return None
        value = settings.get("digitizer")
        if (
            isinstance(value, str)
            and value.startswith("D")
            and value[1:].isdigit()
        ):
            return int(value[1:]) - 1
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

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
        offline = getattr(self, "_offline", False)
        if devices is None and offline:
            devices = ()
        elif devices is None:
            devices = self._external_devices.discover(self._host)
        docks: list[QDockWidget] = []
        for idx, device in enumerate(devices):
            ctrl = (
                ExternalDeviceController(device, auto_start=False)
                if offline
                else ExternalDeviceController(device)
            )
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
        if docks:
            tooltip = ""
        elif offline:
            tooltip = "Recorded external modules appear when a replay bundle contains them."
        else:
            tooltip = "No external Modbus modules were detected on the digitizer host."
        self._window.ui.tabExternal.setToolTip(tooltip)

    def scan_external_devices(self) -> ExternalDeviceScan:
        """Perform a full Modbus scan using a fresh connection owner.

        This method touches no Qt widgets and is safe to call on a background
        thread. The result must later be adopted with
        :meth:`apply_external_device_scan` or explicitly closed.
        """
        if getattr(self, "_offline", False):
            raise RuntimeError("external-device discovery is unavailable in offline mode")
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

    # Bump a state-key suffix when its format or object-name scheme changes.
    # Runtime dock-count changes are handled by the per-host topology signatures.
    _DOCK_STATE_KEY_SCOPE = settings_keys.DOCK_STATE_SCOPE
    _DOCK_STATE_KEY_CURRENT = settings_keys.DOCK_STATE_CURRENT
    _DOCK_STATE_KEY_MCA = settings_keys.DOCK_STATE_MCA
    _DOCK_STATE_KEY_PSD = settings_keys.DOCK_STATE_PSD
    _DOCK_STATE_KEY_PSU = settings_keys.DOCK_STATE_PSU
    _DOCK_STATE_KEY_GLOBAL = settings_keys.DOCK_STATE_GLOBAL
    _DOCK_STATE_KEY_EXTERNAL = settings_keys.DOCK_STATE_EXTERNAL
    _DOCK_TOPOLOGY_KEY_PREFIX = settings_keys.DOCK_TOPOLOGY_PREFIX

    def _dock_state_hosts(self) -> tuple[tuple[str, str, QMainWindow], ...]:
        """Return persistent layout names, keys, and their current dock hosts."""
        return (
            ("scope", self._DOCK_STATE_KEY_SCOPE, self._scope_dock_host),
            ("current", self._DOCK_STATE_KEY_CURRENT, self._current_dock_host),
            ("mca", self._DOCK_STATE_KEY_MCA, self._mca_dock_host),
            ("psd", self._DOCK_STATE_KEY_PSD, self._psd_dock_host),
            ("psu", self._DOCK_STATE_KEY_PSU, self._psu_dock_host),
            ("global", self._DOCK_STATE_KEY_GLOBAL, self._global_dock_host),
            ("external", self._DOCK_STATE_KEY_EXTERNAL, self._external_dock_host),
        )

    @staticmethod
    def _dock_topology(host: QMainWindow) -> list[str]:
        """Identify the exact set of docks a saved host state belongs to."""
        return sorted(dock.objectName() for dock in host.findChildren(QDockWidget))

    @classmethod
    def _dock_topology_key(cls, name: str) -> str:
        return f"{cls._DOCK_TOPOLOGY_KEY_PREFIX}/{name}"

    @classmethod
    def _reset_and_store_dock_host(
        cls,
        settings: QSettings,
        name: str,
        state_key: str,
        host: QMainWindow,
    ) -> None:
        """Replace an incompatible saved layout with the default tab group."""
        cls._reset_dock_host(host)
        settings.setValue(state_key, host.saveState())
        settings.setValue(cls._dock_topology_key(name), cls._dock_topology(host))

    def _save_dock_state(self) -> None:
        settings = QSettings()
        for name, state_key, host in self._dock_state_hosts():
            settings.setValue(state_key, host.saveState())
            settings.setValue(self._dock_topology_key(name), self._dock_topology(host))

    def _restore_dock_state(self) -> None:
        settings = QSettings()
        for name, state_key, host in self._dock_state_hosts():
            state = settings.value(state_key)
            if state is None:
                continue

            saved_topology = settings.value(self._dock_topology_key(name))
            current_topology = self._dock_topology(host)
            if saved_topology != current_topology:
                log.info(
                    "Ignoring saved %s dock layout because its topology changed "
                    "from %r to %r",
                    name,
                    saved_topology,
                    current_topology,
                )
                self._reset_and_store_dock_host(
                    settings,
                    name,
                    state_key,
                    host,
                )
                continue

            if not host.restoreState(state):
                log.warning("Ignoring invalid saved %s dock layout", name)
                self._reset_and_store_dock_host(
                    settings,
                    name,
                    state_key,
                    host,
                )

    @staticmethod
    def _dock_default_order(dock: QDockWidget) -> tuple[int, str]:
        """Sort generated dock names by their numeric channel/device suffix."""
        name = dock.objectName()
        suffix_start = len(name)
        while suffix_start > 0 and name[suffix_start - 1].isdigit():
            suffix_start -= 1
        suffix = name[suffix_start:]
        return (int(suffix) if suffix else -1, name)

    @classmethod
    def _reset_dock_host(cls, host: QMainWindow) -> None:
        """Detach and rebuild one tab group in deterministic default order."""
        docks = sorted(host.findChildren(QDockWidget), key=cls._dock_default_order)
        if not docks:
            return

        # addDockWidget() alone does not dissolve an existing tab group. Fully
        # detach first so a user-arranged order cannot leak into the reset.
        for dock in docks:
            dock.setFloating(False)
            host.removeDockWidget(dock)
        for dock in docks:
            host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
            dock.show()
        for previous, current in zip(docks, docks[1:], strict=False):
            host.tabifyDockWidget(previous, current)
        docks[0].raise_()

    def reset_dock_layout(self) -> None:
        """Clear saved dock state and restore deterministic physical ordering."""
        settings = QSettings()
        dock_hosts = self._dock_state_hosts()
        for name, state_key, _host in dock_hosts:
            settings.remove(state_key)
            settings.remove(self._dock_topology_key(name))

        for _name, _state_key, host in dock_hosts:
            self._reset_dock_host(host)

        log.info("Dock layout reset to default (tabbed)")

    def set_scope_distribution_enabled(self, enabled: bool) -> None:
        """Show and accumulate amplitude distributions on every Scope channel."""
        for controller in self._scope_controllers:
            controller.set_scope_distribution_enabled(enabled)

    def clear_scope_distributions(self) -> None:
        """Clear all per-channel received-waveform amplitude histograms."""
        for controller in self._scope_controllers:
            controller.clear_scope_distribution()

    def set_measurement_series_enabled(self, enabled: bool) -> None:
        """Apply Settings > Enable Series to every Scope, MCA and Current view."""
        for scope_controller in self._scope_controllers:
            scope_controller.set_measurement_series_enabled(enabled)
        for mca_controller in self._mca_controllers:
            mca_controller.set_measurement_series_enabled(enabled)
        for current_controller in self._current_monitor_controllers:
            current_controller.set_measurement_series_enabled(enabled)
        log.info("Measurement series %s", "enabled" if enabled else "disabled")

    def set_roi_visible(self, visible: bool) -> None:
        """Toggle the ROI selection tool + stats panel on every MCA histogram."""
        for ctrl in self._mca_controllers:
            ctrl.set_roi_visible(visible)
        log.info("ROI %s on all MCA histograms", "shown" if visible else "hidden")

    def set_differential_histogram(self, enabled: bool) -> None:
        """Toggle per-refresh differences on every MCA histogram."""
        for ctrl in self._mca_controllers:
            ctrl.set_differential_histogram(enabled)
        log.info(
            "MCA histograms set to %s display",
            "differential" if enabled else "accumulated",
        )

    def set_log_y(self, enabled: bool) -> None:
        """Toggle logarithmic Y-axis on every MCA histogram."""
        for ctrl in self._mca_controllers:
            ctrl.set_log_y(enabled)
        log.info("Histogram Y-axis set to %s", "log" if enabled else "linear")

    def toggle_scope_distribution_log_under_mouse(self) -> bool:
        """Toggle count scaling on the Scope distribution under the pointer."""
        return any(
            controller.toggle_scope_distribution_log_under_mouse()
            for controller in self._scope_controllers
        )

    def mca_histogram_under_mouse(self) -> bool:
        """Return whether the pointer is over any live MCA histogram."""
        return any(ctrl.histogram_under_mouse() for ctrl in self._mca_controllers)

    def _set_board_readbacks_paused(self, reason: str, paused: bool) -> None:
        for mca_controller in self._mca_controllers:
            mca_controller.set_readbacks_paused(reason, paused)
        for psu_controller in self._psu_controllers:
            psu_controller.set_readbacks_paused(reason, paused)
        for global_controller in self._all_global_controllers():
            global_controller.set_readbacks_paused(reason, paused)

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
            "scope_display_mode": int(scope.get("display_mode", DisplayMode.RAW)),
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
            "display_mode": settings.get("scope_display_mode", int(DisplayMode.RAW)),
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
            self._energy_calibration_dialog.compton_fit_requested.connect(
                self._open_compton_edge_fit
            )
        self._energy_calibration_dialog.show_workspace()

    def _open_compton_edge_fit(self, spectra: object) -> None:
        """Open calibration snapshots in the workbench's Compton-edge tab."""
        if not isinstance(spectra, list):
            return
        self.show_mca_peak_analysis()
        dialog = self._mca_peak_analysis_dialog
        if dialog is not None:
            dialog.add_compton_datasets([item for item in spectra if isinstance(item, Spectrum)])

    def _receive_compton_calibration_points(self, transfer: object) -> None:
        """Hand fitted Compton edges (keVee) to the energy-calibration workspace."""
        if not isinstance(transfer, CalibrationPointTransfer):
            return
        self.show_energy_calibration()
        dialog = self._energy_calibration_dialog
        if dialog is not None:
            dialog.add_calibration_points(transfer)

    def _dose_calibration_sources(self) -> list[DoseCalibrationChannelSource]:
        sources: list[DoseCalibrationChannelSource] = []
        for device_index, mca in sorted(self._mca_controller_by_device.items()):
            address = self._physical_channel(device_index)
            sources.append(
                DoseCalibrationChannelSource(
                    label=mca.channel_label,
                    mca=mca,
                    psu=self._psu_controller_by_device.get(device_index),
                    psd=self._psd_controller_by_device.get(device_index),
                    temperatures=self._global_controller_by_digitizer.get(
                        address.digitizer
                    ),
                )
            )
        return sources

    def show_dose_calibration(self) -> None:
        """Open a modeless logger for reference dose-rate calibration rows."""
        if self._dose_calibration_dialog is None:
            sources = self._dose_calibration_sources()
            if not sources:
                QMessageBox.information(
                    self._window,
                    "Dose Calibration",
                    "No MCA channel is available for dose calibration logging.",
                )
                return
            self._dose_calibration_dialog = DoseCalibrationDialog(
                sources,
                parent=self._window,
            )
        self._dose_calibration_dialog.show_workspace()

    def _ensure_mca_peak_analysis_dialog(self) -> McaPeakAnalysisDialog:
        """Return the one workbench, creating it with its cross-window wiring.

        Every creator goes through here (menu, session replay); a workbench created
        elsewhere would silently drop Compton-edge calibration transfers.
        """
        if self._mca_peak_analysis_dialog is None:
            dialog = McaPeakAnalysisDialog(self._mca_controllers, parent=self._window)
            dialog.compton_calibration_requested.connect(
                self._receive_compton_calibration_points
            )
            self._mca_peak_analysis_dialog = dialog
        return self._mca_peak_analysis_dialog

    def show_mca_peak_analysis(self) -> None:
        """Open one modeless peak-analysis workbench for live and saved spectra."""
        self._ensure_mca_peak_analysis_dialog().show_workspace()

    # ------------------------------------------------------------------
    # Versioned session recording and analysis replay
    # ------------------------------------------------------------------

    @property
    def session_recording(self) -> bool:
        recorder = getattr(self, "_session_recorder", None)
        return recorder is not None and recorder.is_recording

    def start_session_recording(self, path: Path) -> Path:
        """Passively record all data already observed by this application."""
        if getattr(self, "_offline", False):
            raise RuntimeError("session recording requires a live hardware connection")
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

    # ------------------------------------------------------------------
    # Session replay (implemented by SessionReplayCoordinator)
    # ------------------------------------------------------------------

    def replay_session(self, path: Path) -> SessionReplay:
        """Replay captured analysis inputs and monitored telemetry once."""
        return self._session_replayer.replay_session(path)

    @property
    def session_replay(self) -> SessionReplay | None:
        replayer = getattr(self, "_session_replayer", None)
        return replayer.session_replay if replayer is not None else None

    @property
    def session_replay_path(self) -> Path | None:
        replayer = getattr(self, "_session_replayer", None)
        return replayer.session_replay_path if replayer is not None else None

    def clear_session_replay(self, *, resume_background: bool = True) -> None:
        """Leave replay mode without resuming any stopped live acquisition."""
        replayer = getattr(self, "_session_replayer", None)
        if replayer is not None:
            replayer.clear_session_replay(resume_background=resume_background)

    def _set_replay_hardware_interlock(
        self,
        enabled: bool,
        *,
        resume_background: bool = True,
    ) -> None:
        """Apply one app-wide hardware-write boundary for session replay."""
        enabled = bool(enabled) or getattr(self, "_offline", False)
        if enabled == getattr(self, "_replay_hardware_interlocked", False):
            return
        targets: list[ReplayInterlock] = []
        coincidence = getattr(self, "_coincidence_controller", None)
        if coincidence is not None:
            targets.append(coincidence)
        targets.extend(getattr(self, "_current_monitor_controllers", []))
        targets.extend(getattr(self, "_scope_controllers", []))
        targets.extend(getattr(self, "_mca_controllers", []))
        targets.extend(self._all_global_controllers())
        targets.extend(getattr(self, "_psu_controllers", []))
        targets.extend(getattr(self, "_external_controllers", []))

        applied: list[ReplayInterlock] = []
        try:
            for target in targets:
                target.set_replay_hardware_interlock(
                    enabled, resume_background=resume_background,
                )
                applied.append(target)
        except Exception:
            for target in reversed(applied):
                try:
                    target.set_replay_hardware_interlock(not enabled, resume_background=False)
                except Exception:
                    log.exception("Failed to roll back replay hardware interlock")
            raise
        self._replay_hardware_interlocked = enabled
        self._sync_coincidence_selector_enabled()

    def _install_offline_external_replay_controllers(
        self,
        replay: SessionReplay,
        events: list[TelemetryReplayEvent] | None = None,
    ) -> None:
        """Build read-only external-module panels from a replay's recorded schema."""
        latest_values: dict[str, dict[str, object]] = {}
        if events is None:
            latest_reader = getattr(replay, "latest_telemetry_events", None)
            latest_events = (
                latest_reader().values() if latest_reader is not None else []
            )
        else:
            latest_events = events
        for event in latest_events:
            if not event.source.startswith("external."):
                continue
            configuration_id = event.source.removeprefix("external.")
            values = event.payload.get("values")
            if not isinstance(values, dict):
                continue
            destination = latest_values.setdefault(configuration_id, {})
            destination.update(
                {
                    str(name): value
                    for name, value in values.items()
                    if isinstance(name, str)
                }
            )
        configuration = replay.configuration_snapshot()
        hardware = configuration.get("hardware")
        hardware = hardware if isinstance(hardware, dict) else {}
        external_hardware = hardware.get("external_devices")
        external_hardware = external_hardware if isinstance(external_hardware, dict) else {}
        application = configuration.get("application")
        application = application if isinstance(application, dict) else {}
        external_application = application.get("external_devices")
        external_application = (
            external_application if isinstance(external_application, dict) else {}
        )
        external_schemas = application.get("external_device_schemas")
        external_schemas = (
            external_schemas if isinstance(external_schemas, dict) else {}
        )

        devices = []
        configuration_ids = (
            set(latest_values) | set(external_hardware) | set(external_schemas)
        )
        for configuration_id in sorted(configuration_ids):
            if not isinstance(configuration_id, str):
                continue
            values = latest_values.get(configuration_id, {})
            recorded_holding = external_hardware.get(configuration_id)
            recorded_holding = (
                recorded_holding if isinstance(recorded_holding, dict) else {}
            )
            holding_values: dict[str, int | float] = {}
            for name, value in recorded_holding.items():
                if not isinstance(name, str) or not isinstance(value, int | float):
                    continue
                holding_values[name] = int(value) if isinstance(value, bool) else value
            schema = external_schemas.get(configuration_id)
            if schema is None:
                devices.append(
                    create_offline_external_device(
                        configuration_id,
                        holding_values,
                        values,
                    )
                )
            else:
                devices.append(
                    create_offline_external_device(
                        configuration_id,
                        holding_values,
                        values,
                        schema,
                    )
                )
        self._build_external_docks(devices)
        for controller in self._external_controllers:
            controller.apply_configuration_settings(
                external_application.get(controller.configuration_id)
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

    def _record_digitizer_temperature_readback(
        self,
        digitizer: int,
        readback: object,
    ) -> None:
        recorder = self._session_recorder
        if recorder is not None:
            recorder.record_telemetry(
                f"digitizer.d{digitizer + 1}.temperature_correction",
                {"readback": readback},
            )

    def _record_digitizer_diagnostics(self, digitizer: int, readings: object) -> None:
        recorder = self._session_recorder
        if recorder is not None:
            recorder.record_telemetry(
                f"digitizer.d{digitizer + 1}.diagnostics",
                {"readings": readings},
            )

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

    def show_scope_sums(self, path: Path | None = None) -> None:
        """Open the hardware-independent Current capture analysis tool."""
        if self._current_capture_dialog is None:
            self._current_capture_dialog = CurrentCaptureDialog(parent=self._window)
        self._current_capture_dialog.show_workspace()
        if path is not None:
            self._current_capture_dialog.open_path(path)

    def show_waveform_analysis(self, path: Path | None = None) -> None:
        """Open the waveform browser and offline PSD reconstruction workbench."""
        if self._waveform_analysis_dialog is None:
            self._waveform_analysis_dialog = WaveformAnalysisDialog(
                parent=self._window, gate_targets=self._waveform_mca_targets,
            )
        self._waveform_analysis_dialog.show_workspace()
        if path is not None:
            self._waveform_analysis_dialog.open_path(path)

    def _check_waveform_mca_target(self, target: MCAController) -> None:
        if (self._offline or self._replay_hardware_interlocked
                or target not in self._mca_controllers):
            raise RuntimeError("MCA target is offline, replay-interlocked, or no longer connected")
        if any(scope.channel == target.channel and scope.measurement_active
               for scope in self._scope_controllers):
            raise RuntimeError("Stop Scope acquisition on the target channel before transfer")

    def _waveform_mca_targets(self) -> list[McaGateTransferTarget]:
        if self._offline or self._replay_hardware_interlocked:
            return []

        def read(target: MCAController) -> McaGateTiming:
            self._check_waveform_mca_target(target)
            return target.gate_transfer_current()

        def apply(target: MCAController, timing: McaGateTiming) -> None:
            self._check_waveform_mca_target(target)
            target.apply_gate_timing(timing)

        return [McaGateTransferTarget(target.channel_label, partial(read, target),
                                      partial(apply, target))
                for target in self._mca_controllers]

    def load_psd_events(self, path: Path) -> None:
        """Compatibility wrapper opening the standalone PSD readback tool."""
        self.show_psd_event_readback(path)

    def load_waveform_file(self, path: Path) -> None:
        """Compatibility wrapper opening the standalone waveform workbench."""
        self.show_waveform_analysis(path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _all_global_controllers(self) -> list[GlobalController]:
        controllers = getattr(self, "_global_controllers", None)
        if controllers is not None:
            return list(controllers)
        controller = getattr(self, "_global_controller", None)
        return [controller] if controller is not None else []

    def _stop_all_workers(self) -> None:
        """Stop all running timers/workers (blocking). Devices stay open."""
        coincidence = getattr(self, "_coincidence_controller", None)
        if coincidence is not None:
            coincidence.request_shutdown()
        # Broadcast the cheap stop requests before waiting for any individual
        # worker. Independent IIO/Modbus calls then finish concurrently rather
        # than making shutdown pay every transport timeout in series.
        for global_controller in self._all_global_controllers():
            global_controller.request_polling_stop()
        for psu_controller in self._psu_controllers:
            psu_controller.request_monitor_stop()
        for external_controller in self._external_controllers:
            external_controller.request_polling_stop()
        for current_controller in self._current_monitor_controllers:
            current_controller.request_monitor_stop()

        for global_controller in self._all_global_controllers():
            global_controller.disarm_sync()
            global_controller.stop_polling_sync()

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
        if self._dose_calibration_dialog is not None:
            self._dose_calibration_dialog.close_without_prompt()
            self._dose_calibration_dialog = None
        if self._mca_peak_analysis_dialog is not None:
            self._mca_peak_analysis_dialog.close_without_prompt()
            self._mca_peak_analysis_dialog = None
        if self._psd_readback_dialog is not None:
            self._psd_readback_dialog.close_without_prompt()
            self._psd_readback_dialog = None
        if self._current_capture_dialog is not None:
            self._current_capture_dialog.close_without_prompt()
            self._current_capture_dialog = None
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

    @staticmethod
    def _validated_endpoint(host: str, port: int, channels: int) -> DigitizerEndpoint:
        endpoint = host.strip()
        if not endpoint:
            raise ValueError("digitizer address cannot be empty")
        if not 1 <= port <= 65_535:
            raise ValueError("digitizer port must be within 1-65535")
        if channels < 1:
            raise ValueError("at least one digitizer channel is required")
        return DigitizerEndpoint(endpoint, port, channels)

    @staticmethod
    def _normalized_host(host: str) -> str:
        """Normalize spelling differences that do not identify another host."""
        return host.strip().removeprefix("[").removesuffix("]").rstrip(".").casefold()

    @staticmethod
    def _normalized_ip(value: str) -> str:
        """Canonicalize an address, including IPv4-mapped IPv6 results."""
        address = ip_address(value.split("%", 1)[0])
        if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        return address.compressed

    @classmethod
    def _resolved_host_addresses(cls, host: str) -> frozenset[str]:
        """Resolve a host for duplicate-board detection, failing closed."""
        normalized_host = cls._normalized_host(host)
        try:
            return frozenset((cls._normalized_ip(normalized_host),))
        except ValueError:
            pass

        try:
            records = socket.getaddrinfo(
                normalized_host,
                None,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise ValueError(
                f"cannot resolve digitizer address {host!r}; the duplicate-device "
                "safety check could not be completed"
            ) from exc

        addresses = frozenset(cls._normalized_ip(str(record[4][0])) for record in records)
        if not addresses:
            raise ValueError(
                f"digitizer address {host!r} resolved to no network addresses; "
                "the duplicate-device safety check could not be completed"
            )
        return addresses

    @classmethod
    def _duplicate_endpoint_index(
        cls,
        candidate: DigitizerEndpoint,
        connected: Sequence[DigitizerEndpoint],
    ) -> int | None:
        """Find an existing endpoint reached by the candidate host alias."""
        same_port = [
            (index, endpoint)
            for index, endpoint in enumerate(connected)
            if endpoint.port == candidate.port
        ]
        candidate_host = cls._normalized_host(candidate.host)
        for index, endpoint in same_port:
            if cls._normalized_host(endpoint.host) == candidate_host:
                return index

        if not same_port:
            return None
        candidate_addresses = cls._resolved_host_addresses(candidate.host)
        for index, endpoint in same_port:
            if candidate_addresses & cls._resolved_host_addresses(endpoint.host):
                return index
        return None

    @staticmethod
    def _close_connection_devices(
        devices: Sequence[Digitizer],
        *,
        context: str,
    ) -> None:
        """Best-effort close without replacing the transition's primary error."""
        for device in devices:
            try:
                device.close()
            except Exception:
                log.warning("%s: closing one channel transport failed", context, exc_info=True)

    def _open_endpoint_devices(
        self,
        endpoint: DigitizerEndpoint,
        digitizer: int,
        *,
        board_label: str | None,
        on_progress: Callable[[str], None] | None,
    ) -> tuple[list[Digitizer], list[DigitizerChannel]]:
        devices: list[Digitizer] = []
        try:
            for channel in range(endpoint.channels):
                if on_progress is not None:
                    message = (
                        f"Connecting channel {channel + 1} of {endpoint.channels}..."
                        if board_label is None
                        else f"Connecting {board_label}:CH{channel} "
                        f"({channel + 1} of {endpoint.channels})..."
                    )
                    on_progress(message)
                devices.append(
                    self._connect_endpoint_channel(endpoint.host, endpoint.port, channel)
                )
        except Exception:
            self._close_connection_devices(
                devices,
                context=f"Opening {endpoint.host}:{endpoint.port}",
            )
            raise
        addresses = [
            DigitizerChannel(digitizer, channel) for channel in range(endpoint.channels)
        ]
        return devices, addresses

    def _open_configured_endpoints(
        self,
        endpoints: Sequence[DigitizerEndpoint],
        *,
        on_progress: Callable[[str], None] | None,
    ) -> tuple[list[Digitizer], list[DigitizerChannel]]:
        devices: list[Digitizer] = []
        addresses: list[DigitizerChannel] = []
        try:
            for digitizer, endpoint in enumerate(endpoints):
                opened, opened_addresses = self._open_endpoint_devices(
                    endpoint,
                    digitizer,
                    board_label=f"D{digitizer + 1}",
                    on_progress=on_progress,
                )
                devices.extend(opened)
                addresses.extend(opened_addresses)
        except Exception:
            self._close_connection_devices(devices, context="Preparing reconnect")
            raise
        return devices, addresses

    def _capture_connection_runtime(self) -> _ConnectionRuntime:
        """Capture the topology and latest coincidence settings before teardown."""
        devices = tuple(self._devices)
        endpoints = tuple(
            getattr(
                self,
                "_endpoints",
                ()
                if self._offline
                else (DigitizerEndpoint(self._host, self._port, len(devices)),),
            )
        )
        addresses = tuple(
            getattr(
                self,
                "_device_channels",
                (DigitizerChannel(0, channel) for channel in range(len(devices))),
            )
        )
        settings = {
            index: dict(value)
            for index, value in getattr(
                self,
                "_coincidence_settings_by_digitizer",
                {},
            ).items()
        }
        selected = getattr(self, "_selected_coincidence_digitizer", 0)
        coincidence = getattr(self, "_coincidence_controller", None)
        if coincidence is not None:
            settings[selected] = coincidence.configuration_settings()
        return _ConnectionRuntime(
            topology=_ConnectionTopology(
                offline=self._offline,
                endpoints=endpoints,
                devices=devices,
                device_channels=addresses,
                offline_port=self._port,
            ),
            coincidence_settings=settings,
            selected_coincidence_digitizer=selected,
            replay_hardware_interlocked=bool(
                getattr(self, "_replay_hardware_interlocked", False)
            ),
        )

    def _install_connection_runtime(self, runtime: _ConnectionRuntime) -> None:
        topology = runtime.topology
        self._devices = list(topology.devices)
        self._endpoints = list(topology.endpoints)
        self._device_channels = list(topology.device_channels)
        self._host = topology.host
        self._port = topology.port
        self._channels = len(topology.devices)
        self._offline = topology.offline
        self._coincidence_settings_by_digitizer = {
            index: dict(value)
            for index, value in runtime.coincidence_settings.items()
        }
        self._selected_coincidence_digitizer = runtime.selected_coincidence_digitizer
        # Reapply the interlock after endpoint-bound widgets have been rebuilt.
        self._replay_hardware_interlocked = False

    def _rebuild_connection_runtime(
        self,
        runtime: _ConnectionRuntime,
        *,
        on_progress: Callable[[str], None] | None,
        normalize_docks: bool,
    ) -> None:
        self._install_connection_runtime(runtime)
        self._rebuild_connection_views(
            on_progress=on_progress,
            normalize_docks=normalize_docks,
        )
        if runtime.topology.offline or runtime.replay_hardware_interlocked:
            self._set_replay_hardware_interlock(True, resume_background=False)

    def _transition_connection_runtime(
        self,
        target: _ConnectionRuntime,
        *,
        previous: _ConnectionRuntime,
        close_on_success: Sequence[Digitizer],
        close_on_failure: Sequence[Digitizer],
        on_progress: Callable[[str], None] | None,
        normalize_docks: bool,
        description: str,
    ) -> None:
        """Atomically rebuild a prepared topology or restore the prior one."""
        try:
            self._teardown_connection_runtime()
            self._rebuild_connection_runtime(
                target,
                on_progress=on_progress,
                normalize_docks=normalize_docks,
            )
        except Exception:
            log.exception("%s failed; restoring the previous workspace", description)
            try:
                self._teardown_connection_runtime()
            except Exception:
                log.exception("Failed to tear down the partial connection workspace")
            self._close_connection_devices(
                close_on_failure,
                context=f"Rolling back {description.lower()}",
            )
            try:
                self._rebuild_connection_runtime(
                    previous,
                    on_progress=on_progress,
                    normalize_docks=normalize_docks,
                )
            except Exception:
                log.exception("Failed to restore the previous connection workspace")
            raise
        self._close_connection_devices(
            close_on_success,
            context=f"Committing {description.lower()}",
        )

    def connect_to_device(
        self,
        host: str,
        port: int,
        channels: int,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Promote an offline workspace to a live IIO hardware session."""
        if not getattr(self, "_offline", False):
            raise RuntimeError("Connect is available only from offline analysis mode")
        endpoint = self._validated_endpoint(host, port, channels)
        previous = self._capture_connection_runtime()
        log.info(
            "Connect: opening IIO endpoint %s:%d with %d channel(s)",
            endpoint.host,
            endpoint.port,
            endpoint.channels,
        )
        connected, addresses = self._open_endpoint_devices(
            endpoint,
            0,
            board_label=None,
            on_progress=on_progress,
        )
        target = _ConnectionRuntime(
            topology=_ConnectionTopology(
                offline=False,
                endpoints=(endpoint,),
                devices=tuple(connected),
                device_channels=tuple(addresses),
                offline_port=endpoint.port,
            ),
            coincidence_settings=previous.coincidence_settings,
            selected_coincidence_digitizer=0,
            replay_hardware_interlocked=False,
        )
        self._transition_connection_runtime(
            target,
            previous=previous,
            close_on_success=previous.topology.devices,
            close_on_failure=connected,
            on_progress=on_progress,
            normalize_docks=False,
            description="Live connection setup",
        )
        log.info("Connect complete: offline workspace promoted to live hardware")

    def add_digitizer(
        self,
        host: str,
        port: int,
        channels: int,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> int:
        """Add another physical board through the atomic topology transition."""
        if getattr(self, "_offline", False):
            raise RuntimeError("connect the first digitizer before adding another")
        endpoint = self._validated_endpoint(host, port, channels)
        previous = self._capture_connection_runtime()
        endpoints = previous.topology.endpoints
        duplicate_index = self._duplicate_endpoint_index(endpoint, endpoints)
        if duplicate_index is not None:
            existing_endpoint = endpoints[duplicate_index]
            raise ValueError(
                f"digitizer {endpoint.host}:{endpoint.port} is already connected as "
                f"D{duplicate_index + 1} "
                f"({existing_endpoint.host}:{existing_endpoint.port})"
            )

        digitizer = len(endpoints)
        board_label = f"D{digitizer + 1}"
        log.info(
            "Connect: opening %s at %s:%d with %d channel(s)",
            board_label,
            endpoint.host,
            endpoint.port,
            endpoint.channels,
        )
        connected, addresses = self._open_endpoint_devices(
            endpoint,
            digitizer,
            board_label=board_label,
            on_progress=on_progress,
        )
        target = _ConnectionRuntime(
            topology=_ConnectionTopology(
                offline=False,
                endpoints=(*endpoints, endpoint),
                devices=(*previous.topology.devices, *connected),
                device_channels=(*previous.topology.device_channels, *addresses),
                offline_port=previous.topology.offline_port,
            ),
            coincidence_settings=previous.coincidence_settings,
            selected_coincidence_digitizer=previous.selected_coincidence_digitizer,
            replay_hardware_interlocked=False,
        )
        self._transition_connection_runtime(
            target,
            previous=previous,
            close_on_success=(),
            close_on_failure=connected,
            on_progress=on_progress,
            normalize_docks=True,
            description=f"Adding {board_label}",
        )
        log.info(
            "Connect complete: %s added at %s:%d",
            board_label,
            endpoint.host,
            endpoint.port,
        )
        return digitizer + 1

    def disconnect_digitizer(
        self,
        digitizer: int,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> bool:
        """Remove one board atomically, returning whether the workspace is offline."""
        if getattr(self, "_offline", False):
            raise RuntimeError("offline mode has no digitizer to disconnect")
        previous = self._capture_connection_runtime()
        endpoints = previous.topology.endpoints
        if not 0 <= digitizer < len(endpoints):
            raise IndexError("digitizer index is out of range")

        board_label = f"D{digitizer + 1}"
        endpoint = endpoints[digitizer]
        paired = tuple(
            zip(
                previous.topology.devices,
                previous.topology.device_channels,
                strict=True,
            )
        )
        removed_devices = tuple(
            device for device, address in paired if address.digitizer == digitizer
        )
        remaining_endpoints = tuple(
            item for index, item in enumerate(endpoints) if index != digitizer
        )
        digitizer_map = {
            old: new
            for new, old in enumerate(
                index for index in range(len(endpoints)) if index != digitizer
            )
        }

        if remaining_endpoints:
            remaining_devices = tuple(
                device for device, address in paired if address.digitizer != digitizer
            )
            remaining_addresses = tuple(
                DigitizerChannel(digitizer_map[address.digitizer], address.channel)
                for _device, address in paired
                if address.digitizer != digitizer
            )
            target_topology = _ConnectionTopology(
                offline=False,
                endpoints=remaining_endpoints,
                devices=remaining_devices,
                device_channels=remaining_addresses,
                offline_port=previous.topology.offline_port,
            )
            target_settings = {
                digitizer_map[index]: settings
                for index, settings in previous.coincidence_settings.items()
                if index in digitizer_map
            }
            selected = digitizer_map.get(
                previous.selected_coincidence_digitizer,
                0,
            )
            failure_devices: Sequence[Digitizer] = ()
            rebuild_message = "Rebuilding remaining digitizer views..."
        else:
            offline_devices: list[Digitizer] = []
            try:
                for channel in range(max(1, endpoint.channels)):
                    offline_devices.append(create_offline_digitizer(channel))
            except Exception:
                self._close_connection_devices(
                    offline_devices,
                    context="Preparing offline workspace",
                )
                raise
            target_topology = _ConnectionTopology(
                offline=True,
                endpoints=(),
                devices=tuple(offline_devices),
                device_channels=tuple(
                    DigitizerChannel(0, channel)
                    for channel in range(len(offline_devices))
                ),
                offline_port=previous.topology.port,
            )
            target_settings = {}
            selected = 0
            failure_devices = offline_devices
            rebuild_message = "Preparing offline analysis workspace..."

        target = _ConnectionRuntime(
            topology=target_topology,
            coincidence_settings=target_settings,
            selected_coincidence_digitizer=selected,
            replay_hardware_interlocked=target_topology.offline,
        )
        if on_progress is not None:
            on_progress(f"Stopping measurements for {board_label}...")
            on_progress(rebuild_message)
        self._transition_connection_runtime(
            target,
            previous=previous,
            close_on_success=removed_devices,
            close_on_failure=failure_devices,
            on_progress=on_progress,
            normalize_docks=True,
            description=f"Disconnecting {board_label}",
        )
        log.info(
            "Disconnect complete: %s at %s:%d removed; %d endpoint(s) remain",
            board_label,
            endpoint.host,
            endpoint.port,
            len(remaining_endpoints),
        )
        return target_topology.offline

    def reconnect(
        self,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Open replacement transports before atomically rebuilding all views."""
        if getattr(self, "_offline", False):
            raise RuntimeError("offline mode has no hardware endpoint; use Connect")
        previous = self._capture_connection_runtime()
        endpoints = previous.topology.endpoints
        log.info("Reconnect: preparing %d replacement endpoint(s)", len(endpoints))
        connected, addresses = self._open_configured_endpoints(
            endpoints,
            on_progress=on_progress,
        )
        target = _ConnectionRuntime(
            topology=_ConnectionTopology(
                offline=False,
                endpoints=endpoints,
                devices=tuple(connected),
                device_channels=tuple(addresses),
                offline_port=previous.topology.offline_port,
            ),
            coincidence_settings=previous.coincidence_settings,
            selected_coincidence_digitizer=previous.selected_coincidence_digitizer,
            replay_hardware_interlocked=False,
        )
        self._transition_connection_runtime(
            target,
            previous=previous,
            close_on_success=previous.topology.devices,
            close_on_failure=connected,
            on_progress=on_progress,
            normalize_docks=False,
            description="Reconnect",
        )
        log.info("Reconnect: all %d device(s) connected", len(connected))

    def _close_connection_analysis_dialogs(self) -> None:
        """Close analysis dialogs whose models refer to the current controllers."""
        if self._energy_calibration_dialog is not None:
            self._energy_calibration_dialog.close_without_prompt()
            self._energy_calibration_dialog = None
        if self._dose_calibration_dialog is not None:
            self._dose_calibration_dialog.close_without_prompt()
            self._dose_calibration_dialog = None
        if self._mca_peak_analysis_dialog is not None:
            self._mca_peak_analysis_dialog.close_without_prompt()
            self._mca_peak_analysis_dialog = None

    def _remove_connection_docks(self) -> None:
        """Detach and schedule deletion of every endpoint-bound dock."""
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
                dock.setParent(None)
                widget = dock.widget()
                if widget is not None:
                    widget.deleteLater()
                dock.deleteLater()

    def _remove_coincidence_workspace(self) -> None:
        """Preserve coincidence settings, then detach its endpoint-bound widgets."""
        coincidence = self._coincidence_controller
        if coincidence is not None:
            self._coincidence_settings_by_digitizer[
                self._selected_coincidence_digitizer
            ] = coincidence.configuration_settings()
            coincidence.set_digitizer_selector_widget(None)
            self._window.ui.layoutTabCoincidence.removeWidget(coincidence)
            coincidence.setParent(None)
            coincidence.deleteLater()
            self._coincidence_controller = None
        selector_row = self._coincidence_selector_row
        if selector_row is not None:
            selector_row.setParent(None)
            selector_row.deleteLater()
        self._coincidence_selector = None
        self._coincidence_selector_row = None
        self._coincidence_options = []

    def _clear_connection_controller_references(self) -> None:
        """Release controller and replay-buffer references after widget teardown."""
        self._global_controller = None
        self._global_controllers = []
        self._global_controller_by_digitizer = {}
        self._coincidence_channels = None
        self._scope_controllers.clear()
        self._current_monitor_controllers.clear()
        self._current_monitor_controller_by_device.clear()
        self._mca_controllers.clear()
        self._mca_controller_by_device.clear()
        self._psd_controllers.clear()
        self._psd_controller_by_device.clear()
        self._psu_controllers.clear()
        self._psu_controller_by_device.clear()
        self._external_controllers.clear()
        self._mca_event_buffers.clear()
        self._scope_frame_buffers.clear()
        self._measurement_state_changed()

    def _teardown_connection_runtime(self) -> None:
        """Stop and detach endpoint-bound views while leaving transports owned."""
        self.clear_session_replay(resume_background=False)
        self._close_connection_analysis_dialogs()
        self._save_dock_state()
        for ctrl in self._scope_controllers:
            ctrl.save_display_settings()

        self._stop_all_workers()
        self._external_devices.close()
        self._external_devices = ExternalDevices()
        self._remove_connection_docks()
        self._remove_coincidence_workspace()
        self._clear_connection_controller_references()

    def _rebuild_connection_views(
        self,
        *,
        on_progress: Callable[[str], None] | None = None,
        normalize_docks: bool = False,
    ) -> None:
        """Recreate every controller that holds endpoint-specific objects."""
        if on_progress is not None:
            on_progress("Preparing global controls...")
        self._build_global_tab()
        if on_progress is not None:
            on_progress("Preparing channel views...")
        self._build_channel_docks()
        if on_progress is not None:
            on_progress("Preparing coincidence view...")
        self._build_coincidence_tab()
        if on_progress is not None:
            on_progress("Discovering external devices...")
        self._build_external_docks()
        if on_progress is not None:
            on_progress("Restoring dock layout...")
        self._restore_dock_state()
        if normalize_docks:
            # The state saved before adding a board knows only the old dock
            # names. Qt can partially restore it and strand one of the new
            # channel docks in a second row. A topology change therefore gets
            # a fresh, deterministic one-row layout which is saved immediately.
            for host in (
                self._scope_dock_host,
                self._current_dock_host,
                self._mca_dock_host,
                self._psd_dock_host,
                self._psu_dock_host,
                self._global_dock_host,
            ):
                self._reset_dock_host(host)
            self._save_dock_state()
        log.info(
            "Connection rebuilt, %d scope / %d current / %d MCA / %d PSD / %d PSU / "
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
        external_schemas = {
            ctrl.configuration_id: ctrl.replay_schema()
            for ctrl in self._external_controllers
        }
        global_controllers = self._all_global_controllers()
        shared_hardware = (
            global_controllers[0].hardware_configuration_settings()
            if global_controllers
            else {}
        )
        global_application = (
            global_controllers[0].configuration_settings() if global_controllers else {}
        )
        digitizer_hardware = {
            f"D{index + 1}": controller.hardware_configuration_settings()
            for index, controller in enumerate(global_controllers)
        }
        digitizer_application = {
            f"D{index + 1}": controller.configuration_settings()
            for index, controller in enumerate(global_controllers)
        }
        coincidence = getattr(self, "_coincidence_controller", None)
        coincidence_application = (
            coincidence.configuration_settings() if coincidence is not None else {}
        )
        if coincidence is not None:
            coincidence_application = dict(coincidence_application)
            coincidence_application["digitizer"] = (
                f"D{self._selected_coincidence_digitizer + 1}"
            )
            coincidence_application["physical_channels"] = [
                self._channel_label(channel)
                for channel in (self._coincidence_channels or ())
            ]
        layout = {
            "scope": self._encoded_dock_state(self._scope_dock_host),
            "current": self._encoded_dock_state(self._current_dock_host),
            "mca": self._encoded_dock_state(self._mca_dock_host),
            "psd": self._encoded_dock_state(self._psd_dock_host),
            "psu": self._encoded_dock_state(self._psu_dock_host),
            "global": self._encoded_dock_state(self._global_dock_host),
            "external": self._encoded_dock_state(self._external_dock_host),
            "topology": {
                name: self._dock_topology(host)
                for name, _state_key, host in self._dock_state_hosts()
            },
        }
        configuration_endpoints = getattr(self, "_endpoints", None)
        if configuration_endpoints is None:
            configuration_endpoints = [
                DigitizerEndpoint(self._host, self._port, self._channels)
            ]
        document = {
            "format_version": FORMAT_VERSION,
            "connection": {
                "ip": self._host,
                "port": self._port,
                "channels": self._channels,
                "digitizers": [
                    {
                        "id": f"D{index + 1}",
                        "ip": endpoint.host,
                        "port": endpoint.port,
                        "channels": endpoint.channels,
                    }
                    for index, endpoint in enumerate(configuration_endpoints)
                ],
                "channel_topology": {
                    str(index): self._channel_label(index)
                    for index in range(len(self._devices))
                },
            },
            "hardware": {
                "channels": hardware_channels,
                "shared_trigger": shared_hardware,
                "digitizers": digitizer_hardware,
                "external_devices": external_hardware,
            },
            "application": {
                "main_window": self._window.configuration_settings(),
                "global": global_application,
                "digitizers": digitizer_application,
                "coincidence": coincidence_application,
                "channels": application_channels,
                "external_devices": external_application,
                "external_device_schemas": external_schemas,
                "dock_layout": layout,
            },
        }
        return document

    def load_all_settings(self, path: Path) -> None:
        """Load settings from YAML and apply to hardware, then refresh UI."""
        if getattr(self, "_offline", False):
            raise RuntimeError("hardware settings cannot be loaded in offline mode")
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
            self._scope_controllers[idx].reload_hardware_state()
            mca_ctrl = self._mca_controller_by_device.get(idx)
            if mca_ctrl is not None:
                mca_ctrl._load_hardware_state()
            channel = self._display_channel(idx)
            hardware = channel_entry(document, channel)
            if hardware is not None:
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

        application = document.get("application", {})
        coincidence_settings = (
            application.get("coincidence") if isinstance(application, dict) else None
        )
        if isinstance(coincidence_settings, dict):
            digitizer = self._saved_coincidence_digitizer(coincidence_settings)
            selector = self._coincidence_selector
            if digitizer is not None and selector is not None:
                selector_index = selector.findData(digitizer)
                if selector_index >= 0:
                    selector.blockSignals(True)
                    selector.setCurrentIndex(selector_index)
                    selector.blockSignals(False)
                    self._install_coincidence_digitizer(digitizer)
        if self._coincidence_controller is not None:
            self._coincidence_controller.apply_configuration_settings(
                coincidence_settings
            )

        hardware_root = document.get("hardware", {})
        global_controllers = self._all_global_controllers()
        hardware_digitizers = (
            hardware_root.get("digitizers", {}) if isinstance(hardware_root, dict) else {}
        )
        hardware_digitizers = (
            hardware_digitizers if isinstance(hardware_digitizers, dict) else {}
        )
        application_digitizers = (
            application.get("digitizers", {}) if isinstance(application, dict) else {}
        )
        application_digitizers = (
            application_digitizers if isinstance(application_digitizers, dict) else {}
        )
        for index, global_controller in enumerate(global_controllers):
            digitizer_key = f"D{index + 1}"
            hardware_settings = hardware_digitizers.get(digitizer_key)
            if hardware_settings is None and index == 0 and isinstance(hardware_root, dict):
                hardware_settings = hardware_root.get("shared_trigger")
            global_controller.apply_hardware_configuration_settings(hardware_settings)
            application_settings = application_digitizers.get(digitizer_key)
            if application_settings is None and index == 0 and isinstance(application, dict):
                application_settings = application.get("global")
            global_controller.apply_configuration_settings(application_settings)
            global_controller.refresh_temperature_state()
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
        saved_topology = settings.get("topology")
        for name, _state_key, host in self._dock_state_hosts():
            encoded = settings.get(name)
            if not isinstance(encoded, str):
                continue

            current_topology = self._dock_topology(host)
            host_topology = (
                saved_topology.get(name) if isinstance(saved_topology, dict) else None
            )
            if host_topology != current_topology:
                log.info(
                    "Ignoring configured %s dock layout because its topology "
                    "changed from %r to %r",
                    name,
                    host_topology,
                    current_topology,
                )
                self._reset_dock_host(host)
                continue

            state = QByteArray.fromBase64(encoded.encode("ascii"))
            if not host.restoreState(state):
                log.warning("Ignoring invalid configured %s dock layout", name)
                self._reset_dock_host(host)

    def _channel_dock_hosts(self) -> tuple[tuple[QWidget, QMainWindow], ...]:
        ui = self._window.ui
        return (
            (ui.tabScope, self._scope_dock_host),
            (ui.tabCurrent, self._current_dock_host),
            (ui.tabMCA, self._mca_dock_host),
            (ui.tabPSD, self._psd_dock_host),
            (ui.tabPSU, self._psu_dock_host),
        )

    def _current_channel_dock_host(self) -> QMainWindow | None:
        page = self._window.ui.mainTabs.currentWidget()
        return next((host for tab, host in self._channel_dock_hosts() if tab is page), None)

    @staticmethod
    def _dock_channel(dock: QDockWidget) -> int | None:
        # Dock names contain the flat channel ID, independent of tab order or
        # missing capabilities (e.g. PSD available only on the second channel).
        workspace, separator, channel = dock.objectName().rpartition("_ch")
        if separator and workspace in {"scope", "current", "mca", "psd", "psu"}:
            if channel.isdecimal():
                return int(channel)
        return None

    def _on_channel_dock_activated(self, dock: QDockWidget, visible: bool = True) -> None:
        # Standalone dock hosts do not participate in workspace channel linking.
        if not visible or not hasattr(self, "_active_channel_dock_host"):
            return
        host = self._current_channel_dock_host()
        if host is None or dock.parentWidget() is not host:
            return
        # Incoming docks become visible before mainTabs.currentChanged fires.
        # Ignore that reveal so the destination's old tab cannot replace the
        # channel chosen in the workspace we are leaving.
        # If navigation started on a non-channel page, the first channel view
        # seeds the selection from its restored tab when it becomes visible.
        if self._selected_channel is not None and host is not self._active_channel_dock_host:
            return
        channel = self._dock_channel(dock)
        if channel is not None:
            self._selected_channel = channel

    def _connect_signals(self) -> None:
        self._active_channel_dock_host = self._current_channel_dock_host()
        for _page, host in self._channel_dock_hosts():
            host.tabifiedDockWidgetActivated.connect(self._on_channel_dock_activated)
        self._window.ui.mainTabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int) -> None:
        tab_name = self._window.ui.mainTabs.tabText(index)
        log.info("Active tab: %s", tab_name)
        host = self._current_channel_dock_host()
        self._active_channel_dock_host = host
        channel = self._selected_channel
        if host is None or channel is None:
            return
        for dock in host.findChildren(QDockWidget):
            if self._dock_channel(dock) == channel:
                dock.raise_()
                break
