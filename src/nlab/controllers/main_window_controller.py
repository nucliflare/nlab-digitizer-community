from __future__ import annotations

import heapq
import logging
import socket
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from ipaddress import IPv6Address, ip_address
from pathlib import Path
from time import monotonic_ns
from typing import TYPE_CHECKING, Any, cast

from PySide6.QtCore import QByteArray, QSettings, Qt, QThread, QThreadPool, QTimer
from PySide6.QtWidgets import QComboBox, QDockWidget, QHBoxLayout, QLabel, QMainWindow, QWidget

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
from nlab.hardware.digitizer.offline import (
    create_offline_digitizer,
    create_offline_external_device,
)
from nlab.hardware.modbus_devices import ExternalDevices, ExternalDeviceScan
from nlab.session.recorder import SessionRecorder
from nlab.session.replay import (
    CurrentReplaySample,
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

_REPLAY_ANALYSIS_EVENTS_PER_TICK = 64
_REPLAY_SCOPE_FRAMES_PER_TICK = 64
_REPLAY_CURRENT_SAMPLES_PER_TICK = 4096
_REPLAY_TELEMETRY_EVENTS_PER_TICK = 1024


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
        self._mca_peak_analysis_dialog: McaPeakAnalysisDialog | None = None
        self._psd_readback_dialog: PsdReadbackDialog | None = None
        self._waveform_analysis_dialog: WaveformAnalysisDialog | None = None
        self._external_devices = ExternalDevices()
        self._thread: QThread | None = None
        self._session_recorder: SessionRecorder | None = None
        self._session_replay: SessionReplay | None = None
        self._session_replay_path: Path | None = None
        self._session_replay_spectrum_ids: set[str] = set()
        self._session_replay_origin_ns = 0
        self._session_replay_started_ns = 0
        self._session_replay_psd_channels: set[int] = set()
        self._session_replay_coincidence_active = False
        self._session_analysis_replay_events: list[McaReplayBatch | McaReplayReadback] = []
        self._session_analysis_replay_iterator: Iterator[
            McaReplayBatch | McaReplayReadback
        ] | None = None
        self._session_analysis_replay_next: McaReplayBatch | McaReplayReadback | None = None
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
        self._session_scope_replay_iterator: Iterator[ScopeReplayFrame] | None = None
        self._session_scope_replay_next: ScopeReplayFrame | None = None
        self._session_scope_replay_duration_ns = 0
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
        self._session_telemetry_replay_iterator: Iterator[TelemetryReplayEvent] | None = None
        self._session_telemetry_replay_next: TelemetryReplayEvent | None = None
        self._session_telemetry_replay_index = 0
        self._session_telemetry_replay_origin_ns = 0
        self._session_telemetry_replay_started_ns = 0
        self._session_telemetry_replay_psu_channels: set[int] = set()
        self._session_telemetry_replay_external_ids: set[str] = set()
        self._session_telemetry_replay_global_active = False
        self._session_telemetry_replay_global_digitizers: set[int] = set()
        self._session_telemetry_replay_timer = QTimer()
        self._session_telemetry_replay_timer.setSingleShot(True)
        self._session_telemetry_replay_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._session_telemetry_replay_timer.timeout.connect(
            self._advance_session_telemetry_replay
        )
        self._session_current_replay_iterator: Iterator[CurrentReplaySample] | None = None
        self._session_current_replay_next: CurrentReplaySample | None = None
        self._session_current_replay_channels: set[int] = set()
        self._session_current_replay_timer = QTimer()
        self._session_current_replay_timer.setSingleShot(True)
        self._session_current_replay_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._session_current_replay_timer.timeout.connect(
            self._advance_session_current_replay
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
            )
            self._set_controller_channel_label(scope_ctrl, ch_label)
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
    _DOCK_STATE_KEY_SCOPE = "docks/v2/scope"
    _DOCK_STATE_KEY_CURRENT = "docks/v1/current"
    _DOCK_STATE_KEY_MCA = "docks/v2/mca"
    _DOCK_STATE_KEY_PSD = "docks/v2/psd"
    _DOCK_STATE_KEY_PSU = "docks/v2/psu"
    _DOCK_STATE_KEY_GLOBAL = "docks/v2/global"
    _DOCK_STATE_KEY_EXTERNAL = "docks/v2/external"
    _DOCK_TOPOLOGY_KEY_PREFIX = "docks/topology/v1"

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

    def mca_histogram_under_mouse(self) -> bool:
        """Return whether the pointer is over any live MCA histogram."""
        return any(ctrl.histogram_under_mouse() for ctrl in self._mca_controllers)

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

    def replay_session(self, path: Path) -> SessionReplay:
        """Replay captured analysis inputs and monitored telemetry once."""
        if self.session_recording:
            raise RuntimeError("stop the active session recording before replay")
        coincidence = getattr(self, "_coincidence_controller", None)
        if coincidence is not None and coincidence.active:
            raise RuntimeError("stop the coincidence measurement before replay")
        replay = SessionReplay(path)
        spectra = replay.replay_mca_spectra()
        recorded_channels_reader = getattr(replay, "recorded_channels", None)
        recorded_channels = (
            recorded_channels_reader() if recorded_channels_reader is not None else []
        )
        if getattr(self, "_offline", False) and recorded_channels:
            unavailable = sorted(
                channel for channel in recorded_channels if channel >= self._channels
            )
            if unavailable:
                required = max(unavailable) + 1
                raise RuntimeError(
                    "this session contains channel(s) unavailable in the offline "
                    f"workspace ({', '.join(map(str, unavailable))}); restart Offline "
                    f"Mode with at least {required} channels"
                )
        legacy_scope_frames: list[ScopeReplayFrame] | None = None
        if getattr(replay, "iter_scope_frames", None) is None:
            legacy_scope_frames = replay.scope_frames()
        legacy_telemetry_events: list[TelemetryReplayEvent] | None = None
        if getattr(replay, "iter_telemetry_events", None) is None:
            telemetry_reader = getattr(replay, "telemetry_events", None)
            legacy_telemetry_events = (
                telemetry_reader() if telemetry_reader is not None else []
            )
        self.clear_session_replay()
        self._session_replay = replay
        self._session_replay_path = path
        try:
            origin_reader = getattr(replay, "replay_origin_ns", None)
            if origin_reader is not None:
                self._session_replay_origin_ns = origin_reader()
            else:
                value = replay.manifest.get("started_monotonic_ns", 0)
                self._session_replay_origin_ns = (
                    int(value)
                    if isinstance(value, int) and not isinstance(value, bool)
                    else 0
                )
            self._session_replay_started_ns = monotonic_ns()
            self._set_replay_hardware_interlock(True)
            if getattr(self, "_offline", False):
                self._install_offline_external_replay_controllers(
                    replay,
                    legacy_telemetry_events,
                )
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
            self._start_session_scope_replay(replay, legacy_scope_frames)
            self._start_session_current_replay(replay)
            self._start_session_telemetry_replay(replay, legacy_telemetry_events)
        except Exception:
            self.clear_session_replay()
            raise
        scope_count = sum(self._session_scope_replay_frame_counts.values())
        telemetry_count = len(self._session_telemetry_replay_events)
        log.info(
            "Session replay loaded: %s (%d MCA spectra, %d Scope frames, "
            "%s telemetry updates, integrity=%s)",
            path,
            len(spectra),
            scope_count,
            telemetry_count if legacy_telemetry_events is not None else "streamed",
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
        enabled = bool(enabled) or getattr(self, "_offline", False)
        if enabled == getattr(self, "_replay_hardware_interlocked", False):
            return
        targets: list[object] = []
        coincidence = getattr(self, "_coincidence_controller", None)
        if coincidence is not None:
            targets.append(coincidence)
        targets.extend(getattr(self, "_current_monitor_controllers", []))
        targets.extend(getattr(self, "_scope_controllers", []))
        targets.extend(getattr(self, "_mca_controllers", []))
        global_controllers = self._all_global_controllers()
        targets.extend(global_controllers)
        psu_controllers = getattr(self, "_psu_controllers", [])
        external_controllers = getattr(self, "_external_controllers", [])
        targets.extend(psu_controllers)
        targets.extend(external_controllers)
        background_targets = {
            id(target)
            for target in [*global_controllers, *psu_controllers, *external_controllers]
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

    def _ensure_session_replay_clock(
        self,
        replay: SessionReplay,
        first_timestamp_ns: int,
    ) -> tuple[int, int]:
        """Create one wall-clock mapping shared by every replay subsystem."""
        started = getattr(self, "_session_replay_started_ns", 0)
        origin = getattr(self, "_session_replay_origin_ns", 0)
        if started:
            return origin, started
        manifest_origin = replay.manifest.get("started_monotonic_ns")
        origin = (
            int(manifest_origin)
            if isinstance(manifest_origin, int)
            and not isinstance(manifest_origin, bool)
            and 0 <= manifest_origin <= first_timestamp_ns
            else first_timestamp_ns
        )
        started = monotonic_ns()
        self._session_replay_origin_ns = origin
        self._session_replay_started_ns = started
        return origin, started

    @staticmethod
    def _close_replay_iterator(iterator: object) -> None:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()

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
        readback_channel_reader = getattr(replay, "mca_readback_channels", None)
        event_channel_reader = getattr(replay, "mca_event_channels", None)
        legacy_readbacks: list[McaReplayReadback] | None = None
        legacy_batches: list[McaReplayBatch] | None = None
        if readback_channel_reader is not None:
            readback_channels = set(readback_channel_reader())
        else:
            legacy_readbacks = replay.mca_readbacks()
            readback_channels = {item.channel for item in legacy_readbacks}
        if event_channel_reader is not None:
            event_channels = set(event_channel_reader())
        else:
            legacy_batches = replay.mca_event_batches()
            event_channels = {item.channel for item in legacy_batches}
        configuration = replay.configuration_snapshot()
        application = configuration.get("application")
        application = application if isinstance(application, dict) else {}
        replay_coincidence_settings = application.get("coincidence")
        replay_digitizer = self._saved_coincidence_digitizer(
            replay_coincidence_settings
        )
        selector = getattr(self, "_coincidence_selector", None)
        if replay_digitizer is not None and selector is not None:
            selector_index = selector.findData(replay_digitizer)
            if selector_index >= 0:
                selector.blockSignals(True)
                selector.setCurrentIndex(selector_index)
                selector.blockSignals(False)
                self._install_coincidence_digitizer(replay_digitizer)
                coincidence = self._coincidence_controller
        channel_settings = application.get("channels")
        channel_settings = channel_settings if isinstance(channel_settings, dict) else {}
        hardware = configuration.get("hardware")
        hardware = hardware if isinstance(hardware, dict) else {}
        hardware_channels = hardware.get("channels")
        hardware_channels = hardware_channels if isinstance(hardware_channels, dict) else {}

        replayed_mca_channels = readback_channels & set(mca_controllers)
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

        coincidence_channels: dict[int, int] = {}
        interval_reader = getattr(replay, "coincidence_intervals", None)
        coincidence_intervals = interval_reader() if interval_reader is not None else []
        explicit_timing_reader = getattr(replay, "has_explicit_mca_batch_timing", None)
        explicit_timing = (
            explicit_timing_reader() if explicit_timing_reader is not None else False
        )
        allow_coincidence = bool(coincidence_intervals) or not explicit_timing
        selected_pair = getattr(self, "_coincidence_channels", None)
        if selected_pair is None:
            ordered_mca = getattr(self, "_mca_controllers", [])
            if len(ordered_mca) >= 2:
                selected_pair = (ordered_mca[0].channel, ordered_mca[1].channel)
        if coincidence is not None and selected_pair is not None and allow_coincidence:
            channels = selected_pair
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
        readback_iterator_reader = getattr(replay, "iter_mca_readbacks", None)
        batch_iterator_reader = getattr(replay, "iter_mca_event_batches", None)
        if readback_iterator_reader is not None:
            readback_iterator = readback_iterator_reader(
                channels=replayed_mca_channels,
            )
        else:
            assert legacy_readbacks is not None
            readback_iterator = iter(
                item
                for item in legacy_readbacks
                if item.channel in replayed_mca_channels
            )
        if batch_iterator_reader is not None:
            batch_iterator = batch_iterator_reader(channels=used_batch_channels)
        else:
            assert legacy_batches is not None
            batch_iterator = iter(
                item for item in legacy_batches if item.channel in used_batch_channels
            )
        def replay_key(
            item: McaReplayBatch | McaReplayReadback,
        ) -> tuple[int, int, int, int]:
            return (
                item.host_monotonic_ns,
                0 if isinstance(item, McaReplayBatch) else 1,
                item.channel,
                item.sequence,
            )

        iterator = heapq.merge(readback_iterator, batch_iterator, key=replay_key)
        first = next(iterator, None)
        if first is None:
            return
        origin, started = self._ensure_session_replay_clock(
            replay,
            first.host_monotonic_ns,
        )
        self._session_analysis_replay_events = []
        self._session_analysis_replay_iterator = iterator
        self._session_analysis_replay_next = first
        self._session_analysis_replay_index = 0
        self._session_analysis_replay_origin_ns = origin
        self._session_analysis_replay_started_ns = started
        self._advance_session_analysis_replay()

    def _advance_session_analysis_replay(self) -> None:
        event = getattr(self, "_session_analysis_replay_next", None)
        iterator = getattr(self, "_session_analysis_replay_iterator", None)
        if event is None:
            legacy_events = getattr(self, "_session_analysis_replay_events", [])
            legacy_index = getattr(self, "_session_analysis_replay_index", 0)
            if legacy_index < len(legacy_events):
                event = legacy_events[legacy_index]
                iterator = iter(legacy_events[legacy_index + 1 :])
                self._session_analysis_replay_iterator = iterator
        if getattr(self, "_session_replay", None) is None or event is None:
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
        processed = 0
        while event is not None and processed < _REPLAY_ANALYSIS_EVENTS_PER_TICK:
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
            self._session_analysis_replay_index += 1
            processed += 1
            event = next(iterator, None) if iterator is not None else None
        self._session_analysis_replay_next = event
        for channel, readback in latest_readback.items():
            controller = mca_controllers.get(channel)
            if controller is not None:
                controller.display_session_replay_readback(readback.readback)

        if event is None:
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
            event.host_monotonic_ns - self._session_analysis_replay_origin_ns,
        )
        remaining_ns = max(0, next_elapsed_ns - elapsed_ns)
        delay_ms = max(1, (remaining_ns + 999_999) // 1_000_000)
        self._session_analysis_replay_timer.start(min(delay_ms, 2_147_000_000))

    def _start_session_scope_replay(
        self,
        replay: SessionReplay,
        frames: list[ScopeReplayFrame] | None = None,
    ) -> None:
        controllers = {
            controller.channel: controller for controller in self._scope_controllers
        }
        frame_counts: dict[int, int]
        if frames is None:
            summary = replay.scope_summary()
            frame_counts = {
                channel: count
                for channel, count in summary.counts.items()
                if channel in controllers
            }
            missing = sorted(set(summary.counts) - controllers.keys())
            iterator = replay.iter_scope_frames(channels=frame_counts)
            last_timestamp = summary.last_timestamp_ns
        else:
            playable = [frame for frame in frames if frame.channel in controllers]
            playable.sort(
                key=lambda frame: (
                    frame.host_monotonic_ns,
                    frame.channel,
                    frame.sequence,
                )
            )
            frame_counts = {}
            for frame in playable:
                frame_counts[frame.channel] = frame_counts.get(frame.channel, 0) + 1
            missing = sorted({frame.channel for frame in frames} - controllers.keys())
            iterator = iter(playable)
            last_timestamp = playable[-1].host_monotonic_ns if playable else None
        if missing:
            log.warning(
                "Session has Scope replay data for unavailable channel(s): %s",
                ", ".join(str(channel) for channel in missing),
            )
        if not frame_counts:
            return
        first = next(iterator, None)
        if first is None:
            return
        origin, started = self._ensure_session_replay_clock(
            replay,
            first.host_monotonic_ns,
        )
        channels = set(frame_counts)
        duration_ns = max(0, (last_timestamp or first.host_monotonic_ns) - origin)

        self._session_scope_replay_frames = []
        self._session_scope_replay_iterator = iterator
        self._session_scope_replay_next = first
        self._session_scope_replay_index = 0
        self._session_scope_replay_origin_ns = origin
        self._session_scope_replay_started_ns = started
        self._session_scope_replay_duration_ns = duration_ns
        self._session_scope_replay_channels = channels
        self._session_scope_replay_frame_counts = frame_counts
        self._session_scope_replay_delivered = dict.fromkeys(channels, 0)
        for channel in sorted(channels):
            controllers[channel].begin_session_replay(
                frame_count=frame_counts[channel],
                duration_ns=duration_ns,
            )

        self._advance_session_scope_replay()

    def _advance_session_scope_replay(self) -> None:
        frame = getattr(self, "_session_scope_replay_next", None)
        iterator = getattr(self, "_session_scope_replay_iterator", None)
        if self._session_replay is None or frame is None:
            return
        elapsed_ns = max(0, monotonic_ns() - self._session_scope_replay_started_ns)
        due_by_channel: dict[int, ScopeReplayFrame] = {}
        processed = 0
        while frame is not None and processed < _REPLAY_SCOPE_FRAMES_PER_TICK:
            recorded_elapsed_ns = max(
                0,
                frame.host_monotonic_ns - self._session_scope_replay_origin_ns,
            )
            if recorded_elapsed_ns > elapsed_ns:
                break
            due_by_channel[frame.channel] = frame
            self._session_scope_replay_delivered[frame.channel] += 1
            self._session_scope_replay_index += 1
            processed += 1
            frame = next(iterator, None) if iterator is not None else None
        self._session_scope_replay_next = frame

        controllers = {
            controller.channel: controller for controller in self._scope_controllers
        }
        duration_ns = self._session_scope_replay_duration_ns
        for channel, displayed_frame in due_by_channel.items():
            controller = controllers.get(channel)
            if controller is None:
                continue
            controller.display_session_replay_frame(
                displayed_frame.samples,
                sample_period_ns=displayed_frame.sample_period_ns,
                frame_index=self._session_scope_replay_delivered[channel],
                frame_count=self._session_scope_replay_frame_counts[channel],
                elapsed_ns=max(
                    0,
                    displayed_frame.host_monotonic_ns
                    - self._session_scope_replay_origin_ns,
                ),
                duration_ns=duration_ns,
                source=displayed_frame.source,
            )

        if frame is None:
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
            frame.host_monotonic_ns - self._session_scope_replay_origin_ns,
        )
        remaining_ns = max(0, next_elapsed_ns - elapsed_ns)
        delay_ms = max(1, (remaining_ns + 999_999) // 1_000_000)
        self._session_scope_replay_timer.start(min(delay_ms, 2_147_000_000))

    def _start_session_current_replay(self, replay: SessionReplay) -> None:
        summary_reader = getattr(replay, "current_summary", None)
        iterator_reader = getattr(replay, "iter_current_samples", None)
        if summary_reader is None or iterator_reader is None:
            return
        controllers = {
            controller.channel: controller
            for controller in getattr(self, "_current_monitor_controllers", [])
        }
        summary = summary_reader()
        channels = set(summary.counts) & set(controllers)
        if not channels:
            return
        configuration = replay.configuration_snapshot()
        application = configuration.get("application")
        application = application if isinstance(application, dict) else {}
        channel_settings = application.get("channels")
        channel_settings = channel_settings if isinstance(channel_settings, dict) else {}
        for channel in sorted(channels):
            saved_channel = channel_settings.get(str(channel), {})
            saved_channel = saved_channel if isinstance(saved_channel, dict) else {}
            controllers[channel].begin_session_replay(
                settings=saved_channel.get("current_monitor"),
                dropped_samples=replay.current_dropped_samples(channel),
            )
        iterator = iterator_reader(channels=channels)
        first = next(iterator, None)
        if first is None:
            return
        origin, _started = self._ensure_session_replay_clock(
            replay,
            first.host_monotonic_ns,
        )
        self._session_current_replay_channels = channels
        self._session_current_replay_iterator = iterator
        self._session_current_replay_next = first
        self._session_replay_origin_ns = origin
        self._advance_session_current_replay()

    def _advance_session_current_replay(self) -> None:
        item = getattr(self, "_session_current_replay_next", None)
        iterator = getattr(self, "_session_current_replay_iterator", None)
        if self._session_replay is None or item is None:
            return
        elapsed_ns = max(0, monotonic_ns() - self._session_replay_started_ns)
        controllers = {
            controller.channel: controller
            for controller in getattr(self, "_current_monitor_controllers", [])
        }
        due: dict[int, list[Any]] = {}
        processed = 0
        while item is not None and processed < _REPLAY_CURRENT_SAMPLES_PER_TICK:
            recorded_elapsed_ns = max(
                0,
                item.host_monotonic_ns - self._session_replay_origin_ns,
            )
            if recorded_elapsed_ns > elapsed_ns:
                break
            due.setdefault(item.channel, []).append(item.sample)
            processed += 1
            item = next(iterator, None) if iterator is not None else None
        self._session_current_replay_next = item
        for channel, samples in due.items():
            controller = controllers.get(channel)
            if controller is not None:
                controller.display_session_replay_samples(tuple(samples))
        if item is None:
            for channel in self._session_current_replay_channels:
                controller = controllers.get(channel)
                if controller is not None:
                    controller.finish_session_replay()
            log.info("Current-monitor session replay reached EOF; final views are held")
            return
        next_elapsed_ns = max(
            0,
            item.host_monotonic_ns - self._session_replay_origin_ns,
        )
        remaining_ns = max(0, next_elapsed_ns - elapsed_ns)
        delay_ms = max(1, (remaining_ns + 999_999) // 1_000_000)
        self._session_current_replay_timer.start(min(delay_ms, 2_147_000_000))

    def _psu_replay_controllers(self) -> dict[int, PSUController]:
        result: dict[int, PSUController] = {}
        for device_index, controller in getattr(
            self,
            "_psu_controller_by_device",
            {},
        ).items():
            result[self._display_channel(device_index)] = controller
        return result

    def _global_replay_sources(
        self,
    ) -> dict[str, tuple[int, GlobalController, str]]:
        result: dict[str, tuple[int, GlobalController, str]] = {}
        for digitizer, controller in enumerate(self._all_global_controllers()):
            prefix = "global" if digitizer == 0 else f"digitizer.d{digitizer + 1}"
            result[f"{prefix}.temperature_correction"] = (
                digitizer,
                controller,
                "temperature",
            )
            result[f"{prefix}.diagnostics"] = (
                digitizer,
                controller,
                "diagnostics",
            )
        return result

    def _start_session_telemetry_replay(
        self,
        replay: SessionReplay,
        events: list[TelemetryReplayEvent] | None = None,
    ) -> None:
        psu_controllers = self._psu_replay_controllers()
        external_controllers = {
            controller.configuration_id: controller
            for controller in getattr(self, "_external_controllers", [])
        }
        psu_channels: set[int] = set()
        external_ids: set[str] = set()
        global_sources = self._global_replay_sources()
        if events is None:
            latest_reader = getattr(replay, "latest_telemetry_events", None)
            latest = latest_reader() if latest_reader is not None else {}
            available_sources = set(latest)
        else:
            available_sources = {event.source for event in events}
        playable_sources: set[str] = set()
        for source in available_sources:
            if source.startswith("psu.ch"):
                try:
                    channel = int(source.removeprefix("psu.ch"))
                except ValueError:
                    continue
                if channel in psu_controllers:
                    playable_sources.add(source)
                    psu_channels.add(channel)
            elif source.startswith("external."):
                configuration_id = source.removeprefix("external.")
                if configuration_id in external_controllers:
                    playable_sources.add(source)
                    external_ids.add(configuration_id)
            elif source in global_sources:
                playable_sources.add(source)
        if not playable_sources:
            return
        if events is None:
            iterator = replay.iter_telemetry_events(sources=playable_sources)
            self._session_telemetry_replay_events = []
        else:
            playable = [event for event in events if event.source in playable_sources]
            playable.sort(key=lambda event: (event.host_monotonic_ns, event.sequence))
            iterator = iter(playable)
            self._session_telemetry_replay_events = playable
        first = next(iterator, None)
        if first is None:
            return
        origin, started = self._ensure_session_replay_clock(
            replay,
            first.host_monotonic_ns,
        )
        self._session_telemetry_replay_iterator = iterator
        self._session_telemetry_replay_next = first
        self._session_telemetry_replay_index = 0
        self._session_telemetry_replay_origin_ns = origin
        self._session_telemetry_replay_started_ns = started
        self._session_telemetry_replay_psu_channels = psu_channels
        self._session_telemetry_replay_external_ids = external_ids
        global_digitizers = {
            global_sources[source][0]
            for source in playable_sources
            if source in global_sources
        }
        self._session_telemetry_replay_global_digitizers = global_digitizers
        self._session_telemetry_replay_global_active = 0 in global_digitizers
        for channel in psu_channels:
            psu_controllers[channel].begin_session_replay()
        for configuration_id in external_ids:
            external_controllers[configuration_id].begin_session_replay()
        for digitizer in sorted(global_digitizers):
            self._all_global_controllers()[digitizer].begin_session_replay()
        self._advance_session_telemetry_replay()

    def _advance_session_telemetry_replay(self) -> None:
        event = getattr(self, "_session_telemetry_replay_next", None)
        iterator = getattr(self, "_session_telemetry_replay_iterator", None)
        if self._session_replay is None or event is None:
            return
        elapsed_ns = max(0, monotonic_ns() - self._session_telemetry_replay_started_ns)
        psu_controllers = self._psu_replay_controllers()
        external_controllers = {
            controller.configuration_id: controller
            for controller in getattr(self, "_external_controllers", [])
        }
        global_sources = self._global_replay_sources()
        processed = 0
        while event is not None and processed < _REPLAY_TELEMETRY_EVENTS_PER_TICK:
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
            elif event.source in global_sources:
                _digitizer, global_replay_controller, kind = global_sources[event.source]
                if kind == "temperature":
                    global_replay_controller.display_session_replay_temperature(
                        event.payload.get("readback")
                    )
                else:
                    global_replay_controller.display_session_replay_diagnostics(
                        event.payload.get("readings")
                    )
            self._session_telemetry_replay_index += 1
            processed += 1
            event = next(iterator, None) if iterator is not None else None
        self._session_telemetry_replay_next = event
        if event is None:
            global_controllers = self._all_global_controllers()
            for digitizer in getattr(
                self,
                "_session_telemetry_replay_global_digitizers",
                set(),
            ):
                if digitizer < len(global_controllers):
                    global_controllers[digitizer].finish_session_replay()
            log.info("Session telemetry replay reached EOF; final readbacks are held")
            return
        next_elapsed_ns = max(
            0,
            event.host_monotonic_ns - self._session_telemetry_replay_origin_ns,
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
        current_timer = getattr(self, "_session_current_replay_timer", None)
        if current_timer is not None:
            current_timer.stop()
        for iterator_name in (
            "_session_analysis_replay_iterator",
            "_session_scope_replay_iterator",
            "_session_current_replay_iterator",
            "_session_telemetry_replay_iterator",
        ):
            self._close_replay_iterator(getattr(self, iterator_name, None))
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
        current_channels: set[int] = getattr(
            self,
            "_session_current_replay_channels",
            set(),
        )
        for controller in getattr(self, "_current_monitor_controllers", []):
            if controller.channel in current_channels:
                try:
                    controller.end_session_replay()
                except Exception:
                    log.exception(
                        "Failed to leave Current replay on channel %d",
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
        global_controllers = self._all_global_controllers()
        replayed_global_digitizers = getattr(
            self,
            "_session_telemetry_replay_global_digitizers",
            {0} if getattr(self, "_session_telemetry_replay_global_active", False) else set(),
        )
        for digitizer in replayed_global_digitizers:
            if digitizer < len(global_controllers):
                global_controllers[digitizer].end_session_replay()
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
        self._session_replay_origin_ns = 0
        self._session_replay_started_ns = 0
        self._session_replay_psd_channels = set()
        self._session_replay_coincidence_active = False
        self._session_analysis_replay_events = []
        self._session_analysis_replay_iterator = None
        self._session_analysis_replay_next = None
        self._session_analysis_replay_index = 0
        self._session_analysis_replay_origin_ns = 0
        self._session_analysis_replay_started_ns = 0
        self._session_analysis_replay_mca_channels = set()
        self._session_analysis_replay_coincidence_channels = {}
        self._session_analysis_replay_coincidence_intervals = None
        self._session_scope_replay_frames = []
        self._session_scope_replay_iterator = None
        self._session_scope_replay_next = None
        self._session_scope_replay_duration_ns = 0
        self._session_scope_replay_index = 0
        self._session_scope_replay_origin_ns = 0
        self._session_scope_replay_started_ns = 0
        self._session_scope_replay_channels = set()
        self._session_scope_replay_frame_counts = {}
        self._session_scope_replay_delivered = {}
        self._session_current_replay_iterator = None
        self._session_current_replay_next = None
        self._session_current_replay_channels = set()
        self._session_telemetry_replay_events = []
        self._session_telemetry_replay_iterator = None
        self._session_telemetry_replay_next = None
        self._session_telemetry_replay_index = 0
        self._session_telemetry_replay_origin_ns = 0
        self._session_telemetry_replay_started_ns = 0
        self._session_telemetry_replay_psu_channels = set()
        self._session_telemetry_replay_external_ids = set()
        self._session_telemetry_replay_global_active = False
        self._session_telemetry_replay_global_digitizers = set()
        if getattr(self, "_offline", False) and getattr(
            self, "_external_controllers", []
        ):
            self._clear_external_docks()
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

    def _connect_signals(self) -> None:
        self._window.ui.mainTabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int) -> None:
        tab_name = self._window.ui.mainTabs.tabText(index)
        log.info("Active tab: %s", tab_name)
