from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QByteArray, QSettings, Qt, QThread, QThreadPool
from PySide6.QtWidgets import QDockWidget, QInputDialog, QMainWindow, QWidget

from nlab.analysis.psd_file import inspect_psd_event_file
from nlab.analysis.waveform_file import inspect_waveform_file
from nlab.controllers.coincidence_controller import CoincidenceController
from nlab.controllers.external_device_controller import ExternalDeviceController
from nlab.controllers.global_controller import GlobalController
from nlab.controllers.mca_controller import MCAController
from nlab.controllers.psd_controller import PSDController
from nlab.controllers.psu_controller import PSUController
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.digitizer import Digitizer
from nlab.hardware.digitizer.dma import IIOMcaDmaStreamer, McaEventBuffer
from nlab.hardware.modbus_devices import ExternalDevices
from nlab.views.energy_calibration_dialog import EnergyCalibrationDialog
from nlab.views.mca_peak_analysis_dialog import McaPeakAnalysisDialog
from nlab.workers.psd_file_worker import PsdFileWorker

if TYPE_CHECKING:
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
        backend: str = "grpc",
        host: str = "",
        port: int = 50050,
        channels: int = 2,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self._window = window
        self._backend = backend
        self._host = host
        self._port = port
        self._channels = channels
        self._devices: list[Digitizer] = []
        self._on_progress = on_progress

        log.info("Connecting via %s to %s:%d, %d channel(s)", backend, host, port, channels)
        for ch in range(1, 1 + self._channels):
            self._report_progress(f"Connecting channel {ch} of {self._channels}...")
            self._devices.append(self._connect_channel(ch))
        log.info("All %d device(s) connected", len(self._devices))

        self._scope_controllers: list[ScopeController] = []
        self._mca_controllers: list[MCAController] = []
        self._psd_controllers: list[PSDController] = []
        self._psd_controller_by_device: dict[int, PSDController] = {}
        self._psu_controllers: list[PSUController] = []
        self._psu_controller_by_device: dict[int, PSUController] = {}
        self._external_controllers: list[ExternalDeviceController] = []
        self._global_controller: GlobalController | None = None
        self._coincidence_controller: CoincidenceController | None = None
        self._energy_calibration_dialog: EnergyCalibrationDialog | None = None
        self._mca_peak_analysis_dialog: McaPeakAnalysisDialog | None = None
        self._external_devices = ExternalDevices()
        self._thread: QThread | None = None
        self._psd_file_thread: QThread | None = None
        self._psd_file_worker: PsdFileWorker | None = None

        self._scope_dock_host = self._make_dock_host()
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
            "UI initialized, %d scope / %d MCA / %d PSD / %d PSU / %d external controllers",
            len(self._scope_controllers),
            len(self._mca_controllers),
            len(self._psd_controllers),
            len(self._psu_controllers),
            len(self._external_controllers),
        )

    def _report_progress(self, message: str) -> None:
        callback = getattr(self, "_on_progress", None)
        if callback is not None:
            callback(message)

    def _connect_channel(self, ch: int) -> Digitizer:
        """Connect one channel using the selected backend.

        gRPC channels are 1-based (GrpcDigitizerBackend's own convention,
        matching hw_def's channel numbering); IIO channels are 0-based
        (ewt-scope0/ewt-scope1 device names), so ch is shifted down by one
        for that backend. Either way self._devices stays a plain list in
        connection order — ch itself is never used as a storage key.
        """
        if self._backend == "iio":
            return Digitizer.from_iio(
                channel=ch - 1,
                uri=f"ip:{self._host}:{self._port}",
                with_ids=True,
            )
        return Digitizer.from_grpc(
            channel=ch,
            hostname=self._host,
            port=self._port,
            with_ids=True,
        )

    def _display_channel(self, idx: int) -> int:
        """Channel number the way the connected backend's own hardware
        labels it -- 0-based for IIO (vdpp_scope instance 0/1, matching
        iio_info's device discovery order), 1-based for gRPC (hw_def's
        channel numbering, unchanged from before). Used for tab labels,
        dock object names, and settings filenames so a channel number
        mentioned anywhere (GUI, logs, saved files, a live debugging
        session cross-referencing the board) means the same physical
        channel everywhere -- instead of the GUI always showing one
        higher than what dmesg/iio_info call it, and needing the
        ch-1 IIO offset in your head to translate between them.
        """
        return idx if self._backend == "iio" else idx + 1

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
        mca_docks: list[QDockWidget] = []
        psd_docks: list[QDockWidget] = []
        psu_docks: list[QDockWidget] = []

        self._psu_controller_by_device = {}
        self._psd_controller_by_device = {}
        missing_psu_channels: list[int] = []

        # IIODigitizerBackend now implements MCABackend against
        # vdpp-pulse-processor.c/vdpp-input-filter.c (see iio_backend.py's
        # module docstring for the few methods that still raise
        # NotImplementedError -- none of them are on MCAController's
        # unconditional default-hydration path, so construction is safe).
        # Only skip MCA docks if this channel's firmware genuinely lacks
        # the pulse-processor/input-filter devices (older/scope-only
        # builds) -- MCAController.__init__ would otherwise crash on the
        # RuntimeError those methods raise instead. Digitizer.mca_available()
        # asks the backend rather than special-casing "iio" by name here,
        # so this doesn't need updating if another backend gains the same
        # per-device variability later.
        build_mca = all(device.mca_available() for device in self._devices)

        for idx, device in enumerate(self._devices):
            ch = self._display_channel(idx)
            self._report_progress(f"Initializing channel {ch} controls...")
            ch_label = f"Ch {ch}"

            scope_ctrl = ScopeController(device.scope, scope_dma=device.scope_dma, channel=ch)
            self._scope_controllers.append(scope_ctrl)
            scope_docks.append(self._make_dock(f"scope_ch{ch}", ch_label, scope_ctrl))

            if build_mca:
                event_buffer = None
                psd_ctrl = None
                if device.mca_dma is not None:
                    event_buffer = McaEventBuffer()
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
                mca_docks.append(self._make_dock(f"mca_ch{ch}", ch_label, mca_ctrl))

            hv = device.hv
            if hv is None:
                missing_psu_channels.append(ch)
                log.warning("PSU controls unavailable for channel %s", ch)
            else:
                psu_ctrl = PSUController(hv)
                self._psu_controllers.append(psu_ctrl)
                self._psu_controller_by_device[idx] = psu_ctrl
                psu_docks.append(self._make_dock(f"psu_ch{ch}", ch_label, psu_ctrl))

        self._populate_dock_host(self._scope_dock_host, scope_docks)
        self._populate_dock_host(self._mca_dock_host, mca_docks)
        self._populate_dock_host(self._psd_dock_host, psd_docks)
        self._populate_dock_host(self._psu_dock_host, psu_docks)

        self._window.ui.layoutTabScope.addWidget(self._scope_dock_host)
        self._window.ui.layoutTabMCA.addWidget(self._mca_dock_host)
        self._window.ui.layoutTabPSD.addWidget(self._psd_dock_host)
        self._window.ui.layoutTabPSU.addWidget(self._psu_dock_host)

        mca_tab_index = self._window.ui.mainTabs.indexOf(self._window.ui.tabMCA)
        self._window.ui.mainTabs.setTabEnabled(mca_tab_index, build_mca)
        self._window.ui.tabMCA.setToolTip(
            ""
            if build_mca
            else "MCA is disabled: no pulse-processor/input-filter device found "
            "for one or more connected channels (older or scope-only firmware)."
        )

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
        dock = self._make_dock("global_panel", "Global", self._global_controller)
        self._populate_dock_host(self._global_dock_host, [dock])
        layout = self._window.ui.layoutTabGlobal
        if layout.indexOf(self._global_dock_host) < 0:
            layout.addWidget(self._global_dock_host)

    def _build_coincidence_tab(self) -> None:
        available = (
            self._backend == "iio"
            and len(self._mca_controllers) >= 2
            and len(self._devices) >= 2
            and all(isinstance(device.mca_dma, IIOMcaDmaStreamer) for device in self._devices[:2])
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
            self._coincidence_controller = CoincidenceController(
                self._devices[:2], self._mca_controllers[:2], self._global_controller
            )
            self._window.ui.layoutTabCoincidence.addWidget(self._coincidence_controller)

    def _build_external_docks(self) -> None:
        """Discover Modbus devices on the digitizer host and dock one tab each.

        The SiPM bias board, Geiger-Mueller probe, and PMT HV supply share the
        digitizer's RS-485 bus via a ser2net TCP bridge — same host as the
        gRPC digitizer connection. A device that doesn't respond (not present,
        or bus not bridged) is simply absent from the discovery results.
        """
        devices = self._external_devices.discover(self._host)
        docks: list[QDockWidget] = []
        for idx, device in enumerate(devices):
            ctrl = ExternalDeviceController(device)
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

    # ------------------------------------------------------------------
    # Dock state persistence
    # ------------------------------------------------------------------

    # Bump suffix when dock object names or topology change so stale layouts
    # are silently discarded rather than corrupting the initial tab arrangement.
    _DOCK_STATE_KEY_SCOPE = "docks/v2/scope"
    _DOCK_STATE_KEY_MCA = "docks/v2/mca"
    _DOCK_STATE_KEY_PSD = "docks/v2/psd"
    _DOCK_STATE_KEY_PSU = "docks/v2/psu"
    _DOCK_STATE_KEY_GLOBAL = "docks/v2/global"
    _DOCK_STATE_KEY_EXTERNAL = "docks/v2/external"

    def _save_dock_state(self) -> None:
        settings = QSettings()
        settings.setValue(self._DOCK_STATE_KEY_SCOPE, self._scope_dock_host.saveState())
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
        settings.remove(self._DOCK_STATE_KEY_MCA)
        settings.remove(self._DOCK_STATE_KEY_PSD)
        settings.remove(self._DOCK_STATE_KEY_PSU)
        settings.remove(self._DOCK_STATE_KEY_GLOBAL)
        settings.remove(self._DOCK_STATE_KEY_EXTERNAL)

        dock_hosts = (
            self._scope_dock_host,
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
        """Auto-range/reset zoom on every scope and MCA plot."""
        for ctrl in self._scope_controllers:
            ctrl.reset_zoom()
        for ctrl in self._mca_controllers:
            ctrl.reset_zoom()
        for ctrl in self._psd_controllers:
            ctrl.reset_zoom()
        log.info("Zoom reset on all plots")

    def refresh_dma_output_settings(self) -> None:
        """Refresh format-dependent controls after the DMA settings dialog."""
        for ctrl in self._mca_controllers:
            ctrl.refresh_dma_output_settings()
        if self._coincidence_controller is not None:
            self._coincidence_controller.refresh_dma_output_settings()

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

    def load_psd_events(self, path: Path) -> None:
        """Reconstruct one PSD view from a saved event file off the GUI thread."""
        if self._psd_file_thread is not None:
            raise RuntimeError("A PSD event file is already being processed")
        if not self._psd_controllers:
            raise RuntimeError("No PSD channel is available for displaying this file")

        info = inspect_psd_event_file(path)
        target = next(
            (ctrl for ctrl in self._psd_controllers if ctrl.channel == info.channel),
            None,
        )
        if target is None and len(self._psd_controllers) == 1:
            target = self._psd_controllers[0]
        if target is None:
            choices = [f"PSD channel {ctrl.channel}" for ctrl in self._psd_controllers]
            selected, accepted = QInputDialog.getItem(
                self._window,
                "Select PSD Display",
                f"Source channel {info.channel!r} has no matching display. Load into:",
                choices,
                0,
                False,
            )
            if not accepted:
                return
            target = self._psd_controllers[choices.index(selected)]

        energy_bins, ratio_bins, energy_right_shift, ratio_range = (
            target.file_analysis_settings()
        )
        worker = PsdFileWorker(
            info,
            energy_bins=energy_bins,
            ratio_bins=ratio_bins,
            energy_right_shift=energy_right_shift,
            ratio_range=ratio_range,
        )
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(target.update_file_load_progress)
        worker.loaded.connect(target.finish_file_load)
        worker.cancelled.connect(target.cancel_file_load)
        worker.error.connect(target.fail_file_load)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_psd_file_thread_finished)
        target.begin_file_load(path)
        self._psd_file_worker = worker
        self._psd_file_thread = thread

        self._window.ui.mainTabs.setCurrentWidget(self._window.ui.tabPSD)
        for dock in self._psd_dock_host.findChildren(QDockWidget):
            if dock.widget() is target:
                dock.raise_()
                break
        thread.start()
        log.info(
            "PSD file load started: %s (%s, %d events, channel=%s)",
            path,
            info.format_name,
            info.total_events,
            info.channel,
        )

    def load_waveform_file(self, path: Path) -> None:
        """Route a CAEN or NLab DMA waveform file to a Scope panel."""
        if not self._scope_controllers:
            raise RuntimeError("No Scope channel is available for displaying this file")
        info = inspect_waveform_file(path)
        target = next(
            (ctrl for ctrl in self._scope_controllers if ctrl.channel == info.channel),
            None,
        )
        if target is None and len(self._scope_controllers) == 1:
            target = self._scope_controllers[0]
        if target is None:
            choices = [f"Scope channel {ctrl.channel}" for ctrl in self._scope_controllers]
            selected, accepted = QInputDialog.getItem(
                self._window,
                "Select Scope Display",
                f"Source channel {info.channel} has no matching display. Load into:",
                choices,
                0,
                False,
            )
            if not accepted:
                return
            target = self._scope_controllers[choices.index(selected)]

        target.open_waveform_file(path)
        self._window.ui.mainTabs.setCurrentWidget(self._window.ui.tabScope)
        for dock in self._scope_dock_host.findChildren(QDockWidget):
            if dock.widget() is target:
                dock.raise_()
                break
        log.info(
            "Waveform file load started: %s (%s, source channel=%d, display channel=%d)",
            path,
            info.format_name,
            info.channel,
            target.channel,
        )

    def _on_psd_file_thread_finished(self) -> None:
        self._psd_file_worker = None
        self._psd_file_thread = None

    def _stop_psd_file_load_sync(self) -> None:
        worker = getattr(self, "_psd_file_worker", None)
        thread = getattr(self, "_psd_file_thread", None)
        if worker is not None:
            worker.stop()
        if thread is not None and not thread.wait(3000):
            log.warning("PSD file worker did not stop in time, terminating")
            thread.terminate()
            thread.wait()
        self._psd_file_worker = None
        self._psd_file_thread = None

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
        for ctrl in self._psu_controllers:
            ctrl.request_monitor_stop()
        for ctrl in self._external_controllers:
            ctrl.request_polling_stop()
        if worker := getattr(self, "_psd_file_worker", None):
            worker.stop()

        if self._global_controller is not None:
            self._global_controller.disarm_sync()
            self._global_controller.stop_polling_sync()

        # 1. Stop scope timers — no new workers will be submitted
        for ctrl in self._scope_controllers:
            ctrl._refresh_timer.stop()

        # 2. Stop DMA workers (blocking)
        for ctrl in self._scope_controllers:
            ctrl.stop_dma_sync()
        for ctrl in self._mca_controllers:
            ctrl.stop_dma_sync()

        # 3. Stop MCA polling workers (blocking)
        for ctrl in self._mca_controllers:
            ctrl.stop_worker_sync()

        if coincidence is not None:
            coincidence.finish_shutdown_sync()

        self._stop_psd_file_load_sync()

        # PSD timers consume only already-decoded display batches. Stop them
        # after MCA DMA tail drain has completed.
        for ctrl in self._psd_controllers:
            ctrl.stop_processing()

        # 4. Wait for in-flight scope workers to finish
        viewer_pool_idle = QThreadPool.globalInstance().waitForDone(3000)
        if viewer_pool_idle:
            for scope_controller in self._scope_controllers:
                scope_controller.close_viewer_client()
        else:
            # An in-flight worker still owns its isolated context. Do not
            # close that context out from under a native libiio read.
            log.warning("Scope viewer workers did not stop within 3 seconds")

        # 5. Stop PSU workers (blocking)
        for ctrl in self._psu_controllers:
            ctrl.stop_monitor_sync()

        # 6. Stop external Modbus device workers (blocking)
        for ctrl in self._external_controllers:
            ctrl.stop_polling_sync()

        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None

    def shutdown(self) -> None:
        """Persist dock layout, stop all workers, close all devices."""
        log.info("Shutdown: persisting state")
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
        self._mca_controllers.clear()
        self._psd_controllers.clear()
        self._psd_controller_by_device.clear()
        self._psu_controllers.clear()
        self._psu_controller_by_device.clear()
        self._external_controllers.clear()

        log.info(
            "Reconnect: connecting via %s to %s:%d, %d channel(s)",
            self._backend,
            self._host,
            self._port,
            self._channels,
        )
        for ch in range(1, 1 + self._channels):
            self._devices.append(self._connect_channel(ch))
        log.info("Reconnect: all %d device(s) connected", len(self._devices))

        self._build_global_tab()
        self._build_channel_docks()
        self._build_external_docks()
        self._restore_dock_state()
        log.info(
            "Reconnect complete, %d scope / %d MCA / %d PSD / %d PSU / %d external controllers",
            len(self._scope_controllers),
            len(self._mca_controllers),
            len(self._psd_controllers),
            len(self._psu_controllers),
            len(self._external_controllers),
        )

    def save_all_settings(self, path) -> None:
        """Save settings for all channels to a single YAML file."""
        from nlab.utils.settings_io import write_configuration

        write_configuration(path, self.configuration_document())
        log.info("All channel and application settings saved to %s", path)

    def configuration_document(self) -> dict[str, object]:
        """Snapshot the same complete document used by Save Settings."""
        from nlab.utils.settings_io import FORMAT_VERSION, collect_channel_hardware

        hardware_channels: dict[str, object] = {}
        application_channels: dict[str, object] = {}
        for idx, device in enumerate(self._devices):
            channel = self._display_channel(idx)
            mca_ctrl = self._mca_controllers[idx] if idx < len(self._mca_controllers) else None
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
            "scope": self._scope_dock_host.saveState().toBase64().data().decode("ascii"),
            "mca": self._mca_dock_host.saveState().toBase64().data().decode("ascii"),
            "psd": self._psd_dock_host.saveState().toBase64().data().decode("ascii"),
            "psu": self._psu_dock_host.saveState().toBase64().data().decode("ascii"),
            "global": self._global_dock_host.saveState().toBase64().data().decode("ascii"),
            "external": self._external_dock_host.saveState().toBase64().data().decode("ascii"),
        }
        document = {
            "format_version": FORMAT_VERSION,
            "connection": {
                "backend": self._backend,
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

    def load_all_settings(self, path) -> None:
        """Load settings from YAML and apply to hardware, then refresh UI."""
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
            if idx < len(self._mca_controllers):
                self._mca_controllers[idx]._load_hardware_state()
            channel = self._display_channel(idx)
            hardware = channel_entry(document, channel)
            if hardware is not None:
                mca_settings = hardware.get("mca", {})
                if idx < len(self._mca_controllers) and isinstance(mca_settings, dict):
                    low_pass = mca_settings.get("low_pass", {})
                    if isinstance(low_pass, dict) and "preset" in low_pass:
                        self._mca_controllers[idx].populate_hardware_configuration_settings(
                            {"low_pass_preset": low_pass["preset"]}
                        )
                psu_ctrl = self._psu_controller_by_device.get(idx)
                if psu_ctrl is not None:
                    psu_settings = hardware.get("psu", {})
                    psu_ctrl.populate_hardware_configuration_settings(psu_settings)

            app_channel = channel_application_entry(document, channel)
            if app_channel is not None:
                self._scope_controllers[idx].apply_configuration_settings(app_channel.get("scope"))
                psu_ctrl = self._psu_controller_by_device.get(idx)
                if psu_ctrl is not None:
                    psu_ctrl.apply_configuration_settings(app_channel.get("psu"))
                if idx < len(self._mca_controllers):
                    self._mca_controllers[idx].apply_configuration_settings(app_channel.get("mca"))
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
