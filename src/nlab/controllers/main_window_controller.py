from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import QSettings, Qt, QThread, QThreadPool
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

from nlab.controllers.external_device_controller import ExternalDeviceController
from nlab.controllers.mca_controller import MCAController
from nlab.controllers.psu_controller import PSUController
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.digitizer import Digitizer
from nlab.hardware.modbus_devices import ExternalDevices

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
    ) -> None:
        self._window = window
        self._backend = backend
        self._host = host
        self._port = port
        self._channels = channels
        self._devices: list[Digitizer] = []

        log.info("Connecting via %s to %s:%d, %d channel(s)", backend, host, port, channels)
        for ch in range(1, 1 + self._channels):
            self._devices.append(self._connect_channel(ch))
        log.info("All %d device(s) connected", len(self._devices))

        self._scope_controllers: list[ScopeController] = []
        self._mca_controllers: list[MCAController] = []
        self._psu_controllers: list[PSUController] = []
        self._external_controllers: list[ExternalDeviceController] = []
        self._external_devices = ExternalDevices()
        self._thread: QThread | None = None

        self._scope_dock_host = self._make_dock_host()
        self._mca_dock_host = self._make_dock_host()
        self._psu_dock_host = self._make_dock_host()
        self._external_dock_host = self._make_dock_host()
        self._build_channel_docks()
        self._build_external_docks()
        self._restore_dock_state()
        self._connect_signals()
        log.info("UI initialized, %d scope / %d MCA / %d PSU / %d external controllers",
                 len(self._scope_controllers), len(self._mca_controllers),
                 len(self._psu_controllers), len(self._external_controllers))

    def _connect_channel(self, ch: int) -> Digitizer:
        """Connect one channel using the selected backend.

        gRPC channels are 1-based (GrpcDigitizerBackend's own convention,
        matching hw_def's channel numbering); IIO channels are 0-based
        (ewt-scope0/ewt-scope1 device names), so ch is shifted down by one
        for that backend. Either way self._devices stays a plain list in
        connection order — ch itself is never used as a storage key.
        """
        if self._backend == "iio":
            return Digitizer.from_iio(channel=ch - 1, uri=f"ip:{self._host}:{self._port}")
        return Digitizer.from_grpc(channel=ch, hostname=self._host, port=self._port)

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
        host.setDockOptions(QMainWindow.DockOption.AllowTabbedDocks | QMainWindow.DockOption.AllowNestedDocks | QMainWindow.DockOption.AnimatedDocks)
        return host

    @staticmethod
    def _make_dock(obj_name: str, title: str, widget: QWidget) -> QDockWidget:
        dock = QDockWidget(title)
        dock.setObjectName(obj_name)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
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
        psu_docks: list[QDockWidget] = []

        # TEMPORARY: IIODigitizerBackend's MCABackend is fully unimplemented
        # (every method raises NotImplementedError) -- MCAController.__init__
        # writes default settings to the backend immediately, so constructing
        # it against an IIO device would crash on startup. Skip MCA entirely
        # for this backend until MCA support lands; scope-only testing in the
        # meantime. Remove this guard once MCA is wired up on the IIO side.
        build_mca = self._backend != "iio"

        for idx, device in enumerate(self._devices):
            ch = self._display_channel(idx)
            ch_label = f"Ch {ch}"

            scope_ctrl = ScopeController(device.scope, scope_dma=device.scope_dma, channel=ch)
            self._scope_controllers.append(scope_ctrl)
            scope_docks.append(self._make_dock(f"scope_ch{ch}", ch_label, scope_ctrl))

            if build_mca:
                mca_ctrl = MCAController(device.mca, mca_dma=device.mca_dma, channel=ch)
                self._mca_controllers.append(mca_ctrl)
                mca_docks.append(self._make_dock(f"mca_ch{ch}", ch_label, mca_ctrl))

            if device.hv is not None:
                psu_ctrl = PSUController(device.hv)
                self._psu_controllers.append(psu_ctrl)
                psu_docks.append(self._make_dock(f"psu_ch{ch}", ch_label, psu_ctrl))

        self._populate_dock_host(self._scope_dock_host, scope_docks)
        self._populate_dock_host(self._mca_dock_host, mca_docks)
        self._populate_dock_host(self._psu_dock_host, psu_docks)

        self._window.ui.layoutTabScope.addWidget(self._scope_dock_host)
        self._window.ui.layoutTabMCA.addWidget(self._mca_dock_host)
        self._window.ui.layoutTabPSU.addWidget(self._psu_dock_host)

        mca_tab_index = self._window.ui.mainTabs.indexOf(self._window.ui.tabMCA)
        self._window.ui.mainTabs.setTabEnabled(mca_tab_index, build_mca)
        self._window.ui.tabMCA.setToolTip(
            "" if build_mca else
            "MCA is temporarily disabled: the IIO backend does not implement it yet."
        )

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

        self._populate_dock_host(self._external_dock_host, docks)
        self._window.ui.layoutTabExternal.addWidget(self._external_dock_host)

    # ------------------------------------------------------------------
    # Dock state persistence
    # ------------------------------------------------------------------

    # Bump suffix when dock object names or topology change so stale layouts
    # are silently discarded rather than corrupting the initial tab arrangement.
    _DOCK_STATE_KEY_SCOPE = "docks/v2/scope"
    _DOCK_STATE_KEY_MCA = "docks/v2/mca"
    _DOCK_STATE_KEY_PSU = "docks/v2/psu"
    _DOCK_STATE_KEY_EXTERNAL = "docks/v2/external"

    def _save_dock_state(self) -> None:
        settings = QSettings()
        settings.setValue(self._DOCK_STATE_KEY_SCOPE, self._scope_dock_host.saveState())
        settings.setValue(self._DOCK_STATE_KEY_MCA, self._mca_dock_host.saveState())
        settings.setValue(self._DOCK_STATE_KEY_PSU, self._psu_dock_host.saveState())
        settings.setValue(self._DOCK_STATE_KEY_EXTERNAL, self._external_dock_host.saveState())

    def _restore_dock_state(self) -> None:
        settings = QSettings()
        if state := settings.value(self._DOCK_STATE_KEY_SCOPE):
            if not self._scope_dock_host.restoreState(state):
                settings.remove(self._DOCK_STATE_KEY_SCOPE)
        if state := settings.value(self._DOCK_STATE_KEY_MCA):
            if not self._mca_dock_host.restoreState(state):
                settings.remove(self._DOCK_STATE_KEY_MCA)
        if state := settings.value(self._DOCK_STATE_KEY_PSU):
            if not self._psu_dock_host.restoreState(state):
                settings.remove(self._DOCK_STATE_KEY_PSU)
        if state := settings.value(self._DOCK_STATE_KEY_EXTERNAL):
            if not self._external_dock_host.restoreState(state):
                settings.remove(self._DOCK_STATE_KEY_EXTERNAL)

    def reset_dock_layout(self) -> None:
        """Clear saved dock state and re-tabify all channel docks."""
        settings = QSettings()
        settings.remove(self._DOCK_STATE_KEY_SCOPE)
        settings.remove(self._DOCK_STATE_KEY_MCA)
        settings.remove(self._DOCK_STATE_KEY_PSU)
        settings.remove(self._DOCK_STATE_KEY_EXTERNAL)

        dock_hosts = (
            self._scope_dock_host, self._mca_dock_host,
            self._psu_dock_host, self._external_dock_host,
        )
        for host in dock_hosts:
            docks = host.findChildren(QDockWidget)
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
        log.info("Zoom reset on all plots")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _stop_all_workers(self) -> None:
        """Stop all running timers/workers (blocking). Devices stay open."""
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

        # 4. Wait for in-flight scope workers to finish
        QThreadPool.globalInstance().waitForDone(3000)

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
        self._save_dock_state()
        for ctrl in self._scope_controllers:
            ctrl.save_display_settings()

        self._stop_all_workers()

        for device in self._devices:
            device.close()
        self._devices.clear()
        self._external_devices.close()
        self._external_devices = ExternalDevices()

        dock_hosts = (
            self._scope_dock_host, self._mca_dock_host,
            self._psu_dock_host, self._external_dock_host,
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
        self._psu_controllers.clear()
        self._external_controllers.clear()

        log.info("Reconnect: connecting via %s to %s:%d, %d channel(s)",
                 self._backend, self._host, self._port, self._channels)
        for ch in range(1, 1 + self._channels):
            self._devices.append(self._connect_channel(ch))
        log.info("Reconnect: all %d device(s) connected", len(self._devices))

        self._build_channel_docks()
        self._build_external_docks()
        self._restore_dock_state()
        log.info("Reconnect complete, %d scope / %d MCA / %d PSU / %d external controllers",
                 len(self._scope_controllers), len(self._mca_controllers),
                 len(self._psu_controllers), len(self._external_controllers))

    def save_all_settings(self, path) -> None:
        """Save settings for all channels to a single YAML file."""
        from nlab.utils.settings_io import save_settings
        for idx, device in enumerate(self._devices):
            ch_path = path.with_stem(f"{path.stem}_ch{self._display_channel(idx)}")
            save_settings(device.scope, device.mca, device.hv, ch_path)
        log.info("All channel settings saved to %s", path.parent)

    def load_all_settings(self, path) -> None:
        """Load settings from YAML and apply to hardware, then refresh UI."""
        from nlab.utils.settings_io import load_settings
        for idx, device in enumerate(self._devices):
            ch_path = path.with_stem(f"{path.stem}_ch{self._display_channel(idx)}")
            if ch_path.exists():
                load_settings(device.scope, device.mca, device.hv, ch_path)
        for ctrl in self._scope_controllers:
            ctrl._load_hardware_state()
        for ctrl in self._mca_controllers:
            ctrl._load_hardware_state()
        log.info("All channel settings loaded and UI refreshed")

    def _connect_signals(self) -> None:
        self._window.ui.mainTabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int) -> None:
        tab_name = self._window.ui.mainTabs.tabText(index)
        log.info("Active tab: %s", tab_name)
