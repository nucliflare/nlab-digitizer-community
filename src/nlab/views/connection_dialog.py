from __future__ import annotations

import logging
import threading
from typing import cast

from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QWidget

from nlab.hardware.digitizer.discovery import (
    USB_GADGET_HOST,
    DiscoveredDigitizer,
    DiscoveryResult,
    discover_iio_digitizers,
)
from nlab.ui.ui_connection_dialog import Ui_ConnectionDialog
from nlab.utils.windows_icon import apply_taskbar_icon

log = logging.getLogger(__name__)

_MAX_RECENT_IPS = 10
_DEFAULT_CHANNELS = 2
_DISCOVERY_ROLE = Qt.ItemDataRole.UserRole

# Backend combo index -> internal key -> default port. gRPC's default here
# matches Digitizer.from_grpc's own default (50050); IIO's matches the
# ewt-scope IIO device tree's iiod port (30431).
_BACKENDS = ("grpc", "iio")
_DEFAULT_BACKEND = "grpc"
_BACKEND_DEFAULT_PORTS = {"grpc": 50050, "iio": 30431}

_KEY_RECENT_IPS = "connection/recent_ips"
_KEY_LAST_BACKEND = "connection/last_backend"
_KEY_LAST_PORT = "connection/last_port"
_KEY_LAST_CHANNELS = "connection/last_channels"


class ConnectionDialog(QDialog):
    """Pre-launch dialog for entering device IP and port.

    IP history is persisted via QSettings.  A background libiio network scan
    populates the same drop-down and separately checks the board's fixed USB
    gadget-network address.
    """

    _discovery_finished = Signal(int, object, object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        ip: str | None = None,
        port: int | None = None,
        backend: str | None = None,
        channels: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._ui = Ui_ConnectionDialog()
        self._ui.setupUi(self)  # type: ignore[no-untyped-call]
        self._ui.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setText("Connect")
        self._ui.buttonBox.accepted.connect(self._on_accept)
        self._ui.buttonBox.rejected.connect(self.reject)
        self._ui.comboBackend.currentIndexChanged.connect(self._on_backend_changed)
        self._ui.comboIp.currentIndexChanged.connect(self._on_address_selected)
        line_edit = self._ui.comboIp.lineEdit()
        if line_edit is not None:
            line_edit.textEdited.connect(self._on_address_edited)
        self._ui.buttonRescan.clicked.connect(self._start_discovery)
        self._discovery_finished.connect(self._on_discovery_finished)
        self._scan_generation = 0
        self._scan_running = False
        self._address_edited_since_scan = False
        self._explicit_ip = ip is not None
        self._recent_ips: set[str] = set()
        self._discovered_hosts: set[str] = set()
        apply_taskbar_icon(self)
        self._load_settings()
        self._apply_initial_values(ip=ip, port=port, backend=backend, channels=channels)
        QTimer.singleShot(0, self._start_discovery)

    # ------------------------------------------------------------------
    # Public properties — read after exec() == Accepted
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Internal backend key: "grpc" or "iio"."""
        return _BACKENDS[self._ui.comboBackend.currentIndex()]

    @property
    def ip(self) -> str:
        return self._ui.comboIp.currentText().strip()

    @property
    def port(self) -> int:
        return self._ui.spinPort.value()

    @property
    def channels(self) -> int:
        return self._ui.spinChannels.value()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_backend_changed(self, index: int) -> None:
        """Snap the port to the newly-selected backend's default.

        Only a live user interaction reaches this (see _load_settings,
        which blocks signals while restoring the saved index), so this
        never fights with a restored last-used port on startup.
        """
        backend = _BACKENDS[index]
        self._ui.spinPort.setValue(_BACKEND_DEFAULT_PORTS[backend])

    def _on_address_edited(self, _text: str) -> None:
        self._address_edited_since_scan = True

    def _on_address_selected(self, index: int) -> None:
        endpoint = self._ui.comboIp.itemData(index, _DISCOVERY_ROLE)
        if not isinstance(endpoint, DiscoveredDigitizer):
            return
        self._ui.comboBackend.setCurrentIndex(_BACKENDS.index("iio"))
        self._ui.spinPort.setValue(endpoint.port)

    def _start_discovery(self) -> None:
        if self._scan_running:
            return
        self._scan_running = True
        self._scan_generation += 1
        generation = self._scan_generation
        self._ui.buttonRescan.setEnabled(False)
        self._ui.buttonRescan.setText("Scanning…")
        self._ui.labelDiscoveryStatus.setText(
            f"Scanning local network (mDNS/Avahi) and USB address {USB_GADGET_HOST}…"
        )
        self._ui.labelDiscoveryStatus.setToolTip("")
        threading.Thread(
            target=self._discover_in_background,
            args=(generation,),
            name="digitizer-discovery",
            daemon=True,
        ).start()

    def _discover_in_background(self, generation: int) -> None:
        try:
            result = discover_iio_digitizers()
            error: Exception | None = None
        except Exception as exc:  # defensive: keep manual entry usable
            result = DiscoveryResult(())
            error = exc
            log.exception("Unexpected digitizer discovery failure")
        try:
            self._discovery_finished.emit(generation, result, error)
        except RuntimeError:
            # The application may have closed while the bounded scan was running.
            pass

    def _on_discovery_finished(
        self,
        generation: int,
        result_object: object,
        error_object: object,
    ) -> None:
        if generation != self._scan_generation:
            return
        self._scan_running = False
        self._ui.buttonRescan.setEnabled(True)
        self._ui.buttonRescan.setText("Rescan")

        result = (
            result_object if isinstance(result_object, DiscoveryResult) else DiscoveryResult(())
        )
        self._replace_discovered_addresses(result.digitizers)

        count = len(result.digitizers)
        if count:
            noun = "digitizer" if count == 1 else "digitizers"
            self._ui.labelDiscoveryStatus.setText(
                f"Found {count} {noun}. Select an address above or enter one manually."
            )
        elif error_object is not None or result.network_error is not None:
            self._ui.labelDiscoveryStatus.setText(
                "Network discovery was unavailable. Enter an address manually or rescan."
            )
        else:
            self._ui.labelDiscoveryStatus.setText(
                "No digitizers found. Enter an address manually or rescan."
            )

        details = result.network_error
        if isinstance(error_object, Exception):
            details = str(error_object) or type(error_object).__name__
        self._ui.labelDiscoveryStatus.setToolTip(details or "")

    def _replace_discovered_addresses(self, digitizers: tuple[DiscoveredDigitizer, ...]) -> None:
        combo = self._ui.comboIp
        current_text = combo.currentText()
        preserve_current = self._explicit_ip or self._address_edited_since_scan

        for index in range(combo.count() - 1, -1, -1):
            host = combo.itemText(index)
            if host in self._discovered_hosts and host not in self._recent_ips:
                combo.removeItem(index)
            else:
                combo.setItemData(index, None, _DISCOVERY_ROLE)
                combo.setItemData(index, None, Qt.ItemDataRole.ToolTipRole)

        self._discovered_hosts = {endpoint.host for endpoint in digitizers}
        by_host = {endpoint.host: endpoint for endpoint in digitizers}
        for endpoint in digitizers:
            index = combo.findText(endpoint.host, Qt.MatchFlag.MatchFixedString)
            if index < 0:
                combo.addItem(endpoint.host)
                index = combo.count() - 1
            combo.setItemData(index, endpoint, _DISCOVERY_ROLE)
            combo.setItemData(
                index,
                f"{endpoint.description} ({endpoint.source}, port {endpoint.port})",
                Qt.ItemDataRole.ToolTipRole,
            )

        if preserve_current:
            combo.setEditText(current_text)
            return
        selected = by_host.get(current_text)
        if selected is None and digitizers:
            selected = digitizers[0]
        if selected is not None:
            index = combo.findText(selected.host, Qt.MatchFlag.MatchFixedString)
            combo.setCurrentIndex(index)
            self._on_address_selected(index)

    def _load_settings(self) -> None:
        settings = QSettings()
        recent = cast(list[str], settings.value(_KEY_RECENT_IPS, [], type=list))
        self._recent_ips = set(recent)
        self._ui.comboIp.addItems(recent)
        if recent:
            self._ui.comboIp.setCurrentIndex(0)

        saved_backend = settings.value(_KEY_LAST_BACKEND, _DEFAULT_BACKEND)
        backend = saved_backend if isinstance(saved_backend, str) else _DEFAULT_BACKEND
        backend_index = (
            _BACKENDS.index(backend) if backend in _BACKENDS else _BACKENDS.index(_DEFAULT_BACKEND)
        )
        self._ui.comboBackend.blockSignals(True)
        self._ui.comboBackend.setCurrentIndex(backend_index)
        self._ui.comboBackend.blockSignals(False)

        default_port = _BACKEND_DEFAULT_PORTS[_BACKENDS[backend_index]]
        saved_port = settings.value(_KEY_LAST_PORT, default_port)
        saved_channels = settings.value(_KEY_LAST_CHANNELS, _DEFAULT_CHANNELS)
        self._ui.spinPort.setValue(
            int(saved_port) if isinstance(saved_port, (int, str)) else default_port
        )
        self._ui.spinChannels.setValue(
            int(saved_channels) if isinstance(saved_channels, (int, str)) else _DEFAULT_CHANNELS
        )

    def _apply_initial_values(
        self,
        *,
        ip: str | None,
        port: int | None,
        backend: str | None,
        channels: int | None,
    ) -> None:
        """Apply YAML/CLI defaults after persistent QSettings are restored."""
        if backend in _BACKENDS:
            self._ui.comboBackend.blockSignals(True)
            self._ui.comboBackend.setCurrentIndex(_BACKENDS.index(backend))
            self._ui.comboBackend.blockSignals(False)
        if ip is not None:
            self._ui.comboIp.setEditText(ip)
        if port is not None:
            self._ui.spinPort.setValue(port)
        if channels is not None:
            self._ui.spinChannels.setValue(channels)

    def _save_settings(self) -> None:
        settings = QSettings()
        recent = cast(list[str], settings.value(_KEY_RECENT_IPS, [], type=list))
        ip = self.ip
        if ip in recent:
            recent.remove(ip)
        recent.insert(0, ip)
        settings.setValue(_KEY_RECENT_IPS, recent[:_MAX_RECENT_IPS])
        settings.setValue(_KEY_LAST_BACKEND, self.backend)
        settings.setValue(_KEY_LAST_PORT, self.port)
        settings.setValue(_KEY_LAST_CHANNELS, self.channels)

    def _on_accept(self) -> None:
        if not self.ip:
            self._ui.comboIp.lineEdit().setFocus()  # type: ignore[union-attr]
            return
        self._save_settings()
        self.accept()
