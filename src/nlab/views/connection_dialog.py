from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QWidget

from nlab.ui.ui_connection_dialog import Ui_ConnectionDialog
from nlab.utils.windows_icon import apply_taskbar_icon

_MAX_RECENT_IPS = 10
_DEFAULT_CHANNELS = 2

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

    IP history is persisted via QSettings so the last-used addresses are
    available in the drop-down on the next run.  Auto-discovery entries
    can be added later by populating the combo from a background scanner.
    """

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
        self._ui.setupUi(self)
        self._ui.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setText("Connect")
        self._ui.buttonBox.accepted.connect(self._on_accept)
        self._ui.buttonBox.rejected.connect(self.reject)
        self._ui.comboBackend.currentIndexChanged.connect(self._on_backend_changed)
        apply_taskbar_icon(self)
        self._load_settings()
        self._apply_initial_values(ip=ip, port=port, backend=backend, channels=channels)

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

    def _load_settings(self) -> None:
        settings = QSettings()
        recent: list[str] = settings.value(_KEY_RECENT_IPS, [], type=list)  # type: ignore[assignment]
        self._ui.comboIp.addItems(recent)
        if recent:
            self._ui.comboIp.setCurrentIndex(0)

        backend = settings.value(_KEY_LAST_BACKEND, _DEFAULT_BACKEND)  # type: ignore[assignment]
        backend_index = (
            _BACKENDS.index(backend) if backend in _BACKENDS else _BACKENDS.index(_DEFAULT_BACKEND)
        )
        self._ui.comboBackend.blockSignals(True)
        self._ui.comboBackend.setCurrentIndex(backend_index)
        self._ui.comboBackend.blockSignals(False)

        default_port = _BACKEND_DEFAULT_PORTS[_BACKENDS[backend_index]]
        self._ui.spinPort.setValue(int(settings.value(_KEY_LAST_PORT, default_port)))  # type: ignore[arg-type]
        self._ui.spinChannels.setValue(int(settings.value(_KEY_LAST_CHANNELS, _DEFAULT_CHANNELS)))  # type: ignore[arg-type]

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
        recent: list[str] = settings.value(_KEY_RECENT_IPS, [], type=list)  # type: ignore[assignment]
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
