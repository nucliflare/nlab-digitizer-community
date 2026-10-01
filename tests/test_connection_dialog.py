from __future__ import annotations

import threading

import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from nlab.hardware.digitizer.discovery import DiscoveredDigitizer, DiscoveryResult
from nlab.views import connection_dialog as connection_dialog_module
from nlab.views.connection_dialog import ConnectionDialog


class _Settings:
    values: dict[str, object] = {}

    def value(
        self,
        key: str,
        default: object = None,
        *,
        type: type[object] | None = None,
    ) -> object:
        del type
        return self.values.get(key, default)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        self.values[key] = value


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _Settings.values = {}
    monkeypatch.setattr(connection_dialog_module, "QSettings", _Settings)


def _digitizer(host: str, *, source: str = "mDNS/Avahi") -> DiscoveredDigitizer:
    return DiscoveredDigitizer(host, 30431, f"Digitizer at {host}", source)


def _wait_for_scan(qtbot: QtBot, dialog: ConnectionDialog) -> None:
    qtbot.waitUntil(
        lambda: (
            dialog._ui.buttonRescan.isEnabled()
            and dialog._ui.labelDiscoveryStatus.text() != "Ready to scan for digitizers."
            and not dialog._ui.labelDiscoveryStatus.text().startswith("Scanning")
        ),
        timeout=3000,
    )


def test_startup_scan_selects_discovered_board_and_iio(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        connection_dialog_module,
        "discover_iio_digitizers",
        lambda: DiscoveryResult((_digitizer("192.168.3.1", source="USB"),)),
    )
    dialog = ConnectionDialog()
    qtbot.addWidget(dialog)
    dialog.show()

    _wait_for_scan(qtbot, dialog)

    assert dialog.ip == "192.168.3.1"
    assert dialog.backend == "iio"
    assert dialog.port == 30431
    assert dialog._ui.buttonRescan.text() == "Rescan"
    assert dialog._ui.labelDiscoveryStatus.text().startswith("Found 1 digitizer")


def test_explicit_address_is_preserved_while_discoveries_are_added(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        connection_dialog_module,
        "discover_iio_digitizers",
        lambda: DiscoveryResult((_digitizer("board.local"),)),
    )
    dialog = ConnectionDialog(ip="configured.local", backend="grpc", port=50050)
    qtbot.addWidget(dialog)
    dialog.show()

    _wait_for_scan(qtbot, dialog)

    assert dialog.ip == "configured.local"
    assert dialog.backend == "grpc"
    assert dialog.port == 50050
    assert dialog._ui.comboIp.findText("board.local", Qt.MatchFlag.MatchFixedString) >= 0


def test_rescan_replaces_stale_discovery_but_keeps_manual_address(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = iter(
        [
            DiscoveryResult((_digitizer("first.local"),)),
            DiscoveryResult((_digitizer("second.local"),)),
        ]
    )

    def discover() -> DiscoveryResult:
        return next(results)

    monkeypatch.setattr(connection_dialog_module, "discover_iio_digitizers", discover)
    dialog = ConnectionDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    _wait_for_scan(qtbot, dialog)

    dialog._ui.comboIp.setEditText("manual.local")
    line_edit = dialog._ui.comboIp.lineEdit()
    assert line_edit is not None
    line_edit.textEdited.emit("manual.local")
    dialog._ui.buttonRescan.click()
    _wait_for_scan(qtbot, dialog)

    assert dialog.ip == "manual.local"
    assert dialog._ui.comboIp.findText("first.local", Qt.MatchFlag.MatchFixedString) < 0
    assert dialog._ui.comboIp.findText("second.local", Qt.MatchFlag.MatchFixedString) >= 0


def test_rescan_button_is_disabled_while_scan_runs(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_scan = threading.Event()

    def discover() -> DiscoveryResult:
        assert release_scan.wait(timeout=3)
        return DiscoveryResult(())

    monkeypatch.setattr(connection_dialog_module, "discover_iio_digitizers", discover)
    dialog = ConnectionDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog._ui.buttonRescan.text() == "Scanning…")

    assert not dialog._ui.buttonRescan.isEnabled()
    release_scan.set()
    _wait_for_scan(qtbot, dialog)
