"""About uses connection snapshots, including during acquisition; no hardware I/O."""

from types import SimpleNamespace
from unittest.mock import Mock

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QMainWindow

from nlab.controllers.main_window_controller import (
    DigitizerChannel,
    DigitizerEndpoint,
    MainWindowController,
)
from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.device_information import (
    DigitizerInformation,
    InformationSection,
    collect_device_information,
)
from nlab.hardware.digitizer.digitizer import Digitizer
from nlab.views import about_dialog


class _ReadOnlyAttribute:
    def __init__(self, value):
        self._value = value
        self.reads = 0

    @property
    def value(self):
        self.reads += 1
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


def _snapshot(uri="ip:board-a:30431", channel=0):
    return DigitizerInformation(uri, channel, (
        InformationSection("Connection and firmware", (
            ("Firmware / image reported by target", "Unknown (not reported by target)"),
        )),
        InformationSection("Reported board / context attributes", (("hw_serial", "BOARD-123"),)),
    ), (
        InformationSection("Scope", (("ip_version", "124"),)),
        InformationSection("Pulse processor", (("hw_version", "101"), ("sw_version", "103023"))),
        InformationSection("List mode", (("frame_records", "1024"), ("frame_bytes", "16384"))),
    ))


def test_identity_collection_is_read_only_and_distinguishes_image_from_pp_registers():
    pp = SimpleNamespace(name="vdpp_pulse_processor", id="iio:device5", attrs={
        "hw_version": _ReadOnlyAttribute("101"),
        "sw_version": _ReadOnlyAttribute("103023"),
        "id_number": _ReadOnlyAttribute("42"),
        "parameter_abi_version": _ReadOnlyAttribute("2"),
        "parameter_units": _ReadOnlyAttribute("adc_samples"),
        "enable": _ReadOnlyAttribute(AssertionError("Acquisition state must not be polled")),
    })
    context = SimpleNamespace(attrs={"hw_model": "Digitizer 4.0", "local,kernel": "6.6.0"},
                              name="network", description="nucliflare-board.local",
                              version=(0, 26, "abc123"), devices=[pp])
    info = collect_device_information(context, uri="ip:board:30431", channel=0,
                                      devices={"Pulse processor": pp})
    connection = dict(info.board_sections[0].rows)
    assert connection["Firmware / image reported by target"].startswith("Unknown")
    assert connection["IIO backend version"] == "0.26 (abc123)"
    assert dict(info.board_sections[1].rows)["local,kernel"] == "6.6.0"
    pulse = next(section for section in info.channel_sections if section.title == "Pulse processor")
    assert dict(pulse.rows)["sw_version"] == "103023"
    assert dict(pulse.rows)["id_number"] == "42"
    assert pp.attrs["enable"].reads == 0


def test_reported_and_user_supplied_images_stay_distinct_and_partial_reads_survive():
    scope = SimpleNamespace(name="vdpp_scope", id="iio:device4", attrs={
        "ip_version": _ReadOnlyAttribute(OSError("Disconnected while reading identity")),
    })
    context = SimpleNamespace(attrs={"firmware_version": "image-build-123"},
                              name="network", description="Board", version=(0, 26, ""),
                              devices=[scope])
    info = collect_device_information(context, uri="ip:board:30431", channel=1,
                                      devices={"Scope": scope}, configured_image="user-build-999")
    connection = dict(info.board_sections[0].rows)
    assert connection["Firmware / image reported by target"] == "firmware_version: image-build-123"
    assert dict(info.channel_sections[0].rows)["Image identity (user supplied)"] == "user-build-999"
    assert "Unavailable:" in dict(info.channel_sections[1].rows)["ip_version"]
    assert dict(info.channel_sections[2].rows)["Availability"] == "Not present"


def test_snapshot_includes_selected_decoder_and_its_actual_qualification():
    context = SimpleNamespace(attrs={}, name="network", description="Board",
                              version=(0, 26, ""), devices=[])
    info = collect_device_information(context, uri="ip:board:30431", channel=0, devices={},
                                      producer_identity={
                                          "qualified_record_schema": "vdpp-zc-calc-q2.14-v1",
                                          "decoder_selection": "application default; unverified",
                                          "firmware_id": "unknown",
                                      })
    decoder = dict(info.channel_sections[0].rows)
    assert decoder["Decoder"] == "vdpp-zc-calc-q2.14-v1"
    assert decoder["Selection / qualification"] == "application default; unverified"
    assert decoder["Decoder image identity"] == "unknown"


def test_about_getters_never_touch_iio_contexts_or_active_readers():
    class NoHardwareAccess:
        def __getattribute__(self, name):
            raise AssertionError(f"Unexpected live access: {name}")
    backend = object.__new__(IIODigitizerBackend)
    backend._device_information = _snapshot()
    backend._ctx = NoHardwareAccess()
    backend._mca_dma_reader = NoHardwareAccess()
    device = object.__new__(Digitizer)
    device._backend = backend
    assert device.get_device_information() is backend._device_information
    backend._device_information = None
    assert device.get_device_information() is None


def test_controller_maps_multiple_boards_and_unknown_cache_without_polling():
    first, second = _snapshot(channel=0), _snapshot(channel=1)
    controller = object.__new__(MainWindowController)
    controller._offline = False
    controller._devices = [SimpleNamespace(get_device_information=lambda: first),
                           SimpleNamespace(get_device_information=lambda: second),
                           SimpleNamespace(get_device_information=lambda: None)]
    controller._device_channels = [
        DigitizerChannel(0, 0), DigitizerChannel(0, 1), DigitizerChannel(1, 0),
    ]
    controller._endpoints = [DigitizerEndpoint("board-a", 30431, 2),
                             DigitizerEndpoint("board-b", 30431, 1)]
    entries = controller.digitizer_information()
    assert [label for label, _info in entries] == ["D1:CH0", "D1:CH1", "D2:CH0"]
    assert entries[0][1] is first and entries[1][1] is second
    assert entries[2][1].uri == "ip:board-b:30431"
    assert "Unknown" in entries[2][1].board_sections[0].rows[0][1]
    controller._offline = True
    forbidden = Mock(side_effect=AssertionError("Offline must not collect live metadata"))
    controller._devices = [SimpleNamespace(get_device_information=forbidden)]
    assert controller.digitizer_information() == ()
    forbidden.assert_not_called()


class _NoNetworkChecker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)

    def check(self):
        raise AssertionError("About must not check updates automatically")

    def cancel(self):
        pass


def test_about_groups_channels_per_board_and_copies_exact_read_only_values(qtbot, monkeypatch):
    monkeypatch.setattr(about_dialog, "GitHubReleaseChecker", _NoNetworkChecker)
    info_a = _snapshot()
    info_b = _snapshot("ip:board-b:30431")
    dialog = about_dialog.AboutDialog("Application info", digitizers=(
        ("D1:CH0", info_a), ("D1:CH1", _snapshot(channel=1)), ("D2:CH0", info_b),
    ))
    qtbot.addWidget(dialog)
    assert dialog.device_tree.topLevelItemCount() == 2
    assert dialog.device_tree.topLevelItem(0).text(0) == "D1"
    assert dialog.device_tree.topLevelItem(0).text(1) == "ip:board-a:30431"
    assert dialog.device_tree.topLevelItem(0).childCount() == 4  # Board sections + channels.
    dialog.copy_button.click()
    copied = QApplication.clipboard().text()
    for value in ("D1:CH0", "D1:CH1", "D2:CH0", "BOARD-123", "103023", "16384", "Unknown"):
        assert value in copied
    assert copied.count("hw_serial: BOARD-123") == 2  # Once per board, not per channel.
    assert not dialog.device_tree.topLevelItem(0).flags() & about_dialog.Qt.ItemFlag.ItemIsEditable


def test_about_offline_shows_no_simulated_hardware_identity(qtbot, monkeypatch):
    monkeypatch.setattr(about_dialog, "GitHubReleaseChecker", _NoNetworkChecker)
    dialog = about_dialog.AboutDialog("Application info")
    qtbot.addWidget(dialog)
    assert "No connected digitizers" in dialog.device_tree.topLevelItem(0).text(0)
    assert not dialog.copy_button.isEnabled()


def test_help_about_receives_cached_connected_digitizer_information(qtbot, monkeypatch):
    from nlab import app as app_module

    cached = (("D1:CH0", _snapshot()),)
    window = QMainWindow()
    qtbot.addWidget(window)
    window._controller = SimpleNamespace(digitizer_information=lambda: cached)
    factory = Mock()
    monkeypatch.setattr(app_module, "AboutDialog", factory)
    app_module.MainAppWindow._on_about(window)
    factory.assert_called_once_with(app_module._ABOUT_TEXT, window, digitizers=cached)
