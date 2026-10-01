from __future__ import annotations

from collections.abc import Mapping

from nlab.hardware.digitizer.discovery import (
    DEFAULT_IIOD_PORT,
    USB_GADGET_HOST,
    discover_iio_digitizers,
)


def test_discovery_combines_mdns_and_direct_usb_probe() -> None:
    probes: list[tuple[str, int, float]] = []

    def scan() -> Mapping[str, str]:
        return {
            "ip:board.local:31000": "Network digitizer",
            "ip:[fe80::1234]:30431": "IPv6 digitizer",
            "usb:1.2.3": "Native USB IIO device",
            "local:": "Local IIO context",
        }

    def probe(host: str, port: int, timeout: float) -> bool:
        probes.append((host, port, timeout))
        return True

    result = discover_iio_digitizers(scan_contexts=scan, probe_endpoint=probe)

    assert [(item.host, item.port, item.source) for item in result.digitizers] == [
        (USB_GADGET_HOST, DEFAULT_IIOD_PORT, "USB"),
        ("board.local", 31000, "mDNS/Avahi"),
        ("fe80::1234", DEFAULT_IIOD_PORT, "mDNS/Avahi"),
    ]
    assert probes == [(USB_GADGET_HOST, DEFAULT_IIOD_PORT, 0.5)]
    assert result.network_error is None


def test_discovery_still_checks_usb_when_mdns_scan_fails() -> None:
    def scan() -> Mapping[str, str]:
        raise OSError("DNS-SD unavailable")

    result = discover_iio_digitizers(
        scan_contexts=scan,
        probe_endpoint=lambda _host, _port, _timeout: True,
    )

    assert [item.host for item in result.digitizers] == [USB_GADGET_HOST]
    assert result.network_error == "DNS-SD unavailable"


def test_discovery_deduplicates_usb_endpoint_advertised_over_mdns() -> None:
    result = discover_iio_digitizers(
        scan_contexts=lambda: {f"ip:{USB_GADGET_HOST}:{DEFAULT_IIOD_PORT}": "Digitizer 4.0"},
        probe_endpoint=lambda _host, _port, _timeout: True,
    )

    assert len(result.digitizers) == 1
    assert result.digitizers[0].source == "mDNS/Avahi + USB"
    assert result.digitizers[0].description == "Digitizer 4.0"
