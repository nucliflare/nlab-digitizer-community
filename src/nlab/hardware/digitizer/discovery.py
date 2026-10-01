"""Read-only discovery of digitizers exposed by remote ``iiod``.

libiio's IP scan uses the platform's DNS-SD/mDNS support (Avahi on Linux).
The USB gadget network is deliberately checked separately because multicast
discovery is not guaranteed to cross that point-to-point interface.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass

import iio

log = logging.getLogger(__name__)

USB_GADGET_HOST = "192.168.3.1"
DEFAULT_IIOD_PORT = 30431
_USB_PROBE_TIMEOUT_SECONDS = 0.5


@dataclass(frozen=True)
class DiscoveredDigitizer:
    """One reachable or advertised network IIO endpoint."""

    host: str
    port: int
    description: str
    source: str


@dataclass(frozen=True)
class DiscoveryResult:
    """Discovery output, including a non-fatal network-scan failure."""

    digitizers: tuple[DiscoveredDigitizer, ...]
    network_error: str | None = None


ScanContexts = Callable[[], Mapping[str, str]]
ProbeEndpoint = Callable[[str, int, float], bool]


def _network_endpoint(uri: str, description: str) -> DiscoveredDigitizer | None:
    """Convert a libiio ``ip:host[:port]`` URI to a GUI endpoint."""
    if not uri.startswith("ip:"):
        return None

    address = uri.removeprefix("ip:").removeprefix("//").strip()
    if not address:
        return None

    host = address
    port = DEFAULT_IIOD_PORT
    if address.startswith("["):
        closing_bracket = address.find("]")
        if closing_bracket < 0:
            return None
        host = address[1:closing_bracket]
        suffix = address[closing_bracket + 1 :]
        if suffix:
            if not suffix.startswith(":") or not suffix[1:].isdigit():
                return None
            port = int(suffix[1:])
    elif address.count(":") == 1:
        possible_host, possible_port = address.rsplit(":", 1)
        if possible_port.isdigit():
            host = possible_host
            port = int(possible_port)

    if not host or not 1 <= port <= 65535:
        return None
    return DiscoveredDigitizer(
        host=host,
        port=port,
        description=description or "IIO digitizer",
        source="mDNS/Avahi",
    )


def _tcp_endpoint_available(host: str, port: int, timeout: float) -> bool:
    """Return whether a TCP endpoint accepts a connection within *timeout*."""
    try:
        with closing(socket.create_connection((host, port), timeout=timeout)):
            return True
    except OSError:
        return False


def discover_iio_digitizers(
    *,
    scan_contexts: ScanContexts | None = None,
    probe_endpoint: ProbeEndpoint | None = None,
) -> DiscoveryResult:
    """Find advertised IIO boards and probe the fixed USB-network endpoint.

    The function creates no IIO context and performs no device or acquisition
    writes.  The explicit USB check is only a bounded TCP connect to iiod.
    Callers should run this function outside the GUI thread because libiio's
    network scan waits briefly for multicast responses.
    """
    scan = iio.scan_contexts if scan_contexts is None else scan_contexts
    probe = _tcp_endpoint_available if probe_endpoint is None else probe_endpoint

    found: dict[tuple[str, int], DiscoveredDigitizer] = {}
    network_error: str | None = None
    try:
        contexts = scan()
    except Exception as exc:
        network_error = str(exc) or type(exc).__name__
        log.warning("IIO mDNS/Avahi discovery failed: %s", network_error)
    else:
        for uri, description in contexts.items():
            endpoint = _network_endpoint(str(uri), str(description))
            if endpoint is not None:
                found[(endpoint.host, endpoint.port)] = endpoint

    if probe(USB_GADGET_HOST, DEFAULT_IIOD_PORT, _USB_PROBE_TIMEOUT_SECONDS):
        key = (USB_GADGET_HOST, DEFAULT_IIOD_PORT)
        existing = found.get(key)
        found[key] = DiscoveredDigitizer(
            host=USB_GADGET_HOST,
            port=DEFAULT_IIOD_PORT,
            description=(existing.description if existing is not None else "Digitizer USB network"),
            source=("mDNS/Avahi + USB" if existing is not None else "USB"),
        )

    digitizers = tuple(
        sorted(
            found.values(),
            key=lambda item: (item.source not in {"USB", "mDNS/Avahi + USB"}, item.host, item.port),
        )
    )
    return DiscoveryResult(digitizers=digitizers, network_error=network_error)
