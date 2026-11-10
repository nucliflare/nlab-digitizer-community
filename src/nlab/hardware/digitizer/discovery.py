"""Read-only discovery of digitizers exposed by remote ``iiod``.

libiio's IP scan uses the platform's DNS-SD/mDNS support (Avahi on Linux).
The USB gadget network is deliberately checked separately because multicast
discovery is not guaranteed to cross that point-to-point interface.
"""

from __future__ import annotations

import logging
import re
import socket
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass, replace
from ipaddress import ip_address

import iio

log = logging.getLogger(__name__)

USB_GADGET_HOST = "192.168.3.1"
DEFAULT_IIOD_PORT = 30431
_USB_PROBE_TIMEOUT_SECONDS = 0.5
_MDNS_HOSTNAME = re.compile(r"\b[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.local\.?\b", re.I)


@dataclass(frozen=True)
class DiscoveredDigitizer:
    """One reachable or advertised network IIO endpoint."""

    host: str
    port: int
    description: str
    source: str
    hostname: str | None = None

    @property
    def display_name(self) -> str:
        return f"{self.hostname} ({self.host})" if self.hostname else self.host


@dataclass(frozen=True)
class DiscoveryResult:
    """Discovery output, including a non-fatal network-scan failure."""

    digitizers: tuple[DiscoveredDigitizer, ...]
    network_error: str | None = None


ScanContexts = Callable[[], Mapping[str, str]]
ProbeEndpoint = Callable[[str, int, float], bool]
ResolveHost = Callable[[str], str | None]


def _is_ip_address(value: str) -> bool:
    try:
        ip_address(value)
    except ValueError:
        return False
    return True


def _resolve_network_address(host: str) -> str | None:
    """Resolve advertised names on the discovery worker, outside the GUI thread."""
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return None
    # Prefer the familiar IPv4 display when both address families are available.
    addresses.sort(key=lambda entry: entry[0] != socket.AF_INET)
    return str(addresses[0][4][0]) if addresses else None


def _network_endpoint(
    uri: str, description: str, resolve_host: ResolveHost,
) -> DiscoveredDigitizer | None:
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
    match = _MDNS_HOSTNAME.search(description)
    hostname = match.group().rstrip(".") if match else None
    if not _is_ip_address(host):
        hostname = host.rstrip(".")
        # Some libiio scans put the numeric address in the description instead
        # of the URI. Prefer that pairing over another DNS lookup.
        resolved_address = next((word for token in description.split()
                                 if _is_ip_address(word := token.strip("[](),;"))), None)
        resolved_address = resolved_address or resolve_host(host)
        if resolved_address is not None and _is_ip_address(resolved_address):
            host = resolved_address
        else:
            hostname = None  # No address established: retain the usable name alone.
    return DiscoveredDigitizer(
        host=host,
        port=port,
        description=description or "IIO digitizer",
        source="mDNS/Avahi",
        hostname=hostname,
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
    resolve_host: ResolveHost | None = None,
) -> DiscoveryResult:
    """Find advertised IIO boards and probe the fixed USB-network endpoint.

    The function creates no IIO context and performs no device or acquisition
    writes.  The explicit USB check is only a bounded TCP connect to iiod.
    Callers should run this function outside the GUI thread because libiio's
    network scan waits briefly for multicast responses.
    """
    scan = iio.scan_contexts if scan_contexts is None else scan_contexts
    probe = _tcp_endpoint_available if probe_endpoint is None else probe_endpoint
    resolve = _resolve_network_address if resolve_host is None else resolve_host

    found: dict[tuple[str, int], DiscoveredDigitizer] = {}
    network_error: str | None = None
    try:
        contexts = scan()
    except Exception as exc:
        network_error = str(exc) or type(exc).__name__
        log.warning("IIO mDNS/Avahi discovery failed: %s", network_error)
    else:
        for uri, description in contexts.items():
            endpoint = _network_endpoint(str(uri), str(description), resolve)
            if endpoint is not None:
                previous = found.get((endpoint.host, endpoint.port))
                if previous is not None and previous.hostname and endpoint.hostname is None:
                    endpoint = replace(endpoint, hostname=previous.hostname)
                found[(endpoint.host, endpoint.port)] = endpoint

    if probe(USB_GADGET_HOST, DEFAULT_IIOD_PORT, _USB_PROBE_TIMEOUT_SECONDS):
        key = (USB_GADGET_HOST, DEFAULT_IIOD_PORT)
        existing = found.get(key)
        found[key] = (
            replace(existing, source="mDNS/Avahi + USB") if existing is not None
            else DiscoveredDigitizer(
                host=USB_GADGET_HOST, port=DEFAULT_IIOD_PORT,
                description="Digitizer USB network", source="USB",
            )
        )

    digitizers = tuple(
        sorted(
            found.values(),
            key=lambda item: (item.source not in {"USB", "mDNS/Avahi + USB"}, item.host, item.port),
        )
    )
    return DiscoveryResult(digitizers=digitizers, network_error=network_error)
