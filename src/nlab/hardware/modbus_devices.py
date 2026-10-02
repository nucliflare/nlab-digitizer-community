from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

from nlab_modbus.core.base_modbus_device import BaseModbusDevice
from nlab_modbus.core.enums import DeviceType
from nlab_modbus.services.manager import DeviceManager

log = logging.getLogger(__name__)

# ser2net TCP ports the digitizer bridges its RS-485 Modbus bus to (one bus,
# multiple daisy-chained devices distinguished by Modbus device_id).
REMOTE_PORTS: tuple[int, ...] = (5001, 5002)


@dataclass(frozen=True)
class ExternalDeviceScan:
    """A fresh manager and the devices it found during one complete scan.

    The manager owns every returned device connection.  Until the GUI adopts
    the result, the caller is responsible for calling ``manager.close()`` if
    the result is discarded.
    """

    manager: ExternalDevices
    devices: tuple[BaseModbusDevice, ...]


class ExternalDevices:
    """Discovers and owns the digitizer's onboard Modbus instruments.

    The SiPM bias board, Geiger-Mueller probe, and PMT HV supply share the
    digitizer's RS-485 bus, bridged to TCP via ser2net — same host as the
    gRPC digitizer connection, different port(s).  Wraps nlab_modbus's
    DeviceManager so the rest of the app only deals with discovered device
    instances and never imports nlab_modbus or touches connection details
    directly.
    """

    def __init__(self) -> None:
        self._manager = DeviceManager()

    @classmethod
    def scan_new(
        cls,
        host: str,
        ports: tuple[int, ...] = REMOTE_PORTS,
    ) -> ExternalDeviceScan:
        """Scan into a new owner so an active device set remains untouched.

        Runtime refresh uses this operation from a background thread after the
        GUI has stopped old polling and closed its ser2net client. Opening a
        second connection while polling can steal a single-client bridge.
        """
        owner = cls()
        try:
            devices = tuple(owner.discover(host, ports))
        except BaseException:
            owner.close()
            raise
        return ExternalDeviceScan(manager=owner, devices=devices)

    def discover(self, host: str, ports: tuple[int, ...] = REMOTE_PORTS) -> list[BaseModbusDevice]:
        """Scan the digitizer host for Modbus devices and connect to all found.

        Uses the manager's remote scan utility per candidate ser2net port;
        a port with nothing attached simply yields no devices. Returns every
        device discovered so far (cumulative across calls).
        """
        for port in ports:
            try:
                self._manager.scan_remote(host, port)
            except Exception:
                log.exception("Modbus scan failed on %s:%d", host, port)
        log.info(
            "Discovered %d external Modbus device(s) on %s",
            len(self._manager.all_devices),
            host,
        )
        return cast(list[BaseModbusDevice], self._manager.all_devices)

    def by_type(self, device_type: DeviceType) -> list[BaseModbusDevice]:
        return cast(list[BaseModbusDevice], self._manager.by_type(device_type))

    def close(self) -> None:
        self._manager.close_all()
