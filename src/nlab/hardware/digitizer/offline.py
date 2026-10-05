"""In-memory device surfaces used by the GUI's offline analysis mode.

These objects exist only to construct the normal views that deterministic
session replay renders into.  They never open a transport, and the application
keeps every hardware control interlocked for their complete lifetime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
from nlab_modbus.core.base_modbus_device import BaseModbusDevice
from nlab_modbus.core.register_specs import RegisterSpec, RegisterType

from nlab.hardware.digitizer.backends.base import DigitizerBackend, IDSBackend
from nlab.hardware.digitizer.digitizer import Digitizer
from nlab.hardware.digitizer.dma import IIOMcaDmaStreamer

if TYPE_CHECKING:
    from nlab.hardware.digitizer.diagnostics import GlobalDiagnosticReading


class _OfflineDigitizerBackend:
    """Stateful duck-typed Scope/MCA backend with no external side effects."""

    def __init__(self, channel: int) -> None:
        self.channel = channel
        self._values: dict[str, object] = {}

    def __getattr__(self, name: str) -> Callable[..., object]:
        if name.startswith("set_"):
            key = name.removeprefix("set_")

            def setter(value: object) -> None:
                self._values[key] = value

            return setter
        if name.startswith("get_"):
            key = name.removeprefix("get_")
            return lambda: self._values.get(key, 0)
        raise AttributeError(name)

    def close(self) -> None:
        pass

    def mca_hardware_present(self) -> bool:
        return True

    def mca_dma_hardware_present(self) -> bool:
        return True

    def scope_dma_active(self) -> bool:
        return False

    def mca_dma_active(self) -> bool:
        return False

    def get_measurement_in_progress(self) -> bool:
        return False

    def get_mem_frame_size(self) -> int:
        return 4096

    def get_mem_amount(self) -> int:
        return 2

    def get_mem_frame_sizes(self) -> np.ndarray:
        configured = self._values.get("dpp_frame_samples", 256)
        samples = (configured if isinstance(configured, int) else 256) // 2
        return np.full(2, max(1, samples), dtype=np.uint32)

    def get_debug_signal_selectors(self) -> tuple[int, ...]:
        return tuple(range(16))

    def get_lp_coeffs_size(self) -> int:
        return 64

    def get_lp_coeffs(self) -> np.ndarray:
        return np.zeros(self.get_lp_coeffs_size(), dtype=np.int32)

    def set_lp_coeffs(self, coeffs: list[int]) -> None:
        self._values["lp_coeffs"] = tuple(coeffs)

    def read_frame(self) -> np.ndarray:
        configured = self._values.get("frame_samples", 1024)
        samples = configured if isinstance(configured, int) else 1024
        return np.zeros(max(1, samples), dtype=np.int16)

    def read_waveform_banks(self) -> tuple[np.ndarray, np.ndarray]:
        sizes = self.get_mem_frame_sizes()
        size = int(sizes[0]) if len(sizes) else 1
        empty = np.zeros(max(1, size), dtype=np.int16)
        return empty.copy(), empty

    def get_histogram_size(self) -> int:
        return 16_384

    def read_histogram(self) -> np.ndarray:
        return np.zeros(self.get_histogram_size(), dtype=np.uint32)

    def clear_histogram(self) -> None:
        pass

    def create_isolated_scope_backend(self) -> _OfflineDigitizerBackend:
        return _OfflineDigitizerBackend(self.channel)

    def get_viewer_frame_samples_limit(self) -> int | None:
        return None

    def edge_det_coeff_is_hardware_backed(self) -> bool:
        return False

    def set_temperature_correction_from_worker(
        self,
        coefficient: float,
        offset: int,
    ) -> None:
        self._values["temp_coeff"] = coefficient
        self._values["temp_offset"] = offset


class _OfflineIDSBackend:
    """In-memory IDS surface used to host PSU replay widgets."""

    def __init__(self) -> None:
        self._values: dict[str, object] = {"sipm_overload": 0}

    def __getattr__(self, name: str) -> Callable[..., object]:
        if name.startswith("set_"):
            key = name.removeprefix("set_")

            def setter(value: object) -> None:
                self._values[key] = value

            return setter
        if name.startswith("get_"):
            key = name.removeprefix("get_")
            return lambda: self._values.get(key, 0.0)
        raise AttributeError(name)

    def get_versions(self) -> list[int]:
        return []

    def get_sipm_overload(self) -> int:
        return 0

    def get_global_diagnostics(self) -> list[GlobalDiagnosticReading]:
        return []

    def close(self) -> None:
        pass


class _OfflineMcaDma:
    """Capability marker; replay never invokes a DMA transport method."""


def create_offline_digitizer(channel: int) -> Digitizer:
    """Create a transport-free channel with every replay target available."""
    backend = cast(DigitizerBackend, _OfflineDigitizerBackend(channel))
    ids_backend = cast(IDSBackend, _OfflineIDSBackend())
    dma = cast(IIOMcaDmaStreamer, _OfflineMcaDma())
    return Digitizer(backend, ids_backend, mca_dma=dma)


def _register_spec(
    register_type: RegisterType,
    address: int,
    value: object,
) -> RegisterSpec:
    is_float = isinstance(value, float)
    is_boolean = isinstance(value, bool)
    if is_boolean:
        minimum, maximum, scale = 0, 1, 1.0
    elif is_float:
        minimum, maximum, scale = -2_000_000_000, 2_000_000_000, 0.001
    else:
        minimum, maximum, scale = -2_000_000_000, 2_000_000_000, 1.0
    return RegisterSpec(
        reg_type=register_type,
        address=address,
        dtype="int32",
        min=minimum,
        max=maximum,
        scale=scale,
    )


class OfflineExternalDevice:
    """Read-only Modbus-device facade synthesized from recorded fields."""

    def __init__(
        self,
        configuration_id: str,
        holding_values: Mapping[str, int | float],
        telemetry_values: Mapping[str, object],
        schema: object = None,
    ) -> None:
        type_name, separator, raw_id = configuration_id.rpartition(":")
        if not separator:
            type_name, raw_id = configuration_id, "replay"
        try:
            device_id: int | str = int(raw_id)
        except ValueError:
            device_id = raw_id
        self.device_type = SimpleNamespace(name=type_name)
        self.device_id = device_id
        self._holding_values = dict(holding_values)

        register_map = self._register_map_from_schema(schema)
        if register_map:
            schema_map = schema if isinstance(schema, Mapping) else {}
            input_addresses = [
                spec.address
                for spec in register_map.values()
                if spec.reg_type == RegisterType.INPUT
            ]
            default_start = min(input_addresses, default=0)
            default_stop = max(input_addresses, default=-1) + 1
            saved_start = schema_map.get("readout_start")
            saved_stop = schema_map.get("readout_stop")
            self.READOUT_START = (
                int(saved_start) if isinstance(saved_start, int) else default_start
            )
            self.READOUT_STOP = (
                int(saved_stop) if isinstance(saved_stop, int) else default_stop
            )
        else:
            for address, (name, value) in enumerate(sorted(telemetry_values.items())):
                register_map[name] = _register_spec(RegisterType.INPUT, address, value)
            self.READOUT_START = 0
            self.READOUT_STOP = len(register_map)
            holding_address = max(1_000, self.READOUT_STOP)
            for offset, (name, value) in enumerate(sorted(self._holding_values.items())):
                register_map.setdefault(
                    name,
                    _register_spec(RegisterType.HOLDING, holding_address + offset, value),
                )
        self.REGISTER_MAP = register_map

    @staticmethod
    def _register_map_from_schema(schema: object) -> dict[str, RegisterSpec]:
        if not isinstance(schema, Mapping):
            return {}
        documents = schema.get("registers")
        if not isinstance(documents, list):
            return {}
        result: dict[str, RegisterSpec] = {}
        for document in documents:
            if not isinstance(document, Mapping) or not isinstance(
                document.get("name"), str
            ):
                continue
            try:
                result[str(document["name"])] = RegisterSpec(
                    reg_type=RegisterType(str(document["register_type"])),
                    address=int(document["address"]),
                    dtype=str(document["dtype"]),
                    min=int(document["min"]),
                    max=int(document["max"]),
                    count=int(document.get("count", 1)),
                    scale=float(document.get("scale", 1.0)),
                    unit=(
                        str(document["unit"])
                        if document.get("unit") is not None
                        else None
                    ),
                    password_protected=bool(
                        document.get("password_protected", False)
                    ),
                    description=str(document.get("description", "")),
                )
            except (KeyError, TypeError, ValueError):
                continue
        return result

    def connection_info(self) -> str:
        return "recorded session (offline)"

    def get_all_holding_registers(self) -> dict[str, int | float]:
        return dict(self._holding_values)

    def read_snapshot(self) -> dict[str, int | float]:
        return {}

    def write(self, _name: str, _value: int | float) -> None:
        raise RuntimeError("recorded external modules are read-only in offline mode")


def create_offline_external_device(
    configuration_id: str,
    holding_values: Mapping[str, int | float],
    telemetry_values: Mapping[str, object],
    schema: object = None,
) -> BaseModbusDevice:
    """Return a typed facade for :class:`ExternalDeviceController`."""
    return cast(
        BaseModbusDevice,
        OfflineExternalDevice(
            configuration_id,
            holding_values,
            telemetry_values,
            schema,
        ),
    )
