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
from nlab.hardware.digitizer.mca import MCA_DEBUG_SIGNAL_SELECTORS
from nlab.hardware.digitizer.mca_parameters import DEFAULT_PARAMETERS, validate_parameters

if TYPE_CHECKING:
    from nlab.hardware.digitizer.diagnostics import GlobalDiagnosticReading


_MCA_ALIASES = {
    "external_trigger_enable": "ext_trig_enable",
    "measurement_time_raw": "time_limit",
    "trigger_source": "trg_source",
    "trigger_level_raw": "dpp_trigger_level",
    "pulse_polarity": "pulse_polarity",
    "baseline_window": "bsln_window",
    "pileup_window_raw": "pileup_window",
    "energy_bin": "energy_bin",
    "pretrigger_samples": "dpp_pretrigger_samples",
    "frame_samples": "dpp_frame_samples",
    "cfd_enable": "cfd_enable",
    "cfd_factor_raw": "cfd_factor",
    "cfd_delay_samples": "cfd_delay",
    "cfd_window_low_samples": "cfd_time_window_low",
    "cfd_window_high_samples": "cfd_time_window_high",
    "charge_comparison_enable": "cc_enable",
    "charge_comparison_samples": "cc_time",
    "crrc2_cdelay": "crrc2_Cdelay",
    "crrc2_fdelay_samples": "crrc2_Fdelay",
    "crrc2_pzc_raw": "crrc2_pzc_coeff",
    "trapezoid_enable": "trapez_enable",
    "trapezoid_r_samples": "trapez_R",
    "trapezoid_m_samples": "trapez_M",
    "trapezoid_beta_raw": "trapez_T",
    "trapezoid_start_offset_samples": "trapez_E",
    "trapezoid_flat_top_window": "trapez_FT",
    "psd_zero_crossing_enable": "psd_zc_enable",
    "psd_zero_crossing_mode": "psd_zc_mode",
    "psd_window_low_samples": "psd_zc_time_window_low",
    "psd_window_high_samples": "psd_zc_time_window_high",
}


class _OfflineDigitizerBackend:
    """Stateful duck-typed Scope/MCA backend with no external side effects."""

    def __init__(self, channel: int) -> None:
        self.channel = channel
        self._values: dict[str, object] = {}
        self.configure_mca_parameters(dict(DEFAULT_PARAMETERS))

    def get_scope_trigger_modes(self) -> tuple[int, ...]:
        return tuple(range(5))

    def get_psd_zc_modes(self) -> tuple[int, ...]:
        return (0, 1, 2)

    def supports_dac(self) -> bool:
        return True

    def acquisition_profile(self) -> dict[str, object]:
        return dict(self._values.get("acquisition_profile", {}))

    def apply_acquisition_profile(self, values: dict[str, object]) -> None:
        from .acquisition_profile import AcquisitionProfile

        self._values["acquisition_profile"] = AcquisitionProfile.from_mapping(values).to_dict()

    def mca_parameter_contract(self) -> dict[str, object]:
        return {"supported": True, "parameter_abi_version": 2, "parameter_units": "adc_samples"}

    def prepare_mca_configuration(self, values: dict[str, int]) -> None:
        validate_parameters(values, complete=True)

    def mca_parameter_snapshot(self) -> dict[str, int]:
        result = {}
        for field, alias in _MCA_ALIASES.items():
            value = cast(int | float, self._values.get(alias, DEFAULT_PARAMETERS[field]))
            if field == "cfd_factor_raw":
                value = round(float(value) * 32768)
            elif field == "measurement_time_raw":
                value = round(float(value) / 0.134217728)
            result[field] = int(value)
        return result

    def configure_mca_parameters(self, values: dict[str, int]) -> None:
        validate_parameters(values)
        for field, value in values.items():
            actual = (
                value / 32768
                if field == "cfd_factor_raw"
                else (value * 0.134217728 if field == "measurement_time_raw" else value)
            )
            if field == "trapezoid_r_samples":
                self.set_trapez_R(value)
            else:
                self._values[_MCA_ALIASES[field]] = actual

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
        samples = (configured if isinstance(configured, int) else 256) // 4
        return np.full(2, max(1, samples), dtype=np.uint32)

    def get_debug_signal_selectors(self) -> tuple[int, ...]:
        return MCA_DEBUG_SIGNAL_SELECTORS

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

    def supports_dc_blocker(self) -> bool:
        return True

    def supports_crrc2_reciprocal_raw(self) -> bool:
        return True

    def supports_trapezoid_reciprocal_raw(self) -> bool:
        return True

    def get_trapezoid_reciprocal_raw(self) -> int:
        return cast(int, self._values.get("trapezoid_reciprocal_raw", 0))

    def set_trapez_R(self, val: int) -> None:  # noqa: N802 - Existing backend API.
        # Emulate only the documented register dependency, not the DSP output.
        if type(val) is not int or not 8 <= val <= 2044 or val % 4:
            raise ValueError("trapezoid R must be 8..2044 ADC samples in steps of 4")
        self._values["trapez_R"] = val
        self._values["trapezoid_reciprocal_raw"] = 32768 // (val // 4)

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
            self.READOUT_START = int(saved_start) if isinstance(saved_start, int) else default_start
            self.READOUT_STOP = int(saved_stop) if isinstance(saved_stop, int) else default_stop
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
            if not isinstance(document, Mapping) or not isinstance(document.get("name"), str):
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
                    unit=(str(document["unit"]) if document.get("unit") is not None else None),
                    password_protected=bool(document.get("password_protected", False)),
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
