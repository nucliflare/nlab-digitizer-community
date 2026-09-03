from __future__ import annotations

import logging
from enum import IntEnum
from typing import TypedDict

from .backends.base import IDSBackend
from .diagnostics import GlobalDiagnosticReading
from .scope import ListSpec, ParameterSpec, RangeSpec

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HV parameter enum — values equal hw_def.json "hv" section register ids
# ---------------------------------------------------------------------------

class HVParam(IntEnum):
    # --- write-only (setters) ---
    SIPM_ENABLE         = 101  # uint32_t, 0–1
    SIPM_VOLTAGE        = 102  # float, 30–70 V
    HV_VOLTAGE          = 103  # float, 100–1250 V
    SIPM_COMPENS_CT     = 113  # float, -0.1–0.1
    SIPM_COMPENS_TREF   = 114  # float, 0–100
    SIPM_COMPENS_MODE   = 115  # temp_compens_mode_t, 0–3
    HV_COMPENS_CT       = 116  # float, -0.1–0.1
    HV_COMPENS_TREF     = 117  # float, 0–100
    HV_COMPENS_MODE     = 118  # temp_compens_mode_t, 0–3
    TEMP_DIGITAL_ENABLE = 119  # temp_digital_mode_t, 0–2
    # --- read-only (getters) ---
    SIPM_ADC_VOLTAGE    = 104  # float, 30–70 V
    SIPM_ADC_CURRENT    = 105  # float, 0–2 A
    SIPM_OVERLOAD       = 106  # uint32_t, 0–1
    SIPM_COMPENS_OUTPUT = 107  # float, 0–70 V
    HV_ADC_VOLTAGE      = 108  # float, 100–1250 V
    HV_COMPENS_OUTPUT   = 109  # float, 0–1250 V
    TEMP_ANALOG         = 110  # float, -40–400
    TEMP_DIGITAL_STATUS = 111  # uint32_t, 0–1
    TEMP_DIGITAL        = 112  # float, -40–400


HV_PARAMETER_SPECS: dict[HVParam, ParameterSpec] = {
    # --- setters (validated before write) ---
    # hw_def: uint32_t, 0–1
    HVParam.SIPM_ENABLE:         ListSpec(items=(0, 1),        default=0),
    # hw_def: float, 30–70, step 0.1
    HVParam.SIPM_VOLTAGE:        RangeSpec(min_val=30.0,  max_val=70.0,   step=0.1,  default=30.0),
    # hw_def: float, 100–1250, step 0.1
    HVParam.HV_VOLTAGE:          RangeSpec(min_val=100.0, max_val=1250.0, step=0.1,  default=100.0),
    # hw_def: float, -0.1–0.1, step 0.01
    HVParam.SIPM_COMPENS_CT:     RangeSpec(min_val=-0.1,  max_val=0.1,    step=0.01, default=0.0),
    HVParam.HV_COMPENS_CT:       RangeSpec(min_val=-0.1,  max_val=0.1,    step=0.01, default=0.0),
    # hw_def: float, 0–100, step 0.01
    HVParam.SIPM_COMPENS_TREF:   RangeSpec(min_val=0.0,   max_val=100.0,  step=0.01, default=20.0),
    HVParam.HV_COMPENS_TREF:     RangeSpec(min_val=0.0,   max_val=100.0,  step=0.01, default=20.0),
    # hw_def: temp_compens_mode_t, 0–3; 0=disabled, 1=digital, 2=analog, 3=?
    HVParam.SIPM_COMPENS_MODE:   ListSpec(items=(0, 1, 2, 3), default=0),
    HVParam.HV_COMPENS_MODE:     ListSpec(items=(0, 1, 2, 3), default=0),
    # hw_def: temp_digital_mode_t, 0–2
    HVParam.TEMP_DIGITAL_ENABLE: ListSpec(items=(0, 1, 2),    default=0),
}


class HVSettingEntry(TypedDict):
    id: int
    value: int | float


# ---------------------------------------------------------------------------
# HVSupply — high-level interface to the IDS subsystem
# ---------------------------------------------------------------------------

class HVSupply:
    """High-voltage / SiPM bias supply and temperature monitoring.

    Wraps the IDS gRPC service — a separate service on the same physical
    device as the DPP engine (different port, different proto).  This
    split is a legacy issue that will be merged in a future firmware
    revision.

    The IDS proto exposes parallel SiPM and HV command sets, each with
    independent voltage control and temperature compensation.  Parameter
    IDs match hw_def.json "hv" section (101–119).
    """

    specs: dict[HVParam, ParameterSpec] = HV_PARAMETER_SPECS

    def __init__(self, backend: IDSBackend) -> None:
        self._b = backend
        self._sipm_available: bool | None = None

    def sipm_available(self) -> bool:
        """Check whether a SiPM module is present on this channel.

        Probes the overload register — valid hardware returns 0 or 1,
        absent hardware returns garbage (uninitialized register).
        Result is cached after first call.
        """
        if self._sipm_available is None:
            overload = self._b.get_sipm_overload()
            self._sipm_available = overload in (0, 1)
        return self._sipm_available

    def _require_sipm(self) -> None:
        if not self.sipm_available():
            raise RuntimeError("SiPM module not available on this channel")

    # ---- versions / info ----

    def get_versions(self) -> list[int]:
        return self._b.get_versions()

    def get_ads_temp(self) -> float:
        return self._b.get_ads_temp()

    def get_ads_temp_for_correction(self) -> float:
        """Read ADS5407 through a worker-dedicated backend path when available."""
        reader = getattr(self._b, "get_ads_temp_for_correction", None)
        return float(reader()) if reader is not None else self.get_ads_temp()

    def get_global_diagnostics(self) -> list[GlobalDiagnosticReading]:
        return self._b.get_global_diagnostics()

    # ---- SiPM bias supply ----

    def set_sipm_enable(self, val: int) -> None:
        self._require_sipm()
        HV_PARAMETER_SPECS[HVParam.SIPM_ENABLE].validate(val, "sipm_enable")
        self._b.set_sipm_enable(val)

    def set_sipm_voltage(self, val: float) -> None:
        self._require_sipm()
        HV_PARAMETER_SPECS[HVParam.SIPM_VOLTAGE].validate(val, "sipm_voltage")
        self._b.set_sipm_voltage(val)

    def get_sipm_adc_voltage(self) -> float:
        self._require_sipm()
        return self._b.get_sipm_adc_voltage()

    def get_sipm_adc_current(self) -> float:
        self._require_sipm()
        return self._b.get_sipm_adc_current()

    def get_sipm_overload(self) -> int:
        self._require_sipm()
        return self._b.get_sipm_overload()

    # ---- SiPM temperature compensation ----

    def set_sipm_compens_ct(self, val: float) -> None:
        self._require_sipm()
        HV_PARAMETER_SPECS[HVParam.SIPM_COMPENS_CT].validate(val, "sipm_compens_ct")
        self._b.set_sipm_compens_ct(val)

    def set_sipm_compens_tref(self, val: float) -> None:
        self._require_sipm()
        HV_PARAMETER_SPECS[HVParam.SIPM_COMPENS_TREF].validate(val, "sipm_compens_tref")
        self._b.set_sipm_compens_tref(val)

    def set_sipm_compens_mode(self, val: int) -> None:
        self._require_sipm()
        HV_PARAMETER_SPECS[HVParam.SIPM_COMPENS_MODE].validate(val, "sipm_compens_mode")
        self._b.set_sipm_compens_mode(val)

    def get_sipm_compens_output(self) -> float:
        self._require_sipm()
        return self._b.get_sipm_compens_output()

    # ---- HV bias supply ----

    def set_hv_voltage(self, val: float) -> None:
        HV_PARAMETER_SPECS[HVParam.HV_VOLTAGE].validate(val, "hv_voltage")
        self._b.set_hv_voltage(val)

    def get_hv_adc_voltage(self) -> float:
        return self._b.get_hv_adc_voltage()

    # ---- HV temperature compensation ----

    def set_hv_compens_ct(self, val: float) -> None:
        HV_PARAMETER_SPECS[HVParam.HV_COMPENS_CT].validate(val, "hv_compens_ct")
        self._b.set_hv_compens_ct(val)

    def set_hv_compens_tref(self, val: float) -> None:
        HV_PARAMETER_SPECS[HVParam.HV_COMPENS_TREF].validate(val, "hv_compens_tref")
        self._b.set_hv_compens_tref(val)

    def set_hv_compens_mode(self, val: int) -> None:
        HV_PARAMETER_SPECS[HVParam.HV_COMPENS_MODE].validate(val, "hv_compens_mode")
        self._b.set_hv_compens_mode(val)

    def get_hv_compens_output(self) -> float:
        return self._b.get_hv_compens_output()

    # ---- temperature sensors ----

    def get_temp_analog(self) -> float:
        return self._b.get_temp_analog()

    def set_temp_digital_enable(self, val: int) -> None:
        HV_PARAMETER_SPECS[HVParam.TEMP_DIGITAL_ENABLE].validate(val, "temp_digital_enable")
        self._b.set_temp_digital_enable(val)

    def get_temp_digital_status(self) -> int:
        return self._b.get_temp_digital_status()

    def get_temp_digital(self) -> float:
        return self._b.get_temp_digital()

    # ---- safety ----

    def safe_shutdown(self) -> None:
        """Ramp HV to minimum and disable SiPM. Call before closing the connection."""
        shutdown_voltage = getattr(self._b, "get_safe_shutdown_voltage", None)
        hv_voltage_spec = HV_PARAMETER_SPECS[HVParam.HV_VOLTAGE]
        assert isinstance(hv_voltage_spec, RangeSpec)
        min_hv = (
            float(shutdown_voltage())
            if shutdown_voltage is not None
            else hv_voltage_spec.min_val
        )
        try:
            log.info("Shutting down HV supply: setting HV to %.1f V", min_hv)
            self._b.set_hv_voltage(min_hv)
        except Exception:
            log.exception("Failed to ramp down HV voltage")
        if self.sipm_available():
            try:
                log.info("Shutting down HV supply: disabling SiPM")
                self._b.set_sipm_enable(0)
            except Exception:
                log.exception("Failed to disable SiPM")

    # ---- GUI helpers ----

    def get_settings(self) -> list[HVSettingEntry]:
        """Return current IDS state for GUI consumption.

        IDs match hw_def.json "hv" section register ids (101–119).
        Write-only parameters (setters with no readback) are excluded.
        """
        entries = [
            HVSettingEntry(id=int(HVParam.HV_ADC_VOLTAGE),      value=self.get_hv_adc_voltage()),
            HVSettingEntry(id=int(HVParam.HV_COMPENS_OUTPUT),    value=self.get_hv_compens_output()),
            HVSettingEntry(id=int(HVParam.TEMP_ANALOG),         value=self.get_temp_analog()),
            HVSettingEntry(id=int(HVParam.TEMP_DIGITAL_STATUS), value=self.get_temp_digital_status()),
            HVSettingEntry(id=int(HVParam.TEMP_DIGITAL),        value=self.get_temp_digital()),
        ]
        if self.sipm_available():
            entries.extend([
                HVSettingEntry(id=int(HVParam.SIPM_ADC_VOLTAGE),    value=self._b.get_sipm_adc_voltage()),
                HVSettingEntry(id=int(HVParam.SIPM_ADC_CURRENT),    value=self._b.get_sipm_adc_current()),
                HVSettingEntry(id=int(HVParam.SIPM_OVERLOAD),       value=self._b.get_sipm_overload()),
                HVSettingEntry(id=int(HVParam.SIPM_COMPENS_OUTPUT),  value=self._b.get_sipm_compens_output()),
            ])
        return entries
