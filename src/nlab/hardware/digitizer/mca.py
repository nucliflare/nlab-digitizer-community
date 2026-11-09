from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TypedDict

import numpy as np

from .backends.base import MCABackend
from .current_monitor import CurrentMonitorClient
from .mca_parameters import DEFAULT_PARAMETERS, validate_parameters
from .scope import ListSpec, ParameterSpec, RangeSpec

# ---------------------------------------------------------------------------
# MCA parameter enum — values equal the corresponding vdpp_config.json widget id
# ---------------------------------------------------------------------------


class MCAParam(IntEnum):
    # --- JSON widget ids (value == id in vdpp_config.json "mca" section) ---
    PULSE_POLARITY = 6
    TRIGGER_LEVEL = 7
    PRETRIGGER_SAMPLES = 8
    FRAME_SAMPLES = 9
    ENERGY_BIN = 10
    TRG_SOURCE = 13
    BASELINE_WINDOW = 14
    CFD_FACTOR = 15
    CFD_DELAY = 16
    CRRC2_CDELAY = 17
    CRRC2_FDELAY = 18
    CC_ENABLE = 19
    CC_TIME = 20
    TRAPEZ_ENABLE = 21
    TRAPEZ_R = 22
    TRAPEZ_M = 23
    TRAPEZ_T = 24
    TRAPEZ_E = 25
    TRAPEZ_FT = 26
    TIME_LIMIT = 28
    CRRC2_PZC = 46
    LP_COEFFS = 51  # coefficients array returned at id 51
    CFD_ENABLE = 52
    CFD_TW_LOW = 53
    CFD_TW_HIGH = 54
    MEM1_SIG_SELECT = 57
    MEM2_SIG_SELECT = 58
    EXT_TRIG_ENABLE = 61
    PILEUP_WINDOW = 62
    EDGE_DET_COEFF = 68
    TEMP_COEFF = 69  # vdpp_config.json "hw" section
    TEMP_OFFSET = 70  # vdpp_config.json "hw" section
    PSD_ZC_ENABLE = 71
    PSD_ZC_MODE = 72
    PSD_ZC_LOW = 73
    PSD_ZC_HIGH = 74
    # --- internal — no JSON widget id ---
    GLOBAL_ENABLE = 200
    LP_PRESET = 202  # preset selector (0=200 MHz, 1=70 MHz, 2=moving average)
    CRRC2_RECIPROCAL = 203  # raw 16-bit 1/C register, no legacy widget id
    TRAPEZ_RECIPROCAL = 204  # read-only derived 1/R, no legacy widget id


MCA_DEBUG_SIGNAL_SELECTORS = tuple(range(9))


# ABI2 native ADC samples (2 ns); C delay/averaging are exponents, beta is raw.
MCA_PARAMETER_SPECS: dict[MCAParam, ParameterSpec] = {
    MCAParam.TRIGGER_LEVEL: RangeSpec(min_val=-32768, max_val=32767, step=1, default=-512),
    MCAParam.PULSE_POLARITY: ListSpec(items=(0, 1), default=0),
    MCAParam.BASELINE_WINDOW: ListSpec(items=(0, 1, 2, 3, 4, 5, 6), default=3),
    MCAParam.PRETRIGGER_SAMPLES: RangeSpec(0, 2044, 4, 12),
    MCAParam.FRAME_SAMPLES: RangeSpec(4, 4096, 4, 128),
    MCAParam.ENERGY_BIN: ListSpec(items=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), default=0),
    MCAParam.PILEUP_WINDOW: RangeSpec(min_val=0, max_val=255, step=1, default=0),
    MCAParam.TIME_LIMIT: RangeSpec(min_val=0, max_val=576460752, step=1, default=0),
    MCAParam.GLOBAL_ENABLE: ListSpec(items=(False, True), default=False),
    MCAParam.TEMP_COEFF: RangeSpec(
        min_val=-1.0, max_val=32767 / 32768, step=2**-15, default=-0.00026735,
    ),
    MCAParam.TEMP_OFFSET: RangeSpec(min_val=-32768, max_val=32767, step=1, default=7),
    MCAParam.TRAPEZ_ENABLE: ListSpec(items=(False, True), default=False),
    MCAParam.TRAPEZ_R: RangeSpec(8, 2044, 4, 32),
    MCAParam.TRAPEZ_M: RangeSpec(0, 2044, 4, 32),
    MCAParam.TRAPEZ_T: RangeSpec(0, 4294967295, 1, 0),
    MCAParam.TRAPEZ_E: RangeSpec(0, 8188, 4, 64),
    MCAParam.TRAPEZ_FT: ListSpec(items=(0, 1, 2, 3, 4, 5, 6), default=0),
    MCAParam.CFD_ENABLE: ListSpec(items=(False, True), default=False),
    MCAParam.CFD_FACTOR: RangeSpec(0.0, 32767 / 32768, 2**-15, 13107 / 32768),
    MCAParam.CFD_DELAY: RangeSpec(5, 132, 1, 5),
    MCAParam.CC_ENABLE: ListSpec(items=(False, True), default=False),
    MCAParam.CC_TIME: RangeSpec(4, 32768, 4, 4),
    MCAParam.TRG_SOURCE: ListSpec(items=(0, 1, 2), default=0),
    MCAParam.CRRC2_CDELAY: RangeSpec(0, 6, 1, 3),
    MCAParam.CRRC2_RECIPROCAL: RangeSpec(min_val=0, max_val=65535, step=1, default=0),
    MCAParam.CRRC2_FDELAY: RangeSpec(4, 508, 4, 4),
    MCAParam.CRRC2_PZC: RangeSpec(min_val=-32768, max_val=32767, step=1, default=32761),
    MCAParam.CFD_TW_LOW: RangeSpec(0, 32764, 4, 0),
    MCAParam.CFD_TW_HIGH: RangeSpec(0, 32764, 4, 32),
    MCAParam.MEM1_SIG_SELECT: ListSpec(items=MCA_DEBUG_SIGNAL_SELECTORS, default=0),
    MCAParam.MEM2_SIG_SELECT: ListSpec(items=MCA_DEBUG_SIGNAL_SELECTORS, default=1),
    MCAParam.EXT_TRIG_ENABLE: ListSpec(items=(False, True), default=False),
    MCAParam.EDGE_DET_COEFF: RangeSpec(min_val=0, max_val=4294967295, step=1, default=1),
    MCAParam.PSD_ZC_ENABLE: ListSpec(items=(False, True), default=False),
    MCAParam.PSD_ZC_MODE: ListSpec(items=(0, 1), default=0),
    MCAParam.PSD_ZC_LOW: RangeSpec(0, 32764, 4, 0),
    MCAParam.PSD_ZC_HIGH: RangeSpec(0, 32764, 4, 8),
    MCAParam.LP_PRESET: ListSpec(items=(0, 1, 2), default=0),
}



class MCASettingEntry(TypedDict):
    id: int
    value: int | float | str | bool | list[int]


# ---------------------------------------------------------------------------
# Filter sub-objects
# ---------------------------------------------------------------------------


class Trapezoid:
    def __init__(self, backend: MCABackend) -> None:
        self._b = backend

    def get_enable(self) -> bool:
        return self._b.get_trapez_enable()

    def set_enable(self, val: bool) -> None:
        self._b.set_trapez_enable(val)

    def supports_reciprocal_raw(self) -> bool:
        """Whether the driver exposes its derived, read-only raw 1/R value."""
        return bool(getattr(self._b, "supports_trapezoid_reciprocal_raw", lambda: False)())

    def get_reciprocal_raw(self) -> int:
        if not self.supports_reciprocal_raw():
            raise NotImplementedError("the driver does not expose trapezoid 1/R")
        return self._b.get_trapezoid_reciprocal_raw()

    def get_R(self) -> int:
        return self._b.get_trapez_R()

    def set_R(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.TRAPEZ_R].validate(val, "trapezoid_r")
        self._b.set_trapez_R(val)

    def get_M(self) -> int:
        return self._b.get_trapez_M()

    def set_M(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.TRAPEZ_M].validate(val, "trapezoid_m")
        self._b.set_trapez_M(val)

    def get_T(self) -> int:
        return self._b.get_trapez_T()

    def set_T(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.TRAPEZ_T].validate(
            val, "trapezoid_beta_raw"
        )
        self._b.set_trapez_T(val)

    def get_E(self) -> int:
        return self._b.get_trapez_E()

    def set_E(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.TRAPEZ_E].validate(val, "trapezoid_time")
        self._b.set_trapez_E(val)

    def get_FT(self) -> int:
        return self._b.get_trapez_FT()

    def set_FT(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.TRAPEZ_FT].validate(
            val, "trapezoid_flat_top_window"
        )
        self._b.set_trapez_FT(val)


class CFD:
    def __init__(self, backend: MCABackend) -> None:
        self._b = backend

    def get_enable(self) -> bool:
        return self._b.get_cfd_enable()

    def set_enable(self, val: bool) -> None:
        self._b.set_cfd_enable(val)

    def get_factor(self) -> float:
        return self._b.get_cfd_factor()

    def set_factor(self, val: float) -> None:
        MCA_PARAMETER_SPECS[MCAParam.CFD_FACTOR].validate(val, "cfd_factor")
        self._b.set_cfd_factor(val)

    def get_delay(self) -> int:
        return self._b.get_cfd_delay()

    def set_delay(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.CFD_DELAY].validate(val, "cfd_delay")
        self._b.set_cfd_delay(val)

    def get_time_window_low(self) -> int:
        return self._b.get_cfd_time_window_low()

    def set_time_window_low(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.CFD_TW_LOW].validate(val, "cfd_time_walk_low")
        self._b.set_cfd_time_window_low(val)

    def get_time_window_high(self) -> int:
        return self._b.get_cfd_time_window_high()

    def set_time_window_high(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.CFD_TW_HIGH].validate(val, "cfd_time_walk_high")
        self._b.set_cfd_time_window_high(val)


class CRRC2:
    def __init__(self, backend: MCABackend) -> None:
        self._b = backend

    def supports_reciprocal_raw(self) -> bool:
        """Return whether the driver exposes the optional raw 1/C coefficient."""
        return bool(getattr(self._b, "supports_crrc2_reciprocal_raw", lambda: False)())

    def get_reciprocal_raw(self) -> int:
        if not self.supports_reciprocal_raw():
            raise NotImplementedError("the driver does not expose CR-RC2 1/C")
        return self._b.get_crrc2_reciprocal_raw()

    def set_reciprocal_raw(self, val: int) -> None:
        if not isinstance(val, int):
            raise ValueError("CR-RC2 1/C must be an integer")
        MCA_PARAMETER_SPECS[MCAParam.CRRC2_RECIPROCAL].validate(val, "crrc2_1cdelay_raw")
        if not self.supports_reciprocal_raw():
            raise NotImplementedError("the driver does not expose CR-RC2 1/C")
        self._b.set_crrc2_reciprocal_raw(val)

    def get_Cdelay(self) -> int:
        return self._b.get_crrc2_Cdelay()

    def set_Cdelay(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.CRRC2_CDELAY].validate(
            val, "crrc2_cdelay_ns"
        )
        self._b.set_crrc2_Cdelay(val)

    def get_Fdelay(self) -> int:
        return self._b.get_crrc2_Fdelay()

    def set_Fdelay(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.CRRC2_FDELAY].validate(val, "crrc2_fdelay")
        self._b.set_crrc2_Fdelay(val)

    def get_pzc_coeff(self) -> int:
        return self._b.get_crrc2_pzc_coeff()

    def set_pzc_coeff(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.CRRC2_PZC].validate(val, "crrc2_pzc_raw")
        self._b.set_crrc2_pzc_coeff(val)


class ChargeComparison:
    def __init__(self, backend: MCABackend) -> None:
        self._b = backend

    def get_enable(self) -> bool:
        return self._b.get_cc_enable()

    def set_enable(self, val: bool) -> None:
        self._b.set_cc_enable(val)

    def get_time(self) -> int:
        return self._b.get_cc_time()

    def set_time(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.CC_TIME].validate(val, "charge_comparison_time")
        self._b.set_cc_time(val)


class PSDZeroCrossing:
    def __init__(self, backend: MCABackend) -> None:
        self._b = backend

    def get_enable(self) -> bool:
        return self._b.get_psd_zc_enable()

    def set_enable(self, val: bool) -> None:
        self._b.set_psd_zc_enable(val)

    def get_mode(self) -> int:
        return self._b.get_psd_zc_mode()

    def set_mode(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.PSD_ZC_MODE].validate(
            val, "psd_zero_crossing_mode"
        )
        self._b.set_psd_zc_mode(val)

    def get_time_window_low(self) -> int:
        return self._b.get_psd_zc_time_window_low()

    def set_time_window_low(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.PSD_ZC_LOW].validate(val, "psd_time_walk_low")
        self._b.set_psd_zc_time_window_low(val)

    def get_time_window_high(self) -> int:
        return self._b.get_psd_zc_time_window_high()

    def set_time_window_high(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.PSD_ZC_HIGH].validate(val, "psd_time_walk_high")
        self._b.set_psd_zc_time_window_high(val)


class LPFilter:
    """FIR input low-pass filter."""

    def __init__(self, backend: MCABackend) -> None:
        self._b = backend

    def get_size(self) -> int:
        return self._b.get_lp_coeffs_size()

    def get_coeffs(self) -> np.ndarray:
        return self._b.get_lp_coeffs()

    def set_coeffs(self, val: list[int]) -> None:
        """Write custom FIR coefficients."""
        self._b.set_lp_coeffs(val)

    def set_preset(self, preset_index: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.LP_PRESET].validate(preset_index, "lp_preset")
        self._b.set_lp_coeffs_preset(preset_index)

    def get_iir_average(self) -> int:
        """IIR LP input filter average (read-only hardware register)."""
        return self._b.get_iir_lp_average()


class MCAFilters:
    """Groups all DSP filter settings under named sub-objects."""

    def __init__(self, backend: MCABackend) -> None:
        self.trapezoid = Trapezoid(backend)
        self.cfd = CFD(backend)
        self.crrc2 = CRRC2(backend)
        self.charge_comparison = ChargeComparison(backend)
        self.psd_zc = PSDZeroCrossing(backend)
        self.lp = LPFilter(backend)


# ---------------------------------------------------------------------------
# Read-only statistics view
# ---------------------------------------------------------------------------


class MCAStatistics:
    def __init__(self, backend: MCABackend) -> None:
        self._b = backend

    def get_count_rate(self) -> int:
        return self._b.get_count_rate()

    def get_pulse_deadtime_ms(self) -> float:
        """Return dead time in milliseconds when the backend exposes its scale."""
        scaled_reader = getattr(self._b, "get_pulse_deadtime_ms", None)
        if scaled_reader is not None:
            return float(scaled_reader())
        # Retain the generic backend fallback for lightweight test doubles.
        return float(self._b.get_pulse_deadtime())

    def get_events_lost(self) -> int:
        return self._b.get_events_lost()

    def get_elapsed_time(self) -> int:
        """Return elapsed time in 0.1s ticks. Divide by 10 for seconds."""
        return self._b.get_elapsed_time()

    def get_pulse_overrange(self) -> int:
        return self._b.get_pulse_overrange_counter()

    def get_pulse_pileup(self) -> int:
        return self._b.get_pulse_pileup_counter()

    def get_energy_overrange(self) -> int:
        return self._b.get_energy_overrange_counter()

    def get_energy_estimation_error(self) -> int:
        return self._b.get_energy_estimation_error()

    def get_throughput_error(self) -> int:
        return self._b.get_throughput_error_counter()


# ---------------------------------------------------------------------------
# Sync trigger view (channel-independent)
# ---------------------------------------------------------------------------


class SyncTrigger:
    def __init__(self, backend: MCABackend) -> None:
        self._b = backend

    def get_enable(self) -> bool:
        return self._b.get_sync_enable()

    def set_enable(self, val: bool) -> None:
        self._b.set_sync_enable(val)

    def get_sw_trig(self) -> int:
        return self._b.get_sync_sw_trig()

    def set_sw_trig(self, val: int) -> None:
        self._b.set_sync_sw_trig(val)

    def get_trig_src(self) -> int:
        return self._b.get_sync_trig_src()

    def set_trig_src(self, val: int) -> None:
        self._b.set_sync_trig_src(val)

    def get_timestamp(self) -> int:
        # > ⚠️ UNVERIFIED: units — nlab_mca.py divides by 1e9 in get_timestamp()
        return self._b.get_sync_timestamp()

    def fire(self) -> None:
        """Send a software trigger pulse."""
        self._b.set_sync_sw_trig(1)


# ---------------------------------------------------------------------------
# MultiChannelAnalyzer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McaGateTiming:
    """Candidate panel timings in physical ns, not encoded register counts."""

    pretrigger_ns: int
    frame_ns: int
    cc_time_ns: int
    cc_enabled: bool

    def validate(self) -> None:
        for param, value in ((MCAParam.PRETRIGGER_SAMPLES, self.pretrigger_ns),
                             (MCAParam.FRAME_SAMPLES, self.frame_ns),
                             (MCAParam.CC_TIME, self.cc_time_ns)):
            if type(value) is not int:
                raise ValueError(f"{param.name} must be an integer number of ns")
            if value % 2:
                raise ValueError(f"{param.name} must align to 2 ns ADC samples")
            MCA_PARAMETER_SPECS[param].validate(value // 2, param.name)
        if type(self.cc_enabled) is not bool:
            raise ValueError("Charge-comparison enable must be boolean")
        if not 0 <= self.pretrigger_ns < self.frame_ns:
            raise ValueError("Pretrigger must be shorter than the window length")
        if not 0 < self.cc_time_ns <= self.frame_ns:
            raise ValueError("Short gate must be positive and no longer than the window")


class MultiChannelAnalyzer:
    """High-level MCA / DPP interface."""

    specs: dict[MCAParam, ParameterSpec] = MCA_PARAMETER_SPECS

    def __init__(self, backend: MCABackend) -> None:
        self._b = backend
        # Serializes the short stop/write/restart sequence used for live
        # configuration with MCAWorker's measurement-completion poll. Without
        # this, the worker can observe the intentional enable=0 interval and
        # incorrectly report that the hardware time limit ended the run.
        self._configuration_lock = (
            getattr(backend, "mca_configuration_lock", None) or threading.RLock()
        )
        self.filters = MCAFilters(backend)
        self.statistics = MCAStatistics(backend)
        self.sync = SyncTrigger(backend)

    def create_current_monitor_client(self) -> CurrentMonitorClient:
        """Create a transport connection owned by a monitor worker thread."""
        return self._b.create_current_monitor_client()

    # ---- device info ----

    def get_hw_version(self) -> int:
        return self._b.get_hw_version()

    def get_sw_version(self) -> int:
        return self._b.get_sw_version()

    def get_id_number(self) -> int:
        return self._b.get_id_number()

    # ---- signal parameters ----

    def get_trigger_level(self) -> int:
        return self._b.get_dpp_trigger_level()

    def set_trigger_level(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.TRIGGER_LEVEL].validate(val, "trigger_level")
        self._b.set_dpp_trigger_level(val)

    def get_pulse_polarity(self) -> int:
        """Return pulse polarity. 0 = negative (falling edge), 1 = positive (rising edge)."""
        return self._b.get_pulse_polarity()

    def set_pulse_polarity(self, val: int) -> None:
        """Set pulse polarity. 0 = negative (falling edge), 1 = positive (rising edge)."""
        MCA_PARAMETER_SPECS[MCAParam.PULSE_POLARITY].validate(val, "pulse_polarity")
        self._b.set_pulse_polarity(val)

    def get_baseline_window(self) -> int:
        return self._b.get_bsln_window()

    def set_baseline_window(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.BASELINE_WINDOW].validate(val, "baseline_window")
        self._b.set_bsln_window(val)

    def get_pretrigger_samples(self) -> int:
        return self._b.get_dpp_pretrigger_samples()

    def set_pretrigger_samples(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.PRETRIGGER_SAMPLES].validate(val, "pretrigger_samples")
        self._b.set_dpp_pretrigger_samples(val)

    def get_frame_samples(self) -> int:
        return self._b.get_dpp_frame_samples()

    def set_frame_samples(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.FRAME_SAMPLES].validate(val, "frame_samples")
        self._b.set_dpp_frame_samples(val)

    def get_trg_source(self) -> int:
        return self._b.get_trg_source()

    def set_trg_source(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.TRG_SOURCE].validate(val, "trg_source")
        self._b.set_trg_source(val)

    # ---- acquisition control ----

    def get_global_enable(self) -> bool:
        with self._configuration_lock:
            return self._b.get_global_enable()

    def set_global_enable(self, val: bool) -> None:
        with self._configuration_lock:
            self._b.set_global_enable(val)

    def dma_active(self) -> bool:
        """Whether the driver currently owns an armed MCA list buffer."""
        return self._b.mca_dma_active()

    def gate_timing(self) -> McaGateTiming:
        with self._configuration_lock:
            return McaGateTiming(
                self.get_pretrigger_samples() * 2, self.get_frame_samples() * 2,
                self.filters.charge_comparison.get_time() * 2,
                self.filters.charge_comparison.get_enable(),
            )

    def apply_gate_timing(self, timing: McaGateTiming) -> None:
        """Transfer physical ns to native ADC samples while stopped.

        A failed partial write stays stopped; inspect hardware before retrying.
        STOP is an ownership gate, not a downstream-completion acknowledgement.
        """
        timing.validate()
        with self._configuration_lock:
            if self.get_global_enable() or self.dma_active():
                raise RuntimeError("Stop MCA acquisition and close list-mode DMA before transfer")
            if self.parameter_contract().get("parameter_abi_version") == 2:
                validate_parameters(self.parameter_snapshot() | {
                    "pretrigger_samples": timing.pretrigger_ns // 2,
                    "frame_samples": timing.frame_ns // 2,
                    "charge_comparison_samples": timing.cc_time_ns // 2,
                    "charge_comparison_enable": int(timing.cc_enabled),
                }, complete=True)
            self.set_pretrigger_samples(timing.pretrigger_ns // 2)
            self.set_frame_samples(timing.frame_ns // 2)
            self.filters.charge_comparison.set_time(timing.cc_time_ns // 2)
            self.filters.charge_comparison.set_enable(timing.cc_enabled)

    def get_time_limit(self) -> int:
        """Return time limit in seconds. 0 = no limit."""
        return self._b.get_time_limit()

    def set_time_limit(self, val: int) -> None:
        """Set time limit in seconds. 0 = no limit. HW auto-stops measurement when reached."""
        MCA_PARAMETER_SPECS[MCAParam.TIME_LIMIT].validate(val, "time_limit")
        self._b.set_time_limit(val)

    def get_ext_trig_enable(self) -> bool:
        return self._b.get_ext_trig_enable()

    def set_ext_trig_enable(self, val: bool) -> None:
        self._b.set_ext_trig_enable(val)

    def get_energy_bin(self) -> int:
        return self._b.get_energy_bin()

    def set_energy_bin(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.ENERGY_BIN].validate(val, "energy_bin")
        self._b.set_energy_bin(val)

    def get_pileup_window(self) -> int:
        return self._b.get_pileup_window()

    def set_pileup_window(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.PILEUP_WINDOW].validate(val, "pileup_window")
        self._b.set_pileup_window(val)

    def get_measurement_in_progress(self) -> bool:
        with self._configuration_lock:
            return self._b.get_measurement_in_progress()

    def reconfigure_while_running(self, write: Callable[[], None]) -> bool:
        """Apply while idle; ABI2 requires an explicit stop/configure/start.

        A07 downstream-completion fencing is deferred by the driver contract.
        Legacy test/offline surfaces retain successful stop/write/start behavior,
        but a failed write always leaves the core stopped.
        """
        with self._configuration_lock:
            if self._b.mca_dma_active():
                raise RuntimeError("Stop list-mode DMA before configuring MCA")
            running = self._b.get_global_enable()
            if running and self.parameter_contract().get("parameter_abi_version") == 2:
                raise RuntimeError("Stop MCA before changing measurement settings (ABI2)")
            if running:
                self._b.set_global_enable(False)
            write()
            if running:
                self._b.set_global_enable(True)
            return running

    def parameter_contract(self) -> dict[str, object]:
        reader = getattr(self._b, "mca_parameter_contract", None)
        return reader() if reader is not None else {}

    def prepare_configuration(self, values: dict[str, int] | None = None) -> None:
        candidate = dict(DEFAULT_PARAMETERS if values is None else values)
        validate_parameters(candidate, complete=True)
        prepare = getattr(self._b, "prepare_mca_configuration", None)
        if prepare is not None:
            prepare(candidate)

    def parameter_snapshot(self) -> dict[str, int]:
        return self._b.mca_parameter_snapshot()

    def configure_parameters(self, values: dict[str, int]) -> None:
        with self._configuration_lock:
            self._b.configure_mca_parameters(values)

    def dc_blocker_supported(self) -> bool:
        return bool(getattr(self._b, "supports_dc_blocker", lambda: False)())

    def get_dc_blocker_bypass(self) -> bool:
        return self._b.get_dc_blocker_bypass()

    def set_dc_blocker_bypass(self, bypass: bool) -> None:
        with self._configuration_lock:
            if self.get_global_enable() or self.dma_active():
                raise RuntimeError("Stop MCA before changing its DC path")
            self._b.set_dc_blocker_bypass(bypass)

    def get_temp_coeff(self) -> float:
        return self._b.get_temp_coeff()

    def set_temp_coeff(self, val: float) -> None:
        MCA_PARAMETER_SPECS[MCAParam.TEMP_COEFF].validate(val, "temperature_coefficient")
        self._b.set_temp_coeff(val)

    def get_temp_offset(self) -> int:
        return self._b.get_temp_offset()

    def set_temp_offset(self, val: int) -> None:
        self._b.set_temp_offset(val)

    def set_temperature_correction(self, coefficient: float, offset: int) -> None:
        """Set both correction terms through a worker-safe path if provided."""
        MCA_PARAMETER_SPECS[MCAParam.TEMP_COEFF].validate(
            coefficient,
            "temperature_coefficient",
        )
        MCA_PARAMETER_SPECS[MCAParam.TEMP_OFFSET].validate(
            offset,
            "temperature_offset",
        )
        with self._configuration_lock:
            writer = getattr(self._b, "set_temperature_correction_from_worker", None)
            if writer is not None:
                writer(coefficient, offset)
                return
            self._b.set_temp_coeff(coefficient)
            self._b.set_temp_offset(offset)

    # ---- edge detector coefficient ----

    def get_edge_det_coeff(self) -> int:
        return self._b.get_edge_det_coeff()

    def set_edge_det_coeff(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.EDGE_DET_COEFF].validate(val, "edge_det_coeff")
        self._b.set_edge_det_coeff(val)

    def edge_det_coeff_is_hardware_backed(self) -> bool:
        """Whether the selected backend has a real edge-coefficient register."""
        return bool(getattr(self._b, "edge_det_coeff_is_hardware_backed", lambda: True)())

    # ---- pulse memory signal routing ----

    def get_mem1_sig_select(self) -> int:
        return self._b.get_mem1_sig_select()

    def get_debug_signal_selectors(self) -> tuple[int, ...]:
        return self._b.get_debug_signal_selectors()

    def set_mem1_sig_select(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.MEM1_SIG_SELECT].validate(val, "mem1_sig_select")
        self._b.set_mem1_sig_select(val)

    def get_mem2_sig_select(self) -> int:
        return self._b.get_mem2_sig_select()

    def set_mem2_sig_select(self, val: int) -> None:
        MCA_PARAMETER_SPECS[MCAParam.MEM2_SIG_SELECT].validate(val, "mem2_sig_select")
        self._b.set_mem2_sig_select(val)

    def get_mem_amount(self) -> int:
        return self._b.get_mem_amount()

    def get_mem_frame_sizes(self) -> np.ndarray:
        return self._b.get_mem_frame_sizes()

    # ---- start / stop ----

    def start(self) -> None:
        self.set_global_enable(True)

    def stop(self) -> None:
        self.set_global_enable(False)

    # ---- data acquisition ----

    def acquire_spectrum(self) -> np.ndarray:
        """Read the accumulated histogram. Returns uint32 array of shape (n_bins,)."""
        with self._configuration_lock:
            return self._b.read_histogram()

    def clear_spectrum(self) -> None:
        self._b.clear_histogram()

    def get_histogram_size(self) -> int:
        return self._b.get_histogram_size()

    def acquire_waveforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Read pulse memory banks. Returns (bank1, bank2) as int16 arrays."""
        return self._b.read_waveform_banks()

    # ---- GUI helpers ----

    def get_settings(self) -> list[MCASettingEntry]:
        """Return current hardware state as a flat list for GUI consumption.

        Legacy IDs match vdpp_config.json mca/hw section widget ids;
        optional raw 1/C uses internal ID 203; derived read-only 1/R uses 204.
        LP coefficients (LP_COEFFS) are returned as list[int] via .tolist().
        """
        settings = [
            MCASettingEntry(id=int(MCAParam.PULSE_POLARITY), value=self.get_pulse_polarity()),
            MCASettingEntry(id=int(MCAParam.TRIGGER_LEVEL), value=self.get_trigger_level()),
            MCASettingEntry(id=int(MCAParam.PRETRIGGER_SAMPLES), value=self.get_pretrigger_samples()),
            MCASettingEntry(id=int(MCAParam.FRAME_SAMPLES), value=self.get_frame_samples()),
            MCASettingEntry(id=int(MCAParam.ENERGY_BIN), value=self.get_energy_bin()),
            MCASettingEntry(id=int(MCAParam.TRG_SOURCE), value=self.get_trg_source()),
            MCASettingEntry(id=int(MCAParam.BASELINE_WINDOW), value=self.get_baseline_window()),
            MCASettingEntry(id=int(MCAParam.CFD_FACTOR), value=self.filters.cfd.get_factor()),
            MCASettingEntry(id=int(MCAParam.CFD_DELAY), value=self.filters.cfd.get_delay()),
            MCASettingEntry(id=int(MCAParam.CRRC2_CDELAY), value=self.filters.crrc2.get_Cdelay()),
            MCASettingEntry(id=int(MCAParam.CRRC2_FDELAY), value=self.filters.crrc2.get_Fdelay()),
            MCASettingEntry(id=int(MCAParam.CC_ENABLE), value=self.filters.charge_comparison.get_enable()),
            MCASettingEntry(id=int(MCAParam.CC_TIME), value=self.filters.charge_comparison.get_time()),
            MCASettingEntry(id=int(MCAParam.TRAPEZ_ENABLE), value=self.filters.trapezoid.get_enable()),
            MCASettingEntry(id=int(MCAParam.TRAPEZ_R), value=self.filters.trapezoid.get_R()),
            MCASettingEntry(id=int(MCAParam.TRAPEZ_M), value=self.filters.trapezoid.get_M()),
            MCASettingEntry(id=int(MCAParam.TRAPEZ_T), value=self.filters.trapezoid.get_T()),
            MCASettingEntry(id=int(MCAParam.TRAPEZ_E), value=self.filters.trapezoid.get_E()),
            MCASettingEntry(id=int(MCAParam.TRAPEZ_FT), value=self.filters.trapezoid.get_FT()),
            MCASettingEntry(id=int(MCAParam.TIME_LIMIT), value=self.get_time_limit()),
            MCASettingEntry(id=int(MCAParam.CRRC2_PZC), value=self.filters.crrc2.get_pzc_coeff()),
            MCASettingEntry(id=int(MCAParam.LP_COEFFS), value=self.filters.lp.get_coeffs().tolist()),
            MCASettingEntry(id=int(MCAParam.CFD_ENABLE), value=self.filters.cfd.get_enable()),
            MCASettingEntry(id=int(MCAParam.CFD_TW_LOW), value=self.filters.cfd.get_time_window_low()),
            MCASettingEntry(id=int(MCAParam.CFD_TW_HIGH), value=self.filters.cfd.get_time_window_high()),
            MCASettingEntry(id=int(MCAParam.MEM1_SIG_SELECT), value=self.get_mem1_sig_select()),
            MCASettingEntry(id=int(MCAParam.MEM2_SIG_SELECT), value=self.get_mem2_sig_select()),
            MCASettingEntry(id=int(MCAParam.EXT_TRIG_ENABLE), value=self.get_ext_trig_enable()),
            MCASettingEntry(id=int(MCAParam.PILEUP_WINDOW), value=self.get_pileup_window()),
            MCASettingEntry(id=int(MCAParam.EDGE_DET_COEFF), value=self.get_edge_det_coeff()),
            MCASettingEntry(id=int(MCAParam.TEMP_COEFF), value=self.get_temp_coeff()),
            MCASettingEntry(id=int(MCAParam.TEMP_OFFSET), value=self.get_temp_offset()),
            MCASettingEntry(id=int(MCAParam.PSD_ZC_ENABLE), value=self.filters.psd_zc.get_enable()),
            MCASettingEntry(id=int(MCAParam.PSD_ZC_MODE), value=self.filters.psd_zc.get_mode()),
            MCASettingEntry(id=int(MCAParam.PSD_ZC_LOW), value=self.filters.psd_zc.get_time_window_low()),
            MCASettingEntry(id=int(MCAParam.PSD_ZC_HIGH), value=self.filters.psd_zc.get_time_window_high()),
        ]
        if self.filters.crrc2.supports_reciprocal_raw():
            settings.append(MCASettingEntry(
                id=int(MCAParam.CRRC2_RECIPROCAL),
                value=self.filters.crrc2.get_reciprocal_raw(),
            ))
        if self.filters.trapezoid.supports_reciprocal_raw():
            settings.append(MCASettingEntry(
                id=int(MCAParam.TRAPEZ_RECIPROCAL),
                value=self.filters.trapezoid.get_reciprocal_raw(),
            ))
        return settings
