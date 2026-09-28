from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..diagnostics import GlobalDiagnosticReading

if TYPE_CHECKING:
    from ..current_monitor import CurrentMonitorClient


class ScopeBackend(ABC):
    """Scope hardware access in hardware-native samples and ADC counts."""

    def create_isolated_scope_backend(self) -> "ScopeBackend":
        """Return a new connection suitable for a background scope operation.

        Backends that support Scope Auto Setup override this. A separate
        connection is required because remote hardware transports are not
        generally safe to share between the GUI and worker threads.
        """
        raise NotImplementedError("this backend does not support isolated scope access")

    def close(self) -> None:
        """Close this backend's transport resources."""

    @abstractmethod
    def get_ip_version(self) -> int: ...

    @abstractmethod
    def get_mem_frame_size(self) -> int: ...

    @abstractmethod
    def get_enable(self) -> bool: ...

    @abstractmethod
    def set_enable(self, val: bool) -> None: ...

    @abstractmethod
    def get_dma_enable(self) -> bool: ...

    @abstractmethod
    def set_dma_enable(self, val: bool) -> None: ...

    @abstractmethod
    def get_trigger_level(self) -> int: ...

    @abstractmethod
    def set_trigger_level(self, val: int) -> None: ...

    @abstractmethod
    def get_edge(self) -> int: ...

    @abstractmethod
    def set_edge(self, val: int) -> None: ...

    @abstractmethod
    def get_pretrigger_samples(self) -> int: ...

    @abstractmethod
    def set_pretrigger_samples(self, val: int) -> None: ...

    @abstractmethod
    def get_frame_samples(self) -> int: ...

    @abstractmethod
    def set_frame_samples(self, val: int) -> None: ...

    @abstractmethod
    def get_dac_value(self) -> int: ...

    @abstractmethod
    def set_dac_value(self, val: int) -> None: ...

    @abstractmethod
    def read_frame(self) -> np.ndarray:
        """Returns int16 array of shape (n_samples,)."""
        ...


class MCABackend(ABC):
    """Abstract interface for MCA / DPP hardware access."""

    def create_current_monitor_client(self) -> "CurrentMonitorClient":
        """Return an independent connection for background IIR polling.

        A transport must opt in explicitly.  Falling back to ``self`` would
        share one remote connection between the GUI and worker threads, which
        is not safe for libiio and can also serialize unrelated gRPC calls.
        """
        raise NotImplementedError(
            "this backend does not support isolated current-monitor access"
        )

    # ---- device info ----
    @abstractmethod
    def get_hw_version(self) -> int: ...

    @abstractmethod
    def get_sw_version(self) -> int: ...

    @abstractmethod
    def get_id_number(self) -> int: ...

    # ---- signal parameters ----
    @abstractmethod
    def get_dpp_trigger_level(self) -> int: ...

    @abstractmethod
    def set_dpp_trigger_level(self, val: int) -> None: ...

    @abstractmethod
    def get_pulse_polarity(self) -> int: ...

    @abstractmethod
    def set_pulse_polarity(self, val: int) -> None: ...

    @abstractmethod
    def get_bsln_window(self) -> int: ...

    @abstractmethod
    def set_bsln_window(self, val: int) -> None: ...

    @abstractmethod
    def get_dpp_pretrigger_samples(self) -> int: ...

    @abstractmethod
    def set_dpp_pretrigger_samples(self, val: int) -> None: ...

    @abstractmethod
    def get_dpp_frame_samples(self) -> int: ...

    @abstractmethod
    def set_dpp_frame_samples(self, val: int) -> None: ...

    # ---- acquisition control ----
    @abstractmethod
    def get_global_enable(self) -> bool: ...

    @abstractmethod
    def set_global_enable(self, val: bool) -> None: ...

    @abstractmethod
    def get_dpp_dma_enable(self) -> bool: ...

    @abstractmethod
    def set_dpp_dma_enable(self, val: bool) -> None: ...

    @abstractmethod
    def get_time_limit(self) -> int: ...

    @abstractmethod
    def set_time_limit(self, val: int) -> None: ...

    @abstractmethod
    def get_ext_trig_enable(self) -> bool: ...

    @abstractmethod
    def set_ext_trig_enable(self, val: bool) -> None: ...

    @abstractmethod
    def get_trg_source(self) -> int: ...

    @abstractmethod
    def set_trg_source(self, val: int) -> None: ...

    @abstractmethod
    def get_energy_bin(self) -> int: ...

    @abstractmethod
    def set_energy_bin(self, val: int) -> None: ...

    @abstractmethod
    def get_pileup_window(self) -> int: ...

    @abstractmethod
    def set_pileup_window(self, val: int) -> None: ...

    @abstractmethod
    def get_measurement_in_progress(self) -> bool: ...

    # ---- statistics (all read-only) ----
    @abstractmethod
    def get_pulse_deadtime(self) -> int: ...

    @abstractmethod
    def get_events_lost(self) -> int: ...

    @abstractmethod
    def get_count_rate(self) -> int: ...

    @abstractmethod
    def get_elapsed_time(self) -> int: ...

    @abstractmethod
    def get_pulse_overrange_counter(self) -> int: ...

    @abstractmethod
    def get_pulse_pileup_counter(self) -> int: ...

    @abstractmethod
    def get_energy_overrange_counter(self) -> int: ...

    @abstractmethod
    def get_energy_estimation_error(self) -> int: ...

    @abstractmethod
    def get_throughput_error_counter(self) -> int: ...

    # ---- FIR input LP filter ----
    @abstractmethod
    def get_lp_coeffs_size(self) -> int: ...

    @abstractmethod
    def set_lp_coeffs_preset(self, val: int) -> None: ...

    @abstractmethod
    def get_lp_coeffs(self) -> np.ndarray:
        """Returns int32 array of FIR coefficients."""
        ...

    @abstractmethod
    def set_lp_coeffs(self, coeffs: list[int]) -> None: ...

    # ---- IIR LP filter (read-only hardware register) ----
    @abstractmethod
    def get_iir_lp_average(self) -> int: ...

    # ---- temperature compensation ----
    @abstractmethod
    def get_temp_coeff(self) -> float: ...

    @abstractmethod
    def set_temp_coeff(self, val: float) -> None: ...

    @abstractmethod
    def get_temp_offset(self) -> int: ...

    @abstractmethod
    def set_temp_offset(self, val: int) -> None: ...

    # ---- edge detector ----
    @abstractmethod
    def get_edge_det_coeff(self) -> int: ...

    @abstractmethod
    def set_edge_det_coeff(self, val: int) -> None: ...

    # ---- CRRC2 shaping filter ----
    @abstractmethod
    def get_crrc2_Cdelay(self) -> int: ...

    @abstractmethod
    def set_crrc2_Cdelay(self, val: int) -> None: ...

    @abstractmethod
    def get_crrc2_Fdelay(self) -> int: ...

    @abstractmethod
    def set_crrc2_Fdelay(self, val: int) -> None: ...

    @abstractmethod
    def get_crrc2_pzc_coeff(self) -> int: ...

    @abstractmethod
    def set_crrc2_pzc_coeff(self, val: int) -> None: ...

    # ---- trapezoidal filter ----
    @abstractmethod
    def get_trapez_enable(self) -> bool: ...

    @abstractmethod
    def set_trapez_enable(self, val: bool) -> None: ...

    @abstractmethod
    def get_trapez_R(self) -> int: ...

    @abstractmethod
    def set_trapez_R(self, val: int) -> None: ...

    @abstractmethod
    def get_trapez_M(self) -> int: ...

    @abstractmethod
    def set_trapez_M(self, val: int) -> None: ...

    @abstractmethod
    def get_trapez_T(self) -> int: ...

    @abstractmethod
    def set_trapez_T(self, val: int) -> None: ...

    @abstractmethod
    def get_trapez_E(self) -> int: ...

    @abstractmethod
    def set_trapez_E(self, val: int) -> None: ...

    @abstractmethod
    def get_trapez_FT(self) -> int: ...

    @abstractmethod
    def set_trapez_FT(self, val: int) -> None: ...

    # ---- CFD ----
    @abstractmethod
    def get_cfd_enable(self) -> bool: ...

    @abstractmethod
    def set_cfd_enable(self, val: bool) -> None: ...

    @abstractmethod
    def get_cfd_factor(self) -> float: ...

    @abstractmethod
    def set_cfd_factor(self, val: float) -> None: ...

    @abstractmethod
    def get_cfd_delay(self) -> int: ...

    @abstractmethod
    def set_cfd_delay(self, val: int) -> None: ...

    @abstractmethod
    def get_cfd_time_window_low(self) -> int: ...

    @abstractmethod
    def set_cfd_time_window_low(self, val: int) -> None: ...

    @abstractmethod
    def get_cfd_time_window_high(self) -> int: ...

    @abstractmethod
    def set_cfd_time_window_high(self, val: int) -> None: ...

    # ---- Charge Comparison ----
    @abstractmethod
    def get_cc_enable(self) -> bool: ...

    @abstractmethod
    def set_cc_enable(self, val: bool) -> None: ...

    @abstractmethod
    def get_cc_time(self) -> int: ...

    @abstractmethod
    def set_cc_time(self, val: int) -> None: ...

    # ---- PSD Zero Crossing ----
    @abstractmethod
    def get_psd_zc_enable(self) -> bool: ...

    @abstractmethod
    def set_psd_zc_enable(self, val: bool) -> None: ...

    @abstractmethod
    def get_psd_zc_mode(self) -> int: ...

    @abstractmethod
    def set_psd_zc_mode(self, val: int) -> None: ...

    @abstractmethod
    def get_psd_zc_time_window_low(self) -> int: ...

    @abstractmethod
    def set_psd_zc_time_window_low(self, val: int) -> None: ...

    @abstractmethod
    def get_psd_zc_time_window_high(self) -> int: ...

    @abstractmethod
    def set_psd_zc_time_window_high(self, val: int) -> None: ...

    # ---- pulse memory ----
    @abstractmethod
    def get_debug_signal_selectors(self) -> tuple[int, ...]:
        """Return debug-memory selector codes supported by this backend."""
        ...

    @abstractmethod
    def get_mem1_sig_select(self) -> int: ...

    @abstractmethod
    def set_mem1_sig_select(self, val: int) -> None: ...

    @abstractmethod
    def get_mem2_sig_select(self) -> int: ...

    @abstractmethod
    def set_mem2_sig_select(self, val: int) -> None: ...

    @abstractmethod
    def get_mem_amount(self) -> int: ...

    @abstractmethod
    def get_mem_frame_sizes(self) -> np.ndarray:
        """Returns uint32 array of per-channel frame sizes."""
        ...

    @abstractmethod
    def read_waveform_banks(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (bank1, bank2) as int16 arrays."""
        ...

    # ---- histogram ----
    @abstractmethod
    def read_histogram(self) -> np.ndarray:
        """Returns uint32 array of shape (n_bins,)."""
        ...

    @abstractmethod
    def clear_histogram(self) -> None: ...

    @abstractmethod
    def get_histogram_size(self) -> int: ...

    # ---- sync trigger (channel-independent / global) ----
    @abstractmethod
    def get_sync_enable(self) -> bool: ...

    @abstractmethod
    def set_sync_enable(self, val: bool) -> None: ...

    @abstractmethod
    def get_sync_sw_trig(self) -> int: ...

    @abstractmethod
    def set_sync_sw_trig(self, val: int) -> None: ...

    @abstractmethod
    def get_sync_trig_src(self) -> int: ...

    @abstractmethod
    def set_sync_trig_src(self, val: int) -> None: ...

    @abstractmethod
    def get_sync_timestamp(self) -> int: ...


class IDSBackend(ABC):
    """Abstract interface for the IDS subsystem.

    Power-supply and temperature-sensor access, provided either by the
    legacy separate gRPC service or directly by IIO devices.
    """

    # ---- versions ----
    @abstractmethod
    def get_versions(self) -> list[int]: ...

    @abstractmethod
    def get_ads_temp(self) -> float: ...

    @abstractmethod
    def get_global_diagnostics(self) -> list[GlobalDiagnosticReading]:
        """Read channel-independent board sensors and health indicators."""
        ...

    # ---- SiPM bias supply ----
    @abstractmethod
    def set_sipm_enable(self, val: int) -> None: ...

    @abstractmethod
    def set_sipm_voltage(self, val: float) -> None: ...

    @abstractmethod
    def get_sipm_adc_voltage(self) -> float: ...

    @abstractmethod
    def get_sipm_adc_current(self) -> float: ...

    @abstractmethod
    def get_sipm_overload(self) -> int: ...

    # ---- SiPM temperature compensation ----
    @abstractmethod
    def set_sipm_compens_ct(self, val: float) -> None: ...

    @abstractmethod
    def set_sipm_compens_tref(self, val: float) -> None: ...

    @abstractmethod
    def set_sipm_compens_mode(self, val: int) -> None: ...

    @abstractmethod
    def get_sipm_compens_output(self) -> float: ...

    # ---- HV bias supply ----
    @abstractmethod
    def set_hv_voltage(self, val: float) -> None: ...

    @abstractmethod
    def get_hv_adc_voltage(self) -> float: ...

    # ---- HV temperature compensation ----
    @abstractmethod
    def set_hv_compens_ct(self, val: float) -> None: ...

    @abstractmethod
    def set_hv_compens_tref(self, val: float) -> None: ...

    @abstractmethod
    def set_hv_compens_mode(self, val: int) -> None: ...

    @abstractmethod
    def get_hv_compens_output(self) -> float: ...

    # ---- temperature sensors ----
    @abstractmethod
    def get_temp_analog(self) -> float: ...

    @abstractmethod
    def set_temp_digital_enable(self, val: int) -> None: ...

    @abstractmethod
    def get_temp_digital_status(self) -> int: ...

    @abstractmethod
    def get_temp_digital(self) -> float: ...

    def close(self) -> None:
        pass


class DigitizerBackend(ScopeBackend, MCABackend, ABC):
    """Combined backend that satisfies both scope and MCA interfaces.

    All concrete backends (gRPC, IIO, …) should inherit from this class.
    IDSBackend is intentionally separate — it connects to a different server.
    """

    def close(self) -> None:
        pass
