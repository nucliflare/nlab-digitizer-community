"""gRPC backend: talks directly to the Engine gRPC stub.

Replaces VDPP_Engine_gRPC for the digitizer layer. Does not inherit from
VDPP_Engine_Base — it implements DigitizerBackend directly.
"""

from typing import Any

import grpc
import numpy as np

import settings_pb2 as sp
import settings_pb2_grpc
import base_pb2 as bsp

from .base import DigitizerBackend

import logging

log = logging.getLogger(__name__)

# Protobuf registers fields and enum values at runtime via reflection; Pylance
# cannot see them as static attributes or constructor kwargs. Typing as Any
# keeps all _CMD.VDPP_* accesses and _Msg/_Val constructor calls clean.
_CMD: Any = sp.RegisterMessage
_Msg: Any = sp.RegisterMessage
_Val: Any = bsp.ValueMessage


class GrpcDigitizerBackend(DigitizerBackend):
    """Single gRPC connection implementing both ScopeBackend and MCABackend."""

    def __init__(
        self,
        channel: int,
        hostname: str = "192.168.10.20",
        port: int = 50050,
    ) -> None:
        self._ch = channel
        self._gchannel = grpc.insecure_channel(f"{hostname}:{port}")
        self._stub = settings_pb2_grpc.EngineStub(self._gchannel)
        log.info("DPP gRPC backend: connected ch%d to %s:%d", channel, hostname, port)

    def close(self) -> None:
        log.info("DPP gRPC backend: closing ch%d", self._ch)
        self._gchannel.close()

    # ------------------------------------------------------------------
    # Internal helpers — the only place gRPC types appear in this file
    # ------------------------------------------------------------------

    def _reg(
        self,
        cmd: int,
        *,
        t_int32: int | None = None,
        t_double: float | None = None,
        no_channel: bool = False,
    ):
        """Build and send a DPP_reg_access call; raise on error response."""
        kw: dict = {}
        if t_int32 is not None:
            kw["t_int32"] = int(t_int32)
        elif t_double is not None:
            kw["t_double"] = float(t_double)
        if not no_channel:
            kw["channel"] = self._ch

        resp = self._stub.DPP_reg_access(
            _Msg(command=cmd, value=_Val(**kw))
        )
        if resp.WhichOneof("response") == "error_response":
            raise RuntimeError(resp.error_response.message)
        return resp

    # Typed scalar extractors — eliminate int|float ambiguity at every call site
    def _int(self, cmd: int, **kw: object) -> int:
        resp = self._reg(cmd, **kw)  # type: ignore[arg-type]
        return int(getattr(resp.value, resp.value.WhichOneof("value")))

    def _float(self, cmd: int, **kw: object) -> float:
        resp = self._reg(cmd, **kw)  # type: ignore[arg-type]
        return float(getattr(resp.value, resp.value.WhichOneof("value")))

    def _bool(self, cmd: int, **kw: object) -> bool:
        resp = self._reg(cmd, **kw)  # type: ignore[arg-type]
        return bool(getattr(resp.value, resp.value.WhichOneof("value")))

    # ------------------------------------------------------------------
    # ScopeBackend
    # ------------------------------------------------------------------

    def get_ip_version(self) -> int:
        return self._int(_CMD.VDPP_scope_get_ip_version)

    def get_mem_frame_size(self) -> int:
        return self._int(_CMD.VDPP_scope_get_mem_frame_size, no_channel=True)

    def get_enable(self) -> bool:
        return self._bool(_CMD.VDPP_scope_get_enable)

    def set_enable(self, val: bool) -> None:
        self._reg(_CMD.VDPP_scope_set_enable, t_int32=int(val))

    def get_dma_enable(self) -> bool:
        return self._bool(_CMD.VDPP_scope_get_dma_enable)

    def set_dma_enable(self, val: bool) -> None:
        self._reg(_CMD.VDPP_scope_set_dma_enable, t_int32=int(val))

    def get_trigger_level(self) -> int:
        return self._int(_CMD.VDPP_scope_get_trigger_level)

    def set_trigger_level(self, val: int) -> None:
        self._reg(_CMD.VDPP_scope_set_trigger_level, t_int32=val)

    def get_edge(self) -> int:
        return self._int(_CMD.VDPP_scope_get_edge)

    def set_edge(self, val: int) -> None:
        self._reg(_CMD.VDPP_scope_set_edge, t_int32=val)

    def get_pretrigger_samples(self) -> int:
        return self._int(_CMD.VDPP_scope_get_pretrigger_len)

    def set_pretrigger_samples(self, val: int) -> None:
        self._reg(_CMD.VDPP_scope_set_pretrigger_len, t_int32=val)

    def get_frame_samples(self) -> int:
        return self._int(_CMD.VDPP_scope_get_frame_len)

    def set_frame_samples(self, val: int) -> None:
        self._reg(_CMD.VDPP_scope_set_frame_len, t_int32=val)

    def get_dac_value(self) -> int:
        return self._int(_CMD.VDPP_dac_get_value)

    def set_dac_value(self, val: int) -> None:
        self._reg(_CMD.VDPP_dac_set_value, t_int32=val)

    def read_frame(self) -> np.ndarray:
        resp = self._reg(_CMD.VDPP_scope_get_mem_frame)
        return np.asarray(resp.int32_array, dtype=np.int16)

    # ------------------------------------------------------------------
    # MCABackend — device info
    # ------------------------------------------------------------------

    def get_hw_version(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_hw_version)

    def get_sw_version(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_sw_version)

    def get_id_number(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_id_number)

    # ---- signal parameters (dpp_ prefix avoids name clash with ScopeBackend) ----

    def get_dpp_trigger_level(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_trigger_level)

    def set_dpp_trigger_level(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_trigger_level, t_int32=val)

    def get_pulse_polarity(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_pulse_polarity)

    def set_pulse_polarity(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_pulse_polarity, t_int32=val)

    def get_bsln_window(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_bsln_window)

    def set_bsln_window(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_bsln_window, t_int32=val)

    def get_dpp_pretrigger_samples(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_pretrigger_len)

    def set_dpp_pretrigger_samples(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_pretrigger_len, t_int32=val)

    def get_dpp_frame_samples(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_frame_len)

    def set_dpp_frame_samples(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_frame_len, t_int32=val)

    # ---- acquisition control ----

    def get_global_enable(self) -> bool:
        return self._bool(_CMD.VDPP_dpp_get_global_enable)

    def set_global_enable(self, val: bool) -> None:
        self._reg(_CMD.VDPP_dpp_set_global_enable, t_int32=int(val))

    def get_dpp_dma_enable(self) -> bool:
        return self._bool(_CMD.VDPP_dpp_get_dma_enable)

    def set_dpp_dma_enable(self, val: bool) -> None:
        self._reg(_CMD.VDPP_dpp_set_dma_enable, t_int32=int(val))

    def get_time_limit(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_set_time_enable)

    def set_time_limit(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_set_time_enable, t_int32=val)

    def get_ext_trig_enable(self) -> bool:
        return self._bool(_CMD.VDPP_dpp_get_ext_trig_enable)

    def set_ext_trig_enable(self, val: bool) -> None:
        self._reg(_CMD.VDPP_dpp_set_ext_trig_enable, t_int32=int(val))

    def get_trg_source(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_trg_source)

    def set_trg_source(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_trg_source, t_int32=val)

    def get_energy_bin(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_energy_bin)

    def set_energy_bin(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_energy_bin, t_int32=val)

    def get_pileup_window(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_pileup_detector_window)

    def set_pileup_window(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_pileup_detector_window, t_int32=val)

    def get_measurement_in_progress(self) -> bool:
        return self._bool(_CMD.VDPP_dpp_get_measurement_in_progress)

    # ---- statistics ----

    def get_pulse_deadtime(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_pulse_deadtime)

    def get_events_lost(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_events_lost)

    def get_count_rate(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_count_rate)

    def get_elapsed_time(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_time_enable)

    def get_pulse_overrange_counter(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_pulse_overrange_counter)

    def get_pulse_pileup_counter(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_pulse_pileup_counter)

    def get_energy_overrange_counter(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_energy_overrange_counter)

    def get_energy_estimation_error(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_energy_estimation_error)

    def get_throughput_error_counter(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_throughput_error_counter)

    # ---- FIR LP filter ----

    def get_lp_coeffs_size(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_lp_coeffs_size, no_channel=True)

    def set_lp_coeffs_preset(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_lp_coeffs_preset, t_int32=val)

    def get_lp_coeffs(self) -> np.ndarray:
        resp = self._reg(_CMD.VDPP_dpp_get_lp_coeffs)
        return np.asarray(resp.int32_array, dtype=np.int32)

    def set_lp_coeffs(self, coeffs: list[int]) -> None:
        # Uploads custom FIR coefficients via RegisterMessage.int32_array.
        resp = self._stub.DPP_reg_access(
            _Msg(
                command=_CMD.VDPP_dpp_set_lp_coeffs,
                value=_Val(channel=self._ch),
                int32_array=list(coeffs),
            )
        )
        if resp.WhichOneof("response") == "error_response":
            raise RuntimeError(resp.error_response.message)

    # ---- IIR LP filter ----

    def get_iir_lp_average(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_iir_lp_average)

    # ---- temperature compensation ----

    def get_temp_coeff(self) -> float:
        return self._float(_CMD.VDPP_dpp_get_temp_coeff)

    def set_temp_coeff(self, val: float) -> None:
        self._reg(_CMD.VDPP_dpp_set_temp_coeff, t_double=val)

    def get_temp_offset(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_temp_offset)

    def set_temp_offset(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_temp_offset, t_int32=val)

    # ---- edge detector ----

    def get_edge_det_coeff(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_edge_det_coeff)

    def set_edge_det_coeff(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_edge_det_coeff, t_int32=val)

    # ---- CRRC2 ----

    def get_crrc2_Cdelay(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_crrc2_Cdelay)

    def set_crrc2_Cdelay(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_crrc2_Cdelay, t_int32=val)

    def get_crrc2_Fdelay(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_crrc2_Fdelay)

    def set_crrc2_Fdelay(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_crrc2_Fdelay, t_int32=val)

    def get_crrc2_pzc_coeff(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_crrc2_pzc_coeff)

    def set_crrc2_pzc_coeff(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_crrc2_pzc_coeff, t_int32=val)

    # ---- trapezoidal filter ----

    def get_trapez_enable(self) -> bool:
        return self._bool(_CMD.VDPP_dpp_get_trapez_enable)

    def set_trapez_enable(self, val: bool) -> None:
        self._reg(_CMD.VDPP_dpp_set_trapez_enable, t_int32=int(val))

    def get_trapez_R(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_trapez_R)

    def set_trapez_R(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_trapez_R, t_int32=val)

    def get_trapez_M(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_trapez_M)

    def set_trapez_M(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_trapez_M, t_int32=val)

    def get_trapez_T(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_trapez_T)

    def set_trapez_T(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_trapez_T, t_int32=val)

    def get_trapez_E(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_trapez_E)

    def set_trapez_E(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_trapez_E, t_int32=val)

    def get_trapez_FT(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_trapez_FT)

    def set_trapez_FT(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_trapez_FT, t_int32=val)

    # ---- CFD ----

    def get_cfd_enable(self) -> bool:
        return self._bool(_CMD.VDPP_dpp_get_cfd_enable)

    def set_cfd_enable(self, val: bool) -> None:
        self._reg(_CMD.VDPP_dpp_set_cfd_enable, t_int32=int(val))

    def get_cfd_factor(self) -> float:
        return self._float(_CMD.VDPP_dpp_get_cfd_factor)

    def set_cfd_factor(self, val: float) -> None:
        self._reg(_CMD.VDPP_dpp_set_cfd_factor, t_double=val)

    def get_cfd_delay(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_cfd_delay)

    def set_cfd_delay(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_cfd_delay, t_int32=val)

    def get_cfd_time_window_low(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_cfd_time_window_low)

    def set_cfd_time_window_low(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_cfd_time_window_low, t_int32=val)

    def get_cfd_time_window_high(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_cfd_time_window_high)

    def set_cfd_time_window_high(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_cfd_time_window_high, t_int32=val)

    # ---- Charge Comparison ----

    def get_cc_enable(self) -> bool:
        return self._bool(_CMD.VDPP_dpp_get_cc_enable)

    def set_cc_enable(self, val: bool) -> None:
        self._reg(_CMD.VDPP_dpp_set_cc_enable, t_int32=int(val))

    def get_cc_time(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_cc_time)

    def set_cc_time(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_cc_time, t_int32=val)

    # ---- PSD Zero Crossing ----

    def get_psd_zc_enable(self) -> bool:
        return self._bool(_CMD.VDPP_dpp_get_psd_zc_enable)

    def set_psd_zc_enable(self, val: bool) -> None:
        self._reg(_CMD.VDPP_dpp_set_psd_zc_enable, t_int32=int(val))

    def get_psd_zc_mode(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_psd_zc_mode)

    def set_psd_zc_mode(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_psd_zc_mode, t_int32=val)

    def get_psd_zc_time_window_low(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_psd_zc_time_window_low)

    def set_psd_zc_time_window_low(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_psd_zc_time_window_low, t_int32=val)

    def get_psd_zc_time_window_high(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_psd_zc_time_window_high)

    def set_psd_zc_time_window_high(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_psd_zc_time_window_high, t_int32=val)

    # ---- pulse memory ----

    def get_debug_signal_selectors(self) -> tuple[int, ...]:
        """The legacy gRPC API exposes the original eight selector codes."""
        return tuple(range(8))

    def get_mem1_sig_select(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_mem1_sig_select)

    def set_mem1_sig_select(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_mem1_sig_select, t_int32=val)

    def get_mem2_sig_select(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_mem2_sig_select)

    def set_mem2_sig_select(self, val: int) -> None:
        self._reg(_CMD.VDPP_dpp_set_mem2_sig_select, t_int32=val)

    def get_mem_amount(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_mem_amount, no_channel=True)

    def get_mem_frame_sizes(self) -> np.ndarray:
        resp = self._reg(_CMD.VDPP_dpp_get_mem_frame_sizes, no_channel=True)
        return np.asarray(resp.uint32_array, dtype=np.uint32)

    def read_waveform_banks(self) -> tuple[np.ndarray, np.ndarray]:
        resp = self._stub.DPP_reg_access(
            _Msg(
                command=_CMD.VDPP_dpp_get_mem_frame,
                value=_Val(channel=self._ch),
            )
        )
        if resp.WhichOneof("response") == "error_response":
            raise RuntimeError(resp.error_response.message)
        return (
            np.asarray(resp.int32_array,  dtype=np.int16),
            np.asarray(resp.int32_array2, dtype=np.int16),
        )

    # ---- histogram ----

    def read_histogram(self) -> np.ndarray:
        resp = self._reg(_CMD.VDPP_dpp_get_mem_hist_frame)
        return np.asarray(resp.uint32_array, dtype=np.uint32)

    def clear_histogram(self) -> None:
        self._reg(_CMD.VDPP_dpp_clear_mem_hist_frame)

    def get_histogram_size(self) -> int:
        return self._int(_CMD.VDPP_dpp_get_mem_hist_frame_size, no_channel=True)

    # ---- sync trigger (global — no channel) ----

    def get_sync_enable(self) -> bool:
        return self._bool(_CMD.VDPP_sync_get_enable, no_channel=True)

    def set_sync_enable(self, val: bool) -> None:
        self._reg(_CMD.VDPP_sync_set_enable, t_int32=int(val), no_channel=True)

    def get_sync_sw_trig(self) -> int:
        return self._int(_CMD.VDPP_sync_get_sw_trig, no_channel=True)

    def set_sync_sw_trig(self, val: int) -> None:
        self._reg(_CMD.VDPP_sync_set_sw_trig, t_int32=val, no_channel=True)

    def get_sync_trig_src(self) -> int:
        return self._int(_CMD.VDPP_sync_get_trig_src, no_channel=True)

    def set_sync_trig_src(self, val: int) -> None:
        self._reg(_CMD.VDPP_sync_set_trig_src, t_int32=val, no_channel=True)

    def get_sync_timestamp(self) -> int:
        return self._int(_CMD.VDPP_sync_get_timestamp, no_channel=True)
