"""gRPC backend for the IDS subsystem.

Same physical device as the DPP engine but exposed as a separate gRPC
service on a different port (default :50040).  Uses its own proto
(IDS.proto) and command enum.  This split is a legacy issue — the two
services will be merged in a future firmware revision.
"""

import logging
from typing import Any

import base_pb2 as bsp
import grpc
import IDS_pb2 as ids
import IDS_pb2_grpc

from ..diagnostics import GlobalDiagnosticReading
from .base import IDSBackend

log = logging.getLogger(__name__)

_CMD: Any = ids.IDSMessage
_Msg: Any = ids.IDSMessage
_Val: Any = bsp.ValueMessage


class GrpcIDSBackend(IDSBackend):
    """Single gRPC connection implementing IDSBackend."""

    def __init__(
        self,
        channel: int,
        hostname: str = "192.168.10.20",
        port: int = 50040,
    ) -> None:
        self._ch = channel
        self._gchannel = grpc.insecure_channel(f"{hostname}:{port}")
        self._stub = IDS_pb2_grpc.IDSServerStub(self._gchannel)
        log.info("IDS gRPC backend: connected ch%d to %s:%d", channel, hostname, port)

    def close(self) -> None:
        log.info("IDS gRPC backend: closing ch%d", self._ch)
        self._gchannel.close()

    def _reg(
        self,
        cmd: int,
        *,
        t_int32: int | None = None,
        t_uint32: int | None = None,
        t_double: float | None = None,
        no_channel: bool = False,
    ):
        kw: dict = {}
        if t_int32 is not None:
            kw["t_int32"] = int(t_int32)
        elif t_uint32 is not None:
            kw["t_uint32"] = int(t_uint32)
        elif t_double is not None:
            kw["t_double"] = float(t_double)
        if not no_channel:
            kw["channel"] = self._ch

        resp = self._stub.IDS_access(
            _Msg(command=cmd, value=_Val(**kw))
        )
        if resp.WhichOneof("response") == "error_response":
            raise RuntimeError(resp.error_response.message)
        return resp

    def _int(self, cmd: int, **kw: object) -> int:
        resp = self._reg(cmd, **kw)  # type: ignore[arg-type]
        return int(getattr(resp.value, resp.value.WhichOneof("value")))

    def _float(self, cmd: int, **kw: object) -> float:
        resp = self._reg(cmd, **kw)  # type: ignore[arg-type]
        return float(getattr(resp.value, resp.value.WhichOneof("value")))

    def _bool(self, cmd: int, **kw: object) -> bool:
        resp = self._reg(cmd, **kw)  # type: ignore[arg-type]
        return bool(getattr(resp.value, resp.value.WhichOneof("value")))

    # ---- versions ----

    def get_versions(self) -> list[int]:
        resp = self._reg(_CMD.VDPP_IDS_get_versions)
        return list(resp.value.ids_ip)

    def get_ads_temp(self) -> float:
        return self._float(_CMD.VDPP_IDS_ADS5407_get_temp, no_channel=True)

    def get_global_diagnostics(self) -> list[GlobalDiagnosticReading]:
        """The legacy IDS service exposes only the common ADS temperature."""
        return [
            GlobalDiagnosticReading(
                key="ads_temperature",
                label="ADS5407 temperature",
                value=self.get_ads_temp(),
                unit="°C",
                precision=1,
            )
        ]

    # ---- SiPM bias supply ----

    def set_sipm_enable(self, val: int) -> None:
        self._reg(_CMD.VDPP_IDS_sipm_set_enable, t_int32=val)

    def set_sipm_voltage(self, val: float) -> None:
        self._reg(_CMD.VDPP_IDS_sipm_set_voltage, t_double=val)

    def get_sipm_adc_voltage(self) -> float:
        return self._float(_CMD.VDPP_IDS_sipm_get_adc_voltage)

    def get_sipm_adc_current(self) -> float:
        return self._float(_CMD.VDPP_IDS_sipm_get_adc_current)

    def get_sipm_overload(self) -> int:
        return self._int(_CMD.VDPP_IDS_sipm_get_overload)

    # ---- SiPM temperature compensation ----

    def set_sipm_compens_ct(self, val: float) -> None:
        self._reg(_CMD.VDPP_IDS_sipm_compens_set_c_t, t_double=val)

    def set_sipm_compens_tref(self, val: float) -> None:
        self._reg(_CMD.VDPP_IDS_sipm_compens_set_t_ref, t_double=val)

    def set_sipm_compens_mode(self, val: int) -> None:
        self._reg(_CMD.VDPP_IDS_sipm_compens_set_mode, t_int32=val)

    def get_sipm_compens_output(self) -> float:
        return self._float(_CMD.VDPP_IDS_sipm_compens_get_output)

    # ---- HV bias supply ----

    def set_hv_voltage(self, val: float) -> None:
        self._reg(_CMD.VDPP_IDS_hv_set_voltage, t_double=val)

    def get_hv_adc_voltage(self) -> float:
        return self._float(_CMD.VDPP_IDS_hv_get_adc_voltage)

    # ---- HV temperature compensation ----

    def set_hv_compens_ct(self, val: float) -> None:
        self._reg(_CMD.VDPP_IDS_hv_compens_set_c_t, t_double=val)

    def set_hv_compens_tref(self, val: float) -> None:
        self._reg(_CMD.VDPP_IDS_hv_compens_set_t_ref, t_double=val)

    def set_hv_compens_mode(self, val: int) -> None:
        self._reg(_CMD.VDPP_IDS_hv_compens_set_mode, t_int32=val)

    def get_hv_compens_output(self) -> float:
        return self._float(_CMD.VDPP_IDS_hv_compens_get_output)

    # ---- temperature sensors ----

    def get_temp_analog(self) -> float:
        return self._float(_CMD.VDPP_IDS_temp_analog_get_adc_measure)

    def set_temp_digital_enable(self, val: int) -> None:
        self._reg(_CMD.VDPP_IDS_temp_digital_set_enable, t_uint32=val)

    def get_temp_digital_status(self) -> int:
        return self._int(_CMD.VDPP_IDS_temp_digital_get_communication_status)

    def get_temp_digital(self) -> float:
        return self._float(_CMD.VDPP_IDS_temp_digital_get_measure)
