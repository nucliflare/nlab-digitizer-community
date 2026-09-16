from __future__ import annotations

import logging

from .backends.base import DigitizerBackend, IDSBackend
from .backends.grpc_backend import GrpcDigitizerBackend
from .diagnostics import GlobalDiagnosticReading
from .dma import IIOMcaDmaStreamer, IIOScopeDmaStreamer, McaDmaStreamer, ScopeDmaStreamer
from .hv import HVSupply
from .mca import MultiChannelAnalyzer
from .scope import Scope

log = logging.getLogger(__name__)


class Digitizer:
    """Entry point for instrument access.

    Construct via a factory classmethod, then access subsystems through
    .scope, .mca, and .hv.  The scope/mca share one gRPC service
    (DPP_reg_access on port 50050); the HV supply uses a separate IDS
    service (IDS_access on port 50040) on the **same physical device**.
    The split is a legacy issue that will be merged in a future firmware
    revision.

    Hardware I/O uses explicit get_*/set_* methods rather than properties.
    This makes the I/O cost visible at call sites and avoids surprising side
    effects inside expressions.

    Example — gRPC::

        d = Digitizer.from_grpc(channel=1)
        d.scope.set_trigger_level(5000)
        d.mca.set_energy_bin(4)
        d.hv.set_voltage(30.0)

    Example — without HV (e.g. bench testing DPP only)::

        d = Digitizer.from_grpc(channel=1, with_ids=False)
        # d.hv is None
    """

    def __init__(
        self,
        backend: DigitizerBackend,
        ids_backend: IDSBackend | None = None,
        scope_dma: ScopeDmaStreamer | IIOScopeDmaStreamer | None = None,
        mca_dma: McaDmaStreamer | IIOMcaDmaStreamer | None = None,
    ) -> None:
        self._backend = backend
        self._ids_backend = ids_backend
        self.scope = Scope(backend)
        self.mca   = MultiChannelAnalyzer(backend)
        self.hv: HVSupply | None = HVSupply(ids_backend) if ids_backend else None
        self.scope_dma = scope_dma
        self.mca_dma = mca_dma

    def close(self) -> None:
        if self.hv is not None:
            self.hv.safe_shutdown()
        self._backend.close()
        if self._ids_backend is not None:
            self._ids_backend.close()

    def mca_available(self) -> bool:
        """Whether d.mca's methods are expected to work on this channel.

        True for backends that always implement MCABackend in full (e.g.
        gRPC). Backends where MCA hardware presence varies per channel
        (currently only the IIO backend, whose pulse-processor/input-filter
        devices depend on the firmware build) opt in via an optional
        mca_hardware_present() extension method; delegating here instead of
        checking the backend type by name keeps this generic across future
        backends with the same variability.
        """
        check = getattr(self._backend, "mca_hardware_present", None)
        return check() if check is not None else True

    def get_global_diagnostics(self) -> list[GlobalDiagnosticReading]:
        """Read sensors shared by the complete digitizer, not one channel."""
        if self.hv is None:
            return []
        return self.hv.get_global_diagnostics()

    @classmethod
    def from_grpc(
        cls,
        channel: int,
        hostname: str = "192.168.10.20",
        port: int = 50050,
        *,
        ids_port: int = 50040,
        with_ids: bool = True,
    ) -> Digitizer:
        """Create a Digitizer backed by gRPC.

        Both the DPP and IDS services run on the same device (same
        hostname, different ports).  Pass with_ids=False to skip the
        IDS connection (d.hv will be None).
        """
        from .backends.grpc_ids_backend import GrpcIDSBackend

        dpp = GrpcDigitizerBackend(channel, hostname, port)
        ids = GrpcIDSBackend(channel, hostname, ids_port) if with_ids else None
        scope_dma = ScopeDmaStreamer(channel, hostname)
        mca_dma = McaDmaStreamer(channel, hostname)
        return cls(dpp, ids, scope_dma, mca_dma)

    @classmethod
    def from_iio(
        cls,
        channel: int,
        uri: str = "ip:192.168.10.128:30431",
        *,
        with_ids: bool = True,
    ) -> Digitizer:
        """Create a Digitizer backed by the on-FPGA IIO device tree.

        The HV module is accessed through its AD5686R, MCP3564(R), TMP117,
        and ADS5407 IIO devices; pass ``with_ids=False`` to leave ``d.hv``
        as ``None``. If a required IDS device is absent, construction logs a
        warning and also leaves ``d.hv`` as ``None`` so scope/MCA operation
        remains available. The removed SiPM controls are reported unavailable.
        When a channel exposes
        both vdpp_pulse_processor and vdpp_lm_frame, d.mca_dma is an
        IIOMcaDmaStreamer implementing the fixed 1024-record lifecycle in
        mca-architecture.md; it remains None on older firmware without
        lm_frame. d.mca itself is usable: IIODigitizerBackend
        implements MCABackend against vdpp-pulse-processor.c/
        vdpp-input-filter.c, except for a handful of methods with no
        matching hardware register (see iio_backend.py's module docstring)
        — and only if this channel's firmware actually has those two
        devices; if not, d.mca's methods raise RuntimeError instead of
        NotImplementedError, since that's a firmware/device-tree gap on a
        specific board, not a gap in this backend.
        d.scope_dma is an IIOScopeDmaStreamer (see dma.py) — pulls
        full-resolution frames by looping read_dma_raw_frame() rather than
        subscribing to a continuous push like the gRPC ZMQ streamers, since
        the IIO scope core has no continuous-streaming hardware path.
        """
        from .backends.iio_backend import IIODigitizerBackend
        from .backends.iio_ids_backend import IIOIDSBackend, IIOIDSUnavailableError

        backend = IIODigitizerBackend(channel, uri)
        ids = None
        if with_ids:
            try:
                ids = IIOIDSBackend(channel, uri)
            except IIOIDSUnavailableError as exc:
                # Scope and MCA are independent IIO cores. A firmware image
                # may legitimately omit one of the IDS converters/sensors;
                # keep the acquired DPP backend and disable only this
                # channel's PSU functionality.
                log.warning(
                    "IIO channel %d: IDS/PSU unavailable; continuing without it: %s",
                    channel,
                    exc,
                )
        scope_dma = IIOScopeDmaStreamer(backend, channel)
        mca_dma = (
            IIOMcaDmaStreamer(backend, channel)
            if backend.mca_dma_hardware_present()
            else None
        )
        return cls(backend, ids, scope_dma=scope_dma, mca_dma=mca_dma)
