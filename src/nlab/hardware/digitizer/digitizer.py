from __future__ import annotations

import logging

from .backends.base import DigitizerBackend, IDSBackend
from .diagnostics import GlobalDiagnosticReading
from .dma import IIOMcaDmaStreamer, IIOScopeDmaStreamer
from .hv import HVSupply
from .mca import MultiChannelAnalyzer
from .scope import Scope

log = logging.getLogger(__name__)


class Digitizer:
    """Entry point for instrument access.

    Construct with :meth:`from_iio`, then access subsystems through
    ``scope``, ``mca``, and ``hv``. Scope/MCA and IDS/HV use the digitizer's
    IIO device tree through independent context owners where threading or
    blocking I/O requires it.

    Hardware I/O uses explicit get_*/set_* methods rather than properties.
    This makes the I/O cost visible at call sites and avoids surprising side
    effects inside expressions.

    Example::

        d = Digitizer.from_iio(channel=0, uri="ip:192.168.3.1:30431")
        d.scope.set_trigger_level(5000)
        d.mca.set_energy_bin(4)
        d.hv.set_voltage(30.0)

    Example — without HV (e.g. bench testing DPP only)::

        d = Digitizer.from_iio(channel=0, with_ids=False)
        # d.hv is None
    """

    def __init__(
        self,
        backend: DigitizerBackend,
        ids_backend: IDSBackend | None = None,
        scope_dma: IIOScopeDmaStreamer | None = None,
        mca_dma: IIOMcaDmaStreamer | None = None,
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

        IIO pulse-processor/input-filter devices depend on the firmware
        build. Backends with variable MCA presence opt in via an optional
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
        d.scope_dma is an IIOScopeDmaStreamer (see dma.py) and pulls
        full-resolution frames by looping read_dma_raw_frame().
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
