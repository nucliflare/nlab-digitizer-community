"""Direct IIO control and DMA for Digitizer 4.0.

MCA uses parameter ABI2 (PetaLinux c546f14): ADC samples are native 2 ns
positions, C delay is an exponent, and trapezoid beta is a raw u32.
Register encoding belongs to the kernel. Historical ABI1 live checks do not
qualify ABI2. Scope raw/mean4 and structured periodic sums remain distinct.
"""

from __future__ import annotations

import ctypes
import errno
import logging
import math
import threading
import time
from collections.abc import Callable

import iio
import numpy as np

from ..acquisition_profile import AcquisitionProfile
from ..current_monitor import CurrentMonitorClient
from ..device_information import DigitizerInformation, collect_device_information
from ..dma import ScopeDmaGeometry
from ..iio_listmode import IIO_LM_EVENT_DTYPE, VDPP_ZC_CALC_SCHEMA
from ..iio_scope_stream import IiodScopeStream
from ..mca_parameters import PARAMETERS, validate_parameters
from ..scope_sums import ScopeSumGeometry
from .base import DigitizerBackend

log = logging.getLogger(__name__)


class DmaReaderActiveError(RuntimeError):
    """A native reader still owns its buffer and context after cancellation."""


# Registers genuinely do not exist in the MCA driver sources (checked
# against every #define/PP_F_* in vdpp-pulse-processor.c and every attribute
# in vdpp-input-filter.c) -- these methods stay stubbed rather than guessing
# at a mapping. See each raise site's docstring for the specific reasoning.
_MCA_EVENTS_LOST_UNSUPPORTED = (
    "IIO MCA backend: vdpp-pulse-processor.c (IP version 101) has no "
    "events-lost counter register -- throughput_error_counter is a "
    "different, separate statistic and would be misleading to alias here."
)
_MCA_HISTOGRAM_CLEAR_UNSUPPORTED = (
    "IIO MCA backend: vdpp-pulse-processor.c has no histogram-clear "
    "register. Per mca-architecture.md, this driver revision deliberately "
    "replaced the legacy library's implicit stop/start-based clear with "
    "explicit start()/stop() -- reintroducing an implicit restart here "
    "would silently reset elapsed_time/statistics too, not just the "
    "spectrum, so that trade-off is left to the caller."
)

_SCOPE_DEVICE_NAME = "vdpp_scope"

# ``vdpp-scope.c`` serializes viewer samples with ``"%d "`` into one
# 4096-byte sysfs page and stops once fewer than 16 bytes remain. A signed
# int16 needs at most seven bytes including its trailing space. This is the
# largest frame length whose decimated (one entry per four ADC samples)
# viewer representation is guaranteed to fit regardless of signal values.
# The DMA path is binary and retains the full 8188-sample hardware range.
_SCOPE_VIEWER_PAGE_BYTES = 4096
_SCOPE_VIEWER_TAIL_BYTES = 16
_SCOPE_VIEWER_MAX_ENTRY_BYTES = len("-32768 ")
_SCOPE_VIEWER_SAFE_ENTRIES = (
    _SCOPE_VIEWER_PAGE_BYTES - _SCOPE_VIEWER_TAIL_BYTES
) // _SCOPE_VIEWER_MAX_ENTRY_BYTES
_SCOPE_VIEWER_SAFE_FRAME_SAMPLES = _SCOPE_VIEWER_SAFE_ENTRIES * 4
_SCOPE_MIN_FRAME_SAMPLES = 4
_SCOPE_MAX_FRAME_SAMPLES = 8188

# The AFE DAC now has a real driver: vdpp_afe_dac, added to the device tree
# after this backend was first written (the ad5686r device present on the
# live tree is a different, unrelated DAC -- not this one). Confirmed live:
# a single vdpp_afe_dac device instance (not one per scope, unlike
# vdpp_scope) exposes two output channels, voltage0 and voltage1, one per
# scope channel -- each with `raw` (the underlying DAC output code, full
# range, not what the GUI's DC-offset control means) and `baseline` (the
# actual DC-offset control: 0-1023 confirmed via voltage0/1's
# baseline_available, matching Scope's DAC_VALUE spec of 0-1023).
_DAC_DEVICE_NAME = "vdpp_afe_dac"

# MCA: two more per-channel IIO devices, one pipeline stage each (see
# mca-architecture.md: "ADC -> input_filters -> pulse_processor -> ...").
# Both device names are literal strings hardcoded once in their driver's
# probe() (indio_dev->name = "..."), identical across every channel
# instance -- confirmed from vdpp-pulse-processor.c / vdpp-input-filter.c --
# Current pulse-processor/lm-frame drivers additionally expose channel_index;
# _device_for_channel() uses it as the authoritative remote pairing key.
_PULSE_PROCESSOR_DEVICE_NAME = "vdpp_pulse_processor"
_INPUT_FILTER_DEVICE_NAME = "vdpp_input_filter"
_LM_FRAME_DEVICE_NAME = "vdpp_lm_frame"

# One channel-independent device shared by both MCA pulse processors. Per
# vdpp-sync-trigger.c, this is a level-sensitive common measurement-start
# controller, not an IIO trigger provider and not a scope trigger.
_SYNC_TRIGGER_DEVICE_NAME = "vdpp_sync_trigger"

# Current vdpp-lm-frame.c: Linux transports one opaque repeated-u8 scan
# element (u8[16]), not the superseded five semantic scan channels. Linux
# 5.15 tracks IIO_TIMESTAMP separately from ordinary scan masks, so the old
# five-bit mask could never be accepted and buffer creation returned EINVAL.
# One refill remains one unchanged 16 KiB frame. _LM_EVENT_DTYPE is only the
# application's selected decoder for the currently deployed producer schema;
# it is not the kernel/IIO transport ABI.
_LM_FRAME_RECORDS = 1024
_LM_RECORD_BYTES = 16
_LM_FRAME_BYTES = _LM_FRAME_RECORDS * _LM_RECORD_BYTES
_LM_IP_VERSION = 121
_LM_KERNEL_BUFFER_COUNT = 8
# Old or capability-unknown Scope images must retain one block.  The current
# provider-qualified driver advertises a recommended/max depth through IIO;
# use that only after validating the complete capability tuple.
_SCOPE_FALLBACK_KERNEL_BUFFER_COUNT = 1
# The IP122/.128 coverage sweep found that 64-frame READBUF requests maximize
# receiver throughput without changing the exact-frame DMA ABI. The server
# still emits its bounded batches as frames become available.
_SCOPE_IIOD_BATCH_FRAMES = 64
_LM_RECORD_LAYOUT = "opaque[16]"
_LM_EVENT_DTYPE = IIO_LM_EVENT_DTYPE

# vdpp-pulse-processor.c: PP_MEM_DEBUG_ENTRIES / PP_MEM_HISTOGRAM_ENTRIES.
# Both memories are fixed-size (no sysfs attribute reports these counts
# directly -- they're only implicit in debug_data's/histogram_data's fixed
# bin_attribute byte length), so read the driver's #define values directly
# rather than trying to derive them from a byte count at runtime.
_PP_MEM_DEBUG_ENTRIES = 2048
_PP_MEM_HISTOGRAM_ENTRIES = 16384
_PP_DEBUG_SNAPSHOT_BYTES = 2 * _PP_MEM_DEBUG_ENTRIES * 2  # two s16 memories
_PP_HISTOGRAM_BYTES = _PP_MEM_HISTOGRAM_ENTRIES * 4  # one u32 memory
_PP_HISTOGRAM_CHUNK_COUNT = 4
_PP_HISTOGRAM_CHUNK_BYTES = _PP_HISTOGRAM_BYTES // _PP_HISTOGRAM_CHUNK_COUNT
_PP_HISTOGRAM_CHUNK_NAMES = tuple(
    f"histogram_data{index}" for index in range(_PP_HISTOGRAM_CHUNK_COUNT)
)

# vdpp-input-filter.c: VDPP_INPUT_FILTER_COEFFICIENTS -- also independently
# readable live via the fir_coefficient_count attribute (get_lp_coeffs_size()
# does that instead of trusting this constant), used here only to give
# set_lp_coeffs() a clear client-side error instead of an opaque EINVAL from
# the wire when the caller passes the wrong number of coefficients.
_INPUT_FILTER_FIR_COEFFICIENTS = 12

# Matches the reference stub's (scope_backend_iio.py) DMA_CLOSE_DRAIN_SECONDS /
# DMA_CLOSE_JOIN_SECONDS. Our own close path was missing this drain step
# entirely (see _drain_for_close()'s docstring) -- ported from the stub once
# that gap was identified by comparing against it directly.
_DMA_CLOSE_DRAIN_SECONDS = 0.050
_DMA_CLOSE_JOIN_SECONDS = 1.0
# Matches the reference stub's DMA_CLOSE_RELEASE_SECONDS /
# DMA_CLOSE_RELEASE_POLL_SECONDS -- 3 seconds per user-api.md's cleanup
# sequence ("wait, for at most 3 seconds, until both dma_buffer_active=0
# and dma_enable=0"), not the 1.25 s an earlier revision of that same
# reference used. The stock Xilinx DMA terminate path polls for HALTED for
# about one second; with a network context iiod can acknowledge buffer
# destruction before that target-side cleanup becomes visible through
# sysfs, so _wait_for_dma_release() polls for this long rather than
# trusting the buffer object being gone.
_DMA_CLOSE_RELEASE_SECONDS = 3.0
_DMA_CLOSE_RELEASE_POLL_SECONDS = 0.050

# Per scope-architecture.md: "Before viewer-only operation transitions to
# DMA, the reference API writes enable=0, waits 1 ms and passively verifies
# that both DMA ownership gates are zero." Distinct from
# _DMA_CLOSE_RELEASE_SECONDS above -- this one covers the viewer-only FSM
# path specifically, whose worst-case frame is far under 1 ms and cannot be
# extended by AXI backpressure (no AXI receiver in viewer-only mode), not
# the up-to-3s post-buffer-destruction teardown.
_VIEWER_TO_DMA_SETTLE_SECONDS = 0.001

# How long to wait for _start_reader_then_enable()'s background reader
# thread to confirm it has entered its blocking refill() call before
# giving up on writing enable=1 at all. Reusing _DMA_CLOSE_JOIN_SECONDS's
# value (thread-start confirmation, same order of magnitude) rather than
# inventing an unrelated number.
_DMA_FIRST_REFILL_ENTER_SECONDS = _DMA_CLOSE_JOIN_SECONDS

# mca-architecture.md/user-api.md define one second without a completed
# block as the conservative end-of-tail boundary because lm_frame v121 has
# no drained/end-seen status. Unlike the scope's fixed 50 ms close window,
# this timeout resets after every list-mode block received during drain.
_LM_CLOSE_INACTIVITY_SECONDS = 1.0
_LM_CLOSE_POLL_SECONDS = 0.050
_LM_CLOSE_JOIN_SECONDS = 1.25
_LM_CLOSE_RELEASE_SECONDS = 3.0
# Operational grace for a nonblocking STOP status transition, not an A07 fence.
_MCA_STOP_STATUS_SECONDS = 0.25
# A network refill can report either the target's Linux ETIMEDOUT=110 through
# the iiod protocol or the local socket timeout. The latter is also 110 on
# Linux but WSAETIMEDOUT=10060 on Windows. Both mean "no block arrived in the
# current wait window"; comparing only with the host errno constant caused
# target-side 110 to terminate legitimate external-trigger waits on Windows.
_REMOTE_ETIMEDOUT = 110
_LIBIIO_TIMEOUT_ERRNOS = frozenset((_REMOTE_ETIMEDOUT, errno.ETIMEDOUT))
_LM_EXPECTED_CANCEL_ERRNOS = frozenset((9, 125, *_LIBIIO_TIMEOUT_ERRNOS))


def _disable_mca_dma_client_timeout(context: iio.Context) -> None:
    """Disable the client-side timeout on the list-mode stream context.

    The project uses pylibiio/libiio 0.x (currently 0.26). Its documented
    ``iio_context_set_timeout()`` contract assigns zero to "no timeout".
    This affects the client's dedicated stream socket. The v0.26 network
    backend opens the device stream on another iiod connection, however, and
    does not forward the context's TIMEOUT command to it (network.c's
    network_open()/network_set_timeout()). That remote connection can still
    return ETIMEDOUT and is handled by _refill_mca_dma_buffer().
    """
    context.set_timeout(0)


def _device_number(device: iio.Device) -> int:
    """Return the numeric part of an ``iio:deviceN`` identifier.

    This matches the PetaLinux reference client and makes index-based selection
    deterministic. It does *not* turn probe order into physical
    channel identity: user-api.md explicitly says that A/B pairing requires
    platform-device links and the ewt,pulse-processor phandle, neither of
    which is exposed by the tested remote IIO context.
    """
    try:
        return int(device.id.rsplit("device", 1)[1])
    except (AttributeError, IndexError, ValueError):
        return 1 << 30


def _devices_named(context: iio.Context, name: str) -> list[iio.Device]:
    return sorted(
        (device for device in context.devices if device.name == name),
        key=_device_number,
    )


def _device_for_channel(
    context: iio.Context,
    name: str,
    channel: int,
) -> iio.Device | None:
    """Return a same-name device by stable channel index when available.

    Current MCA drivers expose ``channel_index`` specifically so remote iiod
    clients can pair ``vdpp_pulse_processor`` with ``vdpp_lm_frame`` without
    relying on probe-derived ``iio:deviceN`` identifiers. Older drivers (and
    the current scope/input-filter drivers) lack that attribute, so retain the
    deterministic sorted-device fallback for compatibility.
    """
    devices = _devices_named(context, name)
    indexed = [device for device in devices if "channel_index" in device.attrs]
    if indexed:
        if len(indexed) != len(devices):
            raise RuntimeError(
                f"'{name}' exposes channel_index on only {len(indexed)} of "
                f"{len(devices)} devices; refusing ambiguous channel pairing"
            )
        matches = [
            device
            for device in indexed
            if int(device.attrs["channel_index"].value) == channel
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one '{name}' device with channel_index="
                f"{channel}, found {len(matches)}"
            )
        return matches[0]
    return devices[channel] if channel < len(devices) else None


class _IIOCurrentMonitorClient(CurrentMonitorClient):
    """Worker-owned direct-mode reader for one input-filter IIR register."""

    def __init__(self, channel: int, uri: str) -> None:
        self._channel_index = channel
        self._uri = uri
        self._device: iio.Device | None = None
        self._input_channel: iio.Channel | None = None
        context = iio.Context(uri)
        self._context: iio.Context | None = context
        device = _device_for_channel(
            context,
            _INPUT_FILTER_DEVICE_NAME,
            channel,
        )
        if device is None:
            self.close()
            raise RuntimeError(
                f"no {_INPUT_FILTER_DEVICE_NAME} device for channel {channel} at {uri}"
            )
        input_channel = device.find_channel("voltage0")
        if input_channel is None:
            self.close()
            raise RuntimeError(
                f"{_INPUT_FILTER_DEVICE_NAME} channel {channel} has no voltage0 channel"
            )
        self._device = device
        self._input_channel = input_channel

    def read_raw(self) -> int:
        """Read the driver's signed 16-bit IIR code without rescaling it.

        ``vdpp-input-filter.c::vdpp_input_filter_read_raw()`` casts the low
        register halfword to ``s16`` and advertises a separate 1/32768 scale.
        The monitor keeps the code intact because conversion to amperes also
        needs the deployment-specific sensor/front-end calibration.
        """
        channel = self._input_channel
        if channel is None:
            raise RuntimeError("current-monitor IIO client is closed")
        return int(channel.attrs["raw"].value)

    def close(self) -> None:
        # pylibiio 0.x releases devices with their owning Context.  Drop
        # child references first so the network connection can be destroyed
        # deterministically when the context reference is cleared.
        self._input_channel = None
        self._device = None
        self._context = None

# vdpp-scope.c's trigger_mode is a plain integer register (0..4), not a
# string attribute like the previous driver. Confirmed from the driver
# source: SCOPE_TRIG_LEVEL_ABOVE=0, SCOPE_TRIG_LEVEL_BELOW=1,
# SCOPE_TRIG_EDGE_FALL=2, SCOPE_TRIG_EDGE_RISE=3, SCOPE_TRIG_PERIODIC=4 --
# these already match scope.TriggerMode's int values 1:1
# (ANY_ABOVE/ANY_BELOW/FALLING_EDGE/RISING_EDGE/TIMED), so get_edge/set_edge
# can pass the int straight through with no lookup table.


class IIODigitizerBackend(DigitizerBackend):
    """Scope/MCA backend for the rewritten IIO device set.

    MCA pulse-processor/list-frame instances are selected by the stable
    ``channel_index`` exported by current drivers. Same-name device sorting is
    retained only for older drivers and for scope/input-filter devices, which
    still expose no authoritative channel key through the remote context.
    """

    def __init__(
        self,
        channel: int,
        uri: str = "ip:192.168.10.128:30431",
        dac_name: str | None = _DAC_DEVICE_NAME,
        dac_channel: int | None = None,
        *,
        _scope_only: bool = False,
        profile: AcquisitionProfile | None = None,
    ) -> None:
        self._ch = channel
        self._uri = uri
        self._dac_name = dac_name
        self._dac_channel = dac_channel
        self._scope_only = _scope_only
        self._acquisition_profile = profile
        self._device_information: DigitizerInformation | None = None
        self._ctx = iio.Context(uri)

        scopes = _devices_named(self._ctx, _SCOPE_DEVICE_NAME)
        if channel >= len(scopes):
            raise RuntimeError(
                f"only {len(scopes)} '{_SCOPE_DEVICE_NAME}' device(s) found at "
                f"{uri}, channel {channel} out of range"
            )
        self._scope = scopes[channel]

        chan = self._scope.find_channel("voltage0")
        if chan is None:
            raise RuntimeError(f"{_SCOPE_DEVICE_NAME} has no voltage0 channel")
        self._chan = chan
        self._chan.enabled = True

        # DAC defaults to vdpp_afe_dac (see module-level comment above
        # _DAC_DEVICE_NAME) -- one device shared by both scope channels,
        # selected by output channel index rather than by device instance
        # like vdpp_scope. The legacy API defaults to the sorted scope index;
        # this does not prove physical wiring. Keep the application's established
        # mapping; an optional profile can override it for a different deployment.
        # Pass dac_name=None
        # to disable DAC support entirely (e.g. against a board without
        # this device yet).
        self._dac_dev = self._ctx.find_device(dac_name) if dac_name else None
        selected_output = profile.dac_output if profile is not None else (
            dac_channel if dac_channel is not None else channel
        )
        self._dac_ch = (
            self._dac_dev.find_channel(
                f"voltage{selected_output}", True,
            )
            if self._dac_dev is not None and selected_output is not None else None
        )

        # DMA capture buffer -- created lazily on first read_dma_frame()
        # call and reused across calls (see read_dma_frame's docstring for
        # why that's safe on this driver, unlike the previous one).
        #
        # Buffer sample count: exactly frame_samples, confirmed live via
        # dmesg after a board/driver update ("buffer length is 1024, must
        # be 512 for a 512 sample frame") -- this used to need 2 *
        # frame_samples (the dmaengine buffer core splitting the request
        # into two double-buffered blocks, with one refill() delivering
        # both), but the driver's required_length formula changed at some
        # point and one refill() now delivers exactly one frame. The current
        # remote transport batches requests, but each returned chunk and each
        # kernel DMA block remain exactly one frame.
        self._dma_buf: iio.Buffer | IiodScopeStream | None = None
        self._dma_buf_frame_samples: int | None = None
        self._dma_capture_transport: str | None = None
        self._dma_capture_kernel_buffers: int | None = None
        self._dma_capture_geometry: ScopeDmaGeometry | ScopeSumGeometry | None = None
        self._dma_last_timestamp: int | None = None
        # Set by IIOScopeDmaWorker.stop() through the streamer. Buffer.cancel()
        # is the libiio-supported way to interrupt a refill blocked in another
        # thread; this event lets that expected cancellation be distinguished
        # from a genuine EBADF/EIO capture failure.
        self._dma_stop_requested = threading.Event()

        # Set by _refill_dma_buffer() on a genuine EIO (not the expected -9
        # a close-time Buffer.cancel() produces) -- see its docstring and
        # acknowledge_dma_recovery()/dma_fault_is_latched().
        self._dma_fault_latched = False
        self._scope_dma_reader: threading.Thread | None = None

        # Auto Setup and the live Scope viewer need worker-owned connections
        # but no DMA or MCA devices. Returning here avoids opening the
        # backend's other role-specific contexts while preserving the
        # no-shared-context rule.
        if self._scope_only:
            log.info(
                "IIO backend: connected lightweight scope worker ch%d to %s",
                channel,
                uri,
            )
            return

        # A single IIO network context is not safe for concurrent use from
        # multiple threads. Confirmed live, twice, independently: this is
        # a *different* problem from the "never write dma_enable directly"
        # bug the driver author's architecture doc identifies -- that bug
        # is real and fixed (see scope_dma_active()'s docstring), but fixing
        # it and then reverting to a single context (on the theory that it
        # was the *only* problem) reintroduced this one. The GUI
        # deliberately restarts the viewer-refresh timer alongside every
        # DMA session (ScopeController._on_dma_ready() -- "DMA running":
        # raw DMA and viewer both active is a documented, supported
        # state), which runs viewer polling on a QThreadPool worker thread
        # concurrently with the DMA worker thread. The single-threaded
        # reference implementation this backend is modeled on never
        # exercises that combination, so its own single-context design
        # doesn't cover it. Two independent connections, one used only by
        # DMA-buffer-lifecycle code (_create_dma_buffer(),
        # _start_reader_then_enable(), _drain_for_close(),
        # _close_dma_buffer(), _wait_for_dma_release(), and their internal
        # enable/dma_enable/dma_buffer_active/frame_samples reads via the
        # _dma_* helpers below) and one used by everything reachable from
        # the GUI thread (self._scope, unchanged), means neither thread's
        # traffic can corrupt the other's, regardless of how the dma_enable
        # ownership rules are enforced on top of that.
        self._dma_ctx = iio.Context(uri)
        dma_scopes = _devices_named(self._dma_ctx, _SCOPE_DEVICE_NAME)
        self._dma_scope = dma_scopes[channel]
        dma_chan = self._dma_scope.find_channel("voltage0")
        if dma_chan is None:
            raise RuntimeError(f"{_SCOPE_DEVICE_NAME} has no voltage0 channel")
        dma_chan.enabled = True

        # MCA (pulse-processor + input-filter) -- optional, mirroring the
        # DAC's graceful-degradation pattern above: older firmware builds
        # that only have the scope core still connect fine, they just get
        # self._pp/self._input_filter = None and every MCABackend method
        # raises RuntimeError (via _pp_attr_get/_if_attr_get) instead of
        # crashing __init__. Both cores are instantiated once per hardware
        # channel (indio_dev->name is the identical literal string
        # "vdpp_pulse_processor"/"vdpp_input_filter" for every instance,
        # confirmed from vdpp-pulse-processor.c's pp_probe() and
        # vdpp-input-filter.c's vdpp_input_filter_probe()), so channel
        # Current pulse-processor drivers expose a stable channel_index over
        # iiod. _device_for_channel() uses it when present and keeps sorted
        # probe order only as an older-driver fallback.
        self._pp = _device_for_channel(
            self._ctx, _PULSE_PROCESSOR_DEVICE_NAME, channel,
        )

        input_filters = _devices_named(self._ctx, _INPUT_FILTER_DEVICE_NAME)
        self._input_filter = input_filters[channel] if channel < len(input_filters) else None
        self._input_filter_ch = (
            self._input_filter.find_channel("voltage0") if self._input_filter is not None else None
        )

        # TemperatureCorrectionWorker writes continuously from its own
        # QThread. It must not share the GUI/config context above or the MCA
        # polling context below; one remote IIO context per active thread is
        # a confirmed requirement for this backend.
        self._temperature_correction_ctx: iio.Context | None = None
        self._temperature_correction_filter: iio.Device | None = None

        # Trigger sync is one global core shared by both MCA channels, not a
        # per-channel device. sync_trigger_smoke_test.sh explicitly requires
        # exactly one instance. Keep absence compatible with older firmware,
        # but reject multiple instances because selecting either would make
        # channel-independent control ambiguous.
        sync_triggers = _devices_named(self._ctx, _SYNC_TRIGGER_DEVICE_NAME)
        if len(sync_triggers) > 1:
            raise RuntimeError(
                f"expected at most one '{_SYNC_TRIGGER_DEVICE_NAME}' device, "
                f"found {len(sync_triggers)}"
            )
        self._sync_trigger = sync_triggers[0] if sync_triggers else None

        # Software-only shadow for get/set_edge_det_coeff() -- see the
        # module docstring, no matching register exists in either driver.
        self._edge_det_coeff = 0

        # Second context dedicated to MCAWorker's background polling
        # thread (read_histogram(), read_waveform_banks(), the statistics
        # getters, get_measurement_in_progress()) -- same rationale as
        # self._dma_ctx above: MCAController wires those exact methods to
        # a QTimer running on its own QThread (MCAWorker), which now runs
        # concurrently with GUI-thread calls into this same backend
        # instance's config setters (and, during shutdown/reconnect,
        # MCAController._ensure_disarmed()'s own get_measurement_in_progress()
        # call can race the still-running worker thread -- see
        # MainWindowController._stop_all_workers()'s ordering). Only
        # opened when the pulse-processor core was actually found, so a
        # scope-only firmware build doesn't pay for a connection nothing
        # will use.
        if self._pp is not None:
            self._mca_ctx = iio.Context(uri)
            self._mca_pp = _device_for_channel(
                self._mca_ctx, _PULSE_PROCESSOR_DEVICE_NAME, channel,
            )
            if self._mca_pp is None:
                raise RuntimeError(
                    f"no {_PULSE_PROCESSOR_DEVICE_NAME} device for channel {channel} "
                    "in the MCA polling context"
                )
        else:
            self._mca_ctx = None
            self._mca_pp = None

        # List-mode DMA uses two more contexts with one transport role each.
        # The stream context owns only vdpp_lm_frame/Buffer.refill(); the
        # control context owns pulse_processor.enable and its gate reads.
        # mca_listmode_capture.py explicitly forbids multiplexing a blocked
        # refill and measurement control over one remote iiod connection.
        # MCA polling uses _mca_ctx and GUI/config calls use _ctx, so neither
        # is shared with the DMA worker either.
        self._mca_dma_ctx: iio.Context | None = None
        self._mca_dma_control_ctx: iio.Context | None = None
        self._mca_dma_control_lock = threading.RLock()
        self._mca_dma_pp: iio.Device | None = None
        self._lm_frame: iio.Device | None = None
        self._lm_status_device: iio.Device | None = None
        self._mca_dma_buf: iio.Buffer | None = None
        self._mca_dma_reader: threading.Thread | None = None
        self._mca_dma_fault_latched = False
        self._mca_stop_uncertain = False
        self._mca_cleanup_errors: list[dict[str, int | str]] = []
        self._mca_dma_stop_requested = threading.Event()
        self._mca_dma_lifecycle_lock = threading.RLock()
        self._mca_dma_cancel_timer: threading.Timer | None = None
        self._mca_dma_cancel_explained = False
        if self._pp is not None:
            mca_dma_ctx = iio.Context(uri)
            mca_dma_control_ctx = iio.Context(uri)
            mca_dma_pp = _device_for_channel(
                mca_dma_control_ctx, _PULSE_PROCESSOR_DEVICE_NAME, channel,
            )
            lm_frame = _device_for_channel(
                mca_dma_ctx, _LM_FRAME_DEVICE_NAME, channel,
            )
            if mca_dma_pp is not None and lm_frame is not None:
                # refill() may legitimately wait for an operator-driven
                # shared software trigger for an arbitrary amount of time.
                # Keep only the buffer stream unbounded; control I/O stays
                # on mca_dma_control_ctx with its normal network timeout.
                _disable_mca_dma_client_timeout(mca_dma_ctx)
                log.debug(
                    "IIO backend ch%d: MCA DMA external-trigger wait "
                    "configured; retryable timeout errnos=%s",
                    channel,
                    sorted(_LIBIIO_TIMEOUT_ERRNOS),
                )
                self._mca_dma_ctx = mca_dma_ctx
                self._mca_dma_control_ctx = mca_dma_control_ctx
                self._mca_dma_pp = mca_dma_pp
                self._lm_frame = lm_frame
                self._lm_status_device = _device_for_channel(
                    mca_dma_control_ctx, _LM_FRAME_DEVICE_NAME, channel,
                )

        log.info("IIO backend: connected ch%d (%s) to %s", channel, _SCOPE_DEVICE_NAME, uri)
        if self._pp is None:
            log.warning(
                "IIO backend ch%d: no %s device found -- MCA methods will raise "
                "RuntimeError until this channel's firmware includes the "
                "pulse-processor core", channel, _PULSE_PROCESSOR_DEVICE_NAME,
            )
        elif self._lm_frame is None:
            log.warning(
                "IIO backend ch%d: no paired %s device found -- MCA "
                "configuration works, but list-mode DMA is unavailable",
                channel, _LM_FRAME_DEVICE_NAME,
            )
        elif "channel_index" not in self._pp.attrs:
            log.warning(
                "IIO backend ch%d: same-name devices are selected by sorted "
                "IIO probe index because this older pulse-processor driver "
                "does not expose channel_index; physical A/B identity is not "
                "guaranteed",
                channel,
            )
        if self._sync_trigger is None:
            log.warning(
                "IIO backend ch%d: no %s device found -- MCA trigger-sync "
                "methods are unavailable",
                channel,
                _SYNC_TRIGGER_DEVICE_NAME,
            )
        # Capture identity before GUI/poll/DMA workers can use these handles.
        # About must never borrow an active worker's context or block the GUI
        # on a new remote request. Failure here cannot make acquisition unusable.
        try:
            producer_identity = None
            if self._mca_dma_pp is not None and self._lm_frame is not None:
                try:
                    producer_identity = self.mca_producer_identity()
                except Exception as exc:
                    producer_identity = {"decoder_selection": f"Unavailable: {exc}"}
            self._device_information = collect_device_information(
                self._ctx, uri=uri, channel=channel,
                devices={
                    "Scope": self._scope, "Pulse processor": self._pp,
                    "List mode": self._lm_frame, "Input filter": self._input_filter,
                    "AFE DAC": self._dac_dev, "Shared START": self._sync_trigger,
                },
                dac_output=str(self._dac_ch.id) if self._dac_ch is not None else None,
                configured_image=profile.firmware_id if profile is not None else "",
                producer_identity=producer_identity,
            )
        except Exception as exc:
            self._device_information = None
            log.warning("IIO ch%d: device information unavailable: %s", channel, exc)

    def get_device_information(self) -> DigitizerInformation | None:
        information = getattr(self, "_device_information", None)
        return information if isinstance(information, DigitizerInformation) else None

    def close(self) -> None:
        log.info("IIO backend: closing ch%d", self._ch)
        if self._scope_only:
            self._close_dma_buffer()
            return
        self._close_mca_dma_buffer()
        self._close_dma_buffer()

    def create_isolated_scope_backend(self) -> IIODigitizerBackend:
        """Open an independent IIO context for a Scope background worker.

        IIO contexts are intentionally not shared across threads; this is
        the same isolation rule used by Auto Setup, the viewer, and DMA.
        """
        return type(self)(
            self._ch,
            self._uri,
            dac_name=self._dac_name,
            dac_channel=self._dac_channel,
            _scope_only=True,
            profile=self._acquisition_profile,
        )

    def create_current_monitor_client(self) -> CurrentMonitorClient:
        """Open the monitor's independent, input-filter-only IIO context."""
        return _IIOCurrentMonitorClient(self._ch, self._uri)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_large_attr(self, name: str, size: int = 8192) -> bytes:
        """Work around pylibiio 0.25's DeviceAttr._read() hardcoding a
        1024-byte buffer for every attribute regardless of its real size
        (same issue this project hit against the previous driver rewrite).
        viewer_data is a text attribute the driver itself caps at
        PAGE_SIZE (4096 bytes, see vdpp-scope.c's viewer_data_show()),
        which already exceeds pylibiio's 1024-byte default -- read via the
        private ctypes binding with an 8192-byte buffer instead.
        """
        buf = ctypes.create_string_buffer(size)
        n = iio._d_read_attr(self._scope._device, name.encode("ascii"), buf, size)
        return buf.raw[:n]

    def _join_dma_reader(self, attribute: str, timeout: float) -> None:
        """Retain the native owner until join proves it has exited.

        PetaLinux c546f14 application-migration-guide.md, sections 4/5:
        cancellation and a bounded wait do not transfer buffer ownership.
        """
        worker = getattr(self, attribute, None)
        if worker is None:
            return
        worker.join(timeout)
        if worker.is_alive():
            if attribute == "_scope_dma_reader":
                self._dma_fault_latched = True
            else:
                self._mca_dma_fault_latched = True
            raise DmaReaderActiveError(
                "DMA reader did not stop; native buffer and context remain owned"
            )
        setattr(self, attribute, None)

    def _drain_for_close(self, buf: iio.Buffer | IiodScopeStream) -> int:
        """Keep refilling *buf* for a short window before cancelling it,
        instead of cancelling immediately.

        Ported from the reference stub's (scope_backend_iio.py)
        _drain_for_close(). If a frame the core already started sending is
        not consumed before the DMA receiver disappears, the core can be left
        blocked in its AXI send state -- not just for DMA, but for the shared
        state machine that viewer-only frames also depend on to return idle.

        A background thread keeps calling refill() (discarding the
        result -- this is drain, not read) until cancelled, while this
        thread sleeps DMA_CLOSE_DRAIN_SECONDS to give one already-in-
        flight frame time to land. cancel() then unblocks the worker's
        final blocking refill() the same way it always has. Returns the
        number of frames drained (0 or 1 in the common case).
        """
        stop = threading.Event()
        entered = threading.Event()
        errors: list[tuple[int, str]] = []
        drained: list[None] = []

        def refill_until_cancelled() -> None:
            entered.set()
            while not stop.is_set():
                try:
                    self._refill_scope_buffer(buf)
                except Exception as exc:
                    # cancel() is expected to interrupt the final
                    # blocking refill -- only an earlier failure is a
                    # real error.
                    if not (stop.is_set() and getattr(exc, "errno", None)
                            in _LM_EXPECTED_CANCEL_ERRNOS):
                        errors.append((getattr(exc, "errno", None) or errno.EIO, str(exc)))
                    return
                drained.append(None)

        worker = threading.Thread(
            target=refill_until_cancelled,
            name=f"vdpp-scope-ch{self._ch}-dma-drain",
            daemon=True,
        )
        self._scope_dma_reader = worker
        worker.start()
        try:
            if not entered.wait(_DMA_CLOSE_JOIN_SECONDS):
                raise RuntimeError("Scope DMA drain reader did not enter refill")
            time.sleep(_DMA_CLOSE_DRAIN_SECONDS)
        finally:
            stop.set()
            try:
                buf.cancel()
            finally:
                self._join_dma_reader("_scope_dma_reader", _DMA_CLOSE_JOIN_SECONDS)
        if errors:
            raise OSError(*errors[0]) from None
        log.debug(
            "IIO backend ch%d: DMA close-drain consumed %d frame(s)",
            self._ch, len(drained),
        )
        return len(drained)

    def _close_dma_buffer(self, *, drain: bool = True) -> None:
        buf = self._dma_buf
        if buf is None:
            return

        # Per the driver author's own architecture note (scope-architecture
        # .md): dma_enable is driver-owned state for this whole lifecycle --
        # "a GUI control labelled DMA represents buffer arm/close, not a
        # direct write to that register", and "writing dma_enable=0 before
        # buffer destruction is an application bug and is rejected with
        # EBUSY". An earlier revision of this method force-wrote both
        # enable=0 and dma_enable=0 here as a "safety net" -- that write was
        # itself the bug: it raced the driver's own teardown, and retrying
        # an operation the driver never intends to allow just spent the
        # whole retry budget on something that was never going to succeed.
        # Never write dma_enable directly. Disable the trigger gate, drain,
        # drop the buffer reference (which is what actually owns clearing
        # dma_enable board-side), then wait -- never write -- for that to
        # become visible.
        log.debug("IIO backend ch%d: closing DMA buffer", self._ch)
        errors: list[tuple[int, str]] = []
        try:
            if self._dma_get_enable():
                self._dma_set_enable(False)
        except Exception as exc:
            errors.append((getattr(exc, "errno", None) or errno.EIO, f"Stop: {exc}"))
            drain = False

        # A failed first refill can still own this buffer. Never create a
        # second reader, or destroy storage, until the old reader has exited.
        if getattr(self, "_scope_dma_reader", None) is not None:
            try:
                buf.cancel()
            finally:
                self._join_dma_reader("_scope_dma_reader", _DMA_CLOSE_JOIN_SECONDS)
            drain = False

        if drain and not isinstance(getattr(self, "_dma_capture_geometry", None), ScopeSumGeometry):
            try:
                self._drain_for_close(buf)
            except DmaReaderActiveError:
                raise
            except Exception as exc:
                errors.append((getattr(exc, "errno", None) or errno.EIO, f"Drain: {exc}"))
        else:
            # A requested cancellation or failed normal refill has no
            # trustworthy tail to drain. Cancel the existing descriptor and
            # proceed directly to buffer destruction.
            try:
                buf.cancel()
            except Exception as exc:
                errors.append((getattr(exc, "errno", None) or errno.EIO, f"Cancel: {exc}"))

        self._dma_buf = None
        self._dma_buf_frame_samples = None
        self._dma_capture_geometry = None
        if isinstance(buf, IiodScopeStream):
            # Closing the dedicated data connection makes iiod destroy the
            # remote IIO buffer. Control/gate polling remains on _dma_ctx.
            buf.close()
        else:
            # Do not depend on Buffer.__del__ timing here. In pylibiio 0.25 a
            # cancelled refill can leave Python/threading references alive
            # long enough for the complete gate wait to expire. Destroy the
            # native buffer explicitly after every reader has stopped, then
            # null the wrapper pointer so its eventual __del__ is idempotent.
            native_buffer = buf._buffer
            buf._buffer = None
            if native_buffer is not None:
                iio._buffer_destroy(native_buffer)
        del buf

        try:
            self._wait_for_dma_release()
            scope = getattr(self, "_dma_scope", None)
            if scope is not None and "dma_poisoned" in scope.attrs and int(
                self._dma_attr_get("dma_poisoned")
            ):
                raise RuntimeError("Scope DMA poisoned during close; reboot required")
        except Exception as exc:
            errors.append((getattr(exc, "errno", None) or errno.EIO, f"Release: {exc}"))
        if errors:
            self._dma_fault_latched = True
            raise OSError(errors[0][0], "; ".join(message for _, message in errors))
        log.debug("IIO backend ch%d: DMA gate released after close", self._ch)

    def _wait_for_dma_release(self) -> None:
        """Wait for target-side buffer destruction to release the DMA gate.

        Ported from the driver author's reference implementation
        (scope_backend_iio.py's _wait_for_dma_release()): a network iiod
        request can acknowledge buffer destruction before the target-side
        teardown (predisable() -> the stock Xilinx DMA terminate path,
        which can take up to about a second -- see vdpp-scope.c's
        preenable() comment on "Cannot stop channel") is actually visible
        through sysfs. Poll dma_buffer_active and dma_enable instead of
        assuming either is already clear the instant the buffer object is
        gone, and never write either -- see _close_dma_buffer()'s comment.
        """
        deadline = time.monotonic() + _DMA_CLOSE_RELEASE_SECONDS
        while True:
            active = int(self._dma_attr_get("dma_buffer_active")) != 0
            if not active and not self._dma_get_dma_enable():
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "DMA buffer close did not release the hardware gate "
                    f"within {_DMA_CLOSE_RELEASE_SECONDS:.2f} seconds"
                )
            time.sleep(_DMA_CLOSE_RELEASE_POLL_SECONDS)

    def close_dma_capture(self) -> None:
        """Public wrapper over _close_dma_buffer(), for callers outside
        this class (IIOScopeDmaStreamer in dma.py) that need to disarm
        DMA capture without going through read_dma_frame()/
        capture_to_file(). Per vdpp-scope.c's postenable()/predisable():
        closing the buffer is what clears ENABLE and DMA_ENABLE for the
        DMA case -- there is no separate "stop DMA" step.
        """
        self._close_dma_buffer()

    def prepare_dma_capture(self) -> None:
        """Begin a new streamer session only from a fully released state."""
        if self._dma_buf is not None:
            raise RuntimeError("cannot start scope DMA: a capture buffer is already open")
        if self._dma_fault_latched:
            raise RuntimeError(
                "DMA fault is latched from a previous session -- call "
                "acknowledge_dma_recovery() before capturing again"
            )
        if "dma_poisoned" in self._dma_scope.attrs and int(self._dma_attr_get("dma_poisoned")):
            raise RuntimeError("Scope DMA is poisoned; reboot the target before reuse")
        self._wait_for_dma_release()
        self._sum_draining = False
        self._dma_stop_requested.clear()

    def get_scope_trigger_modes(self) -> tuple[int, ...]:
        if "trigger_mode_available" not in self._scope.attrs:
            return tuple(range(5))
        tokens = self._attr_get("trigger_mode_available").split()
        # Driver emits alternating numeric selectors and descriptive names.
        advertised = tuple(int(item) for item in tokens if item.isdecimal())
        ip = self.get_ip_version()
        return tuple(
            mode for mode in advertised if mode in range(6) and (mode != 5 or ip in (123, 124))
        )

    def request_scope_sum_stop(self) -> None:
        """Control-context STOP; keep the single DMA reader alive for the tail."""
        self._sum_draining = True
        self.set_enable(False)

    def stop_scope_sum_from_reader(self) -> None:
        """Worker-owned context STOP, distinct from the GUI stop hook."""
        self._sum_draining = True
        self._dma_set_enable(False)

    def scope_sum_status(self) -> tuple[int, int]:
        # A failed transport may already have disarmed/reset the FPGA. Its
        # now-clear pending bit cannot certify that the old tail was drained.
        if self._dma_fault_latched or self._dma_buf is None:
            raise RuntimeError("Cannot confirm sum drain after a DMA fault or closed buffer")
        return (int(self._dma_attr_get("sum_status")),
                int(self._dma_attr_get("sum_missed_windows")))

    def scope_sum_identity(self) -> dict[str, object]:
        return {
            "uri": self._uri,
            "channel": self._ch,
            "device": self._dma_scope.id,
            "ip_version": int(self._dma_attr_get("ip_version")),
            "driver_reference": "PetaLinux bb8c57e",
            "channel_mapping": "sorted discovery; physical A/B mapping not independently verified",
        }

    def close_scope_sum_capture(self) -> None:
        self._close_dma_buffer(drain=False)
        if "dma_poisoned" in self._dma_scope.attrs and int(self._dma_attr_get("dma_poisoned")):
            self._dma_fault_latched = True
            raise RuntimeError("Scope DMA poisoned during teardown; reboot required")

    def request_dma_stop(self) -> None:
        """Stop acquisition and interrupt a possibly blocked DMA refill.

        Called from the controller thread. ``enable=0`` uses the normal
        control context, while Buffer.cancel() is explicitly designed to
        interrupt a refill running in the DMA worker thread. A short grace
        interval lets an already-completing frame return normally first.
        """
        self._dma_stop_requested.set()
        try:
            if self.get_enable():
                self.set_enable(False)
        except OSError:
            log.warning(
                "IIO backend ch%d: failed to clear enable during DMA stop",
                self._ch,
                exc_info=True,
            )
        time.sleep(_DMA_CLOSE_DRAIN_SECONDS)
        buf = self._dma_buf
        if buf is not None:
            buf.cancel()

    def dma_fault_is_latched(self) -> bool:
        """Extension method (not part of ScopeBackend), matching
        user-api.md's naming: whether a genuine EIO from a normal capture
        refill() has blocked automatic DMA rearm -- see
        _refill_dma_buffer()'s docstring. Call acknowledge_dma_recovery()
        to clear it.
        """
        return self._dma_fault_latched

    def acknowledge_dma_recovery(self) -> None:
        """Extension method (not part of ScopeBackend), matching
        user-api.md's naming and contract: "keeps enable=0, passively
        waits for both driver-owned gates to become zero and only then
        permits a new arm. It never writes dma_enable. This is a
        userspace lifecycle check, not proof of internal AXI DMA health."

        Requires the faulted buffer to already be closed (it is, by the
        time _refill_dma_buffer() raises -- see its docstring) and
        acquisition stopped; raises if either still holds the gate, since
        this method does not do that cleanup itself.
        """
        if "dma_poisoned" in self._scope.attrs and int(self._attr_get("dma_poisoned")):
            raise RuntimeError("Scope DMA is poisoned; reboot required before recovery")
        if self._dma_buf is not None:
            raise RuntimeError(
                "cannot acknowledge DMA recovery while a capture buffer "
                "is still open -- call close_dma_capture() first"
            )
        if self._dma_get_enable():
            raise RuntimeError(
                "cannot acknowledge DMA recovery while acquisition is "
                "running -- call set_enable(False) first"
            )
        self._wait_for_dma_release()
        self._dma_fault_latched = False
        log.info(
            "IIO backend ch%d: DMA fault acknowledged, rearm permitted",
            self._ch,
        )

    def _attr_get(self, name: str) -> str:
        return self._scope.attrs[name].value

    def _attr_set(self, name: str, value: str) -> None:
        self._scope.attrs[name].value = value

    # DMA-context attribute helpers -- see __init__'s comment on
    # self._dma_scope/self._dma_ctx. Used exclusively by the DMA-buffer
    # lifecycle (_create_dma_buffer(), _start_reader_then_enable(),
    # _close_dma_buffer(), _wait_for_dma_release(), _refill_dma_buffer()),
    # which only ever runs on the DMA worker thread, so these never
    # contend with the GUI thread's self._scope/_attr_get()/_attr_set()
    # calls. No _dma_set_dma_enable() -- dma_enable is never written on
    # either context; see scope_dma_active()'s docstring.

    def _dma_attr_get(self, name: str) -> str:
        return self._dma_scope.attrs[name].value

    def _dma_attr_set(self, name: str, value: str) -> None:
        self._dma_scope.attrs[name].value = value

    def _dma_get_enable(self) -> bool:
        return bool(int(self._dma_attr_get("enable")))

    def _dma_set_enable(self, val: bool) -> None:
        self._dma_attr_set("enable", "1" if val else "0")

    def _dma_get_dma_enable(self) -> bool:
        return bool(int(self._dma_attr_get("dma_enable")))

    def _dma_get_frame_samples(self) -> int:
        return int(self._dma_attr_get("frame_samples"))

    # MCA attribute helpers -- self._pp/self._input_filter (GUI-thread
    # context, self._ctx) are used by every MCABackend setter and by the
    # getters MCAController calls directly (config hydration, get_settings()).
    # self._mca_pp (dedicated self._mca_ctx) is used only by the handful of
    # getters MCAWorker's background QThread polls every tick -- see
    # __init__'s comment on self._mca_ctx for why that split exists.

    def _pp_attr_get(self, name: str) -> str:
        if self._pp is None:
            raise RuntimeError(
                f"no {_PULSE_PROCESSOR_DEVICE_NAME} device bound -- pulse-"
                "processor core not present on this channel's firmware"
            )
        return self._pp.attrs[name].value

    def _pp_attr_set(self, name: str, value: str) -> None:
        if self._pp is None:
            raise RuntimeError("Pulse processor absent")
        with self.mca_configuration_lock:
            if name not in ("debug_signal1", "debug_signal2") and not (
                name == "enable" and value == "0"
            ):
                self._require_mca_contract(self._pp)
                if name in PARAMETERS:
                    validate_parameters({name: int(value)})
                    if name == "psd_zero_crossing_mode":
                        self._validate_psd_mode(int(value))
                if name == "enable" and value == "1":
                    validate_parameters(self.mca_parameter_snapshot(), complete=True)
                    self._validate_psd_mode(int(self._pp_attr_get("psd_zero_crossing_mode")))
                if name in PARAMETERS or (name == "enable" and value == "1"):
                    self._assert_mca_idle(self._pp)
            self._pp.attrs[name].value = value

    def _if_attr_get(self, name: str) -> str:
        if self._input_filter is None:
            raise RuntimeError(
                f"no {_INPUT_FILTER_DEVICE_NAME} device bound -- input-"
                "filter core not present on this channel's firmware"
            )
        return self._input_filter.attrs[name].value

    def _if_attr_set(self, name: str, value: str) -> None:
        if self._input_filter is None:
            raise RuntimeError(
                f"no {_INPUT_FILTER_DEVICE_NAME} device bound -- input-"
                "filter core not present on this channel's firmware"
            )
        with self.mca_configuration_lock:
            self._input_filter.attrs[name].value = value

    def _sync_attr_get(self, name: str) -> str:
        if self._sync_trigger is None:
            raise RuntimeError(
                f"no {_SYNC_TRIGGER_DEVICE_NAME} device bound -- shared MCA "
                "trigger-sync core not present on this firmware"
            )
        return str(self._sync_trigger.attrs[name].value)

    def _sync_attr_set(self, name: str, value: str) -> None:
        if self._sync_trigger is None:
            raise RuntimeError(
                f"no {_SYNC_TRIGGER_DEVICE_NAME} device bound -- shared MCA "
                "trigger-sync core not present on this firmware"
            )
        self._sync_trigger.attrs[name].value = value

    def _mca_pp_attr_get(self, name: str) -> str:
        return self._require_mca_pp().attrs[name].value

    def _require_mca_pp(self) -> iio.Device:
        """Narrows self._mca_pp from `iio.Device | None` to `iio.Device`
        for callers (read_histogram(), read_waveform_banks()) that need
        the Device object itself, not just an attribute value.
        """
        if self._mca_pp is None:
            raise RuntimeError(
                f"no {_PULSE_PROCESSOR_DEVICE_NAME} device bound -- pulse-"
                "processor core not present on this channel's firmware"
            )
        return self._mca_pp

    def _read_binary_attr(self, device: iio.Device, name: str, size: int) -> bytes:
        """Read an exact-size binary IIO device attribute.

        Unlike _read_large_attr()'s text truncation at the first NUL
        (legacy viewer_data is a C string), binary snapshots may contain
        embedded NULs, so
        embedded NULs must be preserved. The updated remote iiod transport
        needs approximately twice the raw-payload capacity: confirmed live,
        debug_data fails with EIO at capacities 8192/8193 but succeeds at
        16383, and each 16384-byte histogram chunk succeeds at 32767. It
        returns payload_size + 1 and the final byte is a C NUL terminator.
        Older firmware returned exactly payload_size. Allocate the larger
        capacity and accept both result forms, removing only that verified
        final terminator rather than truncating binary data at the first NUL.

        Takes the high-level iio.Device (not its raw ._device ctypes
        handle) -- confirmed live that passing the wrapper object itself
        to _d_read_attr() raises `ctypes.ArgumentError: expected
        LP__Device instance instead of Device`; _read_large_attr() avoids
        this the same way, via self._scope._device rather than self._scope.
        """
        capacity = 2 * size + 1
        buf = ctypes.create_string_buffer(capacity)
        n = iio._d_read_attr(device._device, name.encode("ascii"), buf, capacity)
        if n == size:
            return buf.raw[:size]
        if n == size + 1 and buf.raw[size] == 0:
            return buf.raw[:size]
        raise RuntimeError(
            f"'{name}' returned {n} bytes; expected {size} bytes, optionally "
            "followed by one NUL transport terminator"
        )

    def _read_large_pp_attr(self, device: iio.Device, name: str, size: int) -> bytes:
        """Compatibility-named wrapper for MCA binary attributes."""
        return self._read_binary_attr(device, name, size)

    # MCA list-mode DMA helpers run on IIOMcaDmaWorker's thread. Buffer I/O
    # exclusively uses _mca_dma_ctx, while pulse-processor control uses the
    # independent _mca_dma_control_ctx. GUI configuration and MCAWorker
    # polling remain on two further contexts.

    def _require_lm_frame(self) -> iio.Device:
        if self._lm_frame is None:
            raise RuntimeError(
                f"no {_LM_FRAME_DEVICE_NAME} device bound -- list-mode "
                "DMA is not present on this channel's firmware"
            )
        return self._lm_frame

    def _require_mca_dma_pp(self) -> iio.Device:
        if self._mca_dma_pp is None:
            raise RuntimeError(
                f"no {_PULSE_PROCESSOR_DEVICE_NAME} handle bound in the "
                "dedicated MCA DMA control context"
            )
        return self._mca_dma_pp

    def _lm_attr_get(self, name: str) -> str:
        return str(self._require_lm_frame().attrs[name].value)

    def _lm_status_attr_get(self, name: str) -> str:
        # Status remains readable even when a cancelled native reader has not
        # exited. Never reuse its stream context to collect failure metadata.
        device = getattr(self, "_lm_status_device", None) or self._require_lm_frame()
        with self._mca_dma_control_lock:
            return str(device.attrs[name].value)

    def _mca_dma_pp_attr_get(self, name: str) -> str:
        with self._mca_dma_control_lock:
            return str(self._require_mca_dma_pp().attrs[name].value)

    def _mca_dma_pp_attr_set(self, name: str, value: str) -> None:
        # Whole histogram assemblies and control transitions share ownership,
        # but READBUF keeps its independent connection and never holds this lock.
        with self.mca_configuration_lock, self._mca_dma_control_lock:
            pp = self._require_mca_dma_pp()
            if not (name == "enable" and value == "0"):
                self._require_mca_contract(pp)
                if name == "enable" and value == "1":
                    self._assert_mca_idle(pp)
            pp.attrs[name].value = value

    def _validate_lm_geometry(self) -> None:
        """Validate the fixed v121 ABI before creating a DMA buffer.

        The constants are not client preferences: lm_probe() rejects a
        different IP version/frame size and lm_buffer_preenable() rejects
        every scan size or buffer length except one opaque u8[16] element x
        1024 records. Reading the attributes here makes the superseded
        five-channel driver fail precisely before iio.Buffer() returns its
        otherwise opaque EINVAL.
        """
        self._require_mca_contract(self._require_mca_dma_pp())
        if "dma_poisoned" in self._require_lm_frame().attrs and int(
            self._lm_attr_get("dma_poisoned")
        ):
            raise RuntimeError("MCA DMA is poisoned; reboot the target before reuse")
        version = int(self._lm_attr_get("ip_version"))
        records = int(self._lm_attr_get("frame_records"))
        frame_bytes = int(self._lm_attr_get("frame_bytes"))
        if (version, records, frame_bytes) != (
            _LM_IP_VERSION,
            _LM_FRAME_RECORDS,
            _LM_FRAME_BYTES,
        ):
            raise RuntimeError(
                "unsupported list-mode geometry: "
                f"ip_version={version}, frame_records={records}, "
                f"frame_bytes={frame_bytes}; expected {_LM_IP_VERSION}, "
                f"{_LM_FRAME_RECORDS}, {_LM_FRAME_BYTES}"
            )

        lm_frame = self._require_lm_frame()
        pulse_processor = self._require_mca_dma_pp()
        if "channel_index" in lm_frame.attrs and "channel_index" in pulse_processor.attrs:
            lm_channel = int(lm_frame.attrs["channel_index"].value)
            pp_channel = int(pulse_processor.attrs["channel_index"].value)
            if lm_channel != self._ch or pp_channel != self._ch:
                raise RuntimeError(
                    "mismatched MCA DMA device pair: "
                    f"backend channel={self._ch}, pulse_processor={pp_channel}, "
                    f"lm_frame={lm_channel}"
                )

        channels = lm_frame.channels
        layout = self._lm_attr_get("record_layout").strip()
        if layout != _LM_RECORD_LAYOUT:
            raise RuntimeError(
                f"unsupported {_LM_FRAME_DEVICE_NAME} record_layout={layout!r}; "
                f"expected {_LM_RECORD_LAYOUT!r}. Targets exposing five "
                "semantic channels use the superseded Linux 5.15 ABI whose "
                "timestamp scan mask makes buffer creation fail with EINVAL"
            )
        if len(channels) != 1:
            raise RuntimeError(
                f"{_LM_FRAME_DEVICE_NAME} exposes {len(channels)} scan "
                "channels; the current v121 transport requires exactly one "
                "opaque u8[16] scan element"
            )
        channels[0].enabled = True
        if lm_frame.sample_size != _LM_RECORD_BYTES:
            raise RuntimeError(
                f"{_LM_FRAME_DEVICE_NAME} scan mask produces "
                f"{lm_frame.sample_size} bytes per record; "
                f"v121 requires exactly {_LM_RECORD_BYTES}"
            )

    def _start_mca_reader_then_enable(
        self,
        buf: iio.Buffer,
        on_started: Callable[[], None] | None = None,
    ) -> int:
        """Start the first refill before pulse_processor.enable=1.

        This is the list-mode form of _start_reader_then_enable(). It
        follows mca-architecture.md steps 5-6 and lm_buffer_postenable()'s
        guarantee that the first DMA descriptor is queued before the
        private list gate opens. Starting the producer before a blocking
        reader exists is the same ordering race already confirmed live on
        the scope path.
        """
        entered = threading.Event()
        result: list[int] = []
        errors: list[tuple[int | None, str]] = []

        def reader() -> None:
            entered.set()
            try:
                result.append(self._refill_mca_dma_buffer(buf))
            except OSError as exc:
                # Do not carry the exception object across threads: its
                # traceback frame owns this closure and therefore ``buf``.
                # Keeping it alive would prevent Buffer.__del__ during fault
                # cleanup and make the DMA gate-release wait self-deadlock.
                errors.append((exc.errno, str(exc)))
            except BaseException as exc:
                errors.append((None, f"{type(exc).__name__}: {exc}"))

        thread = threading.Thread(
            target=reader,
            name=f"vdpp-lm-frame-ch{self._ch}-first-refill",
            daemon=True,
        )
        self._mca_dma_reader = thread
        thread.start()
        if not entered.wait(_DMA_FIRST_REFILL_ENTER_SECONDS):
            try:
                self._cancel_mca_dma_refill(buf, "reader thread did not enter refill")
            finally:
                self._join_dma_reader("_mca_dma_reader", _LM_CLOSE_JOIN_SECONDS)
            raise RuntimeError(
                "MCA list-mode reader did not start within "
                f"{_DMA_FIRST_REFILL_ENTER_SECONDS:.2f} seconds -- refusing "
                "to write pulse_processor.enable=1 without a listener"
            )

        try:
            if not bool(int(self._mca_dma_pp_attr_get("enable"))):
                self._mca_dma_pp_attr_set("enable", "1")
            # Report readiness only after the buffer is armed, a reader is
            # waiting, and the producer has been enabled. Previously the
            # streamer emitted ready before iio.Buffer() even ran, allowing
            # the polling worker to observe a false completed measurement.
            if on_started is not None:
                on_started()
        except BaseException as primary:
            try:
                self._mca_dma_pp_attr_set("enable", "0")
            except OSError:
                log.warning(
                    "IIO backend ch%d: failed to stop MCA after start error",
                    self._ch,
                    exc_info=True,
                )
            try:
                try:
                    self._cancel_mca_dma_refill(buf, "MCA start failed")
                finally:
                    self._join_dma_reader("_mca_dma_reader", _LM_CLOSE_JOIN_SECONDS)
            except Exception as cleanup:
                self._record_mca_cleanup_error("Startup reader", cleanup)
                primary.add_note(f"Startup cleanup: {cleanup}")
            raise

        thread.join()
        self._mca_dma_reader = None
        if errors:
            error_errno, error_text = errors[0]
            if error_errno is not None:
                raise OSError(error_errno, error_text) from None
            raise RuntimeError(error_text)
        return result[0]

    def _create_mca_dma_buffer(self) -> None:
        """Arm the fixed 1024-record list-mode buffer while stopped."""
        if getattr(self, "_mca_dma_fault_latched", False):
            raise RuntimeError("MCA DMA fault is latched; acknowledge_mca_dma_recovery() required")
        self._close_mca_dma_buffer()
        self._mca_cleanup_errors = []
        self._mca_dma_stop_requested.clear()
        self._mca_dma_cancel_explained = False

        if bool(int(self._mca_dma_pp_attr_get("enable"))):
            self._mca_dma_pp_attr_set("enable", "0")

        lm_frame = self._require_lm_frame()
        # user-api.md's production lifecycle and mca_listmode_capture.py both
        # require several mmap blocks so the driver can rearm Simple DMA from
        # its completion callback without a network round-trip. Eight is the
        # reference value; unlike vdpp_scope, list mode must not force one.
        lm_frame.set_kernel_buffers_count(_LM_KERNEL_BUFFER_COUNT)
        self._validate_lm_geometry()

        if bool(int(self._mca_dma_pp_attr_get("list_buffer_active"))):
            raise RuntimeError(
                "cannot arm MCA list-mode DMA: pulse processor still "
                "reports list_buffer_active=1 from another buffer owner"
            )
        if bool(int(self._lm_attr_get("buffer_active"))):
            raise RuntimeError(
                "cannot arm MCA list-mode DMA: lm_frame still reports "
                "buffer_active=1 from another buffer owner"
            )

        try:
            self._mca_dma_buf = iio.Buffer(
                lm_frame, _LM_FRAME_RECORDS, False,
            )
        except OSError as exc:
            diagnostics = ", ".join(
                f"{name}={self._lm_attr_get(name)}"
                for name in (
                    "buffer_active",
                    "dma_fault",
                    "dma_error_count",
                    "queued_blocks",
                    "kernel_buffer_blocks",
                )
                if name in lm_frame.attrs
            )
            raise RuntimeError(
                "failed to arm vdpp_lm_frame after selecting eight kernel "
                "blocks and validating the single opaque 16-byte scan with "
                "1024 records; the target rejected buffer creation"
                + (f" ({diagnostics})" if diagnostics else "")
            ) from exc

        if "kernel_buffer_blocks" in lm_frame.attrs:
            blocks = int(self._lm_attr_get("kernel_buffer_blocks"))
            if blocks != _LM_KERNEL_BUFFER_COUNT:
                self._close_mca_dma_buffer(drain=False)
                raise RuntimeError(
                    f"vdpp_lm_frame allocated {blocks} kernel blocks; "
                    f"expected {_LM_KERNEL_BUFFER_COUNT}"
                )
        log.debug(
            "IIO backend ch%d: MCA list-mode buffer armed (%d records, "
            "%d bytes, %d kernel blocks)",
            self._ch, _LM_FRAME_RECORDS, _LM_FRAME_BYTES, _LM_KERNEL_BUFFER_COUNT,
        )

    def _parse_lm_block(self, raw: bytes) -> np.ndarray:
        if len(raw) != _LM_FRAME_BYTES:
            raise RuntimeError(
                f"short MCA list-mode block: got {len(raw)} bytes, "
                f"expected exactly {_LM_FRAME_BYTES}"
            )
        return np.frombuffer(raw, dtype=_LM_EVENT_DTYPE).copy()

    def _refill_mca_dma_buffer(self, buf: iio.Buffer) -> int:
        """Wait for one list-mode block across server-side idle timeouts.

        libiio v0.26's network_open() creates a separate socket for the IIO
        buffer, but unlike network_set_timeout() it does not send a TIMEOUT
        command on that socket. The target therefore periodically returns
        ETIMEDOUT while an armed external-trigger measurement legitimately
        has no data. The buffer and its queued DMA descriptors remain valid,
        so retry the same refill. Buffer.cancel() still wins during Stop;
        once the stop event is set, its ETIMEDOUT/EBADF/ECANCELED propagates
        to the existing expected-cancellation path instead of being retried.

        Confirmed live on 2026-08-11: ch0 list-mode DMA remained armed for
        nine seconds (across two remote ETIMEDOUT responses) while ch1 was
        armed normally; the shared software trigger then produced complete
        DMA frames and the capture closed with valid continuity diagnostics.
        """
        timeout_count = 0
        while True:
            try:
                return iio._buffer_refill(buf._buffer)
            except OSError as exc:
                if (
                    exc.errno not in _LIBIIO_TIMEOUT_ERRNOS
                    or self._mca_dma_stop_requested.is_set()
                ):
                    raise
                timeout_count += 1
                if timeout_count == 1 or timeout_count % 15 == 0:
                    log.debug(
                        "IIO backend ch%d: MCA list-mode refill still "
                        "waiting for data after %d server timeout(s); retrying",
                        self._ch,
                        timeout_count,
                    )

    def start_mca_dma_capture(
        self,
        on_started: Callable[[], None] | None = None,
    ) -> np.ndarray:
        """Arm list mode, start its reader, enable MCA, and return frame one.

        ``on_started`` runs only after the buffer owns the list gate, the
        first refill thread is present, and ``pulse_processor.enable`` reads
        as enabled. It lets the GUI start status polling without the previous
        pre-arm race. The call then remains blocked until the first complete
        16384-byte frame arrives.
        """
        if self._mca_dma_buf is not None:
            raise RuntimeError("MCA list-mode capture is already active")

        self._create_mca_dma_buffer()
        buf = self._mca_dma_buf
        assert buf is not None
        try:
            nbytes = self._start_mca_reader_then_enable(buf, on_started)
            if nbytes == 0 and self._mca_dma_stop_requested.is_set():
                self._close_mca_dma_buffer(drain=False)
                return np.empty(0, dtype=_LM_EVENT_DTYPE)
            raw = ctypes.string_at(iio._buffer_start(buf._buffer), nbytes)
            return self._parse_lm_block(raw)
        except BaseException as exc:
            expected_stop = (
                self._mca_dma_stop_requested.is_set()
                and isinstance(exc, OSError)
                and exc.errno in _LM_EXPECTED_CANCEL_ERRNOS
            )
            if not expected_stop:
                self._mca_dma_fault_latched = True
            try:
                self._close_mca_dma_buffer(drain=False)
            except Exception as cleanup:
                self._record_mca_cleanup_error("Capture close", cleanup)
                exc.add_note(f"Capture cleanup: {cleanup}")
                if expected_stop:
                    raise
            if expected_stop:
                if self._mca_dma_buf is not None:
                    raise
                return np.empty(0, dtype=_LM_EVENT_DTYPE)
            raise

    def read_mca_dma_frame(self) -> np.ndarray:
        """Return one parsed 1024-record list-mode DMA block.

        This is an IIO-backend extension, not part of MCABackend. New stream
        sessions should call start_mca_dma_capture() for the first block so
        readiness can be reported at the correct lifecycle boundary. The
        lazy-start fallback is retained for direct callers. The returned
        structured dtype follows vdpp-lm-frame.c exactly and deliberately
        differs from the legacy NDMA v1 event dtype.
        """
        if self._mca_dma_buf is None:
            return self.start_mca_dma_capture()

        buf = self._mca_dma_buf
        assert buf is not None
        try:
            nbytes = self._refill_mca_dma_buffer(buf)
            if nbytes == 0 and self._mca_dma_stop_requested.is_set():
                self._close_mca_dma_buffer(drain=False)
                return np.empty(0, dtype=_LM_EVENT_DTYPE)
            raw = ctypes.string_at(iio._buffer_start(buf._buffer), nbytes)
            return self._parse_lm_block(raw)
        except BaseException as exc:
            expected_stop = (
                self._mca_dma_stop_requested.is_set()
                and isinstance(exc, OSError)
                and exc.errno in _LM_EXPECTED_CANCEL_ERRNOS
            )
            if not expected_stop:
                self._mca_dma_fault_latched = True
            try:
                self._close_mca_dma_buffer(drain=False)
            except Exception as cleanup:
                self._record_mca_cleanup_error("Capture close", cleanup)
                exc.add_note(f"Capture cleanup: {cleanup}")
                if expected_stop:
                    raise
            if expected_stop:
                return np.empty(0, dtype=_LM_EVENT_DTYPE)
            raise

    def _cancel_mca_dma_refill(self, buf: iio.Buffer, reason: str) -> None:
        """Cancel one native refill with context for libiio's stderr noise.

        libiio 0.26's IIO_ERROR macro writes directly to the process stderr
        and exposes no runtime callback in the shipped DLL. Redirecting that
        descriptor would be process-wide and could swallow unrelated errors
        from other threads. Log the meaning immediately before cancellation
        instead, once per DMA session.
        """
        with self._mca_dma_lifecycle_lock:
            explain = not self._mca_dma_cancel_explained
            self._mca_dma_cancel_explained = True
        if explain:
            log.info(
                "IIO backend ch%d: cancelling blocked MCA DMA refill (%s). "
                "Native libiio may now print 'READ LINE: -9' and/or "
                "'READ INTEGER: -9'; -9 is the expected EBADF from "
                "Buffer.cancel(), not a DMA acquisition fault",
                self._ch,
                reason,
            )
        buf.cancel()

    def _drain_mca_for_close(
        self,
        buf: iio.Buffer,
        on_frame: Callable[[np.ndarray], None] | None,
    ) -> int:
        """Drain complete blocks until one second passes without data.

        Payload zeros are never inspected: mca-architecture.md explicitly
        says an exactly full final frame has no padding and real records may
        themselves contain zeros. Only bounded refill inactivity defines
        the v121 end boundary.
        """
        stop = threading.Event()
        entered = threading.Event()
        errors: list[tuple[int, str]] = []
        drained = 0
        last_activity = [time.monotonic()]

        def refill_until_inactive() -> None:
            nonlocal drained
            entered.set()
            while not stop.is_set():
                try:
                    nbytes = iio._buffer_refill(buf._buffer)
                    raw = ctypes.string_at(iio._buffer_start(buf._buffer), nbytes)
                    frame = self._parse_lm_block(raw)
                    if on_frame is not None:
                        on_frame(frame)
                    drained += 1
                    last_activity[0] = time.monotonic()
                except BaseException as exc:
                    if getattr(exc, "errno", None) in (*_LIBIIO_TIMEOUT_ERRNOS, errno.EAGAIN):
                        # Inactivity is bounded by the owner, not by one
                        # server timeout. Final latched faults are checked too.
                        stop.wait(0.01)
                        continue
                    if not (stop.is_set() and getattr(exc, "errno", None)
                            in _LM_EXPECTED_CANCEL_ERRNOS):
                        # Do not retain a traceback that owns this closure and
                        # its Buffer reference across native buffer teardown.
                        errors.append((getattr(exc, "errno", None) or errno.EIO, str(exc)))
                    return

        worker = threading.Thread(
            target=refill_until_inactive,
            name=f"vdpp-lm-frame-ch{self._ch}-drain",
            daemon=True,
        )
        self._mca_dma_reader = worker
        worker.start()
        try:
            if not entered.wait(_LM_CLOSE_JOIN_SECONDS):
                raise RuntimeError("MCA drain reader did not enter refill")
            while time.monotonic() - last_activity[0] < _LM_CLOSE_INACTIVITY_SECONDS:
                time.sleep(_LM_CLOSE_POLL_SECONDS)
        finally:
            stop.set()
            try:
                self._cancel_mca_dma_refill(buf, "tail-drain inactivity window elapsed")
            finally:
                self._join_dma_reader("_mca_dma_reader", _LM_CLOSE_JOIN_SECONDS)
        if errors:
            raise OSError(*errors[0]) from None
        return drained

    def _wait_for_mca_dma_release(self) -> None:
        deadline = time.monotonic() + _LM_CLOSE_RELEASE_SECONDS
        while True:
            frame_active = bool(int(self._lm_status_attr_get("buffer_active")))
            list_active = bool(int(self._mca_dma_pp_attr_get("list_buffer_active")))
            if not frame_active and not list_active:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "MCA list-mode buffer close did not release buffer_active/"
                    f"list_buffer_active within {_LM_CLOSE_RELEASE_SECONDS:.1f} seconds"
                )
            time.sleep(_DMA_CLOSE_RELEASE_POLL_SECONDS)

    def _close_mca_dma_buffer(
        self,
        on_frame: Callable[[np.ndarray], None] | None = None,
        *,
        drain: bool = True,
    ) -> int:
        """Stop/drain/join/destroy; retain primary errno and report cleanup failures."""
        with self._mca_dma_lifecycle_lock:
            timer = self._mca_dma_cancel_timer
            self._mca_dma_cancel_timer = None
            if timer is not None:
                timer.cancel()
        buf = self._mca_dma_buf
        if buf is None:
            return 0
        errors: list[tuple[int, str]] = []
        drained = 0
        try:
            self._mca_dma_pp_attr_set("enable", "0")
        except Exception as exc:
            errors.append((getattr(exc, "errno", None) or errno.EIO, f"Stop: {exc}"))
            drain = False
        try:
            if getattr(self, "_mca_dma_reader", None) is not None:
                try:
                    self._cancel_mca_dma_refill(buf, "joining retained DMA reader")
                finally:
                    self._join_dma_reader("_mca_dma_reader", _LM_CLOSE_JOIN_SECONDS)
                drain = False
            if drain:
                drained = self._drain_mca_for_close(buf, on_frame)
            else:
                self._cancel_mca_dma_refill(buf, "capture cleanup without tail drain")
        except DmaReaderActiveError as exc:
            self._record_mca_cleanup_error("Reader", exc)
            raise
        except Exception as exc:
            errors.append((getattr(exc, "errno", None) or errno.EIO, f"Drain: {exc}"))
        self._mca_dma_buf = None
        native = buf._buffer
        buf._buffer = None
        if native is not None:
            iio._buffer_destroy(native)
        del buf
        try:
            self._wait_for_mca_dma_release()
            with self.mca_configuration_lock, self._mca_dma_control_lock:
                self._assert_mca_idle(self._mca_dma_pp, check_fault=False)
            if "dma_poisoned" in self._require_lm_frame().attrs and int(
                self._lm_attr_get("dma_poisoned")
            ):
                raise RuntimeError("MCA DMA poisoned during close; reboot required")
        except Exception as exc:
            errors.append((getattr(exc, "errno", None) or errno.EIO, f"Release: {exc}"))
        finally:
            for channel in self._require_lm_frame().channels:
                channel.enabled = False
        if errors:
            self._mca_dma_fault_latched = True
            for number, message in errors:
                self._record_mca_cleanup_error("Close", OSError(number, message))
            raise OSError(errors[0][0], "; ".join(message for _, message in errors))
        return drained

    def _record_mca_cleanup_error(self, phase: str, error: BaseException) -> None:
        if not hasattr(self, "_mca_cleanup_errors"):
            self._mca_cleanup_errors = []
        self._mca_cleanup_errors.append({
            "phase": phase, "errno": getattr(error, "errno", None) or errno.EIO,
            "message": str(error),
        })

    def _assert_mca_idle(self, pp: iio.Device | None, *, check_fault: bool = True) -> None:
        """Reject uncertain rearm; progress=0 still supplies no completion fence.

        PetaLinux c546f14 digitizer-128-verification-2026-11-07.md records
        persistent progress=1 after STOP. The bounded poll only tolerates the
        normal nonblocking transition; it never certifies final histogram data.
        """
        if check_fault and self.mca_recovery_required():
            raise RuntimeError(
                "MCA recovery required; verify target state and acknowledge recovery"
            )
        if pp is None or "measurement_in_progress" not in pp.attrs:
            return
        if int(pp.attrs["enable"].value):
            return  # The kernel retains its EBUSY check for live configuration.
        deadline = time.monotonic() + _MCA_STOP_STATUS_SECONDS
        while int(pp.attrs["measurement_in_progress"].value):
            if time.monotonic() >= deadline:
                self._mca_stop_uncertain = True
                raise RuntimeError(
                    "MCA remains measurement_in_progress=1 after STOP; "
                    "rearm blocked, final data unverified; target recovery required"
                )
            time.sleep(0.01)

    def ensure_mca_stopped(self) -> None:
        with self.mca_configuration_lock:
            if self._pp is not None and int(self._pp.attrs["enable"].value):
                raise RuntimeError("Stop MCA before checking stopped state")
            self._assert_mca_idle(self._pp)

    def mca_recovery_required(self) -> bool:
        return bool(getattr(self, "_mca_dma_fault_latched", False)
                    or getattr(self, "_mca_stop_uncertain", False))

    def acknowledge_mca_dma_recovery(self) -> None:
        """Passive acknowledgement after externally validated recovery; never reset hardware."""
        with self.mca_configuration_lock:
            if self._mca_dma_buf is not None or getattr(self, "_mca_dma_reader", None) is not None:
                raise RuntimeError("Close the owned MCA reader/buffer before recovery")
            if self._lm_frame is not None and "dma_poisoned" in self._lm_frame.attrs:
                if int(self._lm_attr_get("dma_poisoned")):
                    raise RuntimeError("MCA DMA poisoned; reboot required")
            if self._pp is None or int(self._pp.attrs["enable"].value):
                raise RuntimeError("Stop MCA before recovery")
            if int(self._pp.attrs["list_buffer_active"].value):
                raise RuntimeError("Close list DMA before recovery")
            self._wait_for_mca_dma_release()
            self._assert_mca_idle(self._pp, check_fault=False)
            self._mca_dma_fault_latched = False
            self._mca_stop_uncertain = False

    def close_mca_dma_capture(
        self,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> int:
        """Public stop/drain/close wrapper used by IIOMcaDmaStreamer."""
        return self._close_mca_dma_buffer(on_frame)

    def abort_mca_dma_capture(self) -> int:
        """A failed session has no trustworthy healthy-drain operation."""
        self._mca_dma_fault_latched = True
        return self._close_mca_dma_buffer(drain=False)

    def request_mca_dma_stop(self) -> None:
        """Stop the producer without cancelling the list-mode reader.

        The in-band stop closes the current hardware frame. The worker keeps
        the existing IIO buffer/refill alive so that complete final frame can
        be consumed before close, per mca-architecture.md steps 8-10.
        """
        if self._mca_dma_pp is None:
            return
        self._mca_dma_stop_requested.set()
        if bool(int(self._mca_dma_pp_attr_get("enable"))):
            self._mca_dma_pp_attr_set("enable", "0")

        # The reference client owns a continuous reader on another thread,
        # waits one second without a complete frame, then cancels it. Our
        # pull worker can itself be blocked in the first/next refill, so the
        # same bounded inactivity rule needs an out-of-band cancellation.
        # If a final frame arrives first, stream_events() enters close and
        # cancels this timer before starting its ordinary drain reader.
        with self._mca_dma_lifecycle_lock:
            if self._mca_dma_buf is None or self._mca_dma_cancel_timer is not None:
                return

            timer: threading.Timer

            def cancel_inactive_refill() -> None:
                with self._mca_dma_lifecycle_lock:
                    if self._mca_dma_cancel_timer is not timer:
                        return
                    self._mca_dma_cancel_timer = None
                    buf = self._mca_dma_buf
                    if buf is not None:
                        self._cancel_mca_dma_refill(
                            buf,
                            "stop requested and no complete frame arrived",
                        )

            timer = threading.Timer(
                _LM_CLOSE_INACTIVITY_SECONDS,
                cancel_inactive_refill,
            )
            timer.daemon = True
            self._mca_dma_cancel_timer = timer
            timer.start()

    def get_mca_dma_capture_diagnostics(self) -> tuple[int, int, int, int]:
        """Return DMA fault/error/frame/deadtime diagnostics after close.

        ``dma_error_count`` is retained for the capture audit trail. It is a
        cumulative driver diagnostic, so unlike the three continuity checks
        it is not by itself grounds for rejecting the just-finished capture.
        """
        return (
            int(self._lm_status_attr_get("dma_fault")),
            int(self._lm_status_attr_get("dma_error_count")),
            int(self._lm_status_attr_get("completed_frames")),
            int(self._lm_status_attr_get("list_deadtime_raw")),
        )

    def get_mca_dma_capture_status(self) -> dict[str, object]:
        """Best-effort final status, including failures; never retain tracebacks."""
        status: dict[str, object] = {
            "dma_fault_kind": "lm_category",
            "recovery_required": self.mca_recovery_required(),
            "reader_active": bool(getattr(self, "_mca_dma_reader", None)),
            "cleanup_errors": list(getattr(self, "_mca_cleanup_errors", [])),
            "final_histogram_verified": False,  # A07 remains deferred.
        }
        errors: list[str] = []
        device = getattr(self, "_lm_status_device", None) or self._lm_frame
        for name in ("dma_fault", "dma_error_count", "completed_frames", "list_deadtime_raw",
                     "dma_poisoned", "queued_blocks", "kernel_buffer_blocks", "dma_queue_mode"):
            if device is not None and name in device.attrs:
                try:
                    value = self._lm_status_attr_get(name)
                    status[name] = int(value) if value.strip().isdecimal() else value
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
        try:
            status["measurement_in_progress"] = int(
                self._mca_dma_pp_attr_get("measurement_in_progress")
            )
        except Exception as exc:
            errors.append(f"measurement_in_progress: {exc}")
        if errors:
            status["status_errors"] = errors
        return status

    # ------------------------------------------------------------------
    # ScopeBackend
    # ------------------------------------------------------------------

    def get_ip_version(self) -> int:
        return int(self._attr_get("ip_version"))

    def get_mem_frame_size(self) -> int:
        return int(self._attr_get("viewer_mem_entries"))

    def get_enable(self) -> bool:
        return bool(int(self._attr_get("enable")))

    def set_enable(self, val: bool) -> None:
        """Toggle the ENABLE register directly -- a real, persistent
        hardware bit on this driver.

        Lifecycle changed board-side. A prior driver revision's
        postenable() set ENABLE=1 itself as part of arming a DMA buffer,
        so callers never touched this directly during DMA capture. The
        revision documented in vdpp-scope.c deliberately separates the
        two: postenable() arms only DMA_ENABLE, and acquisition is
        "userspace owns the acquisition gate" -- the caller must call
        this explicitly after the buffer is armed (see
        _create_dma_buffer()) and again with val=False before closing it
        (see _close_dma_buffer()) to get a clean stop -> drain -> close
        sequence instead of the driver's own crash-path fallback, which
        force-disarms in predisable() and logs a warning if it had to.

        A prior revision of this method retried val=True through -EBUSY
        for up to 2.5 s, on the theory that DMA_ENABLE could still read 1
        for up to ~1 s after a DMA buffer closes. That theory was wrong --
        per the driver author's own architecture note, dma_enable is
        driver-owned and this backend used to write it directly during
        close, which is what actually produced the stuck window the retry
        was papering over. _close_dma_buffer() now waits
        (_wait_for_dma_release()) for the previous buffer's teardown to be
        confirmed complete before returning, so a plain write here is
        correct: a genuine -EBUSY at this point means something is
        actually wrong, not a race worth retrying through.
        """
        self._attr_set("enable", "1" if val else "0")

    def scope_dma_active(self) -> bool:
        """Read-only diagnostic: the driver-owned DMA stream-gate state.

        Per the driver author's architecture note (scope-architecture.md):
        dma_enable belongs entirely to the buffer arm/close lifecycle
        (_create_dma_buffer()/_close_dma_buffer()) -- "a GUI control
        labelled DMA represents buffer arm/close, not a direct write to
        that register." There is deliberately no public write path for this
        gate; the DMA buffer lifecycle is its only owner.
        """
        return bool(int(self._attr_get("dma_enable")))

    def get_trigger_level(self) -> int:
        return int(self._attr_get("trigger_level_raw"))

    def set_trigger_level(self, val: int) -> None:
        """Write the raw threshold; the driver permits changes while armed."""
        self._attr_set("trigger_level_raw", str(val))

    def get_edge(self) -> int:
        return int(self._attr_get("trigger_mode"))

    def set_edge(self, val: int) -> None:
        """Write the mode, subject to vdpp-scope.c trigger_mode_store() guards.

        Sum transitions require stopped acquisition. An owned DMA buffer also
        forbids a transition that changes the transport format.
        """
        self._attr_set("trigger_mode", str(val))

    def get_frame_period_cycles(self) -> int:
        """Extension method (not part of ScopeBackend), called via
        d._backend -- only meaningful with trigger_mode=TriggerMode.TIMED
        (SCOPE_TRIG_PERIODIC=4). Gap after a frame, in 8 ns datapath
        clocks. Per vdpp-scope.c, the complete periodic interval is this
        value plus frame_samples / 4 clocks.
        """
        return int(self._attr_get("frame_period_cycles"))

    def set_frame_period_cycles(self, val: int) -> None:
        """Write the gap; running or buffer-owned sum geometry is immutable."""
        self._attr_set("frame_period_cycles", str(val))

    def get_pretrigger_samples(self) -> int:
        return int(self._attr_get("pretrigger_samples"))

    def set_pretrigger_samples(self, val: int) -> None:
        """Write native ADC samples under pretrigger_samples_store() guards.

        Sum mode requires zero and disallows writes while acquisition or its
        DMA buffer is active. Switch to a waveform mode before restoring a
        nonzero waveform pretrigger.
        """
        self._attr_set("pretrigger_samples", str(val))

    def get_frame_samples(self) -> int:
        return int(self._attr_get("frame_samples"))

    def get_scope_dma_geometry(self) -> ScopeDmaGeometry | ScopeSumGeometry:
        """Advertised mode-specific record geometry (PetaLinux bb8c57e).

        Raw, mean4, and structured sum layouts are distinct. A legacy fallback
        is permitted only when all three geometry attributes are absent.
        The kernel owns register encoding and advertises the exact scan count.
        """
        frame_samples = self._dma_get_frame_samples()
        mode = (
            int(self._dma_attr_get("trigger_mode"))
            if "trigger_mode" in self._dma_scope.attrs
            else 0
        )
        if mode == 5:
            version = int(self._dma_attr_get("ip_version"))
            advertised = tuple(
                int(self._dma_attr_get(name))
                for name in ("dma_frame_bytes", "dma_waveform_samples", "dma_sample_decimation")
            )
            if advertised != (4096, 0, 0):
                raise RuntimeError(f"Invalid periodic-sum geometry: {advertised}")
            return ScopeSumGeometry(
                frame_samples, version, int(self._dma_attr_get("frame_period_cycles"))
            )
        names = (
            "dma_frame_bytes",
            "dma_waveform_samples",
            "dma_sample_decimation",
        )
        present = tuple(name in self._dma_scope.attrs for name in names)
        if not any(present):
            return ScopeDmaGeometry.legacy(frame_samples)
        if not all(present):
            missing = ", ".join(name for name, available in zip(names, present) if not available)
            raise RuntimeError(f"incomplete Scope DMA geometry capability; missing {missing}")

        frame_bytes = int(self._dma_attr_get("dma_frame_bytes"))
        waveform_samples = int(self._dma_attr_get("dma_waveform_samples"))
        sample_decimation = int(self._dma_attr_get("dma_sample_decimation"))
        if frame_bytes % np.dtype("<i2").itemsize:
            raise RuntimeError(f"Scope DMA frame size {frame_bytes} is not int16-aligned")
        payload_bytes = 8 + waveform_samples * np.dtype("<i2").itemsize
        padding_bytes = frame_bytes - payload_bytes
        try:
            geometry = ScopeDmaGeometry(
                frame_samples=frame_samples,
                buffer_samples=frame_bytes // np.dtype("<i2").itemsize,
                frame_bytes=frame_bytes,
                waveform_samples=waveform_samples,
                sample_decimation=sample_decimation,
                padding_bytes=padding_bytes,
            )
        except ValueError as exc:
            raise RuntimeError(
                "driver advertised invalid Scope DMA geometry: "
                f"frame_samples={frame_samples}, frame_bytes={frame_bytes}, "
                f"waveform_samples={waveform_samples}, "
                f"sample_decimation={sample_decimation}"
            ) from exc
        if geometry.buffer_samples % 4:
            raise RuntimeError(
                "Scope DMA buffer length is not aligned to the driver's "
                f"8-byte requirement: {geometry.buffer_samples} int16 scans"
            )
        return geometry

    def get_scope_dma_runtime_metadata(self) -> dict[str, object]:
        """Return transport choices and a queue snapshot for this arm.

        This is intentionally read after the first successful frame: buffer
        creation has then selected the real transport and qualified kernel
        queue depth.  It contains scalar diagnostics only and performs no
        buffer or acquisition state change.
        """
        def optional_integer(name: str) -> int:
            if name not in self._dma_scope.attrs:
                return 0
            return int(self._dma_attr_get(name))

        return {
            "uri": self._uri,
            "device": self._dma_scope.id,
            "channel": self._ch,
            "ip_version": int(self._dma_attr_get("ip_version")),
            "transport": self._dma_capture_transport or "unarmed",
            "kernel_buffers": self._dma_capture_kernel_buffers or 0,
            "queued_blocks": optional_integer("queued_blocks"),
            "queue_high_watermark": optional_integer("queue_high_watermark"),
            "readbuf_batch_frames": (
                _SCOPE_IIOD_BATCH_FRAMES
                if self._dma_capture_transport == "iiod-batched"
                else 1
            ),
        }

    def set_frame_samples(self, val: int) -> None:
        """Closes any open DMA capture buffer first, then writes directly
        -- no disarm/re-arm of plain viewer-only `enable` needed anymore.

        Per vdpp-scope.c's frame_samples_store(): this is guarded by
        `st->buffer_active` alone now -- rejected with -EBUSY only while a
        DMA capture buffer is armed, since frame length is fixed for that
        buffer's whole armed lifetime (the DMA transfer size was sized
        from it in preenable()). A prior driver revision also rejected
        this while plain viewer-only `enable=1` with no buffer involved
        (`st->running || scope_is_running(st)`); that half of the guard
        is gone, so a plain write no longer needs disarming first unless
        a buffer is actually open.

        predisable()/postdisable() clear the buffer's hardware state as
        part of closing it, but confirmed live that a fresh write right
        after buffer.cancel() can still transiently race that teardown
        and hit -EBUSY (the same brief in-flight window documented
        elsewhere against this driver) -- retried briefly rather than
        surfacing that as a spurious failure.
        """
        if self._dma_buf is not None:
            self._close_dma_buffer()
        self._retry_ebusy(
            lambda: self._attr_set("frame_samples", str(val))
        )

    @staticmethod
    def _retry_errno(fn, errnos: tuple[int, ...], attempts: int = 10, delay_s: float = 0.02):
        """Retry fn() briefly on any of the given errno(s), returning its
        result. Two known transient windows on this driver use this:

        - EBUSY (16): the buffer-teardown callback chain (predisable()
          etc.) can still be settling for a short window after
          buffer.cancel() returns to this client -- see
          set_frame_samples()'s docstring.
        - EINVAL (22): confirmed live against the driver revision that
          separates arming a DMA buffer from starting acquisition (see
          _refill_dma_buffer()'s docstring) -- the very first
          iio_buffer_refill() right after set_enable(True) can fail with
          EINVAL if issued too soon; a prior driver revision set ENABLE
          atomically inside postenable() before ever returning to
          userspace, so this window did not exist there. A ~100ms sleep
          reliably cleared it in testing; retrying is more robust than
          hardcoding that number.
        """
        for attempt in range(attempts):
            try:
                return fn()
            except OSError as e:
                if e.errno not in errnos or attempt == attempts - 1:
                    raise
                time.sleep(delay_s)

    def _retry_ebusy(self, fn, attempts: int = 10, delay_s: float = 0.02) -> None:
        self._retry_errno(fn, (16,), attempts, delay_s)

    def get_dac_value(self) -> int:
        """DC-offset control on the AFE DAC (vdpp_afe_dac) -- see the
        module-level comment above _DAC_DEVICE_NAME for the raw-vs-baseline
        distinction. Reads the channel's `baseline` attribute, not `raw`
        (which is a different, full-range DAC output code, not what
        Scope's DAC_VALUE spec means).

        Raises RuntimeError if no DAC channel was found -- unlike the
        previous no-op-stub behavior, since a real driver now exists and a
        missing channel here means something is actually wrong (wrong
        dac_name/dac_channel, or the device tree doesn't have vdpp_afe_dac
        yet), not an expected, permanent gap.
        """
        if self._dac_ch is None:
            raise RuntimeError(
                "no AFE DAC channel bound -- pass a valid dac_name/dac_channel "
                "to IIODigitizerBackend, or check vdpp_afe_dac is present "
                "on the device tree"
            )
        return int(self._dac_ch.attrs["baseline"].value)

    def set_dac_value(self, val: int) -> None:
        """See get_dac_value()'s docstring."""
        if self._dac_ch is None:
            raise RuntimeError(
                "no AFE DAC channel bound -- pass a valid dac_name/dac_channel "
                "to IIODigitizerBackend, or check vdpp_afe_dac is present "
                "on the device tree"
            )
        self._dac_ch.attrs["baseline"].value = str(val)

    def get_dac_raw_code(self) -> int:
        """Extension diagnostic (not part of ScopeBackend): the full
        16-bit AD5689R code this channel's `raw` attribute reports, i.e.
        `baseline * baseline_code_multiplier` if nothing but baseline
        writes have touched this channel. Useful to rule out a
        baseline_code_multiplier mismatch when a baseline change appears
        to have no effect -- compare against get_dac_code_multiplier().
        """
        if self._dac_ch is None:
            raise RuntimeError(
                "no AFE DAC channel bound -- pass a valid dac_name/dac_channel "
                "to IIODigitizerBackend, or check vdpp_afe_dac is present "
                "on the device tree"
            )
        return int(self._dac_ch.attrs["raw"].value)

    def set_dac_raw_code(self, val: int) -> None:
        """Write a full-resolution AD5689R code directly, bypassing the
        baseline scaling -- diagnostic/advanced use only, see
        get_dac_raw_code()'s docstring.
        """
        if self._dac_ch is None:
            raise RuntimeError(
                "no AFE DAC channel bound -- pass a valid dac_name/dac_channel "
                "to IIODigitizerBackend, or check vdpp_afe_dac is present "
                "on the device tree"
            )
        if not 0 <= val <= 0xFFFF:
            raise ValueError("DAC raw code is outside unsigned 16-bit range")
        self._dac_ch.attrs["raw"].value = str(val)

    def get_dac_code_multiplier(self) -> int:
        """Extension diagnostic: raw AD5689R codes per baseline unit
        (vdpp-afe-dac.c's ewt,baseline-code-multiplier, DT-configurable,
        default 64 -- not necessarily what PARAMETER_SPECS assumes if the
        device tree overrides it).
        """
        if self._dac_dev is None:
            raise RuntimeError("no AFE DAC device bound")
        return int(self._dac_dev.attrs["baseline_code_multiplier"].value)

    def get_dac_command_count(self) -> int:
        """Extension diagnostic: total AD5689R SPI commands issued since
        the vdpp_afe_dac driver probed, across both channels.

        Per vdpp-afe-dac.c's own module docstring, there is no SPI
        completion flag and register readback only reports the last
        *requested* code, not a value verified against the physical DAC
        chip. vdpp_afe_dac_write_code_locked() also silently skips the
        actual write whenever the new code equals the value already in
        the register -- a legitimate optimisation, but indistinguishable
        from "nothing happened" without this counter.

        If this does not increase while changing the DAC value, the write
        never reached the HLS core (suppressed as a no-op, or landed on
        the wrong channel). If it does increase but the signal still does
        not move, the SPI command is reaching the AD5689R and the break is
        downstream, in the analogue bipolar-baseline stage the driver's
        module docstring explicitly says it does not model ("Their
        transfer function is not represented here").
        """
        if self._dac_dev is None:
            raise RuntimeError("no AFE DAC device bound")
        return int(self._dac_dev.attrs["command_count"].value)

    def get_dac_ready(self) -> bool:
        """Extension diagnostic: whether vdpp_afe_dac_probe() completed
        its own initial force-write of both channels. False here would
        mean every baseline/raw write so far has been going to a DAC that
        never got its documented reset-to-midscale synchronisation step.
        """
        if self._dac_dev is None:
            raise RuntimeError("no AFE DAC device bound")
        return bool(int(self._dac_dev.attrs["ready"].value))

    def get_viewer_frame_samples_limit(self) -> int | None:
        """Return the legacy text-viewer limit, or None for binary readout.

        Updated drivers expose ``viewer_data_raw`` and can return every
        averaged entry through the full 8188-sample hardware range. Older
        drivers retain the PAGE_SIZE-limited ``viewer_data`` text ABI.
        """
        if "viewer_data_raw" in self._scope.attrs:
            return None
        return _SCOPE_VIEWER_SAFE_FRAME_SAMPLES

    def read_frame(self) -> np.ndarray:
        """Read the on-chip pulse-viewer memory: one entry per core clock
        (frame_samples / 4 entries, capped at viewer_mem_entries=2048).

        Updated drivers expose ``viewer_data_raw`` as copied little-endian
        int16 data and can return the complete viewer memory. Confirmed live
        on both channels at frame_samples=4096 (1024 entries) and the
        hardware maximum 8188 (2047 entries). The driver performs the
        viewer semaphore/snapshot handshake as part of the attribute read.

        Older drivers fall back to the space-separated ``viewer_data``
        attribute. That ABI is capped at PAGE_SIZE, so long frames return the
        available prefix for the GUI to display explicitly as truncated.
        ``viewer_samples`` is frame_samples // 4 in both ABIs. Computing it
        locally avoids an extra network read while preserving the driver's
        geometry.
        """
        expected = min(self.get_frame_samples() // 4, self.get_mem_frame_size())
        if "viewer_data_raw" in self._scope.attrs:
            payload = self._read_binary_attr(
                self._scope,
                "viewer_data_raw",
                expected * np.dtype("<i2").itemsize,
            )
            samples = np.frombuffer(payload, dtype="<i2").copy()
            if len(samples) != expected:
                raise RuntimeError(
                    f"viewer_data_raw returned {len(samples)} of {expected} entries"
                )
            return samples

        raw = self._read_large_attr("viewer_data")
        # The oversized read buffer comes back padded with trailing NUL
        # bytes beyond the real string content (confirmed live: the byte
        # count _d_read_attr reports doesn't line up with the text's real
        # length over the network transport) -- truncate at the first NUL,
        # same convention as a C string.
        text = raw.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
        if not text:
            return np.empty(0, dtype=np.int16)
        samples = np.array(text.split(), dtype=np.int16)
        if len(samples) > expected:
            raise RuntimeError(
                f"viewer_data returned {len(samples)} entries; expected at most {expected}"
            )
        if len(samples) < expected:
            log.debug(
                "IIO scope ch%d: legacy viewer_data returned a truncated "
                "prefix (%d of %d entries)",
                self._ch,
                len(samples),
                expected,
            )
        return samples

    # ------------------------------------------------------------------
    # DMA capture -- extension method, not part of the ABC, used by
    # IIOScopeDmaStreamer in dma.py
    # ------------------------------------------------------------------

    def read_dma_frame(self) -> tuple[int, np.ndarray]:
        """Read one advertised waveform frame from the DMA path, creating the
        capture buffer on first use and reusing it on subsequent calls.

        Safe to reuse here -- unlike the previous driver rewrite, which
        needed a fresh create/refill/destroy cycle on every single call
        because reusing a buffer there corrupted data. This driver uses
        the standard IIO dmaengine buffer helper
        (devm_iio_dmaengine_buffer_alloc), explicitly designed for
        repeated refill() per vdpp-scope.c's preenable() comment: "every
        read() returns exactly one waveform" -- confirmed live (after a
        board/driver update) that one refill() now really does deliver
        exactly one frame, matching the buffer sizing in
        _refill_dma_buffer()'s docstring.

        Returns (timestamp_raw, samples), excluding the first four int16
        timestamp slots and any advertised alignment padding. On v121 this is
        ``frame_samples - 4`` full-rate ADC values. Periodic v122 firmware can
        instead advertise a decimated waveform count and sample period.
        Timestamp is decoded directly from the raw bytes to avoid signed-word
        extension corrupting the packed uint64 value.
        """
        raw = self.read_dma_raw_frame()
        geometry = self._dma_capture_geometry
        if geometry is None:
            raise RuntimeError("Scope DMA capture has no cached record geometry")
        timestamp = int.from_bytes(raw[:8], "little")
        samples = np.frombuffer(
            raw,
            dtype="<i2",
            count=geometry.waveform_samples,
            offset=8,
        )
        return timestamp, samples

    def read_dma_raw_frame(self) -> bytes:
        """Return one validated, exact raw frame without decoding/repacking.

        The file streamer uses this hot path directly. The immutable byte
        string remains independent of either the native libiio buffer or the
        batched iiod receive buffer when the next refill starts.
        """
        return self._refill_dma_buffer()

    def _refill_dma_buffer(self) -> bytes:
        """Shared by read_dma_frame() and capture_to_file(): ensure the
        persistent capture buffer exists for the current frame_samples,
        refill it, and return exactly the bytes iio_buffer_refill()
        reports for the completed frame (raw, timestamp header included,
        unparsed).

        Buffer sizing is capability-driven. Source-backed v121 requires the
        IIO scan count to equal ``frame_samples``. Deployed v122 periodic mode
        advertises ``dma_frame_bytes``, ``dma_waveform_samples`` and
        ``dma_sample_decimation``; for example, 512 configured full-rate
        samples use a 132-scan transport record. One refill still delivers
        exactly one complete frame. The remote path batches complete refills
        without splitting or enlarging a DMA block.

        Deliberately reads the exact byte count iio_buffer_refill()
        reports rather than trusting the public Buffer.read(), which
        blindly copies the entire allocated buffer regardless of how many
        bytes the refill actually delivered -- the same gotcha already
        confirmed against the previous driver (a refill can legitimately
        deliver less than the full requested buffer).

        Arming ENABLE: the driver's documented lifecycle is "arm buffer ->
        ENABLE=1 -> read -> ENABLE=0 -> drain -> close buffer" -- a newer
        driver revision arms only DMA_ENABLE in postenable() and leaves
        ENABLE for the caller to set explicitly, unlike an older revision
        which set both automatically. Checking get_enable() first instead
        of writing unconditionally works against either: already True on
        the older revision (no-op), False needing an explicit write on the
        newer one.

        Native libiio fallback refills retain the bounded retry for transient
        EINVAL (errno 22). The batched iiod transport never retries a socket
        or protocol error because its response boundary is then unknown.

        A persistent EINVAL after the bounded transient retries ends the
        session. An earlier implementation tried to recreate the buffer and
        continue in the same output file, but that cannot prove continuity
        after a framing fault and retained the old Buffer through the caught
        exception's traceback while waiting for its driver gates to clear.

        A genuine native EIO (errno 5) is handled differently from both cases
        above: per user-api.md, "An EIO returned by a normal capture
        refill() latches a fault in the Python backend. Cleanup is
        attempted exactly once, automatic rearm is blocked, and a second
        close-drain error does not replace the original EIO." This is
        distinct from the expected errno -9 (EBADF) a close-time
        Buffer.cancel() produces when it interrupts _drain_for_close()'s
        blocked refill -- that one is normal cancellation noise, not a
        session fault, and _drain_for_close() already only treats a
        refill failure as an error when it happens before stop.is_set(). Any
        batched-iio socket/protocol failure is likewise latched because the
        stream cannot safely issue a new request after a partial response.

        The first refill() after any (re)arm goes through
        _start_reader_then_enable() instead of a plain refill -- see its
        docstring for why writing enable=1 before the reader is listening
        was itself a bug, not a race worth retrying through.
        """
        if self._dma_fault_latched:
            log.warning(
                "IIO backend ch%d: capture attempted while a DMA fault is "
                "still latched -- call acknowledge_dma_recovery() first",
                self._ch,
            )
            raise RuntimeError(
                "DMA fault is latched from a previous session -- call "
                "acknowledge_dma_recovery() before capturing again"
            )
        if self._dma_stop_requested.is_set():
            raise InterruptedError(errno.ECANCELED, "scope DMA stop requested")

        # Frame geometry is immutable for the lifetime of an armed buffer:
        # vdpp-scope rejects frame_samples writes while buffer_active is set,
        # and set_frame_samples() closes our buffer before writing. Reading the
        # remote attribute on every frame adds one iiod control round trip and
        # defeats much of READBUF batching, so only read it for the initial arm.
        first = self._dma_buf is None
        if first:
            self._create_dma_buffer(self._dma_get_frame_samples())
        if self._dma_buf_frame_samples is None:
            raise RuntimeError("armed Scope DMA buffer has no cached geometry")
        n = self._dma_buf_frame_samples

        buf = self._dma_buf
        assert buf is not None
        batched_iiod = isinstance(buf, IiodScopeStream)
        refill_failed = False
        failure_errno: int | None = None
        failure_text = ""
        try:
            nbytes = self._refill_once(buf, first)
        except OSError as exc:
            # Store scalar diagnostics only. Keeping ``exc`` outside this
            # block would retain its traceback, including _refill_once()'s
            # local Buffer reference, and prevent native buffer destruction.
            refill_failed = True
            failure_errno = exc.errno
            failure_text = str(exc)

        if refill_failed:
            server_timeout = not batched_iiod or getattr(buf, "server_timed_out", False)
            del buf
            if (
                isinstance(getattr(self, "_dma_capture_geometry", None), ScopeSumGeometry)
                and getattr(self, "_sum_draining", False)
                and failure_errno in _LIBIIO_TIMEOUT_ERRNOS
                and server_timeout
            ):
                # Inspect pending/invalid status before closing a drained sum
                # buffer. Socket timeouts do not establish a complete tail.
                raise OSError(errno.ETIMEDOUT, failure_text) from None
            if self._dma_stop_requested.is_set():
                self._close_dma_buffer(drain=False)
                raise InterruptedError(
                    errno.ECANCELED, "scope DMA refill cancelled by Stop"
                ) from None

            # Any batched socket error leaves the READBUF response boundary
            # indeterminate, so that transport is terminal even when the
            # errno is a timeout or protocol error. Native libiio retains its
            # established EIO/EINVAL latch policy.
            if batched_iiod or failure_errno in (errno.EIO, errno.EINVAL):
                self._dma_fault_latched = True
                log.error(
                    "IIO backend ch%d: DMA session fault (errno %s) during "
                    "refill -- latching and closing the buffer",
                    self._ch,
                    failure_errno,
                )
                try:
                    self._close_dma_buffer(drain=False)
                except Exception:
                    log.warning(
                        "IIO backend ch%d: cleanup after a latched DMA "
                        "fault raised its own error",
                        self._ch,
                        exc_info=True,
                    )
                raise OSError(failure_errno or errno.EIO, failure_text) from None

            self._close_dma_buffer(drain=False)
            raise OSError(failure_errno or errno.EIO, failure_text) from None

        expected_bytes = n * np.dtype("<i2").itemsize
        if nbytes != expected_bytes:
            del buf
            self._dma_fault_latched = True
            try:
                self._close_dma_buffer(drain=False)
            except Exception:
                log.warning(
                    "IIO backend ch%d: cleanup after a short DMA frame "
                    "raised its own error",
                    self._ch,
                    exc_info=True,
                )
            raise OSError(
                errno.EIO,
                f"scope DMA returned {nbytes} bytes, expected {expected_bytes}",
            )

        if isinstance(buf, IiodScopeStream):
            raw = buf.read()
        else:
            start = iio._buffer_start(buf._buffer)
            raw = ctypes.string_at(start, nbytes)

        if isinstance(getattr(self, "_dma_capture_geometry", None), ScopeSumGeometry):
            return raw  # Packet decoder owns modulo timestamp and sequence validation.
        timestamp = int.from_bytes(raw[:8], "little")
        previous = getattr(self, "_dma_last_timestamp", None)
        if not timestamp or (previous is not None and timestamp <= previous):
            del buf
            self._dma_fault_latched = True
            try:
                self._close_dma_buffer(drain=False)
            except Exception:
                log.warning(
                    "IIO backend ch%d: cleanup after an invalid DMA timestamp "
                    "raised its own error",
                    self._ch,
                    exc_info=True,
                )
            raise OSError(
                errno.EIO,
                "scope DMA returned a zero, stale, or non-increasing timestamp",
            )
        self._dma_last_timestamp = timestamp
        return raw

    @staticmethod
    def _refill_scope_buffer(buf: iio.Buffer | IiodScopeStream) -> int:
        """Refill either transport and return its exact completed byte count."""
        if isinstance(buf, IiodScopeStream):
            return buf.refill()
        return iio._buffer_refill(buf._buffer)  # type: ignore[no-any-return]

    def _refill_once(self, buf: iio.Buffer | IiodScopeStream, first: bool) -> int:
        """Refill *buf* once. *first* selects _start_reader_then_enable()
        (buffer was just (re)armed, enable is not known to be 1 yet) vs a
        plain retry-tolerant refill (steady-state reuse of an already-
        running buffer) -- see _refill_dma_buffer()'s and
        _start_reader_then_enable()'s docstrings.
        """
        if first:
            return self._start_reader_then_enable(buf)
        if isinstance(buf, IiodScopeStream):
            # A socket/protocol error makes the batch boundary indeterminate;
            # IiodScopeStream marks itself failed and must never be retried.
            return self._refill_scope_buffer(buf)
        return self._retry_errno(  # type: ignore[no-any-return]
            lambda: self._refill_scope_buffer(buf), (22,)
        )

    def _start_reader_then_enable(self, buf: iio.Buffer | IiodScopeStream) -> int:
        """Issue the first blocking refill() on a freshly armed buffer
        *before* writing enable=1, per user-api.md's "Running a
        measurement" step 6: "start the blocking reader, then write
        enable=1."

        Every earlier revision of this backend (and, before this fix, this
        one) did it the other way around: _create_dma_buffer() wrote
        enable=1 itself, and only afterward did the caller issue the first
        refill(). That ordering is exactly the "started too early" fault
        vdpp-scope.c's own preenable() comment describes: creating the
        IIO buffer queues a DMA descriptor, but nothing tells the DMA
        engine to actually start processing it
        (dma_async_issue_pending()) until refill() is called. Writing
        enable=1 first opens a window where a trigger can fire and the
        core starts streaming before the engine is listening for it --
        confirmed live as the cause of a "just armed, first refill faults
        immediately" failure that left no trace in dmesg (unlike the
        confirmed hardware-fault cases this session also hit), meaning it
        was a client-side ordering bug, not a channel fault, all along.

        A background thread issues the blocking refill() first; once it
        has confirmed entering that call (an Event, not just thread
        start -- matching _drain_for_close()'s existing pattern, with the
        same small, accepted residual race between "thread scheduled" and
        "the C call actually dispatched"), only then does this method
        write enable=1. The reader thread's result or exception is joined
        back here.
        """
        entered = threading.Event()
        result: list[int] = []
        errors: list[tuple[int, str]] = []

        def reader() -> None:
            entered.set()
            try:
                result.append(self._refill_scope_buffer(buf))
            except BaseException as exc:
                errors.append((getattr(exc, "errno", None) or errno.EIO, str(exc)))

        thread = threading.Thread(
            target=reader,
            name=f"vdpp-scope-ch{self._ch}-dma-first-refill",
            daemon=True,
        )
        self._scope_dma_reader = thread
        thread.start()
        if not entered.wait(_DMA_FIRST_REFILL_ENTER_SECONDS):
            try:
                buf.cancel()
            finally:
                self._join_dma_reader("_scope_dma_reader", _DMA_CLOSE_JOIN_SECONDS)
            raise RuntimeError(
                "DMA reader did not enter its first refill; acquisition remains stopped"
            )

        try:
            sum_cancelled = (
                isinstance(getattr(self, "_dma_capture_geometry", None), ScopeSumGeometry)
                and getattr(self, "_sum_draining", False)
            )
            if self._dma_stop_requested.is_set() or sum_cancelled:
                raise InterruptedError(errno.ECANCELED, "scope DMA stop requested before enable")
            if not self._dma_get_enable():
                self._dma_set_enable(True)
        except BaseException as primary:
            try:
                try:
                    buf.cancel()
                finally:
                    self._join_dma_reader("_scope_dma_reader", _DMA_CLOSE_JOIN_SECONDS)
            except Exception as cleanup:
                primary.add_note(f"Scope startup cleanup: {cleanup}")
            raise
        thread.join()
        self._scope_dma_reader = None
        if errors:
            raise OSError(*errors[0]) from None
        return result[0]

    def _create_dma_buffer(self, n: int) -> None:
        """Close any existing buffer and arm the layout for configured n.

        The actual IIO scan count comes from :meth:`get_scope_dma_geometry`,
        so dynamic v122 periodic records are not confused with the configured
        full-rate capture length. Does not touch enable itself; see
        _start_reader_then_enable()'s docstring for why that write moved
        out of here and into the caller, ordered after the first refill()
        has already started.

        Per scope-architecture.md, transitioning from viewer-only into DMA
        needs its own settle step, distinct from _wait_for_dma_release()'s
        (much longer) post-buffer-destruction wait: write enable=0, wait
        1 ms, then passively verify both DMA ownership gates already read
        zero. The v121 FSM has no M_IDLE status, but the longest
        viewer-only frame completes far sooner than 1 ms and (unlike a DMA
        frame) can't be extended by AXI backpressure, since viewer-only
        mode has no AXI receiver.
        """
        if self._dma_fault_latched:
            # Reachable defense-in-depth: _refill_dma_buffer() already
            # checks this before ever calling here, so in practice this
            # path is not hit through the normal call chain. Not logged
            # separately -- see _refill_dma_buffer()'s own check for the
            # one that actually fires.
            raise RuntimeError(
                "DMA fault is latched from a previous session -- call "
                "acknowledge_dma_recovery() before arming a new capture"
            )
        if self._dma_stop_requested.is_set():
            raise InterruptedError(errno.ECANCELED, "scope DMA stop requested")
        if (
            n < _SCOPE_MIN_FRAME_SAMPLES
            or n > _SCOPE_MAX_FRAME_SAMPLES
            or n % 4
        ):
            raise ValueError(
                f"invalid Scope DMA frame geometry {n}; expected a multiple "
                f"of 4 in {_SCOPE_MIN_FRAME_SAMPLES}..{_SCOPE_MAX_FRAME_SAMPLES}"
            )

        self._close_dma_buffer()

        if self._dma_get_enable():
            self._dma_set_enable(False)
        time.sleep(_VIEWER_TO_DMA_SETTLE_SECONDS)
        if self._dma_get_dma_enable() or int(self._dma_attr_get("dma_buffer_active")):
            # Confirmed live: this exact condition has produced a genuine
            # xilinx-vdma channel fault (dmesg "has errors", not just a
            # bounded "Cannot stop channel") when ignored -- ERROR, since
            # it means the settle check this method exists for just caught
            # a real problem, not routine timing noise.
            log.error(
                "IIO backend ch%d: dma_enable/dma_buffer_active did not "
                "read back as zero %.0f ms after stopping acquisition -- "
                "refusing to arm DMA", self._ch,
                _VIEWER_TO_DMA_SETTLE_SECONDS * 1000,
            )
            raise RuntimeError(
                "cannot arm DMA capture: dma_enable/dma_buffer_active did "
                "not read back as zero after stopping acquisition -- the "
                "scope may still be finishing a viewer-only frame, or a "
                "prior session was not fully torn down"
            )

        geometry = self.get_scope_dma_geometry()
        if geometry.frame_samples != n:
            raise RuntimeError(
                "Scope DMA geometry changed while arming: "
                f"configured {n} samples, advertised {geometry.frame_samples}"
            )
        buffer_samples = geometry.buffer_samples

        kernel_buffers = self._scope_dma_kernel_buffers()
        # PetaLinux bb8c57e reference arm_capture(): queued remote Scope
        # READBUF batching applies to raw, mean4 and structured sum packets.
        use_batched_iiod = (
            self._uri.startswith("ip:")
            and "dma_queue_mode" in self._dma_scope.attrs
            and self._dma_attr_get("dma_queue_mode") == "dmaengine"
        )
        try:
            self._dma_scope.set_kernel_buffers_count(kernel_buffers)
        except AttributeError as exc:
            raise RuntimeError(
                "the Python libiio binding cannot select the required Scope "
                "DMA kernel-buffer depth"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "failed to select Scope DMA kernel-buffer depth before arming"
            ) from exc

        if isinstance(geometry, ScopeSumGeometry):
            self._dma_ctx.set_timeout(2000)
        if use_batched_iiod:
            self._dma_buf = IiodScopeStream(
                self._uri,
                self._dma_scope.id,
                buffer_samples,
                kernel_buffers,
                _SCOPE_IIOD_BATCH_FRAMES,
                timeout_ms=2000 if isinstance(geometry, ScopeSumGeometry) else 3000,
            )
            self._dma_capture_transport = "iiod-batched"
        else:
            self._dma_buf = iio.Buffer(self._dma_scope, buffer_samples, False)
            self._dma_capture_transport = "libiio"
        self._dma_buf_frame_samples = buffer_samples
        self._dma_capture_geometry = geometry
        self._dma_capture_kernel_buffers = kernel_buffers
        log.debug(
            "IIO backend ch%d: DMA buffer armed, frame_samples=%d, "
            "buffer_samples=%d, waveform_samples=%d, decimation=%d, "
            "transport=%s, kernel_buffers=%d",
            self._ch,
            n,
            buffer_samples,
            geometry.waveform_samples,
            geometry.sample_decimation,
            self._dma_capture_transport,
            kernel_buffers,
        )

    def _scope_dma_kernel_buffers(self) -> int:
        """Select queue depth only from the provider-qualified driver ABI."""
        attrs = self._dma_scope.attrs
        maximum = recommended = _SCOPE_FALLBACK_KERNEL_BUFFER_COUNT
        if (
            "dma_queue_mode" in attrs
            and self._dma_attr_get("dma_queue_mode") == "dmaengine"
        ):
            capability_names = (
                "dma_kernel_buffers_max",
                "dma_kernel_buffers_recommended",
            )
            if all(name in attrs for name in capability_names):
                maximum = int(self._dma_attr_get("dma_kernel_buffers_max"))
                recommended = int(self._dma_attr_get("dma_kernel_buffers_recommended"))
                if (
                    maximum not in (2, 4)
                    or recommended not in (1, 2, 4)
                    or recommended > maximum
                ):
                    raise RuntimeError("invalid Scope DMA queue capability")
        return recommended

    def capture_to_file(
        self, path: str, duration_s: float | None = None, max_frames: int | None = None,
    ) -> int:
        """One measurement, one file -- extension method mirroring the
        reference stub's (scope_backend_iio.py) capture_to_file(). Returns
        the number of frames written.

        Writes each frame's raw bytes exactly as the hardware produced
        them (64-bit timestamp header included, unmodified), so the file
        is self-describing for an offline reader the same way the stub's
        is -- one frame is exactly frame_samples int16 values, the first 4
        being the packed timestamp.

        Runs until *duration_s* elapses, *max_frames* frames have been
        written, or both are None and it never stops on its own (pass at
        least one). Always closes the capture buffer afterward, success
        or failure -- same cleanup guarantee as read_dma_frame(), just
        wrapped around a loop instead of a single call.

        One refill() per frame (see _refill_dma_buffer()'s docstring), so
        one loop iteration is exactly one frame written to file.
        """
        self.prepare_dma_capture()
        frames = 0
        deadline = time.monotonic() + duration_s if duration_s else None
        try:
            with open(path, "wb") as f:
                while True:
                    if deadline is not None and time.monotonic() > deadline:
                        break
                    if max_frames is not None and frames >= max_frames:
                        break
                    f.write(self._refill_dma_buffer())
                    frames += 1
        finally:
            self._close_dma_buffer()

        return frames

    # ------------------------------------------------------------------
    # MCABackend -- vdpp-pulse-processor.c + vdpp-input-filter.c
    # (see module docstring for the handful of methods that stay stubbed)
    #
    # PetaLinux bb8c57e, mca-parameter-abi-v2.md: native ADC samples.
    # Never reapply the kernel's register encoding in these methods.

    def mca_hardware_present(self) -> bool:
        """Unknown MCA contracts are read-only; independent Scope remains usable."""
        return self._input_filter is not None and bool(self.mca_parameter_contract()["supported"])

    def mca_parameter_contract(self) -> dict[str, object]:
        pp = self._pp
        if pp is None:
            return {"supported": False, "reason": "Pulse processor absent"}
        try:
            version = int(pp.attrs["parameter_abi_version"].value)
            units = str(pp.attrs["parameter_units"].value).strip()
        except (KeyError, ValueError, OSError) as exc:
            return {"supported": False, "reason": f"Cannot discover MCA ABI: {exc}"}
        return {"supported": version == 2 and units == "adc_samples",
                "parameter_abi_version": version, "parameter_units": units}

    @property
    def mca_configuration_lock(self) -> threading.RLock:
        # Shared by complete histogram assemblies and all control-context writes.
        if not hasattr(self, "_configuration_lock"):
            self._configuration_lock = threading.RLock()
        return self._configuration_lock

    @staticmethod
    def _require_mca_contract(pp: iio.Device | None, *, acknowledged: bool = True) -> None:
        if pp is None:
            raise RuntimeError("MCA parameter ABI2 discovery failed; pulse processor absent")
        try:
            valid = (
                int(pp.attrs["parameter_abi_version"].value) == 2
                and str(pp.attrs["parameter_units"].value).strip() == "adc_samples"
            )
            ack = int(pp.attrs["parameter_abi_ack"].value) if acknowledged else 2
        except (AttributeError, KeyError, ValueError, OSError) as exc:
            raise RuntimeError(
                "MCA parameter ABI2 discovery failed; no configuration permitted"
            ) from exc
        if not valid or ack != 2:
            raise RuntimeError("MCA requires ABI2/adc_samples and an explicit idle acknowledgement")

    def prepare_mca_configuration(self, values: dict[str, int]) -> None:
        validate_parameters(values, complete=True)
        self._validate_psd_mode(values["psd_zero_crossing_mode"])
        with self.mca_configuration_lock:
            pp = self._pp
            self._require_mca_contract(pp, acknowledged=False)
            assert pp is not None
            if int(pp.attrs["enable"].value) or int(pp.attrs["list_buffer_active"].value):
                raise RuntimeError("Stop MCA and close list DMA before acknowledging ABI2")
            if self._lm_frame is not None and "dma_poisoned" in self._lm_frame.attrs:
                if int(self._lm_frame.attrs["dma_poisoned"].value):
                    raise RuntimeError("MCA DMA is poisoned; reboot the target before reuse")
            pp.attrs["parameter_abi_ack"].value = "2"
            self._require_mca_contract(pp)

    def mca_parameter_snapshot(self) -> dict[str, int]:
        with self.mca_configuration_lock:
            self._require_mca_contract(self._pp, acknowledged=False)
            return {name: int(self._pp_attr_get(name)) for name in PARAMETERS}

    def configure_mca_parameters(self, values: dict[str, int]) -> None:
        values = validate_parameters(values)
        with self.mca_configuration_lock:
            candidate = self.mca_parameter_snapshot() | values
            self.prepare_mca_configuration(candidate)
            self._assert_mca_idle(self._pp)
            # Disable algorithms first; final admission uses the complete candidate.
            enables = [name for name in PARAMETERS if name.endswith("_enable")]
            for name in enables:
                self._pp_attr_set(name, "0")
            try:
                for name, value in candidate.items():
                    if name not in enables:
                        self._pp_attr_set(name, str(value))
                for name in enables:
                    self._pp_attr_set(name, str(candidate[name]))
            except Exception:
                self._pp_attr_set("enable", "0")
                raise

    def supports_dc_blocker(self) -> bool:
        return self._input_filter is not None and "dc_blocker_disable" in self._input_filter.attrs

    def get_dc_blocker_bypass(self) -> bool:
        return bool(int(self._if_attr_get("dc_blocker_disable")))

    def set_dc_blocker_bypass(self, value: bool) -> None:
        self._if_attr_set("dc_blocker_disable", "1" if value else "0")

    def mca_producer_identity(self) -> dict[str, int | str]:
        """Bind the selected event decoder to a source-derived producer identity.

        Keep the existing application decoder for its supported producer family.
        PetaLinux c546f14 coincidence-frontend-contract.md distinguishes this
        source-derived interpretation from physical qualification of an image.
        Missing image identity is recorded as unknown, never invented or required
        merely to connect. An optional deployment profile overrides the default.
        """
        with self._mca_dma_control_lock:
            pp = self._require_mca_dma_pp()
            identity: dict[str, int | str] = {
                "pulse_hw_version": int(pp.attrs["hw_version"].value),
                "pulse_sw_version": int(pp.attrs["sw_version"].value),
                "ip_version": int(self._lm_status_attr_get("ip_version")),
                "parameter_abi_version": int(pp.attrs["parameter_abi_version"].value),
                "parameter_units": str(pp.attrs["parameter_units"].value).strip(),
            }
        profile = getattr(self, "_acquisition_profile", None)
        if profile is not None:
            identity["firmware_id"] = profile.firmware_id
            if profile.record_schema:
                # Retain the public selector key; this is source-derived
                # interpretation, not a claim of live physical qualification.
                identity["qualified_record_schema"] = profile.record_schema
                identity["decoder_selection"] = (
                    "explicit firmware profile; physical qualification unverified"
                )
        elif (
            identity["pulse_hw_version"], identity["pulse_sw_version"],
            identity["ip_version"], identity["parameter_abi_version"], identity["parameter_units"]
        ) == (101, 103023, 121, 2, "adc_samples"):
            # Preserve the pre-migration application default without treating
            # matching register versions as proof of deployed producer packing.
            # The historical selector key is retained for existing consumers.
            identity["qualified_record_schema"] = VDPP_ZC_CALC_SCHEMA
            identity["firmware_id"] = "unknown"
            identity["decoder_selection"] = (
                "application default; source-derived; deployment unverified"
            )
        identity["energy_field"] = "selected estimator; not universally trapezoid energy"
        return identity

    def mca_capture_provenance(self) -> dict[str, object]:
        """Record effective settings through the session control connection."""
        with self.mca_configuration_lock, self._mca_dma_control_lock:
            pp = self._require_mca_dma_pp()
            return {
                "uri": self._uri, "channel_index": self._ch,
                "producer_identity": self.mca_producer_identity(),
                "parameters": {name: int(pp.attrs[name].value) for name in PARAMETERS},
                "parameter_abi_version": 2, "parameter_units": "adc_samples",
                "acquisition_profile": self.acquisition_profile(),
                "timebase": "nominal 500 MSPS ADC / 125 MHz processing; physically unverified",
            }

    def supports_dac(self) -> bool:
        return self._dac_ch is not None

    def acquisition_profile(self) -> dict[str, object]:
        profile = getattr(self, "_acquisition_profile", None)
        return profile.to_dict() if profile is not None else {}

    def apply_acquisition_profile(self, values: dict[str, object]) -> None:
        profile = AcquisitionProfile.from_mapping(values)
        with self.mca_configuration_lock:
            if self.get_enable() or self.scope_dma_active():
                raise RuntimeError(
                    "Stop Scope and close DMA before changing its hardware profile"
                )
            if self._pp is not None and (self.get_global_enable() or self.mca_dma_active()):
                raise RuntimeError(
                    "Stop MCA and close list DMA before changing its hardware profile"
                )
            selected = None
            if profile.dac_output is not None:
                if self._dac_dev is not None:
                    selected = self._dac_dev.find_channel(f"voltage{profile.dac_output}", True)
                if selected is None:
                    raise ValueError("Selected verified DAC output is absent")
            self._dac_ch = selected
            self._acquisition_profile = profile

    def mca_dma_hardware_present(self) -> bool:
        """Whether the dedicated context found both paired DMA devices."""
        return self._mca_dma_pp is not None and self._lm_frame is not None

    def get_lm_ip_version(self) -> int:
        """vdpp_lm_frame diagnostic extension; expected value is 121."""
        return int(self._lm_status_attr_get("ip_version"))

    def get_lm_frame_records(self) -> int:
        return int(self._lm_status_attr_get("frame_records"))

    def get_lm_frame_bytes(self) -> int:
        return int(self._lm_status_attr_get("frame_bytes"))

    def get_list_deadtime_raw(self) -> int:
        """Records dropped while lm_frame's output was full (user-api.md)."""
        return int(self._lm_attr_get("list_deadtime_raw"))

    def get_lm_buffer_active(self) -> bool:
        return bool(int(self._lm_attr_get("buffer_active")))

    def mca_dma_measurement_in_progress(self) -> bool:
        """DMA-context status used to detect hardware time-limit stop."""
        return bool(int(self._mca_dma_pp_attr_get("measurement_in_progress")))

    def get_hw_version(self) -> int:
        return int(self._pp_attr_get("hw_version"))

    def get_sw_version(self) -> int:
        return int(self._pp_attr_get("sw_version"))

    def get_id_number(self) -> int:
        return int(self._pp_attr_get("id_number"))

    def get_dpp_trigger_level(self) -> int:
        return int(self._pp_attr_get("trigger_level_raw"))

    def set_dpp_trigger_level(self, val: int) -> None:
        self._pp_attr_set("trigger_level_raw", str(val))

    def get_pulse_polarity(self) -> int:
        return int(self._pp_attr_get("pulse_polarity"))

    def set_pulse_polarity(self, val: int) -> None:
        self._pp_attr_set("pulse_polarity", str(val))

    def get_bsln_window(self) -> int:
        return int(self._pp_attr_get("baseline_window"))

    def set_bsln_window(self, val: int) -> None:
        self._pp_attr_set("baseline_window", str(val))

    def get_dpp_pretrigger_samples(self) -> int:
        """ABI2 native ADC samples (2 ns each); kernel owns register encoding."""
        return int(self._pp_attr_get("pretrigger_samples"))

    def set_dpp_pretrigger_samples(self, val: int) -> None:
        self._pp_attr_set("pretrigger_samples", str(val))

    def get_dpp_frame_samples(self) -> int:
        """ABI2 native ADC samples (2 ns each); kernel owns register encoding."""
        return int(self._pp_attr_get("frame_samples"))

    def set_dpp_frame_samples(self, val: int) -> None:
        self._pp_attr_set("frame_samples", str(val))

    def get_global_enable(self) -> bool:
        return bool(int(self._pp_attr_get("enable")))

    def set_global_enable(self, val: bool) -> None:
        self._pp_attr_set("enable", "1" if val else "0")

    def mca_dma_active(self) -> bool:
        """Read the driver-owned MCA list-buffer ownership gate.

        Per mca-architecture.md: "The list_dma_enable register physically
        lives in this core but is owned by vdpp-lm-frame... Keeping that
        gate out of userspace prevents the AXI stream from being opened
        before a DMA descriptor exists." vdpp-pulse-processor.c exposes no
        writable list_dma_enable attribute at all -- only the read-only
        list_buffer_active, set internally via the exported
        vdpp_pulse_set_list_buffer() when a vdpp-lm-frame buffer is
        armed/closed. The IIO list-mode extension methods in this backend
        create and destroy that buffer; callers never write the gate directly.
        """
        return bool(int(self._pp_attr_get("list_buffer_active")))

    def get_time_limit(self) -> int:
        """Seconds, per mca.py's documented contract for this method.

        measurement_time_raw (PP_FMT_RAW, no built-in scaling) and the
        read-only measurement_time_scale attribute (seconds per raw tick,
        "0.134217728" -- confirmed from vdpp-pulse-processor.c) together
        give the physical value; scale is read live rather than hardcoded
        so a firmware change to the tick period doesn't silently go stale
        here.
        """
        raw = int(self._pp_attr_get("measurement_time_raw"))
        scale = float(self._pp_attr_get("measurement_time_scale"))
        return round(raw * scale)

    def set_time_limit(self, val: int) -> None:
        scale = float(self._pp_attr_get("measurement_time_scale"))
        self._pp_attr_set("measurement_time_raw", str(round(val / scale)))

    def get_ext_trig_enable(self) -> bool:
        return bool(int(self._pp_attr_get("external_trigger_enable")))

    def set_ext_trig_enable(self, val: bool) -> None:
        self._pp_attr_set("external_trigger_enable", "1" if val else "0")

    def get_trg_source(self) -> int:
        return int(self._pp_attr_get("trigger_source"))

    def set_trg_source(self, val: int) -> None:
        self._pp_attr_set("trigger_source", str(val))

    def get_energy_bin(self) -> int:
        return int(self._pp_attr_get("energy_bin"))

    def set_energy_bin(self, val: int) -> None:
        self._pp_attr_set("energy_bin", str(val))

    def get_pileup_window(self) -> int:
        return int(self._pp_attr_get("pileup_window_raw"))

    def set_pileup_window(self, val: int) -> None:
        self._pp_attr_set("pileup_window_raw", str(val))

    def get_measurement_in_progress(self) -> bool:
        """Routed through self._mca_pp (background-poll context, see
        __init__'s comment on self._mca_ctx) since MCAWorker's tick calls
        this every interval. MCAController._ensure_disarmed() also calls
        this once, from the GUI thread, during shutdown/reconnect -- per
        MainWindowController._stop_all_workers()'s ordering, that can
        happen while the worker thread is still ticking, so this one
        method has no fully contention-free context to use. Accepted as a
        narrow, low-probability residual race (a single blocking read, not
        a sustained stream) rather than adding locking for it.
        """
        return bool(int(self._mca_pp_attr_get("measurement_in_progress")))

    # Statistics -- all read every MCAWorker tick (see MCAReadback), so all
    # nine go through self._mca_pp (background-poll context), not self._pp.

    def get_pulse_deadtime(self) -> int:
        """Raw tick count from ``dead_time_raw``.

        The ABC retains this legacy raw-counter method. GUI code uses
        :meth:`get_pulse_deadtime_ms` so the documented 0.524288 ms/count
        scale is not lost.
        """
        return int(self._mca_pp_attr_get("dead_time_raw"))

    def get_pulse_deadtime_ms(self) -> float:
        """Physical milliseconds from the live raw value and driver scale.

        ``user-api.md`` and ``vdpp-pulse-processor.c`` define
        ``dead_time_scale`` as milliseconds per ``dead_time_raw`` count.
        Reading the scale keeps the GUI correct if a later compatible image
        changes that constant.
        """
        raw = int(self._mca_pp_attr_get("dead_time_raw"))
        scale = float(self._mca_pp_attr_get("dead_time_scale"))
        return raw * scale

    def get_events_lost(self) -> int:
        raise NotImplementedError(_MCA_EVENTS_LOST_UNSUPPORTED)

    def get_count_rate(self) -> int:
        return int(self._mca_pp_attr_get("count_rate_raw"))

    def get_elapsed_time(self) -> int:
        """Return the elapsed register unchanged in deciseconds (0.1 s).

        PetaLinux 167e90f vdpp-pulse-processor.c exposes PP_REG_ENABLE_TIME
        unscaled via PP_STATUS_SHOW. Do not reuse measurement_time_scale:
        it describes the separate programmed-duration register, not elapsed
        status. Read-only qualification on PP hw101/sw103023 (2026-10-22)
        found enable_time_raw=999 after a watch-timed 100 s run, while the
        duration setting was 745 * 0.134217728 = 99.992 s. Scaling elapsed
        with that duration scale incorrectly displayed 134.1 s.

        This preserves MCAStatistics' existing decisecond API; the GUI and
        elapsed-normalized consumers divide by ten exactly once.
        """
        return int(self._mca_pp_attr_get("enable_time_raw"))

    def get_pulse_overrange_counter(self) -> int:
        return int(self._mca_pp_attr_get("pulse_overrange_counter"))

    def get_pulse_pileup_counter(self) -> int:
        return int(self._mca_pp_attr_get("pulse_pileup_counter"))

    def get_energy_overrange_counter(self) -> int:
        return int(self._mca_pp_attr_get("energy_overrange_counter"))

    def get_energy_estimation_error(self) -> int:
        return int(self._mca_pp_attr_get("energy_estimation_error_counter"))

    def get_throughput_error_counter(self) -> int:
        return int(self._mca_pp_attr_get("throughput_error_counter"))

    def get_lp_coeffs_size(self) -> int:
        return int(self._if_attr_get("fir_coefficient_count"))

    def set_lp_coeffs_preset(self, val: int) -> None:
        """Preset index passes straight through to fir_preset -- the
        driver's own preset order (vdpp_input_filter_presets[]: 0="200mhz",
        1="70mhz", 2="moving_average") already matches index-for-index
        what MultiChannelAnalyzer.filters.lp.set_preset() validates against
        (MCAParam.LP_PRESET's ListSpec is (0, 1, 2)), despite mca.py's own
        comment there mislabeling index 1 as "700 MHz" instead of 70 MHz.
        """
        self._if_attr_set("fir_preset", str(val))

    def get_lp_coeffs(self) -> np.ndarray:
        raw = self._if_attr_get("fir_coefficients_raw")
        return np.array([int(v) for v in raw.split()], dtype=np.int32)

    def set_lp_coeffs(self, coeffs: list[int]) -> None:
        """vdpp_input_filter_parse_coefficients() rejects anything but
        exactly VDPP_INPUT_FILTER_COEFFICIENTS (12) values with -EINVAL;
        checked here first for a clear client-side error instead of an
        opaque OSError from the wire.
        """
        if len(coeffs) != _INPUT_FILTER_FIR_COEFFICIENTS:
            raise ValueError(
                f"expected {_INPUT_FILTER_FIR_COEFFICIENTS} FIR coefficients, "
                f"got {len(coeffs)}"
            )
        self._if_attr_set("fir_coefficients_raw", " ".join(str(int(c)) for c in coeffs))

    def get_iir_lp_average(self) -> int:
        """vdpp-input-filter.c's only IIO_CHAN_INFO_RAW channel: reads
        IIR_OUT_REG through the standard channel-attribute path (same
        pattern as get_dac_raw_code()'s `self._dac_ch.attrs["raw"].value`),
        not a plain device attribute.
        """
        if self._input_filter_ch is None:
            raise RuntimeError(
                f"no {_INPUT_FILTER_DEVICE_NAME} channel bound -- input-"
                "filter core not present on this channel's firmware"
            )
        return int(self._input_filter_ch.attrs["raw"].value)

    def get_temp_coeff(self) -> float:
        """temperature_coefficient_raw (s16) x temperature_coefficient_scale
        (0.000030517578125 = 2^-15, read live rather than hardcoded, same
        reasoning as get_time_limit()'s measurement_time_scale read).
        """
        raw = int(self._if_attr_get("temperature_coefficient_raw"))
        scale = float(self._if_attr_get("temperature_coefficient_scale"))
        return raw * scale

    @staticmethod
    def _encode_temperature_coefficient(value: float, scale: float) -> int:
        """Encode the driver's signed 16-bit scaled value, rejecting overflow.

        vdpp-input-filter.c accepts s16 raw values with scale 2^-15. Validate
        physical bounds before rounding so +1.0 cannot become raw +32768.
        """
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("temperature coefficient scale must be finite and positive")
        minimum, maximum = -32768 * scale, 32767 * scale
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"temperature coefficient {value} out of range [{minimum}, {maximum}]"
            )
        return round(value / scale)

    def set_temp_coeff(self, val: float) -> None:
        scale = float(self._if_attr_get("temperature_coefficient_scale"))
        raw = self._encode_temperature_coefficient(val, scale)
        self._if_attr_set("temperature_coefficient_raw", str(raw))

    def get_temp_offset(self) -> int:
        return int(self._if_attr_get("temperature_offset_raw"))

    def set_temp_offset(self, val: int) -> None:
        self._if_attr_set("temperature_offset_raw", str(val))

    def set_temperature_correction_from_worker(
        self,
        coefficient: float,
        offset: int,
    ) -> None:
        """Write both correction terms through the worker-only IIO context."""
        if self._temperature_correction_ctx is None:
            context = iio.Context(self._uri)
            filters = _devices_named(context, _INPUT_FILTER_DEVICE_NAME)
            if self._ch >= len(filters):
                raise RuntimeError(
                    f"no {_INPUT_FILTER_DEVICE_NAME} device for channel "
                    f"{self._ch} in the temperature-correction context"
                )
            self._temperature_correction_ctx = context
            self._temperature_correction_filter = filters[self._ch]
        device = self._temperature_correction_filter
        if device is None:
            raise RuntimeError(
                f"no {_INPUT_FILTER_DEVICE_NAME} device bound for MCA "
                "temperature correction"
            )
        scale = float(device.attrs["temperature_coefficient_scale"].value)
        raw = self._encode_temperature_coefficient(coefficient, scale)
        device.attrs["temperature_coefficient_raw"].value = str(raw)
        device.attrs["temperature_offset_raw"].value = str(offset)

    def get_edge_det_coeff(self) -> int:
        """Software-only shadow, no hardware effect -- see module docstring.
        Neither vdpp-pulse-processor.c nor vdpp-input-filter.c expose an
        edge-detector-coefficient register in this HLS core revision.
        MCAController._send_defaults()/_load_hardware_state() call
        get/set_edge_det_coeff() unconditionally during construction, so
        raising NotImplementedError here (the honest answer -- there is no
        register) would crash every MCA dock at startup; a plain in-memory
        value keeps the shared, backend-agnostic MCAController working
        without a per-backend special case.
        """
        return self._edge_det_coeff

    def set_edge_det_coeff(self, val: int) -> None:
        self._edge_det_coeff = val

    def edge_det_coeff_is_hardware_backed(self) -> bool:
        """The current IIO pulse processor exposes no such register."""
        return False

    def supports_crrc2_reciprocal_raw(self) -> bool:
        """Detect the additive vdpp-pulse-processor CR-RC2 1/C control ABI."""
        return self._pp is not None and "crrc2_1cdelay_raw" in self._pp.attrs

    def get_crrc2_reciprocal_raw(self) -> int:
        """Read the driver shadow of the v101 0x00f8 unsigned 16-bit register.

        The generated xpulse_processor_hw.h defines its width but not its
        numerical scale. Preserve raw units rather than guessing a Q format.
        """
        if not self.supports_crrc2_reciprocal_raw():
            raise NotImplementedError("update vdpp-mca: driver lacks crrc2_1cdelay_raw")
        return int(self._pp_attr_get("crrc2_1cdelay_raw"))

    def set_crrc2_reciprocal_raw(self, val: int) -> None:
        if not isinstance(val, int) or not 0 <= val <= 65535:
            raise ValueError("CR-RC2 1/C must be an unsigned 16-bit integer")
        if not self.supports_crrc2_reciprocal_raw():
            raise NotImplementedError("update vdpp-mca: driver lacks crrc2_1cdelay_raw")
        # Raw passthrough; the driver retains the normal measurement/DMA busy gate.
        self._pp_attr_set("crrc2_1cdelay_raw", str(val))

    def get_crrc2_Cdelay(self) -> int:
        """ABI2 native value; kernel owns register encoding."""
        return int(self._pp_attr_get("crrc2_cdelay"))

    def set_crrc2_Cdelay(self, val: int) -> None:
        validate_parameters({"crrc2_cdelay": val})
        self._pp_attr_set("crrc2_cdelay", str(val))

    def get_crrc2_Fdelay(self) -> int:
        """ABI2 native value; kernel owns register encoding."""
        return int(self._pp_attr_get("crrc2_fdelay_samples"))

    def set_crrc2_Fdelay(self, val: int) -> None:
        self._pp_attr_set("crrc2_fdelay_samples", str(val))

    def get_crrc2_pzc_coeff(self) -> int:
        return int(self._pp_attr_get("crrc2_pzc_raw"))

    def set_crrc2_pzc_coeff(self, val: int) -> None:
        self._pp_attr_set("crrc2_pzc_raw", str(val))

    def get_trapez_enable(self) -> bool:
        return bool(int(self._pp_attr_get("trapezoid_enable")))

    def set_trapez_enable(self, val: bool) -> None:
        self._pp_attr_set("trapezoid_enable", "1" if val else "0")

    def supports_trapezoid_reciprocal_raw(self) -> bool:
        return self._pp is not None and "trapezoid_1r_raw" in self._pp.attrs

    def get_trapezoid_reciprocal_raw(self) -> int:
        """Read the driver shadow of 0x0110, derived when trapezoid_r is set.

        Optional readback is distinct from the ABI2 calculated 32768/(R/4).
        The current reference does not export this optional attribute.
        """
        if not self.supports_trapezoid_reciprocal_raw():
            raise NotImplementedError("update vdpp-mca: driver lacks trapezoid_1r_raw")
        return int(self._pp_attr_get("trapezoid_1r_raw"))

    def get_trapez_R(self) -> int:
        """ABI2 native value; kernel owns register encoding."""
        return int(self._pp_attr_get("trapezoid_r_samples"))

    def set_trapez_R(self, val: int) -> None:
        self._pp_attr_set("trapezoid_r_samples", str(val))

    def get_trapez_M(self) -> int:
        """ABI2 native value; kernel owns register encoding."""
        return int(self._pp_attr_get("trapezoid_m_samples"))

    def set_trapez_M(self, val: int) -> None:
        self._pp_attr_set("trapezoid_m_samples", str(val))

    def get_trapez_T(self) -> int:
        """ABI2 native value; kernel owns register encoding."""
        return int(self._pp_attr_get("trapezoid_beta_raw"))

    def set_trapez_T(self, val: int) -> None:
        validate_parameters({"trapezoid_beta_raw": val})
        self._pp_attr_set("trapezoid_beta_raw", str(val))

    def get_trapez_E(self) -> int:
        """ABI2 native value; kernel owns register encoding."""
        return int(self._pp_attr_get("trapezoid_start_offset_samples"))

    def set_trapez_E(self, val: int) -> None:
        self._pp_attr_set("trapezoid_start_offset_samples", str(val))

    def get_trapez_FT(self) -> int:
        return int(self._pp_attr_get("trapezoid_flat_top_window"))

    def set_trapez_FT(self, val: int) -> None:
        self._pp_attr_set("trapezoid_flat_top_window", str(val))

    def get_cfd_enable(self) -> bool:
        return bool(int(self._pp_attr_get("cfd_enable")))

    def set_cfd_enable(self, val: bool) -> None:
        self._pp_attr_set("cfd_enable", "1" if val else "0")

    def get_cfd_factor(self) -> float:
        """cfd_factor_raw (0..32767; the register is NOT scaled by
        the driver itself) x cfd_factor_scale (0.000030517578125 = 2^-15,
        read live, same reasoning as get_time_limit()'s scale read).
        """
        raw = int(self._pp_attr_get("cfd_factor_raw"))
        scale = float(self._pp_attr_get("cfd_factor_scale"))
        return raw * scale

    def set_cfd_factor(self, val: float) -> None:
        scale = float(self._pp_attr_get("cfd_factor_scale"))
        self._pp_attr_set("cfd_factor_raw", str(round(val / scale)))

    def get_cfd_delay(self) -> int:
        """ABI2 native value; kernel owns register encoding."""
        return int(self._pp_attr_get("cfd_delay_samples"))

    def set_cfd_delay(self, val: int) -> None:
        self._pp_attr_set("cfd_delay_samples", str(val))

    def get_cfd_time_window_low(self) -> int:
        return int(self._pp_attr_get("cfd_window_low_samples"))

    def set_cfd_time_window_low(self, val: int) -> None:
        self._pp_attr_set("cfd_window_low_samples", str(val))

    def get_cfd_time_window_high(self) -> int:
        return int(self._pp_attr_get("cfd_window_high_samples"))

    def set_cfd_time_window_high(self, val: int) -> None:
        self._pp_attr_set("cfd_window_high_samples", str(val))

    def get_cc_enable(self) -> bool:
        return bool(int(self._pp_attr_get("charge_comparison_enable")))

    def set_cc_enable(self, val: bool) -> None:
        self._pp_attr_set("charge_comparison_enable", "1" if val else "0")

    def get_cc_time(self) -> int:
        """ABI2 native ADC samples (2 ns each); kernel owns register encoding."""
        return int(self._pp_attr_get("charge_comparison_samples"))

    def set_cc_time(self, val: int) -> None:
        self._pp_attr_set("charge_comparison_samples", str(val))

    def get_psd_zc_enable(self) -> bool:
        return bool(int(self._pp_attr_get("psd_zero_crossing_enable")))

    def set_psd_zc_enable(self, val: bool) -> None:
        self._pp_attr_set("psd_zero_crossing_enable", "1" if val else "0")

    def get_psd_zc_mode(self) -> int:
        return int(self._pp_attr_get("psd_zero_crossing_mode"))

    def get_psd_zc_modes(self) -> tuple[int, ...]:
        # Additive ABI2 capability, PetaLinux c546f14 pp_*_mode_available.
        # An older ABI2 driver without this attribute establishes only 0/1.
        if self._pp is None or "psd_zero_crossing_mode_available" not in self._pp.attrs:
            return (0, 1)
        tokens = self._pp_attr_get("psd_zero_crossing_mode_available").split()
        return tuple(int(token) for token in tokens if token in ("0", "1", "2"))

    def _validate_psd_mode(self, value: int) -> None:
        if value not in self.get_psd_zc_modes():
            raise ValueError(f"PSD mode {value} is not advertised by this driver")

    def set_psd_zc_mode(self, val: int) -> None:
        self._pp_attr_set("psd_zero_crossing_mode", str(val))

    def get_psd_zc_time_window_low(self) -> int:
        """ABI2 native ADC samples (2 ns each); kernel owns register encoding."""
        return int(self._pp_attr_get("psd_window_low_samples"))

    def set_psd_zc_time_window_low(self, val: int) -> None:
        self._pp_attr_set("psd_window_low_samples", str(val))

    def get_psd_zc_time_window_high(self) -> int:
        return int(self._pp_attr_get("psd_window_high_samples"))

    def set_psd_zc_time_window_high(self, val: int) -> None:
        self._pp_attr_set("psd_window_high_samples", str(val))

    def get_mem1_sig_select(self) -> int:
        return int(self._pp_attr_get("debug_signal1"))

    def get_debug_signal_selectors(self) -> tuple[int, ...]:
        """Read selector capability from the deployed pulse driver.

        The corrected driver advertises 0..8. Older drivers advertise 0..7;
        their names for codes 6 and 7 were wrong, so callers use this method
        only for capability and apply the canonical hardware labels.
        """
        if self._pp is None:
            self._require_mca_pp()
        assert self._pp is not None
        attr = self._pp.attrs.get("debug_signal1_available")
        if attr is None:
            return tuple(range(8))
        tokens = str(attr.value).split()
        try:
            selectors = tuple(int(tokens[index], 0) for index in range(0, len(tokens), 2))
        except (ValueError, IndexError) as exc:
            raise RuntimeError(
                "invalid debug_signal1_available value: " + str(attr.value)
            ) from exc
        if not selectors:
            raise RuntimeError("debug_signal1_available contains no selector codes")
        return selectors

    def set_mem1_sig_select(self, val: int) -> None:
        self._pp_attr_set("debug_signal1", str(val))

    def get_mem2_sig_select(self) -> int:
        return int(self._pp_attr_get("debug_signal2"))

    def set_mem2_sig_select(self, val: int) -> None:
        self._pp_attr_set("debug_signal2", str(val))

    def get_mem_amount(self) -> int:
        """Fixed at 2 on this core: debug_data concatenates exactly two
        PP_MEM_DEBUG_ENTRIES-deep memories (PP_MEM_DEBUG1/PP_MEM_DEBUG2,
        routed by debug_signal1/debug_signal2) -- no sysfs attribute
        reports this count directly, see _PP_MEM_DEBUG_ENTRIES's comment.
        """
        return 2

    def get_mem_frame_sizes(self) -> np.ndarray:
        return np.array([_PP_MEM_DEBUG_ENTRIES, _PP_MEM_DEBUG_ENTRIES], dtype=np.uint32)

    def _read_pp_bin_attr(self, name: str, size: int) -> bytes:
        """Shared by read_histogram()/read_waveform_banks(). Translates
        confirmed-live transport failures into actionable diagnostics.
        """
        try:
            return self._read_large_pp_attr(self._require_mca_pp(), name, size)
        except OSError as e:
            if e.errno == 2:  # ENOENT
                raise RuntimeError(
                    f"'{name}' is not advertised by vdpp_pulse_processor in "
                    "this IIO context. Reconnect after a target driver reload "
                    "or update so libiio can rediscover its attributes. The "
                    "backend supports both the earlier monolithic "
                    "histogram_data ABI and the newer histogram_data0..3 ABI."
                ) from e
            if e.errno == 27:  # EFBIG
                raise RuntimeError(
                    f"'{name}' is advertised by vdpp_pulse_processor but "
                    "the remote iiod rejects its binary payload with EFBIG. "
                    "Confirmed live with iiod 0.25 for the 65536-byte "
                    "histogram_data attribute even when the client supplies "
                    "up to 131072 bytes of buffer capacity. This is a "
                    "target-side attribute-transport size limit, not a "
                    "short client buffer."
                ) from e
            raise

    def read_waveform_banks(self) -> tuple[np.ndarray, np.ndarray]:
        """Reads the debug_data bin_attribute (8192 bytes: two back-to-back
        2048-entry s16 memories, PP_MEM_DEBUG1 then PP_MEM_DEBUG2) via
        self._mca_pp -- MCAWorker's tick calls this every interval.

        Confirmed live over remote iiod 0.25 for both pulse processors,
        with debug_signal1=input and debug_signal2=trigger. Per
        vdpp-pulse-processor.c's pp_debug_snapshot_read(), there is no
        `st->enabled` guard; stopped and running reads are both supported.
        """
        raw = self._read_pp_bin_attr("debug_data", _PP_DEBUG_SNAPSHOT_BYTES)
        samples = np.frombuffer(raw, dtype="<i2")
        return samples[:_PP_MEM_DEBUG_ENTRIES].copy(), samples[_PP_MEM_DEBUG_ENTRIES:].copy()

    def read_histogram(self) -> np.ndarray:
        with self.mca_configuration_lock:
            return self._read_histogram_serialized()

    def _read_histogram_serialized(self) -> np.ndarray:
        """Reads the 65536-byte, 16384-bin u32 histogram via self._mca_pp.

        The updated board ABI exposes four ordered 16384-byte attributes,
        histogram_data0..3, because remote iiod rejects the old monolithic
        65536-byte value with EFBIG. Confirmed live on both pulse processors:
        concatenating chunks 0 through 3 reconstructs all 16384 bins. The
        monolithic histogram_data path remains as a compatibility fallback
        for earlier drivers.

        The current driver source explicitly permits live reads: it copies
        bins while hardware may be updating them, so a running snapshot is
        informational rather than atomic. STOP clears ownership but A07
        downstream-completion acknowledgement is deferred: no final snapshot
        guarantee is implied, including after enable=0. The shared lock covers
        every part and every application configuration/START/STOP write.
        """
        device = self._require_mca_pp()
        if all(name in device.attrs for name in _PP_HISTOGRAM_CHUNK_NAMES):
            raw = b"".join(
                self._read_pp_bin_attr(name, _PP_HISTOGRAM_CHUNK_BYTES)
                for name in _PP_HISTOGRAM_CHUNK_NAMES
            )
        else:
            raw = self._read_pp_bin_attr("histogram_data", _PP_HISTOGRAM_BYTES)
        return np.frombuffer(raw, dtype="<u4").copy()

    def clear_histogram(self) -> None:
        raise NotImplementedError(_MCA_HISTOGRAM_CLEAR_UNSUPPORTED)

    def get_histogram_size(self) -> int:
        return _PP_MEM_HISTOGRAM_ENTRIES

    def get_sync_enable(self) -> bool:
        return bool(int(self._sync_attr_get("start_enable")))

    def set_sync_enable(self, val: bool) -> None:
        self._sync_attr_set("start_enable", "1" if val else "0")

    def get_sync_sw_trig(self) -> int:
        return int(self._sync_attr_get("software_start_state"))

    def set_sync_sw_trig(self, val: int) -> None:
        self._sync_attr_set("software_start_state", str(val))

    def get_sync_trig_src(self) -> int:
        return int(self._sync_attr_get("start_source"))

    def set_sync_trig_src(self, val: int) -> None:
        """Select software (0) or synchronized hardware input (1).

        Per vdpp-sync-trigger.c's start_source_store(), the driver rejects
        this write with EBUSY while start_enable is set. Do not hide that
        global hardware-state transition by cycling start_enable here.
        """
        self._sync_attr_set("start_source", str(val))

    def get_sync_timestamp(self) -> int:
        """Return the free-running 64-bit ap_clk counter.

        vdpp-sync-trigger.c reads the split hardware counter with a
        high/low/high retry under its mutex, so timestamp_raw is already a
        coherent integer and needs no client-side reconstruction.
        """
        return int(self._sync_attr_get("timestamp_raw"))
