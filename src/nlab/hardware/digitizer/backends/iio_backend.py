"""IIO backend for the vdpp_scope driver (board driver rewrite).

This is a from-scratch rewrite of the scope half, following a full rewrite
of the on-FPGA driver (vdpp-scope.c). Nothing from the previous
ewt-scope-iio-based implementation carries over: different device name
("vdpp_scope" instead of "ewt-scope{N}"), different attribute names, a
plain-integer trigger_mode register instead of a string attribute, a
text-based (not binary) viewer attribute, and a standard IIO dmaengine
buffer instead of the old custom cyclic frame-ring. Written from the driver
source (vdpp-scope.c) plus a reference client stub (scope_backend_iio.py)
supplied alongside it, and confirmed live against the rewritten board.

MCABackend is implemented against vdpp-pulse-processor.c (signal
parameters, acquisition control, statistics, filters, pulse memory,
histogram) and vdpp-input-filter.c (FIR/IIR low-pass filter, temperature
compensation). The channel-independent trigger-sync interface is implemented
against vdpp-sync-trigger.c. The extension methods below also implement the
fixed-frame list-mode DMA path owned by vdpp-lm-frame.c -- see
mca-architecture.md for how the three per-channel cores relate, and
notebooks/mca_walkthrough.ipynb for a live-tested, cell-by-cell
verification of the register/configuration half. A few MCABackend methods stay stubbed
because no matching register exists in either driver: see
_MCA_EVENTS_LOST_UNSUPPORTED and _MCA_HISTOGRAM_CLEAR_UNSUPPORTED below for
the specifics. get_edge_det_coeff/
set_edge_det_coeff() are a software-only shadow value for the same reason
(no register), kept non-raising only so MCAController's unconditional
default-hydration pass doesn't need a backend-specific special case.

Every register-based get/set pair (signal parameters, all five filter
groups, temperature compensation) is confirmed live and correct, INCLUDING
one real bug the live run caught and fixed: an earlier revision of this
file reimplemented the driver's PP_FMT_X2/X8/PLUS1_X8/TRAPEZOID_R
conversions (pp_field_to_user()/pp_field_from_user()) client-side, on the
wrong assumption that the sysfs attribute held the raw register value.
It doesn't -- pp_field_show()/pp_field_store() already do that conversion
*inside the kernel*, so the sysfs value already is the physical/user
value. The old code was applying the same conversion a second time,
silently writing a roughly-halved-or-eighthed value to hardware while
still round-tripping correctly through get_*() (a symmetric bug, invisible
to a get-after-set check) -- see the NOTE at the top of the MCABackend
section for the full account and get_trapez_R()'s docstring for the
clearest single example.

read_waveform_banks() and read_histogram() are confirmed live over the IIO
network transport after the driver began registering the binary attributes
explicitly. The newer ABI splits the histogram into histogram_data0..3 to
fit iiod's attribute-transport limit; the client retains compatibility with
the earlier monolithic histogram_data ABI. The current remote transport also
needs roughly twice the raw-payload capacity and returns a counted trailing
NUL; _read_large_pp_attr() handles that without truncating embedded NULs.
"""

from __future__ import annotations

import ctypes
import errno
import logging
import threading
import time
from collections.abc import Callable

import iio
import numpy as np

from .base import DigitizerBackend

log = logging.getLogger(__name__)

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
_SCOPE_KERNEL_BUFFER_COUNT = 1
_LM_RECORD_LAYOUT = "opaque[16]"
_LM_EVENT_DTYPE = np.dtype([
    ("flags", "<u2"),
    ("cfd_q2", "<u2"),
    ("charge_energy", "<u2"),
    ("trapezoid_energy", "<u2"),
    ("timestamp", "<u8"),
])

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

    This matches hw_description/scope_backend_iio.py and makes index-based
    selection deterministic. It does *not* turn probe order into physical
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
    ) -> None:
        self._ch = channel
        self._uri = uri
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
        # like vdpp_scope. dac_channel defaults to this scope's channel
        # index, confirmed live to be the right 1:1 mapping (voltage0 ->
        # scope channel 0, voltage1 -> scope channel 1). Pass dac_name=None
        # to disable DAC support entirely (e.g. against a board without
        # this device yet).
        self._dac_dev = self._ctx.find_device(dac_name) if dac_name else None
        self._dac_ch = (
            self._dac_dev.find_channel(
                f"voltage{dac_channel if dac_channel is not None else channel}", True,
            )
            if self._dac_dev is not None else None
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
        # point and one refill() now delivers exactly one frame. No more
        # per-frame pending queue needed on the client side.
        self._dma_buf: iio.Buffer | None = None
        self._dma_buf_frame_samples: int | None = None
        # Set by IIOScopeDmaWorker.stop() through the streamer. Buffer.cancel()
        # is the libiio-supported way to interrupt a refill blocked in another
        # thread; this event lets that expected cancellation be distinguished
        # from a genuine EBADF/EIO capture failure.
        self._dma_stop_requested = threading.Event()

        # Set by _refill_dma_buffer() on a genuine EIO (not the expected -9
        # a close-time Buffer.cancel() produces) -- see its docstring and
        # acknowledge_dma_recovery()/dma_fault_is_latched().
        self._dma_fault_latched = False

        # A single IIO network context is not safe for concurrent use from
        # multiple threads. Confirmed live, twice, independently: this is
        # a *different* problem from the "never write dma_enable directly"
        # bug the driver author's architecture doc identifies -- that bug
        # is real and fixed (see set_dma_enable()'s docstring), but fixing
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
        self._mca_dma_buf: iio.Buffer | None = None
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

    def close(self) -> None:
        log.info("IIO backend: closing ch%d", self._ch)
        self._close_mca_dma_buffer()
        self._close_dma_buffer()

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

    def _drain_for_close(self, buf: iio.Buffer) -> int:
        """Keep refilling *buf* for a short window before cancelling it,
        instead of cancelling immediately.

        Ported from the reference stub's (scope_backend_iio.py)
        _drain_for_close() -- our own close path was missing this step
        entirely. ADI file-I/O owns one DMA block, and per the stub's own
        comment, if a frame the core already started sending is not
        consumed before the DMA receiver disappears, the core can be left
        blocked in its AXI send state -- not just for DMA, but for the
        shared state machine that viewer-only frames also depend on to
        return to idle.

        A background thread keeps calling refill() (discarding the
        result -- this is drain, not read) until cancelled, while this
        thread sleeps DMA_CLOSE_DRAIN_SECONDS to give one already-in-
        flight frame time to land. cancel() then unblocks the worker's
        final blocking refill() the same way it always has. Returns the
        number of frames drained (0 or 1 in the common case).
        """
        stop = threading.Event()
        entered = threading.Event()
        errors: list[Exception] = []
        drained: list[None] = []

        def refill_until_cancelled() -> None:
            entered.set()
            while not stop.is_set():
                try:
                    buf.refill()
                except Exception as exc:
                    # cancel() is expected to interrupt the final
                    # blocking refill -- only an earlier failure is a
                    # real error.
                    if not stop.is_set():
                        errors.append(exc)
                    return
                drained.append(None)

        worker = threading.Thread(
            target=refill_until_cancelled,
            name=f"vdpp-scope-ch{self._ch}-dma-drain",
            daemon=True,
        )
        worker.start()
        try:
            if not entered.wait(_DMA_CLOSE_JOIN_SECONDS):
                log.warning(
                    "IIO backend ch%d: DMA close-drain thread did not "
                    "start", self._ch,
                )
                return 0
            time.sleep(_DMA_CLOSE_DRAIN_SECONDS)
        finally:
            stop.set()
            buf.cancel()
            worker.join(_DMA_CLOSE_JOIN_SECONDS)

        if worker.is_alive():
            log.warning(
                "IIO backend ch%d: DMA close-drain thread did not stop "
                "after cancellation", self._ch,
            )
        if errors:
            log.warning(
                "IIO backend ch%d: DMA close-drain refill failed: %s",
                self._ch, errors[0],
            )
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
        try:
            if self._dma_get_enable():
                self._dma_set_enable(False)
        except OSError:
            pass

        if drain:
            try:
                self._drain_for_close(buf)
            except Exception:
                log.warning(
                    "IIO backend ch%d: DMA close-drain failed",
                    self._ch, exc_info=True,
                )
        else:
            # A requested cancellation or failed normal refill has no
            # trustworthy tail to drain. Cancel the existing descriptor and
            # proceed directly to buffer destruction.
            buf.cancel()

        self._dma_buf = None
        self._dma_buf_frame_samples = None
        # Do not depend on Buffer.__del__ timing here. In pylibiio 0.25 a
        # cancelled refill can leave Python/threading references alive long
        # enough for the complete gate wait to expire. Destroy the native
        # buffer explicitly after every reader has stopped, then null the
        # wrapper pointer so its eventual __del__ is idempotent.
        native_buffer = buf._buffer
        buf._buffer = None
        if native_buffer is not None:
            iio._buffer_destroy(native_buffer)
        del buf

        try:
            self._wait_for_dma_release()
        except RuntimeError:
            self._dma_fault_latched = True
            log.error(
                "IIO backend ch%d: DMA buffer close did not release the "
                "hardware gate in time -- may indicate a genuine channel "
                "fault (check dmesg for xilinx-vdma errors), not something "
                "a client-side retry can fix", self._ch, exc_info=True,
            )
            raise
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
        self._wait_for_dma_release()
        self._dma_stop_requested.clear()

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
    # either context; see set_dma_enable()'s docstring.

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
            raise RuntimeError(
                f"no {_PULSE_PROCESSOR_DEVICE_NAME} device bound -- pulse-"
                "processor core not present on this channel's firmware"
            )
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

    def _mca_dma_pp_attr_get(self, name: str) -> str:
        with self._mca_dma_control_lock:
            return str(self._require_mca_dma_pp().attrs[name].value)

    def _mca_dma_pp_attr_set(self, name: str, value: str) -> None:
        with self._mca_dma_control_lock:
            self._require_mca_dma_pp().attrs[name].value = value

    def _validate_lm_geometry(self) -> None:
        """Validate the fixed v121 ABI before creating a DMA buffer.

        The constants are not client preferences: lm_probe() rejects a
        different IP version/frame size and lm_buffer_preenable() rejects
        every scan size or buffer length except one opaque u8[16] element x
        1024 records. Reading the attributes here makes the superseded
        five-channel driver fail precisely before iio.Buffer() returns its
        otherwise opaque EINVAL.
        """
        version = int(self._lm_attr_get("ip_version"))
        records = int(self._lm_attr_get("frame_records"))
        frame_bytes = int(self._lm_attr_get("frame_bytes"))
        if (version, records, frame_bytes) != (
            _LM_IP_VERSION, _LM_FRAME_RECORDS, _LM_FRAME_BYTES,
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
        thread.start()
        if not entered.wait(_DMA_FIRST_REFILL_ENTER_SECONDS):
            self._cancel_mca_dma_refill(buf, "reader thread did not enter refill")
            thread.join(_LM_CLOSE_JOIN_SECONDS)
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
        except BaseException:
            try:
                self._mca_dma_pp_attr_set("enable", "0")
            except OSError:
                log.warning(
                    "IIO backend ch%d: failed to stop MCA after start error",
                    self._ch,
                    exc_info=True,
                )
            self._cancel_mca_dma_refill(buf, "MCA start failed")
            thread.join(_LM_CLOSE_JOIN_SECONDS)
            raise

        thread.join()
        if errors:
            error_errno, error_text = errors[0]
            if error_errno is not None:
                raise OSError(error_errno, error_text) from None
            raise RuntimeError(error_text)
        return result[0]

    def _create_mca_dma_buffer(self) -> None:
        """Arm the fixed 1024-record list-mode buffer while stopped."""
        self._close_mca_dma_buffer()
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
            self._close_mca_dma_buffer(drain=False)
            if expected_stop:
                return np.empty(0, dtype=_LM_EVENT_DTYPE)
            raise

    def read_mca_dma_frame(self) -> np.ndarray:
        """Return one parsed 1024-record list-mode DMA block.

        This is an IIO-backend extension, not part of MCABackend. New stream
        sessions should call start_mca_dma_capture() for the first block so
        readiness can be reported at the correct lifecycle boundary. The
        lazy-start fallback is retained for direct callers. The returned
        structured dtype follows vdpp-lm-frame.c exactly and deliberately
        differs from the legacy gRPC/ZMQ event dtype.
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
            self._close_mca_dma_buffer(drain=False)
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
        errors: list[str] = []
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
                    if not stop.is_set():
                        # Do not retain a traceback that owns this closure and
                        # its Buffer reference across native buffer teardown.
                        errors.append(f"{type(exc).__name__}: {exc}")
                    return

        worker = threading.Thread(
            target=refill_until_inactive,
            name=f"vdpp-lm-frame-ch{self._ch}-drain",
            daemon=True,
        )
        worker.start()
        try:
            if not entered.wait(_LM_CLOSE_JOIN_SECONDS):
                log.warning(
                    "IIO backend ch%d: MCA list-mode drain thread did not start",
                    self._ch,
                )
                return 0
            while time.monotonic() - last_activity[0] < _LM_CLOSE_INACTIVITY_SECONDS:
                time.sleep(_LM_CLOSE_POLL_SECONDS)
        finally:
            stop.set()
            self._cancel_mca_dma_refill(buf, "tail-drain inactivity window elapsed")
            worker.join(_LM_CLOSE_JOIN_SECONDS)

        if worker.is_alive():
            log.warning(
                "IIO backend ch%d: MCA list-mode drain did not stop after cancellation",
                self._ch,
            )
        if errors:
            log.warning(
                "IIO backend ch%d: MCA list-mode drain refill failed: %s",
                self._ch, errors[0],
            )
        return drained

    def _wait_for_mca_dma_release(self) -> None:
        deadline = time.monotonic() + _LM_CLOSE_RELEASE_SECONDS
        while True:
            frame_active = bool(int(self._lm_attr_get("buffer_active")))
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
        """Stop, drain, then destroy the list-mode buffer in driver order."""
        with self._mca_dma_lifecycle_lock:
            cancel_timer = self._mca_dma_cancel_timer
            self._mca_dma_cancel_timer = None
            if cancel_timer is not None:
                cancel_timer.cancel()

        buf = self._mca_dma_buf
        if buf is None:
            return 0

        try:
            if bool(int(self._mca_dma_pp_attr_get("enable"))):
                self._mca_dma_pp_attr_set("enable", "0")
        except OSError:
            log.warning(
                "IIO backend ch%d: failed to stop pulse processor before "
                "MCA list-mode drain",
                self._ch, exc_info=True,
            )

        if drain:
            drained = self._drain_mca_for_close(buf, on_frame)
        else:
            # Failed arm/start or refill has no trustworthy tail to drain.
            # Interrupt the outstanding descriptor before native teardown.
            self._cancel_mca_dma_refill(buf, "capture cleanup without tail drain")
            drained = 0

        self._mca_dma_buf = None
        # Match the hardened scope path: Buffer.__del__ timing is not a
        # lifecycle boundary after cancellation. Destroy explicitly once all
        # refill threads have stopped, then null the wrapper for idempotency.
        native_buffer = buf._buffer
        buf._buffer = None
        if native_buffer is not None:
            iio._buffer_destroy(native_buffer)
        del buf
        try:
            self._wait_for_mca_dma_release()
        except RuntimeError:
            log.error(
                "IIO backend ch%d: MCA list-mode buffer did not release in time",
                self._ch, exc_info=True,
            )
        finally:
            # mca_listmode_capture.py: the scan element remains selected for
            # the complete drain and is released only after buffer teardown.
            for channel in self._require_lm_frame().channels:
                channel.enabled = False
        return drained

    def close_mca_dma_capture(
        self,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> int:
        """Public stop/drain/close wrapper used by IIOMcaDmaStreamer."""
        return self._close_mca_dma_buffer(on_frame)

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
            int(self._lm_attr_get("dma_fault")),
            int(self._lm_attr_get("dma_error_count")),
            int(self._lm_attr_get("completed_frames")),
            int(self._lm_attr_get("list_deadtime_raw")),
        )

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

    def get_dma_enable(self) -> bool:
        """Read-only diagnostic: the driver-owned DMA stream-gate state.

        Per the driver author's architecture note (scope-architecture.md):
        dma_enable belongs entirely to the buffer arm/close lifecycle
        (_create_dma_buffer()/_close_dma_buffer()) -- "a GUI control
        labelled DMA represents buffer arm/close, not a direct write to
        that register." There is deliberately no set_dma_enable() write
        path left in this backend; see set_dma_enable()'s docstring.
        """
        return bool(int(self._attr_get("dma_enable")))

    def set_dma_enable(self, val: bool) -> None:
        """No-op for this backend -- see get_dma_enable()'s docstring.

        Part of the shared ScopeBackend/MCABackend interface (also used
        by the gRPC backend, where a direct write is the correct and only
        way to control this bit) and still wired to a GUI checkbox in
        ScopeController for backend-agnostic code -- kept present so both
        remain usable unmodified. For vdpp_scope specifically, writing
        this register directly while no buffer is open sets a state the
        driver never expects a client to set on its own (dma_enable=1
        with no IIO descriptor behind it), and writing it while a buffer
        *is* open races the driver's own close/drain and is explicitly
        rejected with -EBUSY. Confirmed live: retrying that -EBUSY (a
        prior revision did, for up to 2.5 s) never succeeds, because the
        write itself is illegitimate, not delayed -- only
        _close_dma_buffer() dropping the buffer reference is allowed to
        clear this bit.
        """
        log.debug(
            "IIO backend ch%d: set_dma_enable(%s) ignored -- dma_enable is "
            "driver-owned, controlled only via buffer arm/close",
            self._ch, val,
        )

    def get_trigger_level(self) -> int:
        return int(self._attr_get("trigger_level_raw"))

    def set_trigger_level(self, val: int) -> None:
        """Plain, unguarded write. A prior driver revision briefly
        rejected this with -EBUSY while armed (hit live via the GUI,
        worked around at the time with a disarm/write/re-arm wrapper);
        the revision documented in vdpp-scope.c's trigger_level_raw_store()
        removed that guard entirely -- only frame_samples is still
        restricted (see its own docstring for why).
        """
        self._attr_set("trigger_level_raw", str(val))

    def get_edge(self) -> int:
        return int(self._attr_get("trigger_mode"))

    def set_edge(self, val: int) -> None:
        """See set_trigger_level()'s docstring -- same history, same
        current lack of a guard.
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
        """See set_trigger_level()'s docstring -- same history, same
        current lack of a guard.
        """
        self._attr_set("frame_period_cycles", str(val))

    def get_pretrigger_samples(self) -> int:
        return int(self._attr_get("pretrigger_samples"))

    def set_pretrigger_samples(self, val: int) -> None:
        """See set_trigger_level()'s docstring -- same history, same
        current lack of a guard.
        """
        self._attr_set("pretrigger_samples", str(val))

    def get_frame_samples(self) -> int:
        return int(self._attr_get("frame_samples"))

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
        """Read one full-resolution frame from the DMA path, creating the
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

        Returns (timestamp_raw, samples) where samples excludes the first
        4 elements: per vdpp-scope.c's channel comment and the reference
        stub's frame_timestamp() helper, the core overwrites the first 4
        int16 slots of every frame with a 64-bit trigger timestamp
        (little-endian), passed through otherwise unchanged -- so samples
        here is frame_samples - 4 real ADC values, not the full
        frame_samples. Timestamp is decoded directly from the raw bytes
        (not via the stub's own uint64-cast-of-int16 approach, which
        sign-extends on any negative-looking sample word and silently
        corrupts the packing).
        """
        raw = self._refill_dma_buffer()
        timestamp = int.from_bytes(raw[:8], "little")
        samples = np.frombuffer(raw[8:], dtype="<i2")
        return timestamp, samples

    def _refill_dma_buffer(self) -> bytes:
        """Shared by read_dma_frame() and capture_to_file(): ensure the
        persistent capture buffer exists for the current frame_samples,
        refill it, and return exactly the bytes iio_buffer_refill()
        reports for the completed frame (raw, timestamp header included,
        unparsed).

        Buffer sizing: confirmed live via dmesg ("buffer length is 1024,
        must be 512 for a 512 sample frame") that scope_buffer_preenable()
        now requires the buffer's sample count to be exactly
        frame_samples, not 2 * frame_samples like an earlier driver
        revision needed -- one refill() delivers exactly one frame now,
        no client-side splitting/queuing required. The buffer is recreated
        automatically if frame_samples changes between calls.

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

        The refill() itself is wrapped in the same retry-on-transient-error
        helper (this time for EINVAL, errno 22): confirmed live that the
        very first refill() right after set_enable(True) can fail this way
        if issued too soon -- the older driver revision never had this gap
        since ENABLE was set synchronously inside postenable(), before
        control ever returned to userspace, so no refill() could race it.

        A persistent EINVAL after the bounded transient retries ends the
        session. An earlier implementation tried to recreate the buffer and
        continue in the same output file, but that cannot prove continuity
        after a framing fault and retained the old Buffer through the caught
        exception's traceback while waiting for its driver gates to clear.

        A genuine EIO (errno 5) is handled differently from both cases
        above: per user-api.md, "An EIO returned by a normal capture
        refill() latches a fault in the Python backend. Cleanup is
        attempted exactly once, automatic rearm is blocked, and a second
        close-drain error does not replace the original EIO." This is
        distinct from the expected errno -9 (EBADF) a close-time
        Buffer.cancel() produces when it interrupts _drain_for_close()'s
        blocked refill -- that one is normal cancellation noise, not a
        session fault, and _drain_for_close() already only treats a
        refill failure as an error when it happens before stop.is_set().

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

        n = self._dma_get_frame_samples()
        first = self._dma_buf is None or self._dma_buf_frame_samples != n
        if first:
            self._create_dma_buffer(n)

        buf = self._dma_buf
        assert buf is not None
        failure_errno: int | None = None
        failure_text = ""
        try:
            nbytes = self._refill_once(buf, first)
        except OSError as exc:
            # Store scalar diagnostics only. Keeping ``exc`` outside this
            # block would retain its traceback, including _refill_once()'s
            # local Buffer reference, and prevent native buffer destruction.
            failure_errno = exc.errno
            failure_text = str(exc)

        if failure_errno is not None:
            del buf
            if self._dma_stop_requested.is_set():
                self._close_dma_buffer(drain=False)
                raise InterruptedError(
                    errno.ECANCELED, "scope DMA refill cancelled by Stop"
                ) from None

            if failure_errno in (errno.EIO, errno.EINVAL):
                self._dma_fault_latched = True
                log.error(
                    "IIO backend ch%d: DMA session fault (errno %d) during "
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
                raise OSError(failure_errno, failure_text) from None

            self._close_dma_buffer(drain=False)
            raise OSError(failure_errno, failure_text) from None

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

        start = iio._buffer_start(buf._buffer)
        return ctypes.string_at(start, nbytes)

    def _refill_once(self, buf: iio.Buffer, first: bool) -> int:
        """Refill *buf* once. *first* selects _start_reader_then_enable()
        (buffer was just (re)armed, enable is not known to be 1 yet) vs a
        plain retry-tolerant refill (steady-state reuse of an already-
        running buffer) -- see _refill_dma_buffer()'s and
        _start_reader_then_enable()'s docstrings.
        """
        if first:
            return self._start_reader_then_enable(buf)
        return self._retry_errno(  # type: ignore[no-any-return]
            lambda: iio._buffer_refill(buf._buffer), (22,)
        )

    def _start_reader_then_enable(self, buf: iio.Buffer) -> int:
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
        errors: list[BaseException] = []

        def reader() -> None:
            entered.set()
            try:
                result.append(iio._buffer_refill(buf._buffer))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(
            target=reader,
            name=f"vdpp-scope-ch{self._ch}-dma-first-refill",
            daemon=True,
        )
        thread.start()
        if not entered.wait(_DMA_FIRST_REFILL_ENTER_SECONDS):
            raise RuntimeError(
                "DMA reader thread did not start within "
                f"{_DMA_FIRST_REFILL_ENTER_SECONDS:.2f} seconds -- refusing "
                "to write enable=1 with no confirmed listener"
            )

        if self._dma_stop_requested.is_set():
            buf.cancel()
            thread.join(_DMA_CLOSE_JOIN_SECONDS)
            raise InterruptedError(errno.ECANCELED, "scope DMA stop requested")

        if not self._dma_get_enable():
            self._dma_set_enable(True)

        thread.join()
        if errors:
            raise errors[0]
        return result[0]

    def _create_dma_buffer(self, n: int) -> None:
        """Close any existing buffer and open a fresh one sized for n
        samples -- shared by _refill_dma_buffer()'s normal path and its
        EINVAL/EIO recovery paths. Does not touch enable itself; see
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

        # user-api.md and scope-architecture.md require exactly one mmap
        # block for this Xilinx 5.15 Direct Register DMA path. Libiio 0.25's
        # default of four can report pending-but-never-executed descriptors as
        # completed, producing stale, duplicate, or zero-filled full-size
        # frames without a DMA error. This call works through iiod and must be
        # made after the old buffer is closed and before the new one is opened.
        self._dma_scope.set_kernel_buffers_count(_SCOPE_KERNEL_BUFFER_COUNT)
        self._dma_buf = iio.Buffer(self._dma_scope, n, False)
        self._dma_buf_frame_samples = n
        log.debug(
            "IIO backend ch%d: DMA buffer armed, frame_samples=%d",
            self._ch, n,
        )

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
    # NOTE on PP_FMT_X2/X8/PLUS1_X8/TRAPEZOID_R fields (pretrigger_samples,
    # frame_samples, cfd_delay, both CFD/PSD time-window pairs,
    # charge_comparison_time, crrc2_fdelay, trapezoid_r/m/time): an earlier
    # revision of this section reimplemented vdpp-pulse-processor.c's
    # pp_field_to_user()/pp_field_from_user() client-side for these. That
    # was wrong and has been removed -- pp_field_show()/pp_field_store()
    # already apply that conversion *inside the kernel*, so the sysfs
    # value is already the physical/user value, confirmed live (writing
    # "24" to pretrigger_samples reads back "24", not "12"). Applying the
    # same conversion again client-side silently wrote a wrong (roughly
    # halved or eighthed) value to hardware on every affected field while
    # still round-tripping correctly through get_*() -- the bug was
    # symmetric and invisible to a get-after-set check, only surfacing
    # when the double-converted value happened to fall outside a field's
    # valid range (crrc2_fdelay's minimum, confirmed live as an -ERANGE).
    # Every one of these fields is a plain passthrough now, same as RAW
    # fields -- see get_trapez_R()'s docstring for the fullest example.
    # ------------------------------------------------------------------

    def mca_hardware_present(self) -> bool:
        """Extension method (not part of MCABackend), consumed by
        Digitizer.mca_available(): whether this channel's firmware has the
        pulse-processor/input-filter devices at all. False means every
        MCABackend method that goes through _pp_attr_get/set or
        _if_attr_get/set raises RuntimeError -- callers building GUI docks
        (MainWindowController) should check this before constructing
        anything that unconditionally writes MCA defaults on init (e.g.
        MCAController), same spirit as dma_fault_is_latched()/
        acknowledge_dma_recovery() for the scope DMA fault path.
        """
        return self._pp is not None and self._input_filter is not None

    def mca_dma_hardware_present(self) -> bool:
        """Whether the dedicated context found both paired DMA devices."""
        return self._mca_dma_pp is not None and self._lm_frame is not None

    def get_lm_ip_version(self) -> int:
        """vdpp_lm_frame diagnostic extension; expected value is 121."""
        return int(self._lm_attr_get("ip_version"))

    def get_lm_frame_records(self) -> int:
        return int(self._lm_attr_get("frame_records"))

    def get_lm_frame_bytes(self) -> int:
        return int(self._lm_attr_get("frame_bytes"))

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
        """PP_FMT_X2 -- but pp_field_show()/pp_field_store() already apply
        the raw<->user (x2) conversion inside the kernel; the sysfs value
        IS the sample count, confirmed live (writing "24" reads back "24",
        not "12"). No client-side conversion here -- see the NOTE at the
        top of the MCABackend section for the double-conversion bug this
        used to have.
        """
        return int(self._pp_attr_get("pretrigger_samples"))

    def set_dpp_pretrigger_samples(self, val: int) -> None:
        self._pp_attr_set("pretrigger_samples", str(val))

    def get_dpp_frame_samples(self) -> int:
        """See get_dpp_pretrigger_samples()'s docstring -- same PP_FMT_X2
        field, same already-converted-by-the-kernel sysfs value.
        """
        return int(self._pp_attr_get("frame_samples"))

    def set_dpp_frame_samples(self, val: int) -> None:
        self._pp_attr_set("frame_samples", str(val))

    def get_global_enable(self) -> bool:
        return bool(int(self._pp_attr_get("enable")))

    def set_global_enable(self, val: bool) -> None:
        self._pp_attr_set("enable", "1" if val else "0")

    def get_dpp_dma_enable(self) -> bool:
        """Read-only diagnostic, not the real gate.

        Per mca-architecture.md: "The list_dma_enable register physically
        lives in this core but is owned by vdpp-lm-frame... Keeping that
        gate out of userspace prevents the AXI stream from being opened
        before a DMA descriptor exists." vdpp-pulse-processor.c exposes no
        writable list_dma_enable attribute at all -- only the read-only
        list_buffer_active, set internally via the exported
        vdpp_pulse_set_list_buffer() when a vdpp-lm-frame buffer is
        armed/closed. The IIO list-mode extension methods in this backend
        now create and destroy that buffer; callers still never write the
        gate directly.
        """
        return bool(int(self._pp_attr_get("list_buffer_active")))

    def set_dpp_dma_enable(self, val: bool) -> None:
        """No-op -- see get_dpp_dma_enable()'s docstring. list_dma_enable
        is not a writable sysfs attribute on this driver at all (unlike
        vdpp-scope.c's dma_enable, which is at least readable+driver-owned);
        it is entirely internal to the pulse-processor/lm-frame device
        link. Present so MCAController's `cbDmaEnable.toggled` signal (a
        shared, backend-agnostic connection) has something safe to call.
        """
        log.debug(
            "IIO backend ch%d: set_dpp_dma_enable(%s) ignored -- "
            "list_dma_enable has no writable sysfs attribute on this "
            "driver, see get_dpp_dma_enable()'s docstring",
            self._ch, val,
        )

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
        """Deciseconds (0.1 s ticks) -- matches mca.py's MCAStatistics.
        get_elapsed_time() docstring ("Return elapsed time in 0.1s ticks.
        Divide by 10 for seconds."), which MCAController's readback
        display already relies on (`elapsed_time / 10`). enable_time_raw's
        native tick period is measurement_time_scale (0.134217728 s, the
        same clock as the measurement_time_raw/SET_ENABLE_TIME time-limit
        register it counts against) -- converted to real seconds and then
        re-quantized to deciseconds here so this backend numerically
        matches that pre-existing contract despite the hardware's tick
        period not itself being 0.1 s.
        """
        raw = int(self._mca_pp_attr_get("enable_time_raw"))
        scale = float(self._mca_pp_attr_get("measurement_time_scale"))
        return round(raw * scale * 10)

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

    def set_temp_coeff(self, val: float) -> None:
        scale = float(self._if_attr_get("temperature_coefficient_scale"))
        self._if_attr_set("temperature_coefficient_raw", str(round(val / scale)))

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
        device.attrs["temperature_coefficient_raw"].value = str(
            round(coefficient / scale)
        )
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

    def get_crrc2_Cdelay(self) -> int:
        return int(self._pp_attr_get("crrc2_cdelay"))

    def set_crrc2_Cdelay(self, val: int) -> None:
        self._pp_attr_set("crrc2_cdelay", str(val))

    def get_crrc2_Fdelay(self) -> int:
        """PP_FMT_PLUS1_X8 -- pp_field_show()/pp_field_store() apply the
        (raw+1)*8 conversion inside the kernel already; the sysfs value is
        the physical delay, confirmed live. No client-side conversion.
        """
        return int(self._pp_attr_get("crrc2_fdelay"))

    def set_crrc2_Fdelay(self, val: int) -> None:
        self._pp_attr_set("crrc2_fdelay", str(val))

    def get_crrc2_pzc_coeff(self) -> int:
        return int(self._pp_attr_get("crrc2_pzc_raw"))

    def set_crrc2_pzc_coeff(self, val: int) -> None:
        self._pp_attr_set("crrc2_pzc_raw", str(val))

    def get_trapez_enable(self) -> bool:
        return bool(int(self._pp_attr_get("trapezoid_enable")))

    def set_trapez_enable(self, val: bool) -> None:
        self._pp_attr_set("trapezoid_enable", "1" if val else "0")

    def get_trapez_R(self) -> int:
        """PP_FMT_TRAPEZOID_R -- (raw + 1) * 8, same formula as PLUS1_X8
        (pp_field_to_user()'s switch shares the case) -- but that
        conversion happens inside pp_field_show()/pp_field_store()
        already; the sysfs value is the physical value, confirmed live
        (writing "16" reads back "16", not a re-converted "136"). No
        client-side conversion here.

        Per mca-architecture.md, writing trapezoid_r also writes the
        dependent trapezoid_1r register kernel-side (Rdelay/1Rdelay) --
        there is no separate userspace attribute for that, nothing extra
        needed here. The kernel's real minimum is 16 (8 would make Rdelay
        zero, per the same doc). ``mca.py`` and the GUI use the same
        16..4088 step-8 range, so invalid values are rejected before I/O.
        """
        return int(self._pp_attr_get("trapezoid_r"))

    def set_trapez_R(self, val: int) -> None:
        self._pp_attr_set("trapezoid_r", str(val))

    def get_trapez_M(self) -> int:
        """PP_FMT_X8 -- already converted by the kernel, see
        get_trapez_R()'s docstring. No client-side conversion.
        """
        return int(self._pp_attr_get("trapezoid_m"))

    def set_trapez_M(self, val: int) -> None:
        self._pp_attr_set("trapezoid_m", str(val))

    def get_trapez_T(self) -> int:
        """Maps to trapezoid_beta_raw, not a "trapezoid_time"-named
        attribute -- despite the "_T" suffix. Confirmed by numeric range:
        mca.py's MCAParam.TRAPEZ_T spec is a full-range uint32 (min 0, max
        4294967295, step 1), which only trapezoid_beta_raw's PP_FMT_RAW
        uint32 field matches; the driver's actual "trapezoid_time"
        attribute (PP_F_TRAPEZOID_TIME) is a 0..16376 step-8 field and
        matches MCAParam.TRAPEZ_E's spec instead (see get_trapez_E()).
        This is consistent with mca-architecture.md's own legacy-mapping
        note: "the floating-point legacy mapping for the pole-zero time is
        left in userspace: write trapezoid_beta_raw = round(exp(-8 / T) *
        2^31)" -- i.e. the legacy "T" register a caller writes has always
        been the precomputed beta value itself, raw and unscaled.
        """
        return int(self._pp_attr_get("trapezoid_beta_raw"))

    def set_trapez_T(self, val: int) -> None:
        self._pp_attr_set("trapezoid_beta_raw", str(val))

    def get_trapez_E(self) -> int:
        """Maps to the driver's "trapezoid_time" attribute (PP_FMT_X8) --
        see get_trapez_T()'s docstring for why the naming and the legacy
        ABC method names diverge here; this pairing is what the numeric
        ranges (0..16376 step 8) actually confirm. The kernel already
        applies the x8 conversion (see get_trapez_R()'s docstring), so
        this is a plain passthrough like every other X2/X8/PLUS1_X8 field.
        """
        return int(self._pp_attr_get("trapezoid_time"))

    def set_trapez_E(self, val: int) -> None:
        self._pp_attr_set("trapezoid_time", str(val))

    def get_trapez_FT(self) -> int:
        return int(self._pp_attr_get("trapezoid_flat_top_window"))

    def set_trapez_FT(self, val: int) -> None:
        self._pp_attr_set("trapezoid_flat_top_window", str(val))

    def get_cfd_enable(self) -> bool:
        return bool(int(self._pp_attr_get("cfd_enable")))

    def set_cfd_enable(self, val: bool) -> None:
        self._pp_attr_set("cfd_enable", "1" if val else "0")

    def get_cfd_factor(self) -> float:
        """cfd_factor_raw (u16, PP_FMT_RAW -- the register is NOT scaled by
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
        """PP_FMT_X2 -- already converted by the kernel, see
        get_trapez_R()'s docstring. No client-side conversion.
        """
        return int(self._pp_attr_get("cfd_delay"))

    def set_cfd_delay(self, val: int) -> None:
        self._pp_attr_set("cfd_delay", str(val))

    def get_cfd_time_window_low(self) -> int:
        return int(self._pp_attr_get("cfd_time_walk_low"))

    def set_cfd_time_window_low(self, val: int) -> None:
        self._pp_attr_set("cfd_time_walk_low", str(val))

    def get_cfd_time_window_high(self) -> int:
        return int(self._pp_attr_get("cfd_time_walk_high"))

    def set_cfd_time_window_high(self, val: int) -> None:
        self._pp_attr_set("cfd_time_walk_high", str(val))

    def get_cc_enable(self) -> bool:
        return bool(int(self._pp_attr_get("charge_comparison_enable")))

    def set_cc_enable(self, val: bool) -> None:
        self._pp_attr_set("charge_comparison_enable", "1" if val else "0")

    def get_cc_time(self) -> int:
        """PP_FMT_X2 -- already converted by the kernel, see
        get_trapez_R()'s docstring. No client-side conversion.
        """
        return int(self._pp_attr_get("charge_comparison_time"))

    def set_cc_time(self, val: int) -> None:
        self._pp_attr_set("charge_comparison_time", str(val))

    def get_psd_zc_enable(self) -> bool:
        return bool(int(self._pp_attr_get("psd_zero_crossing_enable")))

    def set_psd_zc_enable(self, val: bool) -> None:
        self._pp_attr_set("psd_zero_crossing_enable", "1" if val else "0")

    def get_psd_zc_mode(self) -> int:
        return int(self._pp_attr_get("psd_zero_crossing_mode"))

    def set_psd_zc_mode(self, val: int) -> None:
        self._pp_attr_set("psd_zero_crossing_mode", str(val))

    def get_psd_zc_time_window_low(self) -> int:
        """PP_FMT_X2 -- already converted by the kernel, see
        get_trapez_R()'s docstring. No client-side conversion.
        """
        return int(self._pp_attr_get("psd_time_walk_low"))

    def set_psd_zc_time_window_low(self, val: int) -> None:
        self._pp_attr_set("psd_time_walk_low", str(val))

    def get_psd_zc_time_window_high(self) -> int:
        return int(self._pp_attr_get("psd_time_walk_high"))

    def set_psd_zc_time_window_high(self, val: int) -> None:
        self._pp_attr_set("psd_time_walk_high", str(val))

    def get_mem1_sig_select(self) -> int:
        return int(self._pp_attr_get("debug_signal1"))

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
        """Reads the 65536-byte, 16384-bin u32 histogram via self._mca_pp.

        The updated board ABI exposes four ordered 16384-byte attributes,
        histogram_data0..3, because remote iiod rejects the old monolithic
        65536-byte value with EFBIG. Confirmed live on both pulse processors:
        concatenating chunks 0 through 3 reconstructs all 16384 bins. The
        monolithic histogram_data path remains as a compatibility fallback
        for earlier drivers.

        The current driver source explicitly permits live reads: it copies
        bins while hardware may be updating them, so a running snapshot is
        informational rather than atomic. After enable=0, the same path
        returns the stable final histogram. This matches user-api.md and
        supersedes an earlier driver revision/docstring that described an
        EBUSY guard while running.
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
