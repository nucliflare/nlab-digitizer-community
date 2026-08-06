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

MCABackend is fully stubbed: this driver rewrite only covers the scope core
so far, no pulse-processor equivalent has been ported yet.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time

import iio
import numpy as np

from .base import DigitizerBackend

log = logging.getLogger(__name__)

_MCA_UNSUPPORTED = (
    "IIO MCA backend not implemented: this driver rewrite (vdpp-scope) only "
    "covers the scope core so far."
)

_SCOPE_DEVICE_NAME = "vdpp_scope"

# The AFE DAC now has a real driver: vdpp_afe_dac, added to the device tree
# after this backend was first written (the ad5686r device present on the
# live tree is a different, unrelated DAC -- not this one). Confirmed live:
# a single vdpp_afe_dac device instance (not one per scope, unlike
# vdpp_scope) exposes two output channels, voltage0 and voltage1, one per
# scope channel -- each with `raw` (the underlying DAC output code, full
# range, not what the GUI's DC-offset control means) and `baseline` (the
# actual DC-offset control: 0-1023 confirmed via voltage0/1's
# baseline_available, matching Scope's DAC_VALUE spec of 0-1024 closely
# enough that the 1023-vs-1024 top-of-range mismatch is left to the driver
# to enforce rather than clamped here).
_DAC_DEVICE_NAME = "vdpp_afe_dac"

# Matches the reference stub's (scope_backend_iio.py) DMA_CLOSE_DRAIN_SECONDS /
# DMA_CLOSE_JOIN_SECONDS. Our own close path was missing this drain step
# entirely (see _drain_for_close()'s docstring) -- ported from the stub once
# that gap was identified by comparing against it directly.
_DMA_CLOSE_DRAIN_SECONDS = 0.050
_DMA_CLOSE_JOIN_SECONDS = 1.0
# Matches the reference stub's DMA_CLOSE_RELEASE_SECONDS /
# DMA_CLOSE_RELEASE_POLL_SECONDS. The stock Xilinx DMA terminate path polls
# for HALTED for about one second; with a network context iiod can
# acknowledge buffer destruction before that target-side cleanup becomes
# visible through sysfs, so _wait_for_dma_release() polls for this long
# rather than trusting the buffer object being gone.
_DMA_CLOSE_RELEASE_SECONDS = 1.25
_DMA_CLOSE_RELEASE_POLL_SECONDS = 0.050

# vdpp-scope.c's trigger_mode is a plain integer register (0..4), not a
# string attribute like the previous driver. Confirmed from the driver
# source: SCOPE_TRIG_LEVEL_ABOVE=0, SCOPE_TRIG_LEVEL_BELOW=1,
# SCOPE_TRIG_EDGE_FALL=2, SCOPE_TRIG_EDGE_RISE=3, SCOPE_TRIG_PERIODIC=4 --
# these already match scope.TriggerMode's int values 1:1
# (ANY_ABOVE/ANY_BELOW/FALLING_EDGE/RISING_EDGE/TIMED), so get_edge/set_edge
# can pass the int straight through with no lookup table.


class IIODigitizerBackend(DigitizerBackend):
    """Single IIO network context implementing ScopeBackend against the
    rewritten vdpp_scope driver. MCABackend is stubbed (see module
    docstring).

    vdpp_scope instances all share the identical device name -- there is
    no per-instance label or channel_index attribute the way the previous
    driver had -- so channel selection is purely by discovery order,
    matching the reference scope_backend_iio.py stub this was rewritten
    from.
    """

    def __init__(
        self,
        channel: int,
        uri: str = "ip:192.168.10.128:30431",
        dac_name: str | None = _DAC_DEVICE_NAME,
        dac_channel: int | None = None,
    ) -> None:
        self._ch = channel
        self._ctx = iio.Context(uri)

        scopes = [d for d in self._ctx.devices if d.name == _SCOPE_DEVICE_NAME]
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

        log.info("IIO backend: connected ch%d (%s) to %s", channel, _SCOPE_DEVICE_NAME, uri)

    def close(self) -> None:
        log.info("IIO backend: closing ch%d", self._ch)
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
        return len(drained)

    def _close_dma_buffer(self) -> None:
        if self._dma_buf is None:
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
        try:
            if self.get_enable():
                self.set_enable(False)
        except OSError:
            pass

        try:
            self._drain_for_close(self._dma_buf)
        except Exception:
            log.warning(
                "IIO backend ch%d: DMA close-drain failed",
                self._ch, exc_info=True,
            )

        self._dma_buf = None
        self._dma_buf_frame_samples = None

        try:
            self._wait_for_dma_release()
        except RuntimeError:
            log.warning(
                "IIO backend ch%d: DMA buffer close did not release the "
                "hardware gate in time -- may indicate a genuine channel "
                "fault (check dmesg for xilinx-vdma errors), not something "
                "a client-side retry can fix", self._ch, exc_info=True,
            )

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
            active = int(self._attr_get("dma_buffer_active")) != 0
            if not active and not self.get_dma_enable():
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

    def _attr_get(self, name: str) -> str:
        return self._scope.attrs[name].value

    def _attr_set(self, name: str, value: str) -> None:
        self._scope.attrs[name].value = value

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
        (SCOPE_TRIG_PERIODIC=4). Period between periodic triggers, in
        8 ns datapath clocks.
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

    def read_frame(self) -> np.ndarray:
        """Read the on-chip pulse-viewer memory: one entry per core clock
        (frame_samples / 4 entries, capped at viewer_mem_entries=2048),
        as a space-separated text attribute rather than the previous
        driver's binary blob.

        Self-contained per call -- per vdpp-scope.c's viewer_data_show():
        reading this attribute itself raises the pulse-viewer semaphore,
        sleeps out the worst-case frame time (~20-25ms) so the read can't
        catch a half-written pulse, copies the memory, and drops the
        semaphore, all synchronously inside the read. No separate arm
        attribute is needed this time (unlike the previous driver
        rewrite's capture_enable/viewer_update_enable pair) -- set_enable
        (True) is the only precondition, and once armed every read_frame()
        call blocks for that ~20-25ms and returns a genuinely fresh
        snapshot.

        The driver caps the text at PAGE_SIZE (4096 bytes); if fewer than
        frame_samples // 4 entries come back, the frame was truncated
        server-side (frame_samples too large for one page) -- logged here,
        not otherwise recoverable.

        NOTE: deliberately does NOT cross-check against the viewer_samples
        attribute, which looks like it should report this same expected
        count but doesn't. Confirmed from the driver source:
        viewer_data_show() computes its loop bound as frame_length /
        SCOPE_N (i.e. // 4, matching its own "one viewer entry is the
        average of SCOPE_N samples" comment) -- but viewer_samples_show()
        returns the raw frame_length register value undivided, despite
        carrying the identical comment. Confirmed live: frame_samples=168
        -> 42 real entries from viewer_data (168 // 4, correct), while
        viewer_samples itself reports 168. Computing the expected count
        locally instead avoids a false "truncated" warning on every read.
        """
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
        expected = min(self.get_frame_samples() // 4, self.get_mem_frame_size())
        if len(samples) < expected:
            log.warning(
                "IIO backend ch%d: viewer_data truncated (%d of %d entries) "
                "-- frame_samples may be too large for one PAGE_SIZE read",
                self._ch, len(samples), expected,
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

        Separately, confirmed live that sustained reuse of the same buffer
        can start failing with a *persistent* EINVAL after roughly 200-235
        successful refills -- unlike the transient case above, this does
        not clear even after 5+ seconds of retrying, contradicting the
        driver's own "every read() returns exactly one waveform" reuse-is-
        safe design comment. Since retrying cannot fix it, one recreation
        of the buffer (close, reopen, re-arm) is tried instead, turning
        what would otherwise be a fatal, capture-ending error into a
        one-frame hiccup. If recreating doesn't help either, the second
        attempt's exception propagates rather than looping forever.
        """
        n = self.get_frame_samples()
        if self._dma_buf is None or self._dma_buf_frame_samples != n:
            self._create_dma_buffer(n)

        def refill(b: iio.Buffer) -> int:
            return iio._buffer_refill(b._buffer)

        buf = self._dma_buf
        assert buf is not None
        try:
            nbytes = self._retry_errno(lambda: refill(buf), (22,))  # type: ignore[arg-type]
        except OSError as e:
            if e.errno != 22:
                raise
            log.warning(
                "IIO backend ch%d: DMA buffer stopped refilling (EINVAL) "
                "after sustained reuse -- recreating and continuing",
                self._ch,
            )
            self._create_dma_buffer(n)
            buf = self._dma_buf
            assert buf is not None
            nbytes = self._retry_errno(lambda: refill(buf), (22,))  # type: ignore[arg-type]

        start = iio._buffer_start(buf._buffer)
        return ctypes.string_at(start, nbytes)

    def _create_dma_buffer(self, n: int) -> None:
        """Close any existing buffer, open a fresh one sized for n
        samples, and arm acquisition if it isn't already -- shared by
        _refill_dma_buffer()'s normal path and its EINVAL recovery path.

        Matches the reference stub's arm_capture()/start_capture(): no
        polling wait for dma_enable and no retry around set_enable(True).
        Earlier revisions of this method needed both, but only because
        _close_dma_buffer() used to return immediately after writing
        enable/dma_enable directly, without confirming the driver had
        actually finished tearing down the previous buffer -- the very
        thing _wait_for_dma_release() now blocks on before returning. With
        that guarantee in place, a freshly created buffer's DMA_ENABLE is
        already up by the time this call returns, the same way it is in
        the reference implementation.
        """
        self._close_dma_buffer()
        self._dma_buf = iio.Buffer(self._scope, n, False)
        self._dma_buf_frame_samples = n
        if not self.get_enable():
            self.set_enable(True)

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
    # MCABackend — not implemented (see module docstring)
    # ------------------------------------------------------------------

    def get_hw_version(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_sw_version(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_id_number(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_dpp_trigger_level(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_dpp_trigger_level(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_pulse_polarity(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_pulse_polarity(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_bsln_window(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_bsln_window(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_dpp_pretrigger_samples(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_dpp_pretrigger_samples(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_dpp_frame_samples(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_dpp_frame_samples(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_global_enable(self) -> bool:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_global_enable(self, val: bool) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_dpp_dma_enable(self) -> bool:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_dpp_dma_enable(self, val: bool) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_time_limit(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_time_limit(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_ext_trig_enable(self) -> bool:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_ext_trig_enable(self, val: bool) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_trg_source(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_trg_source(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_energy_bin(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_energy_bin(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_pileup_window(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_pileup_window(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_measurement_in_progress(self) -> bool:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_pulse_deadtime(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_events_lost(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_count_rate(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_elapsed_time(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_pulse_overrange_counter(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_pulse_pileup_counter(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_energy_overrange_counter(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_energy_estimation_error(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_throughput_error_counter(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_lp_coeffs_size(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_lp_coeffs_preset(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_lp_coeffs(self) -> np.ndarray:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_lp_coeffs(self, coeffs: list[int]) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_iir_lp_average(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_temp_coeff(self) -> float:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_temp_coeff(self, val: float) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_temp_offset(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_temp_offset(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_edge_det_coeff(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_edge_det_coeff(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_crrc2_Cdelay(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_crrc2_Cdelay(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_crrc2_Fdelay(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_crrc2_Fdelay(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_crrc2_pzc_coeff(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_crrc2_pzc_coeff(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_trapez_enable(self) -> bool:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_trapez_enable(self, val: bool) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_trapez_R(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_trapez_R(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_trapez_M(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_trapez_M(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_trapez_T(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_trapez_T(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_trapez_E(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_trapez_E(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_trapez_FT(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_trapez_FT(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_cfd_enable(self) -> bool:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_cfd_enable(self, val: bool) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_cfd_factor(self) -> float:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_cfd_factor(self, val: float) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_cfd_delay(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_cfd_delay(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_cfd_time_window_low(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_cfd_time_window_low(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_cfd_time_window_high(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_cfd_time_window_high(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_cc_enable(self) -> bool:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_cc_enable(self, val: bool) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_cc_time(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_cc_time(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_psd_zc_enable(self) -> bool:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_psd_zc_enable(self, val: bool) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_psd_zc_mode(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_psd_zc_mode(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_psd_zc_time_window_low(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_psd_zc_time_window_low(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_psd_zc_time_window_high(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_psd_zc_time_window_high(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_mem1_sig_select(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_mem1_sig_select(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_mem2_sig_select(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_mem2_sig_select(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_mem_amount(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_mem_frame_sizes(self) -> np.ndarray:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def read_waveform_banks(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def read_histogram(self) -> np.ndarray:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def clear_histogram(self) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_histogram_size(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_sync_enable(self) -> bool:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_sync_enable(self, val: bool) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_sync_sw_trig(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_sync_sw_trig(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_sync_trig_src(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def set_sync_trig_src(self, val: int) -> None:
        raise NotImplementedError(_MCA_UNSUPPORTED)

    def get_sync_timestamp(self) -> int:
        raise NotImplementedError(_MCA_UNSUPPORTED)
