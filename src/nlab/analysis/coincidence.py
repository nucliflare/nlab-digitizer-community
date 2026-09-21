"""Bounded, timestamp-ordered two-channel list-mode coincidence analysis.

The IIO event decoder and 8 ns timestamp unit are client-side contracts; see
``petalinux/docs/mca-architecture.md`` and ``docs/user-api.md``. The IIO driver
transports opaque records and does not define their energy-to-histogram scale.
Recorded channel-0/1 captures with different ``energy_bin`` settings matched
their 16,384-bin MCA photopeaks after a fixed two-bit energy shift. This is a
capture-derived mapping, not a kernel-driver guarantee; validate it on new
firmware with a labelled source. Fine CFD event times are a provisional
client-side reconstruction for controlled live validation, not a driver ABI.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from heapq import merge

import numpy as np

from nlab.hardware.digitizer.iio_listmode import (
    MARKER_CFD_VALID,
    MARKER_PSD_ZC_VALID,
    ZC_FRACTION_SCALE,
    provisional_cfd_correction_q14,
)

TICK_NS = 8
FINE_BIN_NS = 1
_FINE_BIN_UNITS = ZC_FRACTION_SCALE // TICK_NS
# ap_uint<8> offset interpreted provisionally as signed, plus ap_fixed<16,2>.
# Reserve the full possible negative correction when advancing a stream's
# event-time watermark; later DMA frames can contain earlier corrected times.
_MIN_CFD_CORRECTION_UNITS = -130 * ZC_FRACTION_SCALE
HISTOGRAM_BINS = 16_384
DMA_ENERGY_TO_MCA_SHIFT = 2
_MAX_SIGNED_TICK = np.iinfo(np.int64).max


@dataclass(frozen=True)
class CoincidenceSettings:
    operator: str = "AND"
    not_ch0: bool = False
    not_ch1: bool = False
    low_tick: int = -6
    high_tick: int = 6
    offset_ch1_tick: int = 0
    roi_ch0: tuple[int, int] | None = None
    roi_ch1: tuple[int, int] | None = None
    energy_bin_ch0: int = 0
    energy_bin_ch1: int = 0
    fine_timing: bool = False

    @property
    def bin_width_ns(self) -> int:
        return FINE_BIN_NS if self.fine_timing else TICK_NS

    def __post_init__(self) -> None:
        if self.operator not in {"AND", "OR", "XOR"}:
            raise ValueError("operator must be AND, OR, or XOR")
        if self.low_tick >= self.high_tick:
            raise ValueError("coincidence low boundary must precede high boundary")
        if self.operator != "AND" and (self.not_ch0 or self.not_ch1):
            raise ValueError("NOT is only supported with AND")
        if self.not_ch0 and self.not_ch1:
            raise ValueError("both coincidence inputs cannot be negated")
        for energy_bin in (self.energy_bin_ch0, self.energy_bin_ch1):
            if not 0 <= energy_bin <= 9:
                raise ValueError("energy_bin must be in 0..9")


@dataclass
class _Event:
    tick: int
    energy: int
    evaluated: bool = False
    paired: bool = False


@dataclass(frozen=True)
class CoincidenceSnapshot:
    delay_counts: np.ndarray
    energy_ch0: np.ndarray
    energy_ch1: np.ndarray
    rate_seconds: np.ndarray
    rate_counts: np.ndarray
    pairs: int
    accepted_ch0: int
    accepted_ch1: int
    ambiguous: int
    zero_timestamps: int
    cfd_valid: int
    psd_zc_valid: int
    cfd_skipped: int
    outside_roi: int
    energy_overflow: int


class CoincidenceAnalyzer:
    """Incremental matcher; no file-sized arrays are retained.

    AND uses an A-ordered, nearest-available-B, one-to-one policy. Veto and
    XOR use *any* opposite-channel ROI-qualified event, independent of whether
    that event has already participated in another result. Decisions wait for
    the opposite stream's timestamp watermark; sparse list-mode streams may
    therefore not publish veto/XOR results until a frame or stop arrives.
    """

    def __init__(self, settings: CoincidenceSettings, *, max_pending: int = 1_000_000):
        self.settings = settings
        self.max_pending = max_pending
        self._events: tuple[list[_Event], list[_Event]] = ([], [])
        self._times: tuple[list[int], list[int]] = ([], [])
        self._cursor = [0, 0]
        self._watermark: list[int | None] = [None, None]
        self._raw_last: list[int | None] = [None, None]
        self._low_units = settings.low_tick * ZC_FRACTION_SCALE
        self._high_units = settings.high_tick * ZC_FRACTION_SCALE
        self._bin_units = _FINE_BIN_UNITS if settings.fine_timing else ZC_FRACTION_SCALE
        delay_bins = (self._high_units - self._low_units) // self._bin_units + 1
        self.delay_counts = np.zeros(delay_bins, dtype=np.uint64)
        self.energy_counts = (
            np.zeros(HISTOGRAM_BINS, dtype=np.uint64),
            np.zeros(HISTOGRAM_BINS, dtype=np.uint64),
        )
        self.pairs = 0
        self.accepted = [0, 0]
        self.ambiguous = 0
        self.zero_timestamps = 0
        self.cfd_valid = 0
        self.psd_zc_valid = 0
        self.cfd_skipped = 0
        self.outside_roi = 0
        self.energy_overflow = 0
        self._rate_origin: int | None = None
        self._rate: dict[int, int] = {}

    def add_batch(self, channel: int, events: np.ndarray) -> None:
        if channel not in (0, 1):
            raise ValueError("coincidence channel must be 0 or 1")
        names = events.dtype.names or ()
        if not {"timestamp", "marker", "trapezoid_energy"}.issubset(names):
            raise ValueError("IIO list-mode timestamp, marker, and trapezoid energy required")
        if self.settings.fine_timing and not {"zc_offset", "zc_estimation"}.issubset(names):
            raise ValueError("CFD timing requires offset and Q2.14 interpolation fields")
        ticks = events["timestamp"]
        if not len(ticks):
            return
        zero = ticks == 0
        self.zero_timestamps += int(np.count_nonzero(zero))
        valid = ticks[~zero]
        if not len(valid):
            return
        if int(np.max(valid)) > _MAX_SIGNED_TICK:
            raise ValueError("list-mode timestamp exceeds signed matching range")
        offset_units = (
            self.settings.offset_ch1_tick * ZC_FRACTION_SCALE if channel == 1 else 0
        )
        first, last = int(valid[0]), int(valid[-1])
        previous = self._raw_last[channel]
        if previous is not None and first < previous:
            raise ValueError(f"channel {channel} list-mode timestamps reversed")
        if np.any(valid[1:] < valid[:-1]):
            raise ValueError(f"channel {channel} list-mode timestamps reversed within frame")
        self._raw_last[channel] = last
        self._watermark[channel] = (
            last * ZC_FRACTION_SCALE
            + offset_units
            + (_MIN_CFD_CORRECTION_UNITS if self.settings.fine_timing else 0)
        )
        marker = events["marker"]
        psd_valid = (marker & MARKER_PSD_ZC_VALID) != 0
        cfd_valid = ((marker & MARKER_CFD_VALID) != 0) & ~psd_valid
        self.cfd_valid += int(np.count_nonzero(cfd_valid & ~zero))
        self.psd_zc_valid += int(np.count_nonzero(psd_valid & ~zero))
        # Marker bits 6/7 identify an input on ordinary events. The previous
        # 0x6000 mask inspected the *offset byte* and discarded valid pulses
        # whenever a negative CFD offset had its high bits set.
        selected = ~zero
        # The event field already tracks the configured MCA binning; shifting
        # by energy_bin again would put real photopeaks below their MCA ROIs.
        energies = np.right_shift(events["trapezoid_energy"], DMA_ENERGY_TO_MCA_SHIFT)
        roi = self.settings.roi_ch0 if channel == 0 else self.settings.roi_ch1
        if roi is not None:
            inside = (energies >= roi[0]) & (energies <= roi[1])
            self.outside_roi += int(np.count_nonzero(selected & ~inside))
            selected &= inside
        if self.settings.fine_timing:
            self.cfd_skipped += int(np.count_nonzero(selected & ~cfd_valid))
            selected &= cfd_valid
            incoming: list[_Event] = []
            for tick, energy, zc_offset, zc_estimation in zip(
                ticks[selected],
                energies[selected],
                events["zc_offset"][selected],
                events["zc_estimation"][selected],
                strict=True,
            ):
                corrected = (
                    int(tick) * ZC_FRACTION_SCALE
                    + offset_units
                    + provisional_cfd_correction_q14(int(zc_offset), int(zc_estimation))
                )
                incoming.append(_Event(corrected, int(energy)))
            # The offset can reorder close pulses both within and across DMA
            # frames. Keep candidate searches sorted by corrected event time.
            incoming.sort(key=lambda event: event.tick)
            pending = self._events[channel]
            if pending and incoming and incoming[0].tick < pending[-1].tick:
                pending[:] = merge(pending, incoming, key=lambda event: event.tick)
                self._times[channel][:] = [event.tick for event in pending]
                self._cursor[channel] = next(
                    (index for index, event in enumerate(pending) if not event.evaluated),
                    len(pending),
                )
            else:
                pending.extend(incoming)
                self._times[channel].extend(event.tick for event in incoming)
        else:
            for tick, energy in zip(ticks[selected], energies[selected], strict=True):
                corrected = int(tick) * ZC_FRACTION_SCALE + offset_units
                self._events[channel].append(_Event(corrected, int(energy)))
                self._times[channel].append(corrected)
        if sum(map(len, self._events)) > self.max_pending:
            raise RuntimeError("coincidence pending-event limit exceeded; results are invalid")
        self._advance()

    def finish(self) -> None:
        """Finalize unmatched singles only after both producer streams end."""
        terminal = (_MAX_SIGNED_TICK + 131) * ZC_FRACTION_SCALE
        self._watermark = [terminal, terminal]
        self._advance()

    def snapshot(self) -> CoincidenceSnapshot:
        if self._rate:
            latest = max(self._rate)
            absolute_seconds = range(max(min(self._rate), latest - 599), latest + 1)
            origin = self._rate_origin or 0
            seconds = np.array([second - origin for second in absolute_seconds], dtype=np.int64)
            rate_counts = np.array(
                [self._rate.get(second, 0) for second in absolute_seconds], dtype=np.uint64
            )
        else:
            seconds = np.array([], dtype=np.int64)
            rate_counts = np.array([], dtype=np.uint64)
        return CoincidenceSnapshot(
            self.delay_counts.copy(),
            self.energy_counts[0].copy(),
            self.energy_counts[1].copy(),
            seconds,
            rate_counts,
            self.pairs,
            self.accepted[0],
            self.accepted[1],
            self.ambiguous,
            self.zero_timestamps,
            self.cfd_valid,
            self.psd_zc_valid,
            self.cfd_skipped,
            self.outside_roi,
            self.energy_overflow,
        )

    def _opposite_candidates(self, channel: int, event: _Event) -> list[_Event]:
        opposite = 1 - channel
        if channel == 0:
            low = event.tick + self._low_units
            high = event.tick + self._high_units
        else:
            low = event.tick - self._high_units
            high = event.tick - self._low_units
        times = self._times[opposite]
        start = bisect_left(times, low)
        stop = bisect_right(times, high)
        return self._events[opposite][start:stop]

    def _can_finalize(self, channel: int, event: _Event) -> bool:
        own = self._watermark[channel]
        if self.settings.fine_timing and (own is None or own <= event.tick):
            # A future record on this same stream may move back by up to 130
            # coarse ticks. Wait before committing A-ordered one-to-one pairs.
            return False
        other = self._watermark[1 - channel]
        if other is None:
            return False
        limit = (
            event.tick + self._high_units
            if channel == 0
            else event.tick - self._low_units
        )
        # The boundary is inclusive; a future event exactly at it must still
        # be eligible. Only a strictly later watermark closes the window.
        return other > limit

    def _accept(self, channel: int, event: _Event) -> None:
        self.accepted[channel] += 1
        if 0 <= event.energy < HISTOGRAM_BINS:
            self.energy_counts[channel][event.energy] += 1
        else:
            self.energy_overflow += 1
        second = event.tick * TICK_NS // (ZC_FRACTION_SCALE * 1_000_000_000)
        if self._rate_origin is None or second < self._rate_origin:
            self._rate_origin = second
        self._rate[second] = self._rate.get(second, 0) + 1
        if len(self._rate) > 600:
            del self._rate[min(self._rate)]

    def _advance(self) -> None:
        settings = self.settings
        if settings.operator == "OR":
            for channel in (0, 1):
                events = self._events[channel]
                while self._cursor[channel] < len(events):
                    event = events[self._cursor[channel]]
                    if not event.evaluated:
                        self._accept(channel, event)
                        event.evaluated = True
                    self._cursor[channel] += 1
        else:
            for channel in (0, 1):
                if settings.operator == "AND" and not settings.not_ch0 and channel == 1:
                    continue
                if settings.operator == "AND" and settings.not_ch0 and channel == 0:
                    continue
                if settings.operator == "AND" and settings.not_ch1 and channel == 1:
                    continue
                events = self._events[channel]
                while self._cursor[channel] < len(events):
                    event = events[self._cursor[channel]]
                    if event.evaluated:
                        self._cursor[channel] += 1
                        continue
                    if not self._can_finalize(channel, event):
                        break
                    candidates = self._opposite_candidates(channel, event)
                    if settings.operator == "AND" and not (settings.not_ch0 or settings.not_ch1):
                        available = [candidate for candidate in candidates if not candidate.paired]
                        if available:
                            center = (self._low_units + self._high_units) / 2
                            partner = min(
                                available,
                                key=lambda candidate: (
                                    abs(candidate.tick - event.tick - center),
                                    candidate.tick,
                                ),
                            )
                            partner.paired = True
                            event.paired = True
                            self.pairs += 1
                            self.ambiguous += int(len(available) > 1)
                            delay_bin = (
                                partner.tick - event.tick - self._low_units
                            ) // self._bin_units
                            self.delay_counts[delay_bin] += 1
                            self._accept(0, event)
                            self._accept(1, partner)
                        elif candidates:
                            self.ambiguous += 1
                    elif not candidates:
                        self._accept(channel, event)
                    event.evaluated = True
                    self._cursor[channel] += 1
        self._prune()

    def _prune(self) -> None:
        if None in self._watermark:
            return
        assert self._watermark[0] is not None and self._watermark[1] is not None
        span = max(abs(self._low_units), abs(self._high_units))
        cutoff = min(self._watermark[0], self._watermark[1]) - 2 * span - 1
        for channel in (0, 1):
            times = self._times[channel]
            removable = min(bisect_left(times, cutoff), self._cursor[channel])
            if self.settings.operator == "AND" and not self.settings.not_ch0 and channel == 1:
                removable = bisect_left(times, cutoff)
            if self.settings.operator == "AND" and self.settings.not_ch0 and channel == 0:
                removable = bisect_left(times, cutoff)
            if self.settings.operator == "AND" and self.settings.not_ch1 and channel == 1:
                removable = bisect_left(times, cutoff)
            if removable:
                del times[:removable]
                del self._events[channel][:removable]
                self._cursor[channel] = max(0, self._cursor[channel] - removable)
