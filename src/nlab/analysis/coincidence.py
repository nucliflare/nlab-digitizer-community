"""Bounded two-channel list-mode coincidence analysis.

Fine timing follows PetaLinux's explicitly selected
``vdpp-zc-calc-q2.14-v1`` profile. The kernel still transports opaque records;
the producer schema, common epoch, and channel-delay calibration remain
measurement qualifications rather than properties of the IIO transport.
"""

from __future__ import annotations

import math
import warnings
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from heapq import merge

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from nlab.hardware.digitizer.iio_listmode import (
    ADC_SAMPLE_Q,
    COARSE_TICK_NS,
    COARSE_TICK_Q,
    FINE_RAW_MAX,
    FINE_RAW_MIN,
    MARKER_CFD_VALID,
    MARKER_PSD_ZC_VALID,
    TIME_Q_PER_NS,
    VDPP_ZC_CALC_SCHEMA,
    cfd_correction_q,
)

TICK_NS = COARSE_TICK_NS
FINE_BIN_DIVISOR = 16
FINE_BIN_Q = TIME_Q_PER_NS // FINE_BIN_DIVISOR
FINE_BIN_NS = FINE_BIN_Q / TIME_Q_PER_NS
COARSE_BIN_Q = COARSE_TICK_Q
HISTOGRAM_BINS = 16_384
DMA_ENERGY_TO_MCA_SHIFT = 2
COINCIDENCE_MATRIX_BINS = 512
COINCIDENCE_MATRIX_CHANNELS_PER_BIN = HISTOGRAM_BINS // COINCIDENCE_MATRIX_BINS
PEAK_FIT_MIN_PAIRS = 100
PEAK_FIT_HALF_WINDOW_NS = 4.0
_GAUSSIAN_FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))


@dataclass(frozen=True)
class CoincidenceSettings:
    operator: str = "AND"
    not_ch0: bool = False
    not_ch1: bool = False
    low_q: int = -48 * TIME_Q_PER_NS
    high_q: int = 48 * TIME_Q_PER_NS
    channel_delay_q: int = 0
    roi_ch0: tuple[int, int] | None = None
    roi_ch1: tuple[int, int] | None = None
    energy_bin_ch0: int = 0
    energy_bin_ch1: int = 0
    fine_timing: bool = False
    random_sidebands: bool = True
    random_sideband_gap_q: int = 0
    record_schema: str = VDPP_ZC_CALC_SCHEMA

    @property
    def bin_width_ns(self) -> float:
        return FINE_BIN_NS if self.fine_timing else TICK_NS

    @property
    def bin_width_q(self) -> int:
        return FINE_BIN_Q if self.fine_timing else COARSE_BIN_Q

    @property
    def low_ns(self) -> float:
        return self.low_q / TIME_Q_PER_NS

    @property
    def high_ns(self) -> float:
        return self.high_q / TIME_Q_PER_NS

    @property
    def channel_delay_ns(self) -> float:
        return self.channel_delay_q / TIME_Q_PER_NS

    def __post_init__(self) -> None:
        if self.operator not in {"AND", "OR", "XOR"}:
            raise ValueError("operator must be AND, OR, or XOR")
        if self.low_q >= self.high_q:
            raise ValueError("coincidence low boundary must precede high boundary")
        if self.operator != "AND" and (self.not_ch0 or self.not_ch1):
            raise ValueError("NOT is only supported with AND")
        if self.not_ch0 and self.not_ch1:
            raise ValueError("both coincidence inputs cannot be negated")
        if self.record_schema != VDPP_ZC_CALC_SCHEMA:
            raise ValueError(f"unsupported coincidence record schema {self.record_schema!r}")
        for energy_bin in (self.energy_bin_ch0, self.energy_bin_ch1):
            if not 0 <= energy_bin <= 9:
                raise ValueError("energy_bin must be in 0..9")
        if self.random_sideband_gap_q < 0:
            raise ValueError("random sideband gap cannot be negative")

    @property
    def random_scale(self) -> float:
        """Scale two equal-width sidebands to one prompt-window width."""
        return 0.5 if self.random_sidebands else 0.0


@dataclass
class _Event:
    tick: int
    energy: int
    zc_offset: int | None = None
    evaluated: bool = False
    paired: bool = False
    accepted: bool = False


@dataclass(frozen=True)
class CoincidenceSnapshot:
    delay_counts: np.ndarray
    energy_ch0: np.ndarray
    energy_ch1: np.ndarray
    prompt_matrix: np.ndarray
    random_matrix: np.ndarray
    rate_seconds: np.ndarray
    rate_counts: np.ndarray
    pairs: int
    random_pairs: int
    accepted_ch0: int
    accepted_ch1: int
    ambiguous: int
    zero_timestamps: int
    cfd_valid: int
    psd_zc_valid: int
    cfd_skipped: int
    fine_out_of_range: int
    offset_boundary_pairs: int
    outside_roi: int
    energy_overflow: int


@dataclass(frozen=True)
class CoincidencePeakFit:
    """Background-plus-Gaussian fit of the central coincidence peak.

    The result describes the fitted Gaussian core, not automatically the
    complete detector CTR distribution. ``reduced_chi_square`` exposes broad
    tails or multiple components that make that distinction important.
    """

    center_ns: float
    sigma_ns: float
    fwhm_ns: float
    fwhm_uncertainty_ns: float
    amplitude_per_bin: float
    background_per_bin: float
    signal_counts: float
    reduced_chi_square: float
    fit_low_ns: float
    fit_high_ns: float


def _gaussian_with_background(
    values: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    background: float,
) -> np.ndarray:
    return background + amplitude * np.exp(-0.5 * ((values - center) / sigma) ** 2)


def fit_coincidence_peak(
    delay_counts: np.ndarray,
    settings: CoincidenceSettings,
) -> CoincidencePeakFit | None:
    """Fit the dominant fine-timing peak, or return ``None`` if unqualified.

    A constant-background Gaussian is intentionally limited to a local
    +/-4 ns window. At least 100 total pairs, about 50 fitted signal counts,
    and a 3-sigma peak-height significance are required. These guards prevent
    sparse or background-only histograms from being reported as timing
    resolution measurements.
    """
    if not settings.fine_timing or delay_counts.ndim != 1:
        return None
    counts = np.asarray(delay_counts, dtype=np.float64)
    if len(counts) < 8 or float(np.sum(counts)) < PEAK_FIT_MIN_PAIRS:
        return None

    bin_width_ns = settings.bin_width_ns
    centers = settings.low_ns + (np.arange(len(counts), dtype=np.float64) + 0.5) * bin_width_ns
    peak_index = int(np.argmax(counts))
    peak_center = float(centers[peak_index])
    selected = np.abs(centers - peak_center) <= PEAK_FIT_HALF_WINDOW_NS
    x = centers[selected]
    y = counts[selected]
    if len(x) < 8 or np.count_nonzero(y) < 5:
        return None

    background_guess = float(np.percentile(y, 25.0))
    amplitude_guess = float(np.max(y) - background_guess)
    if amplitude_guess <= 0.0:
        return None
    sigma_guess = max(0.4, 2.0 * bin_width_ns)
    lower = (0.0, max(float(x[0]), peak_center - 2.0), bin_width_ns, 0.0)
    upper = (np.inf, min(float(x[-1]), peak_center + 2.0), PEAK_FIT_HALF_WINDOW_NS, np.inf)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            parameters, covariance = curve_fit(
                _gaussian_with_background,
                x,
                y,
                p0=(amplitude_guess, peak_center, sigma_guess, background_guess),
                bounds=(lower, upper),
                sigma=np.sqrt(np.maximum(y, 1.0)),
                absolute_sigma=True,
                maxfev=20_000,
            )
    except (OptimizeWarning, RuntimeError, TypeError, ValueError):
        return None

    amplitude, center, sigma, background = map(float, parameters)
    variance = float(covariance[2, 2])
    if variance < 0.0 or not all(
        math.isfinite(value) for value in (amplitude, center, sigma, background, variance)
    ):
        return None
    signal_counts = amplitude * sigma * math.sqrt(2.0 * math.pi) / bin_width_ns
    peak_significance = amplitude / math.sqrt(max(amplitude + background, 1.0))
    if signal_counts < 50.0 or peak_significance < 3.0:
        return None

    fitted = _gaussian_with_background(x, amplitude, center, sigma, background)
    degrees_of_freedom = max(len(y) - 4, 1)
    reduced_chi_square = float(
        np.sum(np.square(y - fitted) / np.maximum(fitted, 1.0)) / degrees_of_freedom
    )
    return CoincidencePeakFit(
        center_ns=center,
        sigma_ns=sigma,
        fwhm_ns=_GAUSSIAN_FWHM_FACTOR * sigma,
        fwhm_uncertainty_ns=_GAUSSIAN_FWHM_FACTOR * math.sqrt(max(variance, 0.0)),
        amplitude_per_bin=amplitude,
        background_per_bin=background,
        signal_counts=signal_counts,
        reduced_chi_square=reduced_chi_square,
        fit_low_ns=float(x[0]),
        fit_high_ns=float(x[-1]),
    )


class CoincidenceAnalyzer:
    """Incremental matcher with exact integer event-time coordinates.

    Ordinary AND emits every pair inside the inclusive time gate. One event
    may therefore participate in several pairs; accepted event histograms and
    counts contain unique participating events. Veto and XOR accept an event
    only after the opposite stream's event-time watermark proves that its
    inclusive partner window is complete.
    """

    def __init__(self, settings: CoincidenceSettings, *, max_pending: int = 1_000_000):
        self.settings = settings
        self.max_pending = max_pending
        self._events: tuple[list[_Event], list[_Event]] = ([], [])
        self._times: tuple[list[int], list[int]] = ([], [])
        self._cursor = [0, 0]
        self._watermark: list[int | None] = [None, None]
        self._raw_last: list[int | None] = [None, None]
        self._low_units = settings.low_q
        self._high_units = settings.high_q
        self._bin_units = settings.bin_width_q
        delay_bins = (self._high_units - self._low_units) // self._bin_units + 1
        self.delay_counts = np.zeros(delay_bins, dtype=np.uint64)
        self.energy_counts = (
            np.zeros(HISTOGRAM_BINS, dtype=np.uint64),
            np.zeros(HISTOGRAM_BINS, dtype=np.uint64),
        )
        self.prompt_matrix = np.zeros(
            (COINCIDENCE_MATRIX_BINS, COINCIDENCE_MATRIX_BINS), dtype=np.uint64
        )
        self.random_matrix = np.zeros_like(self.prompt_matrix)
        self.pairs = 0
        self.random_pairs = 0
        self.accepted = [0, 0]
        self.ambiguous = 0
        self.zero_timestamps = 0
        self.cfd_valid = 0
        self.psd_zc_valid = 0
        self.cfd_skipped = 0
        self.fine_out_of_range = 0
        self.offset_boundary_pairs = 0
        self.outside_roi = 0
        self.energy_overflow = 0
        self._rate_origin: int | None = None
        self._rate: dict[int, int] = {}

    def add_batch(self, channel: int, events: np.ndarray) -> None:
        if channel not in (0, 1):
            raise ValueError("coincidence channel must be 0 or 1")
        names = events.dtype.names or ()
        if not {"timestamp", "marker", "trapezoid_energy"}.issubset(names):
            raise ValueError("IIO list-mode timestamp, marker, and energy are required")
        if self.settings.fine_timing and not {"zc_offset", "zc_estimation"}.issubset(names):
            raise ValueError("CFD timing requires offset and signed Q2.14 fields")
        ticks = events["timestamp"]
        if not len(ticks):
            return

        zero = ticks == 0
        self.zero_timestamps += int(np.count_nonzero(zero))
        valid = ticks[~zero]
        if not len(valid):
            return
        first, last = int(valid[0]), int(valid[-1])
        previous = self._raw_last[channel]
        if previous is not None and first < previous:
            raise ValueError(f"channel {channel} list-mode timestamps reversed")
        if np.any(valid[1:] < valid[:-1]):
            raise ValueError(f"channel {channel} list-mode timestamps reversed within frame")
        self._raw_last[channel] = last

        calibration_q = -self.settings.channel_delay_q if channel == 1 else 0
        # A valid fine correction has a minimum of -1 ADC sample. Unsigned
        # whole-sample offsets only move an event later in the encoded profile.
        earliest_correction = -ADC_SAMPLE_Q if self.settings.fine_timing else 0
        self._watermark[channel] = last * COARSE_TICK_Q + calibration_q + earliest_correction

        marker = events["marker"]
        psd_valid = (marker & MARKER_PSD_ZC_VALID) != 0
        cfd_valid = ((marker & MARKER_CFD_VALID) != 0) & ~psd_valid
        self.cfd_valid += int(np.count_nonzero(cfd_valid & ~zero))
        self.psd_zc_valid += int(np.count_nonzero(psd_valid & ~zero))

        selected = ~zero
        # Capture-validated application mapping. The producer calls this the
        # selected energy estimator; it is not guaranteed to be trapezoidal.
        energies = np.right_shift(events["trapezoid_energy"], DMA_ENERGY_TO_MCA_SHIFT)
        roi = self.settings.roi_ch0 if channel == 0 else self.settings.roi_ch1
        if roi is not None:
            inside = (energies >= roi[0]) & (energies <= roi[1])
            self.outside_roi += int(np.count_nonzero(selected & ~inside))
            selected &= inside

        if self.settings.fine_timing:
            estimates = events["zc_estimation"]
            fine_in_range = (estimates >= FINE_RAW_MIN) & (estimates <= FINE_RAW_MAX)
            self.fine_out_of_range += int(np.count_nonzero(selected & cfd_valid & ~fine_in_range))
            eligible = cfd_valid & fine_in_range
            self.cfd_skipped += int(np.count_nonzero(selected & ~eligible))
            selected &= eligible
            incoming: list[_Event] = []
            for tick, energy, zc_offset, zc_estimation in zip(
                ticks[selected],
                energies[selected],
                events["zc_offset"][selected],
                estimates[selected],
                strict=True,
            ):
                offset = int(zc_offset)
                corrected = (
                    int(tick) * COARSE_TICK_Q
                    + calibration_q
                    + cfd_correction_q(offset, int(zc_estimation))
                )
                incoming.append(_Event(corrected, int(energy), offset))
            # Fine corrections can reorder nearby records and block edges.
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
                corrected = int(tick) * COARSE_TICK_Q + calibration_q
                self._events[channel].append(_Event(corrected, int(energy)))
                self._times[channel].append(corrected)

        if sum(map(len, self._events)) > self.max_pending:
            raise RuntimeError("coincidence pending-event limit exceeded; results are invalid")
        self._advance()

    def finish(self) -> None:
        """Finalize only after both producer streams have stopped and drained."""
        terminal = (1 << 64) * COARSE_TICK_Q + 256 * ADC_SAMPLE_Q + 1
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
            self.prompt_matrix.copy(),
            self.random_matrix.copy(),
            seconds,
            rate_counts,
            self.pairs,
            self.random_pairs,
            self.accepted[0],
            self.accepted[1],
            self.ambiguous,
            self.zero_timestamps,
            self.cfd_valid,
            self.psd_zc_valid,
            self.cfd_skipped,
            self.fine_out_of_range,
            self.offset_boundary_pairs,
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
        return self._events[opposite][bisect_left(times, low) : bisect_right(times, high)]

    def _random_candidates(self, event: _Event) -> list[_Event]:
        """Return CH1 events in two non-overlapping delayed sidebands.

        Each sideband has the same physical width as the inclusive prompt
        window. Their combined counts are therefore scaled by one half before
        subtraction from the prompt matrix. Prompt boundary events are never
        also classified as random events.
        """
        if not self.settings.random_sidebands:
            return []
        width = self._high_units - self._low_units
        gap = self.settings.random_sideband_gap_q
        lower_start = event.tick + self._low_units - gap - width
        lower_stop = event.tick + self._low_units - gap
        upper_start = event.tick + self._high_units + gap
        upper_stop = event.tick + self._high_units + gap + width
        times = self._times[1]
        events = self._events[1]
        lower = events[bisect_left(times, lower_start) : bisect_left(times, lower_stop)]
        upper = events[bisect_right(times, upper_start) : bisect_right(times, upper_stop)]
        return [*lower, *upper]

    def _can_finalize(self, channel: int, event: _Event) -> bool:
        other = self._watermark[1 - channel]
        if other is None:
            return False
        if (
            channel == 0
            and self.settings.operator == "AND"
            and not (self.settings.not_ch0 or self.settings.not_ch1)
            and self.settings.random_sidebands
        ):
            width = self._high_units - self._low_units
            limit = (
                event.tick
                + self._high_units
                + self.settings.random_sideband_gap_q
                + width
            )
        else:
            limit = event.tick + self._high_units if channel == 0 else event.tick - self._low_units
        # Gate edges are inclusive, so only a strictly later watermark closes it.
        return other > limit

    def _accept(self, channel: int, event: _Event) -> None:
        if event.accepted:
            return
        event.accepted = True
        self.accepted[channel] += 1
        if 0 <= event.energy < HISTOGRAM_BINS:
            self.energy_counts[channel][event.energy] += 1
        else:
            self.energy_overflow += 1
        second = event.tick // (TIME_Q_PER_NS * 1_000_000_000)
        if self._rate_origin is None or second < self._rate_origin:
            self._rate_origin = second
        self._rate[second] = self._rate.get(second, 0) + 1
        if len(self._rate) > 600:
            del self._rate[min(self._rate)]

    def _accept_all_pairs(self, event: _Event, candidates: list[_Event]) -> None:
        if candidates:
            self.ambiguous += int(len(candidates) > 1)
            event.paired = True
            self._accept(0, event)
        for partner in candidates:
            partner.paired = True
            self._accept(1, partner)
            self.pairs += 1
            if (
                event.zc_offset is not None
                and partner.zc_offset is not None
                and abs(partner.zc_offset - event.zc_offset) > 128
            ):
                # Diagnostic only. The contract forbids guessing a modular
                # unwrap; the encoded pair remains at its decoded time.
                self.offset_boundary_pairs += 1
            delay_bin = (partner.tick - event.tick - self._low_units) // self._bin_units
            self.delay_counts[delay_bin] += 1
            self._increment_matrix(self.prompt_matrix, event, partner)
        for partner in self._random_candidates(event):
            self.random_pairs += 1
            self._increment_matrix(self.random_matrix, event, partner)

    @staticmethod
    def _increment_matrix(matrix: np.ndarray, ch0: _Event, ch1: _Event) -> None:
        x = ch0.energy // COINCIDENCE_MATRIX_CHANNELS_PER_BIN
        y = ch1.energy // COINCIDENCE_MATRIX_CHANNELS_PER_BIN
        if 0 <= x < COINCIDENCE_MATRIX_BINS and 0 <= y < COINCIDENCE_MATRIX_BINS:
            matrix[y, x] += 1

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
                        self._accept_all_pairs(event, candidates)
                    elif not candidates:
                        self._accept(channel, event)
                    event.evaluated = True
                    self._cursor[channel] += 1
        self._prune()

    def _prune(self) -> None:
        if None in self._watermark:
            return
        assert self._watermark[0] is not None and self._watermark[1] is not None
        if self.settings.random_sidebands:
            width = self._high_units - self._low_units
            gap = self.settings.random_sideband_gap_q
            span = max(
                abs(self._low_units - gap - width),
                abs(self._high_units + gap + width),
            )
        else:
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
