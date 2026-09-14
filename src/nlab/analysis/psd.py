"""Incremental pulse-shape-discrimination histogram analysis.

The legacy gRPC list-mode record named its two charge estimates ``energy``
and ``short_energy``.  The deployed IIO client decoder names the equivalent
analysis inputs ``trapezoid_energy`` and ``charge_energy``.  The IIO transport
itself exposes only an opaque 16-byte record (see mca-architecture.md); this
module therefore owns the client-side interpretation and never changes the
raw capture path.

The ratio below preserves the old GUI's definition, ``1 - Qshort/Qtotal``.
For IIO records the mapping Qshort=charge_energy and
Qtotal=trapezoid_energy is source-derived and unit-tested but has not yet
been confirmed against a labelled live source.  The GUI consequently labels
the two populations "below cut" and "above cut", not neutron/gamma.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from fast_histogram import histogram2d


@dataclass(frozen=True)
class PsdStatistics:
    """Counters describing which list-mode records entered the PSD matrix."""

    received: int = 0
    accepted: int = 0
    zero_total: int = 0
    outside_range: int = 0


class PsdAccumulator:
    """Accumulate event batches into an energy-versus-ratio matrix."""

    def __init__(
        self,
        *,
        energy_bins: int = 1024,
        ratio_bins: int = 256,
        energy_right_shift: int = 0,
        ratio_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.configure(
            energy_bins=energy_bins,
            ratio_bins=ratio_bins,
            energy_right_shift=energy_right_shift,
            ratio_range=ratio_range,
        )

    def configure(
        self,
        *,
        energy_bins: int,
        ratio_bins: int,
        energy_right_shift: int,
        ratio_range: tuple[float, float],
    ) -> None:
        if energy_bins <= 0 or ratio_bins <= 0:
            raise ValueError("PSD bin counts must be positive")
        if not 0 <= energy_right_shift <= 15:
            raise ValueError("energy_right_shift must be in the range 0..15")
        ratio_low, ratio_high = ratio_range
        if not np.isfinite(ratio_low) or not np.isfinite(ratio_high):
            raise ValueError("PSD ratio limits must be finite")
        if ratio_low >= ratio_high:
            raise ValueError("PSD ratio minimum must be below its maximum")

        self.energy_bins = int(energy_bins)
        self.ratio_bins = int(ratio_bins)
        self.energy_right_shift = int(energy_right_shift)
        self.ratio_range = (float(ratio_low), float(ratio_high))
        # uint16 energy occupies 0..65535. A right shift changes the displayed
        # channel range but does not alter raw event capture.
        self.energy_range = (0.0, float(1 << (16 - self.energy_right_shift)))
        self.reset()

    def reset(self) -> None:
        self.matrix = np.zeros((self.energy_bins, self.ratio_bins), dtype=np.uint64)
        self.statistics = PsdStatistics()

    @staticmethod
    def _energy_fields(events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        names = events.dtype.names
        if names is None:
            raise ValueError("PSD events must use a structured NumPy dtype")
        if {"trapezoid_energy", "charge_energy"}.issubset(names):
            return events["trapezoid_energy"], events["charge_energy"]
        if {"energy", "short_energy"}.issubset(names):
            return events["energy"], events["short_energy"]
        if {"long_gate", "short_gate"}.issubset(names):
            return events["long_gate"], events["short_gate"]
        raise ValueError(f"unsupported MCA event fields: {', '.join(names)}")

    def add_events(self, events: np.ndarray) -> PsdStatistics:
        """Add one decoded DMA batch and return cumulative statistics."""
        total_raw, short_raw = self._energy_fields(events)
        received = len(events)
        if received == 0:
            return self.statistics

        total = total_raw.astype(np.float64)
        short = short_raw.astype(np.float64)
        nonzero = total > 0.0
        ratio = np.empty_like(total)
        ratio.fill(np.nan)
        np.divide(total - short, total, out=ratio, where=nonzero)

        energy = np.right_shift(total_raw.astype(np.uint64), self.energy_right_shift).astype(
            np.float64
        )
        ratio_low, ratio_high = self.ratio_range
        energy_low, energy_high = self.energy_range
        in_range = (
            nonzero
            & np.isfinite(ratio)
            & (energy >= energy_low)
            & (energy < energy_high)
            & (ratio >= ratio_low)
            & (ratio <= ratio_high)
        )
        accepted = int(np.count_nonzero(in_range))
        zero_total = int(received - np.count_nonzero(nonzero))
        outside_range = received - zero_total - accepted

        if accepted:
            # fast_histogram treats the upper range edge as exclusive. A
            # valid event with Qshort=0 has ratio exactly 1, so move only
            # that endpoint into the final bin instead of reporting it as
            # outside the configured view.
            accepted_ratio = np.minimum(
                ratio[in_range], np.nextafter(ratio_high, ratio_low)
            )
            counts = histogram2d(
                energy[in_range],
                accepted_ratio,
                bins=(self.energy_bins, self.ratio_bins),
                range=(self.energy_range, self.ratio_range),
            )
            self.matrix += counts.astype(np.uint64, copy=False)

        old = self.statistics
        self.statistics = PsdStatistics(
            received=old.received + received,
            accepted=old.accepted + accepted,
            zero_total=old.zero_total + zero_total,
            outside_range=old.outside_range + outside_range,
        )
        return self.statistics

    @property
    def energy_edges(self) -> np.ndarray:
        return np.linspace(*self.energy_range, self.energy_bins + 1)

    @property
    def energy_centers(self) -> np.ndarray:
        edges = self.energy_edges
        return cast(np.ndarray, (edges[:-1] + edges[1:]) * 0.5)

    @property
    def ratio_edges(self) -> np.ndarray:
        return np.linspace(*self.ratio_range, self.ratio_bins + 1)

    @property
    def ratio_centers(self) -> np.ndarray:
        edges = self.ratio_edges
        return cast(np.ndarray, (edges[:-1] + edges[1:]) * 0.5)

    def ratio_projection(self, energy_region: tuple[float, float]) -> np.ndarray:
        """Return ratio counts inside the selected displayed-energy region."""
        low, high = sorted(energy_region)
        if high <= low:
            return np.zeros(self.ratio_bins, dtype=np.uint64)
        edges = self.energy_edges
        first = int(np.clip(np.searchsorted(edges, low, side="right") - 1, 0, self.energy_bins))
        last = int(np.clip(np.searchsorted(edges, high, side="left"), 0, self.energy_bins))
        if last <= first:
            return np.zeros(self.ratio_bins, dtype=np.uint64)
        return np.asarray(
            self.matrix[first:last].sum(axis=0, dtype=np.uint64),
            dtype=np.uint64,
        )

    def energy_projections(self, cut: float) -> tuple[np.ndarray, np.ndarray]:
        """Return energy spectra below and above the ratio cut."""
        split = int(np.clip(np.searchsorted(self.ratio_edges, cut), 0, self.ratio_bins))
        below = np.asarray(
            self.matrix[:, :split].sum(axis=1, dtype=np.uint64),
            dtype=np.uint64,
        )
        above = np.asarray(
            self.matrix[:, split:].sum(axis=1, dtype=np.uint64),
            dtype=np.uint64,
        )
        return below, above
