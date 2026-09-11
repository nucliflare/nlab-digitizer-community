from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeAxisScale:
    """Display unit selected for a time span expressed in nanoseconds."""

    unit: str
    ns_per_unit: float

    def from_nanoseconds(self, value_ns: float) -> float:
        return value_ns / self.ns_per_unit


def time_axis_scale(duration_ns: float) -> TimeAxisScale:
    """Choose a readable SI unit for a plot covering ``duration_ns``."""

    duration_ns = abs(duration_ns)
    if duration_ns < 1_000:
        return TimeAxisScale("ns", 1.0)
    if duration_ns < 1_000_000:
        return TimeAxisScale("\N{MICRO SIGN}s", 1_000.0)
    if duration_ns < 1_000_000_000:
        return TimeAxisScale("ms", 1_000_000.0)
    return TimeAxisScale("s", 1_000_000_000.0)


def format_duration_ns(duration_ns: float) -> str:
    """Format a nanosecond duration using the same scale as the plot axis."""

    scale = time_axis_scale(duration_ns)
    value = scale.from_nanoseconds(duration_ns)
    return f"{value:g} {scale.unit}"
