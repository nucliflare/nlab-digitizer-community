"""Explicit, provisional mapping from offline gates to MCA panel timings."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from nlab.hardware.digitizer.mca import MCA_PARAMETER_SPECS, McaGateTiming, MCAParam
from nlab.hardware.digitizer.scope import RangeSpec


@dataclass(frozen=True)
class WaveformGateSource:
    baseline_end: int
    gate_start: int
    short_end: int
    long_end: int
    sample_period_ns: float

    def propose(self, cc_enabled: bool) -> GateTimingProposal:
        period = Decimal(str(self.sample_period_ns))
        if not period.is_finite() or period <= 0:
            raise ValueError("Sample period must be finite and positive")
        if not 0 <= self.baseline_end <= self.gate_start < self.short_end <= self.long_end:
            raise ValueError("Gates must satisfy 0 ≤ baseline end ≤ start < short end ≤ long end")
        start, short, end = (period * index for index in
                             (self.gate_start, self.short_end, self.long_end))
        baseline_end = period * self.baseline_end
        desired = (start - baseline_end, end - start, short - start)
        rounded = []
        for param, value in zip((MCAParam.PRETRIGGER_SAMPLES, MCAParam.FRAME_SAMPLES,
                                 MCAParam.CC_TIME), desired, strict=True):
            spec = MCA_PARAMETER_SPECS[param]
            assert isinstance(spec, RangeSpec)
            minimum, maximum, step = map(lambda x: Decimal(str(x)),
                                         (spec.min_val, spec.max_val, spec.step))
            if not minimum <= value <= maximum:
                raise ValueError(f"{param.name}: {value} ns outside {minimum}–{maximum} ns")
            actual = minimum + ((value - minimum) / step).to_integral_value(
                rounding=ROUND_CEILING,
            ) * step
            if actual > maximum:
                raise ValueError(f"{param.name}: rounded value exceeds {maximum} ns")
            rounded.append(int(actual))
        timing = McaGateTiming(rounded[0], rounded[1], rounded[2], cc_enabled)
        timing.validate()
        # Anchor the preview to the baseline region's right boundary. Rounding
        # the baseline-to-green gap upward moves the candidate green marker right.
        effective_start = baseline_end + timing.pretrigger_ns
        return GateTimingProposal(
            timing, tuple(float(value) for value in desired),
            (float(effective_start), float(effective_start + timing.cc_time_ns),
             float(effective_start + timing.frame_ns)),
        )


@dataclass(frozen=True)
class GateTimingProposal:
    timing: McaGateTiming
    desired_ns: tuple[float, ...]
    boundaries_ns: tuple[float, float, float]
