"""Readable frequency axes for spectrum plots.

In log mode pyqtgraph labels every minor tick in the SI-scaled unit
("0.007 0.008 0.009 0.01", "5·10¹6·10¹…") and the labels overlap. A log axis
here gets explicit ticks instead: labelled 1-2-5 values per decade with their
own units ("100 Hz", "2 kHz", "5 kHz") and unlabelled minor ticks for the grid.
pyqtgraph still drops the 2 and 5 labels where they would overlap.
"""

from __future__ import annotations

import math
from typing import Any

from nlab.analysis.waveform_spectrum import format_frequency

_DECADES = range(-4, 10)  # 0.1 mHz .. 1 GHz covers every supported fs and delta_f


def log_frequency_ticks() -> list[list[tuple[float, str]]]:
    """pyqtgraph tick levels in log10(Hz): decades, then 2 and 5, then minor ticks."""
    decades: list[tuple[float, str]] = []
    halves: list[tuple[float, str]] = []
    minor: list[tuple[float, str]] = []
    for decade in _DECADES:
        base = 10.0**decade
        decades.append((float(decade), format_frequency(base)))
        halves.extend((math.log10(m * base), format_frequency(m * base)) for m in (2, 5))
        minor.extend((math.log10(m * base), "") for m in (3, 4, 6, 7, 8, 9))
    return [decades, halves, minor]


_LOG_TICKS = log_frequency_ticks()


def set_frequency_axis(plot: Any, *, log: bool) -> None:
    """Label a PlotItem's bottom axis for a linear or logarithmic frequency scale."""
    axis = plot.getAxis("bottom")
    if log:
        axis.setTicks(_LOG_TICKS)
        axis.enableAutoSIPrefix(False)
        plot.setLabel("bottom", "Frequency")
    else:
        axis.setTicks(None)
        axis.enableAutoSIPrefix(True)
        plot.setLabel("bottom", "Frequency", units="Hz")
