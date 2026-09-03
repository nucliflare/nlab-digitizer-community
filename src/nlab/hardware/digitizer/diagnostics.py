from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalDiagnosticReading:
    """One channel-independent digitizer diagnostic shown in the Global tab."""

    key: str
    label: str
    value: int | float | bool
    unit: str = ""
    precision: int = 0
    healthy: bool | None = None
