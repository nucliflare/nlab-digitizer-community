from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TypedDict, Union

import numpy as np

from .backends.base import ScopeBackend

# ---------------------------------------------------------------------------
# Parameter spec types — used for validation and GUI settings generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RangeSpec:
    min_val: int | float
    max_val: int | float
    step: int | float
    default: int | float

    def validate(self, val: int | float, name: str = "") -> None:
        if not (self.min_val <= val <= self.max_val):
            raise ValueError(f"{name}: {val} out of range [{self.min_val}, {self.max_val}]")
        if self.step and isinstance(val, int) and isinstance(self.step, int):
            if (val - self.min_val) % self.step != 0:
                raise ValueError(f"{name}: {val} not aligned to step {self.step}")


@dataclass(frozen=True)
class ListSpec:
    items: tuple[bool | int | float | str, ...]
    default: bool | int | float | str

    def validate(self, val: bool | int | float | str, name: str = "") -> None:
        if val not in self.items:
            raise ValueError(f"{name}: '{val}' not in {self.items}")


ParameterSpec = Union[RangeSpec, ListSpec]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TriggerMode(IntEnum):
    ANY_ABOVE = 0
    ANY_BELOW = 1
    FALLING_EDGE = 2
    RISING_EDGE = 3
    TIMED = 4


class ScopeParam(IntEnum):
    # Values equal the corresponding vdpp_config.json "scope" section widget id
    TRIGGER_LEVEL = 1
    PRETRIGGER_SAMPLES = 2
    FRAME_SAMPLES = 3
    EDGE_MODE = 4
    DAC_VALUE = 5
    # Internal — no JSON widget id (scope ids 6/7 are software DMA config, not this flag)
    DMA_ENABLED = 100


# ---------------------------------------------------------------------------
# Parameter specs (hardware limits)
# ---------------------------------------------------------------------------

PARAMETER_SPECS: dict[ScopeParam, ParameterSpec] = {
    # int16_t (hw_def: MIN–MAX, step 1)
    ScopeParam.TRIGGER_LEVEL: RangeSpec(min_val=-32768, max_val=32767, step=1, default=0),
    # uint16_t (hw_def: 0–2046, step 2)
    ScopeParam.PRETRIGGER_SAMPLES: RangeSpec(min_val=0, max_val=2040, step=8, default=32),
    # uint16_t (hw_def: 0–16382, step 2)
    ScopeParam.FRAME_SAMPLES: RangeSpec(min_val=0, max_val=16382, step=8, default=1024),
    ScopeParam.EDGE_MODE: ListSpec(
        items=tuple(m.name for m in TriggerMode),
        default=TriggerMode.ANY_BELOW,
    ),
    # 0-1023: confirmed live against vdpp_afe_dac's baseline_available
    # (min=0, step=1, max=1023) -- 1024 is rejected by the driver with
    # ERANGE, so the spec must match the hardware's real range, not just
    # the round-number ceiling the old gRPC-only comment assumed.
    ScopeParam.DAC_VALUE: RangeSpec(min_val=0, max_val=1023, step=1, default=512),
    ScopeParam.DMA_ENABLED: ListSpec(items=(False, True), default=False),
}


# ---------------------------------------------------------------------------
# TypedDict for GUI settings consumers
# ---------------------------------------------------------------------------


class ScopeSettingEntry(TypedDict):
    id: int
    value: int | str | bool


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class Scope:
    """High-level scope interface. All validation lives here; the backend is
    a thin hardware translator with no business logic."""

    specs: dict[ScopeParam, ParameterSpec] = PARAMETER_SPECS

    def __init__(self, backend: ScopeBackend) -> None:
        self._b = backend

    # ---- read-only device info ----

    def get_ip_version(self) -> int:
        return self._b.get_ip_version()

    def get_mem_frame_size(self) -> int:
        return self._b.get_mem_frame_size()

    # ---- arm / disarm ----

    def get_enable(self) -> bool:
        return self._b.get_enable()

    def set_enable(self, val: bool) -> None:
        self._b.set_enable(val)

    def get_dma_enable(self) -> bool:
        return self._b.get_dma_enable()

    def set_dma_enable(self, val: bool) -> None:
        self._b.set_dma_enable(val)

    def start(self) -> None:
        self._b.set_enable(True)

    def stop(self) -> None:
        self._b.set_enable(False)

    # ---- trigger ----

    def get_trigger_level(self) -> int:
        return self._b.get_trigger_level()

    def set_trigger_level(self, val: int) -> None:
        PARAMETER_SPECS[ScopeParam.TRIGGER_LEVEL].validate(val, "trigger_level")
        self._b.set_trigger_level(val)

    def get_trigger_mode(self) -> TriggerMode:
        return TriggerMode(self._b.get_edge())

    def set_trigger_mode(self, val: TriggerMode) -> None:
        self._b.set_edge(int(val))

    # ---- timing (in samples) ----

    def get_pretrigger_samples(self) -> int:
        return self._b.get_pretrigger_samples()

    def set_pretrigger_samples(self, val: int) -> None:
        PARAMETER_SPECS[ScopeParam.PRETRIGGER_SAMPLES].validate(val, "pretrigger_samples")
        self._b.set_pretrigger_samples(val)

    def get_frame_samples(self) -> int:
        return self._b.get_frame_samples()

    def set_frame_samples(self, val: int) -> None:
        PARAMETER_SPECS[ScopeParam.FRAME_SAMPLES].validate(val, "frame_samples")
        self._b.set_frame_samples(val)

    # ---- DC offset ----

    def get_dac_value(self) -> int:
        return self._b.get_dac_value()

    def set_dac_value(self, val: int) -> None:
        PARAMETER_SPECS[ScopeParam.DAC_VALUE].validate(val, "dac_value")
        self._b.set_dac_value(val)

    # ---- data acquisition ----

    def acquire_frame(self) -> np.ndarray:
        """Read one captured frame. Returns int16 array of shape (n_samples,)."""
        return self._b.read_frame()

    # ---- DMA fault recovery ----
    # Extension methods, only meaningful for backends that latch a fault on
    # a genuine DMA session error (currently the IIO backend -- see
    # IIODigitizerBackend._refill_dma_buffer()'s docstring). Backends
    # without this concept (e.g. the gRPC backend) are no-ops here rather
    # than an AttributeError, so callers don't need an isinstance check.

    def dma_fault_is_latched(self) -> bool:
        return bool(getattr(self._b, "dma_fault_is_latched", lambda: False)())

    def acknowledge_dma_recovery(self) -> None:
        ack = getattr(self._b, "acknowledge_dma_recovery", None)
        if ack is not None:
            ack()

    # ---- GUI helpers ----

    def get_settings(self) -> list[ScopeSettingEntry]:
        """Return current hardware state as a flat list for GUI consumption.

        IDs match vdpp_config.json scope section widget ids.
        """
        return [
            ScopeSettingEntry(id=int(ScopeParam.TRIGGER_LEVEL), value=self.get_trigger_level()),
            ScopeSettingEntry(id=int(ScopeParam.PRETRIGGER_SAMPLES), value=self.get_pretrigger_samples()),
            ScopeSettingEntry(id=int(ScopeParam.FRAME_SAMPLES), value=self.get_frame_samples()),
            ScopeSettingEntry(id=int(ScopeParam.EDGE_MODE), value=self.get_trigger_mode().name),
            ScopeSettingEntry(id=int(ScopeParam.DAC_VALUE), value=self.get_dac_value()),
        ]
