"""Operator-selected wiring and producer provenance; never infer these from IIO IDs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .iio_listmode import VDPP_ZC_CALC_SCHEMA


@dataclass(frozen=True)
class AcquisitionProfile:
    dac_output: int | None = None
    firmware_id: str = ""
    record_schema: str | None = None

    def __post_init__(self) -> None:
        if self.dac_output is not None and (
            type(self.dac_output) is not int or self.dac_output not in (0, 1)
        ):
            raise ValueError("Verified DAC output must be 0/A, 1/B, or absent")
        if not isinstance(self.firmware_id, str):
            raise ValueError("Firmware identity must be text")
        if self.record_schema not in (None, VDPP_ZC_CALC_SCHEMA):
            raise ValueError("Unsupported producer record schema")
        if self.record_schema and not self.firmware_id.strip():
            raise ValueError("Select a firmware/image identity before selecting an event decoder")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> AcquisitionProfile:
        if set(values) - {"dac_output", "firmware_id", "record_schema"}:
            raise ValueError("Unknown acquisition profile field")
        # Dataclass validation intentionally rejects rather than coercing fields.
        return cls(**values)  # type: ignore[arg-type]
