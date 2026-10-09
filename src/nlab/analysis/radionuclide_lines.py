"""Pinned offline decay-photon references for calibrated spectrum overlays.

The shipped SQLite catalogue is generated from one identified NNDC ENSDF API
release. ENSDF relative gamma intensities are normalized to photons per 100
parent decays by the build script. No network access occurs at runtime.

The catalogue does not predict measured peak height. Detector efficiency,
attenuation, geometry, coincidence summing, and dead time remain outside this
model.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal, cast

import numpy as np

from nlab import data as data_package
from nlab.analysis.spectrum import Spectrum

RadiationKind = Literal["gamma", "annihilation"]

CATALOG_ID = "nlab-ensdf-decay-photon-lines"
CATALOG_VERSION = 2
CATALOG_REFERENCE = "NNDC ENSDF; Na-22 annihilation supplement from DDEP/IRDFF-II"
CATALOG_REFERENCE_URL = "https://www.nndc.bnl.gov/ensdf/"
_DDEP_URL = "https://www-nds.iaea.org/IRDFF/Recomm-DecayData-IRDFF-II.pdf"
_DATABASE_NAME = "radionuclide_lines.sqlite3"


@dataclass(frozen=True)
class RadionuclideLine:
    """One evaluated photon line emitted following decay of a parent nuclide."""

    parent: str
    energy_kev: float
    emission_probability_percent: float
    kind: RadiationKind = "gamma"
    daughter: str | None = None
    energy_uncertainty_kev: float | None = None
    probability_uncertainty_percent: float | None = None
    note: str = ""
    probability_qualifier: str = "evaluated"
    decay_mode: str | None = None
    source_dataset: str | None = None
    source_dataset_id: int | None = None
    source_cutoff_date: str | None = None
    source_url: str | None = None
    multipolarity: str | None = None
    existence: str | None = None

    def __post_init__(self) -> None:
        if canonical_nuclide(self.parent) != self.parent:
            raise ValueError(f"parent nuclide must be canonical: {self.parent!r}")
        if self.daughter is not None and canonical_nuclide(self.daughter) != self.daughter:
            raise ValueError(f"daughter nuclide must be canonical: {self.daughter!r}")
        if not np.isfinite(self.energy_kev) or self.energy_kev <= 0:
            raise ValueError("reference-line energy must be positive and finite")
        probability = self.emission_probability_percent
        if not np.isfinite(probability) or probability <= 0 or probability > 200:
            raise ValueError("photon emission probability must be within (0, 200]")
        for name, uncertainty in (
            ("energy uncertainty", self.energy_uncertainty_kev),
            ("probability uncertainty", self.probability_uncertainty_percent),
        ):
            if uncertainty is not None and (
                not np.isfinite(uncertainty) or uncertainty < 0
            ):
                raise ValueError(f"{name} must be non-negative and finite")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe, provenance-ready representation."""
        return asdict(self)


@dataclass(frozen=True)
class RadionuclideRecord:
    """Reference lines associated with one decaying parent nuclide."""

    nuclide: str
    lines: tuple[RadionuclideLine, ...]
    reference: str
    reference_url: str
    state_energy_kev: float = 0.0
    half_life_seconds: float | None = None

    def __post_init__(self) -> None:
        if canonical_nuclide(self.nuclide) != self.nuclide:
            raise ValueError(f"nuclide must be canonical: {self.nuclide!r}")
        if not self.lines:
            raise ValueError("a radionuclide record must contain at least one line")
        if any(line.parent != self.nuclide for line in self.lines):
            raise ValueError("all record lines must use the record parent nuclide")
        energies = [line.energy_kev for line in self.lines]
        if energies != sorted(energies):
            raise ValueError("reference lines must be sorted by energy")


@dataclass(frozen=True)
class RadionuclideSummary:
    """Small list-entry representation that avoids loading every line."""

    nuclide: str
    line_count: int
    state_energy_kev: float
    half_life_seconds: float | None


@dataclass(frozen=True)
class RadionuclideCandidate:
    """One parent nuclide with matching lines inside an energy interval."""

    nuclide: str
    matching_line_count: int
    strongest_probability_percent: float


class BuiltinRadionuclideCatalog:
    """Read-only access to the versioned SQLite catalogue shipped with nlab."""

    def __init__(self, database_path: Path | None = None) -> None:
        if database_path is None:
            resource = resources.files(data_package).joinpath(_DATABASE_NAME)
            database_path = Path(str(resource))
        self._database_path = database_path.resolve()
        self._metadata = self._read_metadata()
        if self._metadata.get("catalog_id") != CATALOG_ID:
            raise RuntimeError("bundled radionuclide catalogue has an unexpected identity")
        if int(self._metadata.get("catalog_version", "-1")) != CATALOG_VERSION:
            raise RuntimeError("bundled radionuclide catalogue version does not match the reader")
        self.catalog_id = CATALOG_ID
        self.version = CATALOG_VERSION
        self.reference = CATALOG_REFERENCE
        self.reference_url = CATALOG_REFERENCE_URL
        self._summaries = self._read_summaries()
        self._nuclide_names = frozenset(summary.nuclide for summary in self._summaries)

    @property
    def metadata(self) -> dict[str, str]:
        """Return a copy of pinned source and catalogue metadata."""
        return dict(self._metadata)

    @property
    def database_path(self) -> Path:
        return self._database_path

    def _connect(self) -> sqlite3.Connection:
        if not self._database_path.is_file():
            raise RuntimeError(f"bundled radionuclide catalogue is missing: {self._database_path}")
        uri = f"{self._database_path.as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _read_metadata(self) -> dict[str, str]:
        with self._connect() as connection:
            return {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }

    def _read_summaries(self) -> tuple[RadionuclideSummary, ...]:
        query = """
            SELECT n.name, n.state_energy_kev, n.half_life_seconds, COUNT(l.id) AS line_count
            FROM nuclides n
            JOIN decays d ON d.parent_id = n.id
            JOIN lines l ON l.decay_id = d.id AND l.recommended = 1
            GROUP BY n.id
            ORDER BY n.z, n.a, n.state_index
        """
        with self._connect() as connection:
            return tuple(
                RadionuclideSummary(
                    nuclide=str(row["name"]),
                    line_count=int(row["line_count"]),
                    state_energy_kev=float(row["state_energy_kev"]),
                    half_life_seconds=(
                        float(row["half_life_seconds"])
                        if row["half_life_seconds"] is not None
                        else None
                    ),
                )
                for row in connection.execute(query)
            )

    def summaries(self) -> tuple[RadionuclideSummary, ...]:
        return self._summaries

    def nuclides(self) -> tuple[str, ...]:
        return tuple(summary.nuclide for summary in self._summaries)

    @lru_cache(maxsize=128)
    def candidates(
        self,
        energy_min_kev: float,
        energy_max_kev: float,
        *,
        minimum_probability_percent: float = 0.0,
    ) -> tuple[RadionuclideCandidate, ...]:
        """Return parents with recommended lines in an inclusive energy interval."""
        values = (energy_min_kev, energy_max_kev, minimum_probability_percent)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("candidate-search bounds must be finite")
        if energy_min_kev < 0 or energy_max_kev < energy_min_kev:
            raise ValueError("candidate-search energy bounds are invalid")
        if not 0 <= minimum_probability_percent <= 200:
            raise ValueError("minimum probability must be within [0, 200]")
        query = """
            SELECT
                n.name,
                COUNT(l.id) AS matching_line_count,
                MAX(l.probability_percent) AS strongest_probability_percent
            FROM lines l
            JOIN decays d ON d.id = l.decay_id
            JOIN nuclides n ON n.id = d.parent_id
            WHERE
                l.recommended = 1
                AND l.energy_kev BETWEEN ? AND ?
                AND l.probability_percent >= ?
            GROUP BY n.id
            ORDER BY matching_line_count DESC, strongest_probability_percent DESC, n.z, n.a,
                n.state_index
        """
        with self._connect() as connection:
            return tuple(
                RadionuclideCandidate(
                    nuclide=str(row["name"]),
                    matching_line_count=int(row["matching_line_count"]),
                    strongest_probability_percent=float(row["strongest_probability_percent"]),
                )
                for row in connection.execute(
                    query,
                    (energy_min_kev, energy_max_kev, minimum_probability_percent),
                )
            )

    def record(self, nuclide: str) -> RadionuclideRecord:
        canonical = canonical_nuclide(nuclide)
        lines = self._lines(canonical, minimum_probability_percent=0.0)
        if not lines:
            message = f"nuclide {canonical} is not in {self.catalog_id} v{self.version}"
            raise KeyError(message)
        summary = next(item for item in self._summaries if item.nuclide == canonical)
        return RadionuclideRecord(
            nuclide=canonical,
            lines=lines,
            reference=self.reference,
            reference_url=self.reference_url,
            state_energy_kev=summary.state_energy_kev,
            half_life_seconds=summary.half_life_seconds,
        )

    def lines(
        self,
        nuclide: str,
        *,
        minimum_probability_percent: float = 0.0,
    ) -> tuple[RadionuclideLine, ...]:
        if not np.isfinite(minimum_probability_percent) or not (
            0 <= minimum_probability_percent <= 200
        ):
            raise ValueError("minimum probability must be within [0, 200]")
        canonical = canonical_nuclide(nuclide)
        lines = self._lines(
            canonical,
            minimum_probability_percent=minimum_probability_percent,
        )
        if canonical not in self._nuclide_names:
            message = f"nuclide {canonical} is not in {self.catalog_id} v{self.version}"
            raise KeyError(message)
        return lines

    @lru_cache(maxsize=256)
    def _lines(
        self,
        nuclide: str,
        *,
        minimum_probability_percent: float,
    ) -> tuple[RadionuclideLine, ...]:
        query = """
            SELECT
                n.name AS parent_name,
                d.daughter_name,
                d.decay_mode,
                d.dataset_name,
                d.dataset_id,
                d.cutoff_date,
                d.source_url,
                l.energy_kev,
                l.energy_uncertainty_kev,
                l.probability_percent,
                l.probability_uncertainty_percent,
                l.probability_qualifier,
                l.radiation_kind,
                l.note,
                l.multipolarity,
                l.existence
            FROM lines l
            JOIN decays d ON d.id = l.decay_id
            JOIN nuclides n ON n.id = d.parent_id
            WHERE n.name = ? AND l.recommended = 1 AND l.probability_percent >= ?
            ORDER BY l.energy_kev, l.id
        """
        with self._connect() as connection:
            return tuple(
                self._line_from_row(row)
                for row in connection.execute(
                    query,
                    (nuclide, minimum_probability_percent),
                )
            )

    @staticmethod
    def _line_from_row(row: sqlite3.Row) -> RadionuclideLine:
        return RadionuclideLine(
            parent=str(row["parent_name"]),
            energy_kev=float(row["energy_kev"]),
            emission_probability_percent=float(row["probability_percent"]),
            kind=cast(RadiationKind, str(row["radiation_kind"])),
            daughter=str(row["daughter_name"]),
            energy_uncertainty_kev=(
                float(row["energy_uncertainty_kev"])
                if row["energy_uncertainty_kev"] is not None
                else None
            ),
            probability_uncertainty_percent=(
                float(row["probability_uncertainty_percent"])
                if row["probability_uncertainty_percent"] is not None
                else None
            ),
            note=str(row["note"]),
            probability_qualifier=str(row["probability_qualifier"]),
            decay_mode=str(row["decay_mode"]),
            source_dataset=str(row["dataset_name"]),
            source_dataset_id=int(row["dataset_id"]),
            source_cutoff_date=(str(row["cutoff_date"]) if row["cutoff_date"] else None),
            source_url=str(row["source_url"]),
            multipolarity=(str(row["multipolarity"]) if row["multipolarity"] else None),
            existence=str(row["existence"]) if row["existence"] else None,
        )

    def selection_document(
        self,
        nuclides: tuple[str, ...],
        *,
        minimum_probability_percent: float,
    ) -> dict[str, object]:
        """Snapshot an overlay selection for a reproducible analysis export."""
        canonical = tuple(canonical_nuclide(nuclide) for nuclide in nuclides)
        return {
            "catalog_id": self.catalog_id,
            "catalog_version": self.version,
            "catalog_reference": self.reference,
            "source_data_generated_at": self._metadata["source_data_generated_at"],
            "minimum_probability_percent": minimum_probability_percent,
            "nuclides": list(canonical),
            "lines": [
                self.line_document(line)
                for nuclide in canonical
                for line in self.lines(
                    nuclide,
                    minimum_probability_percent=minimum_probability_percent,
                )
            ],
        }

    def line_document(self, line: RadionuclideLine) -> dict[str, object]:
        """Return one line together with exact catalogue/source attribution."""
        reference = (
            "DDEP values reproduced by IAEA IRDFF-II"
            if line.kind == "annihilation"
            else self._metadata["source_reference"]
        )
        reference_url = _DDEP_URL if line.kind == "annihilation" else line.source_url
        return {
            **line.to_dict(),
            "reference": reference,
            "reference_url": reference_url or self.reference_url,
            "source_data_generated_at": self._metadata["source_data_generated_at"],
        }


def canonical_nuclide(value: str) -> str:
    """Normalize forms such as ``Na22``, ``22Na``, ``Na-22``, and ``Am242m``."""
    compact = re.sub(r"[\s_-]+", "", str(value))
    match = re.fullmatch(r"([A-Za-z]{1,3})(\d+)(m\d*)?", compact, re.IGNORECASE)
    if match is not None:
        symbol, mass, state = match.groups()
    else:
        match = re.fullmatch(r"(\d+)([A-Za-z]{1,3})(m\d*)?", compact, re.IGNORECASE)
        if match is None:
            raise ValueError(f"invalid nuclide name: {value!r}")
        mass, symbol, state = match.groups()
    suffix = (state or "").lower()
    return f"{symbol.capitalize()}-{int(mass)}{suffix}"


def has_usable_energy_axis(spectrum: Spectrum | None) -> bool:
    """Return whether reference energies can be located on this spectrum."""
    if spectrum is None or bool(spectrum.metadata.get("calibration_stale", False)):
        return False
    return spectrum.axis_unit == "keV" or spectrum.energy_kev is not None


def coordinate_for_energy(spectrum: Spectrum, energy_kev: float) -> float | None:
    """Map a reference energy onto the selected spectrum's displayed x coordinate."""
    if not has_usable_energy_axis(spectrum) or not np.isfinite(energy_kev):
        return None
    calibrated = spectrum.x if spectrum.axis_unit == "keV" else spectrum.energy_kev
    assert calibrated is not None
    if energy_kev < calibrated[0] or energy_kev > calibrated[-1]:
        return None
    return float(np.interp(energy_kev, calibrated, spectrum.x))


def energy_range_for_coordinates(
    spectrum: Spectrum,
    coordinate_a: float,
    coordinate_b: float,
) -> tuple[float, float] | None:
    """Map a coordinate interval onto the spectrum's current calibrated energy axis."""
    if not has_usable_energy_axis(spectrum):
        return None
    low, high = sorted((float(coordinate_a), float(coordinate_b)))
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    low = max(low, float(spectrum.x[0]))
    high = min(high, float(spectrum.x[-1]))
    if low > high:
        return None
    calibrated = spectrum.x if spectrum.axis_unit == "keV" else spectrum.energy_kev
    assert calibrated is not None
    energies = np.interp((low, high), spectrum.x, calibrated)
    energy_low, energy_high = map(float, energies)
    if energy_high < 0:
        return None
    return max(0.0, energy_low), energy_high


BUILTIN_RADIONUCLIDE_CATALOG = BuiltinRadionuclideCatalog()


__all__ = [
    "BUILTIN_RADIONUCLIDE_CATALOG",
    "CATALOG_ID",
    "CATALOG_REFERENCE",
    "CATALOG_REFERENCE_URL",
    "CATALOG_VERSION",
    "BuiltinRadionuclideCatalog",
    "RadionuclideLine",
    "RadionuclideCandidate",
    "RadionuclideRecord",
    "RadionuclideSummary",
    "canonical_nuclide",
    "coordinate_for_energy",
    "energy_range_for_coordinates",
    "has_usable_energy_axis",
]
