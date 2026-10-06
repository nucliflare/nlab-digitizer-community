"""Build the pinned offline radionuclide photon catalogue.

The source is the official NNDC ENSDF API.  ENSDF gamma intensities are often
relative, so this builder combines the gamma intensity (RI), photon
normalization (NR), and decay branching ratio (BR) before storing photons per
100 parent decays.  The generated SQLite file is committed and shipped with
the application; the GUI never needs network access.

Run from the repository root::

    python scripts/build_radionuclide_catalog.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "src" / "nlab" / "data" / "radionuclide_lines.sqlite3"
API_BASE = "https://www.nndc.bnl.gov/ensdf-api"
EXPECTED_DATA_GENERATED_AT = "2026-08-10T15:57:13Z"
CATALOG_ID = "nlab-ensdf-decay-photon-lines"
CATALOG_VERSION = 2
SCHEMA_VERSION = 1
DECAY_DOCUMENT_TYPES = (
    "alphaDecay",
    "betaDecay",
    "protonDecay",
    "delayedParticleDecay",
    "generalDecay",
)
SOURCE_REFERENCE = "NNDC Evaluated Nuclear Structure Data File (ENSDF)"
SOURCE_URL = "https://www.nndc.bnl.gov/ensdf/"
USER_AGENT = "nlab-radionuclide-catalog-builder/1.0"


@dataclass(frozen=True)
class DecaySource:
    dataset_id: int
    document_type: str
    dataset_name: str
    cutoff_date: str | None
    decay_mode: str
    parent: dict[str, Any]
    daughter: dict[str, Any]


@dataclass(frozen=True)
class AbsoluteProbability:
    value: float
    uncertainty: float | None
    qualifier: str
    recommended: bool


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-release",
        default=EXPECTED_DATA_GENERATED_AT,
        help="refuse to build from an unexpected ENSDF dataGeneratedAt value",
    )
    return parser.parse_args()


def _request(path: str, query: dict[str, object] | None = None) -> urllib.request.Request:
    url = f"{API_BASE}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _get_json(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(_request(path), timeout=60) as response:  # noqa: S310
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"unexpected JSON response from {path}")
    return cast(dict[str, Any], value)


def _iter_ndjson(path: str, query: dict[str, object]) -> Iterator[dict[str, Any]]:
    with urllib.request.urlopen(_request(path, query), timeout=180) as response:  # noqa: S310
        for line_number, raw_line in enumerate(response, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid NDJSON at {path}, line {line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"unexpected NDJSON value at {path}, line {line_number}")
            yield value


def _quantity_value(quantity: object, *, default: float | None = None) -> float | None:
    if not isinstance(quantity, dict):
        return default
    value = quantity.get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return float(value)


def _uncertainty_type(quantity: object) -> str:
    if not isinstance(quantity, dict):
        return "missing"
    uncertainty = quantity.get("uncertainty")
    if not isinstance(uncertainty, dict):
        return "missing"
    value = uncertainty.get("type")
    return str(value) if value is not None else "missing"


def _symmetric_uncertainty(quantity: object) -> float | None:
    kind = _uncertainty_type(quantity)
    if kind == "none":
        return 0.0
    if kind != "symmetric" or not isinstance(quantity, dict):
        return None
    uncertainty = quantity.get("uncertainty")
    if not isinstance(uncertainty, dict):
        return None
    value = uncertainty.get("value")
    if not isinstance(value, (int, float)) or value < 0 or not math.isfinite(value):
        return None
    return float(value)


def _absolute_probability(
    intensity: object,
    normalizations: object,
    *,
    existence: str | None,
) -> AbsoluteProbability | None:
    """Convert ENSDF RI/NR/BR into photons per 100 parent decays."""
    if not isinstance(normalizations, dict):
        return None
    gamma = normalizations.get("gamma")
    if not isinstance(gamma, dict):
        return None
    relative = _quantity_value(intensity)
    gamma_factor = _quantity_value(gamma)
    if relative is None or gamma_factor is None or relative <= 0 or gamma_factor <= 0:
        return None

    factors: list[object] = [intensity, gamma]
    absolute = relative * gamma_factor
    if not bool(gamma.get("includesBranchingRatio", False)):
        branching = normalizations.get("branchingRatio")
        branching_factor = _quantity_value(branching)
        if branching_factor is None or branching_factor <= 0:
            return None
        absolute *= branching_factor
        factors.append(branching)

    if not math.isfinite(absolute) or absolute <= 0 or absolute > 200:
        return None

    uncertainty_kinds = {_uncertainty_type(factor) for factor in factors}
    disqualifying = {"limit", "range", "missing"}
    approximate = "approximation" in uncertainty_kinds
    qualifier = "approximation" if approximate else "evaluated"
    if uncertainty_kinds & disqualifying:
        qualifier = "bounded"

    relative_variance = 0.0
    uncertainty_available = True
    for factor in factors:
        factor_value = _quantity_value(factor)
        factor_uncertainty = _symmetric_uncertainty(factor)
        if factor_value is None or factor_uncertainty is None:
            uncertainty_available = False
            break
        relative_variance += (factor_uncertainty / factor_value) ** 2
    uncertainty = absolute * math.sqrt(relative_variance) if uncertainty_available else None

    recommended = not bool(uncertainty_kinds & disqualifying) and existence != "uncertain"
    return AbsoluteProbability(absolute, uncertainty, qualifier, recommended)


def _endpoint_key(endpoint: dict[str, Any]) -> tuple[int, int, str, float]:
    z = int(endpoint["z"])
    a = int(endpoint["a"])
    symbol = str(endpoint["elementSymbol"]).capitalize()
    energy = _quantity_value(endpoint.get("levelEnergy"), default=0.0) or 0.0
    if abs(energy) < 1e-9:
        energy = 0.0
    return z, a, symbol, round(energy, 9)


def _same_parent_state(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_energy = _quantity_value(left.get("levelEnergy"), default=0.0) or 0.0
    right_energy = _quantity_value(right.get("levelEnergy"), default=0.0) or 0.0
    difference = abs(left_energy - right_energy)
    if difference <= 1e-3:
        return True

    left_half_life = _quantity_value(left.get("halfLifeSeconds"))
    right_half_life = _quantity_value(right.get("halfLifeSeconds"))
    if left_half_life is None or right_half_life is None:
        return difference <= 0.02
    scale = max(left_half_life, right_half_life)
    same_half_life = scale > 0 and abs(left_half_life - right_half_life) / scale <= 0.01
    return same_half_life and difference <= 0.1


def _state_names(decays: list[DecaySource]) -> dict[tuple[int, int, str, float], str]:
    parent_endpoints: dict[tuple[int, int, str, float], dict[str, Any]] = {}
    for decay in decays:
        parent_endpoints.setdefault(_endpoint_key(decay.parent), decay.parent)
    grouped: dict[tuple[int, int, str], list[tuple[int, int, str, float]]] = defaultdict(list)
    for key in parent_endpoints:
        grouped[key[:3]].append(key)

    # ENSDF identifies a parent state by excitation energy rather than an m/m2
    # suffix. Cluster small evaluation-to-evaluation shifts for the same state,
    # then assign m labels in increasing excitation-energy order. Only states
    # that actually appear as decay parents participate in the numbering;
    # ordinary excited daughter levels must not turn Am-242m into Am-242m2.
    result: dict[tuple[int, int, str, float], str] = {}
    for (z, a, symbol), keys in grouped.items():
        excited_clusters: list[list[tuple[int, int, str, float]]] = []
        for key in sorted(keys, key=lambda value: value[3]):
            if key[3] == 0:
                result[key] = f"{symbol}-{a}"
                continue
            if excited_clusters and _same_parent_state(
                parent_endpoints[excited_clusters[-1][0]],
                parent_endpoints[key],
            ):
                excited_clusters[-1].append(key)
            else:
                excited_clusters.append([key])
        for index, cluster in enumerate(excited_clusters, start=1):
            suffix = "m" if index == 1 else f"m{index}"
            for key in cluster:
                result[key] = f"{symbol}-{a}{suffix}"

    for decay in decays:
        daughter_key = _endpoint_key(decay.daughter)
        if daughter_key in result:
            continue
        z, a, symbol, energy = daughter_key
        if energy == 0:
            result[daughter_key] = f"{symbol}-{a}"
            continue
        candidates = [
            key
            for key in parent_endpoints
            if key[:3] == (z, a, symbol) and key[3] > 0 and abs(key[3] - energy) <= 0.1
        ]
        result[daughter_key] = (
            result[min(candidates, key=lambda key: abs(key[3] - energy))]
            if candidates
            else f"{symbol}-{a}m"
        )
    return result


def _download_decays() -> list[DecaySource]:
    decays: list[DecaySource] = []
    for document_type in DECAY_DOCUMENT_TYPES:
        print(f"Downloading {document_type} decay records...", flush=True)
        records = _iter_ndjson(
            "/decays/export",
            {
                "document-type": document_type,
                "fields": "decayMode,parent,daughter,datasetSummary",
            },
        )
        for record in records:
            summary = record.get("datasetSummary")
            parent = record.get("parent")
            daughter = record.get("daughter")
            if (
                not isinstance(summary, dict)
                or not isinstance(parent, dict)
                or not isinstance(daughter, dict)
            ):
                continue
            decays.append(
                DecaySource(
                    dataset_id=int(summary["id"]),
                    document_type=str(summary["documentType"]),
                    dataset_name=str(summary["datasetName"]),
                    cutoff_date=(
                        str(summary["cutoffDate"]) if summary.get("cutoffDate") else None
                    ),
                    decay_mode=str(record["decayMode"]),
                    parent=parent,
                    daughter=daughter,
                )
            )
    return sorted(decays, key=lambda value: (value.dataset_id, value.decay_mode))


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA page_size = 4096;
        PRAGMA foreign_keys = ON;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE nuclides (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            z INTEGER NOT NULL,
            a INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            state_index INTEGER NOT NULL,
            state_energy_kev REAL NOT NULL,
            half_life_seconds REAL
        );

        CREATE TABLE decays (
            id INTEGER PRIMARY KEY,
            dataset_id INTEGER NOT NULL,
            parent_id INTEGER NOT NULL REFERENCES nuclides(id),
            daughter_name TEXT NOT NULL,
            decay_mode TEXT NOT NULL,
            document_type TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            cutoff_date TEXT,
            source_url TEXT NOT NULL
        );

        CREATE TABLE lines (
            id INTEGER PRIMARY KEY,
            decay_id INTEGER NOT NULL REFERENCES decays(id),
            source_gamma_id INTEGER,
            energy_kev REAL NOT NULL CHECK (energy_kev > 0),
            energy_uncertainty_kev REAL,
            probability_percent REAL NOT NULL CHECK (
                probability_percent > 0 AND probability_percent <= 200
            ),
            probability_uncertainty_percent REAL,
            probability_qualifier TEXT NOT NULL,
            relative_intensity REAL,
            radiation_kind TEXT NOT NULL CHECK (radiation_kind IN ('gamma', 'annihilation')),
            existence TEXT,
            multipolarity TEXT,
            recommended INTEGER NOT NULL CHECK (recommended IN (0, 1)),
            note TEXT NOT NULL DEFAULT ''
        );

        CREATE INDEX lines_decay_probability
            ON lines(decay_id, recommended, probability_percent, energy_kev);
        CREATE INDEX lines_candidate_search
            ON lines(recommended, energy_kev, probability_percent, decay_id);
        CREATE INDEX decays_parent ON decays(parent_id);
        CREATE INDEX nuclides_order ON nuclides(z, a, state_index);
        """
    )


def _insert_metadata(
    connection: sqlite3.Connection,
    release: dict[str, Any],
    *,
    decay_count: int,
    line_count: int,
    recommended_line_count: int,
) -> None:
    values = {
        "catalog_id": CATALOG_ID,
        "catalog_version": str(CATALOG_VERSION),
        "schema_version": str(SCHEMA_VERSION),
        "source_reference": SOURCE_REFERENCE,
        "source_url": SOURCE_URL,
        "source_api": API_BASE,
        "source_api_version": str(release["apiVersion"]),
        "source_schema_version": str(release["schemaVersion"]),
        "source_data_generated_at": str(release["dataGeneratedAt"]),
        "source_dataset_count": str(release["datasetCount"]),
        "decay_count": str(decay_count),
        "line_count": str(line_count),
        "recommended_line_count": str(recommended_line_count),
        "probability_definition": "photons per 100 parent decays; ENSDF RI * NR * BR",
    }
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        sorted(values.items()),
    )


def _build_database(output: Path, release: dict[str, Any], decays: list[DecaySource]) -> None:
    states = _state_names(decays)
    decay_by_dataset: dict[int, list[DecaySource]] = defaultdict(list)
    for decay in decays:
        decay_by_dataset[decay.dataset_id].append(decay)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    if temporary.exists():
        temporary.unlink()

    connection = sqlite3.connect(temporary)
    try:
        _create_schema(connection)

        parent_groups: dict[str, list[tuple[int, int, str, float]]] = defaultdict(list)
        for decay in decays:
            key = _endpoint_key(decay.parent)
            if key not in parent_groups[states[key]]:
                parent_groups[states[key]].append(key)
        parents = sorted(
            parent_groups,
            key=lambda name: (
                parent_groups[name][0][0],
                parent_groups[name][0][1],
                min(key[3] for key in parent_groups[name]),
            ),
        )
        parent_ids: dict[tuple[int, int, str, float], int] = {}
        for parent_id, name in enumerate(parents, start=1):
            keys = parent_groups[name]
            matching = [decay for decay in decays if _endpoint_key(decay.parent) in keys]
            half_lives = [
                value
                for decay in matching
                if (value := _quantity_value(decay.parent.get("halfLifeSeconds"))) is not None
            ]
            z, a, symbol, _energy = keys[0]
            energy = sum(key[3] for key in keys) / len(keys)
            suffix = name.removeprefix(f"{symbol}-{a}")
            state_index = 0 if not suffix else 1 if suffix == "m" else int(suffix[1:])
            connection.execute(
                """
                INSERT INTO nuclides(
                    id, name, z, a, symbol, state_index, state_energy_kev, half_life_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parent_id,
                    name,
                    z,
                    a,
                    symbol,
                    state_index,
                    energy,
                    half_lives[0] if half_lives else None,
                ),
            )
            for key in keys:
                parent_ids[key] = parent_id

        decay_ids: dict[tuple[int, int], int] = {}
        for decay_id, decay in enumerate(decays, start=1):
            siblings = decay_by_dataset[decay.dataset_id]
            sibling_index = siblings.index(decay)
            decay_ids[(decay.dataset_id, sibling_index)] = decay_id
            connection.execute(
                """
                INSERT INTO decays(
                    id, dataset_id, parent_id, daughter_name, decay_mode, document_type,
                    dataset_name, cutoff_date, source_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decay_id,
                    decay.dataset_id,
                    parent_ids[_endpoint_key(decay.parent)],
                    states[_endpoint_key(decay.daughter)],
                    decay.decay_mode,
                    decay.document_type,
                    decay.dataset_name,
                    decay.cutoff_date,
                    f"{API_BASE}/datasets/{decay.dataset_id}",
                ),
            )

        line_id = 0
        recommended_count = 0
        skipped_without_decay = 0
        for document_type in DECAY_DOCUMENT_TYPES:
            print(f"Downloading normalized {document_type} gamma records...", flush=True)
            gammas = _iter_ndjson(
                "/gammas/export",
                {
                    "document-type": document_type,
                    # The public API's absolute-intensity presence filter only
                    # covers an optional stored field and excludes valid decay
                    # records such as Co-60. Export every gamma with RI and
                    # derive the absolute value from the decay normalization.
                    "has-intensity": "true",
                    "fields": (
                        "id,datasetId,energy,intensity,existence,multipolarityDisplay,"
                        "datasetSummary"
                    ),
                },
            )
            for gamma in gammas:
                dataset_id = int(gamma["datasetId"])
                sources = decay_by_dataset.get(dataset_id, ())
                if not sources:
                    skipped_without_decay += 1
                    continue
                gamma_energy = _quantity_value(gamma.get("energy"))
                if gamma_energy is None or gamma_energy <= 0:
                    continue
                existence = str(gamma["existence"]) if gamma.get("existence") else None
                for sibling_index, decay in enumerate(sources):
                    probability = _absolute_probability(
                        gamma.get("intensity"),
                        decay.parent.get("normalizations"),
                        existence=existence,
                    )
                    if probability is None:
                        continue
                    line_id += 1
                    recommended_count += int(probability.recommended)
                    connection.execute(
                        """
                        INSERT INTO lines(
                            id, decay_id, source_gamma_id, energy_kev,
                            energy_uncertainty_kev, probability_percent,
                            probability_uncertainty_percent, probability_qualifier,
                            relative_intensity, radiation_kind, existence, multipolarity,
                            recommended, note
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'gamma', ?, ?, ?, '')
                        """,
                        (
                            line_id,
                            decay_ids[(dataset_id, sibling_index)],
                            int(gamma["id"]),
                            gamma_energy,
                            _symmetric_uncertainty(gamma.get("energy")),
                            probability.value,
                            probability.uncertainty,
                            probability.qualifier,
                            _quantity_value(gamma.get("intensity")),
                            existence,
                            (
                                str(gamma["multipolarityDisplay"])
                                if gamma.get("multipolarityDisplay")
                                else None
                            ),
                            int(probability.recommended),
                        ),
                    )

        # ENSDF gamma records do not represent positron-annihilation photons.
        # Retain the evaluated DDEP/IRDFF-II Na-22 reference used by the original
        # catalogue as an explicitly attributed supplement.
        na22 = connection.execute(
            "SELECT id FROM nuclides WHERE name = 'Na-22'"
        ).fetchone()
        na22_decay = connection.execute(
            """
            SELECT d.id FROM decays d
            JOIN nuclides n ON n.id = d.parent_id
            WHERE n.name = 'Na-22'
            ORDER BY d.id LIMIT 1
            """
        ).fetchone()
        if na22 is not None and na22_decay is not None:
            line_id += 1
            recommended_count += 1
            connection.execute(
                """
                INSERT INTO lines(
                    id, decay_id, energy_kev, probability_percent,
                    probability_uncertainty_percent, probability_qualifier,
                    radiation_kind, recommended, note
                ) VALUES (?, ?, 511.0, 180.76, 0.04, 'evaluated', 'annihilation', 1, ?)
                """,
                (
                    line_id,
                    int(na22_decay[0]),
                    "DDEP/IRDFF-II two-photon positron-annihilation emission; may exceed 100%.",
                ),
            )

        current_release = _get_json("/meta/release")
        if current_release.get("dataGeneratedAt") != release.get("dataGeneratedAt"):
            raise RuntimeError("ENSDF release changed while the catalogue was being generated")

        _insert_metadata(
            connection,
            release,
            decay_count=len(decays),
            line_count=line_id,
            recommended_line_count=recommended_count,
        )
        connection.commit()
        connection.execute("ANALYZE")
        connection.execute("VACUUM")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError(f"SQLite integrity check failed: {integrity!r}")
        if skipped_without_decay:
            print(
                f"Skipped {skipped_without_decay} gamma record(s) without a decay edge.",
                file=sys.stderr,
            )
    except BaseException:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        connection.close()

    os.replace(temporary, output)
    print(
        f"Wrote {output} ({output.stat().st_size / (1024 * 1024):.2f} MiB, "
        f"{line_id} lines, {recommended_count} recommended)."
    )


def main() -> None:
    options = _parse_arguments()
    release = _get_json("/meta/release")
    generated_at = release.get("dataGeneratedAt")
    if generated_at != options.expected_release:
        raise RuntimeError(
            "ENSDF release changed: expected "
            f"{options.expected_release!r}, received {generated_at!r}. Review the new release "
            "before updating EXPECTED_DATA_GENERATED_AT."
        )
    decays = _download_decays()
    _build_database(options.output.resolve(), release, decays)


if __name__ == "__main__":
    main()
