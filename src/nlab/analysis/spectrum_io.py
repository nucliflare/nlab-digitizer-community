"""Spectrum import/export adapters used by the MCA analysis workbench."""

from __future__ import annotations

import csv
import json
import re
import struct
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np

from nlab.analysis.spectrum import Spectrum, SpectrumAxisUnit


def load_spectrum(path: Path) -> Spectrum:
    """Load one supported spectrum file based on its extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        return load_csv_spectrum(path)
    if suffix == ".txt3":
        return load_caen_text_spectrum(path)
    if suffix == ".spe":
        return load_spe_spectrum(path)
    if suffix == ".wdm":
        return load_wdm_spectrum(path)
    if suffix == ".root":
        spectra = load_root_spectra(path)
        if len(spectra) != 1:
            raise ValueError(
                f"ROOT file contains {len(spectra)} energy spectra; "
                "use load_spectra() to import all of them"
            )
        return spectra[0]
    raise ValueError(f"unsupported spectrum file extension: {path.suffix or '(none)'}")


def load_spectra(path: Path) -> list[Spectrum]:
    """Load every spectrum represented by one supported file."""
    path = Path(path)
    if path.suffix.lower() == ".root":
        return load_root_spectra(path)
    return [load_spectrum(path)]


def load_csv_spectrum(path: Path) -> Spectrum:
    """Load native or ordinary delimited channel/count spectrum text."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    metadata: dict[str, object] = {"path": str(path), "format": "CSV"}
    rows: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            key, separator, raw_value = stripped[1:].strip().partition("=")
            if separator and key:
                metadata[key.strip()] = _parse_metadata_value(raw_value.strip())
            continue
        rows.append(line)
    if not rows:
        raise ValueError("CSV spectrum contains no data rows")
    embedded_metadata = metadata.pop("metadata_json", None)
    if isinstance(embedded_metadata, Mapping):
        metadata = {
            **{str(key): value for key, value in embedded_metadata.items()},
            **metadata,
        }

    sample = "\n".join(rows[:10])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = None
    parsed = (
        [line.split() for line in rows]
        if delimiter is None and any(len(line.split()) > 1 for line in rows)
        else [
            list(map(str.strip, row))
            for row in csv.reader(rows, delimiter=delimiter or ",")
        ]
    )
    width = len(parsed[0])
    if width == 0 or any(len(row) != width for row in parsed):
        raise ValueError("CSV spectrum rows have inconsistent column counts")

    header = parsed[0] if any(not _is_number(value) for value in parsed[0]) else None
    data_rows = parsed[1:] if header is not None else parsed
    if not data_rows:
        raise ValueError("CSV spectrum contains a header but no data")
    try:
        data = np.asarray([[float(value) for value in row] for row in data_rows])
    except ValueError as exc:
        raise ValueError("CSV spectrum data must be numeric") from exc
    if not np.all(np.isfinite(data)):
        raise ValueError("CSV spectrum data contains non-finite values")

    names = [_normalise_column_name(value) for value in header] if header else []
    counts_index = _column_index(names, {"count", "counts", "intensity", "cps"})
    channel_index = _column_index(names, {"channel", "channels", "bin", "bins"})
    energy_index = _column_index(names, {"energy", "energykev", "kev"})
    variance_index = _column_index(names, {"variance", "variances"})
    if header is None:
        if width == 1:
            counts_index = 0
        elif width == 2:
            channel_index, counts_index = 0, 1
        else:
            channel_index, energy_index, counts_index = 0, 1, 2
    elif counts_index is None:
        raise ValueError("CSV spectrum header needs a counts column")
    assert counts_index is not None

    counts = data[:, counts_index]
    if channel_index is not None:
        x = data[:, channel_index]
        axis_unit: SpectrumAxisUnit = "channel"
        energy = data[:, energy_index] if energy_index is not None else None
    elif energy_index is not None:
        x = data[:, energy_index]
        axis_unit = "keV"
        energy = None
    else:
        x = np.arange(len(counts), dtype=np.float64)
        axis_unit = "channel"
        energy = None
    variances = data[:, variance_index] if variance_index is not None else np.clip(counts, 0, None)
    saved_poisson = metadata.pop("poisson_counts", None)
    poisson = (
        bool(saved_poisson)
        if isinstance(saved_poisson, bool)
        else variance_index is None and (not names or names[counts_index] != "cps")
    ) and bool(np.all(counts >= 0))
    raw_history = metadata.pop("history_json", ())
    history = (
        tuple(str(value) for value in raw_history)
        if isinstance(raw_history, Sequence) and not isinstance(raw_history, str | bytes)
        else ()
    )
    saved_label = metadata.pop("label", None)
    return Spectrum.create(
        label=str(saved_label) if isinstance(saved_label, str) else path.stem,
        x=x,
        counts=counts,
        variances=variances,
        axis_unit=axis_unit,
        source="csv",
        metadata=metadata,
        history=history,
        energy_kev=energy,
        poisson_counts=poisson,
    )


def load_spe_spectrum(path: Path) -> Spectrum:
    """Load a Maestro/ORTEC-style ASCII SPE spectrum."""
    path = Path(path)
    sections = _spe_sections(path.read_text(encoding="utf-8-sig", errors="replace"))
    data_lines = sections.get("DATA")
    if not data_lines:
        raise ValueError("SPE spectrum has no $DATA section")
    limits = np.fromstring(data_lines[0], sep=" ", dtype=np.int64)
    if limits.size != 2 or limits[1] < limits[0]:
        raise ValueError("SPE $DATA section needs inclusive first and last channels")
    first, last = map(int, limits)
    counts = np.fromstring(" ".join(data_lines[1:]), sep=" ", dtype=np.float64)
    expected = last - first + 1
    if len(counts) != expected:
        raise ValueError(f"SPE $DATA declares {expected} bins but contains {len(counts)} counts")
    channels = np.arange(first, last + 1, dtype=np.float64)

    metadata: dict[str, object] = {"path": str(path), "format": "SPE ASCII"}
    measurement = np.fromstring(" ".join(sections.get("MEAS_TIM", [])), sep=" ")
    if measurement.size >= 1 and measurement[0] > 0:
        metadata["live_time_s"] = float(measurement[0])
    if measurement.size >= 2 and measurement[1] > 0:
        metadata["elapsed_s"] = float(measurement[1])
    calibration = np.fromstring(" ".join(sections.get("ENER_FIT", [])), sep=" ")
    energy = None
    if calibration.size >= 2:
        candidate = np.polynomial.polynomial.polyval(channels, calibration)
        if np.all(np.isfinite(candidate)) and np.all(np.diff(candidate) > 0):
            energy = np.asarray(candidate, dtype=np.float64)
            metadata["energy_calibration_coefficients_kev"] = calibration.tolist()
    label_lines = sections.get("SPEC_ID", [])
    label = label_lines[0].strip() if label_lines and label_lines[0].strip() else path.stem
    return Spectrum.create(
        label=label,
        x=channels,
        counts=counts,
        axis_unit="channel",
        source="spe",
        metadata=metadata,
        energy_kev=energy,
        poisson_counts=bool(np.all(counts >= 0)),
    )


def load_caen_text_spectrum(path: Path) -> Spectrum:
    """Load the calibrated ASCII spectrum exported by CAEN software."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if len(lines) < 4:
        raise ValueError("CAEN text spectrum needs calibration, timing, and data rows")

    calibration_coefficients, energy_unit = _parse_caen_calibration(lines[0])
    elapsed_s = _parse_caen_duration(lines[1], field="RealTime")
    live_time_s = _parse_caen_duration(lines[2], field="LiveTime")
    rows: list[np.ndarray] = []
    for line_number, line in enumerate(lines[3:], start=4):
        if not line.strip():
            continue
        try:
            row = np.asarray(
                [float(value.replace(",", ".")) for value in line.split()],
                dtype=np.float64,
            )
        except ValueError as exc:
            raise ValueError(
                f"CAEN text spectrum row {line_number} contains non-numeric data"
            ) from exc
        rows.append(row)
    if not rows:
        raise ValueError("CAEN text spectrum contains no data rows")
    width = len(rows[0])
    if width not in {2, 3} or any(len(row) != width for row in rows):
        raise ValueError("CAEN text spectrum rows need channel/count[/energy] columns")
    data = np.vstack(rows)
    if not np.all(np.isfinite(data)):
        raise ValueError("CAEN text spectrum contains non-finite values")
    channels = data[:, 0]
    counts = data[:, 1]
    if np.any(np.diff(channels) <= 0):
        raise ValueError("CAEN text spectrum channels must be strictly increasing")
    if np.any(counts < 0):
        raise ValueError("CAEN text spectrum counts cannot be negative")

    energy_factor = _energy_to_kev_factor(energy_unit)
    energy: np.ndarray | None = None
    if width == 3 and energy_factor is not None:
        energy_candidate = np.asarray(data[:, 2] * energy_factor, dtype=np.float64)
        if np.all(np.diff(energy_candidate) > 0):
            energy = energy_candidate
    elif width == 2 and energy_factor is not None:
        coefficients = np.asarray(calibration_coefficients, dtype=np.float64) * energy_factor
        energy_candidate = np.asarray(
            np.polynomial.polynomial.polyval(channels, coefficients),
            dtype=np.float64,
        )
        if np.all(np.diff(energy_candidate) > 0):
            energy = energy_candidate

    metadata: dict[str, object] = {
        "path": str(path),
        "format": "CAEN ASCII spectrum",
        "elapsed_s": elapsed_s,
        "live_time_s": live_time_s,
        "caen_energy_calibration_coefficients": calibration_coefficients,
        "energy_unit": energy_unit,
    }
    if energy_factor is not None:
        metadata["energy_calibration_coefficients_kev"] = [
            coefficient * energy_factor for coefficient in calibration_coefficients
        ]
    return Spectrum.create(
        label=path.stem,
        x=channels,
        counts=counts,
        axis_unit="channel",
        source="caen_text",
        metadata=metadata,
        energy_kev=energy,
        poisson_counts=True,
    )


def load_root_spectra(path: Path) -> list[Spectrum]:
    """Load every one-dimensional histogram from a ROOT Energy directory."""
    import uproot

    path = Path(path)
    spectra: list[Spectrum] = []
    with uproot.open(path) as root_file:
        classes = root_file.classnames(recursive=True)
        energy_keys = [
            key
            for key, class_name in classes.items()
            if key.split(";", 1)[0].startswith("Energy/")
            and str(class_name).startswith("TH1")
        ]
        for key in energy_keys:
            histogram = root_file[key]
            spectra.append(
                _root_histogram_spectrum(
                    path,
                    key=key,
                    histogram=histogram,
                    root_file=root_file,
                    classes=classes,
                )
            )
    if spectra:
        return spectra
    raise ValueError("ROOT file contains no one-dimensional histograms in an Energy directory")


def load_wdm_spectrum(path: Path) -> Spectrum:
    """Load a legacy Tukan MCA binary spectrum.

    The layout follows Tukan8k v2.2.3.2 help, topic ``Formaty plikow z
    widmem``: big-endian version code, little-endian channel count, one
    little-endian 32-bit count per channel, then a variable metadata trailer.
    The calibration and ROI extension following the documented fixed metadata
    has no published field layout and is deliberately retained only as a byte
    count rather than guessed.
    """
    path = Path(path)
    payload = path.read_bytes()
    if len(payload) < 6:
        raise ValueError("Tukan WDM file is shorter than its six-byte header")
    version_code = struct.unpack_from(">I", payload, 0)[0]
    channel_count = struct.unpack_from("<H", payload, 4)[0]
    if version_code == 0:
        raise ValueError("Tukan WDM version code cannot be zero")
    if channel_count == 0:
        raise ValueError("Tukan WDM channel count cannot be zero")
    data_end = 6 + channel_count * 4
    if len(payload) < data_end:
        available = max((len(payload) - 6) // 4, 0)
        raise ValueError(
            f"Tukan WDM declares {channel_count} channels but contains only "
            f"{available} complete count values"
        )
    counts = np.frombuffer(payload, dtype="<i4", count=channel_count, offset=6)
    if np.any(counts < 0):
        raise ValueError("Tukan WDM channel counts cannot be negative")
    metadata: dict[str, object] = {
        "path": str(path),
        "format": "Tukan WDM",
        "wdm_version_code": version_code,
        "wdm_version_hex": f"0x{version_code:08X}",
        "channel_count": channel_count,
    }
    trailer = payload[data_end:]
    if trailer:
        metadata.update(_parse_wdm_trailer(trailer, version_code=version_code))
    spectrum_name = metadata.get("spectrum_name")
    label = path.stem
    if (
        isinstance(spectrum_name, str)
        and spectrum_name.strip()
        and spectrum_name.strip().casefold() != path.stem.casefold()
    ):
        label = f"{path.stem} ({spectrum_name.strip()})"
    return Spectrum.create(
        label=label,
        x=np.arange(channel_count, dtype=np.float64),
        counts=counts,
        axis_unit="channel",
        source="wdm",
        metadata=metadata,
        poisson_counts=True,
    )


def export_spectrum_csv(path: Path, spectrum: Spectrum) -> None:
    """Export one spectrum without overwriting an existing file."""
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="") as output:
        output.write("# nlab_mca_spectrum=1\n")
        output.write(f"# label={json.dumps(spectrum.label)}\n")
        output.write(f"# axis_unit={spectrum.axis_unit}\n")
        output.write(f"# source={spectrum.source}\n")
        output.write(f"# poisson_counts={json.dumps(spectrum.poisson_counts)}\n")
        output.write(f"# metadata_json={json.dumps(spectrum.metadata, default=str)}\n")
        output.write(f"# history_json={json.dumps(spectrum.history)}\n")
        writer = csv.writer(output)
        if spectrum.axis_unit == "channel" and spectrum.energy_kev is not None:
            writer.writerow(("channel", "energy_kev", "counts", "variance"))
            writer.writerows(
                zip(
                    spectrum.x,
                    spectrum.energy_kev,
                    spectrum.counts,
                    spectrum.variances,
                    strict=True,
                )
            )
        else:
            coordinate = "energy_kev" if spectrum.axis_unit == "keV" else "channel"
            writer.writerow((coordinate, "counts", "variance"))
            writer.writerows(
                zip(spectrum.x, spectrum.counts, spectrum.variances, strict=True)
            )


def _parse_caen_calibration(line: str) -> tuple[list[float], str]:
    fields: dict[str, str] = {}
    for part in line.split(";"):
        name, separator, value = part.partition("=")
        if separator:
            fields[name.strip().casefold()] = value.strip()
    missing = [name for name in ("c0", "c1", "c2", "unit") if name not in fields]
    if missing:
        raise ValueError(
            "CAEN text spectrum calibration header is missing " + ", ".join(missing)
        )
    try:
        coefficients = [float(fields[name].replace(",", ".")) for name in ("c0", "c1", "c2")]
    except ValueError as exc:
        raise ValueError("CAEN text spectrum calibration coefficients must be numeric") from exc
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("CAEN text spectrum calibration coefficients must be finite")
    return coefficients, fields["unit"]


def _parse_caen_duration(line: str, *, field: str) -> float:
    name, separator, raw_value = line.partition("=")
    if not separator or name.strip().casefold() != field.casefold():
        raise ValueError(f"CAEN text spectrum needs a {field} header")
    match = re.fullmatch(
        r"\s*(?P<hours>\d+):(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2}(?:[.,]\d+)?)\s*",
        raw_value,
    )
    if match is None:
        raise ValueError(f"CAEN text spectrum {field} must use hours:minutes:seconds")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds").replace(",", "."))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"CAEN text spectrum {field} has an invalid duration")
    return hours * 3600.0 + minutes * 60.0 + seconds


def _root_histogram_spectrum(
    path: Path,
    *,
    key: str,
    histogram: Any,
    root_file: Any,
    classes: Mapping[str, str],
) -> Spectrum:
    raw_values, raw_edges = histogram.to_numpy(flow=False)
    counts = np.asarray(raw_values, dtype=np.float64)
    edges = np.asarray(raw_edges, dtype=np.float64)
    if counts.ndim != 1 or edges.shape != (len(counts) + 1,):
        raise ValueError(f"ROOT object {key!r} is not a regular one-dimensional histogram")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError(f"ROOT energy histogram {key!r} contains invalid counts")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"ROOT energy histogram {key!r} contains invalid bin edges")

    axis_title = str(histogram.axis().member("fTitle"))
    centers = (edges[:-1] + edges[1:]) / 2.0
    energy_factor = _energy_to_kev_factor(axis_title)
    if energy_factor is not None:
        x = centers * energy_factor
        axis_unit: SpectrumAxisUnit = "keV"
    else:
        rounded_centers = np.rint(centers)
        rounded_lower = np.rint(edges[:-1])
        if np.allclose(centers, rounded_centers, rtol=0, atol=1e-9):
            x = rounded_centers
        elif np.allclose(edges[:-1], rounded_lower, rtol=0, atol=1e-9):
            x = rounded_lower
        else:
            x = centers
        axis_unit = "channel"

    raw_variances = histogram.variances(flow=False)
    variances = (
        np.clip(counts, 0, None)
        if raw_variances is None
        else np.asarray(raw_variances, dtype=np.float64)
    )
    flow = np.asarray(histogram.values(flow=True), dtype=np.float64)
    name = str(histogram.name)
    channel_match = re.search(r"CH(?P<channel>\d+)(?:@|$)", name, flags=re.IGNORECASE)
    channel = int(channel_match.group("channel")) if channel_match is not None else None
    processing_match = re.match(r"_(?P<code>[FR])_", name, flags=re.IGNORECASE)
    metadata: dict[str, object] = {
        "path": str(path),
        "format": "ROOT TH1 energy spectrum",
        "root_object": key.split(";", 1)[0],
        "root_class": str(histogram.classname),
        "root_axis_title": axis_title,
        "underflow_counts": float(flow[0]),
        "overflow_counts": float(flow[-1]),
    }
    if processing_match is not None:
        processing_code = processing_match.group("code").upper()
        metadata["caen_processing_code"] = processing_code
        metadata["caen_processing"] = "filtered" if processing_code == "F" else "raw"
    entries = _root_numeric_member(histogram, "fEntries")
    if entries is not None:
        metadata["root_entries"] = entries
    if channel is not None:
        metadata["channel"] = channel
        if "@" in name:
            metadata["device"] = name.split("@", 1)[1]
        for metadata_name, object_name in (
            ("elapsed_s", f"RealTime_{channel}"),
            ("live_time_s", f"LiveTime_{channel}"),
        ):
            milliseconds = _root_time_milliseconds(root_file, object_name)
            if milliseconds is not None and milliseconds >= 0:
                metadata[metadata_name] = milliseconds / 1000.0
        calibration_prefix = f"Energy/Calibration_{channel};"
        calibration_class = next(
            (
                class_name
                for class_key, class_name in classes.items()
                if class_key.startswith(calibration_prefix)
            ),
            None,
        )
        if calibration_class is not None:
            metadata["root_calibration_class"] = str(calibration_class)

    return Spectrum.create(
        label=f"{path.stem} ({name})",
        x=x,
        counts=counts,
        variances=variances,
        axis_unit=axis_unit,
        source="root",
        metadata=metadata,
        poisson_counts=bool(np.allclose(variances, counts, rtol=1e-10, atol=1e-12)),
    )


def _root_numeric_member(root_object: Any, name: str) -> float | None:
    try:
        value = root_object.member(name)
    except (AttributeError, KeyError):
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _root_time_milliseconds(root_file: Any, name: str) -> float | None:
    try:
        time_object = root_file[name]
    except KeyError:
        return None
    return _root_numeric_member(time_object, "fMilliSec")


def _energy_to_kev_factor(unit_or_title: str) -> float | None:
    normalized = re.sub(r"[^a-z]", "", unit_or_title.casefold())
    if "mev" in normalized:
        return 1000.0
    if "kev" in normalized:
        return 1.0
    if normalized == "ev" or normalized.endswith("energyev"):
        return 0.001
    return None


def _spe_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("$") and ":" in line:
            name, _, remainder = line[1:].partition(":")
            current = sections.setdefault(name.strip().upper(), [])
            if remainder.strip():
                current.append(remainder.strip())
        elif current is not None and line:
            current.append(line)
    return sections


def _parse_wdm_trailer(trailer: bytes, *, version_code: int) -> dict[str, object]:
    metadata: dict[str, object] = {"wdm_trailer_bytes": len(trailer)}
    position = 0

    def remaining() -> int:
        return len(trailer) - position

    def read_struct(format_string: str) -> int | float | None:
        nonlocal position
        size = struct.calcsize(format_string)
        if remaining() < size:
            return None
        value = struct.unpack_from(format_string, trailer, position)[0]
        position += size
        return cast(int | float, value)

    def read_pascal_string(field: str) -> bool:
        nonlocal position
        if remaining() < 1:
            return False
        length = trailer[position]
        position += 1
        if remaining() < length:
            raise ValueError(f"Tukan WDM {field} string is truncated")
        raw_value = trailer[position : position + length]
        position += length
        metadata[field] = raw_value.decode("cp1250", errors="replace")
        return True

    integrity_code = read_struct(">I")
    if integrity_code is None:
        metadata["wdm_unparsed_extension_bytes"] = remaining()
        return metadata
    metadata["wdm_integrity_code"] = int(integrity_code)
    integrity_matches = int(integrity_code) == version_code
    metadata["wdm_integrity_matches_version"] = integrity_matches
    if int(integrity_code) not in {0, version_code}:
        raise ValueError(
            "Tukan WDM integrity code does not match its version code "
            f"(0x{int(integrity_code):08X} != 0x{version_code:08X})"
        )

    mass = read_struct("<d")
    if mass is None:
        metadata["wdm_unparsed_extension_bytes"] = remaining()
        return metadata
    if np.isfinite(float(mass)):
        metadata["sample_mass"] = float(mass)
    mass_unit = read_struct("<B")
    if mass_unit is None:
        return metadata
    metadata["sample_mass_unit_code"] = int(mass_unit)

    for field in (
        "analyzer_type",
        "analyzer_serial_number",
        "geometry_name",
        "spectrum_name",
        "spectrum_description",
    ):
        if not read_pascal_string(field):
            return metadata

    started = read_struct("<d")
    if started is None:
        return metadata
    started_value = float(started)
    metadata["tukan_tdatetime"] = started_value
    if np.isfinite(started_value):
        try:
            started_local = datetime(1899, 12, 30) + timedelta(days=started_value)
            metadata["started_local"] = started_local.isoformat(timespec="milliseconds")
        except (OverflowError, ValueError):
            metadata["started_local_invalid"] = True

    elapsed = read_struct("<i")
    live = read_struct("<i")
    if elapsed is not None and int(elapsed) >= 0:
        metadata["elapsed_s"] = int(elapsed)
    if live is not None and int(live) >= 0:
        metadata["live_time_s"] = int(live)
    calibration_code = read_struct("<I")
    if calibration_code is not None:
        metadata["calibration_indicator_code"] = int(calibration_code)
    if remaining():
        metadata["wdm_unparsed_extension_bytes"] = remaining()
    return metadata


def _normalise_column_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _column_index(names: list[str], choices: set[str]) -> int | None:
    return next((index for index, name in enumerate(names) if name in choices), None)


def _is_number(value: str) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except ValueError:
        return False


def _parse_metadata_value(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return float(value)
        except ValueError:
            return value
