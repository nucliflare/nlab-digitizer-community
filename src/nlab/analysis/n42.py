"""Secure ANSI/IEEE N42.42 spectrum validation, import, and export.

The bundled schema files are the NIST-published ANSI/IEEE N42.42-2020
artifacts.  The standard intentionally retains the 2011 XML namespace.
External entities, schemas, and location hints are never resolved.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from importlib import resources
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4
from xml.etree import ElementTree as ET

import numpy as np
import xmlschema

from nlab import _version
from nlab.analysis.spectrum import Spectrum

if TYPE_CHECKING:
    from nlab.analysis.peak_fitting import PeakFitResult, PeakFitSpec

N42_NAMESPACE = "http://physics.nist.gov/N42/2011/N42"
_Q = f"{{{N42_NAMESPACE}}}"
_MAX_XML_BYTES = 64 * 1024 * 1024
_MAX_CHANNELS = 10_000_000
_DURATION_RE = re.compile(
    r"P(?:(?P<days>\d+(?:\.\d+)?)D)?(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?"
)

ET.register_namespace("", N42_NAMESPACE)


class N42ValidationError(ValueError):
    """Raised when an N42 document fails safe parsing or conformance checks."""


@dataclass(frozen=True)
class N42ExportMetadata:
    """Instrument and detector identity written into an exported N42 document."""

    instrument_manufacturer: str = "Eastern Wall Technologies"
    instrument_model: str = "Nuclear Lab Digitizer"
    instrument_identifier: str = "Nuclear Lab Digitizer"
    instrument_description: str = "Nuclear Lab Digitizer MCA"
    instrument_class: str = "Other"
    detector_name: str = "MCA detector input"
    detector_category: str = "Other"
    detector_kind: str = "Other"
    detector_description: str = "Detector material and geometry not specified"
    measurement_class: str = "NotSpecified"


def validate_n42(path: Path) -> None:
    """Validate a local N42 document without resolving external resources."""
    _validated_root(_read_limited(Path(path)))


def load_n42_spectra(path: Path) -> list[Spectrum]:
    """Validate and load every Spectrum in an N42 RadInstrumentData document."""
    path = Path(path)
    root = _validated_root(_read_limited(path))
    calibrations = {
        element.attrib["id"]: element
        for element in root.iter(_Q + "EnergyCalibration")
        if "id" in element.attrib
    }
    detectors = {
        element.attrib["id"]: _detector_metadata(element)
        for element in root.iter(_Q + "RadDetectorInformation")
        if "id" in element.attrib
    }
    instrument = root.find(_Q + "RadInstrumentInformation")
    instrument_metadata = _instrument_metadata(instrument)
    contexts: dict[int, ET.Element] = {}
    for container_name in ("RadMeasurement", "DerivedData"):
        for container in root.iter(_Q + container_name):
            for spectrum_element in container.findall(_Q + "Spectrum"):
                contexts[id(spectrum_element)] = container

    spectra: list[Spectrum] = []
    for index, element in enumerate(root.iter(_Q + "Spectrum"), start=1):
        context = contexts.get(id(element))
        if context is None:
            raise N42ValidationError("N42 Spectrum is outside RadMeasurement or DerivedData")
        calibration_id = element.get("energyCalibrationReference", "")
        calibration = calibrations.get(calibration_id)
        if calibration is None:
            raise N42ValidationError(
                f"N42 Spectrum {element.get('id', index)!r} references missing "
                f"EnergyCalibration {calibration_id!r}"
            )
        channel_data = element.find(_Q + "ChannelData")
        assert channel_data is not None  # Required by the validated schema.
        counts = _decode_channel_data(channel_data)
        channels = np.arange(len(counts), dtype=np.float64)
        energy, calibration_metadata = _calibrated_energies(calibration, channels)

        elapsed_s = _duration_child(context, "RealTimeDuration")
        live_time_s = _duration_child(element, "LiveTimeDuration")
        detector_id = element.get("radDetectorInformationReference")
        detector = detectors.get(detector_id or "", {})
        measurement_id = context.get("id", "")
        spectrum_id = element.get("id", f"Spectrum-{index}")
        measurement_class = _child_text(context, "MeasurementClassCode")
        label_parts = [path.stem, measurement_class, str(detector.get("name", "")), spectrum_id]
        label = " — ".join(part for part in label_parts if part)
        metadata: dict[str, object] = {
            "path": str(path),
            "format": "ANSI/IEEE N42.42",
            "n42_document_uuid": root.get("n42DocUUID", ""),
            "n42_document_datetime": root.get("n42DocDateTime", ""),
            "n42_measurement_id": measurement_id,
            "n42_spectrum_id": spectrum_id,
            "n42_measurement_class": measurement_class,
            "n42_start_datetime": _child_text(context, "StartDateTime"),
            "n42_energy_calibration_id": calibration_id,
            "n42_detector_id": detector_id or "",
            "n42_instrument": instrument_metadata,
            "n42_detector": detector,
            "elapsed_s": elapsed_s,
            "live_time_s": live_time_s,
            **calibration_metadata,
        }
        spectra.append(
            Spectrum.create(
                label=label,
                x=channels,
                counts=counts,
                axis_unit="channel",
                source="n42",
                metadata=metadata,
                energy_kev=energy,
                poisson_counts=bool(
                    np.all(counts >= 0)
                    and np.allclose(counts, np.rint(counts), rtol=0, atol=1e-12)
                ),
            )
        )
    if not spectra:
        raise ValueError("N42 document contains no spectra")
    return spectra


def export_n42_spectrum(
    path: Path,
    spectrum: Spectrum,
    *,
    export_metadata: N42ExportMetadata | None = None,
) -> None:
    """Export one calibrated, timed spectrum as a conformant N42 document."""
    root, _measurement_id = _build_document(spectrum, export_metadata=export_metadata)
    _write_validated(Path(path), root)


def export_n42_peak_fit(
    path: Path,
    *,
    spectrum: Spectrum,
    spec: PeakFitSpec,
    result: PeakFitResult,
    export_metadata: N42ExportMetadata | None = None,
) -> None:
    """Export a spectrum and its fitted peak results in one N42 document."""
    root, measurement_id = _build_document(spectrum, export_metadata=export_metadata)
    analysis = ET.SubElement(
        root,
        _Q + "AnalysisResults",
        {"id": "AnalysisResults-1", "radMeasurementReferences": measurement_id},
    )
    _sub(analysis, "AnalysisStartDateTime", _utc_now())
    _sub(analysis, "AnalysisAlgorithmName", "NLab MCA constrained peak fit")
    _sub(analysis, "AnalysisAlgorithmCreatorName", "Eastern Wall Technologies")
    _sub(analysis, "AnalysisAlgorithmDescription", _fit_description(spec, result))
    version = ET.SubElement(analysis, _Q + "AnalysisAlgorithmVersion")
    _sub(version, "AnalysisAlgorithmComponentName", "nlab")
    _sub(version, "AnalysisAlgorithmComponentVersion", _version.__version__)
    _sub(analysis, "AnalysisResultStatusCode", "Success" if result.success else "Failure")
    peaks_element = ET.SubElement(analysis, _Q + "SpectrumPeakAnalysisResults")
    for peak in result.peaks:
        energy_kev = peak.energy_kev
        if energy_kev is None:
            energy_kev = spectrum.energy(peak.center)
        if energy_kev is None or not math.isfinite(energy_kev) or energy_kev <= 0:
            raise ValueError("N42 peak export requires positive calibrated peak energies")
        peak_element = ET.SubElement(peaks_element, _Q + "SpectrumPeak")
        _sub(peak_element, "SpectrumPeakEnergyValue", _number(energy_kev), units="keV")
        fwhm_kev = peak.energy_fwhm_kev
        if fwhm_kev is None:
            fwhm_kev = _energy_width(spectrum, peak.center, peak.fwhm)
        if fwhm_kev is not None and math.isfinite(fwhm_kev) and fwhm_kev > 0:
            _sub(peak_element, "SpectrumPeakFWHMValue", _number(fwhm_kev), units="keV")
        if math.isfinite(peak.area) and peak.area >= 0:
            _sub(peak_element, "SpectrumPeakNetAreaValue", _number(peak.area))
        if (
            peak.area_stderr is not None
            and math.isfinite(peak.area_stderr)
            and peak.area_stderr >= 0
        ):
            _sub(
                peak_element,
                "SpectrumPeakNetAreaUncertaintyValue",
                _number(peak.area_stderr),
            )
    _write_validated(Path(path), root)


def n42_export_metadata_from_spectrum(spectrum: Spectrum) -> N42ExportMetadata:
    """Build export identity from explicit spectrum metadata, using honest defaults."""
    raw_instrument = spectrum.metadata.get("n42_instrument", {})
    raw_detector = spectrum.metadata.get("n42_detector", {})
    instrument = raw_instrument if isinstance(raw_instrument, Mapping) else {}
    detector = raw_detector if isinstance(raw_detector, Mapping) else {}
    measurement_class = spectrum.metadata.get("n42_measurement_class", "NotSpecified")
    return N42ExportMetadata(
        instrument_manufacturer=_mapping_text(
            instrument,
            "manufacturer",
            "Eastern Wall Technologies",
        ),
        instrument_model=_mapping_text(instrument, "model", "Nuclear Lab Digitizer"),
        instrument_identifier=_mapping_text(
            instrument,
            "identifier",
            "Nuclear Lab Digitizer",
        ),
        instrument_description=_mapping_text(
            instrument,
            "description",
            "Nuclear Lab Digitizer MCA",
        ),
        instrument_class=_mapping_text(instrument, "class", "Other"),
        detector_name=_mapping_text(detector, "name", "MCA detector input"),
        detector_category=_mapping_text(detector, "category", "Other"),
        detector_kind=_mapping_text(detector, "kind", "Other"),
        detector_description=_mapping_text(
            detector,
            "description",
            "Detector material and geometry not specified",
        ),
        measurement_class=(
            str(measurement_class)
            if str(measurement_class)
            in {"Foreground", "Background", "Calibration", "IntrinsicActivity", "NotSpecified"}
            else "NotSpecified"
        ),
    )


def _schema() -> xmlschema.XMLSchema10:
    return _cached_schema()


@lru_cache(maxsize=1)
def _cached_schema() -> xmlschema.XMLSchema10:
    data = resources.files("nlab.data")
    with resources.as_file(data) as data_path:
        base_uri = (data_path / "n42.xsd").resolve().as_uri()
        addin_uri = (data_path / "N42SchemaAddin-2020.xsd").resolve().as_uri()
        return xmlschema.XMLSchema10(
            [base_uri, addin_uri],
            allow="sandbox",
            defuse="always",
            use_location_hints=False,
            use_fallback=False,
            locations={N42_NAMESPACE: base_uri},
        )


def _validated_root(payload: bytes) -> ET.Element:
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise N42ValidationError("N42 XML declarations and entities are not allowed")
    try:
        resource = xmlschema.XMLResource(payload, allow="none", defuse="always")
    except (ET.ParseError, xmlschema.XMLResourceError) as exc:
        raise N42ValidationError(f"N42 XML is not safely parseable: {exc}") from exc
    errors = list(islice(_schema().iter_errors(resource, use_location_hints=False), 11))
    if errors:
        details = "; ".join(
            f"{error.path or '/'}: {error.reason}" for error in errors[:10]
        )
        if len(errors) > 10:
            details += "; additional errors omitted"
        raise N42ValidationError(f"N42 schema validation failed: {details}")
    root = resource.root
    if root.tag != _Q + "RadInstrumentData":
        raise N42ValidationError("N42 root element must be RadInstrumentData in the N42 namespace")
    _validate_supported_semantics(root)
    return root


def _validate_supported_semantics(root: ET.Element) -> None:
    calibrations = {
        element.get("id", ""): element for element in root.iter(_Q + "EnergyCalibration")
    }
    detectors = {
        element.get("id", "") for element in root.iter(_Q + "RadDetectorInformation")
    }
    for container_name in ("RadMeasurement", "DerivedData"):
        for container in root.iter(_Q + container_name):
            real_time_s = _duration_child(container, "RealTimeDuration")
            for spectrum in container.findall(_Q + "Spectrum"):
                live_time_s = _duration_child(spectrum, "LiveTimeDuration")
                if live_time_s > real_time_s * (1.0 + 1e-12):
                    raise N42ValidationError(
                        f"Spectrum {spectrum.get('id', '(unnamed)')!r} has live time "
                        "greater than its measurement real time"
                    )
    for element in root.iter(_Q + "Spectrum"):
        spectrum_id = element.get("id", "(unnamed)")
        calibration_id = element.get("energyCalibrationReference", "")
        calibration = calibrations.get(calibration_id)
        if calibration is None:
            raise N42ValidationError(
                f"Spectrum {spectrum_id!r} does not reference an EnergyCalibration"
            )
        detector_id = element.get("radDetectorInformationReference")
        if detector_id and detector_id not in detectors:
            raise N42ValidationError(
                f"Spectrum {spectrum_id!r} references a non-detector ID {detector_id!r}"
            )
        channel_data = element.find(_Q + "ChannelData")
        if channel_data is None:
            continue
        counts = _decode_channel_data(channel_data)
        boundaries = calibration.find(_Q + "EnergyBoundaryValues")
        if boundaries is not None:
            values = _double_list(boundaries.text, field="EnergyBoundaryValues")
            if len(values) != len(counts) + 1:
                raise N42ValidationError(
                    f"Spectrum {spectrum_id!r} has {len(counts)} channels but its "
                    f"EnergyBoundaryValues has {len(values)} entries"
                )
            if np.any(np.diff(values) <= 0):
                raise N42ValidationError("EnergyBoundaryValues must be strictly increasing")
        _calibrated_energies(
            calibration,
            np.arange(len(counts), dtype=np.float64),
        )


def _read_limited(path: Path) -> bytes:
    with path.open("rb") as source:
        payload = source.read(_MAX_XML_BYTES + 1)
    if len(payload) > _MAX_XML_BYTES:
        raise N42ValidationError(
            f"N42 XML exceeds the {_MAX_XML_BYTES // (1024 * 1024)} MiB safety limit"
        )
    return payload


def _decode_channel_data(element: ET.Element) -> np.ndarray:
    encoded = _double_list(element.text, field="ChannelData")
    compression = element.get("compressionCode", "None")
    if compression in {"", "None"}:
        if len(encoded) > _MAX_CHANNELS:
            raise N42ValidationError("N42 spectrum exceeds the channel-count safety limit")
        return encoded
    if compression != "CountedZeroes":
        raise N42ValidationError(f"unsupported N42 ChannelData compression {compression!r}")
    values: list[float] = []
    index = 0
    while index < len(encoded):
        value = float(encoded[index])
        if value != 0:
            values.append(value)
            index += 1
            continue
        if index + 1 >= len(encoded):
            raise N42ValidationError("CountedZeroes ChannelData ends with a zero marker")
        run_value = float(encoded[index + 1])
        run = int(run_value)
        if run_value != run or run <= 0:
            raise N42ValidationError("CountedZeroes run lengths must be positive integers")
        if len(values) + run > _MAX_CHANNELS:
            raise N42ValidationError("N42 spectrum exceeds the channel-count safety limit")
        values.extend([0.0] * run)
        index += 2
    return np.asarray(values, dtype=np.float64)


def _calibrated_energies(
    calibration: ET.Element,
    channels: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    coefficients = calibration.find(_Q + "CoefficientValues")
    if coefficients is not None:
        values = _double_list(coefficients.text, field="CoefficientValues")
        energy = np.asarray(np.polynomial.polynomial.polyval(channels, values), dtype=np.float64)
        metadata: dict[str, object] = {
            "energy_calibration_coefficients_kev": values.tolist(),
        }
    else:
        boundaries = calibration.find(_Q + "EnergyBoundaryValues")
        if boundaries is None:
            raise N42ValidationError("EnergyCalibration has no supported calibration values")
        edges = _double_list(boundaries.text, field="EnergyBoundaryValues")
        if len(edges) != len(channels) + 1:
            raise N42ValidationError("EnergyBoundaryValues length does not match ChannelData")
        energy = (edges[:-1] + edges[1:]) / 2.0
        metadata = {"energy_calibration_boundaries_kev": edges.tolist()}
    if not np.all(np.isfinite(energy)) or (len(energy) > 1 and np.any(np.diff(energy) <= 0)):
        raise N42ValidationError("N42 energy calibration is not finite and strictly increasing")
    date = _child_text(calibration, "CalibrationDateTime")
    if date:
        metadata["energy_calibration_datetime"] = date
    return energy, metadata


def _build_document(
    spectrum: Spectrum,
    *,
    export_metadata: N42ExportMetadata | None,
) -> tuple[ET.Element, str]:
    if bool(spectrum.metadata.get("calibration_stale")):
        raise ValueError("N42 export refuses a stale energy calibration")
    if not np.all(np.isfinite(spectrum.counts)) or np.any(spectrum.counts < 0):
        raise ValueError("N42 ChannelData requires finite, non-negative counts")
    elapsed_s = _positive_metadata_number(spectrum, "elapsed_s")
    live_time_s = _positive_metadata_number(spectrum, "live_time_s")
    if elapsed_s is None or live_time_s is None:
        raise ValueError("N42 export requires positive elapsed_s and live_time_s metadata")
    if live_time_s > elapsed_s * (1.0 + 1e-9):
        raise ValueError("N42 live_time_s cannot exceed elapsed_s")
    started = spectrum.metadata.get("n42_start_datetime") or spectrum.metadata.get("created_utc")
    if not isinstance(started, str) or not started.strip():
        raise ValueError("N42 export requires a measurement start timestamp")
    metadata = export_metadata or n42_export_metadata_from_spectrum(spectrum)
    calibration_kind, calibration_values = _export_calibration(spectrum)

    root = ET.Element(
        _Q + "RadInstrumentData",
        {"n42DocUUID": str(uuid4()), "n42DocDateTime": _utc_now()},
    )
    _sub(root, "RadInstrumentDataCreatorName", "Nuclear Lab Digitizer Community Edition")
    instrument = ET.SubElement(root, _Q + "RadInstrumentInformation", {"id": "Instrument-1"})
    _sub(instrument, "RadInstrumentManufacturerName", metadata.instrument_manufacturer)
    _sub(instrument, "RadInstrumentIdentifier", metadata.instrument_identifier)
    _sub(instrument, "RadInstrumentModelName", metadata.instrument_model)
    _sub(instrument, "RadInstrumentDescription", metadata.instrument_description)
    _sub(instrument, "RadInstrumentClassCode", metadata.instrument_class)
    version = ET.SubElement(instrument, _Q + "RadInstrumentVersion")
    _sub(version, "RadInstrumentComponentName", "Software")
    _sub(version, "RadInstrumentComponentVersion", _version.__version__)

    detector = ET.SubElement(root, _Q + "RadDetectorInformation", {"id": "Detector-1"})
    _sub(detector, "RadDetectorName", metadata.detector_name)
    _sub(detector, "RadDetectorCategoryCode", metadata.detector_category)
    _sub(detector, "RadDetectorKindCode", metadata.detector_kind)
    _sub(detector, "RadDetectorDescription", metadata.detector_description)

    calibration = ET.SubElement(root, _Q + "EnergyCalibration", {"id": "EnergyCalibration-1"})
    _sub(calibration, calibration_kind, _format_values(calibration_values))
    calibration_date = _calibration_datetime(spectrum)
    if calibration_date:
        _sub(calibration, "CalibrationDateTime", calibration_date)

    measurement_id = "Measurement-1"
    measurement = ET.SubElement(root, _Q + "RadMeasurement", {"id": measurement_id})
    _sub(measurement, "MeasurementClassCode", metadata.measurement_class)
    _sub(measurement, "StartDateTime", started.strip())
    _sub(measurement, "RealTimeDuration", _duration(elapsed_s))
    spectrum_element = ET.SubElement(
        measurement,
        _Q + "Spectrum",
        {
            "id": "Spectrum-1",
            "energyCalibrationReference": "EnergyCalibration-1",
            "radDetectorInformationReference": "Detector-1",
        },
    )
    _sub(spectrum_element, "LiveTimeDuration", _duration(live_time_s))
    compressed, compression = _encode_counts(spectrum.counts)
    attributes = {"compressionCode": compression} if compression != "None" else {}
    _sub(spectrum_element, "ChannelData", _format_values(compressed), **attributes)
    return root, measurement_id


def _export_calibration(spectrum: Spectrum) -> tuple[str, np.ndarray]:
    nested = spectrum.metadata.get("energy_calibration")
    if isinstance(nested, Mapping):
        raw = nested.get("coefficients_kev")
        if isinstance(raw, list | tuple):
            coefficients = np.asarray(raw, dtype=np.float64)
            reference = nested.get("fingerprint")
            reference_binning = reference.get("binning") if isinstance(reference, Mapping) else None
            current_binning = spectrum.metadata.get("binning")
            if (
                isinstance(reference_binning, int)
                and not isinstance(reference_binning, bool)
                and isinstance(current_binning, int)
                and not isinstance(current_binning, bool)
            ):
                scale = 2.0 ** (current_binning - reference_binning)
                coefficients = coefficients * np.power(scale, np.arange(len(coefficients)))
            return _checked_coefficients(coefficients, spectrum)
    raw_coefficients = spectrum.metadata.get("energy_calibration_coefficients_kev")
    if isinstance(raw_coefficients, list | tuple):
        return _checked_coefficients(np.asarray(raw_coefficients, dtype=np.float64), spectrum)
    raw_boundaries = spectrum.metadata.get("energy_calibration_boundaries_kev")
    if isinstance(raw_boundaries, list | tuple):
        boundaries = np.asarray(raw_boundaries, dtype=np.float64)
        if boundaries.shape == (len(spectrum.counts) + 1,):
            return _checked_boundaries(boundaries, spectrum)
    energy = spectrum.x if spectrum.axis_unit == "keV" else spectrum.energy_kev
    if energy is None:
        raise ValueError("N42 export requires an applied energy calibration")
    return _boundaries_from_centers(np.asarray(energy, dtype=np.float64), spectrum)


def _boundaries_from_centers(
    centers: np.ndarray,
    spectrum: Spectrum,
) -> tuple[str, np.ndarray]:
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) < 2:
        raise ValueError("N42 energy-boundary export requires at least two spectrum bins")
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    boundaries = np.concatenate(
        (
            [centers[0] - (midpoints[0] - centers[0])],
            midpoints,
            [centers[-1] + (centers[-1] - midpoints[-1])],
        )
    )
    return _checked_boundaries(boundaries, spectrum)


def _checked_coefficients(coefficients: np.ndarray, spectrum: Spectrum) -> tuple[str, np.ndarray]:
    if not 1 <= len(coefficients) <= 3 or not np.all(np.isfinite(coefficients)):
        raise ValueError("N42 energy calibration coefficients are invalid")
    if len(coefficients) < 3:
        coefficients = np.pad(coefficients, (0, 3 - len(coefficients)))
    coordinates = np.asarray(spectrum.x, dtype=np.float64)
    if len(coordinates) < 2:
        energy = np.asarray(
            np.polynomial.polynomial.polyval(coordinates, coefficients),
            dtype=np.float64,
        )
        return _boundaries_from_centers(energy, spectrum)
    steps = np.diff(coordinates)
    if not np.allclose(steps, steps[0], rtol=1e-10, atol=1e-12):
        energy = np.asarray(
            np.polynomial.polynomial.polyval(coordinates, coefficients),
            dtype=np.float64,
        )
        return _boundaries_from_centers(energy, spectrum)
    offset = coordinates[0]
    scale = steps[0]
    c0, c1, c2 = coefficients
    coefficients = np.asarray(
        (
            c0 + c1 * offset + c2 * offset**2,
            c1 * scale + 2.0 * c2 * offset * scale,
            c2 * scale**2,
        ),
        dtype=np.float64,
    )
    energy = np.polynomial.polynomial.polyval(
        np.arange(len(coordinates), dtype=np.float64),
        coefficients,
    )
    if np.any(energy < 0) or (len(energy) > 1 and np.any(np.diff(energy) <= 0)):
        raise ValueError("N42 energy calibration must be non-negative and increasing")
    return "CoefficientValues", coefficients


def _checked_boundaries(boundaries: np.ndarray, spectrum: Spectrum) -> tuple[str, np.ndarray]:
    if (
        boundaries.shape != (len(spectrum.counts) + 1,)
        or not np.all(np.isfinite(boundaries))
        or np.any(boundaries < 0)
        or np.any(np.diff(boundaries) <= 0)
    ):
        raise ValueError("N42 energy boundaries must be non-negative and strictly increasing")
    return "EnergyBoundaryValues", boundaries


def _encode_counts(counts: np.ndarray) -> tuple[np.ndarray, str]:
    raw = np.asarray(counts, dtype=np.float64)
    compressed: list[float] = []
    index = 0
    while index < len(raw):
        if raw[index] != 0:
            compressed.append(float(raw[index]))
            index += 1
            continue
        stop = index + 1
        while stop < len(raw) and raw[stop] == 0:
            stop += 1
        compressed.extend((0.0, float(stop - index)))
        index = stop
    if len(compressed) < len(raw):
        return np.asarray(compressed, dtype=np.float64), "CountedZeroes"
    return raw, "None"


def _write_validated(path: Path, root: ET.Element) -> None:
    ET.indent(root, space="  ")
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    _validated_root(payload)
    with path.open("xb") as output:
        output.write(payload)
        output.write(b"\n")


def _double_list(text: str | None, *, field: str) -> np.ndarray:
    raw = text or ""
    try:
        values = np.fromstring(raw, sep=" ", dtype=np.float64)
    except ValueError as exc:
        raise N42ValidationError(f"{field} contains invalid numeric values") from exc
    tokens = raw.split()
    if len(values) != len(tokens) or not len(values) or not np.all(np.isfinite(values)):
        raise N42ValidationError(f"{field} must contain a finite numeric list")
    return values


def _duration_child(element: ET.Element, name: str) -> float:
    value = _child_text(element, name)
    match = _DURATION_RE.fullmatch(value)
    if match is None:
        raise N42ValidationError(f"{name} is not a supported ISO 8601 duration")
    seconds = sum(
        float(match.group(field) or 0) * factor
        for field, factor in (("days", 86400), ("hours", 3600), ("minutes", 60), ("seconds", 1))
    )
    if not math.isfinite(seconds) or seconds < 0:
        raise N42ValidationError(f"{name} must be a finite non-negative duration")
    return seconds


def _duration(seconds: float) -> str:
    return f"PT{_number(seconds)}S"


def _child_text(element: ET.Element, name: str) -> str:
    child = element.find(_Q + name)
    return (child.text or "").strip() if child is not None else ""


def _instrument_metadata(element: ET.Element | None) -> dict[str, object]:
    if element is None:
        return {}
    return {
        "id": element.get("id", ""),
        "manufacturer": _child_text(element, "RadInstrumentManufacturerName"),
        "identifier": _child_text(element, "RadInstrumentIdentifier"),
        "model": _child_text(element, "RadInstrumentModelName"),
        "description": _child_text(element, "RadInstrumentDescription"),
        "class": _child_text(element, "RadInstrumentClassCode"),
    }


def _detector_metadata(element: ET.Element) -> dict[str, object]:
    return {
        "id": element.get("id", ""),
        "name": _child_text(element, "RadDetectorName"),
        "category": _child_text(element, "RadDetectorCategoryCode"),
        "kind": _child_text(element, "RadDetectorKindCode"),
        "description": _child_text(element, "RadDetectorDescription"),
    }


def _positive_metadata_number(spectrum: Spectrum, name: str) -> float | None:
    value = spectrum.metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _calibration_datetime(spectrum: Spectrum) -> str:
    value = spectrum.metadata.get("energy_calibration_datetime")
    if isinstance(value, str) and value.strip():
        return value.strip()
    nested = spectrum.metadata.get("energy_calibration")
    if isinstance(nested, Mapping):
        updated = nested.get("updated_utc")
        if isinstance(updated, str) and updated.strip():
            return updated.strip()
    return ""


def _fit_description(spec: PeakFitSpec, result: PeakFitResult) -> str:
    return (
        f"{spec.peak_count} Gaussian peak(s), {spec.background} background, "
        f"{result.statistic} statistic; reduced chi-square {result.reduced_chi_square:.6g}."
    )


def _energy_width(spectrum: Spectrum, center: float, width: float) -> float | None:
    low = spectrum.energy(center - width / 2.0)
    high = spectrum.energy(center + width / 2.0)
    if low is None or high is None:
        return None
    return abs(high - low)


def _mapping_text(mapping: Mapping[object, object], key: str, default: str) -> str:
    value = mapping.get(key)
    return str(value).strip() if isinstance(value, str) and value.strip() else default


def _sub(parent: ET.Element, name: str, text: str, **attributes: str) -> ET.Element:
    element = ET.SubElement(parent, _Q + name, attributes)
    element.text = text
    return element


def _format_values(values: np.ndarray) -> str:
    return " ".join(_number(float(value)) for value in values)


def _number(value: float) -> str:
    return format(float(value), ".17g")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


__all__ = [
    "N42ExportMetadata",
    "N42ValidationError",
    "export_n42_peak_fit",
    "export_n42_spectrum",
    "load_n42_spectra",
    "n42_export_metadata_from_spectrum",
    "validate_n42",
]
