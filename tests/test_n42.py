from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from nlab.analysis.n42 import (
    N42ValidationError,
    export_n42_peak_fit,
    export_n42_spectrum,
    load_n42_spectra,
    validate_n42,
)
from nlab.analysis.peak_fitting import PeakFitSpec, fit_spectrum_peaks
from nlab.analysis.spectrum import Spectrum
from nlab.analysis.spectrum_io import load_spectra

N42 = "http://physics.nist.gov/N42/2011/N42"


def _document(*, boundaries: str | None = None) -> str:
    calibration = (
        f"<EnergyBoundaryValues>{boundaries}</EnergyBoundaryValues>"
        if boundaries is not None
        else "<CoefficientValues>0 2 0</CoefficientValues>"
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<RadInstrumentData xmlns="{N42}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="{N42} https://invalid.example/never-fetch.xsd"
    n42DocUUID="12345678-1234-1234-1234-123456789abc"
    n42DocDateTime="2026-10-14T10:00:00Z">
  <RadInstrumentDataCreatorName>Test</RadInstrumentDataCreatorName>
  <RadInstrumentInformation id="Instrument-1">
    <RadInstrumentManufacturerName>Example</RadInstrumentManufacturerName>
    <RadInstrumentIdentifier>ABC</RadInstrumentIdentifier>
    <RadInstrumentModelName>Model</RadInstrumentModelName>
    <RadInstrumentClassCode>Other</RadInstrumentClassCode>
    <RadInstrumentVersion>
      <RadInstrumentComponentName>Software</RadInstrumentComponentName>
      <RadInstrumentComponentVersion>1.0</RadInstrumentComponentVersion>
    </RadInstrumentVersion>
  </RadInstrumentInformation>
  <RadDetectorInformation id="Detector-1">
    <RadDetectorName>Gamma input</RadDetectorName>
    <RadDetectorCategoryCode>Gamma</RadDetectorCategoryCode>
    <RadDetectorKindCode>NaI</RadDetectorKindCode>
  </RadDetectorInformation>
  <EnergyCalibration id="Calibration-1">{calibration}</EnergyCalibration>
  <RadMeasurement id="Measurement-1">
    <MeasurementClassCode>Foreground</MeasurementClassCode>
    <StartDateTime>2026-10-14T09:59:50Z</StartDateTime>
    <RealTimeDuration>PT10S</RealTimeDuration>
    <Spectrum id="Spectrum-1" energyCalibrationReference="Calibration-1"
        radDetectorInformationReference="Detector-1">
      <LiveTimeDuration>PT9.5S</LiveTimeDuration>
      <ChannelData compressionCode="CountedZeroes">0 2 5 0 1</ChannelData>
    </Spectrum>
    <Spectrum id="Spectrum-2" energyCalibrationReference="Calibration-1"
        radDetectorInformationReference="Detector-1">
      <LiveTimeDuration>PT9S</LiveTimeDuration>
      <ChannelData>1 2 3 4</ChannelData>
    </Spectrum>
  </RadMeasurement>
</RadInstrumentData>
"""


def _spectrum() -> Spectrum:
    channels = np.arange(128, dtype=np.float64)
    counts = 4.0 + 100.0 * np.exp(-0.5 * np.square((channels - 60.0) / 4.0))
    return Spectrum.create(
        label="calibrated spectrum",
        x=channels,
        counts=counts,
        source="mca",
        metadata={
            "elapsed_s": 10.0,
            "live_time_s": 9.5,
            "created_utc": "2026-10-14T09:59:50+00:00",
            "energy_calibration_coefficients_kev": [0.0, 2.0],
            "n42_instrument": {
                "manufacturer": "Eastern Wall Technologies",
                "model": "Nuclear Lab Digitizer",
                "identifier": "D1:CH0@192.0.2.10:30431",
            },
            "n42_detector": {"name": "D1:CH0", "category": "Other", "kind": "Other"},
        },
        energy_kev=channels * 2.0,
    )


def test_load_multi_spectrum_n42_with_counted_zeroes(tmp_path: Path) -> None:
    path = tmp_path / "multi.n42"
    path.write_text(_document(), encoding="utf-8")

    spectra = load_spectra(path)

    assert len(spectra) == 2
    assert spectra[0].source == "n42"
    assert spectra[0].counts.tolist() == [0.0, 0.0, 5.0, 0.0]
    assert spectra[0].energy_kev.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert spectra[0].metadata["elapsed_s"] == pytest.approx(10.0)
    assert spectra[0].metadata["live_time_s"] == pytest.approx(9.5)
    assert spectra[0].metadata["n42_detector"] == {
        "id": "Detector-1",
        "name": "Gamma input",
        "category": "Gamma",
        "kind": "NaI",
        "description": "",
    }
    assert "Spectrum-2" in spectra[1].label


def test_validation_rejects_entities_and_boundary_mismatch(tmp_path: Path) -> None:
    entity_path = tmp_path / "entity.n42"
    entity_path.write_text(
        '<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        f'<RadInstrumentData xmlns="{N42}"/>',
        encoding="utf-8",
    )
    with pytest.raises(N42ValidationError, match="entities are not allowed"):
        validate_n42(entity_path)

    mismatch_path = tmp_path / "mismatch.n42"
    mismatch_path.write_text(_document(boundaries="0 1 2 3"), encoding="utf-8")
    with pytest.raises(N42ValidationError, match="EnergyBoundaryValues has 4 entries"):
        validate_n42(mismatch_path)


def test_n42_spectrum_export_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "spectrum.n42"
    source = _spectrum()

    export_n42_spectrum(path, source)
    validate_n42(path)
    loaded = load_n42_spectra(path)

    assert len(loaded) == 1
    assert np.allclose(loaded[0].counts, source.counts)
    assert np.allclose(loaded[0].energy_kev, source.energy_kev)
    assert loaded[0].metadata["live_time_s"] == pytest.approx(9.5)
    root = ET.parse(path).getroot()
    instrument = root.find(f"{{{N42}}}RadInstrumentInformation")
    detector = root.find(f"{{{N42}}}RadDetectorInformation")
    assert instrument is not None
    assert detector is not None
    assert instrument.findtext(f"{{{N42}}}RadInstrumentIdentifier") == (
        "D1:CH0@192.0.2.10:30431"
    )
    assert detector.findtext(f"{{{N42}}}RadDetectorCategoryCode") == "Other"

    with pytest.raises(FileExistsError):
        export_n42_spectrum(path, source)


def test_n42_export_rebases_calibration_to_exported_channel_indices(tmp_path: Path) -> None:
    coordinates = np.arange(10.0, 20.0)
    spectrum = Spectrum.create(
        label="cropped calibrated spectrum",
        x=coordinates,
        counts=np.arange(1.0, 11.0),
        source="derived",
        metadata={
            "elapsed_s": 10.0,
            "live_time_s": 9.5,
            "created_utc": "2026-10-14T09:59:50+00:00",
            "energy_calibration_coefficients_kev": [0.0, 2.0],
        },
        energy_kev=coordinates * 2.0,
        poisson_counts=False,
    )
    path = tmp_path / "cropped.n42"

    export_n42_spectrum(path, spectrum)
    imported = load_n42_spectra(path)[0]

    assert imported.x.tolist() == list(np.arange(10, dtype=np.float64))
    assert np.allclose(imported.energy_kev, spectrum.energy_kev)


@pytest.mark.parametrize(
    ("metadata_update", "message"),
    [
        ({"calibration_stale": True}, "stale energy calibration"),
        ({"energy_calibration_coefficients_kev": None}, "applied energy calibration"),
        ({"live_time_s": None}, "elapsed_s and live_time_s"),
    ],
)
def test_n42_export_rejects_unreliable_measurement_metadata(
    tmp_path: Path,
    metadata_update: dict[str, object],
    message: str,
) -> None:
    original = _spectrum()
    metadata = {**original.metadata, **metadata_update}
    energy = (
        None
        if metadata_update.get("energy_calibration_coefficients_kev") is None
        else original.energy_kev
    )
    spectrum = Spectrum.create(
        label=original.label,
        x=original.x,
        counts=original.counts,
        source="mca",
        metadata=metadata,
        energy_kev=energy,
    )
    with pytest.raises(ValueError, match=message):
        export_n42_spectrum(tmp_path / "bad.n42", spectrum)


def test_peak_fit_export_contains_n42_peak_results(tmp_path: Path) -> None:
    spectrum = _spectrum()
    spec = PeakFitSpec(
        peak_count=1,
        background="linear",
        fit_min=30.0,
        fit_max=90.0,
        peak_centers=(60.0,),
    )
    result = fit_spectrum_peaks(spectrum, spec)
    path = tmp_path / "fit.n42"

    export_n42_peak_fit(path, spectrum=spectrum, spec=spec, result=result)
    validate_n42(path)

    root = ET.parse(path).getroot()
    peak = root.find(f".//{{{N42}}}SpectrumPeak")
    assert peak is not None
    assert float(peak.findtext(f"{{{N42}}}SpectrumPeakEnergyValue", "0")) == pytest.approx(120)
    assert float(peak.findtext(f"{{{N42}}}SpectrumPeakFWHMValue", "0")) > 0
    assert float(peak.findtext(f"{{{N42}}}SpectrumPeakNetAreaValue", "0")) > 0
