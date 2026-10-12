"""MCA peak-fitting and energy-calibration report tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfReader

from nlab import __version__
from nlab.analysis.energy_calibration import (
    CalibrationPoint,
    SpectrumSnapshot,
    fit_calibration_peak,
    fit_energy_calibration,
)
from nlab.analysis.mca_report import (
    EnergyCalibrationReportPoint,
    EnergyCalibrationReportRequest,
    PeakFitReportRequest,
    _background_formula,
    _formula_png,
    export_energy_calibration_report,
    export_peak_fit_report,
)
from nlab.analysis.peak_fitting import BackgroundModel, PeakFitSpec, fit_spectrum_peaks
from nlab.analysis.spectrum import Spectrum


def _peak_spectrum() -> Spectrum:
    x = np.arange(300, dtype=np.float64)
    counts = 9.0 + 0.015 * x
    counts += 55_000.0 * np.exp(-0.5 * np.square((x - 143.4) / 5.5)) / (
        5.5 * np.sqrt(2.0 * np.pi)
    )
    return Spectrum.create(
        label="Cs-137 spectrum",
        x=x,
        counts=np.rint(counts),
        source="mca",
        metadata={"elapsed_s": 60.0, "channel": 0},
    )


def test_peak_fit_report_writes_pdf_and_complete_fit_tables(tmp_path: Path) -> None:
    spectrum = _peak_spectrum()
    spec = PeakFitSpec(
        peak_count=1,
        background="linear",
        statistic="poisson",
        fit_min=110.0,
        fit_max=180.0,
        peak_centers=(140.0,),
    )
    result = fit_spectrum_peaks(spectrum, spec)
    request = PeakFitReportRequest(
        spectrum=spectrum,
        spec=spec,
        result=result,
        reference_overlay={"nuclides": ["Cs-137"]},
        y_scale="log",
    )

    bundle = export_peak_fit_report(request, tmp_path)

    assert bundle.directory.name == "Cs-137_spectrum_peak_fit_report"
    assert bundle.pdf.read_bytes().startswith(b"%PDF")
    pdf = PdfReader(bundle.pdf)
    assert len(pdf.pages) == 2
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    assert "MCA Peak-Fitting Report" in text
    assert "Eastern Wall Technologies" in text
    assert f"NLab Digitizer {__version__}" in text
    assert "Fitted peak results" in text
    assert "Selected background: Linear" in text
    assert len(pdf.pages[1].images) >= 3

    with bundle.spectrum_fit_csv.open(encoding="utf-8", newline="") as stream:
        curve_rows = list(csv.DictReader(stream))
    assert len(curve_rows) == len(result.x)
    assert [float(row["best_fit"]) for row in curve_rows] == pytest.approx(result.best_fit)
    with bundle.peaks_csv.open(encoding="utf-8", newline="") as stream:
        peaks = list(csv.DictReader(stream))
    assert float(peaks[0]["center"]) == pytest.approx(143.4, abs=0.1)
    metadata = json.loads(bundle.metadata_json.read_text(encoding="utf-8"))
    assert metadata["model"]["peak_centers"] == [140.0]
    assert metadata["display"]["y_scale"] == "log"
    assert metadata["reference_overlay"]["nuclides"] == ["Cs-137"]


@pytest.mark.parametrize(
    ("background", "expected_name", "formula_fragment"),
    [
        ("none", "None", r"b(x)=0"),
        ("constant", "Constant", r"b(x)=c"),
        ("linear", "Linear", r"c+mx"),
        ("exponential", "Centred exponential", r"x_{\mathrm{ref}}"),
        ("compton", "Compton step (error function)", r"\operatorname{erfc}"),
        ("fermi", "Fermi step (logistic)", r"1+\exp"),
    ],
)
def test_peak_report_formula_matches_selected_background(
    background: BackgroundModel,
    expected_name: str,
    formula_fragment: str,
) -> None:
    name, formula = _background_formula(background)

    assert name == expected_name
    assert formula_fragment in formula
    assert _formula_png(formula).getvalue().startswith(b"\x89PNG")


def test_energy_calibration_report_records_seeds_centroids_and_residuals(
    tmp_path: Path,
) -> None:
    x = np.arange(280, dtype=np.float64)
    centers = (64.25, 204.6)
    counts = np.full_like(x, 12.0)
    for center, area, sigma in ((centers[0], 42_000.0, 4.2), (centers[1], 58_000.0, 5.0)):
        counts += area * np.exp(-0.5 * np.square((x - center) / sigma)) / (
            sigma * np.sqrt(2.0 * np.pi)
        )
    snapshot = SpectrumSnapshot.create(
        channel=0,
        counts=np.rint(counts).astype(np.uint32),
        label="Na-22 calibration",
        elapsed_s=90.0,
        live=False,
        fingerprint={"binning": 0},
    )
    seeds = (61.0, 200.0)
    peak_fits = tuple(
        fit_calibration_peak(snapshot, seed, half_width=24.0) for seed in seeds
    )
    points = tuple(
        CalibrationPoint(
            channel=fit.centroid_channel,
            energy_kev=energy,
            label=label,
            source=snapshot.label,
        )
        for fit, energy, label in zip(
            peak_fits,
            (511.0, 1274.537),
            ("annihilation", "Na-22"),
            strict=True,
        )
    )
    calibration = fit_energy_calibration(
        points,
        fingerprint=snapshot.fingerprint,
        max_channel=float(len(snapshot.counts) - 1),
    )
    request = EnergyCalibrationReportRequest(
        channel=0,
        snapshots=(snapshot,),
        calibration=calibration,
        points=tuple(
            EnergyCalibrationReportPoint(point=point, seed_channel=seed, peak_fit=fit)
            for point, seed, fit in zip(points, seeds, peak_fits, strict=True)
        ),
        centroid_half_width=24.0,
    )

    bundle = export_energy_calibration_report(request, tmp_path)

    assert bundle.directory.name == "mca_0_energy_calibration_report"
    pdf = PdfReader(bundle.pdf)
    assert len(pdf.pages) == 2
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    assert "MCA Energy-Calibration Report" in text
    assert "Gaussian centroids used for calibration" in text
    assert "observed minus fitted counts" in text
    assert "Energy calibration and residuals" in text
    assert len(pdf.pages[1].images) >= 3

    with bundle.points_csv.open(encoding="utf-8", newline="") as stream:
        exported_points = list(csv.DictReader(stream))
    assert [float(row["seed_channel"]) for row in exported_points] == list(seeds)
    assert [float(row["centroid_channel"]) for row in exported_points] == pytest.approx(
        centers, abs=0.1
    )
    assert [float(row["residual_kev"]) for row in exported_points] == pytest.approx(
        [0.0, 0.0], abs=1e-8
    )
    with bundle.centroid_fits_csv.open(encoding="utf-8", newline="") as stream:
        centroid_rows = list(csv.DictReader(stream))
    assert len(centroid_rows) == sum(len(fit.x) for fit in peak_fits)
    first_row = centroid_rows[0]
    assert float(first_row["residual"]) == pytest.approx(
        float(first_row["observed"]) - float(first_row["best_fit"])
    )
    metadata = json.loads(bundle.metadata_json.read_text(encoding="utf-8"))
    assert metadata["centroid_method"]["half_width_channel"] == 24.0
    assert metadata["points"][0]["seed_channel"] == seeds[0]
    assert metadata["points"][0]["centroid_fit"]["centroid_channel"] == pytest.approx(
        centers[0], abs=0.1
    )


def test_mca_report_bundles_never_overwrite_existing_directories(tmp_path: Path) -> None:
    spectrum = _peak_spectrum()
    spec = PeakFitSpec(fit_min=110.0, fit_max=180.0, peak_centers=(140.0,))
    result = fit_spectrum_peaks(spectrum, spec)
    request = PeakFitReportRequest(spectrum=spectrum, spec=spec, result=result)

    first = export_peak_fit_report(request, tmp_path)
    second = export_peak_fit_report(request, tmp_path)

    assert first.directory.name == "Cs-137_spectrum_peak_fit_report"
    assert second.directory.name == "Cs-137_spectrum_peak_fit_report_2"
    assert first.pdf.is_file()
    assert second.pdf.is_file()
