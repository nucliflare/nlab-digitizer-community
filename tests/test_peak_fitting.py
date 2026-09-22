"""Constrained MCA peak-model checks using synthetic count spectra."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nlab.analysis.peak_fitting import (
    PeakFitCancelledError,
    PeakFitSpec,
    export_peak_fit_result,
    fit_spectrum_peaks,
)
from nlab.analysis.spectrum import Spectrum, scale_spectrum


def _gaussian(x: np.ndarray, area: float, center: float, sigma: float) -> np.ndarray:
    return area * np.exp(-0.5 * ((x - center) / sigma) ** 2) / (
        sigma * np.sqrt(2.0 * np.pi)
    )


def _synthetic(peaks: list[tuple[float, float, float]]) -> Spectrum:
    x = np.arange(0, 400, dtype=np.float64)
    counts = 15.0 + 0.015 * x
    for area, center, sigma in peaks:
        counts += _gaussian(x, area, center, sigma)
    rounded = np.rint(counts)
    energy = 2.0 + 0.5 * x
    return Spectrum.create(
        label="synthetic",
        x=x,
        counts=rounded,
        source="csv",
        energy_kev=energy,
    )


@pytest.mark.parametrize(
    ("peaks", "centres"),
    [
        ([(80_000, 180, 7)], (180.0,)),
        ([(60_000, 130, 6), (50_000, 225, 9)], (130.0, 225.0)),
        (
            [(50_000, 95, 5), (60_000, 185, 7), (45_000, 285, 8)],
            (95.0, 185.0, 285.0),
        ),
    ],
)
def test_one_two_and_three_gaussian_fits_recover_centres(
    peaks: list[tuple[float, float, float]],
    centres: tuple[float, ...],
) -> None:
    spectrum = _synthetic(peaks)

    result = fit_spectrum_peaks(
        spectrum,
        PeakFitSpec(
            peak_count=len(peaks),
            background="linear",
            fit_min=50,
            fit_max=340,
            peak_centers=centres,
        ),
    )

    assert result.success
    assert result.statistic == "poisson"
    assert [peak.center for peak in result.peaks] == pytest.approx(centres, abs=0.2)
    assert all(peak.area > 0 and peak.fwhm > 0 for peak in result.peaks)
    assert result.peaks[0].energy_kev == pytest.approx(2 + 0.5 * centres[0], abs=0.2)


@pytest.mark.parametrize(
    "background",
    ["none", "constant", "linear", "exponential", "compton", "fermi"],
)
def test_all_background_models_produce_finite_components(background: str) -> None:
    spectrum = _synthetic([(80_000, 180, 7)])

    result = fit_spectrum_peaks(
        spectrum,
        PeakFitSpec(
            peak_count=1,
            background=background,  # type: ignore[arg-type]
            fit_min=130,
            fit_max=230,
            peak_centers=(180,),
        ),
    )

    assert result.success
    assert np.all(np.isfinite(result.best_fit))
    assert "p1_" in result.components
    if background != "none":
        assert "bg_" in result.components


def test_auto_statistic_uses_propagated_variance_for_scaled_spectrum() -> None:
    scaled = scale_spectrum(_synthetic([(80_000, 180, 7)]), 0.5)

    result = fit_spectrum_peaks(
        scaled,
        PeakFitSpec(fit_min=130, fit_max=230, peak_centers=(180,)),
    )

    assert result.statistic == "weighted"


def test_poisson_fit_rejects_derived_non_count_data() -> None:
    scaled = scale_spectrum(_synthetic([(80_000, 180, 7)]), 0.5)

    with pytest.raises(ValueError, match="Poisson"):
        fit_spectrum_peaks(
            scaled,
            PeakFitSpec(
                statistic="poisson",
                fit_min=130,
                fit_max=230,
                peak_centers=(180,),
            ),
        )


def test_fit_can_be_cancelled_before_optimization() -> None:
    spectrum = _synthetic([(80_000, 180, 7)])

    with pytest.raises(PeakFitCancelledError):
        fit_spectrum_peaks(
            spectrum,
            PeakFitSpec(fit_min=130, fit_max=230, peak_centers=(180,)),
            cancelled=lambda: True,
        )


def test_peak_fit_reports_export_to_json_and_csv_without_overwrite(tmp_path: Path) -> None:
    spectrum = _synthetic([(80_000, 180, 7)])
    spec = PeakFitSpec(fit_min=130, fit_max=230, peak_centers=(180,))
    result = fit_spectrum_peaks(spectrum, spec)
    json_path = tmp_path / "fit.json"
    csv_path = tmp_path / "fit.csv"

    export_peak_fit_result(json_path, spectrum=spectrum, spec=spec, result=result)
    export_peak_fit_result(csv_path, spectrum=spectrum, spec=spec, result=result)

    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["kind"] == "nlab_mca_peak_fit"
    assert document["model"]["background"] == "linear"
    assert document["peaks"][0]["center"] == pytest.approx(180, abs=0.2)
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("peak,center")
    with pytest.raises(FileExistsError):
        export_peak_fit_result(json_path, spectrum=spectrum, spec=spec, result=result)
