"""Reference photon catalogue and calibrated-coordinate mapping checks."""

from __future__ import annotations

import numpy as np
import pytest

from nlab.analysis.radionuclide_lines import (
    BUILTIN_RADIONUCLIDE_CATALOG,
    canonical_nuclide,
    coordinate_for_energy,
    energy_range_for_coordinates,
    has_usable_energy_axis,
)
from nlab.analysis.spectrum import Spectrum
from scripts.build_radionuclide_catalog import _absolute_probability


@pytest.mark.parametrize(
    ("value", "expected"),
    (("Na22", "Na-22"), ("22na", "Na-22"), ("cs_137", "Cs-137"), ("Ba-133", "Ba-133")),
)
def test_canonical_nuclide_accepts_common_user_forms(value: str, expected: str) -> None:
    assert canonical_nuclide(value) == expected


def test_builtin_catalog_distinguishes_na22_annihilation_and_decay_gamma() -> None:
    lines = BUILTIN_RADIONUCLIDE_CATALOG.lines("22Na", minimum_probability_percent=1.0)

    assert [line.energy_kev for line in lines] == pytest.approx([511.0, 1274.537])
    assert lines[0].kind == "annihilation"
    assert lines[0].emission_probability_percent > 100
    assert lines[1].daughter == "Ne-22"

    document = BUILTIN_RADIONUCLIDE_CATALOG.selection_document(
        ("Na-22",),
        minimum_probability_percent=100.0,
    )
    assert document["catalog_version"] == 2
    assert document["source_data_generated_at"] == "2026-08-10T15:57:13Z"
    assert document["nuclides"] == ["Na-22"]
    assert [line["energy_kev"] for line in document["lines"]] == [511.0]


def test_bundled_catalog_is_the_full_pinned_ensdf_snapshot() -> None:
    metadata = BUILTIN_RADIONUCLIDE_CATALOG.metadata

    assert len(BUILTIN_RADIONUCLIDE_CATALOG.nuclides()) == 2065
    assert int(metadata["recommended_line_count"]) == 87481
    assert BUILTIN_RADIONUCLIDE_CATALOG.database_path.stat().st_size == 13598720

    co60 = BUILTIN_RADIONUCLIDE_CATALOG.lines(
        "Co60",
        minimum_probability_percent=1.0,
    )
    assert [line.energy_kev for line in co60] == pytest.approx([1173.228, 1332.492])
    assert [line.emission_probability_percent for line in co60] == pytest.approx(
        [99.85, 99.9826]
    )
    assert all(line.source_dataset == "60CO B- DECAY (1925.28 D)" for line in co60)

    assert "Am-242m" in BUILTIN_RADIONUCLIDE_CATALOG.nuclides()
    assert "Am-242m2" not in BUILTIN_RADIONUCLIDE_CATALOG.nuclides()


def test_ensdf_relative_intensity_is_normalized_per_parent_decay() -> None:
    intensity = {"value": 100.0, "uncertainty": {"type": "symmetric", "value": 5.0}}
    normalizations = {
        "gamma": {
            "value": 0.076,
            "uncertainty": {"type": "symmetric", "value": 0.008},
            "includesBranchingRatio": False,
        },
        "branchingRatio": {
            "value": 0.5,
            "uncertainty": {"type": "none"},
            "includesBranchingRatio": False,
        },
    }

    probability = _absolute_probability(intensity, normalizations, existence=None)

    assert probability is not None
    assert probability.value == pytest.approx(3.8)
    assert probability.uncertainty == pytest.approx(
        3.8 * np.sqrt((5.0 / 100.0) ** 2 + (0.008 / 0.076) ** 2)
    )
    assert probability.recommended


def test_normalization_does_not_apply_branching_ratio_twice() -> None:
    probability = _absolute_probability(
        {"value": 100.0, "uncertainty": {"type": "none"}},
        {
            "gamma": {
                "value": 0.1,
                "uncertainty": {"type": "none"},
                "includesBranchingRatio": True,
            },
            "branchingRatio": {
                "value": 0.5,
                "uncertainty": {"type": "none"},
                "includesBranchingRatio": False,
            },
        },
        existence=None,
    )

    assert probability is not None
    assert probability.value == pytest.approx(10.0)


def test_limit_only_or_uncertain_lines_are_not_recommended() -> None:
    probability = _absolute_probability(
        {"value": 4.0, "uncertainty": {"type": "limit", "limitType": "upper"}},
        {
            "gamma": {
                "value": 1.0,
                "uncertainty": {"type": "none"},
                "includesBranchingRatio": True,
            }
        },
        existence="uncertain",
    )

    assert probability is not None
    assert probability.qualifier == "bounded"
    assert not probability.recommended


def test_catalog_finds_and_ranks_candidates_in_energy_interval() -> None:
    candidates = BUILTIN_RADIONUCLIDE_CATALOG.candidates(
        660.0,
        663.0,
        minimum_probability_percent=10.0,
    )
    cs137 = next(candidate for candidate in candidates if candidate.nuclide == "Cs-137")

    assert cs137.matching_line_count == 1
    assert cs137.strongest_probability_percent == pytest.approx(85.1)
    assert all(candidate.nuclide != "Na-22" for candidate in candidates)
    assert [candidate.matching_line_count for candidate in candidates] == sorted(
        (candidate.matching_line_count for candidate in candidates),
        reverse=True,
    )


def test_reference_energy_maps_to_channel_only_with_current_calibration() -> None:
    spectrum = Spectrum.create(
        label="calibrated",
        counts=np.ones(5),
        x=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        source="mca",
        energy_kev=np.asarray([0.0, 500.0, 1000.0, 1500.0, 2000.0]),
    )

    assert has_usable_energy_axis(spectrum)
    assert coordinate_for_energy(spectrum, 1274.537) == pytest.approx(2.549074)
    assert coordinate_for_energy(spectrum, 2500.0) is None

    stale = Spectrum.create(
        label="stale",
        counts=np.ones(5),
        source="mca",
        metadata={"calibration_stale": True},
        energy_kev=np.linspace(0.0, 2000.0, 5),
    )
    assert not has_usable_energy_axis(stale)
    assert coordinate_for_energy(stale, 511.0) is None


def test_reference_energy_uses_native_kev_axis() -> None:
    spectrum = Spectrum.create(
        label="energy spectrum",
        counts=np.ones(4),
        x=np.asarray([100.0, 200.0, 300.0, 400.0]),
        axis_unit="keV",
        source="csv",
    )

    assert coordinate_for_energy(spectrum, 250.0) == pytest.approx(250.0)


def test_coordinate_interval_maps_to_current_energy_calibration() -> None:
    spectrum = Spectrum.create(
        label="calibrated",
        counts=np.ones(5),
        x=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        source="mca",
        energy_kev=np.asarray([0.0, 500.0, 1000.0, 1500.0, 2000.0]),
    )

    assert energy_range_for_coordinates(spectrum, 1.2, 2.8) == pytest.approx((600.0, 1400.0))
    assert energy_range_for_coordinates(spectrum, -10.0, 0.5) == pytest.approx((0.0, 250.0))
    assert energy_range_for_coordinates(spectrum, 10.0, 20.0) is None


def test_coordinate_interval_is_clipped_to_physical_energy_domain() -> None:
    spectrum = Spectrum.create(
        label="calibrated",
        counts=np.ones(4),
        source="csv",
        energy_kev=np.asarray([-100.0, 100.0, 300.0, 500.0]),
    )

    assert energy_range_for_coordinates(spectrum, 0.0, 2.0) == pytest.approx(
        (0.0, 300.0)
    )
    assert energy_range_for_coordinates(spectrum, 0.0, 0.25) is None
