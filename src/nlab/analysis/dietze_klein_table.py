"""Dietze & Klein (1982) Table 3: Compton maximum and half height relative to the edge.

Source: G. Dietze, H. Klein, "Gamma-calibration of NE 213 scintillation
counters", Nucl. Instrum. Methods 193 (1982) 549-556,
doi:10.1016/0029-554X(82)90249-X, Table 3 (p. 552) and Table 1 (p. 551).

The values are GRESP Monte Carlo results for BA1-type NE 213 cells, including
multiple Compton scattering, folded with the pulse-height resolution ``ΔL/L``
(FWHM) at the Compton edge ``L_c``; Dietze (1979, IEEE Trans. Nucl. Sci. 26,
398, doi:10.1109/TNS.1979.4329665) describes that folding as Gaussian.
``L_max`` is the Compton maximum and ``L_1/2`` the half-maximum point on its
high side. Only the Cs-137 (0.662 MeV) and Na-22 annihilation (0.511 MeV)
blocks are reproduced here.

Values were transcribed from the published page image and checked against it
cell by cell. The printed Cs-137 / detector 3 ``L_max`` column is irregularly
spaced between 10 % and 18 % (0.950, 0.949, 0.927) and at 14 % exceeds
detector 4; it is reproduced as printed.

The ratios refer to light output ``L = c(E_e − E_0)`` with ``E_0 = 5 keV``,
i.e. to pulse height measured from zero light output. They are indicative for
other organic scintillators and detector sizes only.
"""

from __future__ import annotations

from dataclasses import dataclass

CITATION = (
    "G. Dietze, H. Klein, Nucl. Instrum. Methods 193 (1982) 549, "
    "doi:10.1016/0029-554X(82)90249-X, Table 3"
)


@dataclass(frozen=True)
class DietzeKleinDetector:
    """One scintillator geometry of Dietze & Klein Table 1."""

    number: int
    radius_cm: float
    height_cm: float

    @property
    def description(self) -> str:
        return (
            f"Detector {self.number}: NE 213, Ø{2.0 * self.radius_cm:.4g} × "
            f"{self.height_cm:.4g} cm"
        )


@dataclass(frozen=True)
class DietzeKleinBlock:
    """Tabulated ratios for one photon line and one detector."""

    source: str
    gamma_energy_kev: float
    detector: int
    resolution_fwhm_percent: tuple[float, ...]
    lmax_over_edge: tuple[float, ...]
    lhalf_over_edge: tuple[float, ...]


DETECTORS: tuple[DietzeKleinDetector, ...] = (
    DietzeKleinDetector(1, 1.90, 3.80),
    DietzeKleinDetector(2, 2.53, 5.07),
    DietzeKleinDetector(3, 6.35, 5.07),
    DietzeKleinDetector(4, 12.70, 5.08),
)

_CS137_RESOLUTION = (10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0)
_NA22_511_RESOLUTION = (14.0, 18.0, 22.0, 26.0, 30.0, 34.0)

BLOCKS: tuple[DietzeKleinBlock, ...] = (
    DietzeKleinBlock(
        "Cs-137", 661.657, 1, _CS137_RESOLUTION,
        (0.939, 0.922, 0.905, 0.886, 0.867, 0.846, 0.824),
        (1.025, 1.035, 1.045, 1.053, 1.061, 1.069, 1.076),
    ),
    DietzeKleinBlock(
        "Cs-137", 661.657, 2, _CS137_RESOLUTION,
        (0.939, 0.922, 0.905, 0.887, 0.869, 0.849, 0.828),
        (1.026, 1.038, 1.049, 1.059, 1.067, 1.074, 1.081),
    ),
    DietzeKleinBlock(
        "Cs-137", 661.657, 3, _CS137_RESOLUTION,
        (0.950, 0.949, 0.927, 0.913, 0.901, 0.886, 0.869),
        (1.046, 1.063, 1.076, 1.088, 1.098, 1.108, 1.117),
    ),
    DietzeKleinBlock(
        "Cs-137", 661.657, 4, _CS137_RESOLUTION,
        (0.955, 0.948, 0.941, 0.933, 0.924, 0.912, 0.899),
        (1.059, 1.079, 1.095, 1.108, 1.120, 1.130, 1.141),
    ),
    DietzeKleinBlock(
        "Na-22 annihilation", 511.0, 1, _NA22_511_RESOLUTION,
        (0.918, 0.900, 0.881, 0.860, 0.840, 0.817),
        (1.033, 1.043, 1.052, 1.060, 1.067, 1.074),
    ),
    DietzeKleinBlock(
        "Na-22 annihilation", 511.0, 2, _NA22_511_RESOLUTION,
        (0.917, 0.900, 0.882, 0.862, 0.842, 0.818),
        (1.036, 1.047, 1.058, 1.066, 1.074, 1.082),
    ),
    DietzeKleinBlock(
        "Na-22 annihilation", 511.0, 3, _NA22_511_RESOLUTION,
        (0.933, 0.921, 0.909, 0.895, 0.883, 0.869),
        (1.061, 1.077, 1.090, 1.101, 1.112, 1.121),
    ),
    DietzeKleinBlock(
        "Na-22 annihilation", 511.0, 4, _NA22_511_RESOLUTION,
        (0.938, 0.931, 0.924, 0.914, 0.903, 0.892),
        (1.076, 1.096, 1.110, 1.124, 1.136, 1.147),
    ),
)


def detector(number: int) -> DietzeKleinDetector:
    for item in DETECTORS:
        if item.number == number:
            return item
    raise ValueError(f"Dietze-Klein detector must be 1-4, not {number!r}")


def block_for(gamma_energy_kev: float, detector_number: int) -> DietzeKleinBlock | None:
    """Return the tabulated block for a photon line within 1 keV, if any."""
    detector(detector_number)
    for block in BLOCKS:
        if block.detector == detector_number and abs(
            block.gamma_energy_kev - gamma_energy_kev
        ) <= 1.0:
            return block
    return None
