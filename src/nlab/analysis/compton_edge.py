"""Klein–Nishina Compton-edge fits for organic-scintillator light-output calibration.

Organic scintillators (EJ-309, EJ-276D, stilbene, organic glass) have almost no
photoabsorption, so their gamma spectra contain Compton continua without a
usable photopeak. Their light output is calibrated in electron-equivalent
energy (keVee) from the Compton edges. Heuristics such as "edge = Compton
maximum" or "edge = half height" are biased by an amount that depends on the
resolution (Knox & Miller 1972; Dietze & Klein 1982). This module therefore
forward-fits the electron spectrum folded with the detector resolution.

Physics model
=============
For a photon of energy ``E_γ`` with ``α = E_γ / m_e c²`` (``m_e c²`` =
510.998950 keV) and ``s = T / E_γ``, the kinetic-energy distribution of
singly scattered Compton electrons is (Klein & Nishina 1929)::

    dσ/dT ∝ 2 + s² / (α² (1 − s)²) + s / (1 − s) · (s − 2/α),   0 ≤ T ≤ T_max
    T_max = E_γ · 2α / (1 + 2α)                                  (Compton edge)

and zero above ``T_max`` (477.334 keV for 661.657 keV). The shape is finite at
``T_max`` because ``1 − s ≥ 1/(1 + 2α)`` there. Its integral is analytic: with
``v = 1 − s``,
``∫ f ds = −[(2/α + 1/α²) v + v²/2 − 1/(α² v) + (1 − 2/α − 2/α²) ln v]``.
The electron-energy grid uses cells of at most 1 keV whose weights are these
exact cell integrals, normalised to one per gamma line and cached per line
energy.

Detector response. A true deposited energy ``T`` (keVee) appears in the
spectrum's x units as a Gaussian with centre and width::

    μ(T) = g · T + o                           (o may be fixed, normally to 0)
    σ_E/E = sqrt(a² + b²/E + c²/E²)            (a: light collection/position,
                                                b: photoelectron statistics,
                                                c: noise, fixed at 0 by default)
    σ_x(T) = g · T · σ_E/E(T)

Dietze & Klein (1982) write the same form for the FWHM,
``ΔL/L = sqrt(α² + β²/L + γ²/L²)``. This module fits the σ terms and reports
both conventions; each FWHM term is ``2.3548 ×`` the σ term.

Binned prediction. Every Gaussian is integrated over the actual bin edges::

    model_bin = Σ_lines A_line Σ_T w_line(T) [Φ((x_hi − μ)/σ_x) − Φ((x_lo − μ)/σ_x)]
                + background

With ``u = (x − o)/g`` the argument is ``(u − T)/(T · σ_E/E(T))``, which does not
depend on ``g`` or ``o``. The folded cumulative response ``F(u)`` is evaluated as
one (grid × nodes) matrix shared by all lines and datasets. The nodes are the
bin edges themselves for ordinary windows. For very large windows they are a
1 keVee grid with cubic Hermite interpolation that uses the exact derivative
``F'(u)``. The analytic Jacobian (``F'``, ``∂F/∂a``, ``∂F/∂b``, ``∂F/∂c``) keeps
interactive fits fast.

Statistic: Poisson deviance by default, using the same ``auto | poisson |
weighted | unweighted`` resolution as the photopeak fitter. Several frozen
spectra can be fitted jointly. They share ``g, o, a, b, c`` and keep their own
line amplitudes and background. Joint fits over several edge energies are what
constrain gain and offset together.

Limitations. Single scattering ignores multiple Compton scattering, which fills
the region above the edge and biases the fitted edge upward in finite detectors
(Kudomi 1999; Dietze & Klein 1982). A simulated deposited-energy response
(e.g. Geant4/OpenGATE) can replace the analytic shape per line (Mauritzson et
al. 2022). The model-independent shape diagnostics in this module (Compton
maximum, half height, Dietze–Klein resolution rule, edge height) help to judge
that bias. Organic-scintillator electron light yield is not exactly linear
above 500 keV, so a linear keVee fit has a positive energy intercept (Swiderski
et al. 2012).

References
==========
- O. Klein, Y. Nishina, Z. Phys. 52 (1929) 853, doi:10.1007/BF01366453
- H. H. Knox, T. G. Miller, Nucl. Instrum. Methods 101 (1972) 519,
  doi:10.1016/0029-554X(72)90040-7
- G. Dietze, IEEE Trans. Nucl. Sci. 26 (1979) 398, doi:10.1109/TNS.1979.4329665
- G. Dietze, H. Klein, Nucl. Instrum. Methods 193 (1982) 549,
  doi:10.1016/0029-554X(82)90249-X
- N. Kudomi, Nucl. Instrum. Methods A 430 (1999) 96,
  doi:10.1016/S0168-9002(99)00200-4
- E. R. Siciliano et al., Nucl. Instrum. Methods A 594 (2008) 232,
  doi:10.1016/j.nima.2008.06.031
- Ł. Swiderski et al., Radiat. Meas. 45 (2010) 605,
  doi:10.1016/j.radmeas.2009.10.015
- Ł. Swiderski et al., JINST 7 (2012) P06011, doi:10.1088/1748-0221/7/06/P06011
- N. Mauritzson et al., Nucl. Instrum. Methods A 1023 (2022) 165962,
  doi:10.1016/j.nima.2021.165962
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np

from nlab.analysis import dietze_klein_table
from nlab.analysis._fit_common import (
    FIT_STATISTICS,
    FitStatistic,
    ResolvedFitStatistic,
    centred_exponential,
    finite_optional,
    poisson_deviance_residual_derivative,
    readonly,
    resolve_statistic,
    statistic_residual,
    statistic_uncertainties,
)
from nlab.analysis.energy_calibration import CalibrationPoint, EnergyUnit
from nlab.analysis.peak_fitting import FitParameter, PeakFitCancelledError
from nlab.analysis.spectrum import Spectrum

# scipy is imported inside the fitting functions: it is only needed when a fit
# runs, not to start the application.

ELECTRON_REST_ENERGY_KEV = 510.998950
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
COMPTON_EDGE_ENERGY_UNIT: EnergyUnit = "keVee"
COMPTON_EDGE_POINT_SOURCE = "compton-edge fit"
DEFAULT_GRID_STEP_KEV = 1.0
REDUCED_DEVIANCE_WARNING = 3.0
EDGE_HEIGHT_EXPECTED = (0.65, 0.90)
"""Edge-height fractions outside this range raise a warning (see Swiderski 2010)."""
SWIDERSKI_EDGE_HEIGHT = (0.74, 0.82)
"""Compton-coincidence edge heights for BC408 (0.74–0.78) and BC501A (0.78–0.82)."""
COMPTON_PRESET_NUCLIDES = (
    "Cs-137",
    "Na-22",
    "Co-60",
    "Mn-54",
    "Ba-133",
    "Bi-207",
    "Y-88",
    "Zn-65",
    "K-40",
    "Hg-203",
)
PRESET_MINIMUM_PROBABILITY_PERCENT = 5.0
PRESET_MINIMUM_ENERGY_KEV = 150.0
REFERENCES = (
    "O. Klein, Y. Nishina, Z. Phys. 52 (1929) 853, doi:10.1007/BF01366453",
    "H. H. Knox, T. G. Miller, Nucl. Instrum. Methods 101 (1972) 519, "
    "doi:10.1016/0029-554X(72)90040-7",
    "G. Dietze, IEEE Trans. Nucl. Sci. 26 (1979) 398, doi:10.1109/TNS.1979.4329665",
    "G. Dietze, H. Klein, Nucl. Instrum. Methods 193 (1982) 549, "
    "doi:10.1016/0029-554X(82)90249-X",
    "N. Kudomi, Nucl. Instrum. Methods A 430 (1999) 96, doi:10.1016/S0168-9002(99)00200-4",
    "E. R. Siciliano et al., Nucl. Instrum. Methods A 594 (2008) 232, "
    "doi:10.1016/j.nima.2008.06.031",
    "Ł. Swiderski et al., Radiat. Meas. 45 (2010) 605, doi:10.1016/j.radmeas.2009.10.015",
    "Ł. Swiderski et al., JINST 7 (2012) P06011, doi:10.1088/1748-0221/7/06/P06011",
    "N. Mauritzson et al., Nucl. Instrum. Methods A 1023 (2022) 165962, "
    "doi:10.1016/j.nima.2021.165962",
)

ComptonBackground = Literal["none", "constant", "linear", "exponential"]
AmplitudeMode = Literal["free", "linked"]
ResolutionTermName = Literal["a", "b", "c"]
EvaluationPath = Literal["direct", "tabulated"]

_BACKGROUNDS = frozenset({"none", "constant", "linear", "exponential"})
_RESOLUTION_NAMES: tuple[ResolutionTermName, ...] = ("a", "b", "c")
_RESOLUTION_UNITS = {"a": "", "b": "keV^1/2", "c": "keV"}
_RESOLUTION_BOUNDS = {"a": (0.0, 0.5), "b": (0.0, 20.0), "c": (0.0, 200.0)}
_GAUSS_SPAN = 8.5
_TABULATION_STEPS_KEV = (2.0, 1.0, 0.5)
_TABULATION_POINTS_PER_SIGMA = 8.0
_MIN_DATASET_BINS = 12
_DIAGNOSTIC_SMOOTHING_DIVISOR = 14.0


class ComptonEdgeFitCancelledError(PeakFitCancelledError):
    """Raised when the caller asks an active Compton-edge fit to stop."""


ComptonEdgeFitStage = Literal["prepare", "gain_scan", "optimize", "finalize"]
_GAIN_SCAN_CANDIDATES = 160


@dataclass(frozen=True)
class ComptonEdgeFitProgress:
    """Progress of one fit stage, reported from the fitting thread.

    ``total`` is known only for the starting-gain scan (one step per candidate
    gain). The optimizer's evaluation count cannot be predicted, so ``optimize``
    reports residual evaluations done with ``total=None``; ``prepare`` and
    ``finalize`` report no count.
    """

    stage: ComptonEdgeFitStage
    done: int = 0
    total: int | None = None


ComptonEdgeProgressCallback = Callable[[ComptonEdgeFitProgress], None]


def _no_progress(_update: ComptonEdgeFitProgress) -> None:
    return None


class _CancelledError(Exception):
    pass


# --------------------------------------------------------------------- physics


def compton_edge_kev(e_gamma_kev: float) -> float:
    """Return the Compton-edge electron energy ``T_max`` for one photon energy."""
    energy = float(e_gamma_kev)
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("photon energy must be positive and finite")
    alpha = energy / ELECTRON_REST_ENERGY_KEV
    return energy * 2.0 * alpha / (1.0 + 2.0 * alpha)


def klein_nishina_electron_spectrum(
    t_kev: float | np.ndarray,
    e_gamma_kev: float,
) -> np.ndarray:
    """Return the single-scatter Compton-electron shape ``dσ/dT`` up to a constant.

    The value is the dimensionless bracket
    ``2 + s²/(α²(1 − s)²) + s/(1 − s)(s − 2/α)``. It is zero outside
    ``0 ≤ T ≤ T_max``.
    """
    energy = float(e_gamma_kev)
    edge = compton_edge_kev(energy)
    alpha = energy / ELECTRON_REST_ENERGY_KEV
    t = np.atleast_1d(np.asarray(t_kev, dtype=np.float64))
    result = np.zeros_like(t)
    inside = (t >= 0.0) & (t <= edge)
    s = t[inside] / energy
    result[inside] = (
        2.0 + s**2 / (alpha**2 * (1.0 - s) ** 2) + s / (1.0 - s) * (s - 2.0 / alpha)
    )
    return result


def _kn_cumulative(t_kev: np.ndarray, e_gamma_kev: float) -> np.ndarray:
    """Exact integral of :func:`klein_nishina_electron_spectrum` from 0 to ``t``."""
    energy = float(e_gamma_kev)
    alpha = energy / ELECTRON_REST_ENERGY_KEV
    s_max = compton_edge_kev(energy) / energy
    s = np.clip(np.asarray(t_kev, dtype=np.float64) / energy, 0.0, s_max)

    def antiderivative(v: np.ndarray | float) -> np.ndarray:
        v_array = np.asarray(v, dtype=np.float64)
        return np.asarray(
            (2.0 / alpha + 1.0 / alpha**2) * v_array
            + 0.5 * v_array**2
            - 1.0 / (alpha**2 * v_array)
            + (1.0 - 2.0 / alpha - 2.0 / alpha**2) * np.log(v_array),
            dtype=np.float64,
        )

    return np.asarray(energy * (antiderivative(1.0) - antiderivative(1.0 - s)))


@lru_cache(maxsize=256)
def _kn_grid_weights(e_gamma_kev: float, step_kev: float, cells: int) -> np.ndarray:
    edges = np.arange(cells + 1, dtype=np.float64) * step_kev
    cumulative = _kn_cumulative(edges, e_gamma_kev)
    total = float(_kn_cumulative(np.asarray([compton_edge_kev(e_gamma_kev)]), e_gamma_kev)[0])
    weights = np.diff(cumulative) / total
    weights.setflags(write=False)
    return weights


@runtime_checkable
class ResponseShape(Protocol):
    """Deposited-electron-energy distribution of one gamma line.

    ``weights`` receives increasing cell edges in keV and returns one
    non-negative weight per cell. The weights are normalised so that a grid
    covering ``[0, max_energy_kev]`` sums to one.
    """

    @property
    def label(self) -> str: ...

    @property
    def max_energy_kev(self) -> float: ...

    def weights(self, cell_edges_kev: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class KleinNishinaResponse:
    """Analytic single-scatter Klein–Nishina electron spectrum."""

    gamma_energy_kev: float

    @property
    def label(self) -> str:
        return "Klein–Nishina single scatter"

    @property
    def max_energy_kev(self) -> float:
        return compton_edge_kev(self.gamma_energy_kev)

    def weights(self, cell_edges_kev: np.ndarray) -> np.ndarray:
        edges = np.asarray(cell_edges_kev, dtype=np.float64)
        total = float(
            _kn_cumulative(np.asarray([self.max_energy_kev]), self.gamma_energy_kev)[0]
        )
        return np.diff(_kn_cumulative(edges, self.gamma_energy_kev)) / total


@dataclass(frozen=True, eq=False)
class SimulatedResponse:
    """Tabulated deposited-energy histogram, for example from Geant4 or OpenGATE.

    ``energies_kev`` are bin centres and ``bin_weights`` their contents. The
    histogram is treated as uniform within each bin; bin boundaries lie midway
    between centres and the outer bins are symmetric about their centres.
    """

    energies_kev: np.ndarray
    bin_weights: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies_kev, dtype=np.float64).copy()
        weights = np.asarray(self.bin_weights, dtype=np.float64).copy()
        if energies.ndim != 1 or weights.shape != energies.shape or len(energies) < 2:
            raise ValueError("a simulated response needs at least two energy/weight rows")
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(weights)):
            raise ValueError("simulated response values must be finite")
        if np.any(np.diff(energies) <= 0):
            raise ValueError("simulated response energies must be strictly increasing")
        if np.any(weights < 0) or float(np.sum(weights)) <= 0:
            raise ValueError("simulated response weights must be non-negative with a positive sum")
        boundaries = _centres_to_edges(energies)
        if boundaries[0] < 0:
            raise ValueError("simulated response bins must not extend below 0 keV")
        energies.setflags(write=False)
        weights.setflags(write=False)
        boundaries.setflags(write=False)
        object.__setattr__(self, "energies_kev", energies)
        object.__setattr__(self, "bin_weights", weights)
        object.__setattr__(self, "_boundaries", boundaries)

    @property
    def bin_edges_kev(self) -> np.ndarray:
        return cast(np.ndarray, object.__getattribute__(self, "_boundaries"))

    @property
    def label(self) -> str:
        name = Path(self.source).name if self.source else "table"
        return f"simulated response ({name})"

    @property
    def max_energy_kev(self) -> float:
        return float(self.bin_edges_kev[-1])

    def weights(self, cell_edges_kev: np.ndarray) -> np.ndarray:
        boundaries = self.bin_edges_kev
        cumulative = np.concatenate(([0.0], np.cumsum(self.bin_weights)))
        cumulative /= cumulative[-1]
        at_edges = np.interp(np.asarray(cell_edges_kev, dtype=np.float64), boundaries, cumulative)
        return np.asarray(np.diff(at_edges), dtype=np.float64)


def load_simulated_response(path: Path) -> SimulatedResponse:
    """Load a two-column ``energy_kev, weight`` deposited-energy histogram (CSV/TSV)."""
    path = Path(path)
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([item.strip() for item in re.split(r"[,;\t ]+", stripped) if item.strip()])
    if not rows:
        raise ValueError(f"{path.name} contains no response rows")
    header = rows[0] if any(not _is_float(value) for value in rows[0]) else None
    data = rows[1:] if header is not None else rows
    energy_column, weight_column = 0, 1
    if header is not None:
        names = [value.casefold() for value in header]
        energy_column = next(
            (index for index, name in enumerate(names) if name.startswith("energy")), 0
        )
        weight_column = next(
            (
                index
                for index, name in enumerate(names)
                if name in {"weight", "weights", "counts", "probability", "value"}
            ),
            1 if energy_column != 1 else 0,
        )
    try:
        energies = [float(row[energy_column]) for row in data]
        weights = [float(row[weight_column]) for row in data]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path.name} needs numeric energy_kev and weight columns") from exc
    return SimulatedResponse(
        np.asarray(energies, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        source=str(path),
    )


# ------------------------------------------------------------------ definitions


@dataclass(frozen=True)
class ComptonLine:
    """One gamma line whose Compton continuum is fitted."""

    energy_kev: float
    label: str = ""
    nuclide: str | None = None
    emission_probability_percent: float | None = None
    response: ResponseShape | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        energy = float(self.energy_kev)
        if not math.isfinite(energy) or energy <= 0:
            raise ValueError("gamma-line energy must be positive and finite")
        object.__setattr__(self, "energy_kev", energy)
        probability = self.emission_probability_percent
        if probability is not None and (not math.isfinite(probability) or probability <= 0):
            raise ValueError("emission probability must be positive and finite")
        if not self.label:
            prefix = f"{self.nuclide} " if self.nuclide else ""
            object.__setattr__(self, "label", f"{prefix}{energy:g} keV")

    @property
    def edge_kev(self) -> float:
        return compton_edge_kev(self.energy_kev)

    @property
    def shape(self) -> ResponseShape:
        return self.response if self.response is not None else KleinNishinaResponse(
            self.energy_kev
        )

    def with_response(self, response: ResponseShape | None) -> ComptonLine:
        return replace(self, response=response)


def nuclide_compton_lines(
    nuclide: str,
    *,
    minimum_probability_percent: float = PRESET_MINIMUM_PROBABILITY_PERCENT,
    minimum_energy_kev: float = PRESET_MINIMUM_ENERGY_KEV,
) -> tuple[ComptonLine, ...]:
    """Return catalogue gamma lines useful as Compton-edge references for a nuclide."""
    from nlab.analysis.radionuclide_lines import BUILTIN_RADIONUCLIDE_CATALOG

    lines = BUILTIN_RADIONUCLIDE_CATALOG.lines(
        nuclide,
        minimum_probability_percent=minimum_probability_percent,
    )
    canonical = lines[0].parent if lines else nuclide
    return tuple(
        ComptonLine(
            energy_kev=line.energy_kev,
            label=f"{line.parent} {line.energy_kev:g} keV",
            nuclide=line.parent,
            emission_probability_percent=line.emission_probability_percent,
        )
        for line in lines
        if line.energy_kev >= minimum_energy_kev and line.parent == canonical
    )


def parse_custom_energies(text: str) -> tuple[float, ...]:
    """Parse comma/semicolon/space separated keV values for custom lines."""
    values: list[float] = []
    for item in re.split(r"[,;\s]+", text.strip()):
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"invalid gamma energy {item!r}") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("custom gamma energies must be positive keV values")
        values.append(value)
    return tuple(values)


@dataclass(frozen=True, eq=False)
class ComptonEdgeDataset:
    """One frozen spectrum, its fit window, gamma lines and background model."""

    spectrum: Spectrum
    lines: tuple[ComptonLine, ...]
    fit_min: float | None = None
    fit_max: float | None = None
    background: ComptonBackground = "exponential"

    def __post_init__(self) -> None:
        object.__setattr__(self, "lines", tuple(self.lines))
        if not self.lines:
            raise ValueError("a Compton-edge dataset needs at least one gamma line")
        if self.background not in _BACKGROUNDS:
            raise ValueError(f"unsupported Compton-edge background {self.background!r}")
        if (
            self.fit_min is not None
            and self.fit_max is not None
            and not float(self.fit_min) < float(self.fit_max)
        ):
            raise ValueError("Compton-edge fit range must have fit_min < fit_max")

    @property
    def label(self) -> str:
        return self.spectrum.label


@dataclass(frozen=True, eq=False)
class ComptonEdgeFitSpec:
    """Global model choices for a (joint) Compton-edge fit."""

    datasets: tuple[ComptonEdgeDataset, ...]
    offset_free: bool = True
    resolution_terms: tuple[ResolutionTermName, ...] = ("a", "b")
    statistic: FitStatistic = "auto"
    amplitude_mode: AmplitudeMode = "free"
    initial_gain: float | None = None
    initial_offset: float = 0.0
    initial_resolution: tuple[float, float, float] = (0.05, 1.0, 0.0)
    grid_step_kev: float = DEFAULT_GRID_STEP_KEV
    dietze_klein_detector: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "datasets", tuple(self.datasets))
        object.__setattr__(self, "resolution_terms", tuple(self.resolution_terms))
        if not self.datasets:
            raise ValueError("a Compton-edge fit needs at least one dataset")
        if not set(self.resolution_terms) <= set(_RESOLUTION_NAMES):
            raise ValueError("resolution terms must be chosen from a, b and c")
        if self.statistic not in FIT_STATISTICS:
            raise ValueError(f"unsupported fit statistic {self.statistic!r}")
        if self.amplitude_mode not in {"free", "linked"}:
            raise ValueError(f"unsupported amplitude mode {self.amplitude_mode!r}")
        if self.initial_gain is not None and (
            not math.isfinite(self.initial_gain) or self.initial_gain <= 0
        ):
            raise ValueError("initial gain must be positive and finite")
        if not math.isfinite(self.initial_offset):
            raise ValueError("initial offset must be finite")
        if len(self.initial_resolution) != 3 or not all(
            math.isfinite(value) and value >= 0 for value in self.initial_resolution
        ):
            raise ValueError("initial resolution terms must be three finite values >= 0")
        for name, value in zip(_RESOLUTION_NAMES, self.initial_resolution, strict=True):
            low, high = _RESOLUTION_BOUNDS[name]
            if not low <= value <= high:
                raise ValueError(f"initial resolution term {name} must be within [{low}, {high}]")
        if not 0 < self.grid_step_kev <= 1.0:
            raise ValueError("the electron-energy grid step must be within (0, 1] keV")
        if self.dietze_klein_detector is not None:
            dietze_klein_table.detector(self.dietze_klein_detector)


# ---------------------------------------------------------------------- results


@dataclass(frozen=True)
class ResolutionTerm:
    """One resolution term in the fitted σ convention, with its FWHM equivalent."""

    name: str
    sigma_value: float
    sigma_stderr: float | None
    varying: bool
    unit: str

    @property
    def fwhm_value(self) -> float:
        return FWHM_PER_SIGMA * self.sigma_value

    @property
    def fwhm_stderr(self) -> float | None:
        return None if self.sigma_stderr is None else FWHM_PER_SIGMA * self.sigma_stderr


@dataclass(frozen=True)
class ComptonEdgeLineResult:
    dataset_index: int
    dataset_label: str
    label: str
    nuclide: str | None
    energy_kev: float
    edge_kev: float
    edge_x: float
    edge_x_stderr: float | None
    amplitude: float
    amplitude_stderr: float | None
    sigma_over_energy: float
    fwhm_over_energy: float
    in_fit_range: bool
    response: str


@dataclass(frozen=True)
class DietzeKleinEstimate:
    """Edge estimate from Dietze & Klein Table 3 ratios at the measured resolution."""

    detector: int
    detector_description: str
    source: str
    resolution_fwhm: float
    lmax_over_edge: float
    lhalf_over_edge: float
    edge_from_lmax: float
    edge_from_lhalf: float
    edge_x: float
    edge_height_fraction: float | None
    citation: str = dietze_klein_table.CITATION


@dataclass(frozen=True)
class ComptonShapeDiagnostics:
    """Model-independent shape points of a single-line Compton continuum."""

    line_energy_kev: float
    background_floor: float
    floor_method: str
    smoothing_sigma_x: float
    l_max: float
    l_half: float
    maximum_height: float
    dk_resolution_fwhm: float
    fitted_edge_x: float
    edge_height_fraction: float | None
    dietze_klein: DietzeKleinEstimate | None
    notes: tuple[str, ...] = ()

    @property
    def fitted_minus_dk_relative(self) -> float | None:
        if self.dietze_klein is None or self.dietze_klein.edge_x <= 0:
            return None
        return (self.fitted_edge_x - self.dietze_klein.edge_x) / self.dietze_klein.edge_x


@dataclass(frozen=True)
class ComptonEdgeDatasetResult:
    index: int
    label: str
    spectrum_id: str
    background: str
    fit_min: float
    fit_max: float
    x: np.ndarray
    bin_edges: np.ndarray
    observed: np.ndarray
    best_fit: np.ndarray
    residual: np.ndarray
    components: dict[str, np.ndarray]
    statistic_value: float
    diagnostics: ComptonShapeDiagnostics | None


@dataclass(frozen=True)
class ComptonEdgeFitResult:
    success: bool
    message: str
    statistic: ResolvedFitStatistic
    gain: float
    gain_stderr: float | None
    offset: float
    offset_stderr: float | None
    offset_free: bool
    gain_offset_covariance: np.ndarray
    resolution: tuple[ResolutionTerm, ...]
    parameters: tuple[FitParameter, ...]
    datasets: tuple[ComptonEdgeDatasetResult, ...]
    lines: tuple[ComptonEdgeLineResult, ...]
    deviance: float
    degrees_of_freedom: int
    reduced_deviance: float
    aic: float
    bic: float
    evaluations: int
    evaluation_path: EvaluationPath
    edge_span_kev: tuple[float, float]
    warnings: tuple[str, ...]
    energy_unit: EnergyUnit = COMPTON_EDGE_ENERGY_UNIT

    def light_output_kevee(self, x: float | np.ndarray) -> float | np.ndarray:
        """Convert spectrum x coordinates to keVee with the fitted linear map."""
        value = (np.asarray(x, dtype=np.float64) - self.offset) / self.gain
        return float(value) if np.ndim(value) == 0 else value

    def edge_position(self, edge_kev: float) -> tuple[float, float | None]:
        """Return ``g·T + o`` and its uncertainty propagated from cov(g, o)."""
        return _edge_position(self.gain, self.offset, self.gain_offset_covariance, edge_kev)

    def resolution_at(self, energy_kevee: float) -> float:
        """Return σ_E/E at one light output."""
        a, b, c = (term.sigma_value for term in self.resolution)
        return math.sqrt(a**2 + b**2 / energy_kevee + c**2 / energy_kevee**2)


@dataclass(frozen=True)
class SummaryRow:
    """One row of the Compton-edge result table (GUI, clipboard and CSV export)."""

    section: str
    quantity: str
    value: float | None
    stderr: float | None
    unit: str
    note: str = ""


# ------------------------------------------------------------------ public API


def fit_compton_edges(
    spec: ComptonEdgeFitSpec,
    *,
    cancelled: Callable[[], bool] | None = None,
    progress: ComptonEdgeProgressCallback | None = None,
) -> ComptonEdgeFitResult:
    """Fit Klein–Nishina (or simulated) Compton continua jointly to frozen spectra.

    ``progress`` is called synchronously on the fitting thread, once per stage
    change, gain candidate and residual evaluation; callers that cross threads
    should throttle it.
    """
    cancel_check = cancelled or (lambda: False)
    report = progress or _no_progress
    report(ComptonEdgeFitProgress("prepare"))
    model = _ComptonModel(spec)
    initial = model.initial_values(cancel_check, report)
    # initial_values() also sets the gain/offset bounds and fixes unfitted terms.
    layout = model.layout
    free = layout.vary
    lower = layout.lower[free]
    upper = layout.upper[free]
    x0 = _strictly_inside(initial[free], lower, upper)

    def full(values: np.ndarray) -> np.ndarray:
        parameters = initial.copy()
        parameters[free] = values
        return parameters

    evaluations = 0

    def fun(values: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        if cancel_check():
            raise _CancelledError
        residuals = model.residuals(full(values))
        # Counts the same residual calls as least_squares' nfev (result.evaluations).
        evaluations += 1
        report(ComptonEdgeFitProgress("optimize", evaluations))
        return residuals

    def jac(values: np.ndarray) -> np.ndarray:
        if cancel_check():
            raise _CancelledError
        return model.jacobian(full(values))[:, free]

    from scipy.optimize import least_squares

    report(ComptonEdgeFitProgress("optimize"))
    try:
        solution = least_squares(
            fun,
            x0,
            jac=jac,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=2_000,
        )
    except _CancelledError as exc:
        raise ComptonEdgeFitCancelledError("Compton-edge fit cancelled") from exc
    if cancel_check():
        raise ComptonEdgeFitCancelledError("Compton-edge fit cancelled")
    report(ComptonEdgeFitProgress("finalize"))
    return model.result(full(np.asarray(solution.x, dtype=np.float64)), solution)


def compton_edge_calibration_points(result: ComptonEdgeFitResult) -> tuple[CalibrationPoint, ...]:
    """Convert fitted edges to energy-calibration points in keVee.

    One point is produced per distinct line energy, because all datasets share
    the same gain and offset. When the offset was fixed, the constraint is
    carried as an explicit zero-light point at that offset so that a linear
    calibration can be fitted from a single edge.
    """
    points: list[CalibrationPoint] = []
    seen: set[float] = set()
    for line in sorted(result.lines, key=lambda item: item.energy_kev):
        key = round(line.energy_kev, 6)
        if key in seen:
            continue
        seen.add(key)
        points.append(
            CalibrationPoint(
                channel=line.edge_x,
                energy_kev=line.edge_kev,
                label=f"Compton edge {line.energy_kev:g} keV",
                source=COMPTON_EDGE_POINT_SOURCE,
            )
        )
    if not result.offset_free:
        points.insert(
            0,
            CalibrationPoint(
                channel=result.offset,
                energy_kev=0.0,
                label=f"Zero light output (offset fixed at {result.offset:g})",
                source=f"{COMPTON_EDGE_POINT_SOURCE} (fixed offset)",
            ),
        )
    return tuple(points)


def extrapolation_warning(
    edge_span_kev: tuple[float, float],
    low_kev: float,
    high_kev: float,
) -> str | None:
    """Warn when a keVee scale is used outside the span of the fitted edges."""
    span_low, span_high = sorted(edge_span_kev)
    low, high = sorted((float(low_kev), float(high_kev)))
    if low >= span_low - 1.0e-9 and high <= span_high + 1.0e-9:
        return None
    return (
        f"The keVee scale is constrained by Compton edges between {span_low:.4g} and "
        f"{span_high:.4g} keVee; values from {low:.4g} to {high:.4g} keVee outside that "
        "span are extrapolated."
    )


def compton_shape_diagnostics(
    x: np.ndarray,
    counts: np.ndarray,
    *,
    edge_x: float,
    edge_sigma_x: float,
    line_energy_kev: float,
    dietze_klein_detector: int | None = None,
) -> ComptonShapeDiagnostics:
    """Measure the Compton maximum, half height and edge height of one continuum.

    The spectrum is lightly smoothed with a Gaussian of width ``σ/14``, where
    ``σ`` is the resolution at the edge in x units. This broadens the edge by
    only 0.25 % in quadrature. The Compton maximum is broad, so ``L_max`` moves
    by about ±1 % between this and several times heavier smoothing, while
    ``L_1/2`` is stable to 0.2 %. The background floor is the median of the
    smoothed spectrum more than 3σ above the edge, or its minimum above the
    maximum when the window ends earlier. ``L_max`` is the highest point within
    ``[edge − 4σ, edge + σ]``, refined by a three-point parabola. ``L_1/2`` is the
    first point above ``L_max`` where the floor-subtracted spectrum falls to half
    that height. All positions are in the spectrum's x units and assume x = 0 at
    zero light output.
    """
    from scipy.ndimage import gaussian_filter1d

    x = np.asarray(x, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if len(x) != len(counts) or len(x) < 7:
        raise ValueError("shape diagnostics need at least seven bins")
    bin_width = float(np.median(np.diff(x)))
    sigma = abs(float(edge_sigma_x))
    if not math.isfinite(sigma) or sigma <= 0 or bin_width <= 0:
        raise ValueError("shape diagnostics need a positive resolution and bin width")
    smoothing = sigma / _DIAGNOSTIC_SMOOTHING_DIVISOR
    smoothed = (
        np.asarray(gaussian_filter1d(counts, smoothing / bin_width, mode="nearest"),
                   dtype=np.float64)
        if smoothing / bin_width >= 0.3
        else counts.copy()
    )
    notes: list[str] = []
    high_side = x >= edge_x + 3.0 * sigma
    if np.count_nonzero(high_side) >= 5:
        floor = float(np.median(smoothed[high_side]))
        floor_method = "median above edge + 3σ"
    else:
        floor = float("nan")
        floor_method = "minimum above the maximum"
    search = (x >= edge_x - 4.0 * sigma) & (x <= edge_x + sigma)
    if not np.any(search):
        raise ValueError("the fit window does not contain the Compton maximum region")
    indices = np.flatnonzero(search)
    peak_index = int(indices[np.argmax(smoothed[indices])])
    if not math.isfinite(floor):
        floor = float(np.min(smoothed[peak_index:]))
        notes.append("Fit window ends less than 3σ above the edge; floor is a minimum.")
    net = smoothed - floor
    l_max = float(x[peak_index])
    if 0 < peak_index < len(x) - 1:
        left, centre, right = net[peak_index - 1 : peak_index + 2]
        curvature = left - 2.0 * centre + right
        if curvature < 0:
            shift = 0.5 * (left - right) / curvature
            l_max = float(x[peak_index] + np.clip(shift, -1.0, 1.0) * bin_width)
    height = float(net[peak_index])
    if height <= 0:
        raise ValueError("no Compton maximum above the background floor")
    half = 0.5 * height
    below = np.flatnonzero(net[peak_index:] <= half)
    if not len(below):
        raise ValueError("the spectrum does not fall to half the Compton maximum in the window")
    crossing = peak_index + int(below[0])
    x0, x1 = x[crossing - 1], x[crossing]
    y0, y1 = net[crossing - 1], net[crossing]
    l_half = float(x0 + (half - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)
    resolution = 1.5 * (l_half - l_max) / l_half if l_half > 0 else float("nan")

    def height_at(position: float) -> float | None:
        if not x[0] <= position <= x[-1]:
            return None
        return float(np.interp(position, x, net)) / height

    edge_height = height_at(float(edge_x))
    if edge_height is None:
        notes.append("The fitted edge lies outside the fit window.")
    estimate: DietzeKleinEstimate | None = None
    if dietze_klein_detector is not None:
        block = dietze_klein_table.block_for(line_energy_kev, dietze_klein_detector)
        percent = 100.0 * resolution
        if block is None:
            notes.append(
                f"Dietze–Klein Table 3 has no {line_energy_kev:g} keV block for this "
                "detector; only Cs-137 and the Na-22 511 keV line are tabulated here."
            )
        elif not (
            block.resolution_fwhm_percent[0] <= percent <= block.resolution_fwhm_percent[-1]
        ):
            notes.append(
                f"Resolution {percent:.3g} % FWHM is outside Dietze–Klein Table 3 "
                f"({block.resolution_fwhm_percent[0]:g}–{block.resolution_fwhm_percent[-1]:g} %)."
            )
        else:
            lmax_ratio = float(np.interp(percent, block.resolution_fwhm_percent,
                                         block.lmax_over_edge))
            lhalf_ratio = float(np.interp(percent, block.resolution_fwhm_percent,
                                          block.lhalf_over_edge))
            from_lmax = l_max / lmax_ratio
            from_lhalf = l_half / lhalf_ratio
            dk_edge = 0.5 * (from_lmax + from_lhalf)
            estimate = DietzeKleinEstimate(
                detector=block.detector,
                detector_description=dietze_klein_table.detector(block.detector).description,
                source=block.source,
                resolution_fwhm=resolution,
                lmax_over_edge=lmax_ratio,
                lhalf_over_edge=lhalf_ratio,
                edge_from_lmax=from_lmax,
                edge_from_lhalf=from_lhalf,
                edge_x=dk_edge,
                edge_height_fraction=height_at(dk_edge),
            )
    return ComptonShapeDiagnostics(
        line_energy_kev=float(line_energy_kev),
        background_floor=floor,
        floor_method=floor_method,
        smoothing_sigma_x=smoothing,
        l_max=l_max,
        l_half=l_half,
        maximum_height=height,
        dk_resolution_fwhm=resolution,
        fitted_edge_x=float(edge_x),
        edge_height_fraction=edge_height,
        dietze_klein=estimate,
        notes=tuple(notes),
    )


def folded_line_response(
    bin_edges_x: np.ndarray,
    line: ComptonLine,
    *,
    gain: float,
    offset: float,
    resolution: tuple[float, float, float],
    grid_step_kev: float = DEFAULT_GRID_STEP_KEV,
) -> np.ndarray:
    """Return the expected fraction of one line's Compton events in each bin.

    This is the exact (grid × bin-edge) evaluation used by the fit.
    """
    edges = np.asarray(bin_edges_x, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin edges must be strictly increasing")
    if not gain > 0:
        raise ValueError("gain must be positive")
    centres, weights = _line_grid(line, grid_step_kev, _grid_cells((line,), grid_step_kev))
    a, b, c = resolution
    u = (edges - offset) / gain
    tables = _fold_tables(centres, weights[None, :], a, b, c, u, derivatives=False)
    return np.asarray(np.diff(tables.cumulative[0]), dtype=np.float64)


def compton_edge_summary_rows(result: ComptonEdgeFitResult) -> tuple[SummaryRow, ...]:
    """Flatten a fit result into the rows shown in the GUI and exported as CSV."""
    rows: list[SummaryRow] = [
        SummaryRow("model", "gain g", result.gain, result.gain_stderr, "x/keVee"),
        SummaryRow(
            "model",
            "offset o",
            result.offset,
            result.offset_stderr,
            "x",
            "" if result.offset_free else "fixed",
        ),
    ]
    for term in result.resolution:
        state = "" if term.varying else "fixed"
        rows.append(
            SummaryRow(
                "resolution",
                f"{term.name} (σ)",
                term.sigma_value,
                term.sigma_stderr,
                term.unit,
                state,
            )
        )
        rows.append(
            SummaryRow(
                "resolution",
                f"{'αβγ'['abc'.index(term.name)]} (FWHM)",
                term.fwhm_value,
                term.fwhm_stderr,
                term.unit,
                "= 2.3548 × σ term (Dietze & Klein convention)",
            )
        )
    for line in result.lines:
        prefix = f"[{line.dataset_index + 1}] {line.label}"
        rows.append(
            SummaryRow(
                "edge",
                f"{prefix} edge",
                line.edge_x,
                line.edge_x_stderr,
                "x",
                f"T_max = {line.edge_kev:.4f} keVee; {line.response}"
                + ("" if line.in_fit_range else "; outside fit range"),
            )
        )
        rows.append(
            SummaryRow("edge", f"{prefix} σ/E at edge", line.sigma_over_energy, None, "")
        )
        rows.append(
            SummaryRow("edge", f"{prefix} FWHM/E at edge", line.fwhm_over_energy, None, "")
        )
        rows.append(
            SummaryRow(
                "edge", f"{prefix} amplitude", line.amplitude, line.amplitude_stderr, "counts"
            )
        )
    for dataset in result.datasets:
        diagnostics = dataset.diagnostics
        if diagnostics is None:
            continue
        prefix = f"[{dataset.index + 1}] {diagnostics.line_energy_kev:g} keV"
        rows.extend(
            (
                SummaryRow("diagnostics", f"{prefix} L_max", diagnostics.l_max, None, "x"),
                SummaryRow("diagnostics", f"{prefix} L_1/2", diagnostics.l_half, None, "x"),
                SummaryRow(
                    "diagnostics",
                    f"{prefix} ΔL/L (FWHM, DK rule)",
                    diagnostics.dk_resolution_fwhm,
                    None,
                    "",
                    "1.5 (L_1/2 − L_max)/L_1/2",
                ),
                SummaryRow(
                    "diagnostics",
                    f"{prefix} fitted-edge height",
                    diagnostics.edge_height_fraction,
                    None,
                    "of maximum",
                    "coincidence measurements: 0.74–0.82",
                ),
            )
        )
        estimate = diagnostics.dietze_klein
        if estimate is not None:
            difference = diagnostics.fitted_minus_dk_relative
            rows.extend(
                (
                    SummaryRow(
                        "diagnostics",
                        f"{prefix} Dietze–Klein edge",
                        estimate.edge_x,
                        None,
                        "x",
                        f"detector {estimate.detector}; mean of L_max/"
                        f"{estimate.lmax_over_edge:.4f} and L_1/2/"
                        f"{estimate.lhalf_over_edge:.4f}",
                    ),
                    SummaryRow(
                        "diagnostics",
                        f"{prefix} DK-edge height",
                        estimate.edge_height_fraction,
                        None,
                        "of maximum",
                    ),
                    SummaryRow(
                        "diagnostics",
                        f"{prefix} fitted − DK edge",
                        difference,
                        None,
                        "relative",
                    ),
                )
            )
    rows.extend(
        (
            SummaryRow("fit", "statistic", None, None, "", result.statistic),
            SummaryRow("fit", "deviance", result.deviance, None, ""),
            SummaryRow("fit", "degrees of freedom", float(result.degrees_of_freedom), None, ""),
            SummaryRow("fit", "reduced deviance", result.reduced_deviance, None, ""),
            SummaryRow("fit", "AIC", result.aic, None, ""),
            SummaryRow("fit", "BIC", result.bic, None, ""),
        )
    )
    return tuple(rows)


def export_compton_edge_result(
    path: Path,
    *,
    spec: ComptonEdgeFitSpec,
    result: ComptonEdgeFitResult,
) -> None:
    """Export a Compton-edge fit as JSON or CSV without overwriting."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        document = {
            "kind": "nlab_compton_edge_fit",
            "energy_unit": result.energy_unit,
            "model": {
                "description": "Klein–Nishina electron spectrum folded with "
                "sigma_E/E = sqrt(a^2 + b^2/E + c^2/E^2); x = g*E + o",
                "offset_free": spec.offset_free,
                "resolution_terms": list(spec.resolution_terms),
                "statistic": spec.statistic,
                "amplitude_mode": spec.amplitude_mode,
                "grid_step_kev": spec.grid_step_kev,
                "dietze_klein_detector": spec.dietze_klein_detector,
                "datasets": [
                    {
                        "spectrum_id": dataset.spectrum.spectrum_id,
                        "label": dataset.spectrum.label,
                        "source": dataset.spectrum.source,
                        "metadata": dataset.spectrum.metadata,
                        "history": dataset.spectrum.history,
                        "fit_min": dataset.fit_min,
                        "fit_max": dataset.fit_max,
                        "background": dataset.background,
                        "lines": [
                            {
                                "energy_kev": line.energy_kev,
                                "label": line.label,
                                "nuclide": line.nuclide,
                                "emission_probability_percent": (
                                    line.emission_probability_percent
                                ),
                                "response": line.shape.label,
                            }
                            for line in dataset.lines
                        ],
                    }
                    for dataset in spec.datasets
                ],
            },
            "statistics": {
                "success": result.success,
                "message": result.message,
                "statistic": result.statistic,
                "deviance": result.deviance,
                "degrees_of_freedom": result.degrees_of_freedom,
                "reduced_deviance": result.reduced_deviance,
                "aic": result.aic,
                "bic": result.bic,
                "evaluations": result.evaluations,
                "evaluation_path": result.evaluation_path,
                "warnings": list(result.warnings),
            },
            "calibration": {
                "gain_x_per_kevee": result.gain,
                "gain_stderr": result.gain_stderr,
                "offset_x": result.offset,
                "offset_stderr": result.offset_stderr,
                "offset_free": result.offset_free,
                "gain_offset_covariance": result.gain_offset_covariance.tolist(),
                "edge_span_kevee": list(result.edge_span_kev),
            },
            "resolution": [
                {
                    "term": term.name,
                    "sigma_value": term.sigma_value,
                    "sigma_stderr": term.sigma_stderr,
                    "fwhm_value": term.fwhm_value,
                    "fwhm_stderr": term.fwhm_stderr,
                    "unit": term.unit,
                    "varying": term.varying,
                }
                for term in result.resolution
            ],
            "lines": [_line_document(line) for line in result.lines],
            "diagnostics": [
                _diagnostics_document(dataset.index, dataset.diagnostics)
                for dataset in result.datasets
                if dataset.diagnostics is not None
            ],
            "parameters": [
                {
                    "name": parameter.name,
                    "value": parameter.value,
                    "stderr": parameter.stderr,
                    "minimum": parameter.minimum,
                    "maximum": parameter.maximum,
                    "varying": parameter.varying,
                    "expression": parameter.expression,
                }
                for parameter in result.parameters
            ],
            "references": list(REFERENCES),
        }
        with path.open("x", encoding="utf-8") as output:
            json.dump(document, output, indent=2, default=str)
        return
    if path.suffix.lower() == ".csv":
        with path.open("x", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("section", "quantity", "value", "stderr", "unit", "note"))
            for row in compton_edge_summary_rows(result):
                writer.writerow(
                    (
                        row.section,
                        row.quantity,
                        "" if row.value is None else repr(float(row.value)),
                        "" if row.stderr is None else repr(float(row.stderr)),
                        row.unit,
                        row.note,
                    )
                )
        return
    raise ValueError("Compton-edge export must use .json or .csv")


# ------------------------------------------------------------------- internals


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _centres_to_edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=np.float64)
    if len(centres) < 2:
        raise ValueError("at least two bin centres are needed to infer bin edges")
    middle = 0.5 * (centres[1:] + centres[:-1])
    first = centres[0] - 0.5 * (centres[1] - centres[0])
    last = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return np.concatenate(([first], middle, [last]))


def _edge_position(
    gain: float,
    offset: float,
    covariance: np.ndarray,
    edge_kev: float,
) -> tuple[float, float | None]:
    position = gain * edge_kev + offset
    if not np.all(np.isfinite(covariance)):
        return position, None
    variance = (
        edge_kev**2 * covariance[0, 0] + covariance[1, 1] + 2.0 * edge_kev * covariance[0, 1]
    )
    return position, math.sqrt(variance) if variance >= 0 else None


def _strictly_inside(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    width = np.where(np.isfinite(upper - lower), upper - lower, np.abs(result) + 1.0)
    margin = np.maximum(width * 1.0e-6, 1.0e-12)
    low = np.where(np.isfinite(lower), lower + margin, -np.inf)
    high = np.where(np.isfinite(upper), upper - margin, np.inf)
    return np.asarray(np.clip(result, low, high), dtype=np.float64)


def _grid_cells(lines: Sequence[ComptonLine], step: float) -> int:
    upper = max(line.shape.max_energy_kev for line in lines)
    return int(math.ceil(upper / step)) + 2


def _line_grid(line: ComptonLine, step: float, cells: int) -> tuple[np.ndarray, np.ndarray]:
    centres = (np.arange(cells, dtype=np.float64) + 0.5) * step
    shape = line.shape
    if isinstance(shape, KleinNishinaResponse):
        weights = _kn_grid_weights(float(shape.gamma_energy_kev), float(step), int(cells))
    else:
        edges = np.arange(cells + 1, dtype=np.float64) * step
        weights = np.asarray(shape.weights(edges), dtype=np.float64)
        if weights.shape != (cells,) or np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError(f"response shape for {line.label} returned invalid weights")
    return centres, weights


@dataclass(frozen=True)
class _FoldTables:
    cumulative: np.ndarray
    density: np.ndarray
    d_a: np.ndarray | None
    d_b: np.ndarray | None
    d_c: np.ndarray | None


def _fold_tables(
    centres: np.ndarray,
    weights: np.ndarray,
    a: float,
    b: float,
    c: float,
    nodes: np.ndarray,
    *,
    derivatives: bool,
) -> _FoldTables:
    """Evaluate the folded cumulative response of every line at light-output nodes.

    ``F_l(u) = Σ_T w_l(T) Φ((u − T)/s(T))`` with ``s(T) = T·σ_E/E(T)``. Rows whose
    Gaussian lies entirely below the nodes only add a constant, which cancels in
    bin differences, and rows entirely above add zero; both are skipped.
    """
    from scipy.special import ndtr

    relative_sq = np.maximum(a**2 + b**2 / centres + c**2 / centres**2, 1.0e-300)
    width = np.maximum(centres * np.sqrt(relative_sq), 1.0e-9)
    low = float(np.min(nodes))
    high = float(np.max(nodes))
    keep = (
        np.any(weights > 0, axis=0)
        & (centres + _GAUSS_SPAN * width > low)
        & (centres - _GAUSS_SPAN * width < high)
    )
    lines = weights.shape[0]
    if not np.any(keep):
        zeros = np.zeros((lines, len(nodes)), dtype=np.float64)
        return _FoldTables(zeros, zeros, zeros, zeros, zeros) if derivatives else _FoldTables(
            zeros, zeros, None, None, None
        )
    t = centres[keep]
    s = width[keep]
    w = weights[:, keep]
    z = (nodes[None, :] - t[:, None]) / s[:, None]
    cumulative = w @ ndtr(z)
    gaussian = np.exp(-0.5 * z * z) * (1.0 / math.sqrt(2.0 * math.pi))
    density = (w / s[None, :]) @ gaussian
    if not derivatives:
        return _FoldTables(cumulative, density, None, None, None)
    gaussian_z = gaussian * z
    r2 = relative_sq[keep]
    d_a = -(w * (a / r2)[None, :]) @ gaussian_z
    d_b = -(w * (b / (t * r2))[None, :]) @ gaussian_z
    d_c = -(w * (c / (t * t * r2))[None, :]) @ gaussian_z
    return _FoldTables(cumulative, density, d_a, d_b, d_c)


@dataclass
class _Layout:
    names: list[str] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)
    varying: list[bool] = field(default_factory=list)

    def add(self, name: str, lower: float, upper: float, *, vary: bool = True) -> int:
        self.names.append(name)
        self.lower.append(lower)
        self.upper.append(upper)
        self.varying.append(vary)
        return len(self.names) - 1

    def freeze(self) -> _FrozenLayout:
        return _FrozenLayout(
            names=tuple(self.names),
            lower=np.asarray(self.lower, dtype=np.float64),
            upper=np.asarray(self.upper, dtype=np.float64),
            vary=np.asarray(self.varying, dtype=bool),
        )


@dataclass(frozen=True)
class _FrozenLayout:
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    vary: np.ndarray


@dataclass
class _Prepared:
    index: int
    dataset: ComptonEdgeDataset
    x: np.ndarray
    edges: np.ndarray
    y: np.ndarray
    variances: np.ndarray
    line_rows: np.ndarray
    coefficients: np.ndarray
    amplitude_names: list[str]
    amplitude_index: list[int]
    background_index: dict[str, int]
    x_ref: float
    span: float
    node_slice: slice = slice(0, 0)


@dataclass(frozen=True)
class _Evaluation:
    models: tuple[np.ndarray, ...]
    responses: tuple[np.ndarray, ...]
    unit_responses: tuple[np.ndarray, ...]
    jacobians: tuple[np.ndarray, ...]


_GAIN, _OFFSET, _RES_A, _RES_B, _RES_C = range(5)


def _safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_") or "line"


class _ComptonModel:
    def __init__(self, spec: ComptonEdgeFitSpec) -> None:
        self.spec = spec
        self.step = float(spec.grid_step_kev)
        all_lines = [line for dataset in spec.datasets for line in dataset.lines]
        self.cells = _grid_cells(all_lines, self.step)
        rows: list[np.ndarray] = []
        centres = (np.arange(self.cells, dtype=np.float64) + 0.5) * self.step
        for line in all_lines:
            _, weights = _line_grid(line, self.step, self.cells)
            rows.append(weights)
        self.centres = centres
        self.weights = np.vstack(rows)
        self.datasets: list[_Prepared] = []
        layout = _Layout()
        layout.add("gain", 0.0, np.inf)
        layout.add("offset", -np.inf, np.inf, vary=spec.offset_free)
        for name in _RESOLUTION_NAMES:
            low, high = _RESOLUTION_BOUNDS[name]
            layout.add(f"res_{name}", low, high, vary=name in spec.resolution_terms)
        statistics: list[ResolvedFitStatistic] = []
        row = 0
        for index, dataset in enumerate(spec.datasets):
            prepared, statistic = self._prepare(index, dataset, layout, row)
            statistics.append(statistic)
            row += len(dataset.lines)
            self.datasets.append(prepared)
        if "weighted" in statistics:
            self.statistic: ResolvedFitStatistic = "weighted"
        elif "unweighted" in statistics:
            self.statistic = "unweighted"
        else:
            self.statistic = "poisson"
        self.layout = layout.freeze()
        node_count = 0
        for prepared in self.datasets:
            prepared.node_slice = slice(node_count, node_count + len(prepared.edges))
            node_count += len(prepared.edges)
        self.node_count = node_count
        self.path: EvaluationPath = "direct"
        self.tabulation_step = _TABULATION_STEPS_KEV[-1]
        self._cache_key: bytes | None = None
        self._cache: _Evaluation | None = None
        self.evaluations = 0
        x_extent = max(
            max(abs(float(item.edges[0])), abs(float(item.edges[-1]))) for item in self.datasets
        )
        self.offset_bound = max(0.5 * x_extent, 1.0)

    def _prepare(
        self,
        index: int,
        dataset: ComptonEdgeDataset,
        layout: _Layout,
        first_row: int,
    ) -> tuple[_Prepared, ResolvedFitStatistic]:
        spectrum = dataset.spectrum
        if len(spectrum.x) < 2:
            raise ValueError(f"dataset {index + 1} spectrum needs at least two bins")
        all_edges = _centres_to_edges(spectrum.x)
        mask = np.ones(len(spectrum.x), dtype=bool)
        if dataset.fit_min is not None:
            mask &= spectrum.x >= float(dataset.fit_min)
        if dataset.fit_max is not None:
            mask &= spectrum.x <= float(dataset.fit_max)
        selected = np.flatnonzero(mask)
        if not len(selected):
            raise ValueError(f"the fit range of dataset {index + 1} contains no bins")
        if len(selected) < _MIN_DATASET_BINS:
            raise ValueError(
                f"the fit range of dataset {index + 1} contains too few bins "
                f"({len(selected)} < {_MIN_DATASET_BINS})"
            )
        start, stop = int(selected[0]), int(selected[-1]) + 1
        x = np.asarray(spectrum.x[start:stop], dtype=np.float64)
        y = np.asarray(spectrum.counts[start:stop], dtype=np.float64)
        variances = np.asarray(spectrum.variances[start:stop], dtype=np.float64)
        edges = np.asarray(all_edges[start : stop + 1], dtype=np.float64)
        if not np.any(y != y[0]):
            raise ValueError(f"the fit range of dataset {index + 1} contains no spectral structure")
        statistic = resolve_statistic(spectrum, self.spec.statistic, y)
        lines = dataset.lines
        groups: dict[str, list[int]] = {}
        if self.spec.amplitude_mode == "linked":
            for line_index, line in enumerate(lines):
                if line.nuclide and line.emission_probability_percent is not None:
                    groups.setdefault(line.nuclide, []).append(line_index)
        columns: list[np.ndarray] = []
        names: list[str] = []
        amplitude_index: list[int] = []
        prefix = f"d{index}_"
        for nuclide, members in groups.items():
            column = np.zeros(len(lines), dtype=np.float64)
            for member in members:
                probability = lines[member].emission_probability_percent
                assert probability is not None
                column[member] = probability / 100.0
            name = f"{prefix}{_safe_name(nuclide)}_scale"
            columns.append(column)
            names.append(name)
            amplitude_index.append(layout.add(name, 0.0, np.inf))
        grouped = {member for members in groups.values() for member in members}
        for line_index, line in enumerate(lines):
            if line_index in grouped:
                continue
            column = np.zeros(len(lines), dtype=np.float64)
            column[line_index] = 1.0
            name = f"{prefix}{_safe_name(line.label)}_amplitude"
            while name in layout.names:
                name += "_"
            columns.append(column)
            names.append(name)
            amplitude_index.append(layout.add(name, 0.0, np.inf))
        span = float(edges[-1] - edges[0])
        x_ref = float(np.mean(x))
        background_index: dict[str, int] = {}
        if dataset.background == "constant":
            background_index["c"] = layout.add(f"{prefix}bg_c", 0.0, np.inf)
        elif dataset.background == "linear":
            background_index["intercept"] = layout.add(f"{prefix}bg_intercept", -np.inf, np.inf)
            background_index["slope"] = layout.add(f"{prefix}bg_slope", -np.inf, np.inf)
            background_index["x_ref"] = layout.add(f"{prefix}bg_x_ref", x_ref, x_ref, vary=False)
        elif dataset.background == "exponential":
            background_index["amplitude"] = layout.add(f"{prefix}bg_amplitude", 0.0, np.inf)
            background_index["rate"] = layout.add(
                f"{prefix}bg_rate", -20.0 / span, 20.0 / span
            )
            background_index["x_ref"] = layout.add(f"{prefix}bg_x_ref", x_ref, x_ref, vary=False)
        prepared = _Prepared(
            index=index,
            dataset=dataset,
            x=x,
            edges=edges,
            y=y,
            variances=variances,
            line_rows=np.arange(first_row, first_row + len(lines)),
            coefficients=np.column_stack(columns),
            amplitude_names=names,
            amplitude_index=amplitude_index,
            background_index=background_index,
            x_ref=x_ref,
            span=span,
        )
        return prepared, statistic

    # ---------------------------------------------------------- evaluation

    def _tables(
        self,
        gain: float,
        offset: float,
        a: float,
        b: float,
        c: float,
        *,
        derivatives: bool,
    ) -> tuple[_FoldTables, np.ndarray | None]:
        if self.path == "direct":
            nodes = np.concatenate([(item.edges - offset) / gain for item in self.datasets])
            return (
                _fold_tables(self.centres, self.weights, a, b, c, nodes, derivatives=derivatives),
                None,
            )
        low = min(float(item.edges[0] - offset) / gain for item in self.datasets)
        high = max(float(item.edges[-1] - offset) / gain for item in self.datasets)
        step = self.tabulation_step
        start = math.floor(low / step) * step - 2.0 * step
        count = int(math.ceil((high - start) / step)) + 3
        grid = start + step * np.arange(count, dtype=np.float64)
        return (
            _fold_tables(self.centres, self.weights, a, b, c, grid, derivatives=derivatives),
            grid,
        )

    def choose_path(self, gain: float, offset: float, a: float, b: float, c: float) -> None:
        """Choose exact bin-edge nodes or a Hermite-interpolated light-output grid.

        The grid step is fixed for the whole fit from the starting resolution:
        at least eight nodes per Gaussian σ at the lowest light output in the
        windows (2, 1 or 0.5 keVee). Tests check it against the exact path.
        """
        low = min(float(item.edges[0] - offset) / gain for item in self.datasets)
        high = max(float(item.edges[-1] - offset) / gain for item in self.datasets)
        light = max(low, 1.0)
        width = light * math.sqrt(a**2 + b**2 / light + c**2 / light**2)
        self.tabulation_step = next(
            (
                step
                for step in _TABULATION_STEPS_KEV
                if width >= _TABULATION_POINTS_PER_SIGMA * step
            ),
            _TABULATION_STEPS_KEV[-1],
        )
        tabulated_nodes = (high - low) / self.tabulation_step + 5.0
        self.path = "direct" if self.node_count <= 1.25 * tabulated_nodes else "tabulated"

    def evaluate(self, values: np.ndarray) -> _Evaluation:
        key = np.asarray(values, dtype=np.float64).tobytes()
        if key == self._cache_key and self._cache is not None:
            return self._cache
        self.evaluations += 1
        gain, offset, a, b, c = (float(values[index]) for index in range(5))
        tables, grid = self._tables(gain, offset, a, b, c, derivatives=True)
        assert tables.d_a is not None and tables.d_b is not None and tables.d_c is not None
        models: list[np.ndarray] = []
        responses: list[np.ndarray] = []
        unit_responses: list[np.ndarray] = []
        jacobians: list[np.ndarray] = []
        parameter_count = len(values)
        for item in self.datasets:
            u = (item.edges - offset) / gain
            rows = item.line_rows
            if grid is None:
                columns = item.node_slice
                cumulative = tables.cumulative[rows, columns]
                density = tables.density[rows, columns]
                d_a = tables.d_a[rows, columns]
                d_b = tables.d_b[rows, columns]
                d_c = tables.d_c[rows, columns]
            else:
                cumulative, density = _hermite(grid, tables.cumulative[rows], tables.density[rows],
                                               u)
                d_a = _linear(grid, tables.d_a[rows], u)
                d_b = _linear(grid, tables.d_b[rows], u)
                d_c = _linear(grid, tables.d_c[rows], u)
            response = np.diff(cumulative, axis=1)
            amplitudes_free = values[item.amplitude_index]
            amplitudes = item.coefficients @ amplitudes_free
            model = amplitudes @ response
            jacobian = np.zeros((len(item.x), parameter_count), dtype=np.float64)
            du_dg = -u / gain
            jacobian[:, _GAIN] = amplitudes @ np.diff(density * du_dg[None, :], axis=1)
            jacobian[:, _OFFSET] = amplitudes @ np.diff(density, axis=1) * (-1.0 / gain)
            jacobian[:, _RES_A] = amplitudes @ np.diff(d_a, axis=1)
            jacobian[:, _RES_B] = amplitudes @ np.diff(d_b, axis=1)
            jacobian[:, _RES_C] = amplitudes @ np.diff(d_c, axis=1)
            jacobian[:, item.amplitude_index] = response.T @ item.coefficients
            background, derivatives = self._background(item, values)
            model = model + background
            for parameter_index, derivative in derivatives.items():
                jacobian[:, parameter_index] = derivative
            models.append(np.asarray(model, dtype=np.float64))
            responses.append(np.asarray(response * amplitudes[:, None], dtype=np.float64))
            unit_responses.append(np.asarray(response, dtype=np.float64))
            jacobians.append(jacobian)
        evaluation = _Evaluation(
            tuple(models), tuple(responses), tuple(unit_responses), tuple(jacobians)
        )
        self._cache_key = key
        self._cache = evaluation
        return evaluation

    def _background(
        self,
        item: _Prepared,
        values: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        kind = item.dataset.background
        x = item.x
        if kind == "none":
            return np.zeros_like(x), {}
        if kind == "constant":
            index = item.background_index["c"]
            return np.full_like(x, float(values[index])), {index: np.ones_like(x)}
        if kind == "linear":
            intercept = item.background_index["intercept"]
            slope = item.background_index["slope"]
            centred = x - float(values[item.background_index["x_ref"]])
            return (
                float(values[intercept]) + float(values[slope]) * centred,
                {intercept: np.ones_like(x), slope: centred},
            )
        amplitude_index = item.background_index["amplitude"]
        rate_index = item.background_index["rate"]
        x_ref = float(values[item.background_index["x_ref"]])
        amplitude = float(values[amplitude_index])
        rate = float(values[rate_index])
        exponent = rate * (x - x_ref)
        unit = np.exp(np.clip(exponent, -100.0, 100.0))
        clipped = (exponent < -100.0) | (exponent > 100.0)
        rate_derivative = np.where(clipped, 0.0, amplitude * (x - x_ref) * unit)
        return (
            centred_exponential(x, amplitude, rate, x_ref),
            {amplitude_index: unit, rate_index: rate_derivative},
        )

    def residuals(self, values: np.ndarray) -> np.ndarray:
        evaluation = self.evaluate(values)
        return np.concatenate(
            [
                statistic_residual(self.statistic, item.y, model, item.variances)
                for item, model in zip(self.datasets, evaluation.models, strict=True)
            ]
        )

    def jacobian(self, values: np.ndarray) -> np.ndarray:
        evaluation = self.evaluate(values)
        blocks: list[np.ndarray] = []
        for item, model, jacobian in zip(
            self.datasets, evaluation.models, evaluation.jacobians, strict=True
        ):
            if self.statistic == "poisson":
                derivative = poisson_deviance_residual_derivative(item.y, model)
            elif self.statistic == "weighted":
                derivative = -1.0 / statistic_uncertainties(item.variances)
            else:
                derivative = -np.ones_like(model)
            blocks.append(jacobian * derivative[:, None])
        return np.vstack(blocks)

    # --------------------------------------------------------- initial values

    def initial_values(
        self,
        cancelled: Callable[[], bool],
        report: ComptonEdgeProgressCallback = _no_progress,
    ) -> np.ndarray:
        spec = self.spec
        values = np.zeros(len(self.layout.names), dtype=np.float64)
        a, b, c = spec.initial_resolution
        values[_OFFSET] = spec.initial_offset
        values[_RES_A], values[_RES_B], values[_RES_C] = a, b, c
        gain = spec.initial_gain or self._scan_gain(
            spec.initial_offset, a, b, c, cancelled, report
        )
        values[_GAIN] = gain
        lower = self.layout.lower.copy()
        upper = self.layout.upper.copy()
        lower[_GAIN], upper[_GAIN] = gain / 3.0, gain * 3.0
        if spec.offset_free:
            bound = self.offset_bound
            lower[_OFFSET] = min(-bound, spec.initial_offset - bound)
            upper[_OFFSET] = max(bound, spec.initial_offset + bound)
        else:
            lower[_OFFSET] = upper[_OFFSET] = spec.initial_offset
        for term_name, index in zip(_RESOLUTION_NAMES, (_RES_A, _RES_B, _RES_C), strict=True):
            if term_name not in spec.resolution_terms:
                lower[index] = upper[index] = values[index]
        self.layout = _FrozenLayout(self.layout.names, lower, upper, self.layout.vary)
        for item in self.datasets:
            for background_name, index in item.background_index.items():
                if background_name == "x_ref":
                    values[index] = item.x_ref
        self.choose_path(gain, spec.initial_offset, a, b, c)
        # Linear amplitudes and background from non-negative least squares at the
        # starting shape; scipy.optimize.nnls needs no starting point.
        from scipy.optimize import nnls

        evaluation = self.evaluate(values)
        self._cache_key = None
        for item, response in zip(self.datasets, evaluation.unit_responses, strict=True):
            # One column per amplitude parameter: Σ_lines C[line, parameter] · R_line.
            line_columns = response.T @ item.coefficients
            background_columns, converter = _background_basis(item)
            design = np.column_stack([line_columns, *background_columns]) if background_columns \
                else line_columns
            weight = 1.0 / np.sqrt(np.maximum(item.y, 1.0))
            coefficients, _ = nnls(design * weight[:, None], item.y * weight, maxiter=50 * len(
                design[0]
            ))
            count = line_columns.shape[1]
            amplitudes = coefficients[:count]
            scale = max(float(np.sum(item.y)), 1.0)
            values[item.amplitude_index] = np.maximum(amplitudes, 1.0e-6 * scale)
            for name, value in converter(coefficients[count:]).items():
                values[item.background_index[name]] = value
        return values

    def _scan_gain(
        self,
        offset: float,
        a: float,
        b: float,
        c: float,
        cancelled: Callable[[], bool],
        report: ComptonEdgeProgressCallback = _no_progress,
    ) -> float:
        from scipy.optimize import nnls

        edges_kev = [line.edge_kev for dataset in self.spec.datasets for line in dataset.lines]
        lowest_edge = min(edges_kev)
        highest_upper = max(
            line.shape.max_energy_kev for dataset in self.spec.datasets for line in dataset.lines
        )
        candidates_low: list[float] = []
        candidates_high: list[float] = []
        for item in self.datasets:
            top = max(line.edge_kev for line in item.dataset.lines)
            lo_x = float(item.edges[0]) - offset
            hi_x = float(item.edges[-1]) - offset
            if hi_x <= 0:
                raise ValueError(
                    f"dataset {item.index + 1} lies below the initial offset; set a starting gain"
                )
            candidates_high.append(hi_x / lowest_edge)
            candidates_low.append(max(lo_x, hi_x * 0.02) / top)
        g_low = max(min(candidates_low) * 0.8, 1.0e-12)
        g_high = max(candidates_high) * 1.25
        if not g_high > g_low:
            g_high = g_low * 10.0
        candidates = np.geomspace(g_low, g_high, _GAIN_SCAN_CANDIDATES)
        report(ComptonEdgeFitProgress("gain_scan", 0, len(candidates)))
        step = 2.0
        u_high = highest_upper * 1.8 + 100.0
        grid = np.arange(0.0, u_high + step, step)
        tables = _fold_tables(self.centres, self.weights, a, b, c, grid, derivatives=False)
        coarse: list[tuple[_Prepared, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for item in self.datasets:
            factor = max(1, int(math.ceil(len(item.x) / 400)))
            index = np.arange(0, len(item.edges), factor)
            if index[-1] != len(item.edges) - 1:
                index = np.append(index, len(item.edges) - 1)
            y = np.add.reduceat(item.y, index[:-1])
            centres = 0.5 * (item.edges[index[1:]] + item.edges[index[:-1]])
            coarse.append((item, item.edges[index], y, centres, 1.0 / np.sqrt(np.maximum(y, 1.0))))
        best_gain = float(candidates[0])
        best_cost = np.inf
        for done, gain in enumerate(candidates, start=1):
            if cancelled():
                raise ComptonEdgeFitCancelledError("Compton-edge fit cancelled")
            cost = 0.0
            for item, edges, y, centres, weight in coarse:
                u = np.clip((edges - offset) / gain, grid[0], grid[-1])
                cumulative = _linear(grid, tables.cumulative[item.line_rows], u)
                response = np.diff(cumulative, axis=1)
                columns = response.T @ item.coefficients
                background_columns, _ = _background_basis(item, centres)
                design = np.column_stack([columns, *background_columns]) if background_columns \
                    else columns
                _, residual = nnls(design * weight[:, None], y * weight, maxiter=50 * len(
                    design[0]
                ))
                cost += float(residual) ** 2
            if cost < best_cost:
                best_cost = cost
                best_gain = float(gain)
            report(ComptonEdgeFitProgress("gain_scan", done, len(candidates)))
        return best_gain

    # ----------------------------------------------------------------- result

    def result(self, values: np.ndarray, solution: object) -> ComptonEdgeFitResult:
        self._cache_key = None
        evaluation = self.evaluate(values)
        residual = self.residuals(values)
        free = self.layout.vary
        jacobian = self.jacobian(values)[:, free]
        data_count = len(residual)
        varying_count = int(np.count_nonzero(free))
        deviance = float(np.sum(residual**2))
        dof = max(data_count - varying_count, 1)
        reduced = deviance / dof
        covariance_free = _covariance(jacobian, reduced)
        covariance = np.full((len(values), len(values)), np.nan, dtype=np.float64)
        if covariance_free is not None:
            free_index = np.flatnonzero(free)
            covariance[np.ix_(free_index, free_index)] = covariance_free
        for index in np.flatnonzero(~free):
            covariance[index, :] = 0.0
            covariance[:, index] = 0.0
        stderr = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        stderr_or_none = [
            finite_optional(float(stderr[index])) if free[index] else None
            for index in range(len(values))
        ]
        gain = float(values[_GAIN])
        offset = float(values[_OFFSET])
        gain_offset = covariance[np.ix_([_GAIN, _OFFSET], [_GAIN, _OFFSET])].copy()
        gain_offset.setflags(write=False)
        resolution = tuple(
            ResolutionTerm(
                name=name,
                sigma_value=float(values[index]),
                sigma_stderr=stderr_or_none[index],
                varying=bool(free[index]),
                unit=_RESOLUTION_UNITS[name],
            )
            for name, index in zip(_RESOLUTION_NAMES, (_RES_A, _RES_B, _RES_C), strict=True)
        )
        a, b, c = (float(values[index]) for index in (_RES_A, _RES_B, _RES_C))
        parameters = tuple(
            FitParameter(
                name=name,
                value=float(values[index]),
                stderr=stderr_or_none[index],
                minimum=float(self.layout.lower[index]),
                maximum=float(self.layout.upper[index]),
                varying=bool(free[index]),
                expression=None,
            )
            for index, name in enumerate(self.layout.names)
        )
        line_results: list[ComptonEdgeLineResult] = []
        dataset_results: list[ComptonEdgeDatasetResult] = []
        offset_ = 0
        for item, model, responses in zip(
            self.datasets, evaluation.models, evaluation.responses, strict=True
        ):
            count = len(item.x)
            item_residual = residual[offset_ : offset_ + count]
            offset_ += count
            amplitude_cov = covariance[np.ix_(item.amplitude_index, item.amplitude_index)]
            amplitudes = item.coefficients @ values[item.amplitude_index]
            amplitude_var = np.einsum(
                "ij,jk,ik->i", item.coefficients, amplitude_cov, item.coefficients
            )
            components: dict[str, np.ndarray] = {}
            for line_index, line in enumerate(item.dataset.lines):
                edge_x, edge_err = _edge_position(gain, offset, gain_offset, line.edge_kev)
                relative = math.sqrt(
                    a**2 + b**2 / line.edge_kev + c**2 / line.edge_kev**2
                )
                variance = float(amplitude_var[line_index])
                line_results.append(
                    ComptonEdgeLineResult(
                        dataset_index=item.index,
                        dataset_label=item.dataset.label,
                        label=line.label,
                        nuclide=line.nuclide,
                        energy_kev=line.energy_kev,
                        edge_kev=line.edge_kev,
                        edge_x=edge_x,
                        edge_x_stderr=finite_optional(edge_err),
                        amplitude=float(amplitudes[line_index]),
                        amplitude_stderr=(
                            finite_optional(math.sqrt(variance))
                            if math.isfinite(variance) and variance >= 0
                            else None
                        ),
                        sigma_over_energy=relative,
                        fwhm_over_energy=FWHM_PER_SIGMA * relative,
                        in_fit_range=bool(item.edges[0] <= edge_x <= item.edges[-1]),
                        response=line.shape.label,
                    )
                )
                name = line.label
                while name in components:
                    name += " "
                components[name] = readonly(responses[line_index])
            background, _ = self._background(item, values)
            if item.dataset.background != "none":
                components["background"] = readonly(background)
            diagnostics = None
            if len(item.dataset.lines) == 1:
                line = item.dataset.lines[0]
                edge_x = gain * line.edge_kev + offset
                sigma_x = gain * line.edge_kev * math.sqrt(
                    a**2 + b**2 / line.edge_kev + c**2 / line.edge_kev**2
                )
                try:
                    diagnostics = compton_shape_diagnostics(
                        item.x,
                        item.y,
                        edge_x=edge_x,
                        edge_sigma_x=sigma_x,
                        line_energy_kev=line.energy_kev,
                        dietze_klein_detector=self.spec.dietze_klein_detector,
                    )
                except ValueError:
                    diagnostics = None
            dataset_results.append(
                ComptonEdgeDatasetResult(
                    index=item.index,
                    label=item.dataset.label,
                    spectrum_id=item.dataset.spectrum.spectrum_id,
                    background=item.dataset.background,
                    fit_min=float(item.x[0]),
                    fit_max=float(item.x[-1]),
                    x=readonly(item.x),
                    bin_edges=readonly(item.edges),
                    observed=readonly(item.y),
                    best_fit=readonly(model),
                    residual=readonly(item.y - model),
                    components=components,
                    statistic_value=float(np.sum(item_residual**2)),
                    diagnostics=diagnostics,
                )
            )
        edges_kev = [line.edge_kev for line in line_results]
        aic, bic = _information_criteria(deviance, data_count, varying_count)
        success = bool(getattr(solution, "success", False))
        message = str(getattr(solution, "message", ""))
        result = ComptonEdgeFitResult(
            success=success,
            message=message,
            statistic=self.statistic,
            gain=gain,
            gain_stderr=stderr_or_none[_GAIN],
            offset=offset,
            offset_stderr=stderr_or_none[_OFFSET],
            offset_free=self.spec.offset_free,
            gain_offset_covariance=gain_offset,
            resolution=resolution,
            parameters=parameters,
            datasets=tuple(dataset_results),
            lines=tuple(line_results),
            deviance=deviance,
            degrees_of_freedom=dof,
            reduced_deviance=reduced,
            aic=aic,
            bic=bic,
            evaluations=int(getattr(solution, "nfev", self.evaluations)),
            evaluation_path=self.path,
            edge_span_kev=(min(edges_kev), max(edges_kev)),
            warnings=(),
        )
        undetermined = tuple(
            name
            for index, name in enumerate(self.layout.names)
            if free[index] and stderr_or_none[index] is None
        )
        return replace(
            result,
            warnings=self._warnings(result, covariance_free is not None, undetermined),
        )

    def _warnings(
        self,
        result: ComptonEdgeFitResult,
        errorbars: bool,
        undetermined: tuple[str, ...],
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        if not result.success:
            warnings.append(f"The fit did not converge: {result.message}")
        if not errorbars:
            warnings.append("Parameter covariance could not be estimated.")
        elif undetermined:
            warnings.append(
                "Uncertainties could not be estimated for: " + ", ".join(undetermined) + "."
            )
        if result.statistic == "weighted" and any(np.any(item.y < 0) for item in self.datasets):
            warnings.append("Negative derived counts were fitted with propagated-variance weights.")
        for parameter in result.parameters:
            if not parameter.varying:
                continue
            tolerance = max(abs(parameter.value) * 1.0e-6, 1.0e-9)
            if math.isfinite(parameter.minimum) and abs(parameter.value - parameter.minimum) <= (
                tolerance
            ):
                warnings.append(f"{parameter.name} reached its lower bound.")
            if math.isfinite(parameter.maximum) and abs(parameter.value - parameter.maximum) <= (
                tolerance
            ):
                warnings.append(f"{parameter.name} reached its upper bound.")
        for line in result.lines:
            if not line.in_fit_range:
                dataset = result.datasets[line.dataset_index]
                warnings.append(
                    f"The Compton edge of {line.label} (x ≈ {line.edge_x:.6g}) lies outside the "
                    f"fit range of dataset {line.dataset_index + 1} "
                    f"[{dataset.bin_edges[0]:.6g}, {dataset.bin_edges[-1]:.6g}]; its amplitude "
                    "and the gain are weakly constrained."
                )
        if self.spec.offset_free and len({round(line.edge_kev, 6) for line in result.lines}) < 2:
            warnings.append(
                "The offset is free but only one distinct Compton-edge energy is fitted, so gain "
                "and offset are degenerate. Fix the offset or add a second source or dataset."
            )
        if result.reduced_deviance > REDUCED_DEVIANCE_WARNING:
            single_scatter = any(
                isinstance(line.shape, KleinNishinaResponse)
                for dataset in self.spec.datasets
                for line in dataset.lines
            )
            if single_scatter:
                warnings.append(
                    f"Reduced deviance {result.reduced_deviance:.3g} exceeds "
                    f"{REDUCED_DEVIANCE_WARNING:g}. The single-scatter Klein–Nishina model "
                    "ignores multiple Compton scattering, which fills the region above the edge "
                    "and biases the fitted edge upward in finite detectors (Kudomi 1999). "
                    "Consider a simulated detector response for each line."
                )
            else:
                warnings.append(
                    f"Reduced deviance {result.reduced_deviance:.3g} exceeds "
                    f"{REDUCED_DEVIANCE_WARNING:g}; the response model does not describe the "
                    "measured shape."
                )
        low, high = EDGE_HEIGHT_EXPECTED
        for dataset in result.datasets:
            diagnostics = dataset.diagnostics
            if diagnostics is None:
                continue
            height = diagnostics.edge_height_fraction
            if height is not None and not low <= height <= high:
                warnings.append(
                    f"Dataset {dataset.index + 1}: the fitted edge crosses the measured spectrum "
                    f"at {height:.0%} of the Compton maximum. Compton-coincidence measurements "
                    f"on organic detectors give {SWIDERSKI_EDGE_HEIGHT[0]:.0%}–"
                    f"{SWIDERSKI_EDGE_HEIGHT[1]:.0%} (Swiderski et al. 2010); outside "
                    f"{low:.0%}–{high:.0%} the edge is probably biased. Consider a simulated "
                    "response."
                )
            warnings.extend(f"Dataset {dataset.index + 1}: {note}" for note in diagnostics.notes)
        return tuple(warnings)


def _background_basis(
    item: _Prepared,
    x: np.ndarray | None = None,
) -> tuple[list[np.ndarray], Callable[[np.ndarray], dict[str, float]]]:
    """Non-negative basis columns for an initial background and their conversion."""
    coordinates = item.x if x is None else x
    low = float(item.edges[0])
    span = item.span
    t = (coordinates - low) / span
    kind = item.dataset.background
    if kind == "none":
        return [], lambda _coefficients: {}
    if kind == "constant":
        return [np.ones_like(t)], lambda coefficients: {"c": float(coefficients[0])}
    if kind == "linear":

        def linear(coefficients: np.ndarray) -> dict[str, float]:
            first, second = (float(value) for value in coefficients)
            t_ref = (item.x_ref - low) / span
            return {
                "intercept": first * (1.0 - t_ref) + second * t_ref,
                "slope": (second - first) / span,
            }

        return [1.0 - t, t], linear

    def exponential(coefficients: np.ndarray) -> dict[str, float]:
        constant, decaying = (float(value) for value in coefficients)

        def level(fraction: float) -> float:
            return constant + decaying * math.exp(-4.0 * fraction)

        first, second = level(0.25), level(0.75)
        scale = max(float(np.mean(np.abs(item.y))), 1.0)
        if first > 0 and second > 0:
            rate = math.log(second / first) / (0.5 * span)
            rate = float(np.clip(rate, -19.0 / span, 19.0 / span))
            t_ref = (item.x_ref - low) / span
            amplitude = max(level(t_ref), 1.0e-6 * scale)
        else:
            rate = 0.0
            amplitude = 1.0e-3 * scale
        return {"amplitude": amplitude, "rate": rate}

    return [np.ones_like(t), np.exp(-4.0 * t)], exponential


def _hermite(
    grid: np.ndarray,
    values: np.ndarray,
    derivatives: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Hermite interpolation on a uniform grid, and its exact derivative."""
    step = float(grid[1] - grid[0])
    position = (query - grid[0]) / step
    index = np.clip(np.floor(position).astype(np.int64), 0, len(grid) - 2)
    t = position - index
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    f0, f1 = values[:, index], values[:, index + 1]
    d0, d1 = derivatives[:, index] * step, derivatives[:, index + 1] * step
    interpolated = h00 * f0 + h10 * d0 + h01 * f1 + h11 * d1
    slope = (
        (6.0 * t2 - 6.0 * t) * f0
        + (3.0 * t2 - 4.0 * t + 1.0) * d0
        + (-6.0 * t2 + 6.0 * t) * f1
        + (3.0 * t2 - 2.0 * t) * d1
    ) / step
    return interpolated, slope


def _linear(grid: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    step = float(grid[1] - grid[0])
    position = (query - grid[0]) / step
    index = np.clip(np.floor(position).astype(np.int64), 0, len(grid) - 2)
    t = position - index
    return np.asarray(values[:, index] * (1.0 - t) + values[:, index + 1] * t, dtype=np.float64)


def _covariance(jacobian: np.ndarray, scale: float) -> np.ndarray | None:
    """Scaled Gauss–Newton covariance from a column-normalised Jacobian.

    Columns are normalised before the SVD: amplitudes (~10⁶) and gains (~10)
    otherwise make ``JᵀJ`` too ill-conditioned for a reliable inverse. A
    parameter whose column is zero has no influence on the data; it gets a NaN
    variance while the others are still estimated.
    """
    count = jacobian.shape[1]
    covariance = np.full((count, count), np.nan, dtype=np.float64)
    if jacobian.size == 0:
        return covariance
    if not np.all(np.isfinite(jacobian)):
        return None
    norms = np.linalg.norm(jacobian, axis=0)
    active = np.flatnonzero(norms > 0)
    if not len(active):
        return None
    scaled = jacobian[:, active] / norms[active]
    _, singular, vt = np.linalg.svd(scaled, full_matrices=False)
    if singular[-1] <= singular[0] * 1.0e-12:
        return None
    inverse = (vt.T / singular**2) @ vt
    covariance[np.ix_(active, active)] = (
        inverse / np.outer(norms[active], norms[active]) * scale
    )
    return covariance


def _information_criteria(deviance: float, data_count: int, varying: int) -> tuple[float, float]:
    chi = max(deviance, 1.0e-250)
    base = data_count * math.log(chi / data_count)
    return base + 2.0 * varying, base + math.log(data_count) * varying


def _line_document(line: ComptonEdgeLineResult) -> dict[str, object]:
    return {
        "dataset_index": line.dataset_index,
        "dataset_label": line.dataset_label,
        "label": line.label,
        "nuclide": line.nuclide,
        "gamma_energy_kev": line.energy_kev,
        "edge_kevee": line.edge_kev,
        "edge_x": line.edge_x,
        "edge_x_stderr": line.edge_x_stderr,
        "amplitude": line.amplitude,
        "amplitude_stderr": line.amplitude_stderr,
        "sigma_over_energy_at_edge": line.sigma_over_energy,
        "fwhm_over_energy_at_edge": line.fwhm_over_energy,
        "in_fit_range": line.in_fit_range,
        "response": line.response,
    }


def _diagnostics_document(
    index: int,
    diagnostics: ComptonShapeDiagnostics,
) -> dict[str, object]:
    estimate = diagnostics.dietze_klein
    return {
        "dataset_index": index,
        "line_energy_kev": diagnostics.line_energy_kev,
        "background_floor": diagnostics.background_floor,
        "floor_method": diagnostics.floor_method,
        "smoothing_sigma_x": diagnostics.smoothing_sigma_x,
        "l_max": diagnostics.l_max,
        "l_half": diagnostics.l_half,
        "maximum_height": diagnostics.maximum_height,
        "dk_resolution_fwhm": diagnostics.dk_resolution_fwhm,
        "fitted_edge_x": diagnostics.fitted_edge_x,
        "edge_height_fraction": diagnostics.edge_height_fraction,
        "fitted_minus_dk_relative": diagnostics.fitted_minus_dk_relative,
        "notes": list(diagnostics.notes),
        "dietze_klein": None
        if estimate is None
        else {
            "detector": estimate.detector,
            "detector_description": estimate.detector_description,
            "source": estimate.source,
            "resolution_fwhm": estimate.resolution_fwhm,
            "lmax_over_edge": estimate.lmax_over_edge,
            "lhalf_over_edge": estimate.lhalf_over_edge,
            "edge_from_lmax": estimate.edge_from_lmax,
            "edge_from_lhalf": estimate.edge_from_lhalf,
            "edge_x": estimate.edge_x,
            "edge_height_fraction": estimate.edge_height_fraction,
            "citation": estimate.citation,
        },
    }


__all__ = [
    "COMPTON_EDGE_ENERGY_UNIT",
    "COMPTON_EDGE_POINT_SOURCE",
    "COMPTON_PRESET_NUCLIDES",
    "ELECTRON_REST_ENERGY_KEV",
    "FWHM_PER_SIGMA",
    "REFERENCES",
    "AmplitudeMode",
    "ComptonBackground",
    "ComptonEdgeDataset",
    "ComptonEdgeDatasetResult",
    "ComptonEdgeFitCancelledError",
    "ComptonEdgeFitProgress",
    "ComptonEdgeFitResult",
    "ComptonEdgeFitSpec",
    "ComptonEdgeLineResult",
    "ComptonLine",
    "ComptonShapeDiagnostics",
    "DietzeKleinEstimate",
    "KleinNishinaResponse",
    "ResolutionTerm",
    "ResponseShape",
    "SimulatedResponse",
    "SummaryRow",
    "compton_edge_calibration_points",
    "compton_edge_kev",
    "compton_edge_summary_rows",
    "compton_shape_diagnostics",
    "export_compton_edge_result",
    "extrapolation_warning",
    "fit_compton_edges",
    "folded_line_response",
    "klein_nishina_electron_spectrum",
    "load_simulated_response",
    "nuclide_compton_lines",
    "parse_custom_energies",
]
