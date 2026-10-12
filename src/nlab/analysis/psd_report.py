"""Portable PDF and tabular report bundles for offline PSD/FOM analysis."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from nlab import __version__
from nlab.analysis.psd_fom import PsdFomFit

_EWT_URL = "https://ewt.tech"
_REPORT_SUFFIX = "_psd_report"


@dataclass(frozen=True)
class PsdReportRequest:
    """All source and presentation state needed for one reproducible report."""

    source_path: Path
    title: str
    matrix: np.ndarray
    energy_range: tuple[float, float]
    ratio_range: tuple[float, float]
    energy_label: str
    cut: float
    energy_roi: tuple[float, float]
    matrix_scale: Literal["linear", "log"]
    energy_log_y: bool
    fit: PsdFomFit
    settings: Mapping[str, object] = field(default_factory=dict)
    digitizer_info: Mapping[str, object] = field(default_factory=dict)
    statistics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PsdReportBundle:
    """Paths produced by :func:`export_psd_report`."""

    directory: Path
    pdf: Path
    psd_matrix_csv: Path
    separated_spectra_csv: Path
    fom_projection_csv: Path
    settings_csv: Path
    metadata_json: Path


@dataclass(frozen=True)
class _ProjectionData:
    energy_edges: np.ndarray
    energy_centers: np.ndarray
    ratio_edges: np.ndarray
    ratio_centers: np.ndarray
    below: np.ndarray
    above: np.ndarray
    ratio_counts: np.ndarray
    cut_bin: int
    first_energy_bin: int
    last_energy_bin: int


def export_psd_report(request: PsdReportRequest, destination: Path) -> PsdReportBundle:
    """Create a uniquely named PDF/CSV report directory under ``destination``."""
    matrix = np.asarray(request.matrix, dtype=np.uint64)
    _validate_request(request, matrix)
    projection = _projection_data(request, matrix)
    report_directory = _unique_report_directory(Path(destination), request.source_path.stem)
    report_directory.mkdir(parents=False, exist_ok=False)

    base_name = _safe_stem(request.source_path.stem)
    bundle = PsdReportBundle(
        directory=report_directory,
        pdf=report_directory / f"{base_name}_psd_report.pdf",
        psd_matrix_csv=report_directory / "psd_matrix.csv",
        separated_spectra_csv=report_directory / "separated_energy_spectra.csv",
        fom_projection_csv=report_directory / "fom_projection_and_fit.csv",
        settings_csv=report_directory / "report_settings.csv",
        metadata_json=report_directory / "report_metadata.json",
    )
    created_utc = datetime.now(UTC).isoformat(timespec="seconds")
    _write_matrix_csv(bundle.psd_matrix_csv, matrix, projection)
    _write_spectra_csv(bundle.separated_spectra_csv, projection)
    _write_fom_csv(bundle.fom_projection_csv, projection, request.fit)
    _write_settings_csv(bundle.settings_csv, request, projection, created_utc)
    _write_metadata(bundle, request, matrix, projection, created_utc)
    _write_pdf(bundle.pdf, request, matrix, projection, created_utc)
    return bundle


def _validate_request(request: PsdReportRequest, matrix: np.ndarray) -> None:
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("PSD report matrix must be a non-empty two-dimensional array")
    if request.matrix_scale not in {"linear", "log"}:
        raise ValueError("PSD report matrix scale must be 'linear' or 'log'")
    for limits, name in (
        (request.energy_range, "energy"),
        (request.ratio_range, "ratio"),
    ):
        low, high = limits
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError(f"PSD report {name} range must be finite and increasing")
    fit = request.fit
    fit_lengths = {
        len(fit.x),
        len(fit.observed),
        len(fit.best_fit),
        len(fit.first_component),
        len(fit.second_component),
        len(fit.background_component),
    }
    if len(fit_lengths) != 1 or not fit.x.size:
        raise ValueError("FOM fit arrays must be non-empty and have matching lengths")


def _projection_data(request: PsdReportRequest, matrix: np.ndarray) -> _ProjectionData:
    energy_edges = np.linspace(*request.energy_range, matrix.shape[0] + 1)
    ratio_edges = np.linspace(*request.ratio_range, matrix.shape[1] + 1)
    energy_centers = (energy_edges[:-1] + energy_edges[1:]) * 0.5
    ratio_centers = (ratio_edges[:-1] + ratio_edges[1:]) * 0.5
    cut_bin = int(np.clip(np.searchsorted(ratio_edges, request.cut), 0, len(ratio_centers)))
    below = matrix[:, :cut_bin].sum(axis=1, dtype=np.uint64)
    above = matrix[:, cut_bin:].sum(axis=1, dtype=np.uint64)

    roi_low, roi_high = sorted(map(float, request.energy_roi))
    first = int(
        np.clip(
            np.searchsorted(energy_edges, roi_low, side="right") - 1,
            0,
            len(energy_centers),
        )
    )
    last = int(
        np.clip(
            np.searchsorted(energy_edges, roi_high, side="left"),
            0,
            len(energy_centers),
        )
    )
    ratio_counts = matrix[first:last].sum(axis=0, dtype=np.uint64)
    return _ProjectionData(
        energy_edges=energy_edges,
        energy_centers=energy_centers,
        ratio_edges=ratio_edges,
        ratio_centers=ratio_centers,
        below=below,
        above=above,
        ratio_counts=ratio_counts,
        cut_bin=cut_bin,
        first_energy_bin=first,
        last_energy_bin=last,
    )


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "psd_analysis"


def _unique_report_directory(parent: Path, source_stem: str) -> Path:
    if not parent.is_dir():
        raise OSError(f"Report destination does not exist: {parent}")
    base = parent / f"{_safe_stem(source_stem)}{_REPORT_SUFFIX}"
    if not base.exists():
        return base
    suffix = 2
    while True:
        candidate = parent / f"{base.name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _write_matrix_csv(path: Path, matrix: np.ndarray, projection: _ProjectionData) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["psd_ratio\\energy_center", *map(_number, projection.energy_centers)])
        for ratio, counts in zip(projection.ratio_centers, matrix.T, strict=True):
            writer.writerow([_number(ratio), *map(int, counts)])


def _write_spectra_csv(path: Path, projection: _ProjectionData) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["energy_center", "below_cut_counts", "above_cut_counts"])
        for energy, below, above in zip(
            projection.energy_centers,
            projection.below,
            projection.above,
            strict=True,
        ):
            writer.writerow([_number(energy), int(below), int(above)])


def _fit_indices(ratio_centers: np.ndarray, fit: PsdFomFit) -> np.ndarray:
    indices = np.searchsorted(ratio_centers, fit.x)
    if np.any(indices >= len(ratio_centers)) or not np.allclose(
        ratio_centers[indices], fit.x, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("FOM fit coordinates do not match the current PSD ratio bins")
    return indices


def _write_fom_csv(path: Path, projection: _ProjectionData, fit: PsdFomFit) -> None:
    fit_indices = _fit_indices(projection.ratio_centers, fit)
    if not np.array_equal(
        projection.ratio_counts[fit_indices].astype(np.float64), fit.observed
    ):
        raise ValueError("FOM fit counts do not match the current PSD projection")
    observed = np.full(len(projection.ratio_centers), np.nan)
    total = np.full(len(projection.ratio_centers), np.nan)
    first = np.full_like(total, np.nan)
    second = np.full_like(total, np.nan)
    background = np.full_like(total, np.nan)
    observed[fit_indices] = fit.observed
    total[fit_indices] = fit.best_fit
    first[fit_indices] = fit.first_component
    second[fit_indices] = fit.second_component
    background[fit_indices] = fit.background_component
    included = np.zeros(len(projection.ratio_centers), dtype=bool)
    included[fit_indices] = True
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "psd_ratio",
                "projected_counts",
                "included_in_fit",
                "fit_observed_counts",
                "fitted_total",
                "population_1",
                "population_2",
                "fitted_background",
            ]
        )
        for row in zip(
            projection.ratio_centers,
            projection.ratio_counts,
            included,
            observed,
            total,
            first,
            second,
            background,
            strict=True,
        ):
            (
                ratio,
                counts,
                is_included,
                fit_observed,
                fitted,
                component_1,
                component_2,
                fitted_background,
            ) = row
            writer.writerow(
                [
                    _number(ratio),
                    int(counts),
                    int(is_included),
                    _optional_number(fit_observed),
                    _optional_number(fitted),
                    _optional_number(component_1),
                    _optional_number(component_2),
                    _optional_number(fitted_background),
                ]
            )


def _write_settings_csv(
    path: Path,
    request: PsdReportRequest,
    projection: _ProjectionData,
    created_utc: str,
) -> None:
    rows = _report_settings_rows(request, projection, created_utc)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["section", "setting", "value"])
        writer.writerows(rows)


def _report_settings_rows(
    request: PsdReportRequest,
    projection: _ProjectionData,
    created_utc: str,
) -> list[tuple[str, str, str]]:
    energy_width = float(np.diff(projection.energy_edges[:2])[0])
    ratio_width = float(np.diff(projection.ratio_edges[:2])[0])
    effective_cut = float(projection.ratio_edges[projection.cut_bin])
    rows = [
        ("report", "created UTC", created_utc),
        ("report", "software", f"NLab Digitizer {__version__}"),
        ("source", "file", str(request.source_path.resolve())),
        ("histogram", "energy bins", str(len(projection.energy_centers))),
        ("histogram", "energy range", _range_text(request.energy_range)),
        ("histogram", "energy bin width", _number(energy_width)),
        ("histogram", "PSD ratio bins", str(len(projection.ratio_centers))),
        ("histogram", "PSD ratio range", _range_text(request.ratio_range)),
        ("histogram", "PSD ratio bin width", _number(ratio_width)),
        ("histogram", "requested PSD cut", _number(request.cut)),
        ("histogram", "cut bin index", str(projection.cut_bin)),
        ("histogram", "effective cut boundary", _number(effective_cut)),
        ("histogram", "projection energy ROI", _range_text(request.energy_roi)),
        ("histogram", "projection first energy bin", str(projection.first_energy_bin)),
        ("histogram", "projection last energy bin exclusive", str(projection.last_energy_bin)),
        ("display", "matrix scale", request.matrix_scale),
        ("display", "energy projection scale", "log" if request.energy_log_y else "linear"),
        ("fit", "input bins", str(len(request.fit.x))),
        ("fit", "input ratio range", _range_text((request.fit.x[0], request.fit.x[-1]))),
    ]
    rows.extend(_mapping_rows("processing", request.settings))
    rows.extend(_mapping_rows("digitizer", request.digitizer_info))
    rows.extend(_mapping_rows("statistics", request.statistics))
    rows.extend(_fit_rows(request.fit))
    return rows


def _mapping_rows(section: str, values: Mapping[str, object]) -> list[tuple[str, str, str]]:
    return [(section, str(key), _value_text(value)) for key, value in values.items()]


def _fit_rows(fit: PsdFomFit) -> list[tuple[str, str, str]]:
    rows = [
        ("fit", "population 1 centroid", _number(fit.centroids[0])),
        ("fit", "population 1 centroid uncertainty", _value_text(fit.centroid_uncertainties[0])),
        ("fit", "population 1 FWHM", _number(fit.fwhms[0])),
        ("fit", "population 1 FWHM uncertainty", _value_text(fit.fwhm_uncertainties[0])),
        ("fit", "population 2 centroid", _number(fit.centroids[1])),
        ("fit", "population 2 centroid uncertainty", _value_text(fit.centroid_uncertainties[1])),
        ("fit", "population 2 FWHM", _number(fit.fwhms[1])),
        ("fit", "population 2 FWHM uncertainty", _value_text(fit.fwhm_uncertainties[1])),
        ("fit", "FOM", _number(fit.fom)),
        ("fit", "reduced Poisson deviance", _number(fit.reduced_deviance)),
        ("fit", "success", str(fit.success)),
        ("fit", "message", fit.message),
        ("fit", "evaluations", str(fit.evaluations)),
        ("fit", "warnings", "; ".join(fit.warnings)),
    ]
    return rows


def _write_metadata(
    bundle: PsdReportBundle,
    request: PsdReportRequest,
    matrix: np.ndarray,
    projection: _ProjectionData,
    created_utc: str,
) -> None:
    fit = request.fit
    metadata = {
        "schema_version": 1,
        "created_utc": created_utc,
        "application": {"name": "NLab Digitizer", "version": __version__, "url": _EWT_URL},
        "source": {
            "path": str(request.source_path.resolve()),
            "file_name": request.source_path.name,
            "size_bytes": request.source_path.stat().st_size
            if request.source_path.exists()
            else None,
        },
        "histogram": {
            "matrix_shape_energy_ratio": list(matrix.shape),
            "energy_range": list(request.energy_range),
            "ratio_range": list(request.ratio_range),
            "cut": request.cut,
            "cut_bin": projection.cut_bin,
            "effective_cut_boundary": float(projection.ratio_edges[projection.cut_bin]),
            "energy_roi": list(request.energy_roi),
            "energy_bin_range": [projection.first_energy_bin, projection.last_energy_bin],
            "matrix_scale": request.matrix_scale,
            "energy_projection_scale": "log" if request.energy_log_y else "linear",
        },
        "processing": _json_mapping(request.settings),
        "digitizer": _json_mapping(request.digitizer_info),
        "statistics": _json_mapping(request.statistics),
        "fit": {
            "success": fit.success,
            "message": fit.message,
            "evaluations": fit.evaluations,
            "centroids": list(fit.centroids),
            "centroid_uncertainties": list(fit.centroid_uncertainties),
            "sigmas": list(fit.sigmas),
            "fwhms": list(fit.fwhms),
            "fwhm_uncertainties": list(fit.fwhm_uncertainties),
            "fom": fit.fom,
            "reduced_poisson_deviance": fit.reduced_deviance,
            "warnings": list(fit.warnings),
            "input_bins": len(fit.x),
            "input_ratio_range": [float(fit.x[0]), float(fit.x[-1])],
        },
        "files": {
            "pdf": bundle.pdf.name,
            "psd_matrix_csv": bundle.psd_matrix_csv.name,
            "separated_spectra_csv": bundle.separated_spectra_csv.name,
            "fom_projection_csv": bundle.fom_projection_csv.name,
            "settings_csv": bundle.settings_csv.name,
        },
    }
    bundle.metadata_json.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_pdf(
    path: Path,
    request: PsdReportRequest,
    matrix: np.ndarray,
    projection: _ProjectionData,
    created_utc: str,
) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportSmall",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            "ReportWarning",
            parent=styles["BodyText"],
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#B54708"),
        )
    )
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=15 * mm,
        title=request.title,
        author="Eastern Wall Technologies",
        subject="PSD figure-of-merit analysis report",
    )
    story: list[Any] = []
    story.extend(_report_header(request, created_utc, styles))
    story.append(Paragraph("PSD matrix and analysis selection", styles["Heading2"]))
    story.append(
        Image(
            _matrix_figure(request, matrix, projection),
            width=180 * mm,
            height=101 * mm,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Cut and binning settings", styles["Heading3"]))
    story.append(_settings_table(request, projection, styles))
    story.append(PageBreak())

    story.append(Paragraph("PSD figure of merit", styles["Heading1"]))
    story.append(
        Image(
            _fom_figure(projection, request.fit),
            width=172 * mm,
            height=100 * mm,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(_fit_table(request.fit, styles))
    if request.fit.poor_convergence:
        warning = "; ".join(request.fit.warnings) or request.fit.message
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(f"<b>Fit warning:</b> {_escape(warning)}", styles["ReportWarning"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Fitting method and FOM", styles["Heading3"]))
    story.extend(_fit_method_explanation(styles))
    story.append(PageBreak())

    story.append(Paragraph("Cut-separated energy spectra", styles["Heading1"]))
    story.append(
        Image(
            _spectra_figure(request, projection),
            width=180 * mm,
            height=105 * mm,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Source and processing information", styles["Heading3"]))
    story.append(_metadata_table(request, styles))
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "The accompanying CSV files contain the complete matrix, both separated "
            "energy spectra, the full PSD projection, and every fitted component. "
            "report_settings.csv and report_metadata.json preserve the analysis settings.",
            styles["ReportSmall"],
        )
    )
    document.build(
        story,
        onFirstPage=_draw_page_footer,
        onLaterPages=_draw_page_footer,
    )


def _report_header(
    request: PsdReportRequest,
    created_utc: str,
    styles: Mapping[str, ParagraphStyle],
) -> list[Any]:
    logo_bytes = _load_logo()
    logo = Image(BytesIO(logo_bytes), width=18 * mm, height=18 * mm)
    brand = Paragraph(
        '<b>Eastern Wall Technologies</b><br/><a href="https://ewt.tech" '
        'color="#2075DF">ewt.tech</a>',
        styles["BodyText"],
    )
    brand_table = Table([[logo, brand]], colWidths=[23 * mm, 145 * mm])
    brand_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return [
        brand_table,
        Spacer(1, 3 * mm),
        Paragraph(_escape(request.title), styles["Title"]),
        Paragraph(
            f"Source: <b>{_escape(request.source_path.name)}</b><br/>"
            f"Generated: {_escape(created_utc)}<br/>"
            f"Software: NLab Digitizer {_escape(__version__)}",
            styles["ReportSmall"],
        ),
        Spacer(1, 4 * mm),
    ]


def _settings_table(
    request: PsdReportRequest,
    projection: _ProjectionData,
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    energy_width = projection.energy_edges[1] - projection.energy_edges[0]
    ratio_width = projection.ratio_edges[1] - projection.ratio_edges[0]
    effective_cut = projection.ratio_edges[projection.cut_bin]
    energy_binning = (
        f"{len(projection.energy_centers)} / {_range_text(request.energy_range)} / "
        f"{_number(energy_width)}"
    )
    ratio_binning = (
        f"{len(projection.ratio_centers)} / {_range_text(request.ratio_range)} / "
        f"{_number(ratio_width)}"
    )
    energy_roi = (
        f"{_range_text(request.energy_roi)} "
        f"(bins {projection.first_energy_bin}:{projection.last_energy_bin})"
    )
    values = [
        ("Energy bins / range / width", energy_binning),
        ("PSD bins / range / width", ratio_binning),
        (
            "PSD cut / effective bin boundary",
            f"{_number(request.cut)} / {_number(effective_cut)} (bin {projection.cut_bin})",
        ),
        ("Projection energy ROI", energy_roi),
        (
            "Display scales",
            f"matrix {request.matrix_scale}; spectra {'log' if request.energy_log_y else 'linear'}",
        ),
    ]
    return _key_value_table(values, styles)



def _fit_method_explanation(styles: Mapping[str, ParagraphStyle]) -> list[Flowable]:
    """Explain the equations implemented by :func:`fit_psd_fom_lm` in the PDF."""
    return [
        Paragraph(
            "Counts in the selected energy ROI are summed by PSD-ratio bin. "
            "Only the selected ratio fit ROI enters the fit. For ratio x, the "
            "expected bin count is a constant background plus two Gaussian peaks:",
            styles["ReportSmall"],
        ),
        _math_formula(
            r"m(x)=B+A_1\exp\!\left[-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right]"
            r"+A_2\exp\!\left[-\frac{(x-\mu_2)^2}{2\sigma_2^2}\right]"
        ),
        Paragraph(
            "The background and peak heights are non-negative; the peak widths "
            "are positive and the fitted centroids remain ordered. Bounded "
            "Levenberg-Marquardt minimizes the sum of squared signed "
            "Poisson-deviance residuals. For each observed bin count, the "
            "expected count is floored at a small positive value:",
            styles["ReportSmall"],
        ),
        _math_formula(r"m_i=\max\!\left(m(x_i),10^{-12}\right)"),
        _math_formula(
            r"r_i=\mathrm{sgn}(y_i-m_i)"
            r"\sqrt{2\!\left[m_i-y_i+y_i\ln\!\left(\frac{y_i}{m_i}\right)\right]}"
        ),
        Paragraph(
            "The logarithmic contribution is defined as zero for an empty bin. "
            "The reported reduced Poisson deviance uses the number of fitted "
            "bins and varied parameters:",
            styles["ReportSmall"],
        ),
        _math_formula(
            r"D_{\mathrm{reduced}}="
            r"\frac{\sum_{i=1}^{n}r_i^2}{\max(n-p,1)}"
        ),
        Paragraph(
            "Each Gaussian width is converted to full width at half maximum "
            "(FWHM). The dimensionless FOM measures centroid separation "
            "relative to the sum of those widths:",
            styles["ReportSmall"],
        ),
        _math_formula(r"\mathrm{FWHM}_j=2\sqrt{2\ln 2}\,\sigma_j"),
        _math_formula(
            r"\mathrm{FOM}="
            r"\frac{\left|\mu_2-\mu_1\right|}{\mathrm{FWHM}_1+\mathrm{FWHM}_2}"
        ),
    ]


def _math_formula(expression: str) -> Image:
    """Render one TeX-style math expression without requiring a LaTeX install."""
    figure = Figure(figsize=(8.0, 0.55), dpi=240)
    figure.text(
        0.01,
        0.5,
        f"${expression}$",
        color="#1F2937",
        fontsize=9.5,
        horizontalalignment="left",
        verticalalignment="center",
    )
    FigureCanvasAgg(figure)
    output = BytesIO()
    figure.savefig(
        output,
        format="png",
        dpi=240,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    output.seek(0)

    image = Image(output)
    scale = min(72 / 240, (168 * mm) / image.imageWidth)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "LEFT"
    image.spaceBefore = 0.2 * mm
    image.spaceAfter = 0.5 * mm
    return image


def _fit_table(fit: PsdFomFit, styles: Mapping[str, ParagraphStyle]) -> Table:
    data = [
        ["Population", "Centroid", "Centroid uncertainty", "FWHM", "FWHM uncertainty"],
        [
            "1",
            _number(fit.centroids[0]),
            _value_text(fit.centroid_uncertainties[0]),
            _number(fit.fwhms[0]),
            _value_text(fit.fwhm_uncertainties[0]),
        ],
        [
            "2",
            _number(fit.centroids[1]),
            _value_text(fit.centroid_uncertainties[1]),
            _number(fit.fwhms[1]),
            _value_text(fit.fwhm_uncertainties[1]),
        ],
    ]
    table = Table(data, colWidths=[24 * mm, 35 * mm, 39 * mm, 32 * mm, 39 * mm], repeatRows=1)
    _apply_table_style(table)
    summary = [
        ("FOM", _number(fit.fom)),
        ("Reduced Poisson deviance", _number(fit.reduced_deviance)),
        ("Fit status", "Poor convergence" if fit.poor_convergence else "Converged"),
        ("LM evaluations", str(fit.evaluations)),
    ]
    wrapper = Table([[table], [_key_value_table(summary, styles)]], colWidths=[172 * mm])
    wrapper.setStyle(
        TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)])
    )
    return wrapper


def _metadata_table(
    request: PsdReportRequest,
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    values: list[tuple[str, str]] = []
    values.extend(
        (f"Digitizer - {key}", _value_text(value)) for key, value in request.digitizer_info.items()
    )
    values.extend(
        (f"Processing - {key}", _value_text(value)) for key, value in request.settings.items()
    )
    values.extend(
        (f"Statistics - {key}", _value_text(value)) for key, value in request.statistics.items()
    )
    if not values:
        values.append(("Metadata", "No additional source metadata was available."))
    return _key_value_table(values, styles)


def _key_value_table(
    values: Sequence[tuple[str, str]],
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    data = [
        [
            Paragraph(f"<b>{_escape(key)}</b>", styles["ReportSmall"]),
            Paragraph(_escape(value), styles["ReportSmall"]),
        ]
        for key, value in values
    ]
    table = Table(data, colWidths=[62 * mm, 108 * mm])
    _apply_table_style(table, header=False)
    return table


def _apply_table_style(table: Table, *, header: bool = True) -> None:
    commands: list[tuple[Any, ...]] = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))


def _matrix_figure(
    request: PsdReportRequest,
    matrix: np.ndarray,
    projection: _ProjectionData,
) -> BytesIO:
    figure = Figure(figsize=(8.0, 4.5), dpi=150, layout="constrained")
    axis = figure.subplots()
    values = np.ma.masked_less_equal(matrix.T.astype(np.float64), 1.0)
    non_sparse = matrix[matrix > 1]
    norm: Normalize
    if request.matrix_scale == "log" and non_sparse.size:
        norm = LogNorm(vmin=2.0, vmax=max(2.0, float(non_sparse.max())))
    else:
        norm = Normalize(vmin=0.0, vmax=max(1.0, float(matrix.max(initial=0))))
    cmap = colormaps["viridis"].with_extremes(bad="white")
    image = axis.pcolormesh(
        projection.energy_edges,
        projection.ratio_edges,
        values,
        shading="flat",
        cmap=cmap,
        norm=norm,
    )
    axis.axhline(request.cut, color="#E63946", linewidth=1.2, label="PSD cut")
    roi_low, roi_high = sorted(request.energy_roi)
    axis.axvspan(roi_low, roi_high, color="#277DA1", alpha=0.12, label="FOM energy ROI")
    axis.set(xlabel=f"{request.energy_label} (raw)", ylabel="PSD ratio")
    axis.set_xlim(request.energy_range)
    axis.set_ylim(request.ratio_range)
    axis.grid(alpha=0.15)
    axis.legend(loc="upper right", fontsize=8)
    figure.colorbar(image, ax=axis, label="Counts")
    return _figure_png(figure)


def _fom_figure(projection: _ProjectionData, fit: PsdFomFit) -> BytesIO:
    _fit_indices(projection.ratio_centers, fit)
    figure = Figure(figsize=(8.0, 4.65), dpi=150, layout="constrained")
    axis = figure.subplots()
    axis.step(
        projection.ratio_centers,
        projection.ratio_counts,
        where="mid",
        color="#6A4C93",
        linewidth=1.1,
        label="Projected counts",
    )
    axis.plot(fit.x, fit.best_fit, color="#2A9D8F", linewidth=2.0, label="Double-Gaussian fit")
    axis.plot(
        fit.x, fit.first_component, "--", color="#277DA1", linewidth=1.2, label="Population 1"
    )
    axis.plot(
        fit.x, fit.second_component, "--", color="#F8961E", linewidth=1.2, label="Population 2"
    )
    axis.plot(
        fit.x, fit.background_component, ":", color="#64748B", linewidth=1.0, label="Background"
    )
    axis.axvspan(
        float(fit.x[0]), float(fit.x[-1]), color="#2A9D8F", alpha=0.07, label="Fit input bins"
    )
    for index, centroid in enumerate(fit.centroids, start=1):
        axis.axvline(centroid, color=("#277DA1", "#F8961E")[index - 1], alpha=0.65, linewidth=1.0)
    axis.set(xlabel="PSD ratio", ylabel="Counts", title=f"FOM = {fit.fom:.5g}")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=7, ncol=2)
    return _figure_png(figure)


def _spectra_figure(request: PsdReportRequest, projection: _ProjectionData) -> BytesIO:
    figure = Figure(figsize=(8.0, 4.65), dpi=150, layout="constrained")
    axis = figure.subplots()
    axis.step(
        projection.energy_centers,
        projection.below,
        where="mid",
        color="#277DA1",
        linewidth=1.1,
        label="Below cut",
    )
    axis.step(
        projection.energy_centers,
        projection.above,
        where="mid",
        color="#F8961E",
        linewidth=1.1,
        label="Above cut",
    )
    if request.energy_log_y and not np.any((projection.below > 0) | (projection.above > 0)):
        axis.set_ylim(1.0, 10.0)
    axis.set_yscale("log" if request.energy_log_y else "linear")
    axis.set(xlabel=f"{request.energy_label} (raw)", ylabel="Counts")
    axis.set_xlim(request.energy_range)
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    return _figure_png(figure)


def _figure_png(figure: Figure) -> BytesIO:
    FigureCanvasAgg(figure)
    output = BytesIO()
    figure.savefig(output, format="png", dpi=150, facecolor="white")
    output.seek(0)
    return output


def _load_logo() -> bytes:
    source_logo = Path(__file__).resolve().parents[3] / "resources" / "ewt.png"
    if source_logo.is_file():
        return source_logo.read_bytes()
    try:
        from PySide6.QtCore import QFile, QIODevice

        resource = QFile(":/ewt.png")
        if resource.open(QIODevice.OpenModeFlag.ReadOnly):
            try:
                return cast(bytes, resource.readAll().data())
            finally:
                resource.close()
    except ImportError:
        pass
    raise OSError("The EWT logo resource is unavailable")


def _draw_page_footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(15 * mm, 11 * mm, A4[0] - 15 * mm, 11 * mm)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(15 * mm, 7 * mm, f"NLab Digitizer {__version__}")
    link_text = "ewt.tech"
    link_x = A4[0] - 35 * mm
    canvas.setFillColor(colors.HexColor("#2075DF"))
    canvas.drawString(link_x, 7 * mm, link_text)
    canvas.linkURL(_EWT_URL, (link_x, 6 * mm, A4[0] - 15 * mm, 10 * mm), relative=0)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawCentredString(A4[0] / 2, 7 * mm, f"Page {document.page}")
    canvas.restoreState()


def _json_mapping(values: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _json_value(value) for key, value in values.items()}


def _json_value(value: object) -> object:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_json_value(item) for item in value]
    return str(value)


def _number(value: float | np.floating[Any]) -> str:
    return f"{float(value):.12g}"


def _optional_number(value: float | np.floating[Any]) -> str:
    return "" if not np.isfinite(value) else _number(value)


def _value_text(value: object) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, float | np.floating):
        return _number(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return " to ".join(_value_text(item) for item in value)
    return str(value)


def _range_text(value: tuple[float, float]) -> str:
    return f"{_number(value[0])} to {_number(value[1])}"


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
