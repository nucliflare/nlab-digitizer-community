"""Portable PDF and tabular reports for MCA fitting and energy calibration."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from nlab import __version__
from nlab.analysis.energy_calibration import (
    CalibrationPeakFit,
    CalibrationPoint,
    EnergyCalibration,
    SpectrumSnapshot,
)
from nlab.analysis.peak_fitting import BackgroundModel, PeakFitResult, PeakFitSpec
from nlab.analysis.spectrum import Spectrum

_EWT_URL = "https://ewt.tech"


@dataclass(frozen=True)
class PeakFitReportRequest:
    """Inputs required to reproduce one MCA peak-fit report."""

    spectrum: Spectrum
    spec: PeakFitSpec
    result: PeakFitResult
    reference_overlay: Mapping[str, object] | None = None
    y_scale: Literal["linear", "log"] = "linear"


@dataclass(frozen=True)
class PeakFitReportBundle:
    directory: Path
    pdf: Path
    spectrum_fit_csv: Path
    peaks_csv: Path
    parameters_csv: Path
    metadata_json: Path


@dataclass(frozen=True)
class EnergyCalibrationReportPoint:
    """One calibration reference plus its optional Gaussian centroid estimate."""

    point: CalibrationPoint
    seed_channel: float
    peak_fit: CalibrationPeakFit | None


@dataclass(frozen=True)
class EnergyCalibrationReportRequest:
    channel: int
    snapshots: tuple[SpectrumSnapshot, ...]
    calibration: EnergyCalibration
    points: tuple[EnergyCalibrationReportPoint, ...]
    centroid_half_width: float


@dataclass(frozen=True)
class EnergyCalibrationReportBundle:
    directory: Path
    pdf: Path
    spectra_csv: Path
    points_csv: Path
    centroid_fits_csv: Path
    metadata_json: Path


def export_peak_fit_report(
    request: PeakFitReportRequest,
    destination: Path,
) -> PeakFitReportBundle:
    """Create a non-overwriting peak-fit PDF/CSV/JSON report bundle."""

    _validate_peak_request(request)
    report_directory = _unique_directory(
        Path(destination),
        f"{_safe_stem(request.spectrum.label)}_peak_fit_report",
    )
    report_directory.mkdir(parents=False, exist_ok=False)
    stem = _safe_stem(request.spectrum.label)
    bundle = PeakFitReportBundle(
        directory=report_directory,
        pdf=report_directory / f"{stem}_peak_fit_report.pdf",
        spectrum_fit_csv=report_directory / "spectrum_and_fit.csv",
        peaks_csv=report_directory / "peak_results.csv",
        parameters_csv=report_directory / "fit_parameters.csv",
        metadata_json=report_directory / "report_metadata.json",
    )
    created_utc = datetime.now(UTC).isoformat(timespec="seconds")
    _write_peak_curve_csv(bundle.spectrum_fit_csv, request)
    _write_peak_results_csv(bundle.peaks_csv, request.result)
    _write_parameters_csv(bundle.parameters_csv, request.result)
    _write_peak_metadata(bundle.metadata_json, request, created_utc)
    _write_peak_pdf(bundle.pdf, request, created_utc)
    return bundle


def export_energy_calibration_report(
    request: EnergyCalibrationReportRequest,
    destination: Path,
) -> EnergyCalibrationReportBundle:
    """Create a non-overwriting energy-calibration PDF/CSV/JSON report bundle."""

    _validate_calibration_request(request)
    report_directory = _unique_directory(
        Path(destination),
        f"mca_{request.channel}_energy_calibration_report",
    )
    report_directory.mkdir(parents=False, exist_ok=False)
    bundle = EnergyCalibrationReportBundle(
        directory=report_directory,
        pdf=report_directory / f"mca_{request.channel}_energy_calibration_report.pdf",
        spectra_csv=report_directory / "source_spectra.csv",
        points_csv=report_directory / "calibration_points.csv",
        centroid_fits_csv=report_directory / "gaussian_centroid_fits.csv",
        metadata_json=report_directory / "report_metadata.json",
    )
    created_utc = datetime.now(UTC).isoformat(timespec="seconds")
    _write_calibration_spectra_csv(bundle.spectra_csv, request.snapshots)
    _write_calibration_points_csv(bundle.points_csv, request)
    _write_centroid_fits_csv(bundle.centroid_fits_csv, request.points)
    _write_calibration_metadata(bundle.metadata_json, request, created_utc)
    _write_calibration_pdf(bundle.pdf, request, created_utc)
    return bundle


def _validate_peak_request(request: PeakFitReportRequest) -> None:
    if request.y_scale not in {"linear", "log"}:
        raise ValueError("peak-fit report scale must be 'linear' or 'log'")
    lengths = {
        len(request.result.x),
        len(request.result.observed),
        len(request.result.best_fit),
        len(request.result.residual),
        *(len(values) for values in request.result.components.values()),
    }
    if len(lengths) != 1 or not request.result.x.size:
        raise ValueError("peak-fit report arrays must be non-empty and equally sized")


def _validate_calibration_request(request: EnergyCalibrationReportRequest) -> None:
    if request.channel < 0:
        raise ValueError("MCA channel must be non-negative")
    if not request.points:
        raise ValueError("energy-calibration report requires calibration points")
    if not np.isfinite(request.centroid_half_width) or request.centroid_half_width <= 0:
        raise ValueError("centroid fit half-width must be positive")
    calibration_points = request.calibration.points
    if len(calibration_points) != len(request.points):
        raise ValueError("report points must match the fitted energy calibration")
    for reported, fitted in zip(request.points, calibration_points, strict=True):
        if reported.point != fitted:
            raise ValueError("report points must preserve the fitted calibration order")


def _write_peak_curve_csv(path: Path, request: PeakFitReportRequest) -> None:
    component_names = tuple(request.result.components)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["coordinate", "observed", "best_fit", "residual", *component_names]
        )
        component_values = tuple(request.result.components[name] for name in component_names)
        for index, coordinate in enumerate(request.result.x):
            writer.writerow(
                [
                    _number(coordinate),
                    _number(request.result.observed[index]),
                    _number(request.result.best_fit[index]),
                    _number(request.result.residual[index]),
                    *(_number(values[index]) for values in component_values),
                ]
            )


def _write_peak_results_csv(path: Path, result: PeakFitResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "peak",
                "center",
                "center_stderr",
                "energy_kev",
                "sigma",
                "fwhm",
                "energy_fwhm_kev",
                "area",
                "statistic",
                "area_stderr",
                "height",
                "resolution_percent",
            ]
        )
        for peak in result.peaks:
            writer.writerow(
                [
                    peak.index,
                    _number(peak.center),
                    _optional_number(peak.center_stderr),
                    _optional_number(peak.energy_kev),
                    _number(peak.sigma),
                    _number(peak.fwhm),
                    _optional_number(peak.energy_fwhm_kev),
                    _number(peak.area),
                    _optional_number(peak.area_stderr),
                    _number(peak.height),
                    _optional_number(peak.resolution_percent),
                ]
            )


def _write_parameters_csv(path: Path, result: PeakFitResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["parameter", "value", "stderr", "minimum", "maximum", "varying", "expression"]
        )
        for parameter in result.parameters:
            writer.writerow(
                [
                    parameter.name,
                    _number(parameter.value),
                    _optional_number(parameter.stderr),
                    _number(parameter.minimum),
                    _number(parameter.maximum),
                    int(parameter.varying),
                    parameter.expression or "",
                ]
            )


def _write_peak_metadata(
    path: Path,
    request: PeakFitReportRequest,
    created_utc: str,
) -> None:
    document = {
        "kind": "nlab_mca_peak_fit_report",
        "created_utc": created_utc,
        "application": {"name": "NLab Digitizer", "version": __version__, "url": _EWT_URL},
        "spectrum": {
            "id": request.spectrum.spectrum_id,
            "label": request.spectrum.label,
            "source": request.spectrum.source,
            "axis_unit": request.spectrum.axis_unit,
            "poisson_counts": request.spectrum.poisson_counts,
            "metadata": request.spectrum.metadata,
            "history": request.spectrum.history,
        },
        "model": asdict(request.spec),
        "display": {"y_scale": request.y_scale},
        "reference_overlay": request.reference_overlay,
        "statistics": {
            "success": request.result.success,
            "message": request.result.message,
            "statistic": request.result.statistic,
            "chi_square": request.result.chi_square,
            "reduced_chi_square": request.result.reduced_chi_square,
            "aic": request.result.aic,
            "bic": request.result.bic,
            "evaluations": request.result.evaluations,
            "warnings": request.result.warnings,
        },
        "peaks": [asdict(peak) for peak in request.result.peaks],
        "parameters": [asdict(parameter) for parameter in request.result.parameters],
    }
    path.write_text(json.dumps(_json_value(document), indent=2), encoding="utf-8")


def _write_calibration_spectra_csv(
    path: Path,
    snapshots: tuple[SpectrumSnapshot, ...],
) -> None:
    maximum = max((len(snapshot.counts) for snapshot in snapshots), default=0)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", *(snapshot.label for snapshot in snapshots)])
        for channel in range(maximum):
            writer.writerow(
                [
                    channel,
                    *(
                        int(snapshot.counts[channel]) if channel < len(snapshot.counts) else ""
                        for snapshot in snapshots
                    ),
                ]
            )


def _write_calibration_points_csv(
    path: Path,
    request: EnergyCalibrationReportRequest,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "point",
                "label",
                "source",
                "enabled",
                "seed_channel",
                "centroid_channel",
                "centroid_stderr",
                "expected_energy_kev",
                "fitted_energy_kev",
                "residual_kev",
                "sigma_channel",
                "fwhm_channel",
                "area",
            ]
        )
        for index, reported in enumerate(request.points, start=1):
            point = reported.point
            fitted_energy = float(request.calibration.energy(point.channel))
            residual = _clean_residual(
                point.energy_kev - fitted_energy,
                point.energy_kev,
            )
            peak_fit = reported.peak_fit
            writer.writerow(
                [
                    index,
                    point.label,
                    point.source,
                    int(point.enabled),
                    _number(reported.seed_channel),
                    _number(point.channel),
                    _optional_number(
                        peak_fit.centroid_stderr if peak_fit is not None else None
                    ),
                    _number(point.energy_kev),
                    _number(fitted_energy),
                    _number(residual),
                    _optional_number(peak_fit.sigma_channel if peak_fit is not None else None),
                    _optional_number(peak_fit.fwhm_channel if peak_fit is not None else None),
                    _optional_number(peak_fit.area if peak_fit is not None else None),
                    peak_fit.statistic if peak_fit is not None else "",
                ]
            )


def _write_centroid_fits_csv(
    path: Path,
    points: tuple[EnergyCalibrationReportPoint, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "point",
                "source",
                "seed_channel",
                "centroid_channel",
                "channel",
                "observed",
                "best_fit",
                "residual",
            ]
        )
        for index, reported in enumerate(points, start=1):
            peak_fit = reported.peak_fit
            if peak_fit is None:
                continue
            for channel, observed, best_fit in zip(
                peak_fit.x,
                peak_fit.observed,
                peak_fit.best_fit,
                strict=True,
            ):
                writer.writerow(
                    [
                        index,
                        reported.point.source,
                        _number(reported.seed_channel),
                        _number(reported.point.channel),
                        _number(channel),
                        _number(observed),
                        _number(best_fit),
                        _number(observed - best_fit),
                    ]
                )


def _write_calibration_metadata(
    path: Path,
    request: EnergyCalibrationReportRequest,
    created_utc: str,
) -> None:
    document = {
        "kind": "nlab_mca_energy_calibration_report",
        "created_utc": created_utc,
        "application": {"name": "NLab Digitizer", "version": __version__, "url": _EWT_URL},
        "channel": request.channel,
        "centroid_method": {
            "model": "Gaussian plus linear background",
            "half_width_channel": request.centroid_half_width,
            "seed_source": "draggable calibration marker",
        },
        "calibration": request.calibration.to_dict(),
        "points": [
            {
                "point": reported.point.to_dict(),
                "seed_channel": reported.seed_channel,
                "centroid_fit": (
                    None
                    if reported.peak_fit is None
                    else {
                        "centroid_channel": reported.peak_fit.centroid_channel,
                        "centroid_stderr": reported.peak_fit.centroid_stderr,
                        "sigma_channel": reported.peak_fit.sigma_channel,
                        "fwhm_channel": reported.peak_fit.fwhm_channel,
                        "area": reported.peak_fit.area,
                        "fit_range": [reported.peak_fit.fit_min, reported.peak_fit.fit_max],
                        "half_width": reported.peak_fit.half_width,
                        "statistic": reported.peak_fit.statistic,
                    }
                ),
            }
            for reported in request.points
        ],
        "snapshots": [
            {
                "id": snapshot.snapshot_id,
                "channel": snapshot.channel,
                "label": snapshot.label,
                "created_utc": snapshot.created_utc,
                "elapsed_s": snapshot.elapsed_s,
                "live": snapshot.live,
                "fingerprint": snapshot.fingerprint,
                "bins": len(snapshot.counts),
            }
            for snapshot in request.snapshots
        ],
    }
    path.write_text(json.dumps(_json_value(document), indent=2), encoding="utf-8")


def _write_peak_pdf(path: Path, request: PeakFitReportRequest, created_utc: str) -> None:
    styles = _report_styles()
    document = _document(
        path,
        title="MCA Peak-Fitting Report",
        subject="MCA Gaussian peak-fitting analysis report",
    )
    story: list[Any] = _report_header(
        "MCA Peak-Fitting Report",
        request.spectrum.label,
        created_utc,
        styles,
    )
    story.append(Paragraph("Spectrum, model, and residual", styles["Heading2"]))
    story.append(Image(_peak_fit_figure(request), width=180 * mm, height=112 * mm))
    story.append(Spacer(1, 3 * mm))
    story.append(_peak_model_table(request, styles))
    story.append(PageBreak())
    story.append(Paragraph("Fitted peak results", styles["Heading1"]))
    story.append(_peak_results_table(request.result, styles))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Fit statistics", styles["Heading2"]))
    story.append(_peak_statistics_table(request.result, styles))
    if request.result.warnings:
        story.append(Spacer(1, 2 * mm))
        story.append(
            Paragraph(
                f"<b>Fit warning:</b> {_escape('; '.join(request.result.warnings))}",
                styles["ReportWarning"],
            )
        )
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Model", styles["Heading2"]))
    story.append(
        Image(
            _formula_png(
                r"f(x)=b(x)+\sum_i \frac{A_i}{\sigma_i\sqrt{2\pi}}"
                r"\exp\!\left[-\frac{(x-\mu_i)^2}{2\sigma_i^2}\right]"
            ),
            width=150 * mm,
            height=13 * mm,
        )
    )
    background_name, background_formula = _background_formula(request.spec.background)
    story.append(
        Paragraph(
            f"Selected background: <b>{_escape(background_name)}</b>",
            styles["ReportSmall"],
        )
    )
    story.append(
        Image(
            _formula_png(background_formula),
            width=140 * mm,
            height=12 * mm,
        )
    )
    story.append(
        Image(
            _formula_png(r"\mathrm{FWHM}_i=2\sqrt{2\ln 2}\,\sigma_i"),
            width=140 * mm,
            height=12 * mm,
        )
    )
    story.append(
        Paragraph(
            "The accompanying CSV files preserve every fitted bin, evaluated component, "
            "peak result, and model parameter. report_metadata.json records spectrum "
            "provenance, model choices, fit statistics, and reference-line selections.",
            styles["ReportSmall"],
        )
    )
    document.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)


def _background_formula(background: BackgroundModel) -> tuple[str, str]:
    """Return the report label and equation for a selectable peak background."""
    formulas: dict[BackgroundModel, tuple[str, str]] = {
        "none": ("None", r"b(x)=0"),
        "constant": ("Constant", r"b(x)=c"),
        "linear": ("Linear", r"b(x)=c+mx"),
        "exponential": (
            "Centred exponential",
            r"b(x)=B\exp\!\left[k(x-x_{\mathrm{ref}})\right]",
        ),
        "compton": (
            "Compton step (error function)",
            r"b(x)=\frac{B}{2}\operatorname{erfc}\!\left("
            r"\frac{x-\mu_1}{\sqrt{2}\,\sigma_1}\right)",
        ),
        "fermi": (
            "Fermi step (logistic)",
            r"b(x)=\frac{B}{1+\exp\!\left[(x-x_0)/w\right]}",
        ),
    }
    return formulas[background]


def _write_calibration_pdf(
    path: Path,
    request: EnergyCalibrationReportRequest,
    created_utc: str,
) -> None:
    styles = _report_styles()
    document = _document(
        path,
        title="MCA Energy-Calibration Report",
        subject="MCA Gaussian-centroid energy-calibration report",
    )
    story: list[Any] = _report_header(
        "MCA Energy-Calibration Report",
        f"MCA channel {request.channel}",
        created_utc,
        styles,
    )
    story.append(
        Paragraph("Source spectra, fitted peak centroids, and residuals", styles["Heading2"])
    )
    story.append(Image(_calibration_spectra_figure(request), width=180 * mm, height=105 * mm))
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "Dashed markers are the draggable initial seeds. Solid markers are the "
            "Gaussian centroids used for calibration. The lower panel shows observed "
            "minus fitted counts for each local centroid fit.",
            styles["ReportSmall"],
        )
    )
    story.append(PageBreak())
    story.append(Paragraph("Energy calibration and residuals", styles["Heading1"]))
    story.append(Image(_calibration_curve_figure(request), width=180 * mm, height=92 * mm))
    story.append(Spacer(1, 3 * mm))
    story.append(_calibration_points_table(request, styles))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Centroid and calibration models", styles["Heading2"]))
    story.append(
        Image(
            _formula_png(
                r"g(c)=b_0+b_1c+\frac{A}{\sigma\sqrt{2\pi}}"
                r"\exp\!\left[-\frac{(c-\mu)^2}{2\sigma^2}\right]"
            ),
            width=148 * mm,
            height=13 * mm,
        )
    )
    polynomial = (
        r"E(c)=a_0+a_1c"
        if request.calibration.model == "linear"
        else r"E(c)=a_0+a_1c+a_2c^2"
    )
    story.append(Image(_formula_png(polynomial), width=140 * mm, height=12 * mm))
    story.append(
        Paragraph(
            f"Each enabled marker seeds a Gaussian-plus-linear-background fit within "
            f"{request.centroid_half_width:g} channels on either side. The fitted "
            "centroid, rather than the seed position, is paired with the entered "
            "reference energy. The "
            "accompanying CSV files contain all source bins, centroid-fit curves, and "
            "calibration residuals.",
            styles["ReportSmall"],
        )
    )
    document.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)


def _peak_fit_figure(request: PeakFitReportRequest) -> BytesIO:
    figure = Figure(figsize=(10.0, 6.2), tight_layout=True)
    spectrum_axis = figure.add_subplot(2, 1, 1)
    residual_axis = figure.add_subplot(2, 1, 2, sharex=spectrum_axis)
    spectrum_axis.step(
        request.spectrum.x,
        request.spectrum.counts,
        where="mid",
        color="#1f77b4",
        linewidth=0.8,
        label="Observed spectrum",
    )
    spectrum_axis.plot(
        request.result.x,
        request.result.best_fit,
        color="#d62728",
        linewidth=1.5,
        label="Best fit",
    )
    component_colors = ("#9467bd", "#2ca02c", "#17becf", "#8c564b")
    for index, (name, values) in enumerate(request.result.components.items()):
        spectrum_axis.plot(
            request.result.x,
            values,
            linewidth=0.9,
            color=component_colors[index % len(component_colors)],
            label=name.rstrip("_"),
        )
    if request.y_scale == "log":
        spectrum_axis.set_yscale("log")
        positives = request.spectrum.counts[request.spectrum.counts > 0]
        if not positives.size:
            spectrum_axis.set_ylim(1.0, 10.0)
    unit = "Energy (keV)" if request.spectrum.axis_unit == "keV" else "MCA channel"
    spectrum_axis.set_ylabel("Counts")
    spectrum_axis.grid(alpha=0.2)
    spectrum_axis.legend(loc="best", fontsize=7, ncols=2)
    residual_axis.axhline(0.0, color="#64748b", linewidth=0.8)
    residual_axis.plot(request.result.x, request.result.residual, color="#334155", linewidth=0.8)
    residual_axis.set(xlabel=unit, ylabel="Residual")
    residual_axis.grid(alpha=0.2)
    return _figure_png(figure)


def _calibration_spectra_figure(request: EnergyCalibrationReportRequest) -> BytesIO:
    figure = Figure(figsize=(10.0, 6.2), tight_layout=True)
    spectrum_axis = figure.add_subplot(2, 1, 1)
    residual_axis = figure.add_subplot(2, 1, 2, sharex=spectrum_axis)
    for snapshot in request.snapshots:
        spectrum_axis.step(
            np.arange(len(snapshot.counts), dtype=np.float64),
            snapshot.counts,
            where="mid",
            linewidth=0.8,
            label=snapshot.label,
        )
    for index, reported in enumerate(request.points, start=1):
        color = ("#2a9d8f", "#7c3aed", "#e76f51", "#0891b2", "#ca8a04")[
            (index - 1) % 5
        ]
        spectrum_axis.axvline(
            reported.seed_channel,
            color="#a66f6f",
            linestyle="--",
            linewidth=0.9,
            label="Initial seed" if index == 1 else None,
        )
        if reported.peak_fit is not None:
            spectrum_axis.plot(
                reported.peak_fit.x,
                reported.peak_fit.best_fit,
                color=color,
                linewidth=1.4,
                label=f"P{index} Gaussian fit",
            )
            spectrum_axis.axvline(
                reported.point.channel,
                color=color,
                linewidth=1.1,
                label="Gaussian centroid" if index == 1 else None,
            )
            residual_axis.plot(
                reported.peak_fit.x,
                reported.peak_fit.observed - reported.peak_fit.best_fit,
                color=color,
                linewidth=0.9,
                label=f"P{index}",
            )
    spectrum_axis.set_ylabel("Counts")
    spectrum_axis.grid(alpha=0.2)
    spectrum_axis.legend(loc="best", fontsize=7, ncols=2)
    residual_axis.axhline(0.0, color="#64748b", linewidth=0.8)
    residual_axis.set(xlabel="MCA channel", ylabel="Residual counts")
    residual_axis.grid(alpha=0.2)
    if residual_axis.lines[1:]:
        residual_axis.legend(loc="best", fontsize=7, ncols=3)
    return _figure_png(figure)


def _calibration_curve_figure(request: EnergyCalibrationReportRequest) -> BytesIO:
    figure = Figure(figsize=(10.0, 5.1), tight_layout=True)
    calibration_axis = figure.add_subplot(2, 1, 1)
    residual_axis = figure.add_subplot(2, 1, 2, sharex=calibration_axis)
    enabled = tuple(point.point for point in request.points if point.point.enabled)
    channels = np.asarray([point.channel for point in enabled], dtype=np.float64)
    energies = np.asarray([point.energy_kev for point in enabled], dtype=np.float64)
    upper = max(
        [float(len(snapshot.counts) - 1) for snapshot in request.snapshots]
        + [float(np.max(channels))]
    )
    coordinate = np.linspace(0.0, upper, 512)
    calibration_axis.plot(
        coordinate,
        request.calibration.energy(coordinate),
        color="#2075df",
        linewidth=1.4,
        label=f"{request.calibration.model.capitalize()} calibration",
    )
    calibration_axis.scatter(channels, energies, color="#c23b22", zorder=3, label="References")
    calibration_axis.set_ylabel("Energy (keV)")
    calibration_axis.grid(alpha=0.2)
    calibration_axis.legend(loc="best", fontsize=8)
    residuals = energies - np.asarray(request.calibration.energy(channels), dtype=np.float64)
    residuals = np.asarray(
        [_clean_residual(value, energy) for value, energy in zip(residuals, energies, strict=True)]
    )
    residual_axis.axhline(0.0, color="#64748b", linewidth=0.8)
    residual_axis.scatter(channels, residuals, color="#334155", s=18)
    if not np.any(residuals):
        residual_axis.set_ylim(-1.0, 1.0)
    residual_axis.set(xlabel="Fitted centroid channel", ylabel="Residual (keV)")
    residual_axis.grid(alpha=0.2)
    return _figure_png(figure)


def _peak_model_table(
    request: PeakFitReportRequest,
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    spec = request.spec
    rows = [
        ["Fit range", f"{_number(request.result.x[0])} to {_number(request.result.x[-1])}"],
        ["Gaussian components", str(spec.peak_count)],
        ["Background", spec.background],
        ["Requested statistic", spec.statistic],
        ["Resolved statistic", request.result.statistic],
        ["Initial centre seeds", ", ".join(_number(value) for value in spec.peak_centers)],
        ["Spectrum source", request.spectrum.source],
    ]
    return _key_value_table(rows, styles)


def _peak_results_table(
    result: PeakFitResult,
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    rows: list[list[object]] = [
        ["Peak", "Centre", "Centre SE", "FWHM", "Area", "Energy (keV)", "Resolution (%)"]
    ]
    for peak in result.peaks:
        rows.append(
            [
                f"P{peak.index}",
                _short_number(peak.center),
                _optional_short_number(peak.center_stderr),
                _short_number(peak.fwhm),
                _short_number(peak.area),
                _optional_short_number(peak.energy_kev),
                _optional_short_number(peak.resolution_percent),
            ]
        )
    return _data_table(rows, styles, [14, 25, 25, 25, 30, 30, 31])


def _peak_statistics_table(
    result: PeakFitResult,
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    return _key_value_table(
        [
            ["Success", str(result.success)],
            ["Message", result.message],
            ["Chi-square / deviance", _number(result.chi_square)],
            ["Reduced statistic", _number(result.reduced_chi_square)],
            ["AIC", _number(result.aic)],
            ["BIC", _number(result.bic)],
            ["Function evaluations", str(result.evaluations)],
        ],
        styles,
    )


def _calibration_points_table(
    request: EnergyCalibrationReportRequest,
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    rows: list[list[object]] = [
        ["Point", "Seed ch.", "Centroid ch.", "Centre SE", "Energy (keV)", "Residual (keV)"]
    ]
    for index, reported in enumerate(request.points, start=1):
        point = reported.point
        fitted = float(request.calibration.energy(point.channel))
        residual = _clean_residual(point.energy_kev - fitted, point.energy_kev)
        rows.append(
            [
                f"P{index}",
                _short_number(reported.seed_channel),
                _short_number(point.channel),
                _optional_short_number(
                    reported.peak_fit.centroid_stderr
                    if reported.peak_fit is not None
                    else None
                ),
                _short_number(point.energy_kev),
                _short_number(residual),
            ]
        )
    table = _data_table(rows, styles, [18, 30, 32, 28, 36, 36])
    return table


def _report_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportSmall",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
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
    return cast(dict[str, ParagraphStyle], styles)


def _document(path: Path, *, title: str, subject: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=15 * mm,
        title=title,
        author="Eastern Wall Technologies",
        subject=subject,
    )


def _report_header(
    title: str,
    source: str,
    created_utc: str,
    styles: Mapping[str, ParagraphStyle],
) -> list[Any]:
    logo = Image(BytesIO(_load_logo()), width=18 * mm, height=18 * mm)
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
        Paragraph(_escape(title), styles["Title"]),
        Paragraph(
            f"Source: <b>{_escape(source)}</b><br/>Generated: {_escape(created_utc)}<br/>"
            f"Software: NLab Digitizer {_escape(__version__)}",
            styles["ReportSmall"],
        ),
        Spacer(1, 4 * mm),
    ]


def _key_value_table(
    rows: Sequence[Sequence[object]],
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    formatted = [
        [
            Paragraph(f"<b>{_escape(str(row[0]))}</b>", styles["ReportSmall"]),
            Paragraph(_escape(str(row[1])), styles["ReportSmall"]),
        ]
        for row in rows
    ]
    table = Table(formatted, colWidths=[50 * mm, 125 * mm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _data_table(
    rows: Sequence[Sequence[object]],
    styles: Mapping[str, ParagraphStyle],
    widths_mm: Sequence[float],
) -> Table:
    formatted = [
        [Paragraph(_escape(str(value)), styles["ReportSmall"]) for value in row]
        for row in rows
    ]
    table = Table(formatted, colWidths=[value * mm for value in widths_mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _formula_png(formula: str) -> BytesIO:
    figure = Figure(figsize=(8.0, 0.7), tight_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    axis.axis("off")
    axis.text(0.5, 0.5, f"${formula}$", ha="center", va="center", fontsize=15)
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


def _draw_footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(15 * mm, 11 * mm, A4[0] - 15 * mm, 11 * mm)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(15 * mm, 7 * mm, f"NLab Digitizer {__version__}")
    link_x = A4[0] - 35 * mm
    canvas.setFillColor(colors.HexColor("#2075DF"))
    canvas.drawString(link_x, 7 * mm, "ewt.tech")
    canvas.linkURL(_EWT_URL, (link_x, 6 * mm, A4[0] - 15 * mm, 10 * mm), relative=0)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawCentredString(A4[0] / 2, 7 * mm, f"Page {document.page}")
    canvas.restoreState()


def _unique_directory(parent: Path, name: str) -> Path:
    if not parent.is_dir():
        raise OSError(f"Report destination does not exist: {parent}")
    base = parent / name
    if not base.exists():
        return base
    suffix = 2
    while True:
        candidate = parent / f"{name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "mca_analysis"


def _number(value: float | np.floating[Any]) -> str:
    return f"{float(value):.12g}"


def _optional_number(value: float | np.floating[Any] | None) -> str:
    return "" if value is None or not np.isfinite(value) else _number(value)


def _short_number(value: float | np.floating[Any]) -> str:
    return f"{float(value):.8g}"


def _optional_short_number(value: float | np.floating[Any] | None) -> str:
    return "" if value is None or not np.isfinite(value) else _short_number(value)


def _clean_residual(value: float | np.floating[Any], reference: float) -> float:
    converted = float(value)
    tolerance = max(1.0e-12, abs(float(reference)) * 1.0e-12)
    return 0.0 if abs(converted) <= tolerance else converted


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
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_json_value(item) for item in value]
    return str(value)


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
