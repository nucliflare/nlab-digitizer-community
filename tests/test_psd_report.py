"""PSD/FOM PDF and tabular report export tests."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from pypdf import PdfReader

import nlab.analysis.psd_report as psd_report
from nlab import __version__
from nlab.analysis.capture_metadata import capture_digitizer_info
from nlab.analysis.psd_fom import fit_psd_fom_lm
from nlab.analysis.psd_report import PsdReportRequest, export_psd_report


def _request(source: Path) -> PsdReportRequest:
    ratio_edges = np.linspace(-0.2, 0.8, 129)
    ratio = (ratio_edges[:-1] + ratio_edges[1:]) * 0.5
    expectation = (
        3.0
        + 120.0 * np.exp(-0.5 * np.square((ratio - 0.12) / 0.035))
        + 85.0 * np.exp(-0.5 * np.square((ratio - 0.48) / 0.055))
    )
    matrix = np.random.default_rng(91).poisson(expectation, size=(16, 128)).astype(np.uint64)
    projection = matrix[2:12].sum(axis=0)
    fit = fit_psd_fom_lm(
        ratio,
        projection,
        roi=(-0.05, 0.65),
        centroid_seeds=(0.1, 0.5),
    )
    return PsdReportRequest(
        source_path=source,
        title="Waveform PSD/FOM Analysis Report",
        matrix=matrix,
        energy_range=(0.0, 16.0),
        ratio_range=(-0.2, 0.8),
        energy_label="Long-gate charge",
        cut=0.3,
        energy_roi=(2.0, 12.0),
        matrix_scale="log",
        energy_log_y=False,
        fit=fit,
        settings={"baseline samples": (0, 8), "event stride used": 1},
        digitizer_info={"source format": "NLab scope NDMA", "channel": 0},
        statistics={"accepted waveforms": int(matrix.sum())},
    )


def test_export_psd_report_writes_pdf_and_complete_plot_data(tmp_path: Path) -> None:
    source = tmp_path / "investigated file.bin"
    source.write_bytes(b"source")
    request = _request(source)

    bundle = export_psd_report(request, tmp_path)

    assert bundle.directory.name == "investigated_file_psd_report"
    assert bundle.pdf.read_bytes().startswith(b"%PDF")
    pdf = PdfReader(bundle.pdf)
    assert len(pdf.pages) == 3
    pdf_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    assert "Eastern Wall Technologies" in pdf_text
    assert "ewt.tech" in pdf_text
    assert "Cut and binning settings" in pdf_text
    assert "PSD figure of merit" in pdf_text
    fit_page = pdf.pages[1].extract_text() or ""
    assert "Fitting method and FOM" in fit_page
    assert "Poisson-deviance residuals" in fit_page
    assert "The logarithmic contribution is defined as zero" in fit_page
    assert "The dimensionless FOM measures centroid separation" in fit_page
    # The plot plus six TeX-style equations are embedded as images on the FOM page.
    assert len(pdf.pages[1].images) >= 7
    assert f"NLab Digitizer {__version__}" in pdf_text

    with bundle.psd_matrix_csv.open(encoding="utf-8", newline="") as stream:
        matrix_rows = list(csv.reader(stream))
    assert len(matrix_rows) == request.matrix.shape[1] + 1
    assert len(matrix_rows[0]) == request.matrix.shape[0] + 1
    exported_matrix = np.asarray(
        [[int(value) for value in row[1:]] for row in matrix_rows[1:]],
        dtype=np.uint64,
    ).T
    np.testing.assert_array_equal(exported_matrix, request.matrix)

    with bundle.separated_spectra_csv.open(encoding="utf-8", newline="") as stream:
        spectra_rows = list(csv.DictReader(stream))
    assert len(spectra_rows) == request.matrix.shape[0]
    assert sum(int(row["below_cut_counts"]) for row in spectra_rows) + sum(
        int(row["above_cut_counts"]) for row in spectra_rows
    ) == int(request.matrix.sum())

    with bundle.fom_projection_csv.open(encoding="utf-8", newline="") as stream:
        fom_rows = list(csv.DictReader(stream))
    assert len(fom_rows) == request.matrix.shape[1]
    assert sum(int(row["included_in_fit"]) for row in fom_rows) == len(request.fit.x)
    assert sum(int(row["projected_counts"]) for row in fom_rows) == int(request.matrix[2:12].sum())
    included_rows = [row for row in fom_rows if row["included_in_fit"] == "1"]
    assert [float(row["fit_observed_counts"]) for row in included_rows] == pytest.approx(
        request.fit.observed
    )
    assert [float(row["fitted_total"]) for row in included_rows] == pytest.approx(
        request.fit.best_fit
    )

    metadata = json.loads(bundle.metadata_json.read_text(encoding="utf-8"))
    assert metadata["application"]["version"] == __version__
    assert metadata["application"]["url"] == "https://ewt.tech"
    assert metadata["histogram"]["cut_bin"] == 64
    assert metadata["processing"]["baseline samples"] == [0, 8]
    assert metadata["fit"]["fom"] == pytest.approx(request.fit.fom)

    settings_text = bundle.settings_csv.read_text(encoding="utf-8")
    assert "energy bin width" in settings_text
    assert "effective cut boundary" in settings_text
    assert "population 1 centroid" in settings_text


@pytest.mark.parametrize("log_y", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_report_energy_spectra_use_panel_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_y: bool, empty: bool
) -> None:
    request = _request(tmp_path / "events.bin")
    if empty:
        request = replace(request, matrix=np.zeros_like(request.matrix))
    request = replace(request, energy_log_y=log_y)
    projection = psd_report._projection_data(request, request.matrix)
    figures = []
    render_figure = psd_report._figure_png
    monkeypatch.setattr(psd_report, "_figure_png", lambda figure: figures.append(figure))

    psd_report._spectra_figure(request, projection)

    axis = figures[0].axes[0]
    assert axis.get_yscale() == ("log" if log_y else "linear")
    if log_y and empty:
        assert axis.get_ylim()[0] > 0
        assert render_figure(figures[0]).getvalue().startswith(b"\x89PNG")


def test_export_psd_report_never_overwrites_an_existing_bundle(tmp_path: Path) -> None:
    source = tmp_path / "events.bin"
    source.write_bytes(b"source")
    request = _request(source)

    first = export_psd_report(request, tmp_path)
    second = export_psd_report(request, tmp_path)

    assert first.directory.name == "events_psd_report"
    assert second.directory.name == "events_psd_report_2"
    assert first.pdf.is_file()
    assert second.pdf.is_file()


def test_capture_digitizer_info_reads_sidecar_and_embedded_configuration(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "scope.bin"
    binary.write_bytes(b"capture")
    binary.with_suffix(".yaml").write_text(
        "connection:\n  ip: 192.168.10.128\n  port: 30431\n  channels: 2\n",
        encoding="utf-8",
    )
    assert capture_digitizer_info(binary) == {
        "device endpoint": "192.168.10.128:30431",
        "configured channels": 2,
    }

    hdf5 = tmp_path / "events.h5"
    with h5py.File(hdf5, "w") as file:
        file.create_dataset(
            "configuration_yaml",
            data="connection:\n  ip: detector.local\n  port: 30431\n",
        )
    assert capture_digitizer_info(hdf5)["device endpoint"] == "detector.local:30431"


def test_capture_digitizer_info_marks_missing_identity(tmp_path: Path) -> None:
    path = tmp_path / "no-metadata.bin"
    path.write_bytes(b"capture")

    assert capture_digitizer_info(path) == {"device endpoint": "Not embedded in source file"}
