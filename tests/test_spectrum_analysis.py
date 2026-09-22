"""Spectrum model, arithmetic, and file-adapter checks."""

from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from nlab.analysis.spectrum import (
    Spectrum,
    combine_spectra,
    crop_spectrum,
    normalize_spectrum,
    rebin_spectrum,
    scale_spectrum,
    subtract_background,
)
from nlab.analysis.spectrum_io import (
    export_spectrum_csv,
    load_caen_text_spectrum,
    load_csv_spectrum,
    load_root_spectra,
    load_spe_spectrum,
    load_spectra,
    load_spectrum,
    load_wdm_spectrum,
)

_TUKAN_VERSION = 0x0007EEEE


def _pascal(value: str) -> bytes:
    encoded = value.encode("cp1250")
    return bytes((len(encoded),)) + encoded


def _wdm_payload(
    counts: list[int],
    *,
    integrity_code: int = _TUKAN_VERSION,
) -> bytes:
    started = datetime(2017, 4, 24, 8, 45, 12, 273000)
    tdatetime = (started - datetime(1899, 12, 30)).total_seconds() / 86400.0
    trailer = (
        struct.pack(">I", integrity_code)
        + struct.pack("<dB", 25.0, 1)
        + _pascal("Tukan8k-USB")
        + _pascal("18")
        + _pascal("geom")
        + _pascal("Cs-137 widmo")
        + _pascal("opis próbki")
        + struct.pack("<diiI", tdatetime, 86, 78, 0)
    )
    return (
        struct.pack(">I", _TUKAN_VERSION)
        + struct.pack("<H", len(counts))
        + np.asarray(counts, dtype="<i4").tobytes()
        + trailer
    )


def _spectrum(label: str, counts: list[float], *, elapsed: float = 10.0) -> Spectrum:
    return Spectrum.create(
        label=label,
        counts=np.asarray(counts),
        source="mca",
        metadata={"elapsed_s": elapsed},
    )


def test_spectrum_is_immutable_and_arithmetic_propagates_variance() -> None:
    left = _spectrum("sample", [10, 20, 30])
    right = _spectrum("background", [2, 4, 6], elapsed=5.0)

    corrected = subtract_background(left, right)

    np.testing.assert_allclose(corrected.counts, [6, 12, 18])
    np.testing.assert_allclose(corrected.variances, [18, 36, 54])
    assert not corrected.poisson_counts
    assert "subtract" in corrected.history[-1]
    with pytest.raises(ValueError):
        corrected.counts[0] = 1


def test_scale_normalize_combine_rebin_and_crop() -> None:
    spectrum = _spectrum("source", [1, 2, 3, 4])

    scaled = scale_spectrum(spectrum, 2)
    normalized = normalize_spectrum(spectrum, "area")
    added = combine_spectra(spectrum, spectrum, operation="add")
    rebinned = rebin_spectrum(spectrum, 3)
    cropped = crop_spectrum(spectrum, 1, 2)

    np.testing.assert_allclose(scaled.counts, [2, 4, 6, 8])
    np.testing.assert_allclose(scaled.variances, [4, 8, 12, 16])
    assert np.sum(normalized.counts) == pytest.approx(1)
    np.testing.assert_allclose(added.counts, [2, 4, 6, 8])
    assert added.poisson_counts
    np.testing.assert_allclose(rebinned.counts, [6, 4])
    np.testing.assert_allclose(rebinned.variances, [6, 4])
    np.testing.assert_allclose(cropped.counts, [2, 3])


def test_arithmetic_rejects_different_coordinate_grids() -> None:
    left = _spectrum("left", [1, 2])
    right = Spectrum.create(
        label="right",
        x=np.asarray([0.0, 2.0]),
        counts=np.asarray([1.0, 2.0]),
        source="csv",
    )

    with pytest.raises(ValueError, match="identical coordinate grids"):
        combine_spectra(left, right, operation="subtract")


def test_native_csv_round_trip_preserves_energy_counts_and_variance(tmp_path: Path) -> None:
    source = Spectrum.create(
        label="calibrated",
        counts=np.asarray([4.0, 9.0, 16.0]),
        variances=np.asarray([5.0, 10.0, 17.0]),
        source="derived",
        metadata={"elapsed_s": 12.5},
        history=("scale(0.5)",),
        energy_kev=np.asarray([100.0, 101.0, 102.0]),
        poisson_counts=False,
    )
    path = tmp_path / "spectrum.csv"

    export_spectrum_csv(path, source)
    loaded = load_csv_spectrum(path)

    assert loaded.axis_unit == "channel"
    np.testing.assert_allclose(loaded.counts, source.counts)
    np.testing.assert_allclose(loaded.variances, source.variances)
    np.testing.assert_allclose(loaded.energy_kev, source.energy_kev)
    assert not loaded.poisson_counts
    assert loaded.elapsed_s == 12.5
    assert loaded.history == ("scale(0.5)",)


def test_plain_counts_csv_and_ascii_spe_import(tmp_path: Path) -> None:
    csv_path = tmp_path / "plain.csv"
    csv_path.write_text("counts\n1\n2\n3\n", encoding="utf-8")
    spe_path = tmp_path / "sample.Spe"
    spe_path.write_text(
        "$SPEC_ID:\nCs137\n$MEAS_TIM:\n8 10\n$DATA:\n2 4\n10\n20\n30\n"
        "$ENER_FIT:\n1.5 0.5\n",
        encoding="utf-8",
    )

    plain = load_csv_spectrum(csv_path)
    spe = load_spe_spectrum(spe_path)

    np.testing.assert_allclose(plain.x, [0, 1, 2])
    np.testing.assert_allclose(plain.counts, [1, 2, 3])
    assert spe.label == "Cs137"
    np.testing.assert_allclose(spe.x, [2, 3, 4])
    np.testing.assert_allclose(spe.energy_kev, [2.5, 3.0, 3.5])
    assert spe.metadata["live_time_s"] == 8
    assert spe.metadata["elapsed_s"] == 10


def test_caen_txt3_import_preserves_calibration_and_timing(tmp_path: Path) -> None:
    path = tmp_path / "CH4@DT5730_666_EspectrumR.txt3"
    path.write_text(
        "C0 = 1.5; C1 = 0.5; C2 = 0; unit = keV\n"
        "RealTime = 0:57:00.295\n"
        "LiveTime = 0:55:31.740\n"
        "0 4 1.5\n"
        "1 9 2.0\n"
        "2 16 2.5\n",
        encoding="utf-8",
    )

    loaded = load_caen_text_spectrum(path)

    assert loaded.source == "caen_text"
    np.testing.assert_allclose(loaded.x, [0, 1, 2])
    np.testing.assert_allclose(loaded.counts, [4, 9, 16])
    np.testing.assert_allclose(loaded.energy_kev, [1.5, 2.0, 2.5])
    assert loaded.metadata["energy_calibration_coefficients_kev"] == [1.5, 0.5, 0.0]
    assert loaded.metadata["elapsed_s"] == pytest.approx(3420.295)
    assert loaded.metadata["live_time_s"] == pytest.approx(3331.740)
    assert load_spectrum(path).source == "caen_text"


@pytest.mark.parametrize(
    ("prefix", "processing"),
    [("_F_", "filtered"), ("_R_", "raw")],
)
def test_root_import_loads_every_energy_histogram_separately(
    tmp_path: Path,
    prefix: str,
    processing: str,
) -> None:
    uproot = pytest.importorskip("uproot")
    path = tmp_path / "compass.root"
    edges = np.arange(5, dtype=np.float64)
    with uproot.recreate(path) as output:
        output[f"Energy/{prefix}EnergyCH0@DT5730_666"] = (np.asarray([2, 3, 5, 7]), edges)
        output[f"Energy/{prefix}EnergyCH1@DT5730_666"] = (np.asarray([0, 0, 0, 0]), edges)
        output[f"Energy/{prefix}EnergyCH2@DT5730_666"] = (
            np.asarray([11, 13, 17, 19]),
            edges,
        )
        output["Time/TimeCH0@DT5730_666"] = (np.asarray([99, 1]), np.arange(3))

    spectra = load_root_spectra(path)

    assert len(spectra) == 3
    assert [spectrum.metadata["channel"] for spectrum in spectra] == [0, 1, 2]
    assert all(spectrum.metadata["caen_processing"] == processing for spectrum in spectra)
    assert all(spectrum.source == "root" for spectrum in spectra)
    np.testing.assert_allclose(spectra[0].x, [0, 1, 2, 3])
    np.testing.assert_allclose(spectra[0].counts, [2, 3, 5, 7])
    np.testing.assert_allclose(spectra[1].counts, [0, 0, 0, 0])
    assert len(load_spectra(path)) == 3
    with pytest.raises(ValueError, match="contains 3 energy spectra"):
        load_spectrum(path)


def test_tukan_wdm_import_decodes_counts_and_fixed_metadata(tmp_path: Path) -> None:
    path = tmp_path / "legacy.WDM"
    path.write_bytes(_wdm_payload([0, 4, 9, 16]))

    loaded = load_wdm_spectrum(path)

    assert loaded.source == "wdm"
    assert loaded.label == "legacy (Cs-137 widmo)"
    np.testing.assert_allclose(loaded.x, [0, 1, 2, 3])
    np.testing.assert_allclose(loaded.counts, [0, 4, 9, 16])
    assert loaded.metadata["wdm_version_hex"] == "0x0007EEEE"
    assert loaded.metadata["wdm_integrity_matches_version"] is True
    assert loaded.metadata["sample_mass"] == 25.0
    assert loaded.metadata["sample_mass_unit_code"] == 1
    assert loaded.metadata["analyzer_type"] == "Tukan8k-USB"
    assert loaded.metadata["analyzer_serial_number"] == "18"
    assert loaded.metadata["spectrum_description"] == "opis próbki"
    assert loaded.metadata["started_local"] == "2017-04-24T08:45:12.273"
    assert loaded.metadata["elapsed_s"] == 86
    assert loaded.metadata["live_time_s"] == 78
    assert loaded.normalization_time_s == 78
    assert load_spectrum(path).source == "wdm"


def test_tukan_wdm_filename_keeps_duplicate_internal_titles_distinct(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "temperature_0.wdm"
    second_path = tmp_path / "temperature_-50.wdm"
    first_path.write_bytes(_wdm_payload([1, 2]))
    second_path.write_bytes(_wdm_payload([3, 4]))

    first = load_wdm_spectrum(first_path)
    second = load_wdm_spectrum(second_path)

    assert first.metadata["spectrum_name"] == second.metadata["spectrum_name"]
    assert first.label == "temperature_0 (Cs-137 widmo)"
    assert second.label == "temperature_-50 (Cs-137 widmo)"


def test_tukan_wdm_import_rejects_truncation_and_corruption(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.wdm"
    truncated.write_bytes(
        struct.pack(">I", _TUKAN_VERSION) + struct.pack("<H", 4) + bytes(12)
    )
    mismatch = tmp_path / "mismatch.wdm"
    mismatch.write_bytes(_wdm_payload([1, 2], integrity_code=0x12345678))
    negative = tmp_path / "negative.wdm"
    negative.write_bytes(
        struct.pack(">I", _TUKAN_VERSION)
        + struct.pack("<H", 1)
        + struct.pack("<i", -1)
    )

    with pytest.raises(ValueError, match="declares 4 channels"):
        load_wdm_spectrum(truncated)
    with pytest.raises(ValueError, match="integrity code"):
        load_wdm_spectrum(mismatch)
    with pytest.raises(ValueError, match="cannot be negative"):
        load_wdm_spectrum(negative)


def test_tukan_wdm_import_accepts_legacy_zero_filled_trailer(tmp_path: Path) -> None:
    path = tmp_path / "old_script.wdm"
    path.write_bytes(
        struct.pack(">I", _TUKAN_VERSION)
        + struct.pack("<H", 3)
        + struct.pack("<iii", 2, 3, 5)
        + bytes(53)
    )

    loaded = load_wdm_spectrum(path)

    assert loaded.label == "old_script"
    np.testing.assert_allclose(loaded.counts, [2, 3, 5])
    assert loaded.metadata["wdm_integrity_code"] == 0
    assert loaded.metadata["wdm_integrity_matches_version"] is False


def test_whitespace_delimited_text_import(tmp_path: Path) -> None:
    path = tmp_path / "columns.txt"
    path.write_text("channel counts\n0 4\n1 9\n2 16\n", encoding="utf-8")

    loaded = load_csv_spectrum(path)

    np.testing.assert_allclose(loaded.x, [0, 1, 2])
    np.testing.assert_allclose(loaded.counts, [4, 9, 16])


def test_exported_raw_counts_remain_poisson_eligible(tmp_path: Path) -> None:
    source = _spectrum("raw", [3, 5, 8])
    path = tmp_path / "raw.csv"

    export_spectrum_csv(path, source)

    assert load_csv_spectrum(path).poisson_counts


def test_csv_export_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "existing.csv"
    path.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_spectrum_csv(path, _spectrum("source", [1, 2]))
