"""Coincidence matrix display, projection, and export checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nlab.analysis.coincidence import COINCIDENCE_MATRIX_BINS
from nlab.analysis.coincidence_matrix import (
    display_matrix,
    export_coincidence_matrix,
    matrix_channel_edges,
    matrix_projections,
    select_matrix,
)


def _matrices() -> tuple[np.ndarray, np.ndarray]:
    prompt = np.zeros((COINCIDENCE_MATRIX_BINS, COINCIDENCE_MATRIX_BINS), dtype=np.uint64)
    random = np.zeros_like(prompt)
    prompt[20, 10] = 7
    random[20, 10] = 2
    return prompt, random


def test_matrix_selection_log_transform_and_raw_channel_edges() -> None:
    prompt, random = _matrices()

    corrected = select_matrix(prompt, random, "corrected", random_scale=0.5)

    assert corrected[20, 10] == 6
    assert matrix_channel_edges()[[0, 1, -1]].tolist() == [0, 32, 16_384]
    assert display_matrix(corrected, logarithmic=True)[20, 10] == pytest.approx(np.log10(7))


def test_matrix_projections_gate_the_opposite_detector_axis() -> None:
    matrix = np.zeros((COINCIDENCE_MATRIX_BINS, COINCIDENCE_MATRIX_BINS), dtype=np.float64)
    matrix[20, 10] = 4
    matrix[30, 10] = 3
    matrix[20, 40] = 2

    ch0, ch1 = matrix_projections(
        matrix,
        ch0_gate=(10 * 32, 11 * 32),
        ch1_gate=(20 * 32, 21 * 32),
    )

    assert ch0[10] == 4
    assert ch0[40] == 2
    assert ch1[20] == 4
    assert ch1[30] == 3


@pytest.mark.parametrize("suffix", [".h5", ".root"])
def test_matrix_export_preserves_prompt_random_and_metadata(tmp_path: Path, suffix: str) -> None:
    prompt, random = _matrices()
    path = tmp_path / f"matrix{suffix}"

    export_coincidence_matrix(
        path,
        prompt=prompt,
        random=random,
        random_scale=0.5,
        metadata={"session_id": "test"},
    )

    if suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as source:
            assert source.attrs["axis_order"] == "rows=ch1, columns=ch0"
            assert source["prompt_counts"][20, 10] == 7
            assert source["random_counts"][20, 10] == 2
            assert source["prompt_minus_scaled_random"][20, 10] == 6
            assert '"session_id": "test"' in source.attrs["metadata_json"]
    else:
        import uproot

        with uproot.open(path) as source:
            assert source["prompt"].values()[10, 20] == 7
            assert source["random"].values()[10, 20] == 2
            assert source["prompt_minus_scaled_random"].values()[10, 20] == 6


def test_matrix_export_never_overwrites(tmp_path: Path) -> None:
    prompt, random = _matrices()
    path = tmp_path / "matrix.h5"
    path.touch()

    with pytest.raises(FileExistsError):
        export_coincidence_matrix(
            path,
            prompt=prompt,
            random=random,
            random_scale=0.5,
            metadata={},
        )
