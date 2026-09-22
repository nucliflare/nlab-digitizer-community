"""Display, projection, and export helpers for coincidence energy matrices."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np

from nlab.analysis.coincidence import (
    COINCIDENCE_MATRIX_BINS,
    COINCIDENCE_MATRIX_CHANNELS_PER_BIN,
    HISTOGRAM_BINS,
)

MatrixMode = Literal["prompt", "random", "corrected"]


def matrix_channel_edges() -> np.ndarray:
    """Return the fixed raw-MCA-channel edges of the 512 matrix bins."""
    return np.arange(COINCIDENCE_MATRIX_BINS + 1, dtype=np.float64) * (
        COINCIDENCE_MATRIX_CHANNELS_PER_BIN
    )


def select_matrix(
    prompt: np.ndarray,
    random: np.ndarray,
    mode: MatrixMode,
    *,
    random_scale: float = 0.5,
) -> np.ndarray:
    """Select prompt/random counts or calculate prompt-minus-random counts."""
    _validate_matrix_pair(prompt, random)
    if mode == "prompt":
        return np.asarray(prompt)
    if mode == "random":
        return np.asarray(random)
    if mode == "corrected":
        return prompt.astype(np.float64) - random_scale * random.astype(np.float64)
    raise ValueError(f"unsupported coincidence matrix mode {mode!r}")


def display_matrix(values: np.ndarray, *, logarithmic: bool) -> np.ndarray:
    """Transform matrix intensities while preserving corrected-count signs."""
    result = np.asarray(values, dtype=np.float64)
    if not logarithmic:
        return result
    return np.sign(result) * np.log10(1.0 + np.abs(result))


def matrix_projections(
    matrix: np.ndarray,
    *,
    ch0_gate: tuple[float, float],
    ch1_gate: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Project CH0 for a CH1 gate and CH1 for a CH0 gate.

    Matrices use row-major ``[ch1, ch0]`` order. Returned arrays therefore
    contain ``(ch0_projection, ch1_projection)`` respectively.
    """
    _validate_matrix(matrix)
    ch0_slice = _gate_slice(ch0_gate)
    ch1_slice = _gate_slice(ch1_gate)
    ch0_projection = np.sum(matrix[ch1_slice, :], axis=0)
    ch1_projection = np.sum(matrix[:, ch0_slice], axis=1)
    return np.asarray(ch0_projection), np.asarray(ch1_projection)


def export_coincidence_matrix(
    path: Path,
    *,
    prompt: np.ndarray,
    random: np.ndarray,
    random_scale: float,
    metadata: Mapping[str, object],
) -> None:
    """Write prompt, random, and corrected matrices to HDF5 or ROOT.

    HDF5 preserves the native ``[ch1, ch0]`` array order. ROOT TH2 objects use
    the conventional ``[x=ch0, y=ch1]`` order and are transposed accordingly.
    Existing files are never overwritten.
    """
    _validate_matrix_pair(prompt, random)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    corrected = select_matrix(prompt, random, "corrected", random_scale=random_scale)
    edges = matrix_channel_edges()
    encoded_metadata = json.dumps(dict(metadata), sort_keys=True, default=str)

    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "x") as output:
            output.attrs["kind"] = "two_channel_coincidence_energy_matrix"
            output.attrs["axis_order"] = "rows=ch1, columns=ch0"
            output.attrs["random_scale"] = float(random_scale)
            output.attrs["metadata_json"] = encoded_metadata
            output.create_dataset("energy_channel_edges", data=edges)
            output.create_dataset("prompt_counts", data=prompt, compression="gzip", shuffle=True)
            output.create_dataset("random_counts", data=random, compression="gzip", shuffle=True)
            output.create_dataset(
                "prompt_minus_scaled_random",
                data=corrected,
                compression="gzip",
                shuffle=True,
            )
        return
    if suffix == ".root":
        import uproot

        with uproot.recreate(path) as output:
            output["prompt"] = (prompt.T.astype(np.float64), edges, edges)
            output["random"] = (random.T.astype(np.float64), edges, edges)
            output["prompt_minus_scaled_random"] = (corrected.T, edges, edges)
            output["metadata"] = encoded_metadata
        return
    raise ValueError("coincidence matrix export must use .h5, .hdf5, or .root")


def _gate_slice(gate: tuple[float, float]) -> slice:
    low, high = sorted(map(float, gate))
    low = float(np.clip(low, 0.0, HISTOGRAM_BINS))
    high = float(np.clip(high, 0.0, HISTOGRAM_BINS))
    first = int(np.floor(low / COINCIDENCE_MATRIX_CHANNELS_PER_BIN))
    last = int(np.ceil(high / COINCIDENCE_MATRIX_CHANNELS_PER_BIN))
    first = int(np.clip(first, 0, COINCIDENCE_MATRIX_BINS))
    last = int(np.clip(last, first, COINCIDENCE_MATRIX_BINS))
    return slice(first, last)


def _validate_matrix(matrix: np.ndarray) -> None:
    if np.shape(matrix) != (COINCIDENCE_MATRIX_BINS, COINCIDENCE_MATRIX_BINS):
        raise ValueError(
            f"coincidence matrix must be {COINCIDENCE_MATRIX_BINS}x{COINCIDENCE_MATRIX_BINS}"
        )


def _validate_matrix_pair(prompt: np.ndarray, random: np.ndarray) -> None:
    _validate_matrix(prompt)
    _validate_matrix(random)
