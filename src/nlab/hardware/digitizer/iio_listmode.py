"""Client-side interpretation of the opaque IIO MCA list-mode record.

The PetaLinux ``vdpp-lm-frame`` driver transfers opaque 16-byte records. The
layout below follows the maintainer-supplied ``vdpp_zc_calc.hpp/.cpp`` output
structure and was checked against a CFD-enabled NDMA capture. It is not an
IIO scan-channel contract; keep transport geometry and event semantics apart.
The optional ``flags/cfd_q2`` interpretation in PetaLinux's user-api notes
disagrees with this output structure and the captured marker bytes.
"""

from __future__ import annotations

import numpy as np

IIO_LM_EVENT_DTYPE = np.dtype(
    [
        ("marker", "u1"),
        ("zc_offset", "u1"),
        ("zc_estimation", "<i2"),
        ("charge_energy", "<u2"),
        ("trapezoid_energy", "<u2"),
        ("timestamp", "<u8"),
    ]
)

MARKER_CFD_VALID = 1 << 1
MARKER_PSD_ZC_VALID = 1 << 3
MARKER_INPUT1 = 1 << 6
MARKER_INPUT0 = 1 << 7
ZC_FRACTION_SCALE = 1 << 14  # ap_fixed<16, 2>: 14 fractional bits


def cfd_valid(events: np.ndarray) -> np.ndarray:
    """Return CFD results selected by the HLS PSD-before-CFD branch order."""
    marker = events["marker"]
    return np.asarray(
        ((marker & MARKER_CFD_VALID) != 0) & ((marker & MARKER_PSD_ZC_VALID) == 0),
        dtype=np.bool_,
    )


def cfd_interpolation_ticks(events: np.ndarray) -> np.ndarray:
    """Decode the signed CFD interpolation term, or NaN if CFD was not selected.

    The supplied HLS stage defines the Q2.14 estimate but only forwards the
    unsigned offset and coarse timestamp from an unavailable upstream stage.
    A capture contains wrapped-looking offset bytes such as 0xFD; their
    time origin and signedness cannot safely be inferred from that alone.
    """
    estimate = events["zc_estimation"].astype(np.float64) / ZC_FRACTION_SCALE
    return np.where(cfd_valid(events), estimate, np.nan)


def provisional_cfd_correction_q14(zc_offset: int, zc_estimation: int) -> int:
    """Candidate offset in Q14 8 ns ticks for live CFD timing validation.

    The HLS output declares ``zc_offset`` unsigned, but captured 0xFD..0xFF
    values look like wrapped negative sample offsets. Interpret the byte as
    signed two's-complement and add the signed Q2.14 interpolation. The
    upstream timestamp anchor is unavailable, so this is an *experimental*
    client reconstruction, not a firmware-verified absolute event time.
    """
    if not 0 <= zc_offset <= 255 or not -32768 <= zc_estimation <= 32767:
        raise ValueError("CFD correction fields are outside their wire ranges")
    signed_offset = zc_offset - 256 if zc_offset >= 128 else zc_offset
    return signed_offset * ZC_FRACTION_SCALE + zc_estimation
