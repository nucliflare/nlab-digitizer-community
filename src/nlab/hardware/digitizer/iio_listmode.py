"""Selected client profile for the opaque IIO MCA list-mode record.

The kernel promises only ``opaque[16]`` transport.  The semantic profile below
is PetaLinux's source-derived ``vdpp-zc-calc-q2.14-v1`` contract and must be
selected explicitly by an application; IP121 alone does not identify it.
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
VDPP_ZC_CALC_SCHEMA = "vdpp-zc-calc-q2.14-v1"

COARSE_TICK_NS = 8
ADC_SAMPLE_NS = 2
ZC_FRACTION_SCALE = 1 << 14  # ap_fixed<16, 2>: 14 fractional bits per ADC sample
TIME_Q_PER_NS = ZC_FRACTION_SCALE // ADC_SAMPLE_NS  # exact coordinate: 1/8192 ns
ADC_SAMPLE_Q = ZC_FRACTION_SCALE
COARSE_TICK_Q = COARSE_TICK_NS * TIME_Q_PER_NS
FINE_RAW_MIN = -ZC_FRACTION_SCALE
FINE_RAW_MAX = 0


def cfd_valid(events: np.ndarray) -> np.ndarray:
    """Return CFD results selected by the HLS PSD-before-CFD branch order."""
    marker = events["marker"]
    return np.asarray(
        ((marker & MARKER_CFD_VALID) != 0) & ((marker & MARKER_PSD_ZC_VALID) == 0),
        dtype=np.bool_,
    )


def cfd_fine_in_range(events: np.ndarray) -> np.ndarray:
    """Return whether the signed interpolation is in the producer's valid range."""
    estimate = events["zc_estimation"]
    return np.asarray(
        (estimate >= FINE_RAW_MIN) & (estimate <= FINE_RAW_MAX),
        dtype=np.bool_,
    )


def cfd_interpolation_samples(events: np.ndarray) -> np.ndarray:
    """Decode the signed fractional-sample term, or NaN when it is ineligible."""
    estimate = events["zc_estimation"].astype(np.float64) / ZC_FRACTION_SCALE
    return np.where(cfd_valid(events) & cfd_fine_in_range(events), estimate, np.nan)


def cfd_interpolation_ticks(events: np.ndarray) -> np.ndarray:
    """Compatibility alias; values are ADC-sample fractions, not coarse ticks."""
    return cfd_interpolation_samples(events)


def cfd_correction_q(zc_offset: int, zc_estimation: int) -> int:
    """Return the source-defined CFD correction in exact 1/8192 ns units.

    ``zc_offset`` is an *unsigned* whole ADC-sample count.  It must not be
    heuristically sign-extended when the producer wraps at the uint8 boundary.
    """
    if not 0 <= zc_offset <= 255 or not -32768 <= zc_estimation <= 32767:
        raise ValueError("CFD correction fields are outside their wire ranges")
    if not FINE_RAW_MIN <= zc_estimation <= FINE_RAW_MAX:
        raise ValueError("CFD interpolation is outside the qualified [-1, 0] sample range")
    return zc_offset * ADC_SAMPLE_Q + zc_estimation


def cfd_time_q(timestamp_ticks: int, zc_offset: int, zc_estimation: int) -> int:
    """Reconstruct one CFD coordinate using arbitrary-precision integer math."""
    if timestamp_ticks < 0:
        raise ValueError("coarse timestamp cannot be negative")
    return timestamp_ticks * COARSE_TICK_Q + cfd_correction_q(zc_offset, zc_estimation)
