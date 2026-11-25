"""MCA parameter ABI2 domains from PetaLinux c546f14; no transport access.

Sample fields are original ADC positions (2 ns each), not encoded registers.
This module validates requests; it does not infer detector calibration.

The blocks whose domains are listed here belong to standard digital
pulse-processing families: trapezoidal shaping (Jordanov and Knoll 1994;
Jordanov et al. 1994), recursive CR-(RC)^n shaping (Nakhostin 2011) and
constant-fraction timing (Gedcke and McDonald 1968; digital form in
Fallu-Labruyere et al. 2007). The PetaLinux driver and firmware remain the
authority on their exact implementation.

References
----------
Jordanov, V. T., Knoll, G. F. (1994). Digital synthesis of pulse shapes in
    real time for high resolution radiation spectroscopy. Nucl. Instrum.
    Methods Phys. Res. A 345, 337-345. doi:10.1016/0168-9002(94)91011-1
Jordanov, V. T., Knoll, G. F., Huber, A. C., Pantazis, J. A. (1994).
    Digital techniques for real-time pulse shaping in radiation measurements.
    Nucl. Instrum. Methods Phys. Res. A 353, 261-264.
    doi:10.1016/0168-9002(94)91652-7
Nakhostin, M. (2011). Recursive algorithms for real-time digital
    CR-(RC)^n pulse shaping. IEEE Trans. Nucl. Sci. 58, 2378-2381.
    doi:10.1109/TNS.2011.2164556
Gedcke, D. A., McDonald, W. J. (1968). Design of the constant fraction of
    pulse height trigger for optimum time resolution. Nucl. Instrum. Methods
    58, 253-260. doi:10.1016/0029-554X(68)90473-4
Fallu-Labruyere, A., Tan, H., Hennig, W., Warburton, W. K. (2007). Time
    resolution studies using digital constant fraction discrimination.
    Nucl. Instrum. Methods Phys. Res. A 579, 247-251.
    doi:10.1016/j.nima.2007.04.048
"""

from __future__ import annotations

from collections.abc import Mapping

PARAMETER_ABI_VERSION = 2
PARAMETER_UNITS = "adc_samples"
PARAMETERS: dict[str, tuple[int, int, int]] = {
    "external_trigger_enable": (0, 1, 1),
    "measurement_time_raw": (0, 4294967295, 1),
    "trigger_source": (0, 2, 1),
    "trigger_level_raw": (-32768, 32767, 1),
    "pulse_polarity": (0, 1, 1),
    "baseline_window": (0, 6, 1),
    "pileup_window_raw": (0, 255, 1),
    "energy_bin": (0, 9, 1),
    "pretrigger_samples": (0, 2044, 4),
    "frame_samples": (4, 4096, 4),
    "cfd_enable": (0, 1, 1),
    "cfd_factor_raw": (0, 32767, 1),
    "cfd_delay_samples": (5, 132, 1),
    "cfd_window_low_samples": (0, 32764, 4),
    "cfd_window_high_samples": (0, 32764, 4),
    "charge_comparison_enable": (0, 1, 1),
    "charge_comparison_samples": (4, 32768, 4),
    "crrc2_cdelay": (0, 6, 1),
    "crrc2_fdelay_samples": (4, 508, 4),
    "crrc2_pzc_raw": (-32768, 32767, 1),
    "trapezoid_enable": (0, 1, 1),
    "trapezoid_r_samples": (8, 2044, 4),
    "trapezoid_m_samples": (0, 2044, 4),
    "trapezoid_beta_raw": (0, 4294967295, 1),
    "trapezoid_start_offset_samples": (0, 8188, 4),
    "trapezoid_flat_top_window": (0, 6, 1),
    "psd_zero_crossing_enable": (0, 1, 1),
    "psd_zero_crossing_mode": (0, 2, 1),
    "psd_window_low_samples": (0, 32764, 4),
    "psd_window_high_samples": (0, 32764, 4),
}

# Explicit, disabled-algorithm startup configuration, not a detector calibration.
DEFAULT_PARAMETERS = {name: bounds[0] for name, bounds in PARAMETERS.items()}
DEFAULT_PARAMETERS.update(
    {
        "trigger_level_raw": -512,
        "baseline_window": 3,
        "pretrigger_samples": 12,
        "frame_samples": 128,
        "cfd_factor_raw": 13107,
        "cfd_delay_samples": 5,
        # Window starts at the trigger beat (low >= pretrigger, P2).
        "cfd_window_low_samples": 12,
        "cfd_window_high_samples": 44,
        "crrc2_cdelay": 3,
        "crrc2_pzc_raw": 32761,
        "trapezoid_r_samples": 32,
        "trapezoid_m_samples": 32,
        "trapezoid_start_offset_samples": 64,
        "psd_window_high_samples": 8,
    }
)


def validate_parameters(values: Mapping[str, object], *, complete: bool = False) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, value in values.items():
        if name not in PARAMETERS:
            raise ValueError(f"Unknown or obsolete MCA parameter: {name}")
        lo, hi, step = PARAMETERS[name]
        if type(value) is not int or not lo <= value <= hi or (value - lo) % step:
            raise ValueError(f"{name}: expected integer {lo}..{hi}, step {step}; got {value!r}")
        result[name] = value
    if complete:
        missing = set(PARAMETERS) - set(result)
        if missing:
            raise ValueError(f"Incomplete MCA configuration: {sorted(missing)}")
        validate_active_configuration(result)
    return result


def validate_active_configuration(values: Mapping[str, int]) -> None:
    """Mirror kernel admission, without claiming downstream STOP completion."""
    frame = values["frame_samples"]
    for prefix, enable in (("cfd", "cfd_enable"), ("psd", "psd_zero_crossing_enable")):
        if values[enable]:
            low, high = (
                values[f"{prefix}_window_low_samples"],
                values[f"{prefix}_window_high_samples"],
            )
            if not 0 <= low <= high < frame:
                raise ValueError(
                    f"Active {prefix.upper()} window must satisfy low <= high < pulse length"
                )
    if values["cfd_enable"] and values["cfd_delay_samples"] >= frame:
        raise ValueError("Active CFD delay must be shorter than the pulse length")
    if values["charge_comparison_enable"] and values["charge_comparison_samples"] > frame:
        raise ValueError("Active CC integration must fit the pulse length")
    if values["trapezoid_enable"]:
        extent = values["trapezoid_start_offset_samples"] + 4 * (
            1 << values["trapezoid_flat_top_window"]
        )
        if extent > frame:
            raise ValueError("Trapezoid start plus averaging window must fit the pulse length")
    pileup = values["pileup_window_raw"]
    # pp_validate_config_locked(): each pile-up extension adds frame/4 - 1
    # beats to an 8-bit HLS counter.
    beats = (frame // 4 - 1) * (pileup + 1)
    if pileup and beats > 255:
        longest = 4 * (255 // (pileup + 1) + 1)
        raise ValueError(
            f"Pile-ups {pileup} with a {2 * frame} ns window ({frame} samples) overflows "
            f"the 8-bit pile-up counter: ({frame}/4 - 1) x ({pileup} + 1) = {beats} > 255. "
            f"Use a window of at most {2 * longest} ns or fewer pile-ups."
        )


def validate_document(document: Mapping[str, object]) -> dict[str, int]:
    if (
        type(document.get("parameter_abi_version")) is not int
        or document["parameter_abi_version"] != 2
    ):
        raise ValueError(
            "MCA settings require parameter_abi_version=2; review older presets explicitly"
        )
    if document.get("parameter_units") != PARAMETER_UNITS:
        raise ValueError("MCA settings require parameter_units=adc_samples")
    values = document.get("parameters")
    if not isinstance(values, dict) or not values:
        raise ValueError("MCA settings require a nonempty parameters mapping")
    return validate_parameters(values, complete=True)
