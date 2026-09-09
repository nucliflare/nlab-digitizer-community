"""Versioned, human-readable YAML persistence for GUI and hardware settings.

Version 2 stores every connected channel in one document. Hardware-backed
values live under ``hardware``; presentation and worker values live under
``application`` so non-hardware state is explicit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from nlab.hardware.digitizer.hv import HVSupply
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.hardware.digitizer.scope import Scope, TriggerMode

log = logging.getLogger(__name__)
FORMAT_VERSION = 2


def read_configuration(path: Path) -> dict[str, Any]:
    """Read and minimally validate a configuration document."""
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Settings YAML must contain a mapping at its top level")
    return loaded


def write_configuration(path: Path, settings: Mapping[str, Any]) -> None:
    """Write one complete configuration document."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(dict(settings), stream, sort_keys=False, allow_unicode=True)
    log.info("Settings saved to %s", path)


def connection_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    value = settings.get("connection", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, {})
    return value if isinstance(value, Mapping) else {}


def channel_entry(settings: Mapping[str, Any], channel: int) -> dict[str, Any] | None:
    """Find a v2 hardware channel, accepting YAML string or integer keys."""
    channels = _mapping(_mapping(settings, "hardware"), "channels")
    value = channels.get(str(channel))
    return dict(value) if isinstance(value, Mapping) else None


def channel_application_entry(
    settings: Mapping[str, Any],
    channel: int,
) -> dict[str, Any] | None:
    channels = _mapping(_mapping(settings, "application"), "channels")
    value = channels.get(str(channel))
    return dict(value) if isinstance(value, Mapping) else None


def collect_channel_hardware(
    scope: Scope,
    mca: MultiChannelAnalyzer | None,
    hv: HVSupply | None,
    *,
    mca_lp_preset: int | None = None,
    psu_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read every user-configurable hardware value for one channel.

    IDS setpoints and the MCA LP preset are write-only, so their controllers
    supply current editor values explicitly.
    """
    settings: dict[str, Any] = {
        "scope": {
            "trigger_level": scope.get_trigger_level(),
            "pretrigger_samples": scope.get_pretrigger_samples(),
            "frame_samples": scope.get_frame_samples(),
            "frame_period_cycles": scope.get_frame_period_cycles(),
            "edge_mode": scope.get_trigger_mode().value,
            "dac_value": scope.get_dac_value(),
            "dma_enabled": scope.get_dma_enable(),
        }
    }
    if mca is not None:
        mca_settings: dict[str, Any] = {
            "signal": {
                "pulse_polarity": mca.get_pulse_polarity(),
                "trigger_level": mca.get_trigger_level(),
                "baseline_window": mca.get_baseline_window(),
                "pretrigger_samples": mca.get_pretrigger_samples(),
                "frame_samples": mca.get_frame_samples(),
                "trg_source": mca.get_trg_source(),
                "ext_trig_enable": mca.get_ext_trig_enable(),
                "edge_det_coeff": mca.get_edge_det_coeff(),
            },
            "acquisition": {
                "energy_bin": mca.get_energy_bin(),
                "pileup_window": mca.get_pileup_window(),
                "time_limit": mca.get_time_limit(),
                "dma_enabled": mca.get_dma_enable(),
            },
            "debug": {
                "mem1_sig_select": mca.get_mem1_sig_select(),
                "mem2_sig_select": mca.get_mem2_sig_select(),
            },
            "crrc2": {
                "cdelay": mca.filters.crrc2.get_Cdelay(),
                "fdelay": mca.filters.crrc2.get_Fdelay(),
                "pzc": mca.filters.crrc2.get_pzc_coeff(),
            },
            "cfd": {
                "enable": mca.filters.cfd.get_enable(),
                "factor": mca.filters.cfd.get_factor(),
                "delay": mca.filters.cfd.get_delay(),
                "tw_low": mca.filters.cfd.get_time_window_low(),
                "tw_high": mca.filters.cfd.get_time_window_high(),
            },
            "trapezoid": {
                "enable": mca.filters.trapezoid.get_enable(),
                "r": mca.filters.trapezoid.get_R(),
                "m": mca.filters.trapezoid.get_M(),
                "t": mca.filters.trapezoid.get_T(),
                "e": mca.filters.trapezoid.get_E(),
                "ft": mca.filters.trapezoid.get_FT(),
            },
            "charge_comparison": {
                "enable": mca.filters.charge_comparison.get_enable(),
                "time": mca.filters.charge_comparison.get_time(),
            },
            "psd_zc": {
                "enable": mca.filters.psd_zc.get_enable(),
                "mode": mca.filters.psd_zc.get_mode(),
                "low": mca.filters.psd_zc.get_time_window_low(),
                "high": mca.filters.psd_zc.get_time_window_high(),
            },
        }
        if mca_lp_preset is not None:
            mca_settings["low_pass"] = {"preset": mca_lp_preset}
        settings["mca"] = mca_settings
    if hv is not None and psu_settings is not None:
        settings["psu"] = dict(psu_settings)
    return settings


def _apply_ints(
    settings: Mapping[str, Any], setters: tuple[tuple[str, Callable[[int], None]], ...]
) -> None:
    for key, setter in setters:
        if key in settings:
            setter(int(settings[key]))


def _apply_floats(
    settings: Mapping[str, Any], setters: tuple[tuple[str, Callable[[float], None]], ...]
) -> None:
    for key, setter in setters:
        if key in settings:
            setter(float(settings[key]))


def _apply_bools(
    settings: Mapping[str, Any], setters: tuple[tuple[str, Callable[[bool], None]], ...]
) -> None:
    for key, setter in setters:
        if key in settings:
            setter(bool(settings[key]))


def _apply_mca(mca: MultiChannelAnalyzer, settings: Mapping[str, Any]) -> None:
    signal = _mapping(settings, "signal")
    _apply_ints(
        signal,
        (
            ("pulse_polarity", mca.set_pulse_polarity),
            ("trigger_level", mca.set_trigger_level),
            ("baseline_window", mca.set_baseline_window),
            ("pretrigger_samples", mca.set_pretrigger_samples),
            ("frame_samples", mca.set_frame_samples),
            ("trg_source", mca.set_trg_source),
            ("edge_det_coeff", mca.set_edge_det_coeff),
        ),
    )
    _apply_bools(signal, (("ext_trig_enable", mca.set_ext_trig_enable),))

    acquisition = _mapping(settings, "acquisition")
    _apply_ints(
        acquisition,
        (
            ("energy_bin", mca.set_energy_bin),
            ("pileup_window", mca.set_pileup_window),
            ("time_limit", mca.set_time_limit),
        ),
    )
    _apply_bools(acquisition, (("dma_enabled", mca.set_dma_enable),))

    debug = _mapping(settings, "debug")
    if "mem1_sig_select" in debug:
        mca.set_mem1_sig_select(int(debug["mem1_sig_select"]))
    if "mem2_sig_select" in debug:
        mca.set_mem2_sig_select(int(debug["mem2_sig_select"]))

    low_pass = _mapping(settings, "low_pass")
    if "preset" in low_pass:
        mca.filters.lp.set_preset(int(low_pass["preset"]))

    crrc2 = _mapping(settings, "crrc2")
    for key, setter in (
        ("cdelay", mca.filters.crrc2.set_Cdelay),
        ("fdelay", mca.filters.crrc2.set_Fdelay),
        ("pzc", mca.filters.crrc2.set_pzc_coeff),
    ):
        if key in crrc2:
            setter(int(crrc2[key]))

    cfd = _mapping(settings, "cfd")
    _apply_bools(cfd, (("enable", mca.filters.cfd.set_enable),))
    _apply_floats(cfd, (("factor", mca.filters.cfd.set_factor),))
    _apply_ints(
        cfd,
        (
            ("delay", mca.filters.cfd.set_delay),
            ("tw_low", mca.filters.cfd.set_time_window_low),
            ("tw_high", mca.filters.cfd.set_time_window_high),
        ),
    )

    trapezoid = _mapping(settings, "trapezoid")
    _apply_bools(trapezoid, (("enable", mca.filters.trapezoid.set_enable),))
    _apply_ints(
        trapezoid,
        (
            ("r", mca.filters.trapezoid.set_R),
            ("m", mca.filters.trapezoid.set_M),
            ("t", mca.filters.trapezoid.set_T),
            ("e", mca.filters.trapezoid.set_E),
            ("ft", mca.filters.trapezoid.set_FT),
        ),
    )

    charge = _mapping(settings, "charge_comparison")
    if "enable" in charge:
        mca.filters.charge_comparison.set_enable(bool(charge["enable"]))
    if "time" in charge:
        mca.filters.charge_comparison.set_time(int(charge["time"]))

    psd = _mapping(settings, "psd_zc")
    _apply_bools(psd, (("enable", mca.filters.psd_zc.set_enable),))
    _apply_ints(
        psd,
        (
            ("mode", mca.filters.psd_zc.set_mode),
            ("low", mca.filters.psd_zc.set_time_window_low),
            ("high", mca.filters.psd_zc.set_time_window_high),
        ),
    )

    # Backward compatibility with the original per-channel files.
    temperature = _mapping(settings, "temperature")
    if "coeff" in temperature:
        mca.set_temp_coeff(float(temperature["coeff"]))
    if "offset" in temperature:
        mca.set_temp_offset(int(temperature["offset"]))


def apply_channel_hardware(
    scope: Scope,
    mca: MultiChannelAnalyzer | None,
    hv: HVSupply | None,
    settings: Mapping[str, Any],
) -> None:
    """Apply one channel's hardware values; missing keys are ignored."""
    scope_settings = _mapping(settings, "scope")
    _apply_ints(
        scope_settings,
        (
            ("trigger_level", scope.set_trigger_level),
            ("pretrigger_samples", scope.set_pretrigger_samples),
            ("frame_samples", scope.set_frame_samples),
            ("frame_period_cycles", scope.set_frame_period_cycles),
            ("dac_value", scope.set_dac_value),
        ),
    )
    _apply_bools(scope_settings, (("dma_enabled", scope.set_dma_enable),))
    if "edge_mode" in scope_settings:
        scope.set_trigger_mode(TriggerMode(int(scope_settings["edge_mode"])))

    if mca is not None and isinstance(settings.get("mca"), Mapping):
        mca_settings = _mapping(settings, "mca")
        # The pulse-processor rejects configuration writes while enabled.
        # One locked stop/write/restart covers the whole transaction.
        mca.reconfigure_while_running(lambda: _apply_mca(mca, mca_settings))

    if hv is not None and isinstance(settings.get("psu"), Mapping):
        psu = _mapping(settings, "psu")
        _apply_floats(
            psu,
            (
                ("hv_voltage", hv.set_hv_voltage),
                ("hv_compens_ct", hv.set_hv_compens_ct),
                ("hv_compens_tref", hv.set_hv_compens_tref),
            ),
        )
        _apply_ints(
            psu,
            (
                ("hv_compens_mode", hv.set_hv_compens_mode),
                ("temp_digital_enable", hv.set_temp_digital_enable),
            ),
        )
        if hv.sipm_available():
            _apply_ints(
                psu,
                (
                    ("sipm_enable", hv.set_sipm_enable),
                    ("sipm_compens_mode", hv.set_sipm_compens_mode),
                ),
            )
            _apply_floats(
                psu,
                (
                    ("sipm_voltage", hv.set_sipm_voltage),
                    ("sipm_compens_ct", hv.set_sipm_compens_ct),
                    ("sipm_compens_tref", hv.set_sipm_compens_tref),
                ),
            )


# Compatibility helpers for code that still uses one old-style channel file.
def save_settings(scope: Scope, mca: MultiChannelAnalyzer, hv: HVSupply | None, path: Path) -> None:
    write_configuration(path, collect_channel_hardware(scope, mca, hv))


def load_settings(scope: Scope, mca: MultiChannelAnalyzer, hv: HVSupply | None, path: Path) -> None:
    apply_channel_hardware(scope, mca, hv, read_configuration(path))
    log.info("Settings loaded from %s", path)
