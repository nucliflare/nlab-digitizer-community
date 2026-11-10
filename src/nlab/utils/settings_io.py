"""Versioned, human-readable YAML persistence for GUI and hardware settings.

Version 4 keeps DMA mode selection under ``application`` because the IIO
``dma_enable``/``list_buffer_active`` attributes are driver-owned buffer gates,
not writable hardware configuration. Version 3 documents are migrated while
loading. Hardware-backed values live under ``hardware``; presentation and
worker values live under ``application`` so non-hardware state is explicit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import yaml

from nlab.acquisition_policy import MCA_FIXED_LOW_PASS_PRESET
from nlab.hardware.digitizer.hv import HVSupply
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.hardware.digitizer.mca_parameters import validate_document
from nlab.hardware.digitizer.scope import (
    SCOPE_ADC_SAMPLE_PERIOD_NS,
    SCOPE_DATAPATH_CLOCK_PERIOD_NS,
    Scope,
    TriggerMode,
)

log = logging.getLogger(__name__)
FORMAT_VERSION = 5
_MIGRATABLE_FORMAT_VERSIONS = frozenset({3, 4})


def read_configuration(path: Path) -> dict[str, Any]:
    """Read and minimally validate a configuration document."""
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Settings YAML must contain a mapping at its top level")
    return migrate_configuration(loaded)


def migrate_configuration(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return a current-schema copy of a complete settings document."""
    migrated = deepcopy(dict(settings))
    if "hardware" not in migrated or migrated.get("format_version") != 3:
        return migrated

    hardware_channels = _mapping(_mapping(migrated, "hardware"), "channels")
    application = migrated.setdefault("application", {})
    if not isinstance(application, dict):
        application = {}
        migrated["application"] = application
    application_channels = application.setdefault("channels", {})
    if not isinstance(application_channels, dict):
        application_channels = {}
        application["channels"] = application_channels

    for channel, raw_hardware_channel in hardware_channels.items():
        if not isinstance(raw_hardware_channel, dict):
            continue
        app_channel = application_channels.setdefault(str(channel), {})
        if not isinstance(app_channel, dict):
            app_channel = {}
            application_channels[str(channel)] = app_channel

        scope = raw_hardware_channel.get("scope")
        if isinstance(scope, dict) and "dma_enabled" in scope:
            app_scope = app_channel.setdefault("scope", {})
            if isinstance(app_scope, dict):
                app_scope.setdefault("dma_enabled", bool(scope["dma_enabled"]))
            scope.pop("dma_enabled", None)

        mca = raw_hardware_channel.get("mca")
        if isinstance(mca, dict):
            acquisition = mca.get("acquisition")
            if isinstance(acquisition, dict) and "dma_enabled" in acquisition:
                app_mca = app_channel.setdefault("mca", {})
                if isinstance(app_mca, dict):
                    app_mca.setdefault("dma_enabled", bool(acquisition["dma_enabled"]))
                acquisition.pop("dma_enabled", None)

    migrated["format_version"] = FORMAT_VERSION
    return migrated


def write_configuration(path: Path, settings: Mapping[str, Any]) -> None:
    """Write one complete configuration document."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(configuration_yaml(settings))
    log.info("Settings saved to %s", path)


def configuration_yaml(settings: Mapping[str, Any]) -> str:
    """Serialize a settings snapshot for a file or embedded capture metadata."""
    return cast(str, yaml.safe_dump(dict(settings), sort_keys=False, allow_unicode=True))


def validate_configuration_version(settings: Mapping[str, Any]) -> None:
    """Reject full GUI settings written with an incompatible schema.

    The unversioned, single-channel mapping used by ``save_settings`` remains
    supported. Documents containing the full ``hardware`` section must use
    the current format or a schema with an explicit migration path.
    """
    validate_connection_transport(settings)
    if "hardware" not in settings:
        return
    version = settings.get("format_version")
    if version != FORMAT_VERSION and version not in _MIGRATABLE_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported settings format_version {version!r}; "
            f"expected {FORMAT_VERSION}"
        )


def validate_connection_transport(settings: Mapping[str, Any]) -> None:
    """Accept current IIO settings and reject configurations for removed transports."""
    connection = settings.get("connection")
    if not isinstance(connection, Mapping):
        return
    backend = connection.get("backend")
    if backend is None or backend == "iio":
        return
    if backend == "grpc":
        raise ValueError(
            "This settings file targets the removed gRPC backend. "
            "Use the grpc-final branch with legacy firmware."
        )
    raise ValueError(f"Unsupported connection backend {backend!r}; this application uses IIO")


def connection_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    value = settings.get("connection", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, {})
    return value if isinstance(value, Mapping) else {}


def channel_entry(settings: Mapping[str, Any], channel: int) -> dict[str, Any] | None:
    """Find a hardware channel, accepting YAML string or integer keys."""
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
            "pretrigger_ns": (
                scope.get_pretrigger_samples() * SCOPE_ADC_SAMPLE_PERIOD_NS
            ),
            "frame_ns": scope.get_frame_samples() * SCOPE_ADC_SAMPLE_PERIOD_NS,
            "frame_gap_ns": (
                scope.get_frame_period_cycles() * SCOPE_DATAPATH_CLOCK_PERIOD_NS
            ),
            "edge_mode": scope.get_trigger_mode().value,
        }
    }
    if scope.dac_available():
        settings["scope"]["dac_value"] = scope.get_dac_value()
    profile = scope.acquisition_profile()
    if profile:
        settings["acquisition_profile"] = profile
    if mca is not None:
        mca_settings: dict[str, Any] = {
            "parameter_abi_version": 2,
            "parameter_units": "adc_samples",
            "parameters": mca.parameter_snapshot(),
            "debug": {"mem1_sig_select": mca.get_mem1_sig_select(),
                      "mem2_sig_select": mca.get_mem2_sig_select()},
            "low_pass": {"preset": MCA_FIXED_LOW_PASS_PRESET},
        }
        if mca.filters.crrc2.supports_reciprocal_raw():
            mca_settings["optional_parameters"] = {
                "crrc2_1cdelay_raw": mca.filters.crrc2.get_reciprocal_raw(),
            }
        if mca.dc_blocker_supported():
            mca_settings["input_filter"] = {"dc_blocker_bypass": mca.get_dc_blocker_bypass()}
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


def _apply_time_ns(
    settings: Mapping[str, Any],
    key: str,
    hardware_unit_ns: int,
    setter: Callable[[int], None],
) -> None:
    if key not in settings:
        return
    value_ns = int(settings[key])
    if value_ns % hardware_unit_ns:
        raise ValueError(
            f"{key}: {value_ns} ns is not aligned to the "
            f"{hardware_unit_ns} ns hardware step"
        )
    setter(value_ns // hardware_unit_ns)


def _apply_mca(mca: MultiChannelAnalyzer, settings: Mapping[str, Any]) -> None:
    values = validate_document(settings)
    optional = _mapping(settings, "optional_parameters")
    if optional:
        if set(optional) != {"crrc2_1cdelay_raw"}:
            raise ValueError("Unknown optional MCA parameter")
        value = optional["crrc2_1cdelay_raw"]
        if type(value) is not int or not 0 <= value <= 65535:
            raise ValueError("CR-RC2 1/C must be an unsigned 16-bit integer")
        if not mca.filters.crrc2.supports_reciprocal_raw():
            raise NotImplementedError("This driver does not advertise CR-RC2 1/C")
    mca.configure_parameters(values)
    if optional:
        mca.filters.crrc2.set_reciprocal_raw(optional["crrc2_1cdelay_raw"])
    mca.filters.lp.set_preset(MCA_FIXED_LOW_PASS_PRESET)
    debug = _mapping(settings, "debug")
    for key, setter in (("mem1_sig_select", mca.set_mem1_sig_select),
                        ("mem2_sig_select", mca.set_mem2_sig_select)):
        if key in debug:
            setter(int(debug[key]))
    input_filter = _mapping(settings, "input_filter")
    if "dc_blocker_bypass" in input_filter:
        mca.set_dc_blocker_bypass(bool(input_filter["dc_blocker_bypass"]))


def apply_mca_hardware(
    mca: MultiChannelAnalyzer,
    settings: Mapping[str, Any],
) -> None:
    """Apply only one channel's saved MCA settings.

    This is useful for focused command-line diagnostics which must reproduce
    an MCA configuration without writing unrelated Scope or PSU registers.
    ``settings`` is the complete per-channel hardware mapping returned by
    :func:`channel_entry`; missing MCA data is ignored. The input low-pass
    response always uses the application's fixed 70 MHz preset, including
    when an older document requests a different response.
    """

    if not isinstance(settings.get("mca"), Mapping):
        return
    mca_settings = _mapping(settings, "mca")
    mca.reconfigure_while_running(lambda: _apply_mca(mca, mca_settings))


def apply_psu_hardware(
    hv: HVSupply,
    settings: Mapping[str, Any],
) -> None:
    """Apply only one channel's saved PSU settings; ignore missing PSU data."""

    if not isinstance(settings.get("psu"), Mapping):
        return
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


def apply_channel_hardware(
    scope: Scope,
    mca: MultiChannelAnalyzer | None,
    hv: HVSupply | None,
    settings: Mapping[str, Any],
) -> None:
    """Apply one channel's hardware values; missing keys are ignored."""
    # Reject unidentified MCA semantics before writing any other hardware.
    if mca is not None and isinstance(settings.get("mca"), Mapping):
        parameters = validate_document(_mapping(settings, "mca"))
        if (parameters["psd_zero_crossing_mode"] == 2
                and 2 not in mca.filters.psd_zc.available_modes()):
            raise ValueError("PSD mode 2 is not advertised by this driver")
    if "acquisition_profile" in settings:
        profile = settings["acquisition_profile"]
        if not isinstance(profile, dict):
            raise ValueError("Acquisition profile must be a mapping")
        scope.apply_acquisition_profile(profile)
    scope_settings = _mapping(settings, "scope")
    if "dac_value" in scope_settings and not scope.dac_available():
        raise ValueError("Select verified DAC wiring before applying baseline settings")
    _apply_ints(
        scope_settings,
        (
            ("trigger_level", scope.set_trigger_level),
            ("dac_value", scope.set_dac_value),
        ),
    )
    _apply_time_ns(
        scope_settings,
        "pretrigger_ns",
        SCOPE_ADC_SAMPLE_PERIOD_NS,
        scope.set_pretrigger_samples,
    )
    _apply_time_ns(
        scope_settings,
        "frame_ns",
        SCOPE_ADC_SAMPLE_PERIOD_NS,
        scope.set_frame_samples,
    )
    _apply_time_ns(
        scope_settings,
        "frame_gap_ns",
        SCOPE_DATAPATH_CLOCK_PERIOD_NS,
        scope.set_frame_period_cycles,
    )
    if "edge_mode" in scope_settings:
        scope.set_trigger_mode(TriggerMode(int(scope_settings["edge_mode"])))

    if mca is not None:
        # The pulse-processor rejects configuration writes while enabled.
        # One locked stop/write/restart covers the whole transaction.
        apply_mca_hardware(mca, settings)

    if hv is not None:
        apply_psu_hardware(hv, settings)


# Compatibility helpers for code that still uses one old-style channel file.
def save_settings(scope: Scope, mca: MultiChannelAnalyzer, hv: HVSupply | None, path: Path) -> None:
    write_configuration(path, collect_channel_hardware(scope, mca, hv))


def load_settings(scope: Scope, mca: MultiChannelAnalyzer, hv: HVSupply | None, path: Path) -> None:
    apply_channel_hardware(scope, mca, hv, read_configuration(path))
    log.info("Settings loaded from %s", path)
