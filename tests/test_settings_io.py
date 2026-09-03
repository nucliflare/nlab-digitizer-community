from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import yaml

from nlab.hardware.digitizer.scope import TriggerMode
from nlab.utils.settings_io import (
    FORMAT_VERSION,
    apply_channel_hardware,
    channel_application_entry,
    channel_entry,
    read_configuration,
    write_configuration,
)


def test_configuration_round_trip_keeps_channels_and_application_separate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.yaml"
    document = {
        "format_version": FORMAT_VERSION,
        "connection": {"ip": "192.0.2.5", "port": 30431},
        "hardware": {"channels": {"0": {"scope": {"trigger_level": -12}}}},
        "application": {
            "channels": {"0": {"scope": {"refresh_rate_hz": 20}}},
            "main_window": {"show_roi": True},
        },
    }

    write_configuration(path, document)
    loaded = read_configuration(path)

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == document
    assert channel_entry(loaded, 0) == {"scope": {"trigger_level": -12}}
    assert channel_application_entry(loaded, 0) == {"scope": {"refresh_rate_hz": 20}}
    assert channel_entry(loaded, 1) is None


def test_read_configuration_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    try:
        read_configuration(path)
    except ValueError as exc:
        assert "top level" in str(exc)
    else:
        raise AssertionError("non-mapping YAML was accepted")


def test_apply_channel_hardware_includes_write_only_and_previously_missing_values() -> None:
    scope = Mock()
    mca = Mock()
    mca.filters = SimpleNamespace(
        lp=Mock(),
        crrc2=Mock(),
        cfd=Mock(),
        trapezoid=Mock(),
        charge_comparison=Mock(),
        psd_zc=Mock(),
    )
    mca.reconfigure_while_running.side_effect = lambda write: write()
    hv = Mock()
    hv.sipm_available.return_value = True
    settings = {
        "scope": {
            "trigger_level": -42,
            "pretrigger_samples": 24,
            "frame_samples": 1024,
            "edge_mode": 1,
            "dac_value": 7,
            "dma_enabled": True,
        },
        "mca": {
            "signal": {"ext_trig_enable": True},
            "acquisition": {"dma_enabled": False},
            "low_pass": {"preset": 2},
        },
        "psu": {
            "hv_voltage": 500.0,
            "hv_compens_tref": 21.5,
            "sipm_enable": 1,
            "sipm_compens_mode": 2,
        },
    }

    apply_channel_hardware(scope, mca, hv, settings)

    scope.set_trigger_level.assert_called_once_with(-42)
    scope.set_trigger_mode.assert_called_once_with(TriggerMode(1))
    scope.set_dma_enable.assert_called_once_with(True)
    mca.reconfigure_while_running.assert_called_once()
    mca.set_ext_trig_enable.assert_called_once_with(True)
    mca.set_dma_enable.assert_called_once_with(False)
    mca.filters.lp.set_preset.assert_called_once_with(2)
    hv.set_hv_voltage.assert_called_once_with(500.0)
    hv.set_hv_compens_tref.assert_called_once_with(21.5)
    hv.set_sipm_enable.assert_called_once_with(1)
    hv.set_sipm_compens_mode.assert_called_once_with(2)
