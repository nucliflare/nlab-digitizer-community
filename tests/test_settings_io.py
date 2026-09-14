from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from nlab.hardware.digitizer.scope import TriggerMode
from nlab.utils.settings_io import (
    FORMAT_VERSION,
    apply_channel_hardware,
    channel_application_entry,
    channel_entry,
    collect_channel_hardware,
    read_configuration,
    validate_configuration_version,
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


def test_full_configuration_rejects_incompatible_format_version() -> None:
    try:
        validate_configuration_version({"format_version": 2, "hardware": {}})
    except ValueError as exc:
        assert "expected 3" in str(exc)
    else:
        raise AssertionError("incompatible settings format was accepted")


def test_scope_settings_reject_time_not_aligned_to_hardware_step() -> None:
    scope = Mock()

    with pytest.raises(ValueError, match="8 ns hardware step"):
        apply_channel_hardware(
            scope,
            None,
            None,
            {"scope": {"frame_gap_ns": 10}},
        )

    scope.set_frame_period_cycles.assert_not_called()


def test_nanosecond_settings_survive_yaml_save_and_load(tmp_path: Path) -> None:
    source_scope = Mock()
    source_scope.get_trigger_level.return_value = -12
    source_scope.get_pretrigger_samples.return_value = 24
    source_scope.get_frame_samples.return_value = 1024
    source_scope.get_frame_period_cycles.return_value = 1234
    source_scope.get_trigger_mode.return_value = TriggerMode.RISING_EDGE
    source_scope.get_dac_value.return_value = 7
    source_scope.get_dma_enable.return_value = False

    source_mca = Mock()
    source_mca.filters = SimpleNamespace(
        lp=Mock(),
        crrc2=Mock(),
        cfd=Mock(),
        trapezoid=Mock(),
        charge_comparison=Mock(),
        psd_zc=Mock(),
    )
    source_mca.get_pulse_polarity.return_value = 1
    source_mca.get_trigger_level.return_value = 100
    source_mca.get_baseline_window.return_value = 64
    source_mca.get_pretrigger_samples.return_value = 32
    source_mca.get_frame_samples.return_value = 4096
    source_mca.get_trg_source.return_value = 0
    source_mca.get_ext_trig_enable.return_value = True
    source_mca.get_edge_det_coeff.return_value = 3
    source_mca.get_energy_bin.return_value = 2
    source_mca.get_pileup_window.return_value = 10
    source_mca.get_time_limit.return_value = 60
    source_mca.get_dma_enable.return_value = True
    source_mca.get_mem1_sig_select.return_value = 6
    source_mca.get_mem2_sig_select.return_value = 8
    source_mca.filters.crrc2.get_Cdelay.return_value = 128
    source_mca.filters.crrc2.get_Fdelay.return_value = 16
    source_mca.filters.crrc2.get_pzc_coeff.return_value = 9
    source_mca.filters.cfd.get_enable.return_value = True
    source_mca.filters.cfd.get_factor.return_value = 0.5
    source_mca.filters.cfd.get_delay.return_value = 8
    source_mca.filters.cfd.get_time_window_low.return_value = 16
    source_mca.filters.cfd.get_time_window_high.return_value = 64
    source_mca.filters.trapezoid.get_enable.return_value = True
    source_mca.filters.trapezoid.get_R.return_value = 64
    source_mca.filters.trapezoid.get_M.return_value = 128
    source_mca.filters.trapezoid.get_T.return_value = 96
    source_mca.filters.trapezoid.get_E.return_value = 256
    source_mca.filters.trapezoid.get_FT.return_value = 2
    source_mca.filters.charge_comparison.get_enable.return_value = True
    source_mca.filters.charge_comparison.get_time.return_value = 80
    source_mca.filters.psd_zc.get_enable.return_value = True
    source_mca.filters.psd_zc.get_mode.return_value = 1
    source_mca.filters.psd_zc.get_time_window_low.return_value = 24
    source_mca.filters.psd_zc.get_time_window_high.return_value = 160

    channel = collect_channel_hardware(source_scope, source_mca, None)
    document = {
        "format_version": FORMAT_VERSION,
        "hardware": {"channels": {"0": channel}},
    }
    path = tmp_path / "settings.yaml"
    write_configuration(path, document)
    loaded = read_configuration(path)
    validate_configuration_version(loaded)
    loaded_channel = channel_entry(loaded, 0)
    assert loaded_channel is not None

    saved_scope = loaded_channel["scope"]
    assert saved_scope["pretrigger_ns"] == 48
    assert saved_scope["frame_ns"] == 2048
    assert saved_scope["frame_gap_ns"] == 9872

    saved_mca = loaded_channel["mca"]
    assert saved_mca["signal"]["pretrigger_ns"] == 32
    assert saved_mca["signal"]["frame_ns"] == 4096
    assert saved_mca["crrc2"] == {"cdelay_ns": 128, "fdelay_ns": 16, "pzc": 9}
    assert saved_mca["cfd"]["delay_ns"] == 8
    assert saved_mca["cfd"]["tw_low_ns"] == 16
    assert saved_mca["cfd"]["tw_high_ns"] == 64
    assert saved_mca["trapezoid"]["rise_ns"] == 64
    assert saved_mca["trapezoid"]["flat_top_ns"] == 128
    assert saved_mca["trapezoid"]["pole_zero_ns"] == 96
    assert saved_mca["trapezoid"]["energy_time_ns"] == 256
    assert saved_mca["charge_comparison"]["time_ns"] == 80
    assert saved_mca["psd_zc"]["low_ns"] == 24
    assert saved_mca["psd_zc"]["high_ns"] == 160

    target_scope = Mock()
    target_mca = Mock()
    target_mca.filters = SimpleNamespace(
        lp=Mock(),
        crrc2=Mock(),
        cfd=Mock(),
        trapezoid=Mock(),
        charge_comparison=Mock(),
        psd_zc=Mock(),
    )
    target_mca.reconfigure_while_running.side_effect = lambda write: write()
    apply_channel_hardware(target_scope, target_mca, None, loaded_channel)

    target_scope.set_pretrigger_samples.assert_called_once_with(24)
    target_scope.set_frame_samples.assert_called_once_with(1024)
    target_scope.set_frame_period_cycles.assert_called_once_with(1234)
    target_mca.set_pretrigger_samples.assert_called_once_with(32)
    target_mca.set_frame_samples.assert_called_once_with(4096)
    target_mca.filters.crrc2.set_Cdelay.assert_called_once_with(128)
    target_mca.filters.crrc2.set_Fdelay.assert_called_once_with(16)
    target_mca.filters.cfd.set_delay.assert_called_once_with(8)
    target_mca.filters.cfd.set_time_window_low.assert_called_once_with(16)
    target_mca.filters.cfd.set_time_window_high.assert_called_once_with(64)
    target_mca.filters.trapezoid.set_R.assert_called_once_with(64)
    target_mca.filters.trapezoid.set_M.assert_called_once_with(128)
    target_mca.filters.trapezoid.set_T.assert_called_once_with(96)
    target_mca.filters.trapezoid.set_E.assert_called_once_with(256)
    target_mca.filters.charge_comparison.set_time.assert_called_once_with(80)
    target_mca.filters.psd_zc.set_time_window_low.assert_called_once_with(24)
    target_mca.filters.psd_zc.set_time_window_high.assert_called_once_with(160)


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
            "pretrigger_ns": 48,
            "frame_ns": 2048,
            "frame_gap_ns": 9872,
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
    scope.set_pretrigger_samples.assert_called_once_with(24)
    scope.set_frame_samples.assert_called_once_with(1024)
    scope.set_frame_period_cycles.assert_called_once_with(1234)
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
