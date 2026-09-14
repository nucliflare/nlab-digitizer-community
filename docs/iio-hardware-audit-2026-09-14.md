# IIO hardware audit — 2026-09-14

This is a read-only live inventory of the iiod endpoints at
`192.168.10.128:30431`, `192.168.10.135:30431`, and
`10.7.0.121:30431`. No DAC or configuration attributes were written.

## Functional module inventory

| Module or sensor | 192.168.10.128 | 192.168.10.135 | 10.7.0.121 |
|---|---:|---:|---:|
| `vdpp_scope` | 2 | 2 | 2 |
| `vdpp_pulse_processor` | 2 | 2 | 2 |
| `vdpp_input_filter` | 2 | 2 | 2 |
| `vdpp_lm_frame` | 2 | 2 | 2 |
| `vdpp_sync_trigger` | 1 | 1 | 1 |
| `vdpp_afe_dac` | 1 | 1 | 1 |
| `ad5686r` HV DAC | 1 | 1 | 1 |
| `mcp3564r` HV/temperature ADC | 1 | 1 | 1 |
| `ads5407` | 1 | 1 | 1 |
| `ltc6951` clock | 1 | 1 | 1 |
| `xadc` | 2 | 2 | 2 |
| `ina228` hwmon | 1 | 1 | 1 |
| Ethernet-temperature hwmon | 1 | 1 | 1 |
| `tmp117` | 3 | 1 | 1 |

Both pulse processors and both list-mode devices expose stable
`channel_index=0/1` on all three targets. Both scope, MCA, list-mode, and HV
control/readback paths are therefore present on every audited board.

## PSU discovery inputs

| Required signal | 192.168.10.128 | 192.168.10.135 | 10.7.0.121 |
|---|---|---|---|
| AD5686R outputs | `voltage0..3` | `voltage0..3` | `voltage0..3` |
| MCP3564R channel-A feedback | `CHA_HV_get_Vout` | present | present |
| MCP3564R channel-B feedback | `CHB_HV_get_Vout` | present | present |
| MCP3564R internal temperature | `temperature` | present | present |
| TMP117 `HAT_temp` | present | present | present |
| TMP117 `cha_temp` | present | absent | absent |
| TMP117 `chb_temp` | present | absent | absent |

The previous backend required `cha_temp`/`chb_temp`, which incorrectly made
the otherwise complete PSU hardware unavailable on the latter two boards.
Discovery now prefers the channel sensor, falls back to `HAT_temp`, and keeps
HV control available even when no TMP117 is exposed.

## Read-only sensor snapshot

Values are a point-in-time diagnostic snapshot, not calibration results.

| Reading | 192.168.10.128 | 192.168.10.135 | 10.7.0.121 |
|---|---:|---:|---:|
| TMP117 `HAT_temp` | 42.82 °C | 51.50 °C | 48.41 °C |
| TMP117 `cha_temp` | 25.22 °C | unavailable | unavailable |
| TMP117 `chb_temp` | 24.86 °C | unavailable | unavailable |
| ADS5407 temperature raw code | 66 | 71 | 67 |
| LTC6951 `pll_locked` / `ref_ok` | 1 / 1 | 1 / 1 | 1 / 1 |
| AD5686R `voltage0` / `voltage1` raw | 0 / 0 | 0 / 0 | 0 / 0 |

TMP117 physical values use the live IIO `raw * scale / 1000` conversion;
each board reported a scale of `7.8125` millidegrees Celsius per count.
