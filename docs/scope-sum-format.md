# Scope periodic-sum capture format (version 1)

The HDF5 root attribute `format` is `nlab-scope-sums`; `format_version` is 1.
This container stores measured Scope aggregates. It is independent of NDMA
waveform and MCA event formats. IP123 and IP124 wire packets are decoded by
`nlab.hardware.digitizer.scope_sums`.

| Dataset | Type / shape | Meaning |
| --- | --- | --- |
| `packets` | uint8, `(packet_count, 4096)` | Original, complete DMA packets, including header and padding. |
| `valid_counts` | little-endian uint16, `(packet_count,)` | Valid window records in each packet; padding is excluded. |
| `timestamps` | little-endian uint64, `(window_count,)` | Native window timestamps, in 8 ns counter ticks, modulo 2^64. |
| `sums` | little-endian uint64, `(window_count,)` | Sum of all N offset-binary ADC samples in each window. |

For signed int16 ADC samples, unsigned code `u = x + 32768`, so the signed
window mean is `S / N - 32768`. ADC samples are 2 ns apart. Each interval
starts every `(N / 4 + gap_cycles) * 8 ns`. Exposure is the number of measured
samples times 2 ns; missed windows and gap intervals do not contribute.
Session means weight measured samples rather than packets. No waveform,
noise extrema, or absolute current calibration is reconstructed.

Root attributes include `signal_origin=measurement`, `created_utc`,
`timestamp_unit`, `sum_unit`, `metadata_json`, `complete`, `cleanup_error`,
and `final_missed_windows`. The UTC field is host file-creation time; it does
not synchronize the FPGA counter to UTC. `metadata_json` records channel,
IP, N, gap, ADC/counter periods, producer/endpoint identity, calibration status,
and the driver reference used by this client. Physical Scope A/B mapping
still requires independent qualification.

A new file starts with `complete=false`. Only successful STOP tail drain and
buffer release can mark it complete. Unexpected read/drain/release errors
retain the accepted prefix and an error string. Existing files are protected
by exclusive creation. HDF5 is not a crash-transaction log: a process or storage
failure can leave an unreadable/truncated file even with incomplete status.

`read_sum_capture(path)` iterates validated packets. It checks wire geometry,
magic, valid counts, padding, sums, modulo sequence/timestamps, and agreement
between original packets and decoded datasets. Valid incomplete prefixes can
be inspected. File > Open Scope sum capture provides offline analysis/replay.

The 8192-point display history is bounded independently of the recording and
sample-weighted totals. GUI updates do not discard recorded windows. IP123
and IP124 host fixtures test the declared wire assumptions; live smoke tests
exercise IP124 on both channels, without claiming detector calibration.
