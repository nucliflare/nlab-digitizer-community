# PetaLinux IIO application update plan

Prepare nlab-community for the corrected MCA parameter ABI and Scope periodic
sum capture. Implement discovery and configuration ownership first, then the
MCA migration and the new Scope data path. Adding trigger mode 5 to a combo box
alone would route structured sum packets into waveform code.

Date: 2026-11-07. Application baseline: `217d5c4` / v0.8.2. Update branch:
`update/petalinux-iio`. PetaLinux reference: `bb8c57e` on `ewt_iio`.
Application changes are prepared and verified on this branch. PetaLinux remains read-only.

## Governing contracts

Use the current [PetaLinux AGENTS.md](../../../petalinux/AGENTS.md),
[application migration guide](../../../petalinux/docs/application-migration-guide.md),
[attribute reference](../../../petalinux/docs/user-api.md),
[API and HLS contract](../../../petalinux/docs/iio-hls-api-contract.md), and
[MCA ABI2 specification](../../../petalinux/docs/mca-parameter-abi-v2.md).
The [maintained backlog](../../../petalinux/docs/iio-remaining-recommendations-2026-11-06.md)
supersedes historical audit proposals. The older consumer review examined a
different app revision; the gaps below refer to this branch's v0.8.2 baseline.

Retain these decisions:

- Driver/HLS semantics define register meanings. Do not port the excluded
  non-IIO/gRPC implementation or add direct MMIO fallbacks.
- Eight IIO corrections are source-complete; their combined image and both
  physical channels still need qualification. IP124 sum hardware has separate
  deployment gates. PetaLinux builds remain manual.
- A07 processing-completion acknowledgement and A11 automatic Scope/DAC pairing
  remain deferred. Debug data is a best-effort preview. Histogram reads require
  one serialized application operation per device; no new snapshot protocol.
- Keep hardware and decoding in nlab-community. Internal pulse algorithms stay
  in ewt_waveform_workbench and use public base interfaces.
- Preserve the app's fixed 70 MHz MCA filter policy, live debug selector changes
  without accumulation restart, explicit units, and worker cancellation/join.

## Confirmed differences in the current app

| Area | Current v0.8.2 behavior | Required change |
|---|---|---|
| MCA admission | Backend writes attributes directly; MCAController initializes hardware before ABI discovery. | Require `parameter_abi_version=2` and `parameter_units=adc_samples`; validate settings, establish idle ownership, then acknowledge 2 before configuration or arm. Unsupported MCA must not prevent independent Scope use. |
| MCA units and names | Public methods, bounds, timing guides and saved fields retain the old ns interpretation and removed attribute names. | Use canonical ADC-sample values and each field's grid; convert to ns explicitly at presentation boundaries. |
| C delay and CFD | C delay divides ns by 8; its default 64 writes 8. CFD permits fractions above 1. | C delay is exponent 0..6; CFD raw is 0..32767, fraction at most 32767/32768. |
| Shaping readouts | Optional 1/C and 1/R controls expect attributes added in an unpublished driver patch. | The current reference exports neither attribute. Keep 1/C unsupported; distinguish any calculated 1/R from a driver readback. Do not merge the cancelled driver patch. |
| Scope modes and geometry | TriggerMode has values 0..4. Waveform geometry already supports IP122 mean4 and driver-advertised alignment. | Preserve that working path; add a distinct sum format for mode 5 on IP123/IP124, where waveform count and decimation are both zero. |
| Scope recording | NDMA v1 is accepted only for compatible waveforms; mean4 is currently online-only. | Keep that refusal. Add an explicitly versioned sum recording/replay path instead of storing sums as int16 pulses. |
| MCA reconfiguration | A failed write is followed by unconditional restart; gate transfer attempts automatic rollback. | Failed/partial configuration stays stopped and is reported. Do not claim STOP or progress=0 proves downstream completion. |
| Histogram ownership | acquire_spectrum does not hold the configuration lock across all histogram parts. | Share session serialization across complete reads, START/STOP/configuration and any independent readers. Export copies of completed application snapshots. |
| LM teardown | Main-refill fatal errors already terminate capture. Unexpected drain/release errors are logged; poison is not checked. | Extend cleanup/status reporting and refuse rearm after quarantine. Retain the existing transport and successful refill path. |
| Common START | Startup describes start_enable=0 as forcing a gated low output. | It actually selects external bypass. Configure source only with both MCAs stopped; arm both receivers/processors before asserting the selected level. |
| Settings and event interpretation | App settings are already YAML format 4; the driver review discusses older format 3. Q2.14 event decoding already exists. | Version hardware semantics independently of the GUI schema; retain the existing decoder, but bind its selection to producer identity. |

The main edit points are [iio_backend.py](../../src/nlab/hardware/digitizer/backends/iio_backend.py),
[mca.py](../../src/nlab/hardware/digitizer/mca.py),
[scope.py](../../src/nlab/hardware/digitizer/scope.py),
[dma.py](../../src/nlab/hardware/digitizer/dma.py),
[settings_io.py](../../src/nlab/utils/settings_io.py), controllers, workers and
their Designer forms. Reference source is linked below each implementation phase.

## Phase 1 Discovery and ownership

Add immutable capability/identity descriptions to the public backend surfaces:
MCA ABI and units, paired channel_index, Scope IP and supported trigger values,
DMA format/queue limits, filter identity, and available fault/poison status.
Keep blocking readers on separate IIO contexts from control. Share session
coordination across those contexts instead of sharing a native context.

Move discovery ahead of Scope/MCA defaults, filter writes, global sync reset and
automatic temperature correction. Build a complete settings candidate and
validate it before acknowledging ABI2. An existing device-wide acknowledgement
is not permission to interpret old settings as ABI2. Report unsupported MCA
clearly while retaining usable Scope/offline functionality.

Acceptance: missing/unknown ABI or units causes no configuration, START or arm;
valid settings acknowledge only while idle with list DMA closed. Test both
channels, partial discovery, reconnect, replay interlocks and constructor order.
Update the app AGENTS.md rules that currently describe superseded IIO semantics.

## Phase 2 MCA parameters and configuration files

Replace old IIO algorithm attribute names with the exact ABI2 names. Python
must not repeat the kernel's sample-to-register conversions, including CFD's
nonmonotonic phase encoding. Keep sample units canonical in the hardware model;
where the UI displays ns, use 2 ns per ADC sample and the actual field grid.

| Parameter | ABI2 samples or selector | Corresponding UI time |
|---|---|---|
| Pretrigger | 0..2044, step 4 | 0..4088 ns, step 8 |
| Pulse length | 4..4096, step 4 | 8..8192 ns, step 8 |
| CFD delay | 5..132, step 1 | 10..264 ns, step 2 |
| CFD/PSD inclusive low and high | 0..32764, step 4 | 0..65528 ns, step 8 |
| CC integration | 4..32768, step 4 | 8..65536 ns, step 8 |
| CR-RC2 F delay | 4..508, step 4 | 8..1016 ns, step 8 |
| Trapezoid R | 8..2044, step 4 | 16..4088 ns, step 8 |
| Trapezoid M | 0..2044, step 4 | 0..4088 ns, step 8 |
| Trapezoid averaging start | 0..8188, step 4 | 0..16376 ns, step 8 |
| Baseline, C delay, flat-top average | exponent 0..6 | 8, 16, 32, 64, 128, 256, 512 ns |

For example, C delay 64 ns means exponent 3; this is a mapping example, not a
new calibrated preset. CFD's physical maximum becomes 0.999969482421875.
Keep the already corrected temperature coefficient range and base precision.

Keep trapezoid_beta_raw canonical as a raw u32 coefficient. The existing
Pole-zero time control assumes an exponential-to-register conversion inherited
from the earlier interface; the current contract declares raw bits, not that
time law. Expose the raw coefficient or qualify a time adapter against the
current authorized HLS before retaining pole_zero_ns as an editable setting.
Do not carry the old assumption into an ABI2 preset merely because it round-trips.

Mirror active-configuration constraints before START/arm: inclusive window
bounds fit the pulse; CFD delay is shorter; CC fits; trapezoid start plus
`4*(1<<flat_top_exponent)` fits. For nonzero pileup P require
`(frame_samples/4-1)*(P+1)<=255`. Do not invent pretrigger<=frame.
Driver admission remains authoritative.

Keep the existing user preference for visible widget snapping when editing
controls: show the nearest valid value before applying it. Backend calls and
imported settings must reject invalid/fractional/off-grid values without
silently repairing them. Cross-field violations need an actionable explanation.

Version the hardware block with parameter_abi_version=2 and explicit units.
Plan a GUI schema 5 for the changed hardware representation; preserve format
3/4 presentation settings while requiring an explicit review of old MCA values.
An old frame512/pre24 maps to 260/12 samples when preserving the preceding IIO
selectors, not by merely renaming keys. Preserve the original file and conversion
provenance. Never automatically apply an unidentified preset.

Update timing guides, debug cropping (`frame_samples/4` beats), ns axes, gate
transfer, waveform/FOM optimization transfers, calibration fingerprints, reports,
offline devices, replay and examples together. Preserve PSD ROI/cut/fit positions
and flag calibration changes explicitly. Probe reset values are diagnostic
state, not a ready-to-load preset; inactive R=4 must not be silently presented
as an applied valid rise time by a clamped widget.

For 1/R, propose a label **1/R (calculated)** derived only from accepted R
readback: `32768 // (R_samples // 4)`. R=16 samples gives 8192. Initial inactive
R=4 is unconfigured; do not show its mathematical result as the register value.
Keep optional raw readback support distinct when an actual driver advertises it,
and replace the misleading "update driver" tooltip when the current ABI simply
does not export the attribute. Leave 1/C editing unavailable on this baseline.

Reference: [ABI2 tables and migration](../../../petalinux/docs/mca-parameter-abi-v2.md),
[parameter helper](../../../petalinux/docs/examples/mca_parameter_contract.py), and
[pulse driver](../../../petalinux/petalinux_project/project-spec/meta-user/recipes-modules/vdpp-mca/files/vdpp-pulse-processor.c).

## Phase 3 Session lifecycle and diagnostics

Use one owner/lock for each complete histogram assembly and MCA measurement
change. Preserve exact four-part lengths and the single transport terminator;
differentiate stale preview fallback from a current read. Remove the current
claim that enable=0 makes the histogram final. Debug selectors remain live and
do not restart accumulation or require a new generation protocol.

Remove unconditional restart/rollback after configuration failure. A07 stays
deferred: no sleep, progress polling, equal RAM reads or invented ready attribute
can supply its missing completion guarantee. Before enabling automatic successful
stop/write/restart on ABI2, qualify the operating policy on the final image or
retain explicit operator-controlled stop/configure/start. This policy is a
release decision, not an additional kernel implementation task.

For LM, inspect dma_poisoned before arm and after close, keep the first refill
errno plus later cleanup errors, and mark retained capture prefixes incomplete
on unexpected drain/release failure. A poisoned channel requires reboot; no
retry/rearm loop. Keep 1024 opaque records / 16384 bytes, complete-frame checks,
reader-before-enable ordering, cancellation/join and the existing Q2.14 decoder.
Unknown producer permits raw capture only. Metadata must distinguish the selected
energy estimator; the raw energy field is not always trapezoid energy.

Correct Global help/startup logic: start_enable=0 is external bypass, not common
STOP. Configure software source/low with both MCAs stopped, arm both channels
and readers, then assert the common software level. A short pulse is not a
latched run. Preserve the separate per-channel STOP controls and sync IP110
absence/version diagnostics.

Add the missing DC-blocker interface/metadata with direct polarity (1=bypass),
and FIR preset readback. Retain fixed 70 MHz policy. The IIR monitor is before
the output FIR; input correction is `(0.5-temp_coeff)*ADC+temp_offset`.
Scope data follows the ADC branch, so FIR changes must not be presented as
filtering Scope packets. Serialize filter/temperature changes through ownership
and record effective settings without implying an atomic multi-register commit.

Reference: [migration lifecycle and faults](../../../petalinux/docs/application-migration-guide.md),
[LM driver](../../../petalinux/petalinux_project/project-spec/meta-user/recipes-modules/vdpp-mca/files/vdpp-lm-frame.c),
[input filter](../../../petalinux/petalinux_project/project-spec/meta-user/recipes-modules/vdpp-input-filter/files/vdpp-input-filter.c),
and [API/HLS behavior](../../../petalinux/docs/iio-hls-api-contract.md).

## Phase 4 Scope packet formats and periodic sum transport

Discover trigger_mode_available rather than using combo indices as a protocol.
Add mode5 only when supported. Keep raw waveform modes on IP121..124 and the
existing mean4 periodic mode4 on IP122..124. Add a separate Scope sum packet
type/decoder; do not weaken waveform geometry validation to admit zero
decimation as if it described a waveform.

| Format | Packet geometry | Decode rule |
|---|---|---|
| Raw waveform | N samples configured, 2*N DMA bytes | First 8 bytes are timestamp; N-4 signed int16 samples |
| Periodic mean4 | Driver-advertised aligned bytes | N/4-1 signed int16 means at 8 ns; exclude padding |
| IP123 mode5 | 4096 bytes / 2048 int16 IIO scans | Magic 0x31534d5553504456, 32-byte header, up to 254 timestamp/sum pairs |
| IP124 mode5 | 4096 bytes / 2048 int16 IIO scans | Magic 0x32534d5553504456, 32-byte header, one first timestamp, up to 508 uint64 sums |

Preserve unsigned 64-bit packet sums/timestamps. IP124 window i starts at
`timestamp0+i*(N//4+gap)` in 8 ns ticks, modulo 2^64. Only valid_count windows
are measurements. Validate magic against armed IP, N/gap, sequence wrap,
counts, partial STOP flag, sum<=N*65535, required padding and inter-packet timing.
IP124 padding repeats the last valid sum; IP123 uses its separate pair/zero-pad
rules. Real zero sums remain valid. Never infer count/EOF from payload values.

Compute signed mean codes as `S/N-32768` only after exact unsigned accumulation;
each sum includes all N original samples, including the first four. Keep native
integers through file/API boundaries; convert only for presentation/calibration.
Packet-boundary losses must remain visible via timestamps/missed counters.
Neither gap0 nor a quiet transport proves continuous capture.

Arm from dma_frame_bytes/2 scans and the advertised queue limits. Set pretrigger
to zero for sums; lock window/gap/pretrigger/format changes while armed. Stop,
drain, close and rearm to change payload format. Disable viewer reads in sum
mode (EOPNOTSUPP); all ordinary viewer modes still need valid START geometry.

Implement normal sum STOP/drain before buffer destruction: retain every complete
tail packet, inspect pending/fault/missed status and preserve completed prefixes
on error. A clear FPGA pending bit alone does not empty kernel/network queues.
Use the reference terminal-timeout drain semantics with bounded cancellation
and worker joins; do not reuse a timed-out transport or run a healthy drain loop
on a faulted buffer. Expose Scope dma_fault as errno, LM dma_fault as category.

Reference: [format helper](../../../petalinux/petalinux_project/project-spec/meta-user/recipes-modules/vdpp-scope/files/vdpp-scope-format.h),
[Scope stores and guards](../../../petalinux/petalinux_project/project-spec/meta-user/recipes-modules/vdpp-scope/files/vdpp-scope.c),
[packet decoder](../../../petalinux/docs/examples/scope_frame_format.py),
[reference transport/drain](../../../petalinux/docs/examples/scope_backend_iio.py), and
[sum contract](../../../petalinux/docs/scope-periodic-sum-v123.md).

## Phase 5 Scope UI current analysis and recording

Expose **Periodic sums** only in Current (not in Scope waveform triggers),
with window N, gap, effective interval, valid window
counts and missed-window status. Show window mean versus native window time;
do not display sums as a recovered pulse. Disable waveform-only persistence,
single-frame viewer/pulse analysis and raw waveform min/max assumptions in
this mode. A timestamp-only ordinary waveform frame (N=4) remains a separate
valid format, not an empty sum packet.

Extend the existing receiver-owned current accumulator with an explicit sum
input. Weight each interval by measured sample counts, preserve raw sums and
exposure/coverage, and exclude padding/missed windows. Do not manufacture raw
waveforms or noise extrema from aggregate sums. Physical current requires the
existing channel calibration and an identified DC path. Keep the working
waveform/mean4 consumers and bounded UI queues intact.

For recording, propose a versioned HDF5 sum container using the existing h5py
dependency: original 4096-byte packets plus lossless uint64 sums/timestamps,
valid counts, format/IP/ABI, effective settings, channel/clock/producer identity,
missed counts, calibration and completion/cleanup status. Reopen through a sum
reader/replay path. Retain NDMA v1 only where its waveform layout is compatible;
do not label a sum file as a waveform measurement.

Review [scope_controller.py](../../src/nlab/controllers/scope_controller.py),
[current_monitor.py](../../src/nlab/hardware/digitizer/current_monitor.py),
[current monitor controller](../../src/nlab/controllers/current_monitor_controller.py),
[dma.py](../../src/nlab/hardware/digitizer/dma.py), file readers and their forms.
Add capability-specific F1 help, recording metadata and user documentation.

## Phase 6 Verification and release

Add focused tests with independently checked expected values:

- Discovery/init: unsupported ABI, partial capability sets, reconnect, shared
  ownership, no writes before validation, both channels, replay/offline mode.
- MCA: exact field names and boundary grids, C-delay exponent 3 for 64 ns,
  CFD fraction limits, R=16 samples/coefficient 8192, inclusive low=0/high=12
  admitting 16 positions, pileup boundaries, reset R, old-preset refusal, and
  preservation of visible widget snapping and PSD selections.
- Lifecycle: paused/interleaved histogram parts and configuration, failed
  partial writes staying stopped, live debug selectors, refill/drain/release
  faults, poison-before-arm/after-close, first-error preservation and joins.
- Scope: IP121/122 regression coverage; IP123/IP124 packets, sizes, valid_count,
  sequence/timestamp wrap and values above 2^53, true zero/full-scale sums,
  corrupt magic/config/padding, partial STOP packets and missed-window gaps.
- Capture/UI: no sum-to-waveform dispatch, no viewer reads in sum mode, exact
  scans/queue limits, format switches through teardown, bounded queues,
  sample-weighted current summaries, lossless recording/replay and cancelled
  close with retained tail/error metadata.

Regenerate modified Qt forms. Run focused Ruff/mypy checks against existing
debt, relevant tests and then the full application suite. Update README,
CHANGELOG, F1 help, settings/file-format docs, examples and app AGENTS.md to
describe the final behavior. Source-derived host fixtures do not qualify
detector response or the booted image.

Before production release, the user builds/deploys the coherent PetaLinux set
and identifies the running IP/DT/kernel/modules/bitstream. Qualify both channels
with known pulses/current, MCA windows and calibration, common START, Scope
raw/mean4/sum modes as supported, local/iiod transport, STOP/drain/rearm and
controlled fault/quarantine cases. Scope sums require the matching sum IP;
the MCA Linux ABI corrections do not themselves require new MCA HLS.

Implementation now covers ABI2 admission/controls/settings, explicit-stop
configuration, histogram serialization, cleanup/poison handling, Scope sum
transport/analysis, lossless HDF5, and inline Current replay. The Scope controller owns a
separate bounded sample-weighted accumulator and reports ADC codes; physical
current calibration remains in the existing waveform Current workflow until the
raw Scope DC path is qualified. It does not invent waveform extrema from sums.

Live smoke testing on 192.168.10.128 verified native boundary writes and live
debug routing on both MCA channels, and clean IP124 sum STOP/reopen on both
Scope channels (1381 and 1254 windows in the final repeat). All final enable/ownership gates were
zero. Reset-only MCA R=4 and Scope N=0 cannot be restored with valid setters;
the test reports these exceptions. Final cleanup set their nearest legal
values, R=8 and N=4, with ABI acknowledgement restored to zero and all four
ownership/enable gates cleared on both channels.

Validation: the full suite passed 1321 tests. The subsequent shared-routing
restore guard passed its seven-test group; final sum recording/replay and early
cleanup coverage passed 28 tests. New modules pass mypy; changed existing
modules retain 17 pre-existing errors. No new Ruff findings were added. All 17
referenced PetaLinux source hashes remain unchanged.

Remaining image/board qualification includes
physical input mapping, calibrated pulses/current, sustained throughput,
controlled faults, simultaneous channels, and A07 completion policy.
