# Nuclear Lab Digitizer — Community Edition

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

Nuclear Lab Digitizer is a PySide6 desktop application for acquiring and
analysing data from the Eastern Wall Technologies Digitizer 4.0. It combines a
two-channel oscilloscope, multi-channel analyser (MCA), pulse-shape
discrimination (PSD), high-voltage control, board diagnostics, and optional
Modbus instruments in one dockable interface.

The application is developed around the board's **direct Linux IIO interface**.
It connects to `iiod` with libiio and uses the board's IIO attributes and
buffers for configuration, live data, and DMA capture.

The gRPC backend is **legacy compatibility support for boards running older
firmware**. It remains usable, but new firmware integration and feature work
target IIO.

## Backend status

| Backend | Status | Transport | Intended use |
|---|---|---|---|
| **IIO** | Recommended | libiio over `iiod`, normally port `30431` | Current Digitizer 4.0 firmware; Scope, MCA, PSD/list mode, IDS/HV, and diagnostics |
| **gRPC** | Legacy | gRPC control plus ZMQ DMA | Compatibility with old firmware exposing the Engine/IDS services |

The high-level `Digitizer`, `Scope`, `MultiChannelAnalyzer`, and `HVSupply`
interfaces are shared by both backends. Backend-specific DMA implementations
preserve the lifecycle required by each firmware generation.

## Main features

### Scope

- Live waveform viewer with level, edge, and periodic trigger modes
- A draggable dashed trigger-threshold line on the Scope plot, synchronized
  with the threshold slider and spin box; the hardware value is committed
  when the line is released
- A muted green dashed pretrigger marker, synchronized with the nanosecond
  control. Dragging it previews the horizontal waveform shift in 8 ns steps;
  releasing it updates hardware. The marker and threshold labels match their
  respective line colors.
- In the live raw waveform view, grab the trace to adjust pretrigger time
  horizontally (8 ns steps) and DAC baseline vertically. The trace is a
  frozen preview while dragging; settings are written on release, then the
  next acquired frame shows the actual signal. Auto Setup supplies a measured
  DAC sensitivity; without it, the vertical preview is approximate.
- Scope Auto Setup recognizes narrow, randomly arriving detector pulses and
  performs a bounded provisional-trigger search when its initial forced-frame
  survey contains only baseline.
- Configurable frame length, pretrigger position, frame gap, and analogue offset
- Persistence and raw display modes
- Full-resolution IIO DMA recording with capability-gated queued buffers,
  batched remote reads, bounded stop, tail drain, and recovery
- Independent viewing while a DMA capture is active

### MCA and PSD

- The MCA debug viewer has draggable, dashed trigger-threshold and pretrigger-
  offset markers. Their muted red/green labels match the corresponding controls;
  the offset marker previews a horizontal shift of both debug traces, and
  hardware settings are applied on release.
- Pulse-processor, input-filter, CFD, charge-comparison, and trapezoid controls
- Live 16,384-bin spectra with logarithmic display and ROI statistics
- Modeless multi-spectrum energy calibration under **Tools**, with draggable
  reference lines, linear or quadratic fits, residuals, per-channel calibrated
  top axes, and raw-channel-preserving CSV/YAML metadata
- Modeless MCA peak-analysis workbench under **Tools** with automatic frozen
  snapshots of current MCA spectra, CSV/ASCII SPE/Tukan WDM/CAEN TXT3/ROOT
  loading, one-to-three Gaussian fits, selectable count-aware backgrounds,
  residuals, parameter uncertainties, and variance-preserving spectrum
  arithmetic
- Debug waveform-bank readout
- Synchronized stop/write/restart when changing MCA settings during acquisition
- List-mode IIO DMA using fixed 1,024-record frames
- Live PSD classification from charge-comparison and energy measurements
- Per-measurement MCA output as NDMA, ROOT TTree, HDF5, or online-only DMA
- Incremental bounded-memory file writing with embedded/same-stem YAML settings
- Per-channel live record rate and end-of-run integrity summary; IIO driver
  frame/loss diagnostics are retained for every list-mode output format
- Two-channel IIO coincidence acquisition with automatic shared software
  start, MCA ROI energy gates, AND/veto/OR/XOR logic, and live timing and
  accepted-energy histograms. Qualified CFD timing uses exact 62.5 ps bins
  and reports a guarded constant-background Gaussian-core center, FWHM,
  uncertainty, fitted counts, and reduced chi-square.

### Instrument control

- Per-channel high-voltage output, feedback, and temperature monitoring when the
  IDS IIO devices are present
- Shared trigger control and board-wide diagnostic readings
- Optional RS-485 device discovery through the board's ser2net bridge
- Support for SiPM bias, Geiger-Mueller, and PMT high-voltage Modbus devices
- Safe acquisition and power-supply shutdown when the application closes

### Application workflow

- Dockable and floatable channel panels, including multi-monitor layouts
- Scope, MCA, PSD, Coincidence, PSU, Global, External, and System Log workspaces
- Matching launch and connection-progress splashes, showing startup stages
  while channels and instrument views are initialized
- Save and restore hardware plus application settings in YAML
- Convert Scope and MCA binary captures to HDF5 from the File menu
- Remote board reboot and shutdown controls
- Persistent layout, connection, plotting, and developer settings

## Hardware and firmware requirements

For the recommended IIO path, the board must run a firmware and Linux image
that exposes the relevant devices through `iiod`.

| Function | Expected IIO device |
|---|---|
| Scope control, viewer, and DMA | `vdpp_scope` |
| Scope analogue offset | `vdpp_afe_dac` |
| MCA control and histogram | `vdpp_pulse_processor` |
| MCA input filter | `vdpp_input_filter` |
| MCA list-mode DMA and PSD | `vdpp_lm_frame` |
| Shared MCA trigger | `vdpp_sync_trigger` |
| IDS/HV and diagnostics | `ad5686r`, `mcp3564`/`mcp3564r`, `tmp117`, and optional diagnostic devices |

Scope remains available if MCA or IDS devices are absent. The GUI disables the
workspaces that the detected firmware cannot support instead of failing the
whole connection.

The standard project configuration uses:

- IIO/iiod: TCP port `30431`
- Legacy DPP gRPC: TCP port `50050`
- Legacy IDS gRPC: TCP port `50040`
- Optional ser2net bridges: TCP ports `5001` and `5002`

The board address is deployment-specific. Examples below use
`192.168.10.128`; older images commonly used `192.168.10.20`.

### Check the IIO endpoint

Install the libiio command-line tools and inspect the remote context before
troubleshooting the GUI:

```bash
iio_info -u "ip:192.168.10.128:30431"
```

Confirm that the expected device names are present for every requested
channel. Current MCA and list-mode drivers expose `channel_index` for stable
pairing. Scope and input-filter instances without that attribute use sorted
`iio:deviceN` order as a compatibility fallback, so verify physical A/B mapping
after a firmware or device-tree change.

### List-mode firmware compatibility

Current list-mode support requires `vdpp_lm_frame` IP version 121 with:

- `frame_records=1024`
- `frame_bytes=16384`
- `record_layout=opaque[16]`
- one repeated-u8 scan element carrying each 16-byte record unchanged

Older list-mode firmware may expose five semantic scan channels. That ABI is
not compatible with the Linux 5.15 IIO timestamp handling used by the target:
buffer allocation fails immediately with target-side `EINVAL` (`Open unlocked:
-22`), including with the official `iio_readdev` utility. Upgrade the target
driver/firmware to the opaque-record ABI; do not change the Python frame length
or request a partial scan mask as a workaround.

## Installation from source

### Prerequisites

- Python 3.12 or newer
- [`uv`](https://github.com/astral-sh/uv)
- The platform-appropriate IIO drivers and native runtime from
  [Analog Devices libiio v0.26](https://github.com/analogdevicesinc/libiio/releases/tag/v0.26)
- The `pylibiio` Python package, installed automatically with the project
- Qt runtime libraries; Python Qt bindings are installed through PySide6

On Linux, the system may also need the OpenGL, EGL, XKB, and XCB libraries used
by Qt.

### Set up the project

```bash
git clone https://github.com/nucliflare/nlab-digitizer-community.git
cd nlab-digitizer-community

uv sync --locked --extra dev

python scripts/build_ui.py
python scripts/generate_proto.py
```

The generated Qt modules and protobuf stubs are intentionally not committed.
Generate both after a fresh clone. The protobuf files are still packaged so
the legacy backend can be selected at runtime.

## Running the application

```bash
uv run nlab
```

In the connection dialog:

1. Select **IIO**.
2. Enter the board address, for example `192.168.10.128`.
3. Use port `30431` unless the target's `iiod` configuration differs.
4. Select the number of channels exposed by the board and connect.

The dialog remembers a successful selection. For compatibility with existing
installations, a completely new settings profile may initially show gRPC;
switch it to IIO for current firmware.

A saved configuration can pre-fill the backend, address, port, and channel
count, while explicit address options override values from the file:

```bash
uv run nlab --config experiment.yaml --ip 192.168.10.128 --port 30431
```

You can also run the module directly:

```bash
python -m nlab.main
```

## Programmatic IIO access

The hardware layer can be used without the GUI. IIO channel numbers are
zero-based.

```python
from nlab.hardware.digitizer import Digitizer

digitizer = Digitizer.from_iio(
    channel=0,
    uri="ip:192.168.10.128:30431",
)

try:
    digitizer.scope.set_trigger_level(5000)
    digitizer.scope.set_frame_samples(1024)
    frame = digitizer.scope.acquire_frame()

    if digitizer.mca_available():
        digitizer.mca.set_energy_bin(4)
        digitizer.mca.start()
        spectrum = digitizer.mca.acquire_spectrum()
        digitizer.mca.stop()

    if digitizer.hv is not None:
        digitizer.hv.set_hv_voltage(800.0)
        print(digitizer.hv.get_hv_adc_voltage())
finally:
    digitizer.scope.stop()
    if digitizer.mca_available():
        digitizer.mca.stop()
    digitizer.close()
```

Pass `with_ids=False` to `Digitizer.from_iio()` when only Scope/MCA access is
needed. If required IDS devices are missing, construction logs a warning and
continues with `digitizer.hv` set to `None`.

For a headless periodic Scope DMA recording, use the direct-IIO example:

```bash
uv run python examples/dma_scope_periodic.py capture.bin --duration-s 10
```

It defaults to channel 0 at `192.168.10.128:30431`, Periodic trigger,
16,376 ns frames (8,188 samples), 1,000 ns gap, Raw NDMA output, and DMA on.
Without `--config`, it runs Scope Auto Setup to calibrate the DAC and trigger
threshold before switching to the requested trigger mode. A GUI v3 settings
YAML or a single-channel YAML with `scope.dac_value` and
`scope.trigger_level` skips Auto Setup; YAML frame/gap/mode values apply unless
overridden by `--frame-ns`, `--gap-ns`, or `--trigger-mode`. For example:

```bash
uv run python examples/dma_scope_periodic.py capture.bin --config scope.yaml --frame-ns 16376 --gap-ns 1000 --trigger-mode periodic
```

The example records for 10 seconds by default, refuses to overwrite an
existing file or take over an active Scope, and restores the original Scope
settings after capture. A time-based progress bar shows complete frames
written and the current logical file size (including the NDMA header).
It has no GUI display; `--display-mode raw` and
`--dma on` name the only supported output path. A 10-second maximum-frame
recording may use roughly 330 MB. Inspect the NDMA result with
`notebooks/check_dma.py`.

For an older firmware board, the legacy factory remains available:

```python
digitizer = Digitizer.from_grpc(
    channel=1,
    hostname="192.168.10.20",
    port=50050,
    ids_port=50040,
)
```

Legacy gRPC channel numbers are one-based, matching the old service API.

## Capture files and analysis

Scope recording remains binary with an `NDMA` header. Scope files are not
rotated at an application-defined size; the GUI reports KiB, MiB, or GiB while
the same file grows until recording stops or storage reports an error. MCA
list-mode output is selected under **Settings → DMA Settings...** and creates a
new file for every measurement. Available modes are binary NDMA, a ROOT
`TTree`, appendable HDF5 with SWMR metadata, and online-only DMA with no file.
Binary, ROOT, and HDF5 save list-mode events whether or not Charge Comparison
is enabled. With Charge
Comparison enabled, DMA also feeds the live PSD view when available. Online-only
DMA with Charge Comparison disabled feeds live coincidence analysis when a
coincidence run owns the channels; otherwise the events are discarded. All file formats are written
incrementally through a bounded queue rather than accumulated in RAM.

### MCA energy calibration

Open **Tools → MCA Energy Calibration...**, choose an MCA channel, and copy its
current spectrum into the calibration workspace. Double-click a known peak to
add a draggable reference line, then enter its expected energy in keV. The
channel value can also be edited numerically. At least two enabled references
are required for a linear least-squares fit and three for a quadratic fit; the
table reports each residual and the tool reports RMS and maximum residuals.

Use **Add current as overlay** after changing radioactive sources to combine
lines from several spectra. Existing points remain in place. Overlays are
accepted only when their MCA binning, polarity, low-pass, and trapezoid settings
match. A spectrum copied during acquisition is labelled as a live, non-atomic
snapshot because histogram chunks are transferred sequentially while counts may
still change.

Applying a calibration does not modify FPGA registers, histogram bins, ROI
coordinates, or raw list-mode records. The MCA histogram keeps raw channels on
the bottom axis and displays calibrated keV on the top axis. ROI statistics and
CSV exports include calibrated values, and Save Settings plus subsequent capture
metadata preserve the reference points, coefficients, fit residuals, and energy-
processing fingerprint. If a relevant MCA setting changes, the top axis is
marked stale until the original settings are restored or a new calibration is
applied. Binning is handled specially: changing its power-of-two factor
automatically rescales the channel coordinates, so a calibration made at (for
example) binning 16 remains valid at binning 32 or 8. Other energy-processing
changes still mark the calibration stale.

### MCA peak-analysis workbench

Open **Tools -> MCA Peak Analysis Workbench...** to copy the latest presented
spectrum from every available MCA channel. These are frozen analysis snapshots:
**Refresh selected MCA** or **Refresh all MCA spectra** copies newer data, but
the workbench never starts, stops, clears, or reconfigures acquisition. A copy
taken while acquisition is running is identified as live and non-atomic for the
same reason as the calibration workspace.

The workbench also loads NLab or ordinary delimited CSV spectra,
Maestro/ORTEC-style ASCII `.Spe` files, legacy binary Tukan `.wdm` spectra,
CAEN calibrated ASCII `.txt3` exports, and CAEN ROOT energy histograms. It
accepts counts-only CSV, channel and counts columns, or channel/energy/counts
columns. A ROOT file contributes every `TH1` object from its `Energy` directory
as a separate spectrum, including empty channel histograms; unrelated Time/PSD
objects are omitted. CAEN calibration coordinates and real/live measurement
times are retained when the source represents them. CAEN `_F_` and `_R_`
histogram prefixes are recorded as filtered and raw provenance respectively.
The documented fixed
Tukan metadata includes the analyzer identity, spectrum description,
acquisition start time, and real/live measurement times. The undocumented
variable Tukan calibration/ROI tail and opaque CAEN ROOT calibration objects
are not guessed or applied. Visible spectra can be overlaid when they share
channel or energy coordinates; spectra using another axis are not silently
resampled.

Select one, two, or three Gaussian components and a background of none,
constant, linear, centred exponential, an error-function Compton step, or a
logistic Fermi step. Draggable markers seed and bound the peak centres. Raw
non-negative count spectra default to a Poisson-deviance fit; scaled,
background-subtracted, or otherwise derived spectra use their propagated
variance. Results include channel and calibrated-energy centroids, FWHM,
integrated area, resolution, parameter uncertainty, reduced fit statistic,
AIC/BIC, component curves, residuals, and boundary/covariance warnings.

Spectrum operations create new immutable entries rather than changing their
sources. Available operations are scalar scaling, area/maximum/acquisition-time
normalization, addition, subtraction, live- or elapsed-time-scaled background
subtraction, integer rebinning, and cropping to the fit range. Arithmetic
requires identical coordinate grids, propagates variance, and never performs
implicit interpolation. Spectra export to CSV; fit summaries export to JSON or
CSV without overwriting existing files.

### Live coincidence measurement

The **Coincidence** tab requires two current-IIO MCA list-mode channels and
the shared software-start core. Set the energy ROIs in the two MCA histograms,
then select **Use CH0/CH1 MCA ROI** in the coincidence view. A hidden ROI, or
an unchecked Use ROI option, accepts the channel's full MCA-channel range.
Each accepted-energy plot shows its MCA ROI as a dashed, shaded band; an
unchecked Use ROI option leaves a muted reference outline without applying
the gate. The band follows MCA ROI dragging and disappears when that ROI is hidden.
The ROI and accepted-energy plots use MCA histogram-bin units. Current IIO
list-mode records map their 16-bit selected-energy field to a 14-bit MCA
channel with `energyRaw >> 2`, **not** by shifting again by the MCA binning
selector. The producer may select trapezoidal or integration energy according
to its configuration. This fixed mapping matched paired saved captures with
different binning settings, but is not specified by the opaque-record IIO
driver and should be checked with a labelled-source capture on new firmware.
The accepted-energy plots remain in raw MCA channels.

`CH0 AND CH1` emits every pair inside the inclusive timing gate. An event may
participate in several pairs; the GUI reports pair count separately from the
unique participating-event counts and reports CH0 anchors with multiple
partners. A channel's **NOT** checkbox
with AND makes the other channel an anti-coincidence anchor: `NOT CH0 AND CH1`
accepts a CH1 event only if no ROI-qualified CH0 event falls in its timing
window. NOT does not invert the energy ROI. `OR` shows the union of eligible
singles; `XOR` shows eligible singles with no opposite-channel event in their
window. Both-NOT and NOT with OR/XOR are deliberately unavailable.

Ordinary `CH0 AND CH1` analysis also fills a 512x512 energy-correlation matrix.
Rows are CH1 and columns are CH0; each matrix bin spans 32 raw MCA channels.
The bottom and left axes remain raw channels, while applied MCA calibrations
provide CH0 and CH1 keV labels on the top and right axes. Movable vertical and
horizontal gates project the selected CH1 and CH0 populations respectively;
these display gates do not change acquisition or the prompt-pair count.

The matrix view can show prompt pairs, delayed-random pairs, or prompt minus
scaled random. Random estimation uses two non-overlapping delayed sidebands,
each equal in width to the prompt timing gate, with a configurable gap. Their
combined matrix is multiplied by 0.5 before subtraction. Linear and logarithmic
colour scales are available; corrected matrices use a sign-preserving
`sign(count) * log10(1 + abs(count))` transform in logarithmic mode. The current
prompt, random, and corrected matrices can be exported without overwriting an
existing file to HDF5 or ROOT together with raw-channel edges, gate settings,
timing configuration, calibration metadata, and session diagnostics. Matrix
analysis is deliberately unavailable for veto, OR, and XOR modes because those
modes do not produce two-member coincidence pairs.

The signed raw difference is CH1 event time minus CH0 event time. The measured
CH1-minus-CH0 common-input delay is then **subtracted** as a calibration. Timing
bounds and calibration accept decimal nanoseconds and are converted to exact
integer coordinates; displayed gate bounds are inclusive. AND plots the pair-
delay histogram. Veto, OR, and XOR have no delay for accepted unpaired events,
so the top panel instead plots accepted counts versus elapsed time. The lower
plots show unique participating CH0 and CH1 events. Changing an ROI, rule, or
timing bound resets only live analysis; raw recording continues.

The IIO driver exposes only `opaque[16]`. The GUI therefore uses the one
qualified producer profile, `vdpp-zc-calc-q2.14-v1`, without presenting an
unusable schema selector. The profile name remains in capture/session metadata;
IP121 alone does not identify it. Precision defaults to **Coarse (8 ns)**;
fine mode must be selected explicitly and requires CFD on both MCAs. Fine event
coordinates follow the PetaLinux contract exactly:

```text
eventTime_ns = 8 * timestampTicks
             + 2 * (uint8(zcOffset) + int16(fineRaw) / 16384)
```

`fineRaw` must be in [-16384, 0]. PSD marker `0x08` takes priority over CFD
marker `0x02`; ineligible or out-of-range records are counted and excluded.
Timing arithmetic remains integer in units of 1/8192 ns until bounded display
differences are formed, so uint64 timestamps above 2^53 retain adjacent ticks.
The delay histogram uses exact 62.5 ps bins. With at least 100 pairs and a
significant local peak, the GUI overlays a constant-background Gaussian-core
fit and reports its center, FWHM and formal uncertainty in picoseconds, fitted
signal count, and reduced chi-square. A reduced chi-square above 3 is labelled
non-Gaussian/poor fit; the displayed core width is not automatically a complete
detector CTR characterization.

`zcOffset` is unsigned and can wrap modulo 256. The client never sign-extends
or heuristically unwraps it. Accepted pairs crossing opposite sides of that
boundary are reported; settings that permit producer overflow are not qualified
for fine timing. If the offset cannot be kept in range, the producer must carry
the wrap into the coarse timestamp or widen/normalize the field. The session
manifest records the fixed qualified schema, exact timing equation, channel-delay
sign, inclusive gate, pairing policy, and available firmware/transport identity.

Start automatically sets external start on both MCAs, arms both IIO DMA
readers, then raises the shared software-start level only after both report
ready. Stop, timeout, or failure gates the start off and drains both channels.
The selected MCA DMA output setting applies to both streams: each recorded
run creates `..._ch0` and `..._ch1` raw files plus a common `_session.yaml`
manifest. Binary files keep their per-channel YAML/JSON sidecars; HDF5 and
ROOT files retain embedded settings. Online-only creates no files. The
list-mode firmware publishes fixed 1024-record frames, so low-rate live
updates, especially veto/XOR decisions, can wait for the next complete frame
or stop. Live results are labelled provisional until both streams stop and
drain. Zero timestamps in a final padded frame cannot be distinguished
perfectly from real zero-time events and are reported as ambiguous exclusions;
the current producer has no fully qualified control-record or heartbeat ABI.

IIO MCA NDMA files use format version 2 to distinguish the opaque IIO record
from the same-sized legacy gRPC/ZMQ event record. A JSON sidecar stores IIO
capture geometry, schema, continuity, and driver diagnostics, while a
same-stem YAML file stores the complete digitizer configuration. ROOT and
HDF5 files embed that YAML snapshot and expose `timestamp`, `long_gate`, and
`short_gate` fields for analysis. New HDF5 and ROOT files also preserve the
raw IIO marker, zero-crossing offset, and signed Q2.14 estimate for offline
analysis. Structured output names the fixed client profile and records the
offset as unsigned 2 ns samples and the Q2.14 term as a signed fractional ADC
sample. Existing capture files remain readable.

Every recorded MCA run also writes a same-stem `.run.json` with channel,
format, start/end time, record count, average rate, and continuity status.
The MCA panel shows this outcome after stopping and its tooltip lists IIO
driver counters (`dma_error_count` is cumulative since driver initialization).
Legacy gRPC runs are marked **unverified** because that
transport does not expose equivalent loss counters. IIO record counts include
any zero-padded slots in the final complete frame; they are not an exact
accepted-pulse count. Online-only DMA still creates no file and shows the
summary only in the panel. A missing `.run.json` after a crash means the run
was not finalized; it is not evidence that the capture is complete.

Use **Tools → PSD Event Readback...** to reconstruct the PSD matrix and both
projections from NDMA, CAEN CoMPASS, legacy `caen.py`, HDF5, or ROOT events in
a standalone workbench. Saved-event analysis no longer replaces or pauses a
live PSD measurement.
The import runs in a background worker: binary files are memory-mapped, HDF5
is read in dataset slices, and ROOT uses chunked tree iteration, so event
memory does not grow with file size. CAEN PSD import requires raw Energy and
Energy Short fields; waveform samples are skipped. Headerless 24-byte files
produced by single-channel extraction tools are accepted only when sampled
records have one stable board/channel, monotonic timestamps, coherent gate
values, and zero reserved words; every reserved word and channel is then
validated during bounded-memory iteration.

Use **Developer → Validate Two-Channel Timing...** with two native NLab MCA
list-mode files from a known pulse split between channels (lower-index channel
in A). The background
check scans the full files for backwards timestamps, verifies that their time
ranges overlap, and plots nearest-neighbor delays from bounded samples. A
fixed channel-B offset in 8 ns steps is saved in application settings and
YAML profiles. The dialog checks saved device identity where available, but
neither matching metadata nor a histogram alone proves shared-clock timing;
verify the observed peak against the known split-pulse setup. The plotted
pairs are diagnostic and are **not** coincidence counts.

Use **Tools → Waveform Analysis Workbench...** to browse native NLab scope NDMA
captures or waveform-bearing CAEN CoMPASS binaries without occupying a live
Scope panel. Variable-length CAEN events are indexed in a background worker,
the file remains memory-mapped, and only the selected frame is copied for
display. NLab DMA files use their known 8 ns point period. CoMPASS binaries do
not store the ADC sample period; the tool starts at 2 ns for the common DT5730,
but the operator must set the correct value for the originating digitizer.

The same workbench reconstructs PSD directly from waveforms. It supports a
draggable baseline region with median or mean offset removal, automatic or
explicit pulse polarity, and draggable integration-start, short-end, and
long-end markers. Each event produces `Qshort`, `Qlong`, and
`(Qlong - Qshort) / Qlong`; accepted values feed a configurable energy-versus-
ratio matrix and linked projections. Event stride and maximum-event controls
bound exploratory processing time. Integration uses vectorized, bounded-memory
background batches: fixed-frame NDMA waveforms are exposed as zero-copy views,
while variable-length CoMPASS records are copied into a dynamically sized batch.
When a CoMPASS event also stores gate sums, the
result reports a computed-versus-stored long-gate scale as a format/setting
cross-check. These operations never alter Scope DMA files or live PSD settings.
With **Auto-recalculate while dragging** enabled, moving any baseline or gate
marker produces a debounced preview from every twentieth selected event. Releasing
the marker cancels any stale preview and queues a full calculation using the
configured event stride and maximum-event limit. The initial PSD ratio view spans
-1 to 1, and the workbench sizes its waveform and full PSD panes to the available
desktop.

For Scope NDMA timing checks, edit `FILENAME` in
[`notebooks/check_dma.py`](notebooks/check_dma.py) and run it from the repository
root. Its sample fill factor is the stored waveform duration
(`(frame_samples - 4) * 2 ns`) divided by the mean interval between valid
frame timestamps. This is nominal observed time coverage, not the fraction of
trigger opportunities accepted. NDMA v1 does not record trigger mode; set the
optional `TRIGGER_MODE_LABEL` when comparing files from different modes.

Use **File → Convert Binary to HDF5...** in the GUI for portable analysis
files. Quarto examples are provided in:

- [`docs/analyze_scope.qmd`](docs/analyze_scope.qmd)
- [`docs/analyze_listmode.qmd`](docs/analyze_listmode.qmd)

Hardware walkthrough notebooks are available in
[`notebooks/`](notebooks/). Their saved outputs come from real-board execution
and may reflect the address and firmware installed at the time they were run.

## Architecture

```text
PySide6 GUI
  └─ controllers and background workers
       └─ Digitizer / Scope / MCA / HV interfaces
            ├─ IIO backend (current)
            │    └─ libiio → iiod → FPGA and converter IIO devices
            └─ gRPC backend (legacy)
                 └─ Engine/IDS gRPC + ZMQ DMA services

External workspace
  └─ nlab-modbus → Modbus TCP / ser2net → RS-485 instruments
```

The IIO implementation uses separate libiio contexts for GUI operations,
background polling, and blocking DMA reads. This is intentional: a blocking
refill and a control write must not share one remote context.
The Scope live viewer also owns a separate IIO connection while it is active;
recording progress is displayed at up to ten updates per second so high frame
rates do not flood the GUI event queue.

On current Scope firmware, the driver advertises a qualified four-block DMA
queue. Remote Scope recording uses a bounded 32-frame iiod request to amortize
Ethernet round trips while preserving one exact hardware frame per IIO block.
Firmware without that capability stays on the conservative one-block path.

Important source locations:

| Path | Purpose |
|---|---|
| `src/nlab/hardware/digitizer/backends/iio_backend.py` | Scope and MCA IIO control plus DMA lifecycle |
| `src/nlab/hardware/digitizer/backends/iio_ids_backend.py` | IDS/HV IIO devices and diagnostics |
| `src/nlab/hardware/digitizer/dma.py` | IIO and legacy capture streamers and NDMA formats |
| `src/nlab/controllers/` | GUI coordination and acquisition state |
| `src/nlab/workers/` | Background polling and DMA workers |
| `tests/` | Unit, lifecycle, format, settings, and GUI tests |

## Development

After editing a Qt Designer form or resource file, regenerate the UI modules:

```bash
python scripts/build_ui.py
```

After editing a legacy `.proto` file, regenerate its compatibility stubs:

```bash
python scripts/generate_proto.py
```

Run the test suite with an offscreen Qt platform on headless systems:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest
```

PowerShell equivalent:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
uv run pytest
```

Local static checks are currently advisory while the existing findings are
being baselined:

```bash
uv run ruff check .
uv run mypy src/nlab
```

Tests that require a real digitizer are kept separate from ordinary unit
tests. When running hardware checks, always stop an armed MCA/Scope in a
`finally` block so a failed script does not leave the target enabled.

## Standalone builds

Install the development dependencies and generate the UI/protobuf modules
before packaging.

The reviewed PyInstaller script is used for release builds. It produces a
single-file executable, retains the required SciPy, Matplotlib, and Pillow
stacks, embeds Windows version information, and reports the artifact hash:

```bash
python scripts/build_pyinstaller_reviewed.py --clean
```

The release executable is written to `dist/nlab.exe` on Windows or `dist/nlab`
on Linux.

Nuitka remains available as an optional standalone-directory build:

```bash
python scripts/build_nuitka.py
```

The original
[`scripts/build_pyinstaller.py`](scripts/build_pyinstaller.py) remains as a
minimal fallback.

Tagged releases trigger Windows and Ubuntu builds through the repository's
GitHub and Gitea workflows. The single-file executables are published in
platform-specific archives on the
[Gitea Releases page](https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases).
The Windows Actions artifact contains `nlab.exe` directly, so extracting its
download ZIP once is enough. The Windows Release asset remains a single ZIP
containing `nlab.exe`.

## Known target-side limitations

- Superseded five-channel `vdpp_lm_frame` firmware cannot allocate a list-mode
  IIO buffer on the target Linux 5.15 stack and returns `EINVAL`; use the
  `opaque[16]` driver ABI described above.
- Sustained IIO Scope DMA on one tested channel-0 target can expose a Xilinx
  VDMA `DMA_INT_ERR`/`EOF_EARLY_ERR`. The client rejects partial records,
  releases buffer gates, and requires explicit recovery acknowledgement, but
  it cannot repair a target DMA fault.
- Remote IIO binary reads are informational while acquisition is running.
  For example, a four-part MCA histogram is transferred sequentially rather
  than as one atomic snapshot. Stop acquisition before reading a stable final
  result.

See [`CHANGELOG.md`](CHANGELOG.md) for release history and the current status
of hardware validation.

## Contributing

Bug reports, documentation improvements, hardware traces, and pull requests
are welcome. Please include the backend, firmware/IP version, device discovery
output, host operating system, and whether a result was reproduced with an
official libiio utility when reporting transport or DMA failures.

Submit changes against `main`, add focused tests where practical, and cite the
corresponding authoritative source for driver-derived claims.

## License

MIT — see [`LICENSE`](LICENSE).

Copyright (c) 2026 Eastern Wall Technologies, Sp. z o. o.

The **Nuclear Lab Digitizer** name and EWT logo are trademarks of Eastern Wall
Technologies and are not covered by the MIT license.

## Related projects

- [`nlab-modbus`](https://github.com/nucliflare/nlab-modbus) — Python support for the RS-485 Modbus instruments
- [Eastern Wall Technologies](https://ewt.tech) — hardware manufacturer
