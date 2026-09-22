# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.5.0] — 2026-09-22

### Added

- Standalone waveform-analysis workbench under Tools with memory-mapped NLab
  Scope NDMA and waveform-bearing CAEN browsing, robust per-event baseline
  removal, automatic/manual polarity normalization, draggable nested
  short/long integration gates, vectorized bounded-memory waveform batches,
  configurable PSD matrix geometry and event subsampling, linked projections,
  and stored-versus-recomputed CAEN gate comparison. Fixed-frame NDMA analysis
  uses zero-copy 32,768-event views; CAEN uses dynamically bounded rectangular
  batches for its variable-length records.
- Standalone PSD event-readback workbench under Tools for chunked NDMA, CAEN,
  HDF5, and ROOT analysis. Offline files no longer replace live Scope or PSD
  measurement displays.
- Compact PSD-readback controls and a larger waveform setup plot. Optional
  gate-drag auto-recalculation uses a debounced every-twentieth-event preview,
  cancels stale work, and queues a full-resolution result on marker release.
  The waveform workbench now starts with the full -1 to 1 PSD-ratio range and
  fits its complete PSD matrix and projections within the available desktop.
- Coincidence energy-matrix workspace with a fixed 512x512 prompt matrix,
  equal-width delayed-random sidebands, scaled background subtraction,
  linear and signed-log pyqtgraph rendering, calibrated secondary axes,
  interactive cross-channel gates and projections, and HDF5/ROOT export with
  analysis metadata.
- MCA peak-analysis workbench with automatic frozen live-spectrum snapshots,
  manual refresh, tolerant CSV, ASCII SPE, legacy binary Tukan WDM, CAEN TXT3,
  and CAEN ROOT energy-histogram import, linear/log pyqtgraph overlays,
  asynchronous lmfit models containing one to three Gaussians and selectable
  constant/linear/exponential/Compton/Fermi backgrounds, calibrated results,
  residual and component plots, JSON/CSV reports, and immutable
  variance-propagating spectrum arithmetic and background subtraction.
- PSD import for strictly validated headerless 24-byte CAEN single-channel
  extraction files, with bounded-memory iteration and full reserved-field and
  channel validation.

### Fixed

- MCA peak-fit worker failures are now shown in a modal error message as well
  as in the workbench status labels and application log.
- MCA peak-result column labels now use a compact two-line header and smaller
  header font so all seven labels remain readable in the default results pane.
- All waveform, spectrum, histogram, PSD matrix, residual, and projection plots
  now consistently support Shift+wheel horizontal-only zoom and Ctrl+wheel
  vertical-only zoom while retaining normal two-axis wheel zoom.
- Auto range on live PSD, offline PSD, and coincidence two-dimensional
  histograms now fits the image bounds without pyqtgraph's empty border.

---

## [0.4.0] — 2026-09-22

### Added

- MCA energy-calibration tool with per-channel spectrum snapshots and overlays,
  draggable reference lines synchronized with an editable point table, linear
  and quadratic least-squares fits, residual reporting, calibrated top axes,
  ROI/CSV integration, settings fingerprints, and YAML/capture persistence.
  Applied calibrations automatically rescale their channel coordinates when the
  MCA power-of-two binning setting changes.
- Fixed qualified `vdpp-zc-calc-q2.14-v1` coincidence decoding, exact integer
  timing in 1/8192 ns units, decimal-nanosecond inclusive gates, channel-delay
  calibration, source/transport identity metadata, and diagnostics for invalid
  fine values and accepted pairs crossing the uint8 offset boundary. Fine mode
  uses exact 62.5 ps histogram bins and a guarded background-plus-Gaussian core
  fit with picosecond FWHM reporting and fit-quality diagnostics; it no longer
  enables itself merely because both CFD controls are checked.
- Coincidence Start and Stop buttons now use the same green/red enabled,
  checked, and disabled color scheme as the Scope and MCA controls.
- Two-channel IIO Coincidence tab with shared software-start coordination,
  MCA ROI energy gates, AND/anti-coincidence/OR/XOR logic, configurable 8 ns
  timing window and channel offset, and three live result plots. Recording
  reuses the MCA binary, HDF5, ROOT, and online-only modes with paired raw
  streams and a session manifest. Accepted-energy plots show the corresponding
  MCA ROI bands, muted when the gate is not applied.
- MCA list-mode runs show live average rate and a final health status. Binary,
  HDF5, and ROOT files receive the same `.run.json` summary with IIO continuity
  counters; online-only runs remain file-free.
- Offline two-channel timing validation under Developer for native list-mode files, with
  timestamp-order checks, overlap detection, a bounded-memory delay histogram,
  and an 8 ns-step channel offset saved in application/YAML settings.

### Fixed

- Scope Auto Setup now detects narrow stochastic detector pulses per event,
  reuses pulses seen during baseline calibration, waits on bounded provisional
  triggers when the forced survey is empty, and verifies against a fresh frame.
  This replaces a whole-survey percentile that rejected clear low-duty-cycle
  scintillator pulses while working with repetitive generator signals.
- Scope Auto Setup now waits for its Qt worker thread to exit before releasing
  the thread wrapper, preventing a native PySide teardown crash that could
  terminate the app without a Python error (observed on channel 1).
- Coincidence Stop now waits for both channels' MCA DMA and polling threads to
  exit before releasing their Qt wrappers, preventing the same silent native
  teardown crash when several channel workers finish together.
- IIO MCA list-mode decoding now matches the supplied 16-byte zero-crossing
  firmware output: one-byte marker and offset, signed Q2.14 estimate, two
  energies, and coarse timestamp. Coincidence no longer mistakes the offset
  byte for input-marker flags and discards valid CFD-tagged pulses. New HDF5
  and ROOT recordings preserve the zero-crossing fields. Fine coincidence now
  follows the PetaLinux equation `8*timestamp + 2*(uint8 offset + Q2.14 fine)`
  instead of sign-extending the offset and multiplying its correction by 8 ns.
- Coincidence AND analysis now emits all pairs inside the inclusive calibrated
  gate instead of greedy one-to-one nearest matches. Pair counts and unique
  participating-event counts are reported separately, and live results remain
  explicitly provisional until both raw streams have stopped and drained.
- Coincidence IIO event energies now use the capture-validated fixed two-bit
  conversion to MCA histogram channels instead of an extra shift by the MCA
  binning selector. ROI gates and accepted-energy plots share the corrected scale.
- MCA online-only DMA can start without Charge Comparison. Binary, ROOT, and
  HDF5 recording remains independent of Charge Comparison; live PSD receives
  events only when Charge Comparison is enabled.

---

## [0.3.2] — 2026-09-16

### Added

- Matching launch and connection-progress splash screens; the latter reports
  device and workspace initialization stages until the main window is ready.

### Fixed

- Windows CI artifacts now contain the executable directly instead of a ZIP
  inside the Actions download ZIP; tagged Release assets remain single ZIPs.

---

## [0.3.1] — 2026-09-16

### Added

- Draggable threshold and pretrigger-offset markers in the MCA debug viewer,
  with matching muted control-label colors and release-to-commit behavior.
- A draggable Scope trigger-threshold marker that follows the threshold
  controls and updates the hardware when released.
- A draggable green Scope pretrigger marker that previews horizontal waveform
  shifts and commits on release; matching, lower-contrast marker and label colors.
- Two-axis raw Scope waveform dragging previews pretrigger and DAC baseline
  changes, then commits both on release and resumes measured display.
- A headless direct-IIO Scope DMA example records raw NDMA frames with the
  tested periodic maximum-frame defaults, optional GUI/minimal YAML
  calibration, Auto Setup when no YAML is provided, a bounded duration, and
  live progress showing written frames and file size.

### Changed

- The uncalibrated Scope waveform-drag DAC preview now uses the board-observed
  direction: decreasing DAC moves the signal up. Auto Setup calibration still
  overrides the approximate fallback.
- Remote IIO Scope DMA now uses the queue capability advertised by the current
  PetaLinux driver: four exact-frame kernel blocks and bounded 32-frame iiod
  `READBUF` batches. Older or capability-unknown firmware remains on the safe
  one-block libiio path.
- Scope recording now passes validated raw DMA frames directly to the bounded
  file-writer queue instead of decoding and reconstructing every record, and
  uses the immutable armed geometry instead of making a remote attribute read
  before every frame.
- The IIO Scope live viewer now uses its own worker-only connection, and DMA
  recording progress is coalesced to at most ten GUI updates per second.
- The Scope DMA checker now reports nominal stored-sample time coverage from
  valid frame timestamps and supports a user-supplied trigger-mode label.

### Fixed

- Scope DMA timed stop now waits for its Qt worker thread to exit before
  restoring controls and releasing the thread wrapper. This addresses a
  native PySide teardown crash observed after a high-rate 20-second capture.

---

## [0.3.0] — 2026-09-15

### Added

- **Direct IIO digitizer backend** — selectable alongside the legacy gRPC backend,
  with separate IIO contexts for GUI access, background polling, and DMA acquisition
- **IIO scope support** — trigger and timing configuration, viewer readout, timestamped
  waveforms, periodic frame gaps, and framed DMA capture
- **IIO MCA support** — pulse-processor and input-filter configuration, statistics,
  synchronized live reconfiguration, debug waveform banks, and 16,384-bin histogram
  readout through both monolithic and four-chunk firmware ABIs
- **IIO MCA list-mode client** — fixed v121 frame validation, the current single-channel
  `opaque[16]` transport ABI, application-selected 16-byte event parsing,
  stop/drain/close lifecycle, NDMA v2 recording, HDF5 conversion, and GUI worker
  integration
- **PSD workspace** — live charge-comparison analysis, energy ROI and ratio cut controls,
  classified histograms, counters, and capture integration with MCA list-mode events
- **Scope Auto Setup** — cancellable, background pulse detection that adjusts the
  baseline DAC for dynamic range, selects a noise-aware rising or falling edge trigger,
  verifies the result, and restores the original settings on failure or cancellation
- **Configurable MCA capture output** — one file per measurement in binary NDMA,
  ROOT `TTree`, or appendable HDF5/SWMR format, plus an online-only PSD mode; file
  writers use bounded queues and include the measurement configuration in embedded or
  same-stem YAML metadata
- **Saved-event PSD analysis** — reconstruct PSD matrices and projections from legacy
  and IIO NDMA, CAEN CoMPASS and legacy `caen.py`, HDF5, or ROOT event files using
  background, bounded-memory readers
- **Waveform file browser** — inspect native NLab scope NDMA and waveform-bearing CAEN
  CoMPASS captures in the Scope view, including multi-board/channel selection, frame
  navigation, background indexing, memory-mapped reads, and adjustable CAEN sample time
- **Global workspace** — shared trigger controls, board diagnostics, temperature
  correction, and support for firmware where IDS/HV hardware is absent
- **Application-wide settings workflow** — hardware and UI state can be saved and
  restored together for all connected channels and external devices
- **Remote board power controls** — restricted-SSH reboot and shutdown actions with
  orderly acquisition shutdown before a command is sent
- **Hardware walkthroughs and focused tests** covering IIO scope, MCA configuration,
  DMA formats and lifecycle, synchronized triggering, PSD analysis, diagnostics,
  settings, and shutdown behavior
- **Reviewed PyInstaller build script** with development-only module exclusions,
  Windows version resources, input validation, optional clean builds, and artifact
  size/hash reporting
### Changed

- The desktop layout now fits 720p displays by sizing the main window against
  available screen space, reacting when a window moves between monitors, moving the
  MCA header controls into its scrollable control column only below 1080p, reflowing
  dense controls, and preserving usable plot sizes
- Scope and MCA DMA now start the first blocking reader before enabling acquisition,
  explicitly destroy native IIO buffers, reject partial frames, and require recovery
  acknowledgement after persistent transport or DMA faults
- MCA configuration changes made during a non-DMA measurement now use a synchronized
  stop/write/restart sequence required by the pulse-processor driver
- Scope viewer frame length and periodic timing controls now follow the current driver
  geometry and register behavior
- Scope and MCA timing controls now display physical nanoseconds, and settings format
  version 3 stores timing fields with explicit `_ns` names and rejects incompatible
  full-document schema versions instead of silently ignoring renamed values
- Current IIO MCA pulse-processor and list-mode devices are paired by their stable
  firmware `channel_index`; older firmware retains deterministic device-index fallback
- IIO MCA CR-RC2 C-stage delay and trapezoid pole-zero controls now preserve the public
  physical-time API while converting the two driver attributes that remain raw
- MCA debug-signal choices now follow the capabilities advertised by each backend,
  including the corrected Charge Comparison Window label and ninth IIO selector
- MCA debug-source selections no longer perform an unrelated hardware reset
- DMA conversion distinguishes legacy gRPC list-mode records from the IIO event layout
  through NDMA format version 2
- PySide6 compatibility was widened for supported older Linux deployments
- Release workflows now package the reviewed PyInstaller single-file executable instead
  of the larger Nuitka standalone directory

### Fixed

- Consecutive scope DMA captures could retain cancelled native buffers and fail to
  re-arm cleanly
- Scope DMA conversion interpreted its frame-length header as bytes instead of int16
  samples and could accept trailing partial frames
- MCA live setting changes could fail with `EBUSY` or be mistaken for time-limit
  completion by the polling worker
- Shutdown could terminate acquisition threads before their IIO buffers and hardware
  gates were released
- IIO deployments without an IDS device could fail application construction instead
  of disabling unavailable diagnostics and power-supply controls
- IIO boards without per-channel `cha_temp`/`chb_temp` sensors no longer lose otherwise
  functional HV control; the backend falls back to `HAT_temp` when available and marks
  digital temperature compensation unavailable when no TMP117 exists
- Scope periodic mode omitted the hardware frame-gap setting
- Linux CI runners omitted the native `libdbus-1-3` and `libiio0` runtimes required to
  import PySide6 and the locked `pylibiio` binding during pytest initialization
- Windows self-hosted CI could fail every `tmp_path`-using test when pytest's shared
  user temp directory had incompatible permissions; CI now uses a workspace-local
  temporary root
- Windows release packaging no longer requires PowerShell Core (`pwsh`) and runs with
  the built-in Windows PowerShell available on self-hosted runners

### Known limitations

- Superseded `vdpp_lm_frame` firmware with five semantic scan channels cannot allocate
  a list-mode buffer on the target Linux 5.15 IIO stack and returns `EINVAL`. Compatible
  targets must provide the current single-channel `opaque[16]` ABI; changing the Python
  frame length or requesting a partial scan mask is not a valid workaround.
- Sustained IIO scope DMA on the tested channel 0 can encounter target-side Xilinx VDMA
  `DMA_INT_ERR`/`EOF_EARLY_ERR`. The client now preserves complete frames, releases all
  gates, and supports a clean recovery acknowledgement, but the underlying target
  fault remains.
- Current firmware still does not expose stable channel identifiers for same-name scope
  and input-filter IIO instances; those two device types can therefore still depend on
  probe order. Pulse-processor and list-mode instances now use `channel_index`.

---

## [0.2.0] — 2026-07-09

### Added

- GitHub Actions and Gitea Actions build matrices for Windows and Linux
- Centralized package versioning, tagged-release validation, and automated release
  artifact publication
- Comprehensive installation, hardware connection, development, build, and usage
  documentation

### Changed

- DMA controls initialize consistently with the active acquisition mode
- External-device polling shutdown waits for its worker lifecycle to complete
- Development and release dependencies are locked through `uv`

---

## [0.1.0] — 2026-06-30

Initial open-source release of the Nuclear Lab Digitizer Community Edition.

### Added

- **Main application window** with tabbed layout: Scope, MCA, Power Supply, External
  devices, and System Log views
- **Scope view** — real-time waveform display with trigger/timing/acquisition controls,
  persistence mode, DMA streaming to binary and HDF5 files
- **MCA view** — pulse-height histogram with ROI selection and statistics, list-mode
  DMA streaming, configurable acquisition parameters
- **Power Supply view** — SiPM bias and HV bias control with temperature compensation,
  real-time voltage/current monitoring with live trend plot
- **External Modbus devices** — auto-discovery of SiPM bias board, Geiger-Mueller probe
  and PMT HV supply over ser2net TCP bridge; generic register table view with live
  telemetry polling and per-register trend plotting (`nlab-modbus` integration)
- **Undockable channels** — each digitizer channel's Scope/MCA/PSU view floats
  independently onto a second monitor and re-docks freely
- **HDF5 support** — convert binary DMA recordings to HDF5 via File menu
- **Settings persistence** — save/load hardware register settings to YAML; dock layout
  state persisted across sessions via QSettings
- **ROI statistics** — configurable region-of-interest overlay on MCA histograms
- **Log console** — in-app system log with colour-coded levels (System Log tab)
- **Documentation** — `docs/` folder with Quarto examples for data analysis workflows
- **Gitea Actions CI** — automated Windows and Linux builds via Nuitka standalone,
  plus lint (ruff), type-check (mypy), and headless Qt test (pytest) steps

### Fixed

- DMA thread not closing cleanly on MCA stop
- Time-limited measurement stopping prematurely
- Pulse polarity detection and start/stop button state indicators
- ZMQ socket `LINGER` default causing process to hang after close when DMA was active
- Taskbar icon not updating correctly for the main window on Windows (deferred
  `WM_SETICON` to post-`show()` to survive `QMainWindow`'s native HWND replacement)
- Windows process identity for taskbar grouping (`SetCurrentProcessExplicitAppUserModelID`)

### Build

- Nuitka standalone packaging with workarounds for:
  - `PySide6.QtOpenGL` / `QtOpenGLWidgets` not auto-discovered by static analysis
  - protoc-generated `*_pb2.py` flat-import style (pre-3.20 codegen) requiring
    explicit data-file shipping alongside the compiled binary
  - Linux executable name conflict with the `nlab/` package directory
- Cross-platform output: `nlab.exe` (Windows), `nlab-app` (Linux), both as
  `dist/main.dist/` standalone folder artifacts

---

[Unreleased]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/compare/v0.5.0...main
[0.5.0]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.5.0
[0.4.0]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.4.0
[0.3.2]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.3.2
[0.3.1]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.3.1
[0.3.0]: https://github.com/nucliflare/nlab-digitizer-community/releases/tag/v0.3.0
[0.2.0]: https://github.com/nucliflare/nlab-digitizer-community/releases/tag/v0.2.0
[0.1.0]: https://github.com/nucliflare/nlab-digitizer-community/releases/tag/v0.1.0
