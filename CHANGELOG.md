# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- A headless direct-IIO Scope DMA example records raw NDMA frames with the
  tested periodic maximum-frame defaults, optional GUI/minimal YAML
  calibration, Auto Setup when no YAML is provided, a bounded duration, and
  live progress showing written frames and file size.

### Changed

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

[Unreleased]: https://github.com/nucliflare/nlab-digitizer-community/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/nucliflare/nlab-digitizer-community/releases/tag/v0.3.0
[0.2.0]: https://github.com/nucliflare/nlab-digitizer-community/releases/tag/v0.2.0
[0.1.0]: https://github.com/nucliflare/nlab-digitizer-community/releases/tag/v0.1.0
