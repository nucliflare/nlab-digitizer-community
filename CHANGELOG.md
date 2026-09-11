# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Direct IIO digitizer backend** — selectable alongside the legacy gRPC backend,
  with separate IIO contexts for GUI access, background polling, and DMA acquisition
- **IIO scope support** — trigger and timing configuration, viewer readout, timestamped
  waveforms, periodic frame gaps, and framed DMA capture
- **IIO MCA support** — pulse-processor and input-filter configuration, statistics,
  synchronized live reconfiguration, debug waveform banks, and 16,384-bin histogram
  readout through both monolithic and four-chunk firmware ABIs
- **IIO MCA list-mode client** — fixed v121 frame validation, all-five-channel scan
  setup, 16-byte event parsing, stop/drain/close lifecycle, NDMA v2 recording, HDF5
  conversion, and GUI worker integration
- **PSD workspace** — live charge-comparison analysis, energy ROI and ratio cut controls,
  classified histograms, counters, and capture integration with MCA list-mode events
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

- Scope and MCA DMA now start the first blocking reader before enabling acquisition,
  explicitly destroy native IIO buffers, reject partial frames, and require recovery
  acknowledgement after persistent transport or DMA faults
- MCA configuration changes made during a non-DMA measurement now use a synchronized
  stop/write/restart sequence required by the pulse-processor driver
- Scope viewer frame length and periodic timing controls now follow the current driver
  geometry and register behavior
- MCA debug-source selections no longer perform an unrelated hardware reset
- DMA conversion distinguishes legacy gRPC list-mode records from the IIO event layout
  through NDMA format version 2
- PySide6 compatibility was widened for supported older Linux deployments

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
- Scope periodic mode omitted the hardware frame-gap setting

### Known limitations

- **IIO MCA list-mode DMA is blocked on the currently tested target firmware.** The
  `vdpp_lm_frame` device reports the documented v121 geometry and all five scan
  channels correctly, but buffer allocation fails immediately with target-side
  `EINVAL` (`Open unlocked: -22`). The same failure occurs with the official
  `iio_readdev` utility using the complete channel mask and multiple requested buffer
  lengths, before acquisition starts. Client geometry should not be changed as a
  workaround; target kernel diagnostics from `lm_buffer_preenable()` or the DMA buffer
  core are required. Consequently, discovery, format, and lifecycle behavior are
  source-derived and unit-tested, but event refill and final tail draining have not
  yet been confirmed live.
- Sustained IIO scope DMA on the tested channel 0 can encounter target-side Xilinx VDMA
  `DMA_INT_ERR`/`EOF_EARLY_ERR`. The client now preserves complete frames, releases all
  gates, and supports a clean recovery acknowledgement, but the underlying target
  fault remains.
- Current firmware does not expose stable labels for same-name scope, pulse-processor,
  input-filter, and list-mode IIO instances; physical channel identity can therefore
  still depend on device probe order.

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

[Unreleased]: https://github.com/nucliflare/nlab-digitizer-community/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/nucliflare/nlab-digitizer-community/releases/tag/v0.2.0
[0.1.0]: https://github.com/nucliflare/nlab-digitizer-community/releases/tag/v0.1.0
