# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Compton-edge calibration for organic scintillators** — the MCA Peak Analysis
  Workbench has a **Compton edge** tab. It fits the Klein–Nishina Compton-electron
  spectrum folded with `σ/E = sqrt(a² + b²/E + c²/E²)` and integrated over the bin
  edges, with a Poisson deviance. Several spectra (for example Cs-137 and
  Cs-137 + Co-60 + Na-22) are fitted jointly with a shared gain, offset and
  resolution. Results include each edge with its uncertainty from cov(g, o), the
  resolution in σ and Dietze–Klein FWHM conventions, and model-independent edge
  diagnostics: Compton maximum, half height, Dietze–Klein resolution, edge height
  and an optional Dietze & Klein (1982) Table 3 edge estimate. Warnings cover
  single-scatter bias, degenerate gain/offset and edges outside the window.
  Simulated deposited-energy responses (CSV) can replace the analytic shape per
  line. While a fit runs, a progress bar at the bottom of the workbench and a busy
  cursor show its stage (starting-gain scan, optimizer evaluations, uncertainties). Results export as JSON/CSV, and **Open Compton edges in Energy
  Calibration...** opens MCA Energy Calibration with the edges as fixed keVee
  points for review before you apply them; that window can also open its
  spectra in the new tab. On the recorded EJ-309 showcase sessions it reproduces the
  published joint fit (g = 20.04 ± 0.30 raw/keVee).
- **keVee energy unit** — energy calibrations record `energy_unit` (`keV` or
  `keVee`). Compton-edge calibrations are labelled keVee on MCA, coincidence and
  workbench axes, in ROI statistics, calibration and peak reports, CSV export and
  N42 export. Saved calibrations without the field still load as keV.

## [0.9.5] — 2026-11-24

Help describes the binary file formats of Scope waveform, MCA list-mode and
Current sum recordings, with example Python readers. The waveform workbench and
HDF5 conversion now time Scope `.bin` recordings at their true 2 ns per sample;
earlier versions showed those times four times too long.

### Added

- **Binary file formats in Help** — **Help → User Guide…** has three new
  **Files** topics for writing your own readers: Scope waveform recordings
  (NDMA version 1), MCA list-mode recordings (NDMA version 2 and its companion
  files) and Current periodic-sum recordings (NSUM). Each gives the byte layout,
  units and the records that are not events, with a NumPy reader and an example.
  The list-mode topic includes a 16-byte test record with its decoded values.
  Tests run every example against files written by the application.

### Fixed

- **Scope recordings timed at 2 ns per sample** — the waveform workbench and
  **Convert Binary to HDF5** treated the samples of Scope `.bin` recordings as
  8 ns apart, but they are raw ADC samples 2 ns apart (500 MSPS); 8 ns is the
  timestamp tick and the live viewer's point spacing. The workbench time axis,
  gate positions in ns, cluster-export and report metadata, and the HDF5
  `time_ns` axis were four times too long, and **Transfer candidate gates to
  MCA** proposed MCA pretrigger, window and charge-comparison times up to four
  times too long. Sample values, frame timestamps, PSD ratios, FOM values and
  gate positions in samples were not affected. Times in ns from Scope `.bin`
  files analysed with version 0.9.4 or earlier are four times too long; redo any
  gate transfer to the MCA made from them.

## [0.9.4] — 2026-11-23

General Settings lists its groups as categories and can add the start time to
recording names you choose. The Scope plot follows a restored frame length,
formulas inside Help sentences render, long tooltips wrap, panel undock buttons
are full size, and External modules use nlab-modbus 0.3.0. The PSD readback asks
to recalculate after histogram changes and fits FOM versus energy, and Help cites
the literature behind the application's methods.

### Added

- **PSD readback recalculation** — changing a histogram setting (energy bins, ratio
  bins, energy shift, ratio range) in **Tools → PSD Event Readback** now asks whether
  to rebuild the matrix from the loaded file; before, new settings only applied to
  the next file loaded. It asks once you press Enter or leave the settings, not while
  moving between them. **Yes** keeps the PSD cut and energy ROI and clears the FOM
  fit; after **No**, **Analysis → Recalculate PSD matrix** (Ctrl+R) applies them later.
- **PSD FOM versus energy** — **Analysis → FOM vs energy...** (Ctrl+E) in the PSD
  readback divides the FOM energy ROI into slices (equal counts by default, or equal
  width) and fits each one, and the whole ROI, with the FOM panel's current model,
  windows and seeds. PSD bands move and narrow with energy, so one fit over a wide
  ROI merges different band positions and understates the separation: on a
  liquid-scintillator Cs-137 + PuBe capture the whole ROI gave FOM 0.65 with a
  poor-fit flag, while six slices gave 0.55 to 0.75 with good fits. Plots show FOM
  and both band positions (±FWHM/2) against energy; the table can be copied or saved
  as CSV with its settings, and the report gains a FOM-versus-energy page and
  `fom_vs_energy.csv` when slices have been fitted. **Use best slice as ROI** sets
  the readback's FOM energy ROI to the slice with the highest FOM among fits without
  warnings and refits the FOM panel there, reproducing that slice's FOM.
- **Literature references** — help topics and the matching modules cite the papers
  and books their methods follow, with DOIs checked against Crossref and each claim
  checked against the full text:
  - PSD: charge comparison (Brooks 1959; Moszyński et al. 1992), zero crossing, the
    FOM (Winyard et al. 1971) and its Poisson likelihood fit (Baker and Cousins
    1984), FOM versus energy, waveform reconstruction and the gate search.
  - MCA shaping: digital trapezoidal shaping (Jordanov and Knoll 1994), recursive
    CR-(RC)ⁿ shaping (Nakhostin 2011), and constant-fraction timing (Gedcke and
    McDonald 1968; Fallu-Labruyere et al. 2007).
  - Coincidence: constant-fraction fine timing, and chance coincidences estimated
    from delayed sidebands (Knoll 2010).
  - Spectrum analysis: energy calibration and the quadratic model (Gilmore and Joss
    2024; Kong et al. 2010); the peak-fit step background and Poisson statistic
    (Phillips and Marlow 1976; Baker and Cousins 1984); decay data from ENSDF and
    IRDFF-II.
  - Temperature compensation: SiPM breakdown-voltage and gain drift with bias
    compensation, and the temperature dependence of scintillator light output and
    photodetector gain.
  - Current mode, ROOT and Uproot output, and the ANSI/IEEE N42.42-2020 format.
  - About lists NumPy, SciPy, Matplotlib, lmfit and Uproot.
- **Timestamps on chosen file names** — **General Settings → Acquisition files →
  Add the start time to chosen file names** gives Scope DMA, MCA list-mode and
  Current sum recordings with an operator-chosen name the same `_YYYYmmdd_HHMMSS`
  start time that generated names carry, for example `detector_20261123_101500.bin`.
  A stamped name never replaces an existing file or sidecar; it gains `_001`
  instead, without the Current overwrite question. A series keeps the first run's
  time and adds the run number after it. Off by default; coincidence session files
  keep their own names. The choice is stored with the other General Settings and
  in saved configuration files.

### Fixed

- **Enter in PSD readback fields** — pressing Enter in a histogram setting no longer
  also starts a FOM fit through the dialog's default button.
- **Scope plot matches the frame length after a restore** — restoring the remembered
  device configuration at startup, or **Load Settings**, updated the Scope **Frame**
  control but left the plot on the earlier time range (for example 0–2 µs for a
  248 ns frame). The plot and persistence image now follow the loaded frame. The
  same applies when Current DMA switches the Scope to its maximum frame (IP 122)
  and after Auto Setup, which previously refitted the axis but not the persistence
  image.
- **Inline formulas in Help** — formulas written inside a sentence (`$...$`), as in the
  CR-RC² shaping, trapezoidal 1/R and periodic-sum topics, showed as raw TeX such as
  `$8\times 2^e$`. They now render offline as images on the text baseline, like the
  existing display formulas; the same Markdown still renders on the wiki. Prices such
  as "$5 and $10", escaped `\$` and code stay literal.
- **Undock button in panel title bars** — in the EWT light and dark themes the button
  at the right of a panel's title bar (for example **D1:CH0**) was squeezed into an
  18×10 px sliver with a blurred icon. It is now an 18 px button with a sharp themed
  "pop out" icon and a hover highlight, and every panel's button has the tooltip
  **Undock current tab**. Classic keeps the platform's own button.
- **Long tooltips wrap** — a long plain-text tooltip, such as **Repeats** in the Scope,
  MCA and Current views, was shown on one line across the whole screen. Every
  widget's tooltip is now shown wrapped at about 68 characters, including tooltips
  that change while the application runs. The text is not changed: F1 help still
  shows the original, and rich-text tooltips, which Qt already wraps, are unaffected.

### Changed

- **General Settings categories** — the dialog lists its groups (Appearance, Scope
  display, Acquisition and monitoring, Acquisition files, Configuration persistence)
  on the left and shows the selected one on the right, instead of stacking every
  group in one tall window. The settings themselves are unchanged; **OK** still
  applies the edits on every page and **Cancel** discards them.
- **nlab-modbus 0.3.0** (was 0.2.7) for the External SiPM, Geiger and PMT HV PSU
  modules. Its register maps now follow the firmware tables, each poll reads the
  whole input table in one transaction, and older firmware with shorter tables is
  detected. In the External view:
  - Telemetry gains the registers 0.2.7 never refreshed: SiPM **Output Overload**,
    Geiger **Pulse Count**, and the HV PWM pulse width.
  - Rows for registers a board's firmware does not implement are left out.
  - A missing temperature sensor shows **n/a** instead of −300 °C.
  - The new reset-command register is never offered as a setting, saved with a
    configuration or written on restore. Identity words (version, serial number)
    are not listed as telemetry.
  - Discovery waits 0.10 s instead of 0.05 s for each absent Modbus ID, as the
    library now derives the probe timeout from the 9600-baud gateway link.

## [0.9.3] — 2026-11-20

Measurement series: one Start repeats a timed Scope, MCA or Current measurement
and numbers every recording. MCA series runs also save their spectra as CSV
files, and the Current view gains a time limit. The main window now fits 1080p
screens, including Windows' 125% display scaling.

### Added

- **Measurement series** — **Settings → Enable Series** enables a **Repeats** control in
  the Scope, MCA and Current views. One **Start** then performs that many timed runs:
  each run stops on its **Time [s]** limit, and the next run starts only after the
  previous one has stopped and released the board, including the complete drain and
  close of a DMA recording, so every run repeats the full arm sequence. More than one
  run needs a time limit above 0. **Stop**, an error, a lost connection, session
  replay or a coincidence session ends the series; a failed run is never retried.
  While the option is checked, recordings (Scope DMA, MCA list mode and Current sum
  files) keep one name followed by `_001`, `_002`, …: the chosen file or the generated
  default. With **Record list-mode DMA** checked, every MCA run writes its own
  complete list-mode file (and settings snapshot), closed before the next run arms.
  Existing files are never replaced. The option and each view's run count are saved
  with the other settings.
- **Automatic MCA spectrum files in a series** — each MCA run of a series saves its
  spectrum as a CSV in the **Export CSV** format, with the run's numbered name (next
  to its list-mode file, if recorded). The histogram is read again after the run
  stopped; if that read fails, the last live readback is saved and marked
  `# source=last live readback`.
- **Current time limit** — the Current view has a **Time [s]** field that stops IIR,
  Scope DMA and periodic-sum acquisitions after that many seconds (host clock; for
  Scope DMA, counted from the start of streaming). 0 runs until **Stop**, as before.

### Fixed

- **Window larger than the screen with Windows display scaling** — on a 1920×1080
  screen at 125% scaling (the default on many laptops), the main window extended past
  the right and bottom edges of the screen. The MCA view chose its wide layout from the
  physical 1080 lines, although only 1536×816 scaled pixels are available. It now
  chooses from the screen's available area in scaled pixels, and uses the compact
  layout there.
- **Channel tabs drawn over MCA controls at 1920×1080** — the MCA control column made
  the window taller than a 1080p work area, so a maximized window squeezed the channel
  docks and their tab bar covered the bottom buttons. The MCA control column and the
  Coincidence settings column now scroll, so neither sets the window height, and the
  two-channel window fits a maximized 1080p screen at 100% and 125%.

## [0.9.2] — 2026-11-19

Connection-loss handling and a waveform drawing choice: a lost board connection
stops PSU and Scope polling and offers a one-click reconnect, and waveforms can
be drawn as steps or lines. The remote board reboot and shutdown commands are
removed.

### Added

- **Reconnect prompt after a lost board connection** — when the Scope live view,
  MCA polling or list-mode DMA, PSU monitoring or a Global diagnostics or
  temperature loop loses the board connection, the application shows one
  **Connection Lost** message with **Reconnect** and **Not Now**. Reconnect replaces
  every board connection like Connection → Reconnect Device, without a second
  confirmation, and offers another attempt if it fails. Further reports of the
  same loss do not reopen the message.
- **Waveform drawing: steps or lines** — General Settings → Appearance → Waveforms
  draws the Scope trace, MCA debug traces, Current raw pulses and Waveform Analysis
  waveforms either as held steps (the default, as before) or as straight lines
  joining the samples. The choice applies immediately and does not change the
  samples. pyqtgraph has no smoothed-curve drawing, so none is offered.

### Removed

- **Remote board reboot and shutdown** — the Developer menu no longer offers Reboot
  Remote Board or Shut Down Remote Board. They needed an OpenSSH client, a dedicated
  restricted account and key on the board, and a command dispatcher there; reboot
  or power off the board directly instead.

### Fixed

- **PSU monitoring and Scope live view after a lost board connection** — when the
  network connection to the board was reset (Windows error 10054), PSU monitoring
  kept polling the dead IIO context and logged the same traceback every second.
  The Scope live view did the same at its refresh rate. Both now stop after the
  first failure, like the MCA and global readback workers, report the loss in
  their panel, and leave Start available again.

## [0.9.1] — 2026-11-17

Interface reskin to the Eastern Wall Technologies design system, with EWT light,
EWT dark and Classic themes under General Settings > Appearance.

### Added

- Current periodic sums can record to a compact binary file: **Record** now has a
  **Format** choice, **Binary (.bin)** (default) or **HDF5 (.h5)**. The NSUM
  container (`nlab/hardware/digitizer/scope_sum_file.py`, specified in
  `docs/scope-sum-format.md`) stores the original 4096-byte packets unchanged
  between a JSON header and a trailer written only after the data is flushed; a
  file cut off by a crash stays readable up to its last whole packet and reports
  incomplete. It writes a third of the HDF5 bytes at about 1.5 instead of 14 µs per
  packet in the reader process. Live on .128 (IP124, gap 0, plot shown,
  recording): 192-sample windows (384 ns) covered 99.66–99.73% (HDF5 before this
  release: 89.7–94.6%) and 160 samples (320 ns) 97.6–98.5% (HDF5: 81%). Current
  Capture Analysis, `read_sum_capture()` and session replay read both formats;
  the extension selects the format. Below about 160 samples the board's DMA and
  Ethernet path saturates near 26 MB/s whatever the client does.

### Changed

- Interface reskinned to the Eastern Wall Technologies design system, with a theme
  choice under **General Settings > Appearance** (applies at the next start):
  **EWT light** (default), **EWT dark**, or **Classic**, which keeps the native
  platform style and the original colours. The EWT themes use the bundled Roboto
  typeface (SIL OFL), the brand ink/blue/cyan palette on flat surfaces, underlined
  tabs, rounded inputs, group boxes and indicators, and a splash with the EWT
  colour lockup that follows the theme (white ground in light and Classic, brand
  ink in dark; the on-dark lockup changes only the wordmark to brand white until
  an official reversed file exists). Start buttons are brand blue and turn cyan while
  running; Stop stays red. Every colour now comes from `nlab/views/theme.py`:
  Start/Stop buttons carry an `nlabRole` property instead of inline stylesheets,
  and status text, badges, plot canvases, plot guides and the system log read
  theme roles. Status colours and the dark theme's derived values are checked
  for WCAG AA contrast in `tests/test_theme.py`, because the design system does not
  yet define them. Plot data-series colours are unchanged. `NLAB_THEME` overrides
  the saved theme for one launch.
- Global tab: the control and diagnostics columns keep their natural width and
  pack to the left instead of spreading across wide windows.
- Every tool that opens in its own window (energy and dose calibration, peak
  analysis, PSD readback, Current Capture Analysis, waveform analysis, timing
  validation, MCA ROI history, help) has the same minimize, maximize and close
  buttons as floating channel docks, and no "?" button. Validate Timing now opens
  as a modeless window like the other tools and keeps its chosen files when
  reopened.
- Plots drop the repeated tick values and titles on their top and right axes,
  which only mirrored the bottom and left ones; the frame line and ticks stay.
  Every frame-only axis has the same fixed size, so linked projections (PSD
  matrix with its ratio and energy projections, live and offline; Scope waveform
  with its distribution) keep identical plot edges. Energy axes (MCA histogram,
  coincidence matrix, peak analysis) show keV values only when a calibration
  exists, instead of mirroring channels.
- Scope Persistence uses a theme colour map whose empty level is the plot
  background (white to cyan, blue and ink in light; the reverse glow in dark)
  instead of viridis' solid purple block; Classic keeps viridis. The Raw trace
  is brand blue in light and cyan in dark (was a low-contrast sky blue).
- Selection regions (MCA ROI, coincidence ROIs, peak-fit range) use a translucent
  brand-cyan fill instead of pyqtgraph's default blue. Reference-line table text
  in Peak Analysis keeps its line colour's hue but is adjusted to stay readable.
  The energy-calibration point table no longer truncates its column headers.
- Scope opens in **Raw** display mode by default (was Persistence). A display mode
  saved for a channel, or set in a configuration file, still takes precedence.
- Power Supply tab: the SiPM, HV, compensation, temperature and monitoring groups
  stack in one scrollable column beside the voltage plot, as on the Scope tab,
  instead of a band of stretched columns above it.

- Current tab: shorter labels with the detail in tooltips. **Start**, **Set
  zero** and **Pause readbacks** replace the longer button texts; the window
  preview reads "= 2.048 µs every 102.048 µs"; the Scope DMA and sum readouts show
  rate and coverage, and the status bar shows a one-line summary whose tooltip
  holds the full transport, queue and loss diagnostics. The disabled calibration
  box is hidden in periodic-sum mode, where it never applies.
- The sum recorder batches up to 64 packets (was 16) behind a 2048-packet queue
  (was 256). HDF5 recording at 192-sample windows now covers 99.0–99.75%.

### Fixed

- Waveform Analysis retires each worker thread through the thread bound when it
  was wired (`retire_thread_when_finished` in `nlab/workers/worker_thread.py`)
  instead of looking the worker up with `QObject.sender()`, which would have left
  the thread running if it ever returned None.
- Intermittent CI timeouts in the Waveform Analysis tests (GitHub Linux runners):
  traced to the test session, not the application. The GUI garbage-collection
  policy, once installed by an earlier test, collected many tests' leftover
  widgets in a single 5-30 s pass inside a later test's 5 s wait. The test
  suite now holds that collector's timer during each test and runs its
  threshold check between tests (`tests/conftest.py`); on a timeout it also
  reports each widget-owned QThread's state.
- Current periodic sums lost windows while the Current plot was on screen. Live on
  .128 (IP124, 256-sample windows, gap 0) the shown GUI path covered 87.7–93% of the
  time; it now covers 99.7–99.8%, the same as a bare script. The remaining 0.2% is
  one window per 508-window packet, which the IP skips while sealing a packet
  (PetaLinux `scope-periodic-sum-v123.md`). Every stall had fallen at a 64-frame
  READBUF boundary: there iiod stops draining the four-block kernel queue (~1 ms of
  sums) until the next request arrives, and a reader process starved of CPU arrived
  late. Three changes:
  - The live sum plot repaints only when a new snapshot has arrived (at most ten a
    second) instead of on every display tick; the GUI process fell from about one
    full core to about half of one.
  - Sum snapshots carry the 8192 display points as one NumPy array instead of a tuple
    of tuples: the reader process pickles a snapshot in 0.09 ms instead of 1.2 ms
    while holding its interpreter lock, and its per-packet analysis takes 31 instead
    of 54 µs.
  - Remote Scope DMA (waveforms, Current DMA and sums) keeps one READBUF queued
    behind the batch being read, so iiod starts the next batch at once. With the
    GUI fixes in place this cut boundary stalls from 24–34 to 1–3 per 14 s run.
    STOP drain, tail delivery, carry-over discard and sum completion were rechecked
    live with it.
- The first periodic-sum capture after a connect or after waveform DMA is no longer
  abnormal (rechecked live on 0.9.0 in 15 fresh-connect sessions): frames left in the
  core by an undrained close are now discarded before START instead of failing as
  `Unknown periodic-sum magic/ABI`.

## [0.9.0] — 2026-11-16

Includes the client-side rules from the PetaLinux driver review of 2026-11-14/15
(`d7b19a8`, `cde5fe1`: STOP semantics, stream carry-over X2, `list_deadtime_raw` L2,
CFD window P2, filter coefficient F1). The new kernel patches 0006/0007 are not
required by this client; with them, an undrained close is reported as EIO instead of
silently delivering a misaligned block.

### Added

- Current: **Pause MCA, HV and temperature readbacks during capture** (off by
  default, saved with the configuration). While a live Current acquisition runs it
  pauses MCA histogram/statistics, Power Supply (HV, temperature) and Global
  diagnostics/temperature-correction polling on every channel; they share the
  board's iiod with the capture. Live, MCA polling alone lowered periodic-sum
  coverage (256 samples, gap 0) from ~99% to ~92%, and GUI recordings measured
  76.9% (256 samples) and 96.5% (512 samples). Pausing never changes measurements
  or HV setpoints; temperature correction holds its last value while paused. Pause
  requests are tracked per Current channel, and offline sum replay never pauses.

### Changed

- Polling workers (MCA, Power Supply, Global diagnostics and temperature correction,
  Current IIR monitor, external devices) share one thread helper,
  `nlab/workers/worker_thread.py`, for wiring and for the never-terminate join.
- Current Capture Analysis has a **File** menu (Open capture… Ctrl+O, Export
  configuration YAML…, Close Ctrl+W) instead of an Open button, and its title bar
  names the open capture file.
- The System Log console batches records: the handler buffers formatted lines
  (bounded, oldest dropped with a visible count) and wakes the GUI once per burst; the
  console appends the batch in one edit block every 50 ms at most and scrolls once.
  Its undo stack is disabled and its root-logger handler is removed when it closes.
- The MCA spectrum draws a min/max envelope per pixel column when more than two bins
  share a pixel (`nlab/views/histogram_curve.py`), so peaks and single-bin lines stay
  visible; zoomed in, the exact step histogram is drawn. Offscreen, a 16,384-bin
  refresh in a 900 px plot dropped from 4.4 to 2.2 ms (median, set data and paint).
- The live Coincidence Gaussian timing fit (scipy `curve_fit`) runs on the
  coincidence analysis thread instead of the GUI thread: at most every 0.5 s while
  data streams, and once more on the drained snapshot before the thread exits.
- One application-wide event filter (`nlab/views/app_events.py`) replaces the six
  separate ones (main-window log shortcuts, context help, numeric input, and one per
  PSD view and offline PSD plot). Each handler declares the event types it needs, so
  ordinary paint, timer and mouse-move events no longer make a Python call per
  feature; context help still runs first and sees every event only while help mode
  is on.
- The Current monitor keeps its 10 s IIR history as NumPy columns
  (`nlab/utils/sample_history.py`); the trend plot, rate, latency and zero estimates
  are vectorized instead of rebuilding Python lists of every sample each frame.
- Session replay playback (state, the four playback timers and ~950 lines) moved out
  of the main window into `nlab/controllers/session_replay_coordinator.py`. The main
  window keeps the public replay API and the shared hardware interlock; replay
  behaviour is unchanged and covered by end-to-end tests that record a real session
  and replay it in an offline window.
- All QSettings keys live in `nlab/settings_keys.py` (values unchanged, so existing
  user settings still apply) and the measurement folder is read through one helper.
- Scope offers the **Periodic** (timed) waveform trigger again, with its periodic
  gap control enabled only in that mode, reverting the Scope part of the 0.8.4
  restriction. Structured periodic sums remain a Current-only acquisition.
- MCA list-mode DMA now runs in a dedicated reader process per capture session
  (`nlab/hardware/digitizer/listmode_process.py`). In the GUI process every libiio
  call released the interpreter lock and had to win it back from GUI work, while the
  eight kernel blocks hold only ~15 ms at ~530 kcps, so the FPGA dropped records
  (12% in a live online-only GUI run). Live (v121, ch0) next to a CPU-busy Python
  thread: in-process 87 frames in 3 s, reader process ~1590 with every record
  delivered to the live view. The child also writes the capture file. If the GUI
  process dies mid-capture, the child still stops the MCA, drains, closes and
  writes a valid file (validated live). Start-up adds ~0.45 s before Ready.
  `NLAB_DMA_IN_PROCESS=1` restores the in-process readers for diagnostics.
- Scope waveform DMA and Current periodic sums now also run in reader processes
  (`scope_process.py`, sharing `reader_process.py` with list mode). Live on ch0 next
  to a CPU-busy Python thread: raw waveforms 474 -> 9016 frames/s (same as idle);
  periodic sums at 1000 samples/window 367-419k -> ~4.4k missed windows in 5 s, the
  FPGA's normal one per packet at gap 0. Files (NDMA, sum HDF5) are written by the
  child. The Current accumulator and sum analysis stay in the GUI and receive every
  captured record through a lossless, backpressured queue (with a busy GUI thread the
  accumulator sustained ~5200 frames/s, each analysed); display-only waveforms use a
  bounded queue with counted drops. If the GUI dies mid-capture, the reader clears
  `enable`/`dma_enable` and closes its file (validated live).
- Faster Current analysis. The Scope current accumulator reduces each forwarded batch
  in one numpy pass (`append_frames`), and the display buffer only decodes frames it
  can keep (`append_records`): 64-83k -> 205-589k frames/s idle, 8.5-114k next to a
  busy GUI thread. A randomized test checks equality with the original per-frame
  algorithm. Periodic-sum decode and analysis are vectorised (203 + 167 -> 26 + 52
  us per IP124 packet), and sums are now analysed entirely in the reader process;
  the GUI mirrors snapshots at <= 10 Hz instead of re-analysing every packet, so GUI
  load no longer slows the sum reader. Sum mode now saturates at the board's ~25 MB/s
  (~6100 packets/s; previously ~2470 packets/s, client-limited).
- Measured Current limits (v124, ch0, live). Periodic sums, coverage = captured
  window time / wall time over 10 s: gap 0 gives the densest coverage (the FPGA's
  one skipped window per 508 costs less than any gap); N=192 (384 ns) 99.41%,
  N=256 (512 ns) 99.61%, N=1024 99.80%; with a busy GUI thread N=256 still 99.32%.
  Below ~160 samples the ~3.1 M windows/s transport ceiling caps coverage (N=128:
  77-81%). TIMED waveform mode is target-limited: ~20k frames/s per-frame ceiling,
  ~26 MB/s for long frames, and the Scope DMA queue of at most four one-frame blocks
  turns network/target stalls into skipped triggers at any kHz rate (also with an
  in-process reader). Smallest usable TIMED frame is 8 samples (4 averages to none).
- MCA list mode now arms 64 kernel blocks (1 MiB per channel) instead of the
  PetaLinux reference value of eight. Live on v121 ch0, even a reader with no GUI
  saw 20-50 ms stalls that emptied eight blocks (`queued_blocks` 0) and the IP
  dropped 784-60k records per 10-30 s run; 32 blocks dropped once in five runs, 64
  blocks never (queue minimum 57). Both channels together with 64 blocks and a busy
  GUI-side thread: 15.5 M records in 30 s, no FPGA or live-view drops. Capture
  metadata records the new `kernel_buffers` value.
- Current HDF5 sum recordings default to the measurement folder from General
  Settings. A blank filename generates `chN_current_<timestamp>.h5` on Start;
  relative names use that folder and explicit absolute paths retain their location.
  Missing folders are created in the background; existing files still require
  overwrite confirmation.
- Make the PSD offset-model regression test independent of Qt timer scheduling
  while preserving coverage of continuous-fit model selection.
- Ruff and strict mypy now pass for the whole repository and the GitHub CI lint
  step is blocking. Generated Qt modules are excluded from both tools; the shaping
  accessors' physics names (`get_trapez_R`, `get_crrc2_Cdelay`, ...) are kept as
  public API. PSU control wiring now uses one logged write helper, with tests for
  construction defaults, every IDS control write, and the replay interlock.
- Scope persistence uses a fixed 1024-row raster (about 62 ADC codes per row)
  instead of 20 rows per waveform column, with vectorized rasterization and
  screen-resolution downsampling. A 4096-sample frame drops from an 84 MB image
  and ~100 ms per frame to ~4 MB and ~5 ms; the maximum frame from 335 MB to 8 MB.
- GUI responsiveness: the Scope refresh tick no longer reads the trigger mode over
  the network (a mirror shared by the channel's Scope clients tracks writes); PSD
  bins all pending DMA events with one 2-D histogram per tick and defers image
  rendering while hidden; the MCA debug plot rebuilds its time axis only when the
  frame length changes.
- Fewer IIO round trips: driver-fixed `scale` attributes (MCA dead time,
  temperature correction, IDS ADC/DAC/TMP117, XADC rails) and the Scope viewer
  capacity are read once per connection; MCA list-mode reads the producer status
  once per frame instead of twice.
- Faster start-up: lmfit, scipy, xmlschema and the matplotlib/reportlab report
  stack are imported when a fit, N42 validation or report export first runs, not
  at launch. `import nlab.app` drops from ~2.3 s to ~0.8 s and offline
  time-to-window from ~4.1 s to ~2.4 s; a regression test keeps them out of
  start-up. Application-wide event filters return before any per-widget work for
  events they do not handle.
- Faster connect (two channels, measured live with all IIO writes suppressed:
  ~4.9 s -> ~3.65 s): the two ser2net Modbus bridges are probed concurrently
  (external discovery 1.6 s -> 0.8 s with no modules attached) and each channel
  opens its DPP and IDS backends concurrently.
- MCA polling: without a session recording, the poll rate is capped at the
  15 Hz plot refresh (faster readbacks were coalesced away unseen); sessions keep
  the requested rate. The 8 KiB debug-waveform snapshot is read only while the
  debug plot is visible or a session records. A tick measured ~12 ms of IIO time
  on a LAN (histogram 6 ms, nine status reads 4.7 ms, debug 1.1 ms).
- MCA configuration loads write the validated, acknowledged, idle-checked
  candidate without repeating the client ABI/idle checks before every store; the
  kernel's `pp_field_store()` still guards each write. Measured live: 112 ms ->
  37 ms per channel for 30 parameters on a LAN.
- MCA list-mode recording: the writer queue holds 1024 frames (~16 MiB) instead
  of 32 to absorb storage stalls before the 8-block kernel ring backs up; binary
  captures hash bytes as they are written, so Stop no longer re-reads the file for
  the sidecar SHA-256; HDF5/ROOT flush every 64 Ki events or 1 s (HDF5 chunks stay
  8192 events); MCA DMA progress is coalesced to 10 Hz like Scope. Scope and MCA
  binary files use 1 MiB write buffering, and live Scope frames share the
  immutable DMA record instead of copying it.

### Fixed

- Scope waveform DMA STOP no longer strands accepted frames. STOP only closes
  admission: frames the core already accepted (several on IP124 mode 4) keep arriving.
  Stop used to cancel the reader 50 ms after `enable=0` and close without draining, so
  the tail was lost and could be delivered to the next session (PetaLinux X2). Like the
  reference backend, Stop now keeps the reader until no frame has arrived for 0.25 s
  (at most 10 s), and the tail frames go to the file, Current accumulator and display
  like any other frame. A reader blocked with no data is released by a watchdog only
  after it has waited the full idle interval. Live on .128 (IP124, mode 4, frame 1024):
  9-68 frames per Stop that were previously dropped are now kept, Stop completes in
  ~0.31 s, and a quiet trigger stops cleanly. Frames that complete before a session's
  `enable=1` are discarded as left over from an earlier session.
- MCA list mode discards whole frames that precede the session's synthetic START
  record (zero payload, timestamp 1), which belong to an earlier undrained session.
  A START record later than the first record, or none in the first four frames, marks
  the capture's continuity invalid instead of passing it as verified. Live on .128 the
  START record was record 0 of the first frame in every session.
- MCA list mode reads `list_deadtime_raw` before arming. The counter is not reset by
  arm or START, so an earlier session's value used to fail a loss-free run, and a
  cleared counter is not loss; only a change during the session now counts as
  dropped records. The run summary and sidecar record both values
  (`list_deadtime_before_raw`) and the number of discarded stale frames.
- The MCA CFD window now defaults to 24-88 ns, starting at the default 24 ns
  pretrigger. The window timer starts at the first (pretrigger) beat but fine time at
  the trigger, so a window opening earlier reports crossings in the pretrigger history
  as a false offset 0 (PetaLinux P2: 95% of CFD-valid records with the old 0-64 ns
  default). The Low field turns orange with an explanation when CFD is enabled and it
  starts before the pretrigger; the driver does not enforce this, so it only warns.
- The MCA temperature coefficient written to the input filter is limited to
  -0.5 ... 0.49997 (raw -16384 ... 16383), the range the driver accepts since
  `d7b19a8` (F1): at 0.5 the filter gain is zero and beyond it the signal inverts.
- Stopping and quickly restarting a background poller (Power Supply monitor, MCA
  polling, Global diagnostics or temperature correction, external devices, Current
  IIR monitor) no longer orphans the new one. The previous thread's queued
  "finished" notice used to clear the new worker, so Stop could not stop it and it
  was not joined at shutdown. Each owner now ignores a notice from a thread it has
  already replaced.
- Stopping a background poller could crash the application (segmentation fault,
  intermittent, reproduced on Linux): the exiting thread deleted its worker while
  the GUI thread dropped the last Python reference and PySide deleted the same
  object. The owner's finished handler now runs only after the thread has fully
  exited.
- Pressing Enter in a field no longer clicks a button in the Current Capture Analysis,
  Dose Calibration Logger, Timing Validation and Waveform Analysis windows (Qt's dialog
  default button had made Enter in a spin box press Open, Browse, Start CSV recording or
  Download report). Enter now only commits the value.

- MCA polling, Power Supply and external-device threads are no longer killed with
  `QThread.terminate()` when they do not stop within 3 s at shutdown or reconnect;
  termination could interrupt a network IIO/Modbus call mid-transaction. Like the
  Global and Current workers, they now wait for the in-flight call (bounded by its
  transport timeout) and log that they are waiting.
- Automatically named Scope DMA recordings can no longer overwrite an earlier file.
  Scope used a per-launch counter without an existence check, so a restart within the
  same second reused a name. Scope, MCA and Current sum recordings now share one rule
  (`<prefix>_<YYYYmmdd_HHMMSS>.<ext>`, then `_001`, `_002`, ... while the file or any
  of its sidecars exists); the always-present `_001` suffix of Scope/MCA names is gone.
- Stopping a batched-iiod Scope DMA capture no longer reports a DMA fault
  ("Stopped (recovered from DMA fault)", `Drain: [WinError 10053]`). Cancelling the
  dedicated READBUF socket surfaces a platform-specific error (WSAECONNABORTED or
  WSAESHUTDOWN on Windows, EOF on Linux) that the close-drain treated as a real
  failure, and a GUI Stop could cancel it from the controller thread mid-drain.
  `IiodScopeStream` now reports a deliberate cancel as ECANCELED and the drain
  recognises it; real drain errors (e.g. EIO) still latch. Validated live (v124,
  ch0) for frame-limited and GUI-style stops: complete frames, no fault, gates
  released, and back-to-back captures without recovery acknowledgement.
- Stopping MCA list-mode DMA at high rates no longer fails with
  `Drain: [Errno 9] Bad file descriptor` and a recovery latch. The stop watchdog
  cancelled the reader one second after STOP even when it was busy consuming or
  writing its backlog, so the healthy buffer was cancelled and the tail drain then
  refilled a dead descriptor. The watchdog now only cancels a refill that has itself
  waited a full inactivity window, close skips the tail drain on an already
  cancelled buffer, and the drain classifies EBADF after a cancel as expected.
  Reproduced in the GUI at ~560 kcps (ch0, CC enabled); fix covered by unit tests,
  not yet re-validated live.
- MCA list-mode streaming no longer reads `measurement_in_progress` after every
  16 KiB frame (0.56 ms of a 1.86 ms frame live, ~30% of the per-frame budget). A
  watcher thread polls it every 50 ms on the control context; on a hardware
  time-limit stop it ends the loop and starts the ordinary STOP, so the in-band
  tail is still drained. Validated live (v121, ch0): user stop and 2 s time limit,
  continuity valid, streamed = completed frames.
- MCA DMA blocking stop (application exit or reconnect) no longer calls
  `QThread.terminate()` after 3 s, which could bypass the list-mode drain/close
  and strand the native IIO buffer. Like Scope, it waits 5 s and otherwise leaves
  the reader alive to finish its own cleanup.
- The PSU monitor no longer rewrites the HV DAC every poll when the compensated
  code is unchanged (always the case in compensation mode 0). Explicit setpoint,
  compensation changes and safe shutdown still always write.

## [0.8.5] — 2026-11-13

### Added

- GitHub-only LMDE 7 / Debian 13 x86_64 release build in a Debian trixie
  container with Python 3.12. Tagged releases include
  `nlab-lmde7-x86_64.tar.gz`; the dedicated workflow also supports manual builds.

### Fixed

- Current periodic-sum recordings ask before overwriting an existing HDF5 file
  on Start. Cancel preserves the file and plot; approval applies to that run
  only. File replacement happens in the writer after capture setup succeeds.

### Changed

- Point README downloads and wiki documentation only to the public GitHub
  repository.
- Clarify that periodic-sum recording saves in the background and keeps
  acquisition controls locked until the file is closed. GUI regression tests
  cover slow file creation, writes, and final flush, including overwrites.
- Replace the long README with a concise quick-start guide covering release and
  source launches, native IIO installation, connection, first Scope/MCA runs,
  and links to built-in help, use cases, and the wiki.

## [0.8.4] — 2026-11-10

### Changed

- Remove all periodic triggers and gap controls from the Scope panel. Scope
  offers only level/edge waveform triggers. Current retains driver-supported
  periodic DMA and sums. Legacy periodic settings cannot start a hidden
  periodic Scope measurement or single-frame acquisition.

### Added

- About displays read-only firmware/image metadata when reported by the target,
  board/context attributes, per-channel core versions, MCA ABI and list-mode
  geometry, and the detected IIO device inventory. Information is cached at
  connection time and can be copied for support without polling active hardware.
- Scanned connection entries pair the advertised hostname with its IP, for
  example `nucliflare-xxxxx.local (192.168.10.yyy)`, while connections, saved
  history, and board profiles use the separate endpoint address.
- Optional deployment profiles can record Scope/DAC wiring and list-mode
  producer provenance. Standard connections retain the existing defaults and
  require no new selections or firmware identity; missing identity is recorded
  as unknown. Unrecognized producer versions permit raw capture only.
- Discoverable MCA PSD mode 2 (trapezoid) on drivers advertising it, including
  integration-energy operation with a valid driver-derived trapezoid reciprocal.
- Explicit MCA recovery acknowledgement after failed DMA or an uncertain STOP.
  The passive check never resets hardware or establishes A07 final completion.
- Optional accumulated Scope amplitude distribution beside the waveform,
  with a shared Y axis, filled outline, per-channel bounded ADC-code counts,
  100-code display bins, and View-menu show/pause/clear controls.
- Scope histogram context menu selects all-sample, trigger-directed peak,
  or pretrigger-baselined pulse-integral accumulation, with per-channel
  clear/zoom/log controls and explicit integral units and skipped-frame counts.
- Current capture distribution uses a filled step outline beside the signal
  plot with a shared amplitude Y axis and red chunk-mean reference lines.
- Current capture analysis overlays a trailing moving average on the raw
  trace, with selectable kernel size, native-window computation before
  preview sampling, and resets at acquisition gaps.
- Dedicated offline Current capture analysis with microsecond-sized chunks,
  first/previous/next/last navigation, aggregate statistics, adjacent-window
  noise estimate, signal distribution, and cancellable bounded file reads.
- GUI periodic-sum HDF5 recordings embed the complete pre-acquisition digitizer
  configuration, with readback and YAML export in the offline tool. Older
  captures remain readable without an embedded configuration.

### Fixed

- Retain native buffers, contexts, and reader ownership after failed cancellation
  or join, including Scope drains and MCA first-refill/START failures. Native
  destruction waits for successful reader termination; failed sessions cannot rearm.
- Serialize DMA START/STOP with complete histogram reads while retaining separate
  blocking-stream and control connections.
- Preserve primary errno, additional cleanup failures, poison/queue status,
  effective parameters, and firmware/decoder provenance in MCA run diagnostics,
  including failed captures. Scope waveform close also checks reset quarantine.
- Refuse configuration/rearm after persistent MCA progress following STOP; keep
  failed acquisitions incomplete and preserve the deferred completion-fence limit.
- HDF5/ROOT and binary-to-HDF5 metadata distinguish PSD 8 ns beat corrections
  from CFD 2 ns sample corrections using PSD-first marker precedence. Unknown
  producer conversion retains raw fields without deriving CFD timing.
- MCA omits the automatically derived trapezoid 1/R readout and hides the
  optional CR-RC2 1/C row when the driver has no editable coefficient.
- MCA parameters remain editable during non-DMA measurements, including ABI2,
  using serialized stop/apply/restart with a fresh accumulation. DMA settings
  stay locked through buffer close; failed edits stay stopped and restore a
  retryable UI.
- MCA sample-based timing editors and sliders again display nanoseconds, with
  native-sample conversion on read/write and consistent markers, diagnostic
  waveform windows, charge-comparison guides, and calibration fingerprints.
- Integral histogram orientation follows the selected pulse direction in Scope,
  while preserving positive integral values and normal sample/peak axes.
- Scope distributions use 100 ADC codes per displayed bin at every zoom level.
  Hover and press L to toggle linear/log counts while preserving the
  shared linear amplitude axis and accumulated sample counts.
- Current capture kernel sizes use samples, and the distribution axis uses
  Counts; its filled outline omits individual bin borders.
- Current time axes select s, ms, µs, or ns from the visible zoom span, keeping
  top and bottom ticks synchronized without changing sample timestamps.
- Remote periodic-sum DMA uses capability-gated batched iiod READBUF requests,
  retaining exact 4096-byte packets and STOP tail draining. Only an explicit
  server timeout can finish a batched sum drain; socket/protocol failures stay
  incomplete and latch the session fault. Clean recordings save the actual
  transport, batch size and queue snapshot in optional HDF5 runtime metadata.
- Periodic-sum HDF5 recording uses a bounded asynchronous writer and batched
  dataset appends instead of resizing/writing on the DMA reader thread. Stop
  joins the writer after queued data is written; storage/queue failures stop
  capture explicitly and preserve incomplete status.
- Current sum plots break the trace at missing native windows instead of
  drawing lines through acquisition gaps.

These changes are host-tested/source-derived. The upstream PSD 8-bit offset
overflow remains unresolved in all PSD modes; final-image hardware qualification
and A07 downstream completion remain outstanding.

## [0.8.3] — 2026-11-09

### Added

- Current periodic sums have dedicated window-size and window-gap controls,
  duration/interval preview, saved window size, and Scope frame restoration.
- Capability-gated IP123/IP124 Scope periodic-sum analysis, measured window means,
  sample-weighted summaries, bounded display history, joined cancellation, and
  versioned lossless HDF5 recording/offline replay.
- MCA ABI2 discovery/idle acknowledgement, native ADC-sample controls, C-delay
  exponent, raw beta, calculated 1/R, and explicit DC-blocker bypass.
- An opt-in maintained live smoke test for both IIO channels.

### Changed

- Periodic sums use the existing Current plot and Start/Stop controls, with
  inline HDF5 recording/replay and waveform trigger settings restored after
  stop/drain; Scope no longer offers this non-waveform trigger.
- Settings schema 5 records MCA ABI/units and canonical parameters. Old MCA
  hardware presets require explicit review; presentation settings stay readable.
- Measurement settings require explicit stop/configure/start on ABI2. Debug
  routing remains live; event decoding is bound to producer identity.

### Fixed

- Scope startup/reset leaves periodic-sum mode before restoring nonzero waveform
  pretrigger, stops stale acquisition, and preserves an active DMA owner.
- Shared histogram/configuration serialization, failed-write restart/rollback,
  propagated LM drain/release errors, poison checks, common START bypass semantics,
  native gate transfer, debug timing/cropping, and calibration fingerprints.
- Sum packets retain valid STOP tail records without waveform reinterpretation.


## [0.8.2] — 2026-11-06

### Added

- Raw unsigned 16-bit CR-RC² 1/C control across the IIO backend, MCA API,
  settings and GUI, with slider/spin-box editing, F1 help, and capability
  detection for older drivers. Keep the existing zero initialization and
  expose the value independently of C delay without assuming a Q format.

- Read-only trapezoid **1/R (raw)** readout, with optional driver capability
  detection, refresh after R commits/settings loads, and F1 explanation.

### Fixed

- Match applied MCA temperature coefficients to the signed Q1.15 maximum
  of 32767/32768 instead of +1.0. Validate direct and worker IIO writes
  before encoding, while retaining base calibration precision.

## [0.8.1] — 2026-11-06

### Changed

- Fix the MCA input low-pass response at 70 MHz on every channel during
  initialization and reconnect, and preserve it when loading older settings.
  Remove the filter selector and record the fixed response in saved settings
  and energy-calibration fingerprints.

### Added

- Offline LaTeX-style equations in F1 help using Markdown display-math blocks
  and the existing Matplotlib Mathtext renderer, with theme-aware text colors,
  high-resolution images, cached rendering, and readable TeX on rendering errors.

- Optional y0 count offsets for joint and separate PSD Gaussian fits, enabled
  by default and fixed to zero when unchecked. Settings, reports, and waveform
  gate optimization preserve the choice and account for the varied parameters.
- F1 FOM help now shows Gaussian, joint/separate model, FWHM, and FOM formulas,
  with dedicated offset help and detailed convergence-warning implications.

- Optional continuous PSD FOM fitting of the latest projection and fit guides,
  coalesced at up to two fits per second with a bounded evaluation budget.
- Separate Gaussian FOM fitting for well-separated populations, with two
  independent colored fit ranges and local constant backgrounds. The gap is
  excluded; mode and fit ranges are preserved in live PSD settings and reports.

- Native minimize/maximize controls on floating channel and device panels,
  preserving their position, size, visibility, and ability to dock again after
  repeated undocking or saved-layout restoration. Verified with offscreen Qt
  tests and a hidden native-Windows caption check.
- The native **X** on floating panels docks them back into their original tab
  group and selects the panel, including from maximized windows. This uses the
  existing caption controls without adding a button row above the content;
  application shutdown keeps its normal close and cleanup behavior.

### Fixed

- Floating title-bar double-clicks now leave maximize/restore to the native
  window manager instead of also redocking the panel, which could make the
  floating window disappear.

## [0.8.0] — 2026-11-06

### Changed

- Allow Windows/Ubuntu build chores on `chore/build-*` branches on GitHub and
  Gitea without changing the application version. Only `v*` tag pushes verify
  release versions and publish releases.

### Fixed

- Preserve PSD cut, energy ROI, fit ROI, and centroid-guide positions through
  analysis settings edits, replay rebinning, and readback refreshes. Clamp to new
  limits instead of restoring default guides, keep unspecified selections during
  partial configuration updates, keep matrix zoom when rebinning, and skip
  unchanged analysis commits. Explicit saved selections still restore after
  applying their bounds. Covered by
  offscreen Qt tests with generated events; no live hardware validation.
- Apply MCA Debug 1/Debug 2 source changes directly, without stopping or
  resetting histogram/viewer acquisition or list-mode DMA. Preserve spectrum,
  elapsed time, and workers; keep viewer selectors available during DMA while
  locking measurement configuration and all replay/disconnected writes. Failed
  changes report an error without restarting. Correct outdated busy-check
  documentation to match the PetaLinux driver. Verified with simulated I/O and
  offscreen Qt tests; no live hardware validation.
- Keep the selected digitizer/channel when switching between Scope, Current,
  MCA, PSD, and PSU workspaces. Match channel identity rather than tab position,
  retain it through non-channel workspaces and missing capabilities, and cover
  navigation with offscreen Qt tests.
- Schedule cyclic garbage collection on the GUI thread, so allocations during
  background FOM fits cannot destroy Qt plot/widget cycles from a worker and
  deadlock GUI cleanup. Collection remains periodic and restores the previous
  automatic-GC state when the application quits.
- Retire waveform indexing, averaging, PSD, and optimization workers from the
  GUI thread and join their deferred deletion before continuing GUI operations,
  avoiding a Windows Qt/Python lock deadlock during plot-label updates. PSD
  outcomes display after cleanup; cancelled or closed analyses discard results.
  CI test steps now log individual tests, dump stacks after a stalled test,
  and have a ten-minute deadline.
- Use dot decimal separators consistently in External controls and telemetry,
  regardless of the system locale. Label plot traces and axes with their units,
  keep SiPM current plots in uA without automatic prefix changes, and convert
  older A/mA current recordings for plotting while preserving recorded values
  and schemas. Covered by simulated I/O and offscreen Polish/German locale tests.
- Update the shared Modbus register definitions to match the supplied
  `modbus_reg.c` units/comments: SiPM current in uA, voltage setpoints in V,
  temperatures in degC, and compensation in mV/degC. PMT temperature now
  applies its 0.01 factor; fractional setting readbacks are retained.
  External labels distinguish sensor temperature, compensation voltage, and
  integrator voltage. Recorded replay units remain unchanged. Verified with
  simulated register I/O and offscreen Qt tests; no live hardware validation.

## [0.7.8] — 2026-11-04

### Added

- Scope sliders beside the pretrigger, frame length, and periodic-gap spin
  boxes, with synchronized values and ranges. Hardware
  timing changes snap to supported steps and apply once on release; periodic
  mode and replay/acquisition interlocks apply to the sliders too. Verified
  with offscreen Qt tests using simulated devices; no live hardware writes.

- Measurement Showcases wiki section with an illustrated Na-22 511 keV
  LBC-detector coincidence walkthrough, published Zenodo session downloads,
  offline replay instructions, and timing-fit and energy-selection context.

### Fixed

- Numeric input widgets throughout the application now clamp out-of-range
  edits to the nearest boundary on Enter or focus loss. Discrete MCA and Scope
  hardware settings, external-device registers, and the Current DMA gap also
  snap to their supported steps before applying settings; halfway values round
  up and paired sliders stay synchronized. Valid continuous float settings
  retain their precision. Hardware API validation remains strict; covered by
  offscreen Qt tests, without live writes.
- Boards with exported IDS IIO devices but unusable HV ADC/DAC readbacks now
  disable the affected PSU at connection time, retaining the device error in
  a warning while keeping Scope/MCA available. Read-only startup checks catch
  EINVAL/EIO/ENODEV/ENXIO before PSU defaults, polling, or shutdown writes;
  transport and permission errors still propagate. Includes simulated
  regressions for both channels and converters; live hardware is unverified.
- Restoring saved settings no longer opens disabled or hidden workspaces,
  including a previously selected Power Supply tab on a board without a
  usable PSU. An unavailable saved tab keeps the current available workspace
  or falls back to the first available tab. Includes offscreen Qt regressions.

## [0.7.7] — 2026-10-26

### Fixed

- Entering zero in the Power Supply HV control now selects the existing 100 V
  minimum on Enter or focus loss instead of restoring the previous
  setpoint. The voltage tooltip explains the 100–1250 V range and zero-entry
  behavior. ADC readback and the separate safe-shutdown path are unchanged.

## [0.7.6] — 2026-10-22

### Fixed

- Waveform workbench time axes now display ns/µs/ms/s instead of stacked prefixes
  such as "kns", with synchronized top/bottom tick scaling when zooming. Sample
  coordinates, gate calculations and exported timings are unchanged.
- Hiding the MCA CC guide through its legend now persists across measurement
  refreshes, zoom and timing/availability changes. The user visibility choice
  is separate from CC enable, and showing it again restores the current timing.
- MCA CC timing guide now starts at the signal/frame beginning and includes
  pretrigger, ending at CC time rather than pretrigger + CC time. Moving the
  pretrigger marker no longer shifts or hides the short-window guide.
- MCA elapsed time no longer reuses the programmed-duration register's scale.
  The raw elapsed status already reports deciseconds; 999 ticks now displays
  99.9 s rather than 134.1 s. Corrects elapsed-normalized ROI rates and analysis/
  dose snapshots without changing measurement-duration programming. Includes
  raw-counter boundary and snapshot/UI regressions; diagnosis was read-only on
  PP hw101/sw103023, with no new acquisition commanded.

### Added

- Live PSD map logarithmic/linear density-scale selector, matching the waveform
  workbench. Display-only changes preserve accumulation, projections and FOM fits;
  the selected scale is saved with PSD settings and restored around session replay.
- MCA Statistics dead-time percentage alongside the existing millisecond value,
  calculated with consistent time units and shown as unavailable at zero elapsed time.
- Scope Acquisition elapsed-time indicator for continuous viewer and DMA runs,
  using a monotonic host clock without extra hardware polling. Retains the final
  time on stop and resets on the next run; excludes arming, single reads and replay.
- MCA **Record list-mode DMA** moved into Measurement above **DMA File...**,
  preserving its position through compact/regular reflow and acquisition locks.
- Dashed black, stepped MCA charge-comparison timing guide, shown only while
  CC is enabled; inverted lower/upper levels show the configured short window and the
  remaining diagnostic window without influencing auto-range. MCA debug and
  Waveform Analysis signal legends now default to bottom-right. The guide is
  schematic, not measured FPGA integration data.
- Waveform-workbench **Transfer gates to MCA...** with explicit target selection,
  pretrigger derived from baseline-region end to the green integration-start
  line (no manual trigger reference), sample-to-ns duration mapping, upward step rounding,
  current/requested/rounded preview, temporary waveform overlays, and a charge-
  comparison enable choice. Idle-only application rechecks replay/reconnect/DMA
  guards, suppresses duplicate widget writes, attempts rollback on partial write
  failure, and never starts acquisition. Includes bundled help and focused
  simulated tests; FPGA integration equivalence is not live-qualified.
- Waveform workbench source/frame/sample-period controls grouped in a scrollable
  left sidebar; redundant Load/Close file buttons and the top control row removed.
  File-menu commands retain their shortcuts and busy-state guards.
- PSD Event Readback and Waveform Analysis Workbench current filename/no-file
  state in the window title instead
  of a separate label row, reclaiming vertical space for plots.
- PSD Event Readback left-hand settings sidebar for histogram spinners and
  matrix-scale/PSD-cut/ROI controls, freeing plot height while retaining menu/F1
  synchronization and a resizable horizontal split.
- Energy Calibration and MCA Peak Analysis menu bars replacing command-button
  rows. Includes File/Analysis/View/Help commands, peak-reference management,
  synchronized peak counts-scale/panel choices, local shortcuts and bundled help,
  while preserving calibration guards and asynchronous fit/cancellation behavior.
- PSD Event Readback File/Analysis/View/Help menu bar for file operations,
  Gaussian fitting, fit-table copy, synchronized ROI/guide visibility, and bundled
  help. Window-local shortcuts replace the separate file-action toolbar while
  preserving existing action availability and the compact histogram settings.
- Green main-window status-bar badge listing channels with active Scope/MCA
  measurements, including DMA/PSD/coincidence. Tracks start/ready/stop/completion
  without extra hardware polling, deduplicates channels, qualifies multiple boards,
  and remains separate from offline analysis/replay and transient status messages.
- Wrapped waveform/PSD tooltips, including long file paths and menu hints, to
  readable multiline text while preserving complete explanations and F1 help.
- File-menu selected waveform-cluster export to native Scope NDMA binary (with
  JSON provenance), ROOT TTree, or HDF5, retaining original samples/timestamps,
  source-frame indices and PSD coordinates/settings. Includes bounded sample
  batches, background progress/cancellation, exclusive output creation, cleanup,
  and variable-length CAEN waveform support in ROOT/HDF5.
- Waveform Analysis Workbench menu bar with File, Analysis, Selection, View, and
  bundled Help actions; window-local file/reconstruction shortcuts and menu state
  synchronized with the existing controls and operation availability.
- Toggleable selected-cluster average waveform overlay with cancellable background
  reading, per-event baseline removal and reconstruction polarity, preserving
  sample timing/amplitude and using the shared prefix for variable-length records.
- Independent visibility switches for offline PSD ROI/cut overlays and waveform
  cluster-selection controls/overlays. Hiding them preserves cuts, filters, and
  match position. Rectangle/polygon outlines and handles are black, including
  on hover, for improved map contrast.
- Waveform PSD workbench rectangle/polygon cluster selection with exact event-
  coordinate cuts, original source-frame tracking, matching-event navigation,
  current-event map marker, and clear/empty/stale-selection handling. Only compact
  per-event coordinates are retained; waveforms remain file-backed. Manual
  regions are not automatic particle-identification labels.
- Raw-waveform negative-PSD diagnostic script with signed integrations, mean/
  median baseline comparison, explicit offline gates, pulse and cumulative-charge
  plots, and reproducible numerical summaries; source captures remain read-only.
- Thin Jupyter/VS Code cell script `notebooks/explore_mca_psd.py` for browsing
  ROOT list-mode branches, configuration/schema metadata, energy spectra, PSD
  maps, and ratio projections with user-selected ranges. Starts with a bounded
  preview and preserves raw integer data and the source file.

## [0.7.5] — 2026-10-20

### Fixed

- Offline session replay no longer monopolizes the GUI thread with per-batch
  HDF5 reads, coincidence/PSD accumulation, Gaussian timing fits, and redraws.
  An interruptible background worker processes every batch in recorded order;
  a bounded latest-state mailbox throttles display updates to 10 Hz, and hidden
  PSD/coincidence plots paint only when shown. PSD re-binning remains available
  asynchronously during replay and at EOF. Replay exit/replacement joins the
  worker before restoring views; analysis failures are logged and displayed.

### Added

- Searchable offline parameter reference under Help → User Guide, shared with
  the wiki and bundled in wheel, PyInstaller, and Nuitka distributions. F1 enters
  click-to-explain mode for labels and controls in the workspace and owned dialogs;
  selection clicks do not operate acquisition controls. Esc/F1/right-click cancel,
  related topics and Back/Forward aid browsing, and unmapped controls explicitly
  fall back to their workspace guide.
- PSD and waveform workbenches use compact control labels and tooltip/F1
  instructions instead of permanent explanatory banners, retaining visible
  fit diagnostics and exhaustive-search confirmation. The waveform signal-preview
  legend defaults to the top-right and follows plot resizing.
- Waveform and pulse displays now use sample-hold steps in Scope raw/DMA/replay
  and persistence, MCA debug traces, the offline waveform workbench, and Current
  DMA. Sample values/timestamps and analysis are unchanged; DMA gaps remain
  disconnected, and scalar trends and fitted curves retain line rendering.
- Waveform FOM optimization offers an optional exhaustive search over every
  valid integer short/long gate pair within the selected search radius. A
  slow-search warning and confirmation show the candidate count before starting;
  progress, cancellation, fit-quality checks, and full-data confirmation remain
  available. The existing local search remains the default.

## [0.7.4] — 2026-10-16

### Fixed

- PyInstaller builds now bundle Awkward's ctypes-loaded native kernels and
  dynamically imported Uproot behaviors, fixing the ROOT conversion menu failing
  before the file picker opens. ROOT loads on the conversion worker and failures
  are reported visibly. Reviewed builds verify list-mode and waveform conversion
  inside the actual executable using synthetic captures before accepting it.
- ROOT conversion failures now emit System Log `ERROR` records containing
  the source/destination paths, exception type, and traceback even when debug
  mode is off. Start, completion, and cancellation are logged at `INFO`.

## [0.7.3] — 2026-10-16

### Added

- **File → Convert Binary to ROOT...** exports native Scope NDMA v1 and MCA
  list-mode NDMA v1/v2 to bounded-memory ROOT TTrees with background progress,
  cancellation, raw integer fields, configuration and verified capture metadata.
  Incomplete or changing captures are rejected and existing files are protected.

- **Help → About → Check for Updates** checks GitHub's latest published stable
  release without blocking the GUI, compares versions numerically, and offers
  a release-page link when a newer version is available. Connection failures,
  timeouts, rate limits, and missing releases are shown in the About dialog.

## [0.7.2] — 2026-10-15

### Added

- The MCA Peak Analysis workbench now securely validates and imports
  ANSI/IEEE N42.42 `.n42`/`.xml` documents, including multi-spectrum files,
  `CountedZeroes` data, real/live timing, detector and instrument identity,
  and coefficient or energy-boundary calibration. Validation uses bundled
  NIST base/add-in schemas without resolving XML entities, external schemas,
  or instance location hints. Calibrated, timed spectra can be exported as
  non-overwriting N42 documents, and fitted results can include standard
  `SpectrumPeakAnalysisResults` centroid, FWHM, area, and uncertainty fields.
  Export rejects stale or missing calibration, missing timing, and negative
  derived counts instead of manufacturing measurement metadata.

### Fixed

- Acquisition messages from MCA, Scope, Current, PSD, and Coincidence now use
  the bottom status bar instead of expanding labels in the measurement panels.
  Messages identify their panel/channel, retain full diagnostic tooltips, and
  restore the latest panel status when switching tabs.

### Documentation

- Added a goal-oriented use-cases guide and reconciled the getting-started,
  acquisition, analysis, and troubleshooting documentation with the 0.7.x
  calibration, reporting, PSD, offline/live connection, and multi-digitizer
  workflows.
- Refreshed the live MCA, energy-calibration, peak-fit, Scope Auto Setup, and
  PSD images from the CH0 CAEN Na-22 generator on `192.168.10.135`, including
  the new centroid-residual and FOM diagnostics. The MCA examples use binning
  128 to retain both generated lines without energy overrange. Added a
  live-count functional example of the dose-calibration logger without PSU
  data.

## [0.7.1] — 2026-10-14

### Fixed

- The dose-calibration live plot no longer enables viewport-dependent
  downsampling and clipping for its bounded history, preventing short series
  from appearing empty in narrow or headless viewports.

## [0.7.0] — 2026-10-14

### Added

- A dose-calibration logger under **Tools** writes interval-based CSV rows with
  a reference dose rate that changes only after editing finishes. Users select
  the desired MCA count-rate/dead-time statistics, total and visible-ROI
  differential-spectrum integrals, PSU setpoints/readbacks, detector and board
  temperatures, and cumulative PSD below/above-cut integrals over the full
  energy range and selected PSD ROI. An adjacent live plot selects among the
  enabled numeric fields, including the user-entered reference dose. Rows are
  flushed immediately, unavailable values remain explicit blanks, and existing
  files are not overwritten.
- Offline analysis mode now provides **Connection -> Connect...**. It reopens
  digitizer discovery, validates every requested IIO channel before replacing
  the read-only workspace, rebuilds the live measurement controls in place, and
  restores the offline workspace if live controller construction fails. The
  connection progress bar now spans the full dialog content width.
- A live workspace can use **Connection -> Add Digitizer...** to append another
  IIO endpoint. Channels are identified as `D1:CH0`, `D1:CH1`, `D2:CH0`, and so
  on, and each board receives independent global controls. **Disconnect
  Digitizer...** removes a selected endpoint, renumbers remaining boards, and
  returns the app to offline analysis when the final board is removed. The
  Coincidence tab can select any eligible digitizer while retaining per-board
  settings, but each pair remains deliberately restricted to one physical
  digitizer; cross-board synchronization is not assumed or enabled. Global
  controls use a compact two-column layout, with diagnostics beside the
  shared-start and temperature controls; synchronized-start guidance wraps
  beside compact action buttons. Power-supply controls form a compact band
  above the full-width monitoring plot instead of reserving a tall sidebar.
- The MCA peak-analysis workbench can download a non-overwriting report bundle
  after a completed fit. It includes a branded two-page PDF with the spectrum,
  model components, residuals, fitted peak table, statistics, and TeX-style
  Gaussian, FWHM, and selected-background equations, plus complete curve, peak,
  parameter, provenance, and reference-overlay data in CSV and JSON files.
- MCA Energy Calibration now treats each draggable reference marker as an
  initial seed for a local Gaussian-plus-linear-background fit and uses the
  fitted centroid for the channel-to-energy polynomial. The configurable fit
  half-width defaults to 300 channels to avoid locking onto narrow noise peaks;
  separate seed and centroid columns, solid centroid overlays, and explicit
  convergence failures make the refinement visible. Applying the
  calibration snaps the draggable seeds to their centroids and overlays each
  local Gaussian fit with an aligned observed-minus-fit residual plot;
  subsequent edits hide both. A matching report download shows the local fit
  residuals and preserves source spectra, seeds, centroid fits, calibration
  points, energy residuals, settings, and typeset model equations in PDF, CSV,
  and JSON files.
- A persistent **ENERGY CALIBRATED** status-bar badge identifies when one or
  more MCA channels have an applied calibration. Its tooltip lists those
  channels, and double-clicking the badge opens the calibration workspace.
- A dual-channel MCA gain-drift CLI diagnostic tracks the Na-22 511-keV peak
  maximum and local centroid from saved differential ROI spectra while logging
  MCA statistics, temperature correction, detector/board temperatures, power
  telemetry, rails, clocks, and available HV feedback. It reproduces the
  auto-saved MCA and ADC-temperature-correction settings without writing Scope
  registers, can explicitly apply saved per-channel PSU bias with guaranteed
  safe-shutdown cleanup, restores the original MCA state, supports distinct
  channel ROIs, and emits CSV, NPZ, JSON, and PNG investigation artifacts.
- The live MCA histogram can switch between the accumulated spectrum and the
  non-negative per-refresh count difference. The differential reference resets
  on mode, run, clear, and configuration changes, while CSV exports and analysis
  snapshots continue to use the accumulated hardware spectrum. Differential
  display and MCA ROI visibility are controlled from the View menu. Double-
  clicking an MCA ROI opens a modeless pyqtgraph history of its integrated
  counts versus acquisition elapsed time. The history follows the selected
  accumulated or per-refresh differential display mode, and moving the ROI or
  changing modes starts a new trace.
- MCA logarithmic Y-scale control is again in the View menu. Hovering over an
  MCA histogram and pressing `L` toggles the shared linear/logarithmic state.
- Live and Tools/readback PSD maps and coincidence matrices render zero- and
  one-count bins transparently, reducing low-occupancy background clutter
  without changing projections, fits, or exported matrix values.
- Waveform gate-driven PSD recalculation preserves the user's energy-projection
  ROI instead of restoring the default first-quarter region after every update.
- Waveform Analysis can optimize short- and long-gate endpoints for a local
  maximum of the fitted PSD figure of merit. The bounded, cancellable search
  holds the baseline, integration start, energy/fit ROIs, and centroid seeds
  fixed, excludes poor-convergence candidates, then rebuilds and refits the
  full PSD at the selected gates.
- PSD Event Readback and Waveform Analysis now keep the current source file's
  full path visible and include its filename in the workbench window title.
- PSD Event Readback and Waveform Analysis can export a file-named report folder
  after an FOM fit. It contains a branded three-page PDF of the PSD matrix,
  fitted FOM projection, cut-separated spectra, convergence results, source and
  digitizer metadata, software version, cut/binning/ROI settings, plus complete
  matrix, spectrum, projection, fit-component, settings, and JSON metadata files.
  Waveform reports also preserve the exact integration gates and processing
  context used to produce the fit.
  The FOM page now typesets the fit model, Poisson-deviance objective, FWHM,
  and FOM equations in TeX-style mathematical notation.
- Live and offline/readback PSD ratio projections now provide a toggleable
  draggable fit ROI, two draggable centroid seeds, and button-triggered bounded
  Levenberg-Marquardt double-Gaussian fitting with Poisson-deviance residuals,
  strictly interior bound initialization, independently estimated starting
  widths, and ordered but non-midpoint-constrained centroids. The lower-right
  results panel reports both centroids and FWHMs, reduced deviance, and the
  conventional PSD figure of merit, with explicit warnings for weak or poor
  convergence. Waveform Analysis and PSD Event Readback can copy the fitted
  populations and shared FOM diagnostics as a tab-separated table.
- The MCA peak-analysis workbench can overlay a pinned offline NNDC ENSDF
  catalogue containing 2,065 decaying parent nuclides/states and 87,481
  default-quality photon lines. The generated SQLite artifact applies ENSDF
  photon and branch normalization, excludes limit-only/uncertain records from
  the default overlay, retains source-dataset provenance, and adds the DDEP/
  IRDFF-II Na-22 annihilation reference. Multiple nuclides, an emission-
  probability cutoff, source tooltips, calibrated channel mapping, and
  reproducible JSON fit-export metadata are supported. An indexed fit-ROI
  candidate filter composes with text and probability filtering, ranks parents
  by matching-line count and strongest emission, and remains explicitly a
  candidate aid rather than an identification. Persistent, user-editable
  favourites pin common laboratory sources above the full list and matching
  ROI candidates. Candidate searches clip calibrated ROIs to the non-negative
  photon-energy domain. Missing or stale calibrations suppress both calibrated
  features explicitly.

### Fixed

- Adding a digitizer now compares resolved IPv4/IPv6 endpoints as well as the
  entered address. A board already connected by numeric IP cannot be added a
  second time through its Avahi/mDNS hostname (or vice versa).
- Saved dock layouts are now restored only when their digitizer/channel topology
  matches the current workspace. Restarting with fewer digitizers no longer
  reapplies obsolete dock placeholders that stack channel views vertically and
  force the main window beyond the available screen height.
- PSD report energy spectra now always follow the panel's linear or logarithmic
  count-axis selection, including empty spectra.
- PSD report rendering no longer floods the system log with Matplotlib or
  Pillow internals when developer debug mode is enabled.
- Offline analysis no longer advertises nonexistent MCA debug selectors 9-15
  and consequently no longer warns about them during workspace startup.
- Frozen builds now detect a missing or unloadable native libiio runtime after
  Qt starts but before importing hardware modules. A recovery dialog directs
  users to the pinned v0.26 Windows installer instead of exposing a
  PyInstaller missing-DLL traceback. The Python binding is constrained to the
  0.x API, and host runtimes reporting the breaking libiio 1.x API are rejected
  while its compatibility layer and DMA model remain unvalidated.
- Pyqtgraph plot surrounds now follow the active Qt interface palette while
  retaining a distinct light data canvas. Offline PSD projection axes align
  exactly with the PSD map, and hovering the bottom energy projection while
  pressing `L` toggles its count axis between linear and logarithmic scale.
- MCA Stop now keeps Start and configuration controls interlocked until the
  list-mode DMA worker has drained, destroyed its native buffer, and released
  the hardware ownership gate, preventing an immediate restart from failing
  with `EBUSY`. File-backed output also restores the DMA file selector after
  a PSD session finishes.
- Definitive IIO connection-loss errors stop MCA, diagnostics, and temperature-
  correction polling after one report instead of generating a traceback on
  every timer tick. Online list-mode continuity errors no longer claim that a
  nonexistent capture file was retained.

## [0.6.3] — 2026-10-05

### Added

- Startup **Offline / Replay Mode** opens the complete analysis workspace
  without an IIO or Modbus connection. Transport-free channel facades host
  Scope, MCA, PSD, coincidence, and PSU replay while a permanent hardware
  interlock disables acquisition and configuration controls. Recorded external
  modules are reconstructed as read-only panels from the session schema, and a
  persistent `OFFLINE` badge identifies the mode independently of the replay
  badge.
- Versioned `.nlab-session` directory recording from the File menu, with a
  complete configuration snapshot, append-only HDF5 measurement streams,
  JSONL telemetry/command/log chronology, mutable scientific annotations,
  completion/drop state, and SHA-256 integrity metadata. Offline replay
  validates the bundle and deterministically reconstructs MCA spectra in the
  peak-analysis workbench without replaying recorded hardware commands. A
  persistent status-bar badge and window-title suffix identify replay mode,
  which can be explicitly exited without affecting unrelated live hardware
  controls.
  Timestamped Scope viewer frames now play once at their recorded cadence, with
  full DMA frames as a fallback; affected live Scope acquisition is stopped and
  isolated, EOF holds the final frame, and playback never loops or falls back to
  live data. Replay now provides an application-wide hardware interlock: active
  Scope, MCA, and Current acquisition is stopped, automatic temperature writes
  are suspended, and all hardware-setting widgets and menu actions are disabled
  until replay is exited. Recorded MCA polling checkpoints now animate the
  normal histogram/debug/statistics views, and timestamped list-mode batches
  progressively rebuild PSD and two-channel coincidence views using saved
  analysis settings. Older bundles derive frame timing from hardware event
  timestamps. PSU and external module polling is suspended during replay and
  their recorded timestamped telemetry is played once into the normal views,
  holding the final values at EOF and resuming only previously active pollers
  on exit. Current-monitor samples, global diagnostics, and temperature-
  correction readbacks now replay on the same clock as Scope/MCA/telemetry.
  Numerical streams use bounded lazy HDF5 iterators, HV-only PSU recordings no
  longer assume SiPM fields, external-module schemas survive offline replay
  (including configured modules with no samples), and undersized offline
  workspaces reject a session explicitly instead of omitting channels.
- A new **Connection → Refresh Modbus Devices** action rescans both ser2net
  bridges without blocking the GUI, safely replaces external-device polling
  workers and docks, and preserves display preferences for rediscovered
  modules.

### Changed

- Connect, add, disconnect, and reconnect now share one validated topology
  transition with explicit transport ownership and rollback. Replacement
  transports are prepared before the active workspace is stopped, teardown or
  view-rebuild failures restore the previous workspace, and the window derives
  its title, channel count, offline indicator, and connection actions from the
  controller after every successful transition.
- External-device settings and telemetry tables use a compact layout with
  bounded register-name columns, aligned table edges, and full-name tooltips,
  keeping every column visible beside the live plot at the default window size.
- The application now uses the direct IIO backend exclusively. Startup,
  connection settings, acquisition workers, examples, and standalone builds no
  longer expose a transport selector or require gRPC/protobuf/ZMQ components.
- Older settings that omit `connection.backend` or specify `iio` remain
  compatible. Settings that explicitly request `grpc` are rejected with a
  pointer to the preserved `grpc-final` branch.
- Digitizer feature discovery is exposed as one immutable capability snapshot
  instead of repeated backend introspection. GUI coincidence availability now
  consumes that snapshot rather than checking concrete streamer classes.
- Optional Current, MCA, and PSD controllers are keyed to their hardware
  channel, so a scope-only channel no longer disables or shifts MCA-capable
  channels on mixed firmware.
- Scope and MCA DMA checkboxes are application-owned capture choices. Settings
  format v4 moves them out of hardware configuration; v3 files migrate on
  load.
- Coincidence timing controls explicitly disable digit grouping, and the
  sideband gap is shown as whole nanoseconds so locale-specific decimal marks
  cannot make `10 ns` look like `10000 ns`.

### Removed

- Removed the legacy gRPC control backend, ZMQ DMA streamers, protobuf sources
  and generation step, and their direct dependencies. Offline reading of
  existing NDMA v1 files is retained.
- Removed the no-op Scope/MCA DMA setters. The remaining `dma_active()` calls
  report driver-owned buffer gates and cannot write them.

## [0.6.2] — 2026-10-01

### Added

- The startup connection dialog now discovers `iiod` boards through the local
  DNS-SD/mDNS (Avahi) scan, independently probes the USB-network endpoint at
  `192.168.3.1:30431`, and provides a non-blocking **Rescan** button.

## [0.6.1] — 2026-09-30

### Added

- General Settings panel with an opt-in automatic configuration snapshot. It
  saves all connected Scope, MCA, and Power Supply settings when enabled and on
  clean exit, then restores them on startup unless `--config` was supplied.
  Snapshots are isolated by device IP so configurations cannot be mixed between
  digitizers.
- Scope display and refresh defaults, MCA refresh, PSU polling/history, DMA
  destination and output format, and histogram ROI/log options are now edited
  application-wide in General Settings instead of being scattered across
  channel panels and menus. Long explanatory text is available as tooltips.
- Coincidence controls use compact labels; detailed gate, timing, matrix, and
  recording explanations are available as hover tooltips.

### Fixed

- GitHub-hosted Windows release builds now install and checksum the official
  libiio v0.26 native runtime before importing the `pylibiio` wrapper, so test
  collection and PyInstaller packaging no longer depend on a runner-global DLL.

## [0.6.0] — 2026-09-28

### Added

- Per-channel live Current workspace backed by the FPGA input-filter IIR
  register. A worker-owned IIO/gRPC connection targets 1 kHz acquisition into
  a bounded buffer while the GUI renders at no more than 30 Hz, with measured
  rate, median transaction latency, sample age, interval statistics, rolling
  history, overflow reporting, and saved zero/scale/unit calibration.
- Selectable online Scope DMA mode for the Current workspace. Validated frames
  are reduced on the receiver thread into bounded, sample-weighted summaries
  before any replaceable GUI hand-off. IP122 automatically uses the
  live-qualified 8,188-sample/100 us maximum-coverage preset and mirrors it into
  the Scope controls; other firmware retains its visible frame/gap settings.
  Host receive and FPGA timing metrics are reported separately, and
  protocol/analysis/display losses are distinguished. A visible Scope panel
  previews the latest owned DMA frame without a second IIO viewer transaction.
  The IIO backend consumes the v122 driver-advertised dynamic DMA geometry
  rather than assuming that the configured full-rate frame length is the
  transport length, and qualified remote streams use READBUF x64. The Current
  plot retains a bounded 10 ms window of complete raw waveform fragments on
  the FPGA timestamp timeline, with explicit breaks between fragments and no
  additional software waveform averaging. Interval statistics remain
  sample-weighted bins; the UI reports post-frame gap separately from nominal
  and measured start-to-start spacing.
- Current monitors now start stopped and open no polling or DMA transport until
  the operator presses **Start monitor**.
- The stopped Current Scope-DMA mode exposes its post-frame gap as an exact
  8 ns-step control, allowing operators to compare coverage and skipped-trigger
  behavior without editing the Scope workspace or source code.
- A minimal `examples/measure_current.py` demonstrates continuous IIR polling
  and Scope-DMA current measurement, optional callbacks, and where a non-blocking
  consumer worker can be connected.

### Fixed

- Visible Current Scope-DMA plotting no longer starves the receive thread. The
  30 Hz scalar readout is decoupled from a 2 Hz raw-fragment waveform paint;
  every received frame still contributes to scientific sums, while only a
  bounded 10 ms raw display window is retained. The live diagnostic shows the
  offscreen widget so Qt paint regressions are exercised instead of timing
  `setData()` alone.
- Starting Current Scope DMA explicitly reapplies the trigger-level and DAC-
  baseline values displayed in Scope. Periodic acquisition preserves the
  threshold for later trigger modes, and the selected DAC operating point is
  used unchanged for the current waveform.
- Stopping online Current DMA now clears the temporary Scope DMA selection
  after buffer teardown, so stale GUI state no longer prevents Scope Auto
  Setup. The stopped Current toolbar also provides **Reset Scope defaults** to
  restore ordinary trigger, timing, DAC, and non-DMA settings after the IP122
  optimization preset.
- Scope DMA size progress now preserves byte counts beyond the signed 32-bit
  Qt boundary and displays long recordings in GiB instead of appearing to
  restart from zero at multi-gigabyte sizes.
- Background workers are now destroyed during their QThread shutdown instead
  of racing GUI-side reference cleanup. This covers Scope/MCA DMA and polling,
  Scope Auto Setup, PSD and waveform readers, peak fitting, timing validation,
  temperature correction, power-supply monitoring, global diagnostics, and
  external-device polling. GUI-side completion handlers are explicitly queued
  back to the main thread so rendering a completed large-file analysis cannot
  update Qt widgets from the worker thread.

### Documentation

- Rebuilt the project wiki from Quarto sources with dedicated installation,
  acquisition, Current, PSD, coincidence, analysis, data-format, architecture,
  development, and troubleshooting guides.
- Replaced generated demonstration plots with real `.128` hardware captures,
  including Current IIR/DMA views, Cs-137/Na-22 MCA workflows, coarse and CFD
  coincidence timing, the prompt energy matrix, PSD, MCA energy calibration,
  and a completed 511 keV peak fit. Added device front- and back-panel images.

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

[Unreleased]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/compare/v0.8.4...main
[0.8.4]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.8.4
[0.8.3]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.8.3
[0.8.2]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.8.2
[0.8.1]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.8.1
[0.7.6]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.7.6
[0.7.5]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.7.5
[0.7.4]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.7.4
[0.7.3]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.7.3
[0.7.2]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.7.2
[0.7.1]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.7.1
[0.7.0]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.7.0
[0.6.3]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.6.3
[0.6.2]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.6.2
[0.6.1]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.6.1
[0.6.0]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.6.0
[0.5.0]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.5.0
[0.4.0]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.4.0
[0.3.2]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.3.2
[0.3.1]: https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases/tag/v0.3.1
[0.3.0]: https://github.com/nucliflare/nlab-digitizer-community/releases/tag/v0.3.0
[0.2.0]: https://github.com/nucliflare/nlab-digitizer-community/releases/tag/v0.2.0
[0.1.0]: https://github.com/nucliflare/nlab-digitizer-community/releases/tag/v0.1.0
