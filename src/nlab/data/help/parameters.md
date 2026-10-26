## Using contextual help {#help}

Press **F1**, then click a label, number field, checkbox, button, or plot. The
question-mark cursor indicates help mode. The selection click opens an explanation
instead of operating the control. **Esc**, F1 again, or a right-click cancels.
Keyboard editing and wheel adjustments are blocked while choosing an item.
Help does not pause acquisition: use the normal Stop controls if that is required.

**Help → User Guide…** opens this searchable offline reference directly. Search
matches both titles and explanations. Back and Forward revisit previous topics.
Labels and their associated controls lead to the same topic where mapped. Items
without a dedicated entry open their workspace guide, explicitly marked as such.

These pages describe application behavior, not detector-specific operating
limits or a live hardware qualification. Firmware capabilities and the units
shown by each control remain authoritative. Do not copy raw thresholds or
calibration coefficients between different signal paths.

Start with [Connection](#connection), [Scope](#scope), [MCA](#mca),
[PSD](#psd), or [offline waveform analysis](#waveform).

The complete workflow guides and screenshots are also available in the
[project wiki](https://github.com/nucliflare/nlab-digitizer-community/wiki).
Opening that link requires a browser and network access; this reference does not.

## Connection — endpoints and offline mode {#connection}

The main window's green **Measuring: CH0, CH1** status-bar badge lists channels
with active live Scope/MCA runs, including DMA used by PSD and coincidence. It
remains visible across tabs and counts each channel once even when both cores
are running. Multiple connected digitizers use unambiguous **D1:CH0** labels.
The badge follows application start/ready/stop/completion notifications, not
extra hardware polling or external acquisition changes. Idle, offline analysis,
and recorded replay do not activate it.

Choose the digitizer address, IIO port (normally 30431), and requested channels.
**Rescan** refreshes discovery; it does not establish a measurement. Verify the
channel identity after a firmware or device-tree change.

**Offline / Replay Mode** opens analysis without hardware transport. From a live
workspace, **Add Digitizer…** adds an independent board. A shared GUI does not
give independent boards a synchronized clock. Stop affected acquisitions before
disconnecting or reconnecting a board.

See [session replay](#session) and [acquisition ownership](#ownership).

## Acquisition — ownership and safe configuration {#ownership}

Each DMA path has one owner. Current waveform DMA uses the Scope path;
Coincidence owns both selected MCA streams. Stop the current owner and allow
its worker to drain and release the buffer before starting a conflicting task.

Do not reconfigure MCA DSP settings during list-mode DMA. For ordinary histogram
acquisition the application's synchronized configuration path can stop, write,
and restart, beginning a new accumulation. A disabled control may indicate this
ownership interlock, not a missing feature.

The client validates complete frames. A clean plot alone does not establish
continuity: review final run diagnostics and target-side errors. Hardware faults
cannot be repaired by changing a display setting.

## Scope — inspecting a signal {#scope}

Scope shows the ADC waveform, with trigger threshold and pretrigger markers.
Use [Auto Setup](#scope-auto) for an initial view, then inspect the real waveform
before starting a long capture. Scope and MCA thresholds belong to different
signal paths and must not be copied between them.

The raw-view trace and markers support drag previews. Hardware changes are
committed on release; verify the next acquired frame rather than treating a
drag preview as a measurement. See [trigger](#scope-trigger),
[timing](#scope-pretrigger), and [DMA](#scope-dma).

Waveforms use sample-hold steps, transitioning at the next displayed sample's
timestamp rather than drawing a diagonal between codes. Persistence uses the
same timing convention. Averaged or decimated previews retain their actual
display resolution; steps do not reconstruct the analogue waveform.

## Scope — trigger level and mode {#scope-trigger}

**Level (raw)** sets the Scope ADC threshold, not volts or the baseline-subtracted
MCA threshold. The dashed threshold line and number field are synchronized.

**Mode** chooses the available level, edge, or periodic triggering behavior.
For a negative-going pulse, its leading edge falls in amplitude; its recovery
edge rises. Select the desired crossing rather than inferring it solely from
the pulse minimum. Periodic acquisition is not a pulse-threshold selection.

Choose a threshold distinguishable from baseline noise. A threshold beyond
the pulse amplitude can leave acquisition waiting for a trigger. Changing
Scope triggering does not change the separate [MCA trigger](#mca-trigger).

## Scope — DAC baseline {#scope-dac}

The DAC control changes the analogue offset presented to the Scope ADC path.
It is not a mathematical subtraction from a saved waveform. Keep the baseline
and pulse inside the available ADC range to avoid clipping.

Auto Setup measures a DAC sensitivity for vertical trace-drag previews. Without
that measurement the preview is approximate. Verify the acquired waveform after
release. Offline waveform baseline subtraction is a [different operation](#waveform-baseline).

## Scope — pretrigger {#scope-pretrigger}

Pretrigger controls how much of the frame precedes the trigger. The displayed
control is in nanoseconds; the Scope drag preview uses the native 8 ns grid.
Leave enough quiet baseline before the pulse for later analysis, and enough
post-trigger duration for its tail.

Dragging the green marker previews a horizontal shift. The next hardware frame
after release confirms the actual positioning. Pretrigger is a time-window
position, not the pulse-shape integration start or a change of trigger polarity.

## Scope — frame duration {#scope-frame}

**Frame** specifies the recorded window duration in nanoseconds, despite the
internal widget name containing “Samples”. A longer window retains more baseline,
tail, and possible late pulses, but increases bytes per event and analysis cost.

Configure geometry while stopped. A Scope NDMA frame contains timestamp words
as well as ADC samples; do not interpret every stored word as waveform amplitude.
Use the application's readers or [conversion tools](#conversion).

## Scope — periodic frame gap {#scope-gap}

**Periodic gap** sets the post-frame gap in nanoseconds for the applicable periodic
acquisition mode. It is not an inter-event time measured from the detector and
does not make pulse-triggered events uniformly spaced.

The control is enabled only where supported by the current mode and backend.
Current DMA can apply its own [qualified Scope preset](#current-mode).

## Scope — Start, Stop, and Acquire Frame {#scope-run}

**Start** begins the selected Scope workflow. **Acquire Frame** requests an
individual frame for inspection. **Stop** ends the active workflow; when recording,
allow the reader to drain and close before reconfiguring the channel.

The [duration](#duration) and **Record DMA frames** selection affect acquisition.
Scope viewing can use the preview supplied by its DMA owner rather than opening
a second competing buffer.

**Elapsed** reports monotonic host-clock seconds since continuous acquisition
started (since DMA ready for recording), independent of viewer refresh or trigger
rate. It retains the final value after stopping and resets on the next run.
Arming, single-frame reads and session replay do not advance this indicator.
It is not hardware live time or the duration represented by recorded samples.

## Scope — Auto Setup {#scope-auto}

Auto Setup performs a bounded survey of baseline, pulse polarity, threshold,
and analogue offset, including a provisional-trigger search for narrow randomly
arriving pulses. It applies settings; it is not a display-only zoom command.

Run it while ordinary Scope acquisition and file DMA are stopped, and wait for
the worker to finish. Inspect its verification result and the next real pulse.
Noise-only data or an unsuitable signal can prevent a verified result; do not
treat the proposed settings as universal detector calibration.

## Scope — DMA waveform recording {#scope-dma}

**Record DMA frames** selects full-frame waveform capture. Choose a new destination
with **DMA File…**, then use Start and Stop. Existing captures are protected rather
than silently overwritten. The source is the Scope waveform path, not MCA list mode.

Stop and wait for final drain before opening the capture for conversion or
changing geometry. Partial frames are not appended as complete events. Review
the final run summary for continuity and hardware faults. Current waveform DMA
cannot simultaneously own the same Scope buffer.

## Acquisition — duration {#duration}

Duration is in seconds. **0 means run until stopped**, not a zero-length run.
A nonzero value requests a timed acquisition. Network and drain completion can
make the final GUI stop later than the nominal limit.

The field does not change the waveform sample period, trigger threshold, or
histogram polling frequency. Wait for recording completion before treating
an output file as final.

## Display — refresh rate {#refresh}

Refresh controls how often the client requests or presents monitoring updates.
Use the units shown: some panels use Hz, others an interval in milliseconds.
Higher requested rates increase network and rendering work and may be capped
by the application.

Polling times are not a hardware sample clock. A slow plot is not proof of
lost DMA records, and a fast plot is not proof of complete transport. Check
acquisition counters separately.

## Display — persistence, scaling, and history {#display}

Raw Scope display emphasizes the current waveform; persistence accumulates
the visual history with the configured retention. History duration controls
how much monitoring data remains visible.

Linear/logarithmic scaling changes presentation, not raw recorded values or
calibration. Sparse zero bins may be transparent on logarithmic images. In MCA,
the View menu controls logarithmic Y; hovering its plot and pressing L also
toggles it. Live and offline PSD have a **Matrix scale** selector for logarithmic
(`log1p` counts) or linear count-density colors. The default is logarithmic;
zero/one-count bins remain transparent in either mode. Changing matrix scale
does not change accumulation, projections, ROI selections or FOM fits. Matrix
scale and the energy histogram's logarithmic Y axis have separate display state.

## MCA — spectrum and DSP overview {#mca}

MCA accumulates an energy histogram and provides two selectable diagnostic traces.
The Scope ADC signal and the MCA input/filtered/debug signals need not share
amplitude units or timing latency.

Configure polarity, baseline, triggering, and the required shaping path while
stopped. Inspect [debug sources](#mca-debug), then start a histogram or
[list-mode capture](#mca-dma). Configuration changes during ordinary acquisition
can begin a new accumulation; list DMA requires its full stop/drain lifecycle.

## MCA — pulse polarity {#mca-polarity}

Polarity tells the pulse processor whether detector pulses are negative- or
positive-going. Match the actual input signal. It is not the same control as
Scope edge selection, nor a request to trigger on the recovery edge.

Verify the [selected trigger signal](#mca-trigger) and debug source. Changing
polarity can affect trigger and energy interpretation, so recheck the spectrum
and calibration afterward.

## MCA — baseline window {#mca-baseline}

Baseline selects the time window used by the pulse processor to estimate the
signal baseline. A longer estimate can smooth noise but can also respond
differently to baseline motion and nearby pulses.

Its nanosecond setting is distinct from the Scope DAC offset and the offline
workbench's user-selected baseline region. Verify performance at your count rate
rather than assuming a longer window always improves energy or PSD resolution.

## MCA — trigger threshold and source {#mca-trigger}

**Trigger level (raw)** is the threshold in the selected MCA trigger path.
**Trigger source** selects threshold, CR-RC², or CR²-RC² triggering where available.
**Ext. Trigger** enables the external/shared trigger path; it can leave a core
armed and waiting even when no events appear.

The threshold is not a Scope ADC code. A selected debug signal may have its
own processing delay and scale. Do not decide which edge fired just from where
the pretrigger marker intersects a different diagnostic trace.

The legacy edge-detector coefficient is capability-dependent. Disabled or
unsupported settings should not be interpreted as active hardware behavior.

## MCA — debug window and pretrigger offset {#mca-window}

**Window length** sets the diagnostic waveform capture-window duration;
**Pretrigger offset** positions it relative to the diagnostic trigger.
Both controls are displayed in nanoseconds. Debug samples use an 8 ns timebase.

The green line marks the configured offset. It is not a universal proof of
the analogue threshold-crossing instant in every internal signal: filter and
logic paths can have different latencies. These controls do not define offline
short/long PSD integration gates.

## MCA — diagnostic signal sources {#mca-debug}

Debug 1 and Debug 2 select internal inputs such as input signal, trigger signal,
trapezoid output/energy, CFD signal/window, charge-comparison window, PSD
zero-crossing window, or logic trigger, subject to firmware support.

“Trigger signal” is an internal signal, not necessarily a Boolean trigger flag;
two traces can therefore look alike. Choose Logic trigger to inspect the logic
path where available. Shared axes do not guarantee identical signal scales or
processing latency. The debug view is not a continuous Scope recording.

Both debug traces use sample-hold steps. This changes drawing only, not sample
timestamps, filter latency, or the hardware trigger decision.
The MCA debug legend and offline waveform-preview legend default to bottom-right
and remain anchored there when resized.

## MCA — hardware pile-up window {#mca-pileup}

The Pile-ups control sets the hardware interval used to classify a second pulse
as pile-up. Its hardware interpretation is backend-dependent; use the advertised
control units and range rather than assuming it is an offline sample count.

Inspect the pulse-pileup statistic and detector response at the intended rate.
This hardware setting is not a general software waveform-cleaning algorithm.
Changing it does not retrospectively remove events from an existing capture.

## MCA — histogram binning {#mca-binning}

Binning sets the MCA histogram energy-bin width. Larger factors combine more
raw energy codes: this can improve counts per bin but reduces displayed energy
detail. It differs from the client-side [PSD display bin count](#psd-bins).

Do not assume a bin number is keV. Energy calibration and its configuration
fingerprint must remain applicable after DSP or gain changes.

## MCA — input low-pass response {#mca-lowpass}

The low-pass preset selects the available FIR response, including the labelled
bandwidth presets or moving-average response. It can change pulse bandwidth,
noise, and timing shape before subsequent processing.

Recheck triggering and energy/PSD performance after changing it. The option
label is not a measurement of your detector's actual pulse bandwidth.

## MCA — CR-RC² shaping {#mca-crrc}

**C delay** and **F delay** set the C- and F-stage delays in nanoseconds.
**PZC coefficient** is the raw pole-zero correction coefficient for this filter,
not a time in nanoseconds. The current control's tooltip identifies the selected
parameter above this guide.

These settings change a shaped signal and may affect filtered triggering.
Inspect the appropriate debug output and keep the underlying decay and count-rate
conditions in mind. Do not copy the raw coefficient into the trapezoid pole-zero
time field; they are different parameterizations.

## MCA — constant-fraction discriminator {#mca-cfd}

**Enable CFD** enables the constant-fraction path. **Factor** sets the fraction
multiplier; **Delay** sets the delayed signal interval in nanoseconds.
**TW low/high** set the lower and upper time-walk boundaries in nanoseconds.

CFD aims to reduce amplitude-dependent timing shifts, but fine timing still
requires a supported record schema, valid interpolation, and calibration.
Inspect the CFD signal/window and qualified [coincidence timing](#coincidence-timing)
instead of assuming the enable checkbox alone guarantees fine resolution.

## MCA — trapezoidal shaping and energy {#mca-trapezoid}

**Enable Trapezoid** enables this energy-processing path.
**R (rise)** sets the rise time; **M** sets the flat-top time;
**Pole-zero time** sets the time constant; **Energy time** sets the energy-sampling
time. These time controls are in nanoseconds. **FT** selects the flat-top window
duration offered by the hardware.

The public IIO interface handles its register encoding; the GUI's pole-zero time
is not the CR-RC² raw PZC coefficient. Shaping affects noise, pulse overlap,
energy scale, and calibration. Inspect the trapezoid debug outputs and rebuild
or verify calibration after material changes.

## MCA — charge comparison {#mca-charge}

**Enable** activates charge comparison; **Time** sets its integration duration
in nanoseconds. In the current live PSD workflow, this supplies `charge_energy`,
used as the short-charge estimate against `trapezoid_energy`.

While enabled, a dashed black stepped timing guide appears on the MCA waveform
plot. The inverted lower step starts at the signal/frame beginning (0) and ends at CC time,
including pretrigger and clipped to Window length. The pretrigger/threshold marker
is not the short-window origin; moving it does not shift the guide. The upper
step marks the remaining diagnostic window.
It tracks timing edits, unit changes and zoom without changing waveform auto-range
or ADC data, and disappears when CC is disabled or the window length is zero.
Its vertical heights are schematic, not ADC amplitudes. It is not a
measured FPGA gate signal, and the remainder is not a qualified long-charge
integration interval; select the actual charge-comparison debug source to
inspect firmware-produced signals. The legend describes the two step levels.
Click the guide's legend sample to hide or show it without disabling hardware
CC. Your choice is retained across measurement refreshes, zoom, timing edits,
gate transfer, and CC off/on changes for the lifetime of that channel view.
The legend entry remains available while CC is enabled so you can show it again.

Live PSD uses `(Qlong - Qshort) / Qlong`; the exact producer quantities must be
validated for the setup. Enabling charge comparison is required for this live
ratio path. It is not the same function as [PSD zero crossing](#mca-zerocross),
and the two features should not be treated as one mutually exclusive selector.

## MCA — PSD zero crossing {#mca-zerocross}

This block enables hardware zero-crossing analysis. **Mode** selects hardware
mode 0 or 1; **Window low/high** set its timing boundaries in nanoseconds.
Interpret mode-specific results using the matching firmware/record schema;
the application does not infer a particle identity from the mode number.

Zero crossing produces timing information for a separate analysis path. It
does not replace the short-charge estimate needed by the live charge-comparison
PSD ratio. Stop list-mode DMA before changing either hardware block.

## MCA — Start, Stop, and Clear Spectrum {#mca-run}

Start begins histogram accumulation or the selected list-mode workflow. Stop
ends acquisition and, for DMA, starts final drain and cleanup. **Clear Spectrum**
clears the accumulation rather than changing the detector response.

Hardware histogram chunks are read sequentially and are not an atomic snapshot
while running. Stop for a stable final spectrum. Ordinary synchronized DSP
reconfiguration can restart accumulation and resets elapsed/presentation state.

## MCA — list-mode DMA and output format {#mca-dma}

**Record list-mode DMA** selects event streaming from the pulse processor.
**General Settings → MCA DMA output** chooses binary NDMA, HDF5, ROOT, or
online-only. Online-only intentionally creates no raw file; consumers can
analyze the stream without another DMA owner.

Live PSD is a bounded subscriber, not a competing reader. A raw capture retains
records independently of a display cut. Current IIO frames contain 1,024 opaque
16-byte records; sparse sources may update slowly until a frame is ready or Stop
drains the acquisition. Review the final status, fault counters, and continuity
summary before treating the capture as complete.

## MCA — ROI, counts, and differential view {#mca-roi}

The ROI selects an energy-bin interval for integrated counts and peak statistics.
Its coordinates are raw MCA channels unless explicitly shown on a calibrated
axis. Background, overlap, and binning can affect the apparent peak width.

Double-click the visible ROI to inspect integrated-count history against MCA
elapsed time. Moving the ROI or changing display modes starts a new history.
The View menu can show non-negative counts added since the previous refresh;
CSV exports and analysis snapshots still use the accumulated histogram.

## MCA — statistics and integrity {#mca-statistics}

Count rate, elapsed time, pile-up, overrange, energy-estimation, and throughput
indicators describe different aspects of acquisition. A zero in one counter
does not establish zero loss everywhere. An unavailable value is not a measured
zero and must not be replaced with an unrelated counter.

Dead time is shown in milliseconds and as a percentage of elapsed acquisition
time: `100 * dead_time_ms / (1000 * elapsed_s)`. The percentage uses the same
readback as the counters and shows **N/A** until elapsed time is positive.
Values above 100% are not clipped, so inconsistent counter snapshots remain visible.

For DMA, read the final summary and driver counters after Stop. Plot subscribers
can drop display batches without changing a separately recorded raw stream;
conversely, an attractive spectrum does not prove capture continuity.

## PSD — ratio and live event source {#psd}

Live PSD needs MCA list-mode DMA with charge comparison enabled. It displays
energy versus `(Qlong - Qshort) / Qlong`, using the event fields
`trapezoid_energy` and `charge_energy`. The horizontal value is raw producer
energy, not automatically calibrated keVee.

Offline waveform analysis calculates its own charge sums from saved samples;
event readback uses recorded event fields. These are related workflows, not
interchangeable settings. Inspect the [energy ROI](#psd-roi), [cut](#psd-cut),
and [FOM fit](#fom) before interpreting separation as particle discrimination.

## PSD — histogram resolution {#psd-bins}

Energy and ratio bins set the client-side matrix resolution. More bins give
finer sampling but fewer counts per bin and more memory/work. They do not
increase the ADC resolution or change hardware MCA energy binning.

A very sparse ratio projection can destabilize a double-Gaussian fit. Compare
fit diagnostics and sufficient statistics rather than selecting the highest
bin count automatically.

## PSD — energy shift {#psd-shift}

Energy shift right-shifts the raw 16-bit producer energy for display. A shift
of one bit divides integer codes by two with integer quantization. It does not
change the hardware energy calculation or establish a calibrated energy scale.

Recorded raw events are not rewritten by this display control. Recheck the
display ROI when changing the visible energy range or shift.

## PSD — axis ranges {#psd-range}

Ratio minimum/maximum define the accepted displayed ratio interval. The offline
Energy maximum sets the upper charge range of its reconstructed histogram.
These are raw analysis coordinates, not detector-specific calibrated limits.

Events outside the configured ranges do not contribute to the plotted matrix.
Check excluded-event counts before interpreting the displayed population balance.
A narrow axis range can hide a population rather than improve its separation.

## PSD — cut and separated spectra {#psd-cut}

The horizontal PSD cut divides events into above-cut and below-cut energy
projections. Dragging the line and editing its number are synchronized.
The cut classifies the displayed data; it does not reject events from the raw
MCA DMA capture.

Do not label the two groups neutron/gamma solely from their position. Detector,
source, gain, energy, and independent reference data determine that interpretation.
The cut is distinct from Gaussian centroid seeds and the fit ROI.

## PSD — energy selection and projections {#psd-roi}

The blue vertical region on the matrix selects the energy interval used to
build the ratio projection. The projection's shaded ratio interval and centroid
guides then define a separate [FOM fit](#fom-guides).

Broad energy intervals can mix populations with energy-dependent centroids and
widths, distorting a Gaussian FOM. Narrowing the energy ROI reduces this mixing
but also reduces counts. Compare the same energy interval when comparing gates
or detector settings; a different selected population can change FOM by itself.

## PSD — figure of merit {#fom}

**Fit** fits two ordered Gaussian populations plus a constant
background to the selected ratio projection using Poisson-deviance residuals.
It reports centroids, FWHMs, and
`FOM = |mu2 - mu1| / (FWHM1 + FWHM2)`.

A larger FOM indicates greater fitted separation relative to the fitted widths
for those selected events. It is not a classification efficiency, neutron
purity, or a universal detector score. Energy selection, baseline, gates,
histogram resolution, and sample size all matter.

Inspect [fit quality](#fom-quality) before accepting the number.
**Copy table** copies the fit snapshot and diagnostics for comparison.

## PSD — fit ROI and centroid guides {#fom-guides}

The shaded fit region selects the ratio interval used by the Gaussian fit.
Place the two centroid guides near the two candidate peaks, with enough of
each distribution inside the fit ROI to constrain its width and background.

Guides are starting estimates, not imposed population means. Hiding draggable
guides changes their visibility, not the underlying physics or a hardware gate.
These are ratio coordinates, distinct from the energy ROI and waveform time gates.

## PSD — convergence and fit-quality warnings {#fom-quality}

A solver returning a number is not sufficient. Inspect convergence, covariance,
parameter bounds, count statistics, population significance, fitted widths,
separation, and reduced Poisson deviance. The application reports suspect fits
and gate optimization rejects fits that fail its trust checks.

Reduced deviance measures model agreement relative to the fitted degrees of
freedom. A large value can expose skew, background mismatch, or extra populations.
A large FOM from a poor fit is not a qualified detector-performance result.
Changing the ROI until a warning disappears is not independent validation.

## Waveforms — offline reconstruction {#waveform}

The workbench's **File**, **Analysis**, **Selection**, **View**, and **Help** menus
provide the same actions as its visible controls, with synchronized enabled and
checked states. Selection menu actions also work when the cluster toolbar is
hidden. Window-local shortcuts: **Ctrl+O** opens a waveform file, **Ctrl+R**
reconstructs PSD, **Ctrl+S** downloads the report, **Ctrl+Shift+W** closes the
file, and **Ctrl+W** closes the workbench. **Help → Workbench guide** opens this
bundled offline reference. Existing F1 click-to-explain mode remains unchanged.
The source filename (or "No file loaded") appears in the window title instead of
a separate label row, leaving more height for plots. Closing the file resets the
title. The full source path is retained as window file metadata.
Open/close file commands are in the File menu without duplicate buttons. Source,
Frame, and Sample period occupy the Waveform source group at the top of the left
sidebar. Scroll the sidebar to reach lower settings on smaller windows.
Long waveform/PSD tooltips wrap to multiple short lines. Full
explanations are retained in plain text for both hover hints and F1 help.

Open a saved waveform capture, select its source, inspect baseline and polarity,
then set integration start, short-gate end, and long-gate end. **Reconstruct PSD**
integrates the configured event selection in a background worker.

The operation reads the capture without changing the file or live digitizer
settings. Cancel requests worker termination; wait for completion before starting
another job. Use the frame preview to check representative pulses, not just one
especially clean event.

## Waveforms — transfer candidate gates to MCA {#waveform-mca-transfer}

After exploring a representative Scope capture, choose **File → Transfer gates
to MCA...**. Select the target channel explicitly. This is a starting preset to
reduce repeated measurements, not a promise of the same online PSD/FOM.

Pretrigger is derived automatically from the **right edge of the baseline region
to the green integration-start line**: `(integration start − baseline end) ×
sample period`. The baseline end is the region's end boundary (exclusive), not
the final included sample or the width of the baseline region. A baseline end
coincident with integration start gives zero pretrigger. No manual trigger
position or trigger-reference confirmation is needed. Source sample indices are
converted using the workbench's sample period, never copied directly as hardware
register counts.

The proposed mapping is pretrigger = integration start minus baseline end;
window length = long-gate end minus integration start; charge-comparison time =
short-gate end minus integration start. All three round upward to the supported
step (currently 2 ns). Out-of-range values and invalid baseline/gate ordering
are rejected. Inspect current/requested/rounded values and
the dashed green/yellow/red candidate boundaries on the waveform. The underlying
offline gates and analysis are unchanged. The preview anchors at baseline end,
then adds the rounded pretrigger gap to place the candidate green boundary;
short/long durations remain relative to that candidate integration start.
No FOM is recalculated for boundaries
that may lie between recorded samples. Choose whether charge comparison should
be enabled; no baseline/polarity/trigger/filter settings are transferred.
Disconnected/offline use still permits a rounded candidate preview, with Apply
disabled and current hardware values shown as unavailable.

**Apply** asks for final confirmation and requires an idle connected target.
It does not start/stop acquisition. Stop Scope on that channel and stop/drain/
close MCA DMA first. Offline/replay mode and targets invalidated by reconnect
cannot be written. Failed sequential writes attempt to restore prior timings;
warnings report failure and incomplete rollback, if any. Current configuration
values are driver shadows, not independent FPGA readback. Window/integration
equivalence remains unqualified; baseline handling and hardware energy selection
can differ. Timing encoding/ranges are source-derived from PetaLinux `167e90f`;
local tests use simulated backends, not live hardware validation.

## Waveforms — selecting clusters {#waveform-clusters}

After **Reconstruct PSD**, add a **Rectangle** or **Polygon** over the reconstructed
matrix. Move/resize the rectangle or drag polygon vertices; click a polygon edge
to insert another vertex. **Apply selection** selects the exact event coordinates
inside the region, including its boundary—not whole histogram bins.

The rectangle/polygon outline and handles are black, including on hover.
**Show PSD ROI** toggles the matrix's energy ROI and PSD cut line without changing
their values or projections. **Waveform selection tools** shows/hides the cluster
controls, black region, and current-event marker; the active filter and selected
match number are preserved. Show the tools again to resume filtered navigation.
FOM **Fit guides** retain their independent visibility switch.

**Previous**, **Next**, and the selected-event number browse only matching
waveforms. **Frame** still shows the original, zero-based source frame index and
is read-only while filtering. A white point marks that event on the PSD map.
**Clear selection** restores ordinary frame browsing. An empty selection clears
the signal preview rather than leaving an unrelated waveform on screen.

Selection includes only events represented in that reconstruction: its source,
stride, maximum-event limit, and energy/ratio display ranges still apply. A sparse
preview therefore cannot select every event in the file. Reconstruction retains
three scalar columns per plotted event (frame, charge, PSD); waveform samples
remain file-backed. Settings/source changes and new reconstructions clear the
selection to prevent stale frame associations. Auto polarity is held at the
reconstruction's resolved polarity while browsing a selected cluster.

**Show average waveform** overlays the entire selected cluster's arithmetic mean
on the current waveform plot. Each waveform's chosen mean/median baseline is
removed before averaging and the reconstruction polarity is applied. For visual
comparison, the mean is placed at the displayed individual pulse's baseline.
There is no peak alignment or amplitude scaling: sample indices and amplitude
differences are preserved, so timing jitter can broaden the average. For varying
record lengths only the shortest shared prefix is shown, with every selected
event contributing to every sample. Calculation runs in the background with
bounded memory. Click the button again to cancel/hide; browsing matching events
keeps the overlay, while changing/clearing the selection or reconstruction
settings removes it. Averaging does not alter the PSD, FOM, or source file.

**File → Export selected cluster...** saves every selected original waveform,
not the average or just the current match. It is available only for a nonempty,
current applied selection. Source, reconstruction stride/limit and display cuts
still apply. The samples are full stored ADC arrays with original uint64
timestamps; no baseline/polarity processing, alignment, or gate cropping is
applied. Export uses a cancellable background task and never overwrites files.
Failure/cancellation removes only the incomplete files created by that export.

- **nlab binary (.bin)**: standard Scope NDMA v1, reopenable in the workbench.
  Keep the accompanying **.cluster.json** file: it records original frame IDs,
  energy/PSD coordinates, selection vertices, gates and analysis/source metadata.
- **ROOT (.root)**: `waveforms` TTree with `samples`, `timestamp`, `source_frame`,
  reconstructed `energy` (long charge), and `psd`; provenance is in `metadata_json`
  and the field description is in `waveform_schema`.
- **HDF5 (.h5/.hdf5)**: `waveforms`, `timestamps`, `source_frame`, `energy`, and
  `psd` datasets, with a `metadata_json` file attribute.

CAEN sources support ROOT/HDF5, retaining variable-length samples and unconverted
source timestamps, plus available stored gates, flags and waveform codes with
validity columns. Scope NDMA cannot losslessly represent CAEN record geometry
and timestamp semantics, so nlab binary export is available only for NDMA input.
The saved sample period is the workbench's analysis value; NDMA v1 does not encode
the actual sampling rate. ROOT/HDF5 are analysis outputs; reopen the native binary
subset for further workbench browsing.

These regions are exploratory geometric cuts, not automatic neutron, gamma,
alpha, or purity labels. Identifying populations requires suitable calibration
and independent source/reference measurements.

## Waveforms — source, frame, and sample period {#waveform-source}

Source selects the recorded channel. Frame is the event index used for inspection,
not a cut selecting only that event for reconstruction. Sample period converts
sample indices into nanoseconds on the plot.

Use the recorded sample period when available, or the verified acquisition value
for a format without it. Guessing the period gives incorrect physical gate times.
Stored gate coordinates remain sample indices; not every binary format stores
the same timestamp or waveform layout.

## Waveforms — baseline estimator and pulse polarity {#waveform-baseline}

The blue region selects quiet baseline samples before the pulse. Median is
more robust to isolated outliers; mean averages all selected values. Neither
can recover a correct baseline if the selected region contains a substantial
pulse or an unmodelled drift.

Polarity normalizes the baseline-subtracted pulse so the integration is oriented
consistently. “Auto from displayed frame” infers it from the inspected frame;
explicit polarity is safer when representative pulses have a known sign.
This is offline preprocessing, not an analogue DAC or FPGA register change.

## Waveforms — integration gates {#waveform-gates}

Green is integration start, yellow is short-gate end, and red is long-gate end.
The short and long sums share the start; each end is an exclusive sample boundary.
The PSD ratio is `(Qlong - Qshort) / Qlong`.

Keep baseline and gates in valid order and retain the pulse tail needed by your
detector. Too short a long gate loses light; an unnecessarily long one adds
noise and can include subsequent pulses. These markers do not reconfigure live
MCA charge comparison or hardware zero crossing.

The stored and polarity-normalized traces use sample-hold steps, preserving
their original sample times and values without diagonal interpolation.

## Waveforms — stride and maximum events {#waveform-sampling}

Event stride selects every Nth recorded event for analysis. Maximum events
limits the selected sample; **All** means no explicit user count limit.
These controls reduce calculation time but change the statistics represented
by the reconstructed matrix.

A sequential or strided selection is not guaranteed to be a random sample of
a drifting measurement. Reports preserve the selection settings. The optimization
event limit is a separate setting used to compare candidate gates.

## Waveforms — automatic preview {#waveform-preview}

With **Auto preview** enabled, dragging a baseline or gate marker schedules
a debounced sparse reconstruction, using every twentieth selected event.
Releasing the marker queues the normal calculation with your configured stride
and maximum events.

Sparse preview FOM and counts need not match the completed full reconstruction.
Wait for the final result before exporting or comparing measurements. Disable
auto-recalculation to adjust markers first and reconstruct manually afterward.

## Waveforms — FOM gate search {#waveform-optimization}

Optimization evaluates discrete short/long gate endpoints near the current
markers, keeping baseline, integration start, polarity, energy/fit ROIs, and
centroid seeds fixed. Optimization events bounds the sample used to compare
candidates; a working-memory limit may reduce it further.

Gate search radius is measured in samples on each side of each endpoint.
The default local coarse-to-fine search is faster but not an exhaustive optimum
guarantee. **Exhaustive search** tests every valid pair in that bounded
region at one-sample spacing and warns about the candidate count and time.
Cancel is checked between fits; poor-quality candidates are rejected.

The winning gates are reconstructed and refitted on the normal selected dataset.
Its FOM can differ from the sampled search result. The search does not demonstrate
out-of-sample performance or a globally optimal detector configuration.

## Current — monitoring overview {#current}

Current monitoring starts stopped. Choose the acquisition mode, verify the
signal-specific calibration, then Start monitor. Stop releases its worker;
the numeric display and plot are rate-limited independently of acquisition.

Raw values are not amperes until the appropriate zero, scale, and units have
been established. IIR and DMA expose different signals and need separate
qualification. See [mode](#current-mode) and [calibration](#current-calibration).

## Current — IIR versus Scope DMA {#current-mode}

IIR polling reads signed FPGA input-filter output for lightweight monitoring.
Its host polling schedule is not a hardware-time sample grid. Scope DMA reads
complete timestamped waveforms and computes sample-weighted statistics.

DMA owns the Scope buffer and applies its qualified periodic preset. The stopped
view can change the DMA frame gap or Reset Scope defaults afterward. Do not start
another Scope DMA owner on the same channel. The modes' raw scales differ.

## Current — zero, scale, and units {#current-calibration}

The saved linear calibration is `(raw - zero) * scale`. Zero removes a measured
offset; scale converts the remaining signal to the selected physical unit.
**Set zero from recent samples** derives an offset from recent data, so use a
known zero-current condition rather than an active detector signal.

Changing the unit label alone is not a calibration. Validate sensor/front-end
gain and polarity for the selected IIR or waveform-DMA mode.

## Coincidence — synchronized two-channel acquisition {#coincidence}

Coincidence owns both selected MCA list streams and shared start on one physical
digitizer. Both channels must be idle and eligible. An independent digitizer in
the same application is not a synchronized coincidence partner.

Energy gates and event logic select analysis events; raw DMA recording retains
the underlying streams. Stop drains both workers before handing ownership back.
Review [logic](#coincidence-logic), [timing](#coincidence-timing), and
[matrix subtraction](#coincidence-matrix) before interpreting a pair count.

## Coincidence — energy gates and Boolean logic {#coincidence-logic}

Use CH0/CH1 MCA ROI applies the corresponding raw-channel energy gate; a hidden
ROI means the full energy range. AND, OR, XOR, and per-channel NOT/veto options
define the event-selection expression shown in the panel.

These are software analysis rules, not retrospective edits to raw files. The
two-dimensional energy matrix is available for the ordinary CH0 AND CH1 path;
different logic is not automatically equivalent to that matrix interpretation.

## Coincidence — timing window and offset {#coincidence-timing}

Lower/Upper Δt select the accepted delay interval. The CH1−CH0 calibration offset
is measured with a simultaneous common input and subtracted from the raw delay.
A cable or electronics change can invalidate it.

Coarse timing and CFD fine timing use different qualifications. Fine display
bins do not prove that the detector, interpolation, or calibration achieves that
resolution. Use the record schema matched to the firmware and inspect valid
timing counts and fit diagnostics. Keep the lower boundary below the upper one.

## Coincidence — energy matrix and random sidebands {#coincidence-matrix}

The matrix relates accepted CH0 and CH1 energies. Prompt shows the signal window;
Delayed random shows the sidebands; Prompt − scaled random subtracts their
width-normalized estimate. Corrected bins can be negative and are not ordinary
Poisson counts.

Random sidebands enables two delayed windows, each the prompt-window width.
Sideband gap separates each delayed window from the prompt gate. Choose a gap
that avoids genuine correlated structure and check the random-background
assumption. Linear/log scale changes display only. Export preserves matrix
data and associated analysis information.

## Global — per-board settings {#global}

Global settings and diagnostics belong to one physical board, even in a
multi-digitizer workspace. Select the intended source before changing shared
trigger or temperature behavior. Per-board diagnostics are not per-channel
measurements.

See [software start](#global-sync) and [temperature correction](#global-temperature).

## Global — synchronized software start {#global-sync}

The shared source and enable controls select the board's MCA synchronization
path. Prepare Software Sync establishes the software-start arrangement;
Start Armed MCA Channels releases eligible armed channels. Reset Software Level
resets that shared level.

Arming and starting are different states. A channel waiting for a shared trigger
can appear inactive yet still own the core. Coincidence manages this lifecycle
itself; do not independently change its shared-start configuration mid-run.
Software start does not synchronize independent boards' clocks.

## Global — shared temperature correction {#global-temperature}

Coefficient and offset configure the board-wide MCA temperature correction;
Enabled and its update interval control its use. They are not the same as the
PSU bias-compensation slope and reference temperature.

Use a validated calibration and check the correction's source and diagnostics.
Do not assume a correction fitted for one detector or gain setting applies to
another. Hardware availability and units are shown by the supported controls.

## PSU — monitoring and availability {#psu}

PSU exposes the discovered SiPM/HV converters, temperature readings, and bias
compensation. Controls can be absent or disabled when the relevant device is not
available. A blank or unavailable reading is not a measured zero.

Observe detector ratings and safe operating procedures. Help does not enable a
supply, stop acquisition, or place the apparatus in a safe state. See
[bias settings](#psu-bias) and [compensation](#psu-compensation).

## PSU — bias voltage and enable {#psu-bias}

The SiPM and HV voltage settings control their respective discovered supplies;
the SiPM enable control switches its supported bias path. Distinguish a requested
setpoint from measured voltage and current telemetry.

The HV setpoint range is **100–1250 V**. Entering zero selects **100 V** when
you press Enter or move focus to another control.
The ADC voltage continues to show the measured output independently.

Use the displayed units and hardware limits, plus the detector's own rating.
Changing bias changes detector gain and can invalidate energy/PSD calibration.
Never infer a safe voltage from an example configuration or an offline replay.

## PSU — temperature compensation {#psu-compensation}

Mode selects the available compensation behavior. Ct specifies the bias
temperature coefficient and Tref the reference temperature, using the control's
displayed units. SiPM and HV compensation belong to separate supply paths.

Apply coefficients measured for the actual detector and check the temperature
source and resulting voltage. This compensates bias response, not arbitrary
offline spectrum drift. Observe hardware and detector voltage limits.

## External instruments — Modbus settings and telemetry {#external}

External instruments are discovered through the configured ser2net/Modbus bridge.
Settings and telemetry rows come from the instrument schema: consult the row's
description, unit, scale, range, and access restrictions rather than assuming
every value is an unscaled register.

Monitoring refresh and plot history affect polling/presentation. Password-
protected or read-only rows cannot be treated as ordinary writable settings.
Recorded replay modules are read-only and do not represent a live connection.

## Analysis — energy calibration {#energy-calibration}

Commands are in **File**, **Analysis**, **View**, and **Help** menus rather than
button rows. **File** replaces/overlays the current spectrum, downloads reports
(**Ctrl+S**), and closes (**Ctrl+W**). **Analysis** adds/removes/clears reference
points and applies or clears calibration. Model/width controls and the table
remain beside the plots; point edits still refit automatically. **View** resets
plot zoom. **Help → Workspace guide...** opens this bundled reference. Closing
still asks before discarding unapplied edits, and Apply/Report retain fit-validity
and settings-compatibility checks.

Load a spectrum snapshot, identify reference peaks, enter known energies, and
fit the selected calibration model. Inspect residuals and enabled calibration
points before applying the result. Reference lines aid identification but do
not establish peak identity by themselves.

Calibration maps raw channel to energy for the associated configuration
fingerprint. Gain, shaping, or detector changes may invalidate it. Overlaying
spectra is not permission to combine incompatible configurations. The status-bar
energy-calibration indicator opens this workspace on double-click.

## Analysis — dose calibration logger {#dose-calibration}

Record response while an independently known reference dose rate is applied.
The tool associates rate and device telemetry with that reference and records
the selected analysis context. It is a calibration logger, not an independent
dosimetry standard.

Check source geometry, background, stability, duration, and detector response.
Do not interpret a fitted conversion outside its qualified energy, dose-rate,
or hardware range. Exported reports preserve context for review.

## Analysis — MCA peak fitting {#peak-analysis}

The **File** menu loads spectra (**Ctrl+O**), exports spectra/fit results, downloads
reports (**Ctrl+S**), and closes the workspace (**Ctrl+W**). **Analysis** refreshes
or removes spectra, suggests/fits peaks, cancels fits, and creates derived spectra
using the visible Operations settings. **References** adds/removes the highlighted
nuclide from favourites, restores defaults, or clears selected reference lines.
**View** controls linear/log counts, selects a side-panel tab, and resets zoom;
the corresponding visible controls stay synchronized. **Help** opens this guide.
Command-button rows have been removed; parameter controls remain beside plots.
Fit/export availability and background fit cancellation behave as before.

Select a spectrum and region, choose the peak/background model, fit, and inspect
residuals, uncertainty, and convergence. Net area, centroid, and width depend on
background and overlapping peaks; a visually plausible fit can still be biased.

Energy calibration determines physical energy coordinates. Reference lines are
identification aids, not automatic proof of an isotope. Export the fit snapshot
and diagnostics when reporting peak measurements.

## Analysis — PSD event readback {#psd-readback}

The **File** menu contains **Load event file...**, **Cancel loading**,
**Download report...**, and **Close workbench**. **Analysis** provides Gaussian
fitting and fit-table copy. **View** toggles PSD ROI and FOM guides, synchronized
with the plot controls; **Help** opens this bundled guide. Window-local shortcuts
are **Ctrl+O** (load), **Ctrl+S** (report), and **Ctrl+W** (close).
The current filename (or "No file loaded") is shown in the window title, with no
separate file-label row. Histogram spinners and matrix
scale/PSD-cut/ROI controls are in a compact left sidebar beside the full-height
plots; drag the divider to resize it. There is no separate file-action toolbar.
Progress/errors stay at the
bottom. Loading disables the load/settings controls and enables Cancel loading;
report download remains available only after loading and a completed FOM fit.

Readback reconstructs energy-versus-ratio distributions from recorded MCA event
fields. Select the file/source, histogram ranges, and event limits, then inspect
excluded counts, fit ROIs, and diagnostics.

It does not reintegrate waveform samples or rewrite hardware integration gates.
Use the waveform workbench when testing different offline time gates. Raw values
require format- and detector-specific interpretation before physical calibration.

## Analysis — timing validation {#timing-validation}

Timing validation inspects recorded event timing and supported producer schemas.
Use it to check sample/time interpretation and integrity before claiming fine
resolution or coincidence performance.

An apparent narrow distribution does not by itself validate timestamp encoding,
wrap handling, a common clock, or channel-offset calibration. Reports must
distinguish synthetic checks, source-derived behavior, and real hardware results.

## Files — conversion to HDF5 or ROOT {#conversion}

The File menu converts supported binary waveform or list-mode captures to HDF5
or ROOT. Select a completed source and a new destination. Conversion runs in a
background worker with cancellation and visible error reporting.

It preserves raw quantities and available metadata; it does not calibrate energy,
calculate particle identity, or repair incomplete records. Partial or changing
captures are rejected. ROOT conversion produces TTrees for supported event and
waveform layouts; inspect their branches before writing an analysis macro.

## Files — reports, CSV, and destinations {#export}

CSV exports and analysis reports are snapshots with their associated settings.
The default DMA folder controls new capture destinations, not where an already
open source file was recorded. Select a new report/output name to avoid replacing
prior results.

PSD report bundles include plots, fit inputs/components, projections, diagnostics,
and provenance. Check whether the result represents sparse preview, selected
events, or completed full reconstruction. Exports cannot supply a missing
physical calibration or prove raw-capture continuity on their own.

## Settings — saved configuration and restore {#settings}

Save/Load Settings serializes application and device configuration. General
Settings controls display defaults, polling, output destinations/formats, and
optional automatic configuration restore.

Loading settings can write hardware and is subject to acquisition ownership.
Review the selected file and current board rather than assuming a saved setup
is safe or compatible. Automatic restore is distinct from session replay and
does not turn a captured analysis result into a live measurement.

## Sessions — recording, replay, and offline use {#session}

Session recording preserves supported readbacks, settings, and references for
later review. Replay displays recorded information without controlling physical
hardware; hardware controls are interlocked.

Recorded channels and external modules expose only what the session retained.
Do not treat absent telemetry as zero or a replay trace as live validation.
Standalone offline analysis can also read captures without opening a connection.

## About — version, updates, and licenses {#about}

About shows the installed application version and project link. Check for Updates
contacts GitHub for the latest published stable release and reports comparison
or connection errors; it does not silently install an update.

Third-Party Licenses lists bundled dependency notices. Offline help is packaged
with this application version and does not require an update check or network.
