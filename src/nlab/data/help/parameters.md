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
[PSD](#psd), or [offline waveform analysis](#waveform). To read recordings in
your own code, see the binary formats of [Scope waveforms](#format-waveforms),
[MCA list mode](#format-listmode) and [Current sums](#format-current), each
with an example Python reader.

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

If the network connection to a board is lost (for example after a board reboot
or a cable fault), the Scope live view, MCA polling and list-mode DMA, PSU
monitoring and the Global loops stop instead of retrying a connection that
cannot recover. One **Connection Lost** message then offers **Reconnect**, which
replaces every board connection like **Connection → Reconnect Device** but
without asking again; if that fails, the message offers another attempt.
**Not Now** leaves everything stopped; reconnect later from the Connection menu.
A board that rebooted may have lost its configuration, so check the settings
before measuring again.

See [session replay](#session) and [acquisition ownership](#ownership).

## Acquisition — ownership and safe configuration {#ownership}

Each DMA path has one owner. Current waveform DMA uses the Scope path;
Coincidence owns both selected MCA streams. Stop the current owner and allow
its worker to drain and release the buffer before starting a conflicting task.

Stop and close list-mode DMA before configuring its measurement. During ordinary
non-DMA histogram acquisition, MCA controls remain editable, including ABI2.
Committing a measurement setting automatically stops acquisition, applies the
setting, then starts a fresh accumulation with reset counts and elapsed time.
Failed writes stay stopped. DMA locks settings throughout arming and draining,
until the buffer closes. The DMA recording checkbox stays fixed during a run.
Debug selector changes remain live without resetting accumulation. A disabled
control may indicate an ownership, replay, or connection-loss interlock.

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

**Mode** uses the triggers advertised by the connected Scope. Level and edge
modes capture individual waveforms. **Periodic** captures a frame at a fixed
interval set by [Periodic gap](#scope-gap), using the IP122+ mean4 waveform format
where supported. Structured [Periodic sums](#scope-sums) on IP123/IP124 are not
waveforms and are available only in [Current](#current-mode).

**Level (raw)** is an ADC-code threshold, not volts or the MCA threshold.
A negative-going pulse has a falling leading edge. Select a crossing that is
separated from baseline noise. Periodic modes do not select threshold crossings.

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

**Periodic gap** sets the gap between frames in **Periodic** mode, in nanoseconds
(8 ns steps); the frame period is frame length plus gap. It is enabled only in
Periodic mode on supported hardware. It is not an inter-event time measured from
the detector and does not make pulse-triggered events uniformly spaced. Current
DMA applies its own [qualified Scope preset](#current-mode).

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
cannot simultaneously own the same Scope buffer. The file layout and an example
reader are in [Scope waveform binary format](#format-waveforms).

## Acquisition — duration {#duration}

Duration is in seconds. **0 means run until stopped**, not a zero-length run.
A nonzero value requests a timed acquisition. Network and drain completion can
make the final GUI stop later than the nominal limit.

The field does not change the waveform sample period, trigger threshold, or
histogram polling frequency. Wait for recording completion before treating
an output file as final.

## Acquisition — measurement series {#series}

**Settings → Enable Series** enables **Repeats** in Scope, MCA and Current. One
Start then performs that many runs, each lasting the selected duration. Every run
is an ordinary measurement: the next one starts only after the previous run has
stopped and released the board, including the complete drain and close of a DMA
recording. More than one run needs a duration above 0. **Stop**, an error, a lost
connection, session replay or a coincidence session ends the series; a failed run
is never repeated automatically. Unchecking the option makes the run in progress
the last one.

While the option is checked, recordings share one name followed by `_001`,
`_002`, …: the file chosen with **DMA File…** (or typed for Current sums), or the
generated default. An existing file is never replaced; a name already in use gains
a further `_NNN`. With **Record list-mode DMA** checked, every MCA run writes its
own list-mode file, closed and complete before the next run arms. Each MCA run
also saves its spectrum as a CSV with the same name, in the **Export CSV** format,
read again after the run stopped. If the stopped
spectrum cannot be read, the last live readback is saved and marked as such.
A run ended with Stop keeps its recording and spectrum.

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

**General Settings → Appearance → Waveforms** draws sampled waveforms (Scope
trace, MCA debug traces, Current raw pulses and Waveform Analysis) either as
**Steps**, holding each sample until the next one (the default), or as
**Lines** joining consecutive samples directly. It applies immediately and
changes only the drawing: the samples, cursors and recorded data are the same.
The Scope persistence image and the MCA charge-comparison gate guide always use
steps.

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

**References** — Digital pulse processing for spectroscopy.

- Knoll, G. F. (2010). *Radiation Detection and Measurement*, 4th ed. Wiley. ISBN 978-0-470-13148-0
- Nakhostin, M. (2017). *Signal Processing for Radiation Detectors*. Wiley. doi:10.1002/9781119410225

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

MCA timing editors and sliders use **nanoseconds**. **Window length** is
8..8192 ns and **Pretrigger offset** is 0..4088 ns, both in 8 ns steps.
The GUI converts these durations to native 2 ns ADC samples at the API
boundary. Saved hardware settings retain the ABI2 `adc_samples` contract.
The debug plot uses physical time and one displayed point per 8 ns.

The green line shows the configured offset. Different filter/logic paths can
have different latency; it does not prove the analogue threshold-crossing time.
The driver does not impose a universal pretrigger<=window rule. The provisional
offline gate-transfer workflow checks that relationship for its own mapping.

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

The MCA input FIR low-pass response is fixed at **70 MHz**. The application
sets it on every channel during initialization, including after reconnecting.
There is no filter selector; loading settings also keeps 70 MHz even if the
saved document requests another preset.

The response affects noise and pulse shape before subsequent processing.
70 MHz names the filter preset, not a measurement of your detector's actual
pulse bandwidth. Saved configurations and calibration fingerprints record
this fixed response; calibrations made with another response may be stale.

## MCA — CR-RC² shaping {#mca-crrc}

**C delay** is an exponent e=0..6. Its averaging window is $8\times 2^e$ ns:
selector 3 means 64 ns. **F delay** is 8..1016 ns in 8 ns steps.
**PZC coefficient** is a signed raw pole-zero correction coefficient.

CR-RC2 normalizes its moving average by the selected $2^e$ beats in HLS.
The current driver does not update or export a separate editable **1/C**
coefficient, so that control is hidden. An explicitly advertised optional
raw attribute retains unsigned 16-bit editing and persistence, with no
assumed fixed-point scale.

Scope observes the raw ADC branch. MCA sees temperature correction, the optional
DC blocker, and FIR shaping. **Bypass DC blocker** maps to disable=1; unchecked
maps to disable=0. The MCA FIR response remains fixed at 70 MHz.

**References** — Recursive digital CR-(RC)ⁿ shaping.

- Nakhostin, M. (2011). Recursive algorithms for real-time digital CR-(RC)ⁿ pulse shaping. *IEEE Trans. Nucl. Sci.* 58, 2378–2381. doi:10.1109/TNS.2011.2164556

## MCA — constant-fraction discriminator {#mca-cfd}

The constant-fraction multiplier is raw/32768 with raw=0..32767: its maximum
is 0.999969482421875. **Delay** is 10..264 ns in 2 ns steps.
The GUI passes 5..132 native ADC samples to the API; the kernel owns
the phase encoding, so the application does not encode it again.

The inclusive low/high boundaries use ns in 8 ns steps. When enabled,
low<=high<window length and delay<window length. A window of low=24 ns and
high=48 ns admits four beats, or 16 original ADC positions.

Start the window at the trigger: **low must be at least the pretrigger**. The
window counts from the first beat of the frame, including the pretrigger
history, but the fine time counts from the trigger, so a crossing inside the
pretrigger history is reported as a false offset 0. The default window
24-88 ns starts at the default 24 ns pretrigger. The Low field turns orange
when CFD is enabled and the window starts earlier; the hardware does not
enforce the rule.

**References** — Constant-fraction timing, analogue and digital.

- Gedcke, D. A., McDonald, W. J. (1968). Design of the constant fraction of pulse height trigger for optimum time resolution. *Nucl. Instrum. Methods* 58, 253–260. doi:10.1016/0029-554X(68)90473-4
- Fallu-Labruyere, A., Tan, H., Hennig, W., Warburton, W. K. (2007). Time resolution studies using digital constant fraction discrimination. *Nucl. Instrum. Methods Phys. Res. A* 579, 247–251. doi:10.1016/j.nima.2007.04.048

## MCA — trapezoidal shaping and energy {#mca-trapezoid}

**R**, **M**, and **Energy time** use nanoseconds. R is 16..4088 ns,
M is 0..4088 ns, and the averaging start is 0..16376 ns, all in 8 ns steps.
**FT** displays 8..512 ns (hardware exponent 0..6). With the algorithm
enabled, the start plus the FT window must fit within the pulse length.

**Beta (raw)** is an unsigned 32-bit coefficient. The current contract does
not establish the previous editable exponential time-law adapter.

The driver writes **1/R** automatically whenever R changes:
$1/R_{raw}=\lfloor32768/(R_{samples}/4)\rfloor$. GUI R=32 ns
(16 samples) produces raw 8192. This derived coefficient has no separate
panel readout or editor; optional raw readback remains available through
the hardware API.

Shaping changes noise, pileup behavior, and energy calibration. Recheck a
calibration after material changes; saved fingerprints include ABI and units.

**References** — Digital trapezoidal shaping.

- Jordanov, V. T., Knoll, G. F. (1994). Digital synthesis of pulse shapes in real time for high resolution radiation spectroscopy. *Nucl. Instrum. Methods Phys. Res. A* 345, 337–345. doi:10.1016/0168-9002(94)91011-1
- Jordanov, V. T., Knoll, G. F., Huber, A. C., Pantazis, J. A. (1994). Digital techniques for real-time pulse shaping in radiation measurements. *Nucl. Instrum. Methods Phys. Res. A* 353, 261–264. doi:10.1016/0168-9002(94)91652-7

## MCA — charge comparison {#mca-charge}

**Charge comparison time** is 8..65536 ns in 8 ns steps. The API receives
native ADC samples. When enabled the time must not exceed the pulse length. The guide on
the debug plot uses ns and is schematic, not a measured FPGA integration gate.

Stop acquisition before changing the measurement. Offline integration and the
hardware estimator are separate algorithms; transferring gates does not qualify
their equivalence.

**References** — Charge comparison and its comparison with zero crossing.

- Brooks, F. D. (1959). A scintillation counter with neutron and gamma-ray discriminators. *Nucl. Instrum. Methods* 4, 151–163. doi:10.1016/0029-554X(59)90067-9
- Moszyński, M., Bizard, G., Costa, G. J., et al. (1992). Study of n-γ discrimination by digital charge comparison method for a large volume liquid scintillator. *Nucl. Instrum. Methods Phys. Res. A* 317, 262–272. doi:10.1016/0168-9002(92)90617-D
- Szczęśniak, T., Grodzicka, M., Moszyński, M., et al. (2014). Digital neutron-gamma discrimination methods: charge comparison versus zero-crossing. *IEEE NSS/MIC 2014*, 1–4. doi:10.1109/NSSMIC.2014.7431222

## MCA — PSD zero crossing {#mca-zerocross}

This block enables hardware zero-crossing analysis. **Mode** selects hardware
mode 0 or 1; **Window low/high** are inclusive positions in nanoseconds,
with 8 ns steps and low<=high<frame duration when enabled.
Interpret mode-specific results using the matching firmware/record schema;
the application does not infer a particle identity from the mode number.

Zero crossing produces timing information for a separate analysis path. It
does not replace the short-charge estimate needed by the live charge-comparison
PSD ratio. Stop list-mode DMA before changing either hardware block.

**References** — Zero-crossing pulse shape discrimination.

- Alexander, T. K., Goulding, F. S. (1961). An amplitude-insensitive system that distinguishes pulses of different shapes. *Nucl. Instrum. Methods* 13, 244–246. doi:10.1016/0029-554X(61)90198-7
- Nakhostin, M., Walker, P. M. (2010). Application of digital zero-crossing technique for neutron–gamma discrimination in liquid organic scintillation detectors. *Nucl. Instrum. Methods Phys. Res. A* 621, 498–501. doi:10.1016/j.nima.2010.06.252
- Szczęśniak, T., Grodzicka, M., Moszyński, M., et al. (2014). Digital neutron-gamma discrimination methods: charge comparison versus zero-crossing. *IEEE NSS/MIC 2014*, 1–4. doi:10.1109/NSSMIC.2014.7431222

## MCA — Start, Stop, and Clear Spectrum {#mca-run}

Start begins histogram accumulation or the selected list-mode workflow. Stop
ends acquisition and, for DMA, starts final drain and cleanup. **Clear Spectrum**
clears the accumulation rather than changing the detector response.

Hardware histogram chunks are read sequentially and are not an atomic snapshot
while running. STOP releases acquisition ownership but the FPGA provides no
A07 downstream-completion acknowledgement, so the application cannot promise a
final coherent snapshot. Failed settings stay stopped; inspect them before START.

## MCA — list-mode DMA and output format {#mca-dma}

**Record list-mode DMA** selects event streaming from the pulse processor.
**General Settings → Acquisition files → MCA output** chooses binary NDMA, HDF5, ROOT, or
online-only. Online-only intentionally creates no raw file; consumers can
analyze the stream without another DMA owner.

Live PSD is a bounded subscriber, not a competing reader. A raw capture retains
records independently of a display cut. Current IIO frames contain 1,024 opaque
16-byte records; sparse sources may update slowly until a frame is ready or Stop
drains the acquisition. Review the final status, fault counters, and continuity
summary before treating the capture as complete.

Every clean list-mode run starts with a START record. Frames that arrive before
it are left over from an earlier, interrupted run; they are discarded and
counted in the summary, and a run whose START record is missing or not first is
reported as incomplete. The hardware loss counter is not reset by Start, so the
summary counts only records dropped during this run and shows a value carried
over from an earlier run separately.

The binary file layout and an example reader are in
[MCA list-mode binary format](#format-listmode).

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

**References** — The live ratio is a charge-comparison ratio: the method's analogue origin, a two-gate QDC implementation, and a textbook review.

- Brooks, F. D. (1959). A scintillation counter with neutron and gamma-ray discriminators. *Nucl. Instrum. Methods* 4, 151–163. doi:10.1016/0029-554X(59)90067-9
- Moszyński, M., Bizard, G., Costa, G. J., et al. (1992). Study of n-γ discrimination by digital charge comparison method for a large volume liquid scintillator. *Nucl. Instrum. Methods Phys. Res. A* 317, 262–272. doi:10.1016/0168-9002(92)90617-D
- Knoll, G. F. (2010). *Radiation Detection and Measurement*, 4th ed., ch. 8, 15 and 17. Wiley. ISBN 978-0-470-13148-0

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
The fitting functions, for PSD-ratio coordinate `x`, are:

**Gaussian peak**

$$
G_i(x)=A_i\exp\!\left[-\frac{(x-\mu_i)^2}{2\sigma_i^2}\right]
$$

**Joint fit**

$$
m(x)=y_0+G_1(x)+G_2(x)
$$

**Separate fits**

$$
m_i(x)=y_{0,i}+G_i(x),\qquad x\in R_i
$$

**Widths and figure of merit**

$$
\mathrm{FWHM}_i=2\sqrt{2\ln 2}\,\sigma_i
$$

$$
\mathrm{FOM}=\frac{|\mu_2-\mu_1|}{\mathrm{FWHM}_1+\mathrm{FWHM}_2}
$$

`mu_i` is the fitted centroid, `sigma_i` is the positive Gaussian width,
and `A_i` is the peak height in counts per ratio bin, above its background.
The Gaussians here describe the **count projection**, not individual waveform
shapes or normalized probability densities. The offset `y0` is a constant count
background. The [Use y0 offset](#fom-offset) checkbox chooses whether to fit it
or fix it to zero. Joint mode uses one shared offset; separate mode uses one
per selected range. FWHM is the width of the Gaussian above that background.

FOM is dimensionless. A narrow, incorrectly fitted peak can produce an
artificially large FOM. **Poor convergence** means the displayed separation
is not yet trustworthy, even when the optimizer itself reports success.
Inspect the [warning explanations and remedies](#fom-quality) before using it
for comparisons, classification settings, or detector-performance claims.

**Separate Gaussians** fits each population independently in its own colored
ratio range, with its own constant background. Use it for well-separated peaks
whose other population contributes negligible tails inside each fit range.
The two ranges must be ordered and disjoint; the gap is excluded from the fit.
With offsets enabled, reduced deviance uses eight varied parameters in separate
mode and seven in joint mode. With offsets fixed to zero, either mode has six. Reports preserve the model and both ranges.
Waveform gate optimization continues to require the joint fit mode.

**Continuous fit** follows changed projections and completed guide edits using
the latest data at most twice per second. Fits use a limited evaluation budget;
manual **Fit** allows a longer optimization. Automatic updates coalesce and do
not queue fit jobs. Turning continuous fitting off leaves manual fitting available.

A larger FOM indicates greater fitted separation relative to the fitted widths
for those selected events. It is not a classification efficiency, neutron
purity, or a universal detector score. Energy selection, baseline, gates,
histogram resolution, and sample size all matter.

Inspect [fit quality](#fom-quality) before accepting the number.
**Copy table** copies the fit snapshot and diagnostics for comparison.

**References** — FOM definition, Poisson likelihood fit statistic and fit algorithm.

- Winyard, R. A., Lutkin, J. E., McBeth, G. W. (1971). Pulse shape discrimination in inorganic and organic scintillators. I. *Nucl. Instrum. Methods* 95, 141–153. doi:10.1016/0029-554X(71)90054-1
- Baker, S., Cousins, R. D. (1984). Clarification of the use of chi-square and likelihood functions in fits to histograms. *Nucl. Instrum. Methods Phys. Res.* 221, 437–442. doi:10.1016/0167-5087(84)90016-4
- Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters. *J. Soc. Ind. Appl. Math.* 11, 431–441. doi:10.1137/0111030
- Newville, M., Stensitzki, T., Allen, D. B., Ingargiola, A. (2014). LMFIT: Non-linear least-square minimization and curve-fitting for Python. Zenodo. doi:10.5281/zenodo.11813

## PSD — constant count offset y0 {#fom-offset}

**Use y0 offset** is checked by default, preserving the original fit model.
Checked: fit a non-negative constant background (`y0` in the joint model,
or independent `y0_1`, `y0_2` in the separate windows). Unchecked: fix every
offset to exactly zero and fit only peak heights, centroids, and widths.
See the [Gaussian and FOM equations](#fom).

Use zero offset when a zero count background is supported by the selected
projection. Leaving a needless free background near its lower bound can make
covariance estimation unreliable. Conversely, forcing zero when a real
background exists can bias the fitted widths and FOM. Changing this checkbox
invalidates the old fit and triggers a new one when continuous fitting is on.
It changes the count model, not the waveform baseline or stored measurements.

## PSD — fit ROI and centroid guides {#fom-guides}

The shaded fit region selects the ratio interval used by the Gaussian fit.
Place the two centroid guides near the two candidate peaks, with enough of
each distribution inside the fit ROI to constrain its width and background.
In separate mode, drag the blue population-1 range and orange population-2 range
independently, and keep each centroid guide inside its own range. Their bounds
and the fit mode remain selected when histogram settings change.

Guides are starting estimates, not imposed population means. Hiding draggable
guides changes their visibility, not the underlying physics or a hardware gate.
These are ratio coordinates, distinct from the energy ROI and waveform time gates.

## PSD — convergence and fit-quality warnings {#fom-quality}

**Poor convergence** is a combined warning about numerical convergence or fit
quality. It does not always mean the solver stopped early: the optimizer may
converge while the model or its uncertainties remain unreliable. The listed
reason identifies what needs attention.

| Reported reason | Meaning and practical implication | What to check |
| --- | --- | --- |
| LM solver did not converge | The evaluation budget was exhausted or the solver could not settle. Centroids and widths may still move; FOM is provisional. | Place seeds near real peaks, improve the ranges and counts, and try manual Fit with its larger budget. |
| Parameter covariance could not be estimated | Uncertainties are unavailable; the parameters may be poorly constrained or correlated, even with a plausible-looking curve. | Check counts, overlap, truncation, and whether a free y0 is needed. Do not invent error bars or treat missing uncertainties as zero. |
| Reduced Poisson deviance indicates a poor Gaussian fit | The selected model disagrees with the observed shape. Skew, extra populations, non-constant background, or unmodeled overlap may bias the widths. | Inspect the shape and use joint mode for overlapping peaks; reconsider the model or physically justified selection. |
| Fewer than 100 counts / less than 3-sigma peak significance | The selected peak has weak statistical support. The reported width or separation can fluctuate strongly. | Accumulate more events or use a physically appropriate energy selection. Separate mode checks each population independently. |
| Centroid or width pinned near a bound | A fit may be using a range edge or constraint instead of measuring the full peak. Its apparent width can be misleading. | Keep each centroid inside its range and include both tails plus enough baseline to constrain y0. |
| Width below histogram resolution | The fitted width is too narrow relative to the ratio-bin spacing to resolve reliably; FOM may be inflated. | Revisit histogram resolution and collect enough counts at that binning. |
| Populations not independently resolved | The joint fit cannot distinguish the two populations cleanly, so their centroids and widths can trade off. | Improve statistics or selection; separate mode is appropriate only when the other peak's tail is negligible in each range. |

The minimized objective uses signed Poisson-deviance residuals, including empty
bins. Reduced deviance is their squared sum divided by `N - p`: `N` is the
number of included ratio bins and `p` is 7 (joint, offset on), 8 (separate,
offsets on), or 6 (either mode, offsets off). For separate fits the gap is
excluded and the counts and varied parameters from the two ranges are added.
A value near one can be consistent with the counting model, but is not proof
that the detector model or Gaussian assumption is correct. The warning threshold
above 3 is a diagnostic heuristic, not a confidence level or p-value.

A large FOM from a poor fit is not a qualified detector-performance result:
centroid separation, fitted widths, their uncertainties, and the resulting FOM
may all be biased or unstable. The application keeps the numbers available
with a warning, and waveform gate optimization rejects fits with warnings.
Fix the underlying issue before comparing runs or quoting a separation result.
Changing the range or disabling y0 merely to remove a warning is not independent
validation; include enough tails and use selections justified by the data.

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

**References** — Digital charge comparison on sampled pulses.

- Moszyński, M., Bizard, G., Costa, G. J., et al. (1992). Study of n-γ discrimination by digital charge comparison method for a large volume liquid scintillator. *Nucl. Instrum. Methods Phys. Res. A* 317, 262–272. doi:10.1016/0168-9002(92)90617-D
- Kaschuck, Y., Esposito, B. (2005). Neutron/γ-ray digital pulse shape discrimination with organic scintillators. *Nucl. Instrum. Methods Phys. Res. A* 551, 420–428. doi:10.1016/j.nima.2005.05.071
- Söderström, P.-A., Nyberg, J., Wolters, R. (2008). Digital pulse-shape discrimination of fast neutrons and γ rays. *Nucl. Instrum. Methods Phys. Res. A* 594, 79–89. doi:10.1016/j.nima.2008.06.004
- Nakhostin, M. (2017). *Signal Processing for Radiation Detectors*. Wiley. doi:10.1002/9781119410225

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
The saved sample period is the workbench's analysis value: 2 ns for Scope NDMA,
whose header does not store it. ROOT/HDF5 are analysis outputs; reopen the native binary
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
Scope `.bin` recordings hold raw ADC samples 2 ns apart, so the field is fixed at
2 ns for them; it is editable only for formats that do not record it, such as CAEN.
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

**References** — FOM-maximising gate choice and other approaches to choosing integration gates.

- Sénoville, M., Delaunay, F., Pârlog, M., Achouri, N. L., Orr, N. A. (2020). Neutron-γ discrimination with organic scintillators: intrinsic pulse shape and light yield contributions. *Nucl. Instrum. Methods Phys. Res. A* 971, 164080. doi:10.1016/j.nima.2020.164080
- Zhou, J., Abdulaziz, A., Altmann, Y., Di Fulvio, A. (2023). Generalized method for the optimization of pulse shape discrimination parameters. *Nucl. Instrum. Methods Phys. Res. A* 1050, 168184. doi:10.1016/j.nima.2023.168184
- Kaplan, A. D., Blair, B., Glenn, A., Wurtz, R. (2023). Automated integration gate selection for Gaussian mixture model pulse shape discrimination. *Nucl. Instrum. Methods Phys. Res. A* 1055, 168486. doi:10.1016/j.nima.2023.168486
- Graham, C., Todd, E., Wilson, C., et al. (2026). Light output-dependent integration gating for optimized pulse shape discrimination. *Nucl. Instrum. Methods Phys. Res. A* 1088, 171521. doi:10.1016/j.nima.2026.171521

## Current — monitoring overview {#current}

Current monitoring starts stopped. Choose the acquisition mode, verify the
signal-specific calibration, then **Start**. Stop releases its worker;
the numeric display and plot are rate-limited independently of acquisition.
Labels show the essentials; hover a value or the status bar for full diagnostics.

Raw values are not amperes until the appropriate zero, scale, and units have
been established. IIR and DMA expose different signals and need separate
qualification. See [mode](#current-mode) and [calibration](#current-calibration).

## Current — IIR, Scope DMA, and periodic sums {#current-mode}

IIR polling reads signed FPGA input-filter output for lightweight monitoring.
Its host polling schedule is not a hardware-time sample grid. Scope DMA reads
complete timestamped waveforms and computes sample-weighted statistics.
**Periodic sums** (IP123/IP124) uses this plot for [window means](#scope-sums)
with inline binary or HDF5 recording. Set its **Window size** and **Window gap** directly
in Current while stopped; the preview shows window duration and start interval.
Stop drains the reader and restores Scope waveform settings, including frame size. The sum view reports ADC codes
and does not apply the IIR/waveform linear calibration.

DMA owns the Scope buffer and applies its qualified periodic preset. The stopped
view can change the DMA frame gap or Reset Scope defaults afterward. Do not start
another Scope DMA owner on the same channel. The modes' raw scales differ.

**Pause readbacks** stops MCA
histogram/statistics, Power Supply (HV and temperature) and Global diagnostics
polling on every channel while this Current acquisition runs. They all share the
board's IIO daemon with the capture; measured live, MCA polling alone lowered
periodic-sum coverage at 256 samples and gap 0 from about 99% to about 92%.
Measurements and HV setpoints are unaffected, displays resume when Current stops,
and temperature correction holds its last applied value while paused. Offline
sum replay never pauses anything.

**References** — Current-mode (integrating) detector operation.

- Knoll, G. F. (2010). *Radiation Detection and Measurement*, 4th ed. Wiley. ISBN 978-0-470-13148-0

## Current — zero, scale, and units {#current-calibration}

The saved linear calibration is `(raw - zero) * scale`. Zero removes a measured
offset; scale converts the remaining signal to the selected physical unit.
**Set zero** derives an offset from recent data, so use a
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

**References** — Constant-fraction timing, its digital form, and timing with plastic scintillators.

- Gedcke, D. A., McDonald, W. J. (1968). Design of the constant fraction of pulse height trigger for optimum time resolution. *Nucl. Instrum. Methods* 58, 253–260. doi:10.1016/0029-554X(68)90473-4
- Fallu-Labruyere, A., Tan, H., Hennig, W., Warburton, W. K. (2007). Time resolution studies using digital constant fraction discrimination. *Nucl. Instrum. Methods Phys. Res. A* 579, 247–251. doi:10.1016/j.nima.2007.04.048
- Moszyński, M., Bengtson, B. (1979). Status of timing with plastic scintillation detectors. *Nucl. Instrum. Methods* 158, 1–31. doi:10.1016/S0029-554X(79)90170-8

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

**References** — Chance coincidences of uncorrelated events (rate 2τr₁r₂ for resolving time τ, window 2τ) and their measurement in a delayed window.

- Knoll, G. F. (2010). *Radiation Detection and Measurement*, 4th ed. Wiley. ISBN 978-0-470-13148-0

## Global — per-board settings {#global}

Global settings and diagnostics belong to one physical board, even in a
multi-digitizer workspace. Select the intended source before changing shared
trigger or temperature behavior. Per-board diagnostics are not per-channel
measurements.

See [software start](#global-sync) and [temperature correction](#global-temperature).

## Global — synchronized software start {#global-sync}

**start_enable=0 selects external bypass; it does not assert common STOP.**
Change shared source/routing with both MCAs stopped and their list buffers
closed. Startup selects software source at LOW with both cores stopped.

Prepare Software Sync sets the selected software level LOW. Arm both receivers
and pulse processors before Start Armed MCA Channels asserts HIGH. START is a
level participating throughout the run, not a latched pulse. Lowering the level
and stopping each processor are distinct operations. Duration timeout can lock
until each processor's global enable has been cleared.

Coincidence owns this sequence during its run. Software START does not
synchronize clocks on independent boards. STOP/progress status does not provide
the deferred A07 downstream-completion acknowledgement.

## Global — shared temperature correction {#global-temperature}

Coefficient and offset configure the board-wide MCA temperature correction;
Enabled and its update interval control its use. They are not the same as the
PSU bias-compensation slope and reference temperature.

The base coefficient is multiplied by `temperature - 67` to obtain the
applied coefficient. Only that applied value is encoded as signed Q1.15,
with a 1/32768 LSB and limits -1.0 through 0.999969482421875. +1.0 is
outside the hardware range. Invalid applied values are rejected before
writing either correction term; the base calibration retains its precision.

Use a validated calibration and check the correction's source and diagnostics.
Do not assume a correction fitted for one detector or gain setting applies to
another. A linear coefficient approximates the drift only over a limited
temperature range, and the board sensor can lag the detector temperature after
an ambient change. Hardware availability and units are shown by the supported controls.

**References** — Temperature dependence of scintillator light output and photodetector gain; decay-time stabilization without a temperature sensor.

- Moszyński, M., Nassalski, A., Syntfeld-Każuch, A., et al. (2006). Temperature dependences of LaBr₃(Ce), LaCl₃(Ce) and NaI(Tl) scintillators. *Nucl. Instrum. Methods Phys. Res. A* 568, 739–751. doi:10.1016/j.nima.2006.06.039
- Pausch, G., Stein, J., Teofilov, N. (2005). Stabilizing scintillation detector systems by exploiting the temperature dependence of the light pulse decay time. *IEEE Trans. Nucl. Sci.* 52, 1849–1855. doi:10.1109/TNS.2005.856616

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
source and resulting voltage. The breakdown-voltage coefficient can differ
between units of one type and from the datasheet value. This compensates bias response, not arbitrary
offline spectrum drift. Observe hardware and detector voltage limits.

**References** — Temperature dependence of SiPM breakdown voltage and gain, and its compensation by bias adjustment.

- Otte, A. N., Garcia, D., Nguyen, T., Purushotham, D. (2017). Characterization of three high efficiency and blue sensitive silicon photomultipliers. *Nucl. Instrum. Methods Phys. Res. A* 846, 106–125. doi:10.1016/j.nima.2016.09.053
- Licciulli, F., Marzocca, C. (2015). An active compensation system for the temperature dependence of SiPM gain. *IEEE Trans. Nucl. Sci.* 62, 228–235. doi:10.1109/TNS.2015.2388580
- Grodzicka, M., Moszyński, M., Szczęśniak, T., et al. (2014). Performance of FBK high-density SiPMs in scintillation spectrometry. *J. Instrum.* 9, P08004. doi:10.1088/1748-0221/9/08/P08004

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
not establish peak identity by themselves. A quadratic model can absorb mild
scintillator non-linearity within the calibrated range; do not extrapolate the
calibration beyond its outermost points.

Calibration maps raw channel to energy for the associated configuration
fingerprint. Gain, shaping, or detector changes may invalidate it. Overlaying
spectra is not permission to combine incompatible configurations. The status-bar
energy-calibration indicator opens this workspace on double-click.

**References** — Energy calibration and scintillator non-linearity.

- Gilmore, G., Joss, D. (2024). *Practical Gamma-ray Spectrometry*, 3rd ed. Wiley. doi:10.1002/9781394286577
- Kong, Y., Pausch, G., Römer, K., et al. (2010). Linearization of gamma energy spectra in scintillator-based commercial instruments. *IEEE Trans. Nucl. Sci.* 57, 1430–1434. doi:10.1109/TNS.2009.2033684

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

**References** — Peak models, Poisson fitting and decay-data sources.

- Gilmore, G., Joss, D. (2024). *Practical Gamma-ray Spectrometry*, 3rd ed. Wiley. doi:10.1002/9781394286577
- Phillips, G. W., Marlow, K. W. (1976). Automatic analysis of gamma-ray spectra from germanium detectors. *Nucl. Instrum. Methods* 137, 525–536. doi:10.1016/0029-554X(76)90472-9
- Baker, S., Cousins, R. D. (1984). Clarification of the use of chi-square and likelihood functions in fits to histograms. *Nucl. Instrum. Methods Phys. Res.* 221, 437–442. doi:10.1016/0167-5087(84)90016-4
- Tuli, J. K. (1996). Evaluated nuclear structure data file. *Nucl. Instrum. Methods Phys. Res. A* 369, 506–510. doi:10.1016/S0168-9002(96)80040-4
- Trkov, A., Griffin, P. J., Simakov, S. P., et al. (2020). IRDFF-II: a new neutron metrology library. *Nucl. Data Sheets* 163, 1–108. doi:10.1016/j.nds.2019.12.001

## Analysis — PSD event readback {#psd-readback}

The **File** menu contains **Load event file...**, **Cancel loading**,
**Download report...**, and **Close workbench**. **Analysis** provides
**Recalculate PSD matrix**, **FOM vs energy...**, Gaussian fitting and fit-table
copy. **View** toggles
PSD ROI and FOM guides, synchronized with the plot controls; **Help** opens this
bundled guide. Window-local shortcuts are **Ctrl+O** (load), **Ctrl+R**
(recalculate), **Ctrl+E** (FOM vs energy), **Ctrl+S** (report), and **Ctrl+W**
(close).

The histogram settings apply when the matrix is built. After you change them
(press Enter or leave the settings), the workbench asks whether to rebuild the
matrix from the loaded file; moving between the settings does not interrupt you.
**Yes** reads every event again, keeps the PSD cut and energy ROI, and clears the
FOM fit. After **No**, the plot keeps its previous settings until you choose
**Analysis → Recalculate PSD matrix**, and the same values are not asked about
again.
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

## Analysis — PSD FOM versus energy {#psd-energy-fom}

PSD bands move and narrow with energy. A single ratio projection over a wide
energy ROI merges different band positions: its peaks widen, the Gaussian model
stops describing them (the fit is flagged), and the FOM understates the
separation reached at each energy. **Analysis → FOM vs energy...** in the PSD
event readback divides the current FOM energy ROI into slices and fits each one,
and the whole ROI for comparison, with the FOM panel's current model, windows,
seeds and y0 setting.

**Slices** sets how many parts the ROI is divided into. **Equal counts** places
the boundaries so each slice holds about the same number of events, so a
falling spectrum does not leave the high-energy slices nearly empty; **Equal
width** gives slices of equal energy width. Boundaries stay on matrix bin edges.
Slices with fewer than **Minimum events** are listed but not fitted.

The upper plot shows each slice's FOM, with bars spanning the slice; hollow
markers are fits flagged as unreliable, and the dashed line is the whole-ROI
FOM. The lower plot shows both population centroids with bars of ±FWHM/2, so
band movement and overlap are visible. The table lists every slice with its
events, centroids, FWHMs, FOM, reduced Poisson deviance and fit status.

**Use best slice as ROI** sets the readback's FOM energy ROI to the slice with
the highest FOM among fits that converged without warnings (marked with a ring
in the plot and in bold in the table), shows the ROI, and refits the FOM panel
there. Slice edges are matrix bin edges, so the panel reproduces that slice's
FOM; a flagged fit is never chosen, because a poor fit can report an inflated
FOM. The next **Fit slices** divides the new, narrower ROI.

**Fit slices** refits from the matrix, ROI and fit settings shown at that
moment; recalculating the matrix clears the result. **Copy table** copies
tab-separated text, and **Save CSV...** writes the table after `#` lines that
record the ROI, slicing and fit settings. When slices have been fitted, the
readback report adds a FOM-versus-energy page and `fom_vs_energy.csv`.

A slice FOM describes that energy range only. Compare slices of similar
statistics, and do not average slice FOMs into one figure.

**References** — FOM in narrow energy cuts and its energy dependence.

- Grodzicka-Kobylka, M., Szczęśniak, T., Moszyński, M., et al. (2020). Fast neutron and gamma ray pulse shape discrimination in EJ-276 and EJ-276G plastic scintillators. *J. Instrum.* 15, P03030. doi:10.1088/1748-0221/15/03/P03030
- Sénoville, M., Delaunay, F., Pârlog, M., Achouri, N. L., Orr, N. A. (2020). Neutron-γ discrimination with organic scintillators: intrinsic pulse shape and light yield contributions. *Nucl. Instrum. Methods Phys. Res. A* 971, 164080. doi:10.1016/j.nima.2020.164080
- Graham, C., Todd, E., Wilson, C., et al. (2026). Light output-dependent integration gating for optimized pulse shape discrimination. *Nucl. Instrum. Methods Phys. Res. A* 1088, 171521. doi:10.1016/j.nima.2026.171521

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

To read the binary files directly instead, see
[Scope waveform binary format](#format-waveforms),
[MCA list-mode binary format](#format-listmode) and
[Current periodic-sum binary format](#format-current).

**References** — ROOT format and the library that writes it.

- Brun, R., Rademakers, F. (1997). ROOT — an object oriented data analysis framework. *Nucl. Instrum. Methods Phys. Res. A* 389, 81–86. doi:10.1016/S0168-9002(97)00048-X
- Pivarski, J., Schreiner, H., Hollands, A., et al. *Uproot*. Zenodo. doi:10.5281/zenodo.4340632

## Files — Scope waveform binary format {#format-waveforms}

**Record DMA frames** in Scope writes a `.bin` file in NDMA format version 1:
a 24-byte header followed by waveform frames that all have the same size. All
values are little-endian. There is no index or trailer, so frame *i* starts at
byte `24 + i × frame_bytes`. MCA list-mode files share this header; see
[MCA list-mode binary format](#format-listmode).

**Header** (24 bytes)

| Offset | Type | Field | Meaning |
| --- | --- | --- | --- |
| 0 | 4 bytes | magic | `NDMA` |
| 4 | uint16 | version | 1 |
| 6 | uint8 | channel | Scope channel, 0 or 1 |
| 7 | uint8 | reserved | 0 |
| 8 | float64 | created | Unix time in seconds when recording started, from the computer clock |
| 16 | uint32 | frame_samples | 16-bit words per frame, including the timestamp |
| 20 | 4 bytes | padding | 0 |

**Frame** (`frame_bytes = 2 × frame_samples`)

| Offset | Type | Field | Meaning |
| --- | --- | --- | --- |
| 0 | uint64 | timestamp | Trigger time in 8 ns ticks of the board counter |
| 8 | int16 × (frame_samples − 4) | samples | Signed ADC codes, 2 ns apart |

`frame_samples` is the Scope **Frame** length in ADC samples (duration / 2 ns).
The timestamp replaces the frame's first four captured samples, so each frame
stores `frame_samples − 4` samples: a 1024-sample frame holds 1020. The
timestamp marks when the trigger was accepted, not the time of the first stored
sample. Ticks count from an arbitrary origin, so use differences between
frames, computed as integers. The `created` time comes from the computer and is
not synchronized with the tick counter.

The recorder writes only complete frames, so a finished file holds exactly
`(file size − 24) / frame_bytes` frames. A remainder means the file was cut
short, for example by an interrupted copy; ignore the incomplete last frame.
The file stores no trigger level, pretrigger, DAC baseline or other setting:
keep a **Save Settings...** file with the capture ([Settings](#settings)).
Periodic frames that the firmware averages over four samples cannot be stored
in this format, so such a recording is refused.

This reader needs only Python 3 and NumPy. It maps the file instead of loading
it, so large captures open immediately:

```python
import os
import struct

import numpy as np

HEADER = struct.Struct("<4sHBBdI4x")  # 24 bytes, little-endian


def read_scope_ndma(path):
    """Return (header, timestamps, waveforms) of a Scope NDMA v1 file.

    timestamps: uint64 trigger times in 8 ns ticks, one per frame.
    waveforms: int16 array with one row per frame, samples 2 ns apart.
    """
    with open(path, "rb") as f:
        fields = HEADER.unpack(f.read(HEADER.size))
    magic, version, channel, _, created, frame_samples = fields
    if magic != b"NDMA" or version != 1:
        raise ValueError("not a Scope NDMA version 1 file")
    if frame_samples <= 4:
        raise ValueError("no waveform samples (list mode has frame_samples 0)")
    frame = np.dtype([
        ("timestamp", "<u8"),
        ("samples", "<i2", (frame_samples - 4,)),
    ])
    count = (os.path.getsize(path) - HEADER.size) // frame.itemsize
    if count > 0:
        frames = np.memmap(path, dtype=frame, mode="r",
                           offset=HEADER.size, shape=(count,))
    else:
        frames = np.zeros(0, dtype=frame)
    header = {"channel": channel, "created_unix_s": created,
              "frame_samples": frame_samples}
    return header, frames["timestamp"], frames["samples"]
```

For example:

```python
header, ticks, waves = read_scope_ndma("scope.bin")
print(len(waves), "frames of", waves.shape[1], "samples")
t_ns = np.arange(waves.shape[1]) * 2.0    # sample times within a frame
spacing_ns = np.diff(ticks) * 8           # trigger spacing, exact integers
baseline = waves[:, :16].mean(axis=1)     # assumes 16 samples before pulses
height = waves.max(axis=1) - baseline     # positive pulses; min() if negative
```

## Files — MCA list-mode binary format {#format-listmode}

**Record list-mode DMA** in MCA, with **Binary NDMA + YAML** chosen under
**General Settings → Acquisition files → MCA output**, writes a `.bin` file in
NDMA format version 2 and three companion files with the same name:

| File | Contents |
| --- | --- |
| `name.bin` | 24-byte header, then the event records |
| `name.json` | Capture check: frame and record counts, record schema, driver loss and fault counters, `continuity_valid`, SHA-256 of the `.bin` |
| `name.yaml` | Device settings at the start of the run |
| `name.run.json` | Run summary: duration, records, average rate, `continuity` |

**Header** — the same 24 bytes as a [Scope waveform file](#format-waveforms):
magic `NDMA`, uint16 version 2, uint8 MCA channel, a reserved byte, float64
Unix time of the start, and uint32 `frame_samples`, which is 0 here.

**Records** — after the header come whole frames of 1,024 records of 16 bytes
(16,384 bytes per frame), so the record count is `(file size − 24) / 16` and is
a multiple of 1,024:

| Offset | Type | Field | Meaning |
| --- | --- | --- | --- |
| 0 | uint8 | marker | Flags, see below |
| 1 | uint8 | zc_offset | Timing offset in whole ADC samples, unsigned |
| 2 | int16 | zc_estimation | Timing fraction, value / 16384 samples |
| 4 | uint16 | charge_energy | Charge-comparison (short-gate) energy, raw |
| 6 | uint16 | trapezoid_energy | Energy from the configured estimator, raw |
| 8 | uint64 | timestamp | Event time in 8 ns ticks |

Marker bits: `0x02` CFD timing is valid; `0x08` a PSD zero-crossing result is
present, and then `zc_offset` and `zc_estimation` hold that result in 8 ns
units instead of a CFD time; `0x40` and `0x80` are logic inputs 1 and 0. Other
bits have no defined meaning.

The hardware delivers each record as 16 opaque bytes. The field meanings above
are the decoding profile the application selects for known firmware, named by
`client_record_schema` in `name.json`. Decode the fields this way only when it
reads `vdpp-zc-calc-q2.14-v1`; with `unqualified-opaque[16]` the bytes are kept
but their meaning is not asserted.

**Records that are not events:**

- **START** — the first record of a clean run has timestamp 1 and every other
  byte zero.
- **Padding** — when a run stops, the last frame is filled up to 1,024 records
  with all-zero records. There is no end record.
- **Earlier runs** — records before START are left over from an earlier run
  that was not stopped cleanly. The application drops whole frames that come
  before START. If START is not the first record of its frame, the records
  before it stay in the file and the run is reported incomplete.

**Timing** — for a record with `0x02` set and `0x08` clear, the CFD event time is
`8 × timestamp + 2 × (zc_offset + zc_estimation / 16384)` ns. A valid
crossing has `zc_estimation` from −16384 to 0; treat other values as having no
fine time, and records without the CFD flag as having only the 8 ns timestamp.
For exact time differences use integers in units of 1/8192 ns:
`timestamp × 65536 + zc_offset × 16384 + zc_estimation`. To test a decoder:
the bytes `02 13 00 f0 34 12 78 56 64 00 00 00 00 00 00 00` decode to marker 2,
zc_offset 19, zc_estimation −4096, charge_energy 0x1234, trapezoid_energy
0x5678, timestamp 100 and CFD time 837.5 ns.

**Energies** are raw 16-bit values, not keV. The application's coincidence
analysis places `trapezoid_energy >> 2` on the 16,384-channel MCA axis, and its
PSD view uses the ratio `1 − charge_energy / trapezoid_energy`.

**Other list-mode files** — version 1 files from earlier firmware use the same
16-byte positions, named `short_energy` and `energy` instead of `charge_energy`
and `trapezoid_energy`, with no frames, START record or `.json` file. HDF5 and
ROOT recordings store every record, START and padding included, in a dataset or
TTree named `events` with fields `timestamp`, `long_gate` (trapezoid energy),
`short_gate` (charge energy), `marker`, `zc_offset` and `zc_estimation`; in
HDF5 read only the first `committed_events` rows.

This reader needs only Python 3 and NumPy:

```python
import os
import struct

import numpy as np

HEADER = struct.Struct("<4sHBBdI4x")  # 24 bytes, little-endian
FRAME_RECORDS = 1024
RECORD = np.dtype([
    ("marker", "u1"),
    ("zc_offset", "u1"),
    ("zc_estimation", "<i2"),
    ("charge_energy", "<u2"),
    ("trapezoid_energy", "<u2"),
    ("timestamp", "<u8"),
])


def read_listmode_ndma(path):
    """Return (header, records) of an MCA list-mode NDMA v2 file."""
    with open(path, "rb") as f:
        fields = HEADER.unpack(f.read(HEADER.size))
    magic, version, channel, _, created, frame_samples = fields
    if magic != b"NDMA" or version != 2 or frame_samples != 0:
        raise ValueError("not an MCA list-mode NDMA version 2 file")
    frame_bytes = FRAME_RECORDS * RECORD.itemsize
    count = (os.path.getsize(path) - HEADER.size) // frame_bytes
    count *= FRAME_RECORDS  # whole frames only
    if count > 0:
        records = np.memmap(path, dtype=RECORD, mode="r",
                            offset=HEADER.size, shape=(count,))
    else:
        records = np.zeros(0, dtype=RECORD)
    return {"channel": channel, "created_unix_s": created}, records


def events_only(records):
    """Drop the START record, any records before it, and zero padding."""
    empty = ((records["marker"] == 0) & (records["zc_offset"] == 0)
             & (records["zc_estimation"] == 0)
             & (records["charge_energy"] == 0)
             & (records["trapezoid_energy"] == 0))
    keep = ~(empty & (records["timestamp"] == 0))
    start = np.flatnonzero(empty & (records["timestamp"] == 1))
    if len(start):
        keep[:start[0] + 1] = False
    return records[keep]


def cfd_time_ns(events, reference_ticks=None):
    """CFD event times in ns, NaN where a record has no valid CFD time.

    Times count from reference_ticks (default: the first event), which
    keeps float64 precision in long runs.
    """
    if reference_ticks is None:
        reference_ticks = events["timestamp"][0] if len(events) else 0
    ticks = events["timestamp"] - np.uint64(reference_ticks)
    fine = events["zc_estimation"].astype(np.float64)
    marker = events["marker"]
    valid = (((marker & 0x08) == 0) & ((marker & 0x02) != 0)
             & (fine >= -16384) & (fine <= 0))
    time_ns = (8.0 * ticks.astype(np.float64)
               + 2.0 * (events["zc_offset"] + fine / 16384.0))
    return np.where(valid, time_ns, np.nan)
```

For example:

```python
import json

with open("run.json", encoding="utf-8") as f:
    check = json.load(f)
print(check["client_record_schema"], "complete:", check["continuity_valid"])

header, records = read_listmode_ndma("run.bin")
events = events_only(records)
channel = events["trapezoid_energy"] >> 2          # MCA channel, 0..16383
spectrum = np.bincount(channel, minlength=16384)
total = events["trapezoid_energy"].astype(np.float64)
ratio = 1 - events["charge_energy"] / np.where(total > 0, total, np.nan)
times = cfd_time_ns(events)
```

## Files — Current periodic-sum binary format {#format-current}

**Record** in Current **Periodic sums**, with the **Binary (.bin)** format,
writes an NSUM file: the hardware's 4096-byte sum packets, unchanged, between a
JSON header and a JSON trailer. All integers are little-endian. NSUM files
start with `NSUM`, never `NDMA`. **HDF5 (.h5)** recordings hold the same
packets in a `packets` dataset, plus decoded `timestamps`, `sums` and
`valid_counts` columns. See [Scope periodic sums](#scope-sums) for the
measurement itself.

**Container**

| Part | Size | Contents |
| --- | --- | --- |
| Preamble | 12 bytes | `NSUM`, uint16 version (1), uint16 reserved (0), uint32 header_bytes |
| Header | header_bytes | UTF-8 JSON, then zero bytes up to the next multiple of 4096 |
| Packets | 4096 each | The hardware packets, in order |
| Trailer | 4096 | 16-byte marker `NSUM-ENDTRAILER` plus a zero byte, uint32 json_bytes, UTF-8 JSON, zero bytes |

The first packet starts where the header, rounded up to a multiple of 4096
bytes, ends; packet *i* follows at `4096 × i` bytes after it. The trailer is
written last, after every packet is on disk. If the last 4096-byte block starts
with the marker it is the trailer, and its `packets` value equals the number of
packets before it. A file without a trailer was not closed, for example after
a crash or power loss; its whole packets are still valid. Ignore a partial
final block.

The header JSON holds `format` (`nlab-scope-sums`), `format_version` (1),
`container` (`nsum`), `created_utc`, `timestamp_unit`, `sum_unit`,
`packet_bytes`, `metadata` (channel, IP version, N, gap, clock periods and
producer identity) and `configuration_yaml` (the saved settings, or empty). The
trailer JSON holds `complete`, `cleanup_error`, `final_missed_windows`,
`runtime_metadata` and `packets`.

**Packet** (4096 bytes)

| Offset | Type | Field | Meaning |
| --- | --- | --- | --- |
| 0 | uint64 | first_tick | Start of the packet's first window, 8 ns ticks |
| 8 | 8 bytes | magic | `VDPSUMS1` (IP123) or `VDPSUMS2` (IP124) |
| 16 | uint32 | n | ADC samples per window, N |
| 20 | uint16 | gap | Gap between windows, 8 ns clock cycles |
| 22 | uint16 | count | Valid windows in this packet |
| 24 | uint32 | sequence | Packet number from 0, modulo 2^32 |
| 28 | uint32 | missed | Bits 0–30: windows missed since Start, cumulative; bit 31: final packet after Stop |
| 32 | 4064 bytes | windows | See below |

- **IP124** — up to 508 uint64 sums. Window *i* starts at
  `first_tick + i × (N/4 + gap)` ticks. Slots after `count` repeat the last sum
  and must be ignored.
- **IP123** — up to 254 pairs of uint64 start tick and uint64 sum; the rest is
  zero.

Only the final packet after Stop can hold fewer windows than the maximum.

**Values** — a sum S adds N offset-binary ADC codes (signed code + 32768), so a
window's mean signed ADC code is `S / N − 32768`. Each window lasts N × 2 ns,
and windows start every (N/4 + gap) × 8 ns. A longer step between consecutive
window timestamps means windows were missed; packets carry no continuity
guarantee. Timestamps wrap modulo 2^64. Sums are ADC codes: the Current zero
and scale are not applied ([Current units](#current-calibration)).

This reader needs only Python 3 and NumPy:

```python
import json
import os
import struct

import numpy as np

PACKET = 4096
PREAMBLE = struct.Struct("<4sHHI")
TRAILER = struct.Struct("<16sI")
TRAILER_MARKER = b"NSUM-END" + b"TRAILER\0"
IP123 = int.from_bytes(b"VDPSUMS1", "little")
IP124 = int.from_bytes(b"VDPSUMS2", "little")
PACKET_DTYPE = np.dtype([
    ("first_tick", "<u8"), ("magic", "<u8"), ("n", "<u4"),
    ("gap", "<u2"), ("count", "<u2"), ("sequence", "<u4"),
    ("missed", "<u4"), ("words", "<u8", (508,)),
])


def read_nsum(path):
    """Return the header, trailer and window columns of an NSUM file.

    The trailer is None when the recording did not close cleanly.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic, version, _, header_bytes = PREAMBLE.unpack(
            f.read(PREAMBLE.size))
        if magic != b"NSUM" or version != 1:
            raise ValueError("not an NSUM version 1 file")
        header = json.loads(f.read(header_bytes))
        offset = -(-(PREAMBLE.size + header_bytes) // PACKET) * PACKET
        blocks, partial = divmod(max(0, size - offset), PACKET)
        trailer = None
        if blocks and not partial:
            f.seek(offset + (blocks - 1) * PACKET)
            last = f.read(PACKET)
            marker, length = TRAILER.unpack_from(last)
            if marker == TRAILER_MARKER:
                body = last[TRAILER.size:TRAILER.size + length]
                trailer = json.loads(body)
                blocks -= 1
    packets = np.fromfile(path, dtype=PACKET_DTYPE, count=blocks,
                          offset=offset)
    ticks, sums = [], []
    for p in packets:
        count = int(p["count"])
        if p["magic"] == IP124:
            step = np.uint64(int(p["n"]) // 4 + int(p["gap"]))
            index = np.arange(count, dtype=np.uint64)
            ticks.append(p["first_tick"] + index * step)
            sums.append(p["words"][:count])
        elif p["magic"] == IP123:
            pairs = p["words"][:2 * count].reshape(count, 2)
            ticks.append(pairs[:, 0])
            sums.append(pairs[:, 1])
        else:
            raise ValueError("unknown sum packet type")
    none = np.zeros(0, dtype=np.uint64)
    return {
        "header": header,
        "trailer": trailer,
        "samples_per_window": int(packets["n"][0]) if blocks else None,
        "gap_cycles": int(packets["gap"][0]) if blocks else None,
        "missed_windows": int(packets["missed"][-1]) & 0x7FFFFFFF
                          if blocks else 0,
        "timestamps": np.concatenate(ticks) if ticks else none,
        "sums": np.concatenate(sums) if sums else none,
    }
```

For example:

```python
capture = read_nsum("current.bin")
n = capture["samples_per_window"]
mean_code = capture["sums"] / n - 32768.0       # signed ADC code per window
ticks = capture["timestamps"]
t_s = (ticks - ticks[0]) * 8e-9                 # seconds from the first window
step = n // 4 + capture["gap_cycles"]
lost = int((np.diff(ticks) // step - 1).sum())  # windows missing in between
trailer = capture["trailer"]
complete = trailer is not None and trailer["complete"]
```

## Files — reports, CSV, and destinations {#export}

CSV exports and analysis reports are snapshots with their associated settings.
The default DMA folder controls new capture destinations, not where an already
open source file was recorded. Select a new report/output name to avoid replacing
prior results.

A blank Scope DMA, MCA list-mode or Current sum file name gets a generated name
with the start time (`<prefix>_YYYYmmdd_HHMMSS`). **General Settings →
Acquisition files → File names** can add the start time to a name you choose as
well: `detector.bin` records as `detector_20261123_101500.bin`, and a name that
already exists gains `_001` instead of being replaced. Without the option, a
chosen name is used as given.

PSD report bundles include plots, fit inputs/components, projections, diagnostics,
and provenance. Check whether the result represents sparse preview, selected
events, or completed full reconstruction. Exports cannot supply a missing
physical calibration or prove raw-capture continuity on their own.

## Settings — saved configuration and restore {#settings}

Save/Load Settings serializes application and device configuration. General
Settings controls display defaults, polling, output destinations/formats, and
optional automatic configuration restore. Its categories are listed on the left:
Appearance, Scope display, Acquisition and monitoring, Acquisition files, and
Configuration persistence. **OK** applies the edits made on every category's
page, including pages no longer shown; **Cancel** discards all of them.

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

**References** — Scientific software the application is built on.

- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature* 585, 357–362. doi:10.1038/s41586-020-2649-2
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nat. Methods* 17, 261–272. doi:10.1038/s41592-019-0686-2
- Hunter, J. D. (2007). Matplotlib: a 2D graphics environment. *Comput. Sci. Eng.* 9, 90–95. doi:10.1109/MCSE.2007.55
- Newville, M., Stensitzki, T., Allen, D. B., Ingargiola, A. (2014). LMFIT: Non-linear least-square minimization and curve-fitting for Python. Zenodo. doi:10.5281/zenodo.11813
- Pivarski, J., Schreiner, H., Hollands, A., et al. *Uproot*. Zenodo. doi:10.5281/zenodo.4340632

## Scope periodic sums {#scope-sums}

Select **Current > Periodic sums > Start** to use the existing Current plot. This is not a Scope
waveform trigger. Stop the channel's Scope acquisition first. **Stop and drain**
restores the waveform trigger, pretrigger, and gap after releasing DMA.

**Window size** sets N original ADC samples per sum window: 4..8188, step four,
at 2 ns/sample. **Window gap** is the unobserved time between windows, in us,
with 0.008 us resolution and a 524.280 us maximum. Both controls apply on Start
and stay locked through tail drain. Pretrigger is zero. Each 4096-byte packet transports
254 timestamp/sum pairs (IP123) or 508 uniformly spaced sums (IP124). Only the
valid records contribute; STOP padding is excluded.

The signed window mean is $S/N-32768$, where S is the offset-binary sum.
The interval is $(N/4+gap)\times 8$ ns. Native timestamps and sums stay exact
integers in the recording; the Current plot shows time relative to the latest window.
Exposure counts measured samples; gaps and missed windows are excluded. The
session mean is weighted by measured samples, not by packets or screen updates.

Use **Start**, then **Stop and drain**. A healthy STOP retains accepted tail
windows through a terminal transport timeout before closing. Review missed
counts and errors. A poisoned DMA channel requires a board reboot.

**Record** creates a new versioned sum file with the original packets, settings,
producer/channel/clock identity, and completion status, in the **Format** chosen
beside it. **Binary (.bin)**, the default, stores the packets exactly as received:
about a third of the size and about ten times cheaper to write, so use it for
windows shorter than 448 ns. **HDF5 (.h5)** also stores lossless uint64 timestamp
and sum columns and opens in generic HDF5 tools. A binary file cut off by a crash
stays readable up to its last whole packet and is reported incomplete. The
binary layout and an example reader are in
[Current periodic-sum binary format](#format-current). Blank
filenames generate `chN_current_<timestamp>` with the format's extension in the
measurement folder from **General Settings** on each **Start**, with a numbered
suffix if needed to avoid an existing file. Relative names use that folder;
absolute paths keep their chosen location. A `.bin`, `.h5` or `.hdf5` extension
selects that format; other names gain the selected format's extension.
**Browse…** starts in the configured folder, and missing folders are created
in the background. If an explicitly named file already exists, **Start**
asks before replacing it, unless **General Settings → Acquisition files** adds the
start time to chosen names: the timestamped name never replaces a file.
**Cancel** keeps the existing recording and plot unchanged.
File creation, writes, and final flush run on a background writer. Acquisition
controls stay locked through **Stop and drain** until the recording is closed.
**Tools > Current Capture Analysis** opens a dedicated offline browser
with µs-sized chunks, navigation, statistics and a window-mean histogram. New
GUI captures embed the complete digitizer settings for readback/YAML export
in its **Digitizer configuration** tab. Adjacent noise is RMS(Δ)/sqrt(2) for
consecutive windows; it cannot recover within-window ADC noise. An
incomplete recording remains marked incomplete. The plot cannot recover pulse
shapes, waveform extrema, or noise spectra. Physical current conversion needs
qualified channel/DC-path calibration; this view reports ADC codes.

MCA settings use GUI schema 5 and an explicit hardware block with
parameter_abi_version=2 and parameter_units=adc_samples. Old format 3/4
presentation settings remain readable; unidentified MCA hardware values must
be reviewed before applying. No automatic numerical conversion is performed.

**References** — Current-mode (integrating) detector operation.

- Knoll, G. F. (2010). *Radiation Detection and Measurement*, 4th ed. Wiley. ISBN 978-0-470-13148-0
