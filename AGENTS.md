# AGENTS.md — IIO digitizer backend (scope + MCA)

This file is a handoff for continuing `src/nlab/hardware/digitizer/backends/iio_backend.py`
and the MCA GUI wiring around it. It was written after a session that (a) implemented
`MCABackend` for the IIO path, (b) live-tested it against a real board and found/fixed a
real bug, and (c) received the driver source and design docs that are now checked into
`hw_description/`. Read this whole file before touching register code — several
non-obvious, expensive-to-rediscover facts are recorded here specifically so you don't
have to re-derive them from scratch or repeat a mistake that already cost a debugging
session.

## 1. What this project is

`nlab-community` is a PySide6 GUI + hardware abstraction layer for a Digitizer 4.0 board
(FPGA-based scope + MCA/DPP instrument). Two backends implement the same abstract
interface (`src/nlab/hardware/digitizer/backends/base.py`):

- **gRPC backend** (`backends/grpc_backend.py`) — talks to a legacy Engine gRPC server.
  Fully implemented, both `ScopeBackend` and `MCABackend`, including list-mode DMA via
  ZMQ (`McaDmaStreamer` in `dma.py`).
- **IIO backend** (`backends/iio_backend.py`) — talks directly to the board's Linux IIO
  subsystem over a network `iio.Context` (no gRPC server in between). This is the
  from-scratch rewrite this file is about. `ScopeBackend` is complete and live-tested.
  `MCABackend`'s register/config half is complete and live-tested (this session).
  **List-mode DMA for MCA is not implemented yet** — that's the main outstanding task,
  see §7.

Layering, top to bottom: `MCAController`/`ScopeController` (Qt widgets) →
`MultiChannelAnalyzer`/`Scope` (`mca.py`/`scope.py`, validation + unit-friendly wrappers)
→ `MCABackend`/`ScopeBackend` (`backends/base.py`, the ABCs) → concrete backend
(`grpc_backend.py` or `iio_backend.py`) → hardware.

## 2. Reference material — `hw_description/` (read these, don't guess)

These are the actual kernel driver sources and the driver author's own design docs,
checked into the repo. They are the ground truth; if `iio_backend.py`'s docstrings and
one of these files disagree, **the file in `hw_description/` wins** (docstrings can go
stale, the driver source can't).

| File | What it is |
|---|---|
| `hw_description/user-api.md` | **Start here.** Authoritative, concise attribute-level spec for every IIO device on the board (register names, rw/ro, ranges, scales). Covers both scope and MCA. |
| `hw_description/mca-architecture.md` | Design rationale for the MCA split (`pulse_processor` / `lm_frame`), event geometry, the lossless start/stop/drain sequence. |
| `hw_description/scope-architecture.md` | Same, for the scope core. Referenced extensively by the existing scope half's docstrings. |
| `hw_description/vdpp-pulse-processor.c` | Kernel driver: MCA config, statistics, filters, pulse memory, histogram. |
| `hw_description/vdpp-input-filter.c` | Kernel driver: FIR/IIR low-pass filter, temperature compensation. |
| `hw_description/vdpp-lm-frame.c` | Kernel driver: MCA list-mode DMA core. **Not yet ported to `iio_backend.py`** — this is §7. |
| `hw_description/scope_backend_iio.py` | A **reference** client implementation for the scope half, supplied by the driver author. `iio_backend.py`'s scope half was modeled on this but has since diverged in places — see §4.3 for one divergence that matters. |

Note: `user-api.md` references `examples/mca_listmode_capture.py` — that file does not
exist in this repo. Don't go looking for it; use `vdpp-lm-frame.c` + `mca-architecture.md`
+ §7 below instead.

## 3. Current implementation status

- **Scope (IIO):** done, live-tested. `notebooks/scope_walkthrough.ipynb`.
- **MCA config/control/statistics/filters/histogram-size (IIO):** done, live-tested this
  session against `ip:192.168.10.128:30431`. `notebooks/mca_walkthrough.ipynb` (executed
  live via `jupyter nbconvert --execute --inplace`, not hand-written — every output in it
  is real). Two real bugs were found and fixed this session; see §4.1 and §4.2.
- **MCA list-mode DMA (IIO):** client implementation is now present across
  `iio_backend.py`, `dma.py`, `dma_workers.py`, `mca_controller.py`, and
  `digitizer.py`; focused format/lifecycle tests are in `tests/test_iio_mca_dma.py`.
  Live discovery and geometry reads pass, but buffer enable is currently blocked by a
  target-side `EINVAL` on the tested board — see §7.5 before changing client geometry.
- **Reading debug waveforms and histograms:** done and live-tested on both channels,
  stopped and running. The latest driver exposes `debug_data` plus four histogram
  chunks, `histogram_data0..3`; the client supports that ABI and retains a fallback for
  the earlier monolithic `histogram_data`. See §4.2.

## 4. Critical lessons from this session — read before writing register code

### 4.1 The sysfs value is already the physical value. Don't convert it again.

An earlier revision of `iio_backend.py` reimplemented the driver's `PP_FMT_X2`/
`PP_FMT_X8`/`PP_FMT_PLUS1_X8`/`PP_FMT_TRAPEZOID_R` conversions
(`pp_field_to_user()`/`pp_field_from_user()` in `vdpp-pulse-processor.c`) client-side, on
the assumption that the sysfs attribute held the *raw register* value. **It doesn't.**
`pp_field_show()`/`pp_field_store()` already do that conversion *inside the kernel* —
the sysfs value is already the "old API"/physical value. Confirmed live: writing `"24"`
to `pretrigger_samples` reads back `"24"`, not `"12"`. `user-api.md` says the same thing
in its own words for every affected attribute, e.g.:

> `pretrigger_samples` | rw | 0..4094, step 2 | old API value; register is value / 2

"register is value / 2" describes the *internal FPGA register*, not the sysfs
attribute — the sysfs attribute **is** "old API value" directly.

The bug this caused was **symmetric**: applying the same conversion a second time
client-side silently wrote a roughly-halved-or-eighthed value to hardware, while still
round-tripping correctly through `get_*()` (get-after-set looked fine). It only
surfaced because a double-converted `crrc2_fdelay` value happened to land outside that
field's valid range and the driver rejected it with a real `OSError`. **A clean
get-after-set round-trip is not sufficient evidence that a register-level unit
conversion is correct** — cross-check against an independent source (the driver's own
field `min`/`max` in `pp_fields[]`, or `user-api.md`'s table) or, ideally, test at both
ends of the field's documented valid range (a double-converted value is far more likely
to fall outside the range at an extreme than in the middle).

Every field in `iio_backend.py` is now a plain passthrough for this reason — see the
NOTE comment at the top of the `MCABackend` section in that file, and
`get_trapez_R()`'s docstring for the fullest worked example. **If you add a new
register-based field, do not add a client-side conversion unless you have confirmed
via `user-api.md` or a live round-trip-at-the-boundary test that the kernel does *not*
already do it.** (The one format the kernel genuinely does *not* convert is `PP_FMT_RAW`
— e.g. `cfd_factor_raw`/`temperature_coefficient_raw` — where a separate read-only
`*_scale` attribute exists purely for the caller to do the scaling; that pattern is
correctly implemented and confirmed live.)

### 4.2 Binary snapshots after the latest driver update: split histogram ABI

Retested live on 2026-08-06 after another board update. Both pulse processors now
advertise 62 regular attributes. `debug_data` is present; the old monolithic
`histogram_data` is absent and has been replaced by `histogram_data0`,
`histogram_data1`, `histogram_data2`, and `histogram_data3`. Each chunk is 16384 bytes
(4096 little-endian u32 bins), and concatenating chunks in numeric order reconstructs
the fixed 16384-bin histogram. The checked-in `vdpp-pulse-processor.c` and
`user-api.md` still describe only the older monolithic ABI, so this conclusion is from
live discovery and readout rather than those currently stale files.

The updated remote attribute transport also changed its buffer-capacity behavior.
`debug_data` has an 8192-byte payload: capacities 8192 and 8193 return `EIO`, while
16383 succeeds and reports 8193 bytes, the last byte being NUL. Each 16384-byte
histogram chunk similarly fails at 16384/16385, succeeds at 32767, and reports 16385
bytes with one trailing NUL. `_read_large_pp_attr()` therefore allocates
`2 * payload_size + 1`, accepts either the older exact-payload return or the new
payload-plus-NUL return, and removes only the verified final terminator. It never
searches for NUL inside the data, because zero bytes are valid binary samples/bins.

`read_waveform_banks()` is confirmed live on both channels, stopped and running,
returning two `(2048,)` `int16` arrays. `read_histogram()` now detects and assembles the
four-chunk ABI, with a monolithic fallback for older boards, and returns `(16384,)`
`uint32`. Twenty consecutive combined waveform+histogram cycles passed on each channel
while enabled with no `EIO`/`ENOENT`; both cores were stopped afterward. Reads while
running are informational rather than an atomic snapshot because the four histogram
attributes are transferred sequentially.

### 4.3 Channel identity: probe order is not guaranteed (confirmed risk, not yet hardened)

`user-api.md` says explicitly, for every multi-instance device including both MCA cores:

> Instance order within `vdpp_scope` follows the IIO device index, which follows probe
> order, which is not guaranteed. Match on the sysfs path if the channel matters... It
> also applies to both MCA device pairs. Match the platform-device symlinks and the
> device-tree `ewt,pulse-processor` relationship; numeric IIO probe order does not
> identify channel A or B.

The reference implementation (`hw_description/scope_backend_iio.py`) takes this
seriously: it sorts discovered devices by parsed `iio:deviceN` number before indexing
(`_device_number()`), and offers an explicit `scope_id="iio:deviceN"` parameter as the
*preferred* way to select a channel in real deployments, with `scope_index` explicitly
documented as "a convenient fallback," not the recommended approach.

**`iio_backend.py`'s current device discovery (every device: `vdpp_scope`,
`vdpp_input_filter`, `vdpp_pulse_processor`, and eventually `vdpp_lm_frame`) does
neither of those things** — it just takes `[d for d in ctx.devices if d.name == X]` in
whatever order pylibiio's `Context.devices` returns and indexes it with the raw
`channel` argument. This has worked correctly on every live test this session and last
session, but that is not the same as being guaranteed to work after a board reboot, a
different device-tree probe order, or a different board.

Checked and confirmed there is currently **no other identifying information exposed
over the network IIO context** for these devices — no `label` attribute (unlike the
board's `tmp117` sensors, which do have one: `HAT_temp`/`cha_temp`/`chb_temp`), no debug
attributes (`_d_debug_attr_count()` is 0 for `vdpp_pulse_processor`/`vdpp_input_filter`),
and the raw context XML's `id="iio:deviceN"` is exactly the probe-order-derived
identifier the doc warns about, not a platform-path-derived one.

**This was not something to silently fix mid-way through an unrelated task, so it
wasn't changed this session** — it's a real correctness risk across the *entire*
existing backend (not just MCA), flagged here for a deliberate decision. Options worth
considering: (a) at minimum, sort by parsed device number like the reference
implementation does, so behavior is at least deterministic within one session; (b) ask
the board/driver maintainers to add a `label` attribute to these four device types,
matching the pattern already used elsewhere on the same board; (c) confirm with
whoever owns the device tree whether probe order is in practice pinned for this
specific board image (i.e., an accepted assumption, documented as such) rather than a
true hardware guarantee.

### 4.4 Scope DMA client recovery fixes and remaining target fault (2026-08-06)

The scope IIO path was reviewed and retested after reports of failed consecutive
viewer/DMA measurements. Short captures work on both channels, and viewer polling
through its separate context remains healthy during DMA. Sustained channel-0 capture,
however, reproduces a real Xilinx VDMA fault: dmesg reports `DMA_INT_ERR` (`0x10`)
followed by `EOF_EARLY_ERR` (`0x100`). This is below the Python client and remains a
driver/FPGA issue; `Cannot stop channel ...: 10008` during ordinary buffer destruction
is the separate, documented stock-Xilinx bounded stop timeout.

Several client bugs amplified the target fault and are now fixed:

- IIO Stop writes `enable=0`, sets the worker event, and calls `Buffer.cancel()` to
  interrupt a refill waiting for a trigger. The GUI keeps Start/configuration disabled
  until the worker has actually finished; shutdown no longer uses `QThread.terminate()`.
- Buffer destruction is explicit (`iio._buffer_destroy()` followed by nulling the
  wrapper pointer) after every refill thread has stopped. Depending on `Buffer.__del__`
  was insufficient after cancellation: Python traceback/thread references kept the
  native buffer alive for the entire three-second gate wait.
- Cross-thread refill errors are reduced to scalar errno/message values instead of
  retaining exception tracebacks which themselves captured the Buffer.
- A refill must return exactly `frame_samples * 2` bytes. The live fault sometimes
  returned only 0 or 8 bytes; those partial records are now rejected before any bytes
  are appended to the NDMA file.
- Persistent `EINVAL`/`EIO` ends and latches the session instead of attempting an unsafe
  mid-file buffer recreation. A clean `acknowledge_dma_recovery()` is required before
  rearm.
- `convert_scope()` now treats the header field as an int16 sample count (therefore
  multiplies by two for bytes) and rejects trailing partial frames.
- Truncated `viewer_data` now raises instead of silently returning a partial plot.

Live confirmation after the fixes:

- An impossible-trigger capture was stopped while blocked in its first refill. It
  returned normally with zero frames in about 1.08 s, wrote only the 24-byte header,
  and left `enable/dma_enable/dma_buffer_active = 0/0/0` with no latched fault.
- A real sustained channel-0 fault stopped after 17 complete frames. The output remained
  exactly frame-aligned, all three gates cleared, `convert_scope()` read 17 frames,
  recovery acknowledgement succeeded, an immediate five-frame DMA capture succeeded,
  and viewer frames changed normally afterward.

Focused coverage is in `tests/test_iio_scope_dma.py`. Do not reintroduce implicit
destructor-only buffer cleanup or write `dma_enable` directly.

### 4.5 Live MCA configuration uses synchronized stop/write/restart

`vdpp-pulse-processor.c`'s `pp_field_store()` rejects every configuration-field write
with `-EBUSY` while either `enable` or `list_buffer_active` is set. This differs from
the current scope driver, which now allows most ordinary fields to be written live.
For non-DMA MCA measurements, `MCAController._apply_hardware_setting()` routes widget
writes through `MultiChannelAnalyzer.reconfigure_while_running()`: it remembers
`enable`, briefly writes `enable=0`, applies the setting, and restores `enable=1` in a
`finally` block. A shared `RLock` also covers MCAWorker's
`get_measurement_in_progress()` call, so the worker cannot mistake that intentional
stop for hardware time-limit completion.

Restarting begins a new hardware accumulation, so the controller clears the old plotted
histogram and elapsed time. Hardware controls remain disabled during list-mode DMA:
`enable=0` alone does not clear `list_buffer_active`, and reconfiguration there would
require the DMA worker's complete stop/drain/close/re-arm lifecycle rather than this
short wrapper.

Confirmed live on 2026-08-06 on both channels: while enabled, sequential `energy_bin`
writes `1, 2, 3, 1` all read back correctly, with zero false
`measurement_in_progress=False` results from a concurrent 5 ms status poll and no
transport errors. The original binning was restored and both channels were stopped.

## 5. Design conventions to keep following

The existing code (both halves) is written in a very particular, deliberate style — keep
matching it:

- **Cite the exact source for every non-obvious claim.** "Per `vdpp-pulse-processor.c`'s
  `pp_field_store()`" or "confirmed live: ..." — not bare assertions. A reader should be
  able to go verify any claim against `hw_description/` or a live test without asking you.
- **Be honest about confirmed-live vs. derived-from-source-but-untested.** Say which one
  a given piece of code/docstring is. This session's whole value came from that
  discipline — the double-conversion bug was *written* confidently and *wrong*; it only
  got caught because "confirmed live" was treated as a real bar to clear, not decoration.
- **When something has no matching register, say so and raise clearly** — don't guess or
  alias it to a different, plausible-sounding register (e.g. `get_events_lost()` was
  *not* aliased to `throughput_error_counter` even though they're conceptually close,
  because that would silently mislabel a real statistic). Use `NotImplementedError` with
  a dedicated, reasoned message constant (see `_MCA_SYNC_UNSUPPORTED` etc. in
  `iio_backend.py`) for "no register exists"; use `RuntimeError` for "device not present
  on this board/channel" (see `_pp_attr_get()`/`_require_mca_pp()`).
- **A method that a GUI controller calls unconditionally during construction must never
  raise.** `MCAController.__init__()` calls `_send_defaults()`/`_load_hardware_state()`
  unconditionally — every method those touch (trace them before adding a
  `NotImplementedError` anywhere) must have a real implementation or a safe software-only
  fallback (see `get_edge_det_coeff()`/`set_edge_det_coeff()`, which has no hardware
  register at all but still can't raise, for exactly this reason).
- **Two `iio.Context`s, never share one across threads.** Anything a background `QThread`
  polls repeatedly (`MCAWorker`'s tick, the scope DMA worker) must use a *dedicated*
  `iio.Context`/device handle, separate from the one GUI-thread code uses — see
  `self._mca_ctx`/`self._mca_pp` and the extensive comment on `self._dma_ctx` in
  `__init__`. This is a real, previously-hit bug class on this codebase, not
  speculative caution.
- **When a legacy/gRPC-era numeric mapping looks ambiguous, cross-check ranges, don't
  guess by name.** The `trapez_T`/`trapez_E` → `trapezoid_beta_raw`/`trapezoid_time`
  mapping (names don't match, ranges do) was resolved this way and confirmed live at
  both ends of each range afterward — see `get_trapez_T()`'s docstring for the method,
  which generalizes to any other legacy-vs-driver naming mismatch you run into.
- **Test at range boundaries, not just typical values**, when confirming a register
  live — see §4.1 for why the middle of a range can hide a bug the edges expose.

## 6. Testing workflow

Hardware: `ip:192.168.10.128:30431` (IP has moved before, see both walkthrough
notebooks' "Hardware note" cells — re-run `iio_info -u "ip:<host>:30431"` if this
address stops responding, and confirm device names before assuming a notebook/this file
still applies).

Sanity-check the board and see every device + attribute at once:

```bash
iio_info -u "ip:192.168.10.128:30431"
```

Quick inline smoke test pattern (what this session used throughout — direct Python, not
the notebook, for fast iteration):

```python
import sys
sys.path.insert(0, "src")
from nlab.hardware.digitizer import Digitizer

d = Digitizer.from_iio(0, "ip:192.168.10.128:30431")
mca = d.mca
try:
    ...  # test code
finally:
    mca.stop()   # always, even on exception -- see the gotcha below
    d.close()
```

**Gotcha confirmed live, twice:** if a test script crashes while a measurement is
armed (`mca.start()` called, no matching `mca.stop()`), the *next* script's config
writes fail with a real `OSError(EBUSY)` — not because the new code is wrong, but
because `enable=1` from the crashed run is still active server-side
(`get_measurement_in_progress()` can already read `False` while `get_global_enable()`
is still `True` — they're not the same bit). Always wrap test code in `try`/`finally:
mca.stop()`, and if you hit a mystery `EBUSY` at the start of a fresh script, check
`mca.get_global_enable()` first before assuming it's a new bug.

Notebooks are meant to be **executed, not hand-written** — every output in both should
be real:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=120 \
  --ExecutePreprocessor.kernel_name=python3 \
  notebooks/mca_walkthrough.ipynb
```

`jupyter_client`/`nbclient`/`nbconvert`/`ipykernel` are already in the project venv.
If you change `iio_backend.py`'s MCA register behavior, re-run
`mca_walkthrough.ipynb` and check the diff of captured outputs, not just that it
completed — a cell that still "runs" with subtly wrong numbers is the exact failure
mode §4.1 describes.

Compile check after any edit (fast, catches syntax errors before a live round-trip):

```bash
python -m py_compile src/nlab/hardware/digitizer/backends/iio_backend.py \
  src/nlab/workers/mca_worker.py src/nlab/hardware/digitizer/dma.py \
  src/nlab/workers/dma_workers.py src/nlab/controllers/main_window_controller.py
```

## 7. Priority 1: implement MCA list-mode DMA (`vdpp_lm_frame`)

This is the main remaining piece. `hw_description/vdpp-lm-frame.c` +
`hw_description/mca-architecture.md` + `hw_description/user-api.md` (§"List-mode buffer
and record format") together are a complete spec — read all three before starting,
they agree with each other and this section just summarizes them for orientation.

### 7.1 Device facts (from the driver source, confirmed against `iio_info` on the live board)

- IIO name `vdpp_lm_frame`, two instances (`iio_info` showed `iio:device16`/`iio:device17`,
  both "buffer capable"), one per channel, linked kernel-side to its matching
  `vdpp_pulse_processor` instance via the `ewt,pulse-processor` device-tree phandle
  (`lm_probe()`'s `vdpp_pulse_get(dev, "ewt,pulse-processor")`) — so, per §4.3, its
  channel identity is exactly as reliable/unreliable as `vdpp_pulse_processor`'s, no
  worse.
- Fixed geometry, not configurable: `frame_records` = 1024, one record = 16 bytes,
  frame = 16384 bytes. Read-only diagnostics: `ip_version` (must be 121),
  `frame_records`, `frame_bytes`, `list_deadtime_raw` ("records dropped while the frame
  output was full" per `user-api.md` — not currently exposed by any `MCABackend`
  method; add as an extension method, same pattern as `get_frame_period_cycles()` on
  the scope backend), `buffer_active`, `record_layout` (self-describing text, matches
  the table below).
- **Five IIO channels, not one** — `lm_channels[]` in the driver: `flags` (u16),
  `cfd_q2` (`IIO_COUNT`, u16), `charge_comparison` energy (`IIO_ENERGY` channel 0, u16),
  `trapezoid` energy (`IIO_ENERGY` channel 1, u16), timestamp (`IIO_TIMESTAMP`, u64).
  `lm_scan_masks[]` only allows **all five together or none** (`GENMASK(4,0)` or `0`) —
  every channel must be `.enabled = True` before creating the buffer, there is no
  partial-scan option like `vdpp_scope`'s single-channel case.
- One event record, little-endian, 16 bytes:

  | Offset | Field | Format |
  |---:|---|---|
  | 0 | flags | u16 — bit0 CFD marker, bit1 CFD valid, `0x2000` input1, `0x4000` input0 |
  | 2 | CFD time | u16, Q2 fixed-point — physical value is `raw / 4` |
  | 4 | charge-comparison energy | u16 |
  | 6 | trapezoid energy | u16 |
  | 8 | hardware timestamp | u64 |

  **This does not match the existing `_EVENT_DTYPE`/`EVENT_STRUCT` in `dma.py`**, which
  was written for the legacy gRPC/ZMQ event stream:
  `EVENT_STRUCT = struct.Struct("<BBHHHQ")` — two separate `u8` fields (`marker`,
  `zc_offset`) where the real IIO record has one `u16` `flags` field. Same total size
  (16 bytes), different field boundaries. **Do not reuse `_EVENT_DTYPE` for the IIO
  path** — define a new dtype matching the table above, e.g.:
  ```python
  _LM_EVENT_DTYPE = np.dtype([
      ("flags", "<u2"), ("cfd_q2", "<u2"),
      ("charge_energy", "<u2"), ("trapezoid_energy", "<u2"),
      ("timestamp", "<u8"),
  ])
  ```
  (Also note, unrelated to this task but discovered while checking `dma.py`:
  `EVENT_STRUCT.size` is actually 16, not the `14` the existing comment next to it
  claims — a pre-existing stale comment in the legacy gRPC path, not something this
  session introduced or something you need to fix to complete this task, but worth a
  one-line correction if you're already in that file.)

### 7.2 Lifecycle (from `lm_buffer_preenable`/`postenable`/`predisable`/`postdisable`, matches `mca-architecture.md`'s 10-step sequence and `user-api.md`'s summary exactly)

1. Pulse processor must be **stopped** (`enable=0`) before arming the list-mode buffer —
   `lm_buffer_preenable()` rejects with `-EBUSY` if `vdpp_pulse_is_running()`. This is
   the same ordering constraint scope DMA already has, just against a different flag.
2. Configure `vdpp_pulse_processor` normally (trigger, filters, etc.) while stopped.
3. Create the IIO buffer on `vdpp_lm_frame` with **all 5 channels enabled**, length
   exactly 1024. `lm_buffer_preenable()` checks `bytes_per_datum == 16` and
   `buffer->length == 1024` and rejects anything else with `-EINVAL`.
4. `postenable()` calls `vdpp_pulse_set_list_buffer(true)` — this is what raises
   `pulse_processor`'s private `list_dma_enable` gate (visible read-only as
   `list_buffer_active`, already implemented on `MCABackend.get_dpp_dma_enable()`).
   **This happens automatically when the buffer is created/enabled — there is still no
   separate client-visible "arm DMA" register write**, same as scope.
5. Start a blocking reader **before** writing `pulse_processor.enable=1` — reuse the
   exact ordering fix already proven for scope
   (`IIODigitizerBackend._start_reader_then_enable()`): issue the first blocking
   `refill()` on a background thread, wait for it to confirm it's actually blocked in
   the call, *then* write `enable=1`. Skipping this ordering was a real, confirmed bug
   for scope DMA — no reason to expect list-mode DMA is immune, and `lm_buffer_
   postenable()`'s own comment ("The DMA helper has queued its first descriptor before
   this callback") describes the identical race window.
6. Write `pulse_processor.enable=1` (this is `mca.start()`, already implemented,
   nothing new needed here).
7. Read complete 16384-byte blocks continuously — same "one `refill()` per frame"
   pattern already working for scope, just with a fixed size instead of a
   `frame_samples`-derived one.
8. To stop: write `pulse_processor.enable=0` **first**, keep the reader/buffer open,
   and drain. `lm_buffer_predisable()`'s own comment: closing while still running is
   only an error-recovery path, and it warns + sleeps 2-3ms before force-stopping if
   that happens — the *normal* path is caller-driven stop-then-drain, matching
   `IIODigitizerBackend._drain_for_close()`'s existing background-thread-refill pattern
   for scope (reuse it, or a close variant of it — no hardware `drained` status exists
   here either, `mca-architecture.md`/`user-api.md` both reference a ~1 second
   inactivity timeout as the practical reference value, not the ~50ms scope uses, since
   these are much bigger blocks at a much lower rate).
9. Close the buffer (`predisable()`/`postdisable()` both call `vdpp_pulse_stop()` +
   `vdpp_pulse_set_list_buffer(false)` — symmetric with the open path, no register the
   client needs to write directly).

### 7.3 Where this plugs into the existing code

The scope IIO DMA path is the exact template — follow the same three-file split:

- **`backends/iio_backend.py`**: add `vdpp_lm_frame` discovery (same optional,
  graceful-degradation pattern as `_pp`/`_input_filter` — see `mca_hardware_present()`),
  a dedicated buffer-lifecycle path mirroring `_create_dma_buffer()`/
  `_start_reader_then_enable()`/`_drain_for_close()`/`_close_dma_buffer()`/
  `_wait_for_dma_release()`, and an extension method (not part of `MCABackend`, like
  `read_dma_frame()`/`capture_to_file()` for scope) that pulls one 16384-byte block and
  returns it parsed as `_LM_EVENT_DTYPE`.
- **`dma.py`**: add `IIOMcaDmaStreamer`, analogous to `IIOScopeDmaStreamer` — a *pull*
  loop over the new backend extension method, writing raw event bytes to file (and/or
  appending parsed events to a shared buffer, matching `McaDmaStreamer.stream_events()`'s
  `event_buffer` parameter so `MCAController`'s ROI/PSD code paths that already consume
  parsed events keep working unmodified). **Not interchangeable with `McaDmaStreamer`**
  (different signature), exactly like `IIOScopeDmaStreamer` isn't interchangeable with
  `ScopeDmaStreamer`.
- **`workers/dma_workers.py`**: add `IIOMcaDmaWorker`, mirroring `IIOScopeDmaWorker`
  (which already exists as the template — read it first, it's short).
- **`controllers/mca_controller.py`**: branch on `isinstance(self._mca_dma,
  IIOMcaDmaStreamer)` in `_start_with_dma()`/wherever else `McaDmaWorker` is
  constructed today, exactly matching how `scope_controller.py` already branches on
  `isinstance(self._scope_dma, IIOScopeDmaStreamer)` (grep that file for the pattern —
  it's the fully-worked reference for this exact kind of branch).
- **`hardware/digitizer/digitizer.py`**: `Digitizer.from_iio()` currently always leaves
  `mca_dma=None`. Once the above exists, construct `IIOMcaDmaStreamer` there — gated on
  `backend.mca_hardware_present()` and lm_frame actually being found, same
  graceful-degradation precedent as everything else in this file.

### 7.4 Before you start

- Update `notebooks/mca_walkthrough.ipynb` (execute it, don't hand-edit) once basic
  register reads work again, and add a DMA-capture section once list-mode DMA is wired
  up — following §6's execute-in-place workflow, not fabricated output.
- Section 4.3's channel-identity risk applies to `vdpp_lm_frame` discovery too — worth
  resolving that first, or at least being deliberate about carrying the same known risk
  forward, rather than accidentally "fixing" it only for the new device and leaving the
  other three inconsistent.

### 7.5 Continuation result and live board blocker (2026-08-06)

The userspace list-mode path described above is now implemented. It uses a third,
DMA-dedicated IIO context; enables all five scan elements; validates v121/1024/16384;
starts the first blocking refill before `pulse_processor.enable=1`; parses the exact
16-byte dtype; stops, drains until one second of inactivity, and then destroys the
buffer; and wires an `IIOMcaDmaStreamer`/worker into `MCAController` and
`Digitizer.from_iio()`. NDMA version 2 distinguishes the IIO record layout from the
legacy same-sized gRPC record, and `dma_converter.py` handles both versions.

Local verification passes: compile, focused Ruff checks, five pytest tests, and targeted
mypy for the streamer/tests. The existing MCA walkthrough was executed successfully
against `ip:192.168.10.128:30431` using the project venv kernel; register/statistics
behavior still works and the notebook leaves `enable=0`.

Live list-mode discovery also passes: `iio:device16` exposes all five expected scan
channels, `sample_size` is 16 after enabling all five, and the attributes read
`ip_version=121`, `frame_records=1024`, `frame_bytes=16384`, `buffer_active=0`, and
`list_deadtime_raw=0`. `Digitizer.from_iio(0, ...)` constructs
`IIOMcaDmaStreamer` correctly.

However, the deployed target rejects buffer creation before acquisition starts:

```
ERROR: Open unlocked: -22
Unable to allocate buffer: Invalid argument (22)
```

This is independently reproducible with the official `iio_readdev`, both with explicit
all-five channel names and with its default all-channel mask. Requested buffer sizes
512, 1024, and 2048 all return the same immediate target `EINVAL`; both pulse processors
are stopped and the ownership gates remain clear. Therefore do **not** change the Python
record geometry to work around it: the client mask/sample size and documented geometry
were independently verified, and the failure is below the Python binding. The next step
is to inspect target `dmesg` for `lm_buffer_preenable()`'s `dev_err` (or a DMA-buffer-core
error before that callback). SSH access from the development machine was denied, so the
precise kernel-side rejection could not be collected in this session. Until that board
issue is fixed, the data/refill and real stop-tail-drain portions are source-derived and
unit-tested but not confirmed live; do not add fabricated DMA output to the notebook.
