# Scope periodic-trigger DMA audit

Date: 2026-09-15
Client repository: `nlab-community`, commit `ad6975a9615f68e64cb97da7c546197e92b6ff59` (`main`)
Hardware repository: `petalinux`, commit `46da8bb853d1bfe0b66ca5735b0be9697e1d1a41` (`ewt_iio`)
Live target: `ip:192.168.10.128:30431`, Linux `5.15.36-xilinx-v2022.2`, target libiio/iiod 0.25, host libiio 0.26, scope 0 `iio:device13`, IP version 121

No source code was changed in either codebase. The only repository change made by this audit is this report.

## Conclusion

The approximately 1 ms timestamp spacing is real, but it is **not** caused by an incorrect nanosecond-to-register conversion or by `check_dma.py` decoding the timestamp incorrectly.

For the reported 1,024-sample frame and 1,000 ns periodic gap:

```text
frame_period_cycles = 1000 ns / 8 ns = 125 clocks
frame duration       = 1024 samples / 4 samples-per-clock = 256 clocks
trigger grid         = 125 + 256 = 381 clocks = 3.048 us
```

The GUI and IIO backend write the correct value, 125, and the driver defines the register as a **post-frame gap**, not the complete start-to-start period. The saved capture and a new live capture both have timestamp deltas that are exact integer multiples of 381 clocks. This is the signature of periodic trigger opportunities being generated correctly but rejected while the scope state machine is busy.

The dominant steady-state limiter in the current client is the serialized, one-frame-per-`Buffer.refill()` transaction through remote libiio/iiod. A new direct live test with no GUI viewer and no disk writer measured a mean refill time of 1.017 ms and a mean FPGA timestamp interval of 1.019 ms. This closely reproduces the file's 1.041 ms average interval. The file writer and plotting are therefore not the cause of the observed millisecond cadence.

There is also a separate intrinsic hardware limit. The v121 scope captures a complete frame into its internal FIFO and then drains that frame before it returns to `M_IDLE`. For a 1,024-sample frame, the driver documents the minimum no-backpressure gap as:

```text
frame_samples / 4 + 2 = 258 clocks = 2.064 us
```

The selected 125-clock/1 us gap is below that limit. Even with an always-ready DMA receiver, every requested trigger cannot be accepted. The first possible accepted spacing on the 381-clock trigger grid is therefore two periods, 762 clocks or 6.096 us. Both the saved file and the new live test have this exact minimum.

Finally, lossless delivery of every such frame to the present remote client is physically impossible over 1 GbE. The requested grid represents approximately 328,084 frames/s and 671.9 MB/s of 2,048-byte frames. Even a loss-free setting at the core's minimum supported gap would require about 498 MB/s. Both exceed the theoretical 125 MB/s line rate before Ethernet/TCP/IIO overhead.

## Evidence from the reported capture

`notebooks/check_dma.py` currently points to:

```text
D:\work\measurements\scope_ch0_20260915_133802_001.bin
```

The existing checker was run with the noninteractive Matplotlib backend. It reported:

| Item | Result |
|---|---:|
| Frame geometry | 1,024 int16 slots / 2,048 bytes |
| Complete frames | 25,978 |
| Trailing bytes | 0 |
| Zero timestamps | 0 |
| All-zero waveforms | 0 |
| Nonpositive timestamp deltas | 0 |
| Timestamp span | 27.0486378 s |
| Header-to-file-mtime span | 27.1836834 s |
| Overall received rate | 960.381 frames/s |
| Median timestamp interval | 993.648 us |
| Mean timestamp interval | 1,041.253 us (derived from the full span) |
| Maximum interval | 13.341096 ms |

Using the user's 1,000 ns gap and the header's 1,024-sample geometry gives the expected 381-clock grid. A separate full-file calculation found:

| Grid check | Result |
|---|---:|
| Deltas exactly on the 381-clock grid | 25,977 / 25,977 |
| Off-grid deltas | 0 |
| Minimum factor | 2 grids = 6.096 us |
| Median factor | 326 grids |
| Mean factor | 341.619 grids |
| Maximum factor | 4,377 grids |
| One-grid intervals | 0 |
| Inferred rejected trigger opportunities | 8,848,248 |
| Rejected fraction between first and last received frame | 99.7073% |

This is valid, ordered frame data with severe trigger-opportunity loss. It is not timestamp corruption and does not show loss of DMA frames that had already completed. IP v121 has no accepted/completed hardware frame counter, so loss before the first and after the last received frame is not observable.

The NDMA v1 header records frame size but does not record trigger mode, gap, or timestamp clock. The grid check above necessarily uses the acquisition setting supplied by the user. A future file-format revision should store those fields so the checker can report rejected trigger opportunities automatically.

## New live tests

All tests used scope 0 and the current `IIODigitizerBackend`. Each test first required `enable=0` and `dma_enable=0`, used the production one-kernel-buffer path, stopped/drained/closed normally, and restored the original state:

```text
enable=0, dma_enable=0, frame_samples=1024,
pretrigger_samples=32, trigger_mode=1, frame_period_cycles=0
```

The `READ LINE: -9` / `READ INTEGER: -9` messages seen during close are the documented result of `Buffer.cancel()` interrupting the final blocked refill, not a capture error.

### Saturated 1 us-gap reproduction

Configuration: 1,024 samples, 32 pretrigger samples, periodic mode 4, `frame_period_cycles=125`, 3,000 frames, timestamps retained in RAM, no viewer polling and no file write.

| Item | Result |
|---|---:|
| Expected trigger grid | 381 clocks / 3.048 us |
| Positive deltas | 2,999 / 2,999 |
| Deltas on expected grid | 2,999 / 2,999 |
| Received payload throughput (timestamp span) | 2.009 MB/s |
| Delta min / median / mean | 6.096 / 1,002.792 / 1,019.341 us |
| Delta p99 / max | 1,283.513 / 10,018.776 us |
| Grid factor min / median / mean / max | 2 / 329 / 334.429 / 3,287 |
| Steady refill median / mean | 1,008.0 / 1,017.0 us |
| Steady refill p99 / max | 1,292.9 / 9,795.9 us |

The refill mean and FPGA timestamp mean agree within about 2.3 us. This directly locates the steady-state pacing at the refill/transport boundary rather than in the GUI writer.

### Maximum-gap control

Configuration: same geometry, periodic mode, `frame_period_cycles=65535`, 1,500 frames. The expected grid is 65,791 clocks or 526.328 us.

| Item | Result |
|---|---:|
| Deltas on expected grid | 1,499 / 1,499 |
| Received payload throughput (timestamp span) | 2.027 MB/s |
| Delta min / median / mean | 526.328 / 1,052.656 / 1,010.171 us |
| Most common grid factors | 2: 1,299; 1: 163; 3: 34 |
| Steady refill median / mean | 1,011.3 / 1,009.9 us |

Even the largest 16-bit hardware gap is usually serviced only every second opportunity by today's stock remote path. This control again ties the accepted-frame cadence to the approximately 1 ms refill transaction.

### Local regression test

```text
.\.venv\Scripts\python.exe -m pytest tests\test_iio_scope_dma.py -q
10 passed in 0.38 s
```

The tests cover exact frame geometry, cancellation, the reader-before-enable lifecycle, and the client file-writer queue. They do not benchmark remote throughput.

### Inventory and source identity checks

`iio_info -u ip:192.168.10.128:30431` succeeded. It confirmed two IP-v121 `vdpp_scope` devices, exact `le:S16/16` scan format, scope 0 stopped with 1,024 samples, and all scope DMA gates clear. It also showed the current list-mode driver reporting `dma_queue_mode=dmaengine`, but the scope driver exposes no equivalent capability attribute.

The nLab copy of `vdpp-scope.c` and the PetaLinux copy differ in the later raw-viewer binary attribute and updated teardown comments. Their periodic-trigger, geometry, buffer, and DMA submission logic is the same. The PetaLinux repository includes both relevant Xilinx DMA patches in `linux-xlnx_%.bbappend`:

- `0003-dmaengine-xilinx_dma-queue-simple-transfers.patch`
- `0004-dmaengine-xilinx_dma-reset-only-simple-s2mm-on-terminate.patch`

An SSH read-only kernel identity check was not completed because the saved host key for `192.168.10.128` differs from the board's current key. The mismatch was not bypassed. Consequently, a live four-kernel-buffer scope test was deliberately not performed: without verified deployment identity, older unpatched Simple-DMA behavior can return pending but unexecuted buffers as completed, producing duplicate/stale/zero frames. Existing PetaLinux reports provide the multi-buffer measurements below, but they are not relabelled as results from today's `.128` target.

## Code and driver cross-check

### Client timing/configuration is correct

- `src/nlab/hardware/digitizer/scope.py` defines the datapath clock as 8 ns and periodic mode as numeric value 4.
- `src/nlab/controllers/scope_controller.py` displays the gap in ns and writes `spinFrameGap // 8`; 1,000 ns becomes 125 cycles.
- `src/nlab/hardware/digitizer/backends/iio_backend.py` passes `frame_period_cycles` directly to IIO, with no second conversion.
- `notebooks/check_dma.py` reconstructs the little-endian uint64 timestamp and uses 8 ns/tick. Its wall-time comparison confirms that scale.

### Driver behavior explains the multiples

In PetaLinux `vdpp-scope.c`:

- lines 54-68 define the 8 ns clock, gap semantics, complete interval, busy-trigger rejection, and intrinsic no-loss gap;
- lines 206-214 submit one exact DMA block using `iio_dmaengine_buffer_submit_block(..., DMA_DEV_TO_MEM)` and insert no delay;
- lines 249-310 require buffer capacity to equal exactly one hardware frame;
- lines 317-360 arm `DMA_ENABLE` without starting acquisition;
- lines 585-618 accept the 16-bit gap and only warn when it is below the state machine's intrinsic limit;
- lines 629-678 start/stop the acquisition gate; and
- lines 968-985 expose the unmodified frame, including its timestamp, as a single int16 scan stream.

The core has one complete-frame FIFO. It captures a frame, drains it to AXI, and only then returns to idle. AXI backpressure therefore propagates into trigger rejection by design.

### Current client serialization

The production IIO scope path:

1. fixes `set_kernel_buffers_count(1)`;
2. allocates an IIO buffer of exactly `frame_samples` scans;
3. performs one blocking `_buffer_refill()` per frame;
4. copies that frame with `ctypes.string_at()`;
5. decodes timestamp/samples;
6. reconstructs the same raw record with `struct.pack()` plus `samples.tobytes()`; and
7. enqueues it to a separate file-writer thread.

The writer queue is already the correct mechanism for keeping filesystem stalls out of the refill loop, and it does not flush per frame. Increasing that queue cannot repair a producer limited before enqueue. The new no-disk live test confirms this.

Stock remote libiio 0.25 performs one synchronous `READBUF` request per `Buffer.refill()` here. With one kernel block, the completed block is owned by iiod/client handling until the next refill returns it. During that interval no spare destination is continuously available; `TREADY` backpressure reaches the scope FIFO and keeps the FSM out of `M_IDLE`.

## Existing measured optimization evidence from the PetaLinux repository

The September 9-10 reports used another address for the same hardware generation (`192.168.10.135`) and must not be treated as measurements of today's `.128` network path. They are nevertheless controlled evidence for solution selection:

| Configuration | 1,024-sample remote throughput |
|---|---:|
| Stock native libiio, 4 kernel blocks | about 4.56 MB/s |
| Batched iiod `READBUF x64`, 2 blocks | 12.12 MB/s |
| Later `READBUF x64`, 4 blocks, iiod CPU1 | 11.96-12.01 MB/s |

For 8,188-sample frames, stock one-block remote capture measured about 23 MB/s; batched requests, four blocks, and CPU placement reached approximately 45-46 MB/s. Local native full-copy tests reached 171-180 MB/s, while a synthetic TCP stream reached 84-86 MB/s. Function-level profiling attributed about 79% of sampled iiod-context stacks to TCP/socket send paths and only about 4% to IIO functions. These results locate the current practical remote limit in per-request/client/iiod/network processing, not the FPGA-to-DDR DMA engine alone.

The experimental batching did **not** enlarge the IIO buffer or DMA BTT. `OPEN` remained one exact hardware frame; one `READBUF` requested multiple complete frames and iiod satisfied it with repeated exact-frame refills. This is the safe interpretation of “skip client confirmation”: amortize/pipeline network request acknowledgement, not skip frame validation or change TLAST geometry.

## Recommended changes

### 1. Make the current behavior explicit in the client

This is a correctness/usability change, not a throughput fix.

- Show both `gap` and computed frame-start interval: `(gap_cycles + frame_samples/4) * 8 ns`.
- Warn before starting periodic DMA when `gap_cycles < frame_samples/4 + 2`. For the reported settings, say that 1,000 ns is below the 2,064 ns intrinsic minimum and at least alternate trigger opportunities will be rejected even without transport stalls.
- Add trigger mode, gap cycles, timestamp frequency, and preferably an acquisition-settings checksum to a new NDMA header version or sidecar.
- Extend `check_dma.py` to report grid multiples and inferred rejected opportunities when those settings are available. Avoid calling them “missing DMA frames”; they are unaccepted triggers.

### 2. Remove avoidable work from the Python hot path

Expose a public backend method returning the already-copied exact raw frame bytes. The streamer should enqueue those bytes directly instead of parsing and reconstructing the record. For further reduction, use a bounded pool of reusable receive buffers and `recv_into`/`memmove`, returning buffers to the pool after the writer commits them. This reduces copies, allocations, and GC/scheduler jitter, but it does not eliminate one synchronous network transaction per frame.

Make viewer polling optional or lower-rate during maximum-throughput DMA. Today's no-viewer test shows it is not the primary 1 ms limiter, so this should be treated as load/jitter reduction rather than the main fix.

### 3. Productionize multi-frame iiod request batching

Highest-value near-term transport change:

- keep each IIO/DMA block exactly one frame;
- retain separate stream and control contexts;
- request a bounded group such as 32 or 64 complete frames per network operation;
- validate every returned chunk, frame boundary, timestamp, and server error;
- preserve reader-before-enable and stop/drain/close ordering; and
- bound latency/memory and surface partial-batch errors explicitly.

This can be a reviewed direct iiod-0.25 protocol client, a small native helper with a stable client API, or a libiio extension. The prior measured x64 prototype is the strongest available evidence that this will help small frames. It will not make a 672 MB/s source fit through 1 GbE.

### 4. Use more kernel blocks only behind a verifiable capability

The nLab backend currently hardcodes one block. The corrected Xilinx Simple-DMA patch makes multiple pending blocks valid and existing tests show two/four blocks improve local throughput substantially, but blocks alone improved stock remote throughput only a few percent.

Before changing the default, add a scope capability such as read-only `dma_queue_mode=dmaengine` (matching list mode), or a precise ABI/capability attribute tied to the corrected kernel path. Then use two or four blocks only when advertised. Do not infer safety from `uname`, IP version 121, or a successful buffer allocation.

Do **not** enlarge `iio.Buffer(scope, frame_samples, False)`. The driver correctly rejects it, and an oversized Simple-DMA BTT would disagree with hardware TLAST.

### 5. Treat iiod CPU/network tuning as secondary and measured

CPU affinity improved the previous batched maximum-frame result, and profiling found most sampled iiod time in socket/TCP transmission. Test affinity, allocation reduction, send batching, and socket-copy reductions one at a time with full-frame retention, long timestamp runs, and both channels. Avoid installing a permanent affinity policy until simultaneous scope/MCA behavior is qualified.

### 6. If every high-rate frame is required, decouple acquisition from Ethernet

No client-only or current-driver-only change can deliver every 1,024-sample frame on a 3.048 us grid over 1 GbE. A real high-rate design needs:

- FPGA capture and drain overlap (ping-pong frame RAM or direct continuous streaming), rather than the current fill-then-drain state machine;
- SG/cyclic DMA or an explicit descriptor/ring design into a large bounded DDR ring;
- hardware accepted-frame, emitted-frame, overflow/backpressure, and sequence counters;
- an overflow policy that is explicit in saved metadata; and
- asynchronous network/file export from the ring at the sustainable sink rate.

A target-side continuously draining userspace service with a bounded RAM queue is a smaller intermediate step. It removes remote request latency from DMA rearming but still consumes CPU copies and eventually overflows when the long-term producer rate exceeds the network/storage sink. It must report that overflow; silently dropping records would make a valid-looking file misleading.

If low-rate lossless remote capture is the goal instead, choose the periodic interval from measured sustainable throughput, not merely from the FPGA minimum. With today's approximately 1 ms one-frame refill, even the current maximum 65,535-cycle gap is too short for consistently lossless 1,024-sample remote capture, as the live control demonstrated. The 16-bit period register may also need widening if the stock transport must be supported without batching.

## Suggested acceptance tests for a proposed fix

1. Verify deployed kernel/module/device-tree identity and the new scope queue capability.
2. Compare stock refill, batched refill, and two/four kernel blocks independently; do not combine changes in the first A/B test.
3. Test 1,024, 4,096, and 8,188 samples, with slow-grid and deliberate-saturation cases.
4. Retain every full frame and require exact byte geometry, nonzero/strictly increasing timestamps, and deltas that are positive grid multiples.
5. Report received MB/s, refill/receive latency distribution, grid-factor distribution, inferred rejected opportunities, file-queue occupancy, and completed DMA IRQ counts.
6. Run long enough to expose GC/scheduler tails; prior short zero-skip runs did not predict later rare stalls.
7. Repeat with viewer off/on, both scope channels independently, both concurrently, and with MCA traffic.
8. Stop acquisition first, drain the optional final frame, close buffers, and verify all gates, DMA status, interrupt accounting, and kernel logs.
9. Preserve the first frame and cross-run boundary timestamps; never hide a stale-tail defect by automatically discarding frame zero.
