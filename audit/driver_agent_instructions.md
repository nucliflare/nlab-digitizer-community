# Instructions for the Codex agent responsible for the scope DMA driver

Work only in `D:\work\petalinux`. Do not edit `D:\work\nlab-community`; a separate agent will implement the correlated client changes after your driver ABI and validation results are complete. Do not inspect or modify the legacy gRPC backend. Focus only on the scope's IIO, Linux IIO/DMAengine, libiio/iiod server path, device tree, and directly related FPGA-interface contracts.

## Objective

Implement the target-side changes that let remote IIO scope DMA keep several exact-frame receive buffers armed and efficiently deliver multiple complete frames per network request, while preserving frame boundaries, timestamps, error reporting, and the existing stop/drain/close lifecycle.

The later client must be able to discover the capability without guessing. Your primary deliverable is therefore both working target-side support and a stable, documented IIO/transport contract that the nLab client can consume.

Do not claim that the target can deliver every frame from the reported 1,024-sample, 1,000 ns-gap configuration. That configuration creates a 3.048 us trigger grid and 671.9 MB/s of payload, which cannot fit through 1 GbE. The target-side goal is to approach the sustainable Ethernet rate and report rejected/overflowed work honestly.

## Read before changing code

Read these files completely before implementation:

- `dma_audit.md`
- `docs/scope-architecture.md`
- `docs/user-api.md`, especially scope periodic triggering, capture lifecycle, exact geometry, and experimental network batching
- `docs/scope-transport-measurement-2026-09-09.md`
- `docs/scope-queue-fix-validation-2026-09-09.md`
- `docs/scope-single-channel-iiod-2026-09-09.md`
- `docs/scope-throughput-isolation-2026-09-09.md`
- `docs/scope-throughput-conclusions-2026-09-10.md`
- `petalinux_project/project-spec/meta-user/recipes-modules/vdpp-scope/files/vdpp-scope.c`
- `petalinux_project/project-spec/meta-user/recipes-kernel/linux/linux-xlnx/0003-dmaengine-xilinx_dma-queue-simple-transfers.patch`
- `petalinux_project/project-spec/meta-user/recipes-kernel/linux/linux-xlnx/0004-dmaengine-xilinx_dma-reset-only-simple-s2mm-on-terminate.patch`
- relevant ADI Linux 5.15 IIO DMA-buffer, Xilinx DMA, libiio 0.25 client, and iiod server sources from the actual pinned PetaLinux build

Treat checked-in source as authoritative. Clearly distinguish source-derived conclusions from results confirmed on live hardware.

## Invariants that must not change

1. One hardware scope frame remains one DMA transaction and one IIO block.
2. DMA capacity is exactly `frame_samples * 2` bytes. BTT and FPGA TLAST must agree.
3. Do not obtain batching by enlarging the scope IIO buffer length. `OPEN` geometry stays exactly `frame_samples` int16 scans.
4. The first blocking reader/descriptors must be ready before `enable=1`.
5. Stop order remains `enable=0`, continue reading/draining the optional final frame, then close/abort the descriptor armed for a future frame.
6. Never write `dma_enable` from userspace during an owned buffer lifetime.
7. Never silently discard frame zero, duplicate/stale frames, partial frames, or transport errors.
8. Missing new capability attributes must leave old clients safe: one kernel block and the existing protocol remain valid.
9. Preserve viewer operation and frame-geometry `EBUSY` protection while DMA is armed.

## Required implementation

### A. Make safe multi-buffer support discoverable

The repository already contains the corrected Xilinx Simple-DMA pending/active queue patch, but the scope IIO device does not advertise whether the deployed kernel/DT combination provides that behavior. Add an explicit capability contract to `vdpp_scope`, preferably following the list-mode convention:

- a read-only `dma_queue_mode` with unambiguous values such as `single` and `dmaengine`;
- a read-only recommended safe block depth, for example `dma_kernel_buffers_recommended`;
- if enforceable and meaningful, a maximum supported depth.

Tie `dmaengine` to an explicit device-tree/build contract, not merely to IP version 121, `uname`, or successful buffer allocation. A suitable approach is a scope-node DT opt-in that is added only in an image containing the queue fix. Validate the property during probe and document what an old/missing property means.

Do not advertise multi-buffer safety unless the corrected non-SG Simple-DMA behavior is guaranteed for that deployed image. Review whether the kernel patch should have a formal DMAengine capability rather than relying only on a board-specific DT assertion, and choose the least fragile implementation.

### B. Validate and harden the Simple-DMA queue path

Review the existing queue and terminate patches against the exact pinned Xilinx 5.15 source. Ensure that, for non-SG S2MM:

- only the descriptor actually programmed into destination/BTT moves to active;
- one completion retires exactly one descriptor;
- the next queued descriptor is programmed before userspace has to return the completed block;
- one completion interrupt is generated per frame;
- error and terminate paths return every owned block exactly once;
- late callbacks cannot complete an aborted/reused block; and
- receive-only reset does not affect an MM2S peer.

Add focused tests around queue depth 1, 2, 4, and at least one deeper bounded value. Test submit failures, DMA errors, termination with pending and active descriptors, repeated arm/close, and frame-size changes rejected throughout the armed lifetime.

### C. Support efficient multi-frame network reads without changing DMA geometry

Confirm against the actual pinned libiio/iiod 0.25 sources that one `READBUF` request can request `N * frame_bytes` while the `OPEN` sample count remains exactly one frame. Preserve the existing wire protocol if possible.

Optimize the iiod/server hot path for bounded multi-frame reads where justified by profiling. Candidate changes include:

- keeping several exact-frame IIO blocks queued;
- filling one bounded response from repeated exact-frame dequeues;
- reducing per-frame socket sends/copies and allocations;
- using preallocated/gathered response storage; and
- retaining frame-aligned partial/error reporting if the requested group cannot be completed.

Do not invent a new protocol until you have proved the existing larger-`READBUF` behavior cannot satisfy the contract. If server changes are unnecessary because the existing daemon already supports the required behavior efficiently enough, document and test that result instead of making speculative changes.

The client agent needs an exact contract covering:

- maximum/recommended frames per network request;
- response/chunk framing and scan-mask handling;
- timeout and cancellation behavior;
- how a partial batch or server error is reported;
- whether the final drain frame may complete a short batch; and
- backward-compatible fallback behavior.

### D. Add honest diagnostics that are possible at this layer

Expose stable read-only diagnostics where the driver can measure them reliably, preferably following the list-mode naming style. Priorities are:

- completed DMA frames;
- DMA error/fault state or count;
- configured/active kernel block depth;
- queued blocks or a queue high-water mark; and
- transport/queue mode.

Do not fabricate `accepted_triggers` or `rejected_while_busy` in Linux if IP v121 provides no such registers. Record those as FPGA requirements. Software-completed-frame counts must be clearly named and must not be described as accepted-trigger counts.

Counters need defined reset/lifetime semantics, atomic access, overflow behavior, and a documented relationship to buffer arm/close.

### E. Produce the hardware-dependent follow-up specification

Do not attempt to turn the current non-SG hardware into SG/cyclic DMA only through a Linux patch. The generated design has Simple DMA and the current scope FSM captures and then drains one frame before accepting another trigger.

If FPGA/HLS sources required for the following are absent, do not guess or edit generated artifacts. Write a concrete follow-up design specifying:

- ping-pong or ring-backed frame storage so capture can overlap drain;
- SG/cyclic DMA or another bounded DDR descriptor ring;
- frame sequence, accepted, emitted, busy-rejected, and overflow counters;
- explicit ring-overflow policy;
- stop/drain/empty acknowledgement; and
- compatibility/version signalling to Linux and userspace.

This follow-up is required for bounded burst capture above Ethernet rate and for directly measuring trigger rejection. It is not required before delivering the safe multi-buffer/batched-IIO improvements above.

## Performance targets and interpretation

Use decimal MB/s and count the complete DMA frame, including its eight-byte timestamp.

Baseline evidence from previous reports:

- current nLab live path at 1,024 samples: about 2.0 MB/s and 1.01 ms/refill;
- older stock remote native path at 1,024 samples: about 4.6 MB/s;
- experimental `READBUF x64` at 1,024 samples: about 12.1 MB/s;
- maximum-frame remote batching plus CPU placement: about 45-46 MB/s;
- local native full copy: about 68 MB/s at 1,024 samples and 171-180 MB/s at 8,188 samples;
- synthetic TCP on the board: about 84-86 MB/s.

Treat 80-90 MB/s aggregate scope payload as the credible eventual target on the measured board/NIC, not an automatic acceptance threshold for this driver-only step. The theoretical 1 GbE TCP payload ceiling is higher, but the existing board measured only 84-86 MB/s in an isolated TCP test.

For this implementation, require a demonstrated improvement over matched one-buffer/one-frame-request controls. Report separately:

- driver-only gain from additional kernel blocks;
- network-request batching gain;
- any daemon optimization gain; and
- combined result.

Do not substitute timestamp-only retention for full-waveform throughput.

## Verification

Do source/unit/build verification first. Do not build/deploy an image, reboot, modify SSH trust, or run a risky multi-buffer hardware test unless the operator explicitly authorizes it and the deployed kernel/module/DT identity is verified. The current `.128` target presented a changed SSH host key during the preceding audit; do not bypass that warning.

After an operator deploys the image, use remote iiod as the primary data path and test:

1. one-buffer backward compatibility;
2. depths 2 and 4, plus any larger advertised depth;
3. one-frame `READBUF` versus groups such as 8, 32, and 64;
4. frame sizes 1,024, 4,096, and 8,188;
5. a slow periodic grid and deliberate saturation;
6. both scope channels separately and concurrently;
7. viewer reads during DMA; and
8. repeated stop/drain/close/rearm cycles.

For every case retain complete raw bytes and require:

- exact frame length and no trailing bytes;
- nonzero, strictly increasing timestamps;
- periodic deltas that are positive multiples of the configured grid;
- explicit counts of inferred busy-rejected trigger opportunities;
- no duplicate, stale, zero-filled, or off-grid frames;
- completed-frame/IRQ accounting;
- geometry writes rejected while armed;
- all gates and ownership released after close;
- halted/clean DMA state; and
- no new DMA/reset errors in kernel logs.

Measure throughput, refill/receive latency distribution, response batch sizes, queue occupancy/high-water marks, CPU consumption, and loss/jitter. Do not claim loss-free operation from a short zero-skip run.

## Deliverables for the later client agent

1. Source changes confined to `petalinux`.
2. Focused automated tests and their results.
3. Build instructions and build results; deployment instructions separately.
4. A driver/transport implementation report under `docs/` containing exact commands, hardware identity, raw results, limitations, and final target state.
5. An explicit client integration contract containing attribute names/types/values, safe buffer depth selection, batch request semantics, error/cancel behavior, and fallback rules.
6. A concise list of client changes now enabled by the driver work.
7. A separate list of improvements blocked on FPGA/HLS changes.

Do not edit the nLab client, do not commit unless requested, and do not hide unrelated dirty-worktree changes. Preserve all existing user changes.
