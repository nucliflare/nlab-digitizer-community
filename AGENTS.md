# AGENTS.md — nlab-community engineering guide

## Purpose

This file contains project-specific rules for changing the Digitizer 4.0 GUI and
hardware backends. Keep it short and durable. Release history, hardware inventories,
and one-off debugging transcripts belong in `CHANGELOG.md`, focused tests, or a dated
report rather than here.

The direct IIO backend is the current implementation. The gRPC/ZMQ backend remains for
legacy firmware compatibility.

## Source of truth

- For driver registers, units, ranges, scan layouts, and lifecycle rules, the
  authoritative source is the corresponding PetaLinux repository revision.
- When driver work is requested, use the direct PetaLinux repository reference supplied
  by the maintainer. Do not recreate or depend on a copied driver-source snapshot in
  this repository.
- If a code comment disagrees with the driver, follow the driver and update the comment.
- Cite non-obvious driver behavior in comments or docstrings and distinguish clearly
  between source-derived behavior, unit-tested behavior, and live hardware validation.
- Never present generated or simulated output as a live hardware result.

## Relevant code

- `src/nlab/hardware/digitizer/backends/base.py`: backend interfaces.
- `src/nlab/hardware/digitizer/backends/iio_backend.py`: IIO Scope and MCA control,
  readout, and DMA lifecycle.
- `src/nlab/hardware/digitizer/backends/iio_ids_backend.py`: IIO HV/IDS and diagnostics.
- `src/nlab/hardware/digitizer/backends/grpc_backend.py`: legacy backend.
- `src/nlab/hardware/digitizer/dma.py`: streamers and NDMA formats.
- `src/nlab/workers/`: polling and DMA workers.
- `src/nlab/controllers/`: GUI coordination and acquisition state.
- `tests/test_iio_scope_dma.py` and `tests/test_iio_mca_dma.py`: focused IIO DMA coverage.

Scope, MCA configuration, histogram/debug readout, and IIO MCA list-mode userspace
support are implemented. Do not treat list-mode DMA as an outstanding implementation
task. Check `README.md` and `CHANGELOG.md` for current target-side limitations.

## Register and unit rules

- IIO sysfs attributes normally expose the user/physical value after the kernel has
  applied register encoding. Do not repeat the kernel's conversion in Python.
- Attributes explicitly named `*_raw` with a separate `*_scale` are the exception:
  apply the advertised scale in the caller when the public API requires a physical
  value.
- A get-after-set round trip cannot detect a symmetric double conversion. When adding
  or changing a field, compare it with the driver and test documented range boundaries.
- Resolve ambiguous legacy-to-IIO mappings by comparing semantics and valid ranges;
  never guess from similar names alone.
- If the hardware has no corresponding register, raise a clear `NotImplementedError`
  or preserve a deliberate software-only fallback. Never alias an unrelated statistic.
- Methods called unconditionally during controller construction must not raise. Trace
  controller initialization before making an existing method unsupported.

## Device discovery and threading

- Do not share one `iio.Context` across GUI, polling, and blocking DMA threads. Each
  background activity needs its own context and device handles.
- Current pulse-processor and list-mode drivers expose `channel_index`; use it for stable
  pairing.
- Scope and input-filter discovery may fall back to sorted `iio:deviceN` order. Sorting
  is deterministic but does not prove physical channel identity. Verify A/B mapping
  after firmware or device-tree changes.
- Reduce cross-thread exceptions to scalar error data. Retaining exception tracebacks
  can keep native IIO buffers alive unexpectedly.

## DMA invariants

For both Scope and MCA DMA:

1. Stop the acquisition core before creating or arming the IIO buffer.
2. Configure scan elements and validate the advertised geometry.
3. Start the first blocking refill on the DMA thread before enabling acquisition.
4. Accept only complete frames. Never append partial frames to an NDMA file.
5. Stop the acquisition core before draining or closing the buffer.
6. Cancel or drain the reader, wait for its thread to finish, explicitly destroy the
   native buffer, and wait for the hardware ownership gates to clear.

Additional rules:

- Never write `dma_enable` or list-buffer ownership gates directly; buffer callbacks own
  those gates.
- Do not rely on Python destructors for native buffer cleanup.
- Persistent `EINVAL` or `EIO` ends and latches a session. Require the existing recovery
  acknowledgement before rearming rather than recreating a buffer mid-file.
- Scope refills must equal `frame_samples * 2` bytes. Scope NDMA headers store an int16
  sample count, not a byte count; reject trailing partial frames during conversion.
- Current MCA list mode transports one unchanged 16 KiB frame containing 1024 opaque
  16-byte records. Keep the IIO record layout distinct from the same-sized legacy gRPC
  event record and preserve the NDMA version distinction.

## MCA configuration and snapshots

- Pulse-processor configuration writes are rejected while acquisition or list DMA owns
  the core. For non-DMA acquisition, use the existing synchronized
  stop/write/restart helper and shared lock.
- Reconfiguration while running begins a new accumulation. Clear stale histogram and
  elapsed-time presentation state accordingly.
- Do not reconfigure the MCA during list-mode DMA; that requires the complete worker
  stop/drain/close/rearm lifecycle.
- Preserve support for both the current split histogram attributes and the earlier
  monolithic histogram attribute.
- Large binary IIO attributes may include one verified trailing NUL from transport.
  Remove only that final terminator; embedded zero bytes are valid payload.
- Histogram chunks are read sequentially and are not an atomic snapshot while running.

## Local verification

Install the locked development environment with:

```bash
uv sync --locked --extra dev
```

Generated Qt and protobuf modules are intentionally untracked. Regenerate them when
their source forms or proto files change:

```bash
python scripts/build_ui.py
python scripts/generate_proto.py
```

Run tests with an offscreen Qt platform in headless environments:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest
```

On PowerShell, set `$env:QT_QPA_PLATFORM = "offscreen"` before running pytest.

Use focused tests and focused Ruff/mypy checks for changed modules. The repository has
pre-existing global lint and strict-mypy debt, so unrelated findings are not a reason to
rewrite neighboring code.

## Live hardware verification

- Do not perform hardware writes unless the task explicitly includes live testing.
- Begin with read-only discovery using `iio_info -u "ip:<host>:30431"` and verify device
  names, IP versions, geometry, and channel identity.
- Always stop an armed Scope or MCA in a `finally` block and close the `Digitizer`.
  A crashed script can leave `enable=1` and make the next configuration write fail with
  `EBUSY`.
- Test changed register mappings at documented range boundaries, not only typical values.
- Execute walkthrough notebooks in place; do not hand-edit captured outputs. Inspect the
  resulting diff and confirm the final cells leave acquisition stopped.
- Record the firmware/IP version, endpoint, channel, and whether an official libiio tool
  independently reproduced a transport failure.

## Known target limitations

- Superseded five-channel `vdpp_lm_frame` firmware fails buffer allocation with target
  `EINVAL`. The current client expects the `opaque[16]` ABI; do not alter frame geometry
  or request partial scan masks as a workaround.
- Sustained Scope DMA can expose target-side Xilinx VDMA `DMA_INT_ERR` and
  `EOF_EARLY_ERR`. Preserve complete-frame validation, gate cleanup, and recovery
  acknowledgement; the Python client cannot repair a driver/FPGA fault.

## Completion checklist

- Preserve public backend and controller behavior unless the task explicitly changes it.
- Add or update focused tests for lifecycle, format, or settings changes.
- Verify no generated artifacts, captures, audit reports, or build outputs are staged.
- Update `README.md` for user-facing behavior and `CHANGELOG.md` for release-visible
  changes.
- State exactly what was tested locally, what was tested live, and what remains
  source-derived or unverified.
