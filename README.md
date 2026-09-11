# Nuclear Lab Digitizer — Community Edition

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

Nuclear Lab Digitizer is a PySide6 desktop application for acquiring and
analysing data from the Eastern Wall Technologies Digitizer 4.0. It combines a
two-channel oscilloscope, multi-channel analyser (MCA), pulse-shape
discrimination (PSD), high-voltage control, board diagnostics, and optional
Modbus instruments in one dockable interface.

The application is developed around the board's **direct Linux IIO interface**.
It connects to `iiod` with libiio and uses the board's IIO attributes and
buffers for configuration, live data, and DMA capture.

The gRPC backend is **legacy compatibility support for boards running older
firmware**. It remains usable, but new firmware integration and feature work
target IIO.

## Backend status

| Backend | Status | Transport | Intended use |
|---|---|---|---|
| **IIO** | Recommended | libiio over `iiod`, normally port `30431` | Current Digitizer 4.0 firmware; Scope, MCA, PSD/list mode, IDS/HV, and diagnostics |
| **gRPC** | Legacy | gRPC control plus ZMQ DMA | Compatibility with old firmware exposing the Engine/IDS services |

The high-level `Digitizer`, `Scope`, `MultiChannelAnalyzer`, and `HVSupply`
interfaces are shared by both backends. Backend-specific DMA implementations
preserve the lifecycle required by each firmware generation.

## Main features

### Scope

- Live waveform viewer with level, edge, and periodic trigger modes
- Configurable frame length, pretrigger position, frame gap, and analogue offset
- Persistence and raw display modes
- Full-resolution IIO DMA recording with bounded stop, tail drain, and recovery
- Independent viewing while a DMA capture is active

### MCA and PSD

- Pulse-processor, input-filter, CFD, charge-comparison, and trapezoid controls
- Live 16,384-bin spectra with logarithmic display and ROI statistics
- Debug waveform-bank readout
- Synchronized stop/write/restart when changing MCA settings during acquisition
- List-mode IIO DMA using fixed 1,024-record frames
- Live PSD classification from charge-comparison and energy measurements
- NDMA recording with sidecar capture metadata and HDF5 conversion

### Instrument control

- Per-channel high-voltage output, feedback, and temperature monitoring when the
  IDS IIO devices are present
- Shared trigger control and board-wide diagnostic readings
- Optional RS-485 device discovery through the board's ser2net bridge
- Support for SiPM bias, Geiger-Mueller, and PMT high-voltage Modbus devices
- Safe acquisition and power-supply shutdown when the application closes

### Application workflow

- Dockable and floatable channel panels, including multi-monitor layouts
- Scope, MCA, PSD, PSU, Global, External, and System Log workspaces
- Save and restore hardware plus application settings in YAML
- Convert Scope and MCA binary captures to HDF5 from the File menu
- Remote board reboot and shutdown controls
- Persistent layout, connection, plotting, and developer settings

## Hardware and firmware requirements

For the recommended IIO path, the board must run a firmware and Linux image
that exposes the relevant devices through `iiod`.

| Function | Expected IIO device |
|---|---|
| Scope control, viewer, and DMA | `vdpp_scope` |
| Scope analogue offset | `vdpp_afe_dac` |
| MCA control and histogram | `vdpp_pulse_processor` |
| MCA input filter | `vdpp_input_filter` |
| MCA list-mode DMA and PSD | `vdpp_lm_frame` |
| Shared MCA trigger | `vdpp_sync_trigger` |
| IDS/HV and diagnostics | `ad5686r`, `mcp3564`/`mcp3564r`, `tmp117`, and optional diagnostic devices |

Scope remains available if MCA or IDS devices are absent. The GUI disables the
workspaces that the detected firmware cannot support instead of failing the
whole connection.

The standard project configuration uses:

- IIO/iiod: TCP port `30431`
- Legacy DPP gRPC: TCP port `50050`
- Legacy IDS gRPC: TCP port `50040`
- Optional ser2net bridges: TCP ports `5001` and `5002`

The board address is deployment-specific. Examples below use
`192.168.10.128`; older images commonly used `192.168.10.20`.

### Check the IIO endpoint

Install the libiio command-line tools and inspect the remote context before
troubleshooting the GUI:

```bash
iio_info -u "ip:192.168.10.128:30431"
```

Confirm that the expected device names are present for every requested
channel. Current MCA and list-mode drivers expose `channel_index` for stable
pairing. Scope and input-filter instances without that attribute use sorted
`iio:deviceN` order as a compatibility fallback, so verify physical A/B mapping
after a firmware or device-tree change.

### List-mode firmware compatibility

Current list-mode support requires `vdpp_lm_frame` IP version 121 with:

- `frame_records=1024`
- `frame_bytes=16384`
- `record_layout=opaque[16]`
- one repeated-u8 scan element carrying each 16-byte record unchanged

Older list-mode firmware may expose five semantic scan channels. That ABI is
not compatible with the Linux 5.15 IIO timestamp handling used by the target:
buffer allocation fails immediately with target-side `EINVAL` (`Open unlocked:
-22`), including with the official `iio_readdev` utility. Upgrade the target
driver/firmware to the opaque-record ABI; do not change the Python frame length
or request a partial scan mask as a workaround.

The authoritative target ABI and capture lifecycle are documented in
[`hw_description/user-api.md`](hw_description/user-api.md),
[`hw_description/scope-architecture.md`](hw_description/scope-architecture.md),
and [`hw_description/mca-architecture.md`](hw_description/mca-architecture.md).

## Installation from source

### Prerequisites

- Python 3.12 or newer
- [`uv`](https://github.com/astral-sh/uv)
- The platform-appropriate IIO drivers and native runtime from
  [Analog Devices libiio v0.26](https://github.com/analogdevicesinc/libiio/releases/tag/v0.26)
- The `pylibiio` Python package, installed automatically with the project
- Qt runtime libraries; Python Qt bindings are installed through PySide6

On Linux, the system may also need the OpenGL, EGL, XKB, and XCB libraries used
by Qt.

### Set up the project

```bash
git clone https://github.com/nucliflare/nlab-digitizer-community.git
cd nlab-digitizer-community

uv venv
uv pip install -e ".[dev]"

python scripts/build_ui.py
python scripts/generate_proto.py
```

The generated Qt modules and protobuf stubs are intentionally not committed.
Generate both after a fresh clone. The protobuf files are still packaged so
the legacy backend can be selected at runtime.

## Running the application

```bash
uv run nlab
```

In the connection dialog:

1. Select **IIO**.
2. Enter the board address, for example `192.168.10.128`.
3. Use port `30431` unless the target's `iiod` configuration differs.
4. Select the number of channels exposed by the board and connect.

The dialog remembers a successful selection. For compatibility with existing
installations, a completely new settings profile may initially show gRPC;
switch it to IIO for current firmware.

A saved configuration can pre-fill the backend, address, port, and channel
count, while explicit address options override values from the file:

```bash
uv run nlab --config experiment.yaml --ip 192.168.10.128 --port 30431
```

You can also run the module directly:

```bash
python -m nlab.main
```

## Programmatic IIO access

The hardware layer can be used without the GUI. IIO channel numbers are
zero-based.

```python
from nlab.hardware.digitizer import Digitizer

digitizer = Digitizer.from_iio(
    channel=0,
    uri="ip:192.168.10.128:30431",
)

try:
    digitizer.scope.set_trigger_level(5000)
    digitizer.scope.set_frame_samples(1024)
    frame = digitizer.scope.acquire_frame()

    if digitizer.mca_available():
        digitizer.mca.set_energy_bin(4)
        digitizer.mca.start()
        spectrum = digitizer.mca.acquire_spectrum()
        digitizer.mca.stop()

    if digitizer.hv is not None:
        digitizer.hv.set_hv_voltage(800.0)
        print(digitizer.hv.get_hv_adc_voltage())
finally:
    digitizer.scope.stop()
    if digitizer.mca_available():
        digitizer.mca.stop()
    digitizer.close()
```

Pass `with_ids=False` to `Digitizer.from_iio()` when only Scope/MCA access is
needed. If required IDS devices are missing, construction logs a warning and
continues with `digitizer.hv` set to `None`.

For an older firmware board, the legacy factory remains available:

```python
digitizer = Digitizer.from_grpc(
    channel=1,
    hostname="192.168.10.20",
    port=50050,
    ids_port=50040,
)
```

Legacy gRPC channel numbers are one-based, matching the old service API.

## Capture files and analysis

Scope and MCA recording produce binary files with an `NDMA` header. IIO MCA
list-mode files use format version 2 to distinguish the opaque IIO record from
the same-sized legacy gRPC/ZMQ event record. A JSON sidecar stores IIO capture
geometry, schema, continuity, and driver diagnostics.

Use **File → Convert Binary to HDF5...** in the GUI for portable analysis
files. Quarto examples are provided in:

- [`docs/analyze_scope.qmd`](docs/analyze_scope.qmd)
- [`docs/analyze_listmode.qmd`](docs/analyze_listmode.qmd)

Hardware walkthrough notebooks are available in
[`notebooks/`](notebooks/). Their saved outputs come from real-board execution
and may reflect the address and firmware installed at the time they were run.

## Architecture

```text
PySide6 GUI
  └─ controllers and background workers
       └─ Digitizer / Scope / MCA / HV interfaces
            ├─ IIO backend (current)
            │    └─ libiio → iiod → FPGA and converter IIO devices
            └─ gRPC backend (legacy)
                 └─ Engine/IDS gRPC + ZMQ DMA services

External workspace
  └─ nlab-modbus → Modbus TCP / ser2net → RS-485 instruments
```

The IIO implementation uses separate libiio contexts for GUI operations,
background polling, and blocking DMA reads. This is intentional: a blocking
refill and a control write must not share one remote context.

Important source locations:

| Path | Purpose |
|---|---|
| `src/nlab/hardware/digitizer/backends/iio_backend.py` | Scope and MCA IIO control plus DMA lifecycle |
| `src/nlab/hardware/digitizer/backends/iio_ids_backend.py` | IDS/HV IIO devices and diagnostics |
| `src/nlab/hardware/digitizer/dma.py` | IIO and legacy capture streamers and NDMA formats |
| `src/nlab/controllers/` | GUI coordination and acquisition state |
| `src/nlab/workers/` | Background polling and DMA workers |
| `hw_description/` | Driver sources, target API, architecture notes, and reference tools |
| `tests/` | Unit, lifecycle, format, settings, and GUI tests |

## Development

After editing a Qt Designer form or resource file, regenerate the UI modules:

```bash
python scripts/build_ui.py
```

After editing a legacy `.proto` file, regenerate its compatibility stubs:

```bash
python scripts/generate_proto.py
```

Run the test suite with an offscreen Qt platform on headless systems:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest
```

PowerShell equivalent:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
uv run pytest
```

Static checks used by the project are:

```bash
uv run ruff check .
uv run mypy src/nlab
```

Tests that require a real digitizer are kept separate from ordinary unit
tests. When running hardware checks, always stop an armed MCA/Scope in a
`finally` block so a failed script does not leave the target enabled.

## Standalone builds

Install the development dependencies and generate the UI/protobuf modules
before packaging.

Nuitka is used for release-style standalone directory builds:

```bash
python scripts/build_nuitka.py
```

The reviewed PyInstaller script produces a convenient single-file executable,
retains the required SciPy, Matplotlib, and Pillow stacks, embeds Windows
version information, and reports the artifact hash:

```bash
python scripts/build_pyinstaller_reviewed.py --clean
```

Output is `dist/nlab.exe` on Windows or `dist/nlab` on Linux. The original
[`scripts/build_pyinstaller.py`](scripts/build_pyinstaller.py) remains as a
fallback while the reviewed build is adopted.

Tagged releases trigger Windows and Ubuntu builds through the repository's
GitHub and Gitea workflows. Published artifacts are available from the
[Releases page](https://github.com/nucliflare/nlab-digitizer-community/releases).

## Known target-side limitations

- Superseded five-channel `vdpp_lm_frame` firmware cannot allocate a list-mode
  IIO buffer on the target Linux 5.15 stack and returns `EINVAL`; use the
  `opaque[16]` driver ABI described above.
- Sustained IIO Scope DMA on one tested channel-0 target can expose a Xilinx
  VDMA `DMA_INT_ERR`/`EOF_EARLY_ERR`. The client rejects partial records,
  releases buffer gates, and requires explicit recovery acknowledgement, but
  it cannot repair a target DMA fault.
- Remote IIO binary reads are informational while acquisition is running.
  For example, a four-part MCA histogram is transferred sequentially rather
  than as one atomic snapshot. Stop acquisition before reading a stable final
  result.

See [`CHANGELOG.md`](CHANGELOG.md) for release history and the current status
of hardware validation.

## Contributing

Bug reports, documentation improvements, hardware traces, and pull requests
are welcome. Please include the backend, firmware/IP version, device discovery
output, host operating system, and whether a result was reproduced with an
official libiio utility when reporting transport or DMA failures.

Submit changes against `main`, add focused tests where practical, and keep
driver-derived claims linked to the checked-in sources under
[`hw_description/`](hw_description/).

## License

MIT — see [`LICENSE`](LICENSE).

Copyright (c) 2026 Eastern Wall Technologies, Sp. z o. o.

The **Nuclear Lab Digitizer** name and EWT logo are trademarks of Eastern Wall
Technologies and are not covered by the MIT license.

## Related projects

- [`nlab-modbus`](https://github.com/nucliflare/nlab-modbus) — Python support for the RS-485 Modbus instruments
- [Eastern Wall Technologies](https://ewt.tech) — hardware manufacturer
