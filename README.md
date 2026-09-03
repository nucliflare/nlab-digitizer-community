# Nuclear Lab Digitizer — Community Edition

A real-time data acquisition and instrument control desktop application for the
**EWT digitizer board**. Provides oscilloscope, multi-channel analyser (MCA),
high-voltage power supply control, and external Modbus instrument monitoring in
a single dockable GUI.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

---

## Hardware requirements

| Component | Details |
|---|---|
| EWT digitizer board | 2-channel DPP + IDS subsystem |
| Network connection | Static IP, default `192.168.10.20` |
| gRPC — DPP service | port `50050` |
| gRPC — IDS (HV) service | port `50040` |
| ZMQ DMA — Scope streaming | ports `50152+` |
| ZMQ DMA — MCA streaming | ports `50162+` |
| RS-485 Modbus devices (optional) | SiPM bias board, Geiger probe, PMT HV supply bridged via ser2net on ports `5001`/`5002` |

The application connects to the board over Ethernet. All services run on the
same host; only the ports differ.

---

## Features

### Scope
- Real-time waveform display with configurable trigger (level, mode, DAC offset)
- Persistence and raw display modes with adjustable refresh rate
- Single-frame acquisition
- DMA continuous streaming to binary file with configurable duration

### MCA (Multi-Channel Analyser)
- Live pulse-height histogram with logarithmic Y-axis option
- Configurable energy bin, shaping, and threshold parameters
- ROI (region of interest) overlay with integrated count statistics
- List-mode DMA streaming to binary or HDF5 file

### Power Supply
- Independent SiPM bias and PMT high-voltage control
- Temperature compensation (analog and digital sensor modes)
- Real-time HV voltage trend plot with configurable time window
- Safe ramp-down on application close

### External Modbus Devices
- Auto-discovery of daisy-chained RS-485 devices over the digitizer's ser2net bridge
- Generic register table view for each device:
  - **Settings** — writable holding registers (HV setpoint, enable, compensation)
  - **Telemetry** — live-polled input registers with per-register trend plotting
- Supports SiPM bias board, Geiger-Mueller probe, and PMT HV power supply

### General
- Undockable, floatable channel panels (dual-monitor friendly)
- Dock layout persisted across sessions
- Save/load all channels, hardware, and application-only settings in one YAML file
- Convert binary DMA recordings to HDF5 via the File menu
- In-app system log with colour-coded levels

---

## Installation

### Prerequisites

- Python 3.12 or newer
- [`uv`](https://github.com/astral-sh/uv) package manager (`pip install uv`)
- Qt 6.7 runtime libraries (bundled via PySide6)

### From source

```bash
git clone https://github.com/nucliflare/nlub-digitizer-community.git
cd nlub-digitizer-community
# Create virtual environment and install all dependencies
uv venv
uv pip install -e ".[dev]"

# Build the Qt Designer .ui forms and compile Qt resources
python scripts/build_ui.py

# Generate gRPC stubs (required once after clone)
python scripts/generate_proto.py
```

### Running

```bash
uv run nlab
# or
python -m nlab.main

# Pre-fill the connection dialog and apply a saved configuration after connecting
uv run nlab --config experiment.yaml --ip 192.168.10.128 --port 30431
```

The connection dialog appears on startup. Enter the digitizer's IP address and
gRPC port (default `192.168.10.20:50050`), then click **Connect**.
`--config` also supplies the saved backend, address, port, and channel count to
the dialog; explicit `--ip` and `--port` arguments override the file values.

---

## Packaged builds

Pre-built standalone distributions are attached to each
[release](https://github.com/nucliflare/nlub-digitizer-community/releases) as zip archives.
No Python installation required.

| Platform | Archive | Run |
|---|---|---|
| Windows | `nlab-windows.zip` | `nlab.exe` |
| Linux | `nlab-ubuntu.tar.gz` | `./nlab-app` |

Extract the **entire archive** — the executable depends on DLLs and data files
in the same folder and will not run if moved out.

> **Windows note:** Windows 11 Smart App Control may block unsigned executables
> on first run. Right-click `nlab.exe` → Properties → Unblock, or add the
> folder to Windows Defender's exclusion list.

---

## Programmatic API

The hardware communication layer is fully usable without the GUI:

```python
from nlab.hardware.digitizer.digitizer import Digitizer

d = Digitizer.from_grpc(channel=1, hostname="192.168.10.20")

# Scope
d.scope.set_trigger_level(5000)
d.scope.set_frame_samples(1024)

# MCA
d.mca.set_energy_bin(4)
d.mca.start()
spectrum = d.mca.get_histogram()
d.mca.stop()

# HV supply
d.hv.set_hv_voltage(800.0)
print(d.hv.get_hv_adc_voltage())

d.close()
```

External Modbus devices via [`nlab-modbus`](https://github.com/nucliflare/nlab-modbus):

```python
from nlab.hardware.modbus_devices import ExternalDevices
from nlab_modbus.core.enums import DeviceType

ext = ExternalDevices()
devices = ext.discover("192.168.10.20")          # scans ports 5001 and 5002

geiger = next(d for d in devices if d.device_type == DeviceType.GEIGER)
print(geiger.read_snapshot())                     # {dose_rate, pulse_rate, hv_voltage, ...}

ext.close()
```

---

## Development

### Project structure

```
nlab-community/
├── src/nlab/
│   ├── hardware/
│   │   ├── digitizer/          # Scope, MCA, HV backends (gRPC + ZMQ DMA)
│   │   │   ├── backends/       # gRPC client wrappers
│   │   │   ├── scope.py
│   │   │   ├── mca.py
│   │   │   ├── hv.py
│   │   │   └── dma.py          # ZMQ streaming
│   │   ├── grpc/generated/     # protoc-generated gRPC stubs
│   │   └── modbus_devices.py   # nlab-modbus integration
│   ├── controllers/            # View + controller widgets (MVC)
│   ├── workers/                # Background QThread workers
│   ├── views/                  # Utility views (log panel, plot viewbox)
│   ├── ui/                     # Generated UI modules (build_ui.py output)
│   └── utils/                  # Settings I/O, DMA converter, Windows icon helper
├── forms/                      # Qt Designer .ui source files
├── resources/                  # Icons, images (.qrc)
├── scripts/                    # Build and code-generation scripts
├── docs/                       # Quarto data analysis examples
└── .gitea/workflows/           # CI pipeline (build + release, Gitea Actions)
```

### Regenerating UI and gRPC stubs

```bash
# After editing any .ui file in forms/ or .qrc in resources/
python scripts/build_ui.py

# After editing any .proto file
python scripts/generate_proto.py
```

Generated files (`src/nlab/ui/ui_*.py`, `resources_rc.py`, gRPC stubs) are
gitignored — always regenerate after checkout.

### Building a standalone distribution

```bash
# Nuitka standalone (recommended for release)
python scripts/build_nuitka.py
# Output: dist/main.dist/  (ship the whole folder)

# PyInstaller onefile (simpler, faster build)
python scripts/build_pyinstaller.py
# Output: dist/nlab.exe  (Windows) / dist/nlab  (Linux)
```

### Running tests

```bash
uv run pytest

# Headless (no display required)
QT_QPA_PLATFORM=offscreen uv run pytest
```

### Code quality

```bash
uv run ruff check .          # linting
uv run mypy src/nlab         # type checking
```

---

## Architecture overview

```
┌──────────────────────────────────────────────────────┐
│                    PySide6 GUI                       │
│  ┌──────────┐ ┌─────────┐ ┌───────┐  ┌───────────┐   │
│  │  Scope   │ │   MCA   │ │  PSU  │  │ External  │   │
│  │Controller│ │Controler│ │ Ctrl  │  │  Device   │   │
│  └────┬─────┘ └────┬────┘ └───┬───┘  └─────┬─────┘   │
│       │            │          │            │         │
│  ┌────▼────────────▼──────────▼────┐  ┌────▼──────┐  │
│  │         QThread Workers         │  │  nlab-    │  │
│  │  (ScopeWorker, McaWorker, Dma,  │  │  modbus   │  │
│  │   PsuWorker, ExternalDevice)    │  │  Manager  │  │
│  └────┬─────────────┬──────────────┘  └────┬──────┘  │
└───────│─────────────│──────────────────────│─────────┘
        │             │                      │
   gRPC │        ZMQ  │ DMA            Modbus│TCP
   ─────┘         ────┘                 ─────┘
        │                                    │
┌───────▼────────────────────────────────────▼──────────┐
│                 EWT Digitizer Board                   │
│   DPP :50050   IDS :50040   DMA :50152+   ser2net     │
│                                           :5001/:5002 │
│                                      ┌────┴──────────┐│
│                                      │ RS-485 Bus    ││
│                                      │ SiPM / Geiger ││
│                                      │ PMT HV PSU    ││
│                                      └───────────────┘│
└───────────────────────────────────────────────────────┘
```

---

## Contributing

1. Fork and clone the repository
2. Install dev dependencies: `uv pip install -e ".[dev]"`
3. Run `python scripts/build_ui.py` and `python scripts/generate_proto.py`
4. Make your changes, add tests where applicable
5. Run `ruff check .` and `mypy src/nlab` before opening a PR
6. Submit a pull request against `main`

All contributions are welcome — bug reports, feature requests, documentation
improvements, and code.

---

## License

MIT — see [LICENSE](LICENSE).

Copyright (c) 2026 Eastern Wall Technologies.

The **Nuclear Lab Digitizer** name and EWT logo are trademarks of Eastern Wall
Technologies and are not covered by the MIT license.

---

## Related

- [nlab-modbus](https://github.com/nucliflare/nlab-modbus) — Python driver library for the RS-485 Modbus instruments
- [Eastern Wall Technologies](https://ewt.tech) — hardware manufacturer
