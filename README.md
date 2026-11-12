# Nuclear Lab Digitizer — Community Edition

A desktop application for the Eastern Wall Technologies Digitizer 4.0, with
live Scope waveforms, MCA spectra, PSD, recording, and offline analysis.

## Install the IIO dependency

Both the release app and source installation need the **native libiio 0.x
runtime** on your computer.

**Windows:** Download [Analog Devices libiio v0.26](https://github.com/analogdevicesinc/libiio/releases/tag/v0.26),
install `libiio-0.26.ga0eca0d2-setup.exe`, then restart the app if it was open.

**Ubuntu:** Install the [IIO runtime](https://packages.ubuntu.com/noble/libiio0)
and command-line tools:

```bash
sudo apt update
sudo apt install libiio0 libiio-utils
```

## Run the app

### From a release

Download the package for your system from [GitHub Releases](https://github.com/nucliflare/nlab-digitizer-community/releases/latest)
or [Gitea Releases](https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/releases).
A separate Python installation is unnecessary.

- **Windows:** Extract `nlab-windows.zip` and open `nlab.exe`.
- **Ubuntu:** Extract and launch from a terminal:

  ```bash
  tar -xzf nlab-ubuntu.tar.gz
  chmod +x nlab
  ./nlab
  ```

### From source

Install Git, Python **3.12**, and [uv](https://github.com/astral-sh/uv), then run:

```bash
git clone https://github.com/nucliflare/nlab-digitizer-community.git
cd nlab-digitizer-community
uv sync --locked
uv run python scripts/build_ui.py
uv run nlab
```

Generate the UI files after a fresh clone or changes to the forms/resources.
For subsequent launches, run `uv run nlab` from the project directory.

## Connect a digitizer

1. Connect your computer and digitizer to the same network or use the board's
   USB network connection.
2. In the startup dialog, choose a scanned entry such as
   `nucliflare-xxxxx.local (192.168.10.128)`, or enter the board's IP manually.
3. Keep port **30431** unless your board uses a different port, select its
   channel count, and click **Connect**.

Choose **Offline / Replay Mode** to explore analysis tools without a board.
Use **Connection → Connect…** when you want to connect later.

## Start Scope and MCA

### View pulses with Scope

1. Open **Scope** and select your channel.
2. With acquisition stopped and **Record DMA frames** unchecked, run
   **Auto Setup**. Check its result, the signal polarity, and trigger settings.
3. Click **Start** to view waveforms and **Stop** to finish.

### Collect a spectrum with MCA

1. Open **MCA** for the same channel, with **Record list-mode DMA** unchecked.
2. Load a known configuration or set the pulse polarity, trigger threshold,
   shaping, and binning for your detector.
3. Click **Start** to accumulate a spectrum and **Stop** to finish.

For a first measurement, use the live viewers before enabling DMA recording.
The [acquisition guide](https://github.com/nucliflare/nlab-digitizer-community/wiki/acquisition_workflows)
covers recording and advanced workflows.

## Help and examples

- **Built-in help:** Open **Help → User Guide…** for the searchable, offline
  parameter reference. Press **F1**, then click a control or label for its
  explanation; **Esc** exits help selection.
- **[Use cases](https://github.com/nucliflare/nlab-digitizer-community/wiki/use_cases):**
  Find a workflow for calibration, peak analysis, PSD, coincidence, or replay.
- **[Project wiki](https://github.com/nucliflare/nlab-digitizer-community/wiki):**
  Detailed setup, measurement guides, file formats, troubleshooting, and development.
  Also available on [Gitea](https://apps.ewt.cloud:30008/nuclear-lab/nlab-digitizer-community-edition/wiki).
- **[Changelog](CHANGELOG.md):** Release history and hardware-validation limitations.

[MIT license](LICENSE). Copyright © 2026 Eastern Wall Technologies, Sp. z o. o.
The Nuclear Lab Digitizer name and EWT logo are EWT trademarks.
