# Nuclear Lab Digitizer — Community Edition

A desktop application for the Eastern Wall Technologies Digitizer 4.0, with
live Scope waveforms, MCA spectra, PSD, recording, and offline analysis.

## Install the IIO dependency

Both the release app and source installation need the **native libiio 0.x
runtime** on your computer.

**Windows:** Download [Analog Devices libiio v0.26](https://github.com/analogdevicesinc/libiio/releases/tag/v0.26),
install `libiio-0.26.ga0eca0d2-setup.exe`, then restart the app if it was open.

**Ubuntu / LMDE 7 (Debian 13):** Install the IIO runtime
([Ubuntu](https://packages.ubuntu.com/noble/libiio0),
[Debian 13](https://packages.debian.org/trixie/libiio0)) and command-line tools:

```bash
sudo apt update
sudo apt install libiio0 libiio-utils
```

## Run the app

### From a release

Download the package for your system from [GitHub Releases](https://github.com/nucliflare/nlab-digitizer-community/releases/latest).
A separate Python installation is unnecessary.

- **Windows:** Extract `nlab-windows.zip` and open `nlab.exe`.
- **Ubuntu:** Extract and launch from a terminal:

  ```bash
  tar -xzf nlab-ubuntu.tar.gz
  chmod +x nlab
  ./nlab
  ```

- **LMDE 7 / Debian 13 (x86_64):** Extract and launch from a terminal:

  ```bash
  tar -xzf nlab-lmde7-x86_64.tar.gz
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

**Settings > General Settings** lists its categories on the left (Appearance, Scope
display, Acquisition and monitoring, Acquisition files, Configuration persistence) and
shows the selected one on the right; **OK** applies the changes made on every page.

Choose the interface theme under **Settings > General Settings > Appearance**:
**EWT light** (default), **EWT dark** or **Classic** (the platform's native style and
the original colours). The choice applies the next time the application starts;
`NLAB_THEME=light|dark|classic` overrides it for one launch.

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

To check the baseline for noise and interference lines, select the **Periodic**
trigger and turn on **View → Scope Noise Spectrum**. It averages the power
spectrum of the received frames below the Scope plot, in ADC codes. A Scope frame
is short, so it cannot resolve mains or supply ripple: the finest bin spacing is
about 61 kHz. This is software analysis that has not yet been validated on live
hardware.

### Collect a spectrum with MCA

1. Open **MCA** for the same channel, with **Record list-mode DMA** unchecked.
2. Load a known configuration or set the pulse polarity, trigger threshold,
   shaping, and binning for your detector.
3. Click **Start** to accumulate a spectrum and **Stop** to finish.

For a first measurement, use the live viewers before enabling DMA recording.
DMA capture (Scope waveforms, Current periodic sums and MCA list mode) runs in
its own background process, so a busy display cannot slow the reader. If the
application closes unexpectedly during a capture, that process still stops the
acquisition and completes the file. **Stop** keeps receiving the frames the
hardware had already accepted until the stream goes quiet, so a recording keeps its
tail; data left over from an interrupted earlier session is detected and discarded
at the start of the next one.
Current periodic sums record to a compact binary `.bin` file (default) or to
HDF5 (`.h5`), in the measurement folder from General Settings. Leave the
filename blank for `ch0_current_<timestamp>.bin` on Start, or enter your own
name or path. Saving runs in the background and asks before
replacing an existing file. **Time [s]** stops a Current acquisition after that
many seconds; 0 runs until **Stop**.

A name you choose for a Scope DMA, MCA list-mode or Current sum recording is used
as given. To add the start time to it as well, check **Settings > General Settings >
Acquisition files > Add the start time to chosen file names**: `detector.bin` then
records as `detector_20261123_101500.bin`, and an existing file is never replaced.
In a series the run number follows the time (`detector_20261123_101500_001.bin`).

### Repeat a measurement

Check **Settings → Enable Series** to enable **Repeats** in Scope, MCA and Current.
Set **Time [s]** above 0 and the number of runs, then click **Start** once: each
run stops on its time limit, and the next starts after the previous one has fully
stopped (including the DMA drain). **Stop**, an error or a lost connection ends the
series. While the option is checked, recordings keep one name with a run number,
such as `run_001.bin`, `run_002.bin`: the file you chose, or the generated default.
Existing files are never replaced. Each MCA run also saves its spectrum as a CSV
with the same name, for example `run_001.csv`; with **Record list-mode DMA**
checked, every run also writes its own complete list-mode file beside it
(`run_001.bin`, `run_002.bin`, …).

The [acquisition guide](https://github.com/nucliflare/nlab-digitizer-community/wiki/acquisition_workflows)
covers recording and advanced workflows.

## Help and examples

- **Built-in help:** Open **Help → User Guide…** for the searchable, offline
  parameter reference. Press **F1**, then click a control or label for its
  explanation; **Esc** exits help selection. Its **Files** topics describe the
  binary Scope waveform, MCA list-mode and Current sum formats byte by byte,
  with example Python readers.
- **[Use cases](https://github.com/nucliflare/nlab-digitizer-community/wiki/use_cases):**
  Find a workflow for calibration (including keVee Compton-edge calibration of
  organic scintillators), peak analysis, PSD, coincidence, or replay.
- **[Project wiki](https://github.com/nucliflare/nlab-digitizer-community/wiki):**
  Detailed setup, measurement guides, file formats, troubleshooting, and development.
- **[Changelog](CHANGELOG.md):** Release history and hardware-validation limitations.

[MIT license](LICENSE). Copyright © 2026 Eastern Wall Technologies, Sp. z o. o.
The Nuclear Lab Digitizer name and EWT logo are EWT trademarks.
