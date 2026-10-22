"""Run in Jupyter with %run notebooks/explore_mca_psd.py (from the repo root).

Then explore ``events`` (raw branch arrays), ``energy``, ``psd``, and ``metadata``.
Example: plot_psd(energy, psd, energy_range=(1000, 10000), psd_range=(0, 0.3))
Only the first MAX_EVENTS entries are loaded; set it to None for the whole file.
The ROOT file is opened read-only. No calibration or MCA binning shift is applied.
"""

# %% Settings / imports
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib.colors import LogNorm

FILENAME = Path(r"E:\2026_10_21_LS_EWT_PSD\Cs137_0cm_PuBe_0cm_npr.root")
TREE = "events"
MAX_EVENTS = 1_000_000  # None loads everything into RAM.


# %% Load raw branches and derive the app's PSD ratio (float before subtraction).
def load_psd(path=FILENAME, tree_name=TREE, entry_stop=MAX_EVENTS):
    with uproot.open(path) as root:
        tree = root[tree_name]
        print(f"{path}\nTree: {tree_name}; total entries: {tree.num_entries:,}")
        tree.show()
        metadata = json.loads(str(root["metadata_json"])) if "metadata_json" in root else {}
        for key in ("configuration_yaml", "event_schema", "capture_metadata_json"):
            if key in root:
                metadata[key] = str(root[key])
        events = tree.arrays(entry_stop=entry_stop, library="np", how=dict)
    for long_name, short_name in (
        ("long_gate", "short_gate"),
        ("trapezoid_energy", "charge_energy"),
        ("energy", "short_energy"),
    ):
        if long_name in events and short_name in events:
            break
    else:
        raise ValueError(f"No recognized long/short charge fields: {list(events)}")
    energy = events[long_name].astype(np.float64)
    short = events[short_name].astype(np.float64)
    psd = np.full(energy.shape, np.nan)
    np.divide(energy - short, energy, out=psd, where=energy > 0)
    print(f"Loaded {len(energy):,} entries; energy={long_name}, short={short_name}.")
    print(f"Zero-energy entries (including possible padding): {np.count_nonzero(energy <= 0):,}")
    return events, energy, psd, metadata


# %% Re-run this function with different ranges/bins, or pass sliced arrays.
def plot_psd(energy, psd, energy_range=None, psd_range=(-0.1, 0.6), bins=(512, 256)):
    keep = (energy > 0) & np.isfinite(energy) & np.isfinite(psd)
    for values, bounds in ((energy, energy_range), (psd, psd_range)):
        if bounds is not None:
            keep &= (values >= bounds[0]) & (values <= bounds[1])
    if not np.any(keep):
        raise ValueError("No events inside the selected energy/PSD ranges.")
    e, p = energy[keep], psd[keep]
    erange = energy_range or (float(e.min()), float(e.max()) + 1)
    prange = psd_range or (float(p.min()) - 0.01, float(p.max()) + 0.01)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].hist(e, bins=bins[0], range=erange, histtype="step", log=True)
    axes[0].set(xlabel="Long charge / energy (raw)", ylabel="Counts")
    image = axes[1].hist2d(e, p, bins=bins, range=(erange, prange), norm=LogNorm(), cmin=1)
    fig.colorbar(image[3], ax=axes[1], label="Counts")
    axes[1].set(xlabel="Long charge / energy (raw)", ylabel="PSD = 1 - short/long")
    axes[2].hist(p, bins=bins[1], range=prange, histtype="step")
    axes[2].set(xlabel="PSD = 1 - short/long", ylabel="Counts")
    fig.suptitle(f"{len(e):,} selected / {len(energy):,} loaded entries")
    return fig, axes


# %% %run leaves these variables/functions available in your notebook.
if __name__ == "__main__":
    events, energy, psd, metadata = load_psd(entry_stop=None)
    fig, axes = plot_psd(energy, psd)
    plt.show()
