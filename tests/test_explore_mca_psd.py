from pathlib import Path
from runpy import run_path

import matplotlib
import numpy as np
import pytest
import uproot

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "notebooks" / "explore_mca_psd.py"


@pytest.mark.parametrize("names", [
    ("long_gate", "short_gate"),
    ("trapezoid_energy", "charge_energy"),
    ("energy", "short_energy"),
])
def test_explorer_loads_raw_fields_without_unsigned_subtraction(
    tmp_path: Path, names: tuple[str, str],
) -> None:
    script = run_path(str(SCRIPT))
    path = tmp_path / "events.root"
    timestamps = np.arange(4, dtype=np.uint64) + 2**63
    raw = {
        names[0]: np.array([1000, 1000, 0, 4000], dtype=np.uint16),
        names[1]: np.array([800, 1200, 0, 2000], dtype=np.uint16),
        "timestamp": timestamps,
    }
    with uproot.recreate(path) as root:
        root["metadata_json"] = '{"channel": 1}'
        root.mktree("events", {key: value.dtype for key, value in raw.items()})
        root["events"].extend(raw)
    before = path.read_bytes()

    events, energy, psd, metadata = script["load_psd"](path, entry_stop=3)

    np.testing.assert_array_equal(events["timestamp"], timestamps[:3])
    np.testing.assert_array_equal(energy, [1000, 1000, 0])
    np.testing.assert_allclose(psd, [0.2, -0.2, np.nan], equal_nan=True)
    assert metadata == {"channel": 1}
    fig, axes = script["plot_psd"](energy, psd, psd_range=(-0.3, 0.3), bins=(16, 16))
    try:
        fig.canvas.draw()
        assert len(axes) == 3
        # histtype='step' is a Polygon: each histogram height appears twice.
        assert axes[2].patches[0].get_xy()[:-1, 1].sum() / 2 == 2
        with pytest.raises(ValueError, match="No events"):
            script["plot_psd"](energy, psd, energy_range=(2000, 3000))
    finally:
        plt.close(fig)
    assert path.read_bytes() == before
