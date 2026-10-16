"""Offline executable smoke check using explicitly synthetic NDMA captures."""

from __future__ import annotations

import tempfile
from pathlib import Path


def check_root_conversion() -> None:
    """Open real conversion dialogs and verify both ROOT TTrees after writing.

    QApplication must already exist. This never creates an IIO context or
    connects to a digitizer; it checks the frozen Qt/ROOT stack.
    All inputs and outputs are synthetic and live only in a temporary directory.
    """
    import numpy as np
    import uproot

    from nlab.hardware.digitizer.dma import (
        _LM_EVENT_DTYPE,
        FILE_HEADER_STRUCT,
        FILE_MAGIC,
        FILE_VERSION,
        IIO_LM_FILE_VERSION,
        IIO_LM_FRAME_RECORDS,
    )
    from nlab.views.root_conversion_dialog import RootConversionDialog

    events = np.zeros(IIO_LM_FRAME_RECORDS, dtype=_LM_EVENT_DTYPE)
    events[0] = (2, 200, -4096, 65535, 32768, 2**63 + 17)
    waveforms = np.zeros(2, dtype=[("timestamp", "<u8"), ("samples", "<i2", (5,))])
    waveforms["timestamp"] = [2**63 + 1, 2**63 + 2]
    waveforms["samples"] = [-32768, -1, 0, 1, 32767]
    with tempfile.TemporaryDirectory(prefix="nlab-root-check-") as directory:
        folder = Path(directory)
        for kind, version, frame_samples, records in (
            ("events", IIO_LM_FILE_VERSION, 0, events),
            ("waveforms", FILE_VERSION, 9, waveforms),
        ):
            source, destination = folder / f"{kind}.bin", folder / f"{kind}.root"
            source.write_bytes(
                FILE_HEADER_STRUCT.pack(FILE_MAGIC, version, 0, 0, 0.0, frame_samples)
                + records.tobytes(),
            )
            dialog = RootConversionDialog(source, destination)
            try:
                dialog.exec()
                result = dialog.conversion_result
                if dialog.error or result is None or result.count != len(records):
                    raise RuntimeError(f"Frozen {kind} conversion failed: {dialog.error}")
                with uproot.open(destination) as root:
                    tree = root[kind]
                    if tree.classname != "TTree" or tree.num_entries != len(records):
                        raise RuntimeError(f"Incorrect frozen {kind} ROOT tree")
                    for name in records.dtype.names or ():
                        np.testing.assert_array_equal(
                            tree[name].array(library="np"), records[name],
                        )
            finally:
                dialog.deleteLater()
