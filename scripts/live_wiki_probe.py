"""Temporary MCA probe for live Current/Coincidence wiki screenshots."""

from __future__ import annotations

import time

import numpy as np
from scipy.signal import find_peaks

from nlab.hardware.digitizer import Digitizer


def main() -> None:
    uri = "ip:192.168.10.128:30431"
    digitizers: list[Digitizer] = []
    try:
        for channel in (0, 1):
            digitizer = Digitizer.from_iio(channel=channel, uri=uri, with_ids=False)
            digitizers.append(digitizer)
            mca = digitizer.mca
            mca.stop()
            mca.set_pulse_polarity(0)
            for threshold in (-1536, -2048, -2560, -3072):
                mca.stop()
                mca.set_trigger_level(threshold)
                mca.set_energy_bin(4)
                mca.set_energy_bin(5)
                mca.start()
                print(f"CH{channel} threshold {threshold}: acquiring for 5 s", flush=True)
                time.sleep(5)
                mca.stop()
                histogram = np.asarray(mca.acquire_spectrum())
                coarse = histogram.reshape(-1, 64).sum(axis=1)
                peaks, properties = find_peaks(
                    coarse,
                    distance=5,
                    prominence=max(10.0, 0.015 * float(coarse.max(initial=0))),
                )
                ranked = sorted(
                    zip(peaks, properties["prominences"], strict=True),
                    key=lambda item: item[1],
                    reverse=True,
                )[:12]
                print(
                    f"CH{channel} threshold {threshold}: "
                    f"events={int(histogram.sum())}; "
                    f"peaks={[(int(p * 64 + 32), round(float(v), 1)) for p, v in ranked]}",
                    flush=True,
                )
    finally:
        for digitizer in digitizers:
            try:
                digitizer.mca.stop()
                digitizer.scope.stop()
            finally:
                digitizer.close()


if __name__ == "__main__":
    main()
