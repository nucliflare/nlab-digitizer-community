"""Pyqtgraph axis that labels raw MCA channel coordinates in keV or keVee."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg

from nlab.analysis.energy_calibration import EnergyCalibration
from nlab.views.plot_viewbox import set_frame_only


class CalibratedEnergyAxis(pg.AxisItem):  # type: ignore[misc]
    def __init__(self, orientation: str = "top") -> None:
        super().__init__(orientation=orientation)
        # The transformed values are already expressed in keV.  Pyqtgraph's
        # automatic SI scaling would otherwise turn a 1000-keV range into
        # "kkeV" even though tickStrings() supplies keV values directly.
        self.enableAutoSIPrefix(False)
        self._calibration: EnergyCalibration | None = None
        self._binning_index = 0
        self._stale = False
        # Uncalibrated, this axis only mirrors the raw channels on the opposite
        # side: keep it as a frame line until a calibration gives it meaning.
        set_frame_only(self, True)

    def set_calibration(
        self,
        calibration: EnergyCalibration | None,
        *,
        binning_index: int = 0,
        stale: bool = False,
    ) -> None:
        self._calibration = calibration
        self._binning_index = binning_index
        self._stale = stale
        if calibration is None:
            self.setLabel(text="")
            set_frame_only(self, True)
            self.setToolTip("No energy calibration is applied.")
        else:
            set_frame_only(self, False)
            quantity = calibration.energy_quantity
            self.setLabel(
                text=f"{quantity} (stale)" if stale else quantity,
                units=calibration.energy_unit,
            )
            self.setToolTip(
                "Calibration settings no longer match the MCA energy configuration."
                if stale
                else f"{calibration.model.capitalize()} MCA energy calibration "
                f"({calibration.energy_unit})"
            )
        self.picture = None
        self.update()

    def tickStrings(  # noqa: N802 - pyqtgraph API
        self,
        values: list[float],
        scale: float,
        spacing: float,
    ) -> list[str]:
        calibration = self._calibration
        if calibration is None:
            return [str(value) for value in super().tickStrings(values, scale, spacing)]
        energies = np.asarray(
            calibration.energy_at_binning(
                np.asarray(values, dtype=np.float64),
                self._binning_index,
            )
        )
        if values:
            first = float(values[0])
            energy_spacing = abs(
                float(calibration.energy_at_binning(first + spacing, self._binning_index))
                - float(calibration.energy_at_binning(first, self._binning_index))
            )
        else:
            energy_spacing = 1.0
        if not math.isfinite(energy_spacing) or energy_spacing <= 0:
            decimals = 2
        else:
            decimals = max(0, min(6, int(math.ceil(-math.log10(energy_spacing))) + 1))
        return [f"{float(energy):.{decimals}f}" for energy in energies]
