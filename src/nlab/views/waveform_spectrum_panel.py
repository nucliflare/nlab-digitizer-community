"""Spectrum tab of the Waveform Analysis Workbench (offline, never live hardware)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.waveform_file import WaveformFileIndex
from nlab.analysis.waveform_frame_spectrum import (
    FrameSpectrumRequest,
    FrameSpectrumResult,
    SpectrumRegion,
)
from nlab.analysis.waveform_psd import WaveformPsdSettings
from nlab.analysis.waveform_spectrum import (
    UNIT_LABELS,
    SpectrumUnit,
    SpectrumWindow,
    Spur,
    as_density,
    find_spurs,
    format_frequency,
    line_threshold_db,
    parseval_check,
    plotted_bins,
    to_display_units,
)
from nlab.views import theme
from nlab.views.frequency_axis import set_frequency_axis
from nlab.views.plot_viewbox import ModifierZoomViewBox
from nlab.workers.waveform_spectrum_worker import WaveformSpectrumWorker

_NOT_VALIDATED = "Software analysis; not validated on live hardware."
_PULSE_LABEL = "Mean pulse |FFT|² [dB re its peak]"


@dataclass(frozen=True)
class WaveformSpectrumSource:
    """The workbench state a spectrum is computed from."""

    index: WaveformFileIndex
    source_index: int
    sample_period_ns: float
    settings: WaveformPsdSettings
    frame_samples: int
    stride: int
    maximum_events: int
    selected_frames: np.ndarray | None


class WaveformSpectrumPanel(QWidget):
    """Noise PSD of recorded frames, with the mean-pulse spectrum on a second axis.

    ``source_provider`` returns the current workbench state or a message
    explaining why nothing can be computed yet.
    """

    def __init__(
        self,
        source_provider: Callable[[], WaveformSpectrumSource | str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source_provider = source_provider
        self._worker: WaveformSpectrumWorker | None = None
        self._generation = 0
        self.result: FrameSpectrumResult | None = None
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.population_combo = QComboBox(self)
        self.population_combo.addItem("All frames (stride / maximum)", "all")
        self.population_combo.addItem("Selected PSD population", "selected")
        self.population_combo.setToolTip(
            "All frames honours Event stride and Maximum events. The selected population "
            "uses the waveforms inside the PSD selection region."
        )
        self.region_combo = QComboBox(self)
        self.region_combo.addItem("Pretrigger baseline (noise)", SpectrumRegion.PRETRIGGER.value)
        self.region_combo.addItem("Whole frame (pulse + noise)", SpectrumRegion.FRAME.value)
        self.region_combo.setToolTip(
            "The pretrigger baseline region (blue) holds no pulse, so it estimates noise, "
            "but its few samples give coarse bins. Whole frames contain the pulses."
        )
        self.window_combo = QComboBox(self)
        for window, text in (
            (SpectrumWindow.HANN, "Hann"), (SpectrumWindow.BLACKMAN_HARRIS, "Blackman-Harris"),
            (SpectrumWindow.FLATTOP, "Flat-top"), (SpectrumWindow.BOXCAR, "Rectangular"),
        ):
            self.window_combo.addItem(text, window.value)
        self.unit_combo = QComboBox(self)
        for unit in SpectrumUnit:
            self.unit_combo.addItem(UNIT_LABELS[unit], unit.value)
        self.log_checkbox = QCheckBox("Log frequency", self)
        self.pulse_checkbox = QCheckBox("Mean pulse spectrum", self)
        self.pulse_checkbox.setChecked(True)
        self.pulse_checkbox.setToolTip(
            "Also show |FFT|² of the mean baseline-subtracted pulse on the right axis, in dB "
            "relative to its own peak. A different quantity from the noise PSD."
        )
        self.compute_button = QPushButton("Compute spectrum", self)
        for label, widget in (
            ("Frames:", self.population_combo), ("Region:", self.region_combo),
            ("Window:", self.window_combo), ("Units:", self.unit_combo),
        ):
            controls.addWidget(QLabel(label, self))
            controls.addWidget(widget)
        controls.addWidget(self.log_checkbox)
        controls.addWidget(self.pulse_checkbox)
        controls.addWidget(self.compute_button)
        controls.addStretch()
        layout.addLayout(controls)

        self.plot_widget = pg.PlotWidget(self, viewBox=ModifierZoomViewBox())
        self.plot_widget.setBackground(theme.colors().plot_background)
        plot = self.plot_widget.getPlotItem()
        assert plot is not None
        self._plot = plot
        plot.showGrid(x=True, y=True, alpha=0.2)
        set_frequency_axis(plot, log=False)
        self.legend = plot.addLegend(offset=(-10, 10))
        self.noise_curve = plot.plot(pen=pg.mkPen("#277da1", width=1), name="Noise PSD (left)")
        self.markers = plot.plot(
            pen=None, symbol="t", symbolSize=9, symbolBrush=pg.mkBrush("#c0563b"), symbolPen=None,
        )
        # A second view box gives the mean-pulse spectrum its own right axis.
        self.pulse_view = pg.ViewBox()
        plot.scene().addItem(self.pulse_view)
        plot.getAxis("right").linkToView(self.pulse_view)
        self.pulse_view.setXLink(plot)
        plot.getAxis("right").setLabel(_PULSE_LABEL)
        self.pulse_curve = pg.PlotCurveItem(
            pen=pg.mkPen("#8c2d91", width=1.5, style=Qt.PenStyle.DashLine),
        )
        self.pulse_view.addItem(self.pulse_curve)
        plot.getViewBox().sigResized.connect(self._sync_pulse_view)
        self._sync_pulse_view()

        message_page = QWidget(self)
        message_layout = QVBoxLayout(message_page)
        message_layout.addStretch()
        self.message = QLabel(message_page)
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        message_layout.addWidget(self.message)
        message_layout.addStretch()
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self.plot_widget)
        self.stack.addWidget(message_page)
        layout.addWidget(self.stack, 1)
        self.summary = QLabel(self)
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)

        self.compute_button.clicked.connect(self.compute)
        for combo in (self.unit_combo,):
            combo.currentIndexChanged.connect(self._render)
        for checkbox in (self.log_checkbox, self.pulse_checkbox):
            checkbox.toggled.connect(self._render)
        self._render()

    # -- computation ---------------------------------------------------
    def compute(self) -> None:
        """Start a spectrum of the current workbench state; replaces a running one."""
        source = self._source_provider()
        if isinstance(source, str):
            self._show_message(source)
            return
        selected = self.population_combo.currentData() == "selected"
        frames = source.selected_frames if selected else None
        if selected and (frames is None or not len(frames)):
            self._show_message(
                "Select a population in the PSD matrix first (Waveform selection tools)."
            )
            return
        settings = source.settings
        try:
            request = FrameSpectrumRequest(
                sample_period_ns=source.sample_period_ns,
                frame_samples=source.frame_samples,
                region=SpectrumRegion(self.region_combo.currentData()),
                baseline_start=settings.baseline_start,
                baseline_end=settings.baseline_end,
                polarity=settings.polarity,
                baseline_method=settings.baseline_method,
                window=SpectrumWindow(self.window_combo.currentData()),
                mean_pulse=self.pulse_checkbox.isChecked(),
            )
        except ValueError as exc:
            self._show_message(str(exc))
            return
        self.stop_sync()
        self._generation += 1
        population = (
            f"Selected population ({len(frames):,} frames)" if frames is not None
            else f"All frames (stride {source.stride}, maximum "
            f"{source.maximum_events or 'all'})"
        )
        worker = WaveformSpectrumWorker(
            self._generation, source.index, source_index=source.source_index,
            request=request, frames=frames, stride=source.stride,
            maximum_events=source.maximum_events, population=population,
        )
        worker.computed.connect(self._computed)
        worker.failed.connect(self._failed)
        worker.progressed.connect(self._progressed)
        self._worker = worker
        self.summary.setText("Reading frames…")
        if self.result is None:
            self._show_message("Computing the spectrum…")
        worker.start()

    def stop_sync(self) -> None:
        """Cancel a running computation and wait for its thread."""
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.cancel()
            worker.wait()

    def clear(self) -> None:
        """Forget the result, e.g. when the file or source changes."""
        self.stop_sync()
        self._generation += 1
        self.result = None
        self._render()

    def _progressed(self, generation: int, percent: int) -> None:
        if generation == self._generation:
            self.summary.setText(f"Reading frames… {percent}%")

    def _failed(self, generation: int, message: str) -> None:
        if generation == self._generation:
            self.result = None
            self._render()
            self.summary.setText(message)

    def _computed(self, generation: int, result: FrameSpectrumResult) -> None:
        if generation == self._generation:
            self.result = result
            self._render()

    # -- display -------------------------------------------------------
    def _sync_pulse_view(self) -> None:
        plot = self._plot
        self.pulse_view.setGeometry(plot.getViewBox().sceneBoundingRect())
        self.pulse_view.linkedViewChanged(plot.getViewBox(), self.pulse_view.XAxis)

    def _show_message(self, text: str) -> None:
        self.message.setText(text)
        self.stack.setCurrentIndex(1)

    def _render(self, *_args: object) -> None:
        plot = self._plot
        unit = SpectrumUnit(self.unit_combo.currentData())
        log_frequency = self.log_checkbox.isChecked()
        plot.setLogMode(x=log_frequency, y=unit != SpectrumUnit.DB)
        set_frequency_axis(plot, log=log_frequency)
        quantity = "ASD" if unit == SpectrumUnit.ASD else "PSD"
        plot.setLabel("left", f"Noise {quantity} [{UNIT_LABELS[unit]}]")
        self.noise_curve.clear()
        self.markers.clear()
        self.pulse_curve.clear()
        self.legend.removeItem(self.pulse_curve)
        result = self.result
        noise = result.noise if result is not None else None
        if result is None or noise is None:
            self._show_message(
                result.message if result is not None
                else "Load a waveform file, choose frames and region, and press Compute spectrum."
            )
            self.summary.setText("" if result is None else result.message)
            return
        self.stack.setCurrentIndex(0)
        # Drawn as a density: the even-N Nyquist bin doubled (an even/odd sample
        # offset is a line exactly there), DC omitted.
        values = to_display_units(as_density(noise.psd, noise.n), unit)
        bins = plotted_bins(noise.n)
        shown = values[bins]
        if unit != SpectrumUnit.DB:
            shown = np.maximum(shown, np.finfo(np.float64).tiny)
        self.noise_curve.setData(noise.frequencies[bins], shown)
        spurs: list[Spur] = []
        if noise.frames >= 2:
            spurs = find_spurs(
                noise, threshold_db_over_median=line_threshold_db(noise.averages),
                max_count=12,
            )
        if spurs:
            marks = values[[spur.bin for spur in spurs]]
            if unit != SpectrumUnit.DB:
                marks = np.maximum(marks, np.finfo(np.float64).tiny)
            self.markers.setData([spur.frequency for spur in spurs], marks)
        show_pulse = (
            self.pulse_checkbox.isChecked()
            and result.pulse_frequencies is not None and result.pulse_db is not None
        )
        plot.showAxis("right", show_pulse)
        if show_pulse:
            assert result.pulse_frequencies is not None and result.pulse_db is not None
            pulse_bins = plotted_bins(result.request.frame_samples)
            frequencies = result.pulse_frequencies[pulse_bins]
            # The second view box is outside PlotItem's log transform.
            x = np.log10(frequencies) if log_frequency else frequencies
            self.pulse_curve.setData(x, result.pulse_db[pulse_bins])
            self.legend.addItem(self.pulse_curve, "Mean pulse |FFT|² (right, dB re peak)")
            self.pulse_view.enableAutoRange()
        plot.enableAutoRange()
        self.summary.setText(self._summary(result, spurs, unit))

    @staticmethod
    def _summary(result: FrameSpectrumResult, spurs: list[Spur], unit: SpectrumUnit) -> str:
        noise = result.noise
        assert noise is not None
        request = result.request
        start, stop = request.segment
        lines = [
            f"{result.population}: {result.frames:,} frames used, {result.skipped:,} skipped "
            f"(shorter than {request.frame_samples:,} samples or not finite).",
            f"{request.region.value.capitalize()} samples {start}–{stop - 1} · fs "
            f"{format_frequency(noise.fs)} · N {noise.n:,} · Δf {format_frequency(noise.delta_f)}"
            f" · ENBW {format_frequency(noise.enbw)} ({noise.window.value}) · "
            f"±{100 * noise.relative_std:.1f}% per bin · Parseval {parseval_check(noise):.3f}",
            result.message,
        ]
        if result.pulse_db is not None:
            lines.append(
                "Mean pulse: |FFT|² of the mean of the same frames after baseline removal and "
                "polarity normalisation (rectangular window), in dB re its peak. It is the "
                "pulse's energy spectrum, not a noise level: compare shapes, not heights."
            )
        if spurs:
            lines.append("Strongest lines: " + "; ".join(
                f"{format_frequency(spur.frequency)} "
                f"({to_display_units(np.array([spur.level]), unit)[0]:.3g} "
                f"{UNIT_LABELS[unit]}, +{spur.excess_db:.0f} dB)"
                for spur in spurs
            ))
        lines.append(_NOT_VALIDATED)
        return "\n".join(lines)
