from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from nlab.hardware.digitizer.scope import PARAMETER_SPECS, RangeSpec, Scope, ScopeParam, TriggerMode

log = logging.getLogger(__name__)

_NO_CLEAR_PULSE = (
    "no clear unipolar pulse was found; check the input signal and try again"
)
_PULSE_SHAPE_MIN_POINTS = 2
_SIGNAL_PROBE_ATTEMPTS = 8
_SIGNAL_PROBE_WAIT_S = 1.0
_VERIFICATION_WAIT_S = 2.0


class AutoSetupCancelledError(RuntimeError):
    """Raised internally when an Auto Setup cancellation is observed."""


@dataclass(frozen=True)
class ScopeAutoSetupResult:
    dac_value: int
    trigger_level: int
    trigger_mode: TriggerMode
    baseline: float
    noise_sigma: float
    pulse_amplitude: float
    verified: bool
    frame: np.ndarray
    dac_slope: float | None = None  # Measured raw ADC counts per baseline DAC unit.


@dataclass(frozen=True)
class _ScopeState:
    enabled: bool
    dac_value: int
    trigger_level: int
    trigger_mode: TriggerMode
    frame_samples: int
    pretrigger_samples: int


@dataclass(frozen=True)
class _SignalEstimate:
    baseline: float
    noise_sigma: float
    low: float
    high: float
    polarity: int
    amplitude: float


def _range_spec(parameter: ScopeParam) -> RangeSpec:
    spec = PARAMETER_SPECS[parameter]
    assert isinstance(spec, RangeSpec)
    return spec


def _estimate_baseline(frames: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(frame, dtype=np.int16) for frame in frames])
    if values.size < 16:
        raise RuntimeError("the scope viewer returned too few samples")

    integer_values = values.astype(np.int32)
    histogram = np.bincount(integer_values + 32768, minlength=65536)
    # The dominant narrow peak represents the quiet baseline even when a
    # triggered pulse occupies a substantial part of the frame.
    smoothed = np.convolve(histogram, np.ones(33, dtype=np.int64), mode="same")
    peak = int(np.argmax(smoothed)) - 32768
    local = integer_values[np.abs(integer_values - peak) <= 384]
    if local.size < max(8, values.size // 100):
        local = integer_values
    baseline = float(np.median(local))
    mad = float(np.median(np.abs(local - baseline)))
    noise_sigma = max(1.0, 1.4826 * mad)
    return baseline, noise_sigma


def _estimate_signal(frames: list[np.ndarray]) -> _SignalEstimate:
    baseline, noise_sigma = _estimate_baseline(frames)
    detection_floor = max(256.0, 8.0 * noise_sigma)

    amplitudes: dict[int, list[float]] = {-1: [], 1: []}
    for frame in frames:
        integer_values = np.asarray(frame, dtype=np.int16).astype(np.int32)
        for polarity in (-1, 1):
            deviation = polarity * (integer_values - baseline)
            amplitude = float(np.max(deviation, initial=0))
            opposite = float(np.max(-deviation, initial=0))
            if amplitude < detection_floor or amplitude < 1.35 * opposite:
                continue

            # A real viewer pulse spans adjacent 8 ns averaged points. Test
            # its shape below the peak-detection floor so a narrow genuine
            # pulse is retained while a single corrupt sample is rejected.
            shape_floor = max(64.0, 4.0 * noise_sigma, 0.20 * amplitude)
            if _longest_true_run(deviation >= shape_floor) < _PULSE_SHAPE_MIN_POINTS:
                continue
            amplitudes[polarity].append(amplitude)

    detected_polarities = [
        polarity for polarity, values in amplitudes.items() if values
    ]
    if not detected_polarities:
        raise RuntimeError(_NO_CLEAR_PULSE)
    if len(detected_polarities) > 1:
        raise RuntimeError("pulse polarity was inconsistent during Auto Setup")

    polarity = detected_polarities[0]
    amplitude = float(np.median(amplitudes[polarity]))
    low = baseline - amplitude if polarity < 0 else baseline
    high = baseline + amplitude if polarity > 0 else baseline

    return _SignalEstimate(
        baseline=baseline,
        noise_sigma=noise_sigma,
        low=float(low),
        high=float(high),
        polarity=polarity,
        amplitude=amplitude,
    )


def _longest_true_run(mask: np.ndarray) -> int:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return max((int(end - start) for start, end in zip(starts, ends)), default=0)


class ScopeAutoSetupProcedure:
    """Synchronous Auto Setup algorithm, intended to run in a worker thread."""

    def __init__(
        self,
        scope: Scope,
        *,
        cancelled: Callable[[], bool] = lambda: False,
        progress: Callable[[str], None] = lambda _message: None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._scope = scope
        self._cancelled = cancelled
        self._progress = progress
        self._sleep = sleep

    def run(self) -> ScopeAutoSetupResult:
        state = self._read_state()
        if state.enabled:
            raise RuntimeError("stop the scope before running Auto Setup")
        if self._scope.get_dma_enable():
            raise RuntimeError("disable scope DMA before running Auto Setup")

        try:
            self._check_cancelled()
            self._scope.stop()
            survey_samples = min(
                int(_range_spec(ScopeParam.FRAME_SAMPLES).max_val),
                max(4096, state.frame_samples),
            )
            survey_samples -= survey_samples % 4
            if survey_samples != state.frame_samples:
                self._scope.set_frame_samples(survey_samples)

            self._progress("Auto Setup: measuring the baseline response...")
            initial_frames = self._capture_forced_frames(4, 0.03)
            initial_baseline, _initial_noise = _estimate_baseline(initial_frames)
            initial_estimate = self._try_estimate_signal(initial_frames)
            slope, trial_frames = self._measure_dac_slope(
                state.dac_value, initial_baseline
            )
            trial_estimate = self._try_estimate_signal(trial_frames)
            centered_dac = self._dac_for_baseline(
                state.dac_value, initial_baseline, 0.0, slope
            )
            self._scope.set_dac_value(centered_dac)
            self._sleep_checked(0.06)

            self._progress("Auto Setup: searching for a signal...")
            survey_frames = self._capture_forced_frames(14, 0.04)
            survey_baseline, survey_noise = _estimate_baseline(survey_frames)
            survey_estimate = self._try_estimate_signal(survey_frames)
            estimate = self._combine_signal_estimates(
                [initial_estimate, trial_estimate, survey_estimate],
                baseline=survey_baseline,
                noise_sigma=survey_noise,
            )
            if estimate is None:
                self._progress("Auto Setup: waiting for a detector pulse...")
                estimate = self._wait_for_sparse_signal(
                    survey_baseline, survey_noise
                )

            target_baseline = self._target_baseline(
                estimate.polarity, estimate.noise_sigma
            )
            dac_value = self._dac_for_baseline(
                centered_dac, estimate.baseline, target_baseline, slope
            )
            self._scope.set_dac_value(dac_value)
            self._sleep_checked(0.06)
            adjusted_frames = self._capture_forced_frames(6, 0.04)
            adjusted = self._try_estimate_signal(adjusted_frames)
            if adjusted is not None:
                if adjusted.polarity != estimate.polarity:
                    raise RuntimeError("pulse polarity was inconsistent during Auto Setup")
            else:
                # At low event rates the post-adjustment viewer snapshots
                # may contain baseline only. A DAC offset does not change
                # pulse amplitude, so retain the survey pulse estimate and
                # use the newly measured baseline/noise.
                baseline, noise_sigma = _estimate_baseline(adjusted_frames)
                adjusted = _SignalEstimate(
                    baseline=baseline,
                    noise_sigma=max(noise_sigma, estimate.noise_sigma),
                    low=baseline - estimate.amplitude if estimate.polarity < 0 else baseline,
                    high=baseline + estimate.amplitude if estimate.polarity > 0 else baseline,
                    polarity=estimate.polarity,
                    amplitude=estimate.amplitude,
                )

            trigger_level = self._choose_trigger_level(adjusted)
            trigger_mode = (
                TriggerMode.FALLING_EDGE
                if adjusted.polarity < 0
                else TriggerMode.RISING_EDGE
            )

            self._scope.stop()
            self._restore_timing(state)
            verification_reference = self._capture_forced_frames(1, 0.03)[-1]
            self._scope.set_trigger_level(trigger_level)
            self._scope.set_trigger_mode(trigger_mode)

            self._progress("Auto Setup: verifying the trigger...")
            frame = self._capture_frames(1, _VERIFICATION_WAIT_S)[-1]
            fresh = (
                frame.shape == verification_reference.shape
                and not np.array_equal(frame, verification_reference)
            )
            verified = fresh and self._crosses_threshold(
                [frame], trigger_level, adjusted.polarity
            )
            frame = np.asarray(frame, dtype=np.int16).copy()
            self._scope.stop()

            return ScopeAutoSetupResult(
                dac_value=dac_value,
                dac_slope=slope,
                trigger_level=trigger_level,
                trigger_mode=trigger_mode,
                baseline=adjusted.baseline,
                noise_sigma=adjusted.noise_sigma,
                pulse_amplitude=adjusted.amplitude,
                verified=verified,
                frame=frame,
            )
        except BaseException:
            self._restore_state(state)
            raise

    def _read_state(self) -> _ScopeState:
        return _ScopeState(
            enabled=self._scope.get_enable(),
            dac_value=self._scope.get_dac_value(),
            trigger_level=self._scope.get_trigger_level(),
            trigger_mode=self._scope.get_trigger_mode(),
            frame_samples=self._scope.get_frame_samples(),
            pretrigger_samples=self._scope.get_pretrigger_samples(),
        )

    def _restore_timing(self, state: _ScopeState) -> None:
        self._scope.stop()
        self._scope.set_frame_samples(state.frame_samples)
        self._scope.set_pretrigger_samples(state.pretrigger_samples)

    def _restore_state(self, state: _ScopeState) -> None:
        try:
            self._scope.stop()
            self._scope.set_dac_value(state.dac_value)
            self._scope.set_trigger_level(state.trigger_level)
            self._scope.set_trigger_mode(state.trigger_mode)
            self._restore_timing(state)
            if state.enabled:
                self._scope.start()
        except Exception:
            log.exception("Failed to restore scope settings after Auto Setup")

    def _capture_forced_frames(self, count: int, interval_s: float) -> list[np.ndarray]:
        trigger_spec = _range_spec(ScopeParam.TRIGGER_LEVEL)
        self._scope.stop()
        self._scope.set_trigger_level(int(trigger_spec.min_val))
        self._scope.set_trigger_mode(TriggerMode.ANY_ABOVE)
        return self._capture_frames(count, interval_s)

    @staticmethod
    def _try_estimate_signal(frames: list[np.ndarray]) -> _SignalEstimate | None:
        try:
            return _estimate_signal(frames)
        except RuntimeError as exc:
            if str(exc) != _NO_CLEAR_PULSE:
                raise
            return None

    @staticmethod
    def _combine_signal_estimates(
        estimates: list[_SignalEstimate | None],
        *,
        baseline: float,
        noise_sigma: float,
    ) -> _SignalEstimate | None:
        present = [estimate for estimate in estimates if estimate is not None]
        if not present:
            return None
        polarities = {estimate.polarity for estimate in present}
        if len(polarities) != 1:
            raise RuntimeError("pulse polarity was inconsistent during Auto Setup")

        polarity = present[0].polarity
        amplitude = float(np.median([estimate.amplitude for estimate in present]))
        combined_noise = max(
            noise_sigma, *(estimate.noise_sigma for estimate in present)
        )
        return _SignalEstimate(
            baseline=baseline,
            noise_sigma=combined_noise,
            low=baseline - amplitude if polarity < 0 else baseline,
            high=baseline + amplitude if polarity > 0 else baseline,
            polarity=polarity,
            amplitude=amplitude,
        )

    def _wait_for_sparse_signal(
        self, baseline: float, noise_sigma: float
    ) -> _SignalEstimate:
        """Wait on alternating provisional thresholds for a rare pulse.

        Forced viewer frames sample only a few microseconds apiece. Detector
        pulses can therefore be obvious when triggered manually but absent
        from a short forced survey. Before each bounded wait, save a forced
        baseline frame. An unchanged frame after arming means that no new
        viewer capture occurred; a changed frame is inspected for a shaped
        unipolar event.
        """
        trigger_spec = _range_spec(ScopeParam.TRIGGER_LEVEL)
        for attempt in range(_SIGNAL_PROBE_ATTEMPTS):
            reference = self._capture_forced_frames(1, 0.03)[-1]
            reference_estimate = self._try_estimate_signal([reference])
            if reference_estimate is not None:
                combined = self._combine_signal_estimates(
                    [reference_estimate],
                    baseline=baseline,
                    noise_sigma=noise_sigma,
                )
                assert combined is not None
                return combined

            reference_baseline, reference_noise = _estimate_baseline([reference])
            polarity = -1 if attempt % 2 == 0 else 1
            distance = max(256.0, 8.0 * reference_noise)
            level = int(
                np.clip(
                    round(reference_baseline + polarity * distance),
                    trigger_spec.min_val,
                    trigger_spec.max_val,
                )
            )
            self._scope.set_trigger_level(level)
            self._scope.set_trigger_mode(
                TriggerMode.ANY_BELOW if polarity < 0 else TriggerMode.ANY_ABOVE
            )
            candidate = self._capture_frames(1, _SIGNAL_PROBE_WAIT_S)[-1]
            if candidate.shape == reference.shape and np.array_equal(
                candidate, reference
            ):
                continue

            candidate_estimate = self._try_estimate_signal([candidate])
            if candidate_estimate is None:
                continue
            combined = self._combine_signal_estimates(
                [candidate_estimate],
                baseline=baseline,
                noise_sigma=noise_sigma,
            )
            assert combined is not None
            return combined

        raise RuntimeError(_NO_CLEAR_PULSE)

    def _capture_frames(self, count: int, interval_s: float) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        self._scope.start()
        try:
            for _ in range(count):
                self._sleep_checked(interval_s)
                frame = np.asarray(self._scope.acquire_frame(), dtype=np.int16)
                if frame.size:
                    frames.append(frame.copy())
        finally:
            self._scope.stop()
        if not frames:
            raise RuntimeError("the scope viewer did not return a frame")
        return frames

    def _measure_dac_slope(
        self, original_dac: int, baseline: float
    ) -> tuple[float, list[np.ndarray]]:
        dac_spec = _range_spec(ScopeParam.DAC_VALUE)
        step = 12 if original_dac <= int(dac_spec.max_val) - 12 else -12
        trial_dac = original_dac + step
        self._scope.set_dac_value(trial_dac)
        self._sleep_checked(0.05)
        trial_frames = self._capture_forced_frames(4, 0.03)
        trial_baseline, _noise_sigma = _estimate_baseline(trial_frames)
        slope = (trial_baseline - baseline) / step
        if abs(slope) < 0.5:
            raise RuntimeError("the DAC did not produce a measurable baseline shift")
        return slope, trial_frames

    @staticmethod
    def _dac_for_baseline(
        anchor_dac: int, baseline: float, target_baseline: float, slope: float
    ) -> int:
        dac_spec = _range_spec(ScopeParam.DAC_VALUE)
        proposed = anchor_dac + round((target_baseline - baseline) / slope)
        proposed = int(np.clip(proposed, dac_spec.min_val, dac_spec.max_val))
        return proposed

    @staticmethod
    def _target_baseline(polarity: int, noise_sigma: float) -> float:
        """Put the quiet baseline near the rail opposite the pulse.

        Five percent of the ADC span is retained for baseline noise and
        analogue overshoot. A larger ten-sigma allowance wins for unusually
        noisy inputs. The remaining roughly 95% is available for the
        expected unipolar pulse excursion.
        """
        adc_spec = _range_spec(ScopeParam.TRIGGER_LEVEL)
        adc_span = float(adc_spec.max_val - adc_spec.min_val)
        margin = max(0.05 * adc_span, 10.0 * noise_sigma, 1024.0)
        margin = min(margin, 0.20 * adc_span)
        if polarity < 0:
            return float(adc_spec.max_val) - margin
        return float(adc_spec.min_val) + margin

    @staticmethod
    def _choose_trigger_level(estimate: _SignalEstimate) -> int:
        distance = max(6.0 * estimate.noise_sigma, 0.20 * estimate.amplitude, 64.0)
        distance = min(distance, 0.50 * estimate.amplitude)
        level = estimate.baseline + estimate.polarity * distance
        trigger_spec = _range_spec(ScopeParam.TRIGGER_LEVEL)
        return int(np.clip(round(level), trigger_spec.min_val, trigger_spec.max_val))

    @staticmethod
    def _crosses_threshold(
        frames: list[np.ndarray], threshold: int, polarity: int
    ) -> bool:
        if polarity < 0:
            return any(np.min(frame) <= threshold for frame in frames)
        return any(np.max(frame) >= threshold for frame in frames)

    def _sleep_checked(self, seconds: float) -> None:
        remaining = seconds
        while remaining > 0:
            self._check_cancelled()
            interval = min(remaining, 0.02)
            self._sleep(interval)
            remaining -= interval

    def _check_cancelled(self) -> None:
        if self._cancelled():
            raise AutoSetupCancelledError("Auto Setup cancelled")


class ScopeAutoSetupWorker(QObject):
    progress = Signal(str)
    succeeded = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, scope_factory: Callable[[], Scope]) -> None:
        super().__init__()
        self._scope_factory = scope_factory
        self._stop_event = threading.Event()

    @Slot()
    def run(self) -> None:
        scope: Scope | None = None
        try:
            # Construct the hardware transport here, after moveToThread(),
            # so the worker thread owns every connection it uses.
            scope = self._scope_factory()
            procedure = ScopeAutoSetupProcedure(
                scope,
                cancelled=self._stop_event.is_set,
                progress=self.progress.emit,
            )
            self.succeeded.emit(procedure.run())
        except AutoSetupCancelledError:
            self.error.emit("Auto Setup cancelled; original settings restored")
        except Exception as exc:
            log.exception("Scope Auto Setup failed")
            self.error.emit(str(exc))
        finally:
            if scope is not None:
                try:
                    scope.close()
                except Exception:
                    log.exception("Failed to close the Auto Setup scope connection")
            self.finished.emit()

    def stop(self) -> None:
        self._stop_event.set()
