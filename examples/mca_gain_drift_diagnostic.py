#!/usr/bin/env python
"""Correlate Na-22 511-keV MCA peak motion with board telemetry.

The command starts fresh histogram acquisitions on both channels, differences
successive accumulated spectra, and tracks the strongest bin in an inclusive
channel ROI.  It saves CSV telemetry, complete differential ROI spectra (NPZ),
metadata/summary JSON, and a diagnostic PNG.

By default the most recently auto-saved configuration for the selected host is
applied to the MCA cores only.  Scope and PSU settings are not written unless
``--apply-saved-psu`` is explicitly selected.  That option applies the saved
bias and always requests a safe HV shutdown after the capture.  The original
MCA settings are restored, acquisition is stopped in a ``finally`` block, and
the HV monitoring backends are closed directly.

Example::

    uv run python examples/mca_gain_drift_diagnostic.py \
        --host 192.168.10.135 --duration 180 \
        --channel-roi 0 3500 4250 --channel-roi 1 1000 1700 \
        --apply-saved-psu
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import iio
import numpy as np
from numpy.typing import NDArray

from nlab.analysis.mca_gain_drift import differential_counts, estimate_peak_maximum
from nlab.hardware.digitizer import Digitizer
from nlab.hardware.digitizer.backends.iio_ids_backend import (
    IIOIDSBackend,
    IIOIDSUnavailableError,
)
from nlab.hardware.digitizer.hv import HVSupply
from nlab.utils.settings_io import (
    apply_mca_hardware,
    apply_psu_hardware,
    channel_entry,
    collect_channel_hardware,
    read_configuration,
)
from nlab.workers.temperature_correction_worker import T_MAX


@dataclass(frozen=True)
class Options:
    uri: str
    host: str
    channels: tuple[int, ...]
    duration_s: float
    interval_s: float
    roi_min: int
    roi_max: int
    channel_rois: dict[int, tuple[int, int]]
    smoothing_sigma_bins: float
    output_root: Path
    configuration_path: Path
    apply_configuration: bool
    apply_saved_psu: bool
    bias_settle_s: float
    use_saved_temperature_correction: bool

    def roi_for(self, channel: int) -> tuple[int, int]:
        return self.channel_rois.get(channel, (self.roi_min, self.roi_max))


@dataclass
class ChannelState:
    channel: int
    digitizer: Digitizer
    ids_backend: IIOIDSBackend | None
    hv: HVSupply | None
    original_hardware: dict[str, Any]
    original_temperature: tuple[float, int]
    original_lp_coefficients: list[int]
    roi_min: int
    roi_max: int
    psu_applied: bool = False
    started_monotonic: float = 0.0
    previous_histogram: NDArray[np.uint32] = field(
        default_factory=lambda: np.empty(0, dtype=np.uint32)
    )
    elapsed_samples_s: list[float] = field(default_factory=list)
    differential_rois: list[NDArray[np.uint64]] = field(default_factory=list)


class AuxiliaryBoardMonitor:
    """Read standard hwmon telemetry exposed by the remote iiod context."""

    def __init__(self, uri: str) -> None:
        self._context = iio.Context(uri)
        self._ethernet = self._context.find_device("hwmon0")
        self._power = self._context.find_device("ina228") or self._context.find_device(
            "hwmon1"
        )

    @staticmethod
    def _input(device: iio.Device | None, channel_name: str) -> float | None:
        if device is None:
            return None
        channel = device.find_channel(channel_name)
        if channel is None or "input" not in channel.attrs:
            return None
        return float(channel.attrs["input"].value)

    def read(self, errors: list[str]) -> dict[str, float | None]:
        try:
            ethernet_temp = self._input(self._ethernet, "temp1")
            input_mv = self._input(self._power, "in1")
            power_uw = self._input(self._power, "power1")
            current_ma = self._input(self._power, "curr1")
            monitor_temp = self._input(self._power, "temp1")
        except (RuntimeError, OSError, ValueError) as exc:
            errors.append(f"board_hwmon: {type(exc).__name__}: {exc}")
            return {}
        # Linux hwmon ABI units: temperature millidegrees C, voltage mV,
        # current mA, and power microwatts.
        return {
            "ethernet_temperature_c": (
                ethernet_temp / 1000.0 if ethernet_temp is not None else None
            ),
            "board_input_voltage_v": input_mv / 1000.0 if input_mv is not None else None,
            "board_input_current_a": (
                current_ma / 1000.0 if current_ma is not None else None
            ),
            "board_input_power_w": power_uw / 1_000_000.0 if power_uw is not None else None,
            "board_power_monitor_temperature_c": (
                monitor_temp / 1000.0 if monitor_temp is not None else None
            ),
        }


def _auto_configuration_path(host: str) -> Path:
    # Match nlab.main's identity before asking QStandardPaths for the location.
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.setApplicationName("Nuclear Lab Digitizer")
    QCoreApplication.setOrganizationName("EWT")
    QCoreApplication.setOrganizationDomain("ewt.local")
    from nlab.views.general_settings_dialog import auto_configuration_path

    return Path(auto_configuration_path(host))


def _read_optional(
    errors: list[str],
    name: str,
    reader: Callable[[], int | float | bool],
) -> int | float | bool | None:
    try:
        return reader()
    except (NotImplementedError, RuntimeError, OSError, ValueError) as exc:
        errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return None


def _channel_telemetry(state: ChannelState, errors: list[str]) -> dict[str, object]:
    hv = state.hv
    if hv is None:
        return {
            "hv_feedback_v": None,
            "hv_dac_output_v": None,
            "temperature_analog_c": None,
            "temperature_digital_c": None,
            "temperature_digital_status": None,
            "ads_temperature_raw": None,
        }
    return {
        "hv_feedback_v": _read_optional(errors, "hv_feedback_v", hv.get_hv_adc_voltage),
        "hv_dac_output_v": _read_optional(
            errors, "hv_dac_output_v", hv.get_hv_compens_output
        ),
        "temperature_analog_c": _read_optional(
            errors, "temperature_analog_c", hv.get_temp_analog
        ),
        "temperature_digital_c": _read_optional(
            errors, "temperature_digital_c", hv.get_temp_digital
        ),
        "temperature_digital_status": _read_optional(
            errors, "temperature_digital_status", hv.get_temp_digital_status
        ),
        "ads_temperature_raw": _read_optional(
            errors, "ads_temperature_raw", hv.get_ads_temp
        ),
    }


def _global_telemetry(state: ChannelState, errors: list[str]) -> dict[str, object]:
    hv = state.hv
    if hv is None:
        return {}
    try:
        readings = hv.get_global_diagnostics()
    except (RuntimeError, OSError, ValueError) as exc:
        errors.append(f"global_diagnostics: {type(exc).__name__}: {exc}")
        return {}
    return {f"global_{reading.key}": reading.value for reading in readings}


def _statistics(state: ChannelState, errors: list[str]) -> dict[str, object]:
    stats = state.digitizer.mca.statistics
    try:
        events_lost: int | None = stats.get_events_lost()
    except NotImplementedError:
        events_lost = None
    return {
        "count_rate_raw": _read_optional(errors, "count_rate", stats.get_count_rate),
        "pulse_deadtime_ms": _read_optional(
            errors, "pulse_deadtime_ms", stats.get_pulse_deadtime_ms
        ),
        "events_lost": events_lost,
        "mca_elapsed_s": (
            float(value) / 10.0
            if (value := _read_optional(errors, "elapsed_time", stats.get_elapsed_time))
            is not None
            else None
        ),
        "pulse_overrange": _read_optional(
            errors, "pulse_overrange", stats.get_pulse_overrange
        ),
        "pulse_pileup": _read_optional(errors, "pulse_pileup", stats.get_pulse_pileup),
        "energy_overrange": _read_optional(
            errors, "energy_overrange", stats.get_energy_overrange
        ),
        "energy_estimation_error": _read_optional(
            errors, "energy_estimation_error", stats.get_energy_estimation_error
        ),
        "throughput_error": _read_optional(
            errors, "throughput_error", stats.get_throughput_error
        ),
    }


def _saved_temperature_settings(
    configuration: Mapping[str, Any],
) -> tuple[bool, float, int, float]:
    application = configuration.get("application")
    global_settings = application.get("global") if isinstance(application, Mapping) else None
    if not isinstance(global_settings, Mapping):
        return False, 0.0, 0, 1.0
    return (
        bool(global_settings.get("temperature_correction_enabled", False)),
        float(global_settings.get("temperature_coefficient", 0.0)),
        int(global_settings.get("temperature_offset", 0)),
        max(0.001, float(global_settings.get("temperature_interval_ms", 1000)) / 1000.0),
    )


def _apply_temperature_correction(
    states: Sequence[ChannelState],
    base_coefficient: float,
    base_offset: int,
    errors: list[str],
) -> tuple[float | None, float | None, int | None]:
    source_hv = next((state.hv for state in states if state.hv is not None), None)
    if source_hv is None:
        errors.append("temperature_correction: IDS/ADS telemetry unavailable")
        return None, None, None
    temperature = _read_optional(
        errors,
        "temperature_correction_ads_raw",
        source_hv.get_ads_temp_for_correction,
    )
    if temperature is None:
        return None, None, None
    delta = float(temperature) - T_MAX
    coefficient = base_coefficient * delta
    offset = int(base_offset * delta)
    for state in states:
        try:
            state.digitizer.mca.set_temperature_correction(coefficient, offset)
        except (RuntimeError, OSError, ValueError) as exc:
            errors.append(
                f"ch{state.channel}.temperature_correction: {type(exc).__name__}: {exc}"
            )
    return float(temperature), coefficient, offset


def _create_states(
    options: Options,
    configuration: Mapping[str, Any],
) -> list[ChannelState]:
    states: list[ChannelState] = []
    try:
        for channel in options.channels:
            digitizer = Digitizer.from_iio(channel, options.uri, with_ids=False)
            mca = digitizer.mca
            if not digitizer.capabilities.mca:
                digitizer.close()
                raise RuntimeError(f"channel {channel} has no MCA core")
            if mca.get_global_enable() or mca.get_measurement_in_progress():
                digitizer.close()
                raise RuntimeError(
                    f"MCA channel {channel} is already running; stop it before this diagnostic"
                )
            if mca.dma_active():
                digitizer.close()
                raise RuntimeError(
                    f"MCA channel {channel} list-mode DMA is active; stop it first"
                )

            try:
                ids_backend: IIOIDSBackend | None = IIOIDSBackend(channel, options.uri)
            except IIOIDSUnavailableError:
                ids_backend = None
            hv = HVSupply(ids_backend) if ids_backend is not None else None
            original = collect_channel_hardware(digitizer.scope, mca, None)
            state = ChannelState(
                channel=channel,
                digitizer=digitizer,
                ids_backend=ids_backend,
                hv=hv,
                original_hardware=original,
                original_temperature=(mca.get_temp_coeff(), mca.get_temp_offset()),
                original_lp_coefficients=mca.filters.lp.get_coeffs().astype(int).tolist(),
                roi_min=options.roi_for(channel)[0],
                roi_max=options.roi_for(channel)[1],
            )
            states.append(state)

        if options.apply_configuration:
            for state in states:
                saved = channel_entry(configuration, state.channel)
                if saved is None:
                    raise ValueError(
                        f"saved configuration has no hardware channel {state.channel}"
                    )
                apply_mca_hardware(state.digitizer.mca, saved)
        if options.apply_saved_psu:
            for state in states:
                saved = channel_entry(configuration, state.channel)
                if saved is None or not isinstance(saved.get("psu"), Mapping):
                    raise ValueError(
                        f"saved configuration has no PSU settings for channel {state.channel}"
                    )
                if state.hv is None:
                    raise RuntimeError(
                        f"channel {state.channel} has no IDS/HV backend for saved PSU settings"
                    )
            for state in states:
                saved = channel_entry(configuration, state.channel)
                assert saved is not None and state.hv is not None
                # Mark before the first register write so partial application
                # still triggers safe shutdown in the exception path.
                state.psu_applied = True
                apply_psu_hardware(state.hv, saved)
        return states
    except BaseException:
        for state in states:
            if state.psu_applied and state.hv is not None:
                state.hv.safe_shutdown()
            if state.ids_backend is not None:
                state.ids_backend.close()
            state.digitizer.close()
        raise


def _restore_and_close(states: Sequence[ChannelState], *, restore: bool) -> list[str]:
    errors: list[str] = []
    for state in states:
        try:
            state.digitizer.mca.stop()
        except Exception as exc:
            errors.append(f"ch{state.channel}.stop: {type(exc).__name__}: {exc}")
    if restore:
        for state in states:
            try:
                apply_mca_hardware(state.digitizer.mca, state.original_hardware)
                state.digitizer.mca.filters.lp.set_coeffs(state.original_lp_coefficients)
                coefficient, offset = state.original_temperature
                state.digitizer.mca.set_temperature_correction(coefficient, offset)
            except Exception as exc:
                errors.append(f"ch{state.channel}.restore: {type(exc).__name__}: {exc}")
    for state in states:
        if state.psu_applied and state.hv is not None:
            try:
                state.hv.safe_shutdown()
            except Exception as exc:
                errors.append(f"ch{state.channel}.hv_shutdown: {type(exc).__name__}: {exc}")
    for state in states:
        # Close the monitoring backend directly. In monitor-only mode this
        # avoids Digitizer.close() changing bias; explicit PSU mode was already
        # shut down immediately above.
        if state.ids_backend is not None:
            state.ids_backend.close()
        state.digitizer.close()  # with_ids=False: closes DPP only
    return errors


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_values(rows: Sequence[dict[str, object]], key: str) -> NDArray[np.float64]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        try:
            converted = (
                float(value)
                if isinstance(value, (bool, int, float, str))
                else float("nan")
            )
        except (TypeError, ValueError):
            converted = float("nan")
        values.append(converted)
    return np.asarray(values, dtype=np.float64)


def _correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 4:
        return None
    selected_x = x[mask]
    selected_y = y[mask]
    if np.ptp(selected_x) == 0 or np.ptp(selected_y) == 0:
        return None
    return float(np.corrcoef(selected_x, selected_y)[0, 1])


def _summarize(rows: Sequence[dict[str, object]], channels: Sequence[int]) -> dict[str, object]:
    result: dict[str, object] = {
        "interpretation": (
            "Correlations are exploratory, not causal. Compare repeated cold-start and "
            "warm-restart captures and inspect the saved differential spectra."
        )
    }
    candidates = (
        "hv_feedback_v",
        "hv_dac_output_v",
        "temperature_analog_c",
        "temperature_digital_c",
        "ads_temperature_raw",
        "global_hat_temperature",
        "global_fpga_temperature",
        "board_input_voltage_v",
        "board_input_power_w",
        "board_power_monitor_temperature_c",
        "ethernet_temperature_c",
        "temperature_correction_coefficient",
        "temperature_correction_offset",
    )
    channel_results: dict[str, object] = {}
    for channel in channels:
        selected = [row for row in rows if row.get("channel") == channel]
        valid = np.asarray([bool(row.get("peak_valid")) for row in selected])
        peak_results: dict[str, object] = {}
        for peak_key in ("peak_smoothed_channel", "peak_local_centroid_channel"):
            peak = _finite_values(selected, peak_key)
            valid_peak = peak[valid & np.isfinite(peak)]
            count = valid_peak.size
            edge_count = max(1, math.ceil(count * 0.2)) if count else 0
            shift = (
                float(np.median(valid_peak[-edge_count:]) - np.median(valid_peak[:edge_count]))
                if count >= 2
                else None
            )
            correlations: dict[str, float] = {}
            for key in candidates:
                value = _correlation(peak, _finite_values(selected, key))
                if value is not None:
                    correlations[key] = value
            peak_results[peak_key] = {
                "early_to_late_shift_channels": shift,
                "interval_standard_deviation_channels": (
                    float(np.std(valid_peak)) if count else None
                ),
                "telemetry_correlations": correlations,
            }
        telemetry_spans: dict[str, object] = {}
        for key in candidates:
            values = _finite_values(selected, key)
            finite = values[np.isfinite(values)]
            if finite.size:
                telemetry_spans[key] = {
                    "minimum": float(np.min(finite)),
                    "maximum": float(np.max(finite)),
                    "span": float(np.ptp(finite)),
                }
        channel_results[str(channel)] = {
            "valid_intervals": int(np.count_nonzero(valid)),
            "peak_estimators": peak_results,
            "telemetry_ranges": telemetry_spans,
        }
    result["channels"] = channel_results
    return result


def _save_plot(path: Path, rows: Sequence[dict[str, object]], channels: Sequence[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for index, channel in enumerate(channels):
        selected = [row for row in rows if row.get("channel") == channel]
        elapsed = _finite_values(selected, "elapsed_s")
        color = colors[index % len(colors)]
        axes[0].plot(
            elapsed,
            _finite_values(selected, "peak_smoothed_channel"),
            ".-",
            color=color,
            label=f"CH{channel} smoothed maximum",
        )
        axes[0].plot(
            elapsed,
            _finite_values(selected, "peak_maximum_channel"),
            ".",
            color=color,
            alpha=0.3,
            label=f"CH{channel} literal maximum",
        )
        axes[0].plot(
            elapsed,
            _finite_values(selected, "peak_local_centroid_channel"),
            "--",
            color=color,
            alpha=0.65,
            label=f"CH{channel} local centroid",
        )
        axes[1].plot(
            elapsed,
            _finite_values(selected, "hv_feedback_v"),
            ".-",
            color=color,
            label=f"CH{channel} HV feedback",
        )
        axes[2].plot(
            elapsed,
            _finite_values(selected, "temperature_digital_c"),
            ".-",
            color=color,
            label=f"CH{channel} detector-board temp",
        )
        axes[3].plot(
            elapsed,
            _finite_values(selected, "interval_roi_counts"),
            ".-",
            color=color,
            label=f"CH{channel} ROI counts/interval",
        )

    if rows:
        reference = [row for row in rows if row.get("channel") == channels[0]]
        elapsed = _finite_values(reference, "elapsed_s")
        for key, label, style in (
            ("global_hat_temperature", "HAT temp", "--"),
            ("global_fpga_temperature", "FPGA temp", ":"),
        ):
            values = _finite_values(reference, key)
            if np.any(np.isfinite(values)):
                axes[2].plot(elapsed, values, style, label=label)

    axes[0].set_ylabel("511-keV peak\n(channel)")
    axes[1].set_ylabel("HV feedback (V)")
    axes[2].set_ylabel("Temperature (°C)")
    axes[3].set_ylabel("Differential\nROI counts")
    axes[3].set_xlabel("Time after MCA start (s)")
    axes[0].set_title("MCA startup gain-drift diagnostic")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize="small")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_spectrum_comparison(
    path: Path,
    states: Sequence[ChannelState],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(states),
        1,
        figsize=(12, 4.5 * len(states)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for axis, state in zip(axes[:, 0], states, strict=True):
        channels = np.arange(state.roi_min, state.roi_max + 1)
        spectra = np.asarray(state.differential_rois, dtype=np.float64)
        if spectra.size == 0:
            continue
        edge_count = max(1, math.ceil(spectra.shape[0] * 0.2))
        early = np.mean(spectra[:edge_count], axis=0)
        late = np.mean(spectra[-edge_count:], axis=0)
        axis.step(channels, early, where="mid", label="first 20% mean", alpha=0.85)
        axis.step(channels, late, where="mid", label="last 20% mean", alpha=0.85)
        axis.set_ylabel(f"CH{state.channel}\ncounts/interval")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    axes[-1, 0].set_xlabel("MCA channel")
    axes[0, 0].set_title("Early and late differential 511-keV ROI spectra")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(options: Options) -> Path:
    configuration = read_configuration(options.configuration_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = options.output_root / f"mca_gain_drift_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(options.configuration_path, output / "saved_configuration.yaml")

    states = _create_states(options, configuration)
    board_monitor = AuxiliaryBoardMonitor(options.uri)
    rows: list[dict[str, object]] = []
    telemetry_errors: list[str] = []
    restore_errors: list[str] = []
    (
        correction_enabled,
        correction_coefficient,
        correction_offset,
        correction_interval_s,
    ) = (
        _saved_temperature_settings(configuration)
    )
    correction_enabled = correction_enabled and options.use_saved_temperature_correction
    started_channels: list[int] = []
    started_at_utc = datetime.now(UTC)
    started_at = time.monotonic()
    try:
        if options.bias_settle_s > 0:
            print(
                f"Holding saved bias for {options.bias_settle_s:g} s before MCA start...",
                flush=True,
            )
            settle_deadline = time.monotonic() + options.bias_settle_s
            while (remaining := settle_deadline - time.monotonic()) > 0:
                time.sleep(min(remaining, 0.1))
        started_at = time.monotonic()
        started_at_utc = datetime.now(UTC)
        for state in states:
            state.digitizer.mca.start()
            state.started_monotonic = time.monotonic()
            started_channels.append(state.channel)
        for state in states:
            state.previous_histogram = state.digitizer.mca.acquire_spectrum().copy()
            if state.roi_max >= state.previous_histogram.size:
                raise ValueError(
                    f"ROI maximum {state.roi_max} exceeds channel {state.channel} "
                    f"histogram size {state.previous_histogram.size}"
                )

        sample_index = 0
        next_sample = started_at + options.interval_s
        next_correction = started_at
        correction_ads: float | None = None
        applied_coefficient: float | None = None
        applied_offset: int | None = None
        pending_interval_errors: list[str] = []
        while True:
            now = time.monotonic()
            if now >= started_at + options.duration_s:
                break
            if correction_enabled and now >= next_correction:
                correction_ads, applied_coefficient, applied_offset = (
                    _apply_temperature_correction(
                        states,
                        correction_coefficient,
                        correction_offset,
                        pending_interval_errors,
                    )
                )
                next_correction += correction_interval_s
                if next_correction <= time.monotonic():
                    next_correction = time.monotonic() + correction_interval_s
                continue
            remaining = next_sample - now
            if remaining > 0:
                correction_wait = (
                    max(0.0, next_correction - now) if correction_enabled else remaining
                )
                time.sleep(min(remaining, correction_wait, 0.05))
                continue

            sample_index += 1
            interval_errors = pending_interval_errors
            pending_interval_errors = []

            global_values = _global_telemetry(states[0], interval_errors)
            board_values = board_monitor.read(interval_errors)
            elapsed = time.monotonic() - started_at
            captured_at = datetime.now(UTC).isoformat()
            for state in states:
                histogram = state.digitizer.mca.acquire_spectrum()
                delta, negative_bins = differential_counts(
                    histogram, state.previous_histogram
                )
                state.previous_histogram = histogram.copy()
                estimate = estimate_peak_maximum(
                    delta,
                    state.roi_min,
                    state.roi_max,
                    smoothing_sigma_bins=options.smoothing_sigma_bins,
                )
                roi = delta[state.roi_min : state.roi_max + 1].copy()
                state.elapsed_samples_s.append(elapsed)
                state.differential_rois.append(roi)
                row_errors = list(interval_errors)
                row: dict[str, object] = {
                    "timestamp_utc": captured_at,
                    "sample": sample_index,
                    "elapsed_s": elapsed,
                    "channel_elapsed_s": time.monotonic() - state.started_monotonic,
                    "channel": state.channel,
                    "interval_s": options.interval_s,
                    "roi_min_channel": state.roi_min,
                    "roi_max_channel": state.roi_max,
                    "peak_maximum_channel": estimate.maximum_channel,
                    "peak_smoothed_channel": estimate.smoothed_channel,
                    "peak_local_centroid_channel": estimate.local_centroid_channel,
                    "peak_counts": estimate.peak_counts,
                    "peak_background_counts_per_bin": (
                        estimate.background_counts_per_bin
                    ),
                    "peak_prominence_counts": estimate.prominence_counts,
                    "peak_signal_to_noise": estimate.signal_to_noise,
                    "peak_valid": estimate.valid,
                    "interval_roi_counts": estimate.roi_counts,
                    "interval_total_counts": int(np.sum(delta, dtype=np.uint64)),
                    "negative_difference_bins": negative_bins,
                    "measurement_in_progress": (
                        state.digitizer.mca.get_measurement_in_progress()
                    ),
                    "temperature_correction_enabled": correction_enabled,
                    "temperature_correction_ads_raw": correction_ads,
                    "temperature_correction_coefficient": applied_coefficient,
                    "temperature_correction_offset": applied_offset,
                    "temperature_coefficient_readback": (
                        state.digitizer.mca.get_temp_coeff()
                    ),
                    "temperature_offset_readback": (
                        state.digitizer.mca.get_temp_offset()
                    ),
                    **global_values,
                    **board_values,
                    **_channel_telemetry(state, row_errors),
                    **_statistics(state, row_errors),
                }
                row["telemetry_error_count"] = len(row_errors)
                rows.append(row)
                telemetry_errors.extend(
                    f"sample {sample_index} ch{state.channel}: {error}"
                    for error in row_errors
                )
                peak_text = (
                    f"{estimate.smoothed_channel:.2f}"
                    if np.isfinite(estimate.smoothed_channel)
                    else "n/a"
                )
                print(
                    f"t={elapsed:7.2f}s ch{state.channel}: peak={peak_text}, "
                    f"ROI={estimate.roi_counts}, S/N={estimate.signal_to_noise:.1f}"
                )

            next_sample += options.interval_s
            if next_sample <= time.monotonic():
                next_sample = time.monotonic() + options.interval_s
    finally:
        restore_errors = _restore_and_close(
            states,
            restore=options.apply_configuration or correction_enabled,
        )

    _write_csv(output / "samples.csv", rows)
    arrays: dict[str, NDArray[Any]] = {}
    for state in states:
        arrays[f"ch{state.channel}_roi_channels"] = np.arange(
            state.roi_min, state.roi_max + 1
        )
        arrays[f"ch{state.channel}_elapsed_s"] = np.asarray(state.elapsed_samples_s)
        arrays[f"ch{state.channel}_differential_counts"] = np.asarray(
            state.differential_rois,
            dtype=np.uint64,
        )
    np.savez_compressed(
        output / "differential_spectra.npz",
        **arrays,  # type: ignore[arg-type]
    )

    summary = _summarize(rows, options.channels)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    metadata = {
        "started_at_utc": started_at_utc.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "uri": options.uri,
        "channels": list(options.channels),
        "duration_s": options.duration_s,
        "interval_s": options.interval_s,
        "roi_inclusive_by_channel": {
            str(state.channel): [state.roi_min, state.roi_max] for state in states
        },
        "smoothing_sigma_bins": options.smoothing_sigma_bins,
        "configuration_source": str(options.configuration_path.resolve()),
        "configuration_sha256": hashlib.sha256(
            options.configuration_path.read_bytes()
        ).hexdigest(),
        "saved_mca_configuration_applied": options.apply_configuration,
        "saved_psu_applied_and_shutdown_after_capture": options.apply_saved_psu,
        "bias_settle_before_mca_start_s": options.bias_settle_s,
        "saved_temperature_correction_used": correction_enabled,
        "started_channels": started_channels,
        "histogram_snapshot_note": (
            "IIO histogram chunks are sequential and not an atomic running snapshot; "
            "negative interval bins are counted and clipped to zero."
        ),
        "telemetry_errors": telemetry_errors,
        "cleanup_or_restore_errors": restore_errors,
        "channel_hardware_before_test": {
            str(state.channel): state.original_hardware for state in states
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    _save_plot(output / "gain_drift.png", rows, options.channels)
    _save_spectrum_comparison(
        output / "early_late_spectra.png",
        states,
    )
    return output


def _parse_args(argv: Sequence[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.10.128")
    parser.add_argument("--port", type=int, default=30431)
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=(0, 1),
        choices=(0, 1),
        help="zero-based MCA channels (default: 0 1)",
    )
    parser.add_argument("--duration", type=float, default=180.0, metavar="SECONDS")
    parser.add_argument("--interval", type=float, default=1.0, metavar="SECONDS")
    parser.add_argument(
        "--roi",
        type=int,
        nargs=2,
        default=(1000, 1700),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--channel-roi",
        type=int,
        nargs=3,
        action="append",
        default=[],
        metavar=("CHANNEL", "MIN", "MAX"),
        help="override the inclusive ROI for one channel; may be repeated",
    )
    parser.add_argument("--smoothing-sigma", type=float, default=2.0, metavar="BINS")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("measurements") / "mca-gain-drift",
    )
    parser.add_argument(
        "--configuration",
        type=Path,
        help="saved full configuration (default: latest auto-save for --host)",
    )
    parser.add_argument(
        "--no-apply-configuration",
        action="store_true",
        help="use current MCA registers instead of applying/restoring the saved MCA values",
    )
    parser.add_argument(
        "--no-temperature-correction",
        action="store_true",
        help="do not run the saved GUI ADS-temperature correction loop",
    )
    parser.add_argument(
        "--apply-saved-psu",
        action="store_true",
        help=(
            "apply saved PSU/HV settings before capture and safely shut the supply "
            "down afterward"
        ),
    )
    parser.add_argument(
        "--bias-settle",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="hold the applied saved bias before starting either MCA",
    )
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.interval <= 0:
        parser.error("--duration and --interval must be positive")
    if args.interval > args.duration:
        parser.error("--interval cannot exceed --duration")
    roi_min, roi_max = args.roi
    if roi_min < 0 or roi_max < roi_min:
        parser.error("--roi must be a non-negative inclusive MIN MAX range")
    channel_rois: dict[int, tuple[int, int]] = {}
    for channel, channel_min, channel_max in args.channel_roi:
        if channel not in (0, 1):
            parser.error("--channel-roi CHANNEL must be 0 or 1")
        if channel_min < 0 or channel_max < channel_min:
            parser.error("--channel-roi must use a non-negative inclusive MIN MAX range")
        channel_rois[channel] = (channel_min, channel_max)
    if args.smoothing_sigma < 0:
        parser.error("--smoothing-sigma must be non-negative")
    if args.bias_settle < 0:
        parser.error("--bias-settle must be non-negative")
    if args.bias_settle and not args.apply_saved_psu:
        parser.error("--bias-settle requires --apply-saved-psu")
    configuration = args.configuration or _auto_configuration_path(args.host)
    if not configuration.is_file():
        parser.error(f"configuration file does not exist: {configuration}")
    return Options(
        uri=f"ip:{args.host}:{args.port}",
        host=args.host,
        channels=tuple(dict.fromkeys(args.channels)),
        duration_s=args.duration,
        interval_s=args.interval,
        roi_min=roi_min,
        roi_max=roi_max,
        channel_rois=channel_rois,
        smoothing_sigma_bins=args.smoothing_sigma,
        output_root=args.output_root,
        configuration_path=configuration,
        apply_configuration=not args.no_apply_configuration,
        apply_saved_psu=args.apply_saved_psu,
        bias_settle_s=args.bias_settle,
        use_saved_temperature_correction=not args.no_temperature_correction,
    )


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_args(argv)
    output = run(options)
    print(f"Saved diagnostic to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
