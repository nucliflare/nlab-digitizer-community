"""Session replay: deterministic playback of a recorded session into the views.

Moved out of MainWindowController (which keeps the public replay API and
delegates here). The coordinator owns all replay state and its four
playback timers; the main window (``host``) still owns the views, the
shared hardware interlock and the offline external-module docks.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from time import monotonic_ns
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, Qt, QTimer

from nlab.controllers.global_controller import GlobalController
from nlab.controllers.psu_controller import PSUController
from nlab.session.replay import (
    CurrentReplaySample,
    McaReplayBatch,
    McaReplayReadback,
    ScopeReplayFrame,
    SessionReplay,
    TelemetryReplayEvent,
)
from nlab.workers.session_replay_worker import SessionAnalysisReplayThread

if TYPE_CHECKING:
    from nlab.controllers.main_window_controller import MainWindowController

log = logging.getLogger(__name__)

_REPLAY_CURRENT_SAMPLES_PER_TICK = 4096
_REPLAY_SCOPE_FRAMES_PER_TICK = 64
_REPLAY_TELEMETRY_EVENTS_PER_TICK = 1024


class SessionReplayCoordinator(QObject):
    def __init__(self, host: MainWindowController) -> None:
        super().__init__()
        self._host = host
        self._session_replay: SessionReplay | None = None
        self._session_replay_path: Path | None = None
        self._session_replay_spectrum_ids: set[str] = set()
        self._session_replay_origin_ns = 0
        self._session_replay_started_ns = 0
        self._session_replay_psd_channels: set[int] = set()
        self._session_replay_coincidence_active = False
        self._session_analysis_replay_index = 0
        self._session_analysis_replay_worker: SessionAnalysisReplayThread | None = None
        self._session_analysis_replay_complete = False
        self._session_analysis_replay_mca_channels: set[int] = set()
        self._session_analysis_replay_coincidence_channels: dict[int, int] = {}
        self._session_analysis_replay_coincidence_intervals: list[
            tuple[int, int | None]
        ] | None = None
        self._session_analysis_replay_timer = QTimer()
        self._session_analysis_replay_timer.setSingleShot(True)
        self._session_analysis_replay_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._session_analysis_replay_timer.timeout.connect(
            self._advance_session_analysis_replay
        )
        self._session_scope_replay_frames: list[ScopeReplayFrame] = []
        self._session_scope_replay_iterator: Iterator[ScopeReplayFrame] | None = None
        self._session_scope_replay_next: ScopeReplayFrame | None = None
        self._session_scope_replay_duration_ns = 0
        self._session_scope_replay_index = 0
        self._session_scope_replay_origin_ns = 0
        self._session_scope_replay_started_ns = 0
        self._session_scope_replay_channels: set[int] = set()
        self._session_scope_replay_frame_counts: dict[int, int] = {}
        self._session_scope_replay_delivered: dict[int, int] = {}
        self._session_scope_replay_timer = QTimer()
        self._session_scope_replay_timer.setSingleShot(True)
        self._session_scope_replay_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._session_scope_replay_timer.timeout.connect(
            self._advance_session_scope_replay
        )
        self._session_telemetry_replay_events: list[TelemetryReplayEvent] = []
        self._session_telemetry_replay_iterator: Iterator[TelemetryReplayEvent] | None = None
        self._session_telemetry_replay_next: TelemetryReplayEvent | None = None
        self._session_telemetry_replay_index = 0
        self._session_telemetry_replay_origin_ns = 0
        self._session_telemetry_replay_started_ns = 0
        self._session_telemetry_replay_psu_channels: set[int] = set()
        self._session_telemetry_replay_external_ids: set[str] = set()
        self._session_telemetry_replay_global_active = False
        self._session_telemetry_replay_global_digitizers: set[int] = set()
        self._session_telemetry_replay_timer = QTimer()
        self._session_telemetry_replay_timer.setSingleShot(True)
        self._session_telemetry_replay_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._session_telemetry_replay_timer.timeout.connect(
            self._advance_session_telemetry_replay
        )
        self._session_current_replay_iterator: Iterator[CurrentReplaySample] | None = None
        self._session_current_replay_next: CurrentReplaySample | None = None
        self._session_current_replay_channels: set[int] = set()
        self._session_current_replay_timer = QTimer()
        self._session_current_replay_timer.setSingleShot(True)
        self._session_current_replay_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._session_current_replay_timer.timeout.connect(
            self._advance_session_current_replay
        )

    def replay_session(self, path: Path) -> SessionReplay:
        """Replay captured analysis inputs and monitored telemetry once."""
        if self._host.session_recording:
            raise RuntimeError("stop the active session recording before replay")
        coincidence = getattr(self._host, "_coincidence_controller", None)
        if coincidence is not None and coincidence.active:
            raise RuntimeError("stop the coincidence measurement before replay")
        replay = SessionReplay(path)
        spectra = replay.replay_mca_spectra()
        recorded_channels_reader = getattr(replay, "recorded_channels", None)
        recorded_channels = (
            recorded_channels_reader() if recorded_channels_reader is not None else []
        )
        if getattr(self._host, "_offline", False) and recorded_channels:
            unavailable = sorted(
                channel for channel in recorded_channels if channel >= self._host._channels
            )
            if unavailable:
                required = max(unavailable) + 1
                raise RuntimeError(
                    "this session contains channel(s) unavailable in the offline "
                    f"workspace ({', '.join(map(str, unavailable))}); restart Offline "
                    f"Mode with at least {required} channels"
                )
        legacy_scope_frames: list[ScopeReplayFrame] | None = None
        if getattr(replay, "iter_scope_frames", None) is None:
            legacy_scope_frames = replay.scope_frames()
        legacy_telemetry_events: list[TelemetryReplayEvent] | None = None
        if getattr(replay, "iter_telemetry_events", None) is None:
            telemetry_reader = getattr(replay, "telemetry_events", None)
            legacy_telemetry_events = (
                telemetry_reader() if telemetry_reader is not None else []
            )
        self.clear_session_replay()
        self._session_replay = replay
        self._session_replay_path = path
        try:
            origin_reader = getattr(replay, "replay_origin_ns", None)
            if origin_reader is not None:
                self._session_replay_origin_ns = origin_reader()
            else:
                value = replay.manifest.get("started_monotonic_ns", 0)
                self._session_replay_origin_ns = (
                    int(value)
                    if isinstance(value, int) and not isinstance(value, bool)
                    else 0
                )
            self._session_replay_started_ns = monotonic_ns()
            self._host._set_replay_hardware_interlock(True)
            if getattr(self._host, "_offline", False):
                self._host._install_offline_external_replay_controllers(
                    replay,
                    legacy_telemetry_events,
                )
            if spectra:
                self._host._ensure_mca_peak_analysis_dialog().add_spectra(spectra)
            self._session_replay_spectrum_ids = {
                spectrum.spectrum_id for spectrum in spectra
            }
            self._start_session_analysis_replay(replay)
            self._start_session_scope_replay(replay, legacy_scope_frames)
            self._start_session_current_replay(replay)
            self._start_session_telemetry_replay(replay, legacy_telemetry_events)
        except Exception:
            self.clear_session_replay()
            raise
        scope_count = sum(self._session_scope_replay_frame_counts.values())
        telemetry_count = len(self._session_telemetry_replay_events)
        log.info(
            "Session replay loaded: %s (%d MCA spectra, %d Scope frames, "
            "%s telemetry updates, integrity=%s)",
            path,
            len(spectra),
            scope_count,
            telemetry_count if legacy_telemetry_events is not None else "streamed",
            replay.integrity,
        )
        return replay

    def _ensure_session_replay_clock(
        self,
        replay: SessionReplay,
        first_timestamp_ns: int,
    ) -> tuple[int, int]:
        """Create one wall-clock mapping shared by every replay subsystem."""
        started = getattr(self, "_session_replay_started_ns", 0)
        origin = getattr(self, "_session_replay_origin_ns", 0)
        if started:
            return origin, started
        manifest_origin = replay.manifest.get("started_monotonic_ns")
        origin = (
            int(manifest_origin)
            if isinstance(manifest_origin, int)
            and not isinstance(manifest_origin, bool)
            and 0 <= manifest_origin <= first_timestamp_ns
            else first_timestamp_ns
        )
        started = monotonic_ns()
        self._session_replay_origin_ns = origin
        self._session_replay_started_ns = started
        return origin, started

    @staticmethod
    def _close_replay_iterator(iterator: object) -> None:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()

    def _start_session_analysis_replay(self, replay: SessionReplay) -> None:
        """Replay MCA polls and list-mode batches over their captured timeline."""
        mca_controllers = {
            controller.channel: controller
            for controller in getattr(self._host, "_mca_controllers", [])
        }
        psd_controllers = getattr(self._host, "_psd_controllers", [])
        coincidence = getattr(self._host, "_coincidence_controller", None)
        if not mca_controllers and not psd_controllers and coincidence is None:
            return
        readback_channel_reader = getattr(replay, "mca_readback_channels", None)
        event_channel_reader = getattr(replay, "mca_event_channels", None)
        legacy_readbacks: list[McaReplayReadback] | None = None
        legacy_batches: list[McaReplayBatch] | None = None
        if readback_channel_reader is not None:
            readback_channels = set(readback_channel_reader())
        else:
            legacy_readbacks = replay.mca_readbacks()
            readback_channels = {item.channel for item in legacy_readbacks}
        if event_channel_reader is not None:
            event_channels = set(event_channel_reader())
        else:
            legacy_batches = replay.mca_event_batches()
            event_channels = {item.channel for item in legacy_batches}
        configuration = replay.configuration_snapshot()
        application = configuration.get("application")
        application = application if isinstance(application, dict) else {}
        replay_coincidence_settings = application.get("coincidence")
        replay_digitizer = self._host._saved_coincidence_digitizer(
            replay_coincidence_settings
        )
        selector = getattr(self._host, "_coincidence_selector", None)
        if replay_digitizer is not None and selector is not None:
            selector_index = selector.findData(replay_digitizer)
            if selector_index >= 0:
                selector.blockSignals(True)
                selector.setCurrentIndex(selector_index)
                selector.blockSignals(False)
                self._host._install_coincidence_digitizer(replay_digitizer)
                coincidence = self._host._coincidence_controller
        channel_settings = application.get("channels")
        channel_settings = channel_settings if isinstance(channel_settings, dict) else {}
        hardware = configuration.get("hardware")
        hardware = hardware if isinstance(hardware, dict) else {}
        hardware_channels = hardware.get("channels")
        hardware_channels = hardware_channels if isinstance(hardware_channels, dict) else {}

        replayed_mca_channels = readback_channels & set(mca_controllers)
        for channel in replayed_mca_channels:
            mca_controllers[channel].begin_session_replay()
        self._session_analysis_replay_mca_channels = replayed_mca_channels

        replayed_psd: set[int] = set()
        for controller in psd_controllers:
            channel = controller.channel
            if channel not in event_channels:
                continue
            hardware_channel = hardware_channels.get(str(channel), {})
            hardware_channel = hardware_channel if isinstance(hardware_channel, dict) else {}
            mca_hardware = hardware_channel.get("mca", {})
            mca_hardware = mca_hardware if isinstance(mca_hardware, dict) else {}
            charge_comparison = mca_hardware.get("charge_comparison", {})
            if (
                isinstance(charge_comparison, dict)
                and charge_comparison.get("enable") is False
            ):
                continue
            saved_channel = channel_settings.get(str(channel), {})
            saved_channel = saved_channel if isinstance(saved_channel, dict) else {}
            try:
                controller.begin_session_replay(
                    settings=saved_channel.get("psd"),
                )
            except Exception:
                log.exception("PSD session replay failed for channel %d", channel)
            else:
                replayed_psd.add(channel)
        self._session_replay_psd_channels = replayed_psd

        coincidence_channels: dict[int, int] = {}
        settings = None
        interval_reader = getattr(replay, "coincidence_intervals", None)
        coincidence_intervals = interval_reader() if interval_reader is not None else []
        explicit_timing_reader = getattr(replay, "has_explicit_mca_batch_timing", None)
        explicit_timing = (
            explicit_timing_reader() if explicit_timing_reader is not None else False
        )
        allow_coincidence = bool(coincidence_intervals) or not explicit_timing
        selected_pair = getattr(self._host, "_coincidence_channels", None)
        if selected_pair is None:
            ordered_mca = getattr(self._host, "_mca_controllers", [])
            if len(ordered_mca) >= 2:
                selected_pair = (ordered_mca[0].channel, ordered_mca[1].channel)
        if coincidence is not None and selected_pair is not None and allow_coincidence:
            channels = selected_pair
            if set(channels).issubset(event_channels):
                try:
                    settings, display_settings = replay.coincidence_settings(channels)
                    coincidence.begin_session_replay(settings, display_settings)
                except Exception as exc:
                    log.warning(
                        "Coincidence session replay is unavailable: %s",
                        exc,
                        exc_info=True,
                    )
                    status = getattr(coincidence, "status", None)
                    if status is not None:
                        status.setText(f"Session coincidence replay unavailable: {exc}")
                else:
                    self._session_replay_coincidence_active = True
                    coincidence_channels = {channels[0]: 0, channels[1]: 1}
        self._session_analysis_replay_coincidence_channels = coincidence_channels
        self._session_analysis_replay_coincidence_intervals = (
            coincidence_intervals if explicit_timing else None
        )

        used_batch_channels = replayed_psd | set(coincidence_channels)
        readback_iterator_reader = getattr(replay, "iter_mca_readbacks", None)
        batch_iterator_reader = getattr(replay, "iter_mca_event_batches", None)
        if readback_iterator_reader is not None:
            readback_iterator = readback_iterator_reader(
                channels=replayed_mca_channels,
            )
        else:
            assert legacy_readbacks is not None
            readback_iterator = iter(
                item
                for item in legacy_readbacks
                if item.channel in replayed_mca_channels
            )
        if batch_iterator_reader is not None:
            batch_iterator = batch_iterator_reader(channels=used_batch_channels)
        else:
            assert legacy_batches is not None
            batch_iterator = iter(
                item for item in legacy_batches if item.channel in used_batch_channels
            )
        origin, started = self._ensure_session_replay_clock(
            replay, replay.manifest.get("started_monotonic_ns", 0),
        )
        worker = SessionAnalysisReplayThread(
            (readback_iterator, batch_iterator),
            origin_ns=origin,
            started_ns=started,
            psd={
                controller.channel: controller.session_replay_accumulator()
                for controller in psd_controllers
                if controller.channel in replayed_psd
            },
            coincidence_settings=settings if coincidence_channels else None,
            coincidence_channels=coincidence_channels,
            coincidence_intervals=self._session_analysis_replay_coincidence_intervals,
        )
        for controller in psd_controllers:
            if controller.channel in replayed_psd:
                controller.use_session_replay_worker(
                    partial(worker.request_psd_configuration, controller.channel)
                )
        self._session_analysis_replay_worker = worker
        self._session_analysis_replay_complete = False
        self._session_analysis_replay_readback_sequences: dict[int, int] = {}
        worker.start()
        self._session_analysis_replay_timer.start(100)

    def _advance_session_analysis_replay(self) -> None:
        """Paint a coalesced snapshot; never read or analyze a batch on Qt."""
        worker = getattr(self, "_session_analysis_replay_worker", None)
        if worker is None or getattr(self, "_session_replay", None) is None:
            return
        snapshot = worker.take_snapshot()
        if snapshot is not None:
            self._session_analysis_replay_index = snapshot.processed
            for controller in getattr(self._host, "_mca_controllers", []):
                readback = snapshot.readbacks.get(controller.channel)
                if readback is not None:
                    sequences = self._session_analysis_replay_readback_sequences
                    if sequences.get(controller.channel) == readback.sequence:
                        continue
                    sequences[controller.channel] = readback.sequence
                    controller.display_session_replay_readback(readback.readback)
            for controller in getattr(self._host, "_psd_controllers", []):
                accumulator = snapshot.psd.get(controller.channel)
                if accumulator is not None:
                    controller.display_session_replay_snapshot(accumulator, snapshot.complete)
            coincidence = getattr(self._host, "_coincidence_controller", None)
            if coincidence is not None and snapshot.coincidence is not None:
                coincidence.display_session_replay_snapshot(
                    snapshot.coincidence, snapshot.timing_fit, snapshot.complete,
                )
            if snapshot.error is not None:
                message = f"Session analysis replay failed: {snapshot.error}"
                log.error(message)
                self._host._window.statusBar().showMessage(message)
                if coincidence is not None and self._session_replay_coincidence_active:
                    coincidence.status.setText(message)
                for controller in getattr(self._host, "_psd_controllers", []):
                    if controller.channel in self._session_replay_psd_channels:
                        controller.status.setText(message)
                return
            if snapshot.complete and not self._session_analysis_replay_complete:
                self._session_analysis_replay_complete = True
                for controller in getattr(self._host, "_mca_controllers", []):
                    if controller.channel in self._session_analysis_replay_mca_channels:
                        controller.finish_session_replay()
                log.info("MCA/PSD/coincidence replay reached EOF; final views are held")
        # Continue polling at EOF to service asynchronous PSD re-binning.
        self._session_analysis_replay_timer.start(100)

    def _start_session_scope_replay(
        self,
        replay: SessionReplay,
        frames: list[ScopeReplayFrame] | None = None,
    ) -> None:
        controllers = {
            controller.channel: controller for controller in self._host._scope_controllers
        }
        frame_counts: dict[int, int]
        if frames is None:
            summary = replay.scope_summary()
            frame_counts = {
                channel: count
                for channel, count in summary.counts.items()
                if channel in controllers
            }
            missing = sorted(set(summary.counts) - controllers.keys())
            iterator = replay.iter_scope_frames(channels=frame_counts)
            last_timestamp = summary.last_timestamp_ns
        else:
            playable = [frame for frame in frames if frame.channel in controllers]
            playable.sort(
                key=lambda frame: (
                    frame.host_monotonic_ns,
                    frame.channel,
                    frame.sequence,
                )
            )
            frame_counts = {}
            for frame in playable:
                frame_counts[frame.channel] = frame_counts.get(frame.channel, 0) + 1
            missing = sorted({frame.channel for frame in frames} - controllers.keys())
            iterator = iter(playable)
            last_timestamp = playable[-1].host_monotonic_ns if playable else None
        if missing:
            log.warning(
                "Session has Scope replay data for unavailable channel(s): %s",
                ", ".join(str(channel) for channel in missing),
            )
        if not frame_counts:
            return
        first = next(iterator, None)
        if first is None:
            return
        origin, started = self._ensure_session_replay_clock(
            replay,
            first.host_monotonic_ns,
        )
        channels = set(frame_counts)
        duration_ns = max(0, (last_timestamp or first.host_monotonic_ns) - origin)

        self._session_scope_replay_frames = []
        self._session_scope_replay_iterator = iterator
        self._session_scope_replay_next = first
        self._session_scope_replay_index = 0
        self._session_scope_replay_origin_ns = origin
        self._session_scope_replay_started_ns = started
        self._session_scope_replay_duration_ns = duration_ns
        self._session_scope_replay_channels = channels
        self._session_scope_replay_frame_counts = frame_counts
        self._session_scope_replay_delivered = dict.fromkeys(channels, 0)
        for channel in sorted(channels):
            controllers[channel].begin_session_replay(
                frame_count=frame_counts[channel],
                duration_ns=duration_ns,
            )

        self._advance_session_scope_replay()

    def _advance_session_scope_replay(self) -> None:
        frame = getattr(self, "_session_scope_replay_next", None)
        iterator = getattr(self, "_session_scope_replay_iterator", None)
        if self._session_replay is None or frame is None:
            return
        elapsed_ns = max(0, monotonic_ns() - self._session_scope_replay_started_ns)
        due_by_channel: dict[int, ScopeReplayFrame] = {}
        processed = 0
        while frame is not None and processed < _REPLAY_SCOPE_FRAMES_PER_TICK:
            recorded_elapsed_ns = max(
                0,
                frame.host_monotonic_ns - self._session_scope_replay_origin_ns,
            )
            if recorded_elapsed_ns > elapsed_ns:
                break
            due_by_channel[frame.channel] = frame
            self._session_scope_replay_delivered[frame.channel] += 1
            self._session_scope_replay_index += 1
            processed += 1
            frame = next(iterator, None) if iterator is not None else None
        self._session_scope_replay_next = frame

        controllers = {
            controller.channel: controller for controller in self._host._scope_controllers
        }
        duration_ns = self._session_scope_replay_duration_ns
        for channel, displayed_frame in due_by_channel.items():
            controller = controllers.get(channel)
            if controller is None:
                continue
            controller.display_session_replay_frame(
                displayed_frame.samples,
                sample_period_ns=displayed_frame.sample_period_ns,
                frame_index=self._session_scope_replay_delivered[channel],
                frame_count=self._session_scope_replay_frame_counts[channel],
                elapsed_ns=max(
                    0,
                    displayed_frame.host_monotonic_ns
                    - self._session_scope_replay_origin_ns,
                ),
                duration_ns=duration_ns,
                source=displayed_frame.source,
            )

        if frame is None:
            for channel in sorted(self._session_scope_replay_channels):
                controller = controllers.get(channel)
                if controller is not None:
                    controller.finish_session_replay(
                        frame_count=self._session_scope_replay_frame_counts[channel],
                        duration_ns=duration_ns,
                    )
            log.info("Scope session replay reached EOF; final frames are held")
            return

        next_elapsed_ns = max(
            0,
            frame.host_monotonic_ns - self._session_scope_replay_origin_ns,
        )
        remaining_ns = max(0, next_elapsed_ns - elapsed_ns)
        delay_ms = max(1, (remaining_ns + 999_999) // 1_000_000)
        self._session_scope_replay_timer.start(min(delay_ms, 2_147_000_000))

    def _start_session_current_replay(self, replay: SessionReplay) -> None:
        summary_reader = getattr(replay, "current_summary", None)
        iterator_reader = getattr(replay, "iter_current_samples", None)
        if summary_reader is None or iterator_reader is None:
            return
        controllers = {
            controller.channel: controller
            for controller in getattr(self._host, "_current_monitor_controllers", [])
        }
        summary = summary_reader()
        channels = set(summary.counts) & set(controllers)
        if not channels:
            return
        configuration = replay.configuration_snapshot()
        application = configuration.get("application")
        application = application if isinstance(application, dict) else {}
        channel_settings = application.get("channels")
        channel_settings = channel_settings if isinstance(channel_settings, dict) else {}
        for channel in sorted(channels):
            saved_channel = channel_settings.get(str(channel), {})
            saved_channel = saved_channel if isinstance(saved_channel, dict) else {}
            controllers[channel].begin_session_replay(
                settings=saved_channel.get("current_monitor"),
                dropped_samples=replay.current_dropped_samples(channel),
            )
        iterator = iterator_reader(channels=channels)
        first = next(iterator, None)
        if first is None:
            return
        origin, _started = self._ensure_session_replay_clock(
            replay,
            first.host_monotonic_ns,
        )
        self._session_current_replay_channels = channels
        self._session_current_replay_iterator = iterator
        self._session_current_replay_next = first
        self._session_replay_origin_ns = origin
        self._advance_session_current_replay()

    def _advance_session_current_replay(self) -> None:
        item = getattr(self, "_session_current_replay_next", None)
        iterator = getattr(self, "_session_current_replay_iterator", None)
        if self._session_replay is None or item is None:
            return
        elapsed_ns = max(0, monotonic_ns() - self._session_replay_started_ns)
        controllers = {
            controller.channel: controller
            for controller in getattr(self._host, "_current_monitor_controllers", [])
        }
        due: dict[int, list[Any]] = {}
        processed = 0
        while item is not None and processed < _REPLAY_CURRENT_SAMPLES_PER_TICK:
            recorded_elapsed_ns = max(
                0,
                item.host_monotonic_ns - self._session_replay_origin_ns,
            )
            if recorded_elapsed_ns > elapsed_ns:
                break
            due.setdefault(item.channel, []).append(item.sample)
            processed += 1
            item = next(iterator, None) if iterator is not None else None
        self._session_current_replay_next = item
        for channel, samples in due.items():
            controller = controllers.get(channel)
            if controller is not None:
                controller.display_session_replay_samples(tuple(samples))
        if item is None:
            for channel in self._session_current_replay_channels:
                controller = controllers.get(channel)
                if controller is not None:
                    controller.finish_session_replay()
            log.info("Current-monitor session replay reached EOF; final views are held")
            return
        next_elapsed_ns = max(
            0,
            item.host_monotonic_ns - self._session_replay_origin_ns,
        )
        remaining_ns = max(0, next_elapsed_ns - elapsed_ns)
        delay_ms = max(1, (remaining_ns + 999_999) // 1_000_000)
        self._session_current_replay_timer.start(min(delay_ms, 2_147_000_000))

    def _psu_replay_controllers(self) -> dict[int, PSUController]:
        result: dict[int, PSUController] = {}
        for device_index, controller in getattr(
            self._host,
            "_psu_controller_by_device",
            {},
        ).items():
            result[self._host._display_channel(device_index)] = controller
        return result

    def _global_replay_sources(
        self,
    ) -> dict[str, tuple[int, GlobalController, str]]:
        result: dict[str, tuple[int, GlobalController, str]] = {}
        for digitizer, controller in enumerate(self._host._all_global_controllers()):
            prefix = "global" if digitizer == 0 else f"digitizer.d{digitizer + 1}"
            result[f"{prefix}.temperature_correction"] = (
                digitizer,
                controller,
                "temperature",
            )
            result[f"{prefix}.diagnostics"] = (
                digitizer,
                controller,
                "diagnostics",
            )
        return result

    def _start_session_telemetry_replay(
        self,
        replay: SessionReplay,
        events: list[TelemetryReplayEvent] | None = None,
    ) -> None:
        psu_controllers = self._psu_replay_controllers()
        external_controllers = {
            controller.configuration_id: controller
            for controller in getattr(self._host, "_external_controllers", [])
        }
        psu_channels: set[int] = set()
        external_ids: set[str] = set()
        global_sources = self._global_replay_sources()
        if events is None:
            latest_reader = getattr(replay, "latest_telemetry_events", None)
            latest = latest_reader() if latest_reader is not None else {}
            available_sources = set(latest)
        else:
            available_sources = {event.source for event in events}
        playable_sources: set[str] = set()
        for source in available_sources:
            if source.startswith("psu.ch"):
                try:
                    channel = int(source.removeprefix("psu.ch"))
                except ValueError:
                    continue
                if channel in psu_controllers:
                    playable_sources.add(source)
                    psu_channels.add(channel)
            elif source.startswith("external."):
                configuration_id = source.removeprefix("external.")
                if configuration_id in external_controllers:
                    playable_sources.add(source)
                    external_ids.add(configuration_id)
            elif source in global_sources:
                playable_sources.add(source)
        if not playable_sources:
            return
        if events is None:
            iterator = replay.iter_telemetry_events(sources=playable_sources)
            self._session_telemetry_replay_events = []
        else:
            playable = [event for event in events if event.source in playable_sources]
            playable.sort(key=lambda event: (event.host_monotonic_ns, event.sequence))
            iterator = iter(playable)
            self._session_telemetry_replay_events = playable
        first = next(iterator, None)
        if first is None:
            return
        origin, started = self._ensure_session_replay_clock(
            replay,
            first.host_monotonic_ns,
        )
        self._session_telemetry_replay_iterator = iterator
        self._session_telemetry_replay_next = first
        self._session_telemetry_replay_index = 0
        self._session_telemetry_replay_origin_ns = origin
        self._session_telemetry_replay_started_ns = started
        self._session_telemetry_replay_psu_channels = psu_channels
        self._session_telemetry_replay_external_ids = external_ids
        global_digitizers = {
            global_sources[source][0]
            for source in playable_sources
            if source in global_sources
        }
        self._session_telemetry_replay_global_digitizers = global_digitizers
        self._session_telemetry_replay_global_active = 0 in global_digitizers
        for channel in psu_channels:
            psu_controllers[channel].begin_session_replay()
        for configuration_id in external_ids:
            external_controllers[configuration_id].begin_session_replay()
        for digitizer in sorted(global_digitizers):
            self._host._all_global_controllers()[digitizer].begin_session_replay()
        self._advance_session_telemetry_replay()

    def _advance_session_telemetry_replay(self) -> None:
        event = getattr(self, "_session_telemetry_replay_next", None)
        iterator = getattr(self, "_session_telemetry_replay_iterator", None)
        if self._session_replay is None or event is None:
            return
        elapsed_ns = max(0, monotonic_ns() - self._session_telemetry_replay_started_ns)
        psu_controllers = self._psu_replay_controllers()
        external_controllers = {
            controller.configuration_id: controller
            for controller in getattr(self._host, "_external_controllers", [])
        }
        global_sources = self._global_replay_sources()
        processed = 0
        while event is not None and processed < _REPLAY_TELEMETRY_EVENTS_PER_TICK:
            recorded_elapsed_ns = max(
                0,
                event.host_monotonic_ns - self._session_telemetry_replay_origin_ns,
            )
            if recorded_elapsed_ns > elapsed_ns:
                break
            if event.source.startswith("psu.ch"):
                channel = int(event.source.removeprefix("psu.ch"))
                controller = psu_controllers.get(channel)
                readback = event.payload.get("readback")
                if controller is not None and isinstance(readback, dict):
                    controller.display_session_replay_readback(readback)
            elif event.source.startswith("external."):
                configuration_id = event.source.removeprefix("external.")
                controller = external_controllers.get(configuration_id)
                timestamp = event.payload.get("worker_timestamp")
                values = event.payload.get("values")
                if controller is not None and isinstance(timestamp, int | float):
                    controller.display_session_replay_readback(float(timestamp), values)
            elif event.source in global_sources:
                _digitizer, global_replay_controller, kind = global_sources[event.source]
                if kind == "temperature":
                    global_replay_controller.display_session_replay_temperature(
                        event.payload.get("readback")
                    )
                else:
                    global_replay_controller.display_session_replay_diagnostics(
                        event.payload.get("readings")
                    )
            self._session_telemetry_replay_index += 1
            processed += 1
            event = next(iterator, None) if iterator is not None else None
        self._session_telemetry_replay_next = event
        if event is None:
            global_controllers = self._host._all_global_controllers()
            for digitizer in getattr(
                self,
                "_session_telemetry_replay_global_digitizers",
                set(),
            ):
                if digitizer < len(global_controllers):
                    global_controllers[digitizer].finish_session_replay()
            log.info("Session telemetry replay reached EOF; final readbacks are held")
            return
        next_elapsed_ns = max(
            0,
            event.host_monotonic_ns - self._session_telemetry_replay_origin_ns,
        )
        remaining_ns = max(0, next_elapsed_ns - elapsed_ns)
        delay_ms = max(1, (remaining_ns + 999_999) // 1_000_000)
        self._session_telemetry_replay_timer.start(min(delay_ms, 2_147_000_000))

    @property
    def session_replay(self) -> SessionReplay | None:
        return getattr(self, "_session_replay", None)

    @property
    def session_replay_path(self) -> Path | None:
        return getattr(self, "_session_replay_path", None)

    def clear_session_replay(self, *, resume_background: bool = True) -> None:
        """Leave replay mode without resuming any stopped live acquisition."""
        timer = getattr(self, "_session_scope_replay_timer", None)
        if timer is not None:
            timer.stop()
        analysis_timer = getattr(self, "_session_analysis_replay_timer", None)
        if analysis_timer is not None:
            analysis_timer.stop()
        analysis_worker = getattr(self, "_session_analysis_replay_worker", None)
        if analysis_worker is not None:
            analysis_worker.request_stop()
            analysis_worker.join()
            self._session_analysis_replay_worker = None
        telemetry_timer = getattr(self, "_session_telemetry_replay_timer", None)
        if telemetry_timer is not None:
            telemetry_timer.stop()
        current_timer = getattr(self, "_session_current_replay_timer", None)
        if current_timer is not None:
            current_timer.stop()
        for iterator_name in (
            "_session_scope_replay_iterator",
            "_session_current_replay_iterator",
            "_session_telemetry_replay_iterator",
        ):
            self._close_replay_iterator(getattr(self, iterator_name, None))
        replay_channels: set[int] = getattr(
            self,
            "_session_scope_replay_channels",
            set(),
        )
        for controller in getattr(self._host, "_scope_controllers", []):
            if controller.channel in replay_channels:
                try:
                    controller.end_session_replay()
                except Exception:
                    log.exception(
                        "Failed to leave Scope replay on channel %d",
                        controller.channel,
                    )
        current_channels: set[int] = getattr(
            self,
            "_session_current_replay_channels",
            set(),
        )
        for controller in getattr(self._host, "_current_monitor_controllers", []):
            if controller.channel in current_channels:
                try:
                    controller.end_session_replay()
                except Exception:
                    log.exception(
                        "Failed to leave Current replay on channel %d",
                        controller.channel,
                    )
        mca_channels: set[int] = getattr(
            self,
            "_session_analysis_replay_mca_channels",
            set(),
        )
        for controller in getattr(self._host, "_mca_controllers", []):
            if controller.channel in mca_channels:
                try:
                    controller.end_session_replay()
                except Exception:
                    log.exception("Failed to leave MCA replay on channel %d", controller.channel)
        psd_channels: set[int] = getattr(self, "_session_replay_psd_channels", set())
        for controller in getattr(self._host, "_psd_controllers", []):
            if controller.channel in psd_channels:
                try:
                    controller.end_session_replay()
                except Exception:
                    log.exception("Failed to leave PSD replay on channel %d", controller.channel)
        coincidence = getattr(self._host, "_coincidence_controller", None)
        if coincidence is not None and getattr(
            self,
            "_session_replay_coincidence_active",
            False,
        ):
            try:
                coincidence.end_session_replay()
            except Exception:
                log.exception("Failed to leave coincidence replay")
        psu_controllers = self._psu_replay_controllers()
        for channel in getattr(self, "_session_telemetry_replay_psu_channels", set()):
            controller = psu_controllers.get(channel)
            if controller is not None:
                controller.end_session_replay()
        external_controllers = {
            controller.configuration_id: controller
            for controller in getattr(self._host, "_external_controllers", [])
        }
        for configuration_id in getattr(
            self,
            "_session_telemetry_replay_external_ids",
            set(),
        ):
            controller = external_controllers.get(configuration_id)
            if controller is not None:
                controller.end_session_replay()
        global_controllers = self._host._all_global_controllers()
        replayed_global_digitizers = getattr(
            self,
            "_session_telemetry_replay_global_digitizers",
            {0} if getattr(self, "_session_telemetry_replay_global_active", False) else set(),
        )
        for digitizer in replayed_global_digitizers:
            if digitizer < len(global_controllers):
                global_controllers[digitizer].end_session_replay()
        spectrum_ids: set[str] = getattr(
            self,
            "_session_replay_spectrum_ids",
            set(),
        )
        dialog = getattr(self._host, "_mca_peak_analysis_dialog", None)
        if dialog is not None and spectrum_ids:
            dialog.remove_spectra(spectrum_ids)
        self._session_replay = None
        self._session_replay_path = None
        self._session_replay_spectrum_ids = set()
        self._session_replay_origin_ns = 0
        self._session_replay_started_ns = 0
        self._session_replay_psd_channels = set()
        self._session_replay_coincidence_active = False
        self._session_analysis_replay_index = 0
        self._session_analysis_replay_complete = False
        self._session_analysis_replay_readback_sequences = {}
        self._session_analysis_replay_mca_channels = set()
        self._session_analysis_replay_coincidence_channels = {}
        self._session_analysis_replay_coincidence_intervals = None
        self._session_scope_replay_frames = []
        self._session_scope_replay_iterator = None
        self._session_scope_replay_next = None
        self._session_scope_replay_duration_ns = 0
        self._session_scope_replay_index = 0
        self._session_scope_replay_origin_ns = 0
        self._session_scope_replay_started_ns = 0
        self._session_scope_replay_channels = set()
        self._session_scope_replay_frame_counts = {}
        self._session_scope_replay_delivered = {}
        self._session_current_replay_iterator = None
        self._session_current_replay_next = None
        self._session_current_replay_channels = set()
        self._session_telemetry_replay_events = []
        self._session_telemetry_replay_iterator = None
        self._session_telemetry_replay_next = None
        self._session_telemetry_replay_index = 0
        self._session_telemetry_replay_origin_ns = 0
        self._session_telemetry_replay_started_ns = 0
        self._session_telemetry_replay_psu_channels = set()
        self._session_telemetry_replay_external_ids = set()
        self._session_telemetry_replay_global_active = False
        self._session_telemetry_replay_global_digitizers = set()
        if getattr(self._host, "_offline", False) and getattr(
            self, "_external_controllers", []
        ):
            self._host._clear_external_docks()
        self._host._set_replay_hardware_interlock(
            False,
            resume_background=resume_background,
        )
