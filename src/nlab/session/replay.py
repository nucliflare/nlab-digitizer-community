"""Read-only validation and deterministic analysis replay for session bundles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import yaml

from nlab.analysis.coincidence import CoincidenceSettings
from nlab.analysis.spectrum import Spectrum
from nlab.hardware.digitizer.dma import IIO_LM_FRAME_RECORDS
from nlab.hardware.digitizer.iio_listmode import (
    COARSE_TICK_NS,
    IIO_LM_EVENT_DTYPE,
    TIME_Q_PER_NS,
    VDPP_ZC_CALC_SCHEMA,
)
from nlab.session.recorder import SESSION_FORMAT, SESSION_FORMAT_VERSION
from nlab.workers.mca_worker import MCAReadback

ScopeReplaySource = Literal["viewer", "dma"]


@dataclass(frozen=True)
class ScopeReplayFrame:
    """One immutable Scope waveform positioned on the recorded host timeline."""

    channel: int
    sequence: int
    host_monotonic_ns: int
    samples: np.ndarray
    sample_period_ns: int
    source: ScopeReplaySource

    def __post_init__(self) -> None:
        if self.channel < 0 or self.sequence < 0:
            raise ValueError("Scope replay channel and sequence must be non-negative")
        if self.host_monotonic_ns < 0 or self.sample_period_ns <= 0:
            raise ValueError("Scope replay timing values are invalid")
        samples = np.asarray(self.samples, dtype="<i2").reshape(-1)
        if len(samples) == 0:
            raise ValueError("Scope replay frame cannot be empty")
        object.__setattr__(
            self,
            "samples",
            np.frombuffer(bytes(samples.tobytes()), dtype="<i2"),
        )


@dataclass(frozen=True)
class TelemetryReplayEvent:
    """One telemetry update positioned on the session host timeline."""

    sequence: int
    host_monotonic_ns: int
    source: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class McaReplayReadback:
    channel: int
    sequence: int
    host_monotonic_ns: int
    readback: MCAReadback


@dataclass(frozen=True)
class McaReplayBatch:
    channel: int
    sequence: int
    host_monotonic_ns: int
    events: np.ndarray

    def __post_init__(self) -> None:
        events = np.asarray(self.events, dtype=IIO_LM_EVENT_DTYPE).reshape(-1)
        immutable = np.frombuffer(bytes(events.tobytes()), dtype=IIO_LM_EVENT_DTYPE)
        object.__setattr__(self, "events", immutable)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SessionReplay:
    """Validated, immutable view of a recorded session.

    Replay deliberately means rerunning analysis from recorded inputs.  It does
    not issue commands to the digitizer or claim to reproduce analog conditions.
    """

    def __init__(self, directory: Path, *, verify_hashes: bool = True) -> None:
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid session manifest: {manifest_path}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("session manifest must be a JSON object")
        if loaded.get("format") != SESSION_FORMAT:
            raise ValueError(f"not an {SESSION_FORMAT} bundle")
        if loaded.get("format_version") != SESSION_FORMAT_VERSION:
            raise ValueError(
                f"unsupported session format_version {loaded.get('format_version')!r}; "
                f"expected {SESSION_FORMAT_VERSION}"
            )
        self.manifest: dict[str, Any] = loaded
        files = loaded.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("session manifest has no files mapping")
        self.streams_path = self.directory / str(files.get("streams", "streams.h5"))
        self.timeline_path = self.directory / str(files.get("timeline", "timeline.jsonl"))
        self.configuration_path = self.directory / str(
            files.get("configuration", "configuration.yaml")
        )
        self.annotations_path = self.directory / str(
            files.get("annotations", "annotations.yaml")
        )
        for path in (self.streams_path, self.timeline_path, self.configuration_path):
            if not path.is_file():
                raise ValueError(f"session member is missing: {path.name}")
        if verify_hashes:
            self._verify_hashes()
        with h5py.File(self.streams_path, "r") as h5:
            if h5.attrs.get("format") != SESSION_FORMAT:
                raise ValueError("streams.h5 has an incompatible format marker")
            h5_session_id = h5.attrs.get("session_id")
            if h5_session_id != loaded.get("session_id"):
                raise ValueError("manifest and stream session IDs do not match")
            if self.capture_complete:
                marker = h5.get("capture_complete")
                if not isinstance(marker, h5py.Dataset) or int(marker[()]) != 1:
                    raise ValueError("manifest and stream completion markers do not match")

    @property
    def capture_complete(self) -> bool:
        return bool(self.manifest.get("capture_complete", False))

    @property
    def integrity(self) -> str:
        return str(self.manifest.get("integrity", "unknown"))

    def _verify_hashes(self) -> None:
        hashes = self.manifest.get("sha256")
        if hashes is None:
            if self.capture_complete:
                raise ValueError("complete session has no content hashes")
            return
        if not isinstance(hashes, Mapping):
            raise ValueError("session sha256 entry must be a mapping")
        for name, expected in hashes.items():
            path = self.directory / str(name)
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"session integrity check failed for {name}")

    def timeline(self) -> Iterator[dict[str, Any]]:
        with self.timeline_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid timeline record on line {line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"timeline line {line_number} is not an object")
                yield record

    def configuration_snapshot(self, *, final: bool = True) -> dict[str, Any]:
        """Return the captured application/hardware configuration.

        Final settings are preferred for derived analysis because they describe
        the controls visible at the end of the recorded interval. Older or
        incomplete sessions transparently fall back to ``configuration.yaml``.
        """
        if final:
            latest: dict[str, Any] | None = None
            for record in self.timeline():
                if record.get("kind") != "configuration_snapshot":
                    continue
                payload = record.get("payload")
                if not isinstance(payload, Mapping) or payload.get("phase") != "final":
                    continue
                configuration = payload.get("configuration")
                if isinstance(configuration, dict):
                    latest = dict(configuration)
            if latest is not None:
                return latest
        try:
            loaded = yaml.safe_load(self.configuration_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError("invalid session configuration") from exc
        if not isinstance(loaded, dict):
            raise ValueError("session configuration must contain a mapping")
        return dict(loaded)

    def telemetry_events(self) -> list[TelemetryReplayEvent]:
        """Return timestamped PSU/global/external telemetry in recorded order."""
        events: list[TelemetryReplayEvent] = []
        for record in self.timeline():
            if record.get("kind") != "telemetry":
                continue
            sequence = record.get("sequence")
            timestamp = record.get("host_monotonic_ns")
            source = record.get("source")
            payload = record.get("payload")
            if (
                not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or sequence < 0
                or not isinstance(timestamp, int)
                or isinstance(timestamp, bool)
                or timestamp < 0
                or not isinstance(source, str)
                or not isinstance(payload, Mapping)
            ):
                raise ValueError("session contains an invalid telemetry timeline record")
            events.append(
                TelemetryReplayEvent(
                    sequence=sequence,
                    host_monotonic_ns=timestamp,
                    source=source,
                    payload=dict(payload),
                )
            )
        events.sort(key=lambda event: (event.host_monotonic_ns, event.sequence))
        return events

    def coincidence_intervals(self) -> list[tuple[int, int | None]]:
        """Return host-time intervals explicitly marked as coincidence runs."""
        intervals: list[tuple[int, int | None]] = []
        started: int | None = None
        for record in self.timeline():
            timestamp = record.get("host_monotonic_ns")
            if not isinstance(timestamp, int) or isinstance(timestamp, bool):
                continue
            kind = record.get("kind")
            if kind == "coincidence_started":
                if started is not None:
                    intervals.append((started, timestamp))
                started = timestamp
            elif kind == "coincidence_observation":
                payload = record.get("payload")
                if (
                    started is None
                    and isinstance(payload, Mapping)
                    and payload.get("state") == "already_active"
                ):
                    started = timestamp
            elif kind == "coincidence_finished" and started is not None:
                intervals.append((started, timestamp))
                started = None
        if started is not None:
            intervals.append((started, None))
        return intervals

    def has_explicit_mca_batch_timing(self) -> bool:
        """Whether the bundle records producer-time boundaries for MCA batches."""
        with h5py.File(self.streams_path, "r") as h5:
            root = h5.get("mca")
            if not isinstance(root, h5py.Group):
                return False
            return any(
                isinstance(group, h5py.Group) and "list_mode/batches" in group
                for group in root.values()
            )

    def coincidence_settings(
        self,
        channels: tuple[int, int],
    ) -> tuple[CoincidenceSettings, dict[str, Any]]:
        """Recreate the qualified coincidence settings saved with a session."""
        configuration = self.configuration_snapshot()
        application = configuration.get("application")
        hardware = configuration.get("hardware")
        application = application if isinstance(application, Mapping) else {}
        hardware = hardware if isinstance(hardware, Mapping) else {}
        display = application.get("coincidence")
        display = dict(display) if isinstance(display, Mapping) else {}
        app_channels = application.get("channels")
        app_channels = app_channels if isinstance(app_channels, Mapping) else {}
        hardware_channels = hardware.get("channels")
        hardware_channels = hardware_channels if isinstance(hardware_channels, Mapping) else {}

        use_roi_value = display.get("use_roi", [False, False])
        use_roi = (
            [bool(value) for value in use_roi_value]
            if isinstance(use_roi_value, list) and len(use_roi_value) == 2
            else [False, False]
        )
        rois: list[tuple[int, int] | None] = []
        energy_bins: list[int] = []
        for index, channel in enumerate(channels):
            app_channel = app_channels.get(str(channel), {})
            app_channel = app_channel if isinstance(app_channel, Mapping) else {}
            mca_application = app_channel.get("mca", {})
            mca_application = (
                mca_application if isinstance(mca_application, Mapping) else {}
            )
            raw_roi = mca_application.get("roi")
            if use_roi[index] and isinstance(raw_roi, list) and len(raw_roi) == 2:
                low, high = sorted(
                    (
                        max(0, min(16_383, round(float(raw_roi[0])))),
                        max(0, min(16_383, round(float(raw_roi[1])))),
                    )
                )
                rois.append((low, high))
            else:
                rois.append(None)

            hardware_channel = hardware_channels.get(str(channel), {})
            hardware_channel = (
                hardware_channel if isinstance(hardware_channel, Mapping) else {}
            )
            mca_hardware = hardware_channel.get("mca", {})
            mca_hardware = mca_hardware if isinstance(mca_hardware, Mapping) else {}
            acquisition = mca_hardware.get("acquisition", {})
            acquisition = acquisition if isinstance(acquisition, Mapping) else {}
            energy_bins.append(int(acquisition.get("energy_bin", 0)))

        window = display.get("window_ns", [-48.0, 48.0])
        if not isinstance(window, list) or len(window) != 2:
            window = [-48.0, 48.0]
        delay = float(display.get("channel_delay_ch1_minus_ch0_ns", 0.0))

        def time_q(value: object, rounding: str) -> int:
            scaled = Decimal(str(value)) * TIME_Q_PER_NS
            return int(scaled.to_integral_value(rounding=rounding))

        settings = CoincidenceSettings(
            operator=str(display.get("operator", "AND")),
            not_ch0=bool(display.get("not_ch0", False)),
            not_ch1=bool(display.get("not_ch1", False)),
            low_q=time_q(window[0], ROUND_CEILING),
            high_q=time_q(window[1], ROUND_FLOOR),
            channel_delay_q=time_q(delay, ROUND_HALF_UP),
            roi_ch0=rois[0],
            roi_ch1=rois[1],
            energy_bin_ch0=energy_bins[0],
            energy_bin_ch1=energy_bins[1],
            fine_timing=str(display.get("timing_mode", "coarse")) == "cfd",
            random_sidebands=bool(display.get("matrix_random_sidebands", True)),
            random_sideband_gap_q=time_q(
                display.get("matrix_random_gap_ns", 0.0),
                ROUND_HALF_UP,
            ),
            record_schema=str(display.get("record_schema", VDPP_ZC_CALC_SCHEMA)),
        )
        return settings, display

    def annotations(self) -> dict[str, Any]:
        """Load mutable scientific labels kept outside capture-integrity hashes."""
        if not self.annotations_path.is_file():
            return {}
        try:
            loaded = yaml.safe_load(self.annotations_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError("invalid session annotations") from exc
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError("session annotations must contain a mapping")
        return dict(loaded)

    def mca_event_channels(self) -> list[int]:
        with h5py.File(self.streams_path, "r") as h5:
            root = h5.get("mca")
            if not isinstance(root, h5py.Group):
                return []
            channels: list[int] = []
            for name, group in root.items():
                if (
                    name.startswith("ch")
                    and isinstance(group, h5py.Group)
                    and "list_mode/events" in group
                ):
                    channels.append(int(name[2:]))
            return sorted(channels)

    @staticmethod
    def _committed_rows(dataset: h5py.Dataset) -> np.ndarray:
        committed = int(dataset.attrs.get("committed_records", len(dataset)))
        if committed < 0 or committed > len(dataset):
            raise ValueError(f"invalid committed-record count for {dataset.name}")
        return np.asarray(dataset[:committed])

    @staticmethod
    def _scope_frames_from_group(
        group: h5py.Group,
        *,
        channel: int,
        source: ScopeReplaySource,
    ) -> list[ScopeReplayFrame]:
        rows = group.get("frames")
        sample_values = group.get("samples")
        if not isinstance(rows, h5py.Dataset) or not isinstance(
            sample_values,
            h5py.Dataset,
        ):
            return []
        result: list[ScopeReplayFrame] = []
        for row in SessionReplay._committed_rows(rows):
            offset = int(row["sample_offset"])
            count = int(row["sample_count"])
            end = offset + count
            if offset < 0 or count <= 0 or end > len(sample_values):
                raise ValueError(
                    f"Scope ch{channel} {source} frame references invalid samples"
                )
            result.append(
                ScopeReplayFrame(
                    channel=channel,
                    sequence=int(row["sequence"]),
                    host_monotonic_ns=int(row["host_monotonic_ns"]),
                    samples=np.asarray(sample_values[offset:end], dtype="<i2"),
                    sample_period_ns=int(row["sample_period_ns"]),
                    source=source,
                )
            )
        return result

    def scope_frames(self) -> list[ScopeReplayFrame]:
        """Return display-ready frames ordered on one deterministic timeline.

        The viewer stream is the exact input originally shown by the Scope GUI
        and is therefore preferred. Full DMA waveforms are used only for a
        channel that has no recorded viewer frames.
        """
        frames: list[ScopeReplayFrame] = []
        with h5py.File(self.streams_path, "r") as h5:
            root = h5.get("scope")
            if not isinstance(root, h5py.Group):
                return []
            for name, channel_group in root.items():
                if not name.startswith("ch") or not isinstance(channel_group, h5py.Group):
                    continue
                try:
                    channel = int(name[2:])
                except ValueError:
                    continue
                viewer = channel_group.get("viewer")
                channel_frames = (
                    self._scope_frames_from_group(
                        viewer,
                        channel=channel,
                        source="viewer",
                    )
                    if isinstance(viewer, h5py.Group)
                    else []
                )
                if not channel_frames:
                    dma = channel_group.get("dma")
                    if isinstance(dma, h5py.Group):
                        channel_frames = self._scope_frames_from_group(
                            dma,
                            channel=channel,
                            source="dma",
                        )
                frames.extend(channel_frames)
        frames.sort(
            key=lambda frame: (
                frame.host_monotonic_ns,
                frame.channel,
                frame.sequence,
            )
        )
        return frames

    def mca_readback_channels(self) -> list[int]:
        with h5py.File(self.streams_path, "r") as h5:
            root = h5.get("mca")
            if not isinstance(root, h5py.Group):
                return []
            return sorted(
                int(name[2:])
                for name, group in root.items()
                if name.startswith("ch")
                and isinstance(group, h5py.Group)
                and "readback/checkpoints" in group
            )

    def mca_events(self, channel: int) -> np.ndarray:
        with h5py.File(self.streams_path, "r") as h5:
            path = f"mca/ch{int(channel)}/list_mode/events"
            if path not in h5:
                raise KeyError(f"session has no MCA list-mode events for channel {channel}")
            events = np.asarray(h5[path], dtype=IIO_LM_EVENT_DTYPE)
        events.setflags(write=False)
        return events

    def mca_event_batches(self) -> list[McaReplayBatch]:
        """Return recorded list-mode batches on their captured host timeline.

        New bundles carry an explicit batch index. For older bundles, complete
        1024-record DMA frames are timed from their first hardware timestamp so
        they can still be played progressively rather than collapsed to EOF.
        """
        result: list[McaReplayBatch] = []
        origin_value = self.manifest.get("started_monotonic_ns")
        origin = (
            int(origin_value)
            if isinstance(origin_value, int)
            and not isinstance(origin_value, bool)
            and origin_value >= 0
            else 0
        )
        with h5py.File(self.streams_path, "r") as h5:
            for channel in self.mca_event_channels():
                base = f"mca/ch{channel}/list_mode"
                event_dataset = h5[f"{base}/events"]
                assert isinstance(event_dataset, h5py.Dataset)
                committed_events = int(
                    event_dataset.attrs.get("committed_records", len(event_dataset))
                )
                batches = h5.get(f"{base}/batches")
                if isinstance(batches, h5py.Dataset):
                    for row in self._committed_rows(batches):
                        offset = int(row["event_offset"])
                        count = int(row["event_count"])
                        end = offset + count
                        if offset < 0 or count < 0 or end > committed_events:
                            raise ValueError(
                                f"MCA ch{channel} batch references invalid events"
                            )
                        result.append(
                            McaReplayBatch(
                                channel=channel,
                                sequence=int(row["sequence"]),
                                host_monotonic_ns=int(row["host_monotonic_ns"]),
                                events=np.asarray(event_dataset[offset:end]),
                            )
                        )
                    continue

                events = np.asarray(
                    event_dataset[:committed_events],
                    dtype=IIO_LM_EVENT_DTYPE,
                )
                first_tick: int | None = None
                sequence = 0
                for offset in range(0, len(events), IIO_LM_FRAME_RECORDS):
                    batch = events[offset : offset + IIO_LM_FRAME_RECORDS]
                    valid_ticks = batch["timestamp"][batch["timestamp"] != 0]
                    tick = int(valid_ticks[0]) if len(valid_ticks) else first_tick
                    if first_tick is None and tick is not None:
                        first_tick = tick
                    elapsed_ns = (
                        0
                        if tick is None or first_tick is None
                        else max(0, tick - first_tick) * COARSE_TICK_NS
                    )
                    result.append(
                        McaReplayBatch(
                            channel=channel,
                            sequence=sequence,
                            host_monotonic_ns=origin + elapsed_ns,
                            events=batch,
                        )
                    )
                    sequence += 1
        result.sort(
            key=lambda batch: (
                batch.host_monotonic_ns,
                batch.channel,
                batch.sequence,
            )
        )
        return result

    def mca_readbacks(self) -> list[McaReplayReadback]:
        """Return every captured MCA poll for progressive histogram replay."""
        result: list[McaReplayReadback] = []
        with h5py.File(self.streams_path, "r") as h5:
            for channel in self.mca_readback_channels():
                base = f"mca/ch{channel}/readback"
                checkpoints = h5[f"{base}/checkpoints"]
                assert isinstance(checkpoints, h5py.Dataset)
                for row in self._committed_rows(checkpoints):
                    arrays: dict[str, np.ndarray] = {}
                    for name, dtype in (
                        ("histogram", "<u4"),
                        ("debug1", "<i2"),
                        ("debug2", "<i2"),
                    ):
                        offset = int(row[f"{name}_offset"])
                        count = int(row[f"{name}_count"])
                        dataset = h5[f"{base}/{name}_values"]
                        assert isinstance(dataset, h5py.Dataset)
                        end = offset + count
                        if offset < 0 or count < 0 or end > len(dataset):
                            raise ValueError(
                                f"MCA ch{channel} readback references invalid {name} values"
                            )
                        values = np.asarray(dataset[offset:end], dtype=dtype)
                        values.setflags(write=False)
                        arrays[name] = values
                    events_lost = int(row["events_lost"])
                    result.append(
                        McaReplayReadback(
                            channel=channel,
                            sequence=int(row["sequence"]),
                            host_monotonic_ns=int(row["host_monotonic_ns"]),
                            readback=MCAReadback(
                                histogram=arrays["histogram"],
                                debug1=arrays["debug1"],
                                debug2=arrays["debug2"],
                                count_rate=int(row["count_rate"]),
                                pulse_deadtime_ms=float(row["pulse_deadtime_ms"]),
                                events_lost=None if events_lost < 0 else events_lost,
                                elapsed_time=int(row["elapsed_time"]),
                                pulse_overrange=int(row["pulse_overrange"]),
                                pulse_pileup=int(row["pulse_pileup"]),
                                energy_overrange=int(row["energy_overrange"]),
                                energy_estimation_error=int(
                                    row["energy_estimation_error"]
                                ),
                                throughput_error=int(row["throughput_error"]),
                            ),
                        )
                    )
        result.sort(
            key=lambda readback: (
                readback.host_monotonic_ns,
                readback.channel,
                readback.sequence,
            )
        )
        return result

    def mca_event_spectrum(self, channel: int, *, bins: int = 16384) -> np.ndarray:
        """Rebuild a spectrum with the capture-derived list-mode mapping."""
        if bins <= 0:
            raise ValueError("bins must be positive")
        events = self.mca_events(channel)
        if len(events) == 0:
            empty = np.zeros(bins, dtype=np.uint64)
            empty.setflags(write=False)
            return empty
        # DMA frames can contain zero padding.  Preserve a timestamp-zero event
        # when any other field is populated; discard only records that are zero
        # in every byte.
        populated = np.any(events.view(np.uint8).reshape(len(events), -1) != 0, axis=1)
        channels = np.right_shift(events["trapezoid_energy"][populated], 2)
        counts = np.bincount(channels, minlength=bins)
        if len(counts) > bins:
            raise ValueError("recorded event maps outside the requested spectrum")
        result = np.asarray(counts[:bins], dtype=np.uint64)
        result.setflags(write=False)
        return result

    def mca_checkpoint(self, channel: int, index: int = -1) -> np.ndarray:
        with h5py.File(self.streams_path, "r") as h5:
            base = f"mca/ch{int(channel)}/readback"
            checkpoints_path = f"{base}/checkpoints"
            if checkpoints_path not in h5:
                raise KeyError(f"session has no MCA readbacks for channel {channel}")
            checkpoints = h5[checkpoints_path]
            assert isinstance(checkpoints, h5py.Dataset)
            if len(checkpoints) == 0:
                raise IndexError("session has no MCA checkpoints")
            row = checkpoints[index]
            offset = int(row["histogram_offset"])
            count = int(row["histogram_count"])
            values = np.asarray(h5[f"{base}/histogram_values"][offset : offset + count])
        values.setflags(write=False)
        return values

    def replay_mca_spectra(self) -> list[Spectrum]:
        """Return one deterministic analysis input for every recorded MCA channel."""
        event_channels = set(self.mca_event_channels())
        readback_channels = set(self.mca_readback_channels())
        spectra: list[Spectrum] = []
        for channel in sorted(event_channels | readback_channels):
            if channel in event_channels:
                counts = self.mca_event_spectrum(channel)
                replay_source = "list_mode_events"
                history = ("session replay: trapezoid_energy >> 2",)
            else:
                counts = self.mca_checkpoint(channel)
                replay_source = "hardware_histogram_checkpoint"
                history = ("session replay: final recorded hardware histogram",)
            spectra.append(
                Spectrum.create(
                    label=f"Session {self.manifest['session_id'][:8]} MCA ch{channel}",
                    counts=counts,
                    source="mca",
                    metadata={
                        "channel": channel,
                        "session_id": self.manifest["session_id"],
                        "session_path": str(self.directory),
                        "replay_source": replay_source,
                        "capture_complete": self.capture_complete,
                        "integrity": self.integrity,
                        "annotations": self.annotations(),
                    },
                    history=history,
                )
            )
        return spectra
