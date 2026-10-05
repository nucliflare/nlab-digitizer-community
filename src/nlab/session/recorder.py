"""Crash-evident, versioned recording of measurements already observed by the GUI.

The recorder is deliberately passive: it subscribes to the existing bounded DMA
publishers and accepts copies of controller readbacks.  It never starts, stops, or
reconfigures hardware.  One private thread owns both output files so callers on Qt,
DMA, and polling threads never perform storage I/O.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import h5py
import numpy as np

from nlab import __version__
from nlab.hardware.digitizer.dma import (
    McaEventBuffer,
    ScopeDmaFrame,
    ScopeFrameBuffer,
)
from nlab.hardware.digitizer.iio_listmode import IIO_LM_EVENT_DTYPE, VDPP_ZC_CALC_SCHEMA
from nlab.utils.settings_io import configuration_yaml

log = logging.getLogger(__name__)

SESSION_FORMAT = "nlab-session"
SESSION_FORMAT_VERSION = 1
TIMELINE_FORMAT_VERSION = 1
STREAM_FORMAT_VERSION = 1
ANNOTATION_FORMAT_VERSION = 1

_STOP = object()
_MCA_CHECKPOINT_DTYPE = np.dtype(
    [
        ("sequence", "<u8"),
        ("host_monotonic_ns", "<u8"),
        ("host_utc_ns", "<u8"),
        ("histogram_offset", "<u8"),
        ("histogram_count", "<u4"),
        ("debug1_offset", "<u8"),
        ("debug1_count", "<u4"),
        ("debug2_offset", "<u8"),
        ("debug2_count", "<u4"),
        ("count_rate", "<u8"),
        ("pulse_deadtime_ms", "<f8"),
        ("events_lost", "<i8"),
        ("elapsed_time", "<u8"),
        ("pulse_overrange", "<u8"),
        ("pulse_pileup", "<u8"),
        ("energy_overrange", "<u8"),
        ("energy_estimation_error", "<u8"),
        ("throughput_error", "<u8"),
    ]
)
_MCA_EVENT_BATCH_DTYPE = np.dtype(
    [
        ("sequence", "<u8"),
        ("host_monotonic_ns", "<u8"),
        ("host_utc_ns", "<u8"),
        ("event_offset", "<u8"),
        ("event_count", "<u4"),
    ]
)
_SCOPE_FRAME_DTYPE = np.dtype(
    [
        ("sequence", "<u8"),
        ("host_monotonic_ns", "<u8"),
        ("host_utc_ns", "<u8"),
        ("device_timestamp", "<u8"),
        ("sample_offset", "<u8"),
        ("sample_count", "<u4"),
        ("buffer_samples", "<u4"),
        ("frame_samples", "<u4"),
        ("waveform_samples", "<u4"),
        ("sample_decimation", "<u4"),
        ("sample_period_ns", "<u4"),
        ("padding_bytes", "<u4"),
    ]
)
_VIEWER_FRAME_DTYPE = np.dtype(
    [
        ("sequence", "<u8"),
        ("host_monotonic_ns", "<u8"),
        ("host_utc_ns", "<u8"),
        ("sample_offset", "<u8"),
        ("sample_count", "<u4"),
        ("sample_period_ns", "<u4"),
    ]
)
_CURRENT_SAMPLE_DTYPE = np.dtype(
    [
        ("sequence", "<u8"),
        ("timestamp_ns", "<u8"),
        ("raw_code", "<f8"),
        ("read_latency_ns", "<u8"),
        ("coverage_ns", "<u8"),
        ("hardware_timestamp", "<u8"),
        ("hardware_timestamp_valid", "u1"),
        ("sample_count", "<u4"),
    ]
)

_EMPTY_ANNOTATIONS: dict[str, object] = {
    "format": "nlab-session-annotations",
    "format_version": ANNOTATION_FORMAT_VERSION,
    "label_status": "unlabeled",
    "sample_id": None,
    "isotopes": [],
    "background_only": None,
    "detector_id": None,
    "geometry": {
        "source_detector_distance_mm": None,
        "orientation": None,
    },
    "shielding": [],
    "environment": {"location": None},
    "split_group": None,
    "notes": None,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json_value(value: object) -> object:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append(dataset: h5py.Dataset, values: np.ndarray) -> int:
    values = np.asarray(values, dtype=dataset.dtype).reshape(-1)
    offset = int(dataset.shape[0])
    dataset.resize((offset + len(values),))
    if len(values):
        dataset[offset:] = values
    return offset


def _one_dimensional_dataset(
    group: h5py.Group,
    name: str,
    dtype: np.dtype[Any] | str,
    *,
    chunk_size: int = 4096,
) -> h5py.Dataset:
    return group.create_dataset(
        name,
        shape=(0,),
        maxshape=(None,),
        chunks=(chunk_size,),
        dtype=dtype,
    )


@dataclass(frozen=True)
class _McaReadbackItem:
    channel: int
    host_monotonic_ns: int
    host_utc_ns: int
    values: dict[str, object]


@dataclass(frozen=True)
class _ScopeViewerItem:
    channel: int
    host_monotonic_ns: int
    host_utc_ns: int
    samples: np.ndarray
    sample_period_ns: int


@dataclass(frozen=True)
class _TimelineItem:
    record: dict[str, object]


@dataclass(frozen=True)
class _CurrentSamplesItem:
    channel: int
    samples: np.ndarray
    dropped_samples: int


class _SessionLogHandler(logging.Handler):
    def __init__(self, recorder: SessionRecorder) -> None:
        super().__init__(level=logging.DEBUG)
        self._recorder = recorder

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(__name__):
            return
        try:
            payload: dict[str, object] = {
                "logger": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
            }
            if record.exc_info is not None:
                payload["exception"] = logging.Formatter().formatException(record.exc_info)
            self._recorder.record_timeline("log", record.name, payload)
        except Exception:
            self.handleError(record)


class SessionRecorder:
    """Record a passive snapshot of one application session.

    ``create`` refuses to reuse an existing path.  ``stop`` is the only operation
    that writes ``capture_complete=true``; a crash therefore leaves an explicit
    incomplete bundle rather than an apparently valid truncated replay.
    """

    def __init__(
        self,
        directory: Path,
        configuration: Mapping[str, object],
        *,
        mca_event_buffers: Mapping[int, McaEventBuffer] | None = None,
        scope_frame_buffers: Mapping[int, ScopeFrameBuffer] | None = None,
        queue_capacity: int = 4096,
        application_version: str = __version__,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self.directory = Path(directory)
        self._configuration = dict(configuration)
        self._application_version = application_version
        self._session_id = str(uuid4())
        self._started_utc = _utc_now()
        self._started_monotonic_ns = time.monotonic_ns()
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_capacity)
        self._stop_requested = threading.Event()
        self._ready = threading.Event()
        self._finished = threading.Event()
        self._writer_error: BaseException | None = None
        self._state_lock = threading.Lock()
        self._accepting = True
        self._queue_dropped = 0
        self._integrity_reasons: set[str] = set()
        self._stream_counts: dict[str, int] = {}
        self._mca_subscriptions: dict[int, tuple[McaEventBuffer, McaEventBuffer]] = {}
        self._scope_subscriptions: dict[int, tuple[ScopeFrameBuffer, ScopeFrameBuffer]] = {}
        self._mca_drop_totals: dict[int, int] = {}
        self._scope_drop_totals: dict[int, int] = {}
        self._timeline_sequence = 0

        self.directory.mkdir(parents=False, exist_ok=False)
        (self.directory / "configuration.yaml").write_text(
            configuration_yaml(self._configuration),
            encoding="utf-8",
            newline="\n",
        )
        # Scientific labels are intentionally separate and mutable: an
        # operator can curate isotope/sample truth later without changing or
        # invalidating the immutable acquisition evidence used for replay.
        (self.directory / "annotations.yaml").write_text(
            configuration_yaml(_EMPTY_ANNOTATIONS),
            encoding="utf-8",
            newline="\n",
        )
        self._write_manifest(capture_complete=False, integrity="recording")

        for channel, mca_publisher in (mca_event_buffers or {}).items():
            mca_subscriber = McaEventBuffer(max_batches=512)
            mca_publisher.subscribe(mca_subscriber)
            self._mca_subscriptions[int(channel)] = (mca_publisher, mca_subscriber)
            self._mca_drop_totals[int(channel)] = 0
        for channel, scope_publisher in (scope_frame_buffers or {}).items():
            scope_subscriber = ScopeFrameBuffer(max_frames=512)
            scope_publisher.subscribe(scope_subscriber)
            self._scope_subscriptions[int(channel)] = (scope_publisher, scope_subscriber)
            self._scope_drop_totals[int(channel)] = 0

        self._thread = threading.Thread(
            target=self._writer_main,
            name=f"session-recorder-{self._session_id[:8]}",
            daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self._detach_buffers()
            raise RuntimeError("session recorder did not initialize within 10 seconds")
        if self._writer_error is not None:
            self._detach_buffers()
            raise RuntimeError("session recorder failed to initialize") from self._writer_error

        self._log_handler = _SessionLogHandler(self)
        logging.getLogger().addHandler(self._log_handler)
        self.record_timeline(
            "session_started",
            "session",
            {"session_id": self._session_id, "configuration": "configuration.yaml"},
        )

    @property
    def is_recording(self) -> bool:
        with self._state_lock:
            return self._accepting and self._writer_error is None

    @property
    def writer_error(self) -> BaseException | None:
        return self._writer_error

    def record_timeline(
        self,
        kind: str,
        source: str,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        """Queue one JSON-safe operational, command, telemetry, or log record."""
        record = {
            "format_version": TIMELINE_FORMAT_VERSION,
            "host_utc": _utc_now(),
            "host_monotonic_ns": time.monotonic_ns(),
            "kind": str(kind),
            "source": str(source),
            "payload": _json_value(dict(payload or {})),
        }
        self._put(_TimelineItem(record))

    def record_command(
        self,
        source: str,
        command: str,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        self.record_timeline(
            "command",
            source,
            {"command": command, "parameters": dict(parameters or {})},
        )

    def record_telemetry(self, source: str, values: Mapping[str, object]) -> None:
        self.record_timeline("telemetry", source, values)

    def mark_integrity_issue(self, reason: str) -> None:
        """Make a known capture gap explicit in the finalized manifest."""
        self._mark_invalid(str(reason))
        self.record_timeline(
            "integrity_issue",
            "session",
            {"reason": str(reason)},
        )

    def record_mca_readback(self, channel: int, readback: object) -> None:
        values = {
            name: getattr(readback, name)
            for name in (
                "histogram",
                "debug1",
                "debug2",
                "count_rate",
                "pulse_deadtime_ms",
                "events_lost",
                "elapsed_time",
                "pulse_overrange",
                "pulse_pileup",
                "energy_overrange",
                "energy_estimation_error",
                "throughput_error",
            )
        }
        for name in ("histogram", "debug1", "debug2"):
            values[name] = np.asarray(values[name]).copy()
        self._put(
            _McaReadbackItem(
                int(channel),
                time.monotonic_ns(),
                time.time_ns(),
                values,
            )
        )

    def record_scope_viewer(
        self,
        channel: int,
        samples: object,
        *,
        sample_period_ns: int,
    ) -> None:
        self._put(
            _ScopeViewerItem(
                int(channel),
                time.monotonic_ns(),
                time.time_ns(),
                np.asarray(samples, dtype="<i2").reshape(-1).copy(),
                int(sample_period_ns),
            )
        )

    def record_current_samples(self, channel: int, batch: object) -> None:
        source_samples = tuple(getattr(batch, "samples"))
        rows = np.zeros(len(source_samples), dtype=_CURRENT_SAMPLE_DTYPE)
        for index, sample in enumerate(source_samples):
            rows[index]["sequence"] = int(sample.sequence)
            rows[index]["timestamp_ns"] = int(sample.timestamp_ns)
            rows[index]["raw_code"] = float(sample.raw_code)
            rows[index]["read_latency_ns"] = int(sample.read_latency_ns)
            rows[index]["coverage_ns"] = int(sample.coverage_ns)
            if sample.hardware_timestamp is not None:
                rows[index]["hardware_timestamp"] = int(sample.hardware_timestamp)
                rows[index]["hardware_timestamp_valid"] = 1
            rows[index]["sample_count"] = int(sample.sample_count)
        dropped = int(getattr(batch, "dropped_samples"))
        error = getattr(batch, "error")
        if dropped:
            self._mark_invalid(f"current_ch{int(channel)}_dropped_samples")
        if dropped or error:
            self.record_timeline(
                "current_monitor_status",
                f"current.ch{int(channel)}",
                {"dropped_samples": dropped, "error": error},
            )
        self._put(_CurrentSamplesItem(int(channel), rows, dropped))

    def stop(
        self,
        *,
        final_configuration: Mapping[str, object] | None = None,
        timeout: float = 30.0,
    ) -> Path:
        """Detach producers, drain accepted records, and atomically finalize."""
        with self._state_lock:
            if not self._accepting:
                if self._writer_error is not None:
                    raise RuntimeError("session recorder failed") from self._writer_error
                return self.directory
        if final_configuration is not None:
            self.record_timeline(
                "configuration_snapshot",
                "application",
                {"phase": "final", "configuration": dict(final_configuration)},
            )
        self.record_timeline("session_stopping", "session", {})
        logging.getLogger().removeHandler(self._log_handler)
        self._detach_buffers()
        with self._state_lock:
            self._accepting = False
        self._stop_requested.set()
        try:
            self._queue.put(_STOP, timeout=min(timeout, 1.0))
        except queue.Full:
            # The writer also observes _stop_requested and will finish after
            # draining the full queue; a missing sentinel is safe.
            pass
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self._mark_invalid("writer_stop_timeout")
            self._write_manifest(capture_complete=False, integrity="invalid")
            raise TimeoutError("session recorder did not stop within the timeout")
        if self._writer_error is not None:
            raise RuntimeError("session recorder failed") from self._writer_error
        return self.directory

    def _put(self, item: object) -> None:
        with self._state_lock:
            accepting = self._accepting
        if not accepting:
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._state_lock:
                self._queue_dropped += 1
                self._integrity_reasons.add("recorder_queue_overflow")

    def _mark_invalid(self, reason: str) -> None:
        with self._state_lock:
            self._integrity_reasons.add(reason)

    def _detach_buffers(self) -> None:
        for mca_publisher, mca_subscriber in self._mca_subscriptions.values():
            mca_publisher.unsubscribe(mca_subscriber)
        for scope_publisher, scope_subscriber in self._scope_subscriptions.values():
            scope_publisher.unsubscribe(scope_subscriber)

    def _base_manifest(self) -> dict[str, object]:
        with self._state_lock:
            reasons = sorted(self._integrity_reasons)
            queue_dropped = self._queue_dropped
        return {
            "format": SESSION_FORMAT,
            "format_version": SESSION_FORMAT_VERSION,
            "session_id": self._session_id,
            "application": {"name": "nlab", "version": self._application_version},
            "host": {"platform": platform.platform(), "python": platform.python_version()},
            "started_utc": self._started_utc,
            "started_monotonic_ns": self._started_monotonic_ns,
            "capture_complete": False,
            "integrity": "recording",
            "integrity_reasons": reasons,
            "recorder_queue_dropped": queue_dropped,
            "files": {
                "configuration": "configuration.yaml",
                "annotations": "annotations.yaml",
                "streams": "streams.h5",
                "timeline": "timeline.jsonl",
            },
            "mutable_files": ["annotations.yaml"],
            "stream_counts": dict(sorted(self._stream_counts.items())),
            "replay": {
                "mca_energy_mapping": "trapezoid_energy >> 2",
                "mca_bins": 16384,
                "list_mode_schema": VDPP_ZC_CALC_SCHEMA,
            },
        }

    def _write_manifest(
        self,
        *,
        capture_complete: bool,
        integrity: str,
        hashes: Mapping[str, str] | None = None,
    ) -> None:
        manifest = self._base_manifest()
        manifest["capture_complete"] = capture_complete
        manifest["integrity"] = integrity
        if capture_complete:
            manifest["finished_utc"] = _utc_now()
        if hashes is not None:
            manifest["sha256"] = dict(hashes)
        _atomic_json(self.directory / "manifest.json", manifest)

    def _writer_main(self) -> None:
        streams_path = self.directory / "streams.h5"
        timeline_path = self.directory / "timeline.jsonl"
        try:
            with (
                h5py.File(streams_path, "w", libver="latest") as h5,
                timeline_path.open("w", encoding="utf-8", newline="\n") as timeline,
            ):
                h5.attrs["format"] = SESSION_FORMAT
                h5.attrs["format_version"] = STREAM_FORMAT_VERSION
                h5.attrs["session_id"] = self._session_id
                h5.attrs["list_mode_schema"] = VDPP_ZC_CALC_SCHEMA
                complete = h5.create_dataset("capture_complete", data=np.uint8(0))
                self._ready.set()
                last_flush = time.monotonic()
                stopping = False
                while not stopping:
                    try:
                        item = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        item = None
                    if item is _STOP:
                        stopping = True
                    elif item is not None:
                        self._write_item(h5, timeline, item)
                    self._drain_dma_buffers(h5)
                    if self._stop_requested.is_set() and self._queue.empty():
                        stopping = True
                    if time.monotonic() - last_flush >= 1.0:
                        h5.flush()
                        timeline.flush()
                        last_flush = time.monotonic()

                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is not _STOP:
                        self._write_item(h5, timeline, item)
                self._drain_dma_buffers(h5)
                complete[()] = np.uint8(1)
                h5.flush()
                timeline.flush()
                os.fsync(timeline.fileno())

            with self._state_lock:
                valid = not self._integrity_reasons
            hashes = {
                name: _sha256(self.directory / filename)
                for name, filename in {
                    "configuration.yaml": "configuration.yaml",
                    "streams.h5": "streams.h5",
                    "timeline.jsonl": "timeline.jsonl",
                }.items()
            }
            self._write_manifest(
                capture_complete=True,
                integrity="verified" if valid else "invalid",
                hashes=hashes,
            )
        except BaseException as exc:
            self._writer_error = exc
            self._mark_invalid(f"writer_error:{type(exc).__name__}")
            log.exception("Session recorder writer failed")
            try:
                self._write_manifest(capture_complete=False, integrity="invalid")
            except Exception:
                log.exception("Failed to update the incomplete session manifest")
        finally:
            self._ready.set()
            self._finished.set()

    def _write_item(
        self,
        h5: h5py.File,
        timeline: Any,
        item: object,
    ) -> None:
        if isinstance(item, _TimelineItem):
            record = dict(item.record)
            record["sequence"] = self._timeline_sequence
            self._timeline_sequence += 1
            timeline.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
            self._stream_counts["timeline_records"] = self._timeline_sequence
        elif isinstance(item, _McaReadbackItem):
            self._write_mca_readback(h5, item)
        elif isinstance(item, _ScopeViewerItem):
            self._write_scope_viewer(h5, item)
        elif isinstance(item, _CurrentSamplesItem):
            self._write_current_samples(h5, item)
        else:
            raise TypeError(f"unsupported session item {type(item).__name__}")

    def _drain_dma_buffers(self, h5: h5py.File) -> None:
        for channel, (_mca_publisher, mca_subscriber) in self._mca_subscriptions.items():
            batches, dropped = mca_subscriber.drain_timed()
            previous = self._mca_drop_totals[channel]
            if dropped > previous:
                self._mark_invalid(f"mca_ch{channel}_subscriber_dropped_records")
                self._mca_drop_totals[channel] = dropped
            for received_ns, events in batches:
                self._write_mca_events(
                    h5,
                    channel,
                    events,
                    host_monotonic_ns=received_ns,
                )
        for channel, (
            _scope_publisher,
            scope_subscriber,
        ) in self._scope_subscriptions.items():
            frames, dropped = scope_subscriber.drain()
            if dropped:
                self._scope_drop_totals[channel] += dropped
                self._mark_invalid(f"scope_ch{channel}_subscriber_dropped_frames")
            for frame in frames:
                self._write_scope_dma_frame(h5, channel, frame)

    def _write_mca_events(
        self,
        h5: h5py.File,
        channel: int,
        events: np.ndarray,
        *,
        host_monotonic_ns: int,
    ) -> None:
        group = h5.require_group(f"mca/ch{channel}/list_mode")
        dataset = group.get("events")
        if dataset is None:
            dataset = _one_dimensional_dataset(
                group,
                "events",
                IIO_LM_EVENT_DTYPE,
                chunk_size=4096,
            )
            _one_dimensional_dataset(
                group,
                "batches",
                _MCA_EVENT_BATCH_DTYPE,
                chunk_size=256,
            )
        assert isinstance(dataset, h5py.Dataset)
        batches = group["batches"]
        assert isinstance(batches, h5py.Dataset)
        typed = np.asarray(events, dtype=IIO_LM_EVENT_DTYPE).reshape(-1)
        offset = _append(dataset, typed)
        row = np.zeros(1, dtype=_MCA_EVENT_BATCH_DTYPE)
        row["sequence"] = len(batches)
        row["host_monotonic_ns"] = host_monotonic_ns
        row["host_utc_ns"] = time.time_ns()
        row["event_offset"] = offset
        row["event_count"] = len(typed)
        _append(batches, row)
        dataset.attrs["committed_records"] = int(dataset.shape[0])
        batches.attrs["committed_records"] = int(batches.shape[0])
        self._stream_counts[f"mca_ch{channel}_list_mode_events"] = int(dataset.shape[0])
        self._stream_counts[f"mca_ch{channel}_list_mode_batches"] = int(batches.shape[0])

    def _write_mca_readback(self, h5: h5py.File, item: _McaReadbackItem) -> None:
        group = h5.require_group(f"mca/ch{item.channel}/readback")
        checkpoints = group.get("checkpoints")
        if checkpoints is None:
            checkpoints = _one_dimensional_dataset(
                group, "checkpoints", _MCA_CHECKPOINT_DTYPE, chunk_size=64
            )
            _one_dimensional_dataset(group, "histogram_values", "<u4")
            _one_dimensional_dataset(group, "debug1_values", "<i2")
            _one_dimensional_dataset(group, "debug2_values", "<i2")
        assert isinstance(checkpoints, h5py.Dataset)
        histogram = np.asarray(item.values["histogram"], dtype="<u4").reshape(-1)
        debug1 = np.asarray(item.values["debug1"], dtype="<i2").reshape(-1)
        debug2 = np.asarray(item.values["debug2"], dtype="<i2").reshape(-1)
        histogram_values = group["histogram_values"]
        debug1_values = group["debug1_values"]
        debug2_values = group["debug2_values"]
        assert isinstance(histogram_values, h5py.Dataset)
        assert isinstance(debug1_values, h5py.Dataset)
        assert isinstance(debug2_values, h5py.Dataset)
        row = np.zeros(1, dtype=_MCA_CHECKPOINT_DTYPE)
        row["sequence"] = len(checkpoints)
        row["host_monotonic_ns"] = item.host_monotonic_ns
        row["host_utc_ns"] = item.host_utc_ns
        row["histogram_offset"] = _append(histogram_values, histogram)
        row["histogram_count"] = len(histogram)
        row["debug1_offset"] = _append(debug1_values, debug1)
        row["debug1_count"] = len(debug1)
        row["debug2_offset"] = _append(debug2_values, debug2)
        row["debug2_count"] = len(debug2)
        for name in (
            "count_rate",
            "pulse_deadtime_ms",
            "elapsed_time",
            "pulse_overrange",
            "pulse_pileup",
            "energy_overrange",
            "energy_estimation_error",
            "throughput_error",
        ):
            row[name] = item.values[name]
        events_lost = item.values["events_lost"]
        row["events_lost"] = (
            -1
            if events_lost is None or isinstance(events_lost, bool)
            else int(cast(int, events_lost))
        )
        _append(checkpoints, row)
        checkpoints.attrs["committed_records"] = int(checkpoints.shape[0])
        self._stream_counts[f"mca_ch{item.channel}_readbacks"] = int(checkpoints.shape[0])

    def _write_scope_dma_frame(
        self,
        h5: h5py.File,
        channel: int,
        frame: ScopeDmaFrame,
    ) -> None:
        group = h5.require_group(f"scope/ch{channel}/dma")
        rows = group.get("frames")
        if rows is None:
            rows = _one_dimensional_dataset(group, "frames", _SCOPE_FRAME_DTYPE, chunk_size=128)
            _one_dimensional_dataset(group, "samples", "<i2", chunk_size=16384)
        assert isinstance(rows, h5py.Dataset)
        samples = group["samples"]
        assert isinstance(samples, h5py.Dataset)
        row = np.zeros(1, dtype=_SCOPE_FRAME_DTYPE)
        geometry = frame.geometry
        row["sequence"] = len(rows)
        row["host_monotonic_ns"] = frame.received_ns
        row["host_utc_ns"] = time.time_ns()
        row["device_timestamp"] = frame.timestamp
        row["sample_offset"] = _append(samples, frame.samples)
        row["sample_count"] = len(frame.samples)
        row["buffer_samples"] = geometry.buffer_samples
        row["frame_samples"] = geometry.frame_samples
        row["waveform_samples"] = geometry.waveform_samples
        row["sample_decimation"] = geometry.sample_decimation
        row["sample_period_ns"] = geometry.sample_period_ns
        row["padding_bytes"] = geometry.padding_bytes
        _append(rows, row)
        rows.attrs["committed_records"] = int(rows.shape[0])
        self._stream_counts[f"scope_ch{channel}_dma_frames"] = int(rows.shape[0])

    def _write_scope_viewer(self, h5: h5py.File, item: _ScopeViewerItem) -> None:
        group = h5.require_group(f"scope/ch{item.channel}/viewer")
        rows = group.get("frames")
        if rows is None:
            rows = _one_dimensional_dataset(group, "frames", _VIEWER_FRAME_DTYPE, chunk_size=128)
            _one_dimensional_dataset(group, "samples", "<i2", chunk_size=4096)
        assert isinstance(rows, h5py.Dataset)
        samples = group["samples"]
        assert isinstance(samples, h5py.Dataset)
        row = np.zeros(1, dtype=_VIEWER_FRAME_DTYPE)
        row["sequence"] = len(rows)
        row["host_monotonic_ns"] = item.host_monotonic_ns
        row["host_utc_ns"] = item.host_utc_ns
        row["sample_offset"] = _append(samples, item.samples)
        row["sample_count"] = len(item.samples)
        row["sample_period_ns"] = item.sample_period_ns
        _append(rows, row)
        rows.attrs["committed_records"] = int(rows.shape[0])
        self._stream_counts[f"scope_ch{item.channel}_viewer_frames"] = int(rows.shape[0])

    def _write_current_samples(self, h5: h5py.File, item: _CurrentSamplesItem) -> None:
        group = h5.require_group(f"current/ch{item.channel}")
        samples = group.get("samples")
        if samples is None:
            samples = _one_dimensional_dataset(
                group,
                "samples",
                _CURRENT_SAMPLE_DTYPE,
                chunk_size=4096,
            )
            group.attrs["dropped_samples"] = 0
        assert isinstance(samples, h5py.Dataset)
        _append(samples, item.samples)
        group.attrs["dropped_samples"] = int(group.attrs["dropped_samples"]) + int(
            item.dropped_samples
        )
        samples.attrs["committed_records"] = int(samples.shape[0])
        self._stream_counts[f"current_ch{item.channel}_samples"] = int(samples.shape[0])
