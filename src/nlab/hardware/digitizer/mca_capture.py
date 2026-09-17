"""Incremental, bounded-memory MCA list-mode file writers."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class McaDmaOutputMode(StrEnum):
    BINARY = "binary"
    ROOT = "root"
    HDF5 = "hdf5"
    ONLINE = "online"

    @property
    def extension(self) -> str:
        return {
            self.BINARY: ".bin",
            self.ROOT: ".root",
            self.HDF5: ".h5",
            self.ONLINE: "",
        }[self]


@dataclass(frozen=True)
class McaRunSummary:
    """Format-independent outcome of one MCA list-mode run.

    A successful legacy transport is *unverified*, not lossless: only the
    current IIO driver exposes completed-frame and list-deadtime counters.
    """

    channel: int
    mode: McaDmaOutputMode
    path: Path | None
    started_utc: str
    finished_utc: str
    duration_s: float
    records: int
    continuity: str
    diagnostics: dict[str, int | bool]
    error: str | None = None
    metadata_error: str | None = None

    @property
    def rate_hz(self) -> float:
        return self.records / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def sidecar_path(self) -> Path | None:
        return self.path.with_suffix(".run.json") if self.path is not None else None

    def status_text(self) -> str:
        label = {"verified": "Verified", "invalid": "Incomplete", "unverified": "Unverified"}[
            self.continuity
        ]
        detail = f"{label}: {self.records:,} records, {self.rate_hz:,.0f}/s avg"
        dropped = int(self.diagnostics.get("list_deadtime_raw", 0))
        fault = int(self.diagnostics.get("dma_fault", 0))
        if dropped:
            detail += f"; {dropped:,} dropped"
        if fault:
            detail += f"; DMA fault {fault}"
        completed = self.diagnostics.get("completed_frames")
        streamed = self.diagnostics.get("streamed_frames")
        if completed is not None and streamed is not None and completed != streamed:
            detail += f"; frame mismatch {streamed}/{completed}"
        if self.error:
            detail += f"; {self.error}"
        if self.metadata_error:
            detail += f"; summary not saved: {self.metadata_error}"
        return detail

    def write_sidecar(self) -> Path | None:
        """Save the same summary beside binary, HDF5, and ROOT captures.

        Online-only mode intentionally creates no file.
        """
        sidecar = self.sidecar_path
        if sidecar is None or self.path is None or not self.path.exists():
            return None
        document = {
            "format": "nlab-mca-run-summary-v1",
            "capture_file": self.path.name,
            "channel": self.channel,
            "output_mode": self.mode.value,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "duration_s": self.duration_s,
            "records": self.records,
            "average_rate_hz": self.rate_hz,
            "continuity": self.continuity,
            "diagnostics": self.diagnostics,
            "error": self.error,
        }
        with sidecar.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return sidecar


_CANONICAL_EVENT_DTYPE = np.dtype(
    [
        ("timestamp", "<u8"),
        ("long_gate", "<u2"),
        ("short_gate", "<u2"),
    ]
)
_WRITER_QUEUE_BATCHES = 32
_STRUCTURED_FLUSH_EVENTS = 8192
_STRUCTURED_FLUSH_SECONDS = 1.0
_QUEUE_POLL_SECONDS = 0.05


def _canonical_events(events: np.ndarray) -> np.ndarray:
    names = events.dtype.names or ()
    converted = np.empty(len(events), dtype=_CANONICAL_EVENT_DTYPE)
    converted["timestamp"] = events["timestamp"]
    if {"energy", "short_energy"}.issubset(names):
        converted["long_gate"] = events["energy"]
        converted["short_gate"] = events["short_energy"]
    elif {"trapezoid_energy", "charge_energy"}.issubset(names):
        converted["long_gate"] = events["trapezoid_energy"]
        converted["short_gate"] = events["charge_energy"]
    else:
        raise ValueError(f"unsupported MCA event fields: {', '.join(names)}")
    return converted


class _CaptureSink(ABC):
    @abstractmethod
    def append(self, raw: bytes, events: np.ndarray) -> None: ...

    @abstractmethod
    def close(self, complete: bool) -> None: ...


class _BinarySink(_CaptureSink):
    def __init__(self, path: Path, header: bytes) -> None:
        self._file = open(path, "xb", buffering=128 * 1024)
        self._file.write(header)
        self._file.flush()

    def append(self, raw: bytes, events: np.ndarray) -> None:
        del events
        self._file.write(raw)

    def close(self, complete: bool) -> None:
        del complete
        self._file.close()


class _BufferedStructuredSink(_CaptureSink, ABC):
    def __init__(self) -> None:
        self._pending: list[np.ndarray] = []
        self._pending_events = 0
        self._last_flush = time.monotonic()

    def append(self, raw: bytes, events: np.ndarray) -> None:
        del raw
        canonical = _canonical_events(events)
        self._pending.append(canonical)
        self._pending_events += len(canonical)
        if (
            self._pending_events >= _STRUCTURED_FLUSH_EVENTS
            or time.monotonic() - self._last_flush >= _STRUCTURED_FLUSH_SECONDS
        ):
            self._flush_pending()

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        batch = self._pending[0] if len(self._pending) == 1 else np.concatenate(self._pending)
        self._write_batch(batch)
        self._pending.clear()
        self._pending_events = 0
        self._last_flush = time.monotonic()

    @abstractmethod
    def _write_batch(self, events: np.ndarray) -> None: ...


class _Hdf5Sink(_BufferedStructuredSink):
    def __init__(self, path: Path, configuration_yaml: str) -> None:
        super().__init__()
        import h5py

        self._file = h5py.File(path, "x", libver="latest")
        text_type = h5py.string_dtype(encoding="utf-8")
        self._file.create_dataset("configuration_yaml", data=configuration_yaml, dtype=text_type)
        self._events = self._file.create_dataset(
            "events",
            shape=(0,),
            maxshape=(None,),
            chunks=(_STRUCTURED_FLUSH_EVENTS,),
            dtype=_CANONICAL_EVENT_DTYPE,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        self._events.attrs["timestamp_unit"] = "8 ns ticks"
        self._events.attrs["gate_unit"] = "raw uint16"
        self._committed = self._file.create_dataset("committed_events", data=np.uint64(0))
        self._complete = self._file.create_dataset("capture_complete", data=np.bool_(False))
        self._file.attrs["format"] = "nlab-mca-listmode"
        self._file.attrs["format_version"] = 1
        self._file.swmr_mode = True
        self._count = 0

    def _write_batch(self, events: np.ndarray) -> None:
        new_count = self._count + len(events)
        self._events.resize((new_count,))
        self._events[self._count:new_count] = events
        self._events.flush()
        self._committed[()] = np.uint64(new_count)
        self._committed.flush()
        self._count = new_count

    def close(self, complete: bool) -> None:
        try:
            self._flush_pending()
            self._complete[()] = np.bool_(complete)
            self._complete.flush()
            self._file.flush()
        finally:
            self._file.close()


class _RootSink(_BufferedStructuredSink):
    def __init__(self, path: Path, configuration_yaml: str) -> None:
        super().__init__()
        try:
            import uproot
        except ImportError as exc:
            raise RuntimeError(
                "ROOT output requires the 'uproot' package; reinstall nlab with its "
                "current dependencies"
            ) from exc

        self._file = uproot.create(path)
        self._file["configuration_yaml"] = configuration_yaml
        self._file["event_schema"] = (
            '{"timestamp":"uint64, 8 ns ticks","long_gate":"uint16, raw",'
            '"short_gate":"uint16, raw"}'
        )
        self._tree = self._file.mktree(
            "events",
            {"timestamp": "uint64", "long_gate": "uint16", "short_gate": "uint16"},
            title="NLab MCA list-mode events",
        )

    def _write_batch(self, events: np.ndarray) -> None:
        self._tree.extend(
            {
                "timestamp": events["timestamp"],
                "long_gate": events["long_gate"],
                "short_gate": events["short_gate"],
            }
        )

    def close(self, complete: bool) -> None:
        del complete
        try:
            self._flush_pending()
        finally:
            self._file.close()


class McaCaptureWriter:
    """Own a file sink on a dedicated thread with a bounded input queue."""

    def __init__(
        self,
        *,
        path: Path,
        mode: McaDmaOutputMode,
        configuration_yaml: str,
        binary_header: bytes,
    ) -> None:
        if mode is McaDmaOutputMode.ONLINE:
            raise ValueError("online mode has no capture writer")
        self._path = path
        self._mode = mode
        self._configuration_yaml = configuration_yaml
        self._binary_header = binary_header
        self._queue: queue.Queue[object] = queue.Queue(maxsize=_WRITER_QUEUE_BATCHES)
        self._ready = threading.Event()
        self._done = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"mca-{mode.value}-writer",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        self._raise_if_failed()

    def _make_sink(self) -> _CaptureSink:
        if self._mode is McaDmaOutputMode.BINARY:
            return _BinarySink(self._path, self._binary_header)
        if self._mode is McaDmaOutputMode.HDF5:
            return _Hdf5Sink(self._path, self._configuration_yaml)
        if self._mode is McaDmaOutputMode.ROOT:
            return _RootSink(self._path, self._configuration_yaml)
        raise AssertionError(f"unsupported MCA output mode {self._mode}")

    def _run(self) -> None:
        sink: _CaptureSink | None = None
        try:
            sink = self._make_sink()
            self._ready.set()
            complete = False
            while True:
                item = self._queue.get()
                if isinstance(item, _Finish):
                    complete = item.complete
                    break
                if not isinstance(item, _WriteBatch):
                    raise AssertionError(f"unexpected MCA writer item {item!r}")
                sink.append(item.raw, item.events)
            closing_sink = sink
            sink = None
            closing_sink.close(complete)
        except BaseException as exc:
            self._error = exc
            log.exception("MCA %s writer failed for %s", self._mode.value, self._path)
            if sink is not None:
                try:
                    sink.close(False)
                except BaseException:
                    log.exception("MCA writer also failed while closing %s", self._path)
        finally:
            self._ready.set()
            self._done.set()

    def append(self, raw: bytes, events: np.ndarray) -> None:
        if self._mode is McaDmaOutputMode.BINARY:
            item = _WriteBatch(raw=raw, events=np.empty(0, dtype=np.uint8))
        else:
            item = _WriteBatch(raw=b"", events=events.copy())
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(item, timeout=_QUEUE_POLL_SECONDS)
                return
            except queue.Full:
                continue

    def close(self, *, complete: bool) -> None:
        if not self._done.is_set():
            while True:
                self._raise_if_failed()
                try:
                    self._queue.put(_Finish(complete), timeout=_QUEUE_POLL_SECONDS)
                    break
                except queue.Full:
                    continue
            self._done.wait()
        self._thread.join()
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"MCA {self._mode.value} writer failed for {self._path}: {self._error}"
            ) from self._error


@dataclass(frozen=True)
class _WriteBatch:
    raw: bytes
    events: np.ndarray


@dataclass(frozen=True)
class _Finish:
    complete: bool
