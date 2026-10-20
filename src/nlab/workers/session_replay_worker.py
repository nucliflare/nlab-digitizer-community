"""Ordered offline MCA analysis, independent of Qt painting and hardware."""

from __future__ import annotations

import heapq
import threading
from collections.abc import Iterator
from dataclasses import dataclass, fields
from time import monotonic_ns

import numpy as np

from nlab.analysis.coincidence import (
    CoincidenceAnalyzer,
    CoincidencePeakFit,
    CoincidenceSettings,
    CoincidenceSnapshot,
    fit_coincidence_peak,
)
from nlab.analysis.psd import PsdAccumulator
from nlab.session.replay import McaReplayBatch, McaReplayReadback


@dataclass(frozen=True)
class SessionAnalysisSnapshot:
    readbacks: dict[int, McaReplayReadback]
    psd: dict[int, PsdAccumulator]
    coincidence: CoincidenceSnapshot | None
    timing_fit: CoincidencePeakFit | None
    processed: int
    complete: bool
    error: str | None = None


class SessionAnalysisReplayThread(threading.Thread):
    """Consume every batch, but publish only the latest display state at 10 Hz.

    This thread exclusively advances/closes its HDF5 iterators and owns all
    mutable analyzers. The single-slot mailbox cannot flood Qt's event queue.
    PSD batches are retained here (as before replay's GUI-side retention) so
    re-binning is also asynchronous, including after EOF. Stop wakes timestamp
    waits and the EOF wait; callers join before restoring the live views.
    """

    def __init__(
        self,
        streams: tuple[Iterator[McaReplayBatch | McaReplayReadback], ...],
        *,
        origin_ns: int,
        started_ns: int,
        psd: dict[int, PsdAccumulator],
        coincidence_settings: CoincidenceSettings | None = None,
        coincidence_channels: dict[int, int] | None = None,
        coincidence_intervals: list[tuple[int, int | None]] | None = None,
    ) -> None:
        super().__init__(name="session-analysis-replay", daemon=False)
        self._streams = streams
        self._origin_ns = origin_ns
        self._started_ns = started_ns
        self._psd = psd
        self._batches: dict[int, list[np.ndarray]] = {channel: [] for channel in psd}
        self._settings = coincidence_settings
        self._analyzer = (
            CoincidenceAnalyzer(coincidence_settings)
            if coincidence_settings is not None else None
        )
        self._channels = coincidence_channels or {}
        self._intervals = coincidence_intervals
        self._readbacks: dict[int, McaReplayReadback] = {}
        self._processed = 0
        self._complete = False
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._wake = threading.Event()
        self._pending_psd: dict[int, PsdAccumulator] = {}
        self._latest: SessionAnalysisSnapshot | None = None
        self._last_publish_ns = 0
        self._dirty = False

    def request_stop(self) -> None:
        self._stop_requested.set()
        self._wake.set()

    def request_psd_configuration(self, channel: int, accumulator: PsdAccumulator) -> None:
        """Transfer a fresh, empty accumulator to the analysis thread."""
        with self._lock:
            self._pending_psd[channel] = accumulator
        self._wake.set()

    def take_snapshot(self) -> SessionAnalysisSnapshot | None:
        with self._lock:
            latest, self._latest = self._latest, None
        return latest

    def _reconfigure(self) -> bool:
        with self._lock:
            pending, self._pending_psd = self._pending_psd, {}
        for channel, accumulator in pending.items():
            for events in self._batches[channel]:
                if self._stop_requested.is_set():
                    return False
                accumulator.add_events(events)
            self._psd[channel] = accumulator
        return bool(pending)

    def _publish(self, *, error: str | None = None) -> None:
        coincidence = self._analyzer.snapshot() if self._analyzer is not None else None
        fit = None
        if coincidence is not None:
            # Snapshot arrays never alias the analyzer. Freeze them before handoff.
            for field in fields(coincidence):
                value = getattr(coincidence, field.name)
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
            if self._settings is not None and error is None:
                fit = fit_coincidence_peak(coincidence.delay_counts, self._settings)
        snapshot = SessionAnalysisSnapshot(
            readbacks=dict(self._readbacks),
            psd={channel: accumulator.snapshot() for channel, accumulator in self._psd.items()},
            coincidence=coincidence,
            timing_fit=fit,
            processed=self._processed,
            complete=self._complete,
            error=error,
        )
        with self._lock:
            self._latest = snapshot
        self._last_publish_ns = monotonic_ns()
        self._dirty = False

    @staticmethod
    def _key(item: McaReplayBatch | McaReplayReadback) -> tuple[int, int, int, int]:
        return (
            item.host_monotonic_ns,
            0 if isinstance(item, McaReplayBatch) else 1,
            item.channel,
            item.sequence,
        )

    def run(self) -> None:
        try:
            for item in heapq.merge(*self._streams, key=self._key):
                while not self._stop_requested.is_set():
                    self._wake.clear()
                    if self._reconfigure():
                        self._publish()
                    if self._dirty and monotonic_ns() - self._last_publish_ns >= 100_000_000:
                        self._publish()
                    due = self._started_ns + max(0, item.host_monotonic_ns - self._origin_ns)
                    remaining = due - monotonic_ns()
                    if remaining <= 0:
                        break
                    self._wake.wait(min(remaining / 1e9, 0.1))
                if self._stop_requested.is_set():
                    return
                if isinstance(item, McaReplayReadback):
                    self._readbacks[item.channel] = item
                else:
                    accumulator = self._psd.get(item.channel)
                    if accumulator is not None:
                        self._batches[item.channel].append(item.events)
                        accumulator.add_events(item.events)
                    channel = self._channels.get(item.channel)
                    inside = self._intervals is None or any(
                        start <= item.host_monotonic_ns
                        and (end is None or item.host_monotonic_ns <= end)
                        for start, end in self._intervals
                    )
                    if self._analyzer is not None and channel is not None and inside:
                        self._analyzer.add_batch(channel, item.events)
                self._processed += 1
                self._dirty = True
                if monotonic_ns() - self._last_publish_ns >= 100_000_000:
                    self._publish()
            if self._stop_requested.is_set():
                return
            if self._analyzer is not None:
                self._analyzer.finish()
            self._complete = True
            self._publish()
        except Exception as exc:
            # Never retain a traceback/native reader in the cross-thread mailbox.
            self._publish(error=f"{type(exc).__name__}: {exc}")
            return
        finally:
            for stream in self._streams:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
        # Keep only PSD reconfiguration available at EOF, with no open HDF5 files.
        while not self._stop_requested.is_set():
            self._wake.clear()
            try:
                if self._reconfigure():
                    self._publish()
            except Exception as exc:
                self._publish(error=f"{type(exc).__name__}: {exc}")
                return
            self._wake.wait(0.1)
