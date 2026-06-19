# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Recorder — RAM-first capture with transparent spill-to-disk shards.

Buffering is gated on recording: nothing is accumulated unless a
recording is active. When the RAM buffer grows past internal thresholds, the
recorder transparently activates ``.npy`` shard storage so capture continues
uninterrupted. The internal queue is bounded; on overflow capture
continues and a single ``RECORDER_OVERFLOW`` event is surfaced.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from data_forwarder_host.core.error_log import ErrorCategory, ErrorLog, ErrorSummary
from data_forwarder_host.protocol.base import DecodedMessage
from data_forwarder_host.utils.paths import app_cache_dir
from data_forwarder_host.utils.timeutil import utc_iso8601_now

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage interfaces
# ---------------------------------------------------------------------------

Row = tuple[int, int | None, int | None, str | None, tuple[float, ...]]


class RecordingStorage(Protocol):
    def iter_rows(self) -> Iterable[Row]:
        """Yields ``(t_host_ms, t_device_ms, seq, label, channels)``."""


@dataclass(frozen=True)
class Recording:
    session_tag: str
    started_utc: str
    stopped_utc: str
    session_info: dict[str, Any] | None
    channel_names: tuple[str, ...]
    storage: RecordingStorage
    error_summary: ErrorSummary
    incomplete: bool


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------


class RAMStorage:
    """All rows kept in process memory."""

    def __init__(self) -> None:
        self._rows: list[Row] = []
        self._bytes_estimate: int = 0

    def append(self, row: Row) -> None:
        self._rows.append(row)
        # Rough estimate: 8B per float channel + ~32B fixed.
        self._bytes_estimate += 32 + 8 * len(row[4])

    def extend(self, rows: Iterable[Row]) -> None:
        for r in rows:
            self.append(r)

    @property
    def size_bytes(self) -> int:
        return self._bytes_estimate

    @property
    def row_count(self) -> int:
        return len(self._rows)

    def iter_rows(self) -> Iterator[Row]:
        yield from self._rows


class SpillStorage:
    """Append-only ``.npy`` shards under ``<user_cache_dir>/spill/<id>/``.

    This is an internal scratch area (not the user's CSV output directory).
    """

    SHARD_BYTES = 32 * 1024 * 1024  # ~32 MB

    def __init__(self, session_id: str, channel_count: int) -> None:
        self._root = app_cache_dir() / "spill" / session_id
        self._root.mkdir(parents=True, exist_ok=True)
        self._channel_count = channel_count
        self._shard_idx = 0
        self._shard_rows: list[Row] = []
        self._shard_bytes = 0
        self._sealed_shards: list[Path] = []

    @property
    def root(self) -> Path:
        return self._root

    def append(self, row: Row) -> None:
        self._shard_rows.append(row)
        self._shard_bytes += 32 + 8 * len(row[4])
        if self._shard_bytes >= self.SHARD_BYTES:
            self._flush_shard()

    def extend(self, rows: Iterable[Row]) -> None:
        for r in rows:
            self.append(r)

    def seal(self) -> None:
        if self._shard_rows:
            self._flush_shard()

    def cleanup(self) -> None:
        try:
            shutil.rmtree(self._root, ignore_errors=True)
        except Exception:
            log.exception("failed to remove spill directory %s", self._root)

    def iter_rows(self) -> Iterator[Row]:
        for shard in self._sealed_shards:
            try:
                arr = np.load(shard, allow_pickle=True)
            except Exception:
                log.exception("failed to read shard %s", shard)
                continue
            for entry in arr:
                yield (
                    int(entry["t_host_ms"]),
                    None if entry["t_device_ms"] < 0 else int(entry["t_device_ms"]),
                    None if entry["seq"] < 0 else int(entry["seq"]),
                    None if entry["label"] == "" else str(entry["label"]),
                    tuple(float(v) for v in entry["channels"]),
                )
        # Any unflushed tail kept in RAM
        for row in self._shard_rows:
            yield row

    def _flush_shard(self) -> None:
        if not self._shard_rows:
            return
        dtype = np.dtype([
            ("t_host_ms", np.int64),
            ("t_device_ms", np.int64),
            ("seq", np.int64),
            ("label", "U64"),
            ("channels", np.float64, (self._channel_count,)),
        ])
        arr = np.empty(len(self._shard_rows), dtype=dtype)
        for i, (t, td, seq, lbl, ch) in enumerate(self._shard_rows):
            arr[i]["t_host_ms"] = t
            arr[i]["t_device_ms"] = td if td is not None else -1
            arr[i]["seq"] = seq if seq is not None else -1
            arr[i]["label"] = lbl or ""
            if len(ch) == self._channel_count:
                arr[i]["channels"] = ch
            else:
                # Channel count mismatch — pad/truncate.
                padded = list(ch) + [0.0] * self._channel_count
                arr[i]["channels"] = padded[: self._channel_count]
        path = self._root / f"shard_{self._shard_idx:06d}.npy"
        try:
            np.save(path, arr, allow_pickle=False)
            self._sealed_shards.append(path)
        except Exception:
            log.exception("failed to write shard %s", path)
        self._shard_idx += 1
        self._shard_rows = []
        self._shard_bytes = 0


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class Recorder(QObject):
    """Records ``sensor_data`` messages from a single session.

    RAM-first; spills to disk shards when either internal threshold is exceeded
    (256 MB OR 10 minutes — whichever comes first). These thresholds are fixed
    internal constants and are NOT user-configurable.
    """

    started = Signal()
    stopped = Signal(object)            # Recording

    SPILL_BYTES = 256 * 1024 * 1024
    SPILL_SECONDS = 10 * 60
    QUEUE_BOUND = 4096

    def __init__(
        self,
        *,
        session_id: str,
        session_tag: str,
        error_log: ErrorLog,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._session_tag = session_tag
        self._error_log = error_log

        self._recording: bool = False
        self._started_utc: str | None = None
        self._session_info: dict[str, Any] | None = None
        self._channel_names: tuple[str, ...] = ()
        self._channel_count: int = 0
        self._t_started_monotonic: float = 0.0
        self._overflow_emitted: bool = False

        self._ram: RAMStorage | None = None
        self._spill: SpillStorage | None = None
        self._using_spill: bool = False

        self._queue: queue.Queue[Row | None] = queue.Queue(maxsize=self.QUEUE_BOUND)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        self._current: Recording | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_recording(self) -> bool:
        return self._recording

    @property
    def using_spill(self) -> bool:
        """True once capture has switched from RAM to on-disk spill shards."""
        return self._using_spill

    @property
    def buffered_rows(self) -> int:
        """Rows currently held in RAM (0 once spilling has freed them)."""
        with self._lock:
            return self._ram.row_count if self._ram is not None else 0

    @property
    def buffered_bytes(self) -> int:
        """Estimated RAM bytes currently held (0 once spilling has freed them)."""
        with self._lock:
            return self._ram.size_bytes if self._ram is not None else 0

    def elapsed_seconds(self) -> float | None:
        """Return seconds since this recording started, or *None* if not recording."""
        if not self._recording:
            return None
        return time.monotonic() - self._t_started_monotonic

    def current(self) -> Recording | None:
        return self._current

    def set_session_info(self, info: dict[str, Any], channel_names: tuple[str, ...]) -> None:
        with self._lock:
            self._session_info = info
            if channel_names:
                self._channel_names = channel_names

    def start(self, *, channel_names: tuple[str, ...]) -> None:
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._overflow_emitted = False
            self._started_utc = utc_iso8601_now()
            self._t_started_monotonic = time.monotonic()
            self._channel_names = channel_names or self._channel_names
            self._channel_count = len(self._channel_names) if self._channel_names else 0
            self._ram = RAMStorage()
            self._spill = None
            self._using_spill = False
            self._stop_event.clear()
            # Drain anything left over from a previous run.
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
            self._thread = threading.Thread(
                target=self._writer_loop, name=f"recorder-{self._session_id}", daemon=True
            )
            self._thread.start()
        self.started.emit()

    def stop(self) -> Recording:
        with self._lock:
            if not self._recording:
                if self._current is not None:
                    return self._current
                raise RuntimeError("recorder is not running")
            self._recording = False

        # Signal writer thread to drain and exit.
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        with self._lock:
            if self._spill is not None:
                self._spill.seal()
            storage: RecordingStorage
            if self._using_spill and self._spill is not None:
                storage = self._spill
            else:
                storage = self._ram if self._ram is not None else RAMStorage()

            # If recording started before any channel metadata was known (e.g.
            # the v1-style "Record" begins streaming and capture in one click,
            # before the device's session_info arrives), infer the channel
            # names from the first captured row so the CSV header always matches
            # the data width. Without this the header collapses to just the
            # device-time column.
            channel_names = self._channel_names
            if not channel_names:
                for row in storage.iter_rows():
                    n = len(row[4])
                    if n:
                        channel_names = tuple(f"ch{i}" for i in range(n))
                    break

            summary = self._error_log.summary()
            rec = Recording(
                session_tag=self._session_tag,
                started_utc=self._started_utc or utc_iso8601_now(),
                stopped_utc=utc_iso8601_now(),
                session_info=self._session_info,
                channel_names=channel_names,
                storage=storage,
                error_summary=summary,
                incomplete=summary.incomplete,
            )
            self._current = rec
        self.stopped.emit(rec)
        return rec

    def cleanup_spill(self) -> None:
        """Delete the on-disk spill folder. Caller must confirm with the user."""
        with self._lock:
            if self._spill is not None:
                self._spill.cleanup()
                self._spill = None
            self._current = None

    # ------------------------------------------------------------------
    # Hot path (gated on recording)
    # ------------------------------------------------------------------

    def on_message(self, msg: DecodedMessage) -> None:
        # Buffering is gated on recording: while not recording, nothing is
        # accumulated (live plots/stats may still update elsewhere).
        if not self._recording or msg.kind != "sensor_data" or msg.channels is None:
            return
        row: Row = (msg.t_host_ms, msg.t_device_ms, msg.seq, msg.label, msg.channels)
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            # Bounded queue overflow: keep capturing (drop the oldest queued row
            # to make room) and surface a single RECORDER_OVERFLOW event.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(row)
            except queue.Full:
                pass
            if not self._overflow_emitted:
                self._overflow_emitted = True
                self._error_log.add(
                    ErrorCategory.RECORDER_OVERFLOW,
                    "recorder queue overflow; writer falling behind",
                )

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            with self._lock:
                if self._using_spill:
                    # Once spilling, rows go to disk shards only. RAM was freed
                    # in _begin_spill so the in-process object count (and the
                    # cyclic-GC scan cost) stops growing during long recordings.
                    if self._spill is not None:
                        self._spill.append(item)
                    continue
                if self._ram is None:
                    continue
                self._ram.append(item)
                if self._should_spill():
                    self._begin_spill()

    def _should_spill(self) -> bool:
        if self._ram is None:
            return False
        if self._ram.size_bytes >= self.SPILL_BYTES:
            return True
        elapsed = time.monotonic() - self._t_started_monotonic
        return elapsed >= self.SPILL_SECONDS

    def _begin_spill(self) -> None:
        if self._channel_count == 0 and self._ram is not None and self._ram.row_count:
            # Infer channel count from the first row.
            for row in self._ram.iter_rows():
                self._channel_count = len(row[4])
                break
        self._spill = SpillStorage(self._session_id, max(self._channel_count, 1))
        self._using_spill = True
        if self._ram is not None:
            self._spill.extend(self._ram.iter_rows())
            # Release the RAM rows now that they live in the spill shards.
            # Keeping them would let the list of tuples grow for the whole
            # recording, driving unbounded memory use and ever-worsening
            # cyclic-GC pauses that stall the entire GUI event loop.
            self._ram = None


# ---------------------------------------------------------------------------
# Non-blocking recording-stop CSV dump
# ---------------------------------------------------------------------------


class CsvDumpJob(QObject):
    """Drive a chunked CSV dump from the Qt event loop without blocking it.

    Writing a finished recording to CSV happens at the recording-stop event. For
    a large capture that write can take long enough to freeze the GUI if done in
    one blocking call, so this job advances a :class:`RecordingCsvDump`-style
    *dump* one bounded chunk per zero-delay timer tick: between ticks the event
    loop is free to paint (the saving progress bar) and service input, so the UI
    stays responsive while the file is written. ``progress`` reports
    ``(rows_written, total)`` after every chunk (``total`` is ``-1`` when the
    recording spilled to disk and the row count is not known up-front);
    ``finished`` carries the written path and ``failed`` an error message.

    The *dump* only needs ``step() -> bool``, ``rows_written``, ``total_rows``
    and ``path`` — :class:`~data_forwarder_host.core.csv_writer.RecordingCsvDump`
    satisfies this.
    """

    progress = Signal(int, int)   # rows_written, total (-1 if unknown)
    finished = Signal(str)        # written CSV path
    failed = Signal(str)          # error message

    def __init__(self, dump: Any, *, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._dump = dump
        self._timer = QTimer(self)
        self._timer.setInterval(0)  # fire as soon as the event loop is idle
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        """Begin writing; chunks are processed on subsequent event-loop ticks."""
        self._timer.start()

    def cancel(self) -> None:
        """Stop processing further chunks (the partial file is left in place)."""
        self._timer.stop()

    def _tick(self) -> None:
        try:
            more = self._dump.step()
        except Exception as exc:  # noqa: BLE001 — surfaced to the GUI verbatim
            self._timer.stop()
            log.exception("recording CSV dump failed")
            self.failed.emit(str(exc))
            return
        total = self._dump.total_rows if self._dump.total_rows is not None else -1
        self.progress.emit(self._dump.rows_written, total)
        if not more:
            self._timer.stop()
            self.finished.emit(str(self._dump.path))
