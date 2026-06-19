# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""The pure source-acquisition loop run by the child process.

:func:`run_source_loop` reads byte chunks from a source, feeds them through the
COBS/CBOR decoder, applies the boundary :class:`BoundaryDropPolicy`, and pushes
**batched** decoded frames (plus decode-error, stats, raw-byte-count, overflow
and lifecycle envelopes) onto an outbound sink. It is written against duck-typed
collaborators — a ``source`` (open/is_open/chunks/close), a ``decoder``
(feed/errors_drained/stats), an outbound ``sink`` (put/qsize), and a ``should_run``
predicate — so the whole loop can run with in-process fakes, no real OS
process, serial port or BLE radio required. The real child entry point
(``child_main``) merely builds the concrete collaborators and calls this.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from data_forwarder_host.pipeline.drop_policy import BoundaryDropPolicy
from data_forwarder_host.pipeline.ipc import (
    FAILED,
    OPENED,
    STOPPED,
    DecodeErrors,
    FrameBatch,
    Lifecycle,
    Overflow,
    RawByteCount,
    StatsUpdate,
)

log = logging.getLogger(__name__)

#: Default maximum frames per outbound batch.
DEFAULT_BATCH_MAX = 256
#: Emit a stats snapshot every this many processed chunks.
_STATS_EVERY_CHUNKS = 32


class _Sink(Protocol):
    def put(self, item: Any) -> None: ...
    def qsize(self) -> int: ...


class _SourceLike(Protocol):
    @property
    def is_open(self) -> bool: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
    def chunks(self) -> Iterator[bytes]: ...


class _DecoderLike(Protocol):
    def feed(self, chunk: bytes) -> Iterator[Any]: ...
    def errors_drained(self) -> Iterator[Any]: ...
    def stats(self) -> Any: ...


def run_source_loop(
    *,
    source: _SourceLike,
    decoder: _DecoderLike,
    sink: _Sink,
    drop_policy: BoundaryDropPolicy,
    should_run: Callable[[], bool],
    batch_max: int = DEFAULT_BATCH_MAX,
) -> None:
    """Run the read→decode→drop→batch loop until the source ends or stop is asked.

    Lifecycle envelopes bracket the run: :data:`OPENED` after a successful open,
    then either :data:`FAILED` (open/loop error, with detail) or :data:`STOPPED`
    on a clean finish. The outbound queue depth (``sink.qsize()``) is the
    back-pressure signal handed to *drop_policy* for each ``sensor_data`` frame.
    """
    try:
        if not source.is_open:
            source.open()
    except Exception as exc:  # open failure is terminal
        log.exception("source open failed in child loop")
        sink.put(Lifecycle(FAILED, str(exc)))
        sink.put(Lifecycle(STOPPED))
        return

    sink.put(Lifecycle(OPENED))
    batch: list[Any] = []

    def flush() -> None:
        if batch:
            sink.put(FrameBatch(tuple(batch)))
            batch.clear()

    try:
        chunk_count = 0
        for chunk in source.chunks():
            if not should_run():
                break
            if not chunk:
                continue
            sink.put(RawByteCount(len(chunk)))
            for msg in decoder.feed(chunk):
                decision = drop_policy.evaluate(msg.kind, sink.qsize())
                if decision.overflow_edge:
                    sink.put(Overflow(drop_policy.dropped))
                if not decision.keep:
                    continue
                batch.append(msg)
                if len(batch) >= batch_max:
                    flush()
            errs = tuple(decoder.errors_drained())
            if errs:
                # Flush frames first so error/frame ordering is preserved.
                flush()
                sink.put(DecodeErrors(errs))
            chunk_count += 1
            if chunk_count % _STATS_EVERY_CHUNKS == 0:
                flush()
                sink.put(StatsUpdate(decoder.stats()))
    except Exception as exc:
        log.exception("source child loop crashed")
        flush()
        sink.put(Lifecycle(FAILED, str(exc)))
    finally:
        flush()
        # A final stats snapshot so the GUI sees the closing counts.
        try:
            sink.put(StatsUpdate(decoder.stats()))
        except Exception:
            log.exception("final stats snapshot failed")
        try:
            source.close()
        except Exception:
            log.exception("source.close() failed in child loop")
        sink.put(Lifecycle(STOPPED))
