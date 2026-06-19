# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Child-process entry point for out-of-process acquisition.

:func:`source_child_main` is the **module-level** target the GUI process launches
with the ``spawn`` start method. It rebuilds the concrete :class:`Source` and the
COBS/CBOR decoder from the picklable :class:`ChildSpec`, wraps the boundary queue
in a :class:`_QueueSink`, and runs the pure :func:`run_source_loop`. No Qt is
imported here and every argument that crosses the boundary is picklable.

The outbound backlog is tracked with a shared integer counter rather than
``Queue.qsize()`` because ``qsize`` is not implemented on macOS — the counter is
incremented by the child on each :class:`FrameBatch` enqueued and decremented by
the GUI host as batches are consumed.
"""

from __future__ import annotations

import logging
from typing import Any

from data_forwarder_host.pipeline.drop_policy import BoundaryDropPolicy
from data_forwarder_host.pipeline.ipc import (
    OPENED,
    STOPPED,
    ChildSpec,
    FrameBatch,
    Lifecycle,
)
from data_forwarder_host.pipeline.source_loop import run_source_loop

log = logging.getLogger(__name__)


class _QueueSink:
    """Adapts a multiprocessing queue + shared counter to the loop's sink API.

    ``put`` enqueues an envelope and, for a :class:`FrameBatch`, bumps the shared
    pending counter; ``qsize`` returns that counter (the GUI-process backlog in
    batches), falling back to ``0`` where neither is available.
    """

    def __init__(self, queue: Any, pending: Any | None = None) -> None:
        self._q = queue
        self._pending = pending

    def put(self, item: Any) -> None:
        if self._pending is not None and isinstance(item, FrameBatch):
            with self._pending.get_lock():
                self._pending.value += 1
        self._q.put(item)

    def qsize(self) -> int:
        if self._pending is not None:
            return int(self._pending.value)
        try:
            return int(self._q.qsize())
        except (NotImplementedError, OSError):
            return 0


def source_child_main(spec: ChildSpec, queue: Any, pending: Any, stop: Any) -> None:
    """Spawn-target: build the real source/decoder and run the acquisition loop."""
    # Imported lazily so the child only pays for what it uses and the module's
    # import graph stays Qt-free.
    from data_forwarder_host.protocol.cobs_cbor_v1 import CobsCborV1
    from data_forwarder_host.source import source_for_kind

    try:
        source = source_for_kind(spec.source_kind)(**spec.source_params)
    except Exception as exc:  # bad config is terminal for the child
        log.exception("child failed to build source")
        sink = _QueueSink(queue, pending)
        from data_forwarder_host.pipeline.ipc import FAILED

        sink.put(Lifecycle(FAILED, str(exc)))
        sink.put(Lifecycle(STOPPED))
        return

    decoder = CobsCborV1(expect_crc=spec.expect_crc)
    sink = _QueueSink(queue, pending)
    policy = BoundaryDropPolicy(spec.max_pending)
    run_source_loop(
        source=source,
        decoder=decoder,
        sink=sink,
        drop_policy=policy,
        should_run=lambda: not stop.is_set(),
        batch_max=spec.batch_max,
    )


def _demo_child(spec: ChildSpec, queue: Any, pending: Any, stop: Any) -> None:
    """A hardware-free spawn target: a demo entry point for the process host.

    Emits a small fixed number of synthetic :class:`FrameBatch` envelopes
    bracketed by lifecycle events, so a real spawned child can be exercised
    without a serial port or BLE radio.
    """
    from data_forwarder_host.protocol.base import DecodedMessage

    sink = _QueueSink(queue, pending)
    sink.put(Lifecycle(OPENED))
    for i in range(3):
        if stop.is_set():
            break
        msg = DecodedMessage(
            kind="sensor_data", t_host_ms=i, t_host_utc="z", t_device_ms=i,
            seq=i, label=None, channels=(float(i),), raw={},
        )
        sink.put(FrameBatch((msg,)))
    sink.put(Lifecycle(STOPPED))
