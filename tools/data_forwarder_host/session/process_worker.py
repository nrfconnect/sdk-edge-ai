# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Qt worker that drains the out-of-process source host.

:class:`ProcessIoWorker` runs on a ``QThread`` and is the GUI-process counterpart
of the in-process :class:`IoWorker`. Each tick it drains a batch of picklable
envelopes from a :class:`SourceProcessHost` and re-emits them as Qt signals. The
crucial difference from :class:`IoWorker` is that decoded frames are re-emitted
**as a batch** (one :pyattr:`messages` emission per drained group) rather than
one signal per frame — so a flood in the child cannot grow the GUI thread's event
queue without bound and lock up the UI.

The host is injectable so this worker can run without spawning a real
process.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from data_forwarder_host.pipeline.ipc import (
    FAILED,
    STOPPED,
    DecodeErrors,
    FrameBatch,
    Lifecycle,
    Overflow,
    RawByteCount,
    StatsUpdate,
)
from data_forwarder_host.pipeline.process_host import SourceProcessHost

log = logging.getLogger(__name__)


class ProcessIoWorker(QObject):
    """Polls a :class:`SourceProcessHost` and re-emits batched Qt signals."""

    messages = Signal(list)          # list[DecodedMessage] (one batch)
    decode_errors = Signal(list)     # list[DecodeError]
    stats = Signal(object)           # DecodeStats
    raw_byte_count = Signal(int)     # raw transport bytes received by the child
    overflow = Signal()              # child began dropping frames
    failed = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        host: SourceProcessHost,
        *,
        poll_timeout: float = 0.05,
        inbox: Any | None = None,
    ) -> None:
        super().__init__()
        self._host = host
        self._poll_timeout = poll_timeout
        self._running = True
        # When set, decoded frames are appended **directly** into this
        # thread-safe FrameInbox from the worker thread instead of being
        # re-emitted as a Qt signal per poll group. This removes *all*
        # high-frequency cross-thread Qt traffic from the hot path, so a flood
        # in the child can never starve the GUI event loop.
        self._inbox = inbox

    @Slot()
    def run(self) -> None:
        try:
            self._host.start()
        except Exception as exc:
            log.exception("failed to start source process")
            self.failed.emit(str(exc))
            self.stopped.emit()
            return

        try:
            while self._running:
                envelopes = self._host.poll(self._poll_timeout)
                if not envelopes:
                    continue
                if self._dispatch(envelopes):
                    break  # child reported STOPPED
        except Exception as exc:
            log.exception("ProcessIoWorker crashed")
            self.failed.emit(str(exc))
        finally:
            self._host.request_stop()
            self._host.join(2.0)
            self.stopped.emit()

    def _dispatch(self, envelopes: list) -> bool:
        """Route a drained group of envelopes; return ``True`` once STOPPED seen."""
        batch: list = []
        saw_stop = False
        for env in envelopes:
            if isinstance(env, FrameBatch):
                batch.extend(env.messages)
            elif isinstance(env, DecodeErrors):
                if env.errors:
                    self.decode_errors.emit(list(env.errors))
            elif isinstance(env, StatsUpdate):
                self.stats.emit(env.stats)
            elif isinstance(env, RawByteCount):
                self.raw_byte_count.emit(env.n_bytes)
            elif isinstance(env, Overflow):
                self.overflow.emit()
            elif isinstance(env, Lifecycle):
                if env.event == FAILED:
                    self.failed.emit(env.detail)
                elif env.event == STOPPED:
                    saw_stop = True
        if batch:
            if self._inbox is not None:
                # Direct, lock-guarded hand-off to the GUI's pull buffer — no Qt
                # signal, so the GUI event loop is never flooded under load.
                self._inbox.append(batch)
            else:
                self.messages.emit(batch)
        return saw_stop

    @Slot()
    def request_stop(self) -> None:
        self._running = False
        self._host.request_stop()
