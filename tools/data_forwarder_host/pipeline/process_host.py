# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""GUI-side host for the acquisition child process.

:class:`SourceProcessHost` owns the ``spawn`` multiprocessing context, the
boundary queue, the shared backlog counter and the stop event, and launches the
child running :func:`source_child_main`. It is **Qt-free** — the session-layer
QThread worker calls :meth:`poll` to drain envelopes and re-emit Qt signals — so
this stays in the ``pipeline`` layer. The context and child target are injectable
so the host's drain/backlog/stop logic can run without a real OS
process, serial port or BLE radio.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from queue import Empty
from typing import Any, Callable

from data_forwarder_host.pipeline.child_main import source_child_main
from data_forwarder_host.pipeline.ipc import ChildSpec, FrameBatch

log = logging.getLogger(__name__)

#: Default blocking timeout (seconds) for a single :meth:`SourceProcessHost.poll`.
DEFAULT_POLL_TIMEOUT = 0.05

#: Setting this environment variable forces single-process (in-GUI) acquisition.
SINGLE_PROCESS_ENV = "DFH_SINGLE_PROCESS"


def process_mode_enabled() -> bool:
    """True unless out-of-process acquisition is disabled via the environment.

    Out-of-process acquisition is the default so the GUI stays
    responsive regardless of backend load; set ``DFH_SINGLE_PROCESS``
    to fall back to the legacy in-GUI :class:`IoWorker` path.
    """
    return os.environ.get(SINGLE_PROCESS_ENV, "").strip().lower() in ("", "0", "false", "no")


class SourceProcessHost:
    """Launches and drains the out-of-process acquisition child.

    Parameters
    ----------
    spec
        Picklable description of the source/decoder the child must build.
    ctx
        A multiprocessing context. Defaults to the ``spawn`` context so behaviour
        matches Windows/macOS even on Linux. Injectable by callers.
    target
        The child entry point. Defaults to :func:`source_child_main`. Injectable
        by callers that need a custom entry point.
    """

    def __init__(
        self,
        spec: ChildSpec,
        *,
        ctx: Any | None = None,
        target: Callable[..., None] | None = None,
    ) -> None:
        self._spec = spec
        self._ctx = ctx or multiprocessing.get_context("spawn")
        self._target = target or source_child_main
        self._queue = self._ctx.Queue()
        self._pending = self._ctx.Value("i", 0)
        self._stop = self._ctx.Event()
        self._proc: Any | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Spawn the child process."""
        if self._proc is not None:
            return
        self._proc = self._ctx.Process(
            target=self._target,
            args=(self._spec, self._queue, self._pending, self._stop),
            daemon=True,
        )
        self._proc.start()

    def request_stop(self) -> None:
        """Ask the child to finish its loop (idempotent)."""
        try:
            self._stop.set()
        except Exception:
            log.exception("failed to set child stop event")

    def join(self, timeout: float = 2.0) -> None:
        """Wait up to *timeout* seconds for the child to exit."""
        if self._proc is not None:
            self._proc.join(timeout)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    # -- draining ----------------------------------------------------------
    def poll(self, timeout: float = DEFAULT_POLL_TIMEOUT) -> list[Any]:
        """Return all envelopes currently available, blocking up to *timeout*.

        Blocks for at most *timeout* waiting for the first envelope, then drains
        everything else already queued without blocking. Consuming a
        :class:`FrameBatch` decrements the shared backlog counter so the child's
        drop policy sees the GUI catching up.
        """
        out: list[Any] = []
        try:
            out.append(self._queue.get(timeout=timeout))
        except Empty:
            return out
        while True:
            try:
                out.append(self._queue.get_nowait())
            except Empty:
                break
        consumed = sum(1 for it in out if isinstance(it, FrameBatch))
        if consumed:
            with self._pending.get_lock():
                self._pending.value = max(0, self._pending.value - consumed)
        return out

    @property
    def backlog(self) -> int:
        """Current outbound backlog in batches (frames buffered for the GUI)."""
        return int(self._pending.value)
