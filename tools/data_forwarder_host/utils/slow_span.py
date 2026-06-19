# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Always-on, near-zero-cost stopwatch for GUI-thread spans.

A *span* is a synchronous block of work that runs on the GUI (main) thread. If
any single span takes longer than a threshold (default 50 ms) it has, by
definition, blocked the Qt event loop for that long — paint and input events
could not run, so the UI was unresponsive for the duration. This helper times
such spans and, only when one is slow, writes a single compact, greppable line
to the ``data_forwarder_host.debug`` logger::

    slow-span name=drain_inbox ms=4211.3 thresh=50 gc=1 gc_ms=4180.4

so the exact span that froze the UI can be found with ``grep slow-span`` in the
metrics log (``~/.local/state/data_forwarder/log/data_forwarder_host.log``).

Garbage-collection attribution
------------------------------
A span timed by ``monotonic()`` includes any stop-the-world CPython garbage
collection that happened to run during it — a long span is often a GC pause the
span merely *witnessed*, not work it performed. To make that visible, a global
``gc.callbacks`` hook accumulates the wall time and count of collections; each
span captures that baseline on entry and, when it logs as slow, appends how many
collections ran during it and their total pause time (``gc=`` / ``gc_ms=``).
When ``gc_ms`` is close to ``ms`` the span did not really do the work — GC did.
The hook is registered once at import; disable it with ``DFH_GC_TRACE=0``.

Note: a collection can be triggered by another thread, but CPython holds the GIL
for the whole collection, so it still froze the GUI thread — attributing it to
the concurrently-open span is the intended signal.

Design goals
------------
* **Zero allocation / logging on the fast path.** When a span finishes inside
  the threshold nothing is logged and no formatting happens — only a
  ``monotonic()`` subtraction, two integer reads and a comparison.
* **Two ergonomic forms.** A context manager (``with slow_span("drain"):``) for
  wrapping inline blocks, and a decorator (``@slow_span_fn("name")`` / bare
  ``@slow_span_fn``) for whole functions.
* **No Qt import.** Usable from any layer in isolation.

The helper does not try to be a profiler: it deliberately reports only the spans
that actually exceeded the budget, which is exactly the signal needed to locate
a freeze.
"""

from __future__ import annotations

import gc
import logging
import os
from functools import wraps
from time import monotonic
from typing import Any, Callable, TypeVar

log = logging.getLogger("data_forwarder_host.debug")

# --- Global garbage-collection accounting -------------------------------
#
# These module-level counters are advanced by a ``gc.callbacks`` hook so any
# span can cheaply read "how much GC happened during me" as the delta of two
# integers, with no per-span ``gc`` calls on the fast path. A collection only
# fires the hook a handful of times per second under churn, and each call does a
# ``monotonic()`` and two additions, so the steady-state cost is negligible.
_gc_collections = 0      # number of completed collections since import
_gc_pause_ms = 0.0       # total wall time (ms) spent inside collections
_gc_phase_t0 = 0.0       # monotonic() captured at the most recent "start" phase

#: GC attribution is on unless explicitly disabled, so diagnostic logs include
#: it without any extra setup. Set ``DFH_GC_TRACE=0`` to skip registering the
#: hook entirely (e.g. to rule it out as a source of overhead).
_GC_TRACE = os.environ.get("DFH_GC_TRACE", "1") not in ("0", "", "false", "False")


def _gc_callback(phase: str, _info: dict) -> None:
    """``gc.callbacks`` hook: accumulate collection count and wall pause time."""
    global _gc_collections, _gc_pause_ms, _gc_phase_t0
    if phase == "start":
        _gc_phase_t0 = monotonic()
    else:  # "stop"
        _gc_pause_ms += (monotonic() - _gc_phase_t0) * 1000.0
        _gc_collections += 1


if _GC_TRACE and _gc_callback not in gc.callbacks:
    gc.callbacks.append(_gc_callback)

#: Default slow-span threshold in milliseconds. A span faster than this is
#: considered to have left the event loop responsive and is never logged. The
#: 50 ms value is ~1 missed frame at 20 Hz — small enough to catch a stutter,
#: large enough that normal per-tick work stays silent. It can be lowered for a
#: diagnostic session via the ``DFH_SLOW_SPAN_MS`` environment variable (read
#: once at import) to attribute cost to finer sub-spans.
DEFAULT_THRESHOLD_MS = float(os.environ.get("DFH_SLOW_SPAN_MS", "50") or "50")

F = TypeVar("F", bound=Callable[..., Any])


class slow_span:  # noqa: N801 — context-manager spelled like a verb on purpose
    """Time a synchronous span; log ``slow-span`` only if it exceeds *threshold_ms*.

    Use as a context manager::

        with slow_span("drain_inbox"):
            ...                      # GUI-thread work

        with slow_span("refresh", extra="charts=6") as span:
            ...
            span.note("decimate")    # optional sub-label appended to the line

    The optional ``extra`` string and any :meth:`note` labels are appended to the
    logged line (still only when the span is slow), so a single span can carry a
    little context about *what* it was doing when it blocked.
    """

    __slots__ = ("name", "_threshold_ms", "_extra", "_notes", "_t0", "elapsed_ms",
                 "_gc_n0", "_gc_ms0")

    def __init__(
        self,
        name: str,
        *,
        threshold_ms: float = DEFAULT_THRESHOLD_MS,
        extra: str = "",
    ) -> None:
        self.name = name
        self._threshold_ms = threshold_ms
        self._extra = extra
        self._notes: list[str] | None = None
        self._t0 = 0.0
        self.elapsed_ms = 0.0
        self._gc_n0 = 0
        self._gc_ms0 = 0.0

    def note(self, label: str) -> None:
        """Attach a sub-label, surfaced on the log line if the span is slow."""
        if self._notes is None:
            self._notes = []
        self._notes.append(label)

    def __enter__(self) -> "slow_span":
        self._gc_n0 = _gc_collections
        self._gc_ms0 = _gc_pause_ms
        self._t0 = monotonic()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.elapsed_ms = (monotonic() - self._t0) * 1000.0
        if self.elapsed_ms < self._threshold_ms:
            return
        parts = [f"slow-span name={self.name}", f"ms={self.elapsed_ms:.1f}",
                 f"thresh={self._threshold_ms:g}"]
        gc_n = _gc_collections - self._gc_n0
        if gc_n:
            parts.append(f"gc={gc_n} gc_ms={_gc_pause_ms - self._gc_ms0:.1f}")
        if self._extra:
            parts.append(self._extra)
        if self._notes:
            parts.append("notes=" + ",".join(self._notes))
        log.warning(" ".join(parts))


def slow_span_fn(
    name: str | Callable[..., Any] | None = None,
    *,
    threshold_ms: float = DEFAULT_THRESHOLD_MS,
) -> Any:
    """Decorator form of :class:`slow_span`.

    Supports both bare and parametrised use::

        @slow_span_fn
        def _tick(self): ...

        @slow_span_fn("data_model.on_message", threshold_ms=20)
        def on_message(self, msg): ...

    The span name defaults to the wrapped function's qualified name.
    """

    def _decorate(func: F, span_name: str) -> F:
        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            gc_n0 = _gc_collections
            gc_ms0 = _gc_pause_ms
            t0 = monotonic()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (monotonic() - t0) * 1000.0
                if elapsed_ms >= threshold_ms:
                    gc_n = _gc_collections - gc_n0
                    gc_part = (
                        f" gc={gc_n} gc_ms={_gc_pause_ms - gc_ms0:.1f}"
                        if gc_n
                        else ""
                    )
                    log.warning(
                        "slow-span name=%s ms=%.1f thresh=%g%s",
                        span_name,
                        elapsed_ms,
                        threshold_ms,
                        gc_part,
                    )

        return _wrapper  # type: ignore[return-value]

    # Bare @slow_span_fn (called with the function directly).
    if callable(name):
        func = name
        return _decorate(func, func.__qualname__)

    # Parametrised @slow_span_fn("name", ...)
    def _outer(func: F) -> F:
        return _decorate(func, name or func.__qualname__)

    return _outer
