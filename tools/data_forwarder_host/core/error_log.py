# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Per-session error journal.

The single, structured journal of per-session abnormalities. Decoder, recorder
and source wrapper all funnel events here, and the GUI ``ErrorPanel`` binds to
this object.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from data_forwarder_host.session.states import SessionState
from data_forwarder_host.utils.timeutil import utc_iso8601_now


class ErrorCategory(Enum):
    COBS = auto()
    CRC = auto()
    MALFORMED = auto()
    CBOR = auto()
    CHANNEL_MISMATCH = auto()
    SESSION_INFO_INVALID = auto()
    SESSION_INFO_MISMATCH = auto()
    SENSOR_DATA_INVALID = auto()
    RECORDER_OVERFLOW = auto()
    PRODUCER_DROP = auto()
    PRODUCER_DROP_TOTAL = auto()
    TRANSPORT = auto()


@dataclass(frozen=True)
class ErrorEvent:
    t_host_utc: str
    category: ErrorCategory
    session_state: SessionState
    detail: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ErrorSummary:
    total_messages: int
    counts: dict[ErrorCategory, int]
    percentages: dict[ErrorCategory, float]
    incomplete: bool


_MAX_EVENTS = 10_000
_SUMMARY_RATE_MS = 1000


class ErrorLog(QObject):
    """Per-session event journal with throttled (1 Hz) summary signal."""

    event_added = Signal(object)        # ErrorEvent
    summary_changed = Signal(object)    # ErrorSummary
    cleared = Signal()                  # emitted when all events are purged

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.RLock()
        self._events: deque[ErrorEvent] = deque(maxlen=_MAX_EVENTS)
        self._counts: Counter[ErrorCategory] = Counter()
        self._state_provider: Callable[[], SessionState] | None = None
        self._total_messages: int = 0
        self._dirty: bool = False

        # Throttle summary emissions to ~1 Hz.
        self._timer = QTimer(self)
        self._timer.setInterval(_SUMMARY_RATE_MS)
        self._timer.timeout.connect(self._maybe_emit_summary)
        self._timer.start()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def bind_state_provider(self, provider: Callable[[], SessionState]) -> None:
        """Provide a callable returning the current ``SessionState``."""
        self._state_provider = provider

    def note_message(self) -> None:
        """Count one successfully delivered application message.

        Used as the denominator for ``percentages`` in :class:`ErrorSummary`.
        """
        with self._lock:
            self._total_messages += 1
            self._dirty = True

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, category: ErrorCategory, detail: str, **context: Any) -> None:
        state = self._state_provider() if self._state_provider else SessionState.CONFIGURED
        evt = ErrorEvent(
            t_host_utc=utc_iso8601_now(),
            category=category,
            session_state=state,
            detail=detail,
            context=dict(context),
        )
        with self._lock:
            self._events.append(evt)
            self._counts[category] += 1
            self._dirty = True
        self.event_added.emit(evt)

    def add_bulk(
        self, category: ErrorCategory, count: int, detail: str, **context: Any
    ) -> None:
        """Record *count* occurrences of *category* as a single coalesced event.

        A transport flood can confirm thousands of losses in one go; emitting an
        individual :class:`ErrorEvent` per occurrence would itself stall the GUI
        (one signal emission and one widget append each). This appends a single
        summary event, advances the category count by *count* in one step, and
        emits :attr:`event_added` exactly once — keeping the running totals
        accurate while the per-event cost stays O(1) regardless of *count*.
        """
        count = int(count)
        if count <= 0:
            return
        state = self._state_provider() if self._state_provider else SessionState.CONFIGURED
        evt = ErrorEvent(
            t_host_utc=utc_iso8601_now(),
            category=category,
            session_state=state,
            detail=detail,
            context={**context, "count": count},
        )
        with self._lock:
            self._events.append(evt)
            self._counts[category] += count
            self._dirty = True
        self.event_added.emit(evt)

    def set_count(self, category: ErrorCategory, count: int) -> None:
        """Force the running count for *category* to an absolute value.

        Used for cumulative producer-side metrics such as the ``dr`` drop
        count, where the device reports a running total rather than discrete
        events. No :class:`ErrorEvent` is appended.
        """
        count = max(0, int(count))
        with self._lock:
            if self._counts.get(category, 0) == count:
                return
            self._counts[category] = count
            self._dirty = True

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._counts.clear()
            self._total_messages = 0
            # Mark clean so the 1-Hz timer doesn't re-emit a stale dirty flush.
            self._dirty = False
        # Notify GUI immediately — don't wait for the next timer tick.
        self.cleared.emit()
        self.summary_changed.emit(self.summary())

    # ------------------------------------------------------------------
    # Read-only
    # ------------------------------------------------------------------

    def events(self) -> Sequence[ErrorEvent]:
        with self._lock:
            return list(self._events)

    def summary(self) -> ErrorSummary:
        with self._lock:
            total = self._total_messages
            denom = total + self._counts.get(ErrorCategory.RECORDER_OVERFLOW, 0)
            counts = {c: self._counts.get(c, 0) for c in ErrorCategory}
            percentages = {
                c: (100.0 * counts[c] / denom) if denom else 0.0 for c in ErrorCategory
            }
            incomplete = counts[ErrorCategory.RECORDER_OVERFLOW] > 0
            return ErrorSummary(
                total_messages=total,
                counts=counts,
                percentages=percentages,
                incomplete=incomplete,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_emit_summary(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False
        self.summary_changed.emit(self.summary())
