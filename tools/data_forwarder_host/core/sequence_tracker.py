# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Transport message-loss detector based on ``sensor_data`` sequence numbers.

Each ``sensor_data`` message carries a monotonically increasing sequence number
(wire path ``d/seq``). A perfect link delivers consecutive sequence numbers; a
*gap* means one or more messages did not arrive — yet. Because the transport may
reorder or briefly delay frames, a gap is **not** reported immediately: every
missing sequence number is held for a configurable grace period — the **loss
confirmation window** — and only counted as a transport loss if it has not
arrived by the time that window elapses. If the missing number turns up within
the window, its pending record is cleared and no loss is reported.

The tracker is pure Python (no Qt) and its clock is injectable, so the policy is
fully deterministic.
"""

from __future__ import annotations

import time
from collections.abc import Callable

# Guard against pathological re-baselines (counter reset, reconnect, device
# restart). A jump larger than this in either direction is treated as a stream
# discontinuity rather than millions of "missing" sequence numbers.
_MAX_GAP = 100_000

# Hard cap on the number of individually tracked pending (missing) sequence
# numbers. This bounds both memory and the per-confirmation work no matter how
# fast or how lossy the link is: a sustained flood with massive drops can no
# longer grow ``_pending`` without limit (the historical cause of an unbounded
# memory climb → OOM) nor make any single ``observe()`` iterate an arbitrarily
# large gap (the historical cause of multi-second GUI-thread stalls). Once the
# cap is reached, further missing numbers in the current burst are not tracked
# individually; under that much loss they are reported in aggregate instead.
_MAX_PENDING = 8_192


class SequenceGapTracker:
    """Detect missing ``sensor_data`` sequence numbers with a confirmation delay.

    :param window_seconds: the loss confirmation window — how long a missing
        sequence number is awaited before it is confirmed lost.
    :param on_loss: called with the sorted list of sequence numbers confirmed
        lost, once per sweep that confirms at least one loss.
    :param time_fn: monotonic clock (seconds); injectable for determinism.
    """

    def __init__(
        self,
        *,
        window_seconds: float,
        on_loss: Callable[[list[int]], None],
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window = float(window_seconds)
        self._on_loss = on_loss
        self._time_fn = time_fn
        self._expected: int | None = None
        # Missing sequence number -> monotonic deadline by which it must arrive.
        self._pending: dict[int, float] = {}
        # Smallest deadline currently in ``_pending`` (or None when empty). Used
        # to make the per-message sweep O(1) in the common case: if nothing is
        # yet due, ``sweep()`` returns immediately instead of scanning/sorting
        # the whole pending set on every single arriving frame. It is only ever
        # a *lower bound* on the true minimum (recovery may remove the holder),
        # which is safe — a stale-low value costs at most one extra bounded scan
        # that then self-corrects, and it can never cause a due loss to be
        # missed. Aggregate counts of losses that exceeded the tracking cap.
        self._min_deadline: float | None = None
        self._untracked_losses: int = 0

    @property
    def window_seconds(self) -> float:
        return self._window

    @property
    def pending_count(self) -> int:
        """Number of sequence numbers currently awaiting confirmation."""
        return len(self._pending)

    @property
    def untracked_loss_count(self) -> int:
        """Losses that exceeded the tracking cap and were counted in aggregate.

        Under extreme, sustained loss the individually tracked pending set is
        capped (see ``_MAX_PENDING``); any further missing numbers in such a
        burst are not tracked one-by-one but accumulate here so the magnitude of
        the loss is still observable without unbounded memory or CPU.
        """
        return self._untracked_losses

    def set_window_seconds(self, window_seconds: float) -> None:
        """Change the loss confirmation window (applies to future deadlines)."""
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window = float(window_seconds)

    def reset(self) -> None:
        """Forget all state (e.g. when a new stream starts)."""
        self._expected = None
        self._pending.clear()
        self._min_deadline = None
        self._untracked_losses = 0

    def observe(self, seq: int, *, now: float | None = None) -> int:
        """Record an arrived sequence number; return losses confirmed in this call.

        Recovers any matching pending number (a late/out-of-order arrival), then
        records newly missing numbers for any forward gap, and finally sweeps
        expired pending numbers.

        The amount of work done here is bounded regardless of how large a gap is
        or how many numbers are already pending: a forward gap adds at most
        ``_MAX_PENDING`` tracked entries (any excess is counted in aggregate),
        and the trailing sweep is O(1) until something is actually due.
        """
        t = self._time_fn() if now is None else now
        seq = int(seq)

        # A late/out-of-order arrival of a number we were waiting for: recovered.
        # (We intentionally do not refresh ``_min_deadline`` here; it stays a
        # safe lower bound, see the field comment.)
        self._pending.pop(seq, None)

        if self._expected is None:
            self._expected = seq + 1
            return self.sweep(now=t)

        gap = seq - self._expected
        if gap == 0:
            self._expected = seq + 1
        elif 0 < gap <= _MAX_GAP:
            # Every integer in [expected, seq) is missing — await each, but only
            # up to the tracking cap so a huge gap can neither iterate without
            # bound nor grow ``_pending`` without bound. In a forward gap none of
            # these numbers can already be pending, so no membership check.
            deadline = t + self._window
            room = _MAX_PENDING - len(self._pending)
            track_until = self._expected + room if room > 0 else self._expected
            if track_until > seq:
                track_until = seq
            for missing in range(self._expected, track_until):
                self._pending[missing] = deadline
            if self._min_deadline is None and self._pending:
                self._min_deadline = deadline
            # Numbers we had no room to track are, under this much loss, almost
            # certainly genuine losses; record their magnitude in aggregate.
            self._untracked_losses += seq - track_until
            self._expected = seq + 1
        elif gap > _MAX_GAP or -gap > _MAX_GAP:
            # Implausible jump in either direction: treat as a stream
            # discontinuity (reset/reconnect), re-baseline without confirming
            # spurious losses across the boundary.
            self._pending.clear()
            self._min_deadline = None
            self._expected = seq + 1
        # else: small backwards step (seq < expected) that was not pending —
        # a duplicate or an already-confirmed loss; ignore.

        return self.sweep(now=t)

    def sweep(self, *, now: float | None = None) -> int:
        """Confirm and report any pending numbers whose window has elapsed.

        Returns the number of losses confirmed. Safe to call on a timer so that
        losses are confirmed even while no new messages arrive.

        Fast path: when nothing is pending, or the earliest deadline is still in
        the future, this returns in O(1) without scanning the pending set — so
        calling it on every arriving frame (as ``observe`` does) is cheap. The
        bounded scan/sort runs only at the moments a confirmation is actually
        due, and over at most ``_MAX_PENDING`` entries.
        """
        t = self._time_fn() if now is None else now
        if not self._pending:
            self._min_deadline = None
            return 0
        if self._min_deadline is not None and t < self._min_deadline:
            return 0
        due = sorted(s for s, deadline in self._pending.items() if deadline <= t)
        if not due:
            # ``_min_deadline`` was a stale lower bound; recompute and bail.
            self._min_deadline = min(self._pending.values())
            return 0
        for s in due:
            del self._pending[s]
        self._min_deadline = min(self._pending.values()) if self._pending else None
        self._on_loss(due)
        return len(due)
