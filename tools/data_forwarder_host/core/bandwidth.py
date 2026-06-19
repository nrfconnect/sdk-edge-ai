# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Real-time bandwidth measurement over a trailing time window.

:class:`BandwidthMeter` is a pure, GUI-free accumulator: callers feed it byte
and message events as they arrive and read back throughput rates computed over a
configurable trailing window. ``channels/s`` is derived from ``messages/s`` and
the session's fixed per-message channel count.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BandwidthSample:
    """A snapshot of the three throughput rates."""

    bytes_per_second: float
    messages_per_second: float
    channels_per_second: float


class BandwidthMeter:
    """Sliding-window throughput meter.

    Events are timestamped via *time_fn* (monotonic seconds by default) and
    rates are the count within the trailing *window_seconds* divided by the
    **effective** window, which is the smaller of *window_seconds* and the time
    elapsed since measurement started. Early on — e.g. 3 s into a 10 s window —
    the rate is averaged over the 3 s actually observed, not the full 10 s, so
    a partially-filled window is not under-reported.
    """

    def __init__(
        self,
        window_seconds: float = 1.0,
        *,
        time_fn=time.monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window = float(window_seconds)
        self._time_fn = time_fn
        # Each deque holds (timestamp, value); messages use value == 1.
        self._byte_events: deque[tuple[float, int]] = deque()
        self._msg_events: deque[float] = deque()
        # Time of the first event since the last reset; used to cap the divisor
        # to the data actually available.
        self._start_time: float | None = None

    @property
    def window_seconds(self) -> float:
        return self._window

    def set_window_seconds(self, window_seconds: float) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window = float(window_seconds)

    def add_bytes(self, n: int, *, now: float | None = None) -> None:
        """Record *n* received bytes at the current (or supplied) time."""
        if n <= 0:
            return
        t = self._time_fn() if now is None else now
        if self._start_time is None:
            self._start_time = t
        self._byte_events.append((t, int(n)))

    def add_message(self, *, now: float | None = None) -> None:
        """Record one decoded message at the current (or supplied) time."""
        t = self._time_fn() if now is None else now
        if self._start_time is None:
            self._start_time = t
        self._msg_events.append(t)

    def reset(self) -> None:
        self._byte_events.clear()
        self._msg_events.clear()
        self._start_time = None

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._byte_events and self._byte_events[0][0] < cutoff:
            self._byte_events.popleft()
        while self._msg_events and self._msg_events[0] < cutoff:
            self._msg_events.popleft()

    def _effective_window(self, now: float) -> float:
        """Return the divisor: ``min(window, available)``.

        ``available`` is the time elapsed since the first recorded event. Until
        the window has filled, rates are averaged over the data actually seen
        rather than the full (mostly empty) window. Falls back to the full
        window when no positive elapsed time exists yet.
        """
        if self._start_time is None:
            return self._window
        available = now - self._start_time
        if available <= 0:
            return self._window
        return min(self._window, available)

    def bytes_per_second(self, *, now: float | None = None) -> float:
        t = self._time_fn() if now is None else now
        self._evict(t)
        total = sum(v for _, v in self._byte_events)
        return total / self._effective_window(t)

    def messages_per_second(self, *, now: float | None = None) -> float:
        t = self._time_fn() if now is None else now
        self._evict(t)
        return len(self._msg_events) / self._effective_window(t)

    def channels_per_second(
        self, channel_count: int, *, now: float | None = None
    ) -> float:
        return self.messages_per_second(now=now) * max(0, int(channel_count))

    def sample(
        self, channel_count: int, *, now: float | None = None
    ) -> BandwidthSample:
        t = self._time_fn() if now is None else now
        return BandwidthSample(
            bytes_per_second=self.bytes_per_second(now=t),
            messages_per_second=self.messages_per_second(now=t),
            channels_per_second=self.channels_per_second(channel_count, now=t),
        )
