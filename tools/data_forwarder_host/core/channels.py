# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""NumPy ring buffer for one channel of ``(t_host_ms, value)`` samples."""

from __future__ import annotations

import math

import numpy as np


class ChannelBuffer:
    """Fixed-capacity FIFO ring buffer storing ``(t_host_ms, value)`` pairs.

    Designed for the live-plot use case: the GUI thread reads recent samples
    via :meth:`snapshot` while the I/O thread appends via :meth:`append`.
    Appends are O(1); snapshots copy out the contiguous in-order view.

    The buffer is **not** internally synchronised — the owning ``DataModel``
    runs on the GUI thread and is the sole writer.
    """

    DEFAULT_SAMPLE_RATE_HINT_HZ = 1000.0
    SAFETY = 2.0

    def __init__(self, *, plot_window_seconds: float, sample_rate_hint_hz: float | None = None) -> None:
        if plot_window_seconds <= 0:
            raise ValueError("plot_window_seconds must be > 0")
        rate = sample_rate_hint_hz or self.DEFAULT_SAMPLE_RATE_HINT_HZ
        cap = max(1024, int(math.ceil(plot_window_seconds * rate * self.SAFETY)))
        self._capacity = cap
        self._ts = np.zeros(cap, dtype=np.int64)
        self._val = np.zeros(cap, dtype=np.float64)
        self._head = 0    # next write index
        self._size = 0    # current number of valid samples

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    def resize(self, *, plot_window_seconds: float, sample_rate_hint_hz: float | None = None) -> None:
        """Reallocate with a new capacity, preserving the most recent samples."""
        ts, val = self.snapshot()
        new = ChannelBuffer(
            plot_window_seconds=plot_window_seconds,
            sample_rate_hint_hz=sample_rate_hint_hz,
        )
        # Replay the most recent samples that fit.
        n_keep = min(new._capacity, ts.size)
        if n_keep:
            new._ts[:n_keep] = ts[-n_keep:]
            new._val[:n_keep] = val[-n_keep:]
            new._head = n_keep % new._capacity
            new._size = n_keep
        self._capacity = new._capacity
        self._ts = new._ts
        self._val = new._val
        self._head = new._head
        self._size = new._size

    def append(self, t_host_ms: int, value: float) -> None:
        self._ts[self._head] = t_host_ms
        self._val[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return contiguous ``(ts, val)`` arrays in chronological order."""
        if self._size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        if self._size < self._capacity:
            return self._ts[: self._size].copy(), self._val[: self._size].copy()
        # Full ring: unwrap starting from oldest (== head)
        ts = np.concatenate((self._ts[self._head :], self._ts[: self._head]))
        val = np.concatenate((self._val[self._head :], self._val[: self._head]))
        return ts, val

    def latest_window(self, window_ms: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the most recent samples spanning ``window_ms`` milliseconds."""
        ts, val = self.snapshot()
        if ts.size == 0:
            return ts, val
        cutoff = int(ts[-1]) - int(window_ms)
        mask = ts >= cutoff
        return ts[mask], val[mask]

    def clear(self) -> None:
        self._head = 0
        self._size = 0

    def oldest_ts(self) -> int | None:
        """Timestamp of the oldest retained sample, or ``None`` if empty."""
        if self._size == 0:
            return None
        if self._size < self._capacity:
            return int(self._ts[0])
        return int(self._ts[self._head])

    def retain_window(self, window_ms: int) -> None:
        """Drop samples older than ``window_ms`` before the newest sample.

        Used to cap idle (not-recording) retention to a bounded time span
        without resizing the ring. Cheap when nothing needs dropping
        (a single oldest-timestamp check); otherwise rebuilds from the trimmed
        chronological view.
        """
        if window_ms <= 0 or self._size == 0:
            return
        oldest = self.oldest_ts()
        ts, val = self._ts, self._val
        newest = int(ts[(self._head - 1) % self._capacity])
        cutoff = newest - int(window_ms)
        if oldest is None or oldest >= cutoff:
            return
        keep_ts, keep_val = self.snapshot()
        mask = keep_ts >= cutoff
        keep_ts, keep_val = keep_ts[mask], keep_val[mask]
        self.clear()
        n = keep_ts.size
        if n:
            self._ts[:n] = keep_ts
            self._val[:n] = keep_val
            self._head = n % self._capacity
            self._size = n
