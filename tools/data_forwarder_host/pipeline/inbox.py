# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Consumer-paced, bounded frame inbox at the GUI boundary.

The inbox decouples frame *production* (an I/O worker thread draining the
out-of-process queue) from frame *consumption* (the GUI thread, when it is
ready). Producers :meth:`append` decoded frames; the consumer :meth:`drain`s up
to a budget it chooses. This inverts the previous push-flood — the GUI is the
client that pulls, the inbox is the server that buffers and, when the GUI cannot
keep up, drops the *oldest* ``sensor_data`` frames and counts them.

Invariants:

* **Control/metadata frames are never dropped** — they live in a separate
  priority lane and are always returned first by :meth:`drain`.
* The ``sensor_data`` lane is **bounded**: appending beyond ``capacity`` evicts
  the oldest sensor frame and increments :attr:`dropped`.
* The structure holds no Qt and performs no I/O, so it is fully isolated
  and safe to call from any thread (guarded by a single lock).
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Iterable

from data_forwarder_host.protocol.base import DecodedMessage

#: Default maximum number of buffered ``sensor_data`` frames before the inbox
#: starts dropping the oldest. A few thousand frames bounds memory while
#: absorbing momentary bursts between GUI drain ticks.
DEFAULT_CAPACITY = 4096

#: Default number of ``sensor_data`` frames the GUI processes per drain tick.
#: At the ~50 Hz drain cadence this is a ~20 000 frames/s lossless ceiling;
#: above that the inbox sheds the surplus (counted) so the GUI stays responsive.
DEFAULT_BUDGET = 400


def _default_is_control(msg: DecodedMessage) -> bool:
    """Treat everything that is not ``sensor_data`` as a control frame."""

    return getattr(msg, "kind", "") != "sensor_data"


class FrameInbox:
    """Thread-safe bounded buffer pulled by the GUI at its own cadence.

    Parameters
    ----------
    capacity
        Maximum number of buffered ``sensor_data`` frames; older ones are
        dropped (and counted) on overflow.
    is_control
        Predicate selecting the never-dropped priority lane. Defaults to
        "anything that is not ``sensor_data``".
    """

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        *,
        is_control: Callable[[DecodedMessage], bool] | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = int(capacity)
        self._is_control = is_control or _default_is_control
        self._sensor: deque[DecodedMessage] = deque()
        self._control: deque[DecodedMessage] = deque()
        self._lock = threading.Lock()
        self._dropped = 0

    def append(self, messages: Iterable[DecodedMessage]) -> None:
        """Add frames produced by the worker thread.

        Control frames go to the unbounded priority lane; ``sensor_data`` frames
        go to the bounded lane, evicting and counting the oldest on overflow.
        """

        with self._lock:
            for msg in messages:
                if self._is_control(msg):
                    self._control.append(msg)
                    continue
                self._sensor.append(msg)
                if len(self._sensor) > self._capacity:
                    self._sensor.popleft()
                    self._dropped += 1

    def drain(self, budget: int) -> list[DecodedMessage]:
        """Return up to ``budget`` sensor frames plus *all* pending control frames.

        Called by the GUI when it is ready. Control frames are returned first
        (FIFO), then sensor frames (FIFO) up to ``budget``; any remaining sensor
        frames stay buffered for the next tick.
        """

        if budget < 0:
            raise ValueError("budget must be >= 0")
        with self._lock:
            out: list[DecodedMessage] = []
            while self._control:
                out.append(self._control.popleft())
            taken = 0
            while self._sensor and taken < budget:
                out.append(self._sensor.popleft())
                taken += 1
            return out

    @property
    def dropped(self) -> int:
        """Total number of ``sensor_data`` frames dropped on overflow."""

        with self._lock:
            return self._dropped

    @property
    def capacity(self) -> int:
        """Maximum buffered ``sensor_data`` frames before dropping starts."""

        return self._capacity

    def pending(self) -> int:
        """Number of frames currently buffered (control + sensor)."""

        with self._lock:
            return len(self._control) + len(self._sensor)

    def pending_sensor(self) -> int:
        """Number of buffered ``sensor_data`` frames awaiting drain."""

        with self._lock:
            return len(self._sensor)

    def clear(self) -> None:
        """Discard all buffered frames and reset the drop counter."""

        with self._lock:
            self._sensor.clear()
            self._control.clear()
            self._dropped = 0
