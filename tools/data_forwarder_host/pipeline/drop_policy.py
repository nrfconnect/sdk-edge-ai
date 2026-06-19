# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Boundary drop policy for the source-acquisition child.

When the GUI process falls behind, the outbound queue between the child and the
GUI grows. Rather than let it grow without bound (the original freeze), the child
sheds load at the boundary: once the queue is at capacity it drops ``sensor_data``
frames, but **never** control/metadata frames (e.g. ``session_info``) — that
guarantee is the same one made in-process, now enforced at the process
boundary. The policy is pure and deterministic: it reads the current queue depth
and decides keep/drop, counting drops and reporting the rising edge of each
dropping episode so a single overflow event can be surfaced per episode.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Kind string of the only droppable message class.
SENSOR_DATA = "sensor_data"


@dataclass(frozen=True, slots=True)
class DropDecision:
    """Outcome of evaluating one message against the policy."""

    keep: bool
    #: ``True`` only on the transition *into* a dropping episode (rising edge).
    overflow_edge: bool = False


class BoundaryDropPolicy:
    """Queue-depth-based keep/drop decision that protects control frames.

    Parameters
    ----------
    max_pending
        The outbound queue depth at or above which ``sensor_data`` frames are
        dropped. Control frames are kept regardless of depth.
    """

    def __init__(self, max_pending: int) -> None:
        self._max = max(1, int(max_pending))
        self._dropped = 0
        self._active = False

    def evaluate(self, kind: str, pending: int) -> DropDecision:
        """Decide whether to keep a message of *kind* given *pending* queue depth."""
        # Control/metadata frames are never dropped at the boundary.
        if kind != SENSOR_DATA:
            return DropDecision(keep=True)
        if pending < self._max:
            # Back below the limit ends the current dropping episode.
            self._active = False
            return DropDecision(keep=True)
        # At/over capacity: drop this sensor frame.
        self._dropped += 1
        edge = not self._active
        self._active = True
        return DropDecision(keep=False, overflow_edge=edge)

    @property
    def max_pending(self) -> int:
        return self._max

    @property
    def dropped(self) -> int:
        """Total ``sensor_data`` frames dropped since the policy was created."""
        return self._dropped

    @property
    def is_dropping(self) -> bool:
        """``True`` while inside an active dropping episode."""
        return self._active
