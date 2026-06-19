# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Session lifecycle state machine (5 states)."""

from __future__ import annotations

from enum import Enum, auto


class SessionState(Enum):
    CONFIGURED = auto()
    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()


# Allowed transitions. Exactly one component (the session) is the sole writer.
#
# * CONFIGURED → RUNNING   (opening the source is an entry action of RUNNING)
# * RUNNING    → STOPPED   (stopping the source is an entry action of STOPPED)
# * STOPPED    → CONFIGURED / RUNNING (re-run)
# * any state  → ERROR
# * ERROR      → CONFIGURED (reset path)
_ALLOWED: dict[SessionState, frozenset[SessionState]] = {
    SessionState.CONFIGURED: frozenset({SessionState.RUNNING, SessionState.ERROR}),
    SessionState.RUNNING: frozenset({SessionState.STOPPED, SessionState.ERROR}),
    SessionState.STOPPED: frozenset(
        {SessionState.CONFIGURED, SessionState.RUNNING, SessionState.ERROR}
    ),
    SessionState.ERROR: frozenset({SessionState.CONFIGURED}),  # reset() only
}


def can_transition(src: SessionState, dst: SessionState) -> bool:
    """Return ``True`` if ``src -> dst`` is an allowed transition."""
    if src == dst:
        return True
    return dst in _ALLOWED.get(src, frozenset())
