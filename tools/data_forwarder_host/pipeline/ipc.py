# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Picklable envelopes that cross the source-process boundary.

Everything the acquisition child sends to the GUI process travels as one of
these small, immutable, **picklable** dataclasses. No Qt object ever crosses the
boundary, and the ``spawn`` start method (used so behaviour matches Windows and
macOS) requires every payload to pickle cleanly — both invariants must hold for
every payload. Consumers dispatch on the concrete envelope type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_forwarder_host.protocol.base import (
    DecodedMessage,
    DecodeError,
    DecodeStats,
)

#: Lifecycle event names carried by :class:`Lifecycle`.
OPENED = "opened"
FAILED = "failed"
STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class ChildSpec:
    """Picklable description the GUI sends to the child to build its source.

    Carries only plain, picklable data (no Qt, no open handles) so it survives
    the ``spawn`` boundary. ``max_pending`` is the outbound backlog (in batches)
    at or above which the child sheds ``sensor_data`` frames.
    """

    source_kind: str
    source_params: dict[str, Any] = field(default_factory=dict)
    expect_crc: bool = True
    batch_max: int = 256
    max_pending: int = 64



@dataclass(frozen=True, slots=True)
class FrameBatch:
    """A batch of successfully decoded messages.

    Frames are shipped in batches rather than one-per-message so the GUI thread
    pays a single cross-thread hand-off per batch instead of per frame — the key
    to keeping the GUI responsive under a flood.
    """

    messages: tuple[DecodedMessage, ...] = ()


@dataclass(frozen=True, slots=True)
class DecodeErrors:
    """Decode errors drained from the child's decoder since the last report."""

    errors: tuple[DecodeError, ...] = ()


@dataclass(frozen=True, slots=True)
class StatsUpdate:
    """A snapshot of the child decoder's running :class:`DecodeStats`."""

    stats: DecodeStats = field(default_factory=DecodeStats)


@dataclass(frozen=True, slots=True)
class RawByteCount:
    """Number of raw transport bytes the child received (for bandwidth meters).

    Only the *count* crosses the boundary, never the bytes themselves — shipping
    the raw payload would defeat the point of offloading the GUI process.
    """

    n_bytes: int = 0


@dataclass(frozen=True, slots=True)
class Overflow:
    """Rising-edge marker: the child began dropping ``sensor_data`` frames."""

    dropped_total: int = 0


@dataclass(frozen=True, slots=True)
class Lifecycle:
    """A child lifecycle transition (:data:`OPENED`/:data:`FAILED`/:data:`STOPPED`)."""

    event: str
    detail: str = ""


#: Union of every envelope type that may appear on the boundary queue.
Envelope = (
    FrameBatch | DecodeErrors | StatsUpdate | RawByteCount | Overflow | Lifecycle
)
