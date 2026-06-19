# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Shared protocol data contracts (single COBS/CBOR v1 protocol).

There is exactly one protocol, so these are plain data types — no decoder
ABC, no registry, no ``(name, version)`` selection layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


@dataclass(frozen=True, slots=True)
class DecodedMessage:
    """One successfully decoded application-level message.

    Parameters
    ----------
    kind
        One of ``"sensor_data"``, ``"session_info"`` or ``"unknown"``.
    t_host_ms
        Host monotonic milliseconds at decode time.
    t_host_utc
        ISO-8601 UTC timestamp at decode time, microsecond precision.
    t_device_ms
        Optional ``ts`` field from the device, in device milliseconds.
    seq
        Optional ``seq`` field from the device.
    label
        Optional ``lbl`` field from the device.
    channels
        Optional ``val`` field from the device (per-channel sample tuple).
    raw
        Full decoded Python object (used for session-info and debugging).
    """

    kind: str
    t_host_ms: int
    t_host_utc: str
    t_device_ms: int | None
    seq: int | None
    label: str | None
    channels: tuple[float, ...] | None
    raw: dict[str, Any]


@dataclass
class DecodeStats:
    """Running counters maintained by a ``ProtocolDecoder``.

    Alongside the frame *counts* the decoder also accumulates the real number
    of bytes seen per reception category, so the bandwidth details
    view can report how much data — not just how many frames — arrived in each
    category. For decoded frames the byte size is the COBS-decoded frame
    length; for COBS failures it is the length of the raw on-wire chunk that
    could not be decoded.
    """

    frames_ok: int = 0
    cobs_errors: int = 0
    crc_errors: int = 0
    malformed: int = 0
    cbor_errors: int = 0

    bytes_ok: int = 0
    bytes_cobs: int = 0
    bytes_crc: int = 0
    bytes_malformed: int = 0
    bytes_cbor: int = 0

    @property
    def frames_bad(self) -> int:
        return self.cobs_errors + self.crc_errors + self.malformed + self.cbor_errors

    @property
    def bytes_bad(self) -> int:
        return self.bytes_cobs + self.bytes_crc + self.bytes_malformed + self.bytes_cbor


class DecodeErrorKind(Enum):
    COBS = auto()
    CRC = auto()
    MALFORMED = auto()
    CBOR = auto()


@dataclass(frozen=True, slots=True)
class DecodeError:
    """A single decode failure surfaced to the per-session ``ErrorLog``."""

    kind: DecodeErrorKind
    detail: str
    context: dict[str, Any] = field(default_factory=dict)
