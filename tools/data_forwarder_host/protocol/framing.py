# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Framing primitives (COBS framing helpers for the data-forwarder protocol)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from cobs import cobs


# ---------------------------------------------------------------------------
# COBS framing helpers
# ---------------------------------------------------------------------------


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    # Matches Zephyr's crc16_ccitt() — reflected (LSB-first) variant, poly 0x1021.
    # NOT the same as CRC-16/CCITT-FALSE (MSB-first); Zephyr uses the reflected form.
    crc = init
    for b in data:
        e = (crc ^ b) & 0xFF
        f = (e ^ (e << 4)) & 0xFF
        crc = ((crc >> 8) ^ (f << 8) ^ (f << 3) ^ (f >> 4)) & 0xFFFF
    return crc


COBS_ERROR: Any = object()  # sentinel for COBS decode failures
BAD_CRC: Any = object()     # sentinel for CRC mismatches


def cobs_frame_iter(byte_source: Iterable[bytes]) -> Iterator[Any]:
    """Yield COBS-decoded raw frames (LEN + CBOR + optional CRC) or COBS_ERROR."""
    buf = bytearray()
    for chunk in byte_source:
        for b in chunk:
            if b == 0x00:
                if buf:
                    try:
                        yield bytes(cobs.decode(bytes(buf)))
                    except cobs.DecodeError:
                        yield COBS_ERROR  # visible in the bad counter, not silently lost
                    buf.clear()
            else:
                buf.append(b)


def extract_cbor(frame: bytes, expect_crc: bool) -> Any:
    """Return CBOR payload, None (malformed), or BAD_CRC."""
    if len(frame) < 2:
        return None
    length = int.from_bytes(frame[:2], "little")
    end = 2 + length
    if expect_crc:
        if len(frame) < end + 2:
            return None
        rx_crc = int.from_bytes(frame[end:end + 2], "little")
        if rx_crc != crc16_ccitt(frame[2:end]):
            return BAD_CRC
    elif len(frame) < end:
        return None
    return frame[2:end]
