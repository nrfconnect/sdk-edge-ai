# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""COBS + CBOR v1 decoder.

A thin stateful wrapper around the verbatim framing primitives in
``data_forwarder_host.protocol.framing``.
"""

from __future__ import annotations

import traceback
from collections.abc import Iterable
from typing import Any

import cbor2
from cobs import cobs

from data_forwarder_host.protocol.base import (
    DecodedMessage,
    DecodeError,
    DecodeErrorKind,
    DecodeStats,
)
from data_forwarder_host.protocol.framing import (
    BAD_CRC,
    crc16_ccitt,
    extract_cbor,
)
from data_forwarder_host.utils.timeutil import host_monotonic_ms, utc_iso8601_now

# Cap the raw bytestream stored in First Failure Data Capture context so a
# pathological frame cannot bloat the error log.
_FFDC_MAX_RAW = 512


class CobsCborV1:
    """Stateful decoder for the device's COBS + CBOR framing.

    The single, fixed protocol: frames are COBS-delimited (``0x00``), carry a
    2-byte little-endian length prefix and an optional CRC-16/CCITT, and wrap a
    CBOR map. See :mod:`data_forwarder_host.protocol.framing` (verbatim port).
    """

    def __init__(self, *, expect_crc: bool = True) -> None:
        self._expect_crc = expect_crc
        self._stats = DecodeStats()
        self._errors: list[DecodeError] = []
        # Persistent raw-byte accumulator between 0x00 COBS delimiters.
        self._buf = bytearray()

    # ------------------------------------------------------------------
    # ProtocolDecoder API
    # ------------------------------------------------------------------

    def feed(self, chunk: bytes) -> Iterable[DecodedMessage]:
        """Process one raw chunk; yield fully decoded messages."""
        if not chunk:
            return
        for b in chunk:
            if b == 0x00:
                if self._buf:
                    raw = bytes(self._buf)
                    try:
                        frame = bytes(cobs.decode(raw))
                    except cobs.DecodeError:
                        self._stats.cobs_errors += 1
                        self._stats.bytes_cobs += len(raw)
                        self._errors.append(
                            DecodeError(
                                DecodeErrorKind.COBS,
                                "COBS decode failure",
                                context={
                                    "raw": raw[:_FFDC_MAX_RAW],
                                    "raw_len": len(raw),
                                    "expected": "valid COBS-encoded frame",
                                    "actual": (
                                        "chunk could not be COBS-decoded "
                                        f"({len(raw)} bytes between delimiters)"
                                    ),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                        )
                        self._buf.clear()
                        continue
                    self._buf.clear()
                    yield from self._process_frame(frame)
            else:
                self._buf.append(b)

    def _process_frame(self, frame: bytes) -> Iterable[DecodedMessage]:
        payload = extract_cbor(frame, expect_crc=self._expect_crc)
        if payload is None:
            self._stats.malformed += 1
            self._stats.bytes_malformed += len(frame)
            self._errors.append(
                DecodeError(
                    DecodeErrorKind.MALFORMED,
                    f"frame too short (got {len(frame)} bytes)",
                    context={
                        "raw": frame[:_FFDC_MAX_RAW],
                        "raw_len": len(frame),
                        "expected": (
                            "at least 2-byte length prefix + payload"
                            + (" + 2-byte CRC" if self._expect_crc else "")
                        ),
                        "actual": f"{len(frame)} byte(s) total",
                    },
                )
            )
            return
        if payload is BAD_CRC:
            self._stats.crc_errors += 1
            self._stats.bytes_crc += len(frame)
            length = int.from_bytes(frame[:2], "little")
            end = 2 + length
            rx_crc = int.from_bytes(frame[end:end + 2], "little")
            calc_crc = crc16_ccitt(frame[2:end])
            self._errors.append(
                DecodeError(
                    DecodeErrorKind.CRC,
                    "CRC-16 mismatch",
                    context={
                        "raw": frame[:_FFDC_MAX_RAW],
                        "raw_len": len(frame),
                        "expected": f"CRC-16/CCITT 0x{calc_crc:04X} (computed over payload)",
                        "actual": f"received CRC 0x{rx_crc:04X}",
                    },
                )
            )
            return
        try:
            obj = cbor2.loads(payload)
        except cbor2.CBORDecodeError as exc:
            self._stats.cbor_errors += 1
            self._stats.bytes_cbor += len(frame)
            self._errors.append(
                DecodeError(
                    DecodeErrorKind.CBOR,
                    f"CBOR decode error: {exc}",
                    context={
                        "raw": bytes(payload)[:_FFDC_MAX_RAW],
                        "raw_len": len(payload),
                        "expected": "well-formed CBOR map",
                        "actual": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
            )
            return
        self._stats.frames_ok += 1
        self._stats.bytes_ok += len(frame)
        yield self._classify(obj)

    def stats(self) -> DecodeStats:
        return self._stats

    def reset(self) -> None:
        self._stats = DecodeStats()
        self._errors.clear()
        self._buf = bytearray()

    def errors_drained(self) -> Iterable[DecodeError]:
        out, self._errors = self._errors, []
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(obj: Any) -> DecodedMessage:
        t_host_ms = host_monotonic_ms()
        t_host_utc = utc_iso8601_now()

        if not isinstance(obj, dict):
            return DecodedMessage(
                kind="unknown",
                t_host_ms=t_host_ms,
                t_host_utc=t_host_utc,
                t_device_ms=None,
                seq=None,
                label=None,
                channels=None,
                raw={"_": obj},
            )

        msg_type = obj.get("t")
        if msg_type == "sd":
            d = obj.get("d") if isinstance(obj.get("d"), dict) else {}
            vals = d.get("val") if isinstance(d, dict) else None
            channels: tuple[float, ...] | None
            if isinstance(vals, (list, tuple)):
                try:
                    channels = tuple(float(v) for v in vals)
                except (TypeError, ValueError):
                    channels = None
            else:
                channels = None
            return DecodedMessage(
                kind="sensor_data",
                t_host_ms=t_host_ms,
                t_host_utc=t_host_utc,
                t_device_ms=_as_int(d.get("ts")) if isinstance(d, dict) else None,
                seq=_as_int(d.get("seq")) if isinstance(d, dict) else None,
                label=_as_label(d.get("lbl")) if isinstance(d, dict) else None,
                channels=channels,
                raw=obj,
            )

        # Anything that is not "sd" is treated as session-info / metadata.
        return DecodedMessage(
            kind="session_info" if msg_type else "unknown",
            t_host_ms=t_host_ms,
            t_host_utc=t_host_utc,
            t_device_ms=None,
            seq=None,
            label=None,
            channels=None,
            raw=obj,
        )


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):  # bool is subclass of int — exclude
        return None
    if isinstance(v, int):
        return v
    return None


def _as_label(v: Any) -> str | None:
    """Normalise the per-sample label field.

    The device sends ``lbl`` as a ``uint`` where ``0`` means "unlabeled"
    (see samples/data_forwarder/cddl). Map a non-zero integer to its decimal
    string; accept a plain string verbatim; treat everything else as unlabeled.
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return None if v == 0 else str(v)
    if isinstance(v, str):
        return v or None
    return None
