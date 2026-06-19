# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""First Failure Data Capture (FFDC) report building.

When a decode failure first occurs the decoder attaches diagnostic context to
the :class:`~data_forwarder_host.core.error_log.ErrorEvent` (the offending
bytestream, what was expected vs. what actually arrived, and a Python
traceback where one exists). This module turns the per-session error journal
into a structured, GUI-free FFDC report: for each error *category* it keeps the
**first** occurrence (the classic "first failure" capture) and renders the
diagnostics into labelled, individually-focusable panels.

Kept free of Qt so it is isolated; the GUI ``FfdcDialog`` only lays the
resulting :class:`FfdcEntry`/:class:`FfdcPanel` objects out as collapsible
sections.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing Qt-bound types at runtime
    from data_forwarder_host.core.error_log import ErrorEvent

_HEX_BYTES_PER_ROW = 16


@dataclass(frozen=True)
class FfdcPanel:
    """One focusable view within a single failure capture."""

    title: str
    text: str


@dataclass(frozen=True)
class FfdcEntry:
    """The first captured occurrence of one error category."""

    category: str
    timestamp: str
    detail: str
    panels: list[FfdcPanel]


def hexdump(data: bytes, *, bytes_per_row: int = _HEX_BYTES_PER_ROW) -> str:
    """Render *data* as an offset/hex/ascii hexdump (one row per 16 bytes)."""
    if not data:
        return "(empty)"
    rows: list[str] = []
    for off in range(0, len(data), bytes_per_row):
        chunk = data[off : off + bytes_per_row]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        hex_part = hex_part.ljust(bytes_per_row * 3 - 1)
        rows.append(f"{off:08x}  {hex_part}  {ascii_view(chunk)}")
    return "\n".join(rows)


def ascii_view(data: bytes) -> str:
    """Return a printable-ASCII view of *data* (non-printables shown as ``.``)."""
    return "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in data)


def _coerce_bytes(value: object) -> bytes | None:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return None


def _build_panels(detail: str, context: dict) -> list[FfdcPanel]:
    panels: list[FfdcPanel] = []

    # Python error trace, when the failure carried an exception.
    tb = context.get("traceback")
    if isinstance(tb, str) and tb.strip() and tb.strip() != "NoneType: None":
        panels.append(FfdcPanel("Python error trace", tb.rstrip()))

    # Real issue analysis: expected vs. what actually arrived.
    expected = context.get("expected")
    actual = context.get("actual")
    if expected is not None or actual is not None:
        panels.append(
            FfdcPanel(
                "Expected vs. actual",
                f"Detail:   {detail}\n"
                f"Expected: {expected if expected is not None else '(n/a)'}\n"
                f"Actual:   {actual if actual is not None else '(n/a)'}",
            )
        )

    # The offending bytestream, in hex + ascii.
    raw = _coerce_bytes(context.get("raw"))
    if raw is not None:
        raw_len = context.get("raw_len", len(raw))
        truncated = isinstance(raw_len, int) and raw_len > len(raw)
        suffix = (
            f"\n... ({raw_len - len(raw)} more byte(s) not captured)"
            if truncated
            else ""
        )
        panels.append(
            FfdcPanel("Bytestream — hex", hexdump(raw) + suffix)
        )
        panels.append(
            FfdcPanel("Bytestream — ASCII", ascii_view(raw) + suffix)
        )

    return panels


def build_ffdc(events: Sequence[ErrorEvent]) -> list[FfdcEntry]:
    """Return one :class:`FfdcEntry` per category, for its first occurrence.

    Categories appear in the order their first failure was observed. Each entry
    exposes a list of :class:`FfdcPanel` views (Python trace, expected/actual,
    bytestream hex/ascii) so the UI can let the user focus on one at a time.
    """
    entries: list[FfdcEntry] = []
    seen: set[str] = set()
    for evt in events:
        name = evt.category.name
        if name in seen:
            continue
        seen.add(name)
        entries.append(
            FfdcEntry(
                category=name,
                timestamp=evt.t_host_utc,
                detail=evt.detail,
                panels=_build_panels(evt.detail, dict(evt.context)),
            )
        )
    return entries
