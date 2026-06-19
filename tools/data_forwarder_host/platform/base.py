# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Platform adapter base class and shared helpers.

The Nordic / SEGGER detection heuristics (``NRF_HINTS``, ``_looks_like_nrf``,
``_vid_pid``, ``_describe_port``, ``_short_port``) originated in the original
prototype serial receiver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SerialPortInfo:
    """Lightweight, transport-agnostic description of a serial port."""

    device: str
    description: str | None = None
    manufacturer: str | None = None
    product: str | None = None
    serial_number: str | None = None
    vid: int | None = None
    pid: int | None = None
    location: str | None = None
    interface: str | None = None
    hwid: str | None = None
    looks_like_nrf: bool = False


@dataclass(frozen=True, slots=True)
class JLinkInfo:
    """Description of a J-Link probe (currently a stub for v1)."""

    serial: str
    product: str | None = None


# ---------------------------------------------------------------------------
# Nordic / SEGGER detection helpers (verbatim from the prototype)
# ---------------------------------------------------------------------------

# Substrings (case-insensitive) suggesting "this is a Nordic / SEGGER device".
NRF_HINTS: tuple[str, ...] = ("nordic", "segger", "j-link", "jlink", "nrf")


def _looks_like_nrf(p: Any) -> bool:
    """Heuristic: does this port look like a Nordic / SEGGER device?"""
    haystack = " ".join(
        s for s in (p.manufacturer, p.description, p.product) if s
    ).lower()
    return any(h in haystack for h in NRF_HINTS)


def _vid_pid(p: Any) -> str | None:
    if p.vid is not None and p.pid is not None:
        return f"{p.vid:04x}:{p.pid:04x}"
    return None


def _describe_port(p: Any, indent: str = "    ") -> str:
    """Multi-line, detailed description of a serial port."""
    lines = [f"{p.device}"]

    def add(label: str, value: str | None) -> None:
        if value:
            lines.append(f"{indent}  {label:<14}{value}")

    add("description:", p.description if p.description and p.description != "n/a" else None)
    add("manufacturer:", p.manufacturer)
    add("product:", p.product)
    add("serial:", p.serial_number)
    add("vid:pid:", _vid_pid(p))
    add("interface:", getattr(p, "interface", None))
    add("location:", getattr(p, "location", None))
    add("hwid:", p.hwid if p.hwid and p.hwid != "n/a" else None)
    return "\n".join(indent + l if i else l for i, l in enumerate(lines))


def _short_port(p: Any) -> str:
    """One-line port summary for inline messages."""
    bits = [p.device]
    if p.description and p.description != "n/a":
        bits.append(p.description)
    extras = []
    if p.manufacturer:
        extras.append(p.manufacturer)
    vp = _vid_pid(p)
    if vp:
        extras.append(f"VID:PID={vp}")
    if p.serial_number:
        extras.append(f"SN={p.serial_number}")
    if extras:
        bits.append("(" + ", ".join(extras) + ")")
    return "  ".join(bits)


# ---------------------------------------------------------------------------
# Platform adapter ABC
# ---------------------------------------------------------------------------


class PlatformAdapter(ABC):
    """Abstract per-OS adapter exposing serial / J-Link enumeration."""

    name: str

    @abstractmethod
    def list_serial_ports(self) -> list[SerialPortInfo]:
        """Enumerate currently-attached serial ports."""

    @abstractmethod
    def list_jlink_devices(self) -> list[JLinkInfo]:
        """Enumerate attached J-Link probes (stub in v1)."""

    @abstractmethod
    def diagnose_access_error(self, exc: BaseException) -> str:
        """Return a user-friendly diagnostic for a transport-access failure."""


def serial_port_from_pyserial(p: Any) -> SerialPortInfo:
    """Convert a ``serial.tools.list_ports`` entry to a ``SerialPortInfo``."""
    return SerialPortInfo(
        device=p.device,
        description=p.description if p.description and p.description != "n/a" else None,
        manufacturer=p.manufacturer,
        product=p.product,
        serial_number=p.serial_number,
        vid=p.vid,
        pid=p.pid,
        location=getattr(p, "location", None),
        interface=getattr(p, "interface", None),
        hwid=p.hwid if p.hwid and p.hwid != "n/a" else None,
        looks_like_nrf=_looks_like_nrf(p),
    )
