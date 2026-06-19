# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Data-source layer — byte sources (UART implemented, BLE NUS declared).

A single tiny ``source_for_kind`` factory maps the two known source kinds to
their classes. There is no pluggable registry abstraction.
"""

from __future__ import annotations

from data_forwarder_host.source.base import (
    ConfigField,
    ConfigSchema,
    RoleMode,
    Source,
    SourceInfo,
)
from data_forwarder_host.source.ble_nus import BleNusSource
from data_forwarder_host.source.uart import UartSource

# The two — and only two — supported source kinds.
SOURCE_KINDS: tuple[str, ...] = (UartSource.kind, BleNusSource.kind)

_BY_KIND: dict[str, type[Source]] = {
    UartSource.kind: UartSource,
    BleNusSource.kind: BleNusSource,
}


def source_for_kind(kind: str) -> type[Source]:
    """Return the :class:`Source` subclass for *kind* (``"uart"`` or ``"ble"``)."""
    try:
        return _BY_KIND[kind]
    except KeyError:
        raise ValueError(f"unknown source kind: {kind!r}") from None


__all__ = [
    "ConfigField",
    "ConfigSchema",
    "RoleMode",
    "Source",
    "SourceInfo",
    "UartSource",
    "BleNusSource",
    "SOURCE_KINDS",
    "source_for_kind",
]
