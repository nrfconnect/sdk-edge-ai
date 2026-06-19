# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Data-source ABC and shared types.

Only the LISTENER role exists (device-initiated streaming). There is no
client/host-initiated mode anywhere in the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from data_forwarder_host.platform.base import PlatformAdapter


class RoleMode(Enum):
    """Direction of control for a data source.

    Only ``LISTENER`` exists: the device pushes, the host receives.
    """

    LISTENER = auto()


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Lightweight description of a discoverable source endpoint."""

    kind: str
    id: str                              # stable identifier (e.g. "/dev/ttyACM0")
    display: str                         # user-facing label
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConfigField:
    """Description of a single configurable parameter for a source."""

    name: str
    label: str
    kind: str                            # "str" | "int" | "float" | "bool" | "path" | "choice"
    default: Any = None
    required: bool = True
    choices: tuple[str, ...] = ()
    help: str = ""


@dataclass(frozen=True, slots=True)
class ConfigSchema:
    """Ordered set of ``ConfigField`` definitions used by the GUI dialog."""

    fields: tuple[ConfigField, ...] = ()


class Source(ABC):
    """Abstract byte-source (listener role only)."""

    kind: str

    @classmethod
    @abstractmethod
    def discover(cls, platform: PlatformAdapter) -> list[SourceInfo]:
        """Return currently discoverable source endpoints."""

    @abstractmethod
    def open(self) -> None:
        """Open the underlying resource."""

    @abstractmethod
    def close(self) -> None:
        """Close the underlying resource."""

    @abstractmethod
    def chunks(self) -> Iterator[bytes]:
        """Yield raw byte chunks until ``close()`` is called or the source ends."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        ...

    @classmethod
    @abstractmethod
    def config_schema(cls) -> ConfigSchema:
        """Configuration schema used by the New-Session dialog."""
