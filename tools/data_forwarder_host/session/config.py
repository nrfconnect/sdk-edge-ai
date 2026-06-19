# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""SessionConfig and SourceSpec (validation included).

A session is configured with a single data source, a recording label, and a
per-session output directory. There is no protocol or sink configuration — the
protocol is the single fixed COBS/CBOR v1 and output is a single CSV file.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from data_forwarder_host.source.base import RoleMode

# Supported source kinds (UART implemented, BLE NUS declared).
_SOURCE_KINDS = {"uart", "ble"}


class ConfigError(ValueError):
    """Raised when a ``SessionConfig`` fails ``validate()``."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    kind: str                          # "uart" | "ble"
    role: RoleMode = RoleMode.LISTENER
    params: dict[str, Any] = field(default_factory=dict)


# Tag and label share the same character set so both are safe in a filename.
_TAG_RE = re.compile(r"^[A-Za-z0-9_\-\.]{1,64}$")


@dataclass(frozen=True, slots=True)
class SessionConfig:
    tag: str
    source: SourceSpec
    label: str = ""                    # recording label (G4); builds CSV filename
    output_dir: str = ""               # per-session CSV output directory (F4)
    expect_crc: bool = True            # single-protocol option
    plot_window_seconds: float = 10.0
    # Trailing window for the live bandwidth monitor. Defaults to the
    # plot window and is adjustable only from the session window, not the
    # New Session dialog.
    bandwidth_window_seconds: float = 10.0
    # Grace period (seconds) a missing sensor_data sequence number is awaited
    # before it is confirmed a transport loss. Adjustable live from
    # the error panel; reordered/late frames arriving within it are not losses.
    loss_confirmation_window_seconds: float = 1.0
    layout_state: dict = field(default_factory=dict)
    config_version: int = 2

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def with_auto_tag(cls, *, source: SourceSpec, **kwargs: Any) -> "SessionConfig":
        tag = f"{source.kind}-{uuid.uuid4().hex[:6]}"
        return cls(tag=tag, source=source, **kwargs)  # type: ignore[arg-type]

    def with_tag(self, tag: str) -> "SessionConfig":
        return replace(self, tag=tag)

    # ------------------------------------------------------------------
    # Recording label (G4)
    # ------------------------------------------------------------------

    def has_recording_label(self) -> bool:
        """Return ``True`` if a valid, non-empty recording label is defined.

        The Record action is enabled only when this is true.
        """
        lbl = self.label.strip() if self.label else ""
        return bool(lbl) and bool(_TAG_RE.match(lbl))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        errs: list[str] = []

        tag = self.tag.strip() if self.tag else ""
        if not tag:
            errs.append("tag must not be empty")
        elif not _TAG_RE.match(tag):
            errs.append(f"tag must match [A-Za-z0-9_-.]{{1,64}} (got {self.tag!r})")

        if self.source.kind not in _SOURCE_KINDS:
            errs.append(f"unknown source kind: {self.source.kind!r}")
        if self.source.kind == "uart" and not self.source.params.get("port"):
            errs.append("uart source requires a 'port' parameter")

        if self.label.strip() and not _TAG_RE.match(self.label.strip()):
            errs.append(
                f"label must match [A-Za-z0-9_-.]{{1,64}} (got {self.label!r})"
            )

        if not (1.0 <= self.plot_window_seconds <= 600.0):
            errs.append(
                f"plot_window_seconds must be in [1, 600] (got {self.plot_window_seconds})"
            )

        if not (0.1 <= self.bandwidth_window_seconds <= 600.0):
            errs.append(
                f"bandwidth_window_seconds must be in [0.1, 600] (got {self.bandwidth_window_seconds})"
            )

        if not (0.05 <= self.loss_confirmation_window_seconds <= 30.0):
            errs.append(
                "loss_confirmation_window_seconds must be in [0.05, 30] "
                f"(got {self.loss_confirmation_window_seconds})"
            )

        if errs:
            raise ConfigError(
                "invalid session configuration:\n  - " + "\n  - ".join(errs)
            )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def config_to_dict(config: "SessionConfig") -> "dict[str, Any]":
    """Serialize *config* to a plain JSON-compatible dict."""
    return {
        "config_version": config.config_version,
        "tag": config.tag,
        "source": {
            "kind": config.source.kind,
            "role": config.source.role.name,
            "params": dict(config.source.params),
        },
        "label": config.label,
        "output_dir": config.output_dir,
        "expect_crc": config.expect_crc,
        "plot_window_seconds": config.plot_window_seconds,
        "bandwidth_window_seconds": config.bandwidth_window_seconds,
        "loss_confirmation_window_seconds": config.loss_confirmation_window_seconds,
        "layout_state": dict(config.layout_state),
    }


def config_from_dict(d: "dict[str, Any]") -> "SessionConfig":
    """Reconstruct a :class:`SessionConfig` from a serialized dict."""
    s = d["source"]
    return SessionConfig(
        tag=d["tag"],
        source=SourceSpec(
            kind=s["kind"],
            role=RoleMode[s.get("role", "LISTENER")],
            params=dict(s.get("params", {})),
        ),
        label=str(d.get("label", "")),
        output_dir=str(d.get("output_dir", "")),
        expect_crc=bool(d.get("expect_crc", True)),
        plot_window_seconds=float(d.get("plot_window_seconds", 10.0)),
        bandwidth_window_seconds=float(d.get("bandwidth_window_seconds", 10.0)),
        loss_confirmation_window_seconds=float(
            d.get("loss_confirmation_window_seconds", 1.0)
        ),
        layout_state=dict(d.get("layout_state", {})),
        config_version=int(d.get("config_version", 2)),
    )
