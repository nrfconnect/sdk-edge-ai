# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Sidecar session-metadata dump written next to each recording CSV.

When a recording is saved as ``{stem}.csv`` a human-readable companion file
``{stem}.txt`` is written with the same base name. It captures everything about
the capture that the CSV rows themselves do not carry: the transport used, the
device that produced the data (serial port / BLE name + address), the host OS
and host timestamps, the device-reported ``session_info`` envelope, the channel
layout and a short error/loss summary.

The file is plain ``key: value`` lines grouped under ``[Section]`` headers so it
is both easy to read and trivial to parse back if needed.
"""

from __future__ import annotations

import platform
import socket
import sys
from typing import TYPE_CHECKING, Any

from data_forwarder_host.core.error_log import ErrorCategory
from data_forwarder_host.utils.timeutil import utc_iso8601_now

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from data_forwarder_host.core.recorder import Recording


def _transport_details(kind: str, params: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ordered ``(label, value)`` rows describing the device endpoint."""
    kind = (kind or "").lower()
    if kind == "uart":
        return [
            ("transport", "UART / serial"),
            ("serial_port", str(params.get("port", "") or "(unknown)")),
            ("baudrate", str(params.get("baudrate", 115200))),
        ]
    if kind == "ble":
        return [
            ("transport", "BLE (Nordic UART Service)"),
            ("device_name", str(params.get("name", "") or "(unknown)")),
            ("device_address", str(params.get("address", "") or "(unknown)")),
        ]
    return [("transport", kind or "(unknown)")]


def _session_info_rows(session_info: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Flatten the device ``session_info`` ('si') envelope into rows."""
    if not isinstance(session_info, dict):
        return [("session_info", "(not received)")]
    payload = session_info.get("d")
    if not isinstance(payload, dict):
        return [("session_info", "(malformed)")]
    rows: list[tuple[str, str]] = []
    labels = {
        "name": "device_reported_name",
        "sid": "session_id",
        "hz": "sampling_rate_hz",
        "st": "device_start_time",
        "ch": "channel_count",
        "dr": "producer_drop_count",
    }
    for key, label in labels.items():
        if key in payload:
            rows.append((label, str(payload[key])))
    ch_n = payload.get("ch_n")
    if isinstance(ch_n, (list, tuple)) and ch_n:
        rows.append(("channels", ", ".join(str(c) for c in ch_n)))
    # Surface any remaining unknown fields verbatim so nothing is silently lost.
    for key, value in payload.items():
        if key not in labels and key != "ch_n":
            rows.append((f"si_{key}", str(value)))
    return rows


def _error_rows(recording: "Recording") -> list[tuple[str, str]]:
    summary = recording.error_summary
    rows: list[tuple[str, str]] = [
        ("total_messages", str(summary.total_messages)),
        ("incomplete", "yes" if recording.incomplete else "no"),
    ]
    for category in ErrorCategory:
        count = summary.counts.get(category, 0)
        if count:
            rows.append((category.name.lower(), str(count)))
    return rows


def build_metadata_text(
    recording: "Recording",
    *,
    label: str,
    source_kind: str,
    source_params: dict[str, Any],
    source_description: str,
    protocol_description: str,
    csv_filename: str | None = None,
) -> str:
    """Render the sidecar metadata text for a finished *recording*."""
    row_count = getattr(recording.storage, "row_count", None)

    sections: list[tuple[str, list[tuple[str, str]]]] = []

    recording_rows: list[tuple[str, str]] = [
        ("label", label),
        ("session_tag", recording.session_tag),
    ]
    if csv_filename:
        recording_rows.append(("csv_file", csv_filename))
    recording_rows.append(("data_rows", str(row_count) if row_count is not None else "(unknown)"))
    sections.append(("Recording", recording_rows))

    sections.append((
        "Host",
        [
            ("written_utc", utc_iso8601_now()),
            ("os", platform.platform()),
            ("os_system", platform.system()),
            ("os_release", platform.release()),
            ("os_version", platform.version()),
            ("machine", platform.machine()),
            ("hostname", socket.gethostname()),
            ("python", sys.version.split()[0]),
        ],
    ))

    transport_rows = _transport_details(source_kind, source_params)
    transport_rows.append(("source", source_description))
    transport_rows.append(("protocol", protocol_description))
    sections.append(("Transport", transport_rows))

    sections.append((
        "Timing",
        [
            ("started_utc", recording.started_utc),
            ("stopped_utc", recording.stopped_utc),
        ],
    ))

    sections.append(("Device session_info", _session_info_rows(recording.session_info)))
    sections.append(("Channels", [
        ("count", str(len(recording.channel_names))),
        ("names", ", ".join(recording.channel_names) if recording.channel_names else "(none)"),
    ]))
    sections.append(("Errors", _error_rows(recording)))

    lines: list[str] = []
    for i, (title, rows) in enumerate(sections):
        if i:
            lines.append("")
        lines.append(f"[{title}]")
        width = max((len(k) for k, _ in rows), default=0)
        for key, value in rows:
            lines.append(f"{key.ljust(width)} : {value}")
    return "\n".join(lines) + "\n"
