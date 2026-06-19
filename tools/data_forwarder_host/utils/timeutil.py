# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Time helpers — host monotonic milliseconds and UTC ISO-8601 strings."""

from __future__ import annotations

import time
from datetime import datetime, timezone


def host_monotonic_ms() -> int:
    """Host monotonic clock in integer milliseconds."""
    return int(time.monotonic() * 1000.0)


def utc_iso8601_now() -> str:
    """ISO-8601 UTC timestamp with microsecond precision, ``Z`` suffix."""
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond:06d}Z"


def utc_iso8601_from_epoch_us(epoch_us: int) -> str:
    """Convert microseconds since the Unix epoch to ISO-8601 UTC."""
    s, us = divmod(epoch_us, 1_000_000)
    dt = datetime.fromtimestamp(s, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{us:06d}Z"


def stamp_for_filename() -> str:
    """``YYYYMMDD-HHMMSS`` (local time), suitable as a filename component."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")
