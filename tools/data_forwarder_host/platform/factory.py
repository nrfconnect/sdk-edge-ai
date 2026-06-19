# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Platform-adapter factory."""

from __future__ import annotations

import sys

from data_forwarder_host.platform.base import PlatformAdapter
from data_forwarder_host.platform.linux import LinuxPlatform
from data_forwarder_host.platform.macos import MacosPlatform
from data_forwarder_host.platform.windows import WindowsPlatform


def detect_platform() -> PlatformAdapter:
    """Return the ``PlatformAdapter`` matching the running OS."""
    if sys.platform.startswith("linux"):
        return LinuxPlatform()
    if sys.platform == "win32":
        return WindowsPlatform()
    if sys.platform == "darwin":
        return MacosPlatform()
    # Unknown POSIX-ish OS — fall back to the Linux adapter; its diagnostics
    # still tend to make sense.
    return LinuxPlatform()
