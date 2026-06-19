# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Windows platform adapter."""

from __future__ import annotations

import serial.tools.list_ports

from data_forwarder_host.platform.base import (
    JLinkInfo,
    PlatformAdapter,
    SerialPortInfo,
    serial_port_from_pyserial,
)


class WindowsPlatform(PlatformAdapter):
    """Adapter for Windows hosts."""

    name = "windows"

    def list_serial_ports(self) -> list[SerialPortInfo]:
        return [serial_port_from_pyserial(p) for p in serial.tools.list_ports.comports()]

    def list_jlink_devices(self) -> list[JLinkInfo]:
        # TODO: integrate pylink/JLink enumeration in a future revision.
        return []

    def diagnose_access_error(self, exc: BaseException) -> str:
        text = str(exc).lower()
        if "access is denied" in text or "permissiondenied" in text:
            return (
                f"Access denied: {exc}\n"
                f"Another program (for example a serial terminal, IDE or RTT viewer)\n"
                f"may already have the COM port open. Close it and retry."
            )
        if "filenotfound" in text or "cannot find" in text:
            return f"COM port not found: {exc}\nUnplug/replug the device or check Device Manager."
        return str(exc)
