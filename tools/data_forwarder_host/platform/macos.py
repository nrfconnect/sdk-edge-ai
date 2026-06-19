# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""macOS platform adapter."""

from __future__ import annotations

import errno

import serial.tools.list_ports

from data_forwarder_host.platform.base import (
    JLinkInfo,
    PlatformAdapter,
    SerialPortInfo,
    serial_port_from_pyserial,
)


class MacosPlatform(PlatformAdapter):
    """Adapter for macOS hosts."""

    name = "macos"

    def list_serial_ports(self) -> list[SerialPortInfo]:
        return [serial_port_from_pyserial(p) for p in serial.tools.list_ports.comports()]

    def list_jlink_devices(self) -> list[JLinkInfo]:
        # TODO: integrate pylink/JLink enumeration in a future revision.
        return []

    def diagnose_access_error(self, exc: BaseException) -> str:
        eno = getattr(exc, "errno", None)
        if eno == errno.EACCES:
            return (
                f"Permission denied: {exc}\n"
                f"On macOS, prefer the /dev/cu.* device node rather than /dev/tty.*\n"
                f"and ensure no other tool is holding the port."
            )
        if eno == errno.EBUSY:
            return f"Device is busy: {exc}\nAnother program may be holding the port open."
        return str(exc)
