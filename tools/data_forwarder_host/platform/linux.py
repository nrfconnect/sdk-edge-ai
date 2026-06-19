# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Linux platform adapter."""

from __future__ import annotations

import errno
import os

import serial.tools.list_ports

from data_forwarder_host.platform.base import (
    JLinkInfo,
    PlatformAdapter,
    SerialPortInfo,
    serial_port_from_pyserial,
)


class LinuxPlatform(PlatformAdapter):
    """Adapter for Linux hosts."""

    name = "linux"

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
                f"On Linux, add your user to the 'dialout' group:\n"
                f"  sudo usermod -aG dialout {os.environ.get('USER', '$USER')}\n"
                f"Then log out and back in."
            )
        if eno == errno.ENOENT:
            return f"No such device: {exc}\nIs the device plugged in? Try `ls /dev/ttyACM* /dev/ttyUSB*`."
        if eno == errno.EBUSY:
            return f"Device is busy: {exc}\nAnother program may be holding the port open."
        return str(exc)
