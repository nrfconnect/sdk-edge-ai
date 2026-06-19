# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""UART data source (pyserial)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import serial
import serial.tools.list_ports

from data_forwarder_host.platform.base import (
    PlatformAdapter,
    _describe_port,  # ported verbatim helper
    _looks_like_nrf,
    _short_port,
    _vid_pid,
)
from data_forwarder_host.source.base import (
    ConfigField,
    ConfigSchema,
    Source,
    SourceInfo,
)


class UartSource(Source):
    """Bytes from a serial port."""

    kind = "uart"

    def __init__(
        self,
        *,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.1,
        chunk_size: int = 4096,
    ) -> None:
        # Normalize port path: ensure Unix paths start with /
        port = port.strip()
        if port and not port.startswith("/") and not port.startswith("COM"):
            if port.startswith("dev/"):
                port = "/" + port
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._chunk_size = chunk_size
        self._serial: serial.Serial | None = None

    # ------------------------------------------------------------------
    # Discovery — uses the verbatim ported helpers
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls, platform: PlatformAdapter) -> list[SourceInfo]:
        out: list[SourceInfo] = []
        for p in serial.tools.list_ports.comports():
            details: dict[str, Any] = {
                "description": p.description,
                "manufacturer": p.manufacturer,
                "product": p.product,
                "serial_number": p.serial_number,
                "vid_pid": _vid_pid(p),
                "looks_like_nrf": _looks_like_nrf(p),
                "describe": _describe_port(p),
            }
            out.append(
                SourceInfo(
                    kind=cls.kind,
                    id=p.device,
                    display=_short_port(p),
                    details=details,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._serial is not None and self._serial.is_open:
            return
        try:
            self._serial = serial.Serial(self._port, self._baudrate, timeout=self._timeout)
        except serial.SerialException as exc:
            raise RuntimeError(f"failed to open {self._port}: {exc}") from exc

    def close(self) -> None:
        if self._serial is not None and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def chunks(self) -> Iterator[bytes]:
        if self._serial is None:
            raise RuntimeError("source not open")
        while self._serial is not None and self._serial.is_open:
            try:
                data = self._serial.read(self._chunk_size)
            except (TypeError, OSError, serial.SerialException):
                # Port was closed by another thread (fd → None) or an OS error
                # occurred.  Exit the generator cleanly; the caller sees a
                # normal end-of-iteration.
                break
            if data:
                yield data

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            fields=(
                ConfigField("port", "Serial port", "str", required=True,
                            help="e.g. /dev/ttyACM0, COM5"),
                ConfigField("baudrate", "Baud rate", "int", default=115200),
                ConfigField("timeout", "Read timeout (s)", "float", default=0.1),
                ConfigField("chunk_size", "Read chunk (bytes)", "int", default=4096),
            )
        )
