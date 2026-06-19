# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Run a blocking callable on a worker thread and report the result via signals.

Device discovery (UART enumeration, and especially a BLE radio scan that waits
a fixed timeout) blocks for several seconds. Running it on the Qt GUI thread
freezes the whole window. :class:`ScanWorker` wraps any no-argument callable so
it can be executed on a :class:`~PySide6.QtCore.QThread`, emitting exactly one
of ``done`` (with the return value) or ``failed`` (with the raised exception)
back on the thread the signals are connected from.

The worker is deliberately generic and Qt-only: it knows nothing about sources
or discovery, so it is reusable in isolation without a display.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Signal


class ScanWorker(QObject):
    """Execute a no-arg callable, emitting ``done(result)`` or ``failed(exc)``.

    Exactly one signal is emitted per :meth:`run` call. Move the worker to a
    :class:`QThread` and trigger :meth:`run` from the thread's ``started``
    signal to keep the GUI responsive while the callable blocks.
    """

    done = Signal(object)
    failed = Signal(object)

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 - reported to the caller via signal
            self.failed.emit(exc)
        else:
            self.done.emit(result)
