# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Continuous (live) BLE scanner that streams device detections to the GUI.

Unlike the one-shot :meth:`BleNusSource.discover` (which waits a fixed timeout
and returns the whole batch at once), :class:`BleLiveScanner` starts a
``bleak`` scanner and reports every advertisement as it arrives via Qt signals.
The New Session dialog uses it so the device list fills in *as the window is
open* — devices appear within a fraction of a second and there is no Refresh
button or fixed scan wait.

The scanner runs its own asyncio event loop on a worker :class:`QThread`; the
signals are delivered to the GUI thread (queued connections), so widget updates
happen on the main thread.
"""

from __future__ import annotations

import asyncio
from typing import Any

from PySide6.QtCore import QObject, Signal


class BleLiveScanner(QObject):
    """Run a continuous BLE scan, emitting one signal per detection/state.

    Signals (delivered on the GUI thread when connected from it):
    - ``device_found(SourceInfo)`` — a device advertisement was seen;
    - ``state_changed(str)`` — ``"on"`` once scanning started, ``"off"`` when no
      usable Bluetooth backend is available, ``"unknown"`` if it cannot tell.
    """

    device_found = Signal(object)
    state_changed = Signal(str)

    #: Hard cap (seconds) on a single ``bleak`` start/stop call, so a wedged
    #: Bluetooth backend can never hang the worker thread forever — which, on
    #: dialog teardown, would otherwise abort the process with
    #: "QThread: Destroyed while thread is still running".
    _BLEAK_OP_TIMEOUT = 5.0

    def __init__(self) -> None:
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_evt: asyncio.Event | None = None
        # Set from the GUI thread the instant a stop is requested. Read by the
        # scan coroutine so a stop that races *ahead* of loop/event creation is
        # never lost (otherwise the coroutine would block forever and its thread
        # could not be joined on teardown).
        self._stop_requested = False

    # -- worker-thread entry point -----------------------------------------

    def run(self) -> None:
        """Event-loop body; invoke from a worker ``QThread.started`` signal."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._scan())
        except Exception:  # noqa: BLE001 - never let the worker thread die noisily
            pass
        finally:
            try:
                self._loop.close()
            finally:
                asyncio.set_event_loop(None)

    async def _scan(self) -> None:
        try:
            from bleak import BleakScanner
        except ImportError:
            self.state_changed.emit("unknown")
            return

        from data_forwarder_host.source.ble_nus import (
            _NO_BLUETOOTH_BACKEND_MSG,
            BleNusSource,
            _describe_scan_failure,
        )

        self._stop_evt = asyncio.Event()
        # A stop may already have been requested before the event existed.
        if self._stop_requested:
            return

        def _on_detect(device: Any, adv: Any) -> None:
            try:
                info = BleNusSource._info_from_advertisement(
                    str(getattr(device, "address", "") or ""), device, adv
                )
            except Exception:  # noqa: BLE001 - a single bad advert must not stop the scan
                return
            self.device_found.emit(info)

        scanner = BleakScanner(detection_callback=_on_detect)
        try:
            await asyncio.wait_for(scanner.start(), timeout=self._BLEAK_OP_TIMEOUT)
        except asyncio.TimeoutError:
            self.state_changed.emit("unknown")
            return
        except Exception as exc:  # noqa: BLE001 - classified for the UI below
            reason = _describe_scan_failure(exc)
            self.state_changed.emit(
                "off" if reason == _NO_BLUETOOTH_BACKEND_MSG else "unknown"
            )
            return

        # If a stop landed during start(), tear the scanner down at once.
        if self._stop_requested:
            await self._safe_stop_scanner(scanner)
            return

        self.state_changed.emit("on")
        try:
            await self._stop_evt.wait()
        finally:
            await self._safe_stop_scanner(scanner)

    @staticmethod
    async def _safe_stop_scanner(scanner: Any) -> None:
        """Stop *scanner* without ever blocking the worker thread indefinitely."""
        try:
            await asyncio.wait_for(
                scanner.stop(), timeout=BleLiveScanner._BLEAK_OP_TIMEOUT
            )
        except Exception:  # noqa: BLE001 - best-effort teardown (incl. timeout)
            pass

    # -- GUI-thread control -------------------------------------------------

    def stop(self) -> None:
        """Ask the scan loop to stop; safe to call from the GUI thread."""
        self._stop_requested = True
        loop = self._loop
        stop_evt = self._stop_evt
        if loop is not None and stop_evt is not None:
            try:
                loop.call_soon_threadsafe(stop_evt.set)
            except RuntimeError:
                # Loop already closed/stopped — nothing left to wake.
                pass
