# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""BLE NUS (Nordic UART Service) data source.

The host acts as a BLE *central*: it scans for advertising devices, connects to
a chosen one, and subscribes to notifications on the NUS **TX** characteristic.
The device-side sample (``samples/data_forwarder``) advertises the NUS service
UUID under the name ``"nRF DataFwd"`` and streams COBS/CBOR frames out via
``bt_nus_send`` (TX notifications). The raw notification payloads are handed to
the same COBS/CBOR decoder used by the UART source.

The cross-platform BLE backend is :mod:`bleak`, which is async. This source
runs a private asyncio event loop on a background thread and bridges incoming
notifications to the synchronous :meth:`chunks` generator through a bounded
queue, so the rest of the pipeline (the :class:`~data_forwarder_host.session.\
forwarding.IoWorker`) sees the same blocking byte-source contract as UART.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Iterator
from typing import Any

from data_forwarder_host.platform.base import PlatformAdapter
from data_forwarder_host.source.base import (
    ConfigSchema,
    Source,
    SourceInfo,
)

log = logging.getLogger(__name__)

# Nordic UART Service UUIDs (the device is the peripheral; the host subscribes
# to TX notifications to receive streamed data).
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # peripheral → central (notify)

# Default advertised name of the device-side data_forwarder sample.
DEFAULT_DEVICE_NAME = "nRF DataFwd"

# The BLE minimum ATT MTU. When the link reports this value the MTU was never
# upgraded, so every notification carries at most MTU-3 (=20) payload bytes and
# each sensor frame is fragmented across multiple notifications.
_DEFAULT_ATT_MTU = 23

# Number of connect attempts and the delay between them. The Windows (WinRT)
# backend can abort the first fresh GATT discovery (reporting "cancelled" or
# "Unreachable") even when the device is perfectly reachable, but a retry
# usually succeeds; the cost on a genuinely absent device is bounded by
# ``connect_timeout`` per attempt.
_CONNECT_ATTEMPTS = 3
_CONNECT_RETRY_DELAY = 1.0

_MISSING_BLEAK_MSG = (
    "BLE support requires the 'bleak' package. Install it with 'pip install bleak'."
)

# Friendly explanation for the most common scan failure: there is no usable
# Bluetooth backend (no running BlueZ daemon / D-Bus system bus / adapter). On
# Linux this surfaces from the BlueZ backend as a bare
# ``FileNotFoundError: [Errno 2] No such file or directory`` (the D-Bus socket
# is absent) or as a D-Bus/org.bluez connection error — both useless to a user.
_NO_BLUETOOTH_BACKEND_MSG = (
    "no usable Bluetooth backend was found. BLE scanning needs a running "
    "Bluetooth stack (BlueZ + D-Bus system bus) with an adapter, which is "
    "typically unavailable inside a container without host Bluetooth access. "
    "Run the host on a machine with Bluetooth, or grant the container access "
    "to the host's Bluetooth (D-Bus system bus + a BLE adapter)."
)


def _describe_scan_failure(exc: BaseException) -> str:
    """Return a human-readable, actionable reason for a BLE scan failure.

    Recognises the common "no usable Bluetooth backend" signatures (missing
    D-Bus system bus socket, BlueZ daemon, or adapter) and replaces the cryptic
    underlying error — e.g. ``[Errno 2] No such file or directory`` when the
    D-Bus socket is absent — with a clear explanation, instead of leaking a raw
    errno string to the user. Any other failure is reported verbatim.
    """
    text = str(exc).lower()
    no_backend = (
        isinstance(exc, (FileNotFoundError, ConnectionError))
        or "org.bluez" in text
        or "bluez" in text
        or "dbus" in text
        or "d-bus" in text
        or "no such file or directory" in text
        or "connection refused" in text
    )
    if no_backend:
        return _NO_BLUETOOTH_BACKEND_MSG
    return str(exc) or exc.__class__.__name__


def infos_matching_name(infos: list[SourceInfo], name: str) -> list[SourceInfo]:
    """Filter discovered devices to those advertising *name*.

    An empty/blank *name* matches everything (so "Discover…" with no name set
    lists every device in range). Matching is exact on the advertised name.
    """
    wanted = (name or "").strip()
    if not wanted:
        return list(infos)
    return [i for i in infos if (i.details.get("name") or "") == wanted]


def _run_blocking(coro: Any, *, timeout: float | None = None) -> Any:
    """Run *coro* to completion on a throwaway event loop and return its result.

    Works whether or not the calling thread already owns a running event loop
    (it always uses a fresh loop on a dedicated thread), so it is safe to call
    from the Qt GUI thread.
    """
    box: dict[str, Any] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            box["value"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised to the caller below
            box["error"] = exc
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

    thread = threading.Thread(target=_runner, name="ble-oneshot", daemon=True)
    thread.start()
    thread.join(timeout)
    if "error" in box:
        raise box["error"]
    return box.get("value")


class BleNusSource(Source):
    """Bytes from a BLE peripheral's Nordic UART Service (TX notifications)."""

    kind = "ble"

    def __init__(
        self,
        *,
        address: str = "",
        name: str = "",
        scan_timeout: float = 6.0,
        connect_timeout: float = 20.0,
        **_: Any,
    ) -> None:
        # The peripheral is identified by its BLE *address* (chosen from the
        # live device list in the New Session dialog). A *name* may still be
        # supplied for display / legacy resolution; when only a name is given
        # the address is resolved by scanning at open() time.
        self._address = (address or "").strip()
        self._name = name.strip()
        self._scan_timeout = float(scan_timeout)
        self._connect_timeout = float(connect_timeout)

        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=4096)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._client: Any = None
        self._open = False
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls, platform: PlatformAdapter) -> list[SourceInfo]:
        """Scan for advertising BLE devices.

        Returns the list of discoverable devices (possibly empty if the scan
        genuinely saw nothing). A *backend* failure — ``bleak`` not installed,
        or no usable Bluetooth stack (BlueZ/D-Bus/adapter unavailable) — raises
        :class:`RuntimeError` so the caller can tell "BLE is unavailable" apart
        from "BLE works but no devices are in range", instead of both collapsing
        into a misleading empty result.
        """
        try:
            from bleak import BleakScanner
        except ImportError as exc:
            raise RuntimeError(_MISSING_BLEAK_MSG) from exc

        try:
            found = _run_blocking(
                BleakScanner.discover(timeout=6.0, return_adv=True),
                timeout=20.0,
            )
        except Exception as exc:
            reason = _describe_scan_failure(exc)
            # Expected, well-understood condition (e.g. no Bluetooth stack in a
            # container) that is already surfaced to the user by the dialog —
            # log a concise WARNING (not an alarming ERROR/traceback). The full
            # traceback is still available at DEBUG level for diagnosis.
            log.warning("BLE scan failed: %s", reason)
            log.debug("BLE scan failure detail", exc_info=True)
            raise RuntimeError(f"BLE unavailable: {reason}") from exc

        return cls._infos_from_scan(found or {})

    @classmethod
    def probe_bluetooth_state(cls, *, timeout: float = 1.5) -> str:
        """Best-effort, cross-platform read of the system Bluetooth state.

        Returns one of:
        - ``"on"``      — a scan started successfully (adapter present & powered);
        - ``"off"``     — no usable Bluetooth backend (adapter off / BlueZ/D-Bus
                           or radio unavailable);
        - ``"unknown"`` — ``bleak`` is missing or the state could not be probed.

        ``bleak`` exposes no uniform adapter-power API across Linux/Windows/
        macOS, so we run a short scan and classify the outcome with the same
        logic used for discovery failures. This is the "reflect" half of the
        reflect-and-guide Bluetooth toggle; the app never powers the adapter.
        """
        try:
            from bleak import BleakScanner
        except ImportError:
            return "unknown"

        try:
            _run_blocking(
                BleakScanner.discover(timeout=timeout, return_adv=True),
                timeout=timeout + 10.0,
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            reason = _describe_scan_failure(exc)
            if reason == _NO_BLUETOOTH_BACKEND_MSG:
                return "off"
            log.debug("Bluetooth probe inconclusive: %s", reason)
            return "unknown"
        return "on"

    @classmethod
    def _infos_from_scan(cls, found: dict[str, Any]) -> list[SourceInfo]:
        """Map a ``{address: (device, advertisement)}`` mapping to ``SourceInfo``."""
        infos: list[SourceInfo] = []
        for address, pair in found.items():
            device, adv = pair if isinstance(pair, tuple) else (pair, None)
            infos.append(cls._info_from_advertisement(str(address), device, adv))
        # NUS / Nordic devices first, then by signal strength (strongest first).
        infos.sort(
            key=lambda i: (
                0 if i.details.get("looks_like_nrf") else 1,
                -(i.details.get("rssi") or -999),
            )
        )
        return infos

    @classmethod
    def _info_from_advertisement(cls, address: str, device: Any, adv: Any) -> SourceInfo:
        """Build a :class:`SourceInfo` from a single scan detection.

        Shared by the one-shot :meth:`discover` and the live (streaming)
        scanner, so both produce identical device records.
        """
        service_uuids = [u.lower() for u in (getattr(adv, "service_uuids", None) or [])]
        has_nus = NUS_SERVICE_UUID in service_uuids
        name = (
            getattr(device, "name", None)
            or getattr(adv, "local_name", None)
            or ""
        )
        looks_like_nrf = (
            has_nus
            or name == DEFAULT_DEVICE_NAME
            or name.lower().startswith("nrf")
        )
        rssi = getattr(adv, "rssi", None)
        display = f"{name or 'unknown'} [{address}]"
        if rssi is not None:
            display = f"{display}  {rssi} dBm"
        return SourceInfo(
            kind=cls.kind,
            id=str(address),
            display=display,
            details={
                "name": name,
                "rssi": rssi,
                "has_nus": has_nus,
                "looks_like_nrf": looks_like_nrf,
            },
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._open:
            return
        try:
            from bleak import BleakClient
        except ImportError as exc:
            raise RuntimeError(_MISSING_BLEAK_MSG) from exc

        target = self._address or self._resolve_address_by_name()
        if not target:
            name = self._name or DEFAULT_DEVICE_NAME
            raise RuntimeError(
                f"no advertising BLE device named {name!r} was found; make sure "
                "the device is powered and in range, then try again."
            )
        self._address = target

        self._stop.clear()
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="ble-loop", daemon=True
        )
        self._loop_thread.start()

        async def _connect() -> None:
            # ``winrt={"use_cached_services": False}`` forces a *fresh* GATT
            # service discovery on the Windows (WinRT) backend. Windows defaults
            # to its cached service table, and for a freshly-flashed/never-bonded
            # peripheral that cache is stale or empty: the OS Bluetooth driver
            # waits ~0.5 s for the GATT response, gives up, and aborts discovery
            # — surfaced by bleak as the misleading
            # ``[WinError -2147023673] The operation was canceled by the user``.
            # Disabling the cache is the documented bleak workaround for this and
            # is inert on the BlueZ/CoreBluetooth backends, so it is cross-platform
            # safe. A fresh discovery can transiently report "Unreachable" on
            # Windows, so the connect is retried a few times before giving up.
            last_exc: BaseException | None = None
            for attempt in range(1, _CONNECT_ATTEMPTS + 1):
                if self._stop.is_set():
                    return
                self._client = BleakClient(
                    self._address,
                    disconnected_callback=self._on_disconnect,
                    timeout=self._connect_timeout,
                    winrt={"use_cached_services": False},
                )
                try:
                    await self._client.connect()
                    break
                except Exception as exc:  # noqa: BLE001 - retried/re-raised below
                    last_exc = exc
                    log.warning(
                        "BLE connect attempt %d/%d to %s failed: %s",
                        attempt,
                        _CONNECT_ATTEMPTS,
                        self._address,
                        exc,
                    )
                    try:
                        await self._client.disconnect()
                    except Exception:  # noqa: BLE001 - best-effort cleanup
                        pass
                    self._client = None
                    if attempt < _CONNECT_ATTEMPTS:
                        await asyncio.sleep(_CONNECT_RETRY_DELAY)
            else:
                raise last_exc if last_exc is not None else RuntimeError("connect failed")
            # Prefer BlueZ "AcquireNotify" for the high-rate notification stream:
            # it hands back a dedicated socket that bleak reads directly, instead
            # of routing every notification through a D-Bus PropertiesChanged
            # signal. At the default 500 Hz this materially lowers per-notification
            # overhead and reduces coalesced/dropped notifications. bleak falls
            # back to "StartNotify" automatically when the characteristic does not
            # support AcquireNotify, and the argument is inert on the non-BlueZ
            # (Windows/macOS) backends, so this is cross-platform safe.
            await self._client.start_notify(
                NUS_TX_CHAR_UUID,
                self._on_notify,
                bluez={"use_start_notify": False},
            )
            self._log_link_mtu()

        fut = asyncio.run_coroutine_threadsafe(_connect(), self._loop)
        # The worker may retry the connect up to ``_CONNECT_ATTEMPTS`` times, so
        # the outer wait must allow for every attempt (each bounded by
        # ``connect_timeout``) plus the delays between them, otherwise this
        # blocking wait would time out and tear the loop down mid-retry.
        overall_timeout = (
            self._connect_timeout * _CONNECT_ATTEMPTS
            + _CONNECT_RETRY_DELAY * (_CONNECT_ATTEMPTS - 1)
            + 10.0
        )
        try:
            fut.result(timeout=overall_timeout)
        except Exception as exc:
            self._teardown_loop()
            raise RuntimeError(
                f"failed to connect to BLE device {self._address}: {exc}"
            ) from exc
        self._open = True
        log.info("BLE NUS connected to %s", self._address)

    def _resolve_address_by_name(self) -> str:
        """Scan and return the address of the first device matching the name filter."""
        name = self._name or DEFAULT_DEVICE_NAME
        matches = infos_matching_name(self.discover(PlatformAdapter()), name)
        return matches[0].id if matches else ""

    def _log_link_mtu(self) -> None:
        """Best-effort log of the negotiated ATT MTU after connecting.

        Reading ``client.mtu_size`` also drives an MTU acquisition on the BlueZ
        backend where one has not happened yet; on the Windows/macOS backends the
        value is already negotiated by the OS. The host cannot *set* the ATT MTU
        on BlueZ (it is negotiated by the OS Bluetooth stack and must be requested
        by the peripheral), but surfacing the value lets a user see whether the
        link came up with the large MTU the device requests — so each ~90-byte
        sensor frame is carried in a single notification — or silently fell back
        to the 23-byte minimum, in which case frames are fragmented across many
        notifications and throughput suffers. Never raises: this is a diagnostic,
        not a connection requirement.
        """
        client = self._client
        try:
            mtu = getattr(client, "mtu_size", None)
        except Exception:  # noqa: BLE001 - diagnostic only, never fail the connect
            mtu = None
        if not mtu:
            return
        if mtu <= _DEFAULT_ATT_MTU:
            log.warning(
                "BLE NUS link is at the default ATT MTU = %s bytes; the peripheral "
                "did not negotiate a larger MTU, so each sensor frame is fragmented "
                "across multiple notifications and throughput suffers. A larger MTU "
                "must be requested by the device firmware — the host cannot set it.",
                mtu,
            )
        else:
            log.info("BLE NUS link negotiated ATT MTU = %s bytes", mtu)

    def close(self) -> None:
        self._stop.set()
        self._open = False
        # Unblock any chunks() consumer waiting on the queue.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        loop = self._loop
        client = self._client
        if loop is not None and client is not None and loop.is_running():
            async def _disconnect() -> None:
                try:
                    await client.stop_notify(NUS_TX_CHAR_UUID)
                except Exception:
                    pass
                try:
                    await client.disconnect()
                except Exception:
                    pass

            try:
                fut = asyncio.run_coroutine_threadsafe(_disconnect(), loop)
                fut.result(timeout=5.0)
            except Exception:
                log.exception("BLE disconnect failed")
        self._teardown_loop()

    def _teardown_loop(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
        if loop is not None and not loop.is_closed():
            try:
                loop.close()
            except Exception:
                pass
        self._loop = None
        self._loop_thread = None
        self._client = None

    @property
    def is_open(self) -> bool:
        return self._open

    def chunks(self) -> Iterator[bytes]:
        while self._open and not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item

    # ------------------------------------------------------------------
    # BLE callbacks (run on the asyncio loop thread)
    # ------------------------------------------------------------------

    def _on_notify(self, _char: Any, data: bytearray) -> None:
        if not data:
            return
        try:
            self._queue.put_nowait(bytes(data))
        except queue.Full:
            # Drop the oldest queued chunk to make room; capture continues.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(bytes(data))
            except (queue.Empty, queue.Full):
                pass

    def _on_disconnect(self, _client: Any) -> None:
        log.info("BLE NUS device disconnected")
        self._open = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        # No user-editable fields. The peripheral is chosen from the live device
        # list in the New Session dialog (Bluetooth toggle → scan → select →
        # connect), and its address is stored in the source params for reconnect.
        return ConfigSchema(fields=())
