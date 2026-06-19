# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Tiny app-wide publish/subscribe bus (thread-safe, callback-based)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

Callback = Callable[..., None]


class EventBus:
    """A minimal topic-based pub/sub bus.

    Subscribers are invoked synchronously from the publishing thread; cross-
    thread delivery is the responsibility of the subscriber (use Qt signals or
    similar). Useful for app-wide notifications that do not warrant a dedicated
    Qt signal/slot wiring (e.g. settings changes).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: dict[str, list[Callback]] = {}

    def subscribe(self, topic: str, callback: Callback) -> Callable[[], None]:
        with self._lock:
            self._subs.setdefault(topic, []).append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                lst = self._subs.get(topic, [])
                if callback in lst:
                    lst.remove(callback)

        return _unsubscribe

    def publish(self, topic: str, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            subscribers = list(self._subs.get(topic, ()))
        for cb in subscribers:
            try:
                cb(*args, **kwargs)
            except Exception:  # pragma: no cover — never crash publisher
                import logging
                logging.getLogger(__name__).exception("subscriber failed on topic %s", topic)
