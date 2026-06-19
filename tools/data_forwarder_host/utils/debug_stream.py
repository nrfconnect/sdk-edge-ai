# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Rate-limited diagnostics stream for root-causing runtime slowdowns.

Opt-in (off by default) so it never costs anything in normal use. When enabled
via the ``DFH_DEBUG_STREAM`` environment variable, a per-session timer samples a
small set of runtime metrics (recorder RAM growth, live-buffer sizes, GC
generation counts, message rate) at a low cadence and writes one compact line
per tick to the ``data_forwarder_host.debug`` logger.

Two guards keep the stream cheap to *produce* and cheap to *analyse* (the user
asked to cap the message count): a minimum interval between emissions and a hard
ceiling on the total number of lines emitted, after which the stream goes quiet.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QTimer

log = logging.getLogger("data_forwarder_host.debug")

ENABLED_ENV = "DFH_DEBUG_STREAM"

# Defaults chosen so the stream is informative but bounded: a sample every few
# seconds, and at most a few hundred lines per session.
DEFAULT_INTERVAL_S = 3.0
DEFAULT_MAX_MESSAGES = 200


def debug_stream_enabled() -> bool:
    """True when the debug stream is switched on via the environment."""
    return os.environ.get(ENABLED_ENV, "").strip().lower() not in ("", "0", "false", "no")


def format_metrics(metrics: dict[str, Any]) -> str:
    """Render a metrics mapping as a compact, stable ``k=v`` line."""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:g}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


class EmissionLimiter:
    """Pure throttle: cap emissions by both count and (caller-supplied) ticks.

    Kept Qt-free so the gating policy is isolated. The owner calls
    :meth:`allow` once per timer tick; it returns ``False`` once the message
    ceiling is reached so the stream stops adding noise (and cost).
    """

    def __init__(self, max_messages: int = DEFAULT_MAX_MESSAGES) -> None:
        self._max = max(0, int(max_messages))
        self._emitted = 0

    @property
    def emitted(self) -> int:
        return self._emitted

    @property
    def exhausted(self) -> bool:
        return self._emitted >= self._max

    def allow(self) -> bool:
        """Reserve one emission slot; ``False`` once the ceiling is hit."""
        if self._emitted >= self._max:
            return False
        self._emitted += 1
        return True


class DebugStream(QObject):
    """Timer-driven, rate-limited runtime-metrics logger (opt-in)."""

    def __init__(
        self,
        provider: Callable[[], dict[str, Any]],
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._provider = provider
        self._limiter = EmissionLimiter(max_messages)
        self._timer = QTimer(self)
        self._timer.setInterval(int(max(0.1, interval_s) * 1000))
        self._timer.timeout.connect(self._on_tick)

    def start(self) -> None:
        self._timer.start()
        try:
            from data_forwarder_host.utils.paths import log_file

            where = str(log_file())
        except Exception:  # pragma: no cover - never block diagnostics
            where = "<application log>"
        log.info(
            "debug stream started (interval=%dms); writing metrics to %s",
            self._timer.interval(),
            where,
        )

    def stop(self) -> None:
        self._timer.stop()

    def _on_tick(self) -> None:
        if not self._limiter.allow():
            # Ceiling reached: go quiet so analysis stays tractable.
            self._timer.stop()
            log.info("debug stream reached its message ceiling; stopping")
            return
        try:
            metrics = self._provider()
        except Exception:  # diagnostics must never crash the app
            log.exception("debug stream provider failed")
            return
        log.info("%s", format_metrics(metrics))
