# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Logging setup — rotating file + optional Qt log handler bridge."""

from __future__ import annotations

import logging
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from typing import Any

from data_forwarder_host.utils.paths import log_file

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 5

_qt_handler: logging.Handler | None = None

# Dedicated logger for user-initiated actions. Lines logged through it carry a
# clear "USER ACTION" marker so the user's workflow (what they clicked / chose)
# can be reconstructed from the console, file and in-app log — making any later
# error readable in the context of the steps that led to it.
_action_log = logging.getLogger("data_forwarder_host.action")


def log_user_action(message: str, *args: Any) -> None:
    """Record a user-initiated GUI action to the shared log.

    Emitted at INFO so it appears on the console / log file / in-app panel
    alongside the rest of the application log. ``args`` are ``%``-formatted
    lazily by the logging framework (do not pre-format the message).
    """
    _action_log.info("USER ACTION: " + message, *args)



class _CallbackHandler(logging.Handler):
    """Simple handler that forwards formatted records to a callback."""

    def __init__(self, callback: Callable[[logging.LogRecord, str], None]) -> None:
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._callback(record, msg)
        except Exception:  # pragma: no cover — never crash the logger
            self.handleError(record)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger. Idempotent."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    if not any(
        isinstance(h, RotatingFileHandler) and getattr(h, "_data_forwarder", False)
        for h in root.handlers
    ):
        fh = RotatingFileHandler(
            log_file(), maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        setattr(fh, "_data_forwarder", True)
        root.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
               for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(sh)


def install_qt_handler(callback: Callable[[logging.LogRecord, str], Any]) -> None:
    """Add a handler that forwards records to a GUI callback (LogPanel)."""
    global _qt_handler
    if _qt_handler is not None:
        return
    _qt_handler = _CallbackHandler(callback)
    _qt_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.getLogger().addHandler(_qt_handler)


def remove_qt_handler() -> None:
    global _qt_handler
    if _qt_handler is not None:
        logging.getLogger().removeHandler(_qt_handler)
        _qt_handler = None
