# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Cross-platform helpers for opening files and folders.

These are GUI-free so they live in the layered ``utils`` package and stay
isolated. The command builders (:func:`open_command`, :func:`reveal_command`)
are pure functions returning the ``argv`` to launch; the ``open_*`` wrappers run
them with a best-effort fallback chain.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Linux launchers tried in order: a desktop opener first, then common editors as
# a last resort for headless/minimal environments.
_LINUX_OPENERS = ("xdg-open", "gio", "code", "gedit", "nano")


def _linux_opener(target: str) -> list[str] | None:
    for tool in _LINUX_OPENERS:
        exe = shutil.which(tool)
        if exe is None:
            continue
        return [exe, "open", target] if tool == "gio" else [exe, target]
    return None


def open_command(path: str) -> list[str] | None:
    """Return the ``argv`` to open *path* with the OS default handler.

    Returns ``None`` when the platform opens files via a non-``subprocess``
    mechanism (Windows ``os.startfile``) or no launcher is available.
    """
    if sys.platform.startswith("darwin"):
        return ["open", path]
    if os.name == "nt":
        return None
    return _linux_opener(path)


def reveal_command(path: str) -> list[str] | None:
    """Return the ``argv`` to open the folder containing *path*."""
    folder = str(Path(path).resolve().parent)
    if sys.platform.startswith("darwin"):
        return ["open", folder]
    if os.name == "nt":
        return None
    return _linux_opener(folder)


def _run(argv: list[str]) -> bool:
    try:
        subprocess.Popen(argv)  # noqa: S603
        return True
    except Exception:
        log.exception("failed to launch %r", argv)
        return False


def open_file(path: str) -> bool:
    """Open *path* with the system default application; ``True`` on success."""
    if os.name == "nt":
        try:
            os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606
            return True
        except Exception:
            log.exception("os.startfile failed for %s", path)
            return False
    argv = open_command(path)
    return _run(argv) if argv else False


def open_containing_folder(path: str) -> bool:
    """Open the folder containing *path*; ``True`` on success."""
    folder = str(Path(path).resolve().parent)
    if os.name == "nt":
        try:
            os.startfile(folder)  # type: ignore[attr-defined]  # noqa: S606
            return True
        except Exception:
            log.exception("os.startfile failed for %s", folder)
            return False
    argv = reveal_command(path)
    return _run(argv) if argv else False
