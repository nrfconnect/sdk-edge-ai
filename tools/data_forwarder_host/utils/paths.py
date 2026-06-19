# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Standard application paths (wraps :mod:`platformdirs`)."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import (
    user_cache_dir,
    user_config_dir,
    user_log_dir,
)

_APP_NAME = "data_forwarder"
_APP_AUTHOR = "Nordic Semiconductor"


def app_config_dir() -> Path:
    p = Path(user_config_dir(_APP_NAME, _APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def app_cache_dir() -> Path:
    p = Path(user_cache_dir(_APP_NAME, _APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def app_log_dir() -> Path:
    p = Path(user_log_dir(_APP_NAME, _APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def settings_file() -> Path:
    return app_config_dir() / "settings.json"


def default_recordings_dir() -> Path:
    """Return the default CSV output directory ``<package root>/recordings``.

    Used when a session's ``output_dir`` is blank. The directory is
    NOT created here; the caller creates it on demand at CSV-write time.
    """
    # paths.py lives at ``data_forwarder_host/utils/paths.py``; the package root
    # is two levels up.
    package_root = Path(__file__).resolve().parent.parent
    return package_root / "recordings"


def log_file() -> Path:
    return app_log_dir() / "data_forwarder_host.log"


def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def safe_join(base: Path, *parts: str) -> Path:
    """Join ``parts`` onto ``base``, refusing traversal outside ``base``."""
    candidate = (base.joinpath(*parts)).resolve()
    base_resolved = base.resolve()
    if os.path.commonpath([str(candidate), str(base_resolved)]) != str(base_resolved):
        raise ValueError(f"path {candidate} escapes base {base_resolved}")
    return candidate
