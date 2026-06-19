# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Keep the flat-layout package importable without shadowing stdlib modules.

This package directory contains a subpackage named ``platform`` (a layered
subpackage). When the application is launched from *inside* that directory
(``python main.py`` or ``python -m data_forwarder_host`` with the package dir as
CWD), Python places the package directory first on ``sys.path``; a subsequent
``import platform`` then resolves to this subpackage instead of the standard
library, breaking stdlib modules such as ``uuid``.

:func:`corrected_sys_path` returns a repaired copy of ``sys.path`` with the
package directory removed (so stdlib resolves normally) and its parent present
and first (so ``data_forwarder_host`` itself stays importable). The function is
pure; callers assign the result to ``sys.path``.
"""

from __future__ import annotations

import os
import sys


def corrected_sys_path(package_dir: str, path: list[str], *, cwd: str) -> list[str]:
    """Return a copy of ``path`` safe to use from inside ``package_dir``.

    Every entry whose absolute form equals ``package_dir`` is dropped (an empty
    ``""`` entry is interpreted as ``cwd``, matching CPython). The parent of
    ``package_dir`` is ensured to be present and first so the package remains
    importable. Idempotent.
    """
    pkg = os.path.abspath(package_dir)
    parent = os.path.dirname(pkg)
    cleaned = [p for p in path if os.path.abspath(p or cwd) != pkg]
    if not any(os.path.abspath(p or cwd) == parent for p in cleaned):
        cleaned.insert(0, parent)
    return cleaned


def repair_launcher_path() -> None:
    """Apply :func:`corrected_sys_path` to the live ``sys.path`` in place.

    Shared by the two in-tree launchers (``main.py`` and ``__main__.py``) so the
    single definition of *how* the correction is applied — the package directory
    (derived from this module's own location) and the ``cwd`` semantics — cannot
    drift between them. Impure by design: it mutates ``sys.path``.
    """
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:] = corrected_sys_path(package_dir, sys.path, cwd=os.getcwd())
