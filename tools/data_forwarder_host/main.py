# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Top-level launcher: ``python3 main.py``.

Equivalent to ``python -m data_forwarder_host`` and the ``data-forwarder-host``
console script; all three delegate to :func:`data_forwarder_host.app.main`.
"""

from __future__ import annotations

import os
import sys


def _run() -> int:
    package_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(package_dir)
    # Bootstrap: ensure the package is importable even without an install, then
    # repair sys.path so the ``platform/`` subpackage cannot shadow stdlib when
    # launching from inside the package directory.
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from data_forwarder_host.utils.syspath import repair_launcher_path

    repair_launcher_path()

    from data_forwarder_host import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_run())
