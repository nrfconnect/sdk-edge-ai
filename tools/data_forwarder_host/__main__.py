# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Entry point for ``python -m data_forwarder_host``."""

from __future__ import annotations

import sys


def _run() -> int:
    # Repair sys.path so the ``platform/`` subpackage cannot shadow stdlib when
    # the package dir is the CWD (e.g. ``python -m`` run from inside it).
    from data_forwarder_host.utils.syspath import repair_launcher_path

    repair_launcher_path()

    from data_forwarder_host import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_run())
