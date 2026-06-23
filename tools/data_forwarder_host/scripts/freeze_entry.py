# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Entry point used when building the standalone (frozen) binary.

This is the script PyInstaller turns into the single-file executable. It is
deliberately separate from ``main.py`` / ``__main__.py`` because a frozen build
has one extra, non-negotiable requirement:

``multiprocessing.freeze_support()`` **must** be the very first thing that runs.

The application launches its acquisition backend in a child process using the
``spawn`` start method (see ``pipeline/process_host.py``). With ``spawn`` the
child is started by *re-executing this same binary*. ``freeze_support()`` detects
that re-execution, runs the child worker, and exits — *without* falling through
to ``main()``. Omitting it makes every spawned child start a brand-new GUI,
producing an endless cascade of windows (the classic PyInstaller + multiprocessing
bug).

The source-tree ``sys.path`` repair done by the other launchers is unnecessary
here: in a frozen build the package is imported from the embedded archive, not
from the package directory, so the ``platform`` subpackage cannot shadow stdlib.
"""

from __future__ import annotations

import multiprocessing


def _run() -> int:
    from data_forwarder_host.app import main

    return main()


if __name__ == "__main__":
    # MUST be first — handles the re-executed `spawn` child process.
    multiprocessing.freeze_support()
    raise SystemExit(_run())
