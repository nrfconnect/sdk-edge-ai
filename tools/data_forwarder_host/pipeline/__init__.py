# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Out-of-process source-acquisition pipeline.

This Qt-free layer owns source reading + COBS/CBOR decode in a separate child
process so the GUI process stays responsive regardless of backend load.
It sits between ``source``/``protocol`` and ``session`` and must never
import ``gui``.
"""

from __future__ import annotations
