# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Version helper."""

from __future__ import annotations

from data_forwarder_host import __version__


def get_version() -> str:
    return __version__
