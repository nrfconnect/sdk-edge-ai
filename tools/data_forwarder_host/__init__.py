# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Data Forwarder Host — receives, visualises and exports sensor data from an nRF device."""

__version__ = "0.1.0"


def main() -> int:
    """GUI entry point (re-exported from :mod:`data_forwarder_host.app`)."""
    from data_forwarder_host.app import main as _main

    return _main()


__all__ = ["__version__", "main"]
