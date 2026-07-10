#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Print a linker-style memory usage table for an Axon model partition image.

Mirrors Zephyr's "Memory region" build output but for the devicetree partition
size vs. the linked model image binary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def format_size(size: int) -> str:
    if size >= 1024 and size % 1024 == 0:
        return f"{size // 1024} KB"

    return f"{size} B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Partition label shown in the table")
    parser.add_argument("--bin", type=Path, required=True, help="Model partition binary")
    parser.add_argument("--region-size", type=lambda value: int(value, 0), required=True)
    args = parser.parse_args()

    if not args.bin.is_file():
        print(f"model partition binary not found: {args.bin}", file=sys.stderr)
        sys.exit(1)

    used = args.bin.stat().st_size
    region = args.region_size

    if region == 0:
        print("model partition region size is zero", file=sys.stderr)
        sys.exit(1)

    percent = (used * 100.0) / region

    print("Partition region      Used Size  Region Size  %age Used")
    print(f"{args.label:>21}: {format_size(used):>10} {format_size(region):>10} {percent:9.2f}%")

    if used > region:
        print(
            f"warning: model image ({used} B) exceeds partition size ({region} B)",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
