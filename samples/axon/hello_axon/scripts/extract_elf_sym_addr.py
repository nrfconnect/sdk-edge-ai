#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Extract a symbol address from an ELF file using the toolchain nm."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nm", required=True)
    parser.add_argument("--elf", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = subprocess.check_output([args.nm, args.elf], text=True, errors="replace")

    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == args.symbol:
            args.output.write_text(parts[0], encoding="ascii")
            return

    print(f"symbol {args.symbol} not found in {args.elf}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
