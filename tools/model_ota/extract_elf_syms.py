#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Extract symbol addresses from zephyr.elf and emit PROVIDE() linker fragments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

OP_EXTENSION_PREFIX = "nrf_axon_nn_op_extension_"


def lookup_symbol(nm: str, elf: Path, symbol: str) -> int | None:
    output = subprocess.check_output([nm, str(elf)], text=True, errors="replace")

    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == symbol:
            return int(parts[0], 16)

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nm", required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--symbols", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Linker script with PROVIDE() entries")
    args = parser.parse_args()

    symbols = [
        line.strip()
        for line in args.symbols.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if "nrf_axon_interlayer_buffer" not in symbols:
        symbols.insert(0, "nrf_axon_interlayer_buffer")

    missing: list[str] = []
    linker_lines: list[str] = []

    for symbol in symbols:
        addr = lookup_symbol(args.nm, args.elf, symbol)
        if addr is None:
            missing.append(symbol)
            continue

        if symbol.startswith(OP_EXTENSION_PREFIX):
            addr |= 1
        linker_lines.append(f"PROVIDE({symbol} = 0x{addr:X});")

    if missing:
        print(f"symbols not found in {args.elf}: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    args.output.write_text("\n".join(linker_lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
