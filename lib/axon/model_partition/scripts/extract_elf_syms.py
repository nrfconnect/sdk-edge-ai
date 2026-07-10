#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Extract symbol addresses from an ELF file and emit a C header."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def lookup_symbol(nm: str, elf: Path, symbol: str) -> str | None:
    output = subprocess.check_output([nm, str(elf)], text=True, errors="replace")

    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == symbol:
            return parts[0]

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nm", required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--symbols", type=Path, required=True,
                        help="Text file with one symbol name per line")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    symbols = [
        line.strip()
        for line in args.symbols.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if "nrf_axon_interlayer_buffer" not in symbols:
        symbols.insert(0, "nrf_axon_interlayer_buffer")

    lines = [
        "/* Auto-generated Axon model partition symbol addresses. */",
        "#ifndef NRF_AXON_MODEL_PARTITION_SYMS_H_",
        "#define NRF_AXON_MODEL_PARTITION_SYMS_H_",
        "",
    ]

    missing: list[str] = []

    for symbol in symbols:
        addr = lookup_symbol(args.nm, args.elf, symbol)
        if addr is None:
            missing.append(symbol)
            continue

        lines.append(f"#define AXON_SYM_{symbol} 0x{addr}")

    if missing:
        print(f"symbols not found in {args.elf}: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    lines.extend([
        "",
        "#define AXON_INTERLAYER_BUFFER_ADDR AXON_SYM_nrf_axon_interlayer_buffer",
        "",
        "#endif /* NRF_AXON_MODEL_PARTITION_SYMS_H_ */",
        "",
    ])

    args.output.write_text("\n".join(lines), encoding="ascii")


if __name__ == "__main__":
    main()
