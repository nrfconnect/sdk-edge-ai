#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Extract symbol addresses from an ELF file and emit a companion linker script.

The model image link runs after the application ELF exists. Symbols such as
nrf_axon_interlayer_buffer must resolve to the same absolute addresses the
application link produced. We emit PROVIDE() entries consumed as a second -T
fragment when linking model_image_stub.o.

The C header output is intentionally empty; only the linker script carries data.
"""

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
    parser.add_argument("--linker-script", type=Path, required=False)
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

        if args.linker_script is not None:
            link_addr = int(addr, 16)
            if symbol.startswith("nrf_axon_nn_op_extension_"):
                # Thumb code pointers need bit 0 set in the stored address.
                link_addr |= 1
            linker_lines.append(f"PROVIDE({symbol} = 0x{link_addr:X});")

    if missing:
        print(f"symbols not found in {args.elf}: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    args.output.write_text(
        "\n".join([
            "/* Auto-generated companion header for model image link. */",
            "/* Symbol addresses are resolved via the companion linker script (PROVIDE). */",
            "#ifndef NRF_AXON_MODEL_PARTITION_SYMS_H_",
            "#define NRF_AXON_MODEL_PARTITION_SYMS_H_",
            "",
            "#endif /* NRF_AXON_MODEL_PARTITION_SYMS_H_ */",
            "",
        ]),
        encoding="ascii",
    )

    if args.linker_script is not None:
        args.linker_script.write_text("\n".join(linker_lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
