#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Validate Axon model partition image layout after linking."""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import sys
from pathlib import Path

MAGIC = 0x4E4F5841
HEADER_SIZE = struct.calcsize("<4I")


def lookup_symbol(nm: str, elf: Path, symbol: str) -> int | None:
    output = subprocess.check_output([nm, str(elf)], text=True, errors="replace")

    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == symbol:
            return int(parts[0], 16)

    return None


def parse_model_symbol(fixups_header: Path | None, model_symbol: str | None) -> str | None:
    if model_symbol:
        return model_symbol

    if fixups_header is None:
        return None

    text = fixups_header.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#define NRF_AXON_MODEL_IMAGE_MODEL_SYM "):
            return line.split()[-1]

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nm", required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--partition-addr", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--model-symbol", default=None)
    parser.add_argument("--fixups-header", type=Path, default=None)
    parser.add_argument("--defs-header", type=Path, default=None)
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args()

    if not args.elf.is_file():
        print(f"ELF not found: {args.elf}", file=sys.stderr)
        sys.exit(1)

    if not args.bin.is_file():
        print(f"binary not found: {args.bin}", file=sys.stderr)
        sys.exit(1)

    model_symbol = parse_model_symbol(args.fixups_header, args.model_symbol)

    expected_version = args.version
    if expected_version is None and args.defs_header is not None:
        match = re.search(
            r"#define\s+NRF_AXON_MODEL_PARTITION_VERSION\s+(\d+)",
            args.defs_header.read_text(encoding="utf-8"),
        )
        if match is not None:
            expected_version = int(match.group(1))

    if expected_version is None:
        print("expected partition version not provided", file=sys.stderr)
        sys.exit(1)

    start = lookup_symbol(args.nm, args.elf, "__axon_model_image_start")
    end = lookup_symbol(args.nm, args.elf, "__axon_model_image_end")
    hdr_sym = lookup_symbol(args.nm, args.elf, "nrf_axon_model_image_partition_hdr")

    if start is None or end is None:
        print("missing linker anchors __axon_model_image_start/__axon_model_image_end", file=sys.stderr)
        sys.exit(1)

    if start != args.partition_addr:
        print(
            f"partition base mismatch: linker start 0x{start:x} != 0x{args.partition_addr:x}",
            file=sys.stderr,
        )
        sys.exit(1)

    if hdr_sym is not None and hdr_sym != start:
        print(
            f"partition header not at image start: hdr 0x{hdr_sym:x}, start 0x{start:x}",
            file=sys.stderr,
        )
        sys.exit(1)

    image_size = end - start
    if image_size <= HEADER_SIZE:
        print(f"model image too small: {image_size} bytes", file=sys.stderr)
        sys.exit(1)

    header_bytes = args.bin.read_bytes()[:HEADER_SIZE]
    if len(header_bytes) != HEADER_SIZE:
        print("partition binary shorter than header", file=sys.stderr)
        sys.exit(1)

    magic, version, model_offset, image_size_hdr = struct.unpack("<4I", header_bytes)

    errors: list[str] = []

    if magic != MAGIC:
        errors.append(f"magic 0x{magic:08x} != 0x{MAGIC:08x}")

    if version != expected_version:
        errors.append(f"version {version} != expected {expected_version}")

    if model_offset < HEADER_SIZE:
        errors.append(f"model_offset {model_offset} < header size {HEADER_SIZE}")

    if image_size_hdr != image_size:
        errors.append(
            f"header image_size 0x{image_size_hdr:x} != linker extent 0x{image_size:x}"
        )

    model_addr = start + model_offset
    if model_addr < start or model_addr >= end:
        errors.append(
            f"model_offset 0x{model_offset:x} places descriptor outside image "
            f"[0x{start:x}, 0x{end:x})"
        )

    if model_symbol is not None:
        model_sym_addr = lookup_symbol(args.nm, args.elf, model_symbol)
        if model_sym_addr is None:
            errors.append(f"model symbol not found: {model_symbol}")
        elif model_sym_addr != model_addr:
            errors.append(
                f"model_offset points to 0x{model_addr:x}, symbol {model_symbol} at 0x{model_sym_addr:x}"
            )

    bin_size = args.bin.stat().st_size
    if bin_size != image_size:
        errors.append(f"binary size 0x{bin_size:x} != linker extent 0x{image_size:x}")

    if errors:
        for err in errors:
            print(f"layout validation failed: {err}", file=sys.stderr)
        sys.exit(1)

    print(
        f"model partition layout ok: size 0x{image_size:x}, "
        f"model_offset 0x{model_offset:x}"
        + (f", symbol {model_symbol}" if model_symbol else "")
    )


if __name__ == "__main__":
    main()
