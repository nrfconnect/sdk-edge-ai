#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Generate hello_axon model partition image from the compiler header."""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

MAGIC = 0x4E4F5841
VERSION = 1
MODEL_CONST_SIZE = 420
FIELDS = (
    "l00_weights",
    "l00_biasp",
    "l01_weights",
    "l01_biasp",
    "l02_weights",
    "l02_biasp",
)


def parse_model_const(header_text: str) -> bytes:
    values: list[int] = []

    for field in FIELDS:
        match = re.search(rf"\.{field} = \{{([^}}]*)\}}", header_text)
        if match is None:
            raise ValueError(f"Could not find field {field}")

        nums = [int(item.strip()) for item in match.group(1).split(",") if item.strip()]

        if field.endswith("weights"):
            values.extend([(num + 256) % 256 for num in nums])
            continue

        for num in nums:
            values.extend(struct.pack("<i", num))

    if len(values) != MODEL_CONST_SIZE:
        raise ValueError(f"Unexpected model const size: {len(values)}")

    return bytes(values)


def build_image(header_text: str) -> bytes:
    model_const = parse_model_const(header_text)
    header = struct.pack("<4I", MAGIC, VERSION, MODEL_CONST_SIZE, 0)
    return header + model_const


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image = build_image(args.header.read_text())
    args.output.write_bytes(image)


if __name__ == "__main__":
    main()
