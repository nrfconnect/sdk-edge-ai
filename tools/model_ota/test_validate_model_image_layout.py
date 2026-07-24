#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Unit tests for model image layout validation."""

from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import validate_model_image_layout as validator
from axon_elf import ElfSymbol


class LayoutValidationTests(unittest.TestCase):
    BASE = 0x100000
    MODEL_SIZE = 16

    def _files(self, directory: Path) -> tuple[Path, Path, Path]:
        image_size = validator.HEADER_SIZE + self.MODEL_SIZE
        header = struct.pack(
            validator.HEADER_FMT,
            validator.MAGIC,
            3,
            validator.PARAMS_AXON,
            0,
            image_size,
            0,
            self.BASE + validator.HEADER_SIZE,
            0,
            b"test\0" + b"\0" * 11,
            0x10000,
            4,
        )
        data = bytearray(header + b"\0" * self.MODEL_SIZE)
        struct.pack_into("<I", data, 12, zlib.crc32(data) & 0xFFFFFFFF)

        elf = directory / "image.elf"
        binary = directory / "image.bin"
        defs = directory / "model_image.h"
        elf.write_bytes(b"ELF fixture is mocked")
        binary.write_bytes(data)
        defs.write_text("#define MODEL_IMAGE_FORMAT_VERSION 3\n", encoding="ascii")
        return elf, binary, defs

    def _symbol(self, name: str) -> ElfSymbol | None:
        image_size = validator.HEADER_SIZE + self.MODEL_SIZE
        entries = {
            "__model_image_start": ElfSymbol(name, self.BASE, 0, "GLOBAL", "NOTYPE", 1),
            "__model_image_end": ElfSymbol(
                name, self.BASE + image_size, 0, "GLOBAL", "NOTYPE", 1
            ),
            "model_image_hdr": ElfSymbol(
                name, self.BASE, validator.HEADER_SIZE, "GLOBAL", "OBJECT", 1
            ),
            "model_test": ElfSymbol(
                name,
                self.BASE + validator.HEADER_SIZE,
                self.MODEL_SIZE,
                "GLOBAL",
                "OBJECT",
                1,
            ),
        }
        return entries.get(name)

    def test_valid_axon_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            validator, "lookup_symbol", side_effect=lambda _elf, name: self._symbol(name)
        ):
            elf, binary, defs = self._files(Path(tmp))
            result = validator.main(
                [
                    "--elf",
                    str(elf),
                    "--bin",
                    str(binary),
                    "--partition-addr",
                    hex(self.BASE),
                    "--partition-size",
                    "0x1000",
                    "--defs-header",
                    str(defs),
                    "--params-type",
                    str(validator.PARAMS_AXON),
                    "--model-symbol",
                    "model_test",
                ]
            )
            self.assertEqual(result, 0)

    def test_partition_overflow_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            validator, "lookup_symbol", side_effect=lambda _elf, name: self._symbol(name)
        ):
            elf, binary, defs = self._files(Path(tmp))
            with self.assertRaises(SystemExit):
                validator.main(
                    [
                        "--elf",
                        str(elf),
                        "--bin",
                        str(binary),
                        "--partition-addr",
                        hex(self.BASE),
                        "--partition-size",
                        str(validator.HEADER_SIZE),
                        "--defs-header",
                        str(defs),
                        "--params-type",
                        str(validator.PARAMS_AXON),
                        "--model-symbol",
                        "model_test",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
