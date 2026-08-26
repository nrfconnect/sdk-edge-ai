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
from model_contract import MODEL_IMAGE_FORMAT_VERSION


class LayoutValidationTests(unittest.TestCase):
    BASE = 0x100000
    MODEL_SIZE = 16
    NAME = b"test\0"

    def _params_block(
        self,
        num: int = 2,
        elem_size: int = 4,
        p_min: int | None = None,
        p_max: int | None = None,
        p_args: int = 0,
        decode: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> bytes:
        """Parameter block whose scaling arrays sit in the fixture's payload region."""
        if p_min is None:
            p_min = self.BASE + validator.HEADER_SIZE
        if p_max is None:
            p_max = p_min + num * elem_size
        return struct.pack(
            validator.PARAMS_FMT, p_min, p_max, p_args, *decode, num, elem_size, 0
        )

    def _files(self, directory: Path, params: bytes | None = None) -> tuple[Path, Path, Path]:
        model_off = validator.HEADER_SIZE
        name_off = model_off + self.MODEL_SIZE
        image_size = name_off + len(self.NAME)
        name_ptr = self.BASE + name_off
        model_ptr = self.BASE + model_off
        header = struct.pack(
            validator.HEADER_FMT,
            validator.MAGIC,
            MODEL_IMAGE_FORMAT_VERSION,
            validator.PARAMS_AXON,
            0,
            image_size,
            0x10000,
            0xAABBCCDD,
            0,
            name_ptr,
            struct.pack(validator.BACKEND_AXON_FMT, model_ptr, 4, 0, 0, 0),
            b"\0" * validator.PARAMS_SIZE if params is None else params,
        )
        data = bytearray(header + b"\0" * self.MODEL_SIZE + self.NAME)
        struct.pack_into("<I", data, validator.CRC32_OFFSET, zlib.crc32(data) & 0xFFFFFFFF)

        elf = directory / "image.elf"
        binary = directory / "image.bin"
        defs = directory / "model_image.h"
        elf.write_bytes(b"ELF fixture is mocked")
        binary.write_bytes(data)
        defs.write_text(
            "#define MODEL_IMAGE_FORMAT_VERSION %d\n" % MODEL_IMAGE_FORMAT_VERSION,
            encoding="ascii",
        )
        return elf, binary, defs

    def _run(self, elf: Path, binary: Path, defs: Path, partition_size: str = "0x1000") -> int:
        return validator.main(
            [
                "--elf",
                str(elf),
                "--bin",
                str(binary),
                "--partition-addr",
                hex(self.BASE),
                "--partition-size",
                partition_size,
                "--defs-header",
                str(defs),
                "--params-type",
                str(validator.PARAMS_AXON),
                "--model-symbol",
                "model_test",
            ]
        )

    def _symbol(self, name: str) -> ElfSymbol | None:
        image_size = validator.HEADER_SIZE + self.MODEL_SIZE + len(self.NAME)
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
            self.assertEqual(self._run(elf, binary, defs), 0)

    def test_partition_overflow_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            validator, "lookup_symbol", side_effect=lambda _elf, name: self._symbol(name)
        ):
            elf, binary, defs = self._files(Path(tmp))
            with self.assertRaises(SystemExit):
                self._run(elf, binary, defs, partition_size=str(validator.HEADER_SIZE))

    def test_params_block_in_image_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            validator, "lookup_symbol", side_effect=lambda _elf, name: self._symbol(name)
        ):
            elf, binary, defs = self._files(Path(tmp), params=self._params_block())
            self.assertEqual(self._run(elf, binary, defs), 0)

    def test_scale_pointer_outside_image_fails(self) -> None:
        outside = self._params_block(p_min=self.BASE - 8, p_max=self.BASE - 4)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            validator, "lookup_symbol", side_effect=lambda _elf, name: self._symbol(name)
        ):
            elf, binary, defs = self._files(Path(tmp), params=outside)
            with self.assertRaises(SystemExit):
                self._run(elf, binary, defs)

    def test_no_params_with_payload_fails(self) -> None:
        """scale_num 0 says the image carries nothing, so the block must be empty."""
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            validator, "lookup_symbol", side_effect=lambda _elf, name: self._symbol(name)
        ):
            elf, binary, defs = self._files(Path(tmp), params=self._params_block(num=0))
            with self.assertRaises(SystemExit):
                self._run(elf, binary, defs)

    def test_decoded_output_pointer_in_image_ok(self) -> None:
        """An Axon-backed Lab solution bakes decode meta pointers just like a Neuton one."""
        baked = self._params_block(decode=(0, self.BASE + validator.HEADER_SIZE, 0, 0))
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            validator, "lookup_symbol", side_effect=lambda _elf, name: self._symbol(name)
        ):
            elf, binary, defs = self._files(Path(tmp), params=baked)
            self.assertEqual(self._run(elf, binary, defs), 0)

    def test_decoded_output_pointer_outside_image_fails(self) -> None:
        stray = self._params_block(decode=(0, self.BASE - 4, 0, 0))
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            validator, "lookup_symbol", side_effect=lambda _elf, name: self._symbol(name)
        ):
            elf, binary, defs = self._files(Path(tmp), params=stray)
            with self.assertRaises(SystemExit):
                self._run(elf, binary, defs)


if __name__ == "__main__":
    unittest.main()
