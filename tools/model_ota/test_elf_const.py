#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Unit tests for reading a compile-time constant back out of an object.

Compiled with the host compiler rather than the target one: the point of reading the constant out
of the ELF instead of out of the compiler's assembly output is that it does not depend on a
toolchain's textual conventions, so exercising it on the host is both possible and the stronger
check. In particular a value of zero, which GCC emits as `.space 4` rather than as a `.word`, is
just four bytes of section data here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from elf_const import CONTRACT_HASH_SYMBOL, contract_hash_from_probe, read_u32

CC = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")

PROBE_C = """
#include <stdint.h>
struct runtime_thing { uint64_t a; void *b; uint8_t c; };
#define MIX(h, k) (((uint32_t)(h) ^ (uint32_t)(k)) * 16777619u)
const uint32_t %s = %s;
const uint16_t model_ota_too_small = 1;
"""


@unittest.skipUnless(CC, "no host C compiler available")
class ProbeReadTests(unittest.TestCase):
    def _probe(self, directory: Path, value: str) -> Path:
        source = directory / "probe.c"
        source.write_text(PROBE_C % (CONTRACT_HASH_SYMBOL, value), encoding="utf-8")
        obj = directory / "probe.o"
        subprocess.run(
            [CC, "-c", "-Os", "-fdata-sections", str(source), "-o", str(obj)],
            check=True,
            capture_output=True,
        )
        return obj

    def test_reads_a_folded_expression(self) -> None:
        """sizeof() is why the host cannot evaluate the macro itself."""
        with tempfile.TemporaryDirectory() as tmp:
            obj = self._probe(
                Path(tmp), "MIX(MIX(2166136261u, 12u), (uint32_t)sizeof(struct runtime_thing))"
            )
            expected = ((2166136261 ^ 12) * 16777619) & 0xFFFFFFFF
            expected = ((expected ^ 24) * 16777619) & 0xFFFFFFFF
            self.assertEqual(contract_hash_from_probe(obj), expected)

    def test_reads_a_zero_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(contract_hash_from_probe(self._probe(Path(tmp), "0u")), 0)

    def test_reads_a_high_bit_value(self) -> None:
        """The constant is unsigned; the assembler spells it as a negative decimal."""
        with tempfile.TemporaryDirectory() as tmp:
            obj = self._probe(Path(tmp), "0xbfd5cb73u")
            self.assertEqual(contract_hash_from_probe(obj), 0xBFD5CB73)

    def test_wrong_size_symbol_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            obj = self._probe(Path(tmp), "1u")
            with self.assertRaises(ValueError):
                read_u32(obj, "model_ota_too_small")

    def test_missing_symbol_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            obj = self._probe(Path(tmp), "1u")
            with self.assertRaises(ValueError):
                read_u32(obj, "model_ota_not_there")

    def test_missing_probe_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                contract_hash_from_probe(Path(tmp) / "absent.o")


if __name__ == "__main__":
    unittest.main()
