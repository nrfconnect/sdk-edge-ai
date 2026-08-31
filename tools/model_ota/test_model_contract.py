#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Unit tests for the string-hashing helpers left on the host side.

The contract hash itself has no test here by design: it is folded by the compiler and read back
out of a probe object (see test_elf_const.py), so there is no host implementation to pin down.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import model_contract
from model_contract import config_define, solution_id_hash, symbol_name_hash

SOLUTION_ID = "90360"


class StringHashTests(unittest.TestCase):
    def test_the_algorithm_is_pinned(self) -> None:
        """Both hashes are on-flash ABI: firmware bakes them, images have to agree.

        Changing the algorithm invalidates every released image, so it has to be a deliberate
        edit to this value rather than a side effect of touching the helper.
        """
        self.assertEqual(solution_id_hash(SOLUTION_ID), 0x2D8CBC77)
        self.assertEqual(symbol_name_hash("nrf_axon_interlayer_buffer"), 0x8E1329CF)

    def test_hashes_stay_in_range(self) -> None:
        self.assertLess(solution_id_hash(SOLUTION_ID), 1 << 32)

    def test_distinct_solutions_hash_apart(self) -> None:
        """The value feeds the solution contract, so a wrong slot must not look compatible."""
        self.assertNotEqual(solution_id_hash(SOLUTION_ID), solution_id_hash("90449"))

    def test_a_trailing_byte_changes_most_of_the_hash(self) -> None:
        """Binding rows are matched on the hash alone, and Axon symbols share long prefixes.

        A last-byte difference must reach the whole word, which is what bare FNV-1a would not
        give: with no finalizer its final step only propagates carries upwards.
        """
        a = symbol_name_hash("nrf_axon_nn_op_extension_a")
        b = symbol_name_hash("nrf_axon_nn_op_extension_b")
        self.assertGreater(bin(a ^ b).count("1"), 8)

    def test_symbol_and_solution_hash_are_one_function(self) -> None:
        """No domain separation: the two values never share a lookup table."""
        self.assertEqual(symbol_name_hash(SOLUTION_ID), solution_id_hash(SOLUTION_ID))


class ConfigDefineTests(unittest.TestCase):
    def _header(self, directory: Path, text: str) -> Path:
        path = directory / "axon_config.h"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_decimal_and_hex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            header = self._header(
                Path(tmp),
                "#define MODEL_OTA_AXON_PERSISTENT_VARS_CAP 12\n"
                "#define MODEL_OTA_AXON_PACKED_OUTPUT_BYTES 0x40u\n",
            )
            self.assertEqual(config_define(header, "MODEL_OTA_AXON_PERSISTENT_VARS_CAP"), 12)
            self.assertEqual(config_define(header, "MODEL_OTA_AXON_PACKED_OUTPUT_BYTES"), 0x40)

    def test_missing_define_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            header = self._header(Path(tmp), "#define SOMETHING_ELSE 1\n")
            self.assertIsNone(config_define(header, "MODEL_OTA_AXON_PERSISTENT_VARS_CAP"))


class CommandLineTests(unittest.TestCase):
    """The CLI model_ota_solution_id_hash() in model_ota_common.cmake calls."""

    def _run(self, argv: list[str]) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(model_contract.main(argv), 0)
        return out.getvalue().strip()

    def test_prints_the_decimal_hash(self) -> None:
        self.assertEqual(
            self._run(["solution-id-hash", SOLUTION_ID]), str(solution_id_hash(SOLUTION_ID))
        )

    def test_an_id_is_an_argument_not_a_snippet(self) -> None:
        """CMake passes the ID through argv, so quoting cannot change what runs."""
        awkward = "90360'); import os; os.abort() #"
        self.assertEqual(self._run(["solution-id-hash", awkward]), str(solution_id_hash(awkward)))

    def test_non_ascii_id_is_refused(self) -> None:
        """A solution ID names C identifiers too, so it is ASCII by construction."""
        with self.assertRaises(SystemExit):
            self._run(["solution-id-hash", "sol\u00fction"])


if __name__ == "__main__":
    unittest.main()
