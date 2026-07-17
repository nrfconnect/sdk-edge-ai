#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Unit tests for extract_elf_syms.py's PROVIDE()-script generation, mocking out the actual
`nm` subprocess call so these run without a real ELF file or toolchain.

Run with: python3 -m unittest test_extract_elf_syms.py -v
"""
import unittest
from unittest.mock import patch

from extract_elf_syms import build_provide_script, lookup_symbol


def fake_nm_output(symbols):
	"""Builds a synthetic `nm` stdout listing (address type name), one per line, matching
	the format lookup_symbol() parses."""
	lines = []
	for name, addr, sym_type in symbols:
		lines.append("%08x %s %s" % (addr, sym_type, name))
	return "\n".join(lines) + "\n"


class LookupSymbolTests(unittest.TestCase):
	def test_finds_matching_symbol(self):
		output = fake_nm_output([("nrf_axon_interlayer_buffer", 0x200003C0, "B")])
		with patch("extract_elf_syms.subprocess.check_output", return_value=output):
			addr = lookup_symbol("nm", "zephyr.elf", "nrf_axon_interlayer_buffer")
		self.assertEqual(addr, 0x200003C0)

	def test_returns_none_when_missing(self):
		output = fake_nm_output([("some_other_symbol", 0x1000, "T")])
		with patch("extract_elf_syms.subprocess.check_output", return_value=output):
			addr = lookup_symbol("nm", "zephyr.elf", "nrf_axon_interlayer_buffer")
		self.assertIsNone(addr)

	def test_ignores_similarly_prefixed_symbol(self):
		"""A symbol name that is merely a prefix of another must not match - nm's `parts[-1]
		== symbol` check (exact match on the last whitespace-separated field) is what
		guarantees this, since line-splitting rules out any partial-match confusion."""
		output = fake_nm_output([("axon_model_kws_persistent_vars_extra", 0x2000_1000, "B")])
		with patch("extract_elf_syms.subprocess.check_output", return_value=output):
			addr = lookup_symbol("nm", "zephyr.elf", "axon_model_kws_persistent_vars")
		self.assertIsNone(addr)


class BuildProvideScriptTests(unittest.TestCase):
	def test_data_symbol_address_used_verbatim(self):
		output = fake_nm_output([("nrf_axon_interlayer_buffer", 0x200003C0, "B")])
		with patch("extract_elf_syms.subprocess.check_output", return_value=output):
			script, missing = build_provide_script("nm", "zephyr.elf",
								  ["nrf_axon_interlayer_buffer"])
		self.assertEqual(missing, [])
		self.assertIn("PROVIDE(nrf_axon_interlayer_buffer = 0x200003C0);", script)

	def test_op_extension_gets_thumb_bit_set(self):
		output = fake_nm_output([("nrf_axon_nn_op_extension_relu", 0x0000_1234, "T")])
		with patch("extract_elf_syms.subprocess.check_output", return_value=output):
			script, missing = build_provide_script("nm", "zephyr.elf",
								  ["nrf_axon_nn_op_extension_relu"])
		self.assertEqual(missing, [])
		self.assertIn("PROVIDE(nrf_axon_nn_op_extension_relu = 0x1235);", script)

	def test_persistent_vars_address_used_verbatim(self):
		output = fake_nm_output([("axon_model_ww_persistent_vars", 0x2000_4000, "B")])
		with patch("extract_elf_syms.subprocess.check_output", return_value=output):
			script, missing = build_provide_script("nm", "zephyr.elf",
								  ["axon_model_ww_persistent_vars"])
		self.assertEqual(missing, [])
		self.assertIn("PROVIDE(axon_model_ww_persistent_vars = 0x20004000);", script)

	def test_missing_symbol_reported_not_raised(self):
		output = fake_nm_output([])
		with patch("extract_elf_syms.subprocess.check_output", return_value=output):
			script, missing = build_provide_script("nm", "zephyr.elf",
								  ["nrf_axon_interlayer_buffer"])
		self.assertEqual(missing, ["nrf_axon_interlayer_buffer"])
		self.assertNotIn("PROVIDE", script)

	def test_multiple_symbols_mixed_found_and_missing(self):
		output = fake_nm_output([("nrf_axon_interlayer_buffer", 0x2000_0000, "B")])
		with patch("extract_elf_syms.subprocess.check_output", return_value=output):
			script, missing = build_provide_script(
				"nm", "zephyr.elf",
				["nrf_axon_interlayer_buffer", "axon_model_kws_persistent_vars"])
		self.assertEqual(missing, ["axon_model_kws_persistent_vars"])
		self.assertIn("PROVIDE(nrf_axon_interlayer_buffer = 0x20000000);", script)


if __name__ == "__main__":
	unittest.main()
