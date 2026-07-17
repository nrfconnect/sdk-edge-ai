#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Host-side unit tests for package_model_axon.py's model-shape validation and header format.

validate_shape() is exercised directly against synthetic nrf_axon_nn_compiled_model_s instances
built from raw bytes via axon_struct_layout.CompiledModel - no ELF file needed. Everything else
in this module (ElfSymbols, build_package()) reads a real model stub ELF, produced only by an
actual `west build` (see lib/model_ota/cmake/nrf_axon_model_stub.cmake); that integration path is
exercised by building samples/axon/hello_axon and samples/nrf_edgeai/regression, not by a host
unit test here.

Run with: python3 -m unittest test_package_model_axon.py -v
"""
import struct
import unittest
from io import StringIO
from unittest.mock import patch

from axon_struct_layout import CompiledModel
from model_partition_layout import check_package_fits, format_size, report_package_usage
from package_model_axon import HEADER_FMT, HEADER_LEN, MAGIC, validate_shape


def make_model(input_cnt=1, external=(True,), labels=0, persistent_vars_count=0,
	       packed_output_buf=0, extra_output_cnt=0, extra_outputs=0):
	"""Builds a zeroed CompiledModel with just the fields validate_shape() cares about
	populated, mirroring what a real compiled model looks like closely enough for this pure
	function."""
	import ctypes
	buf = bytearray(ctypes.sizeof(CompiledModel))
	model = CompiledModel.from_buffer(buf)

	model.labels = labels
	model.input_cnt = input_cnt
	for i in range(input_cnt):
		model.inputs[i].is_external = 1 if external[i] else 0
	model.packed_output_buf = packed_output_buf
	model.persistent_vars.count = persistent_vars_count
	model.extra_output_cnt = extra_output_cnt
	model.extra_outputs = extra_outputs
	return model


class ValidateShapeTests(unittest.TestCase):
	def test_hello_axon_shape_is_accepted(self):
		model = make_model()
		validate_shape(model, "hello_axon")  # must not raise

	def test_two_external_inputs_accepted(self):
		model = make_model(input_cnt=2, external=(True, True))
		validate_shape(model, "multi_input")

	def test_labels_accepted(self):
		"""Unlike the reference-build era, labels no longer need any host-side relocation
		(the model stub's own link already resolves them), so they are accepted
		unconditionally here."""
		model = make_model(labels=0x0800_1234)
		validate_shape(model, "labeled")  # must not raise

	def test_persistent_vars_accepted(self):
		"""Ditto for persistent_vars: the second-pass link resolves its RAM array's real
		address before packaging ever runs, so there is nothing left for this tool to
		reject."""
		model = make_model(persistent_vars_count=3)
		validate_shape(model, "streaming")  # must not raise

	def test_packed_output_buf_rejected(self):
		model = make_model(packed_output_buf=0x2000_2000)
		with self.assertRaisesRegex(ValueError, "packed_output_buf"):
			validate_shape(model, "packed")

	def test_internal_input_accepted(self):
		"""Ditto: models with an input fed by another layer or a persistent variable (e.g.
		streaming/recurrent models) rather than directly by the application are common and
		nothing here needs to relocate, so is_external is no longer checked at all."""
		model = make_model(input_cnt=2, external=(True, False))
		validate_shape(model, "internal_input")  # must not raise

	def test_zero_input_cnt_rejected(self):
		model = make_model(input_cnt=0, external=())
		with self.assertRaisesRegex(ValueError, "implausible"):
			validate_shape(model, "no_inputs")

	def test_extra_outputs_null_but_cnt_nonzero_rejected(self):
		model = make_model(extra_output_cnt=2, extra_outputs=0)
		with self.assertRaisesRegex(ValueError, "NULL extra_outputs"):
			validate_shape(model, "extra")

	def test_extra_outputs_set_but_cnt_zero_rejected(self):
		model = make_model(extra_output_cnt=0, extra_outputs=0x0800_5000)
		with self.assertRaisesRegex(ValueError, "non-NULL extra_outputs"):
			validate_shape(model, "extra")

	def test_extra_outputs_consistent_accepted(self):
		model = make_model(extra_output_cnt=2, extra_outputs=0x0800_5000)
		validate_shape(model, "extra")  # must not raise


class HeaderFormatTests(unittest.TestCase):
	def test_header_len_matches_format(self):
		self.assertEqual(HEADER_LEN, struct.calcsize(HEADER_FMT))

	def test_round_trip_pack_unpack(self):
		name = b"hello_axon\x00\x00\x00\x00\x00\x00"
		packed = struct.pack(HEADER_FMT, MAGIC, 6, 1, 0, name, 0x0001_0000, 848, 708, 140,
				      0x0010_2030, 0xDEAD_BEEF)
		self.assertEqual(len(packed), HEADER_LEN)

		(magic, format_version, model_type, reserved0, unpacked_name, model_version,
		 payload_size, struct_offset, struct_size, package_base, crc32) = struct.unpack(
			HEADER_FMT, packed)

		self.assertEqual(magic, MAGIC)
		self.assertEqual(format_version, 6)
		self.assertEqual(model_type, 1)
		self.assertEqual(reserved0, 0)
		self.assertEqual(unpacked_name, name)
		self.assertEqual(model_version, 0x0001_0000)
		self.assertEqual(payload_size, 848)
		self.assertEqual(struct_offset, 708)
		self.assertEqual(struct_size, 140)
		self.assertEqual(package_base, 0x0010_2030)
		self.assertEqual(crc32, 0xDEAD_BEEF)


class FormatSizeTests(unittest.TestCase):
	def test_exact_kib_shown_as_kb(self):
		self.assertEqual(format_size(968 * 1024), "968 KB")

	def test_non_kib_multiple_shown_as_bytes(self):
		self.assertEqual(format_size(3604), "3604 B")

	def test_small_size_shown_as_bytes(self):
		self.assertEqual(format_size(0), "0 B")


class ReportPackageUsageTests(unittest.TestCase):
	def test_prints_utilization_percentage(self):
		with patch("sys.stdout", new_callable=StringIO) as out:
			report_package_usage(48 * 1024, 968 * 1024, label="model_storage")
		output = out.getvalue()
		self.assertIn("model_storage", output)
		self.assertIn("48 KB", output)
		self.assertIn("968 KB", output)
		self.assertIn("4.96%", output)

	def test_does_not_raise_when_oversized(self):
		# report_package_usage is purely informational; check_package_fits() (exercised
		# below) is what actually enforces the capacity limit.
		with patch("sys.stdout", new_callable=StringIO):
			report_package_usage(2000, 1000)

	def test_check_package_fits_still_raises_on_overflow(self):
		with self.assertRaisesRegex(ValueError, "exceeds model_storage partition capacity"):
			check_package_fits(2000, 1000, "some_model.bin")


if __name__ == "__main__":
	unittest.main()
