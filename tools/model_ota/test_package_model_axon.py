#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Host-side unit tests for package_model_axon.py's pointer classification/relocation logic.

These exercise validate_shape()/relocate_ram_pointers()/relocate_flash_pointers()/
relocate_cmd_buffer() directly against synthetic nrf_axon_nn_compiled_model_s instances built
from raw bytes via axon_struct_layout.CompiledModel - no ELF file needed. This is the only
regression coverage for model shapes (multi-input, extra_outputs, rejected shapes) that don't
exist as real models anywhere in this repo yet; see package_model_axon.py's module docstring.

Run with: python3 -m unittest test_package_model_axon.py -v
"""
import struct
import unittest
from io import StringIO
from unittest.mock import patch

from axon_struct_layout import CompiledModel
from model_partition_layout import check_package_fits, format_size, report_package_usage
from package_model_axon import (
	build_labels_section,
	check_no_op_extension_refs,
	num_output_classes,
	relocate_cmd_buffer,
	relocate_flash_pointers,
	relocate_ram_pointers,
	validate_shape,
)

INTERLAYER_ADDR = 0x2000_1000


def make_model(input_cnt=1, external=(True,), labels=0, persistent_vars_count=0,
	       packed_output_buf=0, extra_output_cnt=0, extra_outputs=0,
	       input_ptrs=None, output_ptr=None):
	"""Builds a zeroed CompiledModel with just the fields validate_shape()/relocate_*()
	care about populated, mirroring what a real compiled model looks like closely enough for
	these pure functions."""
	buf = bytearray(_sizeof())
	model = CompiledModel.from_buffer(buf)

	model.labels = labels
	model.input_cnt = input_cnt
	for i in range(input_cnt):
		model.inputs[i].is_external = 1 if external[i] else 0
		model.inputs[i].ptr = (input_ptrs[i] if input_ptrs else INTERLAYER_ADDR)
	model.output_ptr = output_ptr if output_ptr is not None else INTERLAYER_ADDR
	model.packed_output_buf = packed_output_buf
	model.persistent_vars.count = persistent_vars_count
	model.extra_output_cnt = extra_output_cnt
	model.extra_outputs = extra_outputs
	return model, buf


def _sizeof():
	import ctypes
	return ctypes.sizeof(CompiledModel)


class ValidateShapeTests(unittest.TestCase):
	def test_hello_axon_shape_is_accepted(self):
		model, _ = make_model()
		validate_shape(model, "hello_axon")  # must not raise

	def test_two_external_inputs_accepted(self):
		model, _ = make_model(input_cnt=2, external=(True, True))
		validate_shape(model, "multi_input")

	def test_labels_accepted(self):
		model, _ = make_model(labels=0x0800_1234)
		validate_shape(model, "labeled")  # must not raise

	def test_persistent_vars_rejected(self):
		model, _ = make_model(persistent_vars_count=3)
		with self.assertRaisesRegex(ValueError, "persistent variable"):
			validate_shape(model, "streaming")

	def test_packed_output_buf_rejected(self):
		model, _ = make_model(packed_output_buf=0x2000_2000)
		with self.assertRaisesRegex(ValueError, "packed_output_buf"):
			validate_shape(model, "packed")

	def test_internal_input_rejected(self):
		model, _ = make_model(input_cnt=2, external=(True, False))
		with self.assertRaisesRegex(ValueError, "not external"):
			validate_shape(model, "internal_input")

	def test_zero_input_cnt_rejected(self):
		model, _ = make_model(input_cnt=0, external=())
		with self.assertRaisesRegex(ValueError, "implausible"):
			validate_shape(model, "no_inputs")

	def test_extra_outputs_null_but_cnt_nonzero_rejected(self):
		model, _ = make_model(extra_output_cnt=2, extra_outputs=0)
		with self.assertRaisesRegex(ValueError, "NULL extra_outputs"):
			validate_shape(model, "extra")

	def test_extra_outputs_set_but_cnt_zero_rejected(self):
		model, _ = make_model(extra_output_cnt=0, extra_outputs=0x0800_5000)
		with self.assertRaisesRegex(ValueError, "non-NULL extra_outputs"):
			validate_shape(model, "extra")

	def test_extra_outputs_consistent_accepted(self):
		model, _ = make_model(extra_output_cnt=2, extra_outputs=0x0800_5000)
		validate_shape(model, "extra")  # must not raise


class RelocateRamPointersTests(unittest.TestCase):
	def test_offset_zero(self):
		model, _ = make_model(input_ptrs=[INTERLAYER_ADDR], output_ptr=INTERLAYER_ADDR)
		relocate_ram_pointers(model, "hello_axon", INTERLAYER_ADDR)
		self.assertEqual(model.inputs[0].ptr, 0)
		self.assertEqual(model.output_ptr, 0)
		self.assertEqual(model.model_name, 0)

	def test_nonzero_offset_preserved(self):
		model, _ = make_model(input_ptrs=[INTERLAYER_ADDR + 16],
				       output_ptr=INTERLAYER_ADDR + 32)
		relocate_ram_pointers(model, "multi_output", INTERLAYER_ADDR)
		self.assertEqual(model.inputs[0].ptr, 16)
		self.assertEqual(model.output_ptr, 32)

	def test_two_inputs_relocated_independently(self):
		model, _ = make_model(input_cnt=2, external=(True, True),
				       input_ptrs=[INTERLAYER_ADDR + 4, INTERLAYER_ADDR + 8],
				       output_ptr=INTERLAYER_ADDR)
		relocate_ram_pointers(model, "multi_input", INTERLAYER_ADDR)
		self.assertEqual(model.inputs[0].ptr, 4)
		self.assertEqual(model.inputs[1].ptr, 8)

	def test_pointer_outside_interlayer_buffer_rejected(self):
		model, _ = make_model(input_ptrs=[0x0800_9999], output_ptr=INTERLAYER_ADDR)
		with self.assertRaisesRegex(ValueError, "suspiciously large"):
			relocate_ram_pointers(model, "hello_axon", INTERLAYER_ADDR)


class RelocateFlashPointersTests(unittest.TestCase):
	def test_cmd_buffer_and_model_const_always_relocated(self):
		model, _ = make_model()
		relocate_flash_pointers(model, cmd_buffer_base=0x10_2100,
					 model_const_base=0x10_2300, extra_outputs_base=0x10_2400)
		self.assertEqual(model.cmd_buffer_ptr, 0x10_2100)
		self.assertEqual(model.model_const_ptr, 0x10_2300)
		self.assertEqual(model.extra_outputs, 0)  # extra_output_cnt is 0

	def test_extra_outputs_relocated_when_present(self):
		model, _ = make_model(extra_output_cnt=2, extra_outputs=0x0800_5000)
		relocate_flash_pointers(model, cmd_buffer_base=0x10_2100,
					 model_const_base=0x10_2300, extra_outputs_base=0x10_2400)
		self.assertEqual(model.extra_outputs, 0x10_2400)

	def test_labels_relocated_when_present(self):
		model, _ = make_model(labels=0x0800_6000)
		relocate_flash_pointers(model, cmd_buffer_base=0x10_2100, model_const_base=0x10_2300,
					 extra_outputs_base=0x10_2400, labels_base=0x10_2500)
		self.assertEqual(model.labels, 0x10_2500)

	def test_labels_left_null_when_absent(self):
		model, _ = make_model(labels=0)
		relocate_flash_pointers(model, cmd_buffer_base=0x10_2100, model_const_base=0x10_2300,
					 extra_outputs_base=0x10_2400, labels_base=0x10_2500)
		self.assertEqual(model.labels, 0)


class NumOutputClassesTests(unittest.TestCase):
	def test_single_dimension_classification(self):
		model, _ = make_model()
		model.output_dimensions.height = 1
		model.output_dimensions.width = 4
		model.output_dimensions.channel_cnt = 1
		self.assertEqual(num_output_classes(model), 4)

	def test_multi_channel(self):
		model, _ = make_model()
		model.output_dimensions.height = 2
		model.output_dimensions.width = 3
		model.output_dimensions.channel_cnt = 2
		self.assertEqual(num_output_classes(model), 12)


class BuildLabelsSectionTests(unittest.TestCase):
	def test_pointers_reference_correct_offsets(self):
		strings_base = 0x0010_2500
		labels_bytes, blob = build_labels_section([b"NON_PERSON", b"PERSON"], strings_base)

		ptrs = struct.unpack("<2I", labels_bytes)
		self.assertEqual(ptrs[0], strings_base)
		self.assertEqual(ptrs[1], strings_base + len(b"NON_PERSON\x00"))
		self.assertEqual(blob, b"NON_PERSON\x00PERSON\x00")

	def test_empty_label_list(self):
		labels_bytes, blob = build_labels_section([], 0x0010_2500)
		self.assertEqual(labels_bytes, b"")
		self.assertEqual(blob, b"")

	def test_embedded_nul_rejected(self):
		with self.assertRaisesRegex(ValueError, "NUL"):
			build_labels_section([b"bad\x00label"], 0x0010_2500)


class RelocateCmdBufferTests(unittest.TestCase):
	def test_matching_words_are_patched(self):
		old_base = 0x0800_0000
		new_base = 0x0010_2100
		span = 16
		words = [old_base + 4, 0xDEAD_BEEF, old_base + 15, old_base + 16]
		cmd_buffer = struct.pack("<%uI" % len(words), *words)

		patched = relocate_cmd_buffer(cmd_buffer, old_base, new_base, span)
		out = struct.unpack("<%uI" % len(words), patched)

		self.assertEqual(out[0], new_base + 4)   # inside [old_base, old_base+span): patched
		self.assertEqual(out[1], 0xDEAD_BEEF)    # untouched
		self.assertEqual(out[2], new_base + 15)  # last byte inside the span: patched
		self.assertEqual(out[3], old_base + 16)  # exactly at the span boundary: untouched


class CheckNoOpExtensionRefsTests(unittest.TestCase):
	def test_no_op_extensions_in_elf_accepted(self):
		cmd_buffer = struct.pack("<2I", 0x0010_2100, 0xDEAD_BEEF)
		check_no_op_extension_refs(cmd_buffer, "hello_axon", [])  # must not raise

	def test_unreferenced_op_extension_accepted(self):
		cmd_buffer = struct.pack("<2I", 0x0010_2100, 0xDEAD_BEEF)
		check_no_op_extension_refs(cmd_buffer, "hello_axon", [0x0800_1234])

	def test_referenced_op_extension_rejected(self):
		op_extension_addr = 0x0800_1234
		cmd_buffer = struct.pack("<2I", 0x0010_2100, op_extension_addr)
		with self.assertRaisesRegex(ValueError, "op extension"):
			check_no_op_extension_refs(cmd_buffer, "relu_model", [op_extension_addr])

	def test_thumb_tagged_reference_rejected(self):
		"""cmd_buffer may store the Thumb-tagged (LSB-set) function pointer."""
		op_extension_addr = 0x0800_1234
		cmd_buffer = struct.pack("<1I", op_extension_addr | 1)
		with self.assertRaisesRegex(ValueError, "op extension"):
			check_no_op_extension_refs(cmd_buffer, "relu_model", [op_extension_addr])


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
