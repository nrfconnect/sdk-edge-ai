#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""ctypes mirror of nrf_axon_nn_compiled_model_s's on-device ABI (see
include/drivers/axon/nrf_axon_nn_infer.h), used by package_model_axon.py to read and patch the
compiler-generated struct's pointer fields in place, at the same byte offsets the firmware
itself uses - instead of hand-decoding/rebuilding a handful of scalar fields, as prior format
versions of this tool did.

All pointer-sized fields are declared as c_uint32 (not real pointers): the target is always a
32-bit ARM Cortex-M, and package_model_axon.py only ever needs the raw 4-byte value to compare
against ELF symbol addresses or rewrite to a computed flash offset, never to actually dereference
it on the host. Every class is a LittleEndianStructure with no 8-byte-aligned members, so ctypes'
default (unpacked) layout naturally matches the target's AAPCS layout on any host - this is
cross-checked at runtime by build_package() (see MODEL_STRUCT_SIZE_MISMATCH below) rather than
trusted blindly, since a hand-mirrored ABI is inherently one refactor-of-nrf_axon_nn_infer.h away
from silently drifting out of sync.
"""
import ctypes

MAX_MODEL_INPUTS = 2  # NRF_AXON_NN_MAX_MODEL_INPUTS


class Dimensions(ctypes.LittleEndianStructure):
	_fields_ = [
		("height", ctypes.c_uint16),
		("width", ctypes.c_uint16),
		("channel_cnt", ctypes.c_uint16),
		("byte_width", ctypes.c_uint8),
	]


class Input(ctypes.LittleEndianStructure):
	_fields_ = [
		("ptr", ctypes.c_uint32),
		("dimensions", Dimensions),
		("quant_mult", ctypes.c_uint32),
		("stride", ctypes.c_uint16),
		("quant_round", ctypes.c_uint8),
		("quant_zp", ctypes.c_int8),
		("is_external", ctypes.c_uint8),
	]


class OutputDesc(ctypes.LittleEndianStructure):
	"""Mirrors nrf_axon_compiled_model_output_s (an element of extra_outputs[])."""
	_fields_ = [
		("ptr", ctypes.c_uint32),
		("dimensions", Dimensions),
		("dequant_mult", ctypes.c_uint32),
		("dequant_round", ctypes.c_uint8),
		("dequant_zp", ctypes.c_int8),
		("stride", ctypes.c_uint16),
	]


class PersistentVars(ctypes.LittleEndianStructure):
	_fields_ = [
		("buf_ptr", ctypes.c_uint32),
		("buf_size", ctypes.c_uint32),
		("vars", ctypes.c_uint32),
		("count", ctypes.c_uint16),
	]


class CompiledModel(ctypes.LittleEndianStructure):
	"""Mirrors nrf_axon_nn_compiled_model_s field-for-field and in the same order. Field
	names match the C struct exactly so callers can read/patch it the same way C code
	would (e.g. `model.inputs[0].ptr`, `model.persistent_vars.count`)."""
	_fields_ = [
		("compiler_version", ctypes.c_uint32),
		("model_name", ctypes.c_uint32),
		("labels", ctypes.c_uint32),
		("inputs", Input * MAX_MODEL_INPUTS),
		("input_cnt", ctypes.c_uint8),
		("external_input_ndx", ctypes.c_int8),
		("output_ptr", ctypes.c_uint32),
		("packed_output_buf", ctypes.c_uint32),
		("interlayer_buffer_needed", ctypes.c_uint32),
		("psum_buffer_needed", ctypes.c_uint32),
		("cmd_buffer_ptr", ctypes.c_uint32),
		("model_const_ptr", ctypes.c_uint32),
		("model_const_size", ctypes.c_uint32),
		("cmd_buffer_len", ctypes.c_uint32),
		("persistent_vars", PersistentVars),
		("output_dimensions", Dimensions),
		("output_dequant_mult", ctypes.c_uint32),
		("output_dequant_round", ctypes.c_uint8),
		("output_dequant_zp", ctypes.c_int8),
		("output_stride", ctypes.c_uint16),
		("is_layer_model", ctypes.c_uint8),
		("extra_output_cnt", ctypes.c_uint16),
		("extra_outputs", ctypes.c_uint32),
		("min_driver_version_required", ctypes.c_uint32),
	]


def check_struct_size(actual_size, context):
	"""Raises ValueError if ctypes.sizeof(CompiledModel) doesn't match a real struct
	instance's size read from an ELF - the only trustworthy ground truth for the ABI this
	module hand-mirrors. Call this before relying on any offset/field computed against
	CompiledModel."""
	expected_size = ctypes.sizeof(CompiledModel)
	if actual_size != expected_size:
		raise ValueError(
			"nrf_axon_nn_compiled_model_s is %u B in %s, but this tool's ctypes mirror "
			"(axon_struct_layout.py) computes %u B - it has drifted out of sync with "
			"include/drivers/axon/nrf_axon_nn_infer.h and must be updated before "
			"packaging is safe to trust" % (actual_size, context, expected_size))
