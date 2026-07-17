#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Unit tests for gen_axon_stub_fixups.py's pure text-processing functions: op-extension/
axonpro-constant/persistent_vars symbol discovery and the persistent_vars-to-extern header
patch.

Run with: python3 -m unittest test_gen_axon_stub_fixups.py -v
"""
import unittest

from gen_axon_stub_fixups import (
	INTERLAYER_BUFFER_SYMBOL,
	collect_app_symbols,
	find_axonpro_const_symbols,
	find_op_extension_symbols,
	find_persistent_vars_symbols,
	patch_persistent_vars_definitions,
)

HEADER_NO_EXTRAS = """
const nrf_axon_nn_compiled_model_s model_hello_axon = {
	.cmd_buffer_ptr = cmd_buffer_hello_axon,
};
"""

HEADER_WITH_OP_EXTENSIONS = """
extern void nrf_axon_nn_op_extension_relu(void);
extern void nrf_axon_nn_op_extension_softmax(void);
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_kws[4] = {
	0x01, (NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_nn_op_extension_relu, 0x02,
	(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_nn_op_extension_softmax,
};
"""

HEADER_WITH_PERSISTENT_VARS = """
int32_t axon_model_ww_persistent_vars[1160];

const nrf_axon_nn_compiled_model_s model_ww = {
	.persistent_vars.vars = (int32_t)axon_model_ww_persistent_vars,
	.persistent_vars.count = 6,
};
"""

HEADER_WITH_INDENTED_PERSISTENT_VARS = """
struct foo {
	uint16_t axon_model_kws_persistent_vars[6648];
};
"""

HEADER_WITH_AXONPRO_CONST = """
extern const int8_t axonpro_int8_packing_filter[4];
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_kws[2] = {
	(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter, 0x02,
};
"""


class FindOpExtensionSymbolsTests(unittest.TestCase):
	def test_no_op_extensions(self):
		self.assertEqual(find_op_extension_symbols(HEADER_NO_EXTRAS), [])

	def test_finds_and_dedupes_and_sorts(self):
		text = HEADER_WITH_OP_EXTENSIONS + "\n" + "nrf_axon_nn_op_extension_relu();"
		self.assertEqual(find_op_extension_symbols(text),
				  ["nrf_axon_nn_op_extension_relu", "nrf_axon_nn_op_extension_softmax"])


class FindAxonproConstSymbolsTests(unittest.TestCase):
	def test_no_axonpro_consts(self):
		self.assertEqual(find_axonpro_const_symbols(HEADER_NO_EXTRAS), [])

	def test_finds_and_dedupes_and_sorts(self):
		text = HEADER_WITH_AXONPRO_CONST + "\n" + "axonpro_int8_packing_filter[0];"
		self.assertEqual(find_axonpro_const_symbols(text), ["axonpro_int8_packing_filter"])


class FindPersistentVarsSymbolsTests(unittest.TestCase):
	def test_no_persistent_vars(self):
		self.assertEqual(find_persistent_vars_symbols(HEADER_NO_EXTRAS), [])

	def test_finds_definition_not_mere_mentions(self):
		result = find_persistent_vars_symbols(HEADER_WITH_PERSISTENT_VARS)
		self.assertEqual(result, [("axon_model_ww_persistent_vars", "int32_t", 1160)])

	def test_indented_definition_matched(self):
		result = find_persistent_vars_symbols(HEADER_WITH_INDENTED_PERSISTENT_VARS)
		self.assertEqual(result, [("axon_model_kws_persistent_vars", "uint16_t", 6648)])

	def test_multiple_models_multiple_definitions(self):
		text = HEADER_WITH_PERSISTENT_VARS + HEADER_WITH_INDENTED_PERSISTENT_VARS
		result = find_persistent_vars_symbols(text)
		names = sorted(name for name, _type, _size in result)
		self.assertEqual(names, ["axon_model_kws_persistent_vars", "axon_model_ww_persistent_vars"])


class CollectAppSymbolsTests(unittest.TestCase):
	def test_interlayer_buffer_always_present(self):
		self.assertEqual(collect_app_symbols(HEADER_NO_EXTRAS), [INTERLAYER_BUFFER_SYMBOL])

	def test_op_extensions_and_interlayer(self):
		result = collect_app_symbols(HEADER_WITH_OP_EXTENSIONS)
		self.assertEqual(result, sorted([
			INTERLAYER_BUFFER_SYMBOL,
			"nrf_axon_nn_op_extension_relu",
			"nrf_axon_nn_op_extension_softmax",
		]))

	def test_persistent_vars_and_interlayer(self):
		result = collect_app_symbols(HEADER_WITH_PERSISTENT_VARS)
		self.assertEqual(result, sorted([
			INTERLAYER_BUFFER_SYMBOL,
			"axon_model_ww_persistent_vars",
		]))

	def test_axonpro_const_and_interlayer(self):
		result = collect_app_symbols(HEADER_WITH_AXONPRO_CONST)
		self.assertEqual(result, sorted([
			INTERLAYER_BUFFER_SYMBOL,
			"axonpro_int8_packing_filter",
		]))

	def test_all_four_kinds_combined_and_deduped(self):
		text = HEADER_WITH_OP_EXTENSIONS + HEADER_WITH_PERSISTENT_VARS + HEADER_WITH_AXONPRO_CONST
		result = collect_app_symbols(text)
		self.assertEqual(result, sorted(set([
			INTERLAYER_BUFFER_SYMBOL,
			"nrf_axon_nn_op_extension_relu",
			"nrf_axon_nn_op_extension_softmax",
			"axon_model_ww_persistent_vars",
			"axonpro_int8_packing_filter",
		])))


class PatchPersistentVarsDefinitionsTests(unittest.TestCase):
	def test_no_persistent_vars_unchanged(self):
		self.assertEqual(patch_persistent_vars_definitions(HEADER_NO_EXTRAS), HEADER_NO_EXTRAS)

	def test_definition_rewritten_to_extern(self):
		patched = patch_persistent_vars_definitions(HEADER_WITH_PERSISTENT_VARS)
		self.assertIn("extern int32_t axon_model_ww_persistent_vars[1160];", patched)
		self.assertNotIn("int32_t axon_model_ww_persistent_vars[1160];\n", patched.replace(
			"extern int32_t axon_model_ww_persistent_vars[1160];", ""))

	def test_only_definition_rewritten_mentions_elsewhere_untouched(self):
		patched = patch_persistent_vars_definitions(HEADER_WITH_PERSISTENT_VARS)
		# The .persistent_vars.vars initializer mentioning the same array name by cast must
		# survive untouched - only the standalone array *definition* line is rewritten.
		self.assertIn("(int32_t)axon_model_ww_persistent_vars", patched)

	def test_indentation_preserved(self):
		patched = patch_persistent_vars_definitions(HEADER_WITH_INDENTED_PERSISTENT_VARS)
		self.assertIn("\textern uint16_t axon_model_kws_persistent_vars[6648];", patched)


if __name__ == "__main__":
	unittest.main()
