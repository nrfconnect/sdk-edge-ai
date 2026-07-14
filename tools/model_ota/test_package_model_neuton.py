#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Host-side unit tests for package_model_neuton.py's generated-.c parsing logic.

Run with: python3 -m unittest test_package_model_neuton.py -v
"""
import json
import re
import unittest
from pathlib import Path

from package_model import build_package, sanity_check
from package_model_neuton import parse_neuton_model_c

MODELS_DIR = Path(__file__).parent / "models"
GENERATED_C = (MODELS_DIR / "regression_v1_generated.c").read_text()
EXPECTED_JSON = json.loads((MODELS_DIR / "regression_v1.json").read_text())


class ParseNeutonModelCTests(unittest.TestCase):
	def test_extracts_same_coefficients_as_hand_written_json(self):
		model = parse_neuton_model_c(GENERATED_C, name="aq_regression", version="1.0.0")

		for key in ("weights", "act_weights", "output_scale_min", "output_scale_max",
			    "neuron_links", "neuron_internal_links_num",
			    "neuron_external_links_num", "output_neurons_indices",
			    "neuron_act_type_mask"):
			self.assertEqual(model[key], EXPECTED_JSON[key], "mismatch in '%s'" % key)

	def test_produces_byte_identical_package_to_hand_written_json(self):
		model = parse_neuton_model_c(GENERATED_C, name="aq_regression", version="1.0.0")
		sanity_check(model)

		expected_model = dict(EXPECTED_JSON)
		expected_model["version"] = "1.0.0"
		sanity_check(expected_model)

		self.assertEqual(build_package(model), build_package(expected_model))

	def test_rejects_non_f32_params_type(self):
		source = GENERATED_C.replace("#define MODEL_PARAMS_TYPE f32",
					      "#define MODEL_PARAMS_TYPE q16")
		with self.assertRaisesRegex(ValueError, "only supports f32"):
			parse_neuton_model_c(source, name="x", version="1.0.0")

	def test_rejects_classification_task(self):
		source = re.sub(r"#define\s+MODEL_TASK\s+\d+", "#define MODEL_TASK 0", GENERATED_C)
		with self.assertRaisesRegex(ValueError, "regression .* and anomaly detection"):
			parse_neuton_model_c(source, name="x", version="1.0.0")

	def test_missing_required_array_raises(self):
		source = re.sub(r"static const uint16_t MODEL_NEURONS_LINKS\[\] = \{[^}]*\};", "",
				 GENERATED_C)
		with self.assertRaisesRegex(ValueError, "MODEL_NEURONS_LINKS"):
			parse_neuton_model_c(source, name="x", version="1.0.0")

	def test_missing_output_scale_for_regression_task_raises(self):
		source = re.sub(r"static const nrf_user_output_t MODEL_OUTPUT_SCALE_MIN\[\] = "
				 r"\{[^}]*\};", "", GENERATED_C)
		with self.assertRaisesRegex(ValueError, "MODEL_OUTPUT_SCALE_MIN"):
			parse_neuton_model_c(source, name="x", version="1.0.0")


if __name__ == "__main__":
	unittest.main()
