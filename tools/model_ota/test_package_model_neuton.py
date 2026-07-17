#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Host-side unit tests for package_model_neuton.py's generated-.c parsing logic.

Run with: python3 -m unittest test_package_model_neuton.py -v
"""
import re
import unittest
from pathlib import Path

from package_model import build_package, sanity_check
from package_model_neuton import parse_neuton_model_c

MODELS_DIR = (Path(__file__).parent / "../../samples/nrf_edgeai/regression/src/"
	      "nrf_edgeai_generated/Neuton").resolve()
GENERATED_C = (MODELS_DIR / "nrf_edgeai_user_model.c").read_text()

# The coefficients nrf_edgeai_user_model.c (solution 90508, air quality regression) is known to
# contain, hand-copied here once as an independent oracle rather than re-deriving them from the
# same source the parser under test reads - see also model_wiring_neuton.c's own
# CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA=n branch, a third independent transcription of the same
# model, compiled directly into the regression sample.
EXPECTED = {
	"weights": [0.5644180, 0.1098376, 0.9692999, -0.4583102],
	"neuron_links": [2, 9, 0, 9],
	"neuron_internal_links_num": [0, 3],
	"neuron_external_links_num": [2, 4],
	"act_weights": [20.0000000, 10.0783634],
	"neuron_act_type_mask": [1],
	"output_neurons_indices": [1],
	"output_scale_min": [0.2000000],
	"output_scale_max": [63.7000008],
}


class ParseNeutonModelCTests(unittest.TestCase):
	def test_extracts_expected_coefficients(self):
		model = parse_neuton_model_c(GENERATED_C, name="aq_regression", version="1.0.0")

		for key, value in EXPECTED.items():
			self.assertEqual(model[key], value, "mismatch in '%s'" % key)

	def test_produces_valid_package(self):
		model = parse_neuton_model_c(GENERATED_C, name="aq_regression", version="1.0.0")
		sanity_check(model)
		build_package(model)  # must not raise

	def test_rejects_unknown_params_type(self):
		source = GENERATED_C.replace("#define MODEL_PARAMS_TYPE f32",
					      "#define MODEL_PARAMS_TYPE q4")
		with self.assertRaisesRegex(ValueError, "f32, q16, q8"):
			parse_neuton_model_c(source, name="x", version="1.0.0")

	def test_parses_q16_params_type(self):
		# MODEL_WEIGHTS/MODEL_NEURON_ACTIVATION_WEIGHTS hold quantized integers for a q16
		# model, not the float literals GENERATED_C (an f32 model) actually has.
		source = GENERATED_C.replace("#define MODEL_PARAMS_TYPE f32",
					      "#define MODEL_PARAMS_TYPE q16")
		source = re.sub(r"static const nrf_user_weight_t MODEL_WEIGHTS\[\] = \{[^}]*\};",
				 "static const nrf_user_weight_t MODEL_WEIGHTS[] = "
				 "{5644, 1098, -9692, -4583};", source)
		source = re.sub(
			r"static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS\[\] = "
			r"\{[^}]*\};",
			"static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS[] = "
			"{20000, 10078};", source)

		model = parse_neuton_model_c(source, name="x", version="1.0.0")

		self.assertEqual(model["params_type"], 1)
		self.assertEqual(model["weights"], [5644, 1098, -9692, -4583])
		self.assertEqual(model["act_weights"], [20000, 10078])
		sanity_check(model)
		build_package(model)  # must not raise

	def test_parses_q8_params_type(self):
		source = GENERATED_C.replace("#define MODEL_PARAMS_TYPE f32",
					      "#define MODEL_PARAMS_TYPE q8")
		source = re.sub(r"static const nrf_user_weight_t MODEL_WEIGHTS\[\] = \{[^}]*\};",
				 "static const nrf_user_weight_t MODEL_WEIGHTS[] = "
				 "{56, 10, -96, -45};", source)
		source = re.sub(
			r"static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS\[\] = "
			r"\{[^}]*\};",
			"static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS[] = "
			"{200, 100};", source)

		model = parse_neuton_model_c(source, name="x", version="1.0.0")

		self.assertEqual(model["params_type"], 2)
		self.assertEqual(model["weights"], [56, 10, -96, -45])
		self.assertEqual(model["act_weights"], [200, 100])
		sanity_check(model)
		build_package(model)  # must not raise

	def test_classification_task_packages_with_empty_output_scale(self):
		# Classification tasks never define MODEL_OUTPUT_SCALE_MIN/MAX to begin with
		# (nrf_edgeai_decoded_output_classif_t has no scale concept) - drop them here to
		# mimic what a real classification codegen output would look like.
		source = re.sub(r"#define\s+MODEL_TASK\s+\d+", "#define MODEL_TASK 0", GENERATED_C)
		source = re.sub(r"static const nrf_user_output_t MODEL_OUTPUT_SCALE_M(IN|AX)\[\] = "
				 r"\{[^}]*\};\s*", "", source)

		model = parse_neuton_model_c(source, name="x", version="1.0.0")

		self.assertEqual(model["output_scale_min"], [])
		self.assertEqual(model["output_scale_max"], [])
		sanity_check(model)
		build_package(model)  # must not raise

	def test_regression_task_packages_with_empty_average_embedding(self):
		# GENERATED_C is a regression task (MODEL_TASK=2); nrf_edgeai_decoded_output_regress_t
		# has no average-embedding concept, so it never defines MODEL_AVERAGE_EMBEDDING.
		model = parse_neuton_model_c(GENERATED_C, name="aq_regression", version="1.0.0")

		self.assertEqual(model["average_embedding"], [])

	def test_anomaly_task_parses_average_embedding(self):
		# Mimic what a real anomaly-detection codegen output would look like: MODEL_TASK=3
		# plus a MODEL_AVERAGE_EMBEDDING array (nrf_edgeai_decoded_output_anomaly_t), inserted
		# right after MODEL_OUTPUT_SCALE_MAX like the generated source actually places it.
		source = re.sub(r"#define\s+MODEL_TASK\s+\d+", "#define MODEL_TASK 3", GENERATED_C)
		source = re.sub(
			r"(static const nrf_user_output_t MODEL_OUTPUT_SCALE_MAX\[\] = \{[^}]*\};)",
			r"\1\nstatic const nrf_user_output_t MODEL_AVERAGE_EMBEDDING[] = "
			r"{0.5000000};", source)

		model = parse_neuton_model_c(source, name="x", version="1.0.0")

		self.assertEqual(model["average_embedding"], [0.5])
		sanity_check(model)
		build_package(model)  # must not raise

	def test_missing_average_embedding_for_anomaly_task_raises(self):
		source = re.sub(r"#define\s+MODEL_TASK\s+\d+", "#define MODEL_TASK 3", GENERATED_C)

		with self.assertRaisesRegex(ValueError, "MODEL_AVERAGE_EMBEDDING"):
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
