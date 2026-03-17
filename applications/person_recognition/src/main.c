/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Person recognition application for nRF54L: runs tinyml_vww (Visual Wake Word)
 * on the Axon NPU and reports whether a person is present in each test picture
 * (demo_picture, demo_2, demo_3).
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <axon/nrf_axon_platform.h>

#include "nrf_axon_model_tinyml_vww_.h"
#include "generated/test_images.h"

LOG_MODULE_REGISTER(person_recognition);

/* VWW classes: 0 = non_person, 1 = person (matches compiler_sample_vww_input.yaml) */
#define VWW_NUM_CLASSES 2
#define VWW_CLASS_PERSON 1

int main(void)
{
	nrf_axon_result_e result;
	const nrf_axon_nn_compiled_model_s *model = &model_tinyml_vww;
	int8_t output_buf[8]; /* VWW output: 1x2 int32 => 8 bytes */

	LOG_INF("Person recognition (nRF54L Axon, tinyml_vww)");

	result = nrf_axon_platform_init();
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Axon platform init failed: %d", result);
		return -1;
	}

	result = nrf_axon_nn_model_validate(model);
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Model validation failed: %d", result);
		return -1;
	}

	if (nrf_axon_nn_model_init_vars(model) != 0) {
		LOG_ERR("Model init_vars failed");
		return -1;
	}

	for (size_t i = 0; i < PERSON_RECOGNITION_NUM_TEST_IMAGES; i++) {
		const int8_t *input = person_recognition_test_inputs[i];
		const char *name = person_recognition_test_names[i];

		result = nrf_axon_nn_model_infer_sync(model, input, output_buf);
		if (result != NRF_AXON_RESULT_SUCCESS) {
			LOG_ERR("%s: inference failed: %d", name, result);
			continue;
		}

		int32_t score = 0;
		int16_t class_idx = nrf_axon_nn_get_classification(model, output_buf, NULL, &score);
		if (class_idx < 0) {
			LOG_ERR("%s: classification failed", name);
			continue;
		}

		if (class_idx < VWW_NUM_CLASSES) {
			LOG_INF("%s: person present: %s (class %d, score %d)",
				name,
				class_idx == VWW_CLASS_PERSON ? "yes" : "no",
				class_idx,
				score);
		} else {
			LOG_WRN("%s: unexpected class index: %d", name, class_idx);
		}
	}

	LOG_INF("Person recognition done.");
	return 0;
}
