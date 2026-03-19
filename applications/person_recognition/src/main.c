/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Person recognition application for nRF54L: runs virat_mobilenetv2 on the Axon NPU
 * and reports whether a person is present in each test picture (demo_picture, demo_2, demo_3).
 * Output is 45x80x3 (spatial x 3 classes); class 1 is treated as "person".
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <axon/nrf_axon_platform.h>

#include "nrf_axon_model_virat_mobilenetv2_.h"
#include "generated/test_images.h"

LOG_MODULE_REGISTER(person_recognition);

/* Virat output: 45 x 80 x 3 (height x width x channels), int32, packed 43200 bytes */
#define VIRAT_OUT_H  45
#define VIRAT_OUT_W  80
#define VIRAT_OUT_C  3
#define VIRAT_OUT_SIZE (VIRAT_OUT_H * VIRAT_OUT_W * VIRAT_OUT_C * sizeof(int32_t))

/* Class index assumed for "person" in virat 3-class output */
#define VIRAT_CLASS_PERSON 1

static int8_t output_buf[VIRAT_OUT_SIZE];

/* Packed output is CHW: ch0[45*80], ch1[45*80], ch2[45*80] */
#define VIRAT_CELLS (VIRAT_OUT_H * VIRAT_OUT_W)

/** Count cells where argmax of 3 channels is VIRAT_CLASS_PERSON. */
static int count_person_cells(const int32_t *out)
{
	int count = 0;
	const int32_t *ch0 = out;
	const int32_t *ch1 = out + VIRAT_CELLS;
	const int32_t *ch2 = out + 2 * VIRAT_CELLS;

	for (int h = 0; h < VIRAT_OUT_H; h++) {
		for (int w = 0; w < VIRAT_OUT_W; w++) {
			int idx = h * VIRAT_OUT_W + w;
			int32_t s0 = ch0[idx];
			int32_t s1 = ch1[idx];
			int32_t s2 = ch2[idx];
			int best_c = (s1 > s0 && s1 >= s2) ? 1 : ((s2 > s0 && s2 >= s1) ? 2 : 0);
			if (best_c == VIRAT_CLASS_PERSON) {
				count++;
			}
		}
	}
	return count;
}

int main(void)
{
	nrf_axon_result_e result;
	const nrf_axon_nn_compiled_model_s *model = &model_virat_mobilenetv2;

	LOG_INF("Person recognition (nRF54L Axon, virat_mobilenetv2)");

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

		int person_cells = count_person_cells((const int32_t *)output_buf);
		LOG_INF("%s: person present: %s (person cells %d / %d)",
			name,
			person_cells > 0 ? "yes" : "no",
			person_cells,
			VIRAT_OUT_H * VIRAT_OUT_W);
	}

	LOG_INF("Person recognition done.");
	return 0;
}
