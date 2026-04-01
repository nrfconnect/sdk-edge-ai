/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Minimal person_det test: run inference on PC-generated int8 tensor from demo_picture.jpeg.
 * Regenerate input with: scripts/embed_demo_input.py (invoked from CMake).
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <stdint.h>

#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <axon/nrf_axon_platform.h>

#include "nrf_axon_model_person_det_.h"
#include "person_det_demo_input.h"
#include "person_det_postprocess.h"

LOG_MODULE_REGISTER(person_det_picture);

#define PACKED_OUTPUT_BYTES NRF_AXON_MODEL_PERSON_DET_PACKED_OUTPUT_SIZE

static int8_t output_buf[PACKED_OUTPUT_BYTES];

int main(void)
{
	const nrf_axon_nn_compiled_model_s *model = &model_person_det;
	struct person_det_box boxes[16];
	const int max_boxes = (int)(sizeof(boxes) / sizeof(boxes[0]));
	nrf_axon_result_e result;

	LOG_INF("person_det picture test (embedded tensor %d bytes)", PERSON_DET_DEMO_INPUT_BYTES);

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

	result = nrf_axon_nn_model_infer_sync(model, person_det_demo_input, output_buf);
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("inference failed: %d", result);
		return -1;
	}

	int n = person_det_decode_and_nms(model, boxes, max_boxes, 0.15f, 0.45f);

	LOG_INF("detections after NMS: %d", n);
	for (int i = 0; i < n; i++) {
		LOG_INF("  box %d: [%.1f, %.1f, %.1f, %.1f] score %.3f", i, (double)boxes[i].x1,
			(double)boxes[i].y1, (double)boxes[i].x2, (double)boxes[i].y2,
			(double)boxes[i].score);
	}

	return 0;
}
