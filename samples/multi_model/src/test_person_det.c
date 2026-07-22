/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "postprocessing.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#include "generated/nrf_axon_model_person_det_.h"

LOG_MODULE_REGISTER(multi_person_det, LOG_LEVEL_INF);

#define MODEL_WIDTH  160
#define MODEL_HEIGHT 128
#define MAX_BOXES    8

static int8_t input_buf[MODEL_WIDTH * MODEL_HEIGHT * 3];
static int8_t output_buf[NRF_AXON_MODEL_PERSON_DET_PACKED_OUTPUT_SIZE];

static inline int8_t quantize(const float value, const nrf_axon_nn_compiled_model_input_s *in)
{
	const uint32_t quant_mult = in->quant_mult;
	const uint8_t quant_round = in->quant_round;
	const int8_t quant_zp = in->quant_zp;
	const float scale = (float)quant_mult / (float)(1 << quant_round);
	const int32_t quantized = (int32_t)(value * scale) + quant_zp;

	return (int8_t)__ssat(quantized, 8);
}

static void prefill_dummy_input(const nrf_axon_nn_compiled_model_input_s *in)
{
	const int8_t gray = quantize(0.0f, in);

	memset(input_buf, gray, sizeof(input_buf));
}

void run_person_det_tests(void)
{
	nrf_axon_result_e result;
	const nrf_axon_nn_compiled_model_s *model = &model_person_det;
	const nrf_axon_nn_compiled_model_input_s *model_input =
		nrf_axon_nn_model_1st_external_input(model);
	struct detection_box boxes[MAX_BOXES];

	result = nrf_axon_platform_init();
	__ASSERT_NO_MSG(result == NRF_AXON_RESULT_SUCCESS);

	result = nrf_axon_nn_model_validate(model);
	__ASSERT_NO_MSG(result == NRF_AXON_RESULT_SUCCESS);

	prefill_dummy_input(model_input);
	decode_init(model);

	result = nrf_axon_nn_model_infer_sync(model, input_buf, output_buf);
	__ASSERT_NO_MSG(result == NRF_AXON_RESULT_SUCCESS);

	const size_t detections = decode_output(model, output_buf, boxes, MAX_BOXES);

	LOG_INF("Person detection inference on dummy gray input: %zu detection(s)",
		detections);
}
