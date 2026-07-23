/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/storage/flash_map.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#if defined(CONFIG_MODEL_OTA_AXON)
#include <model_ota/model_image.h>
#else
#include "generated/nrf_axon_model_okay_nordic.h"
#endif

LOG_MODULE_REGISTER(multi_okay_nordic, LOG_LEVEL_INF);

#define FRAMES_PER_INPUT 3
#define MEL_BINS         40
#define INPUT_SIZE       (FRAMES_PER_INPUT * MEL_BINS)
#define OUTPUT_SIZE      1

static int8_t input_buf[INPUT_SIZE];
static int8_t output_buf[OUTPUT_SIZE];

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
	const int8_t zero_mel = quantize(0.0f, in);

	memset(input_buf, zero_mel, sizeof(input_buf));
}

static float dequantize_output(const int8_t *value, const nrf_axon_nn_compiled_model_s *model)
{
	const uint32_t deq_mult = model->output_dequant_mult;
	const uint8_t deq_round = model->output_dequant_round;
	const int8_t deq_zp = model->output_dequant_zp;
	const float scale = (float)deq_mult / (float)(1 << deq_round);

	return (float)(value[0] - deq_zp) * scale;
}

void run_okay_nordic_tests(void)
{
	nrf_axon_result_e result;
	const nrf_axon_nn_compiled_model_s *model;
	const nrf_axon_nn_compiled_model_input_s *model_input;
	int err;

#if defined(CONFIG_MODEL_OTA_AXON)
	if (model_image_load_axon(PARTITION_ID(model_okay_nordic_storage),
				  (const uint8_t *)PARTITION_ADDRESS(model_okay_nordic_storage),
				  &model) != MODEL_IMAGE_OK ||
	    model == NULL) {
		LOG_WRN("No valid okay_nordic model image in model_okay_nordic_storage - skipping "
			"(flash okay_nordic_model_partition.hex)");
		return;
	}
#else
	model = &model_axon_user_instance_wakeword;
#endif

	model_input = nrf_axon_nn_model_1st_external_input(model);

	result = nrf_axon_platform_init();
	__ASSERT_NO_MSG(result == NRF_AXON_RESULT_SUCCESS);

	err = nrf_axon_nn_model_init_vars(model);
	__ASSERT_NO_MSG(err == 0);

	result = nrf_axon_nn_model_validate(model);
	__ASSERT_NO_MSG(result == NRF_AXON_RESULT_SUCCESS);

	prefill_dummy_input(model_input);

	result = nrf_axon_nn_model_infer_sync(model, input_buf, output_buf);
	__ASSERT_NO_MSG(result == NRF_AXON_RESULT_SUCCESS);

	const float probability = dequantize_output(output_buf, model);

	LOG_INF("okay_nordic inference on dummy mel input: raw %d, probability %.4f",
		output_buf[0], (double)probability);
}
