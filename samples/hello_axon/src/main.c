/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <nrf_axon_driver.h>
#include <nrf_axon_nn_infer.h>
#include <nrf_axon_platform.h>
#include <zephyr/logging/log.h>
#include <math.h>

#include "generated/nrf_axon_model_hello_axon_.h"

LOG_MODULE_REGISTER(hello_axon);

static nrf_axon_nn_model_async_inference_wrapper_s async_wrapper;

static int8_t quantitize(const float value, const nrf_axon_nn_compiled_model_input_s *input_params)
{
	return (int8_t)((uint32_t)(value * input_params->quant_mult) >> input_params->quant_round) +
	       input_params->quant_zp;
}

static float dequantitize(const int8_t value, const nrf_axon_nn_compiled_model_s *model)
{
	return (value - model->output_dequant_zp) *
	       ((float)model->output_dequant_mult / (1 << model->output_dequant_round));
}

static void inference_callback(nrf_axon_result_e result, void *callback_context)
{
	ARG_UNUSED(callback_context);
	LOG_INF("Inference completed, result %u", result);

	nrf_axon_platform_generate_user_event();
}

int main(void)
{
	nrf_axon_result_e result;
	int err;

	LOG_INF("Hello AXON sample");
	LOG_INF("Initializing AXON");

	result = nrf_axon_platform_init();
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("AXON platform init failed, err %u", result);
	}

	if (IS_ENABLED(CONFIG_ASYNC_INFERENCE)) {
		result = nrf_axon_nn_model_async_init(&async_wrapper, &model_hello_axon);
		if (result != NRF_AXON_RESULT_SUCCESS) {
			LOG_ERR("AXON async model init failed, err %u", result);
		}
	}

	/* hello_axon model don't use persistent vars, but initialize them anyway */
	err = nrf_axon_nn_model_init_vars(&model_hello_axon);
	if (err) {
		LOG_ERR("AXON model init vars, err %u", result);
	}

	int8_t input[1], output[1];

	for (float value = 0.0; value < 4.0f; value += 0.1) {

		input[0] = quantitize(value, &model_hello_axon.inputs[0]);

		if (IS_ENABLED(CONFIG_ASYNC_INFERENCE)) {
			result = nrf_axon_nn_model_infer_async(&async_wrapper, input, output,
							       inference_callback, NULL);
			if (result != NRF_AXON_RESULT_SUCCESS) {
				LOG_ERR("AXON schedule async inference failed, err %u", result);
			}

			nrf_axon_platform_wait_for_user_event();
		} else {
			result = nrf_axon_nn_model_infer_sync(&model_hello_axon, input, output);
			if (result != NRF_AXON_RESULT_SUCCESS) {
				LOG_ERR("AXON sync inference failed, err %u", result);
			}
		}

		const float prediction = dequantitize(output[0], &model_hello_axon);

		LOG_INF("prediction: %6.3f,  ideal %6.3f", (double)prediction, (double)sinf(value));
	}

	return 0;
}
