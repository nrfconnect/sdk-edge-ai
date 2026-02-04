/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
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

/* Input size, matching the input layer dimensions in @ref model_hello_axon. */
#define INPUT_SIZE (1)
/* Output size, matching the output dimensions in @ref model_hello_axon. */
#define OUTPUT_SIZE (1)

static const float sample_values[] = {3.07f, 2.14f, 5.76f, 4.62f, 0.00f, 1.60f, 4.55f, 4.13f,
				      0.35f, 3.53f, 6.17f, 5.29f, 0.21f, 6.13f, 3.10f, 1.75f};

/* Used to pass current sample value to inference_callback to compare prediction with ideal. */
static float current_async_sample_value;

static void quantize(const float *values, const size_t length, int8_t *target,
		     const nrf_axon_nn_compiled_model_input_s *input_params)
{
	const uint32_t q_mult = input_params->quant_mult;
	const uint8_t q_round = input_params->quant_round;
	const int8_t q_zp = input_params->quant_zp;

	for (size_t i = 0; i < length; i++) {
		target[i] = (int8_t)((uint32_t)(values[i] * q_mult) >> q_round) + q_zp;
	}
}

static void dequantize(const int8_t *values, const size_t length, float *target,
		       const nrf_axon_nn_compiled_model_s *model)
{
	const uint32_t deq_mult = model->output_dequant_mult;
	const uint8_t deq_round = model->output_dequant_round;
	const int8_t deq_zp = model->output_dequant_zp;

	for (size_t i = 0; i < length; i++) {
		target[i] = (values[i] - deq_zp) * ((float)deq_mult / (1 << deq_round));
	}
}

static void sync_flow(void)
{
	nrf_axon_result_e result;

	result = nrf_axon_nn_model_validate(&model_hello_axon);
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Model validation failed, err %d", result);
		return;
	}

	LOG_INF("Running synchronous inference");

#if IS_ENABLED(CONFIG_AVOID_INPUT_DOUBLE_COPY)
	/* This path requires to know that no other model is running or will be running as we access
	 * internal buffer. Set input to NULL to indicate that data is already placed in buffer.
	 */
	int8_t *input_buffer = model_hello_axon.inputs[0].ptr;
	int8_t *input = NULL;
#else
	/* Input buffer of size 1, matching the input layer dimensions in @ref model_hello_axon. */
	int8_t input_buffer[INPUT_SIZE];
	int8_t *input = input_buffer;
#endif /*IS_ENABLED(CONFIG_AVOID_INPUT_DOUBLE_BUFFER)*/

	/* Output buffer of size 1, matching the output dimensions in @ref model_hello_axon. */
	int8_t output[OUTPUT_SIZE];

	for (size_t i = 0; i < ARRAY_SIZE(sample_values); i++) {
		/* Simulate wait for new data from sensor. */
		k_msleep(50);
		const float sample_value = sample_values[i];

		quantize(&sample_value, INPUT_SIZE, input_buffer,
			 &model_hello_axon.inputs[model_hello_axon.external_input_ndx]);

		result = nrf_axon_nn_model_infer_sync(&model_hello_axon, input, output);
		if (result != NRF_AXON_RESULT_SUCCESS) {
			LOG_ERR("Synchronous inference failed, err %d", result);
			return;
		}

		float prediction;

		dequantize(output, OUTPUT_SIZE, &prediction, &model_hello_axon);
		LOG_INF("prediction: %6.3f, ideal %6.3f", (double)prediction,
			(double)sinf(sample_value));
	}
}

static void inference_callback(nrf_axon_result_e result, void *callback_context)
{
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Inference failed, err %d", result);
		return;
	}

	const uint8_t *output = (const uint8_t *)callback_context;
	float prediction;

	dequantize(output, OUTPUT_SIZE, &prediction, &model_hello_axon);
	LOG_INF("prediction: %6.3f, ideal %6.3f", (double)prediction,
		(double)sinf(current_async_sample_value));

	nrf_axon_platform_generate_user_event();
}

static void async_flow(void)
{
	nrf_axon_nn_model_async_inference_wrapper_s async_wrapper;
	nrf_axon_result_e result;

	result = nrf_axon_nn_model_async_init(&async_wrapper, &model_hello_axon);
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Asynchronous model initialization failed, err %d", result);
		return;
	}

	LOG_INF("Running asynchronous inference");

	/* Input buffer of size 1, matching the input layer dimensions in @ref model_hello_axon. */
	int8_t input[INPUT_SIZE];
	/* Output buffer of size 1, matching the output dimensions in @ref model_hello_axon. */
	int8_t output[OUTPUT_SIZE];

	for (size_t i = 0; i < ARRAY_SIZE(sample_values); i++) {
		/* Simulate wait for new data from sensor. */
		k_msleep(50);
		const float sample_value = sample_values[i];
		void* cb_context = output;

		quantize(&sample_value, INPUT_SIZE, input,
			 &model_hello_axon.inputs[model_hello_axon.external_input_ndx]);

		current_async_sample_value = sample_values[i];
		result = nrf_axon_nn_model_infer_async(&async_wrapper, input, output,
						       inference_callback, cb_context);
		if (result != NRF_AXON_RESULT_SUCCESS) {
			LOG_ERR("Schedule asynchronous inference failed, err %d", result);
			return;
		}

		nrf_axon_platform_wait_for_user_event();
	}
}

int main(void)
{
	nrf_axon_result_e result;
	int err;

	LOG_INF("Hello Axon sample");
	LOG_INF("Initializing Axon NPU");

	__ASSERT(model_hello_axon.inputs[model_hello_axon.external_input_ndx]
				 .dimensions.byte_width == 1,
		 "Model input data type different than expected");
	__ASSERT(model_hello_axon.output_dimensions.byte_width == 1,
		 "Model output data type different than expected");

	result = nrf_axon_platform_init();
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Axon NPU platform initialization failed, err %d", result);
		return -1;
	}

	/* hello_axon_model does not use persistent vars, but initialize them anyway. */
	err = nrf_axon_nn_model_init_vars(&model_hello_axon);
	if (err) {
		LOG_ERR("Model persistent variables initialization failed, err %d", err);
		return -1;
	}

	if (IS_ENABLED(CONFIG_ASYNC_INFERENCE)) {
		async_flow();
	} else {
		sync_flow();
	}

	return 0;
}
