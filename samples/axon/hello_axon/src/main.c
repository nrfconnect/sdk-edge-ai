/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <assert.h>
#include <math.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

LOG_MODULE_REGISTER(hello_axon);

/* Input size, matching the input layer dimensions of the currently loaded model. */
#define INPUT_SIZE  (1)
/* Output size, matching the output dimensions of the currently loaded model. */
#define OUTPUT_SIZE (1)

/* Random samples from 0 to 2π */
static const float sample_values[] = {3.07f, 2.14f, 5.76f, 4.62f, 0.00f, 1.60f, 4.55f, 4.13f,
				      0.35f, 3.53f, 6.17f, 5.29f, 0.21f, 6.13f, 3.10f, 1.75f};

/* Passes the currently running model, and the current sample value, to inference_callback. */
static const nrf_axon_nn_compiled_model_s *current_async_model;
static float current_async_sample_value;

/**
 * @brief Quantize and convert from HWC to CHW format.
 */
static void quantize_and_convert(const float *values, int8_t *target,
				 const nrf_axon_nn_compiled_model_input_s *input_params)
{
	const nrf_axon_nn_model_layer_dimensions_s *dims = &input_params->dimensions;

	if (IS_ENABLED(CONFIG_AVOID_INPUT_DOUBLE_COPY)) {
		__ASSERT(input_params->stride == dims->width,
			 "Stride differs from width, need to adapt fill_input");
	}

	const uint32_t q_mult = input_params->quant_mult;
	const uint8_t q_round = input_params->quant_round;
	const int8_t q_zp = input_params->quant_zp;
	const float scale = (float)q_mult / (float)(1 << q_round);

	for (size_t row = 0; row < input_params->dimensions.height; row++) {
		for (size_t col = 0; col < input_params->dimensions.width; col++) {
			for (size_t chan = 0; chan < input_params->dimensions.channel_cnt; chan++) {
				const size_t input_offset = row * dims->width * dims->channel_cnt +
							    col * dims->channel_cnt + chan;
				const size_t target_offset =
					chan * dims->height * dims->width + row * dims->width + col;

				const int32_t quantized =
					(int32_t)(values[input_offset] * scale) + q_zp;

				target[target_offset] = (int8_t)__ssat(quantized, 8);
			}
		}
	}
}

static void dequantize(const int8_t *values, const size_t length, float *target,
		       const nrf_axon_nn_compiled_model_s *model)
{
	const uint32_t deq_mult = model->output_dequant_mult;
	const uint8_t deq_round = model->output_dequant_round;
	const int8_t deq_zp = model->output_dequant_zp;
	const float scale = (float)deq_mult / (float)(1 << deq_round);

	for (size_t i = 0; i < length; i++) {
		target[i] = (float)(values[i] - deq_zp) * scale;
	}
}

static void sync_flow(const nrf_axon_nn_compiled_model_s *model)
{
	nrf_axon_result_e result;

	result = nrf_axon_nn_model_validate(model);
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Model validation failed, err %d", result);
		return;
	}

	LOG_INF("Running synchronous inference");

#if IS_ENABLED(CONFIG_AVOID_INPUT_DOUBLE_COPY)
	/* This path requires to know that no other model is running or will be running as we access
	 * internal buffer. Set input to NULL to indicate that data is already placed in buffer.
	 */
	int8_t *input_buffer = model->inputs[0].ptr;
	int8_t *input = NULL;
#else
	/* Input buffer of size 1, matching the input layer dimensions of the loaded model. */
	int8_t input_buffer[INPUT_SIZE];
	int8_t *input = input_buffer;
#endif /* IS_ENABLED(CONFIG_AVOID_INPUT_DOUBLE_COPY) */

	/* Output buffer of size 1, matching the output dimensions of the loaded model. */
	int8_t output[OUTPUT_SIZE];

	for (size_t i = 0; i < ARRAY_SIZE(sample_values); i++) {
		/* Simulate wait for new data from sensor. */
		k_msleep(50);
		const float sample_value = sample_values[i];

		quantize_and_convert(&sample_value, input_buffer,
				     &model->inputs[model->external_input_ndx]);

		result = nrf_axon_nn_model_infer_sync(model, input, output);
		if (result != NRF_AXON_RESULT_SUCCESS) {
			LOG_ERR("Synchronous inference failed, err %d", result);
			return;
		}

		float prediction;

		/* Output in CHW format. */
		dequantize(output, OUTPUT_SIZE, &prediction, model);
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

	/* Output in CHW format. */
	dequantize(output, OUTPUT_SIZE, &prediction, current_async_model);
	LOG_INF("prediction: %6.3f, ideal %6.3f", (double)prediction,
		(double)sinf(current_async_sample_value));

	nrf_axon_platform_generate_user_event();
}

static void async_flow(const nrf_axon_nn_compiled_model_s *model)
{
	nrf_axon_nn_model_async_inference_wrapper_s async_wrapper;
	nrf_axon_result_e result;

	current_async_model = model;

	result = nrf_axon_nn_model_async_init(&async_wrapper, model);
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Asynchronous model initialization failed, err %d", result);
		return;
	}

	LOG_INF("Running asynchronous inference");

	/* Input buffer of size 1, matching the input layer dimensions of the loaded model. */
	int8_t input[INPUT_SIZE];
	/* Output buffer of size 1, matching the output dimensions of the loaded model. */
	int8_t output[OUTPUT_SIZE];

	for (size_t i = 0; i < ARRAY_SIZE(sample_values); i++) {
		/* Simulate wait for new data from sensor. */
		k_msleep(50);
		const float sample_value = sample_values[i];
		void *cb_context = output;

		quantize_and_convert(&sample_value, input,
				     &model->inputs[model->external_input_ndx]);

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

static void run_inference(const nrf_axon_nn_compiled_model_s *model)
{
	int err;

	__ASSERT(model->inputs[model->external_input_ndx].dimensions.byte_width == 1,
		 "Model input data type different than expected");
	__ASSERT(model->output_dimensions.byte_width == 1,
		 "Model output data type different than expected");

	/* hello_axon_model does not use persistent vars, but initialize them anyway. */
	err = nrf_axon_nn_model_init_vars(model);
	if (err) {
		LOG_ERR("Model persistent variables initialization failed, err %d", err);
		return;
	}

	if (IS_ENABLED(CONFIG_ASYNC_INFERENCE)) {
		async_flow(model);
	} else {
		sync_flow(model);
	}
}

#if defined(CONFIG_HELLO_AXON_MODEL_OTA)

#include <model_ota/model_pkg.h>

#include <zephyr/storage/flash_map.h>

/*
 * Fail the build with a clear message if this board's devicetree overlay doesn't define the
 * model_partition node model_ota loads from (see the overlays under boards/), instead of a
 * much less obvious error deep inside a flash_map.h macro expansion.
 */
BUILD_ASSERT(FIXED_PARTITION_EXISTS(model_partition),
	     "board devicetree is missing the model_partition node - see boards/*.overlay");

/*
 * The model itself is *not* compiled in: this is only populated (by model_pkg_load_axon())
 * once a valid package has been read from the model_storage flash partition. Not named
 * model_hello_axon to avoid any confusion with the generated header's own model_hello_axon
 * symbol, which this application no longer links in at all (see CMakeLists.txt's
 * nrf_axon_model_stub() call - it is only ever compiled into the standalone model stub).
 */
static nrf_axon_nn_compiled_model_s active_model;

static const nrf_axon_nn_compiled_model_s *model_ota_load(void)
{
	struct model_pkg_axon_info info;
	int err;

	err = model_pkg_load_axon(PARTITION_ID(model_partition),
				   (const uint8_t *)PARTITION_ADDRESS(model_partition),
				   &active_model, &info);
	if (err != MODEL_PKG_OK) {
		LOG_ERR("No usable model in model_storage (err %d)", err);
		return NULL;
	}

	LOG_INF("Active model: '%s' version 0x%08x (%u cmd words, %u B const)", info.name,
		info.version, info.cmd_buffer_len, info.model_const_size);

	return &active_model;
}

int main(void)
{
	nrf_axon_result_e result;

	LOG_INF("Hello Axon sample");
	LOG_INF("Initializing Axon NPU");

	result = nrf_axon_platform_init();
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Axon NPU platform initialization failed, err %d", result);
		return -1;
	}

	/*
	 * The app image never contains a model (see model_ota_load() above): at boot, and
	 * after every inference pass, it (re)loads and validates a "model package" from the
	 * model_storage flash partition, independently of the application image. Flashing a new model
	 * package - without rebuilding or reflashing the application - is enough to change what
	 * the device predicts, and is picked up here without needing a physical reset.
	 */
	while (1) {
		const nrf_axon_nn_compiled_model_s *model = model_ota_load();

		if (model == NULL) {
			LOG_WRN("No valid model in model_storage - waiting for one to be "
				"flashed. Inference is skipped until then.");
		} else {
			run_inference(model);
		}

		k_sleep(K_MSEC(5000));
	}

	return 0;
}

#else /* !CONFIG_HELLO_AXON_MODEL_OTA: compiled-in model, no flash partition or reload loop */

#include "generated/nrf_axon_model_hello_axon_.h"

int main(void)
{
	nrf_axon_result_e result;

	LOG_INF("Hello Axon sample");
	LOG_INF("Initializing Axon NPU");

	result = nrf_axon_platform_init();
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Axon NPU platform initialization failed, err %d", result);
		return -1;
	}

	run_inference(&model_hello_axon);

	return 0;
}

#endif /* CONFIG_HELLO_AXON_MODEL_OTA */
