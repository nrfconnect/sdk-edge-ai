/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "test_regression_data.h"

#include <nrf_edgeai/nrf_edgeai.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>
#include <math.h>

#if defined(CONFIG_MODEL_OTA_AXON)
#include <model_ota/model_ota_axon_edgeai.h>
#include <zephyr/storage/flash_map.h>

MODEL_OTA_AXON_EDGEAI_LOAD_DECL(36025);

BUILD_ASSERT(FIXED_PARTITION_EXISTS(model_regress_axon_storage),
	     "board devicetree is missing model_regress_axon_storage - see boards/*.overlay");
#else
nrf_edgeai_t *nrf_edgeai_user_model_36025(void);
#endif

LOG_MODULE_REGISTER(multi_regress_axon, LOG_LEVEL_INF);

static flt32_t fill_features_buffer(flt32_t *p_buffer, const size_t buffer_size,
				    const size_t sample_index)
{
	__ASSERT_NO_MSG(buffer_size == USER_UNIQ_INPUTS_NUM);

	p_buffer[0] = USER_INPUT_DATA[sample_index].COGT;
	p_buffer[1] = USER_INPUT_DATA[sample_index].PT08S1;
	p_buffer[2] = USER_INPUT_DATA[sample_index].PT08S2;
	p_buffer[3] = USER_INPUT_DATA[sample_index].PT08S3;
	p_buffer[4] = USER_INPUT_DATA[sample_index].PT08S4;
	p_buffer[5] = USER_INPUT_DATA[sample_index].PT08S5;
	p_buffer[6] = USER_INPUT_DATA[sample_index].T;
	p_buffer[7] = USER_INPUT_DATA[sample_index].RH;
	p_buffer[8] = USER_INPUT_DATA[sample_index].AH;

	return USER_INPUT_DATA[sample_index].target;
}

static flt32_t model_predict(nrf_edgeai_t *p_user_model, flt32_t *p_input_features,
			     size_t features_num)
{
	nrf_edgeai_err_t res;
	flt32_t model_prediction = INVALID_PREDICTION_VALUE;

	res = nrf_edgeai_feed_inputs(p_user_model, p_input_features, features_num);

	if (res == NRF_EDGEAI_ERR_SUCCESS) {
		res = nrf_edgeai_run_inference(p_user_model);

		if (res == NRF_EDGEAI_ERR_SUCCESS) {
			const flt32_t *p_output = p_user_model->decoded_output.regression.p_outputs;

			__ASSERT_NO_MSG(p_user_model->decoded_output.regression.outputs_num ==
					USER_MODELS_OUTPUTS_NUM);

			model_prediction = p_output[0];
		}
	}

	return model_prediction;
}

void run_regression_axon_tests(void)
{
#if defined(CONFIG_MODEL_OTA_AXON)
	/* Model-only OTA: the compiled Axon model is loaded from its flash partition (XIP) at
	 * runtime and wired into the app-compiled nrf_edgeai_t wrapper.
	 */
	nrf_edgeai_t *p_user_model = nrf_edgeai_load_user_model_36025(
		PARTITION_ID(model_regress_axon_storage),
		(const uint8_t *)PARTITION_ADDRESS(model_regress_axon_storage));

	if (p_user_model == NULL) {
		LOG_WRN("No valid regression model image in model_regress_axon_storage - "
			"skipping (flash regress_axon_model_partition.hex)");
		return;
	}
#else
	nrf_edgeai_t *p_user_model = nrf_edgeai_user_model_36025();
#endif

	__ASSERT_NO_MSG(p_user_model != NULL);
	__ASSERT_NO_MSG(p_user_model->model.type == NRF_EDGEAI_MODEL_AXON);
	__ASSERT_NO_MSG(nrf_edgeai_input_window_size(p_user_model) == USER_WINDOW_SIZE);
	__ASSERT_NO_MSG(nrf_edgeai_uniq_inputs_num(p_user_model) == USER_UNIQ_INPUTS_NUM);
	__ASSERT_NO_MSG(nrf_edgeai_model_outputs_num(p_user_model) == USER_MODELS_OUTPUTS_NUM);

	nrf_edgeai_err_t res = nrf_edgeai_init(p_user_model);

	__ASSERT_NO_MSG(res == NRF_EDGEAI_ERR_SUCCESS);

	flt32_t input_features[USER_UNIQ_INPUTS_NUM];

	LOG_INF("Using Axon-backed regression model");
	LOG_INF("--- Testing Model Air Quality predictions ---");

	const size_t num_input_samples = ARRAY_SIZE(USER_INPUT_DATA);

	for (size_t i = 0; i < num_input_samples; i++) {
		flt32_t ground_truth =
			fill_features_buffer(input_features, USER_UNIQ_INPUTS_NUM, i);
		flt32_t predicted_value =
			model_predict(p_user_model, input_features, USER_UNIQ_INPUTS_NUM);
		flt32_t abs_err = fabsf(predicted_value - ground_truth);

		__ASSERT_NO_MSG(abs_err <= EXPECTED_MODEL_MAE);

		LOG_INF("Air quality - Predicted: %f, Expected: %f, absolute error %f",
			(double)predicted_value, (double)ground_truth, (double)abs_err);
	}
}
