/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "test_classification_data.h"

#include <nrf_edgeai/nrf_edgeai.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(multi_classif_axon, LOG_LEVEL_INF);

nrf_edgeai_t *nrf_edgeai_user_model_36237(void);

static int32_t model_predict(nrf_edgeai_t *p_user_model, const flt32_t *p_input_data,
			     size_t data_len)
{
	nrf_edgeai_err_t res;

	for (size_t i = 0; i < data_len; i++) {
		flt32_t input_sample = p_input_data[i];

		res = nrf_edgeai_feed_inputs(p_user_model, &input_sample, USER_UNIQ_INPUTS_NUM);

		if (res == NRF_EDGEAI_ERR_SUCCESS) {
			res = nrf_edgeai_run_inference(p_user_model);

			if (res == NRF_EDGEAI_ERR_SUCCESS) {
				uint16_t predicted_class =
					p_user_model->decoded_output.classif.predicted_class;
				uint16_t num_classes =
					p_user_model->decoded_output.classif.num_classes;
				const flt32_t *p_probabilities =
					p_user_model->decoded_output.classif.probabilities.p_f32;
				uint16_t prob_percent =
					(uint16_t)(p_probabilities[predicted_class] * 100.0f);

				LOG_INF("In %u classes, predicted %u with probability %u %%",
					num_classes, predicted_class, prob_percent);

				return predicted_class;
			}
		}
	}

	return -1;
}

void run_classification_axon_tests(void)
{
	nrf_edgeai_t *p_user_model = nrf_edgeai_user_model_36237();

	__ASSERT_NO_MSG(p_user_model != NULL);
	__ASSERT_NO_MSG(p_user_model->model.type == NRF_EDGEAI_MODEL_AXON);
	__ASSERT_NO_MSG(nrf_edgeai_input_window_size(p_user_model) == USER_WINDOW_SIZE);
	__ASSERT_NO_MSG(nrf_edgeai_uniq_inputs_num(p_user_model) == USER_UNIQ_INPUTS_NUM);
	__ASSERT_NO_MSG(nrf_edgeai_model_outputs_num(p_user_model) == USER_MODELS_CLASS_NUM);

	nrf_edgeai_err_t res = nrf_edgeai_init(p_user_model);

	__ASSERT_NO_MSG(res == NRF_EDGEAI_ERR_SUCCESS);

	LOG_INF("Using Axon-backed classification model");

	const size_t data_len = USER_WINDOW_SIZE * USER_UNIQ_INPUTS_NUM;
	int32_t predicted_class;

	LOG_INF("--- Testing IDLE state (parcel at rest) ---");
	predicted_class = model_predict(p_user_model, CLASS_0_PARCEL_IDLE_ACCEL_DATA, data_len);
	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_IDLE);
	LOG_INF("Expected class IDLE - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);

	LOG_INF("--- Testing SHAKING state (parcel vibrating) ---");
	predicted_class = model_predict(p_user_model, CLASS_1_PARCEL_SHAKING_ACCEL_DATA, data_len);
	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_SHAKING);
	LOG_INF("Expected class SHAKING - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);

	LOG_INF("--- Testing IMPACT event (collision detected) ---");
	predicted_class = model_predict(p_user_model, CLASS_2_PARCEL_IMPACT_ACCEL_DATA, data_len);
	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_IMPACT);
	LOG_INF("Expected class IMPACT - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);

	LOG_INF("--- Testing FREE FALL event (parcel in air/unsupported) ---");
	predicted_class = model_predict(p_user_model, CLASS_3_PARCEL_FREE_FALL_ACCEL_DATA, data_len);
	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_FREE_FALL);
	LOG_INF("Expected class FREE FALL - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);

	LOG_INF("--- Testing CARRYING (person carrying) ---");
	predicted_class = model_predict(p_user_model, CLASS_4_PARCEL_CARRYING_ACCEL_DATA, data_len);
	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_CARRYING);
	LOG_INF("Expected class CARRYING - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);

	LOG_INF("--- Testing IN CAR state (vehicle transport) ---");
	predicted_class = model_predict(p_user_model, CLASS_5_PARCEL_IN_CAR_ACCEL_DATA, data_len);
	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_IN_CAR);
	LOG_INF("Expected class IN CAR - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);

	LOG_INF("--- Testing PLACED state (active placement event) ---");
	predicted_class = model_predict(p_user_model, CLASS_6_PARCEL_PLACED_ACCEL_DATA, data_len);
	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_PLACED);
	LOG_INF("Expected class PLACED - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);
}
