/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#include <errno.h>
#include <math.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai_user_model.h>

#include "accel.h"
#include "ble/ble_mds.h"
#include "ble/ble_nus.h"
#include "model_obsv.h"
#include "ux_state_manager.h"

/* One value per feed: accelerometer magnitude sqrt(ax^2 + ay^2 + az^2), m/s^2 */
#define NRF_EDGEAI_INPUT_DATA_LEN (1U)

LOG_MODULE_REGISTER(main);

static K_SEM_DEFINE(accel_data_ready_sem, 0, 1);

static void accel_data_ready_cb(void)
{
	k_sem_give(&accel_data_ready_sem);
}

static nrf_edgeai_t *nrf_edgeai_init_model(void)
{
	nrf_edgeai_t *p_model = nrf_edgeai_user_model();

	__ASSERT_NO_MSG(p_model != NULL);
	__ASSERT_NO_MSG(nrf_edgeai_is_runtime_compatible(p_model));
	/* The model must be trained on the data_forwarder BMI270 float dataset */
	__ASSERT(nrf_edgeai_input_type(p_model) == NRF_EDGEAI_INPUT_F32,
		 "Model input must be float32");
	__ASSERT(nrf_edgeai_uniq_inputs_num(p_model) == NRF_EDGEAI_INPUT_DATA_LEN,
		 "Model must take %u input (acc_mag)", NRF_EDGEAI_INPUT_DATA_LEN);

	nrf_edgeai_err_t res = nrf_edgeai_init(p_model);

	__ASSERT_NO_MSG(res == NRF_EDGEAI_ERR_SUCCESS);

	nrf_edgeai_rt_version_t version = nrf_edgeai_runtime_version();

	LOG_INF("nRF Edge AI Model Observability Demo:");
	LOG_INF("nRF Edge AI Runtime Version: %d.%d.%d", version.field.major, version.field.minor,
		version.field.patch);
	LOG_INF("nRF Edge AI Lab Solution id: %s", nrf_edgeai_solution_id_str(p_model));

	return p_model;
}

static void handle_inference_result(nrf_edgeai_t *p_model)
{
	int err;
	uint16_t predicted_class;
	const flt32_t *p_probabilities;

	predicted_class = p_model->decoded_output.classif.predicted_class;
	p_probabilities = p_model->decoded_output.classif.probabilities.p_f32;
	__ASSERT_NO_MSG(p_probabilities != NULL);

	/* update the LED color based on the predicted class */
	uxsm_set_current_activity(predicted_class);

#if IS_ENABLED(CONFIG_APP_BLE_MDS)
	/* feed the class probabilities to the model observability metrics (MDS)*/
	err = model_obsv_update_probs(p_probabilities);
	if (err != 0) {
		LOG_WRN("Failed to update model observability, error = %d", err);
	}
#else
	uint8_t probability_pct;

	probability_pct = (uint8_t)(100.0f * p_probabilities[predicted_class]);

	/* send the recognized class over NUS */
	err = ble_nus_send_result(predicted_class, probability_pct);
	if ((err != 0) && (err != -ENOTCONN)) {
		LOG_WRN("Failed to send result over NUS, error = %d", err);
	}
#endif
}

int main(void)
{
	int err;
	nrf_edgeai_err_t res;

	err = uxsm_init();
	__ASSERT(err == 0, "Failed to initialize UX state manager");

	nrf_edgeai_t *p_model = nrf_edgeai_init_model();

	__ASSERT_NO_MSG(p_model != NULL);

#if IS_ENABLED(CONFIG_APP_BLE_MDS)
	err = model_obsv_init(p_model);
	__ASSERT(err == 0, "Failed to initialize model observability (err %d)", err);

	err = ble_mds_init(uxsm_set_ble_connected);
	__ASSERT(err == 0, "Failed to initialize BLE MDS service (err %d)", err);

#else
	err = ble_nus_init(uxsm_set_ble_connected);
	__ASSERT(err == 0, "Failed to initialize BLE NUS service (err %d)", err);
#endif

	err = accel_init(accel_data_ready_cb);
	__ASSERT(err == 0, "Failed to initialize accelerometer (err %d)", err);

	float accel[ACCEL_AXES_NUM];
	flt32_t input_data[NRF_EDGEAI_INPUT_DATA_LEN];

	for (;;) {
		/* Wait for the semaphore to be released by accelerometer data ready timer */
		k_sem_take(&accel_data_ready_sem, K_FOREVER);

		if (accel_read(accel) != 0) {
			continue;
		}

		/* accel = ax, ay, az in m/s^2 */
		input_data[0] =
			sqrtf(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]);

		res = nrf_edgeai_feed_inputs(p_model, (void *)input_data,
					     NRF_EDGEAI_INPUT_DATA_LEN);

		if (res != NRF_EDGEAI_ERR_SUCCESS) {
			/* INPROGRESS is expected until INPUT_WINDOW_SHIFT samples are collected */
			if (res != NRF_EDGEAI_ERR_INPROGRESS) {
				LOG_WRN("Failed to feed inputs, error = %d", (int)res);
			}
			continue;
		}

		/* Input window is full: run the model */
		res = nrf_edgeai_run_inference(p_model);

		if (res != NRF_EDGEAI_ERR_SUCCESS) {
			LOG_WRN("Failed to run inference, error = %d", (int)res);
			continue;
		}

		handle_inference_result(p_model);
	}

	return 0;
}
