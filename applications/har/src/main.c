/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stddef.h>
#include <string.h>

#include <zephyr/drivers/sensor.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai_user_model.h>

#include "activity_led.h"
#include "button/button.h"
#include "inference_postprocessing.h"
#include "sensor/imu/imu.h"

#if IS_ENABLED(CONFIG_BLE_NUS_OUTPUT)
#include "ble/nus/ble_nus.h"
#endif
#include "ble/ble_common.h"

LOG_MODULE_REGISTER(main);

#define NRF_EDGEAI_INPUT_DATA_LEN (ACCEL_AXIS_NUM + GYRO_AXIS_NUM)
#define IMU_SAMPLE_RATE_HZ (50)
#define HAR_INPUT_WINDOW_SAMPLES (128)

BUILD_ASSERT(HAR_INPUT_WINDOW_SAMPLES > 0);

typedef struct accel_window_s {
	float sum_g[ACCEL_AXIS_NUM];
	float samples_g[HAR_INPUT_WINDOW_SAMPLES][ACCEL_AXIS_NUM];
	int index;
	int count;
} accel_window_t;

static struct k_sem imu_data_ready_sem;
static accel_window_t accel_window;
static nrf_edgeai_t *p_model;

static void imu_data_ready_cb(void)
{
	k_sem_give(&imu_data_ready_sem);
}

static void accel_window_reset(void)
{
	memset(&accel_window, 0, sizeof(accel_window));
}

static void accel_window_push(float x_g, float y_g, float z_g)
{
	const float sample_g[ACCEL_AXIS_NUM] = {x_g, y_g, z_g};

	if (accel_window.count == HAR_INPUT_WINDOW_SAMPLES) {
		for (int axis = 0; axis < ACCEL_AXIS_NUM; axis++) {
			accel_window.sum_g[axis] -= accel_window.samples_g[accel_window.index][axis];
		}
	} else {
		accel_window.count++;
	}

	for (int axis = 0; axis < ACCEL_AXIS_NUM; axis++) {
		accel_window.samples_g[accel_window.index][axis] = sample_g[axis];
		accel_window.sum_g[axis] += sample_g[axis];
	}

	accel_window.index = (accel_window.index + 1) % HAR_INPUT_WINDOW_SAMPLES;
}

static void accel_window_get_average(float *x_g, float *y_g, float *z_g)
{
	__ASSERT_NO_MSG(accel_window.count > 0);

	float inv_count = 1.0f / (float)accel_window.count;

	*x_g = accel_window.sum_g[0] * inv_count;
	*y_g = accel_window.sum_g[1] * inv_count;
	*z_g = accel_window.sum_g[2] * inv_count;
}

static void on_button_click(button_click_t click)
{
	if (click != BUTTON_CLICK_SHORT) {
		return;
	}

	LOG_INF("next");
	accel_window_reset();

#if IS_ENABLED(CONFIG_BLE_NUS_OUTPUT)
	int err = ble_nus_send_message("next");

	if (err != 0) {
		LOG_WRN("Failed to send phase marker over NUS (err %d)", err);
	}
#endif
}

static void hw_modules_init(void)
{
	int ret;
	imu_config_t imu_config = {
		.accel_fs_g = IMU_ACCEL_SCALE_2G,
		.gyro_fs_dps = IMU_GYRO_SCALE_1000DPS,
		.data_rate_hz = IMU_SAMPLE_RATE_HZ,
	};

	ret = activity_led_init();
	if (ret != 0) {
		LOG_ERR("Failed to initialize activity LED module (err %d)", ret);
	}

	ret = button_init();
	if (ret != 0) {
		LOG_ERR("Failed to initialize button module (err %d)", ret);
	}

	status_t status = imu_init(&imu_config, imu_data_ready_cb);

	if (status != STATUS_SUCCESS) {
		LOG_ERR("Failed to initialize IMU sensor, error = %d", (int)status);
		__ASSERT_NO_MSG(false);
	}

	k_sem_init(&imu_data_ready_sem, 0, 1);
	accel_window_reset();
	button_reg_click_handler(on_button_click);
	ble_common_init();

#if IS_ENABLED(CONFIG_BLE_NUS_OUTPUT)
	ret = ble_nus_init();
	if (ret != 0) {
		LOG_ERR("Failed to initialize BLE NUS service (err %d)", ret);
	}
#endif
}

static void publish_classification(class_label_t class_label, float probability)
{
	const char *class_name = inference_get_class_name(class_label);
	int probability_pct = (int)(100.0f * probability);
	float accel_x_g;
	float accel_y_g;
	float accel_z_g;

	accel_window_get_average(&accel_x_g, &accel_y_g, &accel_z_g);

	LOG_INF("Activity: %s (%d %%) accel avg [g]: %.3f, %.3f, %.3f", class_name,
		probability_pct, (double)accel_x_g, (double)accel_y_g, (double)accel_z_g);
	activity_led_show_class(class_label);

#if IS_ENABLED(CONFIG_BLE_NUS_OUTPUT)
	if (ble_common_is_connected()) {
		int err = ble_nus_send_classification(NULL, class_name, probability_pct, accel_x_g,
						     accel_y_g, accel_z_g);

		if (err != 0 && err != -ENOTCONN) {
			LOG_WRN("Failed to send classification over NUS (err %d)", err);
		}
	}
#endif
}

#if IS_ENABLED(CONFIG_HAR_LOG_RAW_PREDICTIONS)
static void report_raw_prediction(uint16_t predicted_target, float probability)
{
	const char *class_name = inference_get_class_name((class_label_t)predicted_target);
	int probability_pct = (int)(100.0f * probability);
	float accel_x_g;
	float accel_y_g;
	float accel_z_g;

	accel_window_get_average(&accel_x_g, &accel_y_g, &accel_z_g);

	LOG_INF("Raw: %s (%d %%) accel avg [g]: %.3f, %.3f, %.3f", class_name, probability_pct,
		(double)accel_x_g, (double)accel_y_g, (double)accel_z_g);

#if IS_ENABLED(CONFIG_BLE_NUS_OUTPUT)
	if (ble_common_is_connected()) {
		int err = ble_nus_send_classification("RAW,", class_name, probability_pct,
						      accel_x_g, accel_y_g, accel_z_g);

		if (err != 0 && err != -ENOTCONN) {
			LOG_WRN("Failed to send raw prediction over NUS (err %d)", err);
		}
	}
#endif
}
#endif

static void handle_inference_result(nrf_edgeai_t *model)
{
	uint16_t predicted_target;
	const flt32_t *p_probabilities;
	class_label_t class_label;
	float probability;

	__ASSERT_NO_MSG(model != NULL);

	predicted_target = model->decoded_output.classif.predicted_class;
	p_probabilities = model->decoded_output.classif.probabilities.p_f32;
	__ASSERT_NO_MSG(p_probabilities != NULL);

#if IS_ENABLED(CONFIG_HAR_INFERENCE_POSTPROCESSING)
	report_raw_prediction(predicted_target, p_probabilities[predicted_target]);

	prediction_ctx_t result =
		inference_postprocess(predicted_target, p_probabilities[predicted_target]);

	if (result.target >= CLASS_LABEL_COUNT) {
		return;
	}

	class_label = (class_label_t)result.target;
	probability = result.probability;
#else
	if (predicted_target >= CLASS_LABEL_COUNT) {
		LOG_INF("Activity: UNKNOWN");
		return;
	}

	class_label = (class_label_t)predicted_target;
	probability = p_probabilities[predicted_target];
#endif

	publish_classification(class_label, probability);
}

static void execute_inference(flt32_t *input_data)
{
	nrf_edgeai_err_t res;

	res = nrf_edgeai_feed_inputs(p_model, (void *)input_data, NRF_EDGEAI_INPUT_DATA_LEN);

	if (res == NRF_EDGEAI_ERR_SUCCESS) {
		res = nrf_edgeai_run_inference(p_model);

		if (res == NRF_EDGEAI_ERR_SUCCESS) {
			handle_inference_result(p_model);
		} else {
			LOG_WRN("Failed to run inference, error = %d", (int)res);
		}
	} else if (res != NRF_EDGEAI_ERR_INPROGRESS) {
		LOG_WRN("Failed to feed inputs, error = %d", (int)res);
	}
}

int main(void)
{
	hw_modules_init();

	p_model = nrf_edgeai_user_model();
	__ASSERT_NO_MSG(p_model != NULL);
	__ASSERT_NO_MSG(nrf_edgeai_is_runtime_compatible(p_model));

	__maybe_unused nrf_edgeai_err_t res = nrf_edgeai_init(p_model);

	__ASSERT_NO_MSG(res == NRF_EDGEAI_ERR_SUCCESS);
	__ASSERT_NO_MSG(p_model->input.window_size == HAR_INPUT_WINDOW_SAMPLES);

	imu_data_t imu_data = {0};
	flt32_t input_data[NRF_EDGEAI_INPUT_DATA_LEN];

	for (;;) {
		k_sem_take(&imu_data_ready_sem, K_FOREVER);

		if (imu_read(&imu_data) != STATUS_SUCCESS) {
			continue;
		}

		accel_window_push(imu_data.accel[0].g, imu_data.accel[1].g, imu_data.accel[2].g);

		/* Model expects accelerometer in g and gyroscope in rad/s. */
		input_data[0] = (flt32_t)(imu_data.accel[0].g);
		input_data[1] = (flt32_t)(imu_data.accel[1].g);
		input_data[2] = (flt32_t)(imu_data.accel[2].g);
		input_data[3] = (flt32_t)imu_data.gyro[0].rad_s;
		input_data[4] = (flt32_t)imu_data.gyro[1].rad_s;
		input_data[5] = (flt32_t)imu_data.gyro[2].rad_s;

		LOG_DBG("Accelerometer [G]: %f, %f, %f",
			(double)input_data[0], (double)input_data[1], (double)input_data[2]);
		LOG_DBG("Gyroscope [rad/s]: %f, %f, %f",
			(double)input_data[3], (double)input_data[4], (double)input_data[5]);

		execute_inference(input_data);
	}

	return 0;
}
