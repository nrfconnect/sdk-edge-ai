/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <soc.h>
#include <stddef.h>
#include <string.h>
#include <zephyr/types.h>
#include <zephyr/sys/util.h>

#include <stdio.h>
#include <errno.h>
#include <zephyr/device.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/usb/usb_device.h>

#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai_user_model.h>

#include "hw_modules/button/button.h"
#include "hw_modules/led/led.h"
#include "hw_modules/sensor/imu/imu.h"

#include "inference_postprocessing.h"

#if defined(CONFIG_BLE_MODE_HID)
#include "ble/hid/ble_hid.h"
#elif defined(CONFIG_BLE_MODE_NUS)
#include "ble/nus/ble_nus.h"
#elif defined(CONFIG_BLE_MODE_GATT_CUSTOM)
#include "ble/gatt/ble_gatt.h"
#endif

#define NRF_EDGEAI_INPUT_DATA_LEN (ACCEL_AXIS_NUM + GYRO_AXIS_NUM)

#define BLINK_LED_TIMER_PERIOD_MS (30)
#define LED_MAX_BRIGHTNESS (0.2f)
#define LED_BLINK_CHANGE_BRIGHTNESS_STEP (0.005f)

LOG_MODULE_REGISTER(main);

/**
 * @brief Application remote control mode
 */
typedef enum app_remotectrl_mode_e {
	/**
	 * APP_REMOTECTRL_MODE_PRESENTATION used for slides control:
	 * - Swipe Right                       = KEY_ARROW_RIGHT (Next slide)
	 * - Swipe Left                        = KEY_ARROW_LEFT  (Previous slide)
	 * - Double Tap                        = KEY_F5          (Enter fullscreen mode)
	 * - Double Thumb                      = KEY_ESC         (Exit fullscreen mode)
	 * - Rotation Clockwise (Right)        = Not used
	 * - Rotation Counter Clockwise (Left) = Not used
	 *
	 */
	APP_REMOTECTRL_MODE_PRESENTATION = 0,

	/**
	 * NEUTON_REMOTE_CTRL_MUSIC used for music/media control:
	 * - Swipe Right                       = KEY_MEDIA_NEXT_TRACK  (Next track)
	 * - Swipe Left                        = KEY_MEDIA_PREV_TRACK  (Previous track)
	 * - Double Tap                        = KEY_MEDIA_PLAYPAUSE   (Play/Pause)
	 * - Double Thumb                      = KEY_MEDIA_MUTE        (Mute)
	 * - Rotation Clockwise (Right)        = KEY_MEDIA_VOLUMEUP    (Volume Up)
	 * - Rotation Counter Clockwise (Left) = KEY_MEDIA_VOLUMEDOWN  (Volume Down)
	 */
	APP_REMOTECTRL_MODE_MUSIC
} app_remotectrl_mode_t;

typedef int (*led_set_func_t)(float brightness);

static void send_imu_data(int16_t *input_data);
static void execute_inference(int16_t *input_data);
static void hw_modules_init(void);
static void led_glowing_timer_handler(struct k_timer *timer);
static void handle_inference_result(nrf_edgeai_t *model);

static bool ble_connected;
static app_remotectrl_mode_t keyboard_ctrl_mode = APP_REMOTECTRL_MODE_MUSIC;
static struct k_sem imu_data_ready_sem;


/* Work queue items for deferring interrupt context LED operations to thread context */
static struct k_work led_update_work;
static struct k_work button_work;
static nrf_edgeai_t *p_model;

K_TIMER_DEFINE(led_timer, led_glowing_timer_handler, NULL);

int main(void)
{
	hw_modules_init();

	p_model = nrf_edgeai_user_model();
	__ASSERT_NO_MSG(p_model != NULL);
	__ASSERT_NO_MSG(nrf_edgeai_is_runtime_compatible(p_model));

	nrf_edgeai_err_t res = nrf_edgeai_init(p_model);

	__ASSERT_NO_MSG(res == NRF_EDGEAI_ERR_SUCCESS);

	nrf_edgeai_rt_version_t version = nrf_edgeai_runtime_version();

	LOG_INF("nRF Edge AI Gestures Recognition Demo:");
	LOG_INF("nRF Edge AI Runtime Version: %d.%d.%d",
		version.field.major, version.field.minor, version.field.patch);
	LOG_INF("nRF Edge AI Lab Solution id: %s",
		nrf_edgeai_solution_id_str(p_model));

	imu_data_t imu_data = {0};
	int16_t input_data[NRF_EDGEAI_INPUT_DATA_LEN];

	for (;;) {
		/* Wait for the semaphore to be released by IMU data ready interrupt */
		k_sem_take(&imu_data_ready_sem, K_FOREVER);

		if (imu_read(&imu_data) != STATUS_SUCCESS) {
			continue;
		}

		input_data[0] = imu_data.accel[0].raw;
		input_data[1] = imu_data.accel[1].raw;
		input_data[2] = imu_data.accel[2].raw;
		input_data[3] = imu_data.gyro[0].raw;
		input_data[4] = imu_data.gyro[1].raw;
		input_data[5] = imu_data.gyro[2].raw;

		if (IS_ENABLED(CONFIG_DATA_COLLECTION_MODE)) {
			send_imu_data(input_data);
		} else {
			execute_inference(input_data);
		}
	}

	return 0;
}

static void imu_data_ready_cb(void)
{
	k_sem_give(&imu_data_ready_sem);
}

#if !IS_ENABLED(CONFIG_BLE_MODE_NONE)
static void ble_connection_cb(bool connected)
{
	ble_connected = connected;
	led_off();
}
#endif

static void led_glowing_timer_handler(struct k_timer *timer)
{
	(void)timer;
	k_work_submit(&led_update_work);
}

static void led_update_work_handler(struct k_work *work)
{
	static const led_set_func_t LED_VS_KEYBOARD_MODE[] = {
		[APP_REMOTECTRL_MODE_PRESENTATION] = led_set_led2,
		[APP_REMOTECTRL_MODE_MUSIC] = led_set_led1,
	};

	static bool rising = true;
	static float brightness;

	if (ble_connected) {
		__ASSERT_NO_MSG(keyboard_ctrl_mode < ARRAY_SIZE(LED_VS_KEYBOARD_MODE));
		led_set_func_t led_set = LED_VS_KEYBOARD_MODE[keyboard_ctrl_mode];

		__ASSERT_NO_MSG(led_set != NULL);
		led_set(brightness);
	} else {
		led_set_led0(brightness);
	}

	if (rising) {
		brightness += LED_BLINK_CHANGE_BRIGHTNESS_STEP;
		if (brightness >= LED_MAX_BRIGHTNESS) {
			rising = false;
		}
	} else {
		brightness -= LED_BLINK_CHANGE_BRIGHTNESS_STEP;
		if (brightness <= 0) {
			rising = true;
		}
	}
}

static void button_work_handler(struct k_work *work)
{
	keyboard_ctrl_mode ^= 1;
	led_off();
}

static void button_click_handler(bool pressed)
{
	if (pressed) {
		k_work_submit(&button_work);
	}
}

static void hw_modules_init(void)
{
	int ret;

	k_work_init(&led_update_work, led_update_work_handler);
	k_work_init(&button_work, button_work_handler);

	ret = led_init();
	if (ret != 0) {
		LOG_ERR("Failed to initialize LEDs module, error = %d", ret);
	}
	k_timer_start(&led_timer, K_MSEC(BLINK_LED_TIMER_PERIOD_MS),
		     K_MSEC(BLINK_LED_TIMER_PERIOD_MS));

	ret = button_init();
	if (ret != 0) {
		LOG_ERR("Failed to initialize user button, error = %d", ret);
	}
	button_reg_click_handler(button_click_handler);

	imu_config_t imu_config = {
		.accel_fs_g = IMU_ACCEL_SCALE_4G,
		.gyro_fs_dps = IMU_GYRO_SCALE_1000DPS,
		.data_rate_hz = 100
	};

	status_t status = imu_init(&imu_config, imu_data_ready_cb);

	if (status != STATUS_SUCCESS) {
		LOG_ERR("Failed to initialize IMU sensor, error = %d", (int)status);
		__ASSERT_NO_MSG(false);
	}
	k_sem_init(&imu_data_ready_sem, 0, 1);

#if IS_ENABLED(CONFIG_BLE_MODE_HID)
	ret = ble_hid_init(ble_connection_cb);
	if (ret != 0) {
		LOG_ERR("Failed to initialize BLE HID service");
	}
#elif IS_ENABLED(CONFIG_BLE_MODE_NUS)
	ret = ble_nus_init(ble_connection_cb);
	if (ret != 0) {
		LOG_ERR("Failed to initialize BLE NUS service");
	}
#elif IS_ENABLED(CONFIG_BLE_MODE_GATT_CUSTOM)
	ret = ble_gatt_init(ble_connection_cb, NULL);
	if (ret != 0) {
		LOG_ERR("Failed to initialize BLE GATT service");
	}
#endif
}

#if IS_ENABLED(CONFIG_BLE_MODE_HID)
static void send_bt_keyboard_key(const class_label_t class_label)
{
	static const ble_hid_key_t LABEL_VS_KEY_BY_MODE[2][8] = {
		[APP_REMOTECTRL_MODE_PRESENTATION] = {
			[CLASS_LABEL_IDLE] = BLE_HID_KEYS_count,
			[CLASS_LABEL_UNKNOWN] = BLE_HID_KEYS_count,
			[CLASS_LABEL_SWIPE_RIGHT] = BLE_HID_KEY_ARROW_RIGHT,
			[CLASS_LABEL_SWIPE_LEFT] = BLE_HID_KEY_ARROW_LEFT,
			[CLASS_LABEL_DOUBLE_SHAKE] = BLE_HID_KEY_F5,
			[CLASS_LABEL_DOUBLE_THUMB] = BLE_HID_KEY_ESC,
			[CLASS_LABEL_ROTATION_RIGHT] = BLE_HID_KEYS_count,
			[CLASS_LABEL_ROTATION_LEFT] = BLE_HID_KEYS_count,
		},
		[APP_REMOTECTRL_MODE_MUSIC] = {
			[CLASS_LABEL_IDLE] = BLE_HID_KEYS_count,
			[CLASS_LABEL_UNKNOWN] = BLE_HID_KEYS_count,
			[CLASS_LABEL_SWIPE_RIGHT] = BLE_HID_KEY_MEDIA_NEXT_TRACK,
			[CLASS_LABEL_SWIPE_LEFT] = BLE_HID_KEY_MEDIA_PREV_TRACK,
			[CLASS_LABEL_DOUBLE_SHAKE] = BLE_HID_KEY_MEDIA_PLAY_PAUSE,
			[CLASS_LABEL_DOUBLE_THUMB] = BLE_HID_KEY_MEDIA_MUTE,
			[CLASS_LABEL_ROTATION_RIGHT] = BLE_HID_KEY_MEDIA_VOLUME_UP,
			[CLASS_LABEL_ROTATION_LEFT] = BLE_HID_KEY_MEDIA_VOLUME_DOWN,
		},
	};

	__ASSERT_NO_MSG(class_label < CLASS_LABEL_COUNT);
	__ASSERT_NO_MSG(keyboard_ctrl_mode < ARRAY_SIZE(LABEL_VS_KEY_BY_MODE));
	ble_hid_send_key(LABEL_VS_KEY_BY_MODE[keyboard_ctrl_mode][class_label]);
}
#endif

#if IS_ENABLED(CONFIG_BLE_MODE_GATT_CUSTOM)
static void ble_send_recognized_class(const class_label_t class_label, size_t probability_pct)
{
	static char send_buff[15];

	int message_len = snprintf(send_buff, sizeof(send_buff), "%d,%d",
					(int)class_label, (int)probability_pct);

	__ASSERT_NO_MSG(message_len > 0 && message_len < sizeof(send_buff));
	ble_gatt_send_raw_data((const uint8_t *)send_buff, (size_t)message_len);
}
#endif

static void execute_inference(int16_t *input_data)
{
	nrf_edgeai_err_t res;

	res = nrf_edgeai_feed_inputs(p_model, input_data, NRF_EDGEAI_INPUT_DATA_LEN);

	if (res == NRF_EDGEAI_ERR_SUCCESS) {
		res = nrf_edgeai_run_inference(p_model);

		if (res == NRF_EDGEAI_ERR_SUCCESS) {
			handle_inference_result(p_model);
		} else {
			LOG_WRN("Failed to run inference, error = %d", (int)res);
		}
	/* INPROGRESS is expected 32/33 times, as we have 33 samples in the INPUT_WINDOW_SHIFT */
	} else if (res != NRF_EDGEAI_ERR_INPROGRESS) {
		LOG_WRN("Failed to feed inputs, error = %d", (int)res);
	}
}

static void send_imu_data(int16_t *input_data)
{
#if IS_ENABLED(CONFIG_BLE_MODE_NUS)
	(void)ble_nus_send(input_data);
#else
	printk("%d,%d,%d,%d,%d,%d\r\n", input_data[0], input_data[1],
	       input_data[2], input_data[3], input_data[4], input_data[5]);
#endif
}

static bool should_act_on_prediction(const class_label_t class_label)
{
	static const uint32_t PREDICTION_TIMEOUT_MS = 800U;
	static uint32_t last_prediction_time_ms;
	uint32_t current_time_ms;

	if (class_label <= CLASS_LABEL_UNKNOWN) {
		return false;
	}

	current_time_ms = k_uptime_get();

	if ((class_label == CLASS_LABEL_ROTATION_RIGHT) ||
	    (class_label == CLASS_LABEL_ROTATION_LEFT) ||
	    (current_time_ms - last_prediction_time_ms) > PREDICTION_TIMEOUT_MS) {
		last_prediction_time_ms = current_time_ms;
		return true;
	}

	return false;
}

static void log_prediction_message(const char *class_name, const float probability)
{
	__ASSERT_NO_MSG(class_name != NULL);
	LOG_INF("Predicted class: %s, with probability %d %%", class_name,
		(int)(100 * probability));
}

static void handle_inference_result(nrf_edgeai_t *model)
{
	uint16_t predicted_target;
	const flt32_t *p_probabilities;

	__ASSERT_NO_MSG(model != NULL);

	predicted_target = model->decoded_output.classif.predicted_class;
	p_probabilities = model->decoded_output.classif.probabilities.p_f32;
	__ASSERT_NO_MSG(p_probabilities != NULL);

	LOG_DBG("Predicted target: %d, Probability: %f",
		predicted_target, (double)p_probabilities[predicted_target]);

	prediction_ctx_t result = inference_postprocess(predicted_target,
							p_probabilities[predicted_target]);
	if (should_act_on_prediction((class_label_t)result.target)) {
		const char *class_name = inference_get_class_name((class_label_t)result.target);
		log_prediction_message(class_name, result.probability);
#if IS_ENABLED(CONFIG_BLE_MODE_HID)
		send_bt_keyboard_key((class_label_t)result.target);
#elif IS_ENABLED(CONFIG_BLE_MODE_GATT_CUSTOM)
		ble_send_recognized_class((class_label_t)result.target,
					  (size_t)(100.0f * result.probability));
#endif
	}
}
