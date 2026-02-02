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

#include "ble/hid/ble_hid.h"
#include "inference_postprocessing.h"

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/hci.h>
#include <bluetooth/services/nus.h>
#include <zephyr/settings/settings.h>

#define NRF_EDGEAI_INPUT_DATA_LEN (ACCEL_AXIS_NUM + GYRO_AXIS_NUM)

#define BLINK_LED_TIMER_PERIOD_MS (30)
#define LED_MAX_BRIGHTNESS (0.2f)
#define LED_BLINK_CHANGE_BRIGHTNESS_STEP (0.005f)

LOG_MODULE_REGISTER(main);


/**
 * @brief Application remote control mode
 *
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

static void hw_modules_init(void);
static void led_glowing_timer_handler(struct k_timer *timer);
static void imu_data_ready_cb(void);
static void ble_connection_cb(bool connected);
static void button_click_handler(bool pressed);

static void send_bt_keyboard_key(const class_label_t class_label);
static bool should_act_on_prediction(const class_label_t class_label, const char *class_name);
static void log_prediction_message(const char *class_name, const float probability);
static void handle_inference_result(nrf_edgeai_t *model);
static int data_collection_ble_init(void);
static int data_collection_ble_send(const int16_t *input_data);

/* Work queue processing function declarations */
static void led_update_work_handler(struct k_work *work);
static void button_work_handler(struct k_work *work);

static bool ble_connected;
static app_remotectrl_mode_t keyboard_ctrl_mode = APP_REMOTECTRL_MODE_MUSIC;
static struct k_sem imu_data_ready_sem;

static struct bt_conn *nus_conn;
static bool nus_send_enabled;

static const struct bt_data nus_ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

static const struct bt_data nus_sd[] = {
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_NUS_VAL),
};

/* Work queue items for deferring interrupt context LED operations to thread context */
static struct k_work led_update_work;
static struct k_work button_work;
static nrf_edgeai_t *p_model;

K_TIMER_DEFINE(led_timer, led_glowing_timer_handler, NULL);

int main(void)
{
	/* Initialize Board Support Package */
	hw_modules_init();

	/* Get generated user model runtime context */
	p_model = nrf_edgeai_user_model();
	__ASSERT_NO_MSG(p_model != NULL);
	__ASSERT_NO_MSG(nrf_edgeai_is_runtime_compatible(p_model));

	/* Initialize nRF Edge AI library */
	nrf_edgeai_err_t res;

	res = nrf_edgeai_init(p_model);
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

		/* Read IMU sensor data sample */
		if (imu_read(&imu_data) != STATUS_SUCCESS) {
			continue;
		}

		input_data[0] = imu_data.accel[0].raw;
		input_data[1] = imu_data.accel[1].raw;
		input_data[2] = imu_data.accel[2].raw;
		input_data[3] = imu_data.gyro[0].raw;
		input_data[4] = imu_data.gyro[1].raw;
		input_data[5] = imu_data.gyro[2].raw;
		/* Feed and prepare raw sensor inputs for the model inference */
		if (IS_ENABLED(CONFIG_DATA_COLLECTION_MODE)) {
			if (IS_ENABLED(CONFIG_DATA_COLLECTION_BLE_NUS)) {
				(void)data_collection_ble_send(input_data);
			} else {
				printk("%d,%d,%d,%d,%d,%d\r\n", input_data[0], input_data[1],
					input_data[2], input_data[3], input_data[4], input_data[5]);
			}
		} else {
			res = nrf_edgeai_feed_inputs(p_model, input_data, NRF_EDGEAI_INPUT_DATA_LEN);

			/* Check if input data window is ready for inference */
			if (res == NRF_EDGEAI_ERR_SUCCESS) {
				/* Run Neuton model inference */
				res = nrf_edgeai_run_inference(p_model);

				/* Handle inference results if the prediction was successful */
				if (res == NRF_EDGEAI_ERR_SUCCESS) {
					handle_inference_result(p_model);
				}
			}
		}
	}

	return 0;
}

static void hw_modules_init(void)
{
	int ret;

	/* Initialize work queues */
	k_work_init(&led_update_work, led_update_work_handler);
	k_work_init(&button_work, button_work_handler);

	/* Initialize LEDs */
	ret = led_init();
	if (ret != 0) {
		LOG_ERR("Failed to initialize LEDs module, error = %d", ret);
	}
	k_timer_start(&led_timer, K_MSEC(BLINK_LED_TIMER_PERIOD_MS),
		      K_MSEC(BLINK_LED_TIMER_PERIOD_MS));

	/* Initialize user button */
	ret = button_init();
	if (ret != 0) {
		LOG_ERR("Failed to initialize user button, error = %d", ret);
	}
	button_reg_click_handler(button_click_handler);

	/* Initialize IMU sensor  */
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
	k_sem_init(&imu_data_ready_sem, 0, 1); /* Initial count 0, max count 1 */

	/* Initialize BLE HID profile */
	if (IS_ENABLED(CONFIG_DATA_COLLECTION_BLE_NUS)) {
		ret = data_collection_ble_init();
		if (ret != 0) {
			LOG_ERR("Failed to initialize BLE NUS service");
		}
	} else {
		ret = ble_hid_init(ble_connection_cb);
		if (ret != 0) {
			LOG_ERR("Failed to initialize BLE HID service");
		}
	}
}

static void led_glowing_timer_handler(struct k_timer *timer)
{
	(void)timer;

	/* Always submit to work queue to avoid interrupt context issues */
	k_work_submit(&led_update_work);
}

/* Work queue processing function implementations */

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
	/* Handle button logic in thread context */
	keyboard_ctrl_mode ^= 1;
	led_off();
}

static void button_click_handler(bool pressed)
{
	if (pressed) {
		/* In interrupt context, only submit work to queue */
		k_work_submit(&button_work);
	}
}

static void ble_connection_cb(bool connected)
{
	ble_connected = connected;
	led_off();
}

static void imu_data_ready_cb(void)
{
	k_sem_give(&imu_data_ready_sem);
}

static void nus_send_enabled_cb(enum bt_nus_send_status status)
{
	nus_send_enabled = (status == BT_NUS_SEND_STATUS_ENABLED);
}

static void nus_connected(struct bt_conn *conn, uint8_t err)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	if (err) {
		LOG_ERR("NUS connection failed to %s (%u)", addr, err);
		return;
	}

	if (!nus_conn) {
		nus_conn = bt_conn_ref(conn);
	}

	ble_connected = true;
	led_off();
	LOG_INF("NUS connected %s", addr);
}

static void nus_disconnected(struct bt_conn *conn, uint8_t reason)
{
	char addr[BT_ADDR_LE_STR_LEN];
	int err;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	LOG_INF("NUS disconnected from %s (reason 0x%02x)", addr, reason);

	if (nus_conn == conn) {
		bt_conn_unref(nus_conn);
		nus_conn = NULL;
	}

	nus_send_enabled = false;
	ble_connected = false;
	led_off();

	err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, nus_ad, ARRAY_SIZE(nus_ad),
			      nus_sd, ARRAY_SIZE(nus_sd));
	if (err) {
		LOG_ERR("NUS advertising failed to start (err %d)", err);
	}
}

static struct bt_nus_cb nus_cb = {
	.send_enabled = nus_send_enabled_cb,
};

static struct bt_conn_cb nus_conn_callbacks = {
	.connected = nus_connected,
	.disconnected = nus_disconnected,
};

static int data_collection_ble_init(void)
{
	int err;

	err = bt_enable(NULL);
	if (err) {
		LOG_ERR("Bluetooth init failed (err %d)", err);
		return err;
	}

	if (IS_ENABLED(CONFIG_SETTINGS)) {
		settings_load();
	}

	err = bt_nus_init(&nus_cb);
	if (err) {
		LOG_ERR("NUS init failed (err %d)", err);
		return err;
	}

	bt_conn_cb_register(&nus_conn_callbacks);

	err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, nus_ad, ARRAY_SIZE(nus_ad),
			      nus_sd, ARRAY_SIZE(nus_sd));
	if (err) {
		LOG_ERR("NUS advertising failed to start (err %d)", err);
		return err;
	}

	LOG_INF("NUS advertising successfully started");
	return 0;
}

static int data_collection_ble_send(const int16_t *input_data)
{
	char buffer[64];
	int len;
	uint32_t mtu;

	__ASSERT_NO_MSG(input_data != NULL);

	if (!nus_conn || !nus_send_enabled) {
		return -ENOTCONN;
	}

	len = snprintf(buffer, sizeof(buffer), "%d,%d,%d,%d,%d,%d\r\n",
		       input_data[0], input_data[1], input_data[2],
		       input_data[3], input_data[4], input_data[5]);
	if ((len <= 0) || (len >= sizeof(buffer))) {
		return -EINVAL;
	}

	mtu = bt_nus_get_mtu(nus_conn);
	if (len > mtu) {
		return -EMSGSIZE;
	}

	return bt_nus_send(nus_conn, (const uint8_t *)buffer, (uint16_t)len);
}

static void handle_inference_result(nrf_edgeai_t *model)
{
	uint16_t predicted_target;
	const flt32_t *p_probabilities;

	__ASSERT_NO_MSG(model != NULL);

	/* Predicted class */
	predicted_target = model->decoded_output.classif.predicted_class;
	/* Probabilities pointer depend on model output quantization setting */
	p_probabilities = model->decoded_output.classif.probabilities.p_f32;
	__ASSERT_NO_MSG(p_probabilities != NULL);

	LOG_DBG("Predicted target: %d, Probability: %f",
		predicted_target, (double)p_probabilities[predicted_target]);

	prediction_ctx_t result =
		inference_postprocess(predicted_target,
				      p_probabilities[predicted_target]);
	const char *class_name =
		inference_get_class_name((class_label_t)result.target);

	if (should_act_on_prediction((class_label_t)result.target, class_name)) {
		log_prediction_message(class_name, result.probability);
		send_bt_keyboard_key((class_label_t)result.target);
	}
}

static bool should_act_on_prediction(const class_label_t class_label, const char *class_name)
{
	static const uint32_t PREDICTION_TIMEOUT_MS = 800U;
	static uint32_t last_prediction_time_ms;
	uint32_t current_time_ms;

	if (class_label <= CLASS_LABEL_UNKNOWN) {
		return false;
	}

	current_time_ms = k_uptime_get();

	/** For classes CLASS_LABEL_ROTATION_RIGHT &
	 * CLASS_LABEL_ROTATION_LEFT there is no timeout,
	 * since the movements must be repetitive in time
	 */
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


static void send_bt_keyboard_key(const class_label_t class_label)
{
	static const ble_hid_key_t LABEL_VS_KEY_BY_MODE[2][8] = {
		[APP_REMOTECTRL_MODE_PRESENTATION] = {
			/* In case no key is supposed to be send BLE_HID_KEYS_count is used */
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
			/* In case no key is supposed to be send BLE_HID_KEYS_count is used */
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
