/*
 * Copyright (c) 2018 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <soc.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <zephyr/types.h>

#include <stdio.h>
#include <zephyr/device.h>
#include <zephyr/kernel.h>
#include <zephyr/usb/usb_device.h>

#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai_generated/nrf_edgeai_user_model.h>

#include <button/bsp_button.h>
#include <led/bsp_led.h>
#include <sensor/imu/bsp_imu.h>

#include "ble/hid/ble_hid.h"
#include "inference_postprocessing.h"
#include "app_version.h"

//////////////////////////////////////////////////////////////////////////////

#define ACCEL_AXIS_NUM (3U)
#define GYRO_AXIS_NUM (3U)
#define NRF_EDGEAI_INPUT_DATA_LEN (ACCEL_AXIS_NUM + GYRO_AXIS_NUM)

#define BLINK_LED_TIMER_PERIOD_MS (30)
#define LED_MAX_BRIGHTNESS (0.2f)
#define LED_BLINK_CHANGE_BRIGHTNESS_STEP (0.005f)

LOG_MODULE_REGISTER(main);

//////////////////////////////////////////////////////////////////////////////

/**
 * @brief Application remote control mode
 *
 */
typedef enum app_remotectrl_mode_e
{
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

//////////////////////////////////////////////////////////////////////////////

static void board_support_init_(void);
static void led_glowing_timer_handler_(struct k_timer* timer);
static void imu_data_ready_cb_(void);
static void ble_connection_cb_(bool connected);
static void button_click_handler_(bool pressed);
#ifndef CONFIG_DATA_COLLECTION_MODE
static void send_bt_keyboard_key_(const class_label_t class_label);
static void model_prediction_handler_(const class_label_t class_label, 
                                        const float probability,
                                        const char* class_name,
                                        const bool is_raw);
#endif
// Work queue processing function declarations
static void led_update_work_handler(struct k_work *work);
static void button_work_handler(struct k_work *work);

//////////////////////////////////////////////////////////////////////////////

static bool ble_connected_ = false;
static app_remotectrl_mode_t keyboard_ctrl_mode_ = APP_REMOTECTRL_MODE_MUSIC;
static struct k_sem imu_data_ready_sem_;

// Work queue items for deferring interrupt context LED operations to thread context
static struct k_work led_update_work;
static struct k_work button_work;
static nrf_edgeai_t* p_model_ = NULL;

//////////////////////////////////////////////////////////////////////////////

K_TIMER_DEFINE(led_timer_, led_glowing_timer_handler_, NULL);

//////////////////////////////////////////////////////////////////////////////

int main(void)
{
    /** Initialize Board Support Package */
    board_support_init_();

    /** Get generated user model runtime context */
    p_model_ = nrf_edgeai_user_model();
    assert(p_model_ != NULL);
    assert(nrf_edgeai_is_runtime_compatible(p_model_));

    /** Initialize nRF Edge AI library */
    nrf_edgeai_err_t res = nrf_edgeai_init(p_model_);
    assert(res == NRF_EDGEAI_ERR_SUCCESS);
    
    nrf_edgeai_rt_version_t version = nrf_edgeai_runtime_version();

    printk("nRF Edge AI Gestures Recognition Demo: \r\n");
    printk("\t Application version: %d.%d.%d\r\n", APP_VERSION_MAJOR, APP_VERSION_MINOR, APP_VERSION_PATCH);
    printk("\t nRF Edge AI Runtime Version: %d.%d.%d\r\n", version.field.major, version.field.minor, version.field.patch);
    printk("\t nRF Edge AI Lab Solution id: %s\r\n", nrf_edgeai_solution_id_str(p_model_));

    bsp_imu_data_t imu_data = {0};
    int16_t input_data[NRF_EDGEAI_INPUT_DATA_LEN];

    for (;;)
    {
        /** Wait for the semaphore to be released by IMU data ready interrupt */
        k_sem_take(&imu_data_ready_sem_, K_FOREVER);

        /** Read IMU sensor data sample */
        if (bsp_imu_read(&imu_data) != BSP_STATUS_SUCCESS)
            continue;

        input_data[0] = imu_data.accel[0].raw;
        input_data[1] = imu_data.accel[1].raw;
        input_data[2] = imu_data.accel[2].raw;
        input_data[3] = imu_data.gyro[0].raw;
        input_data[4] = imu_data.gyro[1].raw;
        input_data[5] = imu_data.gyro[2].raw;
        /** Feed and prepare raw sensor inputs for the model inference */
#if CONFIG_DATA_COLLECTION_MODE
        printk("%d,%d,%d,%d,%d,%d\r\n",  input_data[0], input_data[1], input_data[2], input_data[3], input_data[4], input_data[5]);
#else        
        res = nrf_edgeai_feed_inputs(p_model_, input_data, NRF_EDGEAI_INPUT_DATA_LEN);

        /** Check if input data window is ready for inference */
        if (res == NRF_EDGEAI_ERR_SUCCESS)
        {
            /** Run Neuton model inference */
            res = nrf_edgeai_run_inference(p_model_);

            /** Handle Neuton inference results if the prediction was
             * successful */
            if (res == NRF_EDGEAI_ERR_SUCCESS)
            {
                /** Predicted class */
                uint16_t predicted_target = p_model_->decoded_output.classif.predicted_class;
                /** Probabilities pointer depend on model output quantization setting */
                const flt32_t* p_probabilities = p_model_->decoded_output.classif.probabilities.p_f32;

                bool do_postprocessing = true;
                inference_postprocess(predicted_target,
                                      p_probabilities[predicted_target],
                                      do_postprocessing,
                                      model_prediction_handler_);
            }
        }
#endif // CONFIG_DATA_COLLECTION_MODE
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////

static void board_support_init_(void)
{
    int ret;

    /** Initialize work queues */
    k_work_init(&led_update_work, led_update_work_handler);
    k_work_init(&button_work, button_work_handler);

    /** Initialize LEDs */
    ret = bsp_led_init();
    if (ret != 0)
    {
        printk("Failed to initialize LEDs module, error = %d\n", ret);
    }
    k_timer_start(&led_timer_, K_MSEC(BLINK_LED_TIMER_PERIOD_MS), K_MSEC(BLINK_LED_TIMER_PERIOD_MS));

    /** Initialize user button */
    ret = bsp_button_init();
    if (ret != 0)
    {
        printk("Failed to initialize user button, error = %d\n", ret);
    }
    bsp_button_reg_click_handler(button_click_handler_);

    /** Initialize IMU sensor  */
    bsp_imu_config_t imu_config = 
    {
        .accel_fs_g = BSP_IMU_ACCEL_SCALE_4G,
        .gyro_fs_dps = BSP_IMU_ACCEL_SCALE_1000DPS,
        .data_rate_hz = 100
    };

    bsp_status_t status = bsp_imu_init(&imu_config, imu_data_ready_cb_);
    if (status != BSP_STATUS_SUCCESS)
    {
        printk("Failed to initialize IMU sensor, error = %d\n", (int)status);
    }
    k_sem_init(&imu_data_ready_sem_, 0, 1); // Initial count 0, max count 1

    /** Initialize BLE HID profile */
    ret = ble_hid_init(ble_connection_cb_);
    if (ret != 0)
    {
        printk("Failed to initialize BLE HID service\n");
    }
}

//////////////////////////////////////////////////////////////////////////////

static void led_glowing_timer_handler_(struct k_timer* timer)
{
    (void)timer;
    
    // Always submit to work queue to avoid interrupt context issues
    k_work_submit(&led_update_work);
}

//////////////////////////////////////////////////////////////////////////////

// Work queue processing function implementations

static void led_update_work_handler(struct k_work *work)
{
    static const led_set_func_t LED_VS_KEYBOARD_MODE[] = 
    {
        [APP_REMOTECTRL_MODE_PRESENTATION] = bsp_led_set_blue,
        [APP_REMOTECTRL_MODE_MUSIC] = bsp_led_set_green,
    };

    static bool rising = true;
    static float brightness = 0;

    if (ble_connected_)
    {
        led_set_func_t led_set = LED_VS_KEYBOARD_MODE[keyboard_ctrl_mode_];
        led_set(brightness);
    }
    else
    {
        bsp_led_set_red(brightness);
    }

    if (rising)
    {
        brightness += LED_BLINK_CHANGE_BRIGHTNESS_STEP;
        if (brightness >= LED_MAX_BRIGHTNESS)
            rising = false;
    } 
    else
    {
        brightness -= LED_BLINK_CHANGE_BRIGHTNESS_STEP;
        if (brightness <= 0)
            rising = true;
    }
}

static void button_work_handler(struct k_work *work)
{
    // Handle button logic in thread context
    keyboard_ctrl_mode_ ^= 1;
    bsp_led_off();
}

//////////////////////////////////////////////////////////////////////////////

static void button_click_handler_(bool pressed)
{
    if (pressed)
    {
        // In interrupt context, only submit work to queue
        k_work_submit(&button_work);
    }
}

//////////////////////////////////////////////////////////////////////////////

static void ble_connection_cb_(bool connected)
{
    ble_connected_ = connected;
    bsp_led_off();
}

//////////////////////////////////////////////////////////////////////////////

static void imu_data_ready_cb_(void)
{
    k_sem_give(&imu_data_ready_sem_); // Release the semaphore
}

//////////////////////////////////////////////////////////////////////////////
#ifndef CONFIG_DATA_COLLECTION_MODE
static void model_prediction_handler_(const class_label_t class_label, 
                                        const float probability,
                                        const char* class_name,
                                        const bool is_raw)
{
    static const uint32_t PREDICTION_TIMEOUT_MS = 800U;

    if (is_raw)
    {
        printk("RAW Prediction %s %d %%\r\n", class_name, (int8_t)(probability * 100.0f));
    }
    else if (class_label > CLASS_LABEL_UNKNOWN)
    {
        static uint32_t last_prediction_time_ms_ = 0;
        uint32_t current_time_ms = k_uptime_get();

        /** For classes CLASS_LABEL_ROTATION_RIGHT & CLASS_LABEL_ROTATION_LEFT there is no timeout,
         * since the movements must be repetitive in time
         */
        if ((class_label == CLASS_LABEL_ROTATION_RIGHT) ||
            (class_label == CLASS_LABEL_ROTATION_LEFT) ||
            (current_time_ms - last_prediction_time_ms_) > PREDICTION_TIMEOUT_MS)
        {
            last_prediction_time_ms_ = current_time_ms;

            printk("Predicted class: %s, with probability %d %%\r\n", class_name, (int)(100 * probability));

            send_bt_keyboard_key_(class_label);
        }
    }
}
//////////////////////////////////////////////////////////////////////////////

static void send_bt_keyboard_key_(const class_label_t class_label)
{
    static const ble_hid_key_t LABEL_VS_KEY_BY_MODE[2][8] = 
    {
        [APP_REMOTECTRL_MODE_PRESENTATION] = 
        {
            [CLASS_LABEL_IDLE] = BLE_HID_KEYS_count, // No key
            [CLASS_LABEL_UNKNOWN] = BLE_HID_KEYS_count, // No key
            [CLASS_LABEL_SWIPE_RIGHT] = BLE_HID_KEY_ARROW_RIGHT, // Next slide
            [CLASS_LABEL_SWIPE_LEFT] = BLE_HID_KEY_ARROW_LEFT, // Previous slide
            [CLASS_LABEL_DOUBLE_SHAKE] = BLE_HID_KEY_F5, // Fullscreen mode
            [CLASS_LABEL_DOUBLE_THUMB] = BLE_HID_KEY_ESC, // Exit Fullscreen mode
            [CLASS_LABEL_ROTATION_RIGHT] = BLE_HID_KEYS_count, // No key
            [CLASS_LABEL_ROTATION_LEFT] = BLE_HID_KEYS_count, // No key
        },
        [APP_REMOTECTRL_MODE_MUSIC] = 
        {
            [CLASS_LABEL_IDLE] = BLE_HID_KEYS_count, // No key
            [CLASS_LABEL_UNKNOWN] = BLE_HID_KEYS_count, // No key
            [CLASS_LABEL_SWIPE_RIGHT] = BLE_HID_KEY_MEDIA_NEXT_TRACK, // Next track
            [CLASS_LABEL_SWIPE_LEFT] = BLE_HID_KEY_MEDIA_PREV_TRACK, // Previous track
            [CLASS_LABEL_DOUBLE_SHAKE] = BLE_HID_KEY_MEDIA_PLAY_PAUSE, // Play-pause stream
            [CLASS_LABEL_DOUBLE_THUMB] = BLE_HID_KEY_MEDIA_MUTE, // Mute stream
            [CLASS_LABEL_ROTATION_RIGHT] = BLE_HID_KEY_MEDIA_VOLUME_UP, // Volume up
            [CLASS_LABEL_ROTATION_LEFT] = BLE_HID_KEY_MEDIA_VOLUME_DOWN, // Volume down
        },
    };

    ble_hid_send_key(LABEL_VS_KEY_BY_MODE[keyboard_ctrl_mode_][class_label]);
}

#endif // CONFIG_DATA_COLLECTION_MODE