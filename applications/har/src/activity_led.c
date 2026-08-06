/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "activity_led.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include "led/led.h"

LOG_MODULE_REGISTER(activity_led, CONFIG_LOG_DEFAULT_LEVEL);

#define ACTIVITY_LED_MAX_BRIGHTNESS (0.35f)
#define CONNECTED_LED_BRIGHTNESS (0.25f)
#define NOTIFY_BLINK_COUNT (3)
#define NOTIFY_BLINK_ON_MS (100)
#define NOTIFY_BLINK_OFF_MS (200)
#define NUS_TX_PULSE_MS (80)
#define NUS_TX_PULSE_BRIGHTNESS (1.0f)

typedef struct activity_color_s {
	float red;
	float green;
	float blue;
} activity_color_t;

static const activity_color_t CLASS_COLORS[] = {
	[CLASS_LABEL_WALKING] = {0.0f, 1.0f, 0.0f},
	[CLASS_LABEL_WALKING_UPSTAIRS] = {0.0f, 1.0f, 1.0f},
	[CLASS_LABEL_WALKING_DOWNSTAIRS] = {1.0f, 1.0f, 0.0f},
	[CLASS_LABEL_SITTING] = {0.0f, 0.0f, 1.0f},
	[CLASS_LABEL_STANDING] = {1.0f, 0.0f, 1.0f},
	[CLASS_LABEL_LYING] = {1.0f, 0.0f, 0.0f},
};

static const activity_color_t DISCONNECTED_COLOR = {0.15f, 0.15f, 0.15f};
static const activity_color_t CONNECTED_COLOR = {0.0f, 0.0f, 1.0f};

static bool ble_connected;
static class_label_t current_class = CLASS_LABEL_UNKNOWN;
static bool notify_blink_active;
static int notify_blink_remaining;
static activity_color_t notify_blink_color;
static float notify_blink_brightness;
static struct k_work_delayable notify_blink_work;
static struct k_work_delayable nus_tx_pulse_work;

static void apply_color(const activity_color_t *color, float brightness_scale)
{
	(void)led_set_leds(color->red * brightness_scale, color->green * brightness_scale,
			   color->blue * brightness_scale);
}

static const activity_color_t *current_display_color(void)
{
	if (!ble_connected) {
		return &DISCONNECTED_COLOR;
	}

	if (current_class < CLASS_LABEL_COUNT) {
		return &CLASS_COLORS[current_class];
	}

	return &CONNECTED_COLOR;
}

static float current_display_brightness(void)
{
	if (ble_connected && (current_class >= CLASS_LABEL_COUNT)) {
		return CONNECTED_LED_BRIGHTNESS;
	}

	return ACTIVITY_LED_MAX_BRIGHTNESS;
}

static void apply_current_state(void)
{
	apply_color(current_display_color(), current_display_brightness());
}

static void notify_blink_step(struct k_work *work)
{
	ARG_UNUSED(work);

	if (notify_blink_remaining <= 0) {
		notify_blink_active = false;
		apply_current_state();
		return;
	}

	if (notify_blink_remaining % 2 == 0) {
		apply_color(&notify_blink_color, notify_blink_brightness);
		notify_blink_remaining--;
		k_work_schedule(&notify_blink_work, K_MSEC(NOTIFY_BLINK_ON_MS));
		return;
	}

	(void)led_off();
	notify_blink_remaining--;
	if (notify_blink_remaining > 0) {
		k_work_schedule(&notify_blink_work, K_MSEC(NOTIFY_BLINK_OFF_MS));
		return;
	}

	notify_blink_active = false;
	apply_current_state();
}

static void start_notify_blink(const activity_color_t *blink_color, float brightness)
{
	k_work_cancel_delayable(&notify_blink_work);
	k_work_cancel_delayable(&nus_tx_pulse_work);

	notify_blink_color = *blink_color;
	notify_blink_brightness = brightness;
	notify_blink_active = true;
	notify_blink_remaining = NOTIFY_BLINK_COUNT * 2;
	k_work_schedule(&notify_blink_work, K_NO_WAIT);
}

static void nus_tx_pulse_end(struct k_work *work)
{
	ARG_UNUSED(work);

	if (!notify_blink_active) {
		apply_current_state();
	}
}

int activity_led_init(void)
{
	int ret = led_init();

	k_work_init_delayable(&notify_blink_work, notify_blink_step);
	k_work_init_delayable(&nus_tx_pulse_work, nus_tx_pulse_end);

	ble_connected = false;
	current_class = CLASS_LABEL_UNKNOWN;
	apply_current_state();

	return ret;
}

void activity_led_show_class(class_label_t class_label)
{
	if (class_label >= CLASS_LABEL_COUNT) {
		current_class = CLASS_LABEL_UNKNOWN;
	} else {
		current_class = class_label;
	}

	if (!notify_blink_active) {
		apply_current_state();
	}
}

void activity_led_set_ble_connected(bool connected)
{
	ble_connected = connected;

	if (connected) {
		start_notify_blink(&CONNECTED_COLOR, CONNECTED_LED_BRIGHTNESS);
		return;
	}

	current_class = CLASS_LABEL_UNKNOWN;
	start_notify_blink(&DISCONNECTED_COLOR, ACTIVITY_LED_MAX_BRIGHTNESS);
}

void activity_led_nus_tx_pulse(void)
{
	if (notify_blink_active) {
		return;
	}

	k_work_cancel_delayable(&nus_tx_pulse_work);
	apply_color(current_display_color(), NUS_TX_PULSE_BRIGHTNESS);
	k_work_schedule(&nus_tx_pulse_work, K_MSEC(NUS_TX_PULSE_MS));
}
