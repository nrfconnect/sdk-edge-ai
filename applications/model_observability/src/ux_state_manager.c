/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ux_state_manager.h"

#include <errno.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/atomic.h>
#include <zephyr/sys/util.h>

#include "nrf_edgeai_user_model_labels.h"

LOG_MODULE_REGISTER(uxsm, CONFIG_LOG_DEFAULT_LEVEL);

/** Blink period per state: 3 Hz before BLE connection, 1 Hz after */
static const uint32_t BLINK_PERIOD_MS[] = {
	[UX_STATE_ADVERTISING] = 1000U / 3U,
	[UX_STATE_CONNECTED] = 1000U,
};

BUILD_ASSERT(UX_LED_ON_MS < (1000U / 3U), "LED ON time must be shorter than the blink period");

/** RGB LED channels, bit positions in the activity color mask */
enum {
	LED_CH_RED,
	LED_CH_GREEN,
	LED_CH_BLUE,
	LED_CH_NUM,
};

#define LED_RED   BIT(LED_CH_RED)
#define LED_GREEN BIT(LED_CH_GREEN)
#define LED_BLUE  BIT(LED_CH_BLUE)

/* led0 is the blue channel of the RGB LED */
static const struct gpio_dt_spec rgb_led[LED_CH_NUM] = {
	[LED_CH_RED] = GPIO_DT_SPEC_GET(DT_NODELABEL(led1_red), gpios),
	[LED_CH_GREEN] = GPIO_DT_SPEC_GET(DT_NODELABEL(led1_green), gpios),
	[LED_CH_BLUE] = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios),
};

/** LED color per model label; idle is the default */
static const uint8_t ACTIVITY_COLOR[MODEL_USER_LABEL_COUNT] = {
	[MODEL_USER_LABEL_IDLE] = LED_BLUE,
	[MODEL_USER_LABEL_UPDOWN] = LED_RED,
	[MODEL_USER_LABEL_CIRCLE] = LED_GREEN,
	[MODEL_USER_LABEL_FLIP] = LED_RED | LED_GREEN,
};

/* Written from BT stack callbacks, read from the system workqueue */
static atomic_t ux_state = ATOMIC_INIT(UX_STATE_ADVERTISING);

/* Written from the inference thread, read from the system workqueue */
static atomic_t led_color = ATOMIC_INIT(LED_BLUE);

/* Activity debounce state, used from the inference thread only */
static uint16_t current_activity = MODEL_USER_LABEL_IDLE;
static uint16_t candidate_activity;
static uint8_t candidate_hits;
static uint8_t candidate_misses;

static struct k_work_delayable led_blink_work;

static void led_blink_work_handler(struct k_work *work)
{
	static bool led_on;
	ux_state_t state = (ux_state_t)atomic_get(&ux_state);

	ARG_UNUSED(work);
	__ASSERT_NO_MSG(state < ARRAY_SIZE(BLINK_PERIOD_MS));

	uint32_t color = (uint32_t)atomic_get(&led_color);

	led_on = !led_on;
	for (size_t ch = 0; ch < LED_CH_NUM; ch++) {
		(void)gpio_pin_set_dt(&rgb_led[ch], led_on && ((color & BIT(ch)) != 0U));
	}

	k_work_schedule(&led_blink_work,
			K_MSEC(led_on ? UX_LED_ON_MS : (BLINK_PERIOD_MS[state] - UX_LED_ON_MS)));
}

int uxsm_init(void)
{
	int err;

	for (size_t ch = 0; ch < LED_CH_NUM; ch++) {
		if (!gpio_is_ready_dt(&rgb_led[ch])) {
			LOG_ERR("Status LED GPIO is not ready (channel %zu)", ch);
			return -ENODEV;
		}

		err = gpio_pin_configure_dt(&rgb_led[ch], GPIO_OUTPUT_INACTIVE);
		if (err != 0) {
			LOG_ERR("Failed to configure status LED (channel %zu, err %d)", ch, err);
			return err;
		}
	}

	atomic_set(&ux_state, (atomic_val_t)UX_STATE_ADVERTISING);
	atomic_set(&led_color, (atomic_val_t)ACTIVITY_COLOR[MODEL_USER_LABEL_IDLE]);
	current_activity = MODEL_USER_LABEL_IDLE;
	candidate_hits = 0U;
	candidate_misses = 0U;

	k_work_init_delayable(&led_blink_work, led_blink_work_handler);
	k_work_schedule(&led_blink_work, K_NO_WAIT);

	return 0;
}

void uxsm_set_ble_connected(bool connected)
{
	ux_state_t new_state = connected ? UX_STATE_CONNECTED : UX_STATE_ADVERTISING;
	ux_state_t old = (ux_state_t)atomic_set(&ux_state, (atomic_val_t)new_state);

	if (old == new_state) {
		return;
	}

	LOG_INF("State transition -> %s",
		(new_state == UX_STATE_CONNECTED) ? "CONNECTED" : "ADVERTISING");

	/* Apply the new blink rate right away instead of after the current phase */
	k_work_reschedule(&led_blink_work, K_NO_WAIT);
}

ux_state_t uxsm_get_state(void)
{
	return (ux_state_t)atomic_get(&ux_state);
}

void uxsm_set_current_activity(uint16_t label)
{
	if (label >= MODEL_USER_LABEL_COUNT) {
		return;
	}

	if (label == current_activity) {
		/* Back to the shown class: drop any pending change */
		candidate_hits = 0U;
		candidate_misses = 0U;
		return;
	}

	if ((candidate_hits > 0U) && (label == candidate_activity)) {
		candidate_hits++;
	} else if ((candidate_hits > 0U) && (candidate_misses < UX_ACTIVITY_FLICKER_TOLERANCE)) {
		/* Single outlier inside a run of the candidate class: tolerate it */
		candidate_misses++;
		return;
	} else {
		candidate_activity = label;
		candidate_hits = 1U;
		candidate_misses = 0U;
	}

	if (candidate_hits < UX_ACTIVITY_CONFIRM_COUNT) {
		return;
	}

	current_activity = candidate_activity;
	candidate_hits = 0U;
	candidate_misses = 0U;
	atomic_set(&led_color, (atomic_val_t)ACTIVITY_COLOR[current_activity]);

	LOG_INF("Activity -> %s", NRF_EDGEAI_USER_LABELS_NAME[current_activity]);
}
