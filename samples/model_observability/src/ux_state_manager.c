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

LOG_MODULE_REGISTER(uxsm, CONFIG_LOG_DEFAULT_LEVEL);

/** Blink period per state: 3 Hz before BLE connection, 1 Hz after */
static const uint32_t BLINK_PERIOD_MS[] = {
	[UX_STATE_ADVERTISING] = 1000U / 3U,
	[UX_STATE_CONNECTED] = 1000U,
};

BUILD_ASSERT(UX_LED_ON_MS < (1000U / 3U), "LED ON time must be shorter than the blink period");

/* led0 is the blue channel of the RGB LED */
static const struct gpio_dt_spec blue_led = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

/* Written from BT stack callbacks, read from the system workqueue */
static atomic_t ux_state = ATOMIC_INIT(UX_STATE_ADVERTISING);

static struct k_work_delayable led_blink_work;

static void led_blink_work_handler(struct k_work *work)
{
	static bool led_on;
	ux_state_t state = (ux_state_t)atomic_get(&ux_state);

	ARG_UNUSED(work);
	__ASSERT_NO_MSG(state < ARRAY_SIZE(BLINK_PERIOD_MS));

	led_on = !led_on;
	(void)gpio_pin_set_dt(&blue_led, led_on);

	k_work_schedule(&led_blink_work,
			K_MSEC(led_on ? UX_LED_ON_MS : (BLINK_PERIOD_MS[state] - UX_LED_ON_MS)));
}

int uxsm_init(void)
{
	int err;

	if (!gpio_is_ready_dt(&blue_led)) {
		LOG_ERR("Status LED GPIO is not ready");
		return -ENODEV;
	}

	err = gpio_pin_configure_dt(&blue_led, GPIO_OUTPUT_INACTIVE);
	if (err != 0) {
		LOG_ERR("Failed to configure status LED (err %d)", err);
		return err;
	}

	atomic_set(&ux_state, (atomic_val_t)UX_STATE_ADVERTISING);

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
