/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/devicetree.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(leds);

static const struct gpio_dt_spec led0 = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

static void led_timer_expiry(struct k_timer *timer);
static K_TIMER_DEFINE(led_timer, led_timer_expiry, NULL);

int leds_init(void)
{
	int err;

	if (!gpio_is_ready_dt(&led0)) {
		LOG_ERR("GPIO is not ready");
		return -ENODEV;
	}

	err = gpio_pin_configure_dt(&led0, GPIO_OUTPUT_ACTIVE);
	if (err) {
		LOG_ERR("Failed to configure GPIO pin (err %d)", err);
		return err;
	}

	return 0;
}

void leds_blink_led0(void)
{
	gpio_pin_set_dt(&led0, 1);
	k_timer_start(&led_timer, K_SECONDS(1), K_NO_WAIT);
}

static void led_timer_expiry(struct k_timer *timer)
{
	gpio_pin_set_dt(&led0, 0);
}
