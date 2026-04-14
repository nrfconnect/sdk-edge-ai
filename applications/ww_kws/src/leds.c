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
static const struct gpio_dt_spec led1 = GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios);

static void led_timer_expiry(struct k_timer *timer);
static K_TIMER_DEFINE(led_timer, led_timer_expiry, NULL);

static int led_init(const struct gpio_dt_spec *spec)
{
	int err;

	if (!gpio_is_ready_dt(spec)) {
		LOG_ERR("GPIO %s is not ready", spec->port->name);
		return -ENODEV;
	}

	err = gpio_pin_configure_dt(spec, GPIO_OUTPUT_INACTIVE);
	if (err) {
		LOG_ERR("Failed to configure GPIO %s pin %u (err %d)", spec->port->name, spec->pin,
			err);
		return err;
	}

	return 0;
}

int leds_init(void)
{
	int err;

	err = led_init(&led0);
	if (err) {
		return err;
	}

	err = led_init(&led1);
	if (err) {
		return err;
	}

	return 0;
}

void leds_blink_led0(void)
{
	gpio_pin_set_dt(&led0, 1);
	k_timer_user_data_set(&led_timer, (void *)&led0);
	k_timer_start(&led_timer, K_SECONDS(1), K_NO_WAIT);
}

void leds_blink_led1(void)
{
	gpio_pin_set_dt(&led1, 1);
	k_timer_user_data_set(&led_timer, (void *)&led1);
	k_timer_start(&led_timer, K_SECONDS(1), K_NO_WAIT);
}

static void led_timer_expiry(struct k_timer *timer)
{
	const struct gpio_dt_spec *led = k_timer_user_data_get(timer);

	gpio_pin_set_dt(led, 0);
}

void leds_on_led0(void)
{
	gpio_pin_set_dt(&led0, 1);
}

void leds_off_led0(void)
{
	gpio_pin_set_dt(&led0, 0);
}
