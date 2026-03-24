/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "instrumentation.h"

#include <zephyr/devicetree.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(axon_low_power);

BUILD_ASSERT(DT_NODE_EXISTS(DT_ALIAS(led0)),
	     "LED indication: board must define led0 in devicetree");

static const struct gpio_dt_spec sweep_led_gpio = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

static void sweep_led_set(int value)
{
	int err = gpio_pin_set_dt(&sweep_led_gpio, value);

	if (err != 0) {
		LOG_WRN("Sweep LED GPIO set failed (%d)", err);
	}
}

void sweep_led_on(void)
{
	sweep_led_set(1);
}

void sweep_led_off(void)
{
	sweep_led_set(0);
}

int led_hw_init(void)
{
	int err;

	if (!gpio_is_ready_dt(&sweep_led_gpio)) {
		LOG_ERR("Sweep LED GPIO device not ready");
		return -ENODEV;
	}

	err = gpio_pin_configure_dt(&sweep_led_gpio, GPIO_OUTPUT_INACTIVE);
	if (err != 0) {
		LOG_ERR("Sweep LED pin configure failed (%d)", err);
		return err;
	}

	return 0;
}
