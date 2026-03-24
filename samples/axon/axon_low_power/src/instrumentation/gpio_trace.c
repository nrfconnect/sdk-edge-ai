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

#define TRACE_USER_NODE DT_PATH(zephyr_user)

BUILD_ASSERT(DT_NODE_HAS_PROP(TRACE_USER_NODE, infer_trace_gpios),
	     "GPIO tracing: add zephyr,user infer-trace-gpios (see board overlay)");

static const struct gpio_dt_spec infer_trace_gpio =
	GPIO_DT_SPEC_GET(TRACE_USER_NODE, infer_trace_gpios);

BUILD_ASSERT(DT_NODE_HAS_PROP(TRACE_USER_NODE, sweep_trace_gpios),
	     "GPIO tracing: add zephyr,user sweep-trace-gpios (see board overlay)");

static const struct gpio_dt_spec sweep_trace_gpio =
	GPIO_DT_SPEC_GET(TRACE_USER_NODE, sweep_trace_gpios);

static void infer_trace_gpio_set(int value)
{
	int err = gpio_pin_set_dt(&infer_trace_gpio, value);

	if (err != 0) {
		LOG_WRN("Infer trace GPIO set failed (%d)", err);
	}
}

static void sweep_trace_gpio_set(int value)
{
	int err = gpio_pin_set_dt(&sweep_trace_gpio, value);

	if (err != 0) {
		LOG_WRN("Sweep trace GPIO set failed (%d)", err);
	}
}

void infer_trace_begin(void)
{
	infer_trace_gpio_set(1);
}

void infer_trace_end(void)
{
	infer_trace_gpio_set(0);
}

void sweep_trace_begin(void)
{
	sweep_trace_gpio_set(1);
}

void sweep_trace_end(void)
{
	sweep_trace_gpio_set(0);
}

int gpio_trace_hw_init(void)
{
	int err;

	if (!gpio_is_ready_dt(&infer_trace_gpio)) {
		LOG_ERR("Infer trace GPIO device not ready");
		return -ENODEV;
	}

	err = gpio_pin_configure_dt(&infer_trace_gpio, GPIO_OUTPUT_INACTIVE);
	if (err != 0) {
		LOG_ERR("Infer trace pin configure failed (%d)", err);
		return err;
	}

	if (!gpio_is_ready_dt(&sweep_trace_gpio)) {
		LOG_ERR("Sweep trace GPIO device not ready");
		return -ENODEV;
	}

	err = gpio_pin_configure_dt(&sweep_trace_gpio, GPIO_OUTPUT_INACTIVE);
	if (err != 0) {
		LOG_ERR("Sweep trace pin configure failed (%d)", err);
		return err;
	}

	return 0;
}
