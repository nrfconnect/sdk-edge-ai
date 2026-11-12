/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "inference_trace.h"

#include <zephyr/devicetree.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/init.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(inference_trace);

#define TRACE_NODE DT_PATH(zephyr_user)

BUILD_ASSERT(DT_NODE_HAS_PROP(TRACE_NODE, infer_trace_gpios),
	     "Add infer-trace-gpios to the zephyr,user node");

static const struct gpio_dt_spec trace_gpio = GPIO_DT_SPEC_GET(TRACE_NODE, infer_trace_gpios);

void inference_trace_set(bool active)
{
	(void)gpio_pin_set_dt(&trace_gpio, active);
}

static int inference_trace_init(void)
{
	if (!gpio_is_ready_dt(&trace_gpio)) {
		LOG_ERR("Trace GPIO not ready");
		return -ENODEV;
	}

	int err = gpio_pin_configure_dt(&trace_gpio, GPIO_OUTPUT_INACTIVE);

	if (err) {
		LOG_ERR("Failed to configure trace GPIO (err %d)", err);
	}

	return err;
}

SYS_INIT(inference_trace_init, APPLICATION, CONFIG_APPLICATION_INIT_PRIORITY);
