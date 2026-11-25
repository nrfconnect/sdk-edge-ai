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

BUILD_ASSERT(DT_NODE_HAS_PROP(TRACE_NODE, ww_trace_gpios) &&
		     DT_NODE_HAS_PROP(TRACE_NODE, kws_trace_gpios),
	     "Add ww-trace-gpios and kws-trace-gpios to the zephyr,user node");

static const struct gpio_dt_spec trace_gpios[] = {
	[INFERENCE_TRACE_WW] = GPIO_DT_SPEC_GET(TRACE_NODE, ww_trace_gpios),
	[INFERENCE_TRACE_KWS] = GPIO_DT_SPEC_GET(TRACE_NODE, kws_trace_gpios),
};

void inference_trace_set(enum inference_trace_line line, bool active)
{
	(void)gpio_pin_set_dt(&trace_gpios[line], active);
}

static int inference_trace_init(void)
{
	for (size_t i = 0; i < ARRAY_SIZE(trace_gpios); i++) {
		if (!gpio_is_ready_dt(&trace_gpios[i])) {
			LOG_ERR("Trace GPIO %u not ready", i);
			return -ENODEV;
		}

		int err = gpio_pin_configure_dt(&trace_gpios[i], GPIO_OUTPUT_INACTIVE);

		if (err) {
			LOG_ERR("Failed to configure trace GPIO %u (err %d)", i, err);
			return err;
		}
	}

	return 0;
}

SYS_INIT(inference_trace_init, APPLICATION, CONFIG_APPLICATION_INIT_PRIORITY);
