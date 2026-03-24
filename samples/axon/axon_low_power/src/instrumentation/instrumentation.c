/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/kernel.h>

#include "instrumentation.h"

void inst_sweep_begin(void)
{
#if IS_ENABLED(CONFIG_AXON_LOW_POWER_LED_INDICATION)
	sweep_led_on();
#endif
#if IS_ENABLED(CONFIG_AXON_LOW_POWER_GPIO_TRACING)
	sweep_trace_begin();
#endif
}

void inst_sweep_end(void)
{
#if IS_ENABLED(CONFIG_AXON_LOW_POWER_GPIO_TRACING)
	sweep_trace_end();
#endif
#if IS_ENABLED(CONFIG_AXON_LOW_POWER_LED_INDICATION)
	sweep_led_off();
#endif
}

void inst_infer_begin(void)
{
#if IS_ENABLED(CONFIG_AXON_LOW_POWER_GPIO_TRACING)
	infer_trace_begin();
#endif
}

void inst_infer_end(void)
{
#if IS_ENABLED(CONFIG_AXON_LOW_POWER_GPIO_TRACING)
	infer_trace_end();
#endif
}

int inst_init(void)
{
	int err = 0;

#if IS_ENABLED(CONFIG_AXON_LOW_POWER_LED_INDICATION)
	err = led_hw_init();
	if (err != 0) {
		return err;
	}
#endif

#if IS_ENABLED(CONFIG_AXON_LOW_POWER_GPIO_TRACING)
	err = gpio_trace_hw_init();
	if (err != 0) {
		return err;
	}
#endif

	return err;
}
