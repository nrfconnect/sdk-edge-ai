/*
 * Copyright (c) 2022 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/kernel.h>
#include <zephyr/irq.h>
#include <zephyr/sys_clock.h>
#include "axon/nrf_axon_platform.h"
extern void base_inference_main(void);
int main(void)
{
	while (1) {
		nrf_axon_platform_printf("Hello world from %s\n", CONFIG_BOARD);
		nrf_axon_platform_printf("ticks per second %u\n",
			nrf_axon_platform_get_clk_hz());

		/* do inference test */
		base_inference_main();
		k_sleep(K_FOREVER);
	}
}
