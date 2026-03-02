/*
 * Copyright (c) 2022 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/kernel.h>
#include <zephyr/irq.h>
#include <zephyr/sys_clock.h>
#include <zephyr/sys/printk.h>

#include <zephyr/logging/log.h>
#include "axon/nrf_axon_platform.h"
// #define MY_LOG_LEVEL LOG_LEVEL_NONE
#define MY_LOG_LEVEL LOG_LEVEL_DBG

LOG_MODULE_REGISTER(kws, MY_LOG_LEVEL);

extern void base_inference_main(void);
int main(void)
{
#if 0
  /**
   * @FIXME!!! IS THIS NEEDED FOR NON-FPGA DEBUGGING?
   */
  printk("DEBUG LOOP\n");
  k_busy_wait(3*1000*1000); // ~8 seconds
#endif
  while (1) {
    printk("Hello world from %s\n", CONFIG_BOARD);
    printk("ticks per second %u\n", nrf_axon_platform_get_clk_hz());

    // Start inference
    base_inference_main();
    k_sleep(K_FOREVER);
  }
}
