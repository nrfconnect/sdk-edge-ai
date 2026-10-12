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

// #define MY_LOG_LEVEL LOG_LEVEL_NONE
#define MY_LOG_LEVEL LOG_LEVEL_DBG

LOG_MODULE_REGISTER(axon_test_app, MY_LOG_LEVEL);

extern void main_intrinsics_test(void);
int main(void)
{
    printk("nrf_axon_app_test_dsp_intrinsics on %s\n", CONFIG_BOARD);

    // Start inference
    main_intrinsics_test();

    k_sleep(K_FOREVER);
  return 0;
}
