/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef __ACTIVITY_LED_H__
#define __ACTIVITY_LED_H__

#include "inference_postprocessing.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

int activity_led_init(void);

void activity_led_show_class(class_label_t class_label);

void activity_led_set_ble_connected(bool connected);

void activity_led_nus_tx_pulse(void);

#ifdef __cplusplus
}
#endif

#endif /* __ACTIVITY_LED_H__ */
