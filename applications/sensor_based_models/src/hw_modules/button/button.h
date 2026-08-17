/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef __BUTTON_H__
#define __BUTTON_H__

#include "../common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BUTTON_SHORT_CLICK_MSEC 500
#define BUTTON_LONG_CLICK_MSEC 2000

typedef enum {
	BUTTON_CLICK_SHORT = 0,
	BUTTON_CLICK_LONG,
} button_click_t;

typedef void (*button_click_handler_t)(button_click_t click);

int button_init(void);
void button_reg_click_handler(button_click_handler_t click_handler);

#ifdef __cplusplus
}
#endif

#endif /* __BUTTON_H__ */
