/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 *
 * @defgroup button Button control functions
 * @{
 * @ingroup bsp
 *
 *
 */
#ifndef __BUTTON_H__
#define __BUTTON_H__

#include <common.h>

typedef void (*button_click_handler_t)(bool is_pressed);

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Initialize button module
 * 
 * @return Operation status result, 0 for success
 */
int button_init(void);

/**
 * @brief Register button click handler
 * 
 * @param click_handler Button click handler
 */
void button_reg_click_handler(button_click_handler_t click_handler);

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __BUTTON_H__ */

/**
 * @}
 */
