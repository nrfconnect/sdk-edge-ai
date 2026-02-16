/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @defgroup button Button control functions
 * @{
 * @ingroup hw_modules
 *
 * @brief This module provides button control functions.
 */
#ifndef __BUTTON_H__
#define __BUTTON_H__

#include "../common.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/**
 * @brief Button click handler type
 *
 * @param is_pressed Indicates if the button is currently pressed (true) or released (false).
 */

typedef void (*button_click_handler_t)(bool is_pressed);

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
