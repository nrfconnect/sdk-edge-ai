/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 *
 * @defgroup bsp_button Button control functions
 * @{
 * @ingroup bsp
 *
 *
 */
#ifndef __BSP_BUTTON_H__
#define __BSP_BUTTON_H__

#include <bsp_common.h>

typedef void (*bsp_button_click_handler_t)(bool is_pressed);

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Initialize button module
 * 
 * @return Operation status result, 0 for success
 */
int bsp_button_init(void);

/**
 * @brief Register button click handler
 * 
 * @param click_handler Button click handler
 */
void bsp_button_reg_click_handler(bsp_button_click_handler_t click_handler);

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __BSP_BUTTON_H__ */

/**
 * @}
 */
