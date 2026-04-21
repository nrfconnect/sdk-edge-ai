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
 * @brief This module provides button control functions with short/long press detection.
 *
 * The button module uses a GPIO interrupt service routine (ISR) to detect button
 * state changes (press / release) and a delayable work item for debouncing and
 * timing. It distinguishes two types of clicks based on press duration:
 *
 *   - SHORT click: press duration shorter than @ref BUTTON_SHORT_CLICK_MSEC
 *   - LONG  click: press duration longer  than @ref BUTTON_LONG_CLICK_MSEC
 *
 * Presses released between @ref BUTTON_SHORT_CLICK_MSEC and
 * @ref BUTTON_LONG_CLICK_MSEC are intentionally ignored (no click is reported).
 * This dead band helps prevent accidental activations from presses that are
 * neither a deliberate short tap nor a sustained long press.
 *
 * A LONG click is reported as soon as the long-press threshold elapses while
 * the button is still held; the subsequent release is then suppressed.
 * A SHORT click is reported on release.
 */
#ifndef __BUTTON_H__
#define __BUTTON_H__

#include "../common.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** Short click maximum duration in milliseconds. */
#define BUTTON_SHORT_CLICK_MSEC 500

/** Long click minimum duration in milliseconds. */
#define BUTTON_LONG_CLICK_MSEC 2000

/**
 * @brief Detected button click type
 */
typedef enum {
	BUTTON_CLICK_SHORT = 0,
	BUTTON_CLICK_LONG,
} button_click_t;

/**
 * @brief Button click handler type
 *
 * Invoked from the system workqueue context when a click is detected.
 *
 * @param click  Detected click type (@ref button_click_t)
 */
typedef void (*button_click_handler_t)(button_click_t click);

/**
 * @brief Initialize the button module and its GPIO ISR.
 *
 * @return 0 on success, negative error code otherwise.
 */
int button_init(void);

/**
 * @brief Register a click handler.
 *
 * Replaces any previously registered handler. Pass NULL to deregister.
 *
 * @param click_handler  Handler to invoke on detected clicks.
 */
void button_reg_click_handler(button_click_handler_t click_handler);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __BUTTON_H__ */

/**
 * @}
 */
