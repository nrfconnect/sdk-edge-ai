/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @defgroup leds LEDs control functions
 * @{
 * @ingroup keyword_spotting
 */

#ifndef __LEDS_H__
#define __LEDS_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Initialize LEDs.
 *
 * @return Operation status, 0 for success.
 */
int leds_init(void);

/**
 * @brief Blink with LED0.
 */
void leds_blink_led0(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __LEDS_H__ */

/**
 * @}
 */
