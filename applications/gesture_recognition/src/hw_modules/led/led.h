/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 *
 * @defgroup led LEDs control functions
 * @{
 * @ingroup bsp
 *
 * @brief This module provides LEDs control functions.
 *
 */
#ifndef __LED_H__
#define __LED_H__

#include "../common.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/**
 * @brief Initialize LEDs
 *
 * @return Operation status, 0 for success
 */
int led_init(void);

/**
 * @brief Turn on LED channel 0 (red on RGB boards, LED0 on discrete boards)
 *
 * @param brightness    LED brightness in range 0 - 1
 *
 * @return Operation status, 0 for success
 */
int led_set_led0(float brightness);

/**
 * @brief Turn on LED channel 1 (green on RGB boards, LED1 on discrete boards)
 *
 * @param brightness    LED brightness in range 0 - 1
 *
 * @return Operation status, 0 for success
 */
int led_set_led1(float brightness);

/**
 * @brief Turn on LED channel 2 (blue on RGB boards, LED2 on discrete boards)
 *
 * @param brightness    LED brightness in range 0 - 1
 *
 * @return Operation status, 0 for success
 */
int led_set_led2(float brightness);

/**
 * @brief Turn on LED channels 0..2 with specific brightness
 *
 * @param led0_brightness    LED0 (red on RGB boards) brightness in range 0 - 1
 * @param led1_brightness    LED1 (green on RGB boards) brightness in range 0 - 1
 * @param led2_brightness    LED2 (blue on RGB boards) brightness in range 0 - 1
 *
 * @return Operation status, 0 for success
 */
int led_set_leds(float led0_brightness, float led1_brightness, float led2_brightness);

/**
 * @brief Turn off all LEDs
 *
 * @return Operation status, 0 for success
 */
int led_off(void);

/**
 * @brief Blink LED channel 0 (red on RGB boards, LED0 on discrete boards)
 *
 * @note This function uses sleep internally and should not be called from ISR context.
 *
 * @param brightness  LED brightness in range 0 - 1
 * @param on_ms       LED turns on time in milliseconds
 * @param off_ms      LED turns off time in milliseconds
 *
 * @return Operation status, 0 for success
 */
int led_blink_led0(float brightness, int32_t on_ms, int32_t off_ms);

/**
 * @brief Blink LED channel 1 (green on RGB boards, LED1 on discrete boards)
 *
 * @note This function uses sleep internally and should not be called from ISR context.
 *
 * @param brightness  LED brightness in range 0 - 1
 * @param on_ms       LED turns on time in milliseconds
 * @param off_ms      LED turns off time in milliseconds
 *
 * @return Operation status, 0 for success
 */
int led_blink_led1(float brightness, int32_t on_ms, int32_t off_ms);

/**
 * @brief Blink LED channel 2 (blue on RGB boards, LED2 on discrete boards)
 *
 * @note This function uses sleep internally and should not be called from ISR context.
 *
 * @param brightness  LED brightness in range 0 - 1
 * @param on_ms       LED turns on time in milliseconds
 * @param off_ms      LED turns off time in milliseconds
 *
 * @return Operation status, 0 for success
 */
int led_blink_led2(float brightness, int32_t on_ms, int32_t off_ms);

/**
 * @brief Blink LED channels 0..2 with specific brightness
 *
 * @note This function uses sleep internally and should not be called from ISR context.
 *
 * @param led0        LED0 (red on RGB boards) brightness in range 0 - 1
 * @param led1        LED1 (green on RGB boards) brightness in range 0 - 1
 * @param led2        LED2 (blue on RGB boards) brightness in range 0 - 1
 * @param on_ms       RGB LED turns on time in milliseconds
 * @param off_ms      RGB LED turns off time in milliseconds
 *
 * @return Operation status, 0 for success
 */
int led_blink_leds(float led0_brightness, float led1_brightness, float led2_brightness, int32_t on_ms, int32_t off_ms);

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __LED_H__ */

/**
 * @}
 */
