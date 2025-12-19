/**
 *
 * @defgroup bsp_led LEDs control functions
 * @{
 * @ingroup bsp
 *
 * @brief This module provides LEDs control functions.
 *
 */
#ifndef __BSP_LED_H__
#define __BSP_LED_H__

#include <bsp_common.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/**
 * @brief Initialize LEDs
 * 
 * @return Operation status, 0 for success 
 */
int bsp_led_init(void);

/**
 * @brief Turn on red LED with specific brightness
 * 
 * @param brightness    LED brightness in range 0 - 1
 * 
 * @return Operation status, 0 for success 
 */
int bsp_led_set_red(float brightness);

/**
 * @brief Turn on green LED with specific brightness
 * 
 * @param brightness    LED brightness in range 0 - 1
 * 
 * @return Operation status, 0 for success 
 */
int bsp_led_set_green(float brightness);

/**
 * @brief Turn on blue LED with specific brightness
 * 
 * @param brightness    LED brightness in range 0 - 1
 * 
 * @return Operation status, 0 for success 
 */
int bsp_led_set_blue(float brightness);

/**
 * @brief Turn on RGB LED with specific brightness for each color
 * 
 * @param r    Red LED brightness in range 0 - 1
 * @param g    Green LED brightness in range 0 - 1
 * @param b    Blue LED brightness in range 0 - 1
 * 
 * @return Operation status, 0 for success 
 */
int bsp_led_set_rgb(float r, float g, float b);

/**
 * @brief Turn off all LEDs
 * 
 * @return Operation status, 0 for success  
 */
int bsp_led_off(void);

/**
 * @brief Blink red LED with specific brightness
 * 
 * @param brightness  LED brightness in range 0 - 1
 * @param on_ms       LED turns on time in milliseconds
 * @param off_ms      LED turns off time in milliseconds
 * 
 * @return Operation status, 0 for success
 */
int bsp_led_blink_red(float brightness, int32_t on_ms, int32_t off_ms);

/**
 * @brief Blink green LED with specific brightness
 * 
 * @param brightness  LED brightness in range 0 - 1
 * @param on_ms       LED turns on time in milliseconds
 * @param off_ms      LED turns off time in milliseconds
 * 
 * @return Operation status, 0 for success
 */
int bsp_led_blink_green(float brightness, int32_t on_ms, int32_t off_ms);

/**
 * @brief Blink blue LED with specific brightness
 * 
 * @param brightness  LED brightness in range 0 - 1
 * @param on_ms       LED turns on time in milliseconds
 * @param off_ms      LED turns off time in milliseconds
 * 
 * @return Operation status, 0 for success
 */
int bsp_led_blink_blue(float brightness, int32_t on_ms, int32_t off_ms);

/**
 * @brief Blink RGB LED with specific brightness for each color
 * 
 * @param r           Red LED brightness in range 0 - 1
 * @param g           Green LED brightness in range 0 - 1
 * @param b           Blue LED brightness in range 0 - 1
 * @param on_ms       RGB LED turns on time in milliseconds
 * @param off_ms      RGB LED turns off time in milliseconds
 * 
 * @return Operation status, 0 for success
 */
int bsp_led_blink_rgb(float r, float g, float b, int32_t on_ms, int32_t off_ms);

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __BSP_LED_H__ */

/**
 * @}
 */
