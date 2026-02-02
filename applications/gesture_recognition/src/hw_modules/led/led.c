/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "led.h"
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/pwm.h>
#include <zephyr/sys/util.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(led, CONFIG_LOG_DEFAULT_LEVEL);


/*
 * LED channels: map to RGB on boards with a tri-color LED, or to LED0..LED2
 * on boards with discrete LEDs.
 */
#define HAS_PWM_LEDS (DT_NODE_HAS_STATUS(DT_ALIAS(pwm_led0), okay) && \
		      DT_NODE_HAS_STATUS(DT_ALIAS(pwm_led1), okay) && \
		      DT_NODE_HAS_STATUS(DT_ALIAS(pwm_led2), okay))

#define HAS_GPIO_LEDS (DT_NODE_HAS_STATUS(DT_ALIAS(led0), okay) && \
		       DT_NODE_HAS_STATUS(DT_ALIAS(led1), okay) && \
		       DT_NODE_HAS_STATUS(DT_ALIAS(led2), okay))

#if HAS_PWM_LEDS
static const struct pwm_dt_spec led0 = PWM_DT_SPEC_GET(DT_ALIAS(pwm_led0));
static const struct pwm_dt_spec led1 = PWM_DT_SPEC_GET(DT_ALIAS(pwm_led1));
static const struct pwm_dt_spec led2 = PWM_DT_SPEC_GET(DT_ALIAS(pwm_led2));

#define led_backend_init init_led_pwm
#define led_backend_set  set_led_pwm

#elif HAS_GPIO_LEDS
static const struct gpio_dt_spec led0 = GPIO_DT_SPEC_GET_OR(DT_ALIAS(led0), gpios, {0});
static const struct gpio_dt_spec led1 = GPIO_DT_SPEC_GET_OR(DT_ALIAS(led1), gpios, {0});
static const struct gpio_dt_spec led2 = GPIO_DT_SPEC_GET_OR(DT_ALIAS(led2), gpios, {0});

#define led_backend_init init_led_gpio
#define led_backend_set  set_led_gpio

#else

#error "No LED backend available in devicetree"

#endif

/* not calibrated, fixed for specific board */
#define PWM_PERIOD PWM_MSEC(20)


__unused static int init_led_pwm(const struct pwm_dt_spec pwm_led)
{
	int ret = 0;

	if (!device_is_ready(pwm_led.dev)) {
		LOG_ERR("PWM LED Init error '%s' device_is_ready()", pwm_led.dev->name);
		return -ENODEV;
	}

	ret = pwm_set_dt(&pwm_led, PWM_PERIOD, 0);
	if (ret) {
		LOG_ERR("Error %d: failed to set pulse width for %s", ret, pwm_led.dev->name);
		return ret;
	}
	return ret;
}

__unused static int init_led_gpio(const struct gpio_dt_spec gpio_led)
{
	int ret = 0;

	if (!device_is_ready(gpio_led.port)) {
		LOG_ERR("GPIO LED Init error '%s' device_is_ready()", gpio_led.port->name);
		return -ENODEV;
	}

	ret = gpio_pin_configure_dt(&gpio_led, GPIO_OUTPUT_INACTIVE);
	if (ret != 0) {
		LOG_ERR("Error %d: failed to configure GPIO LED on %s pin %u",
			ret, gpio_led.port->name, gpio_led.pin);
		return ret;
	}

	return ret;
}

__unused static int set_led_pwm(const struct pwm_dt_spec *pwm_led, float brightness)
{
	int ret = 0;
	float pwm_brightness = CLAMP(brightness, 0.0f, 1.0f);
	uint32_t pulse = pwm_brightness * PWM_PERIOD;

	ret = pwm_set_pulse_dt(pwm_led, pulse);
	if (ret < 0) {
		LOG_ERR("LED Init error pwm_set_pulse_dt()");
		return ret;
	}
	return ret;
}

__unused static int set_led_gpio(const struct gpio_dt_spec *gpio_led, float brightness)
{
	int ret = 0;
	int on = brightness > 0.0f;

	ret = gpio_pin_set_dt(gpio_led, on);
	if (ret < 0) {
		LOG_ERR("LED Init error gpio_pin_set_dt()");
		return ret;
	}

	return ret;
}

int led_init(void)
{
	int ret = 0;

	ret = led_backend_init(led0);
	HW_RETURN_IF(ret != 0, ret);

	ret = led_backend_init(led1);
	HW_RETURN_IF(ret != 0, ret);

	return led_backend_init(led2);
}

int led_set_led0(float brightness)
{
	int ret = 0;

	ret = led_backend_set(&led1, 0);
	HW_RETURN_IF(ret != 0, ret);

	ret = led_backend_set(&led2, 0);
	HW_RETURN_IF(ret != 0, ret);

	return led_backend_set(&led0, brightness);

}

int led_set_led1(float brightness)
{
	int ret = 0;

	ret = led_backend_set(&led0, 0);
	HW_RETURN_IF(ret != 0, ret);

	ret = led_backend_set(&led2, 0);
	HW_RETURN_IF(ret != 0, ret);

	return led_backend_set(&led1, brightness);
}

int led_set_led2(float brightness)
{
	int ret = 0;

	ret = led_backend_set(&led0, 0);
	HW_RETURN_IF(ret != 0, ret);

	ret = led_backend_set(&led1, 0);
	HW_RETURN_IF(ret != 0, ret);

	return led_backend_set(&led2, brightness);
}

int led_set_leds(float led0_brightness, float led1_brightness, float led2_brightness)
{
	int ret = 0;

	ret = led_backend_set(&led0, led0_brightness);
	HW_RETURN_IF(ret != 0, ret);

	ret = led_backend_set(&led1, led1_brightness);
	HW_RETURN_IF(ret != 0, ret);

	return led_backend_set(&led2, led2_brightness);
}

int led_off(void)
{
	int ret = 0;

	ret = led_backend_set(&led0, 0);
	HW_RETURN_IF(ret != 0, ret);

	ret = led_backend_set(&led1, 0);
	HW_RETURN_IF(ret != 0, ret);

	return led_backend_set(&led2, 0);
}

int led_blink_led0(float brightness, int32_t on_ms, int32_t off_ms)
{
	int ret = 0;

	ret = led_set_led0(brightness);
	HW_RETURN_IF(ret != 0, ret);

	k_msleep(on_ms);

	ret = led_off();

	k_msleep(off_ms);
	return ret;
}

int led_blink_led1(float brightness, int32_t on_ms, int32_t off_ms)
{
	int ret;

	ret = led_set_led1(brightness);
	HW_RETURN_IF(ret != 0, ret);

	k_msleep(on_ms);

	ret = led_off();

	k_msleep(off_ms);
	return ret;
}

int led_blink_led2(float brightness, int32_t on_ms, int32_t off_ms)
{
	int ret;

	ret = led_set_led2(brightness);
	HW_RETURN_IF(ret != 0, ret);

	k_msleep(on_ms);

	ret = led_off();

	k_msleep(off_ms);
	return ret;
}

int led_blink_leds(float led0, float led1, float led2, int32_t on_ms, int32_t off_ms)
{
	int ret;

	ret = led_set_leds(led0, led1, led2);

	HW_RETURN_IF(ret != 0, ret);

	k_msleep(on_ms);

	ret = led_off();

	k_msleep(off_ms);
	return ret;
}
