/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "button.h"
#include <inttypes.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

LOG_MODULE_REGISTER(button, CONFIG_LOG_DEFAULT_LEVEL);


#define SW0_NODE DT_ALIAS(sw0)
#if !DT_NODE_HAS_STATUS(SW0_NODE, okay)
#error "Unsupported board: sw0 devicetree alias is not defined"
#endif

#define DEBOUNCE_MS 1000

static const struct gpio_dt_spec button_sw0 = GPIO_DT_SPEC_GET_OR(SW0_NODE, gpios, {0});
static struct gpio_callback button_cb_data;
static button_click_handler_t button_click_handler;
static struct k_timer reenable_timer;

static void reenable_timer_expiry(struct k_timer *timer)
{
	ARG_UNUSED(timer);
	int ret = gpio_pin_interrupt_configure_dt(&button_sw0, GPIO_INT_LEVEL_ACTIVE);

	if (ret != 0) {
		LOG_ERR("Error %d: failed to re-enable interrupt on %s pin %u",
			ret, button_sw0.port->name, button_sw0.pin);
	}
}

static void button_interrupt(const struct device *dev, struct gpio_callback *cb, uint32_t pins)
{
	ARG_UNUSED(dev);
	ARG_UNUSED(cb);
	ARG_UNUSED(pins);

	int ret = gpio_pin_interrupt_configure_dt(&button_sw0, GPIO_INT_DISABLE);
	if (ret != 0) {
		LOG_ERR("Error %d: failed to disable interrupt on %s pin %u",
			ret, button_sw0.port->name, button_sw0.pin);
	}

	if (button_click_handler != NULL) {
		button_click_handler();
	} else {
		LOG_WRN("Button interrupt fired but no click handler is registered");
	}

	k_timer_start(&reenable_timer, K_MSEC(DEBOUNCE_MS), K_NO_WAIT);
}

int button_init(void)
{
	int ret;

	if (!device_is_ready(button_sw0.port)) {
		LOG_ERR("Error: button device %s is not ready", button_sw0.port->name);
		return -ENODEV;
	}

	ret = gpio_pin_configure_dt(&button_sw0, GPIO_INPUT);
	if (ret != 0) {
		LOG_ERR("Error %d: failed to configure %s pin %u", ret,
			button_sw0.port->name, button_sw0.pin);
		return ret;
	}

	k_timer_init(&reenable_timer, reenable_timer_expiry, NULL);

	ret = gpio_pin_interrupt_configure_dt(&button_sw0, GPIO_INT_LEVEL_ACTIVE);

	if (ret != 0) {
		LOG_ERR("Error %d: failed to configure interrupt on %s pin %u",
			ret, button_sw0.port->name, button_sw0.pin);
		return ret;
	}

	gpio_init_callback(&button_cb_data, button_interrupt, BIT(button_sw0.pin));
	gpio_add_callback(button_sw0.port, &button_cb_data);
	LOG_DBG("Set up button at %s pin %u", button_sw0.port->name, button_sw0.pin);

	return ret;
}

void button_reg_click_handler(button_click_handler_t click_handler)
{
	button_click_handler = click_handler;
}
