/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "bsp_button.h"
#include <inttypes.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

LOG_MODULE_REGISTER(bsp_button, CONFIG_LOG_DEFAULT_LEVEL);


#define SW0_NODE DT_ALIAS(sw0)
#if !DT_NODE_HAS_STATUS(SW0_NODE, okay)
#error "Unsupported board: sw0 devicetree alias is not defined"
#endif


static const struct gpio_dt_spec button_sw0_ = GPIO_DT_SPEC_GET_OR(SW0_NODE, gpios, {0});
static struct gpio_callback button_cb_data_;
static bsp_button_click_handler_t button_click_handler_ = NULL;


static bool is_pressed_(void)
{
	return gpio_pin_get_dt(&button_sw0_) > 0;
}

void button_interrupt(const struct device *dev, struct gpio_callback *cb, uint32_t pins)
{
	ARG_UNUSED(dev);
	ARG_UNUSED(cb);
	ARG_UNUSED(pins);

	if (button_click_handler_) {
		button_click_handler_(is_pressed_());
	}
}

int bsp_button_init(void)
{
	int ret;

	if (!device_is_ready(button_sw0_.port)) {
		LOG_ERR("Error: button device %s is not ready", button_sw0_.port->name);
		return ENODEV;
	}

	ret = gpio_pin_configure_dt(&button_sw0_, GPIO_INPUT);
	if (ret != 0) {
		LOG_ERR("Error %d: failed to configure %s pin %d", ret,
			button_sw0_.port->name, button_sw0_.pin);
		return ret;
	}

	ret = gpio_pin_interrupt_configure_dt(&button_sw0_, GPIO_INT_EDGE_BOTH);

	if (ret != 0) {
		LOG_ERR("Error %d: failed to configure interrupt on %s pin %d",
			ret, button_sw0_.port->name, button_sw0_.pin);
		return ret;
	}

	gpio_init_callback(&button_cb_data_, button_interrupt, BIT(button_sw0_.pin));
	gpio_add_callback(button_sw0_.port, &button_cb_data_);
	LOG_INF("Set up button at %s pin %d", button_sw0_.port->name, button_sw0_.pin);

	return ret;
}

void bsp_button_reg_click_handler(bsp_button_click_handler_t click_handler)
{
	button_click_handler_ = click_handler;
}
