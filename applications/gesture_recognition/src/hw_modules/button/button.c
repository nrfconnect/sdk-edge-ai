/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "button.h"

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

/* Debounce delay (ms) used after each interrupt before sampling the pin. */
#define BUTTON_DEBOUNCE_MSEC 20

/* Periodic check interval (ms) used while the button is held to detect long press. */
#define BUTTON_CHECK_PERIOD_MSEC 100

static const struct gpio_dt_spec button_sw0 = GPIO_DT_SPEC_GET_OR(SW0_NODE, gpios, {0});

static struct gpio_callback button_cb_data;
static struct k_work_delayable button_check_work;

static button_click_handler_t button_click_handler;

/* State tracked by the system workqueue; safe to access without locking from
 * inside button_check_work_fn() and the registration helper.
 */
static bool button_pressed_state;     /* current debounced state */
static int64_t press_start_uptime_ms; /* k_uptime_get() at press time, valid when pressed */
static bool long_already_reported;    /* long click already emitted while still held */

static void emit_click(button_click_t click)
{
	if (button_click_handler != NULL) {
		button_click_handler(click);
	} else {
		LOG_WRN("Click detected (%s) but no handler registered",
			click == BUTTON_CLICK_LONG ? "LONG" : "SHORT");
	}
}

static void handle_press_edge(int64_t now_ms)
{
	if (button_pressed_state) {
		/* Already considered pressed; ignore (bounce). */
		return;
	}

	button_pressed_state = true;
	press_start_uptime_ms = now_ms;
	long_already_reported = false;

	/* Schedule periodic check to detect long press while still held. */
	k_work_reschedule(&button_check_work, K_MSEC(BUTTON_CHECK_PERIOD_MSEC));
}

static void handle_release_edge(int64_t now_ms)
{
	if (!button_pressed_state) {
		/* Spurious release (e.g. boot-time bounce); ignore. */
		return;
	}

	int64_t held_ms = now_ms - press_start_uptime_ms;

	button_pressed_state = false;
	(void)k_work_cancel_delayable(&button_check_work);

	if (long_already_reported) {
		/* Long click was already emitted while held; ignore release. */
		return;
	}

	if (held_ms < BUTTON_SHORT_CLICK_MSEC) {
		emit_click(BUTTON_CLICK_SHORT);
	} else {
		/* Held between short and long thresholds: ignore (CLICK_NONE). */
		LOG_DBG("Click ignored, hold time %lld ms (between thresholds)",
			(long long)held_ms);
	}
}

/* Workqueue handler: debounces by re-reading the pin and detects long press. */
static void button_check_work_fn(struct k_work *work)
{
	ARG_UNUSED(work);

	int raw = gpio_pin_get_dt(&button_sw0);

	if (raw < 0) {
		LOG_ERR("Failed to read button pin (err %d)", raw);
		return;
	}

	const bool sampled_pressed = (raw != 0);
	const int64_t now_ms = k_uptime_get();

	if (sampled_pressed != button_pressed_state) {
		if (sampled_pressed) {
			handle_press_edge(now_ms);
		} else {
			handle_release_edge(now_ms);
		}
		return;
	}

	/* No state change: while held, check for long-press timeout. */
	if (button_pressed_state && !long_already_reported) {
		int64_t held_ms = now_ms - press_start_uptime_ms;

		if (held_ms >= BUTTON_LONG_CLICK_MSEC) {
			long_already_reported = true;
			emit_click(BUTTON_CLICK_LONG);
			/* Continue scheduling so we still detect the release. */
		}

		k_work_reschedule(&button_check_work, K_MSEC(BUTTON_CHECK_PERIOD_MSEC));
	}
}

static void button_isr(const struct device *dev, struct gpio_callback *cb, uint32_t pins)
{
	ARG_UNUSED(dev);
	ARG_UNUSED(cb);
	ARG_UNUSED(pins);

	/* Defer handling to the workqueue after a short debounce delay. */
	k_work_reschedule(&button_check_work, K_MSEC(BUTTON_DEBOUNCE_MSEC));
}

int button_init(void)
{
	int ret;

	if (!device_is_ready(button_sw0.port)) {
		LOG_ERR("Button GPIO device %s is not ready", button_sw0.port->name);
		return -ENODEV;
	}

	ret = gpio_pin_configure_dt(&button_sw0, GPIO_INPUT);
	if (ret != 0) {
		LOG_ERR("Failed to configure button pin %u (err %d)", button_sw0.pin, ret);
		return ret;
	}

	k_work_init_delayable(&button_check_work, button_check_work_fn);

	/* Edge interrupt on both transitions to detect press AND release. */
	ret = gpio_pin_interrupt_configure_dt(&button_sw0, GPIO_INT_EDGE_BOTH);
	if (ret != 0) {
		LOG_ERR("Failed to configure interrupt on button pin %u (err %d)",
			button_sw0.pin, ret);
		return ret;
	}

	gpio_init_callback(&button_cb_data, button_isr, BIT(button_sw0.pin));
	ret = gpio_add_callback(button_sw0.port, &button_cb_data);
	if (ret != 0) {
		LOG_ERR("Failed to add GPIO callback (err %d)", ret);
		return ret;
	}

	LOG_DBG("Button module ready on %s pin %u", button_sw0.port->name, button_sw0.pin);
	return 0;
}

void button_reg_click_handler(button_click_handler_t click_handler)
{
	button_click_handler = click_handler;
}
