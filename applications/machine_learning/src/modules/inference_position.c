/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#define MODULE inference_position

#include "ml_result_event.h"

#include <zephyr/devicetree.h>

#include <app_event_manager.h>
#include <caf/events/module_state_event.h>
#include <caf/events/sensor_event.h>
#include <hal/nrf_gpio.h>

#define ZEPHYR_USER_NODE DT_PATH(zephyr_user)

#define ABS_GPIO_PIN_BY_IDX(node_id, prop, idx)                                                    \
	NRF_GPIO_PIN_MAP(DT_PROP(DT_PHANDLE_BY_IDX(node_id, prop, idx), port),                     \
			 DT_GPIO_PIN_BY_IDX(node_id, prop, idx))

static const uint32_t pins[] = {DT_FOREACH_PROP_ELEM_SEP(
	ZEPHYR_USER_NODE, inference_position_gpios, ABS_GPIO_PIN_BY_IDX, (, ))};

BUILD_ASSERT(ARRAY_SIZE(pins) == 2,
	     "Wrong number of phandles in inference_position_gpios attribute");

static bool app_event_handler(const struct app_event_header *aeh)
{

	if (is_sensor_event(aeh)) {
		nrf_gpio_pin_toggle(pins[0]);
		return false;
	}

	if (is_ml_result_event(aeh)) {
		nrf_gpio_pin_toggle(pins[1]);
		return false;
	}

	if (is_module_state_event(aeh)) {
		nrf_gpio_cfg_output(pins[0]);
		nrf_gpio_cfg_output(pins[1]);
		return false;
	}
	__ASSERT_NO_MSG(false);

	return false;
}

APP_EVENT_LISTENER(MODULE, app_event_handler);
APP_EVENT_SUBSCRIBE(MODULE, module_state_event);
APP_EVENT_SUBSCRIBE(MODULE, sensor_event);
APP_EVENT_SUBSCRIBE_FIRST(MODULE, ml_result_event);
