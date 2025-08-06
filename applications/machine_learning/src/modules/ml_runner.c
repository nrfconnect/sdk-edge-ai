/*
 * Copyright (c) 2021 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#define MODULE ml_runner

#include "ml_app_mode_event.h"
#include "runner.h"

#include <zephyr/drivers/sensor.h>
#include <zephyr/kernel.h>

#include <caf/events/module_state_event.h>
#include <caf/events/sensor_data_aggregator_event.h>
#include <caf/events/sensor_event.h>

#define APP_CONTROLS_ML_MODE IS_ENABLED(CONFIG_ML_APP_MODE_EVENTS)

/**
 * @brief Enumeration of possible current module states
 */
enum state {
	/** Module is disabled.
	 *  The state is used only before module is initialized.
	 */
	STATE_DISABLED,
	/** Module is currently processing the data. */
	STATE_ACTIVE,
	/** The module was suspended by user. */
	STATE_SUSPENDED,
	/** Error */
	STATE_ERROR
};

BUILD_ASSERT(ARRAY_SIZE(CONFIG_ML_APP_ML_RUNNER_SENSOR_EVENT_DESCR) > 1);
static const char *handled_sensor_event_descr = CONFIG_ML_APP_ML_RUNNER_SENSOR_EVENT_DESCR;

static enum state state;

static void report_error(void)
{
	state = STATE_ERROR;
	module_set_state(MODULE_STATE_ERROR);
}

static bool continue_on_result_cb(int err)
{
	if (err) {
		report_error();
		return false;
	}

	if (state == STATE_ACTIVE) {
		err = runner_start_prediction();
		if (err && (err != -ENOSYS)) {
			report_error();
		}
	}

	return state != STATE_ERROR;
}

static bool handle_sensor_event(const struct sensor_event *event)
{
	if ((event->descr != handled_sensor_event_descr) &&
	    strcmp(event->descr, handled_sensor_event_descr)) {
		return false;
	}

	if (state != STATE_ACTIVE) {
		return false;
	}

	size_t data_cnt = sensor_event_get_data_cnt(event);
	float float_data[data_cnt];
	const struct sensor_value *data_ptr = sensor_event_get_data_ptr(event);

	for (size_t i = 0; i < data_cnt; i++) {
		float_data[i] = sensor_value_to_double(&data_ptr[i]);
	}

	int err = runner_add_data(float_data, data_cnt);

	if (err) {
		report_error();
		return false;
	}

	return false;
}

static bool handle_sensor_data_aggregator_event(const struct sensor_data_aggregator_event *event)
{
	if (state != STATE_ACTIVE) {
		return false;
	}

	if ((event->sensor_descr != handled_sensor_event_descr) &&
	    strcmp(event->sensor_descr, handled_sensor_event_descr)) {
		return false;
	}

	size_t sensor_value_cnt = (size_t)event->sample_cnt * (size_t)event->values_in_sample;

	float float_data[sensor_value_cnt];
	const struct sensor_value *data_ptr = event->samples;

	for (size_t i = 0; i < sensor_value_cnt; i++) {
		float_data[i] = sensor_value_to_double(&data_ptr[i]);
	}

	int err = runner_add_data(float_data, sensor_value_cnt);

	if (err) {
		report_error();
		return false;
	}

	return false;
}

static bool handle_ml_app_mode_event(const struct ml_app_mode_event *event)
{
	if ((event->mode == ML_APP_MODE_MODEL_RUNNING) && (state == STATE_SUSPENDED)) {
		runner_start_prediction();
		state = STATE_ACTIVE;
	} else if ((event->mode != ML_APP_MODE_MODEL_RUNNING) && (state == STATE_ACTIVE)) {
		int err = runner_stop_prediction();

		if (!err || (err == -EBUSY) || (err == -ENOSYS)) {
			state = STATE_SUSPENDED;
		}
	}

	return false;
}

static bool handle_module_state_event(const struct module_state_event *event)
{
	if (check_state(event, MODULE_ID(main), MODULE_STATE_READY)) {
		__ASSERT_NO_MSG(state == STATE_DISABLED);

		if (!runner_init(continue_on_result_cb)) {
			state = APP_CONTROLS_ML_MODE ? STATE_SUSPENDED : STATE_ACTIVE;
			module_set_state(MODULE_STATE_READY);
		} else {
			report_error();
		}
	}

	return false;
}

static bool app_event_handler(const struct app_event_header *aeh)
{
	if (is_sensor_event(aeh)) {
		return handle_sensor_event(cast_sensor_event(aeh));
	}

	if (IS_ENABLED(CONFIG_CAF_SENSOR_DATA_AGGREGATOR_EVENTS) &&
	    is_sensor_data_aggregator_event(aeh)) {
		return handle_sensor_data_aggregator_event(cast_sensor_data_aggregator_event(aeh));
	}

	if (APP_CONTROLS_ML_MODE && is_ml_app_mode_event(aeh)) {
		return handle_ml_app_mode_event(cast_ml_app_mode_event(aeh));
	}

	if (is_module_state_event(aeh)) {
		return handle_module_state_event(cast_module_state_event(aeh));
	}

	/* If event is unhandled, unsubscribe. */
	__ASSERT_NO_MSG(false);

	return false;
}

APP_EVENT_LISTENER(MODULE, app_event_handler);
APP_EVENT_SUBSCRIBE(MODULE, module_state_event);
APP_EVENT_SUBSCRIBE(MODULE, sensor_event);
#if CONFIG_CAF_SENSOR_DATA_AGGREGATOR_EVENTS
APP_EVENT_SUBSCRIBE(MODULE, sensor_data_aggregator_event);
#endif /* CONFIG_CAF_SENSOR_DATA_AGGREGATOR_EVENTS */
#if APP_CONTROLS_ML_MODE
APP_EVENT_SUBSCRIBE(MODULE, ml_app_mode_event);
#endif /* APP_CONTROLS_ML_MODE */
