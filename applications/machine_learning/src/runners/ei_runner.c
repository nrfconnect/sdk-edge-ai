/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#define MODULE ei_runner

#include "ml_result_event.h"
#include "runner.h"

#include <zephyr/logging/log.h>

#include <ei_wrapper.h>

LOG_MODULE_REGISTER(MODULE, CONFIG_ML_APP_ML_RUNNER_LOG_LEVEL);

#define SHIFT_WINDOWS CONFIG_ML_APP_ML_RUNNER_EI_WINDOW_SHIFT
#define SHIFT_FRAMES  CONFIG_ML_APP_ML_RUNNER_EI_FRAME_SHIFT

#define APP_CONTROLS_ML_MODE IS_ENABLED(CONFIG_ML_APP_MODE_EVENTS)

/* Make sure that event handlers will not be preempted by the EI wrapper's callback. */
BUILD_ASSERT(CONFIG_SYSTEM_WORKQUEUE_PRIORITY < CONFIG_EI_WRAPPER_THREAD_PRIORITY);

enum {
	ML_DROP_RESULT = BIT(0),
	ML_CLEANUP_REQUIRED = BIT(1),
	ML_FIRST_PREDICTION = BIT(2),
	ML_RUNNING = BIT(3),
};

static uint8_t ml_control;
static runner_continue_callback continue_cb;

static void submit_result(void)
{
	struct ml_result_event *evt = new_ml_result_event();

	int err = ei_wrapper_get_next_classification_result(&evt->label, &evt->value, NULL);

	if (!err && ei_wrapper_classifier_has_anomaly()) {
		err = ei_wrapper_get_anomaly(&evt->anomaly);
	} else {
		evt->anomaly = 0.0;
	}

	__ASSERT_NO_MSG(!err);
	ARG_UNUSED(err);

	APP_EVENT_SUBMIT(evt);
}

static int buf_cleanup(void)
{
	bool cancelled = false;
	int err = ei_wrapper_clear_data(&cancelled);

	if (!err) {
		if (cancelled) {
			ml_control &= ~ML_RUNNING;
		}

		if (ml_control & ML_RUNNING) {
			ml_control |= ML_DROP_RESULT;
		}

		ml_control &= ~ML_CLEANUP_REQUIRED;
		ml_control |= ML_FIRST_PREDICTION;
	} else if (err == -EBUSY) {
		__ASSERT_NO_MSG(ml_control & ML_RUNNING);
		ml_control |= ML_DROP_RESULT;
		ml_control |= ML_CLEANUP_REQUIRED;

		err = 0;
	} else {
		LOG_ERR("Cannot cleanup buffer (err: %d)", err);
	}

	return err;
}

static void result_ready_cb(int err)
{
	k_sched_lock();

	const bool drop_result = (err) || (ml_control & ML_DROP_RESULT);

	if (err) {
		LOG_ERR("Result ready callback returned error (err: %d)", err);
	} else {
		ml_control &= ~ML_DROP_RESULT;
		ml_control &= ~ML_RUNNING;
	}

	const bool module_ok = continue_cb(err);

	k_sched_unlock();

	if (!drop_result && module_ok) {
		submit_result();
	}
}

int runner_init(const runner_continue_callback cb)
{
	if (!cb) {
		return -EINVAL;
	}

	ml_control |= ML_FIRST_PREDICTION;

	continue_cb = cb;

	int err = ei_wrapper_init(result_ready_cb);

	if (err) {
		LOG_ERR("Edge Impulse wrapper failed to initialize (err: %d)", err);
	} else if (!APP_CONTROLS_ML_MODE) {
		err = runner_start_prediction();
	}

	return err;
}

int runner_start_prediction(void)
{
	int err;
	size_t window_shift;
	size_t frame_shift;

	if (ml_control & ML_RUNNING) {
		return 0;
	}

	if (ml_control & ML_CLEANUP_REQUIRED) {
		err = buf_cleanup();
		if (err) {
			return err;
		}
	}

	if (ml_control & ML_FIRST_PREDICTION) {
		window_shift = 0;
		frame_shift = 0;
	} else {
		window_shift = SHIFT_WINDOWS;
		frame_shift = SHIFT_FRAMES;
	}

	err = ei_wrapper_start_prediction(window_shift, frame_shift);

	if (!err) {
		ml_control |= ML_RUNNING;
		ml_control &= ~ML_FIRST_PREDICTION;
	} else {
		LOG_ERR("Cannot start prediction (err: %d)", err);
	}

	return err;
}

int runner_stop_prediction(void)
{
	return buf_cleanup();
}

int runner_add_data(const float *data, size_t data_size)
{
	int err = ei_wrapper_add_data(data, data_size);
	if (err) {
		LOG_ERR("Cannot add data for model runner (err %d)", err);
	}
	return err;
}
