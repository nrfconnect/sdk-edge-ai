/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#define MODULE neuton_runner

#include "ml_result_event.h"
#include "model_categories_def.h"
#include "runner.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <neuton/neuton.h>

LOG_MODULE_REGISTER(MODULE, CONFIG_ML_APP_ML_RUNNER_LOG_LEVEL);

/* Make sure that event handlers will not be preempted by the Neuton */
BUILD_ASSERT(CONFIG_SYSTEM_WORKQUEUE_PRIORITY < CONFIG_ML_APP_ML_RUNNER_NEUTON_THREAD_PRIORITY);

BUILD_ASSERT(CONFIG_ML_APP_ML_RUNNER_NEUTON_THREAD_STACK_SIZE > 0);

static K_THREAD_STACK_DEFINE(thread_stack, CONFIG_ML_APP_ML_RUNNER_NEUTON_THREAD_STACK_SIZE);
static struct k_thread thread;

static K_SEM_DEFINE(inference_sem, 0, 1);

static neuton_input_features_t *inference_input;
static runner_continue_callback continue_cb;

static void submit_result(const neuton_u16_t predicted, const neuton_output_t probability)
{
	struct ml_result_event *evt = new_ml_result_event();

	evt->label = model_categories[predicted];
	evt->value = (float)probability;
	evt->anomaly = 0.0;

	APP_EVENT_SUBMIT(evt);
}

static void neuton_thread_fn(void)
{
	while (true) {
		k_sem_take(&inference_sem, K_FOREVER);

		if (!inference_input) {
			const int err = -EINVAL;
			continue_cb(err);
		}

		neuton_u16_t predicted_target;
		const neuton_output_t *probabilities;

		const neuton_i16_t targets_num =
			neuton_nn_run_inference(inference_input, &predicted_target, &probabilities);

		const int err = (targets_num < 0) ? targets_num : 0;
		if (err) {
			LOG_ERR("Inference failed (err: %d)", err);
		}

		const bool module_ok = continue_cb(err);

		if (!err && module_ok) {
			submit_result(predicted_target, probabilities[predicted_target]);
		}
	}
}

int runner_init(const runner_continue_callback cb)
{
	if (!cb) {
		return -EINVAL;
	}

	if (continue_cb) {
		return -EALREADY;
	}

	continue_cb = cb;

	neuton_nn_setup();

	k_thread_create(&thread, thread_stack, CONFIG_ML_APP_ML_RUNNER_NEUTON_THREAD_STACK_SIZE,
			(k_thread_entry_t)neuton_thread_fn, NULL, NULL, NULL,
			CONFIG_ML_APP_ML_RUNNER_NEUTON_THREAD_PRIORITY, 0, K_NO_WAIT);
	k_thread_name_set(&thread, "neuton_thread");

	return 0;
}

int runner_start_prediction(void)
{
	return -ENOSYS;
}

int runner_stop_prediction(void)
{
	return -ENOSYS;
}

int runner_add_data(const neuton_input_t *data, size_t data_size)
{
	inference_input = neuton_nn_feed_inputs(data, data_size);

	if (inference_input) {
		k_sem_give(&inference_sem);
	}

	return 0;
}
