/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#define MODULE stub_runner

#include "ml_result_event.h"
#include "runner.h"

#include <zephyr/kernel.h>

#define THREAD_STACK_SIZE CONFIG_ML_APP_ML_RUNNER_STUB_THREAD_STACK_SIZE
#define THREAD_PRIORITY   CONFIG_ML_APP_ML_RUNNER_STUB_THREAD_PRIORITY

/* Make sure that event handlers will not be preempted by the Neuton */
BUILD_ASSERT(CONFIG_SYSTEM_WORKQUEUE_PRIORITY < THREAD_PRIORITY);

BUILD_ASSERT(THREAD_STACK_SIZE > 0);

static K_THREAD_STACK_DEFINE(thread_stack, THREAD_STACK_SIZE);
static struct k_thread thread;

static K_SEM_DEFINE(inference_sem, 0, 1);

static void stub_thread_fn(void)
{
	while (true) {
		k_sem_take(&inference_sem, K_FOREVER);

		struct ml_result_event *evt = new_ml_result_event();

		evt->label = "stub";
		evt->value = 0.0;
		evt->anomaly = 0.0;

		APP_EVENT_SUBMIT(evt);
	}
}

int runner_init(const runner_continue_callback cb)
{
	k_thread_create(&thread, thread_stack, THREAD_STACK_SIZE, (k_thread_entry_t)stub_thread_fn,
			NULL, NULL, NULL, THREAD_PRIORITY, 0, K_NO_WAIT);
	k_thread_name_set(&thread, "stub_thread");

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

#define FRAME_SIZE  CONFIG_ML_APP_ML_RUNNER_STUB_FRAME_SIZE
#define WINDOW_SIZE CONFIG_ML_APP_ML_RUNNER_STUB_WINDOW_SIZE

int runner_add_data(const float *data, size_t data_size)
{
	static uint8_t count = 0;

	count += data_size / FRAME_SIZE;

	if (count >= WINDOW_SIZE) {
		count -= WINDOW_SIZE;
		k_sem_give(&inference_sem);
	}

	return 0;
}
