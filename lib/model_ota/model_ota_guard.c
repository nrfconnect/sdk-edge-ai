/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <model_ota/model_ota_guard.h>

#include <nrf_edgeai/rt/nrf_edgeai_types.h>
#include <nrf_edgeai/rt/private/nrf_edgeai_gate.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/atomic.h>
#include <zephyr/sys/clock.h>

static K_MUTEX_DEFINE(guard_mutex);
static K_CONDVAR_DEFINE(readers_cv);
static atomic_t global_state = ATOMIC_INIT(MODEL_OTA_GUARD_READY);
static uint32_t global_readers;

static bool state_permits_access(enum model_ota_guard_state state)
{
	return state == MODEL_OTA_GUARD_READY;
}

static enum model_ota_guard_state current_state(void)
{
	return (enum model_ota_guard_state)atomic_get(&global_state);
}

static int reader_acquire(void)
{
	k_mutex_lock(&guard_mutex, K_FOREVER);

	if (!state_permits_access(current_state())) {
		k_mutex_unlock(&guard_mutex);
		return -EBUSY;
	}

	global_readers++;
	k_mutex_unlock(&guard_mutex);

	return 0;
}

static void reader_release(void)
{
	k_mutex_lock(&guard_mutex, K_FOREVER);

	if (global_readers > 0U) {
		global_readers--;
	}

	k_condvar_broadcast(&readers_cv);
	k_mutex_unlock(&guard_mutex);
}

static bool readers_idle(void)
{
	bool idle;

	k_mutex_lock(&guard_mutex, K_FOREVER);
	idle = (global_readers == 0U);
	k_mutex_unlock(&guard_mutex);

	return idle;
}

static bool wait_for_readers_zero(k_timeout_t timeout)
{
	k_timepoint_t deadline = sys_timepoint_calc(timeout);

	k_mutex_lock(&guard_mutex, K_FOREVER);

	while (global_readers > 0U) {
		k_timeout_t remaining = sys_timepoint_timeout(deadline);

		if (K_TIMEOUT_EQ(remaining, K_NO_WAIT) && !K_TIMEOUT_EQ(timeout, K_FOREVER)) {
			k_mutex_unlock(&guard_mutex);
			return false;
		}

		int rc = k_condvar_wait(&readers_cv, &guard_mutex, remaining);

		if (rc != 0) {
			k_mutex_unlock(&guard_mutex);
			return false;
		}
	}

	k_mutex_unlock(&guard_mutex);

	return true;
}

bool model_ota_guard_inference_permitted(void)
{
	return state_permits_access(current_state());
}

int model_ota_guard_acquire(void)
{
	return reader_acquire();
}

void model_ota_guard_release(void)
{
	reader_release();
}

int model_ota_guard_begin_update(void)
{
	atomic_set(&global_state, MODEL_OTA_GUARD_BLOCKED);

	if (!wait_for_readers_zero(K_MSEC(CONFIG_MODEL_OTA_GUARD_DRAIN_TIMEOUT_MS))) {
		return -EBUSY;
	}

	return 0;
}

int model_ota_guard_retry_update(void)
{
	if (current_state() != MODEL_OTA_GUARD_BLOCKED) {
		return -EINVAL;
	}

	return readers_idle() ? 0 : -EBUSY;
}

void model_ota_guard_abort_update(void)
{
	atomic_set(&global_state, MODEL_OTA_GUARD_READY);
}

bool nrf_edgeai_guard_inference_permitted(const nrf_edgeai_t *edgeai)
{
	ARG_UNUSED(edgeai);

	return model_ota_guard_inference_permitted();
}

int nrf_edgeai_guard_inference_session_begin(const nrf_edgeai_t *edgeai)
{
	ARG_UNUSED(edgeai);

	return reader_acquire();
}

void nrf_edgeai_guard_inference_session_end(const nrf_edgeai_t *edgeai)
{
	ARG_UNUSED(edgeai);

	reader_release();
}
