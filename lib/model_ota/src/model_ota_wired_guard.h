/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Model OTA guard reader held by the wired loaders while they read the model partition, so an
 * SMP upload cannot erase or rewrite it mid-load and a load is refused once an upload began.
 */

#ifndef MODEL_OTA_WIRED_GUARD_H_
#define MODEL_OTA_WIRED_GUARD_H_

#include <stdbool.h>

#include <zephyr/sys/util.h>

#if IS_ENABLED(CONFIG_MODEL_OTA_SMP)
#include <model_ota/model_ota_guard.h>
#endif

/** @return True when the partition may be read; pair with model_ota_wired_guard_release(). */
static inline bool model_ota_wired_guard_acquire(void)
{
#if IS_ENABLED(CONFIG_MODEL_OTA_SMP)
	return model_ota_guard_acquire() == 0;
#else
	return true;
#endif
}

static inline void model_ota_wired_guard_release(void)
{
#if IS_ENABLED(CONFIG_MODEL_OTA_SMP)
	model_ota_guard_release();
#endif
}

#endif /* MODEL_OTA_WIRED_GUARD_H_ */
