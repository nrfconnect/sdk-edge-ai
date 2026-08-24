/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <model_ota/model_ota_smp.h>

#include <zephyr/logging/log.h>

#include "model_update.h"
#include "leds.h"

LOG_MODULE_REGISTER(model_update);

static void model_update_upload_notify(bool active)
{
	if (active) {
		leds_off_led2();
		leds_on_led1();
		return;
	}

	if (model_ota_smp_is_pending_reset()) {
		leds_off_led1();
		leds_on_led2();
	}
}

int model_update_init(void)
{
	model_ota_smp_set_upload_notify_cb(model_update_upload_notify);

	return 0;
}

bool model_update_is_pending_reset(void)
{
	return model_ota_smp_is_pending_reset();
}
