/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <model_ota/model_ota_smp.h>

#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

#include "model_update.h"
#include "leds.h"

LOG_MODULE_REGISTER(model_update);

static const struct model_ota_smp_slot ww_kws_smp_slots[] = {
	{
		.image_index = WW_MODEL_IMAGE_INDEX,
		.name = "WW",
	},
	{
		.image_index = KWS_MODEL_IMAGE_INDEX,
		.name = "KWS",
	},
};

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

	int err = model_ota_smp_init(ww_kws_smp_slots, ARRAY_SIZE(ww_kws_smp_slots));

	if (err != 0) {
		LOG_ERR("Model SMP coordination init failed (err %d, slots %u, max %u)", err,
			(unsigned)ARRAY_SIZE(ww_kws_smp_slots),
			(unsigned)CONFIG_MODEL_OTA_SMP_MAX_SLOTS);
	}

	return err;
}

bool model_update_blocks_ww_inference(void)
{
	return model_ota_smp_blocks_inference(WW_MODEL_IMAGE_INDEX);
}

bool model_update_blocks_kws_inference(void)
{
	return model_ota_smp_blocks_inference(KWS_MODEL_IMAGE_INDEX);
}

bool model_update_is_pending_reset(void)
{
	return model_ota_smp_is_pending_reset();
}
