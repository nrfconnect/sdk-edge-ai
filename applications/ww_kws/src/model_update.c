/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/mgmt/mcumgr/mgmt/callbacks.h>
#include <zephyr/mgmt/mcumgr/grp/img_mgmt/img_mgmt.h>
#include <zephyr/mgmt/mcumgr/grp/img_mgmt/img_mgmt_callbacks.h>
#include <zephyr/mgmt/mcumgr/mgmt/mgmt.h>
#include <zephyr/sys/atomic.h>

#include "model_update.h"

LOG_MODULE_DECLARE(main);

#define SMP_UPLOAD_LOG_RATE_MS 1000

static atomic_t ww_smp_upload_active;
static atomic_t kws_smp_upload_active;
static atomic_t smp_upload_pending_reset;
static K_MUTEX_DEFINE(model_inference_lock);

static enum mgmt_cb_return model_smp_upload_callback(uint32_t event, enum mgmt_cb_return prev_status,
						     int32_t *rc, uint16_t *group, bool *abort_more,
						     void *data, size_t data_size)
{
	ARG_UNUSED(prev_status);
	ARG_UNUSED(group);
	ARG_UNUSED(abort_more);

	switch (event) {
	case MGMT_EVT_OP_IMG_MGMT_DFU_CHUNK: {
		const struct img_mgmt_upload_check *upload_check;

		if (data == NULL || data_size != sizeof(struct img_mgmt_upload_check)) {
			break;
		}

		upload_check = data;

		const struct img_mgmt_upload_req *req = upload_check->req;

		if (req == NULL) {
			break;
		}

		if ((req->image == WW_MODEL_IMAGE_INDEX || req->image == KWS_MODEL_IMAGE_INDEX) &&
		    req->off == 0) {
			if (k_mutex_lock(&model_inference_lock, K_SECONDS(60)) != 0) {
				LOG_WRN("Model SMP upload rejected - inference still running");
				*rc = MGMT_ERR_EBUSY;
				return MGMT_CB_ERROR_RC;
			}

			if (req->image == WW_MODEL_IMAGE_INDEX) {
				atomic_set(&ww_smp_upload_active, 1);
				LOG_WRN("WW model SMP upload started - inference paused until reset");
			} else {
				atomic_set(&kws_smp_upload_active, 1);
				LOG_WRN("KWS model SMP upload started - inference paused until reset");
			}
		}
		break;
	}

	case MGMT_EVT_OP_IMG_MGMT_DFU_STOPPED:
		if (atomic_get(&ww_smp_upload_active)) {
			atomic_set(&ww_smp_upload_active, 0);
			atomic_set(&smp_upload_pending_reset, 1);
			k_mutex_unlock(&model_inference_lock);
			LOG_WRN("WW model SMP upload finished - reset device to load new model");
		} else if (atomic_get(&kws_smp_upload_active)) {
			atomic_set(&kws_smp_upload_active, 0);
			atomic_set(&smp_upload_pending_reset, 1);
			k_mutex_unlock(&model_inference_lock);
			LOG_WRN("KWS model SMP upload finished - reset device to load new model");
		}
		break;

	default:
		break;
	}

	return MGMT_CB_OK;
}

static struct mgmt_callback model_smp_upload_mgmt_cb = {
	.callback = model_smp_upload_callback,
	.event_id = (MGMT_EVT_OP_IMG_MGMT_DFU_CHUNK | MGMT_EVT_OP_IMG_MGMT_DFU_STOPPED),
};

void model_update_init(void)
{
	mgmt_callback_register(&model_smp_upload_mgmt_cb);
}

bool model_update_is_pending_reset(void)
{
	return atomic_get(&smp_upload_pending_reset) != 0;
}

static bool model_update_blocks_inference(atomic_t *upload_active, const char *model_name)
{
	if (atomic_get(upload_active)) {
		LOG_WRN_RATELIMIT_RATE(SMP_UPLOAD_LOG_RATE_MS,
				  "%s model SMP upload in progress - inference paused", model_name);
		return true;
	}

	if (model_update_is_pending_reset()) {
		LOG_WRN_RATELIMIT_RATE(SMP_UPLOAD_LOG_RATE_MS,
				  "%s model SMP upload complete - reset device to load new model", model_name);
		return true;
	}

	return false;
}

bool model_update_blocks_ww_inference(void)
{
	return model_update_blocks_inference(&ww_smp_upload_active, "WW");
}

bool model_update_blocks_kws_inference(void)
{
	return model_update_blocks_inference(&kws_smp_upload_active, "KWS");
}
