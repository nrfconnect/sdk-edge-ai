/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <model_ota/model_ota_smp.h>

#include <errno.h>
#include <string.h>

#include <sysflash/sysflash.h>
#include <zephyr/dfu/mcuboot.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/mgmt/mcumgr/mgmt/callbacks.h>
#include <zephyr/mgmt/mcumgr/grp/img_mgmt/img_mgmt.h>
#include <zephyr/mgmt/mcumgr/grp/img_mgmt/img_mgmt_callbacks.h>
#include <zephyr/mgmt/mcumgr/mgmt/mgmt.h>
#include <zephyr/storage/flash_map.h>
#include <zephyr/sys/atomic.h>

LOG_MODULE_REGISTER(model_ota_smp, CONFIG_MODEL_OTA_LOG_LEVEL);

#define SMP_UPLOAD_LOG_RATE_MS 1000

struct model_ota_smp_slot_state {
	uint8_t image_index;
	const char *name;
	atomic_t upload_active;
};

static struct model_ota_smp_slot_state registered_slots[CONFIG_MODEL_OTA_SMP_MAX_SLOTS];
static size_t registered_slot_count;
static atomic_t smp_upload_pending_reset;
static bool smp_initialized;
static model_ota_smp_upload_notify_cb upload_notify_cb;
static K_MUTEX_DEFINE(model_inference_lock);

static struct model_ota_smp_slot_state *slot_for_image(uint8_t image_index)
{
	for (size_t i = 0; i < registered_slot_count; i++) {
		if (registered_slots[i].image_index == image_index) {
			return &registered_slots[i];
		}
	}

	return NULL;
}

static bool slot_blocks_inference(const struct model_ota_smp_slot_state *slot)
{
	if (slot != NULL && atomic_get(&slot->upload_active) != 0) {
		if (slot->name != NULL) {
			LOG_WRN_RATELIMIT_RATE(SMP_UPLOAD_LOG_RATE_MS,
					       "%s model SMP upload in progress - inference paused",
					       slot->name);
		}
		return true;
	}

	if (model_ota_smp_is_pending_reset()) {
		if (slot != NULL && slot->name != NULL) {
			LOG_WRN_RATELIMIT_RATE(
				SMP_UPLOAD_LOG_RATE_MS,
				"%s model SMP upload complete - reset device to load new model",
				slot->name);
		}
		return true;
	}

	return false;
}

static void notify_upload_active(bool active)
{
	if (upload_notify_cb != NULL) {
		upload_notify_cb(active);
	}
}

static struct model_ota_smp_slot_state *active_upload_slot(void)
{
	for (size_t i = 0; i < registered_slot_count; i++) {
		if (atomic_get(&registered_slots[i].upload_active) != 0) {
			return &registered_slots[i];
		}
	}

	return NULL;
}

static int clear_inplace_model_trailer(uint8_t image_index)
{
	const struct flash_area *fa;
	ssize_t trailer_off;
	size_t erase_size;
	int rc;

	rc = flash_area_open(FLASH_AREA_IMAGE_PRIMARY(image_index), &fa);
	if (rc != 0) {
		return rc;
	}

	trailer_off = boot_get_trailer_status_offset(fa->fa_size);
	if (trailer_off < 0) {
		flash_area_close(fa);
		return (int)trailer_off;
	}

	erase_size = fa->fa_size - (size_t)trailer_off;
	rc = flash_area_flatten(fa, (off_t)trailer_off, erase_size);
	flash_area_close(fa);

	return rc;
}

static void finalize_inplace_model_update(struct model_ota_smp_slot_state *slot)
{
	int swap_type_before;
	int swap_type_after;
	int rc;

	if (slot == NULL) {
		return;
	}

	swap_type_before = mcuboot_swap_type_multi(slot->image_index);
	if (swap_type_before == BOOT_SWAP_TYPE_NONE) {
		return;
	}

	rc = clear_inplace_model_trailer(slot->image_index);
	if (rc != 0) {
		if (slot->name != NULL) {
			LOG_WRN("%s model image %u trailer clear failed (err %d)", slot->name,
				slot->image_index, rc);
		} else {
			LOG_WRN("Model image %u trailer clear failed (err %d)", slot->image_index,
				rc);
		}
		return;
	}

	swap_type_after = mcuboot_swap_type_multi(slot->image_index);
	if (slot->name != NULL) {
		LOG_INF("%s model image %u finalized in place (swap %d -> %d)", slot->name,
			slot->image_index, swap_type_before, swap_type_after);
	}
}

static void finalize_all_inplace_model_updates(void)
{
	for (size_t i = 0; i < registered_slot_count; i++) {
		finalize_inplace_model_update(&registered_slots[i]);
	}
}

static void finish_model_upload(bool pending_reset)
{
	struct model_ota_smp_slot_state *slot = active_upload_slot();

	if (slot == NULL) {
		return;
	}

	atomic_set(&slot->upload_active, 0);
	if (pending_reset) {
		atomic_set(&smp_upload_pending_reset, 1);
	}
	notify_upload_active(false);
	k_mutex_unlock(&model_inference_lock);

	if (slot->name != NULL) {
		if (pending_reset) {
			LOG_WRN("%s model SMP upload finished - reset device to load new model",
				slot->name);
		} else {
			LOG_WRN("%s model SMP upload aborted - inference resumed", slot->name);
		}
	}
}

static enum mgmt_cb_return model_ota_smp_callback(uint32_t event, enum mgmt_cb_return prev_status,
						  int32_t *rc, uint16_t *group, bool *abort_more,
						  void *data, size_t data_size)
{
	ARG_UNUSED(prev_status);
	ARG_UNUSED(group);
	ARG_UNUSED(abort_more);

	switch (event) {
	case MGMT_EVT_OP_IMG_MGMT_DFU_CHUNK: {
		const struct img_mgmt_upload_check *upload_check;
		const struct img_mgmt_upload_req *req;
		struct model_ota_smp_slot_state *slot;

		if (data == NULL || data_size != sizeof(struct img_mgmt_upload_check)) {
			break;
		}

		upload_check = data;
		req = upload_check->req;

		if (req == NULL || req->off != 0) {
			break;
		}

		slot = slot_for_image(req->image);
		if (slot == NULL) {
			break;
		}

		if (k_mutex_lock(&model_inference_lock,
				 K_MSEC(CONFIG_MODEL_OTA_SMP_UPLOAD_LOCK_TIMEOUT_MS)) != 0) {
			LOG_WRN("Model SMP upload rejected - inference still running");
			*rc = MGMT_ERR_EBUSY;
			return MGMT_CB_ERROR_RC;
		}

		atomic_set(&slot->upload_active, 1);
		notify_upload_active(true);
		if (slot->name != NULL) {
			LOG_WRN("%s model SMP upload started - inference paused until reset",
				slot->name);
		}
		break;
	}

	case MGMT_EVT_OP_IMG_MGMT_DFU_PENDING:
		finalize_all_inplace_model_updates();
		finish_model_upload(true);
		break;

	case MGMT_EVT_OP_IMG_MGMT_DFU_STOPPED:
		finish_model_upload(false);
		break;

	default:
		break;
	}

	return MGMT_CB_OK;
}

static struct mgmt_callback model_ota_smp_mgmt_cb = {
	.callback = model_ota_smp_callback,
	.event_id = (MGMT_EVT_OP_IMG_MGMT_DFU_CHUNK | MGMT_EVT_OP_IMG_MGMT_DFU_PENDING |
		     MGMT_EVT_OP_IMG_MGMT_DFU_STOPPED),
};

int model_ota_smp_init(const struct model_ota_smp_slot *slots, size_t slot_count)
{
	if (slots == NULL || slot_count == 0) {
		return -EINVAL;
	}

	if (slot_count > CONFIG_MODEL_OTA_SMP_MAX_SLOTS) {
		return -EINVAL;
	}

	if (smp_initialized) {
		return -EALREADY;
	}

	memset(registered_slots, 0, sizeof(registered_slots));
	registered_slot_count = 0;

	for (size_t i = 0; i < slot_count; i++) {
		if (slots[i].image_index == 0) {
			return -EINVAL;
		}

		for (size_t j = 0; j < i; j++) {
			if (slots[j].image_index == slots[i].image_index) {
				return -EINVAL;
			}
		}

		registered_slots[i].image_index = slots[i].image_index;
		registered_slots[i].name = slots[i].name;
		registered_slot_count++;
	}

	mgmt_callback_register(&model_ota_smp_mgmt_cb);
	smp_initialized = true;
	finalize_all_inplace_model_updates();

	return 0;
}

bool model_ota_smp_blocks_inference(uint8_t image_index)
{
	struct model_ota_smp_slot_state *slot = slot_for_image(image_index);

	if (slot == NULL) {
		return false;
	}

	return slot_blocks_inference(slot);
}

bool model_ota_smp_is_pending_reset(void)
{
	return atomic_get(&smp_upload_pending_reset) != 0;
}

void model_ota_smp_set_upload_notify_cb(model_ota_smp_upload_notify_cb cb)
{
	upload_notify_cb = cb;
}
