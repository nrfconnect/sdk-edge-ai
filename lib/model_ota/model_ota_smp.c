/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <model_ota/model_ota_smp.h>
#include <model_ota/model_ota_guard.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/mgmt/mcumgr/mgmt/callbacks.h>
#include <zephyr/mgmt/mcumgr/grp/img_mgmt/img_mgmt.h>
#include <zephyr/mgmt/mcumgr/grp/img_mgmt/img_mgmt_callbacks.h>
#include <zephyr/mgmt/mcumgr/mgmt/mgmt.h>

LOG_MODULE_REGISTER(model_ota_smp, CONFIG_MODEL_OTA_LOG_LEVEL);

/*
 * The inference guard is device-wide and MCUmgr keeps a single upload session
 * (g_img_mgmt_state), so all registered slots share one state machine under one
 * lock. img_mgmt events drive it:
 *
 * IDLE        Guard READY, nothing reserved.
 * DRAIN_RETRY Readers did not drain within CONFIG_MODEL_OTA_GUARD_DRAIN_TIMEOUT_MS
 *             at chunk 0, so that chunk was rejected with MGMT_ERR_EBUSY. The
 *             guard stays BLOCKED, which keeps new readers out while the
 *             in-flight ones finish. img_mgmt dropped the upload session along
 *             with the rejection (DFU_STOPPED plus img_mgmt_reset_upload()), so
 *             a client retry arrives as another chunk 0 that only needs a
 *             no-drain re-check. Nothing was written, so if that retry does not
 *             come within CONFIG_MODEL_OTA_SMP_DRAIN_RETRY_WINDOW_MS, or comes
 *             while readers are still running, the guard is released and
 *             inference continues on the unchanged model.
 * ARMED       Chunk 0 authorised with the guard blocked and drained, waiting for
 *             DFU_STARTED. Still rollback-safe: img_mgmt erases and writes only
 *             after DFU_STARTED, so a DFU_STOPPED here (another upload-check
 *             handler rejecting the same chunk) releases the guard.
 * ACTIVE      DFU_STARTED seen, the model partition is being written.
 * LOCKED      Upload ended - finished, aborted or superseded by a non-model
 *             image. Blocked until device reset.
 */
enum smp_upload_state {
	SMP_UPLOAD_IDLE = 0,
	SMP_UPLOAD_DRAIN_RETRY,
	SMP_UPLOAD_ARMED,
	SMP_UPLOAD_ACTIVE,
	SMP_UPLOAD_LOCKED,
};

struct model_ota_smp_slot_state {
	uint8_t image_index;
	const char *name;
};

static void drain_retry_window_expired(struct k_work *work);

static K_MUTEX_DEFINE(smp_lock);
static K_WORK_DELAYABLE_DEFINE(drain_retry_work, drain_retry_window_expired);

static struct model_ota_smp_slot_state registered_slots[CONFIG_MODEL_OTA_SMP_MAX_SLOTS];
static size_t registered_slot_count;
static bool smp_initialized;
static model_ota_smp_upload_notify_cb upload_notify_cb;
static enum smp_upload_state upload_state;
/** Model slot of the session tracked by @ref upload_state; NULL when IDLE. */
static struct model_ota_smp_slot_state *upload_slot;

static struct model_ota_smp_slot_state *slot_for_image(uint8_t image_index)
{
	for (size_t i = 0; i < registered_slot_count; i++) {
		if (registered_slots[i].image_index == image_index) {
			return &registered_slots[i];
		}
	}

	return NULL;
}

static const char *slot_name(const struct model_ota_smp_slot_state *slot)
{
	if (slot == NULL || slot->name == NULL) {
		return "Unnamed";
	}

	return slot->name;
}

static void notify_upload_active(bool active)
{
	if (upload_notify_cb != NULL) {
		upload_notify_cb(active);
	}
}

/** True while the guard is blocked with readers drained, so flash may be written. */
static bool guard_held(void)
{
	return upload_state == SMP_UPLOAD_ARMED || upload_state == SMP_UPLOAD_ACTIVE ||
	       upload_state == SMP_UPLOAD_LOCKED;
}

/** Hand the guard back: no model flash was modified, so inference resumes. */
static void release_guard(void)
{
	(void)k_work_cancel_delayable(&drain_retry_work);
	model_ota_guard_abort_update();
	upload_slot = NULL;
	upload_state = SMP_UPLOAD_IDLE;
}

static void lock_until_reset(bool upload_completed)
{
	bool was_active = (upload_state == SMP_UPLOAD_ACTIVE);
	const char *name = slot_name(upload_slot);

	(void)k_work_cancel_delayable(&drain_retry_work);
	upload_slot = NULL;
	upload_state = SMP_UPLOAD_LOCKED;

	if (upload_completed) {
		LOG_WRN("%s model SMP upload finished - reset device to load new model", name);
	} else {
		LOG_WRN("%s model SMP upload aborted - reset device to restore model", name);
	}

	if (was_active) {
		notify_upload_active(false);
	}
}

static enum mgmt_cb_return reserve_guard(struct model_ota_smp_slot_state *slot, int32_t *rc)
{
	switch (upload_state) {
	case SMP_UPLOAD_IDLE:
		if (model_ota_guard_begin_update() != 0) {
			upload_slot = slot;
			upload_state = SMP_UPLOAD_DRAIN_RETRY;
			(void)k_work_schedule(&drain_retry_work,
					      K_MSEC(CONFIG_MODEL_OTA_SMP_DRAIN_RETRY_WINDOW_MS));

			LOG_WRN("%s model SMP upload deferred - inference draining, "
				"retry upload within %d ms",
				slot_name(slot), CONFIG_MODEL_OTA_SMP_DRAIN_RETRY_WINDOW_MS);
			*rc = MGMT_ERR_EBUSY;

			return MGMT_CB_ERROR_RC;
		}

		break;

	case SMP_UPLOAD_DRAIN_RETRY:
		if (model_ota_guard_retry_update() != 0) {
			LOG_WRN("%s model SMP upload rejected - inference still running, "
				"model left unchanged",
				slot_name(slot));
			release_guard();
			*rc = MGMT_ERR_EBUSY;

			return MGMT_CB_ERROR_RC;
		}

		(void)k_work_cancel_delayable(&drain_retry_work);
		break;

	case SMP_UPLOAD_ARMED:
	case SMP_UPLOAD_ACTIVE:
		/* Guard already blocked and drained; a restarted transfer needs no drain. */
		upload_slot = slot;

		return MGMT_CB_OK;

	case SMP_UPLOAD_LOCKED:
		/*
		 * Model flash was already modified, so the guard stays blocked however
		 * this upload ends. Do not re-arm: a rollback to READY is no longer safe.
		 */
		return MGMT_CB_OK;
	}

	upload_slot = slot;
	upload_state = SMP_UPLOAD_ARMED;

	return MGMT_CB_OK;
}

static void supersede_by_other_image(uint8_t image_index)
{
	switch (upload_state) {
	case SMP_UPLOAD_DRAIN_RETRY:
	case SMP_UPLOAD_ARMED:
		LOG_INF("Model SMP upload dropped for image %u - inference resumed", image_index);
		release_guard();
		break;

	case SMP_UPLOAD_ACTIVE:
		lock_until_reset(false);
		break;

	default:
		break;
	}
}

static enum mgmt_cb_return handle_dfu_chunk(const struct img_mgmt_upload_check *upload_check,
					    int32_t *rc)
{
	const struct img_mgmt_upload_req *req = upload_check->req;
	struct model_ota_smp_slot_state *slot;

	if (req == NULL) {
		return MGMT_CB_OK;
	}

	slot = slot_for_image(req->image);

	if (req->off != 0U) {
		/* Never let a write land on a model partition a reader may still use. */
		if (slot != NULL && !guard_held()) {
			*rc = MGMT_ERR_EBUSY;

			return MGMT_CB_ERROR_RC;
		}

		return MGMT_CB_OK;
	}

	if (slot == NULL) {
		/*
		 * A non-model upload takes over the single MCUmgr upload session: no
		 * retry of our chunk 0 can arrive any more, and a model transfer that
		 * already wrote flash stays blocked until reset.
		 */
		supersede_by_other_image(req->image);

		return MGMT_CB_OK;
	}

	return reserve_guard(slot, rc);
}

static void handle_dfu_started(void)
{
	if (upload_state != SMP_UPLOAD_ARMED) {
		return;
	}

	upload_state = SMP_UPLOAD_ACTIVE;
	LOG_WRN("%s model SMP upload started - inference blocked until reset",
		slot_name(upload_slot));
	notify_upload_active(true);
}

static void handle_dfu_pending(void)
{
	if (upload_state == SMP_UPLOAD_ARMED || upload_state == SMP_UPLOAD_ACTIVE) {
		lock_until_reset(true);
	}
}

static void handle_dfu_stopped(void)
{
	switch (upload_state) {
	case SMP_UPLOAD_ARMED:
		/* Rejected before any erase or write, so the model is untouched. */
		LOG_INF("%s model SMP upload did not start - inference resumed",
			slot_name(upload_slot));
		release_guard();
		break;

	case SMP_UPLOAD_ACTIVE:
		lock_until_reset(false);
		break;

	default:
		/*
		 * In DRAIN_RETRY this is img_mgmt reporting the upload it dropped
		 * because of our own MGMT_ERR_EBUSY. Keep the guard blocked so the
		 * retry finds the readers drained.
		 */
		break;
	}
}

static void drain_retry_window_expired(struct k_work *work)
{
	ARG_UNUSED(work);

	k_mutex_lock(&smp_lock, K_FOREVER);

	if (upload_state == SMP_UPLOAD_DRAIN_RETRY) {
		LOG_WRN("%s model SMP upload not retried - inference resumed",
			slot_name(upload_slot));
		release_guard();
	}

	k_mutex_unlock(&smp_lock);
}

static enum mgmt_cb_return model_ota_smp_callback(uint32_t event, enum mgmt_cb_return prev_status,
						  int32_t *rc, uint16_t *group, bool *abort_more,
						  void *data, size_t data_size)
{
	enum mgmt_cb_return ret = MGMT_CB_OK;

	ARG_UNUSED(prev_status);
	ARG_UNUSED(group);
	ARG_UNUSED(abort_more);

	k_mutex_lock(&smp_lock, K_FOREVER);

	switch (event) {
	case MGMT_EVT_OP_IMG_MGMT_DFU_CHUNK:
		if (data != NULL && data_size == sizeof(struct img_mgmt_upload_check)) {
			ret = handle_dfu_chunk(data, rc);
		}
		break;

	case MGMT_EVT_OP_IMG_MGMT_DFU_STARTED:
		handle_dfu_started();
		break;

	case MGMT_EVT_OP_IMG_MGMT_DFU_PENDING:
		handle_dfu_pending();
		break;

	case MGMT_EVT_OP_IMG_MGMT_DFU_STOPPED:
		handle_dfu_stopped();
		break;

	default:
		break;
	}

	k_mutex_unlock(&smp_lock);

	return ret;
}

static struct mgmt_callback model_ota_smp_mgmt_cb = {
	.callback = model_ota_smp_callback,
	.event_id = (MGMT_EVT_OP_IMG_MGMT_DFU_CHUNK | MGMT_EVT_OP_IMG_MGMT_DFU_STARTED |
		     MGMT_EVT_OP_IMG_MGMT_DFU_PENDING | MGMT_EVT_OP_IMG_MGMT_DFU_STOPPED),
};

int model_ota_smp_register(const struct model_ota_smp_slot *slot)
{
	if (slot == NULL || slot->image_index == 0U) {
		return -EINVAL;
	}

	k_mutex_lock(&smp_lock, K_FOREVER);

	if (registered_slot_count >= CONFIG_MODEL_OTA_SMP_MAX_SLOTS) {
		k_mutex_unlock(&smp_lock);
		return -ENOMEM;
	}

	for (size_t i = 0; i < registered_slot_count; i++) {
		if (registered_slots[i].image_index == slot->image_index) {
			k_mutex_unlock(&smp_lock);
			return -EALREADY;
		}
	}

	registered_slots[registered_slot_count].image_index = slot->image_index;
	registered_slots[registered_slot_count].name = slot->name;
	registered_slot_count++;

	if (!smp_initialized) {
		mgmt_callback_register(&model_ota_smp_mgmt_cb);
		smp_initialized = true;
	}

	k_mutex_unlock(&smp_lock);

	return 0;
}

bool model_ota_smp_is_pending_reset(void)
{
	return upload_state == SMP_UPLOAD_LOCKED;
}

void model_ota_smp_set_upload_notify_cb(model_ota_smp_upload_notify_cb cb)
{
	upload_notify_cb = cb;
}
