/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_OTA_SMP_H_
#define MODEL_OTA_SMP_H_

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * @file
 * @brief Pause partition-resident model inference during MCUboot SMP uploads.
 *
 * Register MCUboot updateable image indices that carry model partition payloads.
 * While an upload is in progress, or after it completes and a reset is still
 * required, callers must not execute inference from the affected flash region.
 *
 * Requires ``CONFIG_MODEL_OTA_SMP`` (MCUboot, MCUMGR image management, and upload hooks).
 */

/** One MCUboot updateable image that maps to a model partition. */
struct model_ota_smp_slot {
	/** MCUboot updateable image index (image 0 is normally firmware). */
	uint8_t image_index;
	/** Short label for log messages; may be NULL. */
	const char *name;
};

/**
 * @brief Register model image slots and install the MCUMGR upload callback.
 *
 * @param slots Slot descriptors; must remain valid for the lifetime of the application.
 * @param slot_count Number of entries in @p slots.
 *
 * @retval 0 on success.
 * @retval -EINVAL if @p slots is NULL, @p slot_count is zero, or a slot is invalid.
 * @retval -EALREADY if already initialized.
 */
int model_ota_smp_init(const struct model_ota_smp_slot *slots, size_t slot_count);

/**
 * @brief True while the given model image is uploading or any upload awaits reset.
 *
 * When any registered model upload has finished, this returns true for every
 * registered image index until the device resets.
 */
bool model_ota_smp_blocks_inference(uint8_t image_index);

/** True after any registered model SMP upload finished and a reset is required. */
bool model_ota_smp_is_pending_reset(void);

/**
 * @brief Optional notification when a registered model upload starts or finishes.
 *
 * @param active True when upload of a registered model image starts, false when it ends.
 */
typedef void (*model_ota_smp_upload_notify_cb)(bool active);

/** Register @p cb, or NULL to clear. May be called before or after model_ota_smp_init(). */
void model_ota_smp_set_upload_notify_cb(model_ota_smp_upload_notify_cb cb);

#endif /* MODEL_OTA_SMP_H_ */
