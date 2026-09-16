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

#include <zephyr/sys/iterable_sections.h>

/**
 * @file
 * @brief Coordinate MCUboot SMP model uploads with the built-in inference guard.
 *
 * Inference blocking for OTA-wired Edge AI models is enforced inside the Edge
 * AI runtime. Direct Axon driver use and other model flash access require
 * explicit ``model_ota_guard_acquire()`` / ``model_ota_guard_release()`` pairs
 * in application or library code. The functions in this header expose upload
 * state for application policy (LEDs, reboot prompts) only — not for protection.
 *
 * OTA-wired models declare their devicetree-derived image indices in a linker-built
 * iterable section. Applications only need to set an optional upload notify callback.
 *
 * Requires ``CONFIG_MODEL_OTA_SMP`` (MCUboot, MCUMGR image management, and upload hooks).
 */

/** One MCUboot updateable image that maps to a model partition. */
struct model_ota_smp_slot {
	/** MCUboot updateable image index (from bootchain devicetree). */
	uint8_t image_index;
	/** Short label for log messages; may be NULL. */
	const char *name;
};

/** Declare one model image in the linker-built SMP protection table. */
#define MODEL_OTA_SMP_SLOT_DEFINE(var_name, image_idx, slot_name)                                  \
	static const STRUCT_SECTION_ITERABLE(model_ota_smp_slot, var_name) = {                     \
		.image_index = (image_idx),                                                         \
		.name = (slot_name),                                                                \
	}

/** Informational: true when a model upload finished and reset is required. */
bool model_ota_smp_is_pending_reset(void);

typedef void (*model_ota_smp_upload_notify_cb)(bool active);

void model_ota_smp_set_upload_notify_cb(model_ota_smp_upload_notify_cb cb);

#endif /* MODEL_OTA_SMP_H_ */
