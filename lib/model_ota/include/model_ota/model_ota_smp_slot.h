/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_OTA_MODEL_OTA_SMP_SLOT_H_
#define MODEL_OTA_MODEL_OTA_SMP_SLOT_H_

/**
 * @file
 * @brief Linker iterable SMP slot entries for OTA-wired models (library internal).
 *
 * Wired translation units register one slot per model partition. Applications use
 * :file:`include/model_ota/model_ota_smp.h` for upload policy only.
 */

#include <stdint.h>

#include <zephyr/sys/iterable_sections.h>

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

#endif /* MODEL_OTA_MODEL_OTA_SMP_SLOT_H_ */
