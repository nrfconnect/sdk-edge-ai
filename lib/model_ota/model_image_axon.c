/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "model_image_common.h"

#include <model_ota/model_image.h>

#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(model_image, CONFIG_MODEL_OTA_LOG_LEVEL);

static bool binding_entry_matches(const struct model_image_binding_entry *entry,
				  const uint32_t *app_table)
{
	uint32_t app_count = app_table[0];

	for (uint32_t i = 0; i < app_count; i++) {
		if (app_table[1 + (2U * i)] != entry->name_hash) {
			continue;
		}
		if (app_table[1 + (2U * i) + 1U] !=
		    (uint32_t)(uintptr_t)entry->address) {
			LOG_ERR("Binding mismatch for hash 0x%08x (image 0x%08x, app 0x%08x)",
				entry->name_hash, (uint32_t)(uintptr_t)entry->address,
				app_table[1 + (2U * i) + 1U]);
			return false;
		}
		return true;
	}

	LOG_ERR("Binding hash 0x%08x not found in app table", entry->name_hash);
	return false;
}

static int model_image_verify_axon_binding(const struct model_image_header *hdr,
					     const uint32_t *app_binding_table)
{
	const struct model_image_binding_entry *entries;
	uint32_t count = hdr->axon.binding_count;

	if (count == 0U) {
		return MODEL_IMAGE_OK;
	}

	if (app_binding_table == NULL) {
		LOG_ERR("Image has %u binding entries but no app binding table", count);
		return MODEL_IMAGE_ERR_BINDING_MISMATCH;
	}

	if (hdr->axon.binding == NULL) {
		LOG_ERR("binding_count %u but binding pointer is NULL", count);
		return MODEL_IMAGE_ERR_BINDING_MISMATCH;
	}

	entries = hdr->axon.binding;

	for (uint32_t i = 0; i < count; i++) {
		if (!binding_entry_matches(&entries[i], app_binding_table)) {
			return MODEL_IMAGE_ERR_BINDING_MISMATCH;
		}
	}

	return MODEL_IMAGE_OK;
}

int model_image_load_axon(const uint8_t *partition_addr, size_t partition_size,
			  const struct model_image_axon_expect *expect,
			  const nrf_axon_nn_compiled_model_s **out_model)
{
	struct model_image_header hdr;
	const nrf_axon_nn_compiled_model_s *model;
	int rc;

	if (out_model == NULL || expect == NULL) {
		return MODEL_IMAGE_ERR_AXON_VALIDATE;
	}

	*out_model = NULL;

	rc = model_image_read_and_validate(partition_addr, partition_size, &hdr);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	if (hdr.params_type != MODEL_IMAGE_PARAMS_AXON) {
		LOG_ERR("Image is not an Axon model (params_type %u)", hdr.params_type);
		return MODEL_IMAGE_ERR_NOT_AXON_IMAGE;
	}

	if (hdr.contract_hash != expect->contract_hash) {
		LOG_ERR("Contract hash mismatch (image 0x%08x, expected 0x%08x)", hdr.contract_hash,
			expect->contract_hash);
		return MODEL_IMAGE_ERR_CONTRACT_MISMATCH;
	}

	if (hdr.axon.persistent_vars_required > expect->persistent_vars_cap) {
		LOG_ERR("Model needs %u persistent vars, app cap is %u",
			hdr.axon.persistent_vars_required, expect->persistent_vars_cap);
		return MODEL_IMAGE_ERR_PERSISTENT_VARS_TOO_MANY;
	}

	if (hdr.axon.axon_packed_output_bytes > expect->packed_output_cap) {
		LOG_ERR("Model needs %u packed-output bytes, app cap is %u",
			hdr.axon.axon_packed_output_bytes, expect->packed_output_cap);
		return MODEL_IMAGE_ERR_PACKED_OUTPUT_TOO_LARGE;
	}

	rc = model_image_verify_axon_binding(&hdr, expect->binding_table);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	/* The baked pointers here are absolute flash addresses linked at the partition base, and
	 * that base is part of contract_hash, so a wrong-slot image was rejected above.
	 * Containment is a build-time property (validate_model_image_layout.py).
	 */
	model = hdr.axon.model;

	if (nrf_axon_nn_model_validate(model) != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Axon model validate failed for image '%s'", hdr.name);
		return MODEL_IMAGE_ERR_AXON_VALIDATE;
	}

	*out_model = model;

	LOG_INF("Loaded Axon model image '%s' v0x%08x", hdr.name, hdr.model_version);

	return MODEL_IMAGE_OK;
}
