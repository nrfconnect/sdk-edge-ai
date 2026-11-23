/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "model_image_axon.h"
#include "model_image_common.h"

#include <zephyr/logging/log.h>
#include <zephyr/sys/__assert.h>

LOG_MODULE_DECLARE(model_ota, CONFIG_MODEL_OTA_LOG_LEVEL);

/** Firmware symbols an Axon model image may bind to, shared by all model slots. */
extern const struct model_image_binding_entry model_ota_axon_binding_table[];

/** Number of entries in @ref model_ota_axon_binding_table. */
extern const uint32_t model_ota_axon_binding_count;

static bool binding_entry_matches(const struct model_image_binding_entry *entry)
{
	for (uint32_t i = 0; i < model_ota_axon_binding_count; i++) {
		const struct model_image_binding_entry *fw = &model_ota_axon_binding_table[i];

		if (fw->name_hash != entry->name_hash) {
			continue;
		}
		if (fw->address != entry->address) {
			LOG_ERR("Binding 0x%08x mismatch (image %p, firmware %p)", entry->name_hash,
				entry->address, fw->address);
			return false;
		}
		return true;
	}

	LOG_ERR("Binding 0x%08x is not provided by the firmware", entry->name_hash);
	return false;
}

static enum model_image_result verify_binding(const struct model_image_header *hdr)
{
	uint32_t count = hdr->axon.binding_count;

	if (count == 0U) {
		return MODEL_IMAGE_OK;
	}

	if (hdr->axon.binding == NULL) {
		LOG_ERR("Image has no binding table");
		return MODEL_IMAGE_ERR_BINDING_MISMATCH;
	}

	for (uint32_t i = 0; i < count; i++) {
		if (!binding_entry_matches(&hdr->axon.binding[i])) {
			return MODEL_IMAGE_ERR_BINDING_MISMATCH;
		}
	}

	return MODEL_IMAGE_OK;
}

enum model_image_result model_image_load_axon(const uint8_t *image_addr, size_t max_image_size,
					      const struct model_image_axon_expect *expect,
					      const nrf_axon_nn_compiled_model_s **out_model)
{
	__ASSERT(out_model != NULL, "No model output");
	__ASSERT(expect != NULL, "No model slot expectations");

	const struct model_image_header *hdr;
	const nrf_axon_nn_compiled_model_s *model;
	enum model_image_result rc;

	rc = model_image_validate_header(image_addr, max_image_size, &hdr);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	if (hdr->backend != MODEL_IMAGE_BACKEND_AXON) {
		LOG_ERR("Image is not an Axon model");
		return MODEL_IMAGE_ERR_BAD_BACKEND;
	}

	if (hdr->contract_hash != expect->contract_hash) {
		LOG_ERR("Contract hash mismatch (image 0x%08x, firmware 0x%08x)",
			hdr->contract_hash, expect->contract_hash);
		return MODEL_IMAGE_ERR_CONTRACT_MISMATCH;
	}

	if (hdr->axon.persistent_vars_required > expect->persistent_vars_cap) {
		LOG_ERR("Model needs %u persistent vars, firmware provides %u",
			hdr->axon.persistent_vars_required, expect->persistent_vars_cap);
		return MODEL_IMAGE_ERR_PERSISTENT_VARS_TOO_MANY;
	}

	if (hdr->axon.packed_output_bytes > expect->packed_output_cap) {
		LOG_ERR("Model needs %u packed-output bytes, firmware provides %u",
			hdr->axon.packed_output_bytes, expect->packed_output_cap);
		return MODEL_IMAGE_ERR_PACKED_OUTPUT_TOO_LARGE;
	}

	rc = verify_binding(hdr);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	model = (const nrf_axon_nn_compiled_model_s *)hdr->axon.model;
	if (model == NULL) {
		LOG_ERR("Image has no Axon model");
		return MODEL_IMAGE_ERR_AXON_VALIDATE;
	}

	if (nrf_axon_nn_model_validate(model) != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Axon model validation failed for image '%s'", hdr->name);
		return MODEL_IMAGE_ERR_AXON_VALIDATE;
	}

	if (model->input_cnt < 0 || (uint32_t)model->input_cnt > expect->input_vector_cap) {
		LOG_ERR("Model needs %d input vectors, firmware provides %u", model->input_cnt,
			expect->input_vector_cap);
		return MODEL_IMAGE_ERR_AXON_VALIDATE;
	}

	*out_model = model;

	LOG_INF("Loaded Axon model image '%s' version 0x%08x", hdr->name, hdr->model_version);

	return MODEL_IMAGE_OK;
}
