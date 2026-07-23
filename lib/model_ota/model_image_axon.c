/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "model_image_common.h"

#include <model_ota/model_image.h>

#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(model_image, CONFIG_MODEL_OTA_LOG_LEVEL);

int model_image_load_axon(uint8_t fa_id, const uint8_t *partition_addr,
			  const nrf_axon_nn_compiled_model_s **out_model)
{
	struct model_image_header hdr;
	const uint8_t *image_end;
	const uint8_t *model_bytes;
	const nrf_axon_nn_compiled_model_s *model;
	int rc;

	if (out_model == NULL) {
		return MODEL_IMAGE_ERR_AXON_VALIDATE;
	}

	*out_model = NULL;

	rc = model_image_read_and_validate(fa_id, partition_addr, &hdr);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	if (hdr.params_type != MODEL_IMAGE_PARAMS_AXON) {
		LOG_ERR("Image is not an Axon model (params_type %u)", hdr.params_type);
		return MODEL_IMAGE_ERR_NOT_AXON_IMAGE;
	}

	image_end = partition_addr + hdr.image_size;
	model_bytes = (const uint8_t *)hdr.model.axon;

	if (model_bytes < partition_addr ||
	    model_bytes + sizeof(nrf_axon_nn_compiled_model_s) > image_end) {
		LOG_ERR("Header model pointer %p outside image [%p, %p)", (void *)hdr.model.axon,
			(const void *)partition_addr, (const void *)image_end);
		return MODEL_IMAGE_ERR_MODEL_PTR_OUT_OF_RANGE;
	}

	if (hdr.decoded_output != NULL) {
		LOG_ERR("Axon image must have NULL decoded_output, got %p",
			(void *)hdr.decoded_output);
		return MODEL_IMAGE_ERR_PTR_OUT_OF_RANGE;
	}

	model = hdr.model.axon;

	if (nrf_axon_nn_model_validate(model) != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Axon model validate failed for image '%.*s'", MODEL_IMAGE_NAME_LEN,
			hdr.name);
		return MODEL_IMAGE_ERR_AXON_VALIDATE;
	}

	*out_model = model;

	LOG_INF("Loaded Axon model image '%.*s' v0x%08x", MODEL_IMAGE_NAME_LEN, hdr.name,
		hdr.model_version);

	return MODEL_IMAGE_OK;
}
