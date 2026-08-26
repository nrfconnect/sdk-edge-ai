/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Applies the model parameters that live in nrf_edgeai_t (feature scaling factors, decoded-output
 * init) from an already validated model partition image.
 */

#include <model_ota/model_image.h>

#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(model_image, CONFIG_MODEL_OTA_LOG_LEVEL);

int model_image_bind_edgeai_params(const uint8_t *partition_addr, nrf_edgeai_t *edgeai,
				   const struct model_image_scale_expect *expect)
{
	const struct model_image_header *hdr = (const struct model_image_header *)partition_addr;
	/* By value: hdr is __packed, so &hdr->edgeai_params would be a possibly-unaligned
	 * pointer.
	 */
	const struct model_image_edgeai_params params = hdr->edgeai_params;

	if (params.scale_num == 0) {
		/* Image carries no parameters; the app keeps its compiled-in ones. */
		return MODEL_IMAGE_OK;
	}

	/* A model-only update keeps the same solution, so the element count and size of the
	 * scaling arrays are fixed by the application.
	 */
	if (params.scale_num != expect->num || params.scale_elem_size != expect->elem_size) {
		LOG_ERR("Image scale %ux%uB != app %ux%uB", params.scale_num,
			params.scale_elem_size, expect->num, expect->elem_size);
		return MODEL_IMAGE_ERR_SCALE_MISMATCH;
	}

	if (edgeai->p_dsp != NULL) {
		edgeai->p_dsp->features.meta = params.scale.features;
	} else {
		edgeai->input.scale = params.scale.input;
	}

	edgeai->decoded_output = params.decoded_output;

	return MODEL_IMAGE_OK;
}
