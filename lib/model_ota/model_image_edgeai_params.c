/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Applies the model parameters that live in nrf_edgeai_t (feature scaling factors, decoded-output
 * init) from an already validated model partition image, after verifying the one part of the DSP
 * contract the hash cannot express: the feature-extraction mask.
 */

#include <model_ota/model_image.h>

#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(model_image, CONFIG_MODEL_OTA_LOG_LEVEL);

/*
 * The masks decide which features the pipeline extracts, in which order, for each unique input
 * feature - and therefore what each slot of the image's flat feature arrays means. The extraction
 * code itself stays compiled into the application, so an image whose masks differ would have its
 * arrays indexed with the wrong per-slot meaning: wrong numbers, no fault.
 */
static enum model_image_result extraction_mask_matches(const struct model_image_edgeai_params *params,
				   const nrf_edgeai_t *edgeai)
{
	const nrf_edgeai_features_mask_t *app = edgeai->p_dsp->features.p_masks;
	const uint16_t num = edgeai->p_dsp->features.masks_num;

	if (params->p_extraction_mask == NULL || app == NULL) {
		LOG_ERR("Extraction mask missing (image %p, app %p)",
			(const void *)params->p_extraction_mask, (const void *)app);
		return MODEL_IMAGE_ERR_DSP_MASK_MISMATCH;
	}

	for (uint16_t i = 0; i < num; i++) {
		if (params->p_extraction_mask[i].all != app[i].all) {
			LOG_ERR("Extraction mask for input feature %u differs "
				"(image 0x%08x%08x, app 0x%08x%08x)",
				i, (uint32_t)(params->p_extraction_mask[i].all >> 32),
				(uint32_t)params->p_extraction_mask[i].all,
				(uint32_t)(app[i].all >> 32), (uint32_t)app[i].all);
			return MODEL_IMAGE_ERR_DSP_MASK_MISMATCH;
		}
	}

	return MODEL_IMAGE_OK;
}

enum model_image_result model_image_bind_edgeai_params(const uint8_t *partition_addr,
							       nrf_edgeai_t *edgeai)
{
	const struct model_image_header *hdr = (const struct model_image_header *)partition_addr;
	/* By value: hdr is __packed, so &hdr->edgeai_params would be a possibly-unaligned
	 * pointer.
	 */
	const struct model_image_edgeai_params params = hdr->edgeai_params;

	if (params.scale_num == 0) {
		/* Solution scales nothing; the app keeps its compiled-in values. The contract hash
		 * covers scale_num, so this cannot be an image that merely omitted them.
		 */
		return MODEL_IMAGE_OK;
	}

	if (edgeai->p_dsp != NULL) {
		enum model_image_result rc = extraction_mask_matches(&params, edgeai);

		if (rc != MODEL_IMAGE_OK) {
			return rc;
		}

		edgeai->p_dsp->features.meta = params.scale.features;
	} else {
		edgeai->input.scale = params.scale.input;
	}

	edgeai->decoded_output = params.decoded_output;

	return MODEL_IMAGE_OK;
}
