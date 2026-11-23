/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/logging/log.h>
#include <zephyr/sys/__assert.h>

#include "model_image_common.h"

LOG_MODULE_REGISTER(model_ota, CONFIG_MODEL_OTA_LOG_LEVEL);

enum model_image_result model_image_validate_header(const uint8_t *image_addr,
						    size_t max_image_size,
						    const struct model_image_header **hdr_out)
{
	__ASSERT(image_addr != NULL, "No model image address");
	__ASSERT(hdr_out != NULL, "No header output");
	__ASSERT(max_image_size >= sizeof(struct model_image_header),
		 "Model partition cannot hold a model image header");

	const struct model_image_header *hdr = (const struct model_image_header *)image_addr;

	if (hdr->magic != MODEL_IMAGE_MAGIC) {
		LOG_ERR("No model image in partition");
		return MODEL_IMAGE_ERR_BAD_MAGIC;
	}

	if (hdr->format_version != MODEL_IMAGE_FORMAT_VERSION) {
		LOG_ERR("Unsupported image format version %u", hdr->format_version);
		return MODEL_IMAGE_ERR_BAD_FORMAT_VERSION;
	}

	if (hdr->image_size < sizeof(*hdr) || hdr->image_size > max_image_size) {
		LOG_ERR("Image size %u B is outside the valid range [%zu, %zu] B", hdr->image_size,
			sizeof(*hdr), max_image_size);
		return MODEL_IMAGE_ERR_BAD_SIZE;
	}

	*hdr_out = hdr;
	return MODEL_IMAGE_OK;
}
