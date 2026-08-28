/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "model_image_common.h"

#include <stddef.h>
#include <string.h>

#include <zephyr/logging/log.h>
#include <zephyr/sys/crc.h>
#include <zephyr/sys/util.h>

LOG_MODULE_REGISTER(model_image, CONFIG_MODEL_OTA_LOG_LEVEL);

BUILD_ASSERT(sizeof(struct model_image_header) == 84,
	     "model_image_header size must match host layout tools");
BUILD_ASSERT(offsetof(struct model_image_header, edgeai_params) == 48,
	     "edgeai_params offset must match host layout tools");
BUILD_ASSERT(offsetof(struct model_image_header, edgeai_params) % sizeof(uint32_t) == 0,
	     "edgeai_params must be word-aligned");
BUILD_ASSERT(offsetof(struct model_image_header, contract_hash) == 16,
	     "contract_hash offset must match host layout tools");
BUILD_ASSERT(offsetof(struct model_image_header, crc32) == MODEL_IMAGE_CRC32_OFFSET,
	     "crc32 offset must match patch_image_crc.py");
BUILD_ASSERT(offsetof(struct model_image_header, name) % sizeof(uint32_t) == 0,
	     "name must be word-aligned");
BUILD_ASSERT(offsetof(struct model_image_header, neuton.model) % sizeof(uint32_t) == 0,
	     "neuton.model must be word-aligned");
BUILD_ASSERT(offsetof(struct model_image_header, axon.model) % sizeof(uint32_t) == 0,
	     "axon.model must be word-aligned");
BUILD_ASSERT(sizeof(struct model_image_neuton_backend) == 4,
	     "neuton backend size must match host layout tools");
BUILD_ASSERT(sizeof(struct model_image_axon_backend) == 20,
	     "axon backend size must match host layout tools");
BUILD_ASSERT(sizeof(struct model_image_edgeai_params) == 36,
	     "edgeai params block size must match host layout tools");
BUILD_ASSERT(offsetof(struct model_image_edgeai_params, p_extraction_mask) % sizeof(uint32_t) == 0,
	     "p_extraction_mask must be word-aligned");

static bool magic_is_valid(const struct model_image_header *hdr)
{
	return hdr->magic[0] == MODEL_IMAGE_MAGIC0 && hdr->magic[1] == MODEL_IMAGE_MAGIC1 &&
	       hdr->magic[2] == MODEL_IMAGE_MAGIC2 && hdr->magic[3] == MODEL_IMAGE_MAGIC3;
}

int model_image_read_and_validate(const uint8_t *partition_addr, size_t partition_size,
				  struct model_image_header *hdr_out)
{
	struct model_image_header hdr;

	if (partition_addr == NULL) {
		LOG_ERR("partition_addr is NULL");
		return MODEL_IMAGE_ERR_NO_PARTITION;
	}

	memcpy(&hdr, partition_addr, sizeof(hdr));

	if (!magic_is_valid(&hdr)) {
		LOG_WRN("No valid model image in partition (bad magic)");
		return MODEL_IMAGE_ERR_BAD_MAGIC;
	}

	if (hdr.format_version != MODEL_IMAGE_FORMAT_VERSION) {
		LOG_ERR("Unsupported image format version %u", hdr.format_version);
		return MODEL_IMAGE_ERR_BAD_FORMAT_VERSION;
	}

	if (hdr.image_size < sizeof(hdr) || hdr.image_size > partition_size) {
		LOG_ERR("Image size %u B does not fit partition (%zu B)", hdr.image_size,
			partition_size);
		return MODEL_IMAGE_ERR_TOO_LARGE;
	}

	/* CRC32/IEEE over the whole memory-mapped image with the header's crc32 field treated as
	 * 0 - matches how tools/model_ota/patch_image_crc.py computed the stored value. The header
	 * (RAM copy, crc zeroed) and the image tail (read from flash/XIP) are chained because they
	 * are not contiguous once the crc field has been blanked.
	 */
	uint32_t stored_crc = hdr.crc32;
	struct model_image_header hdr_for_crc = hdr;

	hdr_for_crc.crc32 = 0;
	uint32_t computed_crc =
		crc32_ieee_update(0, (const uint8_t *)&hdr_for_crc, sizeof(hdr_for_crc));

	computed_crc = crc32_ieee_update(computed_crc, partition_addr + sizeof(hdr),
					 hdr.image_size - sizeof(hdr));

	if (computed_crc != stored_crc) {
		LOG_ERR("Image CRC mismatch (stored 0x%08x, computed 0x%08x)", stored_crc,
			computed_crc);
		return MODEL_IMAGE_ERR_BAD_CRC;
	}

	*hdr_out = hdr;
	return MODEL_IMAGE_OK;
}

bool model_image_name_in_image(const char *name, const uint8_t *base, const uint8_t *end)
{
	const uint8_t *s;

	if (name == NULL) {
		return false;
	}

	s = (const uint8_t *)name;
	if (s < base || s >= end) {
		return false;
	}

	while (s < end) {
		if (*s == '\0') {
			return true;
		}
		s++;
	}

	return false;
}

bool model_image_span_in_image(const void *p, size_t nbytes, const uint8_t *base,
			       const uint8_t *end)
{
	const uint8_t *s = (const uint8_t *)p;

	return s != NULL && s >= base && (s + nbytes) >= s && (s + nbytes) <= end;
}

int model_image_neuton_params_elem_size(uint8_t params_type, size_t *elem_size_out)
{
	switch (params_type) {
	case MODEL_IMAGE_PARAMS_F32:
		if (elem_size_out != NULL) {
			*elem_size_out = 4;
		}
		return MODEL_IMAGE_OK;
	case MODEL_IMAGE_PARAMS_Q16:
		if (elem_size_out != NULL) {
			*elem_size_out = 2;
		}
		return MODEL_IMAGE_OK;
	case MODEL_IMAGE_PARAMS_Q8:
		if (elem_size_out != NULL) {
			*elem_size_out = 1;
		}
		return MODEL_IMAGE_OK;
	default:
		return MODEL_IMAGE_ERR_BAD_PARAMS_TYPE;
	}
}
