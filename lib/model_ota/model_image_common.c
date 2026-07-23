/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "model_image_common.h"

#include <zephyr/logging/log.h>
#include <zephyr/storage/flash_map.h>
#include <zephyr/sys/crc.h>

LOG_MODULE_REGISTER(model_image, CONFIG_MODEL_OTA_LOG_LEVEL);

static bool magic_is_valid(const struct model_image_header *hdr)
{
	return hdr->magic[0] == MODEL_IMAGE_MAGIC0 && hdr->magic[1] == MODEL_IMAGE_MAGIC1 &&
	       hdr->magic[2] == MODEL_IMAGE_MAGIC2 && hdr->magic[3] == MODEL_IMAGE_MAGIC3;
}

int model_image_read_and_validate(uint8_t fa_id, const uint8_t *partition_addr,
				  struct model_image_header *hdr_out)
{
	const struct flash_area *fa;
	struct model_image_header hdr;
	int rc;

	rc = flash_area_open(fa_id, &fa);
	if (rc != 0) {
		LOG_ERR("Cannot open model partition (err %d)", rc);
		return MODEL_IMAGE_ERR_NO_PARTITION;
	}

	rc = flash_area_read(fa, 0, &hdr, sizeof(hdr));
	if (rc != 0) {
		flash_area_close(fa);
		LOG_ERR("Flash read of image header failed (err %d)", rc);
		return MODEL_IMAGE_ERR_FLASH_READ;
	}

	if (!magic_is_valid(&hdr)) {
		flash_area_close(fa);
		LOG_WRN("No valid model image in partition (bad magic)");
		return MODEL_IMAGE_ERR_BAD_MAGIC;
	}

	if (hdr.format_version != MODEL_IMAGE_FORMAT_VERSION) {
		flash_area_close(fa);
		LOG_ERR("Unsupported image format version %u", hdr.format_version);
		return MODEL_IMAGE_ERR_BAD_FORMAT_VERSION;
	}

	if (hdr.image_size < sizeof(hdr) || hdr.image_size > fa->fa_size) {
		flash_area_close(fa);
		LOG_ERR("Image size %u B does not fit partition (%u B)", hdr.image_size,
			(unsigned)fa->fa_size);
		return MODEL_IMAGE_ERR_TOO_LARGE;
	}

	flash_area_close(fa);

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
