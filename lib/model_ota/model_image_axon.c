/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <model_ota/model_image.h>

#include <zephyr/logging/log.h>
#include <zephyr/storage/flash_map.h>
#include <zephyr/sys/crc.h>

LOG_MODULE_REGISTER(model_image_axon, CONFIG_MODEL_OTA_LOG_LEVEL);

static bool magic_is_valid(const struct model_image_header *hdr)
{
	return hdr->magic[0] == MODEL_IMAGE_MAGIC0 && hdr->magic[1] == MODEL_IMAGE_MAGIC1 &&
	       hdr->magic[2] == MODEL_IMAGE_MAGIC2 && hdr->magic[3] == MODEL_IMAGE_MAGIC3;
}

int model_image_load_axon(uint8_t fa_id, const uint8_t *partition_addr,
			  const nrf_axon_nn_compiled_model_s **out_model)
{
	const struct flash_area *fa;
	struct model_image_header hdr;
	int rc;
	const nrf_axon_nn_compiled_model_s *model;

	if (out_model == NULL) {
		return MODEL_IMAGE_ERR_AXON_VALIDATE;
	}

	*out_model = NULL;

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

	if (hdr.params_type != MODEL_IMAGE_PARAMS_AXON) {
		flash_area_close(fa);
		LOG_ERR("Image is not an Axon model (params_type %u)", hdr.params_type);
		return MODEL_IMAGE_ERR_NOT_AXON_IMAGE;
	}

	if (hdr.image_size < sizeof(hdr) || hdr.image_size > fa->fa_size) {
		flash_area_close(fa);
		LOG_ERR("Image size %u B does not fit partition (%u B)", hdr.image_size,
			(unsigned)fa->fa_size);
		return MODEL_IMAGE_ERR_TOO_LARGE;
	}

	flash_area_close(fa);

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

	const uint8_t *image_end = partition_addr + hdr.image_size;
	const uint8_t *model_bytes = (const uint8_t *)hdr.model.axon;

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
