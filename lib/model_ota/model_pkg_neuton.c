/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <model_ota/model_pkg.h>

#include <string.h>

#include <zephyr/logging/log.h>
#include <zephyr/storage/flash_map.h>
#include <zephyr/sys/crc.h>
#include <zephyr/toolchain.h>

LOG_MODULE_REGISTER(model_pkg, CONFIG_MODEL_OTA_LOG_LEVEL);

static bool magic_is_valid(const struct model_pkg_header *hdr)
{
	return hdr->magic[0] == MODEL_PKG_MAGIC0 && hdr->magic[1] == MODEL_PKG_MAGIC1 &&
	       hdr->magic[2] == MODEL_PKG_MAGIC2 && hdr->magic[3] == MODEL_PKG_MAGIC3;
}

/* Byte size of one MODEL_PKG_NEUTON_SEC_WEIGHTS/ACT_WEIGHTS element for a given params_type;
 * 0 for an out-of-range value, checked by the caller.
 */
static size_t params_elem_size(uint8_t params_type)
{
	switch (params_type) {
	case MODEL_PKG_NEUTON_PARAMS_F32:
		return sizeof(float);
	case MODEL_PKG_NEUTON_PARAMS_Q16:
		return sizeof(uint16_t);
	case MODEL_PKG_NEUTON_PARAMS_Q8:
		return sizeof(uint8_t);
	default:
		return 0;
	}
}

int model_pkg_load_neuton(uint8_t fa_id, const uint8_t *partition_addr,
			   nrf_edgeai_model_neuton_t *out_model, void *neurons_buf,
			   size_t neurons_buf_cap, struct model_pkg_neuton_info *out_info)
{
	const struct flash_area *fa;
	struct model_pkg_header hdr;
	int rc;

	rc = flash_area_open(fa_id, &fa);
	if (rc != 0) {
		LOG_ERR("Cannot open model_storage partition (err %d)", rc);
		return MODEL_PKG_ERR_NO_PARTITION;
	}

	rc = flash_area_read(fa, 0, &hdr, sizeof(hdr));
	if (rc != 0) {
		flash_area_close(fa);
		LOG_ERR("Flash read of package header failed (err %d)", rc);
		return MODEL_PKG_ERR_FLASH_READ;
	}

	if (!magic_is_valid(&hdr)) {
		flash_area_close(fa);
		LOG_WRN("No valid model package in model_storage (bad magic)");
		return MODEL_PKG_ERR_BAD_MAGIC;
	}

	if (hdr.format_version != MODEL_PKG_FORMAT_VERSION) {
		flash_area_close(fa);
		LOG_ERR("Unsupported package format version %u", hdr.format_version);
		return MODEL_PKG_ERR_BAD_FORMAT_VERSION;
	}

	if (hdr.model_type != MODEL_PKG_TYPE_NEUTON) {
		flash_area_close(fa);
		LOG_ERR("Package is not a Neuton model (type %u)", hdr.model_type);
		return MODEL_PKG_ERR_WRONG_TYPE;
	}

	size_t weight_elem_size = params_elem_size(hdr.params_type);

	if (weight_elem_size == 0) {
		flash_area_close(fa);
		LOG_ERR("Unsupported Neuton params_type %u", hdr.params_type);
		return MODEL_PKG_ERR_BAD_PARAMS_TYPE;
	}

	if (hdr.payload_size > fa->fa_size - sizeof(hdr)) {
		flash_area_close(fa);
		LOG_ERR("Package payload (%u B) exceeds model_storage capacity (%u B)",
			hdr.payload_size, (unsigned)(fa->fa_size - sizeof(hdr)));
		return MODEL_PKG_ERR_TOO_LARGE;
	}

	/* Each section starts at the next 4-byte-aligned offset from the payload's own start
	 * (see the comment on enum model_pkg_neuton_section) - replay that same layout here to
	 * get the true expected payload size, padding included, rather than a raw sum of
	 * section_len[].
	 */
	uint32_t section_total = 0;

	for (size_t i = 0; i < MODEL_PKG_NEUTON_SECTION_COUNT; i++) {
		section_total += (4 - (section_total % 4)) % 4;
		section_total += hdr.section_len[i];
	}
	if (section_total != hdr.payload_size) {
		flash_area_close(fa);
		LOG_ERR("Section lengths plus alignment padding (%u) do not add up to "
			"payload_size (%u)", section_total, hdr.payload_size);
		return MODEL_PKG_ERR_BAD_SECTION_LEN;
	}

	uint32_t stored_crc = hdr.crc32;

	flash_area_close(fa);

	/* Validate CRC32 straight over the memory-mapped partition, with the header's crc32
	 * field treated as 0. Chained via crc32_ieee_update() since the header (RAM copy with
	 * crc32 zeroed) and payload (read directly from flash) are not contiguous in memory.
	 */
	struct model_pkg_header hdr_for_crc = hdr;

	hdr_for_crc.crc32 = 0;
	const uint8_t *payload = partition_addr + sizeof(hdr);
	uint32_t computed_crc =
		crc32_ieee_update(0, (const uint8_t *)&hdr_for_crc, sizeof(hdr_for_crc));
	computed_crc = crc32_ieee_update(computed_crc, payload, hdr.payload_size);

	if (computed_crc != stored_crc) {
		LOG_ERR("Package CRC mismatch (stored 0x%08x, computed 0x%08x)", stored_crc,
			computed_crc);
		return MODEL_PKG_ERR_BAD_CRC;
	}

	/* Element counts are derived from section byte lengths rather than stored separately. */
	uint32_t weights_num = hdr.section_len[MODEL_PKG_NEUTON_SEC_WEIGHTS] / weight_elem_size;
	uint32_t neurons_num =
		hdr.section_len[MODEL_PKG_NEUTON_SEC_INTERNAL_LINKS_NUM] / sizeof(uint16_t);
	uint32_t outputs_num =
		hdr.section_len[MODEL_PKG_NEUTON_SEC_OUTPUT_INDICES] / sizeof(uint16_t);

	if (neurons_num > neurons_buf_cap) {
		LOG_ERR("Model needs %u neurons, only %u provided", neurons_num,
			(unsigned)neurons_buf_cap);
		return MODEL_PKG_ERR_NEURONS_BUF_TOO_SMALL;
	}

	const void *section_ptr[MODEL_PKG_NEUTON_SECTION_COUNT];
	uint32_t offset = 0;

	for (size_t i = 0; i < MODEL_PKG_NEUTON_SECTION_COUNT; i++) {
		offset += (4 - (offset % 4)) % 4;
		section_ptr[i] = payload + offset;
		offset += hdr.section_len[i];
	}

	/* Which union member to populate - and neurons_buf's actual element type - follows the
	 * package's own params_type; see model_pkg_load_neuton()'s doc comment. Built as a plain
	 * (non-const) local first, since nrf_edgeai_model_neuton_params_t's members would
	 * otherwise inherit the const-qualification described below.
	 */
	nrf_edgeai_model_neuton_params_t params;

	switch (hdr.params_type) {
	case MODEL_PKG_NEUTON_PARAMS_F32:
		params.f32 = (nrf_edgeai_model_neuton_params_f32_t){
			.p_weights = section_ptr[MODEL_PKG_NEUTON_SEC_WEIGHTS],
			.p_act_weights = section_ptr[MODEL_PKG_NEUTON_SEC_ACT_WEIGHTS],
			.p_neurons = neurons_buf,
		};
		break;
	case MODEL_PKG_NEUTON_PARAMS_Q16:
		params.q16 = (nrf_edgeai_model_neuton_params_q16_t){
			.p_weights = section_ptr[MODEL_PKG_NEUTON_SEC_WEIGHTS],
			.p_act_weights = section_ptr[MODEL_PKG_NEUTON_SEC_ACT_WEIGHTS],
			.p_neurons = neurons_buf,
		};
		break;
	case MODEL_PKG_NEUTON_PARAMS_Q8:
	default: /* hdr.params_type was already validated above */
		params.q8 = (nrf_edgeai_model_neuton_params_q8_t){
			.p_weights = section_ptr[MODEL_PKG_NEUTON_SEC_WEIGHTS],
			.p_act_weights = section_ptr[MODEL_PKG_NEUTON_SEC_ACT_WEIGHTS],
			.p_neurons = neurons_buf,
		};
		break;
	}

	/*
	 * nrf_edgeai_model_neuton_t declares its meta/params members `const`, so an existing
	 * instance of it cannot be assigned to (field-by-field or as a whole) after creation.
	 * Build the final value as a fresh local temporary instead, then copy its bytes into
	 * *out_model via memcpy(), which operates on the object representation and is therefore
	 * unaffected by the const-qualified members.
	 */
	nrf_edgeai_model_neuton_t built = {
		.meta = {
			.p_neuron_internal_links_num =
				section_ptr[MODEL_PKG_NEUTON_SEC_INTERNAL_LINKS_NUM],
			.p_neuron_external_links_num =
				section_ptr[MODEL_PKG_NEUTON_SEC_EXTERNAL_LINKS_NUM],
			.p_output_neurons_indices =
				section_ptr[MODEL_PKG_NEUTON_SEC_OUTPUT_INDICES],
			.p_neuron_links = section_ptr[MODEL_PKG_NEUTON_SEC_NEURON_LINKS],
			.p_neuron_act_type_mask =
				section_ptr[MODEL_PKG_NEUTON_SEC_ACT_TYPE_MASK],
			.outputs_num = (uint16_t)outputs_num,
			.neurons_num = (uint16_t)neurons_num,
			.weights_num = weights_num,
		},
		.params = params,
	};

	memcpy(out_model, &built, sizeof(built));

	if (out_info != NULL) {
		memset(out_info->name, 0, sizeof(out_info->name));
		memcpy(out_info->name, hdr.name, MODEL_PKG_NAME_LEN);
		out_info->version = hdr.model_version;
		out_info->neurons_num = (uint16_t)neurons_num;
		out_info->outputs_num = (uint16_t)outputs_num;
		out_info->weights_num = weights_num;
		out_info->output_scale_min = section_ptr[MODEL_PKG_NEUTON_SEC_OUTPUT_SCALE_MIN];
		out_info->output_scale_max = section_ptr[MODEL_PKG_NEUTON_SEC_OUTPUT_SCALE_MAX];
		out_info->average_embedding =
			section_ptr[MODEL_PKG_NEUTON_SEC_AVERAGE_EMBEDDING];
	}

	LOG_INF("Loaded Neuton model '%.*s' v0x%08x (%u neurons, %u weights, %u outputs)",
		MODEL_PKG_NAME_LEN, hdr.name, hdr.model_version, neurons_num, weights_num,
		outputs_num);

	return MODEL_PKG_OK;
}
