/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <model_ota/model_image.h>

#include <string.h>

#include <zephyr/logging/log.h>
#include <zephyr/storage/flash_map.h>
#include <zephyr/sys/crc.h>

LOG_MODULE_REGISTER(model_image, CONFIG_MODEL_OTA_LOG_LEVEL);

static bool magic_is_valid(const struct model_image_header *hdr)
{
	return hdr->magic[0] == MODEL_IMAGE_MAGIC0 && hdr->magic[1] == MODEL_IMAGE_MAGIC1 &&
	       hdr->magic[2] == MODEL_IMAGE_MAGIC2 && hdr->magic[3] == MODEL_IMAGE_MAGIC3;
}

/* Bytes per weight/neuron element for a given enum model_image_params_type. */
static size_t params_elem_size(uint8_t params_type)
{
	switch (params_type) {
	case MODEL_IMAGE_PARAMS_F32:
		return 4;
	case MODEL_IMAGE_PARAMS_Q16:
		return 2;
	case MODEL_IMAGE_PARAMS_Q8:
		return 1;
	default:
		return 0;
	}
}

/* True iff [p, p + nbytes) lies fully inside the memory-mapped image [base, end). A NULL pointer
 * is never in range - callers allow NULL explicitly where a field is optional. The (p + nbytes)
 * >= p guard rejects a span that would wrap the address space.
 */
static bool span_in_image(const void *p, size_t nbytes, const uint8_t *base, const uint8_t *end)
{
	const uint8_t *s = (const uint8_t *)p;

	return s != NULL && s >= base && (s + nbytes) >= s && (s + nbytes) <= end;
}

int model_image_load_neuton(uint8_t fa_id, const uint8_t *partition_addr,
			    nrf_edgeai_model_neuton_t *out_model, void *neurons_buf,
			    size_t neurons_buf_cap,
			    const struct model_image_neuton_expect *expect,
			    struct model_image_neuton_info *out_info)
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

	/* Reject a model whose task or weight/neuron precision differs from what the app was built
	 * for: a model-only update keeps the same solution, so both must match. The precision also
	 * fixes the neuron-buffer element size, so a mismatch here would otherwise corrupt memory
	 * when p_neurons is patched below.
	 */
	if (expect != NULL) {
		if (hdr.task != expect->task) {
			LOG_ERR("Image task %u != expected %u", hdr.task, expect->task);
			return MODEL_IMAGE_ERR_TASK_MISMATCH;
		}
		if (hdr.params_type != expect->params_type) {
			LOG_ERR("Image params_type %u != expected %u", hdr.params_type,
				expect->params_type);
			return MODEL_IMAGE_ERR_PARAMS_TYPE_MISMATCH;
		}
	}

	/* The baked descriptor is addressed by an absolute flash pointer (the image was linked at
	 * the partition base). Confirm it lies fully inside [base, base + image_size) before we
	 * dereference it.
	 */
	const uint8_t *image_end = partition_addr + hdr.image_size;
	const uint8_t *model_bytes = (const uint8_t *)hdr.model;

	if (model_bytes < partition_addr ||
	    model_bytes + sizeof(nrf_edgeai_model_neuton_t) > image_end) {
		LOG_ERR("Header model pointer %p outside image [%p, %p)", (void *)hdr.model,
			(const void *)partition_addr, (const void *)image_end);
		return MODEL_IMAGE_ERR_MODEL_PTR_OUT_OF_RANGE;
	}

	const nrf_edgeai_model_neuton_t *img_model = hdr.model;
	uint16_t neurons_num = img_model->meta.neurons_num;
	uint16_t outputs_num = img_model->meta.outputs_num;
	uint32_t weights_num = img_model->meta.weights_num;

	if (neurons_num > neurons_buf_cap) {
		LOG_ERR("Model needs %u neurons, only %u provided", neurons_num,
			(unsigned)neurons_buf_cap);
		return MODEL_IMAGE_ERR_NEURONS_BUF_TOO_SMALL;
	}

	/* The app's output buffers (decoded output, probabilities, model_outputs_) are compile-time
	 * sized; an image with more outputs than the app can hold would overflow them downstream.
	 */
	if (expect != NULL && outputs_num > expect->outputs_cap) {
		LOG_ERR("Model has %u outputs, app buffers hold %u", outputs_num,
			expect->outputs_cap);
		return MODEL_IMAGE_ERR_OUTPUTS_TOO_MANY;
	}

	/* Defence in depth: CRC proves the image is intact, but a well-formed image that was linked
	 * at the wrong base (or a crafted one) could still carry pointers into app RAM or past the
	 * partition. Confirm every baked pointer we are about to hand to the inference engine lands
	 * inside [base, image_end). Spans are checked where the element count is known from meta;
	 * p_neuron_links / p_act_weights have variable, meta-implicit lengths, so only their base is
	 * bounded (the CRC still covers their contents). p_neurons is excluded - it is overwritten
	 * with the caller's RAM buffer below.
	 */
	const nrf_nn_neuton_model_meta_t *m = &img_model->meta;
	size_t welem = params_elem_size(hdr.params_type);

	if (!span_in_image(m->p_neuron_internal_links_num, (size_t)neurons_num * sizeof(uint16_t),
			   partition_addr, image_end) ||
	    !span_in_image(m->p_neuron_external_links_num, (size_t)neurons_num * sizeof(uint16_t),
			   partition_addr, image_end) ||
	    !span_in_image(m->p_output_neurons_indices, (size_t)outputs_num * sizeof(uint16_t),
			   partition_addr, image_end) ||
	    !span_in_image(m->p_neuron_links, 1, partition_addr, image_end) ||
	    !span_in_image(m->p_neuron_act_type_mask, 1, partition_addr, image_end) ||
	    !span_in_image(img_model->params.f32.p_weights, (size_t)weights_num * welem,
			   partition_addr, image_end) ||
	    !span_in_image(img_model->params.f32.p_act_weights, 1, partition_addr, image_end)) {
		LOG_ERR("Baked descriptor pointer outside image [%p, %p)",
			(const void *)partition_addr, (const void *)image_end);
		return MODEL_IMAGE_ERR_PTR_OUT_OF_RANGE;
	}

	/* Optional output-scale arrays live in the header; when present they are outputs_num floats
	 * inside this same image.
	 */
	if ((hdr.output_scale_min != NULL &&
	     !span_in_image(hdr.output_scale_min, (size_t)outputs_num * sizeof(float),
			    partition_addr, image_end)) ||
	    (hdr.output_scale_max != NULL &&
	     !span_in_image(hdr.output_scale_max, (size_t)outputs_num * sizeof(float),
			    partition_addr, image_end)) ||
	    (hdr.average_embedding != NULL &&
	     !span_in_image(hdr.average_embedding, (size_t)outputs_num * sizeof(float),
			    partition_addr, image_end))) {
		LOG_ERR("Header scale pointer outside image [%p, %p)", (const void *)partition_addr,
			(const void *)image_end);
		return MODEL_IMAGE_ERR_PTR_OUT_OF_RANGE;
	}

	/* Copy the flash-resident parameter pointer-triple, then overwrite only p_neurons. All
	 * three precision variants place p_neurons as the third (last) pointer, so patching via
	 * any union member is layout-identical - see model_image.h. The union is built as a
	 * non-const local first because nrf_edgeai_model_neuton_t's members are const-qualified.
	 */
	nrf_edgeai_model_neuton_params_t params;

	memcpy(&params, &img_model->params, sizeof(params));
	params.f32.p_neurons = neurons_buf;

	nrf_edgeai_model_neuton_t built = {
		.meta = img_model->meta,
		.params = params,
	};

	memcpy(out_model, &built, sizeof(built));

	if (out_info != NULL) {
		memset(out_info->name, 0, sizeof(out_info->name));
		memcpy(out_info->name, hdr.name, MODEL_IMAGE_NAME_LEN);
		out_info->version = hdr.model_version;
		out_info->neurons_num = neurons_num;
		out_info->outputs_num = img_model->meta.outputs_num;
		out_info->weights_num = img_model->meta.weights_num;
		out_info->output_scale_min = hdr.output_scale_min;
		out_info->output_scale_max = hdr.output_scale_max;
		out_info->average_embedding = hdr.average_embedding;
	}

	LOG_INF("Loaded Neuton model image '%.*s' v0x%08x (%u neurons, %u weights, %u outputs)",
		MODEL_IMAGE_NAME_LEN, hdr.name, hdr.model_version, neurons_num,
		img_model->meta.weights_num, img_model->meta.outputs_num);

	return MODEL_IMAGE_OK;
}
