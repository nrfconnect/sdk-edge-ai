/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "model_image_common.h"

#include <model_ota/model_image.h>

#include <string.h>

#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(model_image, CONFIG_MODEL_OTA_LOG_LEVEL);

static enum model_image_result neuton_patch_neurons_buf(nrf_edgeai_model_neuton_params_t *params, uint8_t params_type,
				    void *neurons_buf)
{
	switch (params_type) {
	case MODEL_IMAGE_PARAMS_F32:
		params->f32.p_neurons = neurons_buf;
		return MODEL_IMAGE_OK;
	case MODEL_IMAGE_PARAMS_Q16:
		params->q16.p_neurons = neurons_buf;
		return MODEL_IMAGE_OK;
	case MODEL_IMAGE_PARAMS_Q8:
		params->q8.p_neurons = neurons_buf;
		return MODEL_IMAGE_OK;
	default:
		return MODEL_IMAGE_ERR_BAD_PARAMS_TYPE;
	}
}

enum model_image_result model_image_load_neuton(const uint8_t *partition_addr, size_t partition_size,
			    nrf_edgeai_t *edgeai, void *neurons_buf, size_t neurons_buf_cap,
			    const struct model_image_neuton_expect *expect)
{
	struct model_image_header hdr;
	const nrf_edgeai_model_neuton_t *img_model;
	nrf_edgeai_model_neuton_t *out_model =
		(nrf_edgeai_model_neuton_t *)edgeai->model.instance.p_void;
	nrf_edgeai_model_neuton_params_t params;
	uint16_t neurons_num;
	enum model_image_result rc;

	rc = model_image_read_and_validate(partition_addr, partition_size, &hdr);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	if (expect == NULL) {
		LOG_ERR("Neuton expect contract is required");
		return MODEL_IMAGE_ERR_CONTRACT_MISMATCH;
	}

	if (hdr.contract_hash != expect->contract_hash) {
		LOG_ERR("Contract hash mismatch (image 0x%08x, expected 0x%08x)", hdr.contract_hash,
			expect->contract_hash);
		return MODEL_IMAGE_ERR_CONTRACT_MISMATCH;
	}

	rc = model_image_neuton_params_elem_size(hdr.params_type, NULL);
	if (rc != MODEL_IMAGE_OK) {
		LOG_ERR("Unsupported Neuton params_type %u", hdr.params_type);
		return rc;
	}

	/* Deliberately redundant with the contract hash: the precision fixes the neuron-buffer
	 * element size, so a mismatch here would corrupt memory when p_neurons is patched below,
	 * and that is not a property to leave to a 32-bit non-cryptographic hash alone.
	 */
	if (hdr.params_type != expect->params_type) {
		LOG_ERR("Image params_type %u != expected %u", hdr.params_type,
			expect->params_type);
		return MODEL_IMAGE_ERR_PARAMS_TYPE_MISMATCH;
	}

	/* The baked pointers below are absolute flash addresses linked at the partition base, and
	 * that base is part of contract_hash, so a wrong-slot image was rejected above.
	 * Containment is a build-time property (validate_model_image_layout.py).
	 */
	img_model = hdr.neuton.model;
	neurons_num = img_model->meta.neurons_num;

	if (neurons_num > expect->neurons_cap) {
		LOG_ERR("Model needs %u neurons, app cap is %u", neurons_num, expect->neurons_cap);
		return MODEL_IMAGE_ERR_NEURONS_BUF_TOO_SMALL;
	}

	if (neurons_num > neurons_buf_cap) {
		LOG_ERR("Model needs %u neurons, only %u provided", neurons_num,
			(unsigned)neurons_buf_cap);
		return MODEL_IMAGE_ERR_NEURONS_BUF_TOO_SMALL;
	}

	memcpy(&params, &img_model->params, sizeof(params));
	rc = neuton_patch_neurons_buf(&params, hdr.params_type, neurons_buf);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	nrf_edgeai_model_neuton_t built = {
		.meta = img_model->meta,
		.params = params,
	};

	memcpy(out_model, &built, sizeof(built));

	LOG_INF("Loaded Neuton model image '%s' v0x%08x (%u neurons, %u weights, %u outputs)",
		hdr.name, hdr.model_version, neurons_num, img_model->meta.weights_num,
		img_model->meta.outputs_num);

	return MODEL_IMAGE_OK;
}
