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

static bool neuton_weight_spans_ok(const nrf_edgeai_model_neuton_t *img_model, uint8_t params_type,
				   uint32_t weights_num, const uint8_t *base, const uint8_t *end)
{
	switch (params_type) {
	case MODEL_IMAGE_PARAMS_F32: {
		const nrf_edgeai_model_neuton_params_f32_t *p = &img_model->params.f32;

		return model_image_span_in_image(p->p_weights, (size_t)weights_num * 4, base,
						 end) &&
		       model_image_span_in_image(p->p_act_weights, 1, base, end);
	}
	case MODEL_IMAGE_PARAMS_Q16: {
		const nrf_edgeai_model_neuton_params_q16_t *p = &img_model->params.q16;

		return model_image_span_in_image(p->p_weights, (size_t)weights_num * 2, base,
						 end) &&
		       model_image_span_in_image(p->p_act_weights, 1, base, end);
	}
	case MODEL_IMAGE_PARAMS_Q8: {
		const nrf_edgeai_model_neuton_params_q8_t *p = &img_model->params.q8;

		return model_image_span_in_image(p->p_weights, (size_t)weights_num * 1, base,
						 end) &&
		       model_image_span_in_image(p->p_act_weights, 1, base, end);
	}
	default:
		return false;
	}
}

static int neuton_patch_neurons_buf(nrf_edgeai_model_neuton_params_t *params, uint8_t params_type,
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

int model_image_load_neuton(uint8_t fa_id, const uint8_t *partition_addr, nrf_edgeai_t *edgeai,
			    void *neurons_buf, size_t neurons_buf_cap,
			    const struct model_image_neuton_expect *expect)
{
	struct model_image_header hdr;
	const uint8_t *image_end;
	const uint8_t *model_bytes;
	const nrf_edgeai_model_neuton_t *img_model;
	const nrf_edgeai_decoded_output_t *img_decoded;
	nrf_edgeai_model_neuton_t *out_model =
		(nrf_edgeai_model_neuton_t *)edgeai->model.instance.p_void;
	nrf_edgeai_model_neuton_params_t params;
	uint16_t neurons_num;
	uint16_t outputs_num;
	uint32_t weights_num;
	int rc;

	rc = model_image_read_and_validate(fa_id, partition_addr, &hdr);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	rc = model_image_neuton_params_elem_size(hdr.params_type, NULL);
	if (rc != MODEL_IMAGE_OK) {
		LOG_ERR("Unsupported Neuton params_type %u", hdr.params_type);
		return rc;
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
	image_end = partition_addr + hdr.image_size;
	model_bytes = (const uint8_t *)hdr.model.neuton;

	if (model_bytes < partition_addr ||
	    model_bytes + sizeof(nrf_edgeai_model_neuton_t) > image_end) {
		LOG_ERR("Header model pointer %p outside image [%p, %p)", (void *)hdr.model.neuton,
			(const void *)partition_addr, (const void *)image_end);
		return MODEL_IMAGE_ERR_MODEL_PTR_OUT_OF_RANGE;
	}

	img_model = hdr.model.neuton;
	neurons_num = img_model->meta.neurons_num;
	outputs_num = img_model->meta.outputs_num;
	weights_num = img_model->meta.weights_num;

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

	if (!model_image_span_in_image(m->p_neuron_internal_links_num,
				       (size_t)neurons_num * sizeof(uint16_t), partition_addr,
				       image_end) ||
	    !model_image_span_in_image(m->p_neuron_external_links_num,
				       (size_t)neurons_num * sizeof(uint16_t), partition_addr,
				       image_end) ||
	    !model_image_span_in_image(m->p_output_neurons_indices,
				       (size_t)outputs_num * sizeof(uint16_t), partition_addr,
				       image_end) ||
	    !model_image_span_in_image(m->p_neuron_links, 1, partition_addr, image_end) ||
	    !model_image_span_in_image(m->p_neuron_act_type_mask, 1, partition_addr, image_end) ||
	    !neuton_weight_spans_ok(img_model, hdr.params_type, weights_num, partition_addr,
				    image_end)) {
		LOG_ERR("Baked descriptor pointer outside image [%p, %p)",
			(const void *)partition_addr, (const void *)image_end);
		return MODEL_IMAGE_ERR_PTR_OUT_OF_RANGE;
	}

	/* Baked NN_DECODED_OUTPUT_INIT lives in the image; confirm the struct and any flash-resident
	 * meta pointers it references lie inside [base, image_end).
	 */
	if (!model_image_span_in_image(hdr.decoded_output, sizeof(nrf_edgeai_decoded_output_t),
				       partition_addr, image_end)) {
		LOG_ERR("Header decoded_output pointer %p outside image [%p, %p)",
			(void *)hdr.decoded_output, (const void *)partition_addr,
			(const void *)image_end);
		return MODEL_IMAGE_ERR_PTR_OUT_OF_RANGE;
	}

	img_decoded = hdr.decoded_output;

	switch (hdr.task) {
	case NRF_EDGEAI_TASK_ANOMALY_DETECTION:
		if (!model_image_span_in_image(img_decoded->anomaly.meta.p_scale_min,
					       (size_t)outputs_num * sizeof(float), partition_addr,
					       image_end) ||
		    !model_image_span_in_image(img_decoded->anomaly.meta.p_scale_max,
					       (size_t)outputs_num * sizeof(float), partition_addr,
					       image_end) ||
		    !model_image_span_in_image(img_decoded->anomaly.meta.p_average_embedding,
					       (size_t)outputs_num * sizeof(float), partition_addr,
					       image_end)) {
			LOG_ERR("Baked anomaly decode meta pointer outside image [%p, %p)",
				(const void *)partition_addr, (const void *)image_end);
			return MODEL_IMAGE_ERR_PTR_OUT_OF_RANGE;
		}
		break;
	case NRF_EDGEAI_TASK_REGRESSION:
		if (!model_image_span_in_image(img_decoded->regression.meta.p_scale_min,
					       (size_t)outputs_num * sizeof(float), partition_addr,
					       image_end) ||
		    !model_image_span_in_image(img_decoded->regression.meta.p_scale_max,
					       (size_t)outputs_num * sizeof(float), partition_addr,
					       image_end)) {
			LOG_ERR("Baked regression decode meta pointer outside image [%p, %p)",
				(const void *)partition_addr, (const void *)image_end);
			return MODEL_IMAGE_ERR_PTR_OUT_OF_RANGE;
		}
		break;
	default:
		break;
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

	switch (hdr.task) {
	case NRF_EDGEAI_TASK_ANOMALY_DETECTION:
		edgeai->decoded_output.anomaly = img_decoded->anomaly;
		break;
	case NRF_EDGEAI_TASK_REGRESSION:
		edgeai->decoded_output.regression = img_decoded->regression;
		break;
	case NRF_EDGEAI_TASK_MULT_CLASS:
	case NRF_EDGEAI_TASK_BIN_CLASS:
		edgeai->decoded_output.classif = img_decoded->classif;
		break;
	default:
		break;
	}

	LOG_INF("Loaded Neuton model image '%.*s' v0x%08x (%u neurons, %u weights, %u outputs)",
		MODEL_IMAGE_NAME_LEN, hdr.name, hdr.model_version, neurons_num,
		img_model->meta.weights_num, img_model->meta.outputs_num);

	return MODEL_IMAGE_OK;
}
