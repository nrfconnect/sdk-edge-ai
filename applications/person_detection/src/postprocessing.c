/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "postprocessing.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(decode);

/*
 * Used person_det model use YOLOv3 style of output having 3 heads, one per stride size
 * Each head have HxW cells and each cell have three anchors.
 * Each anchor contains 4 values for box location and 2 values for box and class score.
 */
#define MODEL_HEADS	  3
#define NUM_ANCHORS	  3
#define VALUES_PER_ANCHOR 6

/* Despite model having 400 anchors in total, alloc space for less and filter by score. */
#define MAX_CANDIDATES 50

struct model_head_desc {
	const enum model_head head_id;
	const uint8_t stride_px;
	const uint8_t output_idx;
	const float anchors[NUM_ANCHORS][2];
};

struct model_head_lut {
	float sigmoid_values[UINT8_MAX + 1U];
	float exp_values[UINT8_MAX + 1U];
};

struct model_input_size {
	float width;
	float height;
};

static const struct model_head_desc head_descs[MODEL_HEADS] = {
	{
		.head_id = MODEL_HEAD_STRIDE_32,
		.stride_px = 32,
		.output_idx = 2,
		.anchors = {{116.f, 90.f}, {156.f, 198.f}, {373.f, 326.f}},
	},
	{
		.head_id = MODEL_HEAD_STRIDE_16,
		.stride_px = 16,
		.output_idx = 1,
		.anchors = {{30.f, 61.f}, {62.f, 45.f}, {59.f, 119.f}},
	},
	{
		.head_id = MODEL_HEAD_STRIDE_8,
		.stride_px = 8,
		.output_idx = 0,
		.anchors = {{10.f, 13.f}, {16.f, 30.f}, {33.f, 23.f}},
	},
};

static struct model_head_lut luts[MODEL_HEADS];

static float dequant(const int8_t q, const uint32_t mult, const uint8_t round, const int8_t zp)
{
	const float scale = (float)mult / (float)(1U << round);

	return (float)((int)q - (int)zp) * scale;
}

static void lut_init(struct model_head_lut *lut, const uint32_t mult, const uint8_t round,
		     const int8_t zp)
{
	for (size_t i = 0; i <= UINT8_MAX; i++) {
		const int8_t q = (int8_t)i;
		const float x = dequant(q, mult, round, zp);

		if (x >= 0.f) {
			const float z = expf(-x);

			lut->sigmoid_values[i] = 1.f / (1.f + z);
		} else {
			const float z = expf(x);

			lut->sigmoid_values[i] = z / (1.f + z);
		}

		lut->exp_values[i] = expf(x);
	}
}

static float sigmoid_q(const struct model_head_lut *lut, int8_t q)
{
	return lut->sigmoid_values[(uint8_t)q];
}

static float exp_q(const struct model_head_lut *lut, int8_t q)
{
	return lut->exp_values[(uint8_t)q];
}

static float box_iou(const struct detection_box *a, const struct detection_box *b)
{
	float xx1 = fmaxf(a->x1, b->x1);
	float yy1 = fmaxf(a->y1, b->y1);
	float xx2 = fminf(a->x2, b->x2);
	float yy2 = fminf(a->y2, b->y2);
	float w = fmaxf(0.f, xx2 - xx1);
	float h = fmaxf(0.f, yy2 - yy1);
	float inter = w * h;
	float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
	float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);

	return inter / (area_a + area_b - inter + 1e-6f);
}

static int cmp_score_desc(const void *va, const void *vb)
{
	const struct detection_box *a = va;
	const struct detection_box *b = vb;

	if (a->score > b->score) {
		return -1;
	}
	if (a->score < b->score) {
		return 1;
	}
	return 0;
}

static size_t nms(struct detection_box *candidates, const size_t n, struct detection_box *boxes,
		  const size_t boxes_size)
{
	qsort(candidates, (size_t)n, sizeof(*candidates), cmp_score_desc);

	const float iou_threshold = CONFIG_IOU_THRESHOLD / 1000.f;
	size_t kept = 0;

	for (size_t i = 0; i < n && kept < boxes_size; i++) {
		bool take = true;

		for (size_t j = 0; j < kept; j++) {
			if (box_iou(&candidates[i], &boxes[j]) > iou_threshold) {
				take = false;
				break;
			}
		}
		if (take) {
			boxes[kept++] = candidates[i];
		}
	}

	return kept;
}

static int8_t get_value(const int8_t *base, const nrf_axon_nn_model_layer_dimensions_s *out_dims,
			const size_t channel, const size_t row, const size_t col)
{
	const size_t channel_offset = channel * out_dims->height * out_dims->width;
	const size_t row_offset = row * out_dims->width;

	return base[channel_offset + row_offset + col];
}

static void decode_head(const struct model_head_desc *desc,
			const struct model_input_size *input_size,
			const nrf_axon_nn_model_layer_dimensions_s *dims, const int8_t *data,
			struct detection_box *cand, size_t *ncand)
{
	const float score_thresh = CONFIG_SCORE_THRESHOLD / 1000.f;
	const struct model_head_lut *lut = &luts[desc->output_idx];

	for (size_t row = 0; row < dims->height; row++) {
		for (size_t col = 0; col < dims->width; col++) {
			for (size_t anchor = 0; anchor < NUM_ANCHORS; anchor++) {
				const size_t channel = anchor * VALUES_PER_ANCHOR;

				const int8_t qcx = get_value(data, dims, channel + 0, row, col);
				const int8_t qcy = get_value(data, dims, channel + 1, row, col);
				const int8_t qw = get_value(data, dims, channel + 2, row, col);
				const int8_t qh = get_value(data, dims, channel + 3, row, col);
				const int8_t qobj = get_value(data, dims, channel + 4, row, col);
				const int8_t qcls = get_value(data, dims, channel + 5, row, col);

				const float conf = sigmoid_q(lut, qobj) * sigmoid_q(lut, qcls);

				if (conf < score_thresh) {
					continue;
				}

				if (*ncand >= MAX_CANDIDATES) {
					LOG_WRN("Skipping some candidates");
					return;
				}

				const float cx =
					(sigmoid_q(lut, qcx) + (float)col) * (float)desc->stride_px;
				const float cy =
					(sigmoid_q(lut, qcy) + (float)row) * (float)desc->stride_px;

				const float hw = exp_q(lut, qw) * desc->anchors[anchor][0] * 0.5f;
				const float hh = exp_q(lut, qh) * desc->anchors[anchor][1] * 0.5f;

				if (hw < 0 || hh < 0) {
					continue;
				}

				const float x1 = fmaxf(cx - hw, 0.f);
				const float y1 = fmaxf(cy - hh, 0.f);

				const float x2 = fminf(cx + hw, input_size->width);
				const float y2 = fminf(cy + hh, input_size->height);

				cand[*ncand] = (struct detection_box){
					.x1 = x1,
					.y1 = y1,
					.x2 = x2,
					.y2 = y2,
					.score = conf,
					.head_id = desc->head_id,
				};
				(*ncand)++;
			}
		}
	}
}

void decode_init(const nrf_axon_nn_compiled_model_s *model)
{
	__ASSERT_NO_MSG(model);
	__ASSERT_NO_MSG(ARRAY_SIZE(head_descs) == model->extra_output_cnt + 1);

	for (size_t i = 0; i < ARRAY_SIZE(head_descs); i++) {
		const uint8_t output_idx = head_descs[i].output_idx;

		if (output_idx == 0) {
			lut_init(&luts[output_idx], model->output_dequant_mult,
				 model->output_dequant_round, model->output_dequant_zp);
		} else {
			__ASSERT_NO_MSG(output_idx - 1 <= model->extra_output_cnt);

			const uint8_t extra_output_idx = output_idx - 1;

			const nrf_axon_compiled_model_output_s *model_output =
				&model->extra_outputs[extra_output_idx];

			lut_init(&luts[output_idx], model_output->dequant_mult,
				 model_output->dequant_round, model_output->dequant_zp);
		}
	}
}

size_t decode_output(const nrf_axon_nn_compiled_model_s *model, const int8_t *packed_data,
		     struct detection_box *boxes, const size_t boxes_size)
{
	__ASSERT_NO_MSG(model);
	__ASSERT_NO_MSG(ARRAY_SIZE(head_descs) == model->extra_output_cnt + 1);
	__ASSERT_NO_MSG(packed_data);
	__ASSERT_NO_MSG(boxes);

	struct detection_box cand[MAX_CANDIDATES];
	size_t ncand = 0;

	const nrf_axon_nn_model_layer_dimensions_s *in_dims =
		&model->inputs[model->external_input_ndx].dimensions;
	const struct model_input_size input_size = {
		.width = (float)in_dims->width,
		.height = (float)in_dims->height,
	};

	for (size_t i = 0; i < ARRAY_SIZE(head_descs); i++) {
		const uint8_t output_idx = head_descs[i].output_idx;
		const nrf_axon_nn_model_layer_dimensions_s *out_dims =
			output_idx == 0 ? &model->output_dimensions
					: &model->extra_outputs[output_idx - 1].dimensions;
		const int8_t *packed_output =
			packed_data + nrf_axon_nn_offset_to_output_ndx(model, output_idx);

		decode_head(&head_descs[i], &input_size, out_dims, packed_output, cand, &ncand);
	}

	return nms(cand, ncand, boxes, boxes_size);
}

const char *model_head_name(enum model_head head_id)
{
	switch (head_id) {
	case MODEL_HEAD_STRIDE_32:
		return "s32";
	case MODEL_HEAD_STRIDE_16:
		return "s16";
	case MODEL_HEAD_STRIDE_8:
		return "s8";
	default:
		return "?";
	}
}
