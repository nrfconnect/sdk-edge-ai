/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "person_det_postprocess.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#define NUM_ANCHORS    3
#define PRED_PER_ANCH  6
#define MAX_CANDIDATES 400

static const float k_anchors_stride32[6] = {116.f, 90.f, 156.f, 198.f, 373.f, 326.f};
static const float k_anchors_stride16[6] = {30.f, 61.f, 62.f, 45.f, 59.f, 119.f};
static const float k_anchors_stride8[6] = {10.f, 13.f, 16.f, 30.f, 33.f, 23.f};

static float dequant(int8_t q, uint32_t mult, uint8_t round, int8_t zp);

struct person_det_sigmoid_lut {
	float sigmoid_values[UINT8_MAX + 1U];
	float exp_values[UINT8_MAX + 1U];
	uint32_t mult;
	uint8_t round;
	int8_t zp;
	bool ready;
};

static void person_det_sigmoid_lut_init(struct person_det_sigmoid_lut *lut, uint32_t mult,
					uint8_t round, int8_t zp)
{
	if (lut->ready && lut->mult == mult && lut->round == round && lut->zp == zp) {
		return;
	}

	for (unsigned i = 0; i <= UINT8_MAX; i++) {
		int8_t q = (int8_t)i;
		float x = dequant(q, mult, round, zp);

		if (x >= 0.f) {
			float z = expf(-x);

			lut->sigmoid_values[i] = 1.f / (1.f + z);
		} else {
			float z = expf(x);

			lut->sigmoid_values[i] = z / (1.f + z);
		}

		lut->exp_values[i] = expf(x);
	}

	lut->mult = mult;
	lut->round = round;
	lut->zp = zp;
	lut->ready = true;
}

static float sigmoid_q(const struct person_det_sigmoid_lut *lut, int8_t q)
{
	return lut->sigmoid_values[(uint8_t)q];
}

static float exp_q(const struct person_det_sigmoid_lut *lut, int8_t q)
{
	return lut->exp_values[(uint8_t)q];
}

static struct person_det_sigmoid_lut g_sigmoid_lut_s32;
static struct person_det_sigmoid_lut g_sigmoid_lut_s16;
static struct person_det_sigmoid_lut g_sigmoid_lut_s8;

struct person_det_input_size {
	float width;
	float height;
};

struct person_det_output_desc {
	const int8_t *base;
	uint16_t height;
	uint16_t width;
	uint16_t row_stride;
	uint32_t dequant_mult;
	uint8_t dequant_round;
	int8_t dequant_zp;
};

struct person_det_scale_desc {
	const int8_t *base;
	uint16_t height;
	uint16_t width;
	uint16_t row_stride;
	int stride_px;
	const float *anchors;
	const struct person_det_sigmoid_lut *sigmoid_lut;
	enum person_det_head head_id;
};

struct person_det_head_desc {
	enum person_det_head head_id;
	int stride_px;
	int extra_output_idx;
	const float *anchors;
	struct person_det_sigmoid_lut *sigmoid_lut;
};

static const struct person_det_head_desc k_head_descs[] = {
	{
		.head_id = PERSON_DET_HEAD_STRIDE_32,
		.stride_px = 32,
		.extra_output_idx = 1,
		.anchors = k_anchors_stride32,
		.sigmoid_lut = &g_sigmoid_lut_s32,
	},
	{
		.head_id = PERSON_DET_HEAD_STRIDE_16,
		.stride_px = 16,
		.extra_output_idx = 0,
		.anchors = k_anchors_stride16,
		.sigmoid_lut = &g_sigmoid_lut_s16,
	},
	{
		.head_id = PERSON_DET_HEAD_STRIDE_8,
		.stride_px = 8,
		.extra_output_idx = -1,
		.anchors = k_anchors_stride8,
		.sigmoid_lut = &g_sigmoid_lut_s8,
	},
};

const char *person_det_head_name(enum person_det_head head)
{
	switch (head) {
	case PERSON_DET_HEAD_STRIDE_32:
		return "s32";
	case PERSON_DET_HEAD_STRIDE_16:
		return "s16";
	case PERSON_DET_HEAD_STRIDE_8:
		return "s8";
	default:
		return "?";
	}
}

static float dequant(int8_t q, uint32_t mult, uint8_t round, int8_t zp)
{
	return (float)((int)q - (int)zp) * (float)mult / (float)(1U << round);
}

static int8_t read_ch(const int8_t *base, uint16_t h, uint16_t row_stride, uint16_t c, uint16_t y,
		      uint16_t x)
{
	return base[(size_t)c * (size_t)h * (size_t)row_stride + (size_t)y * (size_t)row_stride +
		    (size_t)x];
}

static void decode_scale(const struct person_det_scale_desc *scale,
			 const struct person_det_input_size *input_size,
			 const struct person_det_decode_config *config, struct person_det_box *cand,
			 int *ncand);

static bool get_output_desc(const nrf_axon_nn_compiled_model_s *model,
			    const struct person_det_head_desc *head,
			    struct person_det_output_desc *output)
{
	if (head->extra_output_idx >= 0) {
		if (model->extra_output_cnt <= (uint32_t)head->extra_output_idx) {
			return false;
		}

		const nrf_axon_compiled_model_output_s *extra =
			&model->extra_outputs[head->extra_output_idx];

		*output = (struct person_det_output_desc){
			.base = extra->ptr,
			.height = extra->dimensions.height,
			.width = extra->dimensions.width,
			.row_stride = extra->stride,
			.dequant_mult = extra->dequant_mult,
			.dequant_round = extra->dequant_round,
			.dequant_zp = extra->dequant_zp,
		};

		return true;
	}

	*output = (struct person_det_output_desc){
		.base = model->output_ptr,
		.height = model->output_dimensions.height,
		.width = model->output_dimensions.width,
		.row_stride = model->output_stride,
		.dequant_mult = model->output_dequant_mult,
		.dequant_round = model->output_dequant_round,
		.dequant_zp = model->output_dequant_zp,
	};

	return true;
}

static void decode_head_output(const struct person_det_head_desc *head,
			       const struct person_det_output_desc *output,
			       const struct person_det_input_size *input_size,
			       const struct person_det_decode_config *config,
			       struct person_det_box *cand, int *ncand)
{
	person_det_sigmoid_lut_init(head->sigmoid_lut, output->dequant_mult, output->dequant_round,
				    output->dequant_zp);

	const struct person_det_scale_desc scale = {
		.base = output->base,
		.height = output->height,
		.width = output->width,
		.row_stride = output->row_stride,
		.stride_px = head->stride_px,
		.anchors = head->anchors,
		.sigmoid_lut = head->sigmoid_lut,
		.head_id = head->head_id,
	};

	decode_scale(&scale, input_size, config, cand, ncand);
}

static void decode_scale(const struct person_det_scale_desc *scale,
			 const struct person_det_input_size *input_size,
			 const struct person_det_decode_config *config, struct person_det_box *cand,
			 int *ncand)
{
	for (unsigned y = 0; y < scale->height; y++) {
		for (unsigned x = 0; x < scale->width; x++) {
			for (int a = 0; a < NUM_ANCHORS; a++) {
				int o = a * PRED_PER_ANCH;
				int8_t qcx = read_ch(scale->base, scale->height, scale->row_stride,
						   (uint16_t)(o + 0), y, x);
				int8_t qcy = read_ch(scale->base, scale->height, scale->row_stride,
						   (uint16_t)(o + 1), y, x);
				int8_t qw = read_ch(scale->base, scale->height, scale->row_stride,
						  (uint16_t)(o + 2), y, x);
				int8_t qh = read_ch(scale->base, scale->height, scale->row_stride,
						  (uint16_t)(o + 3), y, x);
				int8_t qobj = read_ch(scale->base, scale->height, scale->row_stride,
						    (uint16_t)(o + 4), y, x);
				int8_t qcls = read_ch(scale->base, scale->height, scale->row_stride,
						    (uint16_t)(o + 5), y, x);

				float cx =
					(sigmoid_q(scale->sigmoid_lut, qcx) + (float)x) * (float)scale->stride_px;
				float cy =
					(sigmoid_q(scale->sigmoid_lut, qcy) + (float)y) * (float)scale->stride_px;
				float bw = exp_q(scale->sigmoid_lut, qw) * scale->anchors[a * 2 + 0];
				float bh = exp_q(scale->sigmoid_lut, qh) * scale->anchors[a * 2 + 1];

				float hw = bw * 0.5f;
				float hh = bh * 0.5f;

				float conf = sigmoid_q(scale->sigmoid_lut, qobj) *
					     sigmoid_q(scale->sigmoid_lut, qcls);
				if (conf < config->score_thresh || *ncand >= MAX_CANDIDATES) {
					continue;
				}

				float x1 = cx - hw;
				float y1 = cy - hh;
				float x2 = cx + hw;
				float y2 = cy + hh;

				if (x1 < 0.f) {
					x1 = 0.f;
				}
				if (y1 < 0.f) {
					y1 = 0.f;
				}
				if (x2 > input_size->width) {
					x2 = input_size->width;
				}
				if (y2 > input_size->height) {
					y2 = input_size->height;
				}
				if (x2 <= x1 || y2 <= y1) {
					continue;
				}

				cand[*ncand].x1 = x1;
				cand[*ncand].y1 = y1;
				cand[*ncand].x2 = x2;
				cand[*ncand].y2 = y2;
				cand[*ncand].score = conf;
				cand[*ncand].head = scale->head_id;
				(*ncand)++;
			}
		}
	}
}

static float bbox_iou(const struct person_det_box *a, const struct person_det_box *b)
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
	const struct person_det_box *a = va;
	const struct person_det_box *b = vb;

	if (a->score > b->score) {
		return -1;
	}
	if (a->score < b->score) {
		return 1;
	}
	return 0;
}

static int nms(struct person_det_box *dets, int n, struct person_det_box *out,
	       const struct person_det_decode_config *config)
{
	if (n <= 0) {
		return 0;
	}

	qsort(dets, (size_t)n, sizeof(dets[0]), cmp_score_desc);

	int kept = 0;

	for (int i = 0; i < n && kept < config->max_out; i++) {
		bool take = true;

		for (int j = 0; j < kept; j++) {
			if (bbox_iou(&dets[i], &out[j]) > config->nms_iou) {
				take = false;
				break;
			}
		}
		if (take) {
			out[kept++] = dets[i];
		}
	}

	return kept;
}

int person_det_decode_and_nms(const nrf_axon_nn_compiled_model_s *model, struct person_det_box *out,
			      const struct person_det_decode_config *config)
{
	if (model == NULL || out == NULL || config == NULL || config->max_out <= 0) {
		return 0;
	}

	struct person_det_box cand[MAX_CANDIDATES];
	int ncand = 0;

	const nrf_axon_nn_model_layer_dimensions_s *idim = &model->inputs[model->external_input_ndx].dimensions;
	const struct person_det_input_size input_size = {
		.width = (float)idim->width,
		.height = (float)idim->height,
	};

	/* Same tensor order as run_det_camera.py: stride 32, 16, 8 (see build_det_helper). */
	for (size_t i = 0; i < sizeof(k_head_descs) / sizeof(k_head_descs[0]); i++) {
		struct person_det_output_desc output;

		if (!get_output_desc(model, &k_head_descs[i], &output)) {
			continue;
		}

		decode_head_output(&k_head_descs[i], &output, &input_size, config, cand, &ncand);
	}

	return nms(cand, ncand, out, config);
}
