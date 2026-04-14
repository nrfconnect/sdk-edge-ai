/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "person_det_postprocess.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

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

static void decode_scale(const int8_t *base, uint16_t fh, uint16_t fw, uint16_t row_stride,
			 uint32_t dqm, uint8_t dqr, int8_t dqz, int stride_px, const float *anchors,
			 float in_w, float in_h, struct person_det_box *cand, int *ncand,
			 float score_thresh, enum person_det_head head_id)
{
	const struct person_det_sigmoid_lut *sigmoid_lut;

	switch (head_id) {
	case PERSON_DET_HEAD_STRIDE_32:
		sigmoid_lut = &g_sigmoid_lut_s32;
		break;
	case PERSON_DET_HEAD_STRIDE_16:
		sigmoid_lut = &g_sigmoid_lut_s16;
		break;
	default:
		sigmoid_lut = &g_sigmoid_lut_s8;
		break;
	}

	for (unsigned y = 0; y < fh; y++) {
		for (unsigned x = 0; x < fw; x++) {
			for (int a = 0; a < NUM_ANCHORS; a++) {
				int o = a * PRED_PER_ANCH;
				int8_t qcx = read_ch(base, fh, row_stride, (uint16_t)(o + 0), y, x);
				int8_t qcy = read_ch(base, fh, row_stride, (uint16_t)(o + 1), y, x);
				int8_t qw = read_ch(base, fh, row_stride, (uint16_t)(o + 2), y, x);
				int8_t qh = read_ch(base, fh, row_stride, (uint16_t)(o + 3), y, x);
				int8_t qobj = read_ch(base, fh, row_stride, (uint16_t)(o + 4), y, x);
				int8_t qcls = read_ch(base, fh, row_stride, (uint16_t)(o + 5), y, x);

				// float rw = dequant_q(sigmoid_lut, qw);
				// float rh = dequant_q(sigmoid_lut, qh);

				float cx = (sigmoid_q(sigmoid_lut, qcx) + (float)x) * (float)stride_px;
				float cy = (sigmoid_q(sigmoid_lut, qcy) + (float)y) * (float)stride_px;

				// rw = fminf(rw, 10.f);
				// rh = fminf(rh, 10.f);
				// float bw = expf(rw) * anchors[a * 2 + 0];
				// float bh = expf(rh) * anchors[a * 2 + 1];

				// float bw = rw * anchors[a * 2 + 0];
				// float bh = rh * anchors[a * 2 + 1];

				float bw = exp_q(sigmoid_lut, qw) * anchors[a * 2 + 0];
				float bh = exp_q(sigmoid_lut, qh) * anchors[a * 2 + 1];

				float hw = bw * 0.5f;
				float hh = bh * 0.5f;

				float conf = sigmoid_q(sigmoid_lut, qobj) * sigmoid_q(sigmoid_lut, qcls);
				if (conf < score_thresh || *ncand >= MAX_CANDIDATES) {
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
				if (x2 > in_w) {
					x2 = in_w;
				}
				if (y2 > in_h) {
					y2 = in_h;
				}
				if (x2 <= x1 || y2 <= y1) {
					continue;
				}

				cand[*ncand].x1 = x1;
				cand[*ncand].y1 = y1;
				cand[*ncand].x2 = x2;
				cand[*ncand].y2 = y2;
				cand[*ncand].score = conf;
				cand[*ncand].head = head_id;
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

static int nms(struct person_det_box *dets, int n, struct person_det_box *out, int max_out,
	       float iou_thres)
{
	if (n <= 0) {
		return 0;
	}

	qsort(dets, (size_t)n, sizeof(dets[0]), cmp_score_desc);

	int kept = 0;

	for (int i = 0; i < n && kept < max_out; i++) {
		bool take = true;

		for (int j = 0; j < kept; j++) {
			if (bbox_iou(&dets[i], &out[j]) > iou_thres) {
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
			      int max_out, float score_thresh, float nms_iou)
{
	struct person_det_box cand[MAX_CANDIDATES];
	int ncand = 0;

	const nrf_axon_nn_model_layer_dimensions_s *idim = &model->inputs[model->external_input_ndx].dimensions;
	float in_w = (float)idim->width;
	float in_h = (float)idim->height;

	if (model->extra_output_cnt >= 2) {
		const nrf_axon_compiled_model_output_s *e1 = &model->extra_outputs[1];

		person_det_sigmoid_lut_init(&g_sigmoid_lut_s32, e1->dequant_mult,
					    e1->dequant_round, e1->dequant_zp);
	}
	if (model->extra_output_cnt >= 1) {
		const nrf_axon_compiled_model_output_s *e0 = &model->extra_outputs[0];

		person_det_sigmoid_lut_init(&g_sigmoid_lut_s16, e0->dequant_mult,
					    e0->dequant_round, e0->dequant_zp);
	}
	person_det_sigmoid_lut_init(&g_sigmoid_lut_s8, model->output_dequant_mult,
				    model->output_dequant_round, model->output_dequant_zp);

	/* Same tensor order as run_det_camera.py: stride 32, 16, 8 (see build_det_helper). */
	if (model->extra_output_cnt >= 2) {
		const nrf_axon_compiled_model_output_s *e1 = &model->extra_outputs[1];

		decode_scale(e1->ptr, e1->dimensions.height, e1->dimensions.width, e1->stride,
			     e1->dequant_mult, e1->dequant_round, e1->dequant_zp, 32,
			     k_anchors_stride32, in_w, in_h, cand, &ncand, score_thresh,
			     PERSON_DET_HEAD_STRIDE_32);
	}

	if (model->extra_output_cnt >= 1) {
		const nrf_axon_compiled_model_output_s *e0 = &model->extra_outputs[0];

		decode_scale(e0->ptr, e0->dimensions.height, e0->dimensions.width, e0->stride,
			     e0->dequant_mult, e0->dequant_round, e0->dequant_zp, 16,
			     k_anchors_stride16, in_w, in_h, cand, &ncand, score_thresh,
			     PERSON_DET_HEAD_STRIDE_16);
	}

	decode_scale(model->output_ptr, model->output_dimensions.height, model->output_dimensions.width,
		     model->output_stride, model->output_dequant_mult, model->output_dequant_round,
		     model->output_dequant_zp, 8, k_anchors_stride8, in_w, in_h, cand, &ncand,
		     score_thresh, PERSON_DET_HEAD_STRIDE_8);

	return nms(cand, ncand, out, max_out, nms_iou);
}
