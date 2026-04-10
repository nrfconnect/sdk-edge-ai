/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * YOLOv3-style decode + greedy NMS for MCUNet person_det Axon model (matches
 * mcunet/utils/det_helper.py + run_det_camera.py).
 */

#pragma once

#include <drivers/axon/nrf_axon_nn_infer.h>
#include <stdint.h>

/** Which detection head produced the box (decode order matches tensor order). */
enum person_det_head {
	PERSON_DET_HEAD_STRIDE_32 = 0, /**< extra_outputs[1], coarse grid */
	PERSON_DET_HEAD_STRIDE_16 = 1, /**< extra_outputs[0] */
	PERSON_DET_HEAD_STRIDE_8 = 2,  /**< main output, finest grid */
};

struct person_det_box {
	float x1, y1, x2, y2;
	float score;
	enum person_det_head head;
};

/** Short label for logs (e.g. "s32", "s16", "s8"). */
const char *person_det_head_name(enum person_det_head head);

/**
 * Decode all three detection heads and run NMS. Uses raw output pointers inside
 * @p model (valid immediately after nrf_axon_nn_model_infer_sync()).
 *
 * @param score_thresh  minimum class score (after sigmoid), e.g. 0.25f
 * @param nms_iou       IoU threshold for suppression, e.g. 0.45f
 * @return number of boxes written to @p out (at most @p max_out)
 */
int person_det_decode_and_nms(const nrf_axon_nn_compiled_model_s *model, struct person_det_box *out,
				int max_out, float score_thresh, float nms_iou);
