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

struct person_det_box {
	float x1, y1, x2, y2;
	float score;
};

int person_det_decode_and_nms(const nrf_axon_nn_compiled_model_s *model, struct person_det_box *out,
				int max_out, float score_thresh, float nms_iou);
