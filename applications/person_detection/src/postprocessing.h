/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef __POSTPROCESSING_H__
#define __POSTPROCESSING_H__

#include <stddef.h>
#include <stdint.h>

#include <drivers/axon/nrf_axon_nn_infer.h>

/**
 * @brief Model head ID.
 */
enum model_head {
	MODEL_HEAD_STRIDE_32, /**< coarse grid, extra output 1 */
	MODEL_HEAD_STRIDE_16, /**< medium grid, extra output 0 */
	MODEL_HEAD_STRIDE_8,  /**< fine grid, main output */
};

/**
 * @brief Bounding box for object detection.
 */
struct detection_box {
	/** Coordinates of left-top and right-bottom box vertices. */
	float x1, y1, x2, y2;
	/** Bounding box score. */
	float score;
	/** Model head ID. */
	enum model_head head_id;
};

/**
 * @brief Initialize decode procedure.
 *
 * @param model Axon model.
 */
void decode_init(const nrf_axon_nn_compiled_model_s *model);

/**
 * @brief Decode output from the model and perform NMS.
 *
 * @param model Axon model.
 * @param packed_data Output from the model in packed format.
 * @param[out] boxes Place to store results.
 * @param boxes_size Size of @p boxes array.
 *
 * @return Number of boxes written to @p boxes.
 */
size_t decode_output(const nrf_axon_nn_compiled_model_s *model, const int8_t *packed_data,
		     struct detection_box *boxes, const size_t boxes_size);

/**
 * @brief Gives string name of model's head.
 *
 * @param head_id Model's head ID.
 *
 * @return Head name.
 */
const char *model_head_name(enum model_head head_id);

#endif /* __POSTPROCESSING_H__ */
