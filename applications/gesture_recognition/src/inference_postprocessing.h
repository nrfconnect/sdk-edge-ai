/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef __INFERENCE_POSTPROCESSING_H__
#define __INFERENCE_POSTPROCESSING_H__

#include <stdint.h>

typedef enum {
	CLASS_LABEL_IDLE,
	CLASS_LABEL_UNKNOWN,
	CLASS_LABEL_SWIPE_RIGHT,
	CLASS_LABEL_SWIPE_LEFT,
	CLASS_LABEL_DOUBLE_SHAKE,
	CLASS_LABEL_DOUBLE_THUMB,
	CLASS_LABEL_ROTATION_RIGHT,
	CLASS_LABEL_ROTATION_LEFT,
	CLASS_LABEL_COUNT,
} class_label_t;

typedef struct prediction_ctx_s
{
	/** Prediction target */
	uint16_t target;

	/** Prediction probability */
	float probability;
} prediction_ctx_t;

/**
 * @brief Postprocess the Neuton library RAW inference output
 * 
 * @param[in] predicted_target  Predicted target(class)
 * @param[in] probability       Predicted probability of the target
 *
 * @return Postprocessed inference result
 */
prediction_ctx_t inference_postprocess(const uint16_t predicted_target,
				       const float probability);

/**
 * @brief Get class name by label
 *
 * @param[in] class_label  Class label
 *
 * @return Class name, or NULL for invalid labels
 */
const char *inference_get_class_name(const class_label_t class_label);



#endif /* __INFERENCE_POSTPROCESSING_H__ */
