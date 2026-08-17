/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @defgroup app_sbm Sensor Based Models Application
 *
 * @defgroup inference_postprocessing Inference Postprocessing
 * @{
 * @ingroup app_sbm
 */

#ifndef __INFERENCE_POSTPROCESSING_H__
#define __INFERENCE_POSTPROCESSING_H__

#include <stdint.h>
#include <zephyr/sys/util.h>

#if IS_ENABLED(CONFIG_SBM_HAR_MODEL)
typedef enum {
	CLASS_LABEL_WALKING = 0,
	CLASS_LABEL_WALKING_UPSTAIRS,
	CLASS_LABEL_WALKING_DOWNSTAIRS,
	CLASS_LABEL_SITTING,
	CLASS_LABEL_STANDING,
	CLASS_LABEL_LYING,
	CLASS_LABEL_COUNT,
	CLASS_LABEL_UNKNOWN = 0xFF,
} class_label_t;
#elif IS_ENABLED(CONFIG_SBM_CAPTURE_24_MODEL)
typedef enum {
	CLASS_LABEL_BICYCLING = 0,
	CLASS_LABEL_SLEEP,
	CLASS_LABEL_VEHICLE,
	CLASS_LABEL_WALKING,
	CLASS_LABEL_COUNT,
	CLASS_LABEL_UNKNOWN = 0xFF,
} class_label_t;
#endif

typedef struct prediction_ctx_s {
	uint16_t target;
	float probability;
} prediction_ctx_t;

#if IS_ENABLED(CONFIG_SBM_INFERENCE_POSTPROCESSING)
prediction_ctx_t inference_postprocess(uint16_t predicted_target, float probability);
#endif

const char *inference_get_class_name(class_label_t class_label);

#endif /* __INFERENCE_POSTPROCESSING_H__ */
