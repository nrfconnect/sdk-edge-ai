/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "inference_postprocessing.h"

#include <stdbool.h>
#include <string.h>
#include <zephyr/sys/util.h>

#define PREVIOUS_PREDICTION_NUM                 (3)


typedef struct prediction_tracer_s
{
	/** Current prediction index */
	int index;

	/** Current prediction target */
	uint16_t target;

	/** Previous predictions context */
	prediction_ctx_t prev[PREVIOUS_PREDICTION_NUM];
} prediction_tracer_t;

/**
 * @brief Class prediction conditions for postprocessing
 * 
 */
typedef struct class_prediction_condition_s
{
	/* Minimum number of repetitions of a class for prediction */
	uint16_t min_repeat_count;

	/* Minimum probability threshold for prediction */
	float probability_threshold;
} class_prediction_condition_t;

static const char *get_name_by_target_(uint8_t predicted_target)
{
	static const char * const LABEL_VS_NAME[] = {
		[CLASS_LABEL_IDLE]           = "IDLE",
		[CLASS_LABEL_UNKNOWN]        = "UNKNOWN",
		[CLASS_LABEL_SWIPE_LEFT]     = "SWIPE LEFT",
		[CLASS_LABEL_SWIPE_RIGHT]    = "SWIPE RIGHT",
		[CLASS_LABEL_DOUBLE_SHAKE]   = "DOUBLE SHAKE",
		[CLASS_LABEL_DOUBLE_THUMB]   = "DOUBLE THUMB",
		[CLASS_LABEL_ROTATION_RIGHT] = "ROTATION RIGHT",
		[CLASS_LABEL_ROTATION_LEFT]  = "ROTATION LEFT"
	};

	static const uint8_t LABELS_CNT = ARRAY_SIZE(LABEL_VS_NAME);
	__ASSERT_NO_MSG(LABELS_CNT == CLASS_LABEL_COUNT);

	return (predicted_target < LABELS_CNT) ? LABEL_VS_NAME[predicted_target] : NULL;
}

static const class_prediction_condition_t *get_class_condition_(uint8_t predicted_target)
{
	static const class_prediction_condition_t LABEL_VS_CONFIG[] = {
		[CLASS_LABEL_IDLE]           = {0, 0.0},
		[CLASS_LABEL_UNKNOWN]        = {0, 0.0},
		[CLASS_LABEL_SWIPE_LEFT]     = {2, 0.8},
		[CLASS_LABEL_SWIPE_RIGHT]    = {2, 0.8},
		[CLASS_LABEL_DOUBLE_SHAKE]   = {2, 0.7},
		[CLASS_LABEL_DOUBLE_THUMB]   = {3, 0.7},
		[CLASS_LABEL_ROTATION_RIGHT] = {2, 0.7},
		[CLASS_LABEL_ROTATION_LEFT]  = {2, 0.7},
	};

	static const uint8_t LABELS_CNT = ARRAY_SIZE(LABEL_VS_CONFIG);

	return (predicted_target < LABELS_CNT) ? &LABEL_VS_CONFIG[predicted_target] : NULL;
}

static void reset_tracer_(prediction_tracer_t *tracer, uint16_t target)
{
	__ASSERT_NO_MSG(tracer != NULL);
	tracer->index = 0;
	tracer->target = target;
}

static void reset_tracer_if_index_overflow_(prediction_tracer_t *tracer)
{
	__ASSERT_NO_MSG(tracer != NULL);
	if (tracer->index >= PREVIOUS_PREDICTION_NUM) {
		tracer->index = 0;
	}
}

static void reset_tracer_if_target_changed_(prediction_tracer_t *tracer, uint16_t target)
{
	__ASSERT_NO_MSG(tracer != NULL);
	if (tracer->target != target) {
		reset_tracer_(tracer, target);
	}
}

static void record_prediction_(prediction_tracer_t *tracer, float probability)
{
	__ASSERT_NO_MSG(tracer != NULL);
	__ASSERT_NO_MSG(tracer->index < PREVIOUS_PREDICTION_NUM);
	tracer->prev[tracer->index].probability = probability;
	tracer->index++;
}

static float average_probability_(const prediction_tracer_t *tracer)
{
	float average_prob = 0.0f;

	__ASSERT_NO_MSG(tracer != NULL);
	__ASSERT_NO_MSG(tracer->index > 0);
	for (int i = 0; i < tracer->index; ++i) {
		average_prob += tracer->prev[i].probability;
	}

	return average_prob / tracer->index;
}

static bool is_repetitive_class_(uint16_t target)
{
	return (target == CLASS_LABEL_ROTATION_RIGHT) ||
	       (target == CLASS_LABEL_ROTATION_LEFT);
}

static bool apply_conditions_(const class_prediction_condition_t *condition,
			      const prediction_tracer_t *tracer,
			      uint16_t *target,
			      float *probability)
{
	__ASSERT_NO_MSG(condition != NULL);
	__ASSERT_NO_MSG(tracer != NULL);
	__ASSERT_NO_MSG(target != NULL);
	__ASSERT_NO_MSG(probability != NULL);
	if (tracer->index < condition->min_repeat_count) {
		*target = CLASS_LABEL_UNKNOWN;
		*probability = 0.0f;
		return false;
	}

	float average_prob = average_probability_(tracer);
	if (average_prob < condition->probability_threshold) {
		*target = CLASS_LABEL_UNKNOWN;
		*probability = 0.0f;
	} else {
		*probability = average_prob;
	}

	return true;
}

prediction_ctx_t inference_postprocess(uint16_t target, float probability)
{
	prediction_ctx_t result = {
		.target = CLASS_LABEL_UNKNOWN,
		.probability = 0.0f,
	};
	
	static prediction_tracer_t tracer_;

	if ((target == CLASS_LABEL_UNKNOWN) || (target == CLASS_LABEL_IDLE)) {
		/* Reset tracer for UNKNOWN and IDLE classes */
		reset_tracer_(&tracer_, target);
	} else {
		reset_tracer_if_index_overflow_(&tracer_);
		reset_tracer_if_target_changed_(&tracer_, target);
		record_prediction_(&tracer_, probability);

		const class_prediction_condition_t *class_condition =
			get_class_condition_(target);
		if (class_condition == NULL) {
			return result;
		}

		bool evaluated = apply_conditions_(class_condition, &tracer_, &target, &probability);

		/* Reset tracer index for non-repetitive classes */
		if (evaluated && !is_repetitive_class_(target)) {
			tracer_.index = 0;
		}
	}

	result.target = target;
	result.probability = probability;
	return result;
}

const char *inference_get_class_name(const class_label_t class_label)
{
	return get_name_by_target_((uint8_t)class_label);
}
