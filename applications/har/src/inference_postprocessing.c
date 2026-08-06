/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "inference_postprocessing.h"

#include <stdbool.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/util.h>

/* Consecutive identical raw predictions required before a class is published.
 *
 * The counts are asymmetric on purpose. A class the model often predicts by
 * mistake must clear a higher bar than the class it is taken from, or the
 * false positive always wins the race to the repeat count. WALKING is the
 * dominant confusion for WALKING_UPSTAIRS, so while a stair ascent is in
 * progress a lower count for WALKING lets almost every window be published as
 * level walking.
 *
 * Probabilities are averaged for reporting but never gated: on-device testing
 * showed the model is over-confident on the ambulation classes, so a
 * probability threshold suppresses correct and incorrect predictions alike.
 */
static const uint16_t MIN_REPEAT_COUNT[] = {
	[CLASS_LABEL_WALKING] = 3,
	[CLASS_LABEL_WALKING_UPSTAIRS] = 2,
	[CLASS_LABEL_WALKING_DOWNSTAIRS] = 3,
	[CLASS_LABEL_SITTING] = 3,
	[CLASS_LABEL_STANDING] = 3,
	[CLASS_LABEL_LYING] = 2,
};

BUILD_ASSERT(ARRAY_SIZE(MIN_REPEAT_COUNT) == CLASS_LABEL_COUNT);

/** Current run of identical raw predictions. */
typedef struct prediction_run_s {
	uint16_t target;
	uint16_t count;
	float probability_sum;
} prediction_run_t;

/** Raw predictions accumulated since the last published classification. */
typedef struct prediction_history_s {
	uint16_t count[CLASS_LABEL_COUNT];
	float probability_sum[CLASS_LABEL_COUNT];
	int64_t last_publish_uptime_ms;
} prediction_history_t;

static prediction_run_t prediction_run = {
	.target = CLASS_LABEL_UNKNOWN,
};

static prediction_history_t prediction_history;

static const char *get_name_by_target(uint8_t predicted_target)
{
	static const char *const LABEL_VS_NAME[] = {
		[CLASS_LABEL_WALKING] = "WALKING",
		[CLASS_LABEL_WALKING_UPSTAIRS] = "WALKING_UPSTAIRS",
		[CLASS_LABEL_WALKING_DOWNSTAIRS] = "WALKING_DOWNSTAIRS",
		[CLASS_LABEL_SITTING] = "SITTING",
		[CLASS_LABEL_STANDING] = "STANDING",
		[CLASS_LABEL_LYING] = "LYING",
	};

	BUILD_ASSERT(ARRAY_SIZE(LABEL_VS_NAME) == CLASS_LABEL_COUNT);

	if (predicted_target >= CLASS_LABEL_COUNT) {
		return "UNKNOWN";
	}

	return LABEL_VS_NAME[predicted_target];
}

static void reset_run(uint16_t target)
{
	prediction_run.target = target;
	prediction_run.count = 0;
	prediction_run.probability_sum = 0.0f;
}

static void reset_history(void)
{
	prediction_history = (prediction_history_t){
		.last_publish_uptime_ms = k_uptime_get(),
	};
}

static bool publish_timeout_expired(void)
{
	int64_t elapsed_ms = k_uptime_get() - prediction_history.last_publish_uptime_ms;

	return elapsed_ms >= CONFIG_HAR_POSTPROCESSING_PUBLISH_TIMEOUT_MS;
}

/**
 * @brief Pick the class predicted most often since the last published result.
 *
 * Used when the raw prediction keeps changing and no class ever reaches its
 * repeat count. Ties are resolved in favour of the higher accumulated
 * probability.
 */
static prediction_ctx_t most_frequent_prediction(void)
{
	prediction_ctx_t result = {
		.target = CLASS_LABEL_UNKNOWN,
		.probability = 0.0f,
	};
	uint16_t best_count = 0;
	float best_sum = 0.0f;

	for (uint16_t target = 0; target < CLASS_LABEL_COUNT; target++) {
		uint16_t count = prediction_history.count[target];
		float sum = prediction_history.probability_sum[target];

		if (count == 0) {
			continue;
		}

		if (count > best_count || (count == best_count && sum > best_sum)) {
			best_count = count;
			best_sum = sum;
			result.target = target;
			result.probability = sum / count;
		}
	}

	return result;
}

prediction_ctx_t inference_postprocess(uint16_t target, float probability)
{
	prediction_ctx_t result = {
		.target = CLASS_LABEL_UNKNOWN,
		.probability = 0.0f,
	};

	if (target >= CLASS_LABEL_COUNT) {
		reset_run(CLASS_LABEL_UNKNOWN);
		return result;
	}

	if (prediction_run.target != target) {
		reset_run(target);
	}

	prediction_run.count++;
	prediction_run.probability_sum += probability;
	prediction_history.count[target]++;
	prediction_history.probability_sum[target] += probability;

	if (prediction_run.count >= MIN_REPEAT_COUNT[target]) {
		result.target = target;
		result.probability = prediction_run.probability_sum / prediction_run.count;
	} else if (publish_timeout_expired()) {
		result = most_frequent_prediction();
	} else {
		return result;
	}

	reset_run(target);
	reset_history();
	return result;
}

const char *inference_get_class_name(class_label_t class_label)
{
	if (class_label >= CLASS_LABEL_COUNT) {
		return "UNKNOWN";
	}

	return get_name_by_target((uint8_t)class_label);
}
