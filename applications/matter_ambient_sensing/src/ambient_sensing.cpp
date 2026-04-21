/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ambient_sensing.h"

#include <zephyr/kernel.h>
#include <zephyr/sys/__assert.h>
#include <nrf_edgeai/nrf_edgeai.h>

#include "models/all_models.h"

#if defined(CONFIG_AMBIENT_SENSING_MODEL_SNORING)
#define CONFIDENCE_THRESHOLD      0.9f
#define PREDICTION_NUM_IN_ROW     20
#define MAX_PREDICTION_NUM_IN_ROW 31
#define PRINT_RAW_PROBABILITY     0
#elif defined(CONFIG_AMBIENT_SENSING_MODEL_BABY_CRYING)
#define CONFIDENCE_THRESHOLD      0.996078f
#define PREDICTION_NUM_IN_ROW     3
#define MAX_PREDICTION_NUM_IN_ROW 31
#define PRINT_RAW_PROBABILITY     0
#elif defined(CONFIG_AMBIENT_SENSING_MODEL_DOG_BARKING)
#define CONFIDENCE_THRESHOLD      0.9f
#define PREDICTION_NUM_IN_ROW     10
#define MAX_PREDICTION_NUM_IN_ROW 31
#define PRINT_RAW_PROBABILITY     0
#elif defined(CONFIG_AMBIENT_SENSING_MODEL_CAT_MEOWING)
#define CONFIDENCE_THRESHOLD      0.95f
#define PREDICTION_NUM_IN_ROW     9
#define MAX_PREDICTION_NUM_IN_ROW 31
#define PRINT_RAW_PROBABILITY     0
#endif

namespace
{

#if defined(CONFIG_AMBIENT_SENSING_MODEL_SNORING)

bool snoring_detection_postprocessing(nrf_edgeai_t *p_model)
{
	// Rolling postprocessing state kept across calls.
	// - predicitons_history stores recent boolean detections as bits (newest at bit 0).
	// - prediction_count tracks how many `1` bits are currently in the window.
	static uint32_t prediction_count;
	static uint32_t predicitons_history;

	// Read model confidence for the currently predicted class.
	const uint16_t predicted_class = p_model->decoded_output.classif.predicted_class;
	const float probability =
		p_model->decoded_output.classif.probabilities.p_f32[predicted_class];

	// Convert probability to a binary detection for this frame.
	const bool detected = probability > CONFIDENCE_THRESHOLD;

	// Check the bit that will fall out of the window after the left shift.
	// MAX_PREDICTION_NUM_IN_ROW is the history bit-width limit used by this algorithm.
	const bool oldest_entry = (bool)(predicitons_history & BIT(MAX_PREDICTION_NUM_IN_ROW));

	// Update rolling count in O(1): add newest detection, remove oldest.
	prediction_count = prediction_count + detected - oldest_entry;
	// Shift history left and append current detection at LSB.
	predicitons_history = (predicitons_history << 1) | detected;

#if PRINT_RAW_PROBABILITY
	printk("Predictions count: %2u, probability: %0.3f\n", prediction_count, probability);
#endif

	if (prediction_count >= PREDICTION_NUM_IN_ROW) {
		// Enough positive frames accumulated: emit one detection event
		// and clear state to avoid repeated triggers from stale history.
		prediction_count = 0;
		predicitons_history = 0;

		return true;
	}

	return false;
}

#elif defined(CONFIG_AMBIENT_SENSING_MODEL_BABY_CRYING)

bool baby_crying_detection_postprocessing(nrf_edgeai_t *p_model)
{
	return false;
}

#elif defined(CONFIG_AMBIENT_SENSING_MODEL_DOG_BARKING)

bool dog_barking_detection_postprocessing(nrf_edgeai_t *p_model)
{
	return false;
}

#elif defined(CONFIG_AMBIENT_SENSING_MODEL_CAT_MEOWING)

bool cat_meowing_detection_postprocessing(nrf_edgeai_t *p_model)
{
	return false;
}

#endif
} // namespace

namespace Nrf
{
namespace AmbientSensing
{

bool process(nrf_edgeai_t *p_model)
{
#if defined(CONFIG_AMBIENT_SENSING_MODEL_SNORING)
	return snoring_detection_postprocessing(p_model);
#elif defined(CONFIG_AMBIENT_SENSING_MODEL_BABY_CRYING)
	return baby_crying_detection_postprocessing(p_model);
#elif defined(CONFIG_AMBIENT_SENSING_MODEL_DOG_BARKING)
	return dog_barking_detection_postprocessing(p_model);
#elif defined(CONFIG_AMBIENT_SENSING_MODEL_CAT_MEOWING)
	return cat_meowing_detection_postprocessing(p_model);
#endif
}
} // namespace AmbientSensing
} // namespace Nrf