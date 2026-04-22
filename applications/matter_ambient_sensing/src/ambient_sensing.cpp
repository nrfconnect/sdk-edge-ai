/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ambient_sensing.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/__assert.h>
#include <nrf_edgeai/nrf_edgeai.h>

#include "models/all_models.h"

LOG_MODULE_REGISTER(ambient_sensing, LOG_LEVEL_INF);

#if defined(CONFIG_AMBIENT_SENSING_MODEL_SNORING)
#define CONFIDENCE_THRESHOLD	  0.9f
#define PREDICTION_NUM_IN_ROW	  20
#define MAX_PREDICTION_NUM_IN_ROW 31
#define PRINT_RAW_PROBABILITY	  0
const char *MODEL_NAME = "Snoring";
#elif defined(CONFIG_AMBIENT_SENSING_MODEL_BABY_CRYING)
#define CONFIDENCE_THRESHOLD	  0.996078f
#define PREDICTION_NUM_IN_ROW	  3
#define MAX_PREDICTION_NUM_IN_ROW 31
#define PRINT_RAW_PROBABILITY	  0
const char *MODEL_NAME = "Baby Crying";
#elif defined(CONFIG_AMBIENT_SENSING_MODEL_DOG_BARKING)
#define CONFIDENCE_THRESHOLD	  0.9f
#define PREDICTION_NUM_IN_ROW	  10
#define MAX_PREDICTION_NUM_IN_ROW 31
#define PRINT_RAW_PROBABILITY	  0
const char *MODEL_NAME = "Dog Barking";
#elif defined(CONFIG_AMBIENT_SENSING_MODEL_CAT_MEOWING)
#define CONFIDENCE_THRESHOLD	  0.95f
#define PREDICTION_NUM_IN_ROW	  9
#define MAX_PREDICTION_NUM_IN_ROW 31
#define PRINT_RAW_PROBABILITY	  0
const char *MODEL_NAME = "Cat Meowing";
#endif

namespace Nrf
{
namespace AmbientSensing
{

const char *getModelName()
{
	return MODEL_NAME;
}

bool process(nrf_edgeai_t *p_model)
{
	// Rolling postprocessing state kept across calls.
	// - predictions_history stores recent boolean detections as bits (newest at bit 0).
	// - prediction_count tracks how many `1` bits are currently in the window.
	static uint32_t prediction_count;
	static uint32_t predictions_history;

	// Read model confidence for the currently predicted class.
	const uint16_t predicted_class = p_model->decoded_output.classif.predicted_class;
	const float probability =
		p_model->decoded_output.classif.probabilities.p_f32[predicted_class];

	// Convert probability to a binary detection for this frame.
	const bool detected = probability > CONFIDENCE_THRESHOLD;

	// Check the bit that will fall out of the window after the left shift.
	// MAX_PREDICTION_NUM_IN_ROW is the history bit-width limit used by this algorithm.
	const bool oldest_entry = (bool)(predictions_history & BIT(MAX_PREDICTION_NUM_IN_ROW));

	// Update rolling count in O(1): add newest detection, remove oldest.
	prediction_count = prediction_count + detected - oldest_entry;
	// Shift history left and append current detection at LSB.
	predictions_history = (predictions_history << 1) | detected;

#if PRINT_RAW_PROBABILITY
	LOG_DBG("Predictions count: %2u, probability: %0.3f", prediction_count,
		static_cast<double>(probability));
#endif

	if (prediction_count >= PREDICTION_NUM_IN_ROW) {
		// Enough positive frames accumulated: emit one detection event
		// and clear state to avoid repeated triggers from stale history.
		prediction_count = 0;
		predictions_history = 0;

		return true;
	}

	return false;
}
} // namespace AmbientSensing
} // namespace Nrf
