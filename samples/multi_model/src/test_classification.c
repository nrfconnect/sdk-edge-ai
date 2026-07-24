/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @file test_classification.c
 * @brief Parcel State Classification Model using Acceleration Data
 *
 * This file implements a multi-class neural network model for classifying the state and
 * handling conditions of parcels during delivery using triaxial accelerometer data.
 *
 * @details
 * **Model Purpose:** Classify parcel delivery states into 7 distinct categories based on
 * acceleration magnitude patterns, enabling detection of rough handling and impact events.
 *
 * **Input Features (1 total):**
 *   - Acceleration Magnitude: sqrt(x^2 + y^2 + z^2) from triaxial accelerometer
 *     Removes directional bias, captures only motion intensity
 *   - Window size: 50 consecutive samples for temporal pattern analysis
 *
 * **Output Classes (7 total):**
 *   0. IDLE - Parcel at rest on surface (~1000 mG baseline gravity)
 *   1. SHAKING - Parcel vibrating or experiencing continuous motion (629-3505 mG range)
 *   2. IMPACT - Sudden collision or drop event (extreme spikes, 5900+ mG)
 *   3. FREE FALL - Parcel unsupported in air (~15-82 mG, near-zero gravity)
 *   4. CARRYING - Being held and transported by person (793-1350 mG, rhythmic pattern)
 *   5. IN_CAR - Inside vehicle during transport (920-1200 mG, smooth oscillation)
 *   6. PLACED - Active placement or lowering motion (drop then recovery pattern)
 *
 * **Applications:**
 *   - Detecting rough handling or potential damage to shipments
 *   - Monitoring delivery quality and logistics conditions
 *   - Alerting when parcels experience impacts or free falls
 *   - Tracking parcel state transitions throughout delivery lifecycle
 *
 * **Validation Approach:**
 *   The model is validated on 7 representative acceleration sequences (one per class),
 *   each containing real-world sensor data captured during that specific parcel state.
 *   Classification accuracy and confidence scores are reported for each test case.
 */

#include "test_classification_data.h"

#include <nrf_edgeai/nrf_edgeai.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <stdio.h>

LOG_MODULE_REGISTER(multi_classif, LOG_LEVEL_INF);

nrf_edgeai_t *nrf_edgeai_user_model_90449(void);

#if defined(CONFIG_MODEL_OTA_NEUTON)
#include <model_ota/model_ota_neuton.h>
#include <zephyr/storage/flash_map.h>

MODEL_OTA_NEUTON_LOAD_DECL(90449);

BUILD_ASSERT(FIXED_PARTITION_EXISTS(model_classif_storage),
	     "board devicetree is missing model_classif_storage - see boards/*.overlay");
#endif

/**
 *  @brief Runs the trained parcel state model on acceleration data.
 *
 * Parameters:
 *   @param[in] p_user_model: Pointer to initialized Edge AI model instance
 *   @param[in] p_input_data: Array of acceleration magnitude samples (sqrt(x^2+y^2+z^2))
 *   @param[in] data_len:     Total number of samples to process
 *
 * Process:
 *   1. Feed samples one-by-one (simulates real streaming sensor input)
 *   2. Model accumulates 50 samples into internal window
 *   3. When window is full, inference automatically triggers
 *   4. Extract predicted class and confidence probabilities
 *   5. Print results and return predicted class
 *
 * Return:
 *   Predicted class (0–6) on success, -1 if no prediction made
 */
static int32_t model_predict(nrf_edgeai_t *p_user_model, const flt32_t *p_input_data,
			     size_t data_len)
{
	nrf_edgeai_err_t res;

	/* Feed samples point-by-point to match real-world streaming scenario */
	for (size_t i = 0; i < data_len; i++) {
		/* Accumulate single sample into the model's input window */
		flt32_t input_sample = p_input_data[i];

		res = nrf_edgeai_feed_inputs(p_user_model, &input_sample, 1 * USER_UNIQ_INPUTS_NUM);

		if (res == NRF_EDGEAI_ERR_SUCCESS) {
			/* Window full—run inference on collected 50-sample window */
			res = nrf_edgeai_run_inference(p_user_model);

			/* Check if inference completed and was successful */
			if (res == NRF_EDGEAI_ERR_SUCCESS) {
				/* Extract results from model output */
				uint16_t predicted_class =
					p_user_model->decoded_output.classif.predicted_class;
				uint16_t num_classes =
					p_user_model->decoded_output.classif.num_classes;
				/* Confidence scores (probabilities) for all classes (f32, q16, q8)
				 */
				const flt32_t *p_probabilities =
					p_user_model->decoded_output.classif.probabilities.p_f32;
				/* Convert probability to percentage for easier interpretation */
				uint16_t prob_percent =
					(uint16_t)(p_probabilities[predicted_class] * 100.0f);

				LOG_INF("In %u classes, predicted %u with probability %u %%",
					num_classes, predicted_class, prob_percent);

				return predicted_class;
			}
		}
	}
	return -1;
}

/**
 * @brief Parcel State Detection Model Validation Entry Point
 *
 * This function orchestrates the complete validation workflow for the parcel state
 * classification model. It initializes the neural network, runs inference on 7 representative
 * test cases covering all parcel states, and validates classification accuracy.
 *
 * **Workflow:**
 * 1. Retrieve the pre-trained parcel state classification model
 * 2. Validate model configuration matches expected parameters:
 *    - Input window size: 50 acceleration samples
 *    - Input features: 1 (acceleration magnitude)
 *    - Output classes: 7 (IDLE, SHAKING, IMPACT, FREE_FALL, CARRYING, IN_CAR, PLACED)
 * 3. Initialize the Edge AI runtime for neural network execution
 * 4. For each of the 7 test cases:
 *    a. Feed acceleration data one sample at a time
 *    b. Model accumulates 50 samples into window
 *    c. Run model inference to classify parcel state
 *    d. Extract predicted class and confidence probability
 *    e. Verify prediction matches expected class
 *
 * **Validation Metrics:**
 * The predicted class and confidence score indicate model accuracy:
 * - Correct classification for all 7 parcel states validates model performance
 * - High confidence (probability) indicates strong prediction certainty
 * - Misclassification or low confidence indicates potential model issues
 *
 * **Test Coverage:**
 * The 7 test cases comprehensively cover all parcel delivery scenarios:
 * - Static conditions: IDLE (parcel at rest)
 * - Motion conditions: SHAKING, CARRYING, IN_CAR (various transport modes)
 * - Event conditions: IMPACT (collision), FREE_FALL (drop), PLACED (lowering)
 * - Acceleration ranges: from ~15 mG (free fall) to ~6900 mG (impact)
 * - Ensures model validation across complete delivery lifecycle
 *
 */
void run_classification_tests(void)
{
	/* Get user generated model pointer */
#if defined(CONFIG_MODEL_OTA_NEUTON)
	/* Model-only OTA: load the payload from its flash partition (XIP) at runtime. */
	nrf_edgeai_t *p_user_model = nrf_edgeai_load_user_model_90449(
		PARTITION_ID(model_classif_storage),
		(const uint8_t *)PARTITION_ADDRESS(model_classif_storage));

	if (p_user_model == NULL) {
		LOG_WRN("No valid classification model image in model_classif_storage - skipping "
			"(flash gesture_class_model_partition.hex)");
		return;
	}
#else
	nrf_edgeai_t *p_user_model = nrf_edgeai_user_model_90449();

	__ASSERT_NO_MSG(p_user_model != NULL);
#endif

	/** Validate model parameters against expected configuration */
	__ASSERT_NO_MSG(nrf_edgeai_input_window_size(p_user_model) == USER_WINDOW_SIZE);
	__ASSERT_NO_MSG(nrf_edgeai_uniq_inputs_num(p_user_model) == USER_UNIQ_INPUTS_NUM);
	__ASSERT_NO_MSG(nrf_edgeai_model_outputs_num(p_user_model) == USER_MODELS_CLASS_NUM);

	/** Initialize Edge AI runtime for inference execution */
	nrf_edgeai_err_t res = nrf_edgeai_init(p_user_model);

	__ASSERT_NO_MSG(res == NRF_EDGEAI_ERR_SUCCESS);

	nrf_edgeai_rt_version_t v = nrf_edgeai_runtime_version();
	LOG_INF("nRF Edge AI runtime version: %d.%d.%d", v.field.major, v.field.minor,
		v.field.patch);

	if (p_user_model->model.type == NRF_EDGEAI_MODEL_AXON) {
		LOG_INF("Using Axon model");
	} else {
		LOG_INF("Using Neuton model");
	}

	int32_t predicted_class;
	const size_t DATA_LEN = USER_WINDOW_SIZE * USER_UNIQ_INPUTS_NUM;

	/** TEST 1: Predict class 0 - Parcel in the IDLE state */
	LOG_INF("--- Testing IDLE state (parcel at rest) ---");
	predicted_class = model_predict(p_user_model, CLASS_0_PARCEL_IDLE_ACCEL_DATA, DATA_LEN);

	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_IDLE);
	LOG_INF("Expected class IDLE - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);

	/** TEST 2: Predict class 1 - Parcel is SHAKING */
	LOG_INF("--- Testing SHAKING state (parcel vibrating) ---");
	predicted_class = model_predict(p_user_model, CLASS_1_PARCEL_SHAKING_ACCEL_DATA, DATA_LEN);

	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_SHAKING);
	LOG_INF("Expected class SHAKING - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);

	/** TEST 3: Predict class 2 - Parcel IMPACT event */
	LOG_INF("--- Testing IMPACT event (collision detected) ---");
	predicted_class = model_predict(p_user_model, CLASS_2_PARCEL_IMPACT_ACCEL_DATA, DATA_LEN);

	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_IMPACT);
	LOG_INF("Expected class IMPACT - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);

	/** TEST 4: Predict class 3 - Parcel FREE FALL event */
	LOG_INF("--- Testing FREE FALL event (parcel in air/unsupported) ---");
	predicted_class =
		model_predict(p_user_model, CLASS_3_PARCEL_FREE_FALL_ACCEL_DATA, DATA_LEN);

	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_FREE_FALL);
	LOG_INF("Expected class FREE FALL - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);

	/** TEST 5: Predict class 4 - Parcel TRANSPORTED BY COURIER */
	LOG_INF("--- Testing CARRYING (person carrying) ---");
	predicted_class = model_predict(p_user_model, CLASS_4_PARCEL_CARRYING_ACCEL_DATA, DATA_LEN);

	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_CARRYING);
	LOG_INF("Expected class CARRYING - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);

	/** TEST 6: Predict class 5 - Parcel IN CAR */
	LOG_INF("--- Testing IN CAR state (vehicle transport) ---");
	predicted_class = model_predict(p_user_model, CLASS_5_PARCEL_IN_CAR_ACCEL_DATA, DATA_LEN);

	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_IN_CAR);
	LOG_INF("Expected class IN CAR - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);

	/** TEST 7: Predict class 6 - Parcel PLACED */
	LOG_INF("--- Testing PLACED state (active placement event) ---");
	predicted_class = model_predict(p_user_model, CLASS_6_PARCEL_PLACED_ACCEL_DATA, DATA_LEN);

	__ASSERT_NO_MSG(predicted_class == MODEL_CLASS_PLACED);
	LOG_INF("Expected class PLACED - predicted %s", USER_MODEL_LABELS_STR[predicted_class]);
}
