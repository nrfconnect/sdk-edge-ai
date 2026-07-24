/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @file main.c
 * @brief Air Quality Prediction Regression Model
 *
 * This file implements a regression-based machine learning model for predicting air quality
 * levels based on gas sensor measurements and atmospheric conditions.
 *
 * @details
 * **Model Purpose:** Predict continuous air quality indices using multiple sensor inputs
 * for environmental monitoring and pollution prediction.
 *
 * **Input Features (9 total):**
 *   1. COGT - CO sensor reading (ppm or normalized units)
 *   2-6. PT08S1-5 - Five Pt-8 MOS metal oxide semiconductor sensors:
 *        - PT08S1: CO sensor
 *        - PT08S2: NMHC (non-methane hydrocarbons) sensor
 *        - PT08S3: NOx (nitrogen oxides) sensor
 *        - PT08S4: NO2 (nitrogen dioxide) sensor
 *        - PT08S5: O3 (ozone) sensor
 *   7. T - Temperature (Celsius)
 *   8. RH - Relative Humidity (%)
 *   9. AH - Absolute Humidity (kg/m³)
 *
 * **Output:** Single continuous value representing predicted air quality level
 *
 * **Applications:**
 *   - Environmental monitoring and air quality forecasting
 *   - Indoor air quality assessment
 *   - Pollution level prediction and public health alerts
 *   - Industrial emission monitoring
 *
 * **Validation Approach:**
 *   The model is validated against 29 test samples with known ground truth values.
 *   For each sample, the predicted value is compared with the expected value,
 *   and the absolute error is reported.
 */

#include "test_regression_data.h"

#include <nrf_edgeai/nrf_edgeai.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>
#include <math.h>
#include <stdio.h>

LOG_MODULE_REGISTER(multi_regress, LOG_LEVEL_INF);

nrf_edgeai_t *nrf_edgeai_user_model_90508(void);

#if defined(CONFIG_MODEL_OTA_NEUTON)
#include <model_ota/model_ota_neuton.h>
#include <zephyr/storage/flash_map.h>

MODEL_OTA_NEUTON_LOAD_DECL(90508);

BUILD_ASSERT(FIXED_PARTITION_EXISTS(model_regress_storage),
	     "board devicetree is missing model_regress_storage - see boards/*.overlay");
#endif

/**
 * @brief Extract and Prepare Input Features for Model Inference
 *
 * This function extracts sensor readings and environmental parameters from a test sample
 * and arranges them into a buffer suitable for feeding to the neural network model.
 * The feature order must match the model's expected input layer ordering.
 *
 * @param[out] p_buffer          Pointer to buffer where input features will be stored.
 *                              Must have capacity for at least 9 floating-point values.
 * @param[in]  buffer_size       Size of the output buffer (must equal USER_UNIQ_INPUTS_NUM = 9).
 * @param[in]  sample_index      Index of the test sample to extract from USER_INPUT_DATA array.
 *
 * @return Ground truth air quality value for this sample, used for validation/comparison.
 *
 * @details
 * Feature arrangement in the buffer:
 *   [0] COGT   - CO sensor reading
 *   [1] PT08S1 - CO MOS sensor
 *   [2] PT08S2 - NMHC MOS sensor
 *   [3] PT08S3 - NOx MOS sensor
 *   [4] PT08S4 - NO2 MOS sensor
 *   [5] PT08S5 - O3 MOS sensor
 *   [6] T      - Temperature (Celsius)
 *   [7] RH     - Relative Humidity (%)
 *   [8] AH     - Absolute Humidity (kg/m³)
 *
 * The assertion ensures buffer_size validation to prevent buffer overflows.
 */
static flt32_t fill_features_buffer(flt32_t *p_buffer, const size_t buffer_size,
				    const size_t sample_index)
{
	/* Verify buffer has correct capacity for all 9 input features */
	__ASSERT_NO_MSG(buffer_size == USER_UNIQ_INPUTS_NUM);

	/* Extract and arrange sensor readings into the input buffer */
	p_buffer[0] = USER_INPUT_DATA[sample_index].COGT;   /* CO sensor */
	p_buffer[1] = USER_INPUT_DATA[sample_index].PT08S1; /* PT08 CO sensor */
	p_buffer[2] = USER_INPUT_DATA[sample_index].PT08S2; /* PT08 NMHC sensor */
	p_buffer[3] = USER_INPUT_DATA[sample_index].PT08S3; /* PT08 NOx sensor */
	p_buffer[4] = USER_INPUT_DATA[sample_index].PT08S4; /* PT08 NO2 sensor */
	p_buffer[5] = USER_INPUT_DATA[sample_index].PT08S5; /* PT08 O3 sensor */
	p_buffer[6] = USER_INPUT_DATA[sample_index].T;	    /* Temperature */
	p_buffer[7] = USER_INPUT_DATA[sample_index].RH;	    /* Relative Humidity */
	p_buffer[8] = USER_INPUT_DATA[sample_index].AH;	    /* Absolute Humidity */

	/* Return ground truth value for later comparison with prediction */
	return USER_INPUT_DATA[sample_index].target;
}

/**
 * @brief Run Air Quality Regression Model Inference
 *
 * This function executes the neural network model to predict air quality based on
 * provided sensor inputs. The process follows the Edge AI inference pipeline:
 * 1. Feed sensor data into the model's input window
 * 2. Execute the neural network computation when the window is full
 * 3. Extract the continuous air quality prediction from the regression output
 *
 * @param[in] p_user_model      Pointer to the initialized Edge AI model instance.
 * @param[in] p_input_features  Array containing 9 sensor/environmental input values.
 * @param[in] features_num      Number of features in the input array (must be 9).
 *
 * @return Predicted air quality value as a floating-point number.
 *         Returns -1000.0 (invalid sentinel value) if inference fails or encounters errors.
 *
 * @details
 * **Inference Pipeline:**
 * - The model uses a 1-sample window (USER_WINDOW_SIZE=1), so inference occurs immediately
 *   after feeding a single sensor reading.
 * - Input features are processed through the neural network layers trained on historical
 *   air quality data.
 * - The output is a continuous value representing the predicted air quality level,
 *   suitable for direct comparison with ground truth measurements.
 *
 * **Error Handling:**
 * - Returns an invalid sentinel value (INVALID_PREDICTION_VALUE) if any step fails, allowing the
 * caller to detect and handle prediction failures gracefully.
 * - Assertions validate that the model outputs exactly 1 value as expected for this
 *   regression task.
 */
static flt32_t model_predict(nrf_edgeai_t *p_user_model, flt32_t *p_input_features,
			     size_t features_num)
{
	nrf_edgeai_err_t res;
	/* Invalid default value for error detection */
	flt32_t model_prediction = INVALID_PREDICTION_VALUE;

	/* Step 1: Feed sensor inputs into the model's preprocessing pipeline */
	res = nrf_edgeai_feed_inputs(p_user_model, p_input_features, features_num);

	if (res == NRF_EDGEAI_ERR_SUCCESS) {
		/* Step 2: Execute neural network inference on the accumulated window */
		/* With window size = 1, this occurs after every sample is fed */
		res = nrf_edgeai_run_inference(p_user_model);

		/* Step 3: Extract the regression output if inference was successful */
		if (res == NRF_EDGEAI_ERR_SUCCESS) {
			/* Access the regression model output (continuous value prediction) */
			const flt32_t *p_output = p_user_model->decoded_output.regression.p_outputs;

			__ASSERT_NO_MSG(p_user_model->decoded_output.regression.outputs_num ==
					USER_MODELS_OUTPUTS_NUM);

			/* Extract the single air quality prediction value */
			model_prediction = p_output[0];
		}
	}

	return model_prediction;
}

/**
 * @brief Air Quality Regression Model Validation Entry Point
 *
 * This function orchestrates the complete validation workflow for the air quality
 * prediction model. It initializes the neural network, runs inference on all 29 test
 * samples, and measures prediction accuracy against ground truth values.
 *
 * **Workflow:**
 * 1. Retrieve the pre-trained air quality regression model
 * 2. Validate model configuration matches expected parameters:
 *    - Input window size: 1 sample
 *    - Input features: 9 (sensors + environmental parameters)
 *    - Output values: 1 (air quality prediction)
 * 3. Initialize the Edge AI runtime for neural network execution
 * 4. For each of the 29 test samples:
 *    a. Extract sensor readings and environmental parameters
 *    b. Run model inference to get air quality prediction
 *    c. Compare prediction against ground truth value
 *    d. Calculate and display absolute error
 *
 * **Validation Metrics:**
 * The absolute error between predicted and expected values indicates model accuracy:
 * - Lower error = better predictions
 * - Can be averaged across all 29 samples to assess overall model performance
 *
 * **Sample Coverage:**
 * The 29 test samples span various environmental conditions:
 * - Temperature range: ~0°C to ~43°C
 * - Humidity range: ~15% to ~85% RH
 * - Various sensor readings covering clean to polluted air conditions
 * - Ensures model validation across realistic use cases
 *
 */
void run_regression_tests(void)
{
	/* Retrieve the generated neural network model for air quality prediction */
#if defined(CONFIG_MODEL_OTA_NEUTON)
	/* Model-only OTA: load the payload from its flash partition (XIP) at runtime. */
	nrf_edgeai_t *p_user_model = nrf_edgeai_load_user_model_90508(
		PARTITION_ID(model_regress_storage),
		(const uint8_t *)PARTITION_ADDRESS(model_regress_storage));

	if (p_user_model == NULL) {
		LOG_WRN("No valid regression model image in model_regress_storage - skipping (flash "
			"temp_regress_model_partition.hex)");
		return;
	}
#else
	nrf_edgeai_t *p_user_model = nrf_edgeai_user_model_90508();

	__ASSERT_NO_MSG(p_user_model != NULL);
#endif

	/* Validate model configuration: ensure the generated model matches expected parameters */
	__ASSERT_NO_MSG(nrf_edgeai_input_window_size(p_user_model) == USER_WINDOW_SIZE);
	__ASSERT_NO_MSG(nrf_edgeai_uniq_inputs_num(p_user_model) == USER_UNIQ_INPUTS_NUM);
	__ASSERT_NO_MSG(nrf_edgeai_model_outputs_num(p_user_model) == USER_MODELS_OUTPUTS_NUM);

	/* Initialize the Edge AI runtime to prepare the model for inference execution */
	nrf_edgeai_err_t res = nrf_edgeai_init(p_user_model);

	__ASSERT_NO_MSG(res == NRF_EDGEAI_ERR_SUCCESS);

	nrf_edgeai_rt_version_t v = nrf_edgeai_runtime_version();
	LOG_INF("nRF Edge AI runtime version: %d.%d.%d", v.field.major, v.field.minor,
		v.field.patch);

	/* Allocate buffer for holding the 9 input features before each inference */
	flt32_t input_features[USER_UNIQ_INPUTS_NUM];

	LOG_INF("--- Testing Model Air Quality predictions ---");
	if (p_user_model->model.type == NRF_EDGEAI_MODEL_AXON) {
		LOG_INF("Using Axon model");
	} else {
		LOG_INF("Using Neuton model");
	}
	/* Validation loop: test the model against all 29 sample data points */
	const size_t NUM_INPUT_SAMPLES = ARRAY_SIZE(USER_INPUT_DATA);

	for (size_t i = 0; i < NUM_INPUT_SAMPLES; i++) {
		/* Extract sensor readings and environmental parameters from test sample i */
		flt32_t ground_truth =
			fill_features_buffer(input_features, USER_UNIQ_INPUTS_NUM, i);

		/* Run neural network inference with the extracted features */
		flt32_t predicted_value =
			model_predict(p_user_model, input_features, USER_UNIQ_INPUTS_NUM);

		/* Calculate absolute error: magnitude of difference between prediction and truth */
		flt32_t abs_err = fabsf(predicted_value - ground_truth);

		__ASSERT_NO_MSG(abs_err <= EXPECTED_MODEL_MAE);

		/* Display results for this test sample */
		LOG_INF("Air quality - Predicted: %f, Expected: %f, absolute error %f",
			(double)predicted_value, (double)ground_truth, (double)abs_err);
	}
}
