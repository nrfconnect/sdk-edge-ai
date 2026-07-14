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
 *
 * **Model-only OTA update:**
 *   By default (CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA=y) the application image does not
 *   contain a model: at boot (and periodically thereafter) it loads and validates a "model
 *   package" from the model_storage flash partition (see lib/model_ota/model_pkg_neuton.c /
 *   model_pkg_axon.c), and only then runs inference against it. Flashing a new model package
 *   to model_storage - independently of the application binary - is enough to change what the
 *   device predicts, without rebuilding or reflashing the application. If model_storage does
 *   not currently hold a valid package (missing, corrupted, incompatible, wrong backend), the
 *   application stays alive and simply skips inference instead of crashing. See README.rst for
 *   the packaging/flashing workflow. Build with CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA=n to
 *   restore this sample's original behavior instead: the model is compiled directly into the
 *   image and validated once at boot.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>
#include <math.h>
#include <stdio.h>

LOG_MODULE_REGISTER(regression, LOG_LEVEL_INF);

#include <nrf_edgeai/nrf_edgeai.h>

#include "model_wiring.h"

#if defined(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA)

#include <zephyr/storage/flash_map.h>

/*
 * Fail the build with a clear message if this board's devicetree overlay doesn't define the
 * model_partition node model_ota loads from (see the overlays under boards/), instead of a
 * much less obvious error deep inside a flash_map.h macro expansion.
 */
BUILD_ASSERT(FIXED_PARTITION_EXISTS(model_partition),
	     "board devicetree is missing the model_partition node - see boards/*.overlay");

#endif /* CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA */

/**
 * @brief Model Configuration Constants
 *
 * These constants define the shape and size of the neural network model:
 * - USER_UNIQ_INPUTS_NUM: Total number of input features to the model.
 *   Corresponds to: COGT (1) + PT08S sensors (5) + T, RH, AH (3) = 9 features
 * - USER_MODELS_OUTPUTS_NUM: Number of output values from the regression model.
 *   Set to 1 for single air quality prediction output.
 */
static const size_t USER_UNIQ_INPUTS_NUM = 9;	 /* Gas sensor and environmental input features */
static const size_t USER_MODELS_OUTPUTS_NUM = 1; /* Single air quality prediction output */
static const flt32_t INVALID_PREDICTION_VALUE = -9999.0f; /* Invalid prediction indicator */
#if !defined(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA)
/* Only meaningful for the compiled-in model: an OTA-loaded model is expected to change what the
 * device predicts, so its accuracy is only ever logged, never asserted on (see
 * run_inference_loop() below).
 */
static const flt32_t EXPECTED_MODEL_MAE = 2.0f; /* Expected Mean Absolute Error for validation */
#endif

/**
 * @brief Test Dataset Structure and Values
 *
 * This structure defines the schema for test samples:
 * - COGT: CO sensor (main carbon monoxide detector)
 * - PT08S1-5: Five metal oxide semiconductor (MOS) sensors:
 *   * PT08S1: CO sensor output
 *   * PT08S2: NMHC (non-methane hydrocarbons) sensor
 *   * PT08S3: NOx (nitrogen oxides) sensor
 *   * PT08S4: NO2 (nitrogen dioxide) sensor
 *   * PT08S5: O3 (ozone) sensor
 * - T: Temperature in Celsius (affects pollutant concentration and sensor response)
 * - RH: Relative Humidity in percentage (influences sensor readings and pollutant behavior)
 * - AH: Absolute Humidity in kg/m³ (moisture content in air)
 * - target: Ground truth air quality value (used for validation)
 *
 * @note All sensor values are normalized or raw readings from the measurement device.
 * The 29 test samples cover various environmental conditions to validate model performance
 * across different temperature, humidity, and pollution levels.
 */
struct {
	flt32_t COGT;	/**< CO sensor reading */
	flt32_t PT08S1; /**< PT08 MOS sensor 1 (CO) */
	flt32_t PT08S2; /**< PT08 MOS sensor 2 (NMHC) */
	flt32_t PT08S3; /**< PT08 MOS sensor 3 (NOx) */
	flt32_t PT08S4; /**< PT08 MOS sensor 4 (NO2) */
	flt32_t PT08S5; /**< PT08 MOS sensor 5 (O3) */
	flt32_t T;	/**< Temperature in Celsius */
	flt32_t RH;	/**< Relative Humidity in percentage */
	flt32_t AH;	/**< Absolute Humidity in kg/m³ */
	flt32_t target; /**< Ground truth air quality value */
} static const USER_INPUT_DATA[] = {
	[0] = {2.7, 1146, 1125, 846, 1511, 1016, 33.4, 20.3, 1.027, 14.3},
	[1] = {3.3, 1272, 1328, 567, 2085, 1463, 19.6, 54.3, 1.2278, 21.1},
	[2] = {0.6, 919, 571, 1017, 1082, 521, 13.5, 63.8, 0.9801, 1.8},
	[3] = {1.7, 1162, 1019, 622, 1904, 1178, 27.9, 56.5, 2.0895, 11.1},
	[4] = {2.7, 1381, 1227, 595, 1903, 1845, 29.8, 33.1, 1.3693, 17.6},
	[5] = {1.1, 1010, 679, 854, 1046, 889, 5, 82.9, 0.73, 3.4},
	[6] = {1.9, 1055, 1037, 635, 1632, 1161, 25.3, 41.7, 1.3247, 11.6},
	[7] = {3.4, 1417, 1303, 635, 1964, 1752, 20.7, 40, 0.9639, 20.2},
	[8] = {2, 1273, 988, 563, 1387, 1257, 24.1, 33.3, 0.9881, 10.3},
	[9] = {4.8, 1435, 1429, 499, 2072, 1449, 21.7, 59.1, 1.5115, 25},
	[10] = {1, 1003, 635, 829, 1235, 711, 13.7, 80.6, 1.2528, 2.7},
	[11] = {1.3, 1259, 1152, 610, 1165, 1569, 8.2, 36.8, 0.4016, 15.1},
	[12] = {2.7, 1229, 1114, 598, 1723, 1247, 19.7, 0.73, 0.6708, 13.9},
	[13] = {2.8, 1261, 1258, 629, 1813, 1315, 43.4, 14.8, 1.2882, 18.6},
	[14] = {1.3, 997, 752, 837, 952, 724, 11.8, 34.4, 0.4758, 4.8},
	[15] = {1.3, 942, 846, 980, 1615, 905, 21.7, 50.8, 1.3019, 6.7},
	[16] = {0.6, 883, 518, 1135, 962, 606, 3.3, 84.5, 0.6612, 1.2},
	[17] = {3.2, 1174, 1264, 670, 1598, 1287, 28.9, 21.3, 0.8382, 18.8},
	[18] = {0.5, 748, 595, 1208, 1089, 677, 12.8, 0.598, 0.8801, 2.2},
	[19] = {1.4, 920, 783, 1046, 1550, 588, 24.5, 0.384, 1.1669, 5.4},
	[20] = {1.3, 906, 790, 893, 837, 642, 12.3, 19.3, 0.274, 5.5},
	[21] = {0.6, 882, 563, 978, 936, 660, 8.9, 0.527, 0.6034, 1.7},
	[22] = {3.4, 1403, 1443, 508, 2234, 1811, 25.2, 0.386, 1.2215, 25.6},
	[23] = {3.9, 1297, 1102, 507, 1375, 1583, 18.2, 0.363, 0.7487, 13.6},
	[24] = {1.3, 987, 800, 989, 1462, 658, 15.5, 0.571, 0.996, 5.7},
	[25] = {1.3, 869, 866, 1107, 1212, 596, 0.238, 0.145, 0.4222, 7.2},
	[26] = {3.2, 1336, 1340, 540, 2049, 1400, 21.3, 0.635, 1.5941, 21.6},
	[27] = {1, 955, 723, 1129, 1393, 559, 27.7, 0.258, 0.9467, 4.2},
	[28] = {4.3, 1373, 1364, 597, 2005, 1745, 33.7, 0.226, 1.1658, 22.5}};

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
 * provided sensor inputs.
 *
 * @param[in] p_user_model      Pointer to the initialized Edge AI model instance.
 * @param[in] p_input_features  Array containing 9 sensor/environmental input values.
 * @param[in] features_num      Number of features in the input array (must be 9).
 *
 * @return Predicted air quality value as a floating-point number.
 *         Returns -9999.0 (invalid sentinel value) if inference fails or encounters errors.
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
 * @brief Run one full validation pass of the model against all 29 test samples.
 *
 * For each sample, feeds the sensor readings into the model, runs inference, and logs the
 * predicted value alongside the ground truth and absolute error. With a compiled-in model
 * (CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA=n) accuracy is also asserted on; with an OTA-loaded
 * model it is only ever logged, since a freshly flashed model package is expected to change
 * what the device predicts.
 */
static void run_inference_loop(nrf_edgeai_t *p_user_model)
{
	nrf_edgeai_err_t res = nrf_edgeai_init(p_user_model);

	__ASSERT_NO_MSG(res == NRF_EDGEAI_ERR_SUCCESS);

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

#if !defined(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA)
		__ASSERT_NO_MSG(abs_err <= EXPECTED_MODEL_MAE);
#endif

		/* Display results for this test sample */
		LOG_INF("Air quality - Predicted: %f, Expected: %f, absolute error %f",
			(double)predicted_value, (double)ground_truth, (double)abs_err);
	}

	LOG_INF("========== All test cases completed ==========");
}

#if defined(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA)

int main(void)
{
	nrf_edgeai_rt_version_t v = nrf_edgeai_runtime_version();

	LOG_INF("nRF Edge AI runtime version: %d.%d.%d", v.field.major, v.field.minor,
		v.field.patch);

	while (1) {
		/* The model is not compiled in: load (and validate) it from the model_storage
		 * flash partition every iteration, so a model update flashed while the device
		 * is running is picked up without needing a reboot.
		 */
		nrf_edgeai_t *p_user_model = model_ota_load();

		if (p_user_model == NULL) {
			LOG_WRN("No valid model in model_storage - waiting for one to be "
				"flashed. Inference is skipped until then.");
		} else {
			run_inference_loop(p_user_model);
		}

		k_sleep(K_MSEC(5000));
	}

	return 0;
}

#else /* !CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA: compiled-in model, validated once at boot -
       * this sample's original pre-model-OTA behavior.
       */

int main(void)
{
	nrf_edgeai_rt_version_t v = nrf_edgeai_runtime_version();

	LOG_INF("nRF Edge AI runtime version: %d.%d.%d", v.field.major, v.field.minor,
		v.field.patch);

	/* The model is compiled directly into the image (see model_wiring_*.c): validate it
	 * once against all 29 test cases, asserting on the expected accuracy, then idle.
	 */
	run_inference_loop(model_ota_load());

	while (1) {
		LOG_INF("========== All test cases completed ==========");
		k_sleep(K_MSEC(5000));
	}

	return 0;
}

#endif /* CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA */
