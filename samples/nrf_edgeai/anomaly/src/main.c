/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @file main.c
 * @brief Mechanical Gear Anomaly Detection Model using Vibration Analysis
 *
 * This file implements an anomaly detection neural network model for monitoring gear health
 * using multi-axis vibration data, enabling predictive maintenance and early fault detection.
 *
 * @details
 * **Model Purpose:** Detect abnormal vibration patterns indicating gear degradation, wear,
 * misalignment, or bearing failure using machine learning on dual-axis accelerometer data.
 *
 * **Input Features (2 total):**
 *   - Sensor 1: X-axis vibration (radial direction) in m/s^2
 *   - Sensor 2: Y-axis vibration (axial/tangential direction) in m/s^2
 *   - Data format: Interleaved pairs [X1, Y1, X2, Y2, ...] for streaming input
 *   - Window size: 128 consecutive samples per inference cycle
 *
 * **Output:**
 *   - Anomaly Score: Single floating-point value (0.0 to 1.0+)
 *     - Score < 0.000025: Normal operation (healthy gear, no maintenance needed)
 *     - Score >= 0.000025: Anomaly detected (schedule maintenance alert)
 *
 * **Vibration Characteristics:**
 *   - Healthy Gear: Regular, predictable signature (2.42-2.52 m/s^2, tight distribution)
 *     * Dominated by mesh frequency and harmonics
 *     * Stable periodic oscillation, minimal variation
 *     * No impulsive events or amplitude modulation
 *   - Faulty Gear: Erratic, irregular signature (2.40-2.56 m/s^2, wider spread)
 *     * Sidebands and modulation indicating tooth wear/crack
 *     * Random impulsive events from surface degradation
 *     * Non-stationary pattern indicating bearing failure or misalignment
 *
 * **Applications:**
 *   - Predictive maintenance for rotating machinery (pumps, motors, compressors)
 *   - Early fault detection before catastrophic failure occurs
 *   - Condition-based maintenance scheduling to reduce downtime and costs
 *   - Asset health monitoring across industrial facilities
 *   - Quality control and validation of newly manufactured gear units
 *
 * **Validation Approach:**
 *   The model is validated on two representative vibration datasets: one from a healthy
 *   gear (normal operation) and one from a faulty gear (degraded condition). The model
 *   must correctly distinguish between states to confirm proper anomaly detection.
 */

#include <nrf_edgeai/nrf_edgeai.h>
#include "nrf_edgeai_generated/nrf_edgeai_user_model.h"

#include <zephyr/kernel.h>
#include <stdio.h>
#include <assert.h>

/**
 * @brief Model Configuration Constants
 *
 * These constants define the shape and structure of the gear anomaly detector:
 * - USER_WINDOW_SIZE: Number of vibration samples accumulated before running inference.
 *   A window of 128 samples captures sufficient temporal detail in gear vibration patterns.
 * - USER_UNIQ_INPUTS_NUM: Total number of input features to the model.
 *   Set to 2 for the dual-axis accelerometer (X and Y vibration components).
 * - USER_ANOMALY_THRESHOLD: Classification threshold for anomaly detection.
 *   Scores above this threshold indicate abnormal gear operation requiring maintenance.
 */
#define USER_WINDOW_SIZE       128	 /* Samples per inference window */
#define USER_UNIQ_INPUTS_NUM   2	 /* 2 vibration sensors: axis X and Y */
#define USER_ANOMALY_THRESHOLD 0.000025f /* Anomaly score threshold, high sensitivity */
#define INVALID_ANOMALY_SCORE  10000.0f	 /* Invalid score for test validation */

/**
 *  Healthy Gear Vibration Baseline Data
 *
 * This dataset contains vibration measurements from a well-maintained mechanical gear
 * operating under normal conditions. Used to validate that the model correctly identifies
 * healthy gear operation with a low anomaly score.
 *
 * Data characteristics:
 *   - Sensor 1 (X-axis): Tightly clustered values around 2.42-2.52 m/s^2
 *   - Sensor 2 (Y-axis): Tightly clustered values around 2.42-2.52 m/s^2
 *   - Tight, consistent distribution across both axes
 *   - Regular, predictable oscillation pattern
 *   - Signal periodicity dominated by gear mesh frequency and harmonics
 *   - Stable baseline with no sudden spikes or amplitude modulation
 *
 * Data format:
 *   Interleaved pairs: [X1, Y1, X2, Y2, X3, Y3, ...]
 *   Total 256 values: 128 samples × 2 axes
 *
 * Expected model output:
 *   Anomaly score << USER_ANOMALY_THRESHOLD (well below threshold, indicating healthy state)
 *   Model confidence: High certainty of normal operation
 */
static const flt32_t GOOD_GEAR_MECH_VIBRATION_DATA_2AXIS[USER_WINDOW_SIZE *
							 USER_UNIQ_INPUTS_NUM] = {
	2.523464899, 2.43016754,  2.521493825, 2.430003284, 2.522479362, 2.429674773, 2.521329569,
	2.431810097, 2.522479362, 2.43131733,  2.519358495, 2.430496051, 2.521986593, 2.430660307,
	2.521001056, 2.431153074, 2.521001056, 2.431153074, 2.523957667, 2.431974352, 2.522150849,
	2.430003284, 2.521822337, 2.429839029, 2.524943204, 2.428853495, 2.522643618, 2.429839029,
	2.522479362, 2.430660307, 2.521658081, 2.429839029, 2.520015519, 2.430824563, 2.521001056,
	2.432138608, 2.521822337, 2.431810097, 2.520672544, 2.431645841, 2.521493825, 2.430988818,
	2.522315105, 2.43131733,  2.521986593, 2.431645841, 2.522315105, 2.430496051, 2.522643618,
	2.429510518, 2.522315105, 2.43016754,  2.521165312, 2.429674773, 2.522150849, 2.429510518,
	2.521822337, 2.429839029, 2.521329569, 2.43016754,  2.521493825, 2.429839029, 2.522479362,
	2.429346262, 2.521986593, 2.430660307, 2.521493825, 2.430988818, 2.520344032, 2.430331796,
	2.521165312, 2.429182006, 2.5208368,   2.431974352, 2.522479362, 2.431974352, 2.521329569,
	2.432631375, 2.521165312, 2.431153074, 2.522643618, 2.431810097, 2.523629155, 2.431153074,
	2.52297213,  2.430824563, 2.522479362, 2.430331796, 2.521493825, 2.430331796, 2.520344032,
	2.429510518, 2.522807874, 2.43131733,  2.523629155, 2.429839029, 2.521658081, 2.430824563,
	2.521658081, 2.430824563, 2.521001056, 2.429839029, 2.522479362, 2.430003284, 2.524121923,
	2.428360728, 2.523136386, 2.429510518, 2.521329569, 2.430331796, 2.520672544, 2.431974352,
	2.523629155, 2.431153074, 2.521822337, 2.431153074, 2.521165312, 2.431810097, 2.520508288,
	2.432631375, 2.5208368,	  2.432138608, 2.521329569, 2.431153074, 2.523136386, 2.431153074,
	2.523793411, 2.431645841, 2.520179775, 2.431153074, 2.520672544, 2.431810097, 2.524450436,
	2.429346262, 2.522150849, 2.428689239, 2.521493825, 2.429674773, 2.521493825, 2.429674773,
	2.522150849, 2.430660307, 2.522643618, 2.430496051, 2.522479362, 2.431153074, 2.521165312,
	2.431153074, 2.522807874, 2.429839029, 2.523300642, 2.430824563, 2.52297213,  2.429346262,
	2.522150849, 2.429017751, 2.522150849, 2.430988818, 2.523629155, 2.430496051, 2.523136386,
	2.430660307, 2.522150849, 2.430331796, 2.522479362, 2.430988818, 2.522643618, 2.431974352,
	2.522479362, 2.431153074, 2.522807874, 2.431974352, 2.521329569, 2.432631375, 2.520344032,
	2.431481585, 2.521822337, 2.430003284, 2.522643618, 2.43131733,	 2.523793411, 2.430496051,
	2.522315105, 2.430824563, 2.522807874, 2.430824563, 2.523629155, 2.429510518, 2.523464899,
	2.429182006, 2.521822337, 2.430496051, 2.521822337, 2.430824563, 2.521329569, 2.432302864,
	2.522315105, 2.432467119, 2.521822337, 2.432302864, 2.522315105, 2.432138608, 2.522150849,
	2.430824563, 2.524778948, 2.430003284, 2.522643618, 2.430003284, 2.522479362, 2.430988818,
	2.522807874, 2.432795631, 2.522479362, 2.430496051, 2.523300642, 2.430003284, 2.521658081,
	2.430003284, 2.5208368,	  2.43131733,  2.523629155, 2.431481585, 2.523300642, 2.430660307,
	2.522315105, 2.430660307, 2.522807874, 2.430824563, 2.522315105, 2.430660307, 2.521493825,
	2.430003284, 2.523793411, 2.43016754,  2.523300642, 2.430496051, 2.521493825, 2.430331796,
	2.521822337, 2.43016754,  2.521822337, 2.428853495, 2.523300642, 2.430331796, 2.522150849,
	2.43016754,  2.522315105, 2.431153074, 2.521658081, 2.432302864, 2.523136386, 2.430496051,
	2.521165312, 2.431481585, 2.5208368,   2.430988818,
};

/**
 * Anomalous Gear Vibration Data - Faulty Operation
 *
 * This dataset contains vibration measurements from a defective mechanical gear exhibiting
 * abnormal operating conditions. Used to validate that the model correctly identifies gear
 * faults with an anomaly score above the detection threshold.
 *
 * Data characteristics of defective gear:
 *   - Sensor 1 (X-axis): Wider range 2.40-2.56 m/s^2 (vs. healthy 2.42-2.52 m/s^2)
 *   - Sensor 2 (Y-axis): Wider range 2.40-2.56 m/s^2 (greater variation than healthy)
 *   - Erratic, unpredictable fluctuations across both axes
 *   - Irregular variation pattern with sudden amplitude changes
 *   - Modulation sidebands indicating gear tooth damage or misalignment
 *   - Random impulsive events with high-amplitude spikes (reaching 2.56+ m/s^2)
 *   - Non-stationary characteristics indicating evolving fault condition
 *
 * Root causes of this vibration signature:
 *   - Gear tooth crack or progressive wear
 *   - Bearing failure or severe misalignment
 *   - Gear mesh surface degradation and pitting
 *   - Lubrication failure or contamination
 *
 * Data format:
 *   Interleaved pairs: [X1, Y1, X2, Y2, X3, Y3, ...]
 *   Total 256 values: 128 samples × 2 axes
 *
 * Expected model output:
 *   Anomaly score >= USER_ANOMALY_THRESHOLD (at or above threshold, indicating fault)
 *   Model confidence: High certainty of abnormal operation
 *   Risk Level: High - maintenance or replacement required
 */
static const flt32_t ANOMALOUS_GEAR_MECH_VIBRATION_DATA_2AXIS[USER_WINDOW_SIZE *
							      USER_UNIQ_INPUTS_NUM] = {
	2.514595067, 2.435423722, 2.508846103, 2.433288398, 2.516730396, 2.438216068, 2.542682877,
	2.42638966,  2.520179775, 2.425732638, 2.532006221, 2.424582848, 2.51870147,  2.437723301,
	2.546625028, 2.418998157, 2.497019667, 2.43739479,  2.521493825, 2.428032217, 2.501290324,
	2.431810097, 2.532663246, 2.419326669, 2.523300642, 2.448892691, 2.498169459, 2.435423722,
	2.519687007, 2.437559046, 2.5208368,   2.44544332,  2.526257254, 2.442158206, 2.497676691,
	2.402736858, 2.538412214, 2.434602443, 2.514923579, 2.436573511, 2.530692171, 2.409471334,
	2.51360953,  2.421954758, 2.495377107, 2.423104548, 2.511309945, 2.470245947, 2.546460772,
	2.426553916, 2.495705619, 2.441008415, 2.52510746,  2.427703705, 2.512788249, 2.432795631,
	2.551716974, 2.439858625, 2.526750022, 2.406843246, 2.519851263, 2.441172671, 2.522479362,
	2.422776036, 2.491927731, 2.463511457, 2.549088872, 2.389924931, 2.490613683, 2.460390596,
	2.543668415, 2.404050902, 2.501947348, 2.442486717, 2.52987089,	 2.415877301, 2.508189079,
	2.44544332,  2.505889494, 2.433452653, 2.558451484, 2.422611781, 2.532498989, 2.433781165,
	2.543668415, 2.429346262, 2.531020683, 2.422119014, 2.513281018, 2.425896893, 2.540054777,
	2.427375194, 2.519687007, 2.456284201, 2.505889494, 2.429182006, 2.524614692, 2.418998157,
	2.520344032, 2.449221202, 2.541697339, 2.442815228, 2.528392584, 2.427210938, 2.5208368,
	2.420147947, 2.540876058, 2.450370993, 2.527407047, 2.405364946, 2.504082677, 2.415384535,
	2.514102298, 2.433781165, 2.509667384, 2.457926759, 2.53463432,	 2.426718172, 2.511638457,
	2.411606656, 2.515087835, 2.452834829, 2.537755189, 2.442322461, 2.545146721, 2.411935168,
	2.533648783, 2.42491136,  2.514923579, 2.418505391, 2.52987089,	 2.410128357, 2.521165312,
	2.423268803, 2.528228328, 2.44971397,  2.503589909, 2.431810097, 2.520015519, 2.425404126,
	2.500140532, 2.450042481, 2.527735559, 2.445936087, 2.49981202,	 2.44199395,  2.519358495,
	2.419819435, 2.53857647,  2.444457786, 2.560258304, 2.423104548, 2.504575445, 2.434109676,
	2.54563949,  2.431153074, 2.506382262, 2.436902023, 2.528064072, 2.428360728, 2.519194239,
	2.446428855, 2.498662227, 2.429839029, 2.530856427, 2.439037347, 2.512952506, 2.44544332,
	2.502604372, 2.45710548,  2.530527915, 2.424582848, 2.528064072, 2.415713046, 2.537590933,
	2.451520783, 2.541368827, 2.414891768, 2.500797556, 2.436573511, 2.52855684,  2.417519857,
	2.509667384, 2.435587977, 2.507039286, 2.424582848, 2.526585766, 2.426718172, 2.501618836,
	2.436245,    2.532827502, 2.433781165, 2.537590933, 2.428032217, 2.519194239, 2.436409256,
	2.52642151,  2.428360728, 2.554016562, 2.42228327,  2.509174615, 2.451028016, 2.522315105,
	2.434438188, 2.511145688, 2.432467119, 2.480594069, 2.423925826, 2.546296515, 2.427867961,
	2.506710774, 2.451356527, 2.513938042, 2.421954758, 2.552045486, 2.448892691, 2.507367798,
	2.419983691, 2.56633579,  2.428196472, 2.521658081, 2.433124142, 2.522643618, 2.436245,
	2.540547545, 2.436245,	  2.505232469, 2.428853495, 2.512623993, 2.431645841, 2.521001056,
	2.410456868, 2.513116762, 2.42638966,  2.538412214, 2.442322461, 2.529542378, 2.449549714,
	2.520344032, 2.425404126, 2.566007278, 2.426882427, 2.527899816, 2.432467119, 2.526585766,
	2.436409256, 2.52987089,  2.432467119, 2.480429813, 2.421954758, 2.537426676, 2.421297736,
	2.503918421, 2.430660307, 2.487328563, 2.430988818,
};

/** @brief Runs the trained neural network model on dual-axis vibration data.
 *
 * Parameters:
 *   @param[in] p_user_model: Pointer to pre-trained anomaly detection model
 *   @param[in] p_input_data: Array of interleaved [X, Y] acceleration pairs
 *   @param[in] data_len: Total number of data values (128 samples x 2 axes = 256)
 *
 * Process:
 *   1. Extract sensor pair (X, Y) for each sample
 *   2. Feed one pair at a time to simulate streaming sensor input
 *   3. Model accumulates 128 samples into its input window
 *   4. When window fills, inference automatically triggers
 *   5. Compute anomaly score based on learned feature representation
 *   6. Return single score indicating anomaly likelihood
 *
 * Return:
 *   Anomaly score (float): 0.0 = definitely healthy
 *   Interpretation:
 *     score < USER_ANOMALY_THRESHOLD -> Healthy gear (pass maintenance check)
 *     score >= USER_ANOMALY_THRESHOLD > Anomalous gear (schedule maintenance alert)
 */
static flt32_t model_predict(nrf_edgeai_t *p_user_model, const flt32_t *p_input_data,
			     size_t data_len)
{
	nrf_edgeai_err_t res;

	/* Initialize with high value (indicates failure) */
	flt32_t anomaly_score = INVALID_ANOMALY_SCORE;

	/* Feed vibration samples to Edge AI runtime in streaming mode */
	for (size_t i = 0; i < data_len; i += USER_UNIQ_INPUTS_NUM) {
		/* Extract one sample pair: [X_acceleration, Y_acceleration] */
		flt32_t input_sample[USER_UNIQ_INPUTS_NUM];
		input_sample[0] = p_input_data[i];     /* Sensor 1: X-axis */
		input_sample[1] = p_input_data[i + 1]; /* Sensor 2: Y-axis */

		/* Feed this sample pair into the model's windowing buffer */
		res = nrf_edgeai_feed_inputs(p_user_model, input_sample, 1 * USER_UNIQ_INPUTS_NUM);

		if (res == NRF_EDGEAI_ERR_SUCCESS) {
			/* Input buffer has reached 128 samples - run inference on the complete
			 * window
			 */
			res = nrf_edgeai_run_inference(p_user_model);

			/* Check that inference executed successfully */
			if (res == NRF_EDGEAI_ERR_SUCCESS) {
				/* Extract anomaly score from model's decoded output */
				anomaly_score = p_user_model->decoded_output.anomaly.score;
				break;
			}
		}
	}

	return anomaly_score;
}

/**
 * @brief Gear Anomaly Detection Model Validation Entry Point
 *
 * This function orchestrates the complete validation workflow for the gear anomaly
 * detection model. It initializes the neural network, runs inference on healthy and
 * faulty gear vibration data, and validates anomaly detection accuracy.
 *
 * **Workflow:**
 * 1. Retrieve the pre-trained gear anomaly detection model
 * 2. Validate model configuration matches expected parameters:
 *    - Input window size: 128 vibration samples
 *    - Input features: 2 (X and Y axis accelerometers)
 *    - Output values: 1 (anomaly score)
 * 3. Initialize the Edge AI runtime for neural network execution
 * 4. For each of the 2 test cases:
 *    a. Feed vibration data one sample pair at a time
 *    b. Model accumulates 128 samples into window
 *    c. Run model inference to compute anomaly score
 *    d. Compare score against threshold (USER_ANOMALY_THRESHOLD)
 *    e. Verify detection matches expected gear condition
 *
 * **Validation Metrics:**
 * The anomaly score indicates gear health status:
 * - Score < USER_ANOMALY_THRESHOLD: Normal operation (healthy gear, no maintenance needed)
 * - Score >= USER_ANOMALY_THRESHOLD: Anomaly detected (faulty gear, maintenance required)
 * - Correct classification validates model anomaly detection performance
 * - Model confidence determined by score magnitude relative to threshold
 *
 * **Test Coverage:**
 * The 2 test cases validate critical gear operating scenarios:
 * - Healthy gear: Regular, predictable vibration (2.42-2.52 m/s^2 tight distribution)
 * - Faulty gear: Erratic, irregular vibration (2.40-2.56 m/s^2, wider distribution)
 * - Covers the complete spectrum from normal to degraded gear condition
 * - Validates model suitability for predictive maintenance applications
 *
 * **Practical Application:**
 * Once validated, this model enables continuous monitoring of rotating machinery,
 * detecting early signs of gear wear, cracks, misalignment, or bearing failure,
 * triggering maintenance alerts before catastrophic failures occur.
 *
 */
int main(void)
{
	/*  Get user generated model pointer */
	nrf_edgeai_t *p_user_model = nrf_edgeai_user_model();
	assert(p_user_model != NULL);

	/* Validate that the loaded model matches our expected configuration */
	assert(nrf_edgeai_input_window_size(p_user_model) == USER_WINDOW_SIZE);
	assert(nrf_edgeai_uniq_inputs_num(p_user_model) == USER_UNIQ_INPUTS_NUM);

	/* Initialize Edge AI runtime for inference execution */
	nrf_edgeai_err_t res = nrf_edgeai_init(p_user_model);
	assert(res == NRF_EDGEAI_ERR_SUCCESS);

	flt32_t anomaly_score;
	const size_t DATA_LEN = USER_WINDOW_SIZE * USER_UNIQ_INPUTS_NUM;

	/* ---- TEST 1: Healthy Gear ---- */
	printk("\n--- Testing GOOD gear vibration data ---\r\n");
	printk("Expected: Low anomaly score (normal vibration pattern)\r\n");
	anomaly_score = model_predict(p_user_model, GOOD_GEAR_MECH_VIBRATION_DATA_2AXIS, DATA_LEN);

	printk("Anomaly score for GOOD gear data: %f\r\n", anomaly_score);
	printk("Verdict: %s (score %s threshold)\r\n",
	       anomaly_score < USER_ANOMALY_THRESHOLD ? "NORMAL" : "ANOMALY DETECTED",
	       anomaly_score < USER_ANOMALY_THRESHOLD ? "<" : ">=");
	assert(anomaly_score < USER_ANOMALY_THRESHOLD);

	/* ---- TEST 2: Faulty Gear ---- */
	printk("\n--- Testing ANOMALOUS gear vibration data ---\r\n");
	printk("Expected: High anomaly score (abnormal vibration pattern)\r\n");
	anomaly_score =
		model_predict(p_user_model, ANOMALOUS_GEAR_MECH_VIBRATION_DATA_2AXIS, DATA_LEN);

	printk("Anomaly score for ANOMALOUS gear data: %f\r\n", anomaly_score);
	printk("Verdict: %s (score %s threshold)\r\n",
	       anomaly_score < USER_ANOMALY_THRESHOLD ? "NORMAL" : "ANOMALY DETECTED",
	       anomaly_score >= USER_ANOMALY_THRESHOLD ? ">=" : "<");
	assert(anomaly_score >= USER_ANOMALY_THRESHOLD);

	printk("The model correctly distinguishes between healthy and faulty gears.\r\n");

	while (1) {
		printk("\n========== All test cases completed ==========\r\n");
		k_sleep(K_MSEC(5000));
	}

	return 0;
}
