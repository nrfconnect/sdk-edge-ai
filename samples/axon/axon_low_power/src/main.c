/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <errno.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <axon/nrf_axon_platform.h>

#include "generated/nrf_axon_model_okay_nordic.h"
#include "instrumentation.h"
#include "mel_test_vector.h"

LOG_MODULE_REGISTER(axon_low_power);

/*
 * Model input (from the compiled model):
 *   - Shape: batch 1, height 1, width 3 (time), 40 channels (mel bins), int8.
 *   - One inference needs 3 mel frames in a row; we slide by 1 frame per window.
 *   - Flat length is INPUT_SIZE == 3 * 40 int8 values (see FRAMES_PER_INPUT,
 *     MEL_BINS).
 *
 * quantize_mel_windows() fills input buffer for each window: it reads
 * mel_test_vector[][], reference mel features (float32 stored as int32),
 * quantizes to int8, and stores in CHW order (within each bin, the 3 time steps
 * are consecutive in the buffer).
 */
/** Consecutive mel frames along the model's width (time) axis. */
#define FRAMES_PER_INPUT 3
/** Mel bins per frame; must match MEL_TEST_VECTOR_MEL_BINS and model channel count. */
#define MEL_BINS	 MEL_TEST_VECTOR_MEL_BINS
/** Length of one int8 inference input: FRAMES_PER_INPUT bins-by-time slices (3*40=120). */
#define INPUT_SIZE	 (FRAMES_PER_INPUT * MEL_BINS)
/** Scalar model output for this neural network (single int8 model output value). */
#define OUTPUT_SIZE	 1
/** Rows in mel_test_vector: captured frames M000..M###. */
#define NUM_FRAMES	 MEL_TEST_VECTOR_NUM_FRAMES
/** Sliding windows over NUM_FRAMES with a window length of FRAMES_PER_INPUT. */
#define NUM_WINDOWS	 (NUM_FRAMES - FRAMES_PER_INPUT + 1)

BUILD_ASSERT(MEL_TEST_VECTOR_NUM_FRAMES >= FRAMES_PER_INPUT);
BUILD_ASSERT(NUM_WINDOWS > 0);
BUILD_ASSERT(OUTPUT_SIZE == 1);

/** Map Axon driver result to negative errno for main()/helpers */
static int axon_result_to_neg_errno(nrf_axon_result_e result)
{
	switch (result) {
	case NRF_AXON_RESULT_SUCCESS:
		return 0;
	case NRF_AXON_RESULT_NOT_FINISHED:
	case NRF_AXON_RESULT_EVENT_PENDING:
		return -EAGAIN;
	case NRF_AXON_RESULT_FAILURE_INVALID_LENGTH:
	case NRF_AXON_RESULT_FAILURE_MISALIGNED_BUFFER:
	case NRF_AXON_RESULT_FAILURE_INVALID_ROUNDING:
	case NRF_AXON_RESULT_NULL_BUFFER:
	case NRF_AXON_RESULT_FAILURE_MISSING_NULL_COEF:
	case NRF_AXON_RESULT_INVALID_CMD_BUF:
	case NRF_AXON_RESULT_INVALID_MODEL:
		return -EINVAL;
	case NRF_AXON_RESULT_BUFFER_TOO_SMALL:
		return -ENOBUFS;
	case NRF_AXON_RESULT_FAILURE_UNSUPPORTED_HARDWARE:
		return -ENODEV;
	case NRF_AXON_RESULT_MUTEX_FAILED:
		return -EBUSY;
	case NRF_AXON_RESULT_FAILURE_HARDWARE_ERROR:
	case NRF_AXON_RESULT_FAILURE:
		return -EIO;
	default:
		if (result < 0) {
			return -EIO;
		}
		return -EINVAL;
	}
}

static void dequantize(const int8_t *values, size_t length, float *target,
		       const nrf_axon_nn_compiled_model_s *model)
{
	const uint32_t deq_mult = model->output_dequant_mult;
	const uint8_t deq_round = model->output_dequant_round;
	const int8_t deq_zp = model->output_dequant_zp;

	for (size_t i = 0; i < length; i++) {
		target[i] = (values[i] - deq_zp) * ((float)deq_mult / (1 << deq_round));
	}
}

static const nrf_axon_nn_compiled_model_input_s *get_ww_input_desc(void)
{
	return &model_axon_user_instance_wakeword
			.inputs[model_axon_user_instance_wakeword.external_input_ndx];
}

static int init_axon_platform_and_model(void)
{
	nrf_axon_result_e result;
	int err;

	result = nrf_axon_platform_init();
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Axon NPU platform init failed (err %d)", result);
		return axon_result_to_neg_errno(result);
	}

	err = nrf_axon_nn_model_init_vars(&model_axon_user_instance_wakeword);
	if (err != 0) {
		LOG_ERR("Model persistent vars init failed (err %d)", err);
		return -EINVAL;
	}

	result = nrf_axon_nn_model_validate(&model_axon_user_instance_wakeword);
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Model validation failed (err %d)", result);
		return axon_result_to_neg_errno(result);
	}

	return 0;
}

/**
 * @brief Pre-quantize every sliding mel window into int8 CHW tensors for the NN model.
 *
 * Captured reference mel (mel_test_vector[]) is indexed as [frame][bin]:
 * - each row is one mel frame in time,
 * - each column one frequency bin.
 * Sliding window win uses FRAMES_PER_INPUT consecutive rows starting at row win. With loop index
 * frame=0..FRAMES_PER_INPUT-1, each value is read as mel_test_vector[win+frame][bin].
 *
 * The Axon NN model expects channel-first (CHW) bytes: for each mel bin, store all time steps in
 * the window back-to-back, then the next bin. The captured reference buffer mel_test_vector stays
 * [frame][bin]. For each window win, the function reads mel_test_vector[win+frame][bin] and writes
 * inputs[win] in bin-major order (outer bin loop, inner frame loop), quantizing each element.
 * Index into inputs[win][] for those loops is:
 *
 * @code
 * index = bin * FRAMES_PER_INPUT + frame
 * @endcode
 *
 * @verbatim
 * Example illustration: win=2, FRAMES_PER_INPUT=3, MEL_BINS=40. Rows count mel frames in capture
 * order, columns are mel bins.
 *
 *                 bin0   bin1   bin2  ... bin39
 *              .-------.------.------.  .-------.
 *        row 0 |       |      |      |  |       |
 *            : |   ... mel bins across one instant in time ...
 *        row 2 |%%%%%%%|%%%%%%|%%%%%%|  |%%%%%%%|  <- window start (win=2)
 *        row 3 |%%%%%%%|%%%%%%|%%%%%%|  |%%%%%%%|
 *        row 4 |%%%%%%%|%%%%%%|%%%%%%|  |%%%%%%%|  <- window end (win+2)
 *            : |       |      |      |  |       |
 *
 * Same slice packed channel-first into inputs[win] (MEL_BINS * FRAMES_PER_INPUT int8).
 * For each mel bin, the three values are three mel frames in the window.
 *
 *        [b0:f0][b0:f1][b0:f2]  [b1:f0][b1:f1][b1:f2]  ...  [b39:f0][b39:f1][b39:f2]
 *        \___________________/  \___________________/       \______________________/
 *              mel bin 0              mel bin 1                    mel bin 39
 *
 * fk is offset k inside sliding window win (same as inner loop index frame),
 * k=0..FRAMES_PER_INPUT-1; bN is mel bin N, N=0..MEL_BINS-1 (outer loop index bin).
 *
 * @endverbatim
 *
 * Each mel_test_vector cell is int32 storage holding float bits. Treat as float, map to int8 using
 * quantization parameters (quant_mult, quant_round, quant_zp), then clamp with qmin and qmax, and
 * cast:
 *
 * @code
 * scale = quant_mult / 2^quant_round
 * scaled_value = mel_float * scale + zp
 * @endcode
 *
 * @param[out] inputs Array of NUM_WINDOWS rows; each row inputs[win] has length INPUT_SIZE.
 * @param[in] input_desc NN model input descriptor (quantization fields).
 */
static void quantize_mel_windows(int8_t (*inputs)[INPUT_SIZE],
				 const nrf_axon_nn_compiled_model_input_s *input_desc)
{
	/* Quantization scale */
	const float scale = (float)input_desc->quant_mult / (float)(1u << input_desc->quant_round);
	/* Quantization zero point */
	const float zp = (float)input_desc->quant_zp;
	/* Quantization min/max values */
	const float qmax = (float)INT8_MAX;
	const float qmin = (float)INT8_MIN;

	for (int win = 0; win < NUM_WINDOWS; win++) {
		for (int bin = 0; bin < MEL_BINS; bin++) {
			for (int frame = 0; frame < FRAMES_PER_INPUT; frame++) {
				float mel_float;
				int32_t mel_raw = mel_test_vector[win + frame][bin];
				float scaled_value;

				memcpy(&mel_float, &mel_raw, sizeof(float));
				scaled_value = mel_float * scale + zp;

				if (scaled_value > qmax) {
					scaled_value = qmax;
				} else if (!(scaled_value >= qmin)) {
					/*
					 * Below int8 range, or NaN from bad float bits: mel_raw is
					 * reinterpreted as float (no validation).
					 * NaN >= qmin is false, so we clamp to int8 minimum.
					 */
					scaled_value = qmin;
				}
				inputs[win][bin * FRAMES_PER_INPUT + frame] =
					(int8_t)(int32_t)scaled_value;
			}
		}
	}
}

/**
 * @brief Log the per-window inference results for debugging purposes.
 *
 * @param[in] outputs Array of NUM_WINDOWS rows; each row outputs[win] has length OUTPUT_SIZE.
 * @param[in] model NN model descriptor (quantization fields).
 */
static void log_window_results(const int8_t (*outputs)[OUTPUT_SIZE],
			       const nrf_axon_nn_compiled_model_s *model)
{
	LOG_DBG("Per-window inference results:");
	for (int w = 0; w < NUM_WINDOWS; w++) {
		float prob[OUTPUT_SIZE];

		dequantize(outputs[w], OUTPUT_SIZE, &prob[0], model);
		LOG_DBG("  W%02d prediction: %.2f%% (raw output: %d)", w, (double)(prob[0] * 100),
			outputs[w][0]);
	}
}

static int run_window_sweep(int8_t (*inputs)[INPUT_SIZE], int8_t (*outputs)[OUTPUT_SIZE])
{
	inst_sweep_begin();
	for (int w = 0; w < NUM_WINDOWS; w++) {
		inst_infer_begin();
		nrf_axon_result_e result = nrf_axon_nn_model_infer_sync(
			&model_axon_user_instance_wakeword, inputs[w], outputs[w]);
		inst_infer_end();

		if (result != NRF_AXON_RESULT_SUCCESS) {
			LOG_ERR("Inference %d failed (err %d)", w, result);
			inst_sweep_end();
			return axon_result_to_neg_errno(result);
		}
	}
	inst_sweep_end();

	return 0;
}

static int run_inference_continuously(int8_t (*inputs)[INPUT_SIZE], int8_t (*outputs)[OUTPUT_SIZE])
{
	uint32_t iteration = 0;
	int err = 0;

	for (;;) {
		float prediction[OUTPUT_SIZE];

		LOG_INF("Running %d-window sweep (iteration %u)", NUM_WINDOWS, iteration);

		err = run_window_sweep(inputs, outputs);
		if (err != 0) {
			break;
		}

		dequantize(outputs[NUM_WINDOWS - 1], OUTPUT_SIZE, prediction,
			   &model_axon_user_instance_wakeword);

		log_window_results(outputs, &model_axon_user_instance_wakeword);

		iteration++;

		LOG_INF("Sleeping %d ms", CONFIG_AXON_LOW_POWER_SLEEP_BETWEEN_SWEEPS_MS);
		k_msleep(CONFIG_AXON_LOW_POWER_SLEEP_BETWEEN_SWEEPS_MS);
	}

	/* It should never reach here unless there is an error */
	return err;
}

int main(void)
{
	int err;
	static int8_t inputs[NUM_WINDOWS][INPUT_SIZE];
	static int8_t outputs[NUM_WINDOWS][OUTPUT_SIZE];

	LOG_INF("Model: %s", model_axon_user_instance_wakeword.model_name);
	LOG_INF("Input size: %d", INPUT_SIZE);
	LOG_INF("Output size: %d", OUTPUT_SIZE);
	LOG_INF("%d captured frames -> %d sliding windows", NUM_FRAMES, NUM_WINDOWS);

	err = inst_init();
	if (err != 0) {
		LOG_ERR("Instrumentation init failed (err %d)", err);
		return err;
	}

	err = init_axon_platform_and_model();
	if (err != 0) {
		return err;
	}

	LOG_INF("Axon NPU platform and model ready");

	const nrf_axon_nn_compiled_model_input_s *input_desc = get_ww_input_desc();

	LOG_INF("Quantization params: zp=%d, mult=%u, round=%u", input_desc->quant_zp,
		input_desc->quant_mult, input_desc->quant_round);

	quantize_mel_windows(inputs, input_desc);
	LOG_INF("Quantized %d input windows (CHW layout)", NUM_WINDOWS);

	return run_inference_continuously(inputs, outputs);
}
