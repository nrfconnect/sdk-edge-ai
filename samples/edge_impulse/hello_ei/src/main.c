/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/__assert.h>

#include "ei_classifier_types.h"
#include "model-parameters/model_metadata.h"

#include "ei_wrapper.h"
#include "input_data.h"

LOG_MODULE_REGISTER(hello_ei);

#define INPUT_WINDOW_SIZE    EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE
#define LABEL_COUNT          EI_CLASSIFIER_LABEL_COUNT

static struct {
	float data[CONFIG_HELLO_EI_DATA_BUF_SIZE];
	size_t current_index;
} sample_buffer;

static size_t inference_cnt = 0;

static void print_inference_result(const ei_impulse_result_t *result, int64_t duration)
{
	LOG_INF("=== Inference result ===");

	for (size_t i = 0; i < LABEL_COUNT; i++) {
		LOG_INF("%s => %.5f", result->classification[i].label,
		       result->classification[i].value);
	}

#if EI_CLASSIFIER_HAS_ANOMALY
	LOG_INF("Anomaly: %.5f", result->anomaly);
#endif

	LOG_INF("=== Inference time profiling ===");
	LOG_INF("Full inference completed in %lld ms", duration);
	LOG_INF("Classification completed in %d ms", result->timing.classification);
	LOG_INF("DSP operations completed in %d ms", result->timing.dsp);

#if EI_CLASSIFIER_HAS_ANOMALY
	LOG_INF("Anomaly detection completed in %d ms", result->timing.anomaly);
#endif
}

static void print_model_info(void)
{
	LOG_INF("=== Model info ===");
	LOG_INF("Input frame size: %u", EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME);
	LOG_INF("Input window size: %u", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
	LOG_INF("Input frequency: %u", EI_CLASSIFIER_FREQUENCY);
	LOG_INF("Label count: %u", EI_CLASSIFIER_LABEL_COUNT);
	LOG_INF("Has anomaly: %s", EI_CLASSIFIER_HAS_ANOMALY ? "yes" : "no");
}

static void clear_sample_buffer(void)
{
	memset(sample_buffer.data, 0, sizeof(sample_buffer.data));
	sample_buffer.current_index = 0;
}

static void collect_sample(float val)
{
	sample_buffer.data[sample_buffer.current_index] = val;
	sample_buffer.current_index++;
}

static int get_samples_from_buffer(size_t offset, size_t length, float *out_ptr)
{
	size_t buffer_size = ARRAY_SIZE(sample_buffer.data);
	size_t start_index = (inference_cnt + offset) % buffer_size;
	size_t end_index = start_index + length;

	if (end_index <= buffer_size) {
		// Data fits without wrapping
		memcpy(out_ptr, sample_buffer.data + start_index, length * sizeof(float));
	} else {
		// Data wraps around to the beginning of the buffer
		size_t first_part_len = buffer_size - start_index;
		size_t second_part_len = length - first_part_len;

		memcpy(out_ptr, sample_buffer.data + start_index, first_part_len * sizeof(float));
		memcpy(out_ptr + first_part_len, sample_buffer.data, second_part_len * sizeof(float));
	}

	return 0;
}

static int run_model(const float *input_data, size_t input_data_size)
{
	__ASSERT(input_data != NULL, "Input data pointer is NULL");
	__ASSERT(input_data_size >= INPUT_WINDOW_SIZE, "Not enough input data");

	size_t sample_cnt = 0;
	int64_t start_time, delta;
	ei_impulse_result_t inference_result;
	int err;

	clear_sample_buffer();
	inference_cnt = 0;

	while (sample_cnt < input_data_size) {
		collect_sample(input_data[sample_cnt]);
		sample_cnt++;

		// When there is enough data in the buffer, start running inference
		// in a sliding window fashion.
		if (sample_cnt >= INPUT_WINDOW_SIZE) {
			start_time = k_uptime_get();
			err = ei_wrapper_run_inference(&inference_result, INPUT_WINDOW_SIZE);
			delta = k_uptime_delta(&start_time);

			if (err != 0) {
				LOG_ERR("Inference failed with error code: %d", err);
				return -1;
			}

			print_inference_result(&inference_result, delta);

			// Keep track of how many inferences have been performed
			// to know the offset in the input data.
			inference_cnt++;
		}
	}

	LOG_INF("End of input data reached");
	return 0;
}

int main(void)
{
	print_model_info();

	ei_wrapper_init(&get_samples_from_buffer);

	LOG_INF("=== Running inference on sine wave input data ===");
	run_model(input_data_sine, ARRAY_SIZE(input_data_sine));

	LOG_INF("=== Running inference on triangle wave input data ===");
	run_model(input_data_triangle, ARRAY_SIZE(input_data_triangle));

	return 0;
}
