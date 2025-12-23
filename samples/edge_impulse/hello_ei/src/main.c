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

static float sample_buffer[CONFIG_HELLO_EI_DATA_BUF_SIZE];
static size_t sample_buffer_idx = 0;

static size_t inference_cnt = 0;

void print_inference_result(const ei_impulse_result_t *result, int64_t duration)
{
	LOG_INF("=== Inference result ===");

	for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
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

void print_model_info(void)
{
	LOG_INF("=== Model info ===");
	LOG_INF("Input frame size: %u", EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME);
	LOG_INF("Input window size: %u", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
	LOG_INF("Input frequency: %u", EI_CLASSIFIER_FREQUENCY);
	LOG_INF("Label count: %u", EI_CLASSIFIER_LABEL_COUNT);
	LOG_INF("Has anomaly: %s", EI_CLASSIFIER_HAS_ANOMALY ? "yes" : "no");
}

void collect_sample(float val)
{
	sample_buffer[sample_buffer_idx] = val;
	sample_buffer_idx++;
}

int get_sample(size_t offset, size_t length, float *out_ptr)
{
	if ((offset + length) <= ARRAY_SIZE(sample_buffer)) {
		memcpy(out_ptr, sample_buffer + inference_cnt + offset, length * sizeof(float));
		return 0;
	} else {
		return -1;
	}
}

int run_model(const float *input_data, size_t input_data_size)
{
	__ASSERT(input_data_size >= INPUT_WINDOW_SIZE, "Not enough input data");

	ei_wrapper_init(&get_sample);
	memset(sample_buffer, 0, sizeof(sample_buffer));
	sample_buffer_idx = 0;

	size_t sample_cnt = 0;
	int64_t start_time, delta;
	ei_impulse_result_t inference_result;
	int err;

	inference_cnt = 0;

	while (sample_cnt < input_data_size) {
		collect_sample(input_data[sample_cnt]);
		sample_cnt++;

		if (sample_cnt >= INPUT_WINDOW_SIZE) {
			start_time = k_uptime_get();
			err = ei_wrapper_run_inference(&inference_result, INPUT_WINDOW_SIZE);
			delta = k_uptime_delta(&start_time);

			if (err != 0) {
				LOG_ERR("Inference failed with error code: %d", err);
				return -1;
			}

			print_inference_result(&inference_result, delta);
			inference_cnt++;
		}
	}

	LOG_INF("End of input data reached");
	return 0;
}

int main(void)
{
	print_model_info();

	run_model(input_data_sine, ARRAY_SIZE(input_data_sine));
	run_model(input_data_triangle, ARRAY_SIZE(input_data_triangle));

	return 0;
}
