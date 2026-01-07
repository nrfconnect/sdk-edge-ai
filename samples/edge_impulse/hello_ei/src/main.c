/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include "edge-impulse-sdk/dsp/numpy_types.h"
#include "edge-impulse-sdk/porting/ei_classifier_porting.h"
#include "edge-impulse-sdk/classifier/ei_classifier_types.h"

#include "input_data.h"

#define LOG_MODULE_NAME hello_ei
LOG_MODULE_REGISTER(LOG_MODULE_NAME);

#define INPUT_WINDOW_SIZE    EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE

BUILD_ASSERT(INPUT_WINDOW_SIZE <= CONFIG_HELLO_EI_DATA_BUF_SIZE,
	"Size of data buffer is smaller than input window size");

static struct {
	float data[CONFIG_HELLO_EI_DATA_BUF_SIZE];
	size_t current_index;
} sample_buffer;

static size_t inference_cnt = 0;

extern const char* ei_classifier_inferencing_categories[];

extern EI_IMPULSE_ERROR run_classifier(signal_t *signal, ei_impulse_result_t *result, bool debug);

static void print_inference_result(const ei_impulse_result_t *result, int64_t duration)
{
	LOG_INF("=== Inference result ===");

	for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
		LOG_INF("%s => %.5f", result->classification[i].label,
			(double)result->classification[i].value);
	}

#if EI_CLASSIFIER_HAS_ANOMALY
	LOG_INF("anomaly: %.5f", (double)result->anomaly);
#endif

	LOG_INF("=== Inference time profiling ===");
	LOG_INF("Full inference completed in %lld ms", duration);
	LOG_INF("Classification completed in %d ms", result->timing.classification);
	LOG_INF("DSP operations completed in %d ms", result->timing.dsp);

#if EI_CLASSIFIER_HAS_ANOMALY
	LOG_INF("Anomaly detection completed in %d ms", result->timing.anomaly);
#endif

	LOG_INF("");
}

static void print_model_info(void)
{
	LOG_INF("=== Model info ===");
	LOG_INF("Input frame size: %d", EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME);
	LOG_INF("Input window size: %d", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
	LOG_INF("Input frequency: %d", EI_CLASSIFIER_FREQUENCY);
	LOG_INF("Label count: %d", EI_CLASSIFIER_LABEL_COUNT);
	LOG_INF("Labels:");
	for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
		LOG_INF("- %s", ei_classifier_inferencing_categories[i]);
	}
	LOG_INF("Has anomaly: %s", EI_CLASSIFIER_HAS_ANOMALY ? "yes" : "no");
	LOG_INF("");
}

static void clear_sample_buffer(void)
{
	memset(sample_buffer.data, 0, sizeof(sample_buffer.data));
	sample_buffer.current_index = 0;
}

static void collect_sample(float val)
{
	sample_buffer.data[sample_buffer.current_index] = val;

	/* Move to the next index in circular buffer. */
	if (sample_buffer.current_index < (CONFIG_HELLO_EI_DATA_BUF_SIZE - 1)) {
		sample_buffer.current_index++;
	} else {
		sample_buffer.current_index = 0;
	}
}

static int get_samples_from_buffer(size_t offset, size_t length, float *out_ptr)
{
	__ASSERT_NO_MSG(length <= CONFIG_HELLO_EI_DATA_BUF_SIZE);
	__ASSERT_NO_MSG(out_ptr != NULL);

	/* Calculate start and end indices in the circular buffer. */
	size_t buffer_size = ARRAY_SIZE(sample_buffer.data);
	size_t start_index = (inference_cnt + offset) % buffer_size;
	size_t end_index = start_index + length;

	if (end_index <= buffer_size) {
		/* Data fits without wrapping. */
		memcpy(out_ptr, sample_buffer.data + start_index,
		       length * sizeof(float));
	} else {
		/* Data wraps around to the beginning of the buffer. */
		size_t first_part_len = buffer_size - start_index;
		size_t second_part_len = length - first_part_len;

		memcpy(out_ptr, sample_buffer.data + start_index,
		       first_part_len * sizeof(float));
		memcpy(out_ptr + first_part_len, sample_buffer.data,
		       second_part_len * sizeof(float));
	}

	return 0;
}

static int run_ei_classification(ei_impulse_result_t *ei_result, size_t window_size)
{
	signal_t features_signal;

	features_signal.get_data = get_samples_from_buffer;
	features_signal.total_length = window_size;

	EI_IMPULSE_ERROR err = run_classifier(&features_signal, ei_result,
		IS_ENABLED(CONFIG_EI_WRAPPER_DEBUG_MODE));

	return err;
}

static int run_model(const float *input_data, size_t input_data_size)
{
	__ASSERT_NO_MSG(input_data != NULL);
	__ASSERT_NO_MSG(input_data_size >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);

	size_t sample_cnt = 0;
	int64_t start_time, delta;
	ei_impulse_result_t inference_result;
	int err;

	clear_sample_buffer();
	inference_cnt = 0;

	while (sample_cnt < input_data_size) {
		/* Collect new sample (in this case: from compiled input data). */
		collect_sample(input_data[sample_cnt]);
		sample_cnt++;

		/* When there is enough data in the buffer, start running inference
		   in a sliding window fashion. */
		if (sample_cnt >= INPUT_WINDOW_SIZE) {
			start_time = k_uptime_get();
			err = run_ei_classification(&inference_result, INPUT_WINDOW_SIZE);
			delta = k_uptime_delta(&start_time);

			if (err != 0) {
				LOG_ERR("Classification failed with error code: %d", err);
				return -1;
			}

			print_inference_result(&inference_result, delta);

			/* Keep track of how many inferences have been performed
			   to know the offset in the buffered data. */
			inference_cnt++;
		}
	}

	LOG_INF("End of input data reached");
	return 0;
}

int main(void)
{
	print_model_info();

	LOG_INF("Running inference on sine wave input data");
	run_model(input_data_sine, ARRAY_SIZE(input_data_sine));

	LOG_INF("");
	LOG_INF("Running inference on triangle wave input data");
	run_model(input_data_triangle, ARRAY_SIZE(input_data_triangle));

	return 0;
}
