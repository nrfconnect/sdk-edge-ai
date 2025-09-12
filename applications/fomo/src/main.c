/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
#include "axon_driver.h"
#include "axon_nn_infer.h"
#include "axon_platform.h"
#include "axon_stringization.h"
#include "zephyr/sys/__assert.h"
#include "zephyr/sys/slist.h"

#include <assert.h>
#include <stddef.h>

#include <zephyr/device.h>
#include <zephyr/drivers/video-controls.h>
#include <zephyr/drivers/video.h>
#include <zephyr/drivers/video/arducam_mega.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/printk.h>

LOG_MODULE_REGISTER(fomo, LOG_LEVEL_DBG);

#define AXON_MODEL_FILE_NAME_ROOT              axon_model_
#define AXON_MODEL_LAYERS_FILE_NAME_ROOT       AXON_MODEL_FILE_NAME_ROOT
#define AXON_MODEL_TEST_VECTORS_FILE_NAME_ROOT AXON_MODEL_FILE_NAME_ROOT
#define AXON_MODEL_TEST_VECTORS_FILE_NAME_END  _test_vectors_.h
#define AXON_MODEL_LAYERS_FILE_NAME_TAIL       _layers_.h
#define AXON_MODEL_DOT_H                       _.h

#define AXON_MODEL_FILE_NAME                                                                       \
	STRINGIZE_3_CONCAT(AXON_MODEL_FILE_NAME_ROOT, AXON_MODEL_NAME, AXON_MODEL_DOT_H)
#define AXON_MODEL_FILE_LAYERS_NAME                                                                \
	STRINGIZE_3_CONCAT(AXON_MODEL_LAYERS_FILE_NAME_ROOT, AXON_MODEL_NAME,                      \
			   AXON_MODEL_LAYERS_FILE_NAME_TAIL)
#define AXON_MODEL_TEST_VECTORS_FILE_NAME                                                          \
	STRINGIZE_3_CONCAT(AXON_MODEL_TEST_VECTORS_FILE_NAME_ROOT, AXON_MODEL_NAME,                \
			   AXON_MODEL_TEST_VECTORS_FILE_NAME_END)

// generate structure name model_<model_name>
#define THE_REAL_MODEL_STRUCT_NAME(model_name) model_##model_name
#define THE_MODEL_STRUCT_NAME(model_name)      THE_REAL_MODEL_STRUCT_NAME(model_name)

#include AXON_MODEL_FILE_NAME
#if INCLUDE_VECTORS
#include AXON_MODEL_TEST_VECTORS_FILE_NAME
#if AXON_LAYER_TEST_VECTORS
#include AXON_MODEL_FILE_LAYERS_NAME
#endif
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))
#endif

float f_features[9216]; // 96x96x3
int8_t q_features[9216];
const struct device *video;

typedef struct cube {
	size_t x;
	size_t y;
	size_t width;
	size_t height;
	float confidence;
	int channel;
	sys_snode_t node; // for storing in a sys_slist_t
} cube_t;

/**
 * Checks whether a new section overlaps with a cube,
 * and if so, will **update the cube**
 */
static bool extend_cube(cube_t *c, int x, int y, int width, int height, float confidence)
{
	bool is_overlapping = !(c->x + c->width < x || c->y + c->height < y || c->x > x + width ||
				c->y > y + height);

	if (!is_overlapping) {
		return false;
	}

	// if we overlap, but the x of the new box is lower than the x of the current box
	if (x < c->x) {
		// update x to match new box and make width larger (by the diff between the boxes)
		c->x = x;
		c->width += c->x - x;
	}
	// if we overlap, but the y of the new box is lower than the y of the current box
	if (y < c->y) {
		// update y to match new box and make height larger (by the diff between the boxes)
		c->y = y;
		c->height += c->y - y;
	}
	// if we overlap, and x+width of the new box is higher than the x+width of the current box
	if (x + width > c->x + c->width) {
		// just make the box wider
		c->width += (x + width) - (c->x + c->width);
	}
	// if we overlap, and y+height of the new box is higher than the y+height of the current box
	if (y + height > c->y + c->height) {
		// just make the box higher
		c->height += (y + height) - (c->y + c->height);
	}
	// if the new box has higher confidence, then override confidence of the whole box
	if (confidence > c->confidence) {
		c->confidence = confidence;
	}

	return true;
}

static void handle_detection(sys_slist_t *cubes, int x, int y, float vf, int channel)
{

	bool is_overlapping = false;

	cube_t *c;
	SYS_SLIST_FOR_EACH_CONTAINER(cubes, c, node) {
		if (c->channel != channel) {
			continue;
		}

		if (extend_cube(c, x, y, 1, 1, vf)) {
			is_overlapping = true;
			break;
		}
	}

	if (!is_overlapping) {
		cube_t *cube = k_malloc(sizeof(cube_t));
		if (cube == NULL) {
			LOG_ERR("Failed to allocate memory for cube");
			return;
		}
		cube->x = x;
		cube->y = y;
		cube->width = 1;
		cube->height = 1;
		cube->confidence = vf;
		cube->channel = channel;
		sys_slist_append(cubes, &cube->node);
	}
}

static void extact_features(float *output, size_t output_size, const uint8_t *input,
			    size_t input_size)
{
	__ASSERT_NO_MSG(output != NULL);
	__ASSERT_NO_MSG(input != NULL);

	for (size_t i = 0, j = 0; i < input_size && j < output_size; i += 2) {
		uint16_t rgb565 = input[i] << 8 | input[i + 1];

		// rgb to 0..1
		float r = (float)(rgb565 >> 11 & 0x1F) / 31.0f;
		float g = (float)(rgb565 >> 5 & 0x3F) / 63.0f;
		float b = (float)(rgb565 >> 0 & 0x1F) / 31.0f;

		float v = (0.299f * r) + (0.587f * g) + (0.114f * b);
		output[j++] = v;
	}
}

static float dequantize(const axon_nn_compiled_model_struct *model, int8_t value)
{
	float f_value = (value - model->output_dequant_zp);
	f_value /= ((model->output_dequant_mult >> model->output_dequant_round) - 1);
	return f_value;
}

static void quantize_vector(int8_t *output, const float *vector, uint32_t vector_size,
			    uint32_t quant_mp, uint8_t quant_round, int8_t quant_zp)
{
	for (uint32_t i = 0; i < vector_size; i++) {
		output[i] = (int8_t)((uint32_t)(vector[i] * quant_mp) >> quant_round) + quant_zp;
	}
}

static void print_vector(const char *name, const int8_t *vector, uint32_t vector_size)
{
	LOG_INF("%s=[", name);
	for (uint32_t i = 0; i < vector_size; i++) {
		LOG_RAW("%4d, ", vector[i]);
		if (i % 16 == 15) {
			LOG_RAW("\n");
		}
	}
	LOG_RAW("\n]\n");
}

static void print_vector_f(const char *name, const float *vector, uint32_t vector_size)
{
	LOG_INF("%s=[", name);
	for (uint32_t i = 0; i < vector_size; i++) {
		LOG_RAW("%f, ", vector[i]);
		if (i % 16 == 15) {
			LOG_RAW("\n");
		}
	}
	LOG_RAW("\n]\n");
}

static void print_model_output(const char *name, const axon_nn_compiled_model_struct *model)
{
	int8_t height = model->output_dimensions.height;
	int8_t width = model->output_dimensions.width;
	int8_t channel_cnt = model->output_dimensions.channel_cnt;

	float f_value;
	int8_t value;

	LOG_INF("%s:", name);

	for (int8_t c = 0; c < channel_cnt; c++) {
		LOG_RAW("Channel %d:\n", c);

		for (int8_t y = 0; y < height; y++) {
			for (int8_t x = 0; x < width; x++) {
				value = model->output_ptr[height * width * c + y * width + x];
				f_value = dequantize(model, value);
				LOG_RAW("% .2f (% 4d) ", f_value, value);
			}
			LOG_RAW("\n");
		}

		LOG_RAW("\n");
	}
}

static int process_fomo_output(sys_slist_t *results, float treshold,
			       const axon_nn_compiled_model_struct *model)
{
	int8_t height = model->output_dimensions.height;
	int8_t width = model->output_dimensions.width;
	int8_t channel_cnt = model->output_dimensions.channel_cnt;
	uint8_t width_ratio = model->inputs[0].dimensions.width / model->output_dimensions.width;
	uint8_t height_ratio = model->inputs[0].dimensions.height / model->output_dimensions.height;

	for (int8_t c = 1; c < channel_cnt; c++) {
		for (int8_t y = 0; y < height; y++) {
			for (int8_t x = 0; x < width; x++) {
				int8_t value =
					model->output_ptr[height * width * c + y * width + x];
				float f_value = dequantize(model, value);
				if (f_value >= treshold) {
					handle_detection(results, x, y, f_value, c);
				}
			}
		}
	}

	cube_t *cube;
	SYS_SLIST_FOR_EACH_CONTAINER(results, cube, node) {
		cube->x *= width_ratio;
		cube->y *= height_ratio;
		cube->width *= width_ratio;
		cube->height *= height_ratio;
	}

	return sys_slist_len(results);
}

void take_picture(void)
{
	int err;
	enum video_frame_fragmented_status f_status;
	struct video_buffer *vbuf;

	float *f_features_pos = f_features;
	uint32_t f_size_left = ARRAY_SIZE(f_features);

	err = video_dequeue(video, &vbuf, K_FOREVER);
	if (err) {
		LOG_ERR("Unable to dequeue video buf (%d)", err);
		return;
	}

	f_status = vbuf->flags;

	extact_features(f_features_pos, f_size_left, vbuf->buffer, vbuf->bytesused);
	f_features_pos += vbuf->bytesused / 2;
	f_size_left -= vbuf->bytesused / 2;

	vbuf->type = VIDEO_BUF_TYPE_OUTPUT;
	video_enqueue(video, vbuf);
	while (f_status == VIDEO_BUF_FRAG) {
		video_dequeue(video, &vbuf, K_FOREVER);
		f_status = vbuf->flags;

		extact_features(f_features_pos, f_size_left, vbuf->buffer, vbuf->bytesused);
		f_features_pos += vbuf->bytesused / 2;
		f_size_left -= vbuf->bytesused / 2;

		vbuf->type = VIDEO_BUF_TYPE_OUTPUT;
		video_enqueue(video, vbuf);
	}

	return;
}

void run_inference(void)
{
	LOG_INF("Start Platform!");
	AxonResultEnum result = axon_platform_init();
	if (result != kAxonResultSuccess) {
		LOG_ERR("axon_platform_init failed!");
	}

	void *axon_handle = axon_driver_get_handle();
	LOG_INF("Prepare and run Axon!");

	const axon_nn_compiled_model_struct *model = &THE_MODEL_STRUCT_NAME(AXON_MODEL_NAME);
	axon_nn_model_inference_wrapper_struct wrapper;

	if (axon_nn_model_init(&wrapper, model) < 0) {
		LOG_ERR("axon_nn_model_init failed!");
	}

	while (true) {
		video_stream_start(video, VIDEO_BUF_TYPE_OUTPUT);
		take_picture();
		video_stream_stop(video, VIDEO_BUF_TYPE_OUTPUT);

		// print_vector_f("input_features", f_features, ARRAY_SIZE(f_features));

		quantize_vector(q_features, f_features, ARRAY_SIZE(f_features),
				model->inputs[0].quant_mult, model->inputs[0].quant_round,
				model->inputs[0].quant_zp);

		// print_vector("quantized_features", q_features, ARRAY_SIZE(q_features));

		uint32_t start = axon_platform_get_ticks();
		result = axon_nn_model_infer_sync(axon_handle, model, &(wrapper.cmd_buf_info),
						  q_features, ARRAY_SIZE(q_features));

		uint32_t time =
			(axon_platform_get_ticks() - start) * 1000 / axon_platform_get_clk_hz();
		// LOG_INF("axon_nn_model_infer_sync took %d ms", time);

		if (result != kAxonResultSuccess) {
			LOG_ERR("axon_nn_model_infer_sync failed!\n");
		} else {
			// print_model_output("output", model);

			sys_slist_t results;
			sys_slist_init(&results);

			process_fomo_output(&results, 0.5, model);

			LOG_RAW("\n");
			LOG_INF("Prediction results:");

			if (sys_slist_is_empty(&results)) {
				LOG_INF("no prediction");
			}

			cube_t *c, *nc;
			SYS_SLIST_FOR_EACH_CONTAINER_SAFE(&results, c, nc, node) {
				static const char *const labels[] = {"background", "beer", "can"};
				LOG_INF("%s (%.2f) [x: %d, y: %d, width: %d, height: %d]",
					labels[c->channel], c->confidence, c->x, c->y, c->width,
					c->height);
				sys_slist_remove(&results, NULL, &c->node);
				k_free(c);
			}
		}

		k_msleep(500);
	}

	LOG_INF("Inference complete");
	axon_platform_close();
}

int main(void)
{
	video = DEVICE_DT_GET(DT_NODELABEL(arducam_mega));

	if (!device_is_ready(video)) {
		LOG_ERR("Video device %s not ready.", video->name);
		return -1;
	}

	/* Alloc video buffers and enqueue for capture */
	struct video_buffer *buffers[2];
	for (int i = 0; i < ARRAY_SIZE(buffers); i++) {
		buffers[i] = video_buffer_alloc(1024, K_FOREVER);
		if (buffers[i] == NULL) {
			LOG_ERR("Unable to alloc video buffer");
			return -1;
		}
		buffers[i]->type = VIDEO_BUF_TYPE_OUTPUT;
		video_enqueue(video, buffers[i]);
	}

	run_inference();
	return 0;
}
