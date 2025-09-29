/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */

#include <zephyr/device.h>
#include <zephyr/drivers/video.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/slist.h>

#include <axon_driver.h>
#include <axon_nn_infer.h>
#include <axon_platform.h>
#include <axon_stringization.h>

LOG_MODULE_REGISTER(fomo, LOG_LEVEL_DBG);

#define AXON_MODEL_FILE_NAME_ROOT        axon_model_
#define AXON_MODEL_LAYERS_FILE_NAME_TAIL _layers_.h
#define AXON_MODEL_DOT_H                 _.h

#define AXON_MODEL_FILE_NAME                                                                       \
	STRINGIZE_3_CONCAT(AXON_MODEL_FILE_NAME_ROOT, AXON_MODEL_NAME, AXON_MODEL_DOT_H)

// generate structure name model_<model_name>
#define THE_REAL_MODEL_STRUCT_NAME(model_name) model_##model_name
#define THE_MODEL_STRUCT_NAME(model_name)      THE_REAL_MODEL_STRUCT_NAME(model_name)

#include AXON_MODEL_FILE_NAME

float f_features[9216]; // 96x96
int8_t q_features[9216];
const struct device *video;

struct box {
	size_t x;
	size_t y;
	size_t width;
	size_t height;
	float confidence;
	int channel;
	sys_snode_t node; // for storing in a sys_slist_t
};

struct record {
	uint8_t present_count;
	uint8_t missing_count;
	struct box box;
	sys_snode_t node; // for storing in a sys_slist_t
};

/**
 * Checks whether a new section overlaps with a box,
 * and if so, will **update the box**
 */
static bool extend_box(struct box *box, int x, int y, int width, int height, float confidence)
{
	bool is_overlapping = !(box->x + box->width < x || box->y + box->height < y ||
				box->x > x + width || box->y > y + height);

	if (!is_overlapping) {
		return false;
	}

	// if we overlap, but the x of the new box is lower than the x of the current box
	if (x < box->x) {
		// update x to match new box and make width larger (by the diff between the boxes)
		box->x = x;
		box->width += box->x - x;
	}
	// if we overlap, but the y of the new box is lower than the y of the current box
	if (y < box->y) {
		// update y to match new box and make height larger (by the diff between the boxes)
		box->y = y;
		box->height += box->y - y;
	}
	// if we overlap, and x+width of the new box is higher than the x+width of the current box
	if (x + width > box->x + box->width) {
		// just make the box wider
		box->width += (x + width) - (box->x + box->width);
	}
	// if we overlap, and y+height of the new box is higher than the y+height of the current box
	if (y + height > box->y + box->height) {
		// just make the box higher
		box->height += (y + height) - (box->y + box->height);
	}
	// if the new box has higher confidence, then override confidence of the whole box
	if (confidence > box->confidence) {
		box->confidence = confidence;
	}

	return true;
}

static void handle_detection(sys_slist_t *boxes, int x, int y, float vf, int channel)
{

	bool is_overlapping = false;

	struct box *box;
	SYS_SLIST_FOR_EACH_CONTAINER(boxes, box, node) {
		if (box->channel != channel) {
			continue;
		}

		if (extend_box(box, x, y, 1, 1, vf)) {
			is_overlapping = true;
			break;
		}
	}

	if (!is_overlapping) {
		struct box *box = k_malloc(sizeof(struct box));
		if (box == NULL) {
			LOG_ERR("Failed to allocate memory for box");
			return;
		}
		box->x = x;
		box->y = y;
		box->width = 1;
		box->height = 1;
		box->confidence = vf;
		box->channel = channel;
		sys_slist_append(boxes, &box->node);
	}
}

static void extract_features(float *output, size_t output_size, const uint8_t *input,
			     size_t input_size)
{
	__ASSERT_NO_MSG(output != NULL);
	__ASSERT_NO_MSG(input != NULL);

	// Pre-calculated integer coefficients (scaled by 65536)
	// 0.299 * 65536 / 31 ≈ 631
	// 0.587 * 65536 / 63 ≈ 611
	// 0.114 * 65536 / 31 ≈ 241
	const uint32_t R_COEFF = 631;
	const uint32_t G_COEFF = 611;
	const uint32_t B_COEFF = 241;
	const float SCALE = 1.0f / 65536.0f;

	size_t j = 0;
	size_t max_i = (input_size & ~1); // Ensure even number
	size_t max_j = output_size;

	for (size_t i = 0; i < max_i && j < max_j; i += 2) {
		// Load RGB565 value
		uint16_t rgb565 = (uint16_t)(input[i] << 8) | input[i + 1];

		// Extract components
		uint32_t r = (rgb565 >> 11) & 0x1F;
		uint32_t g = (rgb565 >> 5) & 0x3F;
		uint32_t b = rgb565 & 0x1F;

		// Integer calculation
		uint32_t gray = (R_COEFF * r) + (G_COEFF * g) + (B_COEFF * b);

		// Convert to float only once
		output[j++] = (float)gray * SCALE;
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
		LOG_RAW("%f, ", (double)vector[i]);
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
				LOG_RAW("% .2f (% 4d) ", (double)f_value, value);
			}
			LOG_RAW("\n");
		}

		LOG_RAW("\n");
	}
}

static void print_box(const struct box *box)
{
	static const char *const labels[] = {"background", "beer", "can"};
	LOG_INF("%s (%.2f) [x: %d, y: %d, width: %d, height: %d]", labels[box->channel],
		(double)box->confidence, box->x, box->y, box->width, box->height);
}

static int process_fomo_output(sys_slist_t *results, float threshold,
			       const axon_nn_compiled_model_struct *model)
{
	int8_t height = model->output_dimensions.height;
	int8_t width = model->output_dimensions.width;
	int8_t channel_cnt = model->output_dimensions.channel_cnt;

	for (int8_t c = 1; c < channel_cnt; c++) {
		for (int8_t y = 0; y < height; y++) {
			for (int8_t x = 0; x < width; x++) {
				int8_t value =
					model->output_ptr[height * width * c + y * width + x];
				float f_value = dequantize(model, value);
				if (f_value >= threshold) {
					handle_detection(results, x, y, f_value, c);
				}
			}
		}
	}

	return sys_slist_len(results);
}

static int calc_overlap_size(size_t x1, size_t s1, size_t x2, size_t s2)
{
	int left = MAX(x1, x2);
	int right = MIN(x1 + s1, x2 + s2);
	return right - left;
}

static void save_output_to_records(sys_slist_t *results, sys_slist_t *records)
{
	struct record *record, *safe_record;
	sys_snode_t *prev_r = NULL;

	// update existing records
	SYS_SLIST_FOR_EACH_CONTAINER_SAFE(records, record, safe_record, node) {
		struct box *box, *safe_box;
		sys_snode_t *prev_box_node = NULL;

		record->missing_count += 1;

		SYS_SLIST_FOR_EACH_CONTAINER_SAFE(results, box, safe_box, node) {
			bool same_channel = record->box.channel == box->channel;

			int box_area = box->width * box->height;
			int overlap_width = calc_overlap_size(box->x, box->width, record->box.x,
							      record->box.width);
			int overlap_height = calc_overlap_size(box->y, box->height, record->box.y,
							       record->box.height);

			bool small_adjacent_box =
				box_area == 1 && ((overlap_height == 0 && overlap_width >= 0) ||
						  (overlap_width == 0 && overlap_height >= 0));
			bool high_overlap = overlap_width * overlap_height >= 0.5 * box_area;

			if (same_channel && (small_adjacent_box || high_overlap)) {
				sys_slist_remove(results, prev_box_node, &box->node);
				record->box = *box;
				k_free(box);

				record->present_count = MIN(record->present_count + 1, 3);
				record->missing_count = 0;

				break;
			} else {
				prev_box_node = &box->node;
			}
		}

		if (record->missing_count >= 3) {
			sys_slist_remove(records, prev_r, &record->node);
			k_free(record);
		} else {
			prev_r = &record->node;
		}
	}

	struct box *box, *safe_box;

	// add new records
	SYS_SLIST_FOR_EACH_CONTAINER_SAFE(results, box, safe_box, node) {
		record = k_malloc(sizeof(struct record));
		if (record == NULL) {
			LOG_ERR("Failed to allocate memory for record");
			return;
		}
		record->present_count = 1;
		record->missing_count = 0;
		record->box = *box;
		sys_slist_append(records, &record->node);

		sys_slist_remove(results, NULL, &box->node);
		k_free(box);
	}
}

void take_picture(void)
{
	int err;
	struct video_buffer *vbuf;

	float *f_features_pos = f_features;
	uint32_t f_size_left = ARRAY_SIZE(f_features);

	err = video_dequeue(video, &vbuf, K_FOREVER);
	if (err) {
		LOG_ERR("Unable to dequeue video buf (%d)", err);
		return;
	}

	extract_features(f_features_pos, f_size_left, vbuf->buffer, vbuf->bytesused);
	f_features_pos += vbuf->bytesused / 2;
	f_size_left -= vbuf->bytesused / 2;

	vbuf->type = VIDEO_BUF_TYPE_OUTPUT;
	video_enqueue(video, vbuf);
	while (f_size_left != 0) {
		video_dequeue(video, &vbuf, K_FOREVER);

		extract_features(f_features_pos, f_size_left, vbuf->buffer, vbuf->bytesused);
		f_features_pos += vbuf->bytesused / 2;
		f_size_left -= vbuf->bytesused / 2;

		vbuf->type = VIDEO_BUF_TYPE_OUTPUT;
		video_enqueue(video, vbuf);
	}

	return;
}

void capture_timer_handler(struct k_timer *dummy);

K_SEM_DEFINE(data_sem, 0, 1);
K_TIMER_DEFINE(capture_timer, capture_timer_handler, NULL);

void capture_timer_handler(struct k_timer *dummy)
{
	k_sem_give(&data_sem);
}

void run_inference(void)
{
	// Init moved inside inference loop to reduce idle power consumption
	// LOG_INF("Start Platform!");
	// AxonResultEnum result = axon_platform_init();
	// if (result != kAxonResultSuccess) {
	// 	LOG_ERR("axon_platform_init failed!");
	// }

	void *axon_handle = axon_driver_get_handle();
	LOG_INF("Prepare and run Axon!");

	const axon_nn_compiled_model_struct *model = &THE_MODEL_STRUCT_NAME(AXON_MODEL_NAME);
	axon_nn_model_inference_wrapper_struct wrapper;

	if (axon_nn_model_init(&wrapper, model) < 0) {
		LOG_ERR("axon_nn_model_init failed!");
	}

	k_timer_start(&capture_timer, K_NO_WAIT, K_MSEC(500));

	sys_slist_t results, records;
	sys_slist_init(&results);
	sys_slist_init(&records);

	while (true) {
		k_sem_take(&data_sem, K_FOREVER);

		video_stream_start(video, VIDEO_BUF_TYPE_OUTPUT);
		take_picture();
		video_stream_stop(video, VIDEO_BUF_TYPE_OUTPUT);

		// print_vector_f("input_features", f_features, ARRAY_SIZE(f_features));

		quantize_vector(q_features, f_features, ARRAY_SIZE(f_features),
				model->inputs[0].quant_mult, model->inputs[0].quant_round,
				model->inputs[0].quant_zp);

		// print_vector("quantized_features", q_features, ARRAY_SIZE(q_features));

		AxonResultEnum result = axon_platform_init();
		if (result != kAxonResultSuccess) {
			LOG_ERR("axon_platform_init failed!");
		}

		uint32_t start = axon_platform_get_ticks();
		result = axon_nn_model_infer_sync(axon_handle, model, &(wrapper.cmd_buf_info),
						  q_features, ARRAY_SIZE(q_features));

		axon_platform_close();

		uint32_t time =
			(axon_platform_get_ticks() - start) * 1000 / axon_platform_get_clk_hz();
		// LOG_INF("axon_nn_model_infer_sync took %d ms", time);

		if (result != kAxonResultSuccess) {
			LOG_ERR("axon_nn_model_infer_sync failed!\n");
		} else {
			// print_model_output("output", model);

			process_fomo_output(&results, 0.5, model);

			LOG_RAW("\n");
			LOG_INF("New prediction results:");

			if (sys_slist_is_empty(&results)) {
				LOG_INF("no prediction");
			}

			const struct box *box;
			SYS_SLIST_FOR_EACH_CONTAINER(&results, box, node) {
				print_box(box);
			}

			save_output_to_records(&results, &records);

			LOG_INF("Tracked results:");
			const struct record *record;
			SYS_SLIST_FOR_EACH_CONTAINER(&records, record, node) {
				if (record->present_count < 3) {
					continue;
				}

				print_box(&record->box);
			}
		}
	}

	LOG_INF("Inference complete");
	// axon_platform_close();
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
