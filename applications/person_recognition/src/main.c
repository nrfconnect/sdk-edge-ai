/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Person detection (nRF54L Axon): person_det with Arducam Mega @ 128x128 RGB565.
 * Same camera path as mcunet_vww_320kb (video_set_format, 1024-byte buffers, chunked capture).
 * Model canvas 160x128: 128x128 image centered horizontally; border uses neutral gray (sym -1..1 → 0).
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/video.h>
#include <zephyr/logging/log.h>

#include <stdint.h>

#include <zephyr/sys/atomic.h>

#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <axon/nrf_axon_platform.h>

#include "nrf_axon_model_person_det_.h"
#include "person_det_postprocess.h"

LOG_MODULE_REGISTER(person_recognition);

#define PACKED_OUTPUT_BYTES NRF_AXON_MODEL_PERSON_DET_PACKED_OUTPUT_SIZE

#define CAM_W 128
#define CAM_H 128
#define MODEL_W 160
#define MODEL_H 128
#define PAD_LEFT (((MODEL_W) - (CAM_W)) / 2)
#define PAD_TOP  (((MODEL_H) - (CAM_H)) / 2)

BUILD_ASSERT(MODEL_W == 160 && MODEL_H == 128, "person_det input canvas");
BUILD_ASSERT(PAD_LEFT + CAM_W + PAD_LEFT == MODEL_W, "horizontal pad");
BUILD_ASSERT(PAD_TOP + CAM_H + PAD_TOP == MODEL_H, "vertical pad");

#define FRAME_RGB565_BYTES ((CAM_W) * (CAM_H) * 2)

#define MAX_BOXES_LOG 8

static int8_t output_buf[PACKED_OUTPUT_BYTES];
static int8_t input_buf[MODEL_W * MODEL_H * 3];
static uint8_t frame_rgb565[FRAME_RGB565_BYTES];

static const struct gpio_dt_spec led_capture = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
static const struct gpio_dt_spec led_person = GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios);

static atomic_t capture_led_active;

static void capture_led_timer_fn(struct k_timer *timer)
{
	ARG_UNUSED(timer);

	if (atomic_get(&capture_led_active)) {
		(void)gpio_pin_toggle_dt(&led_capture);
	}
}

K_TIMER_DEFINE(capture_led_timer, capture_led_timer_fn, NULL);

static int8_t quantize_sym(float sym_m1_1, uint32_t quant_mult, uint8_t quant_round, int8_t quant_zp)
{
	double v = (double)sym_m1_1 * (double)quant_mult;
	int64_t qi = (int64_t)v;

	qi >>= quant_round;
	qi += quant_zp;
	if (qi < -128) {
		return -128;
	}
	if (qi > 127) {
		return 127;
	}
	return (int8_t)qi;
}

static inline uint16_t rgb565_read_be(const uint8_t *p)
{
	return (uint16_t)((uint16_t)p[0] << 8 | p[1]);
}

static inline float rgb565_channel_to_01(uint16_t pix, unsigned c)
{
	unsigned r5 = (pix >> 11) & 0x1f;
	unsigned g6 = (pix >> 5) & 0x3f;
	unsigned b5 = pix & 0x1f;
	unsigned r8 = (r5 << 3) | (r5 >> 2);
	unsigned g8 = (g6 << 2) | (g6 >> 4);
	unsigned b8 = (b5 << 3) | (b5 >> 2);

	if (c == 0) {
		return (float)r8 / 255.f;
	}
	if (c == 1) {
		return (float)g8 / 255.f;
	}
	return (float)b8 / 255.f;
}

static void build_model_input_from_frame(const uint8_t *rgb565, const nrf_axon_nn_compiled_model_s *model)
{
	const nrf_axon_nn_compiled_model_input_s *in = &model->inputs[model->external_input_ndx];
	const uint32_t qm = in->quant_mult;
	const uint8_t qr = in->quant_round;
	const int8_t qz = in->quant_zp;

	for (unsigned c = 0; c < 3; c++) {
		for (unsigned y = 0; y < MODEL_H; y++) {
			for (unsigned x = 0; x < MODEL_W; x++) {
				float ch01;

				if (x >= PAD_LEFT && x < PAD_LEFT + CAM_W && y >= PAD_TOP &&
				    y < PAD_TOP + CAM_H) {
					unsigned sx = x - PAD_LEFT;
					unsigned sy = y - PAD_TOP;
					uint16_t pix = rgb565_read_be(&rgb565[(sy * CAM_W + sx) * 2]);

					ch01 = rgb565_channel_to_01(pix, c);
				} else {
					ch01 = 0.5f;
				}

				float sym = ch01 * 2.f - 1.f;
				unsigned plane = c * MODEL_W * MODEL_H;
				unsigned idx = plane + y * MODEL_W + x;

				input_buf[idx] = quantize_sym(sym, qm, qr, qz);
			}
		}
	}
}

static int capture_one_frame(const struct device *video)
{
	size_t total = 0;

	while (total < FRAME_RGB565_BYTES) {
		struct video_buffer *vbuf;
		int err = video_dequeue(video, &vbuf, K_FOREVER);

		if (err != 0) {
			LOG_ERR("video_dequeue failed: %d", err);
			return err;
		}

		size_t room = FRAME_RGB565_BYTES - total;
		size_t chunk = vbuf->bytesused < room ? vbuf->bytesused : room;

		memcpy(frame_rgb565 + total, vbuf->buffer, chunk);
		total += chunk;

		vbuf->type = VIDEO_BUF_TYPE_OUTPUT;
		video_enqueue(video, vbuf);
	}

	return 0;
}

int main(void)
{
	nrf_axon_result_e result;
	const nrf_axon_nn_compiled_model_s *model = &model_person_det;
	const struct device *video = DEVICE_DT_GET(DT_NODELABEL(arducam_mega));
	struct video_buffer *vbufs[2];
	struct video_format fmt = {.pixelformat = VIDEO_PIX_FMT_RGB565,
				   .width = CAM_W,
				   .height = CAM_H,
				   .pitch = CAM_W * 2};

	struct person_det_box boxes[MAX_BOXES_LOG];
	const float score_thresh = 0.25f;
	const float nms_iou = 0.45f;

	if (!gpio_is_ready_dt(&led_capture) || !gpio_is_ready_dt(&led_person)) {
		LOG_ERR("LED GPIO not ready");
		return -1;
	}
	(void)gpio_pin_configure_dt(&led_capture, GPIO_OUTPUT_INACTIVE);
	(void)gpio_pin_configure_dt(&led_person, GPIO_OUTPUT_INACTIVE);

	if (!device_is_ready(video)) {
		LOG_ERR("Video device not ready");
		return -1;
	}

	if (video_set_format(video, &fmt) != 0) {
		LOG_ERR("video_set_format failed");
		return -1;
	}

	for (size_t i = 0; i < ARRAY_SIZE(vbufs); i++) {
		vbufs[i] = video_buffer_alloc(1024, K_FOREVER);
		if (vbufs[i] == NULL) {
			LOG_ERR("video_buffer_alloc failed");
			return -1;
		}
		vbufs[i]->type = VIDEO_BUF_TYPE_OUTPUT;
		video_enqueue(video, vbufs[i]);
	}

	LOG_INF("Person detection (camera %dx%d -> model %dx%d)", CAM_W, CAM_H, MODEL_W, MODEL_H);

	result = nrf_axon_platform_init();
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Axon platform init failed: %d", result);
		return -1;
	}

	result = nrf_axon_nn_model_validate(model);
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Model validation failed: %d", result);
		return -1;
	}

	if (nrf_axon_nn_model_init_vars(model) != 0) {
		LOG_ERR("Model init_vars failed");
		return -1;
	}

	while (true) {
		atomic_set(&capture_led_active, 1);
		k_timer_start(&capture_led_timer, K_NO_WAIT, K_MSEC(55));

		if (video_stream_start(video, VIDEO_BUF_TYPE_OUTPUT) != 0) {
			LOG_ERR("video_stream_start failed");
			atomic_set(&capture_led_active, 0);
			k_timer_stop(&capture_led_timer);
			gpio_pin_set_dt(&led_capture, 0);
			k_msleep(500);
			continue;
		}

		if (capture_one_frame(video) != 0) {
			(void)video_stream_stop(video, VIDEO_BUF_TYPE_OUTPUT);
			atomic_set(&capture_led_active, 0);
			k_timer_stop(&capture_led_timer);
			gpio_pin_set_dt(&led_capture, 0);
			k_msleep(500);
			continue;
		}

		(void)video_stream_stop(video, VIDEO_BUF_TYPE_OUTPUT);

		atomic_set(&capture_led_active, 0);
		k_timer_stop(&capture_led_timer);
		gpio_pin_set_dt(&led_capture, 0);

		build_model_input_from_frame(frame_rgb565, model);

		result = nrf_axon_nn_model_infer_sync(model, input_buf, output_buf);
		if (result != NRF_AXON_RESULT_SUCCESS) {
			LOG_ERR("inference failed: %d", result);
			gpio_pin_set_dt(&led_person, 0);
			k_msleep(300);
			continue;
		}

		int n = person_det_decode_and_nms(model, boxes, MAX_BOXES_LOG, score_thresh, nms_iou);

		if (n > 0) {
			for (int i = 0; i < n; i++) {
				LOG_INF(
					"person box %d: head %s [%.1f, %.1f, %.1f, %.1f] score %.3f",
					i, person_det_head_name(boxes[i].head),
					(double)boxes[i].x1, (double)boxes[i].y1, (double)boxes[i].x2,
					(double)boxes[i].y2, (double)boxes[i].score);
			}
			gpio_pin_toggle_dt(&led_person);
		} else {
			gpio_pin_set_dt(&led_person, 0);
		}

		k_msleep(280);
	}

	return 0;
}
