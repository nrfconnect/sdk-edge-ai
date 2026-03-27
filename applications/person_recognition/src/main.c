/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Person recognition (nRF54L Axon): mcunet_vww_320kb with Arducam Mega @ 128x128 RGB565.
 * Camera ROI is centered on the 144x144 model canvas; border is filled with neutral gray
 * (symmetric_m1_1 value 0). Input quantization matches embed_test_images.py / eval_det.py.
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/video.h>
#include <zephyr/logging/log.h>

#include <math.h>
#include <stdint.h>

#include <zephyr/sys/atomic.h>

#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <axon/nrf_axon_platform.h>

#include "nrf_axon_model_mcunet_vww_320kb_.h"

LOG_MODULE_REGISTER(person_recognition);

#define MCUNET_NUM_CLASSES         2
#define MCUNET_CLASS_PERSON        1
#define MCUNET_PACKED_OUTPUT_BYTES NRF_AXON_MODEL_MCUNET_VWW_320KB_PACKED_OUTPUT_SIZE

#define CAM_W 128
#define CAM_H 128
#define MODEL_W 144
#define MODEL_H 144
#define ROI_PAD (((MODEL_W) - (CAM_W)) / 2)

BUILD_ASSERT(MODEL_W == 144 && MODEL_H == 144, "model canvas");
BUILD_ASSERT(2 * ROI_PAD + CAM_W == MODEL_W, "pad");

#define FRAME_RGB565_BYTES ((CAM_W) * (CAM_H) * 2)

static int8_t output_buf[MCUNET_PACKED_OUTPUT_BYTES];
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

static float dequant_logit(int32_t q, const nrf_axon_nn_compiled_model_s *model)
{
	const uint32_t deq_mult = model->output_dequant_mult;
	const uint8_t deq_round = model->output_dequant_round;
	const int8_t deq_zp = model->output_dequant_zp;

	return (float)((q - deq_zp) * ((float)deq_mult / (1U << deq_round)));
}

static float person_probability_two_class(float logit0, float logit1)
{
	float d = logit1 - logit0;

	if (d >= 0.f) {
		return 1.f / (1.f + expf(-d));
	}
	float ed = expf(d);

	return ed / (1.f + ed);
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

				if (x >= ROI_PAD && x < ROI_PAD + CAM_W && y >= ROI_PAD &&
				    y < ROI_PAD + CAM_H) {
					unsigned sx = x - ROI_PAD;
					unsigned sy = y - ROI_PAD;
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
	const nrf_axon_nn_compiled_model_s *model = &model_mcunet_vww_320kb;
	const struct device *video = DEVICE_DT_GET(DT_NODELABEL(arducam_mega));
	struct video_buffer *vbufs[2];
	struct video_format fmt = {.pixelformat = VIDEO_PIX_FMT_RGB565,
				   .width = CAM_W,
				   .height = CAM_H,
				   .pitch = CAM_W * 2};

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

	LOG_INF("Person recognition (camera %dx%d -> model %dx%d)", CAM_W, CAM_H, MODEL_W, MODEL_H);

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

		float l0 = dequant_logit(output_buf[0], model);
		float l1 = dequant_logit(output_buf[1], model);
		float p_person = person_probability_two_class(l0, l1);

		int32_t score = 0;
		int16_t class_idx = nrf_axon_nn_get_classification(model, output_buf, NULL, &score);

		if (class_idx < 0) {
			LOG_ERR("classification failed");
			gpio_pin_set_dt(&led_person, 0);
			k_msleep(300);
			continue;
		}

		bool person = (class_idx == MCUNET_CLASS_PERSON);

		if (class_idx < MCUNET_NUM_CLASSES) {
			LOG_INF("person: %s (class %d, P(person) %.4f)",
				person ? "yes" : "no", class_idx, (double)p_person);
		}

		if (person) {
			gpio_pin_toggle_dt(&led_person);
		} else {
			gpio_pin_set_dt(&led_person, 0);
		}

		k_msleep(280);
	}

	return 0;
}
