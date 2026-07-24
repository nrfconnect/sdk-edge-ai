/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "postprocessing.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/storage/flash_map.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#if defined(CONFIG_MODEL_OTA_AXON)
#include <model_ota/model_image.h>
#include <model_ota/axon/person_det.h>
#else
/*
 * Non-OTA build: allocate the packed-output buffer inline (this TU owns
 * model_person_det's storage), so model_person_det.packed_output_buf is non-NULL,
 * matching the OTA build's ALLOCATE_PACKED_OUTPUT-wired image.
 */
#define NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER 1
#include "generated/nrf_axon_model_person_det_.h"
#endif

LOG_MODULE_REGISTER(multi_person_det, LOG_LEVEL_INF);

#define MODEL_WIDTH  160
#define MODEL_HEIGHT 128
#define MAX_BOXES    8

static int8_t input_buf[MODEL_WIDTH * MODEL_HEIGHT * 3];

static inline int8_t quantize(const float value, const nrf_axon_nn_compiled_model_input_s *in)
{
	const uint32_t quant_mult = in->quant_mult;
	const uint8_t quant_round = in->quant_round;
	const int8_t quant_zp = in->quant_zp;
	const float scale = (float)quant_mult / (float)(1 << quant_round);
	const int32_t quantized = (int32_t)(value * scale) + quant_zp;

	return (int8_t)__ssat(quantized, 8);
}

static void prefill_dummy_input(const nrf_axon_nn_compiled_model_input_s *in)
{
	const int8_t gray = quantize(0.0f, in);

	memset(input_buf, gray, sizeof(input_buf));
}

void run_person_det_tests(void)
{
	nrf_axon_result_e result;
	const nrf_axon_nn_compiled_model_s *model;
	const nrf_axon_nn_compiled_model_input_s *model_input;
	struct detection_box boxes[MAX_BOXES];

#if defined(CONFIG_MODEL_OTA_AXON)
	if (model_image_load_axon(PARTITION_ID(model_person_det_storage),
				  (const uint8_t *)PARTITION_ADDRESS(model_person_det_storage),
				  &model) != MODEL_IMAGE_OK ||
	    model == NULL) {
		LOG_WRN("No valid person-det model image in model_person_det_storage - skipping "
			"(flash person_det_model_partition.hex)");
		return;
	}
#else
	model = &model_person_det;
#endif

	/*
	 * Use the model's own packed-output buffer (app-owned storage; OTA-wired via
	 * ALLOCATE_PACKED_OUTPUT) rather than a separate local buffer, so this path is
	 * always exercised and any build/wiring regression that leaves it NULL is caught
	 * immediately instead of silently falling back.
	 */
	__ASSERT_NO_MSG(model->packed_output_buf != NULL);

	model_input = nrf_axon_nn_model_1st_external_input(model);

	result = nrf_axon_platform_init();
	__ASSERT_NO_MSG(result == NRF_AXON_RESULT_SUCCESS);

	result = nrf_axon_nn_model_validate(model);
	__ASSERT_NO_MSG(result == NRF_AXON_RESULT_SUCCESS);

	prefill_dummy_input(model_input);
	decode_init(model);

	result = nrf_axon_nn_model_infer_sync(model, input_buf, model->packed_output_buf);
	__ASSERT_NO_MSG(result == NRF_AXON_RESULT_SUCCESS);

	const size_t detections = decode_output(model, model->packed_output_buf, boxes, MAX_BOXES);

	LOG_INF("Person detection inference on dummy gray input: %zu detection(s)",
		detections);
}
