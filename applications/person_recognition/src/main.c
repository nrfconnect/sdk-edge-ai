/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Person recognition (nRF54L Axon): mcunet_vww_320kb on embedded test images.
 * Logs person present (argmax) and approximate person probability from dequantized logits,
 * matching the PC path in eval_det.py (_dequantize_output + softmax on 2 logits).
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <math.h>
#include <stdint.h>

#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <axon/nrf_axon_platform.h>

#include "nrf_axon_model_mcunet_vww_320kb_.h"
#include "generated/test_images.h"

LOG_MODULE_REGISTER(person_recognition);

#define MCUNET_NUM_CLASSES           2
#define MCUNET_CLASS_PERSON          1
#define MCUNET_PACKED_OUTPUT_BYTES   NRF_AXON_MODEL_MCUNET_VWW_320KB_PACKED_OUTPUT_SIZE

static int8_t output_buf[MCUNET_PACKED_OUTPUT_BYTES];

/*
 * Axon header: float_output = (quant - output_dequant_zp) * output_dequant_mult / 2^output_dequant_round
 * (same linear map for every output element, so argmax on int32 matches argmax on float.)
 */
static float dequant_logit(int32_t q, const nrf_axon_nn_compiled_model_s *model)
{
	const uint32_t deq_mult = model->output_dequant_mult;
	const uint8_t deq_round = model->output_dequant_round;
	const int8_t deq_zp = model->output_dequant_zp;

	return (float)((q - deq_zp) * ((float)deq_mult / (1 << deq_round)));
}

/* P(class 1) with 2 logits — same as eval_det.py: softmax then take index 1. */
static float person_probability_two_class(float logit0, float logit1)
{
	float d = logit1 - logit0;

	if (d >= 0.f) {
		return 1.f / (1.f + expf(-d));
	}
	float ed = expf(d);

	return ed / (1.f + ed);
}

int main(void)
{
	nrf_axon_result_e result;
	const nrf_axon_nn_compiled_model_s *model = &model_mcunet_vww_320kb;

	LOG_INF("Person recognition (nRF54L Axon, mcunet_vww_320kb)");

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

	for (size_t i = 0; i < PERSON_RECOGNITION_NUM_TEST_IMAGES; i++) {
		const int8_t *input = person_recognition_test_inputs[i];
		const char *name = person_recognition_test_names[i];

		result = nrf_axon_nn_model_infer_sync(model, input, output_buf);
		if (result != NRF_AXON_RESULT_SUCCESS) {
			LOG_ERR("%s: inference failed: %d", name, result);
			continue;
		}

		const int32_t *q = (const int32_t *)output_buf;

		float l0 = dequant_logit(output_buf[0], model);
		float l1 = dequant_logit(output_buf[1], model);
		float p_person = person_probability_two_class(l0, l1);

		int32_t score = 0;
		int16_t class_idx = nrf_axon_nn_get_classification(model, output_buf, NULL, &score);

		if (class_idx < 0) {
			LOG_ERR("%s: classification failed", name);
			continue;
		}

		if (class_idx < MCUNET_NUM_CLASSES) {
			LOG_INF("%s: person present: %s (class %d, raw score %d, P(person) %.4f, logits %.4f %.4f)",
				name,
				class_idx == MCUNET_CLASS_PERSON ? "yes" : "no",
				class_idx,
				(int)score,
				(double)p_person,
				(double)l0,
				(double)l1);
		} else {
			LOG_WRN("%s: unexpected class index: %d", name, class_idx);
		}
	}

	LOG_INF("Person recognition done.");
	return 0;
}
